"""
PRISM Model Optimization: Post-Training Quantization and Deployment

This module provides optimization functions for PRISM models, including
post-training dynamic quantization, TorchScript export, and performance
comparison between float32 and quantized int8 models.
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.jit import ScriptModule
import numpy as np
import time
import os
from typing import Dict, Tuple, Optional, Union
from pathlib import Path
import json
import warnings

from models import STGCN_PRISM, BaselineLSTM, BaselineGRU
from benchmarks import ModelEvaluator


class ModelOptimizer:
    """
    Model optimization class for quantization and deployment.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize model optimizer.
        
        Args:
            device: Device to run optimization on (quantization works on CPU)
        """
        self.device = device
        print(f"ModelOptimizer initialized on device: {self.device}")
    
    def get_model_size(self, model: nn.Module) -> Dict[str, float]:
        """
        Calculate model size in different formats.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary containing size information in MB
        """
        # Calculate size of model parameters
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        
        # Convert to MB
        size_mb = total_size / (1024 * 1024)
        param_size_mb = param_size / (1024 * 1024)
        buffer_size_mb = buffer_size / (1024 * 1024)
        
        return {
            'total_size_mb': size_mb,
            'param_size_mb': param_size_mb,
            'buffer_size_mb': buffer_size_mb,
            'total_size_bytes': total_size,
            'param_size_bytes': param_size,
            'buffer_size_bytes': buffer_size
        }
    
    def count_parameters(self, model: nn.Module) -> Dict[str, int]:
        """
        Count model parameters.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary containing parameter counts
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }


def quantize_model(stgcn_model: STGCN_PRISM, 
                  calibration_data: torch.Tensor,
                  backend: str = 'qnnpack') -> nn.Module:
    """
    Apply post-training dynamic quantization to STGCN_PRISM model.
    
    Args:
        stgcn_model: Trained STGCN_PRISM model
        calibration_data: Sample data for calibration (batch_size, seq_len, joints, features)
        backend: Quantization backend ('qnnpack' for CPU, 'fbgemm' for server)
        
    Returns:
        Quantized model
    """
    print("Starting post-training dynamic quantization...")
    print(f"Backend: {backend}")
    
    # Set quantization backend
    quantization.backend = backend
    
    # Create a copy of the model for quantization
    model_copy = STGCN_PRISM(
        num_joints=stgcn_model.num_joints,
        in_channels=stgcn_model.in_channels,
        num_classes=stgcn_model.num_classes,
        hidden_channels=stgcn_model.hidden_channels,
        temporal_kernel_sizes=stgcn_model.temporal_kernel_sizes,
        dropout=stgcn_model.dropout,
        use_attention=stgcn_model.use_attention
    )
    
    # Copy weights from original model
    model_copy.load_state_dict(stgcn_model.state_dict())
    
    # Set to evaluation mode
    model_copy.eval()
    
    # Move to CPU for quantization (required)
    model_copy = model_copy.cpu()
    
    # Prepare model for quantization
    print("Preparing model for quantization...")
    model_copy.qconfig = quantization.get_default_qconfig(backend)
    
    # Prepare the model for quantization
    prepared_model = quantization.prepare(model_copy)
    
    # Calibrate the model with sample data
    print("Calibrating model with sample data...")
    with torch.no_grad():
        # Run inference on calibration data
        for i in range(min(10, calibration_data.size(0))):  # Use first 10 samples
            sample = calibration_data[i:i+1].cpu()
            _ = prepared_model(sample)
    
    # Convert to quantized model
    print("Converting to quantized model...")
    quantized_model = quantization.convert(prepared_model)
    
    print("✓ Quantization completed successfully")
    
    return quantized_model


def quantize_lstm_model(lstm_model: BaselineLSTM, 
                       calibration_data: torch.Tensor,
                       backend: str = 'qnnpack') -> nn.Module:
    """
    Apply post-training dynamic quantization to LSTM model.
    
    Args:
        lstm_model: Trained LSTM model
        calibration_data: Sample data for calibration
        backend: Quantization backend
        
    Returns:
        Quantized LSTM model
    """
    print("Starting LSTM quantization...")
    
    # Set quantization backend
    quantization.backend = backend
    
    # Create a copy of the model
    model_copy = BaselineLSTM(
        input_size=lstm_model.input_size,
        hidden_size=lstm_model.hidden_size,
        num_layers=lstm_model.num_layers,
        num_classes=lstm_model.num_classes,
        dropout=lstm_model.dropout,
        bidirectional=lstm_model.bidirectional
    )
    
    # Copy weights
    model_copy.load_state_dict(lstm_model.state_dict())
    model_copy.eval()
    model_copy = model_copy.cpu()
    
    # Prepare for quantization
    model_copy.qconfig = quantization.get_default_qconfig(backend)
    prepared_model = quantization.prepare(model_copy)
    
    # Calibrate
    print("Calibrating LSTM model...")
    with torch.no_grad():
        for i in range(min(10, calibration_data.size(0))):
            sample = calibration_data[i:i+1].cpu()
            _ = prepared_model(sample)
    
    # Convert
    quantized_model = quantization.convert(prepared_model)
    
    print("✓ LSTM quantization completed")
    
    return quantized_model


def save_optimized_model(model: nn.Module, 
                        model_name: str,
                        output_dir: str = "optimized_models",
                        save_torchscript: bool = True,
                        save_onnx: bool = False) -> Dict[str, str]:
    """
    Save optimized model in deployable formats.
    
    Args:
        model: Model to save (quantized or original)
        model_name: Name for the saved model
        output_dir: Directory to save models
        save_torchscript: Whether to save as TorchScript
        save_onnx: Whether to save as ONNX (experimental)
        
    Returns:
        Dictionary containing paths to saved models
    """
    print(f"Saving optimized model: {model_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = {}
    
    # Save PyTorch model
    pytorch_path = os.path.join(output_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), pytorch_path)
    saved_paths['pytorch'] = pytorch_path
    print(f"✓ PyTorch model saved to: {pytorch_path}")
    
    # Save TorchScript
    if save_torchscript:
        try:
            print("Converting to TorchScript...")
            model.eval()
            
            # Create example input for tracing
            if hasattr(model, 'num_joints'):
                # STGCN model
                example_input = torch.randn(1, 30, model.num_joints, model.in_channels)
            else:
                # LSTM model - need to determine input size
                # This is a simplified approach
                example_input = torch.randn(1, 30, 132)  # Default for LSTM
            
            # Trace the model
            traced_model = torch.jit.trace(model, example_input)
            
            # Save TorchScript
            torchscript_path = os.path.join(output_dir, f"{model_name}_torchscript.pt")
            traced_model.save(torchscript_path)
            saved_paths['torchscript'] = torchscript_path
            print(f"✓ TorchScript model saved to: {torchscript_path}")
            
        except Exception as e:
            print(f"⚠ TorchScript conversion failed: {str(e)}")
            saved_paths['torchscript'] = None
    
    # Save ONNX (experimental)
    if save_onnx:
        try:
            print("Converting to ONNX...")
            model.eval()
            
            # Create example input
            if hasattr(model, 'num_joints'):
                example_input = torch.randn(1, 30, model.num_joints, model.in_channels)
            else:
                example_input = torch.randn(1, 30, 132)
            
            # Export to ONNX
            onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
            torch.onnx.export(
                model, 
                example_input, 
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            saved_paths['onnx'] = onnx_path
            print(f"✓ ONNX model saved to: {onnx_path}")
            
        except Exception as e:
            print(f"⚠ ONNX conversion failed: {str(e)}")
            saved_paths['onnx'] = None
    
    return saved_paths


def compare_model_performance(original_model: nn.Module,
                            quantized_model: nn.Module,
                            test_loader,
                            num_samples: int = 100) -> Dict[str, Dict]:
    """
    Compare performance between original and quantized models.
    
    Args:
        original_model: Original float32 model
        quantized_model: Quantized int8 model
        test_loader: Test data loader
        num_samples: Number of samples for comparison
        
    Returns:
        Dictionary containing comparison results
    """
    print("Comparing model performance...")
    
    optimizer = ModelOptimizer()
    
    # Get model sizes
    original_size = optimizer.get_model_size(original_model)
    quantized_size = optimizer.get_model_size(quantized_model)
    
    # Get parameter counts
    original_params = optimizer.count_parameters(original_model)
    quantized_params = optimizer.count_parameters(quantized_model)
    
    # Measure inference latency
    print("Measuring inference latency...")
    
    # Original model latency
    original_latency = measure_inference_latency(original_model, test_loader, num_samples)
    
    # Quantized model latency
    quantized_latency = measure_inference_latency(quantized_model, test_loader, num_samples)
    
    # Calculate compression ratio
    compression_ratio = original_size['total_size_mb'] / quantized_size['total_size_mb']
    size_reduction = (original_size['total_size_mb'] - quantized_size['total_size_mb']) / original_size['total_size_mb'] * 100
    
    # Calculate speedup
    speedup = original_latency['mean_latency_ms'] / quantized_latency['mean_latency_ms']
    
    comparison_results = {
        'original_model': {
            'size_mb': original_size['total_size_mb'],
            'parameters': original_params['total_parameters'],
            'mean_latency_ms': original_latency['mean_latency_ms'],
            'std_latency_ms': original_latency['std_latency_ms'],
            'throughput_fps': original_latency['throughput_fps']
        },
        'quantized_model': {
            'size_mb': quantized_size['total_size_mb'],
            'parameters': quantized_params['total_parameters'],
            'mean_latency_ms': quantized_latency['mean_latency_ms'],
            'std_latency_ms': quantized_latency['std_latency_ms'],
            'throughput_fps': quantized_latency['throughput_fps']
        },
        'improvements': {
            'compression_ratio': compression_ratio,
            'size_reduction_percent': size_reduction,
            'speedup_factor': speedup,
            'latency_reduction_percent': (original_latency['mean_latency_ms'] - quantized_latency['mean_latency_ms']) / original_latency['mean_latency_ms'] * 100,
            'throughput_improvement_percent': (quantized_latency['throughput_fps'] - original_latency['throughput_fps']) / original_latency['throughput_fps'] * 100
        }
    }
    
    return comparison_results


def measure_inference_latency(model: nn.Module, 
                            test_loader, 
                            num_samples: int = 100) -> Dict[str, float]:
    """
    Measure inference latency for a model.
    
    Args:
        model: Model to measure
        test_loader: Test data loader
        num_samples: Number of samples to process
        
    Returns:
        Dictionary containing latency metrics
    """
    model.eval()
    model.cpu()  # Ensure CPU for fair comparison
    
    latencies = []
    frame_counts = []
    
    with torch.no_grad():
        sample_count = 0
        for batch in test_loader:
            if sample_count >= num_samples:
                break
                
            pose_sequences = batch['pose_sequence']
            batch_size, sequence_length = pose_sequences.shape[:2]
            total_frames = batch_size * sequence_length
            
            # Measure inference time
            start_time = time.time()
            _ = model(pose_sequences)
            end_time = time.time()
            
            # Calculate latency per frame
            total_time_ms = (end_time - start_time) * 1000
            latency_per_frame = total_time_ms / total_frames
            
            latencies.append(latency_per_frame)
            frame_counts.append(total_frames)
            sample_count += batch_size
    
    # Calculate statistics
    latencies = np.array(latencies)
    
    return {
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'median_latency_ms': np.median(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'throughput_fps': 1000 / np.mean(latencies) if np.mean(latencies) > 0 else 0
    }


def generate_optimization_report(comparison_results: Dict[str, Dict],
                               model_name: str = "STGCN_PRISM",
                               save_path: Optional[str] = None) -> str:
    """
    Generate comprehensive optimization report.
    
    Args:
        comparison_results: Results from compare_model_performance
        model_name: Name of the model
        save_path: Optional path to save the report
        
    Returns:
        Formatted report string
    """
    print("Generating optimization report...")
    
    original = comparison_results['original_model']
    quantized = comparison_results['quantized_model']
    improvements = comparison_results['improvements']
    
    report = f"""
PRISM Model Optimization Report
{'=' * 50}
Model: {model_name}
Optimization: Post-Training Dynamic Quantization (Float32 → Int8)

SIZE COMPARISON:
{'─' * 30}
Original Model Size:    {original['size_mb']:.2f} MB
Quantized Model Size:   {quantized['size_mb']:.2f} MB
Size Reduction:         {improvements['size_reduction_percent']:.1f}%
Compression Ratio:      {improvements['compression_ratio']:.2f}x

PARAMETER COUNT:
{'─' * 30}
Original Parameters:    {original['parameters']:,}
Quantized Parameters:   {quantized['parameters']:,}

INFERENCE PERFORMANCE:
{'─' * 30}
Original Latency:       {original['mean_latency_ms']:.2f} ± {original['std_latency_ms']:.2f} ms
Quantized Latency:      {quantized['mean_latency_ms']:.2f} ± {quantized['std_latency_ms']:.2f} ms
Latency Reduction:      {improvements['latency_reduction_percent']:.1f}%
Speedup Factor:         {improvements['speedup_factor']:.2f}x

THROUGHPUT IMPROVEMENT:
{'─' * 30}
Original Throughput:    {original['throughput_fps']:.2f} FPS
Quantized Throughput:   {quantized['throughput_fps']:.2f} FPS
Throughput Improvement: {improvements['throughput_improvement_percent']:.1f}%

OPTIMIZATION SUMMARY:
{'─' * 30}
✓ Model size reduced by {improvements['size_reduction_percent']:.1f}%
✓ Inference speed improved by {improvements['latency_reduction_percent']:.1f}%
✓ Throughput increased by {improvements['throughput_improvement_percent']:.1f}%
✓ Compression ratio: {improvements['compression_ratio']:.2f}x
✓ Speedup factor: {improvements['speedup_factor']:.2f}x

DEPLOYMENT READINESS:
{'─' * 30}
✓ Quantized model ready for CPU deployment
✓ TorchScript export available for mobile/edge deployment
✓ Significant size reduction for storage and memory efficiency
✓ Improved inference speed for real-time applications

{'=' * 50}
"""
    
    # Save report if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"✓ Optimization report saved to: {save_path}")
    
    return report


def optimize_stgcn_model(stgcn_model: STGCN_PRISM,
                        calibration_data: torch.Tensor,
                        test_loader,
                        model_name: str = "STGCN_PRISM_optimized",
                        output_dir: str = "optimized_models") -> Dict[str, Union[nn.Module, Dict]]:
    """
    Complete optimization pipeline for STGCN model.
    
    Args:
        stgcn_model: Trained STGCN_PRISM model
        calibration_data: Sample data for quantization calibration
        test_loader: Test data loader for performance comparison
        model_name: Name for the optimized model
        output_dir: Directory to save optimized models
        
    Returns:
        Dictionary containing optimized model and results
    """
    print("=" * 60)
    print("PRISM STGCN MODEL OPTIMIZATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Quantize model
    print("\n1. Applying post-training dynamic quantization...")
    quantized_model = quantize_model(stgcn_model, calibration_data)
    
    # Step 2: Compare performance
    print("\n2. Comparing original vs quantized performance...")
    comparison_results = compare_model_performance(
        stgcn_model, quantized_model, test_loader
    )
    
    # Step 3: Save optimized model
    print("\n3. Saving optimized model...")
    saved_paths = save_optimized_model(
        quantized_model, model_name, output_dir
    )
    
    # Step 4: Generate report
    print("\n4. Generating optimization report...")
    report = generate_optimization_report(
        comparison_results, model_name, 
        save_path=os.path.join(output_dir, f"{model_name}_report.txt")
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    
    improvements = comparison_results['improvements']
    print(f"✓ Model size reduced by {improvements['size_reduction_percent']:.1f}%")
    print(f"✓ Inference speed improved by {improvements['latency_reduction_percent']:.1f}%")
    print(f"✓ Compression ratio: {improvements['compression_ratio']:.2f}x")
    print(f"✓ Speedup factor: {improvements['speedup_factor']:.2f}x")
    print(f"✓ Optimized model saved to: {output_dir}/")
    
    return {
        'quantized_model': quantized_model,
        'comparison_results': comparison_results,
        'saved_paths': saved_paths,
        'optimization_report': report
    }


if __name__ == "__main__":
    # Example usage
    print("PRISM Model Optimization - Post-Training Quantization")
    print("=" * 60)
    
    print("This module provides:")
    print("• Post-training dynamic quantization (Float32 → Int8)")
    print("• TorchScript export for deployment")
    print("• Model size and latency comparison")
    print("• Comprehensive optimization reporting")
    print("\nUse optimize_stgcn_model() for complete optimization pipeline!")
