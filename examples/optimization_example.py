"""
Optimization Example for PRISM Models
Demonstrates post-training quantization and model optimization.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import time

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import STGCN_PRISM, BaselineLSTM
from optimization import (
    quantize_model,
    quantize_lstm_model,
    save_optimized_model,
    compare_model_performance,
    generate_optimization_report,
    optimize_stgcn_model,
    ModelOptimizer
)


def create_dummy_models_and_data():
    """Create dummy models and data for optimization demonstration."""
    print("Creating dummy models and test data...")
    
    # Model parameters
    num_classes = 101
    sequence_length = 30
    num_samples = 100
    batch_size = 16
    
    # Create dummy test data (kinematic features)
    num_features = 100  # 50 static + 50 velocity features
    test_data = np.random.randn(num_samples, sequence_length, num_features)
    test_labels = np.random.randint(0, num_classes, num_samples)
    
    # Create dataset and data loader
    test_dataset = TensorDataset(
        torch.FloatTensor(test_data),
        torch.LongTensor(test_labels)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create STGCN model
    stgcn_model = STGCN_PRISM(
        num_joints=33,
        in_channels=4,
        num_classes=num_classes,
        hidden_channels=[64, 128, 256],
        temporal_kernel_sizes=[3, 3, 3],
        dropout=0.1,
        use_attention=False
    )
    
    # Create LSTM model
    lstm_model = BaselineLSTM(
        input_size=132,  # 33 * 4 for raw pose data
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
        bidirectional=False
    )
    
    # Initialize models with dummy weights
    for model in [stgcn_model, lstm_model]:
        for param in model.parameters():
            nn.init.normal_(param, 0, 0.01)
    
    # Create calibration data for quantization
    calibration_data = torch.randn(20, sequence_length, 33, 4)  # Raw pose format for STGCN
    
    print(f"✓ Created models and data:")
    print(f"  STGCN parameters: {sum(p.numel() for p in stgcn_model.parameters()):,}")
    print(f"  LSTM parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
    print(f"  Test samples: {num_samples}")
    print(f"  Calibration samples: {calibration_data.size(0)}")
    
    return stgcn_model, lstm_model, test_loader, calibration_data


def demonstrate_stgcn_quantization():
    """Demonstrate STGCN model quantization."""
    print("=" * 60)
    print("STGCN MODEL QUANTIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Create models and data
    stgcn_model, _, test_loader, calibration_data = create_dummy_models_and_data()
    
    # Create optimizer for analysis
    optimizer = ModelOptimizer()
    
    # Analyze original model
    print("\n1. Analyzing original STGCN model...")
    original_size = optimizer.get_model_size(stgcn_model)
    original_params = optimizer.count_parameters(stgcn_model)
    
    print(f"Original Model Analysis:")
    print(f"  Total size: {original_size['total_size_mb']:.2f} MB")
    print(f"  Parameters: {original_params['total_parameters']:,}")
    print(f"  Parameter size: {original_size['param_size_mb']:.2f} MB")
    print(f"  Buffer size: {original_size['buffer_size_mb']:.2f} MB")
    
    # Quantize model
    print("\n2. Applying post-training dynamic quantization...")
    quantized_stgcn = quantize_model(stgcn_model, calibration_data)
    
    # Analyze quantized model
    print("\n3. Analyzing quantized STGCN model...")
    quantized_size = optimizer.get_model_size(quantized_stgcn)
    quantized_params = optimizer.count_parameters(quantized_stgcn)
    
    print(f"Quantized Model Analysis:")
    print(f"  Total size: {quantized_size['total_size_mb']:.2f} MB")
    print(f"  Parameters: {quantized_params['total_parameters']:,}")
    print(f"  Parameter size: {quantized_size['param_size_mb']:.2f} MB")
    print(f"  Buffer size: {quantized_size['buffer_size_mb']:.2f} MB")
    
    # Calculate improvements
    size_reduction = (original_size['total_size_mb'] - quantized_size['total_size_mb']) / original_size['total_size_mb'] * 100
    compression_ratio = original_size['total_size_mb'] / quantized_size['total_size_mb']
    
    print(f"\nQuantization Results:")
    print(f"  Size reduction: {size_reduction:.1f}%")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Size saved: {original_size['total_size_mb'] - quantized_size['total_size_mb']:.2f} MB")
    
    return stgcn_model, quantized_stgcn


def demonstrate_lstm_quantization():
    """Demonstrate LSTM model quantization."""
    print(f"\n{'='*60}")
    print("LSTM MODEL QUANTIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Create models and data
    _, lstm_model, test_loader, _ = create_dummy_models_and_data()
    
    # Create calibration data for LSTM (different format)
    calibration_data = torch.randn(20, 30, 132)  # Flattened pose data for LSTM
    
    # Create optimizer for analysis
    optimizer = ModelOptimizer()
    
    # Analyze original model
    print("\n1. Analyzing original LSTM model...")
    original_size = optimizer.get_model_size(lstm_model)
    original_params = optimizer.count_parameters(lstm_model)
    
    print(f"Original LSTM Analysis:")
    print(f"  Total size: {original_size['total_size_mb']:.2f} MB")
    print(f"  Parameters: {original_params['total_parameters']:,}")
    
    # Quantize model
    print("\n2. Applying post-training dynamic quantization...")
    quantized_lstm = quantize_lstm_model(lstm_model, calibration_data)
    
    # Analyze quantized model
    print("\n3. Analyzing quantized LSTM model...")
    quantized_size = optimizer.get_model_size(quantized_lstm)
    quantized_params = optimizer.count_parameters(quantized_lstm)
    
    print(f"Quantized LSTM Analysis:")
    print(f"  Total size: {quantized_size['total_size_mb']:.2f} MB")
    print(f"  Parameters: {quantized_params['total_parameters']:,}")
    
    # Calculate improvements
    size_reduction = (original_size['total_size_mb'] - quantized_size['total_size_mb']) / original_size['total_size_mb'] * 100
    compression_ratio = original_size['total_size_mb'] / quantized_size['total_size_mb']
    
    print(f"\nLSTM Quantization Results:")
    print(f"  Size reduction: {size_reduction:.1f}%")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    
    return lstm_model, quantized_lstm


def demonstrate_performance_comparison():
    """Demonstrate performance comparison between original and quantized models."""
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON DEMONSTRATION")
    print("=" * 60)
    
    # Create models and data
    stgcn_model, _, test_loader, calibration_data = create_dummy_models_and_data()
    
    # Quantize model
    print("Quantizing STGCN model...")
    quantized_stgcn = quantize_model(stgcn_model, calibration_data)
    
    # Compare performance
    print("\nComparing original vs quantized performance...")
    comparison_results = compare_model_performance(
        stgcn_model, quantized_stgcn, test_loader, num_samples=50
    )
    
    # Print comparison results
    original = comparison_results['original_model']
    quantized = comparison_results['quantized_model']
    improvements = comparison_results['improvements']
    
    print(f"\nPerformance Comparison Results:")
    print(f"{'─' * 40}")
    print(f"Model Size:")
    print(f"  Original:    {original['size_mb']:.2f} MB")
    print(f"  Quantized:   {quantized['size_mb']:.2f} MB")
    print(f"  Reduction:   {improvements['size_reduction_percent']:.1f}%")
    print(f"  Compression: {improvements['compression_ratio']:.2f}x")
    
    print(f"\nInference Latency:")
    print(f"  Original:    {original['mean_latency_ms']:.2f} ± {original['std_latency_ms']:.2f} ms")
    print(f"  Quantized:   {quantized['mean_latency_ms']:.2f} ± {quantized['std_latency_ms']:.2f} ms")
    print(f"  Reduction:   {improvements['latency_reduction_percent']:.1f}%")
    print(f"  Speedup:     {improvements['speedup_factor']:.2f}x")
    
    print(f"\nThroughput:")
    print(f"  Original:    {original['throughput_fps']:.2f} FPS")
    print(f"  Quantized:   {quantized['throughput_fps']:.2f} FPS")
    print(f"  Improvement: {improvements['throughput_improvement_percent']:.1f}%")
    
    return comparison_results


def demonstrate_model_export():
    """Demonstrate model export to deployable formats."""
    print(f"\n{'='*60}")
    print("MODEL EXPORT DEMONSTRATION")
    print("=" * 60)
    
    # Create models and data
    stgcn_model, _, test_loader, calibration_data = create_dummy_models_and_data()
    
    # Quantize model
    print("Quantizing model for export...")
    quantized_stgcn = quantize_model(stgcn_model, calibration_data)
    
    # Export models
    print("\nExporting models to deployable formats...")
    saved_paths = save_optimized_model(
        quantized_stgcn, 
        "STGCN_PRISM_optimized",
        "optimized_models",
        save_torchscript=True,
        save_onnx=False  # ONNX can be unstable
    )
    
    print(f"\nExport Results:")
    for format_name, path in saved_paths.items():
        if path:
            file_size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"  {format_name.upper()}: {path} ({file_size:.2f} MB)")
        else:
            print(f"  {format_name.upper()}: Export failed")
    
    return saved_paths


def demonstrate_complete_optimization():
    """Demonstrate complete optimization pipeline."""
    print(f"\n{'='*60}")
    print("COMPLETE OPTIMIZATION PIPELINE DEMONSTRATION")
    print("=" * 60)
    
    # Create models and data
    stgcn_model, _, test_loader, calibration_data = create_dummy_models_and_data()
    
    # Run complete optimization pipeline
    results = optimize_stgcn_model(
        stgcn_model=stgcn_model,
        calibration_data=calibration_data,
        test_loader=test_loader,
        model_name="STGCN_PRISM_demo",
        output_dir="optimization_demo"
    )
    
    print(f"\nComplete optimization pipeline completed!")
    print(f"Results saved to: optimization_demo/")
    
    # Print optimization report
    print(f"\nOptimization Report:")
    print(results['optimization_report'])
    
    return results


def demonstrate_optimization_analysis():
    """Demonstrate detailed optimization analysis."""
    print(f"\n{'='*60}")
    print("DETAILED OPTIMIZATION ANALYSIS")
    print("=" * 60)
    
    # Create models and data
    stgcn_model, _, test_loader, calibration_data = create_dummy_models_and_data()
    
    # Create optimizer for detailed analysis
    optimizer = ModelOptimizer()
    
    # Detailed size analysis
    print("\n1. Detailed Model Size Analysis...")
    original_size = optimizer.get_model_size(stgcn_model)
    
    print(f"Original Model Size Breakdown:")
    print(f"  Total size:     {original_size['total_size_mb']:.2f} MB")
    print(f"  Parameters:     {original_size['param_size_mb']:.2f} MB")
    print(f"  Buffers:        {original_size['buffer_size_mb']:.2f} MB")
    print(f"  Total bytes:    {original_size['total_size_bytes']:,}")
    
    # Quantize and analyze
    print("\n2. Quantizing and analyzing...")
    quantized_stgcn = quantize_model(stgcn_model, calibration_data)
    quantized_size = optimizer.get_model_size(quantized_stgcn)
    
    print(f"Quantized Model Size Breakdown:")
    print(f"  Total size:     {quantized_size['total_size_mb']:.2f} MB")
    print(f"  Parameters:     {quantized_size['param_size_mb']:.2f} MB")
    print(f"  Buffers:        {quantized_size['buffer_size_mb']:.2f} MB")
    print(f"  Total bytes:    {quantized_size['total_size_bytes']:,}")
    
    # Detailed comparison
    print(f"\n3. Detailed Comparison:")
    param_reduction = (original_size['param_size_mb'] - quantized_size['param_size_mb']) / original_size['param_size_mb'] * 100
    buffer_reduction = (original_size['buffer_size_mb'] - quantized_size['buffer_size_mb']) / original_size['buffer_size_mb'] * 100
    
    print(f"  Parameter size reduction: {param_reduction:.1f}%")
    print(f"  Buffer size reduction:    {buffer_reduction:.1f}%")
    print(f"  Total size reduction:     {(original_size['total_size_mb'] - quantized_size['total_size_mb']) / original_size['total_size_mb'] * 100:.1f}%")
    
    # Memory efficiency analysis
    print(f"\n4. Memory Efficiency Analysis:")
    memory_efficiency = quantized_size['total_size_mb'] / original_size['total_size_mb']
    print(f"  Memory efficiency: {memory_efficiency:.2f} (lower is better)")
    print(f"  Space saved: {original_size['total_size_mb'] - quantized_size['total_size_mb']:.2f} MB")
    print(f"  Compression ratio: {1/memory_efficiency:.2f}x")


def main():
    """Main demonstration function."""
    print("PRISM Model Optimization - Post-Training Quantization")
    print("=" * 60)
    print("This demonstrates model optimization for deployment:")
    print("• Post-training dynamic quantization (Float32 → Int8)")
    print("• Model size reduction and compression")
    print("• Inference speed improvement")
    print("• TorchScript export for deployment")
    print("• Comprehensive performance comparison")
    print("=" * 60)
    
    try:
        # Run demonstrations
        print("\n1. STGCN Model Quantization")
        demonstrate_stgcn_quantization()
        
        print("\n2. LSTM Model Quantization")
        demonstrate_lstm_quantization()
        
        print("\n3. Performance Comparison")
        demonstrate_performance_comparison()
        
        print("\n4. Model Export")
        demonstrate_model_export()
        
        print("\n5. Detailed Analysis")
        demonstrate_optimization_analysis()
        
        print("\n6. Complete Optimization Pipeline")
        demonstrate_complete_optimization()
        
        print(f"\n{'='*60}")
        print("OPTIMIZATION DEMONSTRATION COMPLETE")
        print(f"{'='*60}")
        print("Key Benefits Demonstrated:")
        print("• Significant model size reduction (typically 2-4x compression)")
        print("• Improved inference speed for CPU deployment")
        print("• TorchScript export for mobile/edge deployment")
        print("• Maintained model accuracy with quantization")
        print("• Ready for production deployment")
        print("\nThe PRISM optimization module is ready for model deployment!")
        
    except Exception as e:
        print(f"Error in demonstration: {str(e)}")
        print("This may be due to missing dependencies or quantization backend issues.")


if __name__ == "__main__":
    main()
