"""
Benchmark Example for PRISM Models
Demonstrates comprehensive evaluation of all model variants.
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

from models import BaselineLSTM, STGCN_PRISM
from benchmarks import (
    run_comprehensive_benchmark,
    evaluate_baseline_lstm,
    evaluate_stgcn_no_dp,
    evaluate_stgcn_with_dp,
    generate_comparison_table,
    ModelEvaluator
)


def create_dummy_models_and_data():
    """Create dummy models and data for demonstration."""
    print("Creating dummy models and test data...")
    
    # Model parameters
    num_classes = 101
    sequence_length = 30
    num_samples = 200
    batch_size = 16
    
    # Create dummy test data
    if True:  # Use kinematic features
        num_features = 100  # 50 static + 50 velocity features
        test_data = np.random.randn(num_samples, sequence_length, num_features)
    else:  # Use raw pose data
        num_joints = 33
        num_coords = 4
        test_data = np.random.randn(num_samples, sequence_length, num_joints, num_coords)
    
    # Generate labels
    test_labels = np.random.randint(0, num_classes, num_samples)
    
    # Create dataset and data loader
    test_dataset = TensorDataset(
        torch.FloatTensor(test_data),
        torch.LongTensor(test_labels)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✓ Created test dataset:")
    print(f"  Samples: {num_samples}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Features per frame: {test_data.shape[-1] if len(test_data.shape) == 3 else test_data.shape[-2] * test_data.shape[-1]}")
    print(f"  Number of classes: {num_classes}")
    
    # Create dummy models
    print(f"\nCreating dummy models...")
    
    # LSTM Baseline
    lstm_model = BaselineLSTM(
        input_size=test_data.shape[-1] if len(test_data.shape) == 3 else test_data.shape[-2] * test_data.shape[-1],
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
        bidirectional=False
    )
    
    # STGCN (No DP)
    stgcn_no_dp_model = STGCN_PRISM(
        num_joints=33,
        in_channels=4,
        num_classes=num_classes,
        hidden_channels=[64, 128, 256],
        temporal_kernel_sizes=[3, 3, 3],
        dropout=0.1,
        use_attention=False
    )
    
    # STGCN (With DP) - same architecture as no DP
    stgcn_with_dp_model = STGCN_PRISM(
        num_joints=33,
        in_channels=4,
        num_classes=num_classes,
        hidden_channels=[64, 128, 256],
        temporal_kernel_sizes=[3, 3, 3],
        dropout=0.1,
        use_attention=False
    )
    
    # Initialize models with dummy weights
    for model in [lstm_model, stgcn_no_dp_model, stgcn_with_dp_model]:
        for param in model.parameters():
            nn.init.normal_(param, 0, 0.01)
    
    print(f"✓ Created models:")
    print(f"  LSTM parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
    print(f"  STGCN parameters: {sum(p.numel() for p in stgcn_no_dp_model.parameters()):,}")
    
    return lstm_model, stgcn_no_dp_model, stgcn_with_dp_model, test_loader


def demonstrate_individual_evaluations():
    """Demonstrate individual model evaluations."""
    print("=" * 60)
    print("INDIVIDUAL MODEL EVALUATIONS")
    print("=" * 60)
    
    # Create models and data
    lstm_model, stgcn_no_dp_model, stgcn_with_dp_model, test_loader = create_dummy_models_and_data()
    
    # Class names for detailed reporting
    class_names = [f"Action_{i:02d}" for i in range(101)]
    
    # Evaluate LSTM Baseline
    print(f"\n1. Evaluating LSTM Baseline...")
    lstm_metrics = evaluate_baseline_lstm(lstm_model, test_loader, class_names)
    
    print(f"LSTM Results:")
    print(f"  Accuracy: {lstm_metrics['accuracy']:.4f}")
    print(f"  F1-Score (Weighted): {lstm_metrics['f1_weighted']:.4f}")
    print(f"  Mean Latency: {lstm_metrics['mean_latency_ms']:.2f} ms")
    print(f"  Throughput: {lstm_metrics['throughput_fps']:.2f} FPS")
    
    # Evaluate STGCN (No DP)
    print(f"\n2. Evaluating STGCN (No DP)...")
    stgcn_no_dp_metrics = evaluate_stgcn_no_dp(stgcn_no_dp_model, test_loader, class_names)
    
    print(f"STGCN (No DP) Results:")
    print(f"  Accuracy: {stgcn_no_dp_metrics['accuracy']:.4f}")
    print(f"  F1-Score (Weighted): {stgcn_no_dp_metrics['f1_weighted']:.4f}")
    print(f"  Mean Latency: {stgcn_no_dp_metrics['mean_latency_ms']:.2f} ms")
    print(f"  Throughput: {stgcn_no_dp_metrics['throughput_fps']:.2f} FPS")
    
    # Evaluate STGCN (With DP)
    print(f"\n3. Evaluating STGCN (With DP)...")
    privacy_epsilon = 1.0
    privacy_delta = 1e-5
    
    stgcn_with_dp_metrics = evaluate_stgcn_with_dp(
        stgcn_with_dp_model, test_loader, privacy_epsilon, privacy_delta, class_names
    )
    
    print(f"STGCN (With DP) Results:")
    print(f"  Accuracy: {stgcn_with_dp_metrics['accuracy']:.4f}")
    print(f"  F1-Score (Weighted): {stgcn_with_dp_metrics['f1_weighted']:.4f}")
    print(f"  Mean Latency: {stgcn_with_dp_metrics['mean_latency_ms']:.2f} ms")
    print(f"  Throughput: {stgcn_with_dp_metrics['throughput_fps']:.2f} FPS")
    print(f"  Privacy ε: {stgcn_with_dp_metrics['privacy_epsilon']}")
    
    return lstm_metrics, stgcn_no_dp_metrics, stgcn_with_dp_metrics


def demonstrate_comparison_table():
    """Demonstrate comparison table generation."""
    print(f"\n{'='*60}")
    print("COMPARISON TABLE GENERATION")
    print("=" * 60)
    
    # Get metrics from individual evaluations
    lstm_metrics, stgcn_no_dp_metrics, stgcn_with_dp_metrics = demonstrate_individual_evaluations()
    
    # Generate comparison table
    print(f"\nGenerating comparison table...")
    comparison_table = generate_comparison_table(
        lstm_metrics, stgcn_no_dp_metrics, stgcn_with_dp_metrics,
        save_path="benchmark_comparison.csv"
    )
    
    print(f"\nComparison Table:")
    print(comparison_table.to_string(index=False))
    
    return comparison_table


def demonstrate_comprehensive_benchmark():
    """Demonstrate comprehensive benchmark evaluation."""
    print(f"\n{'='*60}")
    print("COMPREHENSIVE BENCHMARK EVALUATION")
    print("=" * 60)
    
    # Create models and data
    lstm_model, stgcn_no_dp_model, stgcn_with_dp_model, test_loader = create_dummy_models_and_data()
    
    # Class names
    class_names = [f"Action_{i:02d}" for i in range(101)]
    
    # Privacy parameters
    privacy_epsilon = 1.0
    privacy_delta = 1e-5
    
    # Run comprehensive benchmark
    results = run_comprehensive_benchmark(
        lstm_model=lstm_model,
        stgcn_no_dp_model=stgcn_no_dp_model,
        stgcn_with_dp_model=stgcn_with_dp_model,
        test_loader=test_loader,
        privacy_epsilon=privacy_epsilon,
        privacy_delta=privacy_delta,
        class_names=class_names,
        output_dir="benchmark_results"
    )
    
    print(f"\nComprehensive benchmark completed!")
    print(f"Results saved to: benchmark_results/")
    
    return results


def demonstrate_advanced_metrics():
    """Demonstrate advanced metrics and analysis."""
    print(f"\n{'='*60}")
    print("ADVANCED METRICS AND ANALYSIS")
    print("=" * 60)
    
    # Create models and data
    lstm_model, stgcn_no_dp_model, stgcn_with_dp_model, test_loader = create_dummy_models_and_data()
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate with detailed metrics
    print("Evaluating with detailed classification metrics...")
    detailed_metrics = evaluator.evaluate_classification_metrics(
        lstm_model, test_loader, class_names=[f"Action_{i:02d}" for i in range(101)]
    )
    
    print(f"\nDetailed Classification Metrics:")
    print(f"  Accuracy: {detailed_metrics['accuracy']:.4f}")
    print(f"  Precision (Weighted): {detailed_metrics['precision_weighted']:.4f}")
    print(f"  Recall (Weighted): {detailed_metrics['recall_weighted']:.4f}")
    print(f"  F1-Score (Weighted): {detailed_metrics['f1_weighted']:.4f}")
    print(f"  F1-Score (Macro): {detailed_metrics['f1_macro']:.4f}")
    print(f"  Cohen's Kappa: {detailed_metrics['kappa']:.4f}")
    
    # Measure detailed latency
    print(f"\nMeasuring detailed inference latency...")
    latency_metrics = evaluator.measure_inference_latency(
        lstm_model, test_loader, num_warmup=5, num_iterations=50
    )
    
    print(f"\nDetailed Latency Metrics:")
    print(f"  Mean Latency: {latency_metrics['mean_latency_ms']:.2f} ms")
    print(f"  Std Latency: {latency_metrics['std_latency_ms']:.2f} ms")
    print(f"  P95 Latency: {latency_metrics['p95_latency_ms']:.2f} ms")
    print(f"  P99 Latency: {latency_metrics['p99_latency_ms']:.2f} ms")
    print(f"  Throughput: {latency_metrics['throughput_fps']:.2f} FPS")


def main():
    """Main demonstration function."""
    print("PRISM Benchmarks - Comprehensive Model Evaluation")
    print("=" * 60)
    print("This demonstrates rigorous evaluation for scientific paper:")
    print("• F1-Score and Classification Accuracy across 101 classes")
    print("• Inference Latency measurements (ms per frame)")
    print("• Statistical significance testing")
    print("• Privacy-utility tradeoff analysis")
    print("• Comprehensive comparison tables")
    print("=" * 60)
    
    try:
        # Run demonstrations
        print("\n1. Individual Model Evaluations")
        demonstrate_individual_evaluations()
        
        print("\n2. Comparison Table Generation")
        demonstrate_comparison_table()
        
        print("\n3. Advanced Metrics Analysis")
        demonstrate_advanced_metrics()
        
        print("\n4. Comprehensive Benchmark (Full Evaluation)")
        demonstrate_comprehensive_benchmark()
        
        print(f"\n{'='*60}")
        print("BENCHMARK DEMONSTRATION COMPLETE")
        print(f"{'='*60}")
        print("Key Features Demonstrated:")
        print("• Comprehensive classification metrics (Accuracy, F1-Score, etc.)")
        print("• Detailed inference latency measurements")
        print("• Statistical significance testing between models")
        print("• Privacy-utility tradeoff analysis")
        print("• Publication-ready comparison tables")
        print("• Visualization plots for scientific papers")
        print("\nThe PRISM benchmarks module is ready for scientific evaluation!")
        
    except Exception as e:
        print(f"Error in demonstration: {str(e)}")
        print("This may be due to missing dependencies or configuration issues.")


if __name__ == "__main__":
    main()
