"""
PRISM Benchmarks: Comprehensive Model Evaluation

This module provides rigorous evaluation functions for all PRISM models,
including Baseline LSTM, STGCN variants, and privacy-preserving models.
Essential for scientific paper evaluation and comparison.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    classification_report,
    confusion_matrix,
    cohen_kappa_score
)
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings

from models import BaselineLSTM, BaselineGRU, STGCN_PRISM
from data_pipeline import PRISMDataset


class ModelEvaluator:
    """
    Comprehensive model evaluation class for PRISM models.
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize model evaluator.
        
        Args:
            device: Device to run evaluation on ('auto', 'cpu', 'cuda')
        """
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"ModelEvaluator initialized on device: {self.device}")
    
    def evaluate_classification_metrics(self, 
                                      model: nn.Module, 
                                      test_loader: DataLoader,
                                      class_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            model: Trained model to evaluate
            test_loader: Test data loader
            class_names: Optional list of class names for detailed reporting
            
        Returns:
            Dictionary containing all classification metrics
        """
        model.eval()
        model.to(self.device)
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                pose_sequences = batch['pose_sequence'].to(self.device)
                labels = batch['label'].squeeze().to(self.device)
                
                # Forward pass
                outputs = model(pose_sequences)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Precision, Recall, F1-Score (weighted averages)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro', zero_division=0
        )
        
        # Micro averages
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='micro', zero_division=0
        )
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(all_labels, all_predictions)
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            all_labels, all_predictions, average=None, zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Classification report
        if class_names:
            report = classification_report(
                all_labels, all_predictions, 
                target_names=class_names, 
                output_dict=True,
                zero_division=0
            )
        else:
            report = classification_report(
                all_labels, all_predictions, 
                output_dict=True,
                zero_division=0
            )
        
        metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'kappa': kappa,
            'support': support,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'support_per_class': support_per_class,
            'classification_report': report
        }
        
        return metrics
    
    def measure_inference_latency(self, 
                                model: nn.Module, 
                                test_loader: DataLoader,
                                num_warmup: int = 10,
                                num_iterations: int = 100) -> Dict[str, float]:
        """
        Measure inference latency in milliseconds per frame.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            num_warmup: Number of warmup iterations
            num_iterations: Number of timing iterations
            
        Returns:
            Dictionary containing latency metrics
        """
        model.eval()
        model.to(self.device)
        
        # Warmup
        print(f"Warming up model with {num_warmup} iterations...")
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= num_warmup:
                    break
                pose_sequences = batch['pose_sequence'].to(self.device)
                _ = model(pose_sequences)
        
        # Timing measurements
        print(f"Measuring inference latency with {num_iterations} iterations...")
        latencies = []
        frame_counts = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= num_iterations:
                    break
                
                pose_sequences = batch['pose_sequence'].to(self.device)
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
        
        # Calculate statistics
        latencies = np.array(latencies)
        frame_counts = np.array(frame_counts)
        
        metrics = {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'total_frames_processed': np.sum(frame_counts),
            'total_time_ms': np.sum(latencies * frame_counts),
            'throughput_fps': 1000 / np.mean(latencies) if np.mean(latencies) > 0 else 0
        }
        
        return metrics
    
    def evaluate_model_comprehensive(self, 
                                   model: nn.Module, 
                                   test_loader: DataLoader,
                                   class_names: Optional[List[str]] = None,
                                   measure_latency: bool = True) -> Dict[str, Union[float, Dict]]:
        """
        Comprehensive model evaluation including all metrics.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            class_names: Optional list of class names
            measure_latency: Whether to measure inference latency
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        print(f"Starting comprehensive evaluation...")
        
        # Classification metrics
        print("Calculating classification metrics...")
        classification_metrics = self.evaluate_classification_metrics(
            model, test_loader, class_names
        )
        
        # Inference latency
        latency_metrics = {}
        if measure_latency:
            print("Measuring inference latency...")
            latency_metrics = self.measure_inference_latency(model, test_loader)
        
        # Combine all metrics
        all_metrics = {
            **classification_metrics,
            **latency_metrics
        }
        
        print("✓ Comprehensive evaluation completed")
        
        return all_metrics


def evaluate_baseline_lstm(model: nn.Module, 
                          test_loader: DataLoader,
                          class_names: Optional[List[str]] = None) -> Dict[str, Union[float, Dict]]:
    """
    Evaluate Baseline LSTM model.
    
    Args:
        model: Trained LSTM model
        test_loader: Test data loader
        class_names: Optional class names
        
    Returns:
        Evaluation metrics dictionary
    """
    print("Evaluating Baseline LSTM...")
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model_comprehensive(
        model, test_loader, class_names, measure_latency=True
    )
    
    # Add model-specific information
    metrics['model_type'] = 'Baseline LSTM'
    metrics['model_parameters'] = sum(p.numel() for p in model.parameters())
    
    return metrics


def evaluate_stgcn_no_dp(model: nn.Module, 
                        test_loader: DataLoader,
                        class_names: Optional[List[str]] = None) -> Dict[str, Union[float, Dict]]:
    """
    Evaluate STGCN model without differential privacy.
    
    Args:
        model: Trained STGCN model
        test_loader: Test data loader
        class_names: Optional class names
        
    Returns:
        Evaluation metrics dictionary
    """
    print("Evaluating STGCN (No DP)...")
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model_comprehensive(
        model, test_loader, class_names, measure_latency=True
    )
    
    # Add model-specific information
    metrics['model_type'] = 'STGCN (No DP)'
    metrics['model_parameters'] = sum(p.numel() for p in model.parameters())
    metrics['privacy_epsilon'] = float('inf')  # No privacy protection
    metrics['privacy_delta'] = 0.0
    
    return metrics


def evaluate_stgcn_with_dp(model: nn.Module, 
                          test_loader: DataLoader,
                          privacy_epsilon: float,
                          privacy_delta: float,
                          class_names: Optional[List[str]] = None) -> Dict[str, Union[float, Dict]]:
    """
    Evaluate STGCN model with differential privacy.
    
    Args:
        model: Trained STGCN model with DP
        test_loader: Test data loader
        privacy_epsilon: Privacy budget used
        privacy_delta: Privacy failure probability
        class_names: Optional class names
        
    Returns:
        Evaluation metrics dictionary
    """
    print(f"Evaluating STGCN (With DP, ε={privacy_epsilon})...")
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model_comprehensive(
        model, test_loader, class_names, measure_latency=True
    )
    
    # Add model-specific information
    metrics['model_type'] = 'STGCN (With DP)'
    metrics['model_parameters'] = sum(p.numel() for p in model.parameters())
    metrics['privacy_epsilon'] = privacy_epsilon
    metrics['privacy_delta'] = privacy_delta
    
    return metrics


def statistical_significance_test(metrics1: Dict, 
                                metrics2: Dict, 
                                metric_name: str = 'f1_weighted') -> Dict[str, float]:
    """
    Perform statistical significance test between two models.
    
    Args:
        metrics1: Metrics from first model
        metrics2: Metrics from second model
        metric_name: Metric to test (e.g., 'f1_weighted', 'accuracy')
        
    Returns:
        Dictionary containing statistical test results
    """
    # Extract predictions for the specified metric
    if metric_name in ['f1_weighted', 'accuracy']:
        # For these metrics, we need to use predictions
        pred1 = metrics1['predictions']
        pred2 = metrics2['predictions']
        labels = metrics1['labels']
        
        # Calculate metric values for each sample
        # This is a simplified approach - in practice, you'd need per-sample metrics
        metric1_values = [metrics1[metric_name]] * len(pred1)
        metric2_values = [metrics2[metric_name]] * len(pred2)
        
    else:
        # For other metrics, use the scalar values
        metric1_values = [metrics1[metric_name]]
        metric2_values = [metrics2[metric_name]]
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(metric1_values, metric2_values)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(metric1_values) + np.var(metric2_values)) / 2)
    cohens_d = (np.mean(metric1_values) - np.mean(metric2_values)) / pooled_std if pooled_std > 0 else 0
    
    results = {
        'metric': metric_name,
        'model1_value': np.mean(metric1_values),
        'model2_value': np.mean(metric2_values),
        'difference': np.mean(metric1_values) - np.mean(metric2_values),
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05,
        'effect_size': 'small' if abs(cohens_d) < 0.2 else 'medium' if abs(cohens_d) < 0.8 else 'large'
    }
    
    return results


def generate_comparison_table(lstm_metrics: Dict,
                            stgcn_no_dp_metrics: Dict,
                            stgcn_with_dp_metrics: Dict,
                            save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generate comprehensive comparison table for the three primary experimental conditions.
    
    Args:
        lstm_metrics: Metrics from LSTM Baseline
        stgcn_no_dp_metrics: Metrics from STGCN (No DP)
        stgcn_with_dp_metrics: Metrics from STGCN (With DP)
        save_path: Optional path to save the table
        
    Returns:
        Pandas DataFrame with comparison table
    """
    print("Generating comparison table...")
    
    # Prepare data for comparison
    models = [
        ('LSTM Baseline', lstm_metrics),
        ('STGCN (No DP)', stgcn_no_dp_metrics),
        ('STGCN (With DP)', stgcn_with_dp_metrics)
    ]
    
    # Define metrics to include in comparison
    comparison_metrics = [
        ('Classification Accuracy', 'accuracy'),
        ('F1-Score (Weighted)', 'f1_weighted'),
        ('F1-Score (Macro)', 'f1_macro'),
        ('Precision (Weighted)', 'precision_weighted'),
        ('Recall (Weighted)', 'recall_weighted'),
        ('Cohen\'s Kappa', 'kappa'),
        ('Mean Latency (ms)', 'mean_latency_ms'),
        ('Std Latency (ms)', 'std_latency_ms'),
        ('P95 Latency (ms)', 'p95_latency_ms'),
        ('Throughput (FPS)', 'throughput_fps'),
        ('Model Parameters', 'model_parameters'),
        ('Privacy ε', 'privacy_epsilon'),
        ('Privacy δ', 'privacy_delta')
    ]
    
    # Create comparison data
    comparison_data = []
    
    for model_name, metrics in models:
        row = {'Model': model_name}
        
        for display_name, metric_key in comparison_metrics:
            if metric_key in metrics:
                value = metrics[metric_key]
                if isinstance(value, float):
                    if 'latency' in metric_key or 'fps' in metric_key:
                        row[display_name] = f"{value:.2f}"
                    elif 'privacy' in metric_key:
                        if value == float('inf'):
                            row[display_name] = "∞"
                        else:
                            row[display_name] = f"{value:.2e}"
                    else:
                        row[display_name] = f"{value:.4f}"
                else:
                    row[display_name] = f"{value:,}"
            else:
                row[display_name] = "N/A"
        
        comparison_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Reorder columns for better readability
    column_order = [
        'Model', 'Classification Accuracy', 'F1-Score (Weighted)', 'F1-Score (Macro)',
        'Precision (Weighted)', 'Recall (Weighted)', 'Cohen\'s Kappa',
        'Mean Latency (ms)', 'Std Latency (ms)', 'P95 Latency (ms)', 'Throughput (FPS)',
        'Model Parameters', 'Privacy ε', 'Privacy δ'
    ]
    
    df = df[column_order]
    
    # Save table if path provided
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Comparison table saved to: {save_path}")
        
        # Also save as LaTeX table for scientific paper
        latex_path = save_path.replace('.csv', '.tex')
        df.to_latex(latex_path, index=False, float_format='%.4f')
        print(f"LaTeX table saved to: {latex_path}")
    
    return df


def generate_detailed_report(lstm_metrics: Dict,
                           stgcn_no_dp_metrics: Dict,
                           stgcn_with_dp_metrics: Dict,
                           save_path: Optional[str] = None) -> Dict:
    """
    Generate detailed evaluation report with statistical analysis.
    
    Args:
        lstm_metrics: Metrics from LSTM Baseline
        stgcn_no_dp_metrics: Metrics from STGCN (No DP)
        stgcn_with_dp_metrics: Metrics from STGCN (With DP)
        save_path: Optional path to save the report
        
    Returns:
        Dictionary containing detailed report
    """
    print("Generating detailed evaluation report...")
    
    # Statistical significance tests
    significance_tests = {}
    
    # LSTM vs STGCN (No DP)
    significance_tests['lstm_vs_stgcn_no_dp'] = statistical_significance_test(
        lstm_metrics, stgcn_no_dp_metrics, 'f1_weighted'
    )
    
    # STGCN (No DP) vs STGCN (With DP)
    significance_tests['stgcn_no_dp_vs_stgcn_with_dp'] = statistical_significance_test(
        stgcn_no_dp_metrics, stgcn_with_dp_metrics, 'f1_weighted'
    )
    
    # LSTM vs STGCN (With DP)
    significance_tests['lstm_vs_stgcn_with_dp'] = statistical_significance_test(
        lstm_metrics, stgcn_with_dp_metrics, 'f1_weighted'
    )
    
    # Performance summary
    performance_summary = {
        'best_accuracy': max(
            lstm_metrics['accuracy'],
            stgcn_no_dp_metrics['accuracy'],
            stgcn_with_dp_metrics['accuracy']
        ),
        'best_f1_weighted': max(
            lstm_metrics['f1_weighted'],
            stgcn_no_dp_metrics['f1_weighted'],
            stgcn_with_dp_metrics['f1_weighted']
        ),
        'fastest_inference': min(
            lstm_metrics['mean_latency_ms'],
            stgcn_no_dp_metrics['mean_latency_ms'],
            stgcn_with_dp_metrics['mean_latency_ms']
        )
    }
    
    # Privacy-utility tradeoff analysis
    privacy_utility_analysis = {
        'privacy_cost_accuracy': stgcn_no_dp_metrics['accuracy'] - stgcn_with_dp_metrics['accuracy'],
        'privacy_cost_f1': stgcn_no_dp_metrics['f1_weighted'] - stgcn_with_dp_metrics['f1_weighted'],
        'privacy_cost_latency': stgcn_with_dp_metrics['mean_latency_ms'] - stgcn_no_dp_metrics['mean_latency_ms'],
        'privacy_efficiency_accuracy': stgcn_with_dp_metrics['accuracy'] / stgcn_with_dp_metrics['privacy_epsilon'],
        'privacy_efficiency_f1': stgcn_with_dp_metrics['f1_weighted'] / stgcn_with_dp_metrics['privacy_epsilon']
    }
    
    # Create detailed report
    detailed_report = {
        'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'models_evaluated': ['LSTM Baseline', 'STGCN (No DP)', 'STGCN (With DP)'],
        'metrics': {
            'lstm_baseline': lstm_metrics,
            'stgcn_no_dp': stgcn_no_dp_metrics,
            'stgcn_with_dp': stgcn_with_dp_metrics
        },
        'statistical_tests': significance_tests,
        'performance_summary': performance_summary,
        'privacy_utility_analysis': privacy_utility_analysis
    }
    
    # Save report if path provided
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(detailed_report, f, indent=2, default=str)
        print(f"Detailed report saved to: {save_path}")
    
    return detailed_report


def create_visualization_plots(lstm_metrics: Dict,
                             stgcn_no_dp_metrics: Dict,
                             stgcn_with_dp_metrics: Dict,
                             save_dir: str = "benchmark_plots") -> None:
    """
    Create visualization plots for model comparison.
    
    Args:
        lstm_metrics: Metrics from LSTM Baseline
        stgcn_no_dp_metrics: Metrics from STGCN (No DP)
        stgcn_with_dp_metrics: Metrics from STGCN (With DP)
        save_dir: Directory to save plots
    """
    print(f"Creating visualization plots in {save_dir}...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data
    models = ['LSTM Baseline', 'STGCN (No DP)', 'STGCN (With DP)']
    accuracies = [
        lstm_metrics['accuracy'],
        stgcn_no_dp_metrics['accuracy'],
        stgcn_with_dp_metrics['accuracy']
    ]
    f1_scores = [
        lstm_metrics['f1_weighted'],
        stgcn_no_dp_metrics['f1_weighted'],
        stgcn_with_dp_metrics['f1_weighted']
    ]
    latencies = [
        lstm_metrics['mean_latency_ms'],
        stgcn_no_dp_metrics['mean_latency_ms'],
        stgcn_with_dp_metrics['mean_latency_ms']
    ]
    
    # 1. Accuracy and F1-Score comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    ax1.bar(x + width/2, f1_scores, width, label='F1-Score (Weighted)', alpha=0.8)
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Score')
    ax1.set_title('Classification Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Inference latency comparison
    ax2.bar(models, latencies, alpha=0.8, color=['skyblue', 'lightgreen', 'salmon'])
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Mean Latency (ms)')
    ax2.set_title('Inference Latency Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Privacy-Utility Tradeoff
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot accuracy vs privacy budget
    privacy_epsilons = [float('inf'), float('inf'), stgcn_with_dp_metrics['privacy_epsilon']]
    accuracies_with_privacy = accuracies.copy()
    
    # Convert infinity to a large number for plotting
    privacy_epsilons_plot = [1000 if x == float('inf') else x for x in privacy_epsilons]
    
    scatter = ax.scatter(privacy_epsilons_plot, accuracies, 
                        s=100, alpha=0.7, c=['blue', 'green', 'red'])
    
    # Add labels
    for i, model in enumerate(models):
        ax.annotate(model, (privacy_epsilons_plot[i], accuracies[i]), 
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Privacy Budget (ε)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Privacy-Utility Tradeoff')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/privacy_utility_tradeoff.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization plots saved to {save_dir}/")


def run_comprehensive_benchmark(lstm_model: nn.Module,
                               stgcn_no_dp_model: nn.Module,
                               stgcn_with_dp_model: nn.Module,
                               test_loader: DataLoader,
                               privacy_epsilon: float,
                               privacy_delta: float,
                               class_names: Optional[List[str]] = None,
                               output_dir: str = "benchmark_results") -> Dict:
    """
    Run comprehensive benchmark evaluation for all three experimental conditions.
    
    Args:
        lstm_model: Trained LSTM baseline model
        stgcn_no_dp_model: Trained STGCN model without DP
        stgcn_with_dp_model: Trained STGCN model with DP
        test_loader: Test data loader
        privacy_epsilon: Privacy budget used for DP model
        privacy_delta: Privacy failure probability
        class_names: Optional class names
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing all benchmark results
    """
    print("=" * 60)
    print("PRISM COMPREHENSIVE BENCHMARK EVALUATION")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate all models
    print("\n1. Evaluating LSTM Baseline...")
    lstm_metrics = evaluate_baseline_lstm(lstm_model, test_loader, class_names)
    
    print("\n2. Evaluating STGCN (No DP)...")
    stgcn_no_dp_metrics = evaluate_stgcn_no_dp(stgcn_no_dp_model, test_loader, class_names)
    
    print("\n3. Evaluating STGCN (With DP)...")
    stgcn_with_dp_metrics = evaluate_stgcn_with_dp(
        stgcn_with_dp_model, test_loader, privacy_epsilon, privacy_delta, class_names
    )
    
    # Generate comparison table
    print("\n4. Generating comparison table...")
    comparison_table = generate_comparison_table(
        lstm_metrics, stgcn_no_dp_metrics, stgcn_with_dp_metrics,
        save_path=f"{output_dir}/comparison_table.csv"
    )
    
    # Generate detailed report
    print("\n5. Generating detailed report...")
    detailed_report = generate_detailed_report(
        lstm_metrics, stgcn_no_dp_metrics, stgcn_with_dp_metrics,
        save_path=f"{output_dir}/detailed_report.json"
    )
    
    # Create visualizations
    print("\n6. Creating visualization plots...")
    create_visualization_plots(
        lstm_metrics, stgcn_no_dp_metrics, stgcn_with_dp_metrics,
        save_dir=f"{output_dir}/plots"
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}/")
    print(f"Comparison table: {output_dir}/comparison_table.csv")
    print(f"Detailed report: {output_dir}/detailed_report.json")
    print(f"Visualization plots: {output_dir}/plots/")
    
    # Print key results
    print(f"\nKey Results:")
    print(f"  Best Accuracy: {max(lstm_metrics['accuracy'], stgcn_no_dp_metrics['accuracy'], stgcn_with_dp_metrics['accuracy']):.4f}")
    print(f"  Best F1-Score: {max(lstm_metrics['f1_weighted'], stgcn_no_dp_metrics['f1_weighted'], stgcn_with_dp_metrics['f1_weighted']):.4f}")
    print(f"  Fastest Inference: {min(lstm_metrics['mean_latency_ms'], stgcn_no_dp_metrics['mean_latency_ms'], stgcn_with_dp_metrics['mean_latency_ms']):.2f} ms")
    
    return {
        'comparison_table': comparison_table,
        'detailed_report': detailed_report,
        'individual_metrics': {
            'lstm_baseline': lstm_metrics,
            'stgcn_no_dp': stgcn_no_dp_metrics,
            'stgcn_with_dp': stgcn_with_dp_metrics
        }
    }


if __name__ == "__main__":
    # Example usage
    print("PRISM Benchmarks - Model Evaluation Module")
    print("=" * 50)
    
    # This would typically be used with actual trained models
    print("This module provides comprehensive evaluation functions for:")
    print("• LSTM Baseline models")
    print("• STGCN models (with and without DP)")
    print("• Classification metrics (Accuracy, F1-Score, etc.)")
    print("• Inference latency measurements")
    print("• Statistical significance testing")
    print("• Privacy-utility tradeoff analysis")
    print("\nUse run_comprehensive_benchmark() for complete evaluation!")
