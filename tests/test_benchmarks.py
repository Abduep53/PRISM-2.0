"""
Unit tests for benchmarks module.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from benchmarks import (
    evaluate_classification_metrics,
    measure_inference_latency,
    statistical_significance_test,
    generate_comparison_table
)


class TestBenchmarks(unittest.TestCase):
    """Test cases for benchmarks module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_samples = 100
        self.num_classes = 10
        
        # Create dummy data
        self.y_true = np.random.randint(0, self.num_classes, self.num_samples)
        self.y_pred = np.random.randint(0, self.num_classes, self.num_samples)
        
        # Create dummy model
        self.sample_model = nn.Linear(10, self.num_classes)
        
        # Create dummy data loader
        X = torch.randn(self.num_samples, 10)
        y = torch.randint(0, self.num_classes, (self.num_samples,))
        dataset = TensorDataset(X, y)
        self.data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
    def test_evaluate_classification_metrics(self):
        """Test classification metrics evaluation."""
        metrics = evaluate_classification_metrics(
            self.y_true, 
            self.y_pred, 
            self.num_classes
        )
        
        # Check that all expected metrics are present
        expected_metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 
                          'recall_weighted', 'cohen_kappa']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertGreaterEqual(metrics[metric], 0.0)
            self.assertLessEqual(metrics[metric], 1.0)
            
    def test_measure_inference_latency(self):
        """Test inference latency measurement."""
        latency_results = measure_inference_latency(
            self.sample_model,
            self.data_loader,
            device='cpu',
            num_warmup=2,
            num_measurements=5
        )
        
        # Check that latency results are present
        expected_keys = ['mean_latency', 'std_latency', 'p95_latency', 'throughput']
        for key in expected_keys:
            self.assertIn(key, latency_results)
            self.assertIsInstance(latency_results[key], (int, float))
            self.assertGreater(latency_results[key], 0)
            
    def test_statistical_significance_test(self):
        """Test statistical significance testing."""
        # Create two sets of results
        results1 = np.random.normal(0.8, 0.1, 10)
        results2 = np.random.normal(0.85, 0.1, 10)
        
        significance_results = statistical_significance_test(
            results1, 
            results2, 
            alpha=0.05
        )
        
        # Check that significance results are present
        expected_keys = ['t_statistic', 'p_value', 'is_significant', 'effect_size']
        for key in expected_keys:
            self.assertIn(key, significance_results)
            
        self.assertIsInstance(significance_results['is_significant'], bool)
        self.assertIsInstance(significance_results['p_value'], (int, float))
        
    def test_generate_comparison_table(self):
        """Test comparison table generation."""
        # Create dummy results
        lstm_results = {
            'accuracy': 0.82,
            'f1_weighted': 0.81,
            'mean_latency': 12.3,
            'model_size': 15.2
        }
        
        stgcn_results = {
            'accuracy': 0.89,
            'f1_weighted': 0.87,
            'mean_latency': 18.7,
            'model_size': 23.8
        }
        
        dp_results = {
            'accuracy': 0.85,
            'f1_weighted': 0.84,
            'mean_latency': 19.2,
            'model_size': 23.8,
            'privacy_epsilon': 1.0
        }
        
        comparison_table = generate_comparison_table(
            lstm_results, 
            stgcn_results, 
            dp_results
        )
        
        # Check that table has expected structure
        self.assertIsInstance(comparison_table, dict)
        self.assertIn('data', comparison_table)
        self.assertIn('columns', comparison_table)
        
        # Check that all models are present
        model_names = [row['Model'] for row in comparison_table['data']]
        self.assertIn('LSTM Baseline', model_names)
        self.assertIn('STGCN (No DP)', model_names)
        self.assertIn('STGCN (With DP)', model_names)


if __name__ == '__main__':
    unittest.main()
