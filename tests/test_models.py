"""
Unit tests for models module.
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import (
    BaselineLSTM,
    BaselineGRU,
    STGCN_PRISM,
    get_human_skeleton_adjacency_matrix,
    TemporalConvolutionalLayer,
    GraphConvolutionalLayer,
    STGCNBlock
)


class TestModels(unittest.TestCase):
    """Test cases for model architectures."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.sequence_length = 30
        self.num_joints = 33
        self.num_features = 4
        self.num_classes = 101
        
        # Create dummy input data
        self.sample_input = torch.randn(
            self.batch_size, 
            self.sequence_length, 
            self.num_joints, 
            self.num_features
        )
        
    def test_baseline_lstm_forward(self):
        """Test BaselineLSTM forward pass."""
        model = BaselineLSTM(
            input_size=self.num_joints * self.num_features,
            hidden_size=128,
            num_layers=2,
            num_classes=self.num_classes,
            dropout=0.3
        )
        
        # Reshape input for LSTM
        input_reshaped = self.sample_input.view(
            self.batch_size, 
            self.sequence_length, 
            -1
        )
        
        output = model(input_reshaped)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        
    def test_baseline_gru_forward(self):
        """Test BaselineGRU forward pass."""
        model = BaselineGRU(
            input_size=self.num_joints * self.num_features,
            hidden_size=128,
            num_layers=2,
            num_classes=self.num_classes,
            dropout=0.3
        )
        
        # Reshape input for GRU
        input_reshaped = self.sample_input.view(
            self.batch_size, 
            self.sequence_length, 
            -1
        )
        
        output = model(input_reshaped)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        
    def test_stgcn_prism_forward(self):
        """Test STGCN_PRISM forward pass."""
        model = STGCN_PRISM(
            num_joints=self.num_joints,
            in_channels=self.num_features,
            num_classes=self.num_classes,
            hidden_channels=[64, 128],
            temporal_kernel_sizes=[3, 3],
            dropout=0.1
        )
        
        output = model(self.sample_input)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        
    def test_adjacency_matrix_shape(self):
        """Test adjacency matrix has correct shape."""
        adj_matrix = get_human_skeleton_adjacency_matrix()
        self.assertEqual(adj_matrix.shape, (self.num_joints, self.num_joints))
        self.assertTrue(torch.all(adj_matrix >= 0))  # Non-negative values
        self.assertTrue(torch.all(adj_matrix <= 1))  # Binary or normalized values
        
    def test_temporal_convolutional_layer(self):
        """Test TemporalConvolutionalLayer."""
        layer = TemporalConvolutionalLayer(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            dropout=0.1
        )
        
        # Input: (batch, channels, time)
        input_tensor = torch.randn(self.batch_size, 64, self.sequence_length)
        output = layer(input_tensor)
        
        self.assertEqual(output.shape, (self.batch_size, 128, self.sequence_length))
        
    def test_graph_convolutional_layer(self):
        """Test GraphConvolutionalLayer."""
        adj_matrix = get_human_skeleton_adjacency_matrix()
        layer = GraphConvolutionalLayer(
            in_features=self.num_features,
            out_features=64,
            adjacency_matrix=adj_matrix
        )
        
        # Input: (batch, time, joints, features)
        input_tensor = torch.randn(
            self.batch_size, 
            self.sequence_length, 
            self.num_joints, 
            self.num_features
        )
        output = layer(input_tensor)
        
        self.assertEqual(output.shape, (self.batch_size, self.sequence_length, self.num_joints, 64))
        
    def test_stgcn_block(self):
        """Test STGCNBlock."""
        adj_matrix = get_human_skeleton_adjacency_matrix()
        block = STGCNBlock(
            in_channels=self.num_features,
            out_channels=64,
            temporal_kernel_size=3,
            adjacency_matrix=adj_matrix,
            dropout=0.1
        )
        
        output = block(self.sample_input)
        self.assertEqual(output.shape, (self.batch_size, self.sequence_length, self.num_joints, 64))


if __name__ == '__main__':
    unittest.main()
