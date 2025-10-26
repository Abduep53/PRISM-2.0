"""
Unit tests for data_pipeline module.
"""

import unittest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_pipeline import (
    extract_and_normalize_pose,
    kinematic_features,
    PRISMDataset,
    calculate_joint_angle,
    calculate_orientation,
    calculate_distance
)


class TestDataPipeline(unittest.TestCase):
    """Test cases for data pipeline functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_pose = np.random.randn(30, 33, 4)  # 30 frames, 33 joints, 4 features
        self.sample_pose[:, :, 3] = np.abs(self.sample_pose[:, :, 3]) % 1.0  # Valid confidence values
        
    def test_kinematic_features_shape(self):
        """Test that kinematic features have correct shape."""
        features = kinematic_features(self.sample_pose)
        self.assertEqual(features.shape[0], 30)  # Same number of frames
        self.assertEqual(features.shape[1], 100)  # Expected feature dimension
        
    def test_calculate_joint_angle(self):
        """Test joint angle calculation."""
        # Test with simple vectors
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        angle = calculate_joint_angle(v1, v2)
        self.assertAlmostEqual(angle, np.pi/2, places=5)
        
    def test_calculate_distance(self):
        """Test distance calculation."""
        p1 = np.array([0, 0, 0])
        p2 = np.array([3, 4, 0])
        distance = calculate_distance(p1, p2)
        self.assertAlmostEqual(distance, 5.0, places=5)
        
    def test_prism_dataset_initialization(self):
        """Test PRISMDataset initialization."""
        # Create dummy data directory
        import tempfile
        import pickle
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy pose files
            for i in range(10):
                pose_data = np.random.randn(30, 33, 4)
                pose_data[:, :, 3] = np.abs(pose_data[:, :, 3]) % 1.0
                np.save(os.path.join(temp_dir, f"pose_sequence_{i:04d}.npy"), pose_data)
            
            # Create labels file
            labels = np.random.randint(0, 101, 10)
            with open(os.path.join(temp_dir, "labels.pkl"), 'wb') as f:
                pickle.dump(labels, f)
            
            # Test dataset initialization
            dataset = PRISMDataset(
                data_dir=temp_dir,
                labels_file=os.path.join(temp_dir, "labels.pkl"),
                sequence_length=30,
                use_kinematics=True
            )
            
            self.assertEqual(len(dataset), 10)
            self.assertEqual(dataset.feature_dims, 100)  # Expected kinematic feature dimension


if __name__ == '__main__':
    unittest.main()
