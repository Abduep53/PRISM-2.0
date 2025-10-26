"""
Unit tests for privacy_module.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from privacy_module import (
    PrivacyConfig,
    PrivacyAccountant,
    ManualDPTrainer,
    PrivacyPreservingTrainer,
    create_privacy_config
)


class TestPrivacyModule(unittest.TestCase):
    """Test cases for privacy module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_model = nn.Linear(10, 2)
        self.privacy_config = PrivacyConfig(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            epochs=1,
            batch_size=32,
            learning_rate=0.001
        )
        
    def test_privacy_config_creation(self):
        """Test PrivacyConfig creation."""
        config = create_privacy_config(epsilon=2.0, delta=1e-4)
        self.assertEqual(config.epsilon, 2.0)
        self.assertEqual(config.delta, 1e-4)
        self.assertIsNotNone(config.noise_multiplier)
        
    def test_privacy_accountant_initialization(self):
        """Test PrivacyAccountant initialization."""
        accountant = PrivacyAccountant(self.privacy_config)
        self.assertEqual(accountant.epsilon, 1.0)
        self.assertEqual(accountant.delta, 1e-5)
        
    def test_privacy_accountant_tracking(self):
        """Test privacy budget tracking."""
        accountant = PrivacyAccountant(self.privacy_config)
        
        # Simulate privacy consumption
        initial_epsilon = accountant.get_current_epsilon()
        accountant.consume_privacy(0.5)
        current_epsilon = accountant.get_current_epsilon()
        
        self.assertLess(current_epsilon, initial_epsilon)
        
    def test_manual_dp_trainer_initialization(self):
        """Test ManualDPTrainer initialization."""
        trainer = ManualDPTrainer(self.sample_model, self.privacy_config)
        self.assertEqual(trainer.model, self.sample_model)
        self.assertEqual(trainer.privacy_config, self.privacy_config)
        
    def test_privacy_preserving_trainer_initialization(self):
        """Test PrivacyPreservingTrainer initialization."""
        trainer = PrivacyPreservingTrainer(self.sample_model, self.privacy_config)
        self.assertIsNotNone(trainer.trainer)
        
    def test_gradient_clipping(self):
        """Test gradient clipping functionality."""
        trainer = ManualDPTrainer(self.sample_model, self.privacy_config)
        
        # Create dummy gradients
        gradients = [torch.randn(10, 2) * 10]  # Large gradients
        
        clipped_gradients = trainer._clip_gradients(gradients)
        
        # Check that gradients are clipped
        grad_norm = torch.norm(torch.cat([g.flatten() for g in clipped_gradients]))
        self.assertLessEqual(grad_norm.item(), self.privacy_config.max_grad_norm + 1e-6)
        
    def test_noise_addition(self):
        """Test noise addition to gradients."""
        trainer = ManualDPTrainer(self.sample_model, self.privacy_config)
        
        # Create dummy gradients
        gradients = [torch.randn(10, 2)]
        
        noisy_gradients = trainer._add_noise(gradients)
        
        # Check that noise was added (gradients should be different)
        original_norm = torch.norm(torch.cat([g.flatten() for g in gradients]))
        noisy_norm = torch.norm(torch.cat([g.flatten() for g in noisy_gradients]))
        
        # They should be different (with high probability)
        self.assertNotEqual(original_norm.item(), noisy_norm.item())


if __name__ == '__main__':
    unittest.main()
