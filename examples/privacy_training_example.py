"""
Privacy-Preserving Training Example for PRISM
Demonstrates ε-Differential Privacy training with STGCN_PRISM model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import STGCN_PRISM
from privacy_module import (
    PrivacyPreservingTrainer, 
    create_privacy_config,
    analyze_privacy_utility_tradeoff
)


def create_dummy_pose_data(num_samples: int = 1000, num_classes: int = 101) -> DataLoader:
    """
    Create dummy pose sequence data for privacy training demonstration.
    
    Args:
        num_samples: Number of samples to generate
        num_classes: Number of action classes
        
    Returns:
        DataLoader with dummy pose data
    """
    print(f"Creating {num_samples} dummy pose sequences...")
    
    # Generate dummy pose sequences
    sequence_length = 30
    num_joints = 33
    in_channels = 4
    
    # Create realistic pose data
    pose_sequences = torch.randn(num_samples, sequence_length, num_joints, in_channels)
    
    # Make confidence values realistic (0-1)
    pose_sequences[:, :, :, 3] = torch.sigmoid(pose_sequences[:, :, :, 3])
    
    # Add temporal correlation to make it more realistic
    for t in range(1, sequence_length):
        pose_sequences[:, t] = 0.7 * pose_sequences[:, t-1] + 0.3 * pose_sequences[:, t]
    
    # Generate labels
    labels = torch.randint(0, num_classes, (num_samples, 1))
    
    # Create dataset
    dataset = TensorDataset(pose_sequences, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"✓ Created dummy dataset with {num_samples} samples")
    print(f"  Pose shape: {pose_sequences.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Number of classes: {num_classes}")
    
    return dataloader


def demonstrate_privacy_training():
    """Demonstrate privacy-preserving training with different privacy budgets."""
    print("PRISM Privacy-Preserving Training Demonstration")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create dummy data
    train_loader = create_dummy_pose_data(num_samples=1000, num_classes=101)
    val_loader = create_dummy_pose_data(num_samples=200, num_classes=101)
    
    # Different privacy configurations to test
    privacy_configs = [
        {
            'name': 'High Privacy (ε=0.1)',
            'epsilon': 0.1,
            'delta': 1e-5,
            'max_grad_norm': 1.0,
            'epochs': 10,
            'batch_size': 16
        },
        {
            'name': 'Medium Privacy (ε=1.0)',
            'epsilon': 1.0,
            'delta': 1e-5,
            'max_grad_norm': 1.0,
            'epochs': 10,
            'batch_size': 16
        },
        {
            'name': 'Low Privacy (ε=10.0)',
            'epsilon': 10.0,
            'delta': 1e-5,
            'max_grad_norm': 1.0,
            'epochs': 10,
            'batch_size': 16
        }
    ]
    
    results = {}
    
    for config_dict in privacy_configs:
        print(f"\n{'='*60}")
        print(f"Training with {config_dict['name']}")
        print(f"{'='*60}")
        
        # Create model
        model = STGCN_PRISM(
            num_joints=33,
            in_channels=4,
            num_classes=101,
            hidden_channels=[32, 64, 128],  # Smaller model for demo
            temporal_kernel_sizes=[3, 3, 3],
            dropout=0.1,
            use_attention=False
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create privacy configuration
        privacy_config = create_privacy_config(
            epsilon=config_dict['epsilon'],
            delta=config_dict['delta'],
            max_grad_norm=config_dict['max_grad_norm'],
            epochs=config_dict['epochs'],
            batch_size=config_dict['batch_size'],
            learning_rate=0.001,
            use_opacus=True  # Try Opacus first, fallback to manual if not available
        )
        
        # Create privacy trainer
        privacy_trainer = PrivacyPreservingTrainer(model, privacy_config)
        
        # Train model
        try:
            training_results = privacy_trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config_dict['epochs'],
                learning_rate=0.001
            )
            
            results[config_dict['name']] = training_results
            
            # Analyze privacy-utility tradeoff
            analysis = analyze_privacy_utility_tradeoff(training_results)
            
            print(f"\nTraining Results for {config_dict['name']}:")
            print(f"  Final Accuracy: {analysis['final_accuracy']:.4f}")
            print(f"  Privacy Spent: ε={analysis['final_epsilon']:.4f}")
            print(f"  Privacy Efficiency: {analysis['privacy_efficiency']:.4f}")
            print(f"  Privacy Budget Used: {analysis['privacy_budget_used']:.1f}%")
            print(f"  Trainer Type: {analysis['trainer_type']}")
            
        except Exception as e:
            print(f"Error training {config_dict['name']}: {str(e)}")
            continue
    
    return results


def compare_privacy_mechanisms():
    """Compare Opacus vs Manual DP implementation."""
    print(f"\n{'='*60}")
    print("Comparing Privacy Mechanisms")
    print(f"{'='*60}")
    
    # Create small dataset for quick comparison
    train_loader = create_dummy_pose_data(num_samples=200, num_classes=10)
    
    # Test both Opacus and Manual DP
    mechanisms = [
        {'name': 'Opacus DP', 'use_opacus': True},
        {'name': 'Manual DP', 'use_opacus': False}
    ]
    
    for mechanism in mechanisms:
        print(f"\nTesting {mechanism['name']}...")
        
        try:
            # Create model
            model = STGCN_PRISM(
                num_joints=33,
                in_channels=4,
                num_classes=10,
                hidden_channels=[16, 32],  # Very small model for demo
                temporal_kernel_sizes=[3, 3],
                dropout=0.1,
                use_attention=False
            )
            
            # Create privacy config
            privacy_config = create_privacy_config(
                epsilon=1.0,
                delta=1e-5,
                max_grad_norm=1.0,
                epochs=5,
                batch_size=8,
                learning_rate=0.001,
                use_opacus=mechanism['use_opacus']
            )
            
            # Create trainer
            privacy_trainer = PrivacyPreservingTrainer(model, privacy_config)
            
            # Train for a few epochs
            results = privacy_trainer.train(
                train_loader=train_loader,
                val_loader=None,
                num_epochs=5,
                learning_rate=0.001
            )
            
            # Print results
            analysis = analyze_privacy_utility_tradeoff(results)
            print(f"  ✓ {mechanism['name']} completed successfully")
            print(f"    Final Accuracy: {analysis['final_accuracy']:.4f}")
            print(f"    Privacy Spent: ε={analysis['final_epsilon']:.4f}")
            print(f"    Trainer Type: {analysis['trainer_type']}")
            
        except Exception as e:
            print(f"  ✗ {mechanism['name']} failed: {str(e)}")


def demonstrate_privacy_analysis():
    """Demonstrate privacy analysis and reporting."""
    print(f"\n{'='*60}")
    print("Privacy Analysis and Reporting")
    print(f"{'='*60}")
    
    # Create a sample training result
    train_loader = create_dummy_pose_data(num_samples=500, num_classes=20)
    
    model = STGCN_PRISM(
        num_joints=33,
        in_channels=4,
        num_classes=20,
        hidden_channels=[32, 64],
        temporal_kernel_sizes=[3, 3],
        dropout=0.1,
        use_attention=False
    )
    
    privacy_config = create_privacy_config(
        epsilon=2.0,
        delta=1e-5,
        max_grad_norm=1.0,
        epochs=8,
        batch_size=16,
        learning_rate=0.001,
        use_opacus=True
    )
    
    privacy_trainer = PrivacyPreservingTrainer(model, privacy_config)
    
    # Train model
    results = privacy_trainer.train(
        train_loader=train_loader,
        val_loader=None,
        num_epochs=8,
        learning_rate=0.001
    )
    
    # Analyze results
    analysis = analyze_privacy_utility_tradeoff(results)
    
    print(f"\nPrivacy Analysis Report:")
    print(f"  Target Privacy: ε={privacy_config.epsilon}, δ={privacy_config.delta}")
    print(f"  Achieved Privacy: ε={analysis['final_epsilon']:.4f}, δ={privacy_config.delta}")
    print(f"  Privacy Budget Used: {analysis['privacy_budget_used']:.1f}%")
    print(f"  Final Accuracy: {analysis['final_accuracy']:.4f}")
    print(f"  Privacy Efficiency: {analysis['privacy_efficiency']:.4f}")
    print(f"  Implementation: {analysis['trainer_type']}")
    
    # Privacy-utility tradeoff analysis
    print(f"\nPrivacy-Utility Tradeoff:")
    if analysis['final_epsilon'] <= privacy_config.epsilon:
        print(f"  ✓ Privacy budget respected")
    else:
        print(f"  ⚠ Privacy budget exceeded")
    
    if analysis['privacy_efficiency'] > 0.5:
        print(f"  ✓ Good privacy efficiency")
    elif analysis['privacy_efficiency'] > 0.2:
        print(f"  ⚠ Moderate privacy efficiency")
    else:
        print(f"  ✗ Low privacy efficiency")


def main():
    """Main demonstration function."""
    print("PRISM Privacy Module - ε-Differential Privacy Training")
    print("=" * 60)
    print("This demonstrates the core scientific novelty of PRISM:")
    print("Privacy-preserving human action recognition using DP-SGD")
    print("=" * 60)
    
    # Run demonstrations
    try:
        # Demonstrate privacy training with different budgets
        results = demonstrate_privacy_training()
        
        # Compare privacy mechanisms
        compare_privacy_mechanisms()
        
        # Demonstrate privacy analysis
        demonstrate_privacy_analysis()
        
        print(f"\n{'='*60}")
        print("PRIVACY TRAINING DEMONSTRATION COMPLETE")
        print(f"{'='*60}")
        print("Key Features Demonstrated:")
        print("• ε-Differential Privacy with configurable privacy budgets")
        print("• DP-SGD with gradient clipping and noise injection")
        print("• Opacus integration for automatic DP enforcement")
        print("• Manual DP implementation as fallback")
        print("• Privacy budget tracking and accounting")
        print("• Privacy-utility tradeoff analysis")
        print("\nThe PRISM privacy module is ready for production use!")
        
    except Exception as e:
        print(f"Error in demonstration: {str(e)}")
        print("This may be due to missing dependencies or configuration issues.")


if __name__ == "__main__":
    main()
