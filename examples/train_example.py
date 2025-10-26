"""
Example training script for PRISM baseline models.
Demonstrates how to train LSTM and GRU models on pose sequence data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
import os

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_pipeline import PRISMDataset
from models import train_baseline, evaluate_model, BaselineLSTM, BaselineGRU, STGCN_PRISM
from privacy_module import PrivacyPreservingTrainer, create_privacy_config


def create_dummy_data(data_dir: str = "dummy_data", num_samples: int = 1000, num_classes: int = 101):
    """
    Create dummy pose sequence data for demonstration.
    
    Args:
        data_dir: Directory to save dummy data
        num_samples: Number of samples to generate
        num_classes: Number of action classes
    """
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Creating {num_samples} dummy pose sequences...")
    
    # Generate dummy pose sequences
    sequence_length = 30
    landmarks = 33
    features = 4
    
    for i in range(num_samples):
        # Generate random normalized pose data
        pose_sequence = np.random.randn(sequence_length, landmarks, features)
        
        # Ensure confidence values are between 0 and 1
        pose_sequence[:, :, 3] = np.abs(pose_sequence[:, :, 3]) % 1.0
        
        # Add some structure to make it more realistic
        # Add temporal correlation
        for j in range(1, sequence_length):
            pose_sequence[j] = 0.7 * pose_sequence[j-1] + 0.3 * pose_sequence[j]
        
        # Save to file
        filename = os.path.join(data_dir, f"pose_sequence_{i:04d}.npy")
        np.save(filename, pose_sequence)
    
    # Create labels file
    labels = np.random.randint(0, num_classes, num_samples)
    labels_file = os.path.join(data_dir, "labels.pkl")
    
    import pickle
    with open(labels_file, 'wb') as f:
        pickle.dump(labels, f)
    
    print(f"✓ Created dummy data in {data_dir}/")
    print(f"  - {num_samples} pose sequences")
    print(f"  - {num_classes} action classes")
    print(f"  - Labels saved to {labels_file}")
    
    return data_dir, labels_file


def main():
    """Main training example."""
    print("PRISM Baseline Model Training Example")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create dummy data
    data_dir, labels_file = create_dummy_data(num_samples=1000, num_classes=101)
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = PRISMDataset(
        data_dir=data_dir,
        labels_file=labels_file,
        sequence_length=30,
        transform=None
    )
    
    print(f"✓ Dataset created with {len(dataset)} samples")
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"✓ Dataset split:")
    print(f"  - Training: {len(train_dataset)} samples")
    print(f"  - Validation: {len(val_dataset)} samples")
    print(f"  - Test: {len(test_dataset)} samples")
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✓ Data loaders created with batch size {batch_size}")
    
    # Training configurations
    configs = [
        {
            'name': 'LSTM Baseline',
            'model_type': 'lstm',
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'bidirectional': False
        },
        {
            'name': 'GRU Baseline',
            'model_type': 'gru',
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'bidirectional': False
        },
        {
            'name': 'STGCN_PRISM',
            'model_type': 'stgcn',
            'hidden_channels': [64, 128, 256],
            'temporal_kernel_sizes': [3, 3, 3],
            'dropout': 0.1,
            'use_attention': False
        },
        {
            'name': 'STGCN_PRISM with Attention',
            'model_type': 'stgcn',
            'hidden_channels': [64, 128, 256],
            'temporal_kernel_sizes': [3, 3, 3],
            'dropout': 0.1,
            'use_attention': True
        },
        {
            'name': 'STGCN_PRISM with Privacy (ε=1.0)',
            'model_type': 'stgcn_privacy',
            'hidden_channels': [64, 128, 256],
            'temporal_kernel_sizes': [3, 3, 3],
            'dropout': 0.1,
            'use_attention': False,
            'privacy_epsilon': 1.0
        }
    ]
    
    # Train models
    results = {}
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Training {config['name']}")
        print(f"{'='*60}")
        
        try:
            # Train model
            train_kwargs = {
                'model_type': config['model_type'],
                'train_loader': train_loader,
                'val_loader': val_loader,
                'num_epochs': 20,  # Reduced for demo
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'num_classes': 101,
                'device': 'auto',
                'save_model': True,
                'model_save_path': f"{config['model_type']}_baseline.pth"
            }
            
            # Handle privacy-preserving training
            if config['model_type'] == 'stgcn_privacy':
                # Create STGCN model
                model = STGCN_PRISM(
                    num_joints=33,
                    in_channels=4,
                    num_classes=101,
                    hidden_channels=config['hidden_channels'],
                    temporal_kernel_sizes=config['temporal_kernel_sizes'],
                    dropout=config['dropout'],
                    use_attention=config['use_attention']
                )
                
                # Create privacy configuration
                privacy_config = create_privacy_config(
                    epsilon=config['privacy_epsilon'],
                    delta=1e-5,
                    max_grad_norm=1.0,
                    epochs=20,
                    batch_size=16,
                    learning_rate=0.001,
                    use_opacus=True
                )
                
                # Create privacy trainer
                privacy_trainer = PrivacyPreservingTrainer(model, privacy_config)
                
                # Train with privacy
                result = privacy_trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=20,
                    learning_rate=0.001
                )
                
            else:
                # Add model-specific parameters for regular training
                if config['model_type'] in ['lstm', 'gru']:
                    train_kwargs.update({
                        'hidden_size': config['hidden_size'],
                        'num_layers': config['num_layers'],
                        'dropout': config['dropout'],
                        'bidirectional': config['bidirectional']
                    })
                elif config['model_type'] == 'stgcn':
                    train_kwargs.update({
                        'hidden_channels': config['hidden_channels'],
                        'temporal_kernel_sizes': config['temporal_kernel_sizes'],
                        'dropout': config['dropout'],
                        'use_attention': config['use_attention']
                    })
                
                result = train_baseline(**train_kwargs)
            
            results[config['name']] = result
            
            # Evaluate on test set
            print(f"\nEvaluating {config['name']} on test set...")
            
            # Handle different result formats
            if config['model_type'] == 'stgcn_privacy':
                # Privacy training results have different structure
                model = result['model']
                final_epsilon = result['final_epsilon']
                print(f"Privacy Spent: ε={final_epsilon:.4f}")
            else:
                # Regular training results
                model = result['model']
            
            test_results = evaluate_model(
                model=model,
                test_loader=test_loader,
                device='auto'
            )
            
            print(f"Test Accuracy: {test_results['accuracy']:.4f}")
            
            # Add privacy information if available
            if config['model_type'] == 'stgcn_privacy':
                print(f"Privacy Budget: ε={final_epsilon:.4f}")
            
        except Exception as e:
            print(f"Error training {config['name']}: {str(e)}")
            continue
    
    # Compare results
    print(f"\n{'='*60}")
    print("TRAINING RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Best Validation Accuracy: {result['best_val_acc']:.2f}%")
        print(f"  Training Time: {result['training_time']:.2f}s")
        print(f"  Model Parameters: {sum(p.numel() for p in result['model'].parameters()):,}")
        print()
    
    print("Training completed!")
    print("\nNext steps:")
    print("1. Use your own pose data by replacing the dummy data generation")
    print("2. Adjust hyperparameters based on your data characteristics")
    print("3. Implement more advanced models for comparison")
    print("4. Add data augmentation techniques")


if __name__ == "__main__":
    main()
