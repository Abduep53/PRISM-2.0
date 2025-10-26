"""
Fast GPU training script for PRISM on NTU RGB+D dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import wandb  # For logging (optional)

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ntu_data_loader import create_ntu_dataloaders
from optimized_models import OptimizedSTGCN_PRISM
from privacy_module import PrivacyPreservingTrainer, create_privacy_config
from benchmarks import ModelEvaluator

class FastTrainer:
    """Fast GPU trainer with optimizations."""
    
    def __init__(self, model, device, use_mixed_precision=True):
        self.model = model.to(device)
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        
        # Mixed precision training
        if use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Optimizer with optimizations
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch with optimizations."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to GPU
            pose_sequences = batch['pose_sequence'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(pose_sequences)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(pose_sequences)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.0 * correct / total:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                pose_sequences = batch['pose_sequence'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(pose_sequences)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(pose_sequences)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs, save_path=None):
        """Train model with optimizations."""
        best_val_acc = 0.0
        start_time = time.time()
        
        print(f"Starting training on {self.device}")
        print(f"Mixed precision: {self.use_mixed_precision}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_path:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'epoch': epoch,
                        'val_acc': val_acc,
                        'train_acc': train_acc
                    }, save_path)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Best Val Acc: {best_val_acc:.2f}%")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 60)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        return {
            'best_val_acc': best_val_acc,
            'training_time': training_time,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Fast GPU training for PRISM on NTU RGB+D')
    parser.add_argument('--data_path', type=str, required=True, help='Path to NTU RGB+D dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--sequence_length', type=int, default=300, help='Sequence length')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker processes')
    parser.add_argument('--use_mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--use_privacy', action='store_true', help='Use differential privacy')
    parser.add_argument('--privacy_epsilon', type=float, default=1.0, help='Privacy budget')
    parser.add_argument('--save_path', type=str, default='ntu_model.pth', help='Path to save model')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb (optional)
    if args.use_wandb:
        wandb.init(project="prism-ntu", config=vars(args))
    
    # Create data loaders
    print("Loading NTU RGB+D dataset...")
    train_loader, val_loader = create_ntu_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sequence_length=args.sequence_length
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = OptimizedSTGCN_PRISM(
        num_joints=25,
        in_channels=3,
        num_classes=60,
        hidden_channels=[64, 128, 256],
        temporal_kernel_sizes=[3, 3, 3],
        dropout=0.1,
        use_attention=False
    )
    
    # Create trainer
    if args.use_privacy:
        # Privacy-preserving training
        privacy_config = create_privacy_config(
            epsilon=args.privacy_epsilon,
            delta=1e-5,
            max_grad_norm=1.0,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=0.001,
            use_opacus=True
        )
        
        trainer = PrivacyPreservingTrainer(model, privacy_config)
        results = trainer.train(train_loader, val_loader, args.num_epochs)
    else:
        # Regular training
        trainer = FastTrainer(
            model=model,
            device=device,
            use_mixed_precision=args.use_mixed_precision
        )
        
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            save_path=args.save_path
        )
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = ModelEvaluator(device=device)
    test_metrics = evaluator.evaluate_model_comprehensive(
        model=trainer.model,
        test_loader=val_loader,
        measure_latency=True
    )
    
    print(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Final Test F1-Score: {test_metrics['f1_weighted']:.4f}")
    print(f"Mean Latency: {test_metrics['mean_latency_ms']:.2f} ms")
    
    # Log to wandb
    if args.use_wandb:
        wandb.log({
            'final_test_accuracy': test_metrics['accuracy'],
            'final_test_f1': test_metrics['f1_weighted'],
            'mean_latency_ms': test_metrics['mean_latency_ms']
        })
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()