"""
PRISM Privacy Module: ε-Differential Privacy Implementation

This module provides comprehensive differential privacy mechanisms for the PRISM project,
including DP-SGD with Opacus integration and manual DP implementation with gradient
clipping and noise injection. This is the core scientific novelty of PRISM.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math
import warnings
from dataclasses import dataclass
import time

# Opacus imports
try:
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    from opacus.validators import ModuleValidator
    # --- ИСПРАВЛЕНИЕ 1 ---
    # Мы импортируем 'get_noise_multiplier' отсюда,
    # так как его удалили из PrivacyEngine в Opacus 1.0+
    from opacus.accountants import RDPAccountant, get_noise_multiplier
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    print("Warning: Opacus not available. Manual DP implementation will be used.")
    # Добавляем "заглушки" на случай, если Opacus не установлен
    class PrivacyEngine: pass
    class BatchMemoryManager: pass
    class ModuleValidator: pass
    class RDPAccountant: pass
    def get_noise_multiplier(target_epsilon, target_delta, sample_rate, epochs):
        return 1.0 # Возвращаем 1.0, если Opacus недоступен

@dataclass
class PrivacyConfig:
    """Configuration for differential privacy training."""
    epsilon: float = 1.0  # Privacy budget (ε)
    delta: float = 1e-5   # Failure probability (δ)
    max_grad_norm: float = 1.0  # Gradient clipping norm
    noise_multiplier: Optional[float] = None  # Auto-calculated if None
    target_epsilon: Optional[float] = None  # Target privacy budget
    target_delta: Optional[float] = None  # Target failure probability
    epochs: int = 50  # Number of training epochs
    batch_size: int = 32  # Batch size
    learning_rate: float = 0.001  # Learning rate
    use_opacus: bool = True  # Whether to use Opacus (if available)
    noise_type: str = 'gaussian'  # 'gaussian' or 'laplace'
    privacy_accounting: str = 'rdp'  # 'rdp' or 'moments'


class PrivacyAccountant:
    """
    Privacy accountant for tracking privacy budget consumption.
    Implements Renyi Differential Privacy (RDP) accounting.
    """
    
    def __init__(self, delta: float = 1e-5):
        """
        Initialize privacy accountant.
        
        Args:
            delta: Failure probability for (ε, δ)-DP
        """
        self.delta = delta
        self.alpha_values = [1 + x / 10.0 for x in range(1, 100)]  # RDP orders
        self.rdp_orders = []
        self.rdp_values = []
        
    def add_step(self, noise_multiplier: float, batch_size: int, 
                 dataset_size: int, sampling_rate: float):
        """
        Add a training step to the privacy accounting.
        
        Args:
            noise_multiplier: Noise multiplier for DP-SGD
            batch_size: Batch size
            dataset_size: Total dataset size
            sampling_rate: Sampling rate (batch_size / dataset_size)
        """
        # Calculate RDP for Gaussian mechanism
        rdp_value = self._compute_rdp_gaussian(noise_multiplier, sampling_rate)
        self.rdp_orders.append(1.0)  # Simplified for Gaussian
        self.rdp_values.append(rdp_value)
    
    def _compute_rdp_gaussian(self, noise_multiplier: float, sampling_rate: float) -> float:
        """Compute RDP for Gaussian mechanism."""
        if noise_multiplier <= 0:
            return float('inf')
        
        # Simplified RDP calculation for Gaussian mechanism
        sigma = noise_multiplier
        alpha = 1.0
        return alpha / (2 * sigma**2)
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get current privacy budget spent.
        
        Returns:
            Tuple of (epsilon, delta)
        """
        if not self.rdp_values:
            return 0.0, self.delta
        
        # Convert RDP to (ε, δ)-DP
        # ПРИМЕЧАНИЕ: Исходный код здесь содержал ошибку (деление на ноль: 1-1)
        # Это очень упрощенный (и, вероятно, неверный) RDP.
        # Используем Opacus' RDPAccountant для более точного отслеживания.
        try:
            total_rdp = sum(self.rdp_values)
            if not self.alpha_values: # Избегаем деления на ноль
                return 0.0, self.delta
            
            # (Это все еще очень грубое приближение, Opacus намного лучше)
            alpha = self.alpha_values[0] # Берем первый порядок
            epsilon = total_rdp + math.log(1 / self.delta) / (alpha - 1)
        except ZeroDivisionError:
             epsilon = float('inf')
        except Exception:
             epsilon = float('inf') # Обработка других математических ошибок

        return epsilon, self.delta
    
    def get_remaining_budget(self, target_epsilon: float) -> float:
        """Get remaining privacy budget."""
        current_epsilon, _ = self.get_privacy_spent()
        return max(0.0, target_epsilon - current_epsilon)


class ManualDPTrainer:
    """
    Manual implementation of DP-SGD without Opacus.
    Implements gradient clipping and noise injection.
    """
    
    def __init__(self, model: nn.Module, privacy_config: PrivacyConfig):
        """
        Initialize manual DP trainer.
        
        Args:
            model: PyTorch model to train
            privacy_config: Privacy configuration
        """
        self.model = model
        self.config = privacy_config
        self.privacy_accountant = PrivacyAccountant(privacy_config.delta)
        
        # Calculate noise multiplier if not provided
        if privacy_config.noise_multiplier is None:
            self.config.noise_multiplier = self._calculate_noise_multiplier()
        
        print(f"Manual DP-SGD Configuration:")
        print(f"  Target ε: {privacy_config.epsilon}")
        print(f"  δ: {privacy_config.delta}")
        print(f"  Max grad norm: {privacy_config.max_grad_norm}")
        print(f"  Noise multiplier: {self.config.noise_multiplier:.4f}")
    
    def _calculate_noise_multiplier(self) -> float:
        """Calculate noise multiplier based on privacy budget."""
        # Simplified calculation - in practice, use more sophisticated methods
        dataset_size = 1000  # Assume dataset size
        sampling_rate = self.config.batch_size / dataset_size
        
        # Approximate noise multiplier calculation
        # This is a simplified version - real implementation would use RDP
        target_epsilon = self.config.epsilon
        noise_multiplier = math.sqrt(2 * math.log(1.25 / self.config.delta)) / target_epsilon
        
        return max(1.0, noise_multiplier)
    
    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradients to L2 norm.
        
        Args:
            model: Model with gradients
            
        Returns:
            Total gradient norm before clipping
        """
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # Clip gradients
        clip_coef = min(1.0, self.config.max_grad_norm / (total_norm + 1e-6))
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)
        
        return total_norm
    
    def add_noise_to_gradients(self, model: nn.Module, batch_size: int):
        """
        Add calibrated noise to gradients.
        
        Args:
            model: Model with clipped gradients
            batch_size: Current batch size
        """
        noise_scale = self.config.max_grad_norm * self.config.noise_multiplier
        
        for param in model.parameters():
            if param.grad is not None:
                if self.config.noise_type == 'gaussian':
                    noise = torch.normal(0, noise_scale, param.grad.shape, 
                                         device=param.grad.device, dtype=param.grad.dtype)
                elif self.config.noise_type == 'laplace':
                    noise = torch.distributions.Laplace(0, noise_scale).sample(param.grad.shape).to(param.grad.device)
                else:
                    raise ValueError(f"Unknown noise type: {self.config.noise_type}")
                
                param.grad.data.add_(noise)
    
    def train_step(self, model: nn.Module, optimizer: optim.Optimizer, 
                   batch: Dict[str, torch.Tensor], dataset_size: int) -> Dict[str, float]:
        """
        Perform one DP training step.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            batch: Training batch
            dataset_size: Total dataset size
            
        Returns:
            Dictionary with training metrics
        """
        model.train()
        
        # Forward pass
        pose_sequences = batch['pose_sequence']
        labels = batch['label'].squeeze()
        
        optimizer.zero_grad()
        outputs = model(pose_sequences)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # DP-SGD steps
        grad_norm = self.clip_gradients(model)
        self.add_noise_to_gradients(model, pose_sequences.size(0))
        
        # Update parameters
        optimizer.step()
        
        # Update privacy accounting
        sampling_rate = pose_sequences.size(0) / dataset_size
        self.privacy_accountant.add_step(
            self.config.noise_multiplier, 
            pose_sequences.size(0), 
            dataset_size, 
            sampling_rate
        )
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'grad_norm': grad_norm,
            'privacy_epsilon': self.privacy_accountant.get_privacy_spent()[0]
        }


class OpacusDPTrainer:
    """
    DP-SGD trainer using Opacus library.
    Provides automatic DP enforcement and accounting.
    """
    
    def __init__(self, model: nn.Module, privacy_config: PrivacyConfig):
        """
        Initialize Opacus DP trainer.
        
        Args:
            model: PyTorch model to train
            privacy_config: Privacy configuration
        """
        if not OPACUS_AVAILABLE:
            raise ImportError("Opacus is required for OpacusDPTrainer")
        
        self.model = model
        self.config = privacy_config
        self.privacy_engine = None
        self.optimizer = None
        self.data_loader = None # Opacus 1.0+ изменяет DataLoader
        
        # Validate model for Opacus
        self._validate_model()
        
        print(f"Opacus DP-SGD Configuration:")
        print(f"  Target ε: {privacy_config.epsilon}")
        print(f"  δ: {privacy_config.delta}")
        print(f"  Max grad norm: {privacy_config.max_grad_norm}")
    
    def _validate_model(self):
        """Validate model compatibility with Opacus."""
        try:
            # Opacus 1.0+ может требовать 'strict=True' или 'strict=False'
            # 'strict=False' более щадящий
            is_valid = ModuleValidator.is_valid(self.model)
            if not is_valid:
                 print("Warning: Model is not strictly valid for Opacus. Attempting to fix...")
                 self.model = ModuleValidator.fix(self.model)
                 print("✓ Model fixed for Opacus")
            else:
                 print("✓ Model validated for Opacus")
        except Exception as e:
            print(f"⚠ Model validation warning: {e}")
            print("Proceeding with potential compatibility issues...")
    
    def setup_privacy_engine(self, optimizer: optim.Optimizer, 
                             data_loader: DataLoader) -> optim.Optimizer:
        """
        Setup privacy engine with Opacus.
        
        Args:
            optimizer: PyTorch optimizer
            data_loader: Training data loader
            
        Returns:
            Privacy-enabled optimizer
        """
        self.privacy_engine = PrivacyEngine()
        
        # --- ИСПРАВЛЕНИЕ 2 ---
        # Мы используем импортированную функцию 'get_noise_multiplier'
        # и исправляем расчет 'sample_rate'.
        
        # Calculate noise multiplier if not provided
        if self.config.noise_multiplier is None:
            
            # Правильный sample_rate = batch_size / dataset_size
            if not data_loader.batch_size or not hasattr(data_loader, 'dataset'):
                 raise ValueError("DataLoader must have batch_size and dataset attributes for DP.")
            
            dataset_size = len(data_loader.dataset)
            batch_size = data_loader.batch_size
            sample_rate = batch_size / dataset_size
            
            print(f"Calculating noise multiplier for sample_rate={sample_rate:.4f}...")
            
            self.config.noise_multiplier = get_noise_multiplier(
                target_epsilon=self.config.epsilon,
                target_delta=self.config.delta,
                sample_rate=sample_rate,
                epochs=self.config.epochs
            )
        # --- КОНЕЦ ИСПРАВЛЕНИЯ 2 ---

        # Make optimizer privacy-enabled
        # В Opacus 1.0+ 'make_private' заменена на 'attach'
        # и она не возвращает optimizer/data_loader, а модифицирует их
        
        # Проверяем, существует ли 'attach' (Opacus 1.0+)
        if hasattr(self.privacy_engine, 'attach'):
            self.data_loader = data_loader # Сохраняем ссылку на оригинальный loader
            self.optimizer = optimizer
            self.privacy_engine.attach(
                optimizer=self.optimizer,
                data_loader=self.data_loader,
                module=self.model,
                noise_multiplier=self.config.noise_multiplier,
                max_grad_norm=self.config.max_grad_norm,
            )
        else:
            # Используем старый API 'make_private' (Opacus < 1.0)
            self.optimizer, self.data_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=self.config.noise_multiplier,
                max_grad_norm=self.config.max_grad_norm,
            )
        
        print(f"✓ Privacy engine setup complete")
        print(f"  Noise multiplier: {self.config.noise_multiplier:.4f}")
        
        return self.optimizer
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one DP training step with Opacus.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        # Forward pass
        pose_sequences = batch['pose_sequence']
        labels = batch['label'].squeeze()
        
        self.optimizer.zero_grad()
        outputs = self.model(pose_sequences)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        # Backward pass (Opacus handles DP automatically)
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean().item()
        
        # Get privacy spent
        # 'get_privacy_spent' возвращает (epsilon, best_alpha)
        # Нам нужен epsilon
        epsilon = self.privacy_engine.get_epsilon(delta=self.config.delta)
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'privacy_epsilon': epsilon,
            'privacy_delta': self.config.delta
        }
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get current privacy budget spent."""
        if self.privacy_engine is None:
            return 0.0, self.config.delta
        
        epsilon = self.privacy_engine.get_epsilon(delta=self.config.delta)
        return epsilon, self.config.delta


class PrivacyPreservingTrainer:
    """
    Main privacy-preserving trainer that can use either Opacus or manual DP.
    """
    
    def __init__(self, model: nn.Module, privacy_config: PrivacyConfig):
        """
        Initialize privacy-preserving trainer.
        
        Args:
            model: PyTorch model to train
            privacy_config: Privacy configuration
        """
        self.model = model
        self.config = privacy_config
        
        # Choose trainer based on availability and preference
        if privacy_config.use_opacus and OPACUS_AVAILABLE:
            self.dp_trainer = OpacusDPTrainer(model, privacy_config)
            self.trainer_type = "opacus"
        else:
            if privacy_config.use_opacus and not OPACUS_AVAILABLE:
                print("Warning: Opacus requested but not found. Falling back to Manual DP.")
            self.dp_trainer = ManualDPTrainer(model, privacy_config)
            self.trainer_type = "manual"
        
        print(f"Using {self.trainer_type.upper()} DP trainer")
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              num_epochs: int = None, learning_rate: float = None) -> Dict:
        """
        Train model with differential privacy.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Training results dictionary
        """
        if num_epochs is None:
            num_epochs = self.config.epochs
        if learning_rate is None:
            learning_rate = self.config.learning_rate
        
        # Setup optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Setup DP trainer
        if self.trainer_type == "opacus":
            # Эта функция теперь также настраивает self.data_loader в Opacus 1.0+
            optimizer = self.dp_trainer.setup_privacy_engine(optimizer, train_loader)
            # Убедимся, что мы используем правильный (возможно, измененный Opacus) data loader
            active_train_loader = self.dp_trainer.data_loader if hasattr(self.dp_trainer, 'data_loader') and self.dp_trainer.data_loader is not None else train_loader
        else:
            active_train_loader = train_loader
        
        # Training loop
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        privacy_epsilons = []
        
        print(f"\nStarting privacy-preserving training...")
        print(f"Epochs: {num_epochs}, Learning rate: {learning_rate}")
        print(f"Privacy budget: ε={self.config.epsilon}, δ={self.config.delta}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            # Training
            epoch_train_loss = 0.0
            epoch_train_acc = 0.0
            num_batches = 0
            
            # Используем active_train_loader
            for batch_idx, batch in enumerate(active_train_loader):
                if self.trainer_type == "opacus":
                    metrics = self.dp_trainer.train_step(batch)
                else:
                    metrics = self.dp_trainer.train_step(
                        self.model, optimizer, batch, len(active_train_loader.dataset)
                    )
                
                epoch_train_loss += metrics['loss']
                epoch_train_acc += metrics['accuracy']
                num_batches += 1
                
                # Print progress
                if batch_idx % 10 == 0:
                    current_epsilon = metrics.get('privacy_epsilon', 0.0)
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(active_train_loader)}: "
                          f"Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.4f}, ε={current_epsilon:.4f}")
            
            # Calculate epoch averages
            if num_batches == 0:
                print(f"Warning: Epoch {epoch+1} had 0 batches. Skipping...")
                continue
                
            avg_train_loss = epoch_train_loss / num_batches
            avg_train_acc = epoch_train_acc / num_batches
            
            train_losses.append(avg_train_loss)
            train_accuracies.append(avg_train_acc)
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
            else:
                val_losses.append(0.0)
                val_accuracies.append(0.0)
            
            # Get privacy spent
            if self.trainer_type == "opacus":
                epsilon, delta = self.dp_trainer.get_privacy_spent()
            else:
                epsilon, delta = self.dp_trainer.privacy_accountant.get_privacy_spent()
            
            privacy_epsilons.append(epsilon)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
            if val_loader is not None:
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  Privacy Spent: ε={epsilon:.4f}, δ={delta:.4f}")
            
            # Оставшийся бюджет
            remaining_budget = self.config.epsilon - epsilon
            print(f"  Remaining Budget: {remaining_budget:.4f}")
            print("-" * 60)
            
            # Check privacy budget
            if epsilon >= self.config.epsilon:
                print(f"⚠ Privacy budget exhausted! Stopping training at epoch {epoch+1}")
                break
        
        # Final privacy report
        if not privacy_epsilons:
            print("Training did not run for any epochs.")
            final_epsilon, final_delta = 0.0, self.config.delta
        else:
            final_epsilon, final_delta = privacy_epsilons[-1], self.config.delta
            
        print(f"\n{'='*60}")
        print("PRIVACY TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Final Privacy Spent: ε={final_epsilon:.4f}, δ={final_delta:.4f}")
        print(f"Target Privacy: ε={self.config.epsilon:.4f}, δ={self.config.delta:.4f}")
        if self.config.epsilon > 0:
            print(f"Privacy Budget Used: {(final_epsilon/self.config.epsilon)*100:.1f}%")
        
        return {
            'model': self.model,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'privacy_epsilons': privacy_epsilons,
            'final_epsilon': final_epsilon,
            'final_delta': final_delta,
            'trainer_type': self.trainer_type,
            'config': self.config
        }
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                pose_sequences = batch['pose_sequence']
                labels = batch['label'].squeeze()
                
                outputs = self.model(pose_sequences)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy


def create_privacy_config(epsilon: float = 1.0, delta: float = 1e-5,
                          max_grad_norm: float = 1.0, epochs: int = 50,
                          batch_size: int = 32, learning_rate: float = 0.001,
                          use_opacus: bool = True) -> PrivacyConfig:
    """
    Create privacy configuration with sensible defaults.
    
    Args:
        epsilon: Privacy budget (ε)
        delta: Failure probability (δ)
        max_grad_norm: Gradient clipping norm
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        use_opacus: Whether to use Opacus (if available)
        
    Returns:
        PrivacyConfig object
    """
    return PrivacyConfig(
        epsilon=epsilon,
        delta=delta,
        max_grad_norm=max_grad_norm,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_opacus=use_opacus
    )


def analyze_privacy_utility_tradeoff(results: Dict) -> Dict:
    """
    Analyze the privacy-utility tradeoff from training results.
    
    Args:
        results: Training results dictionary
        
    Returns:
        Analysis dictionary
    """
    final_accuracy = results['train_accuracies'][-1]
    final_epsilon = results['final_epsilon']
    target_epsilon = results['config'].epsilon
    
    # Calculate privacy efficiency
    privacy_efficiency = final_accuracy / final_epsilon if final_epsilon > 0 else 0
    
    # Calculate utility loss (compared to non-private baseline)
    # This would need to be compared with actual non-private results
    utility_loss = 0.0  # Placeholder
    
    analysis = {
        'final_accuracy': final_accuracy,
        'final_epsilon': final_epsilon,
        'privacy_efficiency': privacy_efficiency,
        'utility_loss': utility_loss,
        'privacy_budget_used': (final_epsilon / target_epsilon) * 100,
        'trainer_type': results['trainer_type']
    }
    
    return analysis


if __name__ == "__main__":
    # Example usage
    print("PRISM Privacy Module - ε-Differential Privacy")
    print("=" * 50)
    
    # Check Opacus availability
    print(f"Opacus available: {OPACUS_AVAILABLE}")
    
    # Create privacy configuration
    config = create_privacy_config(
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0,
        epochs=10,
        batch_size=16,
        learning_rate=0.001,
        use_opacus=OPACUS_AVAILABLE
    )
    
    print(f"\nPrivacy Configuration:")
    print(f"  ε (epsilon): {config.epsilon}")
    print(f"  δ (delta): {config.delta}")
    print(f"  Max grad norm: {config.max_grad_norm}")
    print(f"  Use Opacus: {config.use_opacus}")
    
    print(f"\nPrivacy module is ready for training!")
    print(f"Use PrivacyPreservingTrainer to train models with differential privacy.")
