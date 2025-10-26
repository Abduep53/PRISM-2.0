"""
PRISM: Baseline Models for Anonymous Human Action Recognition

This module contains the baseline RNN models for pose sequence classification,
including LSTM and GRU architectures with training functionality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch Geometric imports
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv
    from torch_geometric.data import Data, Batch
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: PyTorch Geometric not available. STGCN_PRISM will use fallback implementation.")


def get_human_skeleton_adjacency_matrix() -> torch.Tensor:
    """
    Create adjacency matrix for human skeleton based on MediaPipe pose landmarks.
    
    Returns:
        Adjacency matrix of shape (33, 33) representing human joint connections
    """
    # MediaPipe pose landmark indices
    # 0-10: Face landmarks (not used for body pose)
    # 11-16: Upper body (shoulders, elbows, wrists)
    # 17-22: Lower body (hips, knees, ankles)
    # 23-24: Feet
    # 25-32: Hands (not used for main body pose)
    
    # Define skeleton connections based on human anatomy
    skeleton_links = [
        # Head and neck
        (0, 1), (1, 2), (2, 3), (3, 7),  # Face outline
        (0, 4), (1, 5), (2, 6), (3, 7),  # Face features
        
        # Upper body
        (11, 12),  # Shoulders
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (11, 23), (12, 24),  # Shoulder to hip
        
        # Lower body
        (23, 24),  # Hips
        (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
        (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
        
        # Additional connections for better graph structure
        (11, 23), (12, 24),  # Torso
        (13, 15), (14, 16),  # Arms
        (25, 27), (26, 28),  # Thighs
        (27, 29), (28, 30),  # Shins
        (29, 31), (30, 32),  # Feet
    ]
    
    # Create adjacency matrix
    num_joints = 33
    adj_matrix = torch.zeros(num_joints, num_joints)
    
    for i, j in skeleton_links:
        if i < num_joints and j < num_joints:
            adj_matrix[i, j] = 1.0
            adj_matrix[j, i] = 1.0  # Make symmetric
    
    # Add self-connections
    adj_matrix.fill_diagonal_(1.0)
    
    return adj_matrix


class TemporalConvolutionalLayer(nn.Module):
    """
    Temporal Convolutional Layer for time-series analysis of pose sequences.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 padding: int = 1, dilation: int = 1):
        """
        Initialize temporal convolutional layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of temporal convolution kernel
            stride: Stride of convolution
            padding: Padding for convolution
            dilation: Dilation rate for convolution
        """
        super(TemporalConvolutionalLayer, self).__init__()
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal convolution.
        
        Args:
            x: Input tensor of shape (batch_size, channels, time_steps)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, time_steps)
        """
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class GraphConvolutionalLayer(nn.Module):
    """
    Graph Convolutional Layer for spatial analysis of pose joints.
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 adjacency_matrix: torch.Tensor, bias: bool = True):
        """
        Initialize graph convolutional layer.
        
        Args:
            in_features: Number of input features per node
            out_features: Number of output features per node
            adjacency_matrix: Fixed adjacency matrix for the graph
            bias: Whether to use bias term
        """
        super(GraphConvolutionalLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.adjacency_matrix = adjacency_matrix
        
        # Linear transformation
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize layer weights."""
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through graph convolution.
        
        Args:
            x: Input tensor of shape (batch_size, num_nodes, in_features)
            
        Returns:
            Output tensor of shape (batch_size, num_nodes, out_features)
        """
        batch_size, num_nodes, in_features = x.shape
        
        # Apply linear transformation
        x = self.linear(x)  # (batch_size, num_nodes, out_features)
        
        # Apply graph convolution: A * X
        # Reshape for batch matrix multiplication
        x = x.view(batch_size * num_nodes, self.out_features)
        adj = self.adjacency_matrix.to(x.device)
        
        # Graph convolution: A * X
        x = torch.matmul(adj, x.view(batch_size, num_nodes, self.out_features))
        x = x.view(batch_size, num_nodes, self.out_features)
        
        return x


class STGCNBlock(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Block combining spatial and temporal convolutions.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 adjacency_matrix: torch.Tensor, 
                 temporal_kernel_size: int = 3,
                 spatial_dropout: float = 0.1,
                 temporal_dropout: float = 0.1):
        """
        Initialize ST-GCN block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            adjacency_matrix: Fixed adjacency matrix for spatial convolution
            temporal_kernel_size: Size of temporal convolution kernel
            spatial_dropout: Dropout rate for spatial convolution
            temporal_dropout: Dropout rate for temporal convolution
        """
        super(STGCNBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Spatial graph convolution
        self.spatial_conv = GraphConvolutionalLayer(
            in_features=in_channels,
            out_features=out_channels,
            adjacency_matrix=adjacency_matrix
        )
        
        # Temporal convolution
        self.temporal_conv = TemporalConvolutionalLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=temporal_kernel_size
        )
        
        # Batch normalization and activation
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Dropout layers
        self.spatial_dropout = nn.Dropout(spatial_dropout)
        self.temporal_dropout = nn.Dropout(temporal_dropout)
        
        # Residual connection
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ST-GCN block.
        
        Args:
            x: Input tensor of shape (batch_size, num_nodes, in_channels, time_steps)
            
        Returns:
            Output tensor of shape (batch_size, num_nodes, out_channels, time_steps)
        """
        batch_size, num_nodes, in_channels, time_steps = x.shape
        
        # Store residual
        residual = x
        if self.residual is not None:
            residual = self.residual(residual.transpose(1, 2)).transpose(1, 2)
        
        # Spatial convolution: process each time step
        spatial_outputs = []
        for t in range(time_steps):
            # Extract features for time step t
            x_t = x[:, :, :, t]  # (batch_size, num_nodes, in_channels)
            
            # Apply spatial graph convolution
            x_t = self.spatial_conv(x_t)  # (batch_size, num_nodes, out_channels)
            x_t = self.spatial_dropout(x_t)
            
            spatial_outputs.append(x_t)
        
        # Stack temporal outputs
        x = torch.stack(spatial_outputs, dim=-1)  # (batch_size, num_nodes, out_channels, time_steps)
        
        # Temporal convolution: process each node
        temporal_outputs = []
        for n in range(num_nodes):
            # Extract features for node n
            x_n = x[:, n, :, :]  # (batch_size, out_channels, time_steps)
            
            # Apply temporal convolution
            x_n = self.temporal_conv(x_n)  # (batch_size, out_channels, time_steps)
            x_n = self.temporal_dropout(x_n)
            
            temporal_outputs.append(x_n)
        
        # Stack spatial outputs
        x = torch.stack(temporal_outputs, dim=1)  # (batch_size, num_nodes, out_channels, time_steps)
        
        # Add residual connection
        x = x + residual
        
        # Apply batch normalization and activation
        x = x.transpose(1, 2).contiguous()  # (batch_size, out_channels, num_nodes, time_steps)
        x = x.view(batch_size, self.out_channels, -1)  # (batch_size, out_channels, num_nodes * time_steps)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = x.view(batch_size, self.out_channels, num_nodes, time_steps)
        x = x.transpose(1, 2).contiguous()  # (batch_size, num_nodes, out_channels, time_steps)
        
        return x


class STGCN_PRISM(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Network for PRISM action recognition.
    
    This model processes pose sequences by:
    1. Applying spatial graph convolutions to capture joint relationships
    2. Applying temporal convolutions to capture motion patterns
    3. Combining spatial and temporal features for action classification
    """
    
    def __init__(self, 
                 num_joints: int = 33,
                 in_channels: int = 4,  # x, y, z, confidence
                 num_classes: int = 101,
                 hidden_channels: List[int] = [64, 128, 256],
                 temporal_kernel_sizes: List[int] = [3, 3, 3],
                 dropout: float = 0.1,
                 use_attention: bool = False):
        """
        Initialize STGCN_PRISM model.
        
        Args:
            num_joints: Number of pose joints (33 for MediaPipe)
            in_channels: Number of input channels per joint (4 for x,y,z,confidence)
            num_classes: Number of action classes
            hidden_channels: List of hidden channel sizes for each ST-GCN block
            temporal_kernel_sizes: List of temporal kernel sizes for each block
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super(STGCN_PRISM, self).__init__()
        
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.use_attention = use_attention
        
        # Get human skeleton adjacency matrix
        self.adjacency_matrix = get_human_skeleton_adjacency_matrix()
        
        # Input projection
        self.input_projection = nn.Linear(in_channels, hidden_channels[0])
        
        # ST-GCN blocks
        self.stgcn_blocks = nn.ModuleList()
        in_ch = hidden_channels[0]
        
        for i, (out_ch, temp_kernel) in enumerate(zip(hidden_channels, temporal_kernel_sizes)):
            self.stgcn_blocks.append(
                STGCNBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    adjacency_matrix=self.adjacency_matrix,
                    temporal_kernel_size=temp_kernel,
                    spatial_dropout=dropout,
                    temporal_dropout=dropout
                )
            )
            in_ch = out_ch
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_channels[-1],
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1] // 2, hidden_channels[-1] // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1] // 4, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through STGCN_PRISM model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_joints, in_channels)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        batch_size, sequence_length, num_joints, in_channels = x.shape
        
        # Transpose to (batch_size, num_joints, in_channels, sequence_length)
        x = x.transpose(1, 3).transpose(2, 3)  # (batch_size, num_joints, in_channels, sequence_length)
        
        # Input projection
        x = x.transpose(1, 2)  # (batch_size, in_channels, num_joints, sequence_length)
        x = x.permute(0, 2, 1, 3)  # (batch_size, num_joints, in_channels, sequence_length)
        x = x.contiguous().view(-1, in_channels)  # (batch_size * num_joints * sequence_length, in_channels)
        x = self.input_projection(x)  # (batch_size * num_joints * sequence_length, hidden_channels[0])
        x = x.view(batch_size, num_joints, self.hidden_channels[0], sequence_length)
        
        # Apply ST-GCN blocks
        for stgcn_block in self.stgcn_blocks:
            x = stgcn_block(x)  # (batch_size, num_joints, out_channels, sequence_length)
        
        # Global pooling
        x = x.permute(0, 2, 1, 3)  # (batch_size, out_channels, num_joints, sequence_length)
        x = self.global_pool(x)  # (batch_size, out_channels, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (batch_size, out_channels)
        
        # Attention mechanism (optional)
        if self.use_attention:
            # Reshape for attention
            x_att = x.unsqueeze(1)  # (batch_size, 1, out_channels)
            x_att, _ = self.attention(x_att, x_att, x_att)
            x = x_att.squeeze(1)  # (batch_size, out_channels)
        
        # Classification
        output = self.classifier(x)  # (batch_size, num_classes)
        
        return output


class BaselineLSTM(nn.Module):
    """
    2-layer LSTM baseline model for pose sequence classification.
    
    Architecture:
    - Input: (batch_size, sequence_length, 33, 4) pose sequences
    - Flatten: (batch_size, sequence_length, 132) 
    - LSTM layers: 2-layer LSTM with dropout
    - Classifier: Linear layers for 101-class classification
    """
    
    def __init__(self, 
                 input_size: int = 132,  # 33 landmarks * 4 features
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 101,
                 dropout: float = 0.3,
                 bidirectional: bool = False):
        """
        Initialize the LSTM model.
        
        Args:
            input_size: Input feature size (33 * 4 = 132)
            hidden_size: Hidden state size for LSTM
            num_layers: Number of LSTM layers
            num_classes: Number of action classes
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(BaselineLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # Special initialization for LSTM weights
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, 33, 4)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Flatten pose landmarks: (batch_size, seq_len, 33, 4) -> (batch_size, seq_len, 132)
        x = x.view(batch_size, x.size(1), -1)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output from the sequence
        # lstm_out shape: (batch_size, seq_len, hidden_size * directions)
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size * directions)
        
        # Classification
        output = self.classifier(last_output)
        
        return output


class BaselineGRU(nn.Module):
    """
    2-layer GRU baseline model for pose sequence classification.
    
    Architecture:
    - Input: (batch_size, sequence_length, 33, 4) pose sequences
    - Flatten: (batch_size, sequence_length, 132)
    - GRU layers: 2-layer GRU with dropout
    - Classifier: Linear layers for 101-class classification
    """
    
    def __init__(self, 
                 input_size: int = 132,  # 33 landmarks * 4 features
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 101,
                 dropout: float = 0.3,
                 bidirectional: bool = False):
        """
        Initialize the GRU model.
        
        Args:
            input_size: Input feature size (33 * 4 = 132)
            hidden_size: Hidden state size for GRU
            num_layers: Number of GRU layers
            num_classes: Number of action classes
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional GRU
        """
        super(BaselineGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate GRU output size
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    # Special initialization for GRU weights
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, 33, 4)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Flatten pose landmarks: (batch_size, seq_len, 33, 4) -> (batch_size, seq_len, 132)
        x = x.view(batch_size, x.size(1), -1)
        
        # GRU forward pass
        gru_out, hidden = self.gru(x)
        
        # Use the last output from the sequence
        # gru_out shape: (batch_size, seq_len, hidden_size * directions)
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_size * directions)
        
        # Classification
        output = self.classifier(last_output)
        
        return output


class ModelTrainer:
    """
    Training utility class for baseline models.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            device: Device to run training on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, 
                   dataloader: DataLoader, 
                   optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Get data
            pose_sequences = batch['pose_sequence'].to(self.device)
            labels = batch['label'].squeeze().to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(pose_sequences)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, 
                      dataloader: DataLoader, 
                      criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate the model for one epoch.
        
        Args:
            dataloader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Get data
                pose_sequences = batch['pose_sequence'].to(self.device)
                labels = batch['label'].squeeze().to(self.device)
                
                # Forward pass
                outputs = self.model(pose_sequences)
                loss = criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Training Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def train_baseline(model_type: str = 'lstm',
                  train_loader: DataLoader = None,
                  val_loader: DataLoader = None,
                  num_epochs: int = 50,
                  learning_rate: float = 0.001,
                  weight_decay: float = 1e-5,
                  hidden_size: int = 128,
                  num_layers: int = 2,
                  dropout: float = 0.3,
                  bidirectional: bool = False,
                  num_classes: int = 101,
                  device: str = 'auto',
                  save_model: bool = True,
                  model_save_path: str = 'baseline_model.pth',
                  # STGCN specific parameters
                  hidden_channels: List[int] = [64, 128, 256],
                  temporal_kernel_sizes: List[int] = [3, 3, 3],
                  use_attention: bool = False) -> Dict:
    """
    Train a baseline RNN model for action classification.
    
    Args:
        model_type: Type of model ('lstm' or 'gru')
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        hidden_size: Hidden state size
        num_layers: Number of RNN layers
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional RNN
        num_classes: Number of action classes
        device: Device to use ('auto', 'cpu', or 'cuda')
        save_model: Whether to save the trained model
        model_save_path: Path to save the model
        
    Returns:
        Dictionary containing training history and model info
    """
    # Set device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Training on device: {device}")
    print(f"Model type: {model_type.upper()}")
    print(f"Number of classes: {num_classes}")
    print(f"Hidden size: {hidden_size}")
    print(f"Number of layers: {num_layers}")
    print(f"Bidirectional: {bidirectional}")
    print("-" * 50)
    
    # Create model
    if model_type.lower() == 'lstm':
        model = BaselineLSTM(
            input_size=132,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=bidirectional
        )
    elif model_type.lower() == 'gru':
        model = BaselineGRU(
            input_size=132,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=bidirectional
        )
    elif model_type.lower() == 'stgcn':
        model = STGCN_PRISM(
            num_joints=33,
            in_channels=4,
            num_classes=num_classes,
            hidden_channels=hidden_channels,
            temporal_kernel_sizes=temporal_kernel_sizes,
            dropout=dropout,
            use_attention=use_attention
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from 'lstm', 'gru', or 'stgcn'")
    
    # Initialize trainer
    trainer = ModelTrainer(model, device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_acc = 0.0
    start_time = time.time()
    
    print("Starting training...")
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'LR':<10}")
    print("-" * 70)
    
    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, criterion)
        trainer.train_losses.append(train_loss)
        trainer.train_accuracies.append(train_acc)
        
        # Validation
        if val_loader is not None:
            val_loss, val_acc = trainer.validate_epoch(val_loader, criterion)
            trainer.val_losses.append(val_loss)
            trainer.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_model:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'val_acc': val_acc,
                        'model_config': {
                            'model_type': model_type,
                            'hidden_size': hidden_size,
                            'num_layers': num_layers,
                            'num_classes': num_classes,
                            'dropout': dropout,
                            'bidirectional': bidirectional
                        }
                    }, model_save_path)
        else:
            val_loss, val_acc = 0.0, 0.0
            trainer.val_losses.append(val_loss)
            trainer.val_accuracies.append(val_acc)
        
        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"{epoch+1:<6} {train_loss:<12.4f} {train_acc:<12.2f} {val_loss:<12.4f} {val_acc:<12.2f} {current_lr:<10.6f}")
    
    training_time = time.time() - start_time
    print("-" * 70)
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training history
    trainer.plot_training_history(f"{model_save_path}_training_history.png")
    
    # Return training results
    results = {
        'model': model,
        'trainer': trainer,
        'best_val_acc': best_val_acc,
        'training_time': training_time,
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'train_accuracies': trainer.train_accuracies,
        'val_accuracies': trainer.val_accuracies,
        'model_config': {
            'model_type': model_type,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_classes': num_classes,
            'dropout': dropout,
            'bidirectional': bidirectional
        }
    }
    
    return results


def evaluate_model(model: nn.Module, 
                  test_loader: DataLoader, 
                  device: str = 'cpu',
                  class_names: Optional[List[str]] = None) -> Dict:
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run evaluation on
        class_names: Optional list of class names for detailed reporting
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            pose_sequences = batch['pose_sequence'].to(device)
            labels = batch['label'].squeeze().to(device)
            
            outputs = model(pose_sequences)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    # Calculate metrics
    accuracy = total_correct / total_samples
    accuracy_score_val = accuracy_score(all_labels, all_predictions)
    
    # Classification report
    if class_names:
        report = classification_report(
            all_labels, all_predictions, 
            target_names=class_names, 
            output_dict=True
        )
    else:
        report = classification_report(
            all_labels, all_predictions, 
            output_dict=True
        )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'accuracy': accuracy,
        'accuracy_score': accuracy_score_val,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    return results


if __name__ == "__main__":
    # Example usage
    print("PRISM Baseline Models")
    print("=" * 50)
    
    # Create dummy data for demonstration
    from torch.utils.data import TensorDataset, DataLoader
    
    # Generate dummy pose sequences
    batch_size = 8
    sequence_length = 30
    num_samples = 100
    num_classes = 101
    
    # Create dummy data
    pose_sequences = torch.randn(num_samples, sequence_length, 33, 4)
    labels = torch.randint(0, num_classes, (num_samples, 1))
    
    # Create dataset and data loader
    dataset = TensorDataset(pose_sequences, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Created dummy dataset with {num_samples} samples")
    print(f"Pose sequence shape: {pose_sequences.shape}")
    print(f"Number of classes: {num_classes}")
    
    # Test model creation
    print("\nTesting model creation...")
    
    # Test LSTM model
    lstm_model = BaselineLSTM(num_classes=num_classes)
    print(f"LSTM model created: {sum(p.numel() for p in lstm_model.parameters())} parameters")
    
    # Test GRU model
    gru_model = BaselineGRU(num_classes=num_classes)
    print(f"GRU model created: {sum(p.numel() for p in gru_model.parameters())} parameters")
    
    # Test STGCN model
    stgcn_model = STGCN_PRISM(num_classes=num_classes)
    print(f"STGCN model created: {sum(p.numel() for p in stgcn_model.parameters())} parameters")
    
    # Test forward pass
    test_input = torch.randn(2, sequence_length, 33, 4)
    lstm_output = lstm_model(test_input)
    gru_output = gru_model(test_input)
    stgcn_output = stgcn_model(test_input)
    
    print(f"LSTM output shape: {lstm_output.shape}")
    print(f"GRU output shape: {gru_output.shape}")
    print(f"STGCN output shape: {stgcn_output.shape}")
    
    print("\nModels are ready for training!")
    print("Use train_baseline() function to train the models.")
    print("Available models: 'lstm', 'gru', 'stgcn'")
