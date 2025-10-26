"""
Optimized STGCN implementation for GPU training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import math

class OptimizedSTGCNBlock(nn.Module):
    """GPU-optimized ST-GCN block with vectorized operations."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 adjacency_matrix: torch.Tensor, 
                 temporal_kernel_size: int = 3,
                 dropout: float = 0.1):
        super(OptimizedSTGCNBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.adjacency_matrix = adjacency_matrix
        
        # Spatial convolution (vectorized)
        self.spatial_conv = nn.Linear(in_channels, out_channels, bias=False)
        
        # Temporal convolution (1D conv)
        self.temporal_conv = nn.Conv1d(
            out_channels, out_channels, 
            kernel_size=temporal_kernel_size,
            padding=temporal_kernel_size//2,
            bias=False
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optimized forward pass with vectorized operations.
        
        Args:
            x: Input tensor (batch_size, num_joints, in_channels, time_steps)
            
        Returns:
            Output tensor (batch_size, num_joints, out_channels, time_steps)
        """
        batch_size, num_joints, in_channels, time_steps = x.shape
        
        # Store residual
        residual = x
        if self.residual is not None:
            residual = self.residual(residual.transpose(1, 2)).transpose(1, 2)
        
        # Vectorized spatial convolution
        # Reshape: (batch_size, num_joints, in_channels, time_steps) -> (batch_size * time_steps, num_joints, in_channels)
        x_reshaped = x.permute(0, 3, 1, 2).contiguous().view(-1, num_joints, in_channels)
        
        # Apply spatial convolution
        x_spatial = self.spatial_conv(x_reshaped)  # (batch_size * time_steps, num_joints, out_channels)
        
        # Apply adjacency matrix (vectorized)
        adj = self.adjacency_matrix.to(x.device)
        x_spatial = torch.bmm(adj.unsqueeze(0).expand(batch_size * time_steps, -1, -1), x_spatial)
        
        # Reshape back: (batch_size * time_steps, num_joints, out_channels) -> (batch_size, time_steps, num_joints, out_channels)
        x_spatial = x_spatial.view(batch_size, time_steps, num_joints, self.out_channels)
        
        # Transpose for temporal convolution: (batch_size, num_joints, out_channels, time_steps)
        x_spatial = x_spatial.permute(0, 2, 3, 1).contiguous()
        
        # Apply temporal convolution (vectorized across joints)
        # Reshape: (batch_size, num_joints, out_channels, time_steps) -> (batch_size * num_joints, out_channels, time_steps)
        x_temporal = x_spatial.view(batch_size * num_joints, self.out_channels, time_steps)
        
        # Apply temporal convolution
        x_temporal = self.temporal_conv(x_temporal)
        x_temporal = self.batch_norm(x_temporal)
        x_temporal = self.relu(x_temporal)
        x_temporal = self.dropout(x_temporal)
        
        # Reshape back: (batch_size * num_joints, out_channels, time_steps) -> (batch_size, num_joints, out_channels, time_steps)
        x_temporal = x_temporal.view(batch_size, num_joints, self.out_channels, time_steps)
        
        # Add residual connection
        x_temporal = x_temporal + residual
        
        return x_temporal


class OptimizedSTGCN_PRISM(nn.Module):
    """GPU-optimized STGCN for fast training."""
    
    def __init__(self, 
                 num_joints: int = 25,  # NTU RGB+D has 25 joints
                 in_channels: int = 3,  # x, y, z coordinates
                 num_classes: int = 60,  # NTU RGB+D has 60 classes
                 hidden_channels: List[int] = [64, 128, 256],
                 temporal_kernel_sizes: List[int] = [3, 3, 3],
                 dropout: float = 0.1,
                 use_attention: bool = False):
        super(OptimizedSTGCN_PRISM, self).__init__()
        
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        
        # Create NTU RGB+D skeleton adjacency matrix
        self.adjacency_matrix = self._create_ntu_adjacency_matrix()
        
        # Input projection
        self.input_projection = nn.Linear(in_channels, hidden_channels[0])
        
        # ST-GCN blocks
        self.stgcn_blocks = nn.ModuleList()
        in_ch = hidden_channels[0]
        
        for i, (out_ch, temp_kernel) in enumerate(zip(hidden_channels, temporal_kernel_sizes)):
            self.stgcn_blocks.append(
                OptimizedSTGCNBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    adjacency_matrix=self.adjacency_matrix,
                    temporal_kernel_size=temp_kernel,
                    dropout=dropout
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
    
    def _create_ntu_adjacency_matrix(self) -> torch.Tensor:
        """Create adjacency matrix for NTU RGB+D skeleton."""
        # NTU RGB+D skeleton connections
        skeleton_links = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Head
            (0, 5), (5, 6), (6, 7), (7, 8),  # Left arm
            (0, 9), (9, 10), (10, 11), (11, 12),  # Right arm
            (0, 13), (13, 14), (14, 15), (15, 16),  # Torso
            (13, 17), (17, 18), (18, 19), (19, 20),  # Left leg
            (13, 21), (21, 22), (22, 23), (23, 24),  # Right leg
        ]
        
        # Create adjacency matrix
        num_joints = 25
        adj_matrix = torch.zeros(num_joints, num_joints)
        
        for i, j in skeleton_links:
            if i < num_joints and j < num_joints:
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0  # Make symmetric
        
        # Add self-connections
        adj_matrix.fill_diagonal_(1.0)
        
        return adj_matrix
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optimized forward pass.
        
        Args:
            x: Input tensor (batch_size, sequence_length, num_joints, in_channels)
            
        Returns:
            Output tensor (batch_size, num_classes)
        """
        batch_size, sequence_length, num_joints, in_channels = x.shape
        
        # Transpose to (batch_size, num_joints, in_channels, sequence_length)
        x = x.transpose(1, 3).transpose(2, 3)
        
        # Input projection
        x = x.transpose(1, 2)  # (batch_size, in_channels, num_joints, sequence_length)
        x = x.permute(0, 2, 1, 3)  # (batch_size, num_joints, in_channels, sequence_length)
        x = x.contiguous().view(-1, in_channels)  # (batch_size * num_joints * sequence_length, in_channels)
        x = self.input_projection(x)  # (batch_size * num_joints * sequence_length, hidden_channels[0])
        x = x.view(batch_size, num_joints, self.hidden_channels[0], sequence_length)
        
        # Apply ST-GCN blocks
        for stgcn_block in self.stgcn_blocks:
            x = stgcn_block(x)
        
        # Global pooling
        x = x.permute(0, 2, 1, 3)  # (batch_size, out_channels, num_joints, sequence_length)
        x = self.global_pool(x)  # (batch_size, out_channels, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (batch_size, out_channels)
        
        # Classification
        output = self.classifier(x)
        
        return output