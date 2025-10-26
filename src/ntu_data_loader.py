"""
NTU RGB+D Dataset Loader for PRISM
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import pickle
from pathlib import Path

class NTURGBDDataset(Dataset):
    """NTU RGB+D Dataset for PRISM training."""
    
    def __init__(self, 
                 data_path: str,
                 split: str = 'train',  # 'train' or 'val'
                 sequence_length: int = 300,
                 use_kinematics: bool = True,
                 transform=None):
        """
        Initialize NTU RGB+D dataset.
        
        Args:
            data_path: Path to NTU RGB+D dataset
            split: Dataset split ('train' or 'val')
            sequence_length: Fixed sequence length
            use_kinematics: Whether to use kinematic features
            transform: Optional data transformation
        """
        self.data_path = Path(data_path)
        self.split = split
        self.sequence_length = sequence_length
        self.use_kinematics = use_kinematics
        self.transform = transform
        
        # Load data
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load sample paths and labels."""
        samples = []
        
        # NTU RGB+D file structure
        if self.split == 'train':
            data_file = self.data_path / "nturgbd_skeletons_s001_to_s017.h5"
        else:
            data_file = self.data_path / "nturgbd_skeletons_s018_to_s032.h5"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Load HDF5 file
        with h5py.File(data_file, 'r') as f:
            # Get all sample keys
            sample_keys = list(f.keys())
            
            for key in sample_keys:
                # Extract label from key (format: S001C001P001R001A001)
                label = int(key.split('A')[1]) - 1  # Convert to 0-indexed
                
                # Skip invalid samples
                if label < 0 or label >= 60:
                    continue
                
                samples.append((key, label))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """Get sample at index."""
        key, label = self.samples[idx]
        
        # Load skeleton data
        skeleton_data = self._load_skeleton_data(key)
        
        # Apply transformations
        if self.transform:
            skeleton_data = self.transform(skeleton_data)
        
        # Convert to tensor
        skeleton_tensor = torch.FloatTensor(skeleton_data)
        
        return {
            'pose_sequence': skeleton_tensor,
            'label': torch.LongTensor([label])[0]
        }
    
    def _load_skeleton_data(self, key: str) -> np.ndarray:
        """Load skeleton data for a sample."""
        if self.split == 'train':
            data_file = self.data_path / "nturgbd_skeletons_s001_to_s017.h5"
        else:
            data_file = self.data_path / "nturgbd_skeletons_s018_to_s032.h5"
        
        with h5py.File(data_file, 'r') as f:
            # Load skeleton data (N, 25, 3) where N is number of frames
            skeleton_data = f[key][:]
            
            # Normalize skeleton data
            skeleton_data = self._normalize_skeleton(skeleton_data)
            
            # Pad or truncate to fixed length
            skeleton_data = self._pad_or_truncate(skeleton_data)
            
            return skeleton_data
    
    def _normalize_skeleton(self, skeleton_data: np.ndarray) -> np.ndarray:
        """Normalize skeleton data."""
        # Center skeleton around origin
        center = np.mean(skeleton_data, axis=1, keepdims=True)
        skeleton_data = skeleton_data - center
        
        # Scale skeleton
        scale = np.std(skeleton_data)
        if scale > 0:
            skeleton_data = skeleton_data / scale
        
        return skeleton_data
    
    def _pad_or_truncate(self, skeleton_data: np.ndarray) -> np.ndarray:
        """Pad or truncate sequence to fixed length."""
        current_length = skeleton_data.shape[0]
        
        if current_length == self.sequence_length:
            return skeleton_data
        elif current_length > self.sequence_length:
            # Truncate by taking evenly spaced frames
            indices = np.linspace(0, current_length - 1, self.sequence_length, dtype=int)
            return skeleton_data[indices]
        else:
            # Pad with last frame
            padding_needed = self.sequence_length - current_length
            last_frame = skeleton_data[-1:].repeat(padding_needed, axis=0)
            return np.vstack([skeleton_data, last_frame])


def create_ntu_dataloaders(data_path: str,
                          batch_size: int = 32,
                          num_workers: int = 4,
                          sequence_length: int = 300) -> Tuple[DataLoader, DataLoader]:
    """
    Create NTU RGB+D data loaders.
    
    Args:
        data_path: Path to NTU RGB+D dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        sequence_length: Fixed sequence length
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = NTURGBDDataset(
        data_path=data_path,
        split='train',
        sequence_length=sequence_length,
        use_kinematics=True
    )
    
    val_dataset = NTURGBDDataset(
        data_path=data_path,
        split='val',
        sequence_length=sequence_length,
        use_kinematics=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader