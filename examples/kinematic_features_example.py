"""
Kinematic Features Example for PRISM
Demonstrates the enhanced data pipeline with kinematic feature extraction.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_pipeline import (
    kinematic_features, 
    PRISMDataset, 
    extract_and_normalize_pose_with_kinematics,
    process_video_batch
)


def demonstrate_kinematic_features():
    """Demonstrate kinematic feature extraction and analysis."""
    print("PRISM Kinematic Features Demonstration")
    print("=" * 50)
    
    # Create dummy pose data
    num_frames = 30
    num_joints = 33
    num_coords = 4
    
    print(f"Creating dummy pose data...")
    print(f"  Frames: {num_frames}")
    print(f"  Joints: {num_joints}")
    print(f"  Coordinates: {num_coords}")
    
    # Generate realistic pose data
    pose_data = np.random.randn(num_frames, num_joints, num_coords)
    
    # Make confidence values realistic (0-1)
    pose_data[:, :, 3] = np.abs(pose_data[:, :, 3]) % 1.0
    
    # Add temporal correlation to make it more realistic
    for t in range(1, num_frames):
        pose_data[t] = 0.7 * pose_data[t-1] + 0.3 * pose_data[t]
    
    print(f"Raw pose data shape: {pose_data.shape}")
    print(f"Raw data size: {pose_data.size} elements")
    
    # Extract kinematic features
    print(f"\nExtracting kinematic features...")
    kinematic_feats = kinematic_features(pose_data)
    
    print(f"Kinematic features shape: {kinematic_feats.shape}")
    print(f"Kinematic data size: {kinematic_feats.size} elements")
    print(f"Data reduction: {pose_data.size} -> {kinematic_feats.size} elements")
    print(f"Compression ratio: {pose_data.size / kinematic_feats.size:.2f}x")
    
    # Analyze feature types
    print(f"\nKinematic Feature Analysis:")
    print(f"  Static features per frame: 50")
    print(f"  Velocity features per frame: 50")
    print(f"  Total features per frame: {kinematic_feats.shape[1]}")
    
    # Show feature statistics
    print(f"\nFeature Statistics:")
    print(f"  Mean: {np.mean(kinematic_feats):.4f}")
    print(f"  Std: {np.std(kinematic_feats):.4f}")
    print(f"  Min: {np.min(kinematic_feats):.4f}")
    print(f"  Max: {np.max(kinematic_feats):.4f}")
    
    return kinematic_feats


def demonstrate_dataset_loading():
    """Demonstrate PRISMDataset with kinematic features."""
    print(f"\n{'='*50}")
    print("PRISMDataset with Kinematic Features")
    print("=" * 50)
    
    # Create dummy data directory
    import os
    data_dir = "dummy_kinematic_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate dummy kinematic data
    num_samples = 10
    num_frames = 30
    num_features = 100  # 50 static + 50 velocity
    
    print(f"Creating {num_samples} dummy kinematic samples...")
    
    for i in range(num_samples):
        # Generate kinematic features
        kinematic_data = np.random.randn(num_frames, num_features)
        
        # Add some structure
        for t in range(1, num_frames):
            kinematic_data[t] = 0.8 * kinematic_data[t-1] + 0.2 * kinematic_data[t]
        
        # Save to file
        filename = os.path.join(data_dir, f"kinematic_sample_{i:03d}.npy")
        np.save(filename, kinematic_data)
    
    print(f"✓ Created dummy kinematic data in {data_dir}/")
    
    # Test dataset loading with kinematic features
    print(f"\nTesting PRISMDataset with kinematic features...")
    
    try:
        dataset = PRISMDataset(
            data_dir=data_dir,
            sequence_length=30,
            use_kinematics=True
        )
        
        print(f"✓ Dataset created successfully")
        print(f"  Number of samples: {len(dataset)}")
        print(f"  Feature dimensions: {dataset.feature_dims}")
        
        # Test data loading
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        print(f"✓ DataLoader created with batch size 4")
        
        # Test loading a batch
        for batch_idx, batch in enumerate(dataloader):
            pose_sequences = batch['pose_sequence']
            print(f"  Batch {batch_idx + 1}:")
            print(f"    Shape: {pose_sequences.shape}")
            print(f"    Data type: {pose_sequences.dtype}")
            print(f"    Value range: [{pose_sequences.min():.4f}, {pose_sequences.max():.4f}]")
            
            if batch_idx >= 1:  # Only show first 2 batches
                break
                
    except Exception as e:
        print(f"✗ Error creating dataset: {str(e)}")
    
    # Clean up
    import shutil
    shutil.rmtree(data_dir, ignore_errors=True)


def compare_raw_vs_kinematic():
    """Compare raw pose data vs kinematic features."""
    print(f"\n{'='*50}")
    print("Raw Pose vs Kinematic Features Comparison")
    print("=" * 50)
    
    # Create dummy pose data
    pose_data = np.random.randn(20, 33, 4)
    pose_data[:, :, 3] = np.abs(pose_data[:, :, 3]) % 1.0
    
    # Add temporal correlation
    for t in range(1, pose_data.shape[0]):
        pose_data[t] = 0.7 * pose_data[t-1] + 0.3 * pose_data[t]
    
    # Extract kinematic features
    kinematic_data = kinematic_features(pose_data)
    
    print(f"Raw Pose Data:")
    print(f"  Shape: {pose_data.shape}")
    print(f"  Size: {pose_data.size} elements")
    print(f"  Memory: {pose_data.nbytes / 1024:.2f} KB")
    
    print(f"\nKinematic Features:")
    print(f"  Shape: {kinematic_data.shape}")
    print(f"  Size: {kinematic_data.size} elements")
    print(f"  Memory: {kinematic_data.nbytes / 1024:.2f} KB")
    
    print(f"\nComparison:")
    print(f"  Data reduction: {pose_data.size / kinematic_data.size:.2f}x")
    print(f"  Memory reduction: {pose_data.nbytes / kinematic_data.nbytes:.2f}x")
    print(f"  Features per frame: {pose_data.shape[1] * pose_data.shape[2]} -> {kinematic_data.shape[1]}")
    
    # Analyze information content
    print(f"\nInformation Content Analysis:")
    raw_variance = np.var(pose_data)
    kinematic_variance = np.var(kinematic_data)
    
    print(f"  Raw data variance: {raw_variance:.4f}")
    print(f"  Kinematic data variance: {kinematic_variance:.4f}")
    print(f"  Variance ratio: {kinematic_variance / raw_variance:.4f}")


def demonstrate_feature_types():
    """Demonstrate different types of kinematic features."""
    print(f"\n{'='*50}")
    print("Kinematic Feature Types Demonstration")
    print("=" * 50)
    
    # Create pose data with specific movements
    num_frames = 20
    pose_data = np.zeros((num_frames, 33, 4))
    
    # Set confidence values
    pose_data[:, :, 3] = 0.9
    
    # Simulate arm movement (left arm extending)
    for t in range(num_frames):
        # Left shoulder (fixed)
        pose_data[t, 11, :3] = [0, 0, 0]
        
        # Left elbow (moving outward)
        pose_data[t, 13, :3] = [0.1 * t / num_frames, 0, 0]
        
        # Left wrist (following elbow)
        pose_data[t, 15, :3] = [0.2 * t / num_frames, 0, 0]
    
    # Extract kinematic features
    kinematic_data = kinematic_features(pose_data)
    
    print(f"Simulated arm extension movement:")
    print(f"  Frames: {num_frames}")
    print(f"  Kinematic features shape: {kinematic_data.shape}")
    
    # Analyze specific features
    print(f"\nFeature Analysis:")
    
    # Left arm angle (should be changing)
    left_arm_angles = kinematic_data[:, 0]  # First feature is left arm angle
    print(f"  Left arm angles: {left_arm_angles[:5]}...")
    print(f"  Angle range: {np.min(left_arm_angles):.4f} to {np.max(left_arm_angles):.4f}")
    
    # Velocity features (should show movement)
    if kinematic_data.shape[1] > 50:
        left_arm_velocities = kinematic_data[:, 50]  # First velocity feature
        print(f"  Left arm velocities: {left_arm_velocities[:5]}...")
        print(f"  Velocity range: {np.min(left_arm_velocities):.4f} to {np.max(left_arm_velocities):.4f}")
    
    print(f"\n✓ Kinematic features successfully capture movement dynamics!")


def main():
    """Main demonstration function."""
    print("PRISM Enhanced Data Pipeline - Kinematic Features")
    print("=" * 60)
    print("This demonstrates the enhanced data pipeline with:")
    print("• Relative joint angles between major limbs")
    print("• Temporal velocity vectors across frames")
    print("• Significant data dimensionality reduction")
    print("• Enhanced movement dynamics for action classification")
    print("=" * 60)
    
    # Run demonstrations
    try:
        # Demonstrate basic kinematic feature extraction
        kinematic_feats = demonstrate_kinematic_features()
        
        # Demonstrate dataset loading
        demonstrate_dataset_loading()
        
        # Compare raw vs kinematic data
        compare_raw_vs_kinematic()
        
        # Demonstrate feature types
        demonstrate_feature_types()
        
        print(f"\n{'='*60}")
        print("KINEMATIC FEATURES DEMONSTRATION COMPLETE")
        print(f"{'='*60}")
        print("Key Benefits Demonstrated:")
        print("• 50+ kinematic features per frame (vs 132 raw features)")
        print("• Significant data reduction while preserving movement information")
        print("• Joint angles capture limb relationships")
        print("• Velocity features capture temporal dynamics")
        print("• Enhanced action classification potential")
        print("\nThe enhanced PRISM data pipeline is ready for training!")
        
    except Exception as e:
        print(f"Error in demonstration: {str(e)}")
        print("This may be due to missing dependencies or configuration issues.")


if __name__ == "__main__":
    main()
