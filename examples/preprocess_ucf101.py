"""
Preprocessing script for UCF-101 dataset
Extracts pose landmarks and kinematic features from UCF-101 videos
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_pipeline import extract_and_normalize_pose_with_kinematics, save_kinematic_features

def process_single_video(video_path: str, output_dir: str) -> bool:
    """Process a single video and save kinematic features."""
    try:
        # Extract kinematic features
        features = extract_and_normalize_pose_with_kinematics(video_path)
        
        # Create output filename
        video_name = Path(video_path).stem
        output_path = os.path.join(output_dir, f"{video_name}_kinematic.npy")
        
        # Save features
        save_kinematic_features(features, output_path)
        
        return True
        
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return False

def preprocess_ucf101(data_root: str, output_dir: str, num_workers: int = 4):
    """Preprocess UCF-101 dataset."""
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_extensions = ['.avi', '.mp4', '.mov']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(data_root.glob(f"**/*{ext}"))
    
    print(f"Found {len(video_files)} video files")
    
    # Process videos
    if num_workers > 1:
        # Parallel processing
        with mp.Pool(num_workers) as pool:
            process_func = partial(process_single_video, output_dir=str(output_dir))
            results = list(tqdm(
                pool.imap(process_func, [str(f) for f in video_files]),
                total=len(video_files),
                desc="Processing videos"
            ))
    else:
        # Sequential processing
        results = []
        for video_path in tqdm(video_files, desc="Processing videos"):
            result = process_single_video(str(video_path), str(output_dir))
            results.append(result)
    
    # Print statistics
    successful = sum(results)
    failed = len(results) - successful
    
    print(f"\nPreprocessing completed:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {successful/len(results)*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Preprocess UCF-101 dataset')
    parser.add_argument('--data_root', type=str, required=True, 
                       help='Path to UCF-101 dataset root')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed features')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of worker processes')
    
    args = parser.parse_args()
    
    preprocess_ucf101(
        data_root=args.data_root,
        output_dir=args.output_dir,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main()