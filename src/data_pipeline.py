"""
PRISM: Anonymous Human Action Recognition Data Pipeline

This module provides functionality for extracting and normalizing human pose data
from videos using MediaPipe Pose, with spatial and scale invariance through
center-of-mass normalization, and includes the PyTorch Dataset implementation 
integrated with UCF-101 split files.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Dict, Union
import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.transform import Rotation as R
import re
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# V1: UTILITY FUNCTIONS (Kinematic & Math)
# ==============================================================================

def calculate_joint_angle(point1: np.ndarray, point2: np.ndarray, point3: np.ndarray,
                          conf1: float, conf2: float, conf3: float) -> float:
    """Calculate the angle between three points (joint angle)."""
    if min(conf1, conf2, conf3) < 0.3:
        return 0.0
    
    vec1 = point1 - point2
    vec2 = point3 - point2
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cos_angle = dot_product / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle = np.arccos(cos_angle)
    return angle

def calculate_orientation(point1: np.ndarray, point2: np.ndarray, point3: np.ndarray,
                          conf1: float, conf2: float, conf3: float) -> List[float]:
    """Calculate orientation features between three points."""
    if min(conf1, conf2, conf3) < 0.3:
        return [0.0, 0.0, 0.0]
    
    vec1 = point2 - point1
    vec2 = point3 - point1
    normal = np.cross(vec1, vec2)
    normal_norm = np.linalg.norm(normal)
    
    if normal_norm == 0:
        return [0.0, 0.0, 0.0]
    
    normal = normal / normal_norm
    pitch = np.arcsin(np.clip(normal[1], -1.0, 1.0))
    yaw = np.arctan2(normal[0], normal[2])
    roll = np.arctan2(normal[1], normal[0])
    
    return [pitch, yaw, roll]

def calculate_distance(point1: np.ndarray, point2: np.ndarray,
                      conf1: float, conf2: float) -> float:
    """Calculate distance between two points."""
    if min(conf1, conf2) < 0.3:
        return 0.0
    
    return np.linalg.norm(point1 - point2)

def calculate_temporal_velocities(features: np.ndarray) -> np.ndarray:
    """Calculate temporal velocity features across consecutive frames."""
    num_frames, num_features = features.shape
    
    if num_frames < 2:
        return np.zeros((num_frames, num_features))
    
    velocities = np.diff(features, axis=0)
    velocities = np.vstack([np.zeros((1, num_features)), velocities])
    
    return velocities

def calculate_joint_velocity(joint_coords: np.ndarray, frame_idx: int) -> np.ndarray:
    """Calculate velocity for a specific joint across frames."""
    if frame_idx == 0:
        return np.zeros(3)
    
    return joint_coords[frame_idx] - joint_coords[frame_idx - 1]

def calculate_joint_acceleration(joint_coords: np.ndarray, frame_idx: int) -> np.ndarray:
    """Calculate acceleration for a specific joint across frames."""
    if frame_idx < 2:
        return np.zeros(3)
    
    vel_current = joint_coords[frame_idx] - joint_coords[frame_idx - 1]
    vel_previous = joint_coords[frame_idx - 1] - joint_coords[frame_idx - 2]
    
    return vel_current - vel_previous

def calculate_body_center_of_mass(coords: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Calculate body center of mass weighted by confidence."""
    valid_joints = confidence > 0.3
    if not np.any(valid_joints):
        return np.zeros(3)
    
    weights = confidence[valid_joints]
    weighted_coords = coords[valid_joints] * weights[:, np.newaxis]
    
    return np.sum(weighted_coords, axis=0) / np.sum(weights)

def calculate_limb_symmetry(left_length: float, right_length: float) -> float:
    """Calculate symmetry between left and right limbs."""
    if left_length + right_length < 1e-6:
        return 0.0
    
    return abs(left_length - right_length) / (left_length + right_length)

# ==============================================================================
# V2: CORE COMPONENTS (PoseExtractor & Kinematic Features)
# ==============================================================================

class PoseExtractor:
    """Handles pose extraction and normalization using MediaPipe Pose."""
    
    def __init__(self):
        """Initialize MediaPipe pose detection."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hip_landmarks = [23, 24]
        self.shoulder_landmarks = [11, 12]
        
    def extract_pose_landmarks(self, video_path: str) -> List[np.ndarray]:
        """Extract pose landmarks from video frames."""
        cap = cv2.VideoCapture(video_path)
        landmarks_list = []
        
        if not cap.isOpened():
            print(f"Warning: Could not open video file: {video_path}")
            return landmarks_list
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = np.zeros((33, 4))
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    landmarks[i] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
                landmarks_list.append(landmarks)
            else:
                landmarks_list.append(np.zeros((33, 4)))
            
        cap.release()
        return landmarks_list
    
    def calculate_center_of_mass(self, landmarks: np.ndarray) -> Tuple[float, float, float]:
        """Calculate the center of mass using hip landmarks as reference."""
        left_hip = landmarks[self.hip_landmarks[0]]
        right_hip = landmarks[self.hip_landmarks[1]]
        center_x = (left_hip[0] + right_hip[0]) / 2.0
        center_y = (left_hip[1] + right_hip[1]) / 2.0
        center_z = (left_hip[2] + right_hip[2]) / 2.0
        return center_x, center_y, center_z
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalize landmarks relative to center of mass for spatial invariance."""
        center_x, center_y, center_z = self.calculate_center_of_mass(landmarks)
        normalized_landmarks = landmarks.copy()
        normalized_landmarks[:, 0] = landmarks[:, 0] - center_x
        normalized_landmarks[:, 1] = landmarks[:, 1] - center_y
        normalized_landmarks[:, 2] = landmarks[:, 2] - center_z
        normalized_landmarks[:, 3] = landmarks[:, 3]  # Keep confidence
        return normalized_landmarks
    
    def calculate_scale_factor(self, landmarks: np.ndarray) -> float:
        """Calculate scale factor based on shoulder-hip distance for scale invariance."""
        left_shoulder = landmarks[self.shoulder_landmarks[0]]
        right_shoulder = landmarks[self.shoulder_landmarks[1]]
        left_hip = landmarks[self.hip_landmarks[0]]
        right_hip = landmarks[self.hip_landmarks[1]]
        
        shoulder_width = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
        hip_width = np.linalg.norm(left_hip[:2] - right_hip[:2])
        
        scale_reference = (shoulder_width + hip_width) / 2.0
        
        return 1.0 / scale_reference if scale_reference > 1e-6 else 1.0

    def apply_scale_normalization(self, landmarks: np.ndarray, scale_factor: float) -> np.ndarray:
        """Apply scale normalization to landmarks."""
        scaled_landmarks = landmarks.copy()
        scaled_landmarks[:, :3] *= scale_factor
        return scaled_landmarks

def extract_and_normalize_pose(video_path: str) -> np.ndarray:
    """Extract and normalize pose data from video for spatial and scale invariance."""
    # Валидация входных данных
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise ValueError(f"Unsupported video format: {video_path}")
    
    extractor = PoseExtractor()
    raw_landmarks_list = extractor.extract_pose_landmarks(video_path)
    
    if not raw_landmarks_list:
        print(f"Warning: No pose landmarks detected in {video_path}")
        return np.zeros((0, 33, 4))
    
    normalized_sequences = []
    
    for landmarks in raw_landmarks_list:
        if np.mean(landmarks[:, 3]) < 0.3:
            continue
            
        normalized_landmarks = extractor.normalize_landmarks(landmarks)
        scale_factor = extractor.calculate_scale_factor(landmarks)
        final_landmarks = extractor.apply_scale_normalization(normalized_landmarks, scale_factor)
        
        normalized_sequences.append(final_landmarks)
    
    return np.array(normalized_sequences) if normalized_sequences else np.zeros((0, 33, 4))

# ==============================================================================
# V3: COMPREHENSIVE KINEMATIC FEATURES EXTRACTION
# ==============================================================================

def kinematic_features(normalized_poses: np.ndarray) -> np.ndarray:
    """
    Transform normalized pose coordinates into comprehensive kinematic features.
    
    This function extracts 50 static features + 50 temporal features = 100 total features:
    
    Static Features (50):
    1. Joint Angles (15): Arm, leg, and torso angles
    2. Body Orientation (6): Head and torso orientation (pitch, yaw, roll)
    3. Limb Lengths (5): Arm, leg, and torso lengths
    4. Symmetry Features (2): Arm and leg symmetry
    5. Center of Mass (2): X and Y coordinates
    6. Joint Distances (10): Distances between key joints
    7. Body Proportions (5): Ratios of limb lengths
    8. Spatial Features (5): Body width, height, depth
    
    Temporal Features (50):
    1. Joint Velocities (15): Velocities of key joints
    2. Joint Accelerations (15): Accelerations of key joints
    3. Angular Velocities (10): Velocities of joint angles
    4. Temporal Symmetry (5): Changes in symmetry over time
    5. Motion Intensity (5): Overall motion intensity
    
    Args:
        normalized_poses: Normalized pose data (T, 33, 4) where T=time, 33=joints, 4=[x,y,z,confidence]
        
    Returns:
        kinematic_features: Array of shape (T, 100) containing kinematic features
    """
    num_frames, num_joints, num_coords = normalized_poses.shape
    
    if num_frames == 0:
        return np.zeros((0, 100))
    
    # Validate input
    if num_joints != 33:
        raise ValueError(f"Expected 33 joints (MediaPipe), got {num_joints}")
    
    if num_coords != 4:
        raise ValueError(f"Expected 4 coordinates [x,y,z,confidence], got {num_coords}")
    
    coords = normalized_poses[:, :, :3]  # (T, 33, 3)
    confidence = normalized_poses[:, :, 3]  # (T, 33)
    
    # Initialize feature arrays
    static_features = np.zeros((num_frames, 50))
    temporal_features = np.zeros((num_frames, 50))
    
    # Process each frame
    for frame_idx in range(num_frames):
        frame_coords = coords[frame_idx]  # (33, 3)
        frame_confidence = confidence[frame_idx]  # (33,)
        
        # Skip frame if low confidence
        if np.mean(frame_confidence) < 0.3:
            static_features[frame_idx] = 0.0
            temporal_features[frame_idx] = 0.0
            continue
        
        # ========================================================================
        # STATIC FEATURES (50 features)
        # ========================================================================
        
        feature_idx = 0
        
        # 1. JOINT ANGLES (15 features)
        # Arm angles
        left_arm_angle = calculate_joint_angle(
            frame_coords[11], frame_coords[13], frame_coords[15],  # shoulder-elbow-wrist
            frame_confidence[11], frame_confidence[13], frame_confidence[15]
        )
        right_arm_angle = calculate_joint_angle(
            frame_coords[12], frame_coords[14], frame_coords[16],
            frame_confidence[12], frame_confidence[14], frame_confidence[16]
        )
        
        # Leg angles
        left_leg_angle = calculate_joint_angle(
            frame_coords[23], frame_coords[25], frame_coords[27],  # hip-knee-ankle
            frame_confidence[23], frame_confidence[25], frame_confidence[27]
        )
        right_leg_angle = calculate_joint_angle(
            frame_coords[24], frame_coords[26], frame_coords[28],
            frame_confidence[24], frame_confidence[26], frame_confidence[28]
        )
        
        # Torso angle
        torso_angle = calculate_joint_angle(
            frame_coords[11], frame_coords[23], frame_coords[24],  # shoulder-hip-hip
            frame_confidence[11], frame_confidence[23], frame_confidence[24]
        )
        
        # Additional joint angles (10 more)
        # Head angles
        head_angle_1 = calculate_joint_angle(
            frame_coords[0], frame_coords[1], frame_coords[2],  # nose-left_eye-right_eye
            frame_confidence[0], frame_confidence[1], frame_confidence[2]
        )
        head_angle_2 = calculate_joint_angle(
            frame_coords[0], frame_coords[7], frame_coords[8],  # nose-left_ear-right_ear
            frame_confidence[0], frame_confidence[7], frame_confidence[8]
        )
        
        # Shoulder angles
        left_shoulder_angle = calculate_joint_angle(
            frame_coords[11], frame_coords[12], frame_coords[13],  # left_shoulder-right_shoulder-left_elbow
            frame_confidence[11], frame_confidence[12], frame_confidence[13]
        )
        right_shoulder_angle = calculate_joint_angle(
            frame_coords[12], frame_coords[11], frame_coords[14],  # right_shoulder-left_shoulder-right_elbow
            frame_confidence[12], frame_confidence[11], frame_confidence[14]
        )
        
        # Hip angles
        left_hip_angle = calculate_joint_angle(
            frame_coords[23], frame_coords[24], frame_coords[25],  # left_hip-right_hip-left_knee
            frame_confidence[23], frame_confidence[24], frame_confidence[25]
        )
        right_hip_angle = calculate_joint_angle(
            frame_coords[24], frame_coords[23], frame_coords[26],  # right_hip-left_hip-right_knee
            frame_confidence[24], frame_confidence[23], frame_confidence[26]
        )
        
        # Knee angles
        left_knee_angle = calculate_joint_angle(
            frame_coords[25], frame_coords[27], frame_coords[29],  # left_knee-left_ankle-left_foot
            frame_confidence[25], frame_confidence[27], frame_confidence[29]
        )
        right_knee_angle = calculate_joint_angle(
            frame_coords[26], frame_coords[28], frame_coords[30],  # right_knee-right_ankle-right_foot
            frame_confidence[26], frame_confidence[28], frame_confidence[30]
        )
        
        # Wrist angles
        left_wrist_angle = calculate_joint_angle(
            frame_coords[15], frame_coords[17], frame_coords[19],  # left_wrist-left_pinky-left_index
            frame_confidence[15], frame_confidence[17], frame_confidence[19]
        )
        right_wrist_angle = calculate_joint_angle(
            frame_coords[16], frame_coords[18], frame_coords[20],  # right_wrist-right_pinky-right_index
            frame_confidence[16], frame_confidence[18], frame_confidence[20]
        )
        
        # Ankle angles
        left_ankle_angle = calculate_joint_angle(
            frame_coords[27], frame_coords[29], frame_coords[31],  # left_ankle-left_foot-left_heel
            frame_confidence[27], frame_confidence[29], frame_confidence[31]
        )
        right_ankle_angle = calculate_joint_angle(
            frame_coords[28], frame_coords[30], frame_coords[32],  # right_ankle-right_foot-right_heel
            frame_confidence[28], frame_confidence[30], frame_confidence[32]
        )
        
        static_features[frame_idx, feature_idx:feature_idx+15] = [
            left_arm_angle, right_arm_angle, left_leg_angle, right_leg_angle, torso_angle,
            head_angle_1, head_angle_2, left_shoulder_angle, right_shoulder_angle,
            left_hip_angle, right_hip_angle, left_knee_angle, right_knee_angle,
            left_wrist_angle, right_wrist_angle
        ]
        feature_idx += 15
        
        # 2. BODY ORIENTATION (6 features)
        # Head orientation
        head_orientation = calculate_orientation(
            frame_coords[0], frame_coords[1], frame_coords[2],  # nose-left_eye-right_eye
            frame_confidence[0], frame_confidence[1], frame_confidence[2]
        )
        
        # Torso orientation
        torso_orientation = calculate_orientation(
            frame_coords[11], frame_coords[12], frame_coords[23],  # left_shoulder-right_shoulder-left_hip
            frame_confidence[11], frame_confidence[12], frame_confidence[23]
        )
        
        static_features[frame_idx, feature_idx:feature_idx+6] = head_orientation + torso_orientation
        feature_idx += 6
        
        # 3. LIMB LENGTHS (5 features)
        left_arm_length = calculate_distance(
            frame_coords[11], frame_coords[15],  # left_shoulder-left_wrist
            frame_confidence[11], frame_confidence[15]
        )
        right_arm_length = calculate_distance(
            frame_coords[12], frame_coords[16],  # right_shoulder-right_wrist
            frame_confidence[12], frame_confidence[16]
        )
        left_leg_length = calculate_distance(
            frame_coords[23], frame_coords[27],  # left_hip-left_ankle
            frame_confidence[23], frame_confidence[27]
        )
        right_leg_length = calculate_distance(
            frame_coords[24], frame_coords[28],  # right_hip-right_ankle
            frame_confidence[24], frame_confidence[28]
        )
        torso_length = calculate_distance(
            frame_coords[11], frame_coords[23],  # left_shoulder-left_hip
            frame_confidence[11], frame_confidence[23]
        )
        
        static_features[frame_idx, feature_idx:feature_idx+5] = [
            left_arm_length, right_arm_length, left_leg_length, right_leg_length, torso_length
        ]
        feature_idx += 5
        
        # 4. SYMMETRY FEATURES (2 features)
        arm_symmetry = calculate_limb_symmetry(left_arm_length, right_arm_length)
        leg_symmetry = calculate_limb_symmetry(left_leg_length, right_leg_length)
        
        static_features[frame_idx, feature_idx:feature_idx+2] = [arm_symmetry, leg_symmetry]
        feature_idx += 2
        
        # 5. CENTER OF MASS (2 features)
        com = calculate_body_center_of_mass(frame_coords, frame_confidence)
        static_features[frame_idx, feature_idx:feature_idx+2] = [com[0], com[1]]
        feature_idx += 2
        
        # 6. JOINT DISTANCES (10 features)
        # Key joint distances
        joint_pairs = [
            (0, 11), (0, 12),  # nose to shoulders
            (11, 23), (12, 24),  # shoulders to hips
            (23, 24),  # hip to hip
            (11, 12),  # shoulder to shoulder
            (15, 16),  # wrist to wrist
            (27, 28),  # ankle to ankle
            (0, 23)  # nose to left hip
        ]
        
        for i, (joint1, joint2) in enumerate(joint_pairs):
            if i < 10:  # Only take first 10
                distance = calculate_distance(
                    frame_coords[joint1], frame_coords[joint2],
                    frame_confidence[joint1], frame_confidence[joint2]
                )
                static_features[frame_idx, feature_idx] = distance
                feature_idx += 1
        
        # 7. BODY PROPORTIONS (5 features)
        # Ratios of limb lengths
        if torso_length > 0:
            static_features[frame_idx, feature_idx] = left_arm_length / torso_length
            static_features[frame_idx, feature_idx+1] = right_arm_length / torso_length
            static_features[frame_idx, feature_idx+2] = left_leg_length / torso_length
            static_features[frame_idx, feature_idx+3] = right_leg_length / torso_length
        else:
            static_features[frame_idx, feature_idx:feature_idx+4] = 0.0
        
        # Arm to leg ratio
        if left_leg_length + right_leg_length > 0:
            static_features[frame_idx, feature_idx+4] = (left_arm_length + right_arm_length) / (left_leg_length + right_leg_length)
        else:
            static_features[frame_idx, feature_idx+4] = 0.0
        
        feature_idx += 5
        
        # 8. SPATIAL FEATURES (5 features)
        # Body dimensions
        body_width = calculate_distance(
            frame_coords[11], frame_coords[12],  # shoulder width
            frame_confidence[11], frame_confidence[12]
        )
        body_height = calculate_distance(
            frame_coords[0], frame_coords[23],  # nose to hip height
            frame_confidence[0], frame_confidence[23]
        )
        hip_width = calculate_distance(
            frame_coords[23], frame_coords[24],  # hip width
            frame_confidence[23], frame_confidence[24]
        )
        
        # Body area (approximate)
        body_area = body_width * body_height
        
        # Body aspect ratio
        aspect_ratio = body_height / (body_width + 1e-6)
        
        static_features[frame_idx, feature_idx:feature_idx+5] = [
            body_width, body_height, hip_width, body_area, aspect_ratio
        ]
        feature_idx += 5
        
        # Ensure we have exactly 50 static features
        assert feature_idx == 50, f"Expected 50 static features, got {feature_idx}"
    
    # ========================================================================
    # TEMPORAL FEATURES (50 features)
    # ========================================================================
    
    # 1. JOINT VELOCITIES (15 features)
    # Calculate velocities for key joints
    key_joints = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30]  # 15 joints
    
    for frame_idx in range(num_frames):
        feature_idx = 0
        
        for joint_idx in key_joints:
            if frame_idx > 0:
                velocity = calculate_joint_velocity(coords[:, joint_idx, :], frame_idx)
                velocity_magnitude = np.linalg.norm(velocity)
            else:
                velocity_magnitude = 0.0
            
            temporal_features[frame_idx, feature_idx] = velocity_magnitude
            feature_idx += 1
        
        # 2. JOINT ACCELERATIONS (15 features)
        for joint_idx in key_joints:
            if frame_idx > 1:
                acceleration = calculate_joint_acceleration(coords[:, joint_idx, :], frame_idx)
                acceleration_magnitude = np.linalg.norm(acceleration)
            else:
                acceleration_magnitude = 0.0
            
            temporal_features[frame_idx, feature_idx] = acceleration_magnitude
            feature_idx += 1
        
        # 3. ANGULAR VELOCITIES (10 features)
        # Calculate angular velocities for key angles
        key_angles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # First 10 static features (angles)
        
        for angle_idx in key_angles:
            if frame_idx > 0:
                angular_velocity = static_features[frame_idx, angle_idx] - static_features[frame_idx-1, angle_idx]
            else:
                angular_velocity = 0.0
            
            temporal_features[frame_idx, feature_idx] = angular_velocity
            feature_idx += 1
        
        # 4. TEMPORAL SYMMETRY (5 features)
        # Changes in symmetry over time
        symmetry_features = [45, 46]  # arm_symmetry, leg_symmetry indices
        
        for sym_idx in symmetry_features:
            if frame_idx > 0:
                symmetry_change = static_features[frame_idx, sym_idx] - static_features[frame_idx-1, sym_idx]
            else:
                symmetry_change = 0.0
            
            temporal_features[frame_idx, feature_idx] = symmetry_change
            feature_idx += 1
        
        # Additional temporal symmetry features
        if frame_idx > 0:
            # Overall symmetry change
            overall_symmetry_change = np.mean([
                static_features[frame_idx, 45] - static_features[frame_idx-1, 45],  # arm symmetry
                static_features[frame_idx, 46] - static_features[frame_idx-1, 46]   # leg symmetry
            ])
        else:
            overall_symmetry_change = 0.0
        
        temporal_features[frame_idx, feature_idx] = overall_symmetry_change
        feature_idx += 1
        
        # Symmetry variance
        if frame_idx > 2:
            symmetry_variance = np.var(static_features[frame_idx-2:frame_idx+1, 45:47])
        else:
            symmetry_variance = 0.0
        
        temporal_features[frame_idx, feature_idx] = symmetry_variance
        feature_idx += 1
        
        # 5. MOTION INTENSITY (5 features)
        # Overall motion intensity
        if frame_idx > 0:
            # Total velocity magnitude
            total_velocity = np.sum(temporal_features[frame_idx, :15])  # Sum of joint velocities
            
            # Total acceleration magnitude
            total_acceleration = np.sum(temporal_features[frame_idx, 15:30])  # Sum of joint accelerations
            
            # Motion smoothness (inverse of acceleration)
            motion_smoothness = 1.0 / (total_acceleration + 1e-6)
            
            # Motion direction change
            if frame_idx > 1:
                prev_velocity = np.sum(temporal_features[frame_idx-1, :15])
                velocity_change = abs(total_velocity - prev_velocity)
            else:
                velocity_change = 0.0
            
            # Motion consistency
            if frame_idx > 2:
                motion_consistency = 1.0 / (np.std(temporal_features[frame_idx-2:frame_idx+1, :15]) + 1e-6)
            else:
                motion_consistency = 0.0
        else:
            total_velocity = 0.0
            total_acceleration = 0.0
            motion_smoothness = 0.0
            velocity_change = 0.0
            motion_consistency = 0.0
        
        temporal_features[frame_idx, feature_idx:feature_idx+5] = [
            total_velocity, total_acceleration, motion_smoothness, velocity_change, motion_consistency
        ]
        feature_idx += 5
        
        # Ensure we have exactly 50 temporal features
        assert feature_idx == 50, f"Expected 50 temporal features, got {feature_idx}"
    
    # Combine static and temporal features
    final_features = np.concatenate([static_features, temporal_features], axis=1)
    
    return final_features

def extract_and_normalize_pose_with_kinematics(video_path: str) -> np.ndarray:
    """Extract normalized pose data and convert to kinematic features."""
    normalized_poses = extract_and_normalize_pose(video_path)
    kinematic_feats = kinematic_features(normalized_poses)
    
    return kinematic_feats

# ==============================================================================
# V4: UCF-101 UTILITY FUNCTIONS
# ==============================================================================

def load_ucf_class_map(class_ind_path: Union[str, Path]) -> Dict[str, int]:
    """Loads class names and maps them to 0-indexed integer labels."""
    class_map = {}
    with open(class_ind_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_map[parts[1]] = int(parts[0]) - 1
    return class_map

def load_ucf_split_files(data_root: Union[str, Path], split_id: int, subset: str) -> List[Tuple[str, int]]:
    """Loads video file paths and their 0-indexed labels for a specific UCF-101 split."""
    data_root = Path(data_root)
    split_dir = data_root / "ucf_splits"
    
    class_map = load_ucf_class_map(split_dir / "classInd.txt")
    
    list_filename = f"{subset}list{split_id:02d}.txt"
    list_path = split_dir / list_filename
    
    if not list_path.exists():
        raise FileNotFoundError(f"UCF-101 split file not found: {list_path}")
        
    samples = []
    with open(list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            video_rel_path = line.split()[0] if subset == 'test' else line.split()[0]
            class_name = video_rel_path.split('/')[0]
            
            if class_name in class_map:
                label = class_map[class_name]
                samples.append((video_rel_path, label))
            else:
                print(f"Warning: Class {class_name} not found in class map.")
                
    return samples

# ==============================================================================
# V5: DATASET CLASS
# ==============================================================================

class PRISMDataset(Dataset):
    """PyTorch Dataset for loading normalized pose/kinematic sequences based on UCF-101 splits."""
    
    def __init__(self, data_root: str, split_id: int, subset: str, 
                 sequence_length: int = 30, use_kinematics: bool = True):
        
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.use_kinematics = use_kinematics
        self.split_id = split_id
        self.subset = subset
        
        self.load_dir = self.data_root / ("features" if use_kinematics else "processed")
        
        self.samples = load_ucf_split_files(self.data_root, split_id, subset)
        
        self.data_files = []
        self.labels = []
        suffix = "_kinematic.npy" if use_kinematics else "_pose.npy"
        
        for video_rel_path, label in self.samples:
            video_stem = Path(video_rel_path).stem
            file_name = f"{video_stem}{suffix}"
            file_path = self.load_dir / file_name
            
            if file_path.exists():
                self.data_files.append(file_path)
                self.labels.append(label)

        if not self.data_files:
            raise ValueError(f"No corresponding feature files found for UCF-101 Split {split_id}, Subset {subset} in {self.load_dir}")
        
        self.feature_dims = 100 if use_kinematics else 132
        print(f"Loaded {len(self.data_files)} samples for Split {split_id}/{subset}.")
        print(f"Using {'kinematic (100 dims)' if use_kinematics else 'raw pose (132 dims)'} features.")

    def __len__(self) -> int:
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data_file = self.data_files[idx]
        raw_data = np.load(data_file)
        
        if self.use_kinematics and len(raw_data.shape) == 3:
            sequence_data = kinematic_features(raw_data)
        else:
            sequence_data = raw_data
        
        sequence_data = self._pad_or_truncate_sequence(sequence_data)
        pose_tensor = torch.FloatTensor(sequence_data)
        
        sample = {
            'pose_sequence': pose_tensor,
            'label': torch.LongTensor([self.labels[idx]])[0]
        }
        
        return sample
    
    def _pad_or_truncate_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Pad or truncate sequence to fixed length."""
        current_length = sequence.shape[0]
        
        if current_length == self.sequence_length:
            return sequence
        elif current_length > self.sequence_length:
            indices = np.linspace(0, current_length - 1, self.sequence_length, dtype=int)
            return sequence[indices]
        else:
            padding_needed = self.sequence_length - current_length
            last_frame = sequence[-1:].repeat(padding_needed, axis=0)
            return np.vstack([sequence, last_frame])

# ==============================================================================
# V6: UTILITY FUNCTIONS
# ==============================================================================

def save_pose_sequence(pose_sequence: np.ndarray, output_path: str):
    """Save normalized pose sequence to .npy file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, pose_sequence)

def save_kinematic_features(kinematic_features: np.ndarray, output_path: str):
    """Save kinematic features to .npy file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, kinematic_features)

def process_video_batch(video_paths: List[str], output_dir: str, use_kinematics: bool = True):
    """Process a batch of videos and save normalized pose sequences or kinematic features."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, video_path in enumerate(video_paths):
        try:
            print(f"Processing video {i+1}/{len(video_paths)}: {Path(video_path).name}")
            
            if use_kinematics:
                kinematic_features_data = extract_and_normalize_pose_with_kinematics(video_path)
                video_name = Path(video_path).stem
                output_path = os.path.join(output_dir, f"{video_name}_kinematic.npy")
                save_kinematic_features(kinematic_features_data, output_path)
                
            else:
                pose_sequence = extract_and_normalize_pose(video_path)
                video_name = Path(video_path).stem
                output_path = os.path.join(output_dir, f"{video_name}_pose.npy")
                save_pose_sequence(pose_sequence, output_path)
            
            print(f"  -> Saved to {Path(output_path).name}")
            
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            continue

# ==============================================================================
# V7: TESTING AND VALIDATION
# ==============================================================================

def test_kinematic_features():
    """Test kinematic features extraction."""
    print("Testing kinematic features extraction...")
    
    # Create dummy data
    num_frames = 10
    num_joints = 33
    num_coords = 4
    
    # Generate realistic pose data
    dummy_poses = np.random.randn(num_frames, num_joints, num_coords)
    dummy_poses[:, :, 3] = np.abs(dummy_poses[:, :, 3]) % 1.0  # Confidence between 0 and 1
    
    # Add some structure
    for frame_idx in range(1, num_frames):
        dummy_poses[frame_idx] = 0.7 * dummy_poses[frame_idx-1] + 0.3 * dummy_poses[frame_idx]
    
    # Extract kinematic features
    features = kinematic_features(dummy_poses)
    
    print(f"Input shape: {dummy_poses.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected shape: ({num_frames}, 100)")
    
    # Validate output
    assert features.shape == (num_frames, 100), f"Expected shape ({num_frames}, 100), got {features.shape}"
    assert not np.any(np.isnan(features)), "Features contain NaN values"
    assert not np.any(np.isinf(features)), "Features contain infinite values"
    
    print("✓ Kinematic features test passed!")
    
    # Print feature statistics
    print(f"Feature statistics:")
    print(f"  Mean: {np.mean(features):.4f}")
    print(f"  Std: {np.std(features):.4f}")
    print(f"  Min: {np.min(features):.4f}")
    print(f"  Max: {np.max(features):.4f}")
    
    return features

if __name__ == "__main__":
    # Run tests
    test_kinematic_features()
    print("All tests completed successfully!")