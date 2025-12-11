#!/usr/bin/env python3
"""
build_sequences.py - Assemble temporal sequences from pose keypoints.

This script:
1. Groups images/poses into temporal sequences (sliding windows)
2. Normalizes joint positions (subtract mid-hip, divide by torso length)
3. Optionally computes joint velocities as additional features
4. Creates train/val/test splits with matched labels
5. Supports synthetic sequence generation from static images

For static image datasets, the script can generate pseudo-sequences by applying
random temporal augmentations (scale, rotation, noise) to simulate motion.

Usage:
    python build_sequences.py --pose_dir pose_npy --output_dir sequences --T 16 --stride 8
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Build temporal sequences from pose keypoints'
    )
    parser.add_argument(
        '--pose_dir',
        type=str,
        default='pose_npy',
        help='Directory containing pose .npy files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='sequences',
        help='Output directory for sequences'
    )
    parser.add_argument(
        '--labels_dir',
        type=str,
        default=None,
        help='Directory containing label files (YOLO format) for classification'
    )
    parser.add_argument(
        '--data_yaml',
        type=str,
        default=None,
        help='Path to data.yaml for class information'
    )
    parser.add_argument(
        '--T',
        type=int,
        default=16,
        help='Sequence length (number of frames)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=8,
        help='Stride for sliding window'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        default=True,
        help='Normalize keypoints by centering and scaling'
    )
    parser.add_argument(
        '--add_velocity',
        action='store_true',
        help='Add joint velocity features (frame-to-frame differences)'
    )
    parser.add_argument(
        '--flatten',
        action='store_true',
        default=True,
        help='Flatten output to (T, 66) instead of (T, 33, 3)'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Generate synthetic sequences from static images'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Apply data augmentation to sequences'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Splits to process'
    )
    return parser.parse_args()


# MediaPipe keypoint indices for normalization
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def normalize_pose(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalize pose keypoints by centering on mid-hip and scaling by torso length.

    Args:
        keypoints: Array of shape (33, 3) with x, y, visibility

    Returns:
        Normalized keypoints of same shape
    """
    keypoints = keypoints.copy()

    # Get visibility mask (only process visible joints)
    visibility = keypoints[:, 2]

    # Calculate mid-hip center
    if visibility[LEFT_HIP] > 0.5 and visibility[RIGHT_HIP] > 0.5:
        center = (keypoints[LEFT_HIP, :2] + keypoints[RIGHT_HIP, :2]) / 2
    else:
        # Fall back to mean of visible points
        visible_mask = visibility > 0.5
        if visible_mask.sum() > 0:
            center = keypoints[visible_mask, :2].mean(axis=0)
        else:
            center = np.array([0.5, 0.5])

    # Center keypoints
    keypoints[:, 0] -= center[0]
    keypoints[:, 1] -= center[1]

    # Calculate torso length for scaling
    torso_length = 1.0
    if (visibility[LEFT_SHOULDER] > 0.5 and visibility[RIGHT_SHOULDER] > 0.5 and
            visibility[LEFT_HIP] > 0.5 and visibility[RIGHT_HIP] > 0.5):
        shoulder_center = (keypoints[LEFT_SHOULDER, :2] + keypoints[RIGHT_SHOULDER, :2]) / 2
        hip_center = (keypoints[LEFT_HIP, :2] + keypoints[RIGHT_HIP, :2]) / 2
        torso_length = np.linalg.norm(shoulder_center - hip_center)
        if torso_length < 0.01:  # Avoid division by very small values
            torso_length = 1.0

    # Scale by torso length
    keypoints[:, 0] /= torso_length
    keypoints[:, 1] /= torso_length

    return keypoints


def compute_velocity(sequence: np.ndarray) -> np.ndarray:
    """
    Compute joint velocities (frame-to-frame differences).

    Args:
        sequence: Array of shape (T, 33, 3)

    Returns:
        Velocities of shape (T, 33, 2) (only x, y velocities)
    """
    # Compute differences
    velocity = np.zeros((sequence.shape[0], 33, 2), dtype=np.float32)
    velocity[1:] = sequence[1:, :, :2] - sequence[:-1, :, :2]
    return velocity


def extract_frame_number(filename: str) -> Optional[int]:
    """Extract frame number from filename for temporal ordering."""
    # Try common patterns
    patterns = [
        r'_(\d+)$',  # name_123
        r'(\d+)$',  # name123
        r'frame[_-]?(\d+)',  # frame_123 or frame123
        r'(\d+)[_-]?$'  # 123_ or 123-
    ]

    stem = Path(filename).stem

    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            return int(match.group(1))

    return None


def group_by_video(pose_files: List[Path]) -> Dict[str, List[Tuple[int, Path]]]:
    """
    Group pose files by video/sequence ID.

    Assumes filenames follow pattern: video_id_frame_number.npy
    """
    groups = defaultdict(list)

    for pose_path in pose_files:
        stem = pose_path.stem

        # Try to extract video ID and frame number
        # Pattern: videoname_framenum or videoname_framenum_other
        parts = stem.rsplit('_', 1)

        if len(parts) == 2 and parts[1].isdigit():
            video_id = parts[0]
            frame_num = int(parts[1])
        else:
            # Use filename as video ID with extracted frame number
            frame_num = extract_frame_number(stem)
            if frame_num is not None:
                # Remove frame number from video ID
                video_id = re.sub(r'[_-]?\d+$', '', stem)
            else:
                # Single image, no sequence
                video_id = stem
                frame_num = 0

        groups[video_id].append((frame_num, pose_path))

    # Sort each group by frame number
    for video_id in groups:
        groups[video_id].sort(key=lambda x: x[0])

    return groups


def generate_synthetic_sequence(
    base_pose: np.ndarray,
    T: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a synthetic sequence from a single pose by applying
    random temporal augmentations.

    Args:
        base_pose: Single pose array (33, 3)
        T: Sequence length
        seed: Random seed

    Returns:
        Synthetic sequence of shape (T, 33, 3)
    """
    if seed is not None:
        np.random.seed(seed)

    sequence = np.zeros((T, 33, 3), dtype=np.float32)

    # Base transformations
    for t in range(T):
        pose = base_pose.copy()

        # Progressive random noise (simulating small movements)
        noise_scale = 0.005 * (1 + t / T)  # Increasing noise over time
        pose[:, :2] += np.random.randn(33, 2) * noise_scale

        # Slight progressive translation (simulating drift)
        drift_x = np.sin(2 * np.pi * t / T) * 0.02
        drift_y = np.cos(2 * np.pi * t / T) * 0.01
        pose[:, 0] += drift_x
        pose[:, 1] += drift_y

        # Slight rotation around center
        angle = np.sin(2 * np.pi * t / T) * 0.05  # Small oscillation
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        center = pose[:, :2].mean(axis=0)
        centered = pose[:, :2] - center
        rotated = np.stack([
            centered[:, 0] * cos_a - centered[:, 1] * sin_a,
            centered[:, 0] * sin_a + centered[:, 1] * cos_a
        ], axis=1)
        pose[:, :2] = rotated + center

        sequence[t] = pose

    return sequence


def augment_sequence(sequence: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """Apply random augmentation to a sequence."""
    if seed is not None:
        np.random.seed(seed)

    seq = sequence.copy()
    T = seq.shape[0]

    # Random horizontal flip (50% chance)
    if np.random.random() > 0.5:
        seq[:, :, 0] = -seq[:, :, 0]
        # Swap left/right joints
        left_indices = [1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
        right_indices = [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
        for l, r in zip(left_indices, right_indices):
            seq[:, [l, r]] = seq[:, [r, l]]

    # Random scale
    scale = np.random.uniform(0.9, 1.1)
    seq[:, :, :2] *= scale

    # Random translation
    tx = np.random.uniform(-0.1, 0.1)
    ty = np.random.uniform(-0.1, 0.1)
    seq[:, :, 0] += tx
    seq[:, :, 1] += ty

    # Random temporal speed perturbation (by interpolation)
    if np.random.random() > 0.5:
        speed = np.random.uniform(0.8, 1.2)
        new_T = int(T * speed)
        if new_T > 2:
            # Simple linear interpolation
            old_indices = np.linspace(0, T - 1, new_T)
            new_seq = np.zeros((new_T, 33, 3), dtype=np.float32)
            for i, idx in enumerate(old_indices):
                low_idx = int(idx)
                high_idx = min(low_idx + 1, T - 1)
                frac = idx - low_idx
                new_seq[i] = seq[low_idx] * (1 - frac) + seq[high_idx] * frac

            # Resample back to original length
            indices = np.linspace(0, new_T - 1, T).astype(int)
            seq = new_seq[indices]

    # Add Gaussian noise
    noise_scale = 0.01
    seq[:, :, :2] += np.random.randn(T, 33, 2) * noise_scale

    return seq


def get_label_for_image(label_path: Path) -> Optional[int]:
    """Get class label from YOLO format label file."""
    if not label_path.exists():
        return None

    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    return int(parts[0])
    except Exception:
        pass

    return None


def build_sequences_from_group(
    frames: List[Tuple[int, Path]],
    T: int,
    stride: int,
    normalize: bool,
    labels_dir: Optional[Path],
    synthetic: bool = False
) -> List[Tuple[np.ndarray, Optional[int]]]:
    """Build sequences from a group of temporally ordered frames."""
    sequences = []

    if len(frames) == 0:
        return sequences

    # Load all poses
    poses = []
    labels = []
    for _, pose_path in frames:
        try:
            pose = np.load(pose_path)
            if normalize:
                pose = normalize_pose(pose)
            poses.append(pose)

            # Get label if available
            if labels_dir:
                label_path = labels_dir / f'{pose_path.stem}.txt'
                label = get_label_for_image(label_path)
            else:
                # Default label: 1 (aggressive pose) for this dataset
                label = 1
            labels.append(label)
        except Exception as e:
            logger.warning(f"Error loading pose {pose_path}: {e}")
            continue

    if len(poses) == 0:
        return sequences

    poses = np.array(poses)

    if synthetic and len(poses) == 1:
        # Generate synthetic sequence from single image
        syn_seq = generate_synthetic_sequence(poses[0], T)
        label = labels[0] if labels else 1
        sequences.append((syn_seq, label))
    elif len(poses) >= T:
        # Create sliding windows
        for start in range(0, len(poses) - T + 1, stride):
            seq = poses[start:start + T]

            # Get majority label for the sequence
            seq_labels = labels[start:start + T]
            valid_labels = [l for l in seq_labels if l is not None]
            if valid_labels:
                # Use most common label
                from collections import Counter
                label = Counter(valid_labels).most_common(1)[0][0]
            else:
                label = 1

            sequences.append((seq, label))
    elif len(poses) > 1:
        # Pad short sequences
        pad_length = T - len(poses)
        padded = np.zeros((T, 33, 3), dtype=np.float32)
        padded[:len(poses)] = poses
        # Repeat last frame for padding
        padded[len(poses):] = poses[-1]

        label = labels[0] if labels else 1
        sequences.append((padded, label))
    else:
        # Single pose - generate synthetic if enabled
        if synthetic:
            syn_seq = generate_synthetic_sequence(poses[0], T)
            label = labels[0] if labels else 1
            sequences.append((syn_seq, label))

    return sequences


def process_split(
    pose_dir: Path,
    labels_dir: Optional[Path],
    T: int,
    stride: int,
    normalize: bool,
    add_velocity: bool,
    flatten: bool,
    synthetic: bool,
    augment: bool,
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Process a single split and return sequences with labels."""
    np.random.seed(seed)

    # Get all pose files
    pose_files = sorted(pose_dir.glob('*.npy'))
    pose_files = [p for p in pose_files if not p.name.endswith('_all.npy')]

    if not pose_files:
        logger.warning(f"No pose files found in {pose_dir}")
        return np.array([]), np.array([])

    logger.info(f"Found {len(pose_files)} pose files in {pose_dir}")

    # Group by video
    groups = group_by_video(pose_files)
    logger.info(f"Grouped into {len(groups)} videos/sequences")

    all_sequences = []
    all_labels = []

    for video_id, frames in groups.items():
        sequences = build_sequences_from_group(
            frames, T, stride, normalize, labels_dir, synthetic
        )

        for seq, label in sequences:
            all_sequences.append(seq)
            all_labels.append(label)

            # Add augmented versions for training
            if augment:
                aug_seq = augment_sequence(seq)
                all_sequences.append(aug_seq)
                all_labels.append(label)

    if not all_sequences:
        return np.array([]), np.array([])

    sequences = np.array(all_sequences, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    # Add velocity features if requested
    if add_velocity:
        velocities = np.array([compute_velocity(seq) for seq in sequences])
        # Concatenate: (T, 33, 3) + (T, 33, 2) -> (T, 33, 5)
        sequences = np.concatenate([sequences, velocities], axis=-1)

    # Flatten if requested
    if flatten:
        # (N, T, 33, C) -> (N, T, 33*C)
        N, T_len = sequences.shape[:2]
        sequences = sequences.reshape(N, T_len, -1)

    logger.info(f"Created {len(sequences)} sequences with shape {sequences.shape}")

    return sequences, labels


def main():
    """Main function to build sequences."""
    args = parse_args()

    pose_dir = Path(args.pose_dir)
    output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load class info if available
    class_names = ['aggressive_pose']
    if args.data_yaml and Path(args.data_yaml).exists():
        with open(args.data_yaml, 'r') as f:
            config = yaml.safe_load(f)
        class_names = config.get('names', class_names)

    all_info = []

    # Process each split
    for split in args.splits:
        split_pose_dir = pose_dir / split
        labels_dir = Path(args.labels_dir) / split if args.labels_dir else None

        if not split_pose_dir.exists():
            # Try direct directory (no splits)
            if split == 'train' and pose_dir.exists():
                split_pose_dir = pose_dir
                labels_dir = Path(args.labels_dir) if args.labels_dir else None
            else:
                logger.warning(f"Pose directory not found for split '{split}': {split_pose_dir}")
                continue

        logger.info(f"\nProcessing {split} split...")

        sequences, labels = process_split(
            split_pose_dir,
            labels_dir,
            args.T,
            args.stride,
            args.normalize,
            args.add_velocity,
            args.flatten,
            args.synthetic,
            args.augment and split == 'train',  # Only augment training
            args.seed
        )

        if len(sequences) == 0:
            logger.warning(f"No sequences created for split '{split}'")
            continue

        # Save sequences
        split_output = output_dir / split
        split_output.mkdir(exist_ok=True)

        # Save as single numpy files
        np.save(split_output / 'sequences.npy', sequences)
        np.save(split_output / 'labels.npy', labels)

        # Also save as CSV for labels
        labels_df = pd.DataFrame({
            'sequence_idx': range(len(labels)),
            'label': labels,
            'class_name': [class_names[l] if l < len(class_names) else f'class_{l}' for l in labels]
        })
        labels_df.to_csv(split_output / 'labels.csv', index=False)

        logger.info(f"Saved {len(sequences)} sequences to {split_output}")

        # Collect info
        for i, label in enumerate(labels):
            all_info.append({
                'split': split,
                'sequence_idx': i,
                'label': label,
                'class_name': class_names[label] if label < len(class_names) else f'class_{label}'
            })

    # Save combined info
    if all_info:
        info_df = pd.DataFrame(all_info)
        info_df.to_csv(output_dir / 'all_sequences.csv', index=False)

        # Save configuration
        config = {
            'sequence_length': args.T,
            'stride': args.stride,
            'normalized': args.normalize,
            'has_velocity': args.add_velocity,
            'flattened': args.flatten,
            'synthetic': args.synthetic,
            'seed': args.seed,
            'class_names': class_names
        }
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)

        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("SEQUENCE BUILDING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total sequences: {len(all_info)}")
        for split in args.splits:
            split_count = sum(1 for i in all_info if i['split'] == split)
            if split_count > 0:
                logger.info(f"  {split}: {split_count} sequences")
        logger.info(f"\nClass distribution:")
        for label in sorted(set(info_df['label'])):
            count = sum(1 for l in info_df['label'] if l == label)
            name = class_names[label] if label < len(class_names) else f'class_{label}'
            logger.info(f"  {name}: {count}")

    logger.info("\nSequence building complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
