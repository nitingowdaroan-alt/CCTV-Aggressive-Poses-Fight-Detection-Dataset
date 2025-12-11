#!/usr/bin/env python3
"""
split_dataset.py - Create deterministic train/val/test split for YOLO format dataset.

This script:
1. Shuffles dataset with a fixed seed for reproducibility
2. Splits into train/val/test with configurable ratios (default: 70/15/15)
3. Creates yolo_split/{images,labels}/{train,val,test} structure
4. Updates data.yaml with correct paths
5. Saves manifest.csv listing all files with their splits

Usage:
    python split_dataset.py --data_root /path/to/Aggressive_Poses_Dataset --output_dir yolo_split
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import random

import yaml
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create train/val/test split for YOLO format dataset'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='.',
        help='Root directory of the dataset (contains images/, labels/, data.yaml)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='yolo_split',
        help='Output directory for split dataset'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.70,
        help='Ratio of data for training (default: 0.70)'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Ratio of data for validation (default: 0.15)'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='Ratio of data for testing (default: 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of creating symlinks'
    )
    parser.add_argument(
        '--stratify',
        action='store_true',
        help='Stratify split by class distribution (requires parsing labels)'
    )
    return parser.parse_args()


def get_matched_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    """Get list of matched image-label pairs."""
    pairs = []

    # Get all images
    images = {}
    for ext in IMAGE_EXTENSIONS:
        for img_path in images_dir.glob(f'*{ext}'):
            images[img_path.stem] = img_path
        for img_path in images_dir.glob(f'*{ext.upper()}'):
            images[img_path.stem] = img_path

    # Match with labels
    for stem, img_path in images.items():
        label_path = labels_dir / f'{stem}.txt'
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            logger.warning(f"No label file for image: {img_path.name}")

    return pairs


def get_class_from_label(label_path: Path) -> int:
    """Get primary class from label file (for stratification)."""
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    return int(parts[0])
    except Exception:
        pass
    return 0  # Default class


def split_data(
    pairs: List[Tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify: bool = False
) -> Tuple[List, List, List]:
    """Split data into train/val/test sets."""
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        logger.warning(f"Ratios sum to {total_ratio}, normalizing...")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio

    # Set random seed
    random.seed(seed)

    if stratify:
        # Group by class
        class_groups = {}
        for img_path, label_path in pairs:
            cls = get_class_from_label(label_path)
            if cls not in class_groups:
                class_groups[cls] = []
            class_groups[cls].append((img_path, label_path))

        train_data, val_data, test_data = [], [], []

        for cls, class_pairs in class_groups.items():
            random.shuffle(class_pairs)
            n = len(class_pairs)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_data.extend(class_pairs[:n_train])
            val_data.extend(class_pairs[n_train:n_train + n_val])
            test_data.extend(class_pairs[n_train + n_val:])

        # Shuffle each split
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

    else:
        # Simple random split
        shuffled = pairs.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_data = shuffled[:n_train]
        val_data = shuffled[n_train:n_train + n_val]
        test_data = shuffled[n_train + n_val:]

    return train_data, val_data, test_data


def create_split_directory(
    output_dir: Path,
    split_name: str,
    data: List[Tuple[Path, Path]],
    use_copy: bool = False
) -> List[Dict]:
    """Create directory structure and copy/link files for a split."""
    images_dir = output_dir / 'images' / split_name
    labels_dir = output_dir / 'labels' / split_name

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []

    for img_path, label_path in data:
        dst_img = images_dir / img_path.name
        dst_label = labels_dir / label_path.name

        if use_copy:
            shutil.copy2(img_path, dst_img)
            shutil.copy2(label_path, dst_label)
        else:
            # Create symlinks (relative paths for portability)
            try:
                if dst_img.exists():
                    dst_img.unlink()
                if dst_label.exists():
                    dst_label.unlink()

                # Use absolute paths for symlinks
                dst_img.symlink_to(img_path.resolve())
                dst_label.symlink_to(label_path.resolve())
            except OSError:
                # Fall back to copy if symlinks fail (e.g., on Windows without admin)
                shutil.copy2(img_path, dst_img)
                shutil.copy2(label_path, dst_label)

        manifest_entries.append({
            'image_path': str(img_path),
            'label_path': str(label_path),
            'split': split_name,
            'image_name': img_path.name,
            'label_name': label_path.name
        })

    return manifest_entries


def create_data_yaml(
    output_dir: Path,
    original_yaml_path: Path,
    num_classes: int,
    class_names: List[str]
):
    """Create updated data.yaml for the split dataset."""
    # Load original config if exists
    original_config = {}
    if original_yaml_path.exists():
        with open(original_yaml_path, 'r') as f:
            original_config = yaml.safe_load(f) or {}

    # Create new config with updated paths
    output_dir_resolved = output_dir.resolve()

    new_config = {
        'path': str(output_dir_resolved),
        'train': str(output_dir_resolved / 'images' / 'train'),
        'val': str(output_dir_resolved / 'images' / 'val'),
        'test': str(output_dir_resolved / 'images' / 'test'),
        'nc': num_classes,
        'names': class_names
    }

    # Add any additional fields from original
    for key in ['roboflow', 'license', 'version']:
        if key in original_config:
            new_config[key] = original_config[key]

    # Save new config
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Created data.yaml at {yaml_path}")

    return new_config


def print_split_summary(train_data, val_data, test_data):
    """Print summary of the split."""
    total = len(train_data) + len(val_data) + len(test_data)

    logger.info("\n" + "=" * 50)
    logger.info("SPLIT SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total samples: {total}")
    logger.info(f"  Train: {len(train_data)} ({100*len(train_data)/total:.1f}%)")
    logger.info(f"  Val:   {len(val_data)} ({100*len(val_data)/total:.1f}%)")
    logger.info(f"  Test:  {len(test_data)} ({100*len(test_data)/total:.1f}%)")
    logger.info("=" * 50)


def main():
    """Main function to create dataset split."""
    args = parse_args()

    # Setup paths
    data_root = Path(args.data_root)
    images_dir = data_root / 'images'
    labels_dir = data_root / 'labels'
    yaml_path = data_root / 'data.yaml'
    output_dir = Path(args.output_dir)

    logger.info(f"Source dataset: {data_root}")
    logger.info(f"Output directory: {output_dir}")

    # Validate source
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        sys.exit(1)
    if not labels_dir.exists():
        logger.error(f"Labels directory not found: {labels_dir}")
        sys.exit(1)

    # Get matched pairs
    pairs = get_matched_pairs(images_dir, labels_dir)
    if not pairs:
        logger.error("No matched image-label pairs found!")
        sys.exit(1)

    logger.info(f"Found {len(pairs)} matched image-label pairs")

    # Split data
    train_data, val_data, test_data = split_data(
        pairs,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        args.stratify
    )

    print_split_summary(train_data, val_data, test_data)

    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create split directories and copy/link files
    logger.info("\nCreating split directories...")
    manifest_entries = []

    for split_name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        entries = create_split_directory(output_dir, split_name, data, args.copy)
        manifest_entries.extend(entries)
        logger.info(f"  Created {split_name} split: {len(data)} samples")

    # Load original data.yaml for class info
    num_classes = 1
    class_names = ['aggressive_pose']

    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            original_config = yaml.safe_load(f) or {}
        num_classes = original_config.get('nc', num_classes)
        class_names = original_config.get('names', class_names)

    # Create new data.yaml
    create_data_yaml(output_dir, yaml_path, num_classes, class_names)

    # Save manifest
    manifest_df = pd.DataFrame(manifest_entries)
    manifest_path = output_dir / 'manifest.csv'
    manifest_df.to_csv(manifest_path, index=False)
    logger.info(f"Saved manifest to {manifest_path}")

    # Save split info
    split_info = {
        'seed': args.seed,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'total_samples': len(pairs),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'stratified': args.stratify
    }

    with open(output_dir / 'split_info.yaml', 'w') as f:
        yaml.dump(split_info, f, default_flow_style=False)

    logger.info("\nSplit complete!")
    logger.info(f"Dataset ready at: {output_dir.resolve()}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
