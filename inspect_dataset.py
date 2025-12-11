#!/usr/bin/env python3
"""
inspect_dataset.py - Verify images/labels and preview labeled images with bounding boxes.

This script performs the following:
1. Counts images and label files
2. Lists unmatched files (images without labels and vice versa)
3. Validates data.yaml fields
4. Visualizes first N annotated images with bounding boxes
5. Generates dataset statistics report

Usage:
    python inspect_dataset.py --data_root /path/to/Aggressive_Poses_Dataset --preview_count 10
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict

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
        description='Inspect and validate YOLO format dataset for fight detection'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='.',
        help='Root directory of the dataset (contains images/, labels/, data.yaml)'
    )
    parser.add_argument(
        '--preview_count',
        type=int,
        default=10,
        help='Number of images to preview with annotations'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='inspection_output',
        help='Directory to save inspection results'
    )
    parser.add_argument(
        '--save_previews',
        action='store_true',
        help='Save preview images to output directory'
    )
    parser.add_argument(
        '--no_display',
        action='store_true',
        help='Do not display images (useful for headless environments)'
    )
    return parser.parse_args()


def get_image_files(images_dir: Path) -> Dict[str, Path]:
    """Get all image files in directory, keyed by stem (filename without extension)."""
    images = {}
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return images

    for ext in IMAGE_EXTENSIONS:
        for img_path in images_dir.glob(f'*{ext}'):
            images[img_path.stem] = img_path
        for img_path in images_dir.glob(f'*{ext.upper()}'):
            images[img_path.stem] = img_path

    return images


def get_label_files(labels_dir: Path) -> Dict[str, Path]:
    """Get all label files in directory, keyed by stem."""
    labels = {}
    if not labels_dir.exists():
        logger.error(f"Labels directory not found: {labels_dir}")
        return labels

    for label_path in labels_dir.glob('*.txt'):
        labels[label_path.stem] = label_path

    return labels


def parse_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Parse YOLO format label file.

    Returns:
        List of (class_id, x_center, y_center, width, height) tuples
    """
    annotations = []
    try:
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    logger.warning(f"Invalid annotation at {label_path}:{line_num}: {line}")
                    continue
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Validate normalized coordinates
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                            0 <= width <= 1 and 0 <= height <= 1):
                        logger.warning(f"Coordinates out of range at {label_path}:{line_num}")

                    annotations.append((class_id, x_center, y_center, width, height))
                except ValueError as e:
                    logger.warning(f"Parse error at {label_path}:{line_num}: {e}")
    except Exception as e:
        logger.error(f"Error reading {label_path}: {e}")

    return annotations


def load_data_yaml(yaml_path: Path) -> Optional[Dict]:
    """Load and validate data.yaml configuration."""
    if not yaml_path.exists():
        logger.error(f"data.yaml not found at {yaml_path}")
        return None

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Validate required fields
        required_fields = ['nc', 'names']
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            logger.warning(f"Missing fields in data.yaml: {missing_fields}")

        # Validate consistency
        if 'nc' in data and 'names' in data:
            if data['nc'] != len(data['names']):
                logger.warning(
                    f"Mismatch: nc={data['nc']} but {len(data['names'])} names provided"
                )

        return data
    except Exception as e:
        logger.error(f"Error loading data.yaml: {e}")
        return None


def yolo_to_pixel_coords(
    x_center: float, y_center: float, width: float, height: float,
    img_width: int, img_height: int
) -> Tuple[int, int, int, int]:
    """Convert YOLO normalized coordinates to pixel coordinates (x1, y1, x2, y2)."""
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    return x1, y1, x2, y2


def visualize_image_with_boxes(
    image_path: Path,
    annotations: List[Tuple[int, float, float, float, float]],
    class_names: List[str],
    save_path: Optional[Path] = None,
    display: bool = True
) -> np.ndarray:
    """Draw bounding boxes on image and optionally save/display."""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"Could not read image: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img.shape[:2]

    # Color palette for different classes
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(class_names) + 1)))

    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_rgb)

    # Draw bounding boxes
    for class_id, x_center, y_center, width, height in annotations:
        x1, y1, x2, y2 = yolo_to_pixel_coords(
            x_center, y_center, width, height, img_width, img_height
        )

        # Get class name and color
        if class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"class_{class_id}"

        color = colors[class_id % len(colors)][:3]

        # Draw rectangle
        rect = Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # Add label
        ax.text(
            x1, y1 - 5, class_name,
            color='white', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8)
        )

    ax.set_title(f'{image_path.name} ({len(annotations)} annotations)')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved preview to {save_path}")

    if display:
        plt.show()
    else:
        plt.close()

    return img_rgb


def generate_statistics(
    images: Dict[str, Path],
    labels: Dict[str, Path],
    data_config: Optional[Dict]
) -> Dict:
    """Generate comprehensive dataset statistics."""
    stats = {
        'total_images': len(images),
        'total_labels': len(labels),
        'matched_pairs': 0,
        'images_without_labels': [],
        'labels_without_images': [],
        'class_distribution': defaultdict(int),
        'boxes_per_image': [],
        'image_sizes': [],
        'annotation_sizes': []
    }

    # Find matched and unmatched files
    image_stems = set(images.keys())
    label_stems = set(labels.keys())

    matched = image_stems & label_stems
    stats['matched_pairs'] = len(matched)
    stats['images_without_labels'] = list(image_stems - label_stems)
    stats['labels_without_images'] = list(label_stems - image_stems)

    # Analyze annotations
    for stem in matched:
        img_path = images[stem]
        label_path = labels[stem]

        # Get image size
        img = cv2.imread(str(img_path))
        if img is not None:
            stats['image_sizes'].append(img.shape[:2])

        # Parse annotations
        annotations = parse_yolo_label(label_path)
        stats['boxes_per_image'].append(len(annotations))

        for class_id, x_c, y_c, w, h in annotations:
            stats['class_distribution'][class_id] += 1
            stats['annotation_sizes'].append((w, h))

    return stats


def print_report(stats: Dict, data_config: Optional[Dict], output_dir: Path):
    """Print and save inspection report."""
    report_lines = []

    def log_and_append(msg):
        logger.info(msg)
        report_lines.append(msg)

    log_and_append("=" * 60)
    log_and_append("DATASET INSPECTION REPORT")
    log_and_append("=" * 60)

    # Basic counts
    log_and_append(f"\n[File Counts]")
    log_and_append(f"  Total images: {stats['total_images']}")
    log_and_append(f"  Total labels: {stats['total_labels']}")
    log_and_append(f"  Matched pairs: {stats['matched_pairs']}")

    # Unmatched files
    if stats['images_without_labels']:
        log_and_append(f"\n[WARNING] Images without labels ({len(stats['images_without_labels'])}):")
        for stem in stats['images_without_labels'][:10]:
            log_and_append(f"    - {stem}")
        if len(stats['images_without_labels']) > 10:
            log_and_append(f"    ... and {len(stats['images_without_labels']) - 10} more")

    if stats['labels_without_images']:
        log_and_append(f"\n[WARNING] Labels without images ({len(stats['labels_without_images'])}):")
        for stem in stats['labels_without_images'][:10]:
            log_and_append(f"    - {stem}")
        if len(stats['labels_without_images']) > 10:
            log_and_append(f"    ... and {len(stats['labels_without_images']) - 10} more")

    # data.yaml info
    if data_config:
        log_and_append(f"\n[data.yaml Configuration]")
        log_and_append(f"  Number of classes (nc): {data_config.get('nc', 'N/A')}")
        log_and_append(f"  Class names: {data_config.get('names', 'N/A')}")
        if 'train' in data_config:
            log_and_append(f"  Train path: {data_config.get('train')}")
        if 'val' in data_config:
            log_and_append(f"  Val path: {data_config.get('val')}")
        if 'test' in data_config:
            log_and_append(f"  Test path: {data_config.get('test')}")

    # Class distribution
    if stats['class_distribution']:
        log_and_append(f"\n[Class Distribution]")
        class_names = data_config.get('names', []) if data_config else []
        for class_id, count in sorted(stats['class_distribution'].items()):
            name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            log_and_append(f"  {name} (id={class_id}): {count} instances")

    # Annotation statistics
    if stats['boxes_per_image']:
        boxes = stats['boxes_per_image']
        log_and_append(f"\n[Annotation Statistics]")
        log_and_append(f"  Total annotations: {sum(boxes)}")
        log_and_append(f"  Avg boxes per image: {np.mean(boxes):.2f}")
        log_and_append(f"  Min boxes per image: {min(boxes)}")
        log_and_append(f"  Max boxes per image: {max(boxes)}")

    # Image size statistics
    if stats['image_sizes']:
        heights, widths = zip(*[(h, w) for h, w in stats['image_sizes']])
        log_and_append(f"\n[Image Size Statistics]")
        log_and_append(f"  Height - min: {min(heights)}, max: {max(heights)}, avg: {np.mean(heights):.0f}")
        log_and_append(f"  Width - min: {min(widths)}, max: {max(widths)}, avg: {np.mean(widths):.0f}")

    # Box size statistics
    if stats['annotation_sizes']:
        widths, heights = zip(*stats['annotation_sizes'])
        log_and_append(f"\n[Bounding Box Size Statistics (normalized)]")
        log_and_append(f"  Width - min: {min(widths):.4f}, max: {max(widths):.4f}, avg: {np.mean(widths):.4f}")
        log_and_append(f"  Height - min: {min(heights):.4f}, max: {max(heights):.4f}, avg: {np.mean(heights):.4f}")

    log_and_append("\n" + "=" * 60)

    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'inspection_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    logger.info(f"Report saved to {report_path}")


def plot_statistics(stats: Dict, data_config: Optional[Dict], output_dir: Path):
    """Generate and save statistical plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Class distribution
    ax = axes[0, 0]
    if stats['class_distribution']:
        class_names = data_config.get('names', []) if data_config else []
        classes = []
        counts = []
        for class_id, count in sorted(stats['class_distribution'].items()):
            name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            classes.append(name)
            counts.append(count)
        ax.bar(classes, counts, color='steelblue')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution')
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, 'No annotations found', ha='center', va='center')

    # 2. Boxes per image histogram
    ax = axes[0, 1]
    if stats['boxes_per_image']:
        ax.hist(stats['boxes_per_image'], bins=20, color='coral', edgecolor='black')
        ax.set_xlabel('Number of Boxes')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Boxes per Image')

    # 3. Image sizes scatter
    ax = axes[1, 0]
    if stats['image_sizes']:
        heights, widths = zip(*stats['image_sizes'])
        ax.scatter(widths, heights, alpha=0.5, color='green')
        ax.set_xlabel('Width (pixels)')
        ax.set_ylabel('Height (pixels)')
        ax.set_title('Image Size Distribution')

    # 4. Box sizes scatter
    ax = axes[1, 1]
    if stats['annotation_sizes']:
        widths, heights = zip(*stats['annotation_sizes'])
        ax.scatter(widths, heights, alpha=0.3, color='purple')
        ax.set_xlabel('Box Width (normalized)')
        ax.set_ylabel('Box Height (normalized)')
        ax.set_title('Bounding Box Size Distribution')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plot_path = output_dir / 'dataset_statistics.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Statistics plot saved to {plot_path}")


def main():
    """Main function to run dataset inspection."""
    args = parse_args()

    # Setup paths
    data_root = Path(args.data_root)
    images_dir = data_root / 'images'
    labels_dir = data_root / 'labels'
    yaml_path = data_root / 'data.yaml'
    output_dir = Path(args.output_dir)

    logger.info(f"Inspecting dataset at: {data_root}")

    # Get files
    images = get_image_files(images_dir)
    labels = get_label_files(labels_dir)
    data_config = load_data_yaml(yaml_path)

    if not images:
        logger.error("No images found. Check the data_root path.")
        sys.exit(1)

    # Generate statistics
    logger.info("Generating statistics...")
    stats = generate_statistics(images, labels, data_config)

    # Print and save report
    print_report(stats, data_config, output_dir)

    # Generate plots
    plot_statistics(stats, data_config, output_dir)

    # Preview images with annotations
    if args.preview_count > 0:
        logger.info(f"\nGenerating {args.preview_count} preview images...")

        class_names = data_config.get('names', ['person', 'aggressive']) if data_config else ['person', 'aggressive']

        preview_dir = output_dir / 'previews'
        if args.save_previews:
            preview_dir.mkdir(parents=True, exist_ok=True)

        # Get matched images for preview
        matched_stems = set(images.keys()) & set(labels.keys())
        preview_stems = sorted(list(matched_stems))[:args.preview_count]

        for stem in preview_stems:
            img_path = images[stem]
            label_path = labels[stem]
            annotations = parse_yolo_label(label_path)

            save_path = preview_dir / f'{stem}_preview.png' if args.save_previews else None

            visualize_image_with_boxes(
                img_path,
                annotations,
                class_names,
                save_path=save_path,
                display=not args.no_display
            )

    logger.info("\nInspection complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
