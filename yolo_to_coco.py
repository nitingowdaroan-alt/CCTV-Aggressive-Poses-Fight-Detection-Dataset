#!/usr/bin/env python3
"""
yolo_to_coco.py - Convert YOLO format labels to COCO JSON format.

This script converts YOLO txt annotations (x_center, y_center, width, height)
to COCO JSON format (x_min, y_min, width, height) with proper image/annotation IDs.

Usage:
    python yolo_to_coco.py --data_root yolo_split --output_dir coco_format
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

import cv2
import yaml

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
        description='Convert YOLO format dataset to COCO JSON format'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='yolo_split',
        help='Root directory of YOLO split dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='coco_format',
        help='Output directory for COCO JSON files'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Splits to convert (default: train val test)'
    )
    return parser.parse_args()


def get_image_info(image_path: Path, image_id: int) -> Dict:
    """Get COCO image info dictionary."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = img.shape[:2]

    return {
        'id': image_id,
        'file_name': image_path.name,
        'width': width,
        'height': height,
        'date_captured': datetime.now().isoformat(),
        'license': 1,
        'coco_url': '',
        'flickr_url': ''
    }


def yolo_to_coco_bbox(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert YOLO normalized bbox to COCO absolute bbox.

    YOLO: (x_center, y_center, width, height) - normalized [0,1]
    COCO: (x_min, y_min, width, height) - absolute pixels

    Returns:
        (x_min, y_min, width, height) in pixels
    """
    # Convert to absolute
    abs_width = width * img_width
    abs_height = height * img_height
    x_min = (x_center * img_width) - (abs_width / 2)
    y_min = (y_center * img_height) - (abs_height / 2)

    # Ensure non-negative
    x_min = max(0, x_min)
    y_min = max(0, y_min)

    return x_min, y_min, abs_width, abs_height


def parse_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Parse YOLO format label file."""
    annotations = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append((class_id, x_center, y_center, width, height))
    except Exception as e:
        logger.error(f"Error parsing {label_path}: {e}")

    return annotations


def create_coco_annotation(
    annotation_id: int,
    image_id: int,
    category_id: int,
    bbox: Tuple[float, float, float, float],
    area: Optional[float] = None
) -> Dict:
    """Create COCO annotation dictionary."""
    x_min, y_min, width, height = bbox
    if area is None:
        area = width * height

    return {
        'id': annotation_id,
        'image_id': image_id,
        'category_id': category_id,
        'bbox': [round(x, 2) for x in [x_min, y_min, width, height]],
        'area': round(area, 2),
        'segmentation': [],
        'iscrowd': 0
    }


def create_coco_categories(class_names: List[str]) -> List[Dict]:
    """Create COCO categories list."""
    categories = []
    for i, name in enumerate(class_names):
        categories.append({
            'id': i,
            'name': name,
            'supercategory': 'action'
        })
    return categories


def convert_split_to_coco(
    images_dir: Path,
    labels_dir: Path,
    class_names: List[str],
    split_name: str
) -> Dict:
    """Convert a single split to COCO format."""
    coco_data = {
        'info': {
            'description': f'Aggressive Poses Dataset - {split_name}',
            'url': '',
            'version': '1.0',
            'year': datetime.now().year,
            'contributor': '',
            'date_created': datetime.now().isoformat()
        },
        'licenses': [{
            'id': 1,
            'name': 'Unknown',
            'url': ''
        }],
        'categories': create_coco_categories(class_names),
        'images': [],
        'annotations': []
    }

    image_id = 1
    annotation_id = 1

    # Get all images
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(images_dir.glob(f'*{ext}'))
        image_files.extend(images_dir.glob(f'*{ext.upper()}'))

    image_files = sorted(image_files, key=lambda x: x.name)

    for img_path in image_files:
        try:
            # Get image info
            img_info = get_image_info(img_path, image_id)
            coco_data['images'].append(img_info)

            # Get corresponding label
            label_path = labels_dir / f'{img_path.stem}.txt'
            if label_path.exists():
                annotations = parse_yolo_label(label_path)

                for class_id, x_c, y_c, w, h in annotations:
                    # Convert bbox
                    bbox = yolo_to_coco_bbox(
                        x_c, y_c, w, h,
                        img_info['width'], img_info['height']
                    )

                    # Create annotation
                    ann = create_coco_annotation(
                        annotation_id,
                        image_id,
                        class_id,
                        bbox
                    )
                    coco_data['annotations'].append(ann)
                    annotation_id += 1

            image_id += 1

        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            continue

    return coco_data


def main():
    """Main function to convert YOLO to COCO format."""
    args = parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    # Load data.yaml for class names
    yaml_path = data_root / 'data.yaml'
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        class_names = config.get('names', ['aggressive_pose'])
    else:
        logger.warning("data.yaml not found, using default class names")
        class_names = ['aggressive_pose']

    logger.info(f"Class names: {class_names}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert each split
    for split in args.splits:
        images_dir = data_root / 'images' / split
        labels_dir = data_root / 'labels' / split

        if not images_dir.exists():
            logger.warning(f"Images directory not found for split '{split}': {images_dir}")
            continue

        if not labels_dir.exists():
            logger.warning(f"Labels directory not found for split '{split}': {labels_dir}")
            continue

        logger.info(f"Converting {split} split...")

        coco_data = convert_split_to_coco(
            images_dir, labels_dir, class_names, split
        )

        # Save JSON
        output_path = output_dir / f'instances_{split}.json'
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        logger.info(f"  Saved {output_path}")
        logger.info(f"    Images: {len(coco_data['images'])}")
        logger.info(f"    Annotations: {len(coco_data['annotations'])}")

    # Create dataset info file
    info = {
        'source': str(data_root),
        'converted_date': datetime.now().isoformat(),
        'splits': args.splits,
        'classes': class_names
    }
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    logger.info(f"\nConversion complete! COCO format saved to: {output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
