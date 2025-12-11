#!/usr/bin/env python3
"""
extract_mediapipe_pose.py - Extract pose keypoints using MediaPipe.

This script:
1. Uses MediaPipe Pose in static mode for image pose estimation
2. Extracts 33 keypoints (x, y, visibility) per person
3. Supports person detection with YOLO for multi-person crops
4. Saves per-image .npy files with shape (33, 3)
5. Generates summary CSV with detection statistics

Usage:
    python extract_mediapipe_pose.py --data_root /path/to/dataset --output_dir pose_npy
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

import cv2
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MediaPipe landmark indices and names
MEDIAPIPE_LANDMARKS = {
    0: 'nose',
    1: 'left_eye_inner',
    2: 'left_eye',
    3: 'left_eye_outer',
    4: 'right_eye_inner',
    5: 'right_eye',
    6: 'right_eye_outer',
    7: 'left_ear',
    8: 'right_ear',
    9: 'mouth_left',
    10: 'mouth_right',
    11: 'left_shoulder',
    12: 'right_shoulder',
    13: 'left_elbow',
    14: 'right_elbow',
    15: 'left_wrist',
    16: 'right_wrist',
    17: 'left_pinky',
    18: 'right_pinky',
    19: 'left_index',
    20: 'right_index',
    21: 'left_thumb',
    22: 'right_thumb',
    23: 'left_hip',
    24: 'right_hip',
    25: 'left_knee',
    26: 'right_knee',
    27: 'left_ankle',
    28: 'right_ankle',
    29: 'left_heel',
    30: 'right_heel',
    31: 'left_foot_index',
    32: 'right_foot_index'
}

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract MediaPipe pose keypoints from images'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='.',
        help='Root directory of the dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='pose_npy',
        help='Output directory for pose keypoints'
    )
    parser.add_argument(
        '--use_yolo_crops',
        action='store_true',
        help='Use YOLO labels to crop persons before pose estimation'
    )
    parser.add_argument(
        '--min_detection_confidence',
        type=float,
        default=0.5,
        help='Minimum confidence for pose detection (default: 0.5)'
    )
    parser.add_argument(
        '--min_tracking_confidence',
        type=float,
        default=0.5,
        help='Minimum confidence for pose tracking (default: 0.5)'
    )
    parser.add_argument(
        '--model_complexity',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Model complexity (0=lite, 1=full, 2=heavy)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1 for GPU)'
    )
    parser.add_argument(
        '--save_visualizations',
        action='store_true',
        help='Save pose visualization images'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=None,
        help='Process specific splits (train, val, test). If not set, processes images/ directly'
    )
    return parser.parse_args()


def get_pose_extractor(model_complexity: int, min_detection_confidence: float, min_tracking_confidence: float):
    """Initialize MediaPipe pose extractor."""
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
    return pose, mp.solutions.drawing_utils, mp_pose


def extract_pose_from_image(
    image_path: Path,
    pose_model,
    label_path: Optional[Path] = None,
    use_crops: bool = False
) -> Tuple[List[np.ndarray], float]:
    """
    Extract pose keypoints from an image.

    Args:
        image_path: Path to image file
        pose_model: MediaPipe Pose model
        label_path: Optional path to YOLO label file for cropping
        use_crops: Whether to use YOLO boxes to crop persons

    Returns:
        List of pose arrays (each shape 33x3) and average visibility score
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        return [], 0.0

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img.shape[:2]

    poses = []
    total_visibility = 0.0

    if use_crops and label_path and label_path.exists():
        # Process each person crop from YOLO labels
        crops_and_boxes = get_person_crops(img_rgb, label_path, img_width, img_height)

        for crop, (x1, y1, x2, y2) in crops_and_boxes:
            results = pose_model.process(crop)

            if results.pose_landmarks:
                keypoints = landmarks_to_array(results.pose_landmarks)

                # Adjust coordinates back to original image space
                crop_h, crop_w = crop.shape[:2]
                keypoints[:, 0] = (keypoints[:, 0] * crop_w + x1) / img_width
                keypoints[:, 1] = (keypoints[:, 1] * crop_h + y1) / img_height

                poses.append(keypoints)
                total_visibility += keypoints[:, 2].mean()
    else:
        # Process full image
        results = pose_model.process(img_rgb)

        if results.pose_landmarks:
            keypoints = landmarks_to_array(results.pose_landmarks)
            poses.append(keypoints)
            total_visibility = keypoints[:, 2].mean()

    avg_visibility = total_visibility / len(poses) if poses else 0.0
    return poses, avg_visibility


def landmarks_to_array(landmarks) -> np.ndarray:
    """Convert MediaPipe landmarks to numpy array (33, 3)."""
    keypoints = np.zeros((33, 3), dtype=np.float32)
    for i, lm in enumerate(landmarks.landmark):
        keypoints[i] = [lm.x, lm.y, lm.visibility]
    return keypoints


def get_person_crops(
    img: np.ndarray,
    label_path: Path,
    img_width: int,
    img_height: int,
    padding: float = 0.1
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Get person crops from image using YOLO labels."""
    crops = []

    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Convert to pixel coordinates with padding
                    pad_w = width * padding
                    pad_h = height * padding

                    x1 = int(max(0, (x_center - width / 2 - pad_w) * img_width))
                    y1 = int(max(0, (y_center - height / 2 - pad_h) * img_height))
                    x2 = int(min(img_width, (x_center + width / 2 + pad_w) * img_width))
                    y2 = int(min(img_height, (y_center + height / 2 + pad_h) * img_height))

                    if x2 > x1 and y2 > y1:
                        crop = img[y1:y2, x1:x2].copy()
                        crops.append((crop, (x1, y1, x2, y2)))
    except Exception as e:
        logger.error(f"Error reading labels from {label_path}: {e}")

    return crops


def save_pose_visualization(
    image_path: Path,
    poses: List[np.ndarray],
    output_path: Path,
    mp_drawing,
    mp_pose
):
    """Save visualization of detected poses."""
    img = cv2.imread(str(image_path))
    if img is None:
        return

    img_height, img_width = img.shape[:2]

    # Draw keypoints for each detected pose
    for keypoints in poses:
        # Draw skeleton connections
        connections = mp_pose.POSE_CONNECTIONS

        for connection in connections:
            start_idx, end_idx = connection
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]

            # Only draw if both points have good visibility
            if start_point[2] > 0.5 and end_point[2] > 0.5:
                start_xy = (int(start_point[0] * img_width), int(start_point[1] * img_height))
                end_xy = (int(end_point[0] * img_width), int(end_point[1] * img_height))
                cv2.line(img, start_xy, end_xy, (0, 255, 0), 2)

        # Draw keypoints
        for i, kp in enumerate(keypoints):
            if kp[2] > 0.5:  # visibility threshold
                x, y = int(kp[0] * img_width), int(kp[1] * img_height)
                cv2.circle(img, (x, y), 4, (0, 0, 255), -1)

    cv2.imwrite(str(output_path), img)


def process_directory(
    images_dir: Path,
    labels_dir: Optional[Path],
    output_dir: Path,
    pose_model,
    mp_drawing,
    mp_pose,
    use_crops: bool = False,
    save_viz: bool = False
) -> List[Dict]:
    """Process all images in a directory."""
    results = []

    # Get all images
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(images_dir.glob(f'*{ext}'))
        image_files.extend(images_dir.glob(f'*{ext.upper()}'))

    image_files = sorted(image_files, key=lambda x: x.name)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_viz:
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)

    logger.info(f"Processing {len(image_files)} images from {images_dir}")

    for i, img_path in enumerate(image_files):
        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{len(image_files)} images")

        # Get label path if available
        label_path = None
        if labels_dir:
            label_path = labels_dir / f'{img_path.stem}.txt'

        # Extract poses
        poses, avg_visibility = extract_pose_from_image(
            img_path, pose_model, label_path, use_crops
        )

        # Save poses
        if poses:
            # Save primary pose (first detected or highest visibility)
            pose_array = poses[0]
            npy_path = output_dir / f'{img_path.stem}.npy'
            np.save(npy_path, pose_array)

            # If multiple poses, save all
            if len(poses) > 1:
                all_poses = np.stack(poses)
                multi_path = output_dir / f'{img_path.stem}_all.npy'
                np.save(multi_path, all_poses)

            # Save visualization
            if save_viz:
                viz_path = viz_dir / f'{img_path.stem}_pose.jpg'
                save_pose_visualization(img_path, poses, viz_path, mp_drawing, mp_pose)

        results.append({
            'image': img_path.name,
            'detected_persons': len(poses),
            'avg_visibility': round(avg_visibility, 4),
            'pose_saved': len(poses) > 0
        })

    return results


def main():
    """Main function for pose extraction."""
    args = parse_args()

    # Import MediaPipe here to handle potential import errors gracefully
    try:
        import mediapipe as mp
    except ImportError:
        logger.error("MediaPipe not installed. Please install with: pip install mediapipe")
        sys.exit(1)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    # Initialize pose model
    logger.info("Initializing MediaPipe Pose model...")
    pose_model, mp_drawing, mp_pose = get_pose_extractor(
        args.model_complexity,
        args.min_detection_confidence,
        args.min_tracking_confidence
    )

    all_results = []

    if args.splits:
        # Process specific splits
        for split in args.splits:
            images_dir = data_root / 'images' / split
            labels_dir = data_root / 'labels' / split
            split_output = output_dir / split

            if not images_dir.exists():
                logger.warning(f"Split directory not found: {images_dir}")
                continue

            logger.info(f"\nProcessing {split} split...")
            results = process_directory(
                images_dir,
                labels_dir if labels_dir.exists() else None,
                split_output,
                pose_model,
                mp_drawing,
                mp_pose,
                args.use_yolo_crops,
                args.save_visualizations
            )

            for r in results:
                r['split'] = split
            all_results.extend(results)
    else:
        # Process images directory directly
        images_dir = data_root / 'images'
        labels_dir = data_root / 'labels'

        if not images_dir.exists():
            logger.error(f"Images directory not found: {images_dir}")
            sys.exit(1)

        logger.info("\nProcessing images...")
        results = process_directory(
            images_dir,
            labels_dir if labels_dir.exists() else None,
            output_dir,
            pose_model,
            mp_drawing,
            mp_pose,
            args.use_yolo_crops,
            args.save_visualizations
        )
        all_results.extend(results)

    # Save summary CSV
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = output_dir / 'pose_extraction_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\nSummary saved to {summary_path}")

        # Print statistics
        logger.info("\n" + "=" * 50)
        logger.info("EXTRACTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total images processed: {len(all_results)}")
        logger.info(f"Images with poses detected: {sum(r['pose_saved'] for r in all_results)}")
        logger.info(f"Average persons per image: {np.mean([r['detected_persons'] for r in all_results]):.2f}")
        logger.info(f"Average visibility score: {np.mean([r['avg_visibility'] for r in all_results if r['avg_visibility'] > 0]):.4f}")

    pose_model.close()
    logger.info("\nPose extraction complete!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
