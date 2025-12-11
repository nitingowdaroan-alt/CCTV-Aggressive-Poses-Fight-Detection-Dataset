#!/usr/bin/env python3
"""
train_yolov8.py - Training script for YOLOv8 object detection.

This script trains a YOLOv8 model for frame-level fight/aggressive pose detection.

Usage:
    python train_yolov8.py --data yolo_split/data.yaml --model yolov8n.pt --epochs 50

Or using ultralytics CLI directly:
    yolo detect train model=yolov8n.pt data=yolo_split/data.yaml epochs=50 imgsz=640
"""

import argparse
import os
import sys
from pathlib import Path
import logging
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
        description='Train YOLOv8 model for aggressive pose detection'
    )
    parser.add_argument('--data', type=str, default='yolo_split/data.yaml',
                        help='Path to data.yaml file')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                        help='YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device(s) or "cpu"')
    parser.add_argument('--project', type=str, default='outputs/yolov8',
                        help='Project directory for saving results')
    parser.add_argument('--name', type=str, default='train',
                        help='Experiment name')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['SGD', 'Adam', 'AdamW', 'RMSProp'],
                        help='Optimizer')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Weight decay')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Enable data augmentation')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of dataloader workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    return parser.parse_args()


def verify_data_yaml(yaml_path: str) -> bool:
    """Verify data.yaml file exists and has required fields."""
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        logger.error(f"data.yaml not found at: {yaml_path}")
        return False

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    required_fields = ['train', 'val', 'nc', 'names']
    missing = [f for f in required_fields if f not in data]

    if missing:
        logger.error(f"Missing required fields in data.yaml: {missing}")
        return False

    # Check if paths exist
    for split in ['train', 'val']:
        split_path = Path(data[split])
        if not split_path.exists():
            # Try relative to yaml location
            alt_path = yaml_path.parent / split_path
            if not alt_path.exists():
                logger.warning(f"{split} path not found: {split_path}")

    logger.info(f"data.yaml verified: {data['nc']} classes - {data['names']}")
    return True


def main():
    """Main training function."""
    args = parse_args()

    # Verify data.yaml
    if not verify_data_yaml(args.data):
        logger.error("Please run split_dataset.py first to create the data split")
        sys.exit(1)

    # Try to import ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics package not installed. Install with: pip install ultralytics")
        sys.exit(1)

    # Create output directory
    Path(args.project).mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Training configuration
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'project': args.project,
        'name': args.name,
        'patience': args.patience,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'weight_decay': args.weight_decay,
        'augment': args.augment,
        'workers': args.workers,
        'exist_ok': True,
        'verbose': True,
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'plots': True,
    }

    # Add resume if specified
    if args.resume:
        train_args['resume'] = args.resume

    # Configure W&B if requested
    if args.wandb:
        os.environ['WANDB_PROJECT'] = 'fight-detection-yolo'

    # Save training config
    config_path = Path(args.project) / args.name / 'train_config.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(train_args, f)

    logger.info("\nStarting training...")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Data: {args.data}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch}")
    logger.info(f"  Image size: {args.imgsz}")
    logger.info(f"  Device: {args.device}")

    # Train
    results = model.train(**train_args)

    # Log results
    logger.info("\nTraining complete!")
    logger.info(f"Results saved to: {Path(args.project) / args.name}")

    # Validate on test set if available
    test_yaml = Path(args.data).parent / 'data.yaml'
    with open(test_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    if 'test' in data_config:
        logger.info("\nEvaluating on test set...")
        test_results = model.val(data=args.data, split='test')

        # Log test metrics
        logger.info(f"\nTest Results:")
        logger.info(f"  mAP@0.5: {test_results.box.map50:.4f}")
        logger.info(f"  mAP@0.5:0.95: {test_results.box.map:.4f}")
        logger.info(f"  Precision: {test_results.box.mp:.4f}")
        logger.info(f"  Recall: {test_results.box.mr:.4f}")

    # Export best model
    logger.info("\nExporting best model...")
    best_model_path = Path(args.project) / args.name / 'weights' / 'best.pt'

    if best_model_path.exists():
        best_model = YOLO(str(best_model_path))

        # Export to ONNX
        onnx_path = best_model.export(format='onnx', imgsz=args.imgsz, simplify=True)
        logger.info(f"  ONNX model saved to: {onnx_path}")

        # Export to TorchScript
        torchscript_path = best_model.export(format='torchscript', imgsz=args.imgsz)
        logger.info(f"  TorchScript model saved to: {torchscript_path}")

    return 0


def generate_training_commands():
    """Print example training commands."""
    print("""
YOLOv8 Training Commands
========================

1. Basic training with nano model:
   python train_yolov8.py --data yolo_split/data.yaml --model yolov8n.pt --epochs 50

2. Training with larger model for better accuracy:
   python train_yolov8.py --data yolo_split/data.yaml --model yolov8m.pt --epochs 100 --batch 8

3. Training with custom settings:
   python train_yolov8.py --data yolo_split/data.yaml --model yolov8s.pt --epochs 80 --imgsz 640 --batch 16 --lr0 0.01

4. Using Ultralytics CLI directly:
   yolo detect train model=yolov8n.pt data=yolo_split/data.yaml epochs=50 imgsz=640 batch=16

5. Training on CPU:
   python train_yolov8.py --data yolo_split/data.yaml --model yolov8n.pt --device cpu

6. Resume training:
   python train_yolov8.py --data yolo_split/data.yaml --resume outputs/yolov8/train/weights/last.pt

Model Selection Guide:
- yolov8n.pt: Fastest, lowest accuracy (~3M params)
- yolov8s.pt: Fast, good balance (~11M params)
- yolov8m.pt: Medium speed, better accuracy (~25M params)
- yolov8l.pt: Slower, high accuracy (~43M params)
- yolov8x.pt: Slowest, highest accuracy (~68M params)
""")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        generate_training_commands()
    else:
        sys.exit(main())
