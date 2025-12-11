#!/usr/bin/env python3
"""
eval.py - Comprehensive evaluation script for fight detection models.

This script evaluates trained models (LSTM, ST-GCN, Transformer, YOLOv8) and
generates detailed metrics including accuracy, precision, recall, F1, confusion
matrix, ROC curves, and temporal detection metrics.

Usage:
    python eval.py --model outputs/lstm/best_model.pt --data sequences --type lstm
    python eval.py --model outputs/yolov8/train/weights/best.pt --data yolo_split --type yolo
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.metrics import (
    compute_metrics,
    compute_confusion_matrix,
    plot_confusion_matrix,
    compute_roc_auc,
    plot_precision_recall_curve,
    classification_report,
    compute_segment_iou,
    compute_temporal_map
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate fight detection models'
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--type', type=str, required=True,
                        choices=['lstm', 'stgcn', 'transformer', 'yolo'],
                        help='Model type')
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'],
                        help='Data split to evaluate on')
    parser.add_argument('--output', type=str, default='evaluation',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='Generate plots')
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """Get torch device."""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def load_pose_model(
    model_path: str,
    model_type: str,
    device: torch.device
) -> Tuple[nn.Module, Dict]:
    """Load a pose-based model (LSTM, ST-GCN, or Transformer)."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})

    if model_type == 'lstm':
        from models.lstm import create_lstm_model
        model = create_lstm_model(
            model_type=config.get('model_type', 'bilstm'),
            input_size=config.get('input_size', 66),
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 2),
            num_classes=2,
            dropout=config.get('dropout', 0.3)
        )
    elif model_type == 'stgcn':
        from models.stgcn import create_stgcn_model
        model = create_stgcn_model(
            model_type=config.get('model_type', 'lightweight'),
            num_classes=2,
            num_joints=config.get('num_joints', 33),
            dropout=config.get('dropout', 0.0)
        )
    elif model_type == 'transformer':
        from models.transformer import create_transformer_model
        model = create_transformer_model(
            model_type='transformer',
            input_dim=config.get('input_dim', 66),
            embed_dim=config.get('embed_dim', 128),
            num_classes=2,
            dropout=config.get('dropout', 0.1)
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config


def evaluate_pose_model(
    model: nn.Module,
    data_dir: Path,
    split: str,
    device: torch.device,
    batch_size: int = 32,
    model_type: str = 'lstm'
) -> Dict:
    """Evaluate pose-based model."""
    from utils.dataset import PoseSequenceDataset

    # Load dataset
    if model_type == 'stgcn':
        from train_stgcn import STGCNDataset
        dataset = STGCNDataset(
            data_dir / split / 'sequences.npy',
            data_dir / split / 'labels.npy',
            augment=False
        )
    else:
        dataset = PoseSequenceDataset(
            data_dir / split / 'sequences.npy',
            data_dir / split / 'labels.npy',
            normalize=True
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)

            outputs = model(sequences)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }


def evaluate_yolo_model(
    model_path: str,
    data_yaml: str,
    split: str = 'test'
) -> Dict:
    """Evaluate YOLOv8 model."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics package not installed")
        return {}

    model = YOLO(model_path)

    results = model.val(data=data_yaml, split=split, verbose=False)

    return {
        'map50': results.box.map50,
        'map': results.box.map,
        'precision': results.box.mp,
        'recall': results.box.mr,
        'per_class_ap50': results.box.ap50.tolist() if hasattr(results.box, 'ap50') else [],
        'per_class_ap': results.box.ap.tolist() if hasattr(results.box, 'ap') else []
    }


def generate_evaluation_report(
    results: Dict,
    output_dir: Path,
    class_names: List[str],
    model_type: str,
    plot: bool = True
) -> Dict:
    """Generate comprehensive evaluation report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {}

    if model_type == 'yolo':
        # YOLO-specific metrics
        report['detection'] = {
            'mAP@0.5': results['map50'],
            'mAP@0.5:0.95': results['map'],
            'precision': results['precision'],
            'recall': results['recall'],
            'per_class_ap50': dict(zip(class_names, results.get('per_class_ap50', []))),
            'per_class_ap': dict(zip(class_names, results.get('per_class_ap', [])))
        }

        logger.info("\nYOLO Detection Metrics:")
        logger.info(f"  mAP@0.5: {results['map50']:.4f}")
        logger.info(f"  mAP@0.5:0.95: {results['map']:.4f}")
        logger.info(f"  Precision: {results['precision']:.4f}")
        logger.info(f"  Recall: {results['recall']:.4f}")

    else:
        # Classification metrics
        preds = results['predictions']
        labels = results['labels']
        probs = results['probabilities']

        metrics = compute_metrics(labels, preds, probs)
        report['classification'] = metrics

        # Classification report
        cls_report = classification_report(labels, preds, class_names, output_dict=True)
        report['per_class'] = cls_report

        # Confusion matrix
        cm = compute_confusion_matrix(labels, preds)
        report['confusion_matrix'] = cm.tolist()

        logger.info("\nClassification Metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

        # Print classification report
        text_report = classification_report(labels, preds, class_names)
        logger.info(f"\nClassification Report:\n{text_report}")

        # Save text report
        with open(output_dir / 'classification_report.txt', 'w') as f:
            f.write(text_report)

        if plot:
            # Confusion matrix plot
            fig = plot_confusion_matrix(
                cm, class_names,
                save_path=str(output_dir / 'confusion_matrix.png')
            )
            plt.close(fig)

            # ROC curve
            if len(np.unique(labels)) == 2:
                _, fig = compute_roc_auc(
                    labels, probs[:, 1],
                    plot=True,
                    save_path=str(output_dir / 'roc_curve.png')
                )
                if fig:
                    plt.close(fig)

                # PR curve
                fig = plot_precision_recall_curve(
                    labels, probs[:, 1],
                    save_path=str(output_dir / 'pr_curve.png')
                )
                plt.close(fig)

            logger.info(f"\nPlots saved to: {output_dir}")

    # Save full report
    with open(output_dir / 'evaluation_report.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_report = {}
        for key, value in report.items():
            if isinstance(value, np.ndarray):
                json_report[key] = value.tolist()
            elif isinstance(value, dict):
                json_report[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                json_report[key] = value
        json.dump(json_report, f, indent=2)

    return report


def main():
    """Main evaluation function."""
    args = parse_args()

    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output)
    data_dir = Path(args.data)

    # Load class names
    config_path = data_dir / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            data_config = yaml.safe_load(f)
        class_names = data_config.get('class_names', ['normal', 'aggressive'])
    else:
        # Try data.yaml for YOLO
        yaml_path = data_dir / 'data.yaml'
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            class_names = data_config.get('names', ['aggressive_pose'])
        else:
            class_names = ['normal', 'aggressive']

    logger.info(f"Class names: {class_names}")
    logger.info(f"Evaluating model: {args.model}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Split: {args.split}")

    if args.type == 'yolo':
        # YOLO evaluation
        data_yaml = str(data_dir / 'data.yaml')
        results = evaluate_yolo_model(args.model, data_yaml, args.split)
    else:
        # Pose model evaluation
        model, config = load_pose_model(args.model, args.type, device)
        results = evaluate_pose_model(
            model, data_dir, args.split, device,
            args.batch_size, args.type
        )

    # Generate report
    report = generate_evaluation_report(
        results, output_dir, class_names, args.type, args.plot
    )

    logger.info(f"\nEvaluation complete! Results saved to: {output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
