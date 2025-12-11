#!/usr/bin/env python3
"""
train_lstm.py - Training script for LSTM-based pose sequence classifier.

This script trains an LSTM model for fight/aggressive pose detection
using extracted pose sequences.

Usage:
    python train_lstm.py --data sequences --epochs 60 --batch_size 32 --lr 1e-3
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.lstm import create_lstm_model
from utils.dataset import PoseSequenceDataset, AugmentedPoseDataset, create_dataloaders
from utils.metrics import compute_metrics, compute_confusion_matrix, classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train LSTM model for fight detection'
    )
    parser.add_argument('--data', type=str, default='sequences',
                        help='Path to sequences directory')
    parser.add_argument('--output', type=str, default='outputs/lstm',
                        help='Output directory for models and logs')
    parser.add_argument('--model_type', type=str, default='bilstm',
                        choices=['lstm', 'bilstm', 'conv_lstm'],
                        help='LSTM model type')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--early_stop', type=int, default=15,
                        help='Early stopping patience (0 to disable)')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights for imbalanced data')
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='fight-detection',
                        help='W&B project name')
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    """Get torch device."""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (sequences, labels) in enumerate(dataloader):
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(sequences)
        loss = criterion(outputs, labels)

        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * sequences.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * sequences.size(0)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def main():
    """Main training function."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Load data configuration
    data_dir = Path(args.data)
    data_config_path = data_dir / 'config.yaml'
    if data_config_path.exists():
        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)
        class_names = data_config.get('class_names', ['normal', 'aggressive'])
    else:
        class_names = ['normal', 'aggressive']

    num_classes = len(class_names)

    # Determine input size from data
    train_seqs = np.load(data_dir / 'train' / 'sequences.npy')
    input_size = train_seqs.shape[-1]
    logger.info(f"Input size: {input_size}, Num classes: {num_classes}")

    # Create datasets
    if args.augment:
        train_dataset = AugmentedPoseDataset(
            data_dir / 'train' / 'sequences.npy',
            data_dir / 'train' / 'labels.npy',
            augment_prob=0.5,
            normalize=True
        )
    else:
        train_dataset = PoseSequenceDataset(
            data_dir / 'train' / 'sequences.npy',
            data_dir / 'train' / 'labels.npy',
            normalize=True
        )

    val_dataset = PoseSequenceDataset(
        data_dir / 'val' / 'sequences.npy',
        data_dir / 'val' / 'labels.npy',
        normalize=True
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model
    model = create_lstm_model(
        model_type=args.model_type,
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(device)

    logger.info(f"Model: {args.model_type}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function
    if args.use_class_weights:
        class_weights = train_dataset.get_class_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"Using class weights: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    else:
        scheduler = None

    # Initialize W&B if requested
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=f'{args.model_type}_h{args.hidden_size}_l{args.num_layers}',
                config=config
            )
            wandb.watch(model)
        except ImportError:
            logger.warning("wandb not installed, skipping")
            args.wandb = False

    # Training loop
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }

    logger.info("\nStarting training...")
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate
        val_loss, val_acc, val_preds, val_labels, val_probs = evaluate(
            model, val_loader, criterion, device
        )

        # Compute metrics
        metrics = compute_metrics(val_labels, val_preds, val_probs)
        val_f1 = metrics['f1']

        # Update scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Log
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s) - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
        )

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        # W&B logging
        if args.wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'lr': optimizer.param_groups[0]['lr']
            })

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'config': config
            }, output_dir / 'best_model.pt')

            logger.info(f"  Saved best model (F1: {val_f1:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if args.early_stop > 0 and patience_counter >= args.early_stop:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Training complete
    total_time = time.time() - start_time
    logger.info(f"\nTraining complete in {total_time / 60:.1f} minutes")
    logger.info(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, output_dir / 'final_model.pt')

    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f)

    # Final evaluation on best model
    logger.info("\nFinal evaluation on best model...")
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    _, _, val_preds, val_labels, val_probs = evaluate(model, val_loader, criterion, device)

    # Print classification report
    report = classification_report(val_labels, val_preds, class_names)
    logger.info(f"\nClassification Report:\n{report}")

    # Save report
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(report)

    # Confusion matrix
    cm = compute_confusion_matrix(val_labels, val_preds)
    np.save(output_dir / 'confusion_matrix.npy', cm)

    # Plot training curves
    try:
        from utils.visualization import plot_training_curves
        plot_training_curves(history, save_path=str(output_dir / 'training_curves.png'))
    except Exception as e:
        logger.warning(f"Could not plot training curves: {e}")

    if args.wandb:
        wandb.finish()

    logger.info(f"\nAll outputs saved to: {output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
