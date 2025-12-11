"""
Dataset utilities for loading pose sequences for fight detection training.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class PoseSequenceDataset(Dataset):
    """
    PyTorch Dataset for pose sequences.

    Loads sequences from .npy files and returns tensors for training.
    """

    def __init__(
        self,
        sequences_path: Union[str, Path],
        labels_path: Union[str, Path],
        transform: Optional[callable] = None,
        normalize: bool = False,
        return_indices: bool = False
    ):
        """
        Initialize dataset.

        Args:
            sequences_path: Path to sequences.npy file
            labels_path: Path to labels.npy file
            transform: Optional transform to apply to sequences
            normalize: Whether to z-normalize sequences
            return_indices: Whether to return sample indices
        """
        self.sequences = np.load(sequences_path)
        self.labels = np.load(labels_path)
        self.transform = transform
        self.normalize = normalize
        self.return_indices = return_indices

        # Compute normalization statistics
        if normalize:
            self.mean = self.sequences.mean(axis=(0, 1), keepdims=True)
            self.std = self.sequences.std(axis=(0, 1), keepdims=True)
            self.std[self.std < 1e-6] = 1.0
        else:
            self.mean = 0
            self.std = 1

        assert len(self.sequences) == len(self.labels), \
            f"Mismatch: {len(self.sequences)} sequences vs {len(self.labels)} labels"

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, int]]:
        sequence = self.sequences[idx].copy()
        label = self.labels[idx]

        # Normalize
        if self.normalize:
            sequence = (sequence - self.mean.squeeze()) / self.std.squeeze()

        # Apply transform
        if self.transform:
            sequence = self.transform(sequence)

        # Convert to tensor
        sequence = torch.FloatTensor(sequence)
        label = torch.LongTensor([label])[0]

        if self.return_indices:
            return sequence, label, idx
        return sequence, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for handling imbalanced data."""
        unique, counts = np.unique(self.labels, return_counts=True)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(unique)
        return torch.FloatTensor(weights)

    def get_sample_weights(self) -> np.ndarray:
        """Get per-sample weights for weighted sampling."""
        unique, counts = np.unique(self.labels, return_counts=True)
        class_weights = {c: 1.0 / count for c, count in zip(unique, counts)}
        return np.array([class_weights[label] for label in self.labels])


class AugmentedPoseDataset(PoseSequenceDataset):
    """Dataset with online augmentation for training."""

    def __init__(
        self,
        sequences_path: Union[str, Path],
        labels_path: Union[str, Path],
        augment_prob: float = 0.5,
        **kwargs
    ):
        super().__init__(sequences_path, labels_path, **kwargs)
        self.augment_prob = augment_prob

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sequence = self.sequences[idx].copy()
        label = self.labels[idx]

        # Apply random augmentations
        if np.random.random() < self.augment_prob:
            sequence = self._augment(sequence)

        # Normalize
        if self.normalize:
            sequence = (sequence - self.mean.squeeze()) / self.std.squeeze()

        sequence = torch.FloatTensor(sequence)
        label = torch.LongTensor([label])[0]

        return sequence, label

    def _augment(self, sequence: np.ndarray) -> np.ndarray:
        """Apply random augmentations."""
        T = sequence.shape[0]

        # Random noise
        if np.random.random() > 0.5:
            noise = np.random.randn(*sequence.shape) * 0.01
            sequence = sequence + noise

        # Random scale
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            sequence = sequence * scale

        # Temporal jitter (random frame drop and duplicate)
        if np.random.random() > 0.5 and T > 2:
            drop_idx = np.random.randint(0, T)
            sequence = np.delete(sequence, drop_idx, axis=0)
            dup_idx = np.random.randint(0, T - 1)
            sequence = np.insert(sequence, dup_idx, sequence[dup_idx], axis=0)

        return sequence


class RealTimeBuffer:
    """
    Buffer for real-time sequence inference.

    Maintains a sliding window of poses for temporal classification.
    """

    def __init__(
        self,
        window_size: int = 16,
        stride: int = 8,
        normalize_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        self.window_size = window_size
        self.stride = stride
        self.buffer = []
        self.frame_count = 0
        self.mean, self.std = normalize_stats if normalize_stats else (0, 1)

    def add_pose(self, pose: np.ndarray) -> Optional[np.ndarray]:
        """
        Add a pose to the buffer and return sequence if window is full.

        Args:
            pose: Pose array of shape (33, 3) or flattened

        Returns:
            Sequence array if window is full and stride is reached, else None
        """
        self.buffer.append(pose)
        self.frame_count += 1

        # Check if we have enough frames
        if len(self.buffer) >= self.window_size:
            # Check if we've reached the stride
            if (self.frame_count - self.window_size) % self.stride == 0:
                sequence = np.array(self.buffer[-self.window_size:])

                # Normalize
                if isinstance(self.mean, np.ndarray):
                    sequence = (sequence - self.mean) / self.std

                return sequence

        return None

    def reset(self):
        """Reset the buffer."""
        self.buffer = []
        self.frame_count = 0


def create_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    augment_train: bool = True,
    weighted_sampling: bool = True,
    normalize: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, val, and optional test dataloaders.

    Args:
        data_dir: Directory containing train/val/test subdirectories
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        augment_train: Whether to augment training data
        weighted_sampling: Whether to use weighted random sampling
        normalize: Whether to z-normalize sequences

    Returns:
        Tuple of (train_loader, val_loader, test_loader or None)
    """
    data_dir = Path(data_dir)

    # Create training dataset
    train_dir = data_dir / 'train'
    if augment_train:
        train_dataset = AugmentedPoseDataset(
            train_dir / 'sequences.npy',
            train_dir / 'labels.npy',
            augment_prob=0.5,
            normalize=normalize
        )
    else:
        train_dataset = PoseSequenceDataset(
            train_dir / 'sequences.npy',
            train_dir / 'labels.npy',
            normalize=normalize
        )

    # Create sampler for imbalanced data
    if weighted_sampling:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

    # Create validation dataset
    val_dir = data_dir / 'val'
    val_dataset = PoseSequenceDataset(
        val_dir / 'sequences.npy',
        val_dir / 'labels.npy',
        normalize=normalize
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Create test dataset if exists
    test_loader = None
    test_dir = data_dir / 'test'
    if test_dir.exists() and (test_dir / 'sequences.npy').exists():
        test_dataset = PoseSequenceDataset(
            test_dir / 'sequences.npy',
            test_dir / 'labels.npy',
            normalize=normalize
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader, test_loader
