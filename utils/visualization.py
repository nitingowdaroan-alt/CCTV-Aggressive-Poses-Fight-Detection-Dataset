"""
Visualization utilities for fight detection system.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import cv2


# MediaPipe skeleton connections
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Left face
    (0, 4), (4, 5), (5, 6), (6, 8),  # Right face
    (9, 10),  # Mouth
    (11, 12),  # Shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # Left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # Right arm
    (11, 23), (12, 24), (23, 24),  # Torso
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),  # Left leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),  # Right leg
]


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = 'Training History'
) -> plt.Figure:
    """
    Plot training and validation curves.

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc' etc.
        save_path: Path to save the figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax = axes[0]
    if 'train_loss' in history:
        ax.plot(history['train_loss'], label='Train Loss', color='blue')
    if 'val_loss' in history:
        ax.plot(history['val_loss'], label='Val Loss', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{title} - Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy/F1 curves
    ax = axes[1]
    if 'train_acc' in history:
        ax.plot(history['train_acc'], label='Train Acc', color='blue')
    if 'val_acc' in history:
        ax.plot(history['val_acc'], label='Val Acc', color='orange')
    if 'train_f1' in history:
        ax.plot(history['train_f1'], label='Train F1', color='green', linestyle='--')
    if 'val_f1' in history:
        ax.plot(history['val_f1'], label='Val F1', color='red', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric')
    ax.set_title(f'{title} - Accuracy/F1')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_pose(
    keypoints: np.ndarray,
    image: Optional[np.ndarray] = None,
    image_size: Tuple[int, int] = (640, 480),
    save_path: Optional[str] = None,
    title: str = 'Pose',
    show_indices: bool = False
) -> plt.Figure:
    """
    Visualize a single pose.

    Args:
        keypoints: Array of shape (33, 2) or (33, 3) with normalized coordinates
        image: Optional background image
        image_size: Size of output image if no background provided
        save_path: Path to save the figure
        title: Plot title
        show_indices: Whether to show joint indices

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if image is not None:
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        h, w = image.shape[:2]
    else:
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(1.5, -0.5)  # Flip y-axis
        w, h = 1, 1

    # Extract coordinates
    if keypoints.shape[-1] == 3:
        coords = keypoints[:, :2]
        visibility = keypoints[:, 2]
    else:
        coords = keypoints
        visibility = np.ones(len(keypoints))

    # Scale to image size if needed
    if image is not None:
        coords = coords.copy()
        coords[:, 0] *= w
        coords[:, 1] *= h

    # Draw skeleton connections
    for i, j in POSE_CONNECTIONS:
        if visibility[i] > 0.3 and visibility[j] > 0.3:
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color='lime', linewidth=2, alpha=0.7
            )

    # Draw keypoints
    visible_mask = visibility > 0.3
    ax.scatter(
        coords[visible_mask, 0],
        coords[visible_mask, 1],
        c='red', s=30, zorder=5
    )

    # Add indices
    if show_indices:
        for i, (x, y) in enumerate(coords):
            if visibility[i] > 0.3:
                ax.annotate(str(i), (x, y), fontsize=6, color='white')

    ax.set_title(title)
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_sequence(
    sequence: np.ndarray,
    num_frames: int = 8,
    save_path: Optional[str] = None,
    title: str = 'Sequence'
) -> plt.Figure:
    """
    Visualize a sequence of poses.

    Args:
        sequence: Array of shape (T, 33, 2/3) or (T, 66)
        num_frames: Number of frames to display
        save_path: Path to save the figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    T = sequence.shape[0]

    # Reshape if flattened
    if sequence.ndim == 2 and sequence.shape[1] == 66:
        sequence = sequence.reshape(T, 33, 2)

    # Select frames to display
    if T <= num_frames:
        frame_indices = list(range(T))
    else:
        frame_indices = np.linspace(0, T - 1, num_frames, dtype=int)

    # Create subplot grid
    cols = min(4, len(frame_indices))
    rows = (len(frame_indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, frame_idx in enumerate(frame_indices):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        pose = sequence[frame_idx]

        # Handle different input shapes
        if pose.shape[-1] == 3:
            coords = pose[:, :2]
            visibility = pose[:, 2]
        else:
            coords = pose.reshape(-1, 2) if pose.ndim == 1 else pose
            visibility = np.ones(33)

        # Draw skeleton
        for i, j in POSE_CONNECTIONS:
            if visibility[i] > 0.3 and visibility[j] > 0.3:
                ax.plot(
                    [coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    color='lime', linewidth=2, alpha=0.7
                )

        # Draw keypoints
        visible_mask = visibility > 0.3
        ax.scatter(
            coords[visible_mask, 0],
            coords[visible_mask, 1],
            c='red', s=20, zorder=5
        )

        ax.set_xlim(-1, 1)
        ax.set_ylim(1, -1)  # Flip y-axis
        ax.set_title(f'Frame {frame_idx}')
        ax.axis('off')

    # Hide empty subplots
    for idx in range(len(frame_indices), rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def draw_pose_on_frame(
    frame: np.ndarray,
    keypoints: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    confidence_threshold: float = 0.3
) -> np.ndarray:
    """
    Draw pose skeleton on a video frame.

    Args:
        frame: BGR image array
        keypoints: Array of shape (33, 3) with normalized coordinates
        color: BGR color for skeleton
        thickness: Line thickness
        confidence_threshold: Minimum visibility to draw

    Returns:
        Frame with pose drawn
    """
    frame = frame.copy()
    h, w = frame.shape[:2]

    # Extract coordinates
    coords = keypoints[:, :2].copy()
    coords[:, 0] *= w
    coords[:, 1] *= h
    coords = coords.astype(int)

    visibility = keypoints[:, 2] if keypoints.shape[1] == 3 else np.ones(33)

    # Draw connections
    for i, j in POSE_CONNECTIONS:
        if visibility[i] > confidence_threshold and visibility[j] > confidence_threshold:
            pt1 = tuple(coords[i])
            pt2 = tuple(coords[j])
            cv2.line(frame, pt1, pt2, color, thickness)

    # Draw keypoints
    for i, (x, y) in enumerate(coords):
        if visibility[i] > confidence_threshold:
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    return frame


def create_video_with_predictions(
    frames: List[np.ndarray],
    predictions: List[Dict],
    output_path: str,
    fps: int = 30,
    class_names: List[str] = ['Normal', 'Aggressive']
) -> None:
    """
    Create video with prediction overlays.

    Args:
        frames: List of BGR frames
        predictions: List of dicts with 'label', 'confidence', optional 'keypoints'
        output_path: Path to save output video
        fps: Frames per second
        class_names: List of class names
    """
    if not frames:
        return

    h, w = frames[0].shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame_idx, (frame, pred) in enumerate(zip(frames, predictions)):
        frame = frame.copy()

        # Draw pose if available
        if 'keypoints' in pred and pred['keypoints'] is not None:
            color = (0, 255, 0) if pred['label'] == 0 else (0, 0, 255)
            frame = draw_pose_on_frame(frame, pred['keypoints'], color=color)

        # Draw label and confidence
        label = pred.get('label', 0)
        confidence = pred.get('confidence', 1.0)
        class_name = class_names[label] if label < len(class_names) else f'Class {label}'

        # Set text color based on class
        text_color = (0, 255, 0) if label == 0 else (0, 0, 255)

        # Draw background rectangle for text
        text = f'{class_name}: {confidence:.2f}'
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (10, 10), (20 + text_w, 20 + text_h + 10), (0, 0, 0), -1)

        # Draw text
        cv2.putText(
            frame, text, (15, 20 + text_h),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2
        )

        # Add frame number
        cv2.putText(
            frame, f'Frame: {frame_idx}', (w - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )

        writer.write(frame)

    writer.release()


def plot_attention_weights(
    attention_weights: np.ndarray,
    frame_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot attention weights for transformer models.

    Args:
        attention_weights: Array of shape (T, T) or (heads, T, T)
        frame_labels: Optional labels for frames
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    if attention_weights.ndim == 3:
        # Average over heads
        attention_weights = attention_weights.mean(axis=0)

    T = attention_weights.shape[0]

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(attention_weights, cmap='viridis')
    ax.figure.colorbar(im, ax=ax)

    if frame_labels:
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(frame_labels, rotation=45, ha='right')
        ax.set_yticklabels(frame_labels)
    else:
        ax.set_xticks(range(0, T, max(1, T // 10)))
        ax.set_yticks(range(0, T, max(1, T // 10)))

    ax.set_xlabel('Key Frame')
    ax.set_ylabel('Query Frame')
    ax.set_title('Temporal Attention Weights')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
