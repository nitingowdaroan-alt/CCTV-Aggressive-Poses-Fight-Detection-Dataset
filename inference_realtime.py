#!/usr/bin/env python3
"""
inference_realtime.py - Real-time inference pipeline for fight detection.

This script provides real-time fight/aggressive pose detection using:
- Option A (Light): MediaPipe pose + LSTM/ST-GCN classifier
- Option B (Heavy): YOLOv8 detection + optional tracking + classifier

Usage:
    # Webcam inference with pose model
    python inference_realtime.py --source 0 --model outputs/lstm/best_model.pt --type lstm

    # Video file inference
    python inference_realtime.py --source video.mp4 --model outputs/lstm/best_model.pt --type lstm

    # YOLOv8 detection
    python inference_realtime.py --source 0 --model outputs/yolov8/best.pt --type yolo
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Real-time fight detection inference'
    )
    parser.add_argument('--source', type=str, default='0',
                        help='Video source (0 for webcam, or video file path)')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--type', type=str, default='lstm',
                        choices=['lstm', 'stgcn', 'transformer', 'yolo'],
                        help='Model type')
    parser.add_argument('--window_size', type=int, default=16,
                        help='Temporal window size for pose models')
    parser.add_argument('--stride', type=int, default=8,
                        help='Inference stride (process every N frames)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--smooth_window', type=int, default=5,
                        help='Smoothing window for predictions')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (optional)')
    parser.add_argument('--show', action='store_true', default=True,
                        help='Display output window')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='YOLO confidence threshold')
    return parser.parse_args()


class PoseExtractor:
    """MediaPipe pose extraction for real-time inference."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        import mediapipe as mp

        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def extract(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract pose from frame.

        Args:
            frame: BGR image

        Returns:
            Keypoints array (33, 3) or None if no pose detected
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks:
            keypoints = np.zeros((33, 3), dtype=np.float32)
            for i, lm in enumerate(results.pose_landmarks.landmark):
                keypoints[i] = [lm.x, lm.y, lm.visibility]
            return keypoints

        return None

    def draw(self, frame: np.ndarray, keypoints: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw pose on frame."""
        if keypoints is None:
            return frame

        frame = frame.copy()
        h, w = frame.shape[:2]

        # Define skeleton connections
        connections = [
            (11, 12), (11, 23), (12, 24), (23, 24),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
            (23, 25), (25, 27), (24, 26), (26, 28),  # Legs
        ]

        # Draw connections
        for i, j in connections:
            if keypoints[i, 2] > 0.5 and keypoints[j, 2] > 0.5:
                pt1 = (int(keypoints[i, 0] * w), int(keypoints[i, 1] * h))
                pt2 = (int(keypoints[j, 0] * w), int(keypoints[j, 1] * h))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Draw keypoints
        for i in range(33):
            if keypoints[i, 2] > 0.5:
                pt = (int(keypoints[i, 0] * w), int(keypoints[i, 1] * h))
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)

        return frame

    def close(self):
        self.pose.close()


class TemporalBuffer:
    """Sliding window buffer for temporal classification."""

    def __init__(
        self,
        window_size: int = 16,
        stride: int = 8,
        feature_dim: int = 66
    ):
        self.window_size = window_size
        self.stride = stride
        self.feature_dim = feature_dim
        self.buffer = deque(maxlen=window_size)
        self.frame_count = 0

    def add(self, features: np.ndarray) -> Optional[np.ndarray]:
        """
        Add features to buffer.

        Returns:
            Sequence array if ready for inference, else None
        """
        # Flatten features
        if features.ndim > 1:
            features = features.flatten()[:self.feature_dim]

        self.buffer.append(features)
        self.frame_count += 1

        # Check if we have enough frames and it's time to process
        if len(self.buffer) >= self.window_size:
            if (self.frame_count - self.window_size) % self.stride == 0:
                return np.array(self.buffer)

        return None

    def reset(self):
        self.buffer.clear()
        self.frame_count = 0


class PredictionSmoother:
    """Smooth predictions over time."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.scores = deque(maxlen=window_size)

    def add(self, prediction: int, score: float) -> Tuple[int, float]:
        """Add prediction and return smoothed result."""
        self.predictions.append(prediction)
        self.scores.append(score)

        if len(self.predictions) == 0:
            return prediction, score

        # Majority vote for prediction
        smoothed_pred = max(set(self.predictions), key=list(self.predictions).count)

        # Average score
        smoothed_score = np.mean(self.scores)

        return smoothed_pred, smoothed_score


def load_pose_model(model_path: str, model_type: str, device: torch.device):
    """Load pose classification model."""
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
            dropout=0
        )
    elif model_type == 'stgcn':
        from models.stgcn import create_stgcn_model
        model = create_stgcn_model(
            model_type=config.get('model_type', 'lightweight'),
            num_classes=2,
            num_joints=33,
            dropout=0
        )
    elif model_type == 'transformer':
        from models.transformer import create_transformer_model
        model = create_transformer_model(
            model_type='transformer',
            input_dim=config.get('input_dim', 66),
            embed_dim=config.get('embed_dim', 128),
            num_classes=2,
            dropout=0
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config


def run_pose_inference(
    source: str,
    model_path: str,
    model_type: str,
    window_size: int = 16,
    stride: int = 8,
    threshold: float = 0.5,
    smooth_window: int = 5,
    device: str = 'auto',
    output_path: Optional[str] = None,
    show: bool = True
):
    """Run real-time inference with pose-based model."""
    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model: {model_path}")
    model, config = load_pose_model(model_path, model_type, device)

    # Initialize components
    pose_extractor = PoseExtractor()
    temporal_buffer = TemporalBuffer(window_size, stride)
    smoother = PredictionSmoother(smooth_window)

    # Open video source
    if source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error(f"Could not open video source: {source}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup video writer if output specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Class names and colors
    class_names = ['Normal', 'Aggressive']
    colors = [(0, 255, 0), (0, 0, 255)]

    # Tracking variables
    frame_count = 0
    inference_times = []
    current_pred = 0
    current_score = 0.0

    logger.info("Starting inference... Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start_time = time.time()

        # Extract pose
        keypoints = pose_extractor.extract(frame)

        if keypoints is not None:
            # Add to buffer and check if ready for inference
            sequence = temporal_buffer.add(keypoints[:, :2])  # Use only x, y

            if sequence is not None:
                # Prepare input
                if model_type == 'stgcn':
                    # Reshape for ST-GCN: (1, T, V, C)
                    input_tensor = sequence.reshape(1, window_size, 33, 2)
                    # Add confidence channel
                    conf = np.ones((1, window_size, 33, 1))
                    input_tensor = np.concatenate([input_tensor, conf], axis=-1)
                else:
                    # (1, T, features)
                    input_tensor = sequence.reshape(1, window_size, -1)

                input_tensor = torch.FloatTensor(input_tensor).to(device)

                # Inference
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred = probs.argmax(dim=1).item()
                    score = probs[0, pred].item()

                # Smooth prediction
                current_pred, current_score = smoother.add(pred, score)

        # Calculate FPS
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        if len(inference_times) > 30:
            inference_times.pop(0)
        avg_fps = 1.0 / np.mean(inference_times)

        # Draw visualization
        if keypoints is not None:
            frame = pose_extractor.draw(frame, keypoints)

        # Draw prediction
        color = colors[current_pred]
        label = f"{class_names[current_pred]}: {current_score:.2f}"

        # Draw label background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(frame, (10, 10), (20 + text_w, 25 + text_h + 10), (0, 0, 0), -1)
        cv2.putText(frame, label, (15, 25 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Draw FPS
        fps_text = f"FPS: {avg_fps:.1f}"
        cv2.putText(frame, fps_text, (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Write frame
        if writer:
            writer.write(frame)

        # Show frame
        if show:
            cv2.imshow('Fight Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    pose_extractor.close()

    logger.info(f"Processed {frame_count} frames")
    logger.info(f"Average FPS: {1.0 / np.mean(inference_times):.1f}")


def run_yolo_inference(
    source: str,
    model_path: str,
    conf: float = 0.5,
    output_path: Optional[str] = None,
    show: bool = True
):
    """Run real-time inference with YOLOv8."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics package not installed")
        return

    logger.info(f"Loading YOLOv8 model: {model_path}")
    model = YOLO(model_path)

    # Run inference
    results = model(
        source,
        stream=True,
        conf=conf,
        show=show,
        save=output_path is not None,
        project=str(Path(output_path).parent) if output_path else None,
        name=Path(output_path).stem if output_path else None
    )

    # Process results
    for r in results:
        # Results are processed in the stream
        pass

    logger.info("Inference complete")


def main():
    """Main function."""
    args = parse_args()

    if args.type == 'yolo':
        run_yolo_inference(
            args.source,
            args.model,
            args.conf,
            args.output,
            args.show
        )
    else:
        run_pose_inference(
            args.source,
            args.model,
            args.type,
            args.window_size,
            args.stride,
            args.threshold,
            args.smooth_window,
            args.device,
            args.output,
            args.show
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
