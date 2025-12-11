#!/usr/bin/env python3
"""
monitor.py - Performance monitoring for fight detection inference.

This script monitors:
- Inference latency (mean, p50, p95, p99)
- Prediction distribution
- Confidence scores
- Model drift detection
- System resource usage

Usage:
    python scripts/monitor.py --model outputs/lstm/best_model.pt --source 0
"""

import argparse
import json
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading
import sys

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor inference performance metrics."""

    def __init__(
        self,
        log_dir: str = "logs/monitoring",
        window_size: int = 1000,
        alert_latency_ms: float = 100,
        alert_confidence: float = 0.3
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

        self.alert_latency_ms = alert_latency_ms
        self.alert_confidence = alert_confidence

        self.total_inferences = 0
        self.alerts = []
        self.start_time = datetime.now()

        # Baseline for drift detection
        self.baseline_predictions = None

    def log_inference(
        self,
        latency_ms: float,
        prediction: int,
        confidence: float,
        metadata: Optional[Dict] = None
    ):
        """Log a single inference result."""
        self.total_inferences += 1
        timestamp = datetime.now()

        self.latencies.append(latency_ms)
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        self.timestamps.append(timestamp)

        # Check for alerts
        self._check_alerts(latency_ms, confidence)

        # Log to file periodically
        if self.total_inferences % 100 == 0:
            self._log_metrics()

    def _check_alerts(self, latency_ms: float, confidence: float):
        """Check for performance issues and raise alerts."""
        alerts = []

        # High latency alert
        if latency_ms > self.alert_latency_ms:
            alerts.append({
                "type": "HIGH_LATENCY",
                "value": latency_ms,
                "threshold": self.alert_latency_ms,
                "timestamp": datetime.now().isoformat()
            })
            logger.warning(f"High latency alert: {latency_ms:.1f}ms > {self.alert_latency_ms}ms")

        # Low confidence alert
        if confidence < self.alert_confidence:
            alerts.append({
                "type": "LOW_CONFIDENCE",
                "value": confidence,
                "threshold": self.alert_confidence,
                "timestamp": datetime.now().isoformat()
            })
            logger.warning(f"Low confidence alert: {confidence:.2f} < {self.alert_confidence}")

        # Rolling average check
        if len(self.latencies) >= 10:
            avg_latency = np.mean(list(self.latencies)[-10:])
            if avg_latency > self.alert_latency_ms * 0.8:
                alerts.append({
                    "type": "SUSTAINED_HIGH_LATENCY",
                    "value": avg_latency,
                    "threshold": self.alert_latency_ms * 0.8,
                    "timestamp": datetime.now().isoformat()
                })

        self.alerts.extend(alerts)

    def _log_metrics(self):
        """Log current metrics to file."""
        metrics = self.get_stats()
        log_file = self.log_dir / f"metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"

        with open(log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

    def set_baseline(self, predictions: List[int]):
        """Set baseline predictions for drift detection."""
        self.baseline_predictions = np.array(predictions)
        logger.info(f"Baseline set with {len(predictions)} predictions")

    def check_drift(self, alpha: float = 0.05) -> Tuple[bool, float, str]:
        """
        Check for prediction distribution drift.

        Returns:
            (is_drift, p_value, message)
        """
        if self.baseline_predictions is None:
            return False, 1.0, "No baseline set"

        if len(self.predictions) < 50:
            return False, 1.0, "Insufficient data"

        from scipy import stats

        current = np.array(list(self.predictions))

        # Count predictions per class
        baseline_counts = np.bincount(self.baseline_predictions, minlength=2)
        current_counts = np.bincount(current, minlength=2)

        # Normalize to same sample size
        baseline_freq = baseline_counts / len(self.baseline_predictions)
        expected_counts = baseline_freq * len(current)

        # Chi-squared test
        if expected_counts.min() < 5:
            # Use Fisher's exact test for small samples
            from scipy.stats import fisher_exact
            contingency = np.array([baseline_counts, current_counts])
            _, p_value = fisher_exact(contingency[:, :2])
        else:
            chi2, p_value = stats.chisquare(current_counts, expected_counts)

        is_drift = p_value < alpha

        if is_drift:
            message = f"DRIFT DETECTED: p={p_value:.4f}, baseline={dict(enumerate(baseline_counts))}, current={dict(enumerate(current_counts))}"
            logger.warning(message)
        else:
            message = f"No drift: p={p_value:.4f}"

        return is_drift, p_value, message

    def get_stats(self) -> Dict:
        """Get current performance statistics."""
        if not self.latencies:
            return {"status": "no_data", "timestamp": datetime.now().isoformat()}

        latencies = list(self.latencies)
        confidences = list(self.confidences)
        predictions = list(self.predictions)

        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "total_inferences": self.total_inferences,
            "window_size": len(latencies),
            "latency": {
                "mean_ms": float(np.mean(latencies)),
                "std_ms": float(np.std(latencies)),
                "min_ms": float(min(latencies)),
                "max_ms": float(max(latencies)),
                "p50_ms": float(np.percentile(latencies, 50)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "p99_ms": float(np.percentile(latencies, 99))
            },
            "predictions": {
                "class_0_count": sum(1 for p in predictions if p == 0),
                "class_1_count": sum(1 for p in predictions if p == 1),
                "class_0_ratio": sum(1 for p in predictions if p == 0) / len(predictions),
                "class_1_ratio": sum(1 for p in predictions if p == 1) / len(predictions)
            },
            "confidence": {
                "mean": float(np.mean(confidences)),
                "std": float(np.std(confidences)),
                "min": float(min(confidences)),
                "max": float(max(confidences))
            },
            "throughput_fps": len(latencies) / sum(latencies) * 1000 if sum(latencies) > 0 else 0,
            "alerts_count": len(self.alerts)
        }

    def save_report(self) -> str:
        """Save comprehensive monitoring report."""
        stats = self.get_stats()
        stats["alerts"] = self.alerts[-100:]  # Last 100 alerts

        report_path = self.log_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Report saved to {report_path}")
        return str(report_path)

    def print_summary(self):
        """Print current statistics to console."""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("PERFORMANCE MONITORING SUMMARY")
        print("=" * 60)
        print(f"Uptime: {stats.get('uptime_seconds', 0):.1f} seconds")
        print(f"Total Inferences: {stats.get('total_inferences', 0)}")
        print(f"Throughput: {stats.get('throughput_fps', 0):.1f} FPS")

        if 'latency' in stats:
            print(f"\nLatency (ms):")
            print(f"  Mean: {stats['latency']['mean_ms']:.2f}")
            print(f"  P50:  {stats['latency']['p50_ms']:.2f}")
            print(f"  P95:  {stats['latency']['p95_ms']:.2f}")
            print(f"  P99:  {stats['latency']['p99_ms']:.2f}")

        if 'predictions' in stats:
            print(f"\nPredictions:")
            print(f"  Normal: {stats['predictions']['class_0_count']} ({stats['predictions']['class_0_ratio']:.1%})")
            print(f"  Aggressive: {stats['predictions']['class_1_count']} ({stats['predictions']['class_1_ratio']:.1%})")

        if 'confidence' in stats:
            print(f"\nConfidence:")
            print(f"  Mean: {stats['confidence']['mean']:.3f}")
            print(f"  Range: [{stats['confidence']['min']:.3f}, {stats['confidence']['max']:.3f}]")

        print(f"\nAlerts: {stats.get('alerts_count', 0)}")
        print("=" * 60)


class SystemMonitor:
    """Monitor system resources."""

    def __init__(self):
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            self.psutil = None
            logger.warning("psutil not installed, system monitoring disabled")

    def get_stats(self) -> Dict:
        """Get system resource statistics."""
        if self.psutil is None:
            return {}

        stats = {
            "cpu_percent": self.psutil.cpu_percent(interval=0.1),
            "memory": {
                "total_gb": self.psutil.virtual_memory().total / (1024**3),
                "used_gb": self.psutil.virtual_memory().used / (1024**3),
                "percent": self.psutil.virtual_memory().percent
            }
        }

        # GPU stats (if available)
        try:
            import torch
            if torch.cuda.is_available():
                stats["gpu"] = {
                    "name": torch.cuda.get_device_name(0),
                    "memory_allocated_gb": torch.cuda.memory_allocated(0) / (1024**3),
                    "memory_cached_gb": torch.cuda.memory_reserved(0) / (1024**3)
                }
        except Exception:
            pass

        return stats


def run_monitored_inference(
    model_path: str,
    model_type: str,
    source: str,
    monitor: PerformanceMonitor
):
    """Run inference with monitoring."""
    import torch
    import cv2

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            dropout=0
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Initialize pose extractor
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5
    )

    # Open video source
    if source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)

    # Pose buffer
    pose_buffer = []
    window_size = 16

    logger.info("Starting monitored inference... Press 'q' to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()

            # Extract pose
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                keypoints = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark])
                pose_buffer.append(keypoints.flatten())

                if len(pose_buffer) >= window_size:
                    # Prepare input
                    sequence = np.array(pose_buffer[-window_size:])
                    input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)

                    # Inference
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        pred = probs.argmax(dim=1).item()
                        confidence = probs[0, pred].item()

                    latency_ms = (time.time() - start_time) * 1000

                    # Log to monitor
                    monitor.log_inference(latency_ms, pred, confidence)

                    # Display
                    label = "Aggressive" if pred == 1 else "Normal"
                    color = (0, 0, 255) if pred == 1 else (0, 255, 0)
                    cv2.putText(frame, f"{label}: {confidence:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(frame, f"Latency: {latency_ms:.1f}ms", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Monitored Inference', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

        # Save final report
        monitor.print_summary()
        monitor.save_report()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Monitor fight detection inference')
    parser.add_argument('--model', type=str, default='outputs/lstm/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--type', type=str, default='lstm',
                        choices=['lstm', 'stgcn'],
                        help='Model type')
    parser.add_argument('--source', type=str, default='0',
                        help='Video source (0 for webcam)')
    parser.add_argument('--log_dir', type=str, default='logs/monitoring',
                        help='Directory for monitoring logs')
    parser.add_argument('--alert_latency', type=float, default=100,
                        help='Latency alert threshold (ms)')
    parser.add_argument('--alert_confidence', type=float, default=0.3,
                        help='Confidence alert threshold')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Initialize monitor
    monitor = PerformanceMonitor(
        log_dir=args.log_dir,
        alert_latency_ms=args.alert_latency,
        alert_confidence=args.alert_confidence
    )

    logger.info(f"Starting monitored inference")
    logger.info(f"Model: {args.model}")
    logger.info(f"Source: {args.source}")
    logger.info(f"Logs: {args.log_dir}")

    # Run inference with monitoring
    run_monitored_inference(
        args.model,
        args.type,
        args.source,
        monitor
    )


if __name__ == '__main__':
    main()
