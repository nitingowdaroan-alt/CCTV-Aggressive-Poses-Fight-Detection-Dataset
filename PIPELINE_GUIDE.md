# Complete Pipeline Guide: Data Preparation to Monitoring

This guide provides step-by-step instructions to run the entire fight detection pipeline from raw data to production monitoring.

## Table of Contents
1. [Environment Setup](#1-environment-setup)
2. [Data Preparation](#2-data-preparation)
3. [Pose Extraction](#3-pose-extraction)
4. [Sequence Building](#4-sequence-building)
5. [Model Training](#5-model-training)
6. [Model Evaluation](#6-model-evaluation)
7. [Model Export](#7-model-export)
8. [Inference & Deployment](#8-inference--deployment)
9. [Monitoring & Maintenance](#9-monitoring--maintenance)

---

## 1. Environment Setup

### 1.1 Create Virtual Environment

```bash
# Create and activate virtual environment
python -m venv fight_detection_env

# Linux/Mac
source fight_detection_env/bin/activate

# Windows
fight_detection_env\Scripts\activate
```

### 1.2 Install Dependencies

```bash
# Install all dependencies (GPU support)
pip install -r requirements.txt

# Or for CPU-only deployment
pip install -r requirements-cpu.txt

# Verify installations
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import mediapipe; print(f'MediaPipe: {mediapipe.__version__}')"
```

### 1.3 Verify GPU Setup (Optional but Recommended)

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## 2. Data Preparation

### 2.1 Organize Your Dataset

Ensure your dataset follows this structure:
```
Aggressive_Poses_Dataset/
├── images/
│   ├── aggressive_pose_1.jpg
│   ├── aggressive_pose_2.jpg
│   └── ...
├── labels/
│   ├── aggressive_pose_1.txt
│   ├── aggressive_pose_2.txt
│   └── ...
└── data.yaml
```

### 2.2 Inspect Dataset

```bash
# Basic inspection
python inspect_dataset.py \
    --data_root "C:\Users\Nitin N\Aggressive_Poses_Dataset" \
    --preview_count 10 \
    --output_dir inspection_output \
    --save_previews

# For headless environments (no display)
python inspect_dataset.py \
    --data_root /path/to/dataset \
    --preview_count 10 \
    --no_display \
    --save_previews
```

**Expected Output:**
- `inspection_output/inspection_report.txt` - Statistics summary
- `inspection_output/dataset_statistics.png` - Distribution plots
- `inspection_output/previews/` - Annotated sample images

### 2.3 Create Train/Val/Test Split

```bash
# Create split with default ratios (70/15/15)
python split_dataset.py \
    --data_root "C:\Users\Nitin N\Aggressive_Poses_Dataset" \
    --output_dir yolo_split \
    --train_ratio 0.70 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42 \
    --copy  # Use --copy for Windows, symlinks may require admin

# Verify split
ls yolo_split/images/train | wc -l  # Linux
dir yolo_split\images\train | find /c /v ""  # Windows
```

**Expected Output:**
```
yolo_split/
├── images/
│   ├── train/   # ~72 images (70%)
│   ├── val/     # ~15 images (15%)
│   └── test/    # ~16 images (15%)
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── data.yaml    # Updated paths
├── manifest.csv # File listing
└── split_info.yaml
```

### 2.4 Convert to COCO Format (Optional)

```bash
# Convert for frameworks that require COCO format
python yolo_to_coco.py \
    --data_root yolo_split \
    --output_dir coco_format \
    --splits train val test
```

---

## 3. Pose Extraction

### 3.1 Extract MediaPipe Poses

```bash
# Extract poses from split dataset
python extract_mediapipe_pose.py \
    --data_root yolo_split \
    --output_dir pose_npy \
    --splits train val test \
    --model_complexity 1 \
    --min_detection_confidence 0.5 \
    --save_visualizations

# For faster extraction (lower quality)
python extract_mediapipe_pose.py \
    --data_root yolo_split \
    --output_dir pose_npy \
    --splits train val test \
    --model_complexity 0
```

**Expected Output:**
```
pose_npy/
├── train/
│   ├── aggressive_pose_1.npy  # Shape: (33, 3) - x, y, visibility
│   ├── aggressive_pose_2.npy
│   └── visualizations/        # If --save_visualizations
├── val/
├── test/
└── pose_extraction_summary.csv
```

### 3.2 Verify Pose Extraction

```bash
# Check extraction results
python -c "
import numpy as np
from pathlib import Path

pose_dir = Path('pose_npy/train')
poses = list(pose_dir.glob('*.npy'))
print(f'Total poses extracted: {len(poses)}')

if poses:
    sample = np.load(poses[0])
    print(f'Pose shape: {sample.shape}')
    print(f'Sample keypoints (first 5):')
    print(sample[:5])
"
```

---

## 4. Sequence Building

### 4.1 Build Temporal Sequences

For datasets with **sequential frames** (video frames):
```bash
python build_sequences.py \
    --pose_dir pose_npy \
    --output_dir sequences \
    --labels_dir yolo_split/labels \
    --T 16 \
    --stride 8 \
    --normalize \
    --add_velocity \
    --splits train val test
```

For **static images** (like your dataset), generate synthetic sequences:
```bash
python build_sequences.py \
    --pose_dir pose_npy \
    --output_dir sequences \
    --labels_dir yolo_split/labels \
    --T 16 \
    --stride 8 \
    --normalize \
    --synthetic \
    --augment \
    --splits train val test \
    --seed 42
```

**Expected Output:**
```
sequences/
├── train/
│   ├── sequences.npy  # Shape: (N, 16, 66) or (N, 16, 33, 3)
│   ├── labels.npy     # Shape: (N,)
│   └── labels.csv
├── val/
├── test/
├── all_sequences.csv
└── config.yaml
```

### 4.2 Verify Sequences

```bash
python -c "
import numpy as np

train_seq = np.load('sequences/train/sequences.npy')
train_labels = np.load('sequences/train/labels.npy')

print(f'Training sequences shape: {train_seq.shape}')
print(f'Training labels shape: {train_labels.shape}')
print(f'Label distribution: {dict(zip(*np.unique(train_labels, return_counts=True)))}')
"
```

---

## 5. Model Training

### 5.1 Train LSTM Model (Recommended First)

```bash
# Basic training
python train_lstm.py \
    --data sequences \
    --output outputs/lstm \
    --model_type bilstm \
    --hidden_size 128 \
    --num_layers 2 \
    --epochs 60 \
    --batch_size 32 \
    --lr 1e-3 \
    --dropout 0.3 \
    --early_stop 15

# With class weights for imbalanced data
python train_lstm.py \
    --data sequences \
    --output outputs/lstm_weighted \
    --model_type bilstm \
    --epochs 60 \
    --batch_size 32 \
    --use_class_weights \
    --augment

# With Weights & Biases logging
python train_lstm.py \
    --data sequences \
    --output outputs/lstm_wandb \
    --epochs 60 \
    --wandb \
    --wandb_project fight-detection
```

### 5.2 Train ST-GCN Model

```bash
# Lightweight ST-GCN (faster, good for real-time)
python train_stgcn.py \
    --data sequences \
    --output outputs/stgcn_light \
    --model_type lightweight \
    --epochs 80 \
    --batch_size 16 \
    --lr 1e-3 \
    --scheduler step \
    --augment

# Full ST-GCN (better accuracy)
python train_stgcn.py \
    --data sequences \
    --output outputs/stgcn_full \
    --model_type stgcn \
    --epochs 100 \
    --batch_size 8 \
    --lr 5e-4
```

### 5.3 Train YOLOv8 (Frame-level Detection)

```bash
# Nano model (fastest)
python train_yolov8.py \
    --data yolo_split/data.yaml \
    --model yolov8n.pt \
    --epochs 50 \
    --batch 16 \
    --imgsz 640

# Medium model (better accuracy)
python train_yolov8.py \
    --data yolo_split/data.yaml \
    --model yolov8m.pt \
    --epochs 100 \
    --batch 8 \
    --imgsz 640 \
    --patience 30

# Using Ultralytics CLI directly
yolo detect train \
    model=yolov8n.pt \
    data=yolo_split/data.yaml \
    epochs=50 \
    imgsz=640 \
    batch=16 \
    project=outputs/yolov8 \
    name=train
```

### 5.4 Monitor Training

```bash
# View TensorBoard logs (if enabled)
tensorboard --logdir outputs/

# Check training progress
tail -f outputs/lstm/training.log

# View latest metrics
cat outputs/lstm/history.json | python -m json.tool
```

---

## 6. Model Evaluation

### 6.1 Evaluate Pose Models

```bash
# Evaluate LSTM on test set
python eval.py \
    --model outputs/lstm/best_model.pt \
    --data sequences \
    --type lstm \
    --split test \
    --output evaluation/lstm \
    --plot

# Evaluate ST-GCN
python eval.py \
    --model outputs/stgcn_light/best_model.pt \
    --data sequences \
    --type stgcn \
    --split test \
    --output evaluation/stgcn
```

### 6.2 Evaluate YOLOv8

```bash
# Evaluate on test set
python eval.py \
    --model outputs/yolov8/train/weights/best.pt \
    --data yolo_split \
    --type yolo \
    --split test \
    --output evaluation/yolov8

# Using Ultralytics CLI
yolo detect val \
    model=outputs/yolov8/train/weights/best.pt \
    data=yolo_split/data.yaml \
    split=test
```

### 6.3 Compare Models

```bash
# Generate comparison report
python -c "
import json
from pathlib import Path

models = ['lstm', 'stgcn', 'yolov8']
print('Model Comparison:')
print('-' * 50)

for model in models:
    report_path = Path(f'evaluation/{model}/evaluation_report.json')
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        if 'classification' in report:
            print(f'{model.upper()}:')
            print(f'  Accuracy: {report[\"classification\"][\"accuracy\"]:.4f}')
            print(f'  F1 Score: {report[\"classification\"][\"f1\"]:.4f}')
        elif 'detection' in report:
            print(f'{model.upper()}:')
            print(f'  mAP@0.5: {report[\"detection\"][\"mAP@0.5\"]:.4f}')
"
```

**Expected Evaluation Output:**
```
evaluation/lstm/
├── evaluation_report.json
├── classification_report.txt
├── confusion_matrix.png
├── roc_curve.png
└── pr_curve.png
```

---

## 7. Model Export

### 7.1 Export to ONNX

```bash
# Export LSTM
python deploy/export_to_onnx.py \
    --model outputs/lstm/best_model.pt \
    --type lstm \
    --output models/lstm.onnx \
    --seq_len 16 \
    --simplify \
    --verify

# Export ST-GCN
python deploy/export_to_onnx.py \
    --model outputs/stgcn_light/best_model.pt \
    --type stgcn \
    --output models/stgcn.onnx \
    --seq_len 16 \
    --verify

# Export YOLOv8 (built-in)
yolo export \
    model=outputs/yolov8/train/weights/best.pt \
    format=onnx \
    simplify=True
```

### 7.2 Export to TFLite (Edge Devices)

```bash
# Export with quantization for mobile
python deploy/export_to_tflite.py \
    --model outputs/lstm/best_model.pt \
    --type lstm \
    --output models/lstm.tflite \
    --quantize

# Export with FP16 (better accuracy)
python deploy/export_to_tflite.py \
    --model outputs/lstm/best_model.pt \
    --type lstm \
    --output models/lstm_fp16.tflite \
    --fp16
```

### 7.3 Verify Exported Models

```bash
# Test ONNX model
python -c "
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('models/lstm.onnx')
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f'Input: {input_name}, Shape: {input_shape}')

# Test inference
dummy_input = np.random.randn(1, 16, 66).astype(np.float32)
output = session.run(None, {input_name: dummy_input})
print(f'Output shape: {output[0].shape}')
print(f'Predictions: {output[0]}')
"
```

---

## 8. Inference & Deployment

### 8.1 Real-time Webcam Inference

```bash
# Using LSTM model
python inference_realtime.py \
    --source 0 \
    --model outputs/lstm/best_model.pt \
    --type lstm \
    --window_size 16 \
    --stride 8 \
    --threshold 0.5 \
    --smooth_window 5

# Using ST-GCN model
python inference_realtime.py \
    --source 0 \
    --model outputs/stgcn_light/best_model.pt \
    --type stgcn \
    --threshold 0.6

# Using YOLOv8
python inference_realtime.py \
    --source 0 \
    --model outputs/yolov8/train/weights/best.pt \
    --type yolo \
    --conf 0.5
```

### 8.2 Video File Inference

```bash
# Process video and save output
python inference_realtime.py \
    --source input_video.mp4 \
    --model outputs/lstm/best_model.pt \
    --type lstm \
    --output output_annotated.mp4 \
    --threshold 0.5

# Process without display (server environment)
python inference_realtime.py \
    --source input_video.mp4 \
    --model outputs/lstm/best_model.pt \
    --type lstm \
    --output output.mp4 \
    --no-show
```

### 8.3 Start REST API Server

```bash
# Start FastAPI server
python deploy/api_server.py \
    --host 0.0.0.0 \
    --port 8000 \
    --model outputs/lstm/best_model.pt \
    --type lstm \
    --device cpu

# Test API endpoints
# Health check
curl http://localhost:8000/health

# Predict sequence
curl -X POST "http://localhost:8000/predict/sequence" \
    -H "Content-Type: application/json" \
    -d '{"sequence": [[0.5, 0.5, 0.4, 0.6, ...]], "threshold": 0.5}'

# API documentation
open http://localhost:8000/docs
```

### 8.4 Docker Deployment

```bash
# Build images
docker build -f deploy/Dockerfile --target gpu -t fight-detection:gpu .
docker build -f deploy/Dockerfile --target cpu -t fight-detection:cpu .

# Run GPU container
docker run --gpus all -p 8000:8000 \
    -v $(pwd)/models:/app/models:ro \
    fight-detection:gpu \
    --model models/lstm.onnx --type lstm

# Run CPU container
docker run -p 8000:8000 \
    -v $(pwd)/models:/app/models:ro \
    fight-detection:cpu \
    --model models/lstm.onnx --type lstm

# Using docker-compose
docker-compose -f deploy/docker-compose.yml up fight-detection-api
```

---

## 9. Monitoring & Maintenance

### 9.1 Performance Monitoring

Create a monitoring script (`scripts/monitor.py`):

```python
#!/usr/bin/env python3
"""Monitor inference performance and model drift."""

import time
import json
import logging
from datetime import datetime
from pathlib import Path
from collections import deque

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self, log_dir="logs/monitoring", window_size=1000):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)

        self.alert_threshold_latency = 100  # ms
        self.alert_threshold_confidence = 0.3

    def log_inference(self, latency_ms, prediction, confidence):
        """Log single inference result."""
        self.latencies.append(latency_ms)
        self.predictions.append(prediction)
        self.confidences.append(confidence)

        # Check alerts
        self._check_alerts()

    def _check_alerts(self):
        """Check for performance issues."""
        if len(self.latencies) < 10:
            return

        avg_latency = np.mean(list(self.latencies)[-10:])
        avg_confidence = np.mean(list(self.confidences)[-10:])

        if avg_latency > self.alert_threshold_latency:
            logger.warning(f"High latency alert: {avg_latency:.1f}ms")

        if avg_confidence < self.alert_threshold_confidence:
            logger.warning(f"Low confidence alert: {avg_confidence:.2f}")

    def get_stats(self):
        """Get current statistics."""
        if not self.latencies:
            return {}

        return {
            "timestamp": datetime.now().isoformat(),
            "total_inferences": len(self.latencies),
            "latency": {
                "mean": np.mean(self.latencies),
                "p50": np.percentile(self.latencies, 50),
                "p95": np.percentile(self.latencies, 95),
                "p99": np.percentile(self.latencies, 99)
            },
            "predictions": {
                "class_0": sum(1 for p in self.predictions if p == 0),
                "class_1": sum(1 for p in self.predictions if p == 1)
            },
            "confidence": {
                "mean": np.mean(self.confidences),
                "min": min(self.confidences),
                "max": max(self.confidences)
            }
        }

    def save_report(self):
        """Save monitoring report."""
        stats = self.get_stats()
        report_path = self.log_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Report saved to {report_path}")
        return stats

# Usage example
if __name__ == "__main__":
    monitor = PerformanceMonitor()

    # Simulate monitoring
    for i in range(100):
        latency = np.random.exponential(20)  # ~20ms average
        prediction = np.random.choice([0, 1], p=[0.7, 0.3])
        confidence = np.random.beta(5, 2)

        monitor.log_inference(latency, prediction, confidence)
        time.sleep(0.01)

    print(json.dumps(monitor.get_stats(), indent=2))
    monitor.save_report()
```

### 9.2 Model Drift Detection

```python
#!/usr/bin/env python3
"""Detect model drift by comparing prediction distributions."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats
import numpy as np

def detect_drift(baseline_predictions, current_predictions, alpha=0.05):
    """
    Detect distribution drift using Chi-squared test.

    Returns:
        (is_drift, p_value, message)
    """
    # Count predictions per class
    baseline_counts = np.bincount(baseline_predictions, minlength=2)
    current_counts = np.bincount(current_predictions, minlength=2)

    # Chi-squared test
    chi2, p_value = stats.chisquare(current_counts, baseline_counts)

    is_drift = p_value < alpha

    if is_drift:
        message = f"DRIFT DETECTED: p-value={p_value:.4f} < {alpha}"
    else:
        message = f"No drift: p-value={p_value:.4f} >= {alpha}"

    return is_drift, p_value, message

# Example usage
baseline = np.random.choice([0, 1], size=1000, p=[0.7, 0.3])
current = np.random.choice([0, 1], size=1000, p=[0.6, 0.4])  # Drift!

is_drift, p_value, message = detect_drift(baseline, current)
print(message)
```

### 9.3 Logging Configuration

Create `configs/logging_config.yaml`:

```yaml
# Logging configuration
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/fight_detection.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  inference:
    level: INFO
    handlers: [console, file]
    propagate: false

  monitoring:
    level: DEBUG
    handlers: [file]
    propagate: false

root:
  level: INFO
  handlers: [console]
```

### 9.4 Automated Retraining Pipeline

```bash
#!/bin/bash
# scripts/retrain_pipeline.sh

set -e

echo "=== Fight Detection Retraining Pipeline ==="
echo "Started at: $(date)"

# Configuration
DATA_DIR="new_data"
OUTPUT_DIR="outputs/retrain_$(date +%Y%m%d)"
BASELINE_MODEL="outputs/lstm/best_model.pt"

# Step 1: Prepare new data
echo "[1/5] Preparing new data..."
python split_dataset.py \
    --data_root $DATA_DIR \
    --output_dir ${OUTPUT_DIR}/split \
    --seed 42

# Step 2: Extract poses
echo "[2/5] Extracting poses..."
python extract_mediapipe_pose.py \
    --data_root ${OUTPUT_DIR}/split \
    --output_dir ${OUTPUT_DIR}/poses \
    --splits train val test

# Step 3: Build sequences
echo "[3/5] Building sequences..."
python build_sequences.py \
    --pose_dir ${OUTPUT_DIR}/poses \
    --output_dir ${OUTPUT_DIR}/sequences \
    --T 16 --stride 8 \
    --synthetic --augment

# Step 4: Train new model
echo "[4/5] Training model..."
python train_lstm.py \
    --data ${OUTPUT_DIR}/sequences \
    --output ${OUTPUT_DIR}/model \
    --epochs 60 \
    --early_stop 15

# Step 5: Evaluate and compare
echo "[5/5] Evaluating model..."
python eval.py \
    --model ${OUTPUT_DIR}/model/best_model.pt \
    --data ${OUTPUT_DIR}/sequences \
    --type lstm \
    --split test \
    --output ${OUTPUT_DIR}/evaluation

# Compare with baseline
echo "Comparing with baseline model..."
python eval.py \
    --model $BASELINE_MODEL \
    --data ${OUTPUT_DIR}/sequences \
    --type lstm \
    --split test \
    --output ${OUTPUT_DIR}/baseline_eval

echo "=== Pipeline Complete ==="
echo "Results saved to: $OUTPUT_DIR"
echo "Finished at: $(date)"
```

### 9.5 Health Check Script

```bash
#!/bin/bash
# scripts/health_check.sh

echo "=== System Health Check ==="

# Check GPU
echo -n "GPU Status: "
if nvidia-smi &> /dev/null; then
    echo "OK"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
else
    echo "No GPU detected"
fi

# Check disk space
echo -e "\nDisk Space:"
df -h | grep -E "^/dev|Filesystem"

# Check model files
echo -e "\nModel Files:"
for model in outputs/*/best_model.pt models/*.onnx; do
    if [ -f "$model" ]; then
        echo "  ✓ $model ($(du -h $model | cut -f1))"
    fi
done

# Check API health (if running)
echo -e "\nAPI Health:"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "  ✓ API is running"
    curl -s http://localhost:8000/health | python -m json.tool
else
    echo "  ✗ API is not running"
fi

# Check recent logs
echo -e "\nRecent Errors (last 10):"
if [ -f "logs/fight_detection.log" ]; then
    grep -i "error\|warning" logs/fight_detection.log | tail -10
else
    echo "  No log file found"
fi

echo -e "\n=== Health Check Complete ==="
```

---

## Quick Reference Commands

```bash
# === DATA PREPARATION ===
python inspect_dataset.py --data_root /path/to/data --preview_count 5
python split_dataset.py --data_root /path/to/data --output_dir yolo_split

# === POSE EXTRACTION ===
python extract_mediapipe_pose.py --data_root yolo_split --output_dir pose_npy --splits train val test

# === SEQUENCE BUILDING ===
python build_sequences.py --pose_dir pose_npy --output_dir sequences --T 16 --synthetic

# === TRAINING ===
python train_lstm.py --data sequences --output outputs/lstm --epochs 60
python train_stgcn.py --data sequences --output outputs/stgcn --epochs 80
python train_yolov8.py --data yolo_split/data.yaml --model yolov8n.pt --epochs 50

# === EVALUATION ===
python eval.py --model outputs/lstm/best_model.pt --data sequences --type lstm --split test

# === EXPORT ===
python deploy/export_to_onnx.py --model outputs/lstm/best_model.pt --type lstm --output models/lstm.onnx

# === INFERENCE ===
python inference_realtime.py --source 0 --model outputs/lstm/best_model.pt --type lstm

# === DEPLOYMENT ===
python deploy/api_server.py --host 0.0.0.0 --port 8000 --model outputs/lstm/best_model.pt
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| MediaPipe not detecting poses | Lower `--min_detection_confidence` to 0.3 |
| CUDA out of memory | Reduce batch size or use CPU |
| Poor model accuracy | Enable augmentation, try different model |
| High inference latency | Use lightweight model, export to ONNX |
| API 503 error | Check if model file exists at specified path |

For more details, see the main [README.md](README.md).
