# CCTV Aggressive Poses / Fight Detection System

A comprehensive end-to-end system for detecting fights and aggressive poses in CCTV footage using pose-based temporal models (LSTM, ST-GCN, Transformer) and object detection (YOLOv8).

## Features

- **Multiple Model Architectures**:
  - BiLSTM with temporal attention
  - ST-GCN (Spatial-Temporal Graph Convolutional Network)
  - Transformer-based temporal classifier
  - YOLOv8 for frame-level detection

- **MediaPipe Pose Extraction**: Fast and accurate 33-keypoint skeleton extraction

- **Temporal Modeling**: Sliding window sequences for action recognition

- **Real-time Inference**: Webcam and video file support with <100ms latency on GPU

- **Deployment Ready**: ONNX export, TFLite conversion, Docker support, REST API

## Project Structure

```
.
├── configs/                    # Model configuration files
│   ├── lstm_config.yaml
│   ├── stgcn_config.yaml
│   └── yolov8_config.yaml
├── deploy/                     # Deployment scripts
│   ├── api_server.py          # FastAPI REST server
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── export_to_onnx.py
│   └── export_to_tflite.py
├── models/                     # Model architectures
│   ├── lstm.py
│   ├── stgcn.py
│   └── transformer.py
├── notebooks/                  # Jupyter notebooks
│   └── 01_dataset_exploration.ipynb
├── utils/                      # Utility modules
│   ├── dataset.py
│   ├── metrics.py
│   └── visualization.py
├── inspect_dataset.py          # Dataset inspection tool
├── split_dataset.py            # Train/val/test split
├── yolo_to_coco.py            # YOLO to COCO conversion
├── extract_mediapipe_pose.py   # Pose extraction
├── build_sequences.py          # Temporal sequence builder
├── train_lstm.py               # LSTM training
├── train_stgcn.py              # ST-GCN training
├── train_yolov8.py             # YOLOv8 training
├── eval.py                     # Model evaluation
├── inference_realtime.py       # Real-time inference
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.x (for GPU support)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/CCTV-Aggressive-Poses-Fight-Detection-Dataset.git
cd CCTV-Aggressive-Poses-Fight-Detection-Dataset

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For CPU-only deployment
pip install -r requirements-cpu.txt
```

## Quick Start

### 1. Inspect Dataset

```bash
python inspect_dataset.py --data_root /path/to/Aggressive_Poses_Dataset --preview_count 10
```

### 2. Create Train/Val/Test Split

```bash
python split_dataset.py \
    --data_root /path/to/Aggressive_Poses_Dataset \
    --output_dir yolo_split \
    --train_ratio 0.70 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42
```

### 3. Extract Pose Keypoints

```bash
python extract_mediapipe_pose.py \
    --data_root yolo_split \
    --output_dir pose_npy \
    --splits train val test \
    --save_visualizations
```

### 4. Build Temporal Sequences

```bash
python build_sequences.py \
    --pose_dir pose_npy \
    --output_dir sequences \
    --T 16 \
    --stride 8 \
    --normalize \
    --synthetic  # Generate synthetic sequences from static images
```

### 5. Train Models

#### LSTM Baseline
```bash
python train_lstm.py \
    --data sequences \
    --output outputs/lstm \
    --model_type bilstm \
    --hidden_size 128 \
    --num_layers 2 \
    --epochs 60 \
    --batch_size 32 \
    --lr 1e-3
```

#### ST-GCN
```bash
python train_stgcn.py \
    --data sequences \
    --output outputs/stgcn \
    --model_type lightweight \
    --epochs 80 \
    --batch_size 16 \
    --lr 1e-3
```

#### YOLOv8 (Frame-level Detection)
```bash
python train_yolov8.py \
    --data yolo_split/data.yaml \
    --model yolov8n.pt \
    --epochs 50 \
    --batch 16 \
    --imgsz 640
```

### 6. Evaluate Models

```bash
# Evaluate pose model
python eval.py \
    --model outputs/lstm/best_model.pt \
    --data sequences \
    --type lstm \
    --split test \
    --output evaluation/lstm

# Evaluate YOLO
python eval.py \
    --model outputs/yolov8/train/weights/best.pt \
    --data yolo_split \
    --type yolo \
    --split test
```

### 7. Real-time Inference

```bash
# Webcam with LSTM
python inference_realtime.py \
    --source 0 \
    --model outputs/lstm/best_model.pt \
    --type lstm \
    --window_size 16 \
    --threshold 0.5

# Video file with ST-GCN
python inference_realtime.py \
    --source video.mp4 \
    --model outputs/stgcn/best_model.pt \
    --type stgcn \
    --output output_video.mp4

# YOLOv8 detection
python inference_realtime.py \
    --source 0 \
    --model outputs/yolov8/train/weights/best.pt \
    --type yolo \
    --conf 0.5
```

## Model Export

### ONNX Export
```bash
python deploy/export_to_onnx.py \
    --model outputs/lstm/best_model.pt \
    --type lstm \
    --output models/lstm.onnx \
    --simplify \
    --verify
```

### TFLite Export
```bash
python deploy/export_to_tflite.py \
    --model outputs/lstm/best_model.pt \
    --type lstm \
    --output models/lstm.tflite \
    --quantize
```

## Deployment

### Docker

```bash
# Build GPU image
docker build -f deploy/Dockerfile --target gpu -t fight-detection:gpu .

# Build CPU image
docker build -f deploy/Dockerfile --target cpu -t fight-detection:cpu .

# Run with docker-compose
docker-compose -f deploy/docker-compose.yml up fight-detection-gpu
```

### REST API

```bash
# Start API server
python deploy/api_server.py --host 0.0.0.0 --port 8000 --model outputs/lstm/best_model.pt

# Test endpoint
curl -X POST "http://localhost:8000/predict/sequence" \
    -H "Content-Type: application/json" \
    -d '{"sequence": [[...]], "threshold": 0.5}'
```

API documentation available at `http://localhost:8000/docs`

## Model Architectures

### BiLSTM with Attention
- Input: Pose sequences (T, 66) - 33 joints x 2 coordinates
- Architecture: BiLSTM layers -> Temporal attention -> FC classifier
- Parameters: ~500K
- Speed: ~5ms per sequence (GPU)

### ST-GCN (Lightweight)
- Input: Pose sequences (T, 33, 3) - joints with visibility
- Architecture: Graph convolutions on skeleton -> Temporal convolutions
- Parameters: ~200K (lightweight variant)
- Speed: ~10ms per sequence (GPU)

### YOLOv8
- Input: Images (640x640)
- Output: Bounding boxes with class labels
- Variants: n (nano), s (small), m (medium), l (large)
- Speed: 5-50ms per frame depending on variant

## Hyperparameter Guidelines

| Model | LR | Batch Size | Epochs | Hidden | Notes |
|-------|------|------------|--------|--------|-------|
| LSTM | 1e-3 | 32 | 60 | 128 | Use attention for best results |
| ST-GCN | 1e-3 | 16 | 80 | 64 | Lightweight for real-time |
| YOLOv8n | 0.01 | 16 | 50 | - | Best speed/accuracy balance |

## Expected Performance

| Model | Accuracy | F1 Score | Latency (GPU) |
|-------|----------|----------|---------------|
| BiLSTM | ~85% | ~0.82 | ~5ms |
| ST-GCN | ~87% | ~0.84 | ~10ms |
| YOLOv8n | mAP@0.5: ~0.75 | - | ~8ms |

*Note: Actual performance depends on dataset size and quality.*

## Dataset Format

### YOLO Format (labels/*.txt)
```
<class_id> <x_center> <y_center> <width> <height>
0 0.5 0.5 0.3 0.6
```

### data.yaml
```yaml
train: path/to/train/images
val: path/to/val/images
test: path/to/test/images
nc: 1
names: ['aggressive_pose']
```

## Ethics and Safety

### Bias Considerations
- Model trained on specific demographics/settings may not generalize
- CCTV angle, lighting, and resolution affect accuracy
- Consider cultural differences in body language interpretation

### Deployment Guidelines
1. **Threshold Tuning**: Adjust classification threshold to minimize false positives in sensitive deployments
2. **Human-in-the-Loop**: Require human verification before taking action on alerts
3. **Privacy Compliance**: Ensure compliance with local privacy regulations (GDPR, etc.)
4. **Data Retention**: Implement appropriate data retention policies
5. **Transparency**: Inform subjects about surveillance and automated analysis

### False Positive Mitigation
- Use temporal smoothing to reduce sporadic false alerts
- Set minimum event duration thresholds
- Implement confidence-based tiered alerting

## Troubleshooting

### Common Issues

**MediaPipe installation fails**
```bash
pip install mediapipe --no-cache-dir
```

**CUDA out of memory**
- Reduce batch size
- Use lightweight model variant
- Use mixed precision training: `--fp16`

**No poses detected**
- Check image quality and resolution
- Ensure subjects are visible (not occluded)
- Lower `min_detection_confidence`

**Poor model performance**
- Increase training data with augmentation
- Try different model architectures
- Tune hyperparameters using validation set

## Citation

If you use this code in your research, please cite:

```bibtex
@software{fight_detection_2024,
  title={CCTV Aggressive Poses / Fight Detection System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/CCTV-Aggressive-Poses-Fight-Detection-Dataset}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for pose estimation
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [ST-GCN](https://github.com/yysijie/st-gcn) for skeleton-based action recognition
