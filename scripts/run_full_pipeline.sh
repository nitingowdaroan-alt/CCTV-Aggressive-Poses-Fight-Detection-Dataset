#!/bin/bash
# =============================================================================
# Full Pipeline Runner Script
# =============================================================================
# This script runs the complete fight detection pipeline from data to inference
#
# Usage:
#   ./scripts/run_full_pipeline.sh /path/to/Aggressive_Poses_Dataset
#
# =============================================================================

set -e  # Exit on error

# Configuration
DATA_ROOT="${1:-/path/to/Aggressive_Poses_Dataset}"
OUTPUT_BASE="outputs"
EPOCHS_LSTM=60
EPOCHS_STGCN=80
BATCH_SIZE=32
SEQUENCE_LENGTH=16
STRIDE=8

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==============================================================================${NC}"
echo -e "${GREEN}     FIGHT DETECTION PIPELINE - Full Execution${NC}"
echo -e "${GREEN}==============================================================================${NC}"
echo ""
echo "Configuration:"
echo "  Data Root: $DATA_ROOT"
echo "  Output Base: $OUTPUT_BASE"
echo "  LSTM Epochs: $EPOCHS_LSTM"
echo "  ST-GCN Epochs: $EPOCHS_STGCN"
echo "  Sequence Length: $SEQUENCE_LENGTH"
echo ""

# Check data root exists
if [ ! -d "$DATA_ROOT" ]; then
    echo -e "${RED}ERROR: Data root not found: $DATA_ROOT${NC}"
    echo "Usage: $0 /path/to/Aggressive_Poses_Dataset"
    exit 1
fi

# Create output directories
mkdir -p $OUTPUT_BASE
mkdir -p logs

# =============================================================================
# STEP 1: Dataset Inspection
# =============================================================================
echo -e "\n${YELLOW}[1/8] Inspecting Dataset...${NC}"
python inspect_dataset.py \
    --data_root "$DATA_ROOT" \
    --preview_count 5 \
    --output_dir inspection_output \
    --save_previews \
    --no_display 2>&1 | tee logs/01_inspect.log

echo -e "${GREEN}✓ Dataset inspection complete${NC}"

# =============================================================================
# STEP 2: Train/Val/Test Split
# =============================================================================
echo -e "\n${YELLOW}[2/8] Creating Train/Val/Test Split...${NC}"
python split_dataset.py \
    --data_root "$DATA_ROOT" \
    --output_dir yolo_split \
    --train_ratio 0.70 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42 \
    --copy 2>&1 | tee logs/02_split.log

echo -e "${GREEN}✓ Data split complete${NC}"

# =============================================================================
# STEP 3: Pose Extraction
# =============================================================================
echo -e "\n${YELLOW}[3/8] Extracting Pose Keypoints...${NC}"
python extract_mediapipe_pose.py \
    --data_root yolo_split \
    --output_dir pose_npy \
    --splits train val test \
    --model_complexity 1 \
    --min_detection_confidence 0.5 2>&1 | tee logs/03_pose.log

echo -e "${GREEN}✓ Pose extraction complete${NC}"

# =============================================================================
# STEP 4: Build Temporal Sequences
# =============================================================================
echo -e "\n${YELLOW}[4/8] Building Temporal Sequences...${NC}"
python build_sequences.py \
    --pose_dir pose_npy \
    --output_dir sequences \
    --labels_dir yolo_split/labels \
    --T $SEQUENCE_LENGTH \
    --stride $STRIDE \
    --normalize \
    --synthetic \
    --augment \
    --splits train val test \
    --seed 42 2>&1 | tee logs/04_sequences.log

echo -e "${GREEN}✓ Sequence building complete${NC}"

# =============================================================================
# STEP 5: Train LSTM Model
# =============================================================================
echo -e "\n${YELLOW}[5/8] Training LSTM Model...${NC}"
python train_lstm.py \
    --data sequences \
    --output $OUTPUT_BASE/lstm \
    --model_type bilstm \
    --hidden_size 128 \
    --num_layers 2 \
    --epochs $EPOCHS_LSTM \
    --batch_size $BATCH_SIZE \
    --lr 1e-3 \
    --dropout 0.3 \
    --early_stop 15 \
    --augment 2>&1 | tee logs/05_train_lstm.log

echo -e "${GREEN}✓ LSTM training complete${NC}"

# =============================================================================
# STEP 6: Train ST-GCN Model
# =============================================================================
echo -e "\n${YELLOW}[6/8] Training ST-GCN Model...${NC}"
python train_stgcn.py \
    --data sequences \
    --output $OUTPUT_BASE/stgcn \
    --model_type lightweight \
    --epochs $EPOCHS_STGCN \
    --batch_size 16 \
    --lr 1e-3 \
    --scheduler step \
    --early_stop 20 \
    --augment 2>&1 | tee logs/06_train_stgcn.log

echo -e "${GREEN}✓ ST-GCN training complete${NC}"

# =============================================================================
# STEP 7: Evaluate Models
# =============================================================================
echo -e "\n${YELLOW}[7/8] Evaluating Models...${NC}"

# Evaluate LSTM
python eval.py \
    --model $OUTPUT_BASE/lstm/best_model.pt \
    --data sequences \
    --type lstm \
    --split test \
    --output evaluation/lstm \
    --plot 2>&1 | tee logs/07_eval_lstm.log

# Evaluate ST-GCN
python eval.py \
    --model $OUTPUT_BASE/stgcn/best_model.pt \
    --data sequences \
    --type stgcn \
    --split test \
    --output evaluation/stgcn \
    --plot 2>&1 | tee logs/07_eval_stgcn.log

echo -e "${GREEN}✓ Model evaluation complete${NC}"

# =============================================================================
# STEP 8: Export Models
# =============================================================================
echo -e "\n${YELLOW}[8/8] Exporting Models to ONNX...${NC}"
mkdir -p models

# Export LSTM
python deploy/export_to_onnx.py \
    --model $OUTPUT_BASE/lstm/best_model.pt \
    --type lstm \
    --output models/lstm.onnx \
    --seq_len $SEQUENCE_LENGTH \
    --verify 2>&1 | tee logs/08_export.log

# Export ST-GCN
python deploy/export_to_onnx.py \
    --model $OUTPUT_BASE/stgcn/best_model.pt \
    --type stgcn \
    --output models/stgcn.onnx \
    --seq_len $SEQUENCE_LENGTH \
    --verify 2>&1 | tee -a logs/08_export.log

echo -e "${GREEN}✓ Model export complete${NC}"

# =============================================================================
# SUMMARY
# =============================================================================
echo -e "\n${GREEN}==============================================================================${NC}"
echo -e "${GREEN}     PIPELINE COMPLETE!${NC}"
echo -e "${GREEN}==============================================================================${NC}"
echo ""
echo "Output Files:"
echo "  - Split dataset: yolo_split/"
echo "  - Pose keypoints: pose_npy/"
echo "  - Sequences: sequences/"
echo "  - LSTM model: $OUTPUT_BASE/lstm/best_model.pt"
echo "  - ST-GCN model: $OUTPUT_BASE/stgcn/best_model.pt"
echo "  - ONNX models: models/*.onnx"
echo "  - Evaluations: evaluation/"
echo "  - Logs: logs/"
echo ""
echo "To run real-time inference:"
echo "  python inference_realtime.py --source 0 --model $OUTPUT_BASE/lstm/best_model.pt --type lstm"
echo ""
echo "To start API server:"
echo "  python deploy/api_server.py --model $OUTPUT_BASE/lstm/best_model.pt"
echo ""
