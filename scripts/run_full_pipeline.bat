@echo off
REM =============================================================================
REM Full Pipeline Runner Script for Windows
REM =============================================================================
REM Usage:
REM   run_full_pipeline.bat "C:\Users\Nitin N\Aggressive_Poses_Dataset"
REM =============================================================================

setlocal EnableDelayedExpansion

REM Configuration
set "DATA_ROOT=%~1"
if "%DATA_ROOT%"=="" set "DATA_ROOT=C:\Users\Nitin N\Aggressive_Poses_Dataset"
set "OUTPUT_BASE=outputs"
set "EPOCHS_LSTM=60"
set "EPOCHS_STGCN=80"
set "BATCH_SIZE=32"
set "SEQUENCE_LENGTH=16"
set "STRIDE=8"

echo ==============================================================================
echo      FIGHT DETECTION PIPELINE - Full Execution (Windows)
echo ==============================================================================
echo.
echo Configuration:
echo   Data Root: %DATA_ROOT%
echo   Output Base: %OUTPUT_BASE%
echo   LSTM Epochs: %EPOCHS_LSTM%
echo   Sequence Length: %SEQUENCE_LENGTH%
echo.

REM Check data root exists
if not exist "%DATA_ROOT%" (
    echo ERROR: Data root not found: %DATA_ROOT%
    echo Usage: %0 "C:\path\to\Aggressive_Poses_Dataset"
    exit /b 1
)

REM Create output directories
if not exist "%OUTPUT_BASE%" mkdir "%OUTPUT_BASE%"
if not exist "logs" mkdir "logs"

REM =============================================================================
REM STEP 1: Dataset Inspection
REM =============================================================================
echo.
echo [1/8] Inspecting Dataset...
python inspect_dataset.py ^
    --data_root "%DATA_ROOT%" ^
    --preview_count 5 ^
    --output_dir inspection_output ^
    --save_previews ^
    --no_display > logs\01_inspect.log 2>&1

if errorlevel 1 (
    echo ERROR: Dataset inspection failed
    exit /b 1
)
echo Done!

REM =============================================================================
REM STEP 2: Train/Val/Test Split
REM =============================================================================
echo.
echo [2/8] Creating Train/Val/Test Split...
python split_dataset.py ^
    --data_root "%DATA_ROOT%" ^
    --output_dir yolo_split ^
    --train_ratio 0.70 ^
    --val_ratio 0.15 ^
    --test_ratio 0.15 ^
    --seed 42 ^
    --copy > logs\02_split.log 2>&1

if errorlevel 1 (
    echo ERROR: Data split failed
    exit /b 1
)
echo Done!

REM =============================================================================
REM STEP 3: Pose Extraction
REM =============================================================================
echo.
echo [3/8] Extracting Pose Keypoints...
python extract_mediapipe_pose.py ^
    --data_root yolo_split ^
    --output_dir pose_npy ^
    --splits train val test ^
    --model_complexity 1 ^
    --min_detection_confidence 0.5 > logs\03_pose.log 2>&1

if errorlevel 1 (
    echo ERROR: Pose extraction failed
    exit /b 1
)
echo Done!

REM =============================================================================
REM STEP 4: Build Temporal Sequences
REM =============================================================================
echo.
echo [4/8] Building Temporal Sequences...
python build_sequences.py ^
    --pose_dir pose_npy ^
    --output_dir sequences ^
    --labels_dir yolo_split\labels ^
    --T %SEQUENCE_LENGTH% ^
    --stride %STRIDE% ^
    --normalize ^
    --synthetic ^
    --augment ^
    --splits train val test ^
    --seed 42 > logs\04_sequences.log 2>&1

if errorlevel 1 (
    echo ERROR: Sequence building failed
    exit /b 1
)
echo Done!

REM =============================================================================
REM STEP 5: Train LSTM Model
REM =============================================================================
echo.
echo [5/8] Training LSTM Model...
python train_lstm.py ^
    --data sequences ^
    --output %OUTPUT_BASE%\lstm ^
    --model_type bilstm ^
    --hidden_size 128 ^
    --num_layers 2 ^
    --epochs %EPOCHS_LSTM% ^
    --batch_size %BATCH_SIZE% ^
    --lr 1e-3 ^
    --dropout 0.3 ^
    --early_stop 15 ^
    --augment > logs\05_train_lstm.log 2>&1

if errorlevel 1 (
    echo ERROR: LSTM training failed
    exit /b 1
)
echo Done!

REM =============================================================================
REM STEP 6: Train ST-GCN Model
REM =============================================================================
echo.
echo [6/8] Training ST-GCN Model...
python train_stgcn.py ^
    --data sequences ^
    --output %OUTPUT_BASE%\stgcn ^
    --model_type lightweight ^
    --epochs %EPOCHS_STGCN% ^
    --batch_size 16 ^
    --lr 1e-3 ^
    --scheduler step ^
    --early_stop 20 ^
    --augment > logs\06_train_stgcn.log 2>&1

if errorlevel 1 (
    echo ERROR: ST-GCN training failed
    exit /b 1
)
echo Done!

REM =============================================================================
REM STEP 7: Evaluate Models
REM =============================================================================
echo.
echo [7/8] Evaluating Models...

python eval.py ^
    --model %OUTPUT_BASE%\lstm\best_model.pt ^
    --data sequences ^
    --type lstm ^
    --split test ^
    --output evaluation\lstm ^
    --plot > logs\07_eval_lstm.log 2>&1

python eval.py ^
    --model %OUTPUT_BASE%\stgcn\best_model.pt ^
    --data sequences ^
    --type stgcn ^
    --split test ^
    --output evaluation\stgcn ^
    --plot > logs\07_eval_stgcn.log 2>&1

echo Done!

REM =============================================================================
REM STEP 8: Export Models
REM =============================================================================
echo.
echo [8/8] Exporting Models to ONNX...
if not exist "models" mkdir "models"

python deploy\export_to_onnx.py ^
    --model %OUTPUT_BASE%\lstm\best_model.pt ^
    --type lstm ^
    --output models\lstm.onnx ^
    --seq_len %SEQUENCE_LENGTH% ^
    --verify > logs\08_export.log 2>&1

python deploy\export_to_onnx.py ^
    --model %OUTPUT_BASE%\stgcn\best_model.pt ^
    --type stgcn ^
    --output models\stgcn.onnx ^
    --seq_len %SEQUENCE_LENGTH% ^
    --verify >> logs\08_export.log 2>&1

echo Done!

REM =============================================================================
REM SUMMARY
REM =============================================================================
echo.
echo ==============================================================================
echo      PIPELINE COMPLETE!
echo ==============================================================================
echo.
echo Output Files:
echo   - Split dataset: yolo_split\
echo   - Pose keypoints: pose_npy\
echo   - Sequences: sequences\
echo   - LSTM model: %OUTPUT_BASE%\lstm\best_model.pt
echo   - ST-GCN model: %OUTPUT_BASE%\stgcn\best_model.pt
echo   - ONNX models: models\*.onnx
echo   - Evaluations: evaluation\
echo   - Logs: logs\
echo.
echo To run real-time inference:
echo   python inference_realtime.py --source 0 --model %OUTPUT_BASE%\lstm\best_model.pt --type lstm
echo.
echo To start API server:
echo   python deploy\api_server.py --model %OUTPUT_BASE%\lstm\best_model.pt
echo.

endlocal
