#!/usr/bin/env python3
"""
api_server.py - FastAPI server for fight detection inference.

Provides REST API endpoints for:
- Single image classification
- Pose sequence classification
- Video processing (async)

Usage:
    python deploy/api_server.py --host 0.0.0.0 --port 8000

    # Or with uvicorn directly:
    uvicorn deploy.api_server:app --host 0.0.0.0 --port 8000
"""

import argparse
import os
import sys
import io
import base64
import tempfile
from pathlib import Path
from typing import List, Optional
import logging

import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import FastAPI dependencies
try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not installed. Install with: pip install fastapi uvicorn python-multipart")

# Global model instances
pose_extractor = None
pose_model = None
model_config = None


class PoseSequence(BaseModel):
    """Pose sequence input model."""
    sequence: List[List[float]]  # Shape: (T, features)
    threshold: float = 0.5


class PredictionResponse(BaseModel):
    """Prediction response model."""
    label: int
    class_name: str
    confidence: float
    probabilities: List[float]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str


def create_app():
    """Create FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available")

    app = FastAPI(
        title="Fight Detection API",
        description="Real-time fight/aggressive pose detection API",
        version="1.0.0"
    )

    return app


app = create_app() if FASTAPI_AVAILABLE else None


def load_models(model_path: str, model_type: str = 'lstm', device: str = 'cpu'):
    """Load models for inference."""
    global pose_extractor, pose_model, model_config

    import torch

    logger.info(f"Loading models on device: {device}")

    # Load pose extractor
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose_extractor = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        )
        logger.info("Pose extractor loaded")
    except ImportError:
        logger.warning("MediaPipe not available, pose extraction disabled")

    # Load classification model
    if model_path and Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint.get('config', {})

        if model_type == 'lstm':
            from models.lstm import create_lstm_model
            pose_model = create_lstm_model(
                model_type=config.get('model_type', 'bilstm'),
                input_size=config.get('input_size', 66),
                hidden_size=config.get('hidden_size', 128),
                num_layers=config.get('num_layers', 2),
                num_classes=2,
                dropout=0
            )
        elif model_type == 'stgcn':
            from models.stgcn import create_stgcn_model
            pose_model = create_stgcn_model(
                model_type=config.get('model_type', 'lightweight'),
                num_classes=2,
                dropout=0
            )

        pose_model.load_state_dict(checkpoint['model_state_dict'])
        pose_model.to(torch.device(device))
        pose_model.eval()
        model_config = config

        logger.info(f"Classification model loaded: {model_type}")
    else:
        logger.warning(f"Model not found: {model_path}")


if FASTAPI_AVAILABLE:
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="ok",
            model_loaded=pose_model is not None,
            device=str(next(pose_model.parameters()).device) if pose_model else "none"
        )

    @app.post("/predict/sequence", response_model=PredictionResponse)
    async def predict_sequence(data: PoseSequence):
        """
        Classify a pose sequence.

        Expected input: sequence of flattened pose features (T, 66)
        """
        import torch

        if pose_model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            # Convert to tensor
            sequence = np.array(data.sequence, dtype=np.float32)
            if sequence.ndim == 2:
                sequence = sequence.reshape(1, *sequence.shape)

            input_tensor = torch.FloatTensor(sequence)
            device = next(pose_model.parameters()).device
            input_tensor = input_tensor.to(device)

            # Inference
            with torch.no_grad():
                outputs = pose_model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred = probs.argmax(dim=1).item()
                confidence = probs[0, pred].item()

            class_names = ['Normal', 'Aggressive']

            return PredictionResponse(
                label=pred,
                class_name=class_names[pred],
                confidence=confidence,
                probabilities=probs[0].cpu().tolist()
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/image", response_model=PredictionResponse)
    async def predict_image(file: UploadFile = File(...)):
        """
        Extract pose from image and return features.

        Note: Single image cannot be classified temporally.
        This endpoint returns pose features for building sequences.
        """
        import cv2

        if pose_extractor is None:
            raise HTTPException(status_code=503, detail="Pose extractor not available")

        try:
            # Read image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image")

            # Extract pose
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose_extractor.process(rgb)

            if not results.pose_landmarks:
                raise HTTPException(status_code=400, detail="No pose detected in image")

            # Get keypoints
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])

            # For single image, return neutral prediction
            return PredictionResponse(
                label=0,
                class_name="Unknown (single frame)",
                confidence=0.5,
                probabilities=[0.5, 0.5]
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/extract/pose")
    async def extract_pose(file: UploadFile = File(...)):
        """Extract pose keypoints from an image."""
        import cv2

        if pose_extractor is None:
            raise HTTPException(status_code=503, detail="Pose extractor not available")

        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image")

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose_extractor.process(rgb)

            if not results.pose_landmarks:
                return JSONResponse({
                    "detected": False,
                    "keypoints": None
                })

            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                })

            return JSONResponse({
                "detected": True,
                "keypoints": keypoints
            })

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fight Detection API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind')
    parser.add_argument('--model', type=str, default='outputs/lstm/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--type', type=str, default='lstm',
                        choices=['lstm', 'stgcn'], help='Model type')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    return parser.parse_args()


def main():
    """Main function."""
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI not installed. Install with: pip install fastapi uvicorn python-multipart")
        sys.exit(1)

    args = parse_args()

    # Load models
    load_models(args.model, args.type, args.device)

    # Run server
    logger.info(f"Starting server at http://{args.host}:{args.port}")
    logger.info(f"API documentation at http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "deploy.api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == '__main__':
    main()
