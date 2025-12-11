#!/usr/bin/env python3
"""
export_to_tflite.py - Export models to TensorFlow Lite format.

This script exports trained models to TFLite format for mobile/edge deployment.

Usage:
    python deploy/export_to_tflite.py --model outputs/lstm/best_model.pt --type lstm --output models/lstm.tflite
"""

import argparse
import os
import sys
from pathlib import Path
import logging

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Export PyTorch models to TFLite'
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--type', type=str, required=True,
                        choices=['lstm', 'stgcn', 'transformer'],
                        help='Model type')
    parser.add_argument('--output', type=str, required=True,
                        help='Output TFLite file path')
    parser.add_argument('--seq_len', type=int, default=16,
                        help='Sequence length')
    parser.add_argument('--num_joints', type=int, default=33,
                        help='Number of skeleton joints')
    parser.add_argument('--quantize', action='store_true',
                        help='Apply dynamic range quantization')
    parser.add_argument('--fp16', action='store_true',
                        help='Use float16 quantization')
    return parser.parse_args()


def load_model(model_path: str, model_type: str) -> nn.Module:
    """Load PyTorch model from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu')
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
            num_joints=config.get('num_joints', 33),
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
    model.eval()

    return model, config


def pytorch_to_onnx(
    model: nn.Module,
    model_type: str,
    seq_len: int,
    num_joints: int
) -> str:
    """Convert PyTorch model to ONNX (intermediate step)."""
    import tempfile

    # Create dummy input
    if model_type == 'stgcn':
        dummy_input = torch.randn(1, seq_len, num_joints, 3)
    else:
        dummy_input = torch.randn(1, seq_len, num_joints * 2)

    # Export to temp file
    temp_onnx = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)

    torch.onnx.export(
        model,
        dummy_input,
        temp_onnx.name,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )

    return temp_onnx.name, dummy_input


def onnx_to_tflite(
    onnx_path: str,
    output_path: str,
    quantize: bool = False,
    fp16: bool = False
):
    """Convert ONNX model to TFLite."""
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
        import tempfile

        logger.info("Converting ONNX to TensorFlow...")

        # Load ONNX model
        onnx_model = onnx.load(onnx_path)

        # Convert to TF
        tf_rep = prepare(onnx_model)

        # Save to SavedModel format
        temp_dir = tempfile.mkdtemp()
        tf_rep.export_graph(temp_dir)

        logger.info("Converting TensorFlow to TFLite...")

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir)

        if quantize:
            logger.info("  Applying dynamic range quantization")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if fp16:
            logger.info("  Applying float16 quantization")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        logger.info(f"  Saved TFLite model to: {output_path}")

        # Cleanup temp ONNX
        os.unlink(onnx_path)

        # File size
        file_size = os.path.getsize(output_path) / 1024
        logger.info(f"  File size: {file_size:.2f} KB")

        return True

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install onnx-tf tensorflow")
        return False


def verify_tflite(tflite_path: str, pytorch_model: nn.Module, dummy_input: torch.Tensor):
    """Verify TFLite model output."""
    try:
        import tensorflow as tf

        logger.info("Verifying TFLite model...")

        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Get PyTorch output
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(dummy_input).numpy()

        # Get TFLite output
        interpreter.set_tensor(input_details[0]['index'], dummy_input.numpy())
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]['index'])

        # Compare
        max_diff = np.abs(pytorch_output - tflite_output).max()
        logger.info(f"  Max difference: {max_diff:.6f}")

        if max_diff < 1e-4:
            logger.info("  Verification PASSED")
        else:
            logger.warning("  Output differs (expected with quantization)")

    except ImportError:
        logger.warning("TensorFlow not installed, skipping verification")


def main():
    """Main function."""
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model: {args.model}")
    model, config = load_model(args.model, args.type)

    # Convert through ONNX
    logger.info("Converting PyTorch to ONNX...")
    onnx_path, dummy_input = pytorch_to_onnx(
        model, args.type, args.seq_len, args.num_joints
    )

    # Convert to TFLite
    success = onnx_to_tflite(
        onnx_path,
        str(output_path),
        args.quantize,
        args.fp16
    )

    if success:
        verify_tflite(str(output_path), model, dummy_input)
        logger.info(f"\nExport complete: {output_path}")
    else:
        logger.error("Export failed")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
