#!/usr/bin/env python3
"""
export_to_onnx.py - Export PyTorch models to ONNX format.

This script exports trained pose classification models (LSTM, ST-GCN, Transformer)
to ONNX format for deployment.

Usage:
    python deploy/export_to_onnx.py --model outputs/lstm/best_model.pt --type lstm --output models/lstm.onnx
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
        description='Export PyTorch models to ONNX'
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--type', type=str, required=True,
                        choices=['lstm', 'stgcn', 'transformer'],
                        help='Model type')
    parser.add_argument('--output', type=str, required=True,
                        help='Output ONNX file path')
    parser.add_argument('--seq_len', type=int, default=16,
                        help='Sequence length')
    parser.add_argument('--num_joints', type=int, default=33,
                        help='Number of skeleton joints')
    parser.add_argument('--opset', type=int, default=12,
                        help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model (requires onnxsim)')
    parser.add_argument('--dynamic', action='store_true',
                        help='Enable dynamic batch size')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='Verify exported model')
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


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    model_type: str,
    seq_len: int = 16,
    num_joints: int = 33,
    opset_version: int = 12,
    dynamic_batch: bool = False
):
    """Export model to ONNX format."""
    # Create dummy input
    if model_type == 'stgcn':
        # ST-GCN input: (batch, T, V, C)
        dummy_input = torch.randn(1, seq_len, num_joints, 3)
        input_names = ['pose_sequence']
        output_names = ['class_logits']
    else:
        # LSTM/Transformer input: (batch, T, features)
        dummy_input = torch.randn(1, seq_len, num_joints * 2)
        input_names = ['pose_sequence']
        output_names = ['class_logits']

    # Dynamic axes for batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            input_names[0]: {0: 'batch_size'},
            output_names[0]: {0: 'batch_size'}
        }

    # Export
    logger.info(f"Exporting model to ONNX...")
    logger.info(f"  Input shape: {dummy_input.shape}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

    logger.info(f"  Exported to: {output_path}")
    return dummy_input


def simplify_onnx(onnx_path: str):
    """Simplify ONNX model using onnxsim."""
    try:
        import onnx
        from onnxsim import simplify

        logger.info("Simplifying ONNX model...")
        model = onnx.load(onnx_path)
        simplified_model, check = simplify(model)

        if check:
            onnx.save(simplified_model, onnx_path)
            logger.info("  Model simplified successfully")
        else:
            logger.warning("  Simplification check failed, keeping original")

    except ImportError:
        logger.warning("onnxsim not installed, skipping simplification")
        logger.warning("Install with: pip install onnxsim")


def verify_onnx(
    onnx_path: str,
    pytorch_model: nn.Module,
    dummy_input: torch.Tensor,
    tolerance: float = 1e-5
):
    """Verify ONNX model produces same output as PyTorch."""
    try:
        import onnx
        import onnxruntime as ort

        logger.info("Verifying ONNX model...")

        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("  ONNX model validation passed")

        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path)

        # Get PyTorch output
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(dummy_input).numpy()

        # Get ONNX output
        input_name = session.get_inputs()[0].name
        onnx_output = session.run(None, {input_name: dummy_input.numpy()})[0]

        # Compare
        max_diff = np.abs(pytorch_output - onnx_output).max()
        logger.info(f"  Max difference: {max_diff:.6f}")

        if max_diff < tolerance:
            logger.info("  Verification PASSED")
            return True
        else:
            logger.warning(f"  Verification FAILED (diff > {tolerance})")
            return False

    except ImportError:
        logger.warning("onnx or onnxruntime not installed, skipping verification")
        return None


def get_model_info(onnx_path: str):
    """Print ONNX model information."""
    try:
        import onnx

        model = onnx.load(onnx_path)

        logger.info("\nONNX Model Info:")
        logger.info(f"  IR Version: {model.ir_version}")
        logger.info(f"  Opset Version: {model.opset_import[0].version}")

        # Input info
        logger.info("  Inputs:")
        for inp in model.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            logger.info(f"    {inp.name}: {shape}")

        # Output info
        logger.info("  Outputs:")
        for out in model.graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            logger.info(f"    {out.name}: {shape}")

        # File size
        file_size = os.path.getsize(onnx_path) / (1024 * 1024)
        logger.info(f"  File Size: {file_size:.2f} MB")

    except ImportError:
        logger.warning("onnx package not installed")


def main():
    """Main function."""
    args = parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model: {args.model}")
    model, config = load_model(args.model, args.type)

    # Export
    dummy_input = export_to_onnx(
        model,
        str(output_path),
        args.type,
        args.seq_len,
        args.num_joints,
        args.opset,
        args.dynamic
    )

    # Simplify
    if args.simplify:
        simplify_onnx(str(output_path))

    # Verify
    if args.verify:
        verify_onnx(str(output_path), model, dummy_input)

    # Print info
    get_model_info(str(output_path))

    logger.info(f"\nExport complete: {output_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
