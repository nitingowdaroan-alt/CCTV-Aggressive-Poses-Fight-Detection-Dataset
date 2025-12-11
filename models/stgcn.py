"""
Spatial-Temporal Graph Convolutional Network (ST-GCN) for skeleton-based action recognition.

Based on: "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition"
by Yan et al. (AAAI 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


def get_mediapipe_graph() -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Get the adjacency matrix for MediaPipe 33 keypoints.

    Returns:
        Tuple of (adjacency matrix, edge list)
    """
    num_nodes = 33

    # MediaPipe skeleton connections
    edges = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        # Torso
        (11, 12), (11, 23), (12, 24), (23, 24),
        # Left arm
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        # Right arm
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        # Left leg
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
        # Right leg
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    ]

    # Self-connections
    self_edges = [(i, i) for i in range(num_nodes)]
    all_edges = edges + self_edges

    # Build adjacency matrix
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in all_edges:
        A[i, j] = 1
        A[j, i] = 1

    return A, edges


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """Normalize adjacency matrix with degree matrix."""
    D = np.diag(A.sum(axis=1) ** (-0.5))
    D[np.isinf(D)] = 0
    return D @ A @ D


class SpatialGraphConv(nn.Module):
    """
    Spatial Graph Convolution layer.

    Performs graph convolution on skeleton joints at a single timestep.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: np.ndarray,
        kernel_size: int = 1,
        bias: bool = True
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.out_channels = out_channels

        # Register adjacency as buffer (not a parameter)
        A = normalize_adjacency(adjacency)
        self.register_buffer('A', torch.FloatTensor(A))

        # Learnable adjacency component
        self.PA = nn.Parameter(torch.zeros_like(self.A))

        # Convolution weights
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, T, V)

        Returns:
            Output tensor of shape (batch, out_channels, T, V)
        """
        # Effective adjacency
        A = self.A + self.PA

        # Graph convolution
        x = self.conv(x)

        # Split into kernel parts
        batch, _, T, V = x.shape
        x = x.view(batch, self.kernel_size, self.out_channels, T, V)

        # Apply adjacency
        x = torch.einsum('bkctv,vw->bkctw', x, A)

        # Sum over kernel dimension
        x = x.sum(dim=1)

        return x


class TemporalConv(nn.Module):
    """Temporal convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        dilation: int = 1
    ):
        super().__init__()

        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            dilation=(dilation, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class STGCNBlock(nn.Module):
    """
    ST-GCN block consisting of spatial graph convolution + temporal convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: np.ndarray,
        stride: int = 1,
        temporal_kernel_size: int = 9,
        dropout: float = 0.0,
        residual: bool = True
    ):
        super().__init__()

        self.residual = residual

        # Spatial graph convolution
        self.gcn = SpatialGraphConv(in_channels, out_channels, adjacency)
        self.bn_gcn = nn.BatchNorm2d(out_channels)

        # Temporal convolution
        self.tcn = TemporalConv(out_channels, out_channels, temporal_kernel_size, stride)

        # Residual connection
        if not residual:
            self.res = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.res = lambda x: x
        else:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, T, V)

        Returns:
            Output tensor
        """
        res = self.res(x)

        x = self.gcn(x)
        x = self.bn_gcn(x)
        x = self.relu(x)
        x = self.tcn(x)
        x = self.dropout(x)

        x = x + res
        x = self.relu(x)

        return x


class STGCN(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Network for skeleton-based action recognition.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,  # (x, y, confidence)
        num_joints: int = 33,
        num_frames: int = 16,
        dropout: float = 0.0,
        edge_importance: bool = True
    ):
        """
        Initialize ST-GCN.

        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels per joint
            num_joints: Number of skeleton joints
            num_frames: Number of frames in sequence
            dropout: Dropout rate
            edge_importance: Whether to use learnable edge importance
        """
        super().__init__()

        self.num_joints = num_joints
        self.num_frames = num_frames

        # Get graph structure
        A, _ = get_mediapipe_graph()

        # Data normalization
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        # ST-GCN blocks
        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, 64, A, stride=1, dropout=dropout),
            STGCNBlock(64, 64, A, stride=1, dropout=dropout),
            STGCNBlock(64, 64, A, stride=1, dropout=dropout),
            STGCNBlock(64, 128, A, stride=2, dropout=dropout),
            STGCNBlock(128, 128, A, stride=1, dropout=dropout),
            STGCNBlock(128, 128, A, stride=1, dropout=dropout),
            STGCNBlock(128, 256, A, stride=2, dropout=dropout),
            STGCNBlock(256, 256, A, stride=1, dropout=dropout),
            STGCNBlock(256, 256, A, stride=1, dropout=dropout),
        ])

        # Global pooling and classifier
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, T, joints, channels) or (batch, T, joints*channels)

        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Handle different input shapes
        if x.dim() == 3:
            # (batch, T, features) -> (batch, T, joints, channels)
            batch, T, features = x.shape
            if features == self.num_joints * 2:
                # Only (x, y), add confidence channel
                x = x.view(batch, T, self.num_joints, 2)
                conf = torch.ones(batch, T, self.num_joints, 1, device=x.device)
                x = torch.cat([x, conf], dim=-1)
            else:
                x = x.view(batch, T, self.num_joints, -1)

        batch, T, V, C = x.shape

        # Reshape for batch norm: (N, C*V, T)
        x = x.permute(0, 3, 2, 1).contiguous()  # (N, C, V, T)
        x = x.view(batch, C * V, T)
        x = self.data_bn(x)
        x = x.view(batch, C, V, T)
        x = x.permute(0, 1, 3, 2)  # (N, C, T, V)

        # ST-GCN layers
        for layer in self.layers:
            x = layer(x)

        # Global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(batch, -1)

        # Classification
        x = self.fc(x)

        return x


class LightweightSTGCN(nn.Module):
    """
    Lightweight ST-GCN variant for faster inference.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        num_joints: int = 33,
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()

        self.num_joints = num_joints

        A, _ = get_mediapipe_graph()

        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, hidden_dim, A, stride=1, dropout=dropout),
            STGCNBlock(hidden_dim, hidden_dim, A, stride=2, dropout=dropout),
            STGCNBlock(hidden_dim, hidden_dim * 2, A, stride=2, dropout=dropout),
        ])

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle input shape
        if x.dim() == 3:
            batch, T, features = x.shape
            if features == self.num_joints * 2:
                x = x.view(batch, T, self.num_joints, 2)
                conf = torch.ones(batch, T, self.num_joints, 1, device=x.device)
                x = torch.cat([x, conf], dim=-1)
            else:
                x = x.view(batch, T, self.num_joints, -1)

        batch, T, V, C = x.shape

        # Batch norm
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(batch, C * V, T)
        x = self.data_bn(x)
        x = x.view(batch, C, V, T)
        x = x.permute(0, 1, 3, 2)

        # Layers
        for layer in self.layers:
            x = layer(x)

        # Pool and classify
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(batch, -1)
        x = self.fc(x)

        return x


def create_stgcn_model(
    model_type: str = 'stgcn',
    num_classes: int = 2,
    in_channels: int = 3,
    num_joints: int = 33,
    dropout: float = 0.0,
    **kwargs
) -> nn.Module:
    """
    Factory function to create ST-GCN models.

    Args:
        model_type: 'stgcn' or 'lightweight'
        num_classes: Number of output classes
        in_channels: Input channels per joint
        num_joints: Number of skeleton joints
        dropout: Dropout rate

    Returns:
        PyTorch model
    """
    if model_type == 'stgcn':
        return STGCN(
            num_classes=num_classes,
            in_channels=in_channels,
            num_joints=num_joints,
            dropout=dropout
        )
    elif model_type == 'lightweight':
        return LightweightSTGCN(
            num_classes=num_classes,
            in_channels=in_channels,
            num_joints=num_joints,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
