"""
Transformer-based models for pose sequence classification.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class JointEmbedding(nn.Module):
    """Embedding layer for skeleton joints."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_joints: int = 33,
        use_joint_tokens: bool = True
    ):
        super().__init__()

        self.use_joint_tokens = use_joint_tokens
        self.num_joints = num_joints

        self.linear = nn.Linear(input_dim, embed_dim)

        if use_joint_tokens:
            # Learnable joint type embeddings
            self.joint_tokens = nn.Parameter(torch.randn(1, num_joints, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch, T, num_joints, channels) or (batch, T, features)

        Returns:
            Embedded tensor of shape (batch, T, embed_dim)
        """
        if x.dim() == 3:
            # (batch, T, features) - already flattened
            x = self.linear(x)
        else:
            # (batch, T, num_joints, channels)
            batch, T, V, C = x.shape
            x = x.view(batch * T, V, C)
            x = self.linear(x)  # (batch*T, V, embed_dim)

            if self.use_joint_tokens:
                x = x + self.joint_tokens

            # Pool over joints
            x = x.mean(dim=1)  # (batch*T, embed_dim)
            x = x.view(batch, T, -1)

        return x


class PoseTransformer(nn.Module):
    """
    Transformer encoder for pose sequence classification.
    """

    def __init__(
        self,
        input_dim: int = 66,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 256,
        num_classes: int = 2,
        max_len: int = 64,
        dropout: float = 0.1,
        use_cls_token: bool = True
    ):
        """
        Initialize Pose Transformer.

        Args:
            input_dim: Input feature dimension per timestep
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward hidden dimension
            num_classes: Number of output classes
            max_len: Maximum sequence length
            dropout: Dropout rate
            use_cls_token: Use a CLS token for classification
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token

        # Input embedding
        self.input_embed = nn.Linear(input_dim, embed_dim)

        # CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_len + 1, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mask: Optional attention mask

        Returns:
            Output logits of shape (batch, num_classes)
        """
        batch_size = x.shape[0]

        # Embed input
        x = self.input_embed(x)

        # Add CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)

        # Normalize
        x = self.norm(x)

        # Get classification token or mean pool
        if self.use_cls_token:
            x = x[:, 0]  # CLS token
        else:
            x = x.mean(dim=1)  # Mean pooling

        # Classify
        output = self.classifier(x)

        return output

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights from the last layer."""
        batch_size = x.shape[0]

        x = self.input_embed(x)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_encoding(x)

        # Get attention from each layer
        attention_weights = []
        for layer in self.transformer.layers:
            # Self attention
            attn_output, attn_weight = layer.self_attn(
                x, x, x, need_weights=True, average_attn_weights=False
            )
            attention_weights.append(attn_weight)
            x = layer(x)

        # Return last layer attention
        return attention_weights[-1]


class SpatialTemporalTransformer(nn.Module):
    """
    Transformer with separate spatial and temporal attention.
    """

    def __init__(
        self,
        num_joints: int = 33,
        joint_dim: int = 2,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_spatial_layers: int = 2,
        num_temporal_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_joints = num_joints
        self.embed_dim = embed_dim

        # Joint embedding
        self.joint_embed = nn.Linear(joint_dim, embed_dim)

        # Learnable spatial position embedding (one per joint)
        self.spatial_pos = nn.Parameter(torch.randn(1, num_joints, embed_dim) * 0.02)

        # Spatial transformer (attention over joints)
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=num_spatial_layers)

        # Temporal positional encoding
        self.temporal_pos = PositionalEncoding(embed_dim, max_len=128, dropout=dropout)

        # Temporal transformer (attention over time)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=num_temporal_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, T, joints*channels) or (batch, T, joints, channels)

        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Handle input shape
        if x.dim() == 3:
            batch, T, features = x.shape
            x = x.view(batch, T, self.num_joints, -1)

        batch, T, V, C = x.shape

        # Embed joints
        x = self.joint_embed(x)  # (batch, T, V, embed_dim)

        # Spatial attention (per timestep)
        x = x.view(batch * T, V, -1)
        x = x + self.spatial_pos
        x = self.spatial_transformer(x)

        # Pool over joints
        x = x.mean(dim=1)  # (batch*T, embed_dim)
        x = x.view(batch, T, -1)

        # Temporal attention
        x = self.temporal_pos(x)
        x = self.temporal_transformer(x)

        # Mean pool over time
        x = x.mean(dim=1)

        # Classify
        output = self.classifier(x)

        return output


def create_transformer_model(
    model_type: str = 'transformer',
    input_dim: int = 66,
    embed_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    num_classes: int = 2,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create transformer models.

    Args:
        model_type: 'transformer' or 'spatial_temporal'
        input_dim: Input feature dimension
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        num_classes: Number of output classes
        dropout: Dropout rate

    Returns:
        PyTorch model
    """
    if model_type == 'transformer':
        return PoseTransformer(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )
    elif model_type == 'spatial_temporal':
        return SpatialTemporalTransformer(
            num_joints=kwargs.get('num_joints', 33),
            joint_dim=kwargs.get('joint_dim', 2),
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_spatial_layers=num_layers // 2,
            num_temporal_layers=num_layers // 2,
            num_classes=num_classes,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
