"""
LSTM models for pose sequence classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PoseLSTM(nn.Module):
    """
    LSTM-based classifier for pose sequences.

    A simple but effective baseline model for temporal pose classification.
    """

    def __init__(
        self,
        input_size: int = 66,  # 33 joints * 2 (x, y)
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Input feature dimension per timestep
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output layers
        lstm_output_size = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            lengths: Optional sequence lengths for packed sequences

        Returns:
            Output logits of shape (batch, num_classes)
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)

        # LSTM forward
        if lengths is not None:
            # Pack sequences
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(x)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state (concatenate both directions if bidirectional)
        if self.bidirectional:
            # h_n: (num_layers * 2, batch, hidden)
            h_forward = h_n[-2]  # Last layer, forward
            h_backward = h_n[-1]  # Last layer, backward
            hidden = torch.cat([h_forward, h_backward], dim=1)
        else:
            hidden = h_n[-1]

        # Classify
        output = self.classifier(hidden)

        return output

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate features before classification."""
        x = self.input_proj(x)
        lstm_out, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            hidden = h_n[-1]

        return hidden


class BiLSTMClassifier(nn.Module):
    """
    Enhanced BiLSTM with attention mechanism for pose classification.
    """

    def __init__(
        self,
        input_size: int = 66,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        use_attention: bool = True
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.use_attention = use_attention

        # Input projection with residual
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        lstm_output_size = hidden_size * 2

        # Attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=False)
            )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Project input
        x = self.input_proj(x)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        if self.use_attention:
            # Compute attention weights
            attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
            attn_weights = F.softmax(attn_weights, dim=1)

            # Weighted sum
            context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)
        else:
            # Mean pooling
            context = lstm_out.mean(dim=1)

        # Classify
        output = self.classifier(context)

        return output

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for visualization."""
        if not self.use_attention:
            return None

        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)

        return attn_weights.squeeze(-1)


class TemporalConvLSTM(nn.Module):
    """
    LSTM with 1D temporal convolutions for capturing local patterns.
    """

    def __init__(
        self,
        input_size: int = 66,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.3
    ):
        super().__init__()

        # Multi-scale temporal convolutions
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_size, hidden_size // len(kernel_sizes),
                          kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(hidden_size // len(kernel_sizes)),
                nn.ReLU()
            )
            for k in kernel_sizes
        ])

        conv_output_size = hidden_size // len(kernel_sizes) * len(kernel_sizes)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=conv_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Output logits of shape (batch, num_classes)
        """
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)

        # Multi-scale convolutions
        conv_outputs = [conv(x) for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1)

        # Back to (batch, seq_len, features)
        x = x.transpose(1, 2)

        # LSTM
        lstm_out, (h_n, _) = self.lstm(x)

        # Use last hidden states from both directions
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)

        # Classify
        output = self.classifier(hidden)

        return output


def create_lstm_model(
    model_type: str = 'bilstm',
    input_size: int = 66,
    hidden_size: int = 128,
    num_layers: int = 2,
    num_classes: int = 2,
    dropout: float = 0.3,
    **kwargs
) -> nn.Module:
    """
    Factory function to create LSTM models.

    Args:
        model_type: 'lstm', 'bilstm', or 'conv_lstm'
        input_size: Input feature dimension
        hidden_size: Hidden layer size
        num_layers: Number of LSTM layers
        num_classes: Number of output classes
        dropout: Dropout rate

    Returns:
        PyTorch model
    """
    if model_type == 'lstm':
        return PoseLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=False
        )
    elif model_type == 'bilstm':
        return BiLSTMClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            use_attention=kwargs.get('use_attention', True)
        )
    elif model_type == 'conv_lstm':
        return TemporalConvLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
