from __future__ import annotations


def build_cnn(input_length: int, dropout: float = 0.3):
    """Construct a one-dimensional residual CNN when the optional torch extra is installed."""
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("Install the optional CNN dependencies: pip install -e '.[cnn]'") from exc

    class Residual1D(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.block = nn.Sequential(nn.Conv1d(channels, channels, 5, padding=2),
                                       nn.BatchNorm1d(channels), nn.ReLU(),
                                       nn.Dropout(dropout),
                                       nn.Conv1d(channels, channels, 5, padding=2),
                                       nn.BatchNorm1d(channels))
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(x + self.block(x))

    return nn.Sequential(nn.Conv1d(1, 32, 9, padding=4), nn.ReLU(),
                         nn.MaxPool1d(2), Residual1D(32),
                         nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
                         nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                         nn.Dropout(dropout), nn.Linear(64, 1))
