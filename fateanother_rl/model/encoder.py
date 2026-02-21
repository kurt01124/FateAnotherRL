"""Neural network encoders for observation components."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfEncoder(nn.Module):
    """Encode self unit vector → fixed-dim embedding."""

    def __init__(self, input_dim: int = 77, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, x):
        """(B, input_dim) → (B, hidden)"""
        return self.net(x)


class UnitEncoder(nn.Module):
    """Shared encoder for ally/enemy units with mean pooling."""

    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
        )

    def forward(self, x):
        """(B, N, input_dim) → (B, hidden) via mean pooling."""
        # (B, N, input_dim) → (B, N, hidden) → mean over N → (B, hidden)
        return self.net(x).mean(dim=1)


class GridEncoder(nn.Module):
    """2D grid encoder using convolutions."""

    def __init__(self, in_channels: int = 3, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # → (64, 4, 4) = 1024
        )
        self.fc = nn.Linear(64 * 4 * 4, out_dim)

    def forward(self, x):
        """(B, C, H, W) → (B, out_dim)"""
        h = self.conv(x).flatten(start_dim=1)  # (B, 1024)
        return F.relu(self.fc(h))
