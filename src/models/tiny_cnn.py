"""MicroCrack-Tiny: depthwise-separable CNN with BatchNorm + ReLU, GAP and a small linear head."""
from __future__ import annotations
import torch
from torch import nn


def _dw_sep(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU(inplace=False),
        nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=False),
    )


class MicroCrackTiny(nn.Module):
    def __init__(self, num_classes: int = 2, width: float = 1.0, dropout: float = 0.1):
        super().__init__()
        c = [max(8, int(round(v * width))) for v in (16, 32, 64, 64, 128, 128)]
        self.features = nn.Sequential(
            nn.Conv2d(3, c[0], 3, 2, 1, bias=False), nn.BatchNorm2d(c[0]), nn.ReLU(inplace=False),
            _dw_sep(c[0], c[1], 1),
            _dw_sep(c[1], c[2], 2),
            _dw_sep(c[2], c[3], 1),
            _dw_sep(c[3], c[4], 2),
            _dw_sep(c[4], c[5], 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c[5], num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.flatten(self.pool(self.features(x)), 1))
