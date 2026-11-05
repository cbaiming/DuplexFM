import torch
from torch import nn
from torch.nn import functional as F

class CellNorm(nn.Module):
    """Normalize channels at each cell; never mix real cells with padding."""

    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class LocalBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.norm = CellNorm(width)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x, valid):
        z = self.drop(F.gelu(self.norm(self.conv(x))))
        return (x + z).masked_fill(~valid, 0)

def _diagonal_indices(g, t, device):
    i = torch.arange(g, device=device)
    offsets = torch.arange(g + t - 1, device=device)
    j = offsets[None, :] - i[:, None]
    return j, j.clamp(0, t - 1)
