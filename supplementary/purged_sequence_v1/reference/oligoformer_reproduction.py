"""Exact extracted historical corrected reproduction, NOT untouched author code."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def sinusoidal_encoding(length: int, dimension: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    scale = torch.exp(torch.arange(0, dimension, 2, dtype=torch.float32) * (-math.log(10000.0) / dimension))
    encoding = torch.zeros(length, dimension, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * scale)
    encoding[:, 1::2] = torch.cos(position * scale)
    return encoding


class OligoSequenceBranch(nn.Module):
    def __init__(
        self,
        *,
        kind: str,
        input_length: int,
        hidden_dim: int = 32,
        n_head: int = 8,
        transformer_hidden: int = 128,
        dropout: float = 0.02,
    ) -> None:
        super().__init__()
        if kind not in {"guide", "target"}:
            raise ValueError(kind)
        self.kind = kind
        kernel_size = 1 if kind == "guide" else 5
        self.output_length = input_length - kernel_size + 1
        self.conv = nn.Conv1d(5, 64, kernel_size=kernel_size)
        lstm_input = 69 if kind == "guide" else 64
        self.lstm = nn.LSTM(
            lstm_input,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,
            nhead=n_head,
            dim_feedforward=transformer_hidden,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("position", sinusoidal_encoding(self.output_length, hidden_dim * 2), persistent=False)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        one_hot = F.one_hot(token_ids.long(), num_classes=5).float()
        if self.kind == "guide":
            if lengths is None:
                raise ValueError("Guide lengths are required.")
            valid_mask = torch.arange(one_hot.shape[1], device=one_hot.device)[None, :] < lengths.to(one_hot.device)[:, None]
            one_hot = one_hot * valid_mask.unsqueeze(-1)
        else:
            valid_mask = None

        convolved = F.relu(self.conv(one_hot.transpose(1, 2))).transpose(1, 2)
        avg_channels = F.avg_pool1d(convolved, kernel_size=2, stride=2)
        max_channels = F.max_pool1d(convolved, kernel_size=2, stride=2)
        if self.kind == "guide":
            features = torch.cat([one_hot, avg_channels, max_channels], dim=-1)
            packed = pack_padded_sequence(
                features,
                lengths.detach().long().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.lstm(packed)
            encoded, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=self.output_length,
            )
            padding_mask = ~valid_mask
        else:
            features = torch.cat([avg_channels, max_channels], dim=-1)
            encoded, _ = self.lstm(features)
            padding_mask = None
        encoded = self.dropout(F.relu(encoded))
        encoded = encoded + self.position[: encoded.shape[1]].unsqueeze(0)
        encoded = self.transformer(encoded, src_key_padding_mask=padding_mask)
        if valid_mask is not None:
            encoded = encoded * valid_mask.unsqueeze(-1)
        return encoded


class CorrectedOligoFormerSiRNA(nn.Module):
    requires_sequence = True

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.guide_encoder = OligoSequenceBranch(kind="guide", input_length=19, dropout=dropout)
        self.target_encoder = OligoSequenceBranch(kind="target", input_length=57, dropout=dropout)
        merged_dim = 19 * 64 + 53 * 64 + 128 + 128 + 24
        self.classifier = nn.Sequential(
            nn.Linear(merged_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        *,
        guide_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        lengths = torch.full(
            (guide_tokens.shape[0],),
            19,
            dtype=torch.long,
            device="cpu",
        )
        guide = self.guide_encoder(guide_tokens, lengths).flatten(1)
        target = self.target_encoder(target_tokens).flatten(1)
        return torch.sigmoid(self.classifier(torch.cat([guide, target, features], dim=1)).squeeze(-1))
