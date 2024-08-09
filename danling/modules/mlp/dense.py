# DanLing
# Copyright (C) 2022-Present  DanLing

# This program is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from torch import nn


class Dense(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        norm: str = "LayerNorm",
        activation: str = "ReLU",
        dropout: float = 0.1,
        pool: str = "AdaptiveAvgPool1d",
        bias: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm = getattr(nn, norm)(out_features) if norm else None
        self.activation = getattr(nn, activation)() if activation else None
        self.dropout = nn.Dropout(dropout)
        self.pool = getattr(nn, pool)(out_features) if self.residual else None

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        if self.residual:
            out = out + self.pool(x)
        return out
