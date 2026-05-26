from __future__ import annotations

import math

import torch
from torch import nn


class _KnowledgeAttentionBlock(nn.Module):
    """One knowledge-guided attention branch with spatial query smoothing."""

    def __init__(
        self,
        dim: int = 512,
        attn_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.query_proj = nn.Linear(dim, attn_dim)
        self.key_value_proj = nn.Linear(dim, attn_dim)
        self.spatial_reconstruct = nn.Conv2d(attn_dim, attn_dim, kernel_size=3, padding=1)
        self.cross_attn = nn.MultiheadAttention(
            attn_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_proj = nn.Linear(attn_dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def _smooth_queries(self, queries: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = queries.shape
        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            return queries
        grid = queries.transpose(1, 2).reshape(batch_size, channels, side, side)
        grid = self.spatial_reconstruct(grid)
        return grid.flatten(2).transpose(1, 2)

    def forward(self, Zv: torch.Tensor, Zk: torch.Tensor) -> torch.Tensor:
        queries = self._smooth_queries(self.query_proj(Zv))
        keys_values = self.key_value_proj(Zk)
        guided, _ = self.cross_attn(queries, keys_values, keys_values, need_weights=False)
        guided = self.output_proj(guided)
        output = self.norm1(Zv + guided)
        output = self.norm2(output + self.ffn(output))
        return output


class KGALite(nn.Module):
    """
    Knowledge-guided cross-attention module with three parallel branches.

    Inputs:
      Zv: (batch, 196, 512)
      Zk: (batch, 1, 512)
    Output:
      knowledge-guided visual features with shape (batch, 196, 512).
    """

    def __init__(
        self,
        dim: int = 512,
        attn_dim: int = 256,
        num_heads: int = 8,
        num_branches: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                _KnowledgeAttentionBlock(
                    dim=dim,
                    attn_dim=attn_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_branches)
            ]
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * num_branches),
            nn.Linear(dim * num_branches, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, Zv: torch.Tensor, Zk: torch.Tensor) -> torch.Tensor:
        branch_outputs = [branch(Zv, Zk) for branch in self.branches]
        fused = self.fusion(torch.cat(branch_outputs, dim=-1))
        return self.output_norm(Zv + fused)
