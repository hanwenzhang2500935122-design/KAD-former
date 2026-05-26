from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SAM(nn.Module):
    """
    Semantic alignment module with self-attention and bidirectional cross-attention.

    Inputs:
      Xv: (batch, 196, 768)
      Xk: (batch, 1, 256)
    Outputs:
      Zv: (batch, 196, 512)
      Zk: (batch, 1, 512)
    """

    def __init__(
        self,
        vision_dim: int = 768,
        knowledge_dim: int = 256,
        unified_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        alignment_margin: float = 1.0,
        alignment_lambda: float = 0.2,
    ) -> None:
        super().__init__()
        self.alignment_margin = alignment_margin
        self.alignment_lambda = alignment_lambda
        self.vision_proj = nn.Linear(vision_dim, unified_dim)
        self.knowledge_proj = nn.Linear(knowledge_dim, unified_dim)

        self.vision_self_attn = nn.MultiheadAttention(
            unified_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.knowledge_self_attn = nn.MultiheadAttention(
            unified_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.vision_to_knowledge = nn.MultiheadAttention(
            unified_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.knowledge_to_vision = nn.MultiheadAttention(
            unified_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm_v1 = nn.LayerNorm(unified_dim)
        self.norm_k1 = nn.LayerNorm(unified_dim)
        self.norm_v2 = nn.LayerNorm(unified_dim)
        self.norm_k2 = nn.LayerNorm(unified_dim)
        self.ffn_v = nn.Sequential(
            nn.Linear(unified_dim, unified_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(unified_dim * 4, unified_dim),
        )
        self.ffn_k = nn.Sequential(
            nn.Linear(unified_dim, unified_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(unified_dim * 4, unified_dim),
        )
        self.norm_v3 = nn.LayerNorm(unified_dim)
        self.norm_k3 = nn.LayerNorm(unified_dim)

    def compute_alignment_loss(
        self,
        Zv: torch.Tensor,
        Zk: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Compute a batch-level SAM alignment loss.

        The loss follows the paper's center-based idea in a mini-demo form:
        same-class visual and knowledge centers are pulled together, while
        different-class visual/knowledge centers are separated by a margin.
        """
        if labels is None:
            return Zv.new_zeros(())

        labels = labels.to(device=Zv.device, dtype=torch.long)
        visual_samples = Zv.mean(dim=1)
        knowledge_samples = Zk.mean(dim=1)
        class_ids = labels.unique(sorted=True)
        if class_ids.numel() == 0:
            return Zv.new_zeros(())

        visual_centers: list[torch.Tensor] = []
        knowledge_centers: list[torch.Tensor] = []
        for class_id in class_ids:
            mask = labels == class_id
            visual_centers.append(visual_samples[mask].mean(dim=0))
            knowledge_centers.append(knowledge_samples[mask].mean(dim=0))

        visual_center_tensor = F.normalize(torch.stack(visual_centers), dim=-1)
        knowledge_center_tensor = F.normalize(torch.stack(knowledge_centers), dim=-1)
        intra_loss = F.mse_loss(visual_center_tensor, knowledge_center_tensor)

        if class_ids.numel() == 1:
            return intra_loss

        distances = torch.cdist(visual_center_tensor, knowledge_center_tensor, p=2)
        off_diagonal = ~torch.eye(
            class_ids.numel(),
            dtype=torch.bool,
            device=distances.device,
        )
        inter_loss = F.relu(self.alignment_margin - distances[off_diagonal]).pow(2).mean()
        return intra_loss + self.alignment_lambda * inter_loss

    def forward(
        self,
        Xv: torch.Tensor,
        Xk: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_loss: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Zv = self.vision_proj(Xv)
        Zk = self.knowledge_proj(Xk)

        v_self, _ = self.vision_self_attn(Zv, Zv, Zv, need_weights=False)
        k_self, _ = self.knowledge_self_attn(Zk, Zk, Zk, need_weights=False)
        Zv = self.norm_v1(Zv + v_self)
        Zk = self.norm_k1(Zk + k_self)

        v_cross, _ = self.vision_to_knowledge(Zv, Zk, Zk, need_weights=False)
        k_cross, _ = self.knowledge_to_vision(Zk, Zv, Zv, need_weights=False)
        Zv = self.norm_v2(Zv + v_cross)
        Zk = self.norm_k2(Zk + k_cross)

        Zv = self.norm_v3(Zv + self.ffn_v(Zv))
        Zk = self.norm_k3(Zk + self.ffn_k(Zk))
        if return_loss:
            alignment_loss = self.compute_alignment_loss(Zv, Zk, labels)
            return Zv, Zk, alignment_loss
        return Zv, Zk
