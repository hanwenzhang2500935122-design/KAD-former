from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .akg import AKG
from .kga import KGALite
from .sam import SAM
from .vit_backbone import ViTBackbone


class KADFormerLite(nn.Module):
    """
    Minimal KAD-Former demo.

    Image -> ViT -> AKG lookup -> SAM -> KGA -> classifier.
    """

    def __init__(
        self,
        num_classes: int = 4,
        embeddings_path: str | Path = "data/knowledge_graph/node_embeddings.pt",
        vit_model_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.vit = ViTBackbone(model_name=vit_model_name, pretrained=pretrained, output_dim=768)
        self.akg = AKG(embeddings_path)
        self.sam = SAM(vision_dim=768, knowledge_dim=256, unified_dim=512, num_heads=8)
        self.kga = KGALite(dim=512, attn_dim=256, num_heads=8, num_branches=3)
        self.coarse_classifier = nn.Linear(768, num_classes)
        self.classifier = nn.Sequential(
            nn.LayerNorm(512 + 768),
            nn.Linear(512 + 768, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        patch_tokens, cls_token = self.vit(x)

        if labels is None:
            with torch.no_grad():
                labels = self.coarse_classifier(cls_token).argmax(dim=1)

        knowledge_vec = self.akg(labels)
        if return_aux:
            Zv, Zk, alignment_loss = self.sam(
                patch_tokens,
                knowledge_vec,
                labels=labels,
                return_loss=True,
            )
        else:
            Zv, Zk = self.sam(patch_tokens, knowledge_vec)
            alignment_loss = patch_tokens.new_zeros(())
        guided_visual = self.kga(Zv, Zk)
        pooled_visual = guided_visual.mean(dim=1)
        fused = torch.cat([pooled_visual, cls_token], dim=1)
        logits = self.classifier(fused)
        if return_aux:
            return logits, {"alignment_loss": alignment_loss}
        return logits
