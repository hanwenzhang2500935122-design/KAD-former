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
        vit_checkpoint: str | Path | None = None,
        knowledge_gt_mix_ratio: float = 0.5,
        confidence_weighted_alignment: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= knowledge_gt_mix_ratio <= 1.0:
            raise ValueError("knowledge_gt_mix_ratio must be in [0, 1].")
        self.num_classes = num_classes
        self.knowledge_gt_mix_ratio = knowledge_gt_mix_ratio
        self.confidence_weighted_alignment = confidence_weighted_alignment
        self.vit = ViTBackbone(
            model_name=vit_model_name,
            pretrained=pretrained,
            output_dim=768,
            checkpoint_path=vit_checkpoint,
        )
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
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        patch_tokens, cls_token = self.vit(x)

        coarse_logits = self.coarse_classifier(cls_token)
        coarse_probs = torch.softmax(coarse_logits, dim=1)
        knowledge_probs = coarse_probs
        if self.training and labels is not None and self.knowledge_gt_mix_ratio > 0:
            gt_probs = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).to(
                device=coarse_probs.device,
                dtype=coarse_probs.dtype,
            )
            mix_ratio = self.knowledge_gt_mix_ratio
            knowledge_probs = (1.0 - mix_ratio) * coarse_probs + mix_ratio * gt_probs
        knowledge_vec = self.akg(knowledge_probs)
        if return_aux:
            alignment_weights = None
            if self.training and labels is not None and self.confidence_weighted_alignment:
                alignment_weights = coarse_probs.gather(1, labels[:, None]).squeeze(1).detach()
            Zv, Zk, alignment_loss = self.sam(
                patch_tokens,
                knowledge_vec,
                labels=labels,
                alignment_weights=alignment_weights,
                return_loss=True,
            )
        else:
            Zv, Zk = self.sam(patch_tokens, knowledge_vec)
            alignment_loss = patch_tokens.new_zeros(())
        if return_attention:
            guided_visual, kga_attention = self.kga(Zv, Zk, return_attention=True)
        else:
            guided_visual = self.kga(Zv, Zk)
            kga_attention = patch_tokens.new_zeros(())
        pooled_visual = guided_visual.mean(dim=1)
        fused = torch.cat([pooled_visual, cls_token], dim=1)
        logits = self.classifier(fused)
        if return_aux:
            return logits, {
                "alignment_loss": alignment_loss,
                "coarse_logits": coarse_logits,
                "knowledge_probs": knowledge_probs,
                "alignment_weights": alignment_weights
                if alignment_weights is not None
                else patch_tokens.new_zeros(()),
                "kga_attention": kga_attention,
            }
        return logits
