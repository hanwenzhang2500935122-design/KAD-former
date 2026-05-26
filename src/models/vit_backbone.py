from __future__ import annotations

import torch
from torch import nn


class ViTBackbone(nn.Module):
    """
    ViT backbone that returns patch tokens and a CLS token.

    Input: x with shape (batch, 3, 224, 224).
    Output: patch_tokens (batch, 196, 768), cls_token (batch, 768).
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
        output_dim: int = 768,
        trainable_blocks: int = 2,
    ) -> None:
        super().__init__()
        import timm

        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        embed_dim = int(getattr(self.model, "embed_dim", output_dim))
        self.output_proj = nn.Identity() if embed_dim == output_dim else nn.Linear(embed_dim, output_dim)
        self.output_dim = output_dim
        self._freeze_bottom_blocks(trainable_blocks=trainable_blocks)

    def _freeze_bottom_blocks(self, trainable_blocks: int) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        blocks = getattr(self.model, "blocks", None)
        if blocks is not None and trainable_blocks > 0:
            for block in blocks[-trainable_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

        for name in ("norm", "fc_norm"):
            module = getattr(self.model, name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = True

        for parameter in self.output_proj.parameters():
            parameter.requires_grad = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model.forward_features(x)
        if isinstance(features, tuple):
            features = features[0]
        if features.ndim != 3:
            raise RuntimeError(
                "ViTBackbone expected token features with shape (B, N, C). "
                f"Got {tuple(features.shape)} from timm model."
            )

        cls_token = features[:, 0]
        patch_tokens = features[:, 1:]
        patch_tokens = self.output_proj(patch_tokens)
        cls_token = self.output_proj(cls_token)
        return patch_tokens, cls_token
