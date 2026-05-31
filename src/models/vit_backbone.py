from __future__ import annotations

from pathlib import Path

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
        checkpoint_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        import timm

        resolved_checkpoint = self._resolve_checkpoint_path(checkpoint_path)
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained and resolved_checkpoint is None,
            num_classes=0,
        )
        if resolved_checkpoint is not None:
            self._load_local_checkpoint(resolved_checkpoint)
        embed_dim = int(getattr(self.model, "embed_dim", output_dim))
        self.output_proj = nn.Identity() if embed_dim == output_dim else nn.Linear(embed_dim, output_dim)
        self.output_dim = output_dim
        self._freeze_bottom_blocks(trainable_blocks=trainable_blocks)

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path: str | Path | None) -> Path | None:
        """Resolve a local timm checkpoint file or directory."""
        if checkpoint_path is None:
            return None

        path = Path(checkpoint_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"ViT checkpoint path does not exist: {path}")

        if path.is_file():
            return path.resolve()

        candidates = [
            path / "pytorch_model.bin",
            path / "model.safetensors",
            path / "checkpoint.pth",
            path / "model.pth",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        raise FileNotFoundError(
            f"No supported ViT checkpoint file found in {path}. "
            "Expected pytorch_model.bin, model.safetensors, checkpoint.pth, or model.pth."
        )

    def _load_local_checkpoint(self, checkpoint_path: Path) -> None:
        """Load a local timm-style checkpoint and ignore classification-head weights."""
        if checkpoint_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(checkpoint_path))
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(state_dict, dict):
            for key in ("state_dict", "model"):
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key.removeprefix("module.")
            if clean_key.startswith("head."):
                continue
            cleaned_state_dict[clean_key] = value

        missing, unexpected = self.model.load_state_dict(cleaned_state_dict, strict=False)
        filtered_missing = [key for key in missing if not key.startswith("head.")]
        filtered_unexpected = [key for key in unexpected if not key.startswith("head.")]
        if filtered_missing or filtered_unexpected:
            print(
                "Loaded ViT checkpoint with non-head key mismatches: "
                f"missing={filtered_missing[:8]} unexpected={filtered_unexpected[:8]}"
            )
        print(f"Loaded ViT checkpoint: {checkpoint_path}")

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
