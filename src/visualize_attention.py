from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.kad_former import KADFormerLite
from train import PureViTClassifier


CLASS_NAMES = ["scab", "black_rot", "rust", "healthy"]


def build_transform() -> transforms.Compose:
    """Build the same deterministic transform used for evaluation."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path) -> None:
    """Load model weights from a training checkpoint."""
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload.get("model_state_dict", payload)
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("akg.lookup_")
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"checkpoint loaded with missing={missing[:8]} unexpected={unexpected[:8]}")


class LastBlockAttentionCapture:
    """Capture ViT last-block attention from timm's attention dropout module."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.attention: torch.Tensor | None = None
        self.handle: torch.utils.hooks.RemovableHandle | None = None
        blocks = getattr(model.vit.model, "blocks")
        last_attn = blocks[-1].attn
        if hasattr(last_attn, "fused_attn"):
            last_attn.fused_attn = False
        self.handle = last_attn.attn_drop.register_forward_hook(self._hook)

    def _hook(self, _module: torch.nn.Module, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.attention = output.detach().cpu()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.remove()


def vit_attention_to_heatmap(attention: torch.Tensor, image_size: tuple[int, int]) -> np.ndarray:
    """Convert CLS-to-patch attention to a normalized image-sized heatmap."""
    if attention.ndim != 4:
        raise ValueError(f"Expected attention shape (B, heads, tokens, tokens), got {tuple(attention.shape)}")

    cls_to_patch = attention[0, :, 0, 1:].mean(dim=0)
    num_patches = cls_to_patch.numel()
    side = int(num_patches ** 0.5)
    if side * side != num_patches:
        raise ValueError(f"Cannot reshape {num_patches} patch attentions into a square grid.")

    heatmap = cls_to_patch.reshape(1, 1, side, side)
    heatmap = F.interpolate(heatmap, size=image_size[::-1], mode="bilinear", align_corners=False)
    heatmap_np = heatmap.squeeze().numpy()
    heatmap_np = heatmap_np - heatmap_np.min()
    heatmap_np = heatmap_np / max(float(heatmap_np.max()), 1e-8)
    return heatmap_np


def kga_attention_to_heatmap(attention: torch.Tensor, image_size: tuple[int, int]) -> np.ndarray:
    """
    Convert KGA patch-to-knowledge attention into a spatial heatmap.

    Shape is (branches, batch, heads, patches, knowledge_tokens). We use the
    maximum attention over knowledge tokens per patch, averaged over branches
    and heads. This highlights patches that strongly select a specific symptom
    or disease knowledge token.
    """
    if attention.ndim != 5:
        raise ValueError(
            "Expected KGA attention shape (branches, B, heads, patches, knowledge_tokens), "
            f"got {tuple(attention.shape)}"
        )

    patch_scores = attention[:, 0].mean(dim=1).max(dim=-1).values.mean(dim=0)
    num_patches = patch_scores.numel()
    side = int(num_patches ** 0.5)
    if side * side != num_patches:
        raise ValueError(f"Cannot reshape {num_patches} patch attentions into a square grid.")

    heatmap = patch_scores.reshape(1, 1, side, side)
    heatmap = F.interpolate(heatmap, size=image_size[::-1], mode="bilinear", align_corners=False)
    heatmap_np = heatmap.squeeze().numpy()
    heatmap_np = heatmap_np - heatmap_np.min()
    heatmap_np = heatmap_np / max(float(heatmap_np.max()), 1e-8)
    return heatmap_np


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay a jet heatmap on an RGB image."""
    image = image.convert("RGB")
    image_np = np.asarray(image).astype(np.float32) / 255.0
    colored = cm.get_cmap("jet")(heatmap)[..., :3].astype(np.float32)
    overlay = (1.0 - alpha) * image_np + alpha * colored
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    """Build KAD-Former or the pure ViT baseline."""
    if args.baseline:
        return PureViTClassifier(
            num_classes=4,
            vit_model_name=args.vit_model,
            pretrained=args.vit_checkpoint is None,
            vit_checkpoint=args.vit_checkpoint,
        )
    return KADFormerLite(
        num_classes=4,
        embeddings_path=args.embeddings_path,
        vit_model_name=args.vit_model,
        pretrained=args.vit_checkpoint is None,
        vit_checkpoint=args.vit_checkpoint,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a ViT attention heatmap for one image.")
    parser.add_argument("--image", type=str, required=True, help="Input image path.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_kad_former.pt")
    parser.add_argument("--baseline", action="store_true", help="Visualize the pure ViT baseline.")
    parser.add_argument("--embeddings-path", type=str, default="data/knowledge_graph/node_embeddings.pt")
    parser.add_argument("--vit-model", type=str, default="vit_small_patch16_224")
    parser.add_argument("--vit-checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="attention_heatmap.jpg")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument(
        "--source",
        choices=("vit", "kga"),
        default="vit",
        help="Use ViT CLS-to-patch attention or KGA patch-to-knowledge attention.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args)
    load_checkpoint(model, args.checkpoint)
    model.to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    resized_image = image.resize((224, 224))
    tensor = build_transform()(image).unsqueeze(0).to(device)

    if args.source == "vit":
        capture = LastBlockAttentionCapture(model)
        with torch.no_grad():
            logits = model(tensor)
        capture.close()
        if capture.attention is None:
            raise RuntimeError("Failed to capture ViT attention. The timm attention module may have changed.")
        heatmap = vit_attention_to_heatmap(capture.attention, resized_image.size)
    else:
        if args.baseline:
            raise ValueError("KGA attention is only available for KAD-Former, not the pure ViT baseline.")
        with torch.no_grad():
            logits, aux = model(tensor, return_aux=True, return_attention=True)
        heatmap = kga_attention_to_heatmap(aux["kga_attention"].detach().cpu(), resized_image.size)

    pred_idx = int(logits.argmax(dim=1).item())
    pred_prob = float(torch.softmax(logits, dim=1)[0, pred_idx].item())
    overlay = overlay_heatmap(resized_image, heatmap, alpha=args.alpha)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)
    print(f"prediction: {CLASS_NAMES[pred_idx]} ({pred_prob:.4f})")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
