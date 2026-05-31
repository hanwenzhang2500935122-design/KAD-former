from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ApplePlantVillage, build_transforms, ensure_plantvillage
from models.kad_former import KADFormerLite
from train import PureViTClassifier


CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
]
SUBSET_LABELS = [0, 2, 3]
SUBSET_NAMES = ["Apple___Apple_scab", "Apple___Cedar_apple_rust", "Apple___healthy"]


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path) -> None:
    """Load a training checkpoint and ignore regenerated AKG buffers."""
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


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    """Build KAD-Former or pure ViT baseline."""
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
        disable_akg=args.disable_akg,
        disable_sam=args.disable_sam,
        disable_kga=args.disable_kga,
        ablation_knowledge_tokens=args.ablation_knowledge_tokens,
    )


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    """Evaluate model and return predictions and labels."""
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="evaluating"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(targets.tolist())
    return preds, labels


def print_metrics(preds: list[int], labels: list[int]) -> None:
    """Print 4-class strict metrics and 3-class subset metrics."""
    pred_array = np.asarray(preds)
    label_array = np.asarray(labels)

    strict_acc = accuracy_score(label_array, pred_array)
    strict_macro_f1 = f1_score(label_array, pred_array, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    strict_weighted_f1 = f1_score(
        label_array,
        pred_array,
        labels=[0, 1, 2, 3],
        average="weighted",
        zero_division=0,
    )

    subset_mask = np.isin(label_array, SUBSET_LABELS)
    subset_labels = label_array[subset_mask]
    subset_preds = pred_array[subset_mask]
    subset_acc = accuracy_score(subset_labels, subset_preds)
    subset_macro_f1 = f1_score(
        subset_labels,
        subset_preds,
        labels=SUBSET_LABELS,
        average="macro",
        zero_division=0,
    )
    black_rot_on_subset_rate = float(np.mean(subset_preds == 1)) if subset_preds.size else 0.0

    print("\nPlantVillage 4-class strict evaluation")
    print(f"accuracy: {strict_acc:.4f}")
    print(f"macro_f1: {strict_macro_f1:.4f}")
    print(f"weighted_f1: {strict_weighted_f1:.4f}")
    print(
        classification_report(
            label_array,
            pred_array,
            labels=[0, 1, 2, 3],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )
    print("confusion_matrix rows=true cols=pred labels=scab,black_rot,rust,healthy")
    print(confusion_matrix(label_array, pred_array, labels=[0, 1, 2, 3]))

    print("\nPlantVillage 3-class subset evaluation (scab/rust/healthy only)")
    print(f"subset_size: {int(subset_mask.sum())}")
    print(f"subset_accuracy: {subset_acc:.4f}")
    print(f"subset_macro_f1: {subset_macro_f1:.4f}")
    print(f"black_rot_prediction_rate_on_subset: {black_rot_on_subset_rate:.4f}")
    print(
        classification_report(
            subset_labels,
            subset_preds,
            labels=SUBSET_LABELS,
            target_names=SUBSET_NAMES,
            zero_division=0,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on PlantVillage and report 3-class subset accuracy."
    )
    parser.add_argument("--data-root", type=str, default="data/plantvillage")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--embeddings-path", type=str, default="data/knowledge_graph/node_embeddings.pt")
    parser.add_argument("--vit-model", type=str, default="vit_small_patch16_224")
    parser.add_argument("--vit-checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--disable-akg", action="store_true", help="Evaluate a KAD ablation checkpoint trained without AKG.")
    parser.add_argument("--disable-sam", action="store_true", help="Evaluate a KAD ablation checkpoint trained without SAM.")
    parser.add_argument("--disable-kga", action="store_true", help="Evaluate a KAD ablation checkpoint trained without KGA.")
    parser.add_argument(
        "--ablation-knowledge-tokens",
        type=int,
        default=22,
        help="Number of learnable class knowledge tokens used by a --disable-akg checkpoint.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_root = ensure_plantvillage(Path(args.data_root), download=False)
    dataset = ApplePlantVillage(image_root, transform=build_transforms(False))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(args)
    load_checkpoint(model, args.checkpoint)
    model.to(device)

    preds, labels = evaluate(model, loader, device)
    print_metrics(preds, labels)


if __name__ == "__main__":
    main()
