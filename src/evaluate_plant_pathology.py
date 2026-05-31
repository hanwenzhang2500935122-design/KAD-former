from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.kad_former import KADFormerLite
from train import PureViTClassifier


PLANTVILLAGE_CLASS_NAMES = ["scab", "black_rot", "rust", "healthy"]
PLANT_PATHOLOGY_TO_PV = {
    "scab": 0,
    "rust": 2,
    "healthy": 3,
}


def build_eval_transform() -> transforms.Compose:
    """Build the deterministic evaluation transform."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path) -> None:
    """Load a training checkpoint into the model."""
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


def remap_targets(imagefolder_targets: list[int], classes: list[str]) -> list[int]:
    """Map ImageFolder scab/rust/healthy targets to PlantVillage 4-class ids."""
    mapped: list[int] = []
    for target in imagefolder_targets:
        class_name = classes[target]
        if class_name not in PLANT_PATHOLOGY_TO_PV:
            raise ValueError(
                f"Unexpected class `{class_name}`. Expected only {sorted(PLANT_PATHOLOGY_TO_PV)}."
            )
        mapped.append(PLANT_PATHOLOGY_TO_PV[class_name])
    return mapped


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: datasets.ImageFolder,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Run strict 4-output evaluation."""
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    class_lookup = torch.tensor(
        [PLANT_PATHOLOGY_TO_PV[class_name] for class_name in dataset.classes],
        dtype=torch.long,
    )

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="evaluating"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            targets = class_lookup.index_select(0, labels).tolist()
            all_preds.extend(preds)
            all_targets.extend(targets)

    return all_preds, all_targets


def print_metrics(preds: list[int], targets: list[int]) -> None:
    """Print strict metrics and a confusion matrix."""
    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, labels=[0, 2, 3], average="macro", zero_division=0)
    weighted_f1 = f1_score(targets, preds, labels=[0, 2, 3], average="weighted", zero_division=0)
    black_rot_rate = float(np.mean(np.asarray(preds) == 1))

    print("\nStrict 4-output evaluation")
    print(f"accuracy: {acc:.4f}")
    print(f"macro_f1_target_classes: {macro_f1:.4f}")
    print(f"weighted_f1_target_classes: {weighted_f1:.4f}")
    print(f"black_rot_prediction_rate: {black_rot_rate:.4f}")
    print(
        classification_report(
            targets,
            preds,
            labels=[0, 1, 2, 3],
            target_names=PLANTVILLAGE_CLASS_NAMES,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(targets, preds, labels=[0, 1, 2, 3])
    print("confusion_matrix rows=true cols=pred labels=scab,black_rot,rust,healthy")
    print(matrix)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a PlantVillage-trained KAD-Former on Plant Pathology scab/rust/healthy."
    )
    parser.add_argument("--data-root", type=str, default="data/plant_pathology_3class")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_kad_former.pt")
    parser.add_argument("--baseline", action="store_true", help="Evaluate the pure ViT baseline.")
    parser.add_argument("--embeddings-path", type=str, default="data/knowledge_graph/node_embeddings.pt")
    parser.add_argument("--vit-model", type=str, default="vit_small_patch16_224")
    parser.add_argument("--vit-checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = datasets.ImageFolder(args.data_root, transform=build_eval_transform())
    expected_classes = set(PLANT_PATHOLOGY_TO_PV)
    if set(dataset.classes) != expected_classes:
        raise ValueError(f"Expected classes {sorted(expected_classes)}, got {dataset.classes}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if args.baseline:
        model = PureViTClassifier(
            num_classes=4,
            vit_model_name=args.vit_model,
            pretrained=args.vit_checkpoint is None,
            vit_checkpoint=args.vit_checkpoint,
        )
    else:
        model = KADFormerLite(
            num_classes=4,
            embeddings_path=args.embeddings_path,
            vit_model_name=args.vit_model,
            pretrained=args.vit_checkpoint is None,
            vit_checkpoint=args.vit_checkpoint,
        )
    load_checkpoint(model, args.checkpoint)
    model.to(device)

    preds, targets = evaluate(model, loader, dataset, device)
    print_metrics(preds, targets)


if __name__ == "__main__":
    main()
