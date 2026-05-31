from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import get_dataloaders_by_name
from models.kad_former import KADFormerLite
from models.vit_backbone import ViTBackbone
from utils import AverageMeter, compute_metrics, set_seed


class PureViTClassifier(nn.Module):
    """Pure ViT baseline that classifies from the CLS token only."""

    def __init__(
        self,
        num_classes: int = 4,
        vit_model_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
        vit_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.vit = ViTBackbone(
            model_name=vit_model_name,
            pretrained=pretrained,
            output_dim=768,
            checkpoint_path=vit_checkpoint,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.2),
            nn.Linear(768, num_classes),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        _, cls_token = self.vit(x)
        return self.classifier(cls_token)


def default_embeddings_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "knowledge_graph" / "node_embeddings.pt"


def run_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    alignment_loss_weight: float = 0.1,
    coarse_loss_weight: float = 0.2,
) -> tuple[float, float, float, float]:
    """Train or evaluate one epoch and return loss plus metrics."""
    is_train = optimizer is not None
    model.train(is_train)
    loss_meter = AverageMeter("loss")
    all_preds: list[int] = []
    all_labels: list[int] = []

    progress = tqdm(loader, desc="train" if is_train else "eval", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            try:
                outputs = model(images, labels if is_train else None, return_aux=is_train)
            except TypeError:
                outputs = model(images, labels if is_train else None)

            aux_losses: dict[str, torch.Tensor] = {}
            if isinstance(outputs, tuple):
                logits, aux_losses = outputs
            else:
                logits = outputs

            cls_loss = criterion(logits, labels)
            alignment_loss = aux_losses.get("alignment_loss", logits.new_zeros(()))
            coarse_logits = aux_losses.get("coarse_logits")
            coarse_loss = (
                criterion(coarse_logits, labels)
                if coarse_logits is not None
                else logits.new_zeros(())
            )
            loss = cls_loss + alignment_loss_weight * alignment_loss + coarse_loss_weight * coarse_loss
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
        progress.set_postfix(loss=f"{loss_meter.avg:.4f}")

    acc, f1_macro, f1_weighted = compute_metrics(all_preds, all_labels)
    return loss_meter.avg, acc, f1_macro, f1_weighted


def append_log(log_path: Path, row: dict[str, float | int]) -> None:
    """Append one epoch of metrics to a CSV file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()
    with open(log_path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_model(args: argparse.Namespace) -> nn.Module:
    """Create either the baseline or the KAD-Former model."""
    if args.baseline:
        return PureViTClassifier(
            num_classes=4,
            vit_model_name=args.vit_model,
            pretrained=not args.no_pretrained,
            vit_checkpoint=args.vit_checkpoint,
        )
    return KADFormerLite(
        num_classes=4,
        embeddings_path=args.embeddings_path,
        vit_model_name=args.vit_model,
        pretrained=not args.no_pretrained,
        vit_checkpoint=args.vit_checkpoint,
        knowledge_gt_mix_ratio=args.knowledge_gt_mix_ratio,
        confidence_weighted_alignment=not args.no_confidence_weighted_alignment,
        disable_akg=args.disable_akg,
        disable_sam=args.disable_sam,
        disable_kga=args.disable_kga,
        ablation_knowledge_tokens=args.ablation_knowledge_tokens,
    )


def build_run_name(args: argparse.Namespace) -> str:
    """Build a checkpoint/log stem that records the ablated modules."""
    if args.baseline:
        return "vit_baseline"

    disabled_modules = []
    if args.disable_akg:
        disabled_modules.append("no_akg")
    if args.disable_sam:
        disabled_modules.append("no_sam")
    if args.disable_kga:
        disabled_modules.append("no_kga")

    if disabled_modules:
        return "kad_former_" + "_".join(disabled_modules)
    return "kad_former"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train KAD-Former mini demo.")
    parser.add_argument("--baseline", action="store_true", help="Train pure ViT baseline.")
    parser.add_argument(
        "--dataset",
        choices=("plantvillage", "plant_pathology"),
        default="plantvillage",
        help="Dataset to train on.",
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--embeddings-path", type=str, default=str(default_embeddings_path()))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vit-model", type=str, default="vit_small_patch16_224")
    parser.add_argument(
        "--vit-checkpoint",
        type=str,
        default=None,
        help="Local timm checkpoint file or directory for the ViT backbone.",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--alignment-loss-weight", type=float, default=0.1)
    parser.add_argument("--coarse-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--knowledge-gt-mix-ratio",
        type=float,
        default=0.5,
        help=(
            "During KAD training, mix this fraction of GT one-hot knowledge with coarse "
            "probability knowledge. Evaluation and inference always use coarse probabilities only."
        ),
    )
    parser.add_argument(
        "--no-confidence-weighted-alignment",
        action="store_true",
        help=(
            "Disable confidence-weighted SAM alignment. By default, SAM alignment "
            "uses coarse_probs[GT] as a detached sample weight during training."
        ),
    )
    parser.add_argument("--disable-akg", action="store_true", help="Ablation: replace AKG with learnable class knowledge tokens.")
    parser.add_argument("--disable-sam", action="store_true", help="Ablation: bypass SAM attention/alignment and only use its input projections.")
    parser.add_argument("--disable-kga", action="store_true", help="Ablation: bypass KGA and classify from SAM visual features plus ViT CLS.")
    parser.add_argument(
        "--ablation-knowledge-tokens",
        type=int,
        default=22,
        help="Number of learnable class knowledge tokens used when --disable-akg is enabled.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders_by_name(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
        seed=args.seed,
        download=not args.no_download,
    )

    model = build_model(args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = build_run_name(args)
    checkpoint_dir = Path(args.checkpoint_dir)
    log_path = Path(args.log_dir) / f"{run_name}.csv"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1_macro, train_f1_weighted = run_one_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            alignment_loss_weight=args.alignment_loss_weight,
            coarse_loss_weight=args.coarse_loss_weight,
        )
        val_loss, val_acc, val_f1_macro, val_f1_weighted = run_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            alignment_loss_weight=args.alignment_loss_weight,
            coarse_loss_weight=0.0,
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1_macro": train_f1_macro,
            "train_f1_weighted": train_f1_weighted,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1_macro": val_f1_macro,
            "val_f1_weighted": val_f1_weighted,
        }
        append_log(log_path, row)

        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1_macro:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = checkpoint_dir / f"best_{run_name}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"saved best checkpoint: {checkpoint_path} val_acc={val_acc:.4f}")


if __name__ == "__main__":
    main()
