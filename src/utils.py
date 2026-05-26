from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def set_seed(seed: int = 42) -> None:
    """Set common random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(preds: Iterable[int], labels: Iterable[int]) -> tuple[float, float, float]:
    """Return accuracy, macro F1, and weighted F1."""
    pred_array = np.asarray(list(preds))
    label_array = np.asarray(list(labels))
    acc = accuracy_score(label_array, pred_array)
    f1_macro = f1_score(label_array, pred_array, average="macro", zero_division=0)
    f1_weighted = f1_score(label_array, pred_array, average="weighted", zero_division=0)
    return float(acc), float(f1_macro), float(f1_weighted)


@dataclass
class AverageMeter:
    """Track a running average."""

    name: str = "value"
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def reset(self) -> None:
        """Reset all accumulated values."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Add a value observed n times."""
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)
