from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Callable

from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive


CLASS_TO_LABEL: dict[str, int] = {
    "Apple___Apple_scab": 0,
    "Apple___Black_rot": 1,
    "Apple___Cedar_apple_rust": 2,
    "Apple___healthy": 3,
}

PLANTVILLAGE_URL = (
    "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"
)


def _default_data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "plantvillage"


def _find_image_root(root: Path) -> Path | None:
    """Find a directory containing the PlantVillage class folders."""
    candidates = [
        root,
        root / "color",
        root / "PlantVillage-Dataset-master" / "raw" / "color",
        root / "PlantVillage-Dataset-master" / "raw" / "segmented",
        root / "PlantVillage-Dataset-master" / "raw" / "grayscale",
    ]
    for candidate in candidates:
        if candidate.exists() and all((candidate / cls).is_dir() for cls in CLASS_TO_LABEL):
            return candidate

    for candidate in root.rglob("*"):
        if candidate.is_dir() and all((candidate / cls).is_dir() for cls in CLASS_TO_LABEL):
            return candidate
    return None


def download_plantvillage(root: Path) -> None:
    """Download and extract PlantVillage with torchvision's download helper."""
    root.mkdir(parents=True, exist_ok=True)
    archive_name = "PlantVillage-Dataset-master.zip"
    download_and_extract_archive(
        url=PLANTVILLAGE_URL,
        download_root=str(root),
        extract_root=str(root),
        filename=archive_name,
    )


def ensure_plantvillage(root: Path, download: bool = True) -> Path:
    """Return the image-folder root, downloading the dataset if needed."""
    image_root = _find_image_root(root)
    if image_root is not None:
        return image_root

    if not download:
        raise FileNotFoundError(
            f"PlantVillage apple folders were not found under {root}. "
            "Place the dataset there or call get_dataloaders(download=True)."
        )

    download_plantvillage(root)
    image_root = _find_image_root(root)
    if image_root is None:
        raise FileNotFoundError(
            f"Downloaded PlantVillage, but could not find class folders under {root}."
        )
    return image_root


def build_transforms(train: bool) -> transforms.Compose:
    """Build ImageNet-normalized transforms for train or eval."""
    steps: list[Callable] = [transforms.Resize((224, 224))]
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(steps)


class ApplePlantVillage(Dataset[tuple[torch.Tensor, int]]):
    """Filtered PlantVillage apple subset with four fixed labels."""

    def __init__(
        self,
        image_root: str | Path,
        indices: list[int] | None = None,
        transform: Callable | None = None,
    ) -> None:
        self.image_root = Path(image_root)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for class_name, label in CLASS_TO_LABEL.items():
            class_dir = self.image_root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing PlantVillage class folder: {class_dir}")
            for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                self.samples.extend((path, label) for path in class_dir.glob(pattern))

        self.samples.sort(key=lambda item: str(item[0]))
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def targets(self) -> list[int]:
        """Return labels for distribution reporting and stratified splits."""
        return [label for _, label in self.samples]


def _split_indices(targets: list[int], seed: int) -> tuple[list[int], list[int], list[int]]:
    all_indices = list(range(len(targets)))
    train_idx, tmp_idx, train_y, tmp_y = train_test_split(
        all_indices,
        targets,
        train_size=0.70,
        random_state=seed,
        stratify=targets,
    )
    val_idx, test_idx = train_test_split(
        tmp_idx,
        test_size=0.50,
        random_state=seed,
        stratify=tmp_y,
    )
    return list(train_idx), list(val_idx), list(test_idx)


def describe_dataset(name: str, dataset: ApplePlantVillage) -> None:
    """Print sample count and class distribution."""
    counts = Counter(dataset.targets)
    distribution = {class_name: counts[label] for class_name, label in CLASS_TO_LABEL.items()}
    print(f"{name}: {len(dataset)} samples | {distribution}")


def get_dataloaders(
    batch_size: int = 16,
    num_workers: int = 4,
    data_root: str | Path | None = None,
    seed: int = 42,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return train, validation, and test dataloaders for the apple subset."""
    root = Path(data_root) if data_root is not None else _default_data_root()
    image_root = ensure_plantvillage(root, download=download)

    base_dataset = ApplePlantVillage(image_root)
    if len(base_dataset) == 0:
        raise RuntimeError(f"No images found in {image_root}")

    train_idx, val_idx, test_idx = _split_indices(base_dataset.targets, seed=seed)
    train_set = ApplePlantVillage(image_root, indices=train_idx, transform=build_transforms(True))
    val_set = ApplePlantVillage(image_root, indices=val_idx, transform=build_transforms(False))
    test_set = ApplePlantVillage(image_root, indices=test_idx, transform=build_transforms(False))

    describe_dataset("train", train_set)
    describe_dataset("val", val_set)
    describe_dataset("test", test_set)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PlantVillage apple dataloaders.")
    parser.add_argument("--data-root", type=str, default=str(_default_data_root()))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
        download=not args.no_download,
    )


if __name__ == "__main__":
    main()
