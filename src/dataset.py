from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Callable

from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive


CLASS_TO_LABEL: dict[str, int] = {
    "Apple___Apple_scab": 0,
    "Apple___Black_rot": 1,
    "Apple___Cedar_apple_rust": 2,
    "Apple___healthy": 3,
}

PLANT_PATHOLOGY_CLASS_TO_LABEL: dict[str, int] = {
    "scab": 0,
    "rust": 2,
    "healthy": 3,
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


class RemappedImageFolder(Dataset[tuple[torch.Tensor, int]]):
    """ImageFolder wrapper that remaps class indices into the model label space."""

    def __init__(
        self,
        root: str | Path,
        class_to_label: dict[str, int],
        indices: list[int] | None = None,
        transform: Callable | None = None,
    ) -> None:
        self.dataset = ImageFolder(str(root), transform=transform)
        self.class_to_label = class_to_label
        missing = set(self.dataset.classes) - set(class_to_label)
        if missing:
            raise ValueError(f"Unexpected classes in {root}: {sorted(missing)}")
        self.indices = indices if indices is not None else list(range(len(self.dataset.samples)))
        self.remapped_targets = [
            class_to_label[self.dataset.classes[self.dataset.targets[index]]]
            for index in self.indices
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, _ = self.dataset[self.indices[index]]
        return image, self.remapped_targets[index]

    @property
    def targets(self) -> list[int]:
        """Return remapped labels."""
        return self.remapped_targets


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


def describe_dataset(name: str, dataset: Dataset) -> None:
    """Print sample count and class distribution."""
    targets = getattr(dataset, "targets")
    counts = Counter(targets)
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


def get_plant_pathology_dataloaders(
    batch_size: int = 16,
    num_workers: int = 4,
    data_root: str | Path = "data/plant_pathology_3class",
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return train, validation, and test dataloaders for Plant Pathology 3-class data."""
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Plant Pathology ImageFolder root not found: {root}")

    base_dataset = RemappedImageFolder(root, class_to_label=PLANT_PATHOLOGY_CLASS_TO_LABEL)
    train_idx, val_idx, test_idx = _split_indices(base_dataset.targets, seed=seed)
    train_set = RemappedImageFolder(
        root,
        class_to_label=PLANT_PATHOLOGY_CLASS_TO_LABEL,
        indices=train_idx,
        transform=build_transforms(True),
    )
    val_set = RemappedImageFolder(
        root,
        class_to_label=PLANT_PATHOLOGY_CLASS_TO_LABEL,
        indices=val_idx,
        transform=build_transforms(False),
    )
    test_set = RemappedImageFolder(
        root,
        class_to_label=PLANT_PATHOLOGY_CLASS_TO_LABEL,
        indices=test_idx,
        transform=build_transforms(False),
    )

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


def get_dataloaders_by_name(
    dataset_name: str,
    batch_size: int = 16,
    num_workers: int = 4,
    data_root: str | Path | None = None,
    seed: int = 42,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Dispatch dataloader creation by dataset name."""
    if dataset_name == "plantvillage":
        return get_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers,
            data_root=data_root,
            seed=seed,
            download=download,
        )
    if dataset_name == "plant_pathology":
        return get_plant_pathology_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers,
            data_root=data_root or "data/plant_pathology_3class",
            seed=seed,
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PlantVillage apple dataloaders.")
    parser.add_argument("--data-root", type=str, default=str(_default_data_root()))
    parser.add_argument("--dataset", choices=("plantvillage", "plant_pathology"), default="plantvillage")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    get_dataloaders_by_name(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
        download=not args.no_download,
    )


if __name__ == "__main__":
    main()
