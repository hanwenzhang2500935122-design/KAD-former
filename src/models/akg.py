from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


DISEASE_ORDER = ["apple_scab", "black_rot", "cedar_rust", "healthy"]


class AKG(nn.Module):
    """
    Agricultural knowledge injection module backed by precomputed KG embeddings.

    Input: labels with shape (batch,). Output: Xk with shape (batch, 1, 256).
    """

    def __init__(self, embeddings_path: str | Path) -> None:
        super().__init__()
        path = Path(embeddings_path)
        if not path.exists():
            raise FileNotFoundError(
                f"KG embeddings not found: {path}. Run `python src/kg_builder.py` first."
            )
        payload = torch.load(path, map_location="cpu")
        node_embeddings = payload["node_embeddings"].float()
        disease_to_idx = payload["disease_to_idx"]
        lookup_indices = torch.tensor([disease_to_idx[disease_id] for disease_id in DISEASE_ORDER])
        lookup_embeddings = node_embeddings.index_select(0, lookup_indices)
        self.register_buffer("lookup_embeddings", lookup_embeddings)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(device=self.lookup_embeddings.device, dtype=torch.long)
        if torch.any((labels < 0) | (labels >= len(DISEASE_ORDER))):
            raise ValueError("AKG labels must be class indices in [0, 3].")
        return self.lookup_embeddings.index_select(0, labels).unsqueeze(1)
