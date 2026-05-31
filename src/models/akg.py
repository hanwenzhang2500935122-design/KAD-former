from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


DISEASE_ORDER = ["apple_scab", "black_rot", "cedar_rust", "healthy"]


class AKG(nn.Module):
    """
    Agricultural knowledge injection module backed by precomputed KG embeddings.

    Inputs:
      labels with shape (batch,), or
      class probabilities with shape (batch, 4).
    Output: Xk with shape (batch, knowledge_tokens, 256).
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
        if "class_knowledge_indices" in payload:
            lookup_indices = torch.tensor(
                [payload["class_knowledge_indices"][disease_id] for disease_id in DISEASE_ORDER],
                dtype=torch.long,
            )
        else:
            lookup_indices = torch.tensor(
                [[disease_to_idx[disease_id]] for disease_id in DISEASE_ORDER],
                dtype=torch.long,
            )

        lookup_embeddings = node_embeddings.index_select(0, lookup_indices.flatten())
        lookup_embeddings = lookup_embeddings.reshape(*lookup_indices.shape, node_embeddings.size(-1))
        self.register_buffer("lookup_embeddings", lookup_embeddings, persistent=False)
        self.register_buffer("lookup_indices", lookup_indices, persistent=False)

    def forward(self, labels_or_probs: torch.Tensor) -> torch.Tensor:
        """
        Return disease knowledge vectors.

        Integer labels perform a hard lookup. Floating-point class distributions
        perform a soft weighted sum over all disease knowledge token sequences.
        """
        if labels_or_probs.ndim == 1:
            labels = labels_or_probs.to(device=self.lookup_embeddings.device, dtype=torch.long)
            return self.forward_hard(labels)

        if labels_or_probs.ndim != 2 or labels_or_probs.size(1) != len(DISEASE_ORDER):
            raise ValueError(
                "AKG expects labels with shape (batch,) or class probabilities with shape (batch, 4)."
            )

        probs = labels_or_probs.to(device=self.lookup_embeddings.device, dtype=self.lookup_embeddings.dtype)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("bc,ckd->bkd", probs, self.lookup_embeddings)

    def forward_hard(self, labels: torch.Tensor) -> torch.Tensor:
        """Return hard disease lookup vectors for integer class labels."""
        labels = labels.to(device=self.lookup_embeddings.device, dtype=torch.long)
        if torch.any((labels < 0) | (labels >= len(DISEASE_ORDER))):
            raise ValueError("AKG labels must be class indices in [0, 3].")
        return self.lookup_embeddings.index_select(0, labels)
