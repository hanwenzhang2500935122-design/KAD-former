from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm import tqdm


DISEASE_ORDER = ["apple_scab", "black_rot", "cedar_rust", "healthy"]


def default_json_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "knowledge_graph" / "apple_kg.json"


def default_output_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "knowledge_graph" / "node_embeddings.pt"


def load_kg_json(json_path: str) -> dict[str, Any]:
    """Read the knowledge graph JSON file."""
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _flatten_entities(kg: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    entities = kg.get("entities", {})
    flattened: list[tuple[str, str, dict[str, Any]]] = []

    if isinstance(entities, list):
        for item in entities:
            node_id = item["id"]
            node_type = item.get("type") or item.get("class") or "entity"
            flattened.append((node_id, node_type, item))
        return flattened

    for node_type, items in entities.items():
        for item in items:
            node_id = item["id"]
            flattened.append((node_id, node_type, item))
    return flattened


def build_node_index(
    kg: dict[str, Any],
) -> tuple[dict[str, int], list[str], list[str], list[str]]:
    """
    Assign contiguous integer indices to graph nodes.

    Returns (id_to_idx, idx_to_id, idx_to_type, idx_to_description).
    """
    node_descriptions = kg.get("node_descriptions", {})
    id_to_idx: dict[str, int] = {}
    idx_to_id: list[str] = []
    idx_to_type: list[str] = []
    idx_to_description: list[str] = []

    for node_id, node_type, item in _flatten_entities(kg):
        if node_id in id_to_idx:
            continue
        id_to_idx[node_id] = len(idx_to_id)
        idx_to_id.append(node_id)
        idx_to_type.append(str(node_type))
        description = (
            item.get("description")
            or node_descriptions.get(node_id)
            or item.get("desc_short")
            or item.get("label")
            or node_id
        )
        idx_to_description.append(str(description))

    return id_to_idx, idx_to_id, idx_to_type, idx_to_description


def build_edges(kg: dict[str, Any], id_to_idx: dict[str, int]) -> dict[str, list[tuple[int, int]]]:
    """Group directed edges by relation type."""
    edges_by_relation: dict[str, list[tuple[int, int]]] = {}
    for triple in kg.get("triples", []):
        head = triple["head"]
        tail = triple["tail"]
        relation = triple["relation"]
        if head not in id_to_idx or tail not in id_to_idx:
            raise KeyError(f"Triple references an unknown node: {triple}")
        edges_by_relation.setdefault(relation, []).append((id_to_idx[head], id_to_idx[tail]))
    return edges_by_relation


def encode_descriptions_with_bert(
    descriptions: list[str],
    model_name: str = "bert-base-chinese",
    device: str = "cuda",
    batch_size: int = 16,
    force_download: bool = False,
    fallback_random: bool = False,
) -> torch.Tensor:
    """
    Encode node descriptions with Chinese BERT and return [CLS] embeddings.

    Input shape: N descriptions. Output shape: (N, 768).
    """
    from transformers import AutoModel, AutoTokenizer

    resolved_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model_source = _resolve_model_source(model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source, force_download=force_download)
        model = AutoModel.from_pretrained(model_source, force_download=force_download).to(resolved_device)
    except OSError as exc:
        if not fallback_random:
            raise OSError(
                f"Failed to load `{model_name}`. This usually means the HuggingFace cache or mirror "
                "download is incomplete. Try `--force-download`, pass a local directory with "
                "`--model-name`, or use `--fallback-random` only for a smoke test."
            ) from exc
        print(
            f"Warning: failed to load `{model_name}` ({exc}). "
            "Using random 768-d text embeddings; this is only suitable for pipeline testing."
        )
        generator = torch.Generator().manual_seed(42)
        return torch.randn(len(descriptions), 768, generator=generator)
    model.eval()

    embeddings: list[torch.Tensor] = []
    for start in tqdm(range(0, len(descriptions), batch_size), desc="Encoding KG nodes"):
        batch = descriptions[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu()
        embeddings.append(cls_embeddings)

    return torch.cat(embeddings, dim=0)


def _resolve_model_source(model_name: str) -> str:
    """Resolve a HuggingFace model id or a local model directory."""
    path_like = any(token in model_name for token in ("\\", "/", ":")) or model_name.startswith(".")
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    if path_like:
        raise FileNotFoundError(
            f"Local model path does not exist: {candidate}. "
            "Download it first, then pass the existing directory to --model-name."
        )
    return model_name


def build_hetero_graph(
    num_nodes: int,
    edges_by_relation: dict[str, list[tuple[int, int]]],
    bert_embeddings: torch.Tensor,
):
    """Build a PyG HeteroData graph with one node type and relation-specific edges."""
    from torch_geometric.data import HeteroData

    data = HeteroData()
    data["entity"].x = bert_embeddings.float()
    data["entity"].num_nodes = num_nodes

    for relation, edges in edges_by_relation.items():
        if not edges:
            continue
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data["entity", relation, "entity"].edge_index = edge_index
        reverse_relation = f"rev_{relation}"
        data["entity", reverse_relation, "entity"].edge_index = edge_index.flip(0)
    return data


class _HeteroSAGE(nn.Module):
    """Small relation-aware SAGE encoder for precomputing KG node embeddings."""

    def __init__(self, metadata: tuple[list[str], list[tuple[str, str, str]]], hidden_dim: int, num_layers: int):
        super().__init__()
        from torch_geometric.nn import HeteroConv, Linear, SAGEConv

        self.input_proj = Linear(-1, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            convs = {
                edge_type: SAGEConv((-1, -1), hidden_dim)
                for edge_type in metadata[1]
            }
            self.convs.append(HeteroConv(convs, aggr="sum"))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))

    def forward(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[Any, torch.Tensor]) -> torch.Tensor:
        x = self.input_proj(x_dict["entity"])
        x_dict = {"entity": torch.relu(x)}
        for conv, norm in zip(self.convs, self.norms):
            out_dict = conv(x_dict, edge_index_dict)
            out = torch.relu(norm(out_dict["entity"]))
            x_dict = {"entity": out + x_dict["entity"]}
        return x_dict["entity"]


def propagate_gnn(
    data,
    hidden_dim: int = 256,
    num_layers: int = 2,
) -> torch.Tensor:
    """
    Propagate relation-aware messages with HeteroConv + SAGEConv.

    Input dim: 768. Output dim: hidden_dim.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    model = _HeteroSAGE(data.metadata(), hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    model.eval()
    with torch.no_grad():
        node_embeddings = model(data.x_dict, data.edge_index_dict).detach().cpu()
    return node_embeddings


def main(
    json_path: str | None = None,
    output_path: str | None = None,
    model_name: str = "bert-base-chinese",
    force_download: bool = False,
    fallback_random: bool = False,
) -> None:
    """
    Run the full KG precomputation pipeline and save node embeddings.

    Saved keys: node_embeddings, idx_to_id, idx_to_type, disease_to_idx.
    """
    json_file = Path(json_path) if json_path is not None else default_json_path()
    output_file = Path(output_path) if output_path is not None else default_output_path()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    kg = load_kg_json(str(json_file))
    id_to_idx, idx_to_id, idx_to_type, descriptions = build_node_index(kg)
    edges_by_relation = build_edges(kg, id_to_idx)
    bert_embeddings = encode_descriptions_with_bert(
        descriptions,
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        force_download=force_download,
        fallback_random=fallback_random,
    )
    data = build_hetero_graph(len(idx_to_id), edges_by_relation, bert_embeddings)
    node_embeddings = propagate_gnn(data, hidden_dim=256, num_layers=2)
    disease_to_idx = {disease_id: id_to_idx[disease_id] for disease_id in DISEASE_ORDER}

    torch.save(
        {
            "node_embeddings": node_embeddings,
            "idx_to_id": idx_to_id,
            "idx_to_type": idx_to_type,
            "disease_to_idx": disease_to_idx,
        },
        output_file,
    )
    print(f"Saved KG embeddings: {output_file}")
    print(f"nodes={len(idx_to_id)} relations={len(edges_by_relation)} shape={tuple(node_embeddings.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build KAD-Former KG node embeddings.")
    parser.add_argument("--json-path", type=str, default=str(default_json_path()))
    parser.add_argument("--output-path", type=str, default=str(default_output_path()))
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-chinese",
        help="HuggingFace model id or local directory for the Chinese BERT encoder.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore cached files and re-download the BERT tokenizer/model.",
    )
    parser.add_argument(
        "--fallback-random",
        action="store_true",
        help="Use random 768-d text embeddings if BERT loading fails. For smoke tests only.",
    )
    args = parser.parse_args()
    main(
        args.json_path,
        args.output_path,
        model_name=args.model_name,
        force_download=args.force_download,
        fallback_random=args.fallback_random,
    )
