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


def build_class_knowledge_indices(
    kg: dict[str, Any],
    id_to_idx: dict[str, int],
    max_tokens: int | None = None,
) -> tuple[dict[str, list[int]], dict[str, list[str]]]:
    """
    Build disease-specific knowledge token lists.

    Each class receives the disease node plus its local KG neighborhood:
    disease -> pathogen/symptoms and symptom -> location/stage/attributes.
    Lists are padded to the same length by repeating the disease node so AKG
    can return a dense tensor of shape (classes, tokens, dim).
    """
    first_hop_relations = {"disease_cause", "disease_symptom", "differentiated_by"}
    second_hop_relations = {
        "symptom_location",
        "symptom_stage",
        "symptom_has_color",
        "symptom_has_texture",
        "symptom_has_shape",
        "feature_evidence",
    }

    adjacency: dict[str, list[tuple[str, str]]] = {}
    for triple in kg.get("triples", []):
        adjacency.setdefault(triple["head"], []).append((triple["relation"], triple["tail"]))

    raw_ids_by_disease: dict[str, list[str]] = {}
    for disease_id in DISEASE_ORDER:
        ordered_ids = [disease_id]
        seen = {disease_id}

        first_hop = [
            tail
            for relation, tail in adjacency.get(disease_id, [])
            if relation in first_hop_relations
        ]
        second_hop: list[str] = []
        for node_id in first_hop:
            if node_id not in seen:
                ordered_ids.append(node_id)
                seen.add(node_id)
            second_hop.extend(
                tail
                for relation, tail in adjacency.get(node_id, [])
                if relation in second_hop_relations
            )

        for node_id in second_hop:
            if node_id not in seen:
                ordered_ids.append(node_id)
                seen.add(node_id)

        ordered_ids = [node_id for node_id in ordered_ids if node_id in id_to_idx]
        raw_ids_by_disease[disease_id] = ordered_ids

    if max_tokens is None:
        target_len = max(len(ids) for ids in raw_ids_by_disease.values())
    else:
        target_len = max(1, max_tokens)

    ids_by_disease: dict[str, list[str]] = {}
    indices_by_disease: dict[str, list[int]] = {}
    for disease_id, node_ids in raw_ids_by_disease.items():
        clipped = node_ids[:target_len]
        padded = clipped + [disease_id] * max(0, target_len - len(clipped))
        ids_by_disease[disease_id] = padded
        indices_by_disease[disease_id] = [id_to_idx[node_id] for node_id in padded]

    return indices_by_disease, ids_by_disease


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
    max_knowledge_tokens: int | None = None,
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
    class_knowledge_indices, class_knowledge_ids = build_class_knowledge_indices(
        kg,
        id_to_idx,
        max_tokens=max_knowledge_tokens,
    )

    torch.save(
        {
            "node_embeddings": node_embeddings,
            "idx_to_id": idx_to_id,
            "idx_to_type": idx_to_type,
            "disease_to_idx": disease_to_idx,
            "class_knowledge_indices": class_knowledge_indices,
            "class_knowledge_ids": class_knowledge_ids,
        },
        output_file,
    )
    print(f"Saved KG embeddings: {output_file}")
    print(f"nodes={len(idx_to_id)} relations={len(edges_by_relation)} shape={tuple(node_embeddings.shape)}")
    print(
        "knowledge_tokens_per_class="
        f"{ {disease_id: len(indices) for disease_id, indices in class_knowledge_indices.items()} }"
    )


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
    parser.add_argument(
        "--max-knowledge-tokens",
        type=int,
        default=None,
        help="Optional cap for disease-specific knowledge tokens. Defaults to the largest disease subgraph.",
    )
    args = parser.parse_args()
    main(
        args.json_path,
        args.output_path,
        model_name=args.model_name,
        force_download=args.force_download,
        fallback_random=args.fallback_random,
        max_knowledge_tokens=args.max_knowledge_tokens,
    )
