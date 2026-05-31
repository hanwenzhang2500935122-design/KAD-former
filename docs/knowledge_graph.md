# Knowledge Graph

This project uses an apple disease knowledge graph to provide disease-aware semantic tokens for KAD-Former.

## Files

```text
data/knowledge_graph/apple_kg.json
data/knowledge_graph/node_embeddings.pt
```

Only `apple_kg.json` is tracked by Git. The generated `node_embeddings.pt` file is ignored because it is a local artifact.

## Entity Types

The graph contains several types of entities:

- `fruit_tree`
- `disease`
- `pathogen`
- `symptom`
- `symptom_attribute`
- `plant_part`
- `stage`
- `key_visual_feature`

The `key_visual_feature` nodes are expert-style discriminative visual concepts, such as olive velvet scab lesions and orange-yellow rust pustules.

## Relation Types

Important relation types include:

- `species_disease`
- `disease_cause`
- `disease_symptom`
- `symptom_location`
- `symptom_stage`
- `symptom_has_color`
- `symptom_has_texture`
- `symptom_has_shape`
- `differentiated_by`
- `feature_evidence`
- `confused_with`

The `differentiated_by` relation links each disease to discriminative visual features. The `feature_evidence` relation links key visual features to supporting symptoms, colors, textures, shapes, and locations.

## Embedding Pipeline

The KG embedding pipeline is implemented in:

```text
src/kg_builder.py
```

It performs the following steps:

```text
KG JSON
  -> node descriptions
  -> BERT text embeddings
  -> relation-aware GNN propagation
  -> disease-specific knowledge token lists
  -> node_embeddings.pt
```

Run:

```powershell
D:\myenv\Scripts\python.exe src\kg_builder.py `
  --model-name .\src\models\bert-base-chinese
```

The output file contains:

```text
node_embeddings
idx_to_id
idx_to_type
disease_to_idx
class_knowledge_indices
class_knowledge_ids
```

## AKG Usage

AKG loads the generated embeddings and returns knowledge tokens for the current image.

During inference:

```text
knowledge_probs = coarse_probs
knowledge_tokens = AKG(knowledge_probs)
```

During training, optional GT mixing can be used:

```text
knowledge_probs =
  (1 - knowledge_gt_mix_ratio) * coarse_probs
  + knowledge_gt_mix_ratio * GT_one_hot
```

This reduces early training instability while keeping inference logic close to training logic.

## No AKG Ablation

The `--disable-akg` ablation does not feed zero tokens. It replaces KG-derived knowledge with learnable class knowledge tokens:

```text
learnable_knowledge_embeddings: (num_classes, tokens, 256)
```

This keeps SAM and KGA structurally comparable while removing expert KG semantics.
