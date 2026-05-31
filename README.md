# KAD-Former

A lightweight KAD-Former reproduction and extension demo for apple disease recognition. The project combines a ViT visual backbone with an apple disease knowledge graph, semantic alignment, and knowledge-guided attention for fine-grained plant disease classification.

This repository is intended as a research-oriented mini implementation, not an official reproduction of the original paper.

## Overview

KAD-Former follows this pipeline:

```text
Image -> ViT backbone -> coarse classifier -> AKG -> SAM -> KGA -> classifier
```

- **ViT backbone** extracts patch tokens and a CLS token from leaf images.
- **AKG** selects disease-related knowledge tokens from precomputed graph embeddings.
- **SAM** aligns visual patch features and knowledge features in a shared semantic space.
- **KGA** uses knowledge-guided cross-attention to enhance visual features.
- **Classifier** predicts apple disease classes from visual and knowledge-guided features.

The current demo supports four PlantVillage-compatible apple classes:

```text
Apple Scab
Black Rot
Cedar Apple Rust
Healthy
```

It also supports a 3-class Plant Pathology subset:

```text
scab
rust
healthy
```

## Features

- ViT baseline and KAD-Former training.
- Knowledge graph embedding generation with BERT + relation-aware GNN propagation.
- Soft AKG knowledge selection with optional GT/coarse probability mixing.
- Confidence-weighted SAM alignment loss.
- Three-branch KGA cross-attention.
- Plant Pathology subset preparation.
- PlantVillage and Plant Pathology evaluation scripts.
- Visual attention heatmap export.
- Module-level ablation switches for AKG, SAM, and KGA.

## Project Structure

```text
.
├── data/
│   └── knowledge_graph/
│       └── apple_kg.json
├── docs/
│   ├── ablation_experiments.md
│   ├── knowledge_graph.md
│   ├── kg_triples_report.md
│   ├── reproduction_plan.md
│   └── training_and_evaluation.md
├── scripts/
│   ├── eval_plantvillage.ps1
│   ├── prepare_plant_pathology.ps1
│   ├── run_ablation_no_akg.ps1
│   ├── run_ablation_no_kga.ps1
│   ├── run_ablation_no_sam.ps1
│   ├── train_baseline_pathology.ps1
│   └── train_kad_pathology.ps1
├── src/
│   ├── dataset.py
│   ├── evaluate_plant_pathology.py
│   ├── evaluate_plantvillage.py
│   ├── kg_builder.py
│   ├── prepare_plant_pathology.py
│   ├── train.py
│   ├── visualize_attention.py
│   └── models/
│       ├── akg.py
│       ├── kad_former.py
│       ├── kga.py
│       ├── sam.py
│       └── vit_backbone.py
├── requirements.txt
└── README.md
```

## Installation

Create and activate a Python environment, then install dependencies:

```powershell
pip install -r requirements.txt
```

The project was developed with Python 3.11 and PyTorch. A CUDA-enabled GPU is recommended.

## Data

Datasets are not included in this repository.

Expected local paths:

```text
data/plantvillage/
data/plant_pathology_3class/
```

Downloaded model weights and trained checkpoints are also excluded from Git:

```text
models/
checkpoints/
logs/
data/knowledge_graph/node_embeddings.pt
```

## Knowledge Graph Embeddings

Build the KG embeddings before KAD-Former training:

```powershell
D:\myenv\Scripts\python.exe src\kg_builder.py `
  --model-name .\src\models\bert-base-chinese
```

This generates:

```text
data/knowledge_graph/node_embeddings.pt
```

## Training

Train KAD-Former on the Plant Pathology 3-class subset:

```powershell
D:\myenv\Scripts\python.exe src\train.py `
  --dataset plant_pathology `
  --data-root data\plant_pathology_3class `
  --embeddings-path data\knowledge_graph\node_embeddings.pt `
  --vit-checkpoint models\vit\vit_small_patch16_224 `
  --checkpoint-dir checkpoints\plant_pathology_kad `
  --log-dir logs\plant_pathology_kad `
  --epochs 30 `
  --batch-size 32 `
  --lr 1e-4 `
  --weight-decay 1e-4 `
  --knowledge-gt-mix-ratio 0.25
```

Train the ViT baseline:

```powershell
D:\myenv\Scripts\python.exe src\train.py `
  --baseline `
  --dataset plant_pathology `
  --data-root data\plant_pathology_3class `
  --vit-checkpoint models\vit\vit_small_patch16_224 `
  --checkpoint-dir checkpoints\plant_pathology_baseline `
  --log-dir logs\plant_pathology_baseline `
  --epochs 30 `
  --batch-size 32 `
  --lr 1e-4 `
  --weight-decay 1e-4
```

## Evaluation

Evaluate a Plant Pathology-trained KAD model on PlantVillage:

```powershell
D:\myenv\Scripts\python.exe src\evaluate_plantvillage.py `
  --data-root data\plantvillage `
  --checkpoint checkpoints\plant_pathology_kad\best_kad_former.pt `
  --embeddings-path data\knowledge_graph\node_embeddings.pt `
  --vit-checkpoint models\vit\vit_small_patch16_224 `
  --batch-size 32
```

The script reports 4-class strict accuracy and 3-class subset accuracy.

## Attention Visualization

Export a visual attention heatmap from KAD:

```powershell
D:\myenv\Scripts\python.exe src\visualize_attention.py `
  --source vit `
  --image data\plant_pathology_3class\scab\Train_0.jpg `
  --checkpoint checkpoints\plant_pathology_kad\best_kad_former.pt `
  --embeddings-path data\knowledge_graph\node_embeddings.pt `
  --vit-checkpoint models\vit\vit_small_patch16_224 `
  --output logs\kad_visual_attention.jpg
```

Use `--source kga` for KGA patch-to-knowledge attention.

## Ablation Study

The following module ablations are supported:

```text
--disable-akg
--disable-sam
--disable-kga
```

See [docs/ablation_experiments.md](docs/ablation_experiments.md) for training and evaluation commands.

## Limitations

- This is a compact demo implementation, not a full official reproduction.
- KG entities and descriptions are draft expert-style knowledge and should be reviewed by domain experts.
- Results can be sensitive to dataset split, training seed, and cross-dataset domain shift.
- The PlantVillage dataset is relatively simple and may overestimate real-world robustness.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
