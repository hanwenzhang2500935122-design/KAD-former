# Training and Evaluation

This document collects the main commands used to train and evaluate the project.

## Environment

Install dependencies:

```powershell
pip install -r requirements.txt
```

The examples below assume this Python executable:

```text
D:\myenv\Scripts\python.exe
```

Replace it with your own environment path if needed.

## Data Layout

Expected local paths:

```text
data/plantvillage/
data/plant_pathology_3class/
models/vit/vit_small_patch16_224/
src/models/bert-base-chinese/
```

Datasets and downloaded model weights are not tracked by Git.

## Prepare Plant Pathology 3-Class Subset

If the raw Plant Pathology data is under `data/`, run:

```powershell
D:\myenv\Scripts\python.exe src\prepare_plant_pathology.py `
  --source-root data `
  --output-root data\plant_pathology_3class `
  --mode hardlink
```

The expected output structure is:

```text
data/plant_pathology_3class/
├── healthy/
├── rust/
└── scab/
```

## Build KG Embeddings

```powershell
D:\myenv\Scripts\python.exe src\kg_builder.py `
  --model-name .\src\models\bert-base-chinese
```

This creates:

```text
data/knowledge_graph/node_embeddings.pt
```

## Train ViT Baseline

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

Output checkpoint:

```text
checkpoints/plant_pathology_baseline/best_vit_baseline.pt
```

## Train KAD-Former

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
  --knowledge-gt-mix-ratio 0.25 `
  --alignment-loss-weight 0.1 `
  --coarse-loss-weight 0.2
```

Output checkpoint:

```text
checkpoints/plant_pathology_kad/best_kad_former.pt
```

## Important Training Options

```text
--knowledge-gt-mix-ratio
```

Controls how much GT one-hot knowledge is mixed into AKG selection during training. Inference always uses coarse probabilities.

```text
--alignment-loss-weight
```

Controls the strength of SAM visual-knowledge alignment loss.

```text
--coarse-loss-weight
```

Controls the auxiliary CE loss for the coarse classifier. This helps AKG receive more reliable coarse probabilities.

```text
--no-confidence-weighted-alignment
```

Disables confidence-weighted SAM alignment. By default, SAM alignment uses `coarse_probs[GT]` as detached sample weights.

## Evaluate On Plant Pathology

```powershell
D:\myenv\Scripts\python.exe src\evaluate_plant_pathology.py `
  --data-root data\plant_pathology_3class `
  --checkpoint checkpoints\plant_pathology_kad\best_kad_former.pt `
  --embeddings-path data\knowledge_graph\node_embeddings.pt `
  --vit-checkpoint models\vit\vit_small_patch16_224 `
  --batch-size 32
```

## Evaluate On PlantVillage

```powershell
D:\myenv\Scripts\python.exe src\evaluate_plantvillage.py `
  --data-root data\plantvillage `
  --checkpoint checkpoints\plant_pathology_kad\best_kad_former.pt `
  --embeddings-path data\knowledge_graph\node_embeddings.pt `
  --vit-checkpoint models\vit\vit_small_patch16_224 `
  --batch-size 32
```

For baseline evaluation, add:

```powershell
--baseline
```

and use the baseline checkpoint.

## Attention Visualization

Visual attention from the ViT backbone:

```powershell
D:\myenv\Scripts\python.exe src\visualize_attention.py `
  --source vit `
  --image data\plant_pathology_3class\scab\Train_0.jpg `
  --checkpoint checkpoints\plant_pathology_kad\best_kad_former.pt `
  --embeddings-path data\knowledge_graph\node_embeddings.pt `
  --vit-checkpoint models\vit\vit_small_patch16_224 `
  --output logs\kad_visual_attention.jpg
```

KGA patch-to-knowledge attention:

```powershell
D:\myenv\Scripts\python.exe src\visualize_attention.py `
  --source kga `
  --image data\plant_pathology_3class\scab\Train_0.jpg `
  --checkpoint checkpoints\plant_pathology_kad\best_kad_former.pt `
  --embeddings-path data\knowledge_graph\node_embeddings.pt `
  --vit-checkpoint models\vit\vit_small_patch16_224 `
  --output logs\kad_kga_attention.jpg
```
