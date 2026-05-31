# KAD-Former Ablation Experiments

This project supports three module-level ablations for KAD-Former:

- `--disable-akg`: replaces AKG knowledge lookup with learnable class knowledge tokens.
- `--disable-sam`: bypasses SAM attention/alignment and only keeps input projections.
- `--disable-kga`: bypasses KGA and classifies from SAM visual features plus ViT CLS.

Use the same ablation flag during training and evaluation.

## Train On Plant Pathology

Full KAD:

```powershell
D:\myenv\Scripts\python.exe src\train.py `
  --dataset plant_pathology `
  --data-root data\plant_pathology_3class `
  --embeddings-path data\knowledge_graph\node_embeddings.pt `
  --vit-checkpoint models\vit\vit_small_patch16_224 `
  --checkpoint-dir checkpoints\ablation_full_kad `
  --log-dir logs\ablation_full_kad `
  --epochs 30 `
  --batch-size 32 `
  --lr 1e-4 `
  --weight-decay 1e-4 `
  --knowledge-gt-mix-ratio 0.25
```

No AKG:

```powershell
D:\myenv\Scripts\python.exe src\train.py `
  --dataset plant_pathology `
  --data-root data\plant_pathology_3class `
  --embeddings-path data\knowledge_graph\node_embeddings.pt `
  --vit-checkpoint models\vit\vit_small_patch16_224 `
  --checkpoint-dir checkpoints\ablation_no_akg `
  --log-dir logs\ablation_no_akg `
  --epochs 30 `
  --batch-size 32 `
  --lr 1e-4 `
  --weight-decay 1e-4 `
  --knowledge-gt-mix-ratio 0.25 `
  --disable-akg
```

No SAM:

```powershell
D:\myenv\Scripts\python.exe src\train.py `
  --dataset plant_pathology `
  --data-root data\plant_pathology_3class `
  --embeddings-path data\knowledge_graph\node_embeddings.pt `
  --vit-checkpoint models\vit\vit_small_patch16_224 `
  --checkpoint-dir checkpoints\ablation_no_sam `
  --log-dir logs\ablation_no_sam `
  --epochs 30 `
  --batch-size 32 `
  --lr 1e-4 `
  --weight-decay 1e-4 `
  --knowledge-gt-mix-ratio 0.25 `
  --disable-sam
```

No KGA:

```powershell
D:\myenv\Scripts\python.exe src\train.py `
  --dataset plant_pathology `
  --data-root data\plant_pathology_3class `
  --embeddings-path data\knowledge_graph\node_embeddings.pt `
  --vit-checkpoint models\vit\vit_small_patch16_224 `
  --checkpoint-dir checkpoints\ablation_no_kga `
  --log-dir logs\ablation_no_kga `
  --epochs 30 `
  --batch-size 32 `
  --lr 1e-4 `
  --weight-decay 1e-4 `
  --knowledge-gt-mix-ratio 0.25 `
  --disable-kga
```

## Evaluate On PlantVillage

Full KAD:

```powershell
D:\myenv\Scripts\python.exe src\evaluate_plantvillage.py `
  --data-root data\plantvillage `
  --checkpoint checkpoints\ablation_full_kad\best_kad_former.pt `
  --embeddings-path data\knowledge_graph\node_embeddings.pt `
  --vit-checkpoint models\vit\vit_small_patch16_224 `
  --batch-size 32
```

No AKG:

```powershell
D:\myenv\Scripts\python.exe src\evaluate_plantvillage.py `
  --data-root data\plantvillage `
  --checkpoint checkpoints\ablation_no_akg\best_kad_former_no_akg.pt `
  --embeddings-path data\knowledge_graph\node_embeddings.pt `
  --vit-checkpoint models\vit\vit_small_patch16_224 `
  --batch-size 32 `
  --disable-akg
```

No SAM:

```powershell
D:\myenv\Scripts\python.exe src\evaluate_plantvillage.py `
  --data-root data\plantvillage `
  --checkpoint checkpoints\ablation_no_sam\best_kad_former_no_sam.pt `
  --embeddings-path data\knowledge_graph\node_embeddings.pt `
  --vit-checkpoint models\vit\vit_small_patch16_224 `
  --batch-size 32 `
  --disable-sam
```

No KGA:

```powershell
D:\myenv\Scripts\python.exe src\evaluate_plantvillage.py `
  --data-root data\plantvillage `
  --checkpoint checkpoints\ablation_no_kga\best_kad_former_no_kga.pt `
  --embeddings-path data\knowledge_graph\node_embeddings.pt `
  --vit-checkpoint models\vit\vit_small_patch16_224 `
  --batch-size 32 `
  --disable-kga
```
