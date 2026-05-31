$ErrorActionPreference = "Stop"

$Python = "D:\myenv\Scripts\python.exe"

& $Python src\train.py `
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
