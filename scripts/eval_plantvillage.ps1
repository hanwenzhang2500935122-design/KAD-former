$ErrorActionPreference = "Stop"

$Python = "D:\myenv\Scripts\python.exe"

& $Python src\evaluate_plantvillage.py `
  --data-root data\plantvillage `
  --checkpoint checkpoints\plant_pathology_kad\best_kad_former.pt `
  --embeddings-path data\knowledge_graph\node_embeddings.pt `
  --vit-checkpoint models\vit\vit_small_patch16_224 `
  --batch-size 32
