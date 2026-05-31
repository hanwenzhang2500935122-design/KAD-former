$ErrorActionPreference = "Stop"

$Python = "D:\myenv\Scripts\python.exe"

& $Python src\train.py `
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
