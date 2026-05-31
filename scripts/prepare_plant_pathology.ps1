$ErrorActionPreference = "Stop"

$Python = "D:\myenv\Scripts\python.exe"

& $Python src\prepare_plant_pathology.py `
  --source-root data `
  --output-root data\plant_pathology_3class `
  --mode hardlink
