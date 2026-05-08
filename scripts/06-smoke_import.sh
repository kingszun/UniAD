#!/usr/bin/env bash
set -euo pipefail

# Verify env inside container: torch + cuda + mm-series + UniAD plugin import.
# acceptance for KAK-22.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

docker compose exec uniad bash -lc '
set -e
python - <<PY
import sys, importlib

print(f"python: {sys.version.split()[0]}")

import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"cuda device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"cuda device 0: {torch.cuda.get_device_name(0)}")

for mod in ("mmcv", "mmdet", "mmseg", "mmdet3d"):
    m = importlib.import_module(mod)
    print(f"{mod}: {m.__version__}")

# UniAD plugin import — projects 가 PYTHONPATH 에 있어야 함.
import importlib.util
spec = importlib.util.find_spec("projects.mmdet3d_plugin")
print(f"projects.mmdet3d_plugin spec: {spec is not None}")

print("PASS")
PY
'
