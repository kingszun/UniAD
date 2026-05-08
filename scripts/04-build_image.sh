#!/usr/bin/env bash
set -euo pipefail

# Build UniAD image via docker compose (KAK-22).
# image tag default: cu118-py39-torch2.0.1 (controlled via .env / IMAGE_TAG).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ ! -f .env && -f .env.example ]]; then
    echo "WARN: .env not found. copying .env.example -> .env (review before next runs)"
    cp .env.example .env
fi

docker compose build "$@"
echo "done. image:"
docker images "$(grep '^IMAGE_REPO=' .env | cut -d= -f2 || echo uniad)" --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}'
