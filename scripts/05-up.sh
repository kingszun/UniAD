#!/usr/bin/env bash
set -euo pipefail

# Bring up UniAD container (detached). reuses image built by 04-build_image.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
docker compose up -d
docker compose ps
