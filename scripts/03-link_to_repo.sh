#!/usr/bin/env bash
set -euo pipefail

# Symlink repo data/ckpts paths to the host external mount layout.
# Both data and ckpts are gitignored, so symlinks are not tracked.

HOST_DATA_ROOT="${HOST_DATA_ROOT:-/mnt/e/datasets}"
NUSCENES_DIR="${NUSCENES_DIR:-${HOST_DATA_ROOT}/nuscenes}"
UNIAD_ROOT="${UNIAD_ROOT:-${HOST_DATA_ROOT}/uniad}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

link() {
    local target="$1"
    local link_path="$2"
    if [[ -L "${link_path}" ]]; then
        rm "${link_path}"
    elif [[ -e "${link_path}" ]]; then
        echo "ERROR: ${link_path} exists and is not a symlink. aborting." >&2
        exit 1
    fi
    ln -s "${target}" "${link_path}"
    echo "link ${link_path} -> ${target}"
}

mkdir -p "${REPO_ROOT}/data"
link "${NUSCENES_DIR}"        "${REPO_ROOT}/data/nuscenes"
link "${UNIAD_ROOT}/infos"    "${REPO_ROOT}/data/infos"
link "${UNIAD_ROOT}/others"   "${REPO_ROOT}/data/others"
link "${UNIAD_ROOT}/ckpts"    "${REPO_ROOT}/ckpts"

echo "done. expected layout per docs/DATA_PREP.md:"
ls -la "${REPO_ROOT}/data" "${REPO_ROOT}/ckpts"
