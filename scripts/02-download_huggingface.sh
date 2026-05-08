#!/usr/bin/env bash
set -euo pipefail

# Download UniAD-specific artifacts from HuggingFace (no auth required).
# - ckpts (4 files): backbone + bevformer + stage1 + stage2
# - info pkls (2 files): off-the-shelf train/val info from upstream
# - motion anchors (1 file)

HOST_DATA_ROOT="${HOST_DATA_ROOT:-/mnt/e/datasets}"
UNIAD_ROOT="${UNIAD_ROOT:-${HOST_DATA_ROOT}/uniad}"
CKPTS_DIR="${UNIAD_ROOT}/ckpts"
INFOS_DIR="${UNIAD_ROOT}/infos"
OTHERS_DIR="${UNIAD_ROOT}/others"

HF_BASE="https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main"

CKPT_FILES=(
    "r101_dcn_fcos3d_pretrain.pth"
    "bevformer_r101_dcn_24ep.pth"
    "uniad_base_track_map.pth"
    "uniad_base_e2e.pth"
)

INFO_FILES=(
    "nuscenes_infos_temporal_train.pkl"
    "nuscenes_infos_temporal_val.pkl"
)

OTHER_FILES=(
    "motion_anchor_infos_mode6.pkl"
)

mkdir -p "${CKPTS_DIR}" "${INFOS_DIR}" "${OTHERS_DIR}"

download() {
    local url="$1"
    local target="$2"
    if [[ -f "${target}" ]]; then
        echo "skip (exists): ${target}"
        return 0
    fi
    echo "download ${url} -> ${target}"
    if command -v aria2c >/dev/null 2>&1; then
        aria2c --dir="$(dirname "${target}")" --out="$(basename "${target}")" \
               --split=8 --max-connection-per-server=8 --continue=true \
               --auto-file-renaming=false "${url}"
    else
        wget --continue -O "${target}" "${url}"
    fi
}

for f in "${CKPT_FILES[@]}"; do
    download "${HF_BASE}/ckpts/${f}" "${CKPTS_DIR}/${f}"
done

for f in "${INFO_FILES[@]}"; do
    download "${HF_BASE}/data/${f}" "${INFOS_DIR}/${f}"
done

for f in "${OTHER_FILES[@]}"; do
    download "${HF_BASE}/data/${f}" "${OTHERS_DIR}/${f}"
done

echo "done. layout:"
echo "ckpts:"
ls -la "${CKPTS_DIR}"
echo "infos:"
ls -la "${INFOS_DIR}"
echo "others:"
ls -la "${OTHERS_DIR}"
