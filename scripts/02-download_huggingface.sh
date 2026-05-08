#!/usr/bin/env bash
set -euo pipefail

# Download UniAD v1 ckpts + info pkls + motion anchors (no auth).
# 출처: GitHub Releases — OpenDriveLab/UniAD v1.0 / v1.0.1, zhiqi-li/storage v1.0.

HOST_DATA_ROOT="${HOST_DATA_ROOT:-/mnt/e/datasets}"
UNIAD_ROOT="${UNIAD_ROOT:-${HOST_DATA_ROOT}/uniad}"
CKPTS_DIR="${UNIAD_ROOT}/ckpts"
INFOS_DIR="${UNIAD_ROOT}/infos"
OTHERS_DIR="${UNIAD_ROOT}/others"

declare -A DOWNLOADS=(
    ["${CKPTS_DIR}/bevformer_r101_dcn_24ep.pth"]="https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth"
    ["${CKPTS_DIR}/uniad_base_track_map.pth"]="https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/uniad_base_track_map.pth"
    ["${CKPTS_DIR}/uniad_base_e2e.pth"]="https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth"
    ["${INFOS_DIR}/nuscenes_infos_temporal_train.pkl"]="https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/nuscenes_infos_temporal_train.pkl"
    ["${INFOS_DIR}/nuscenes_infos_temporal_val.pkl"]="https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/nuscenes_infos_temporal_val.pkl"
    ["${OTHERS_DIR}/motion_anchor_infos_mode6.pkl"]="https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/motion_anchor_infos_mode6.pkl"
)

mkdir -p "${CKPTS_DIR}" "${INFOS_DIR}" "${OTHERS_DIR}"

download() {
    local target="$1"
    local url="$2"
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

for target in "${!DOWNLOADS[@]}"; do
    download "${target}" "${DOWNLOADS[$target]}"
done

echo "done. layout:"
echo "ckpts:"
ls -la "${CKPTS_DIR}"
echo "infos:"
ls -la "${INFOS_DIR}"
echo "others:"
ls -la "${OTHERS_DIR}"
