#!/usr/bin/env bash
set -euo pipefail

# Download nuScenes archives listed in $NUSCENES_URLS_FILE (default: scripts/nuscenes_urls.txt)
# and extract them into $NUSCENES_DIR. URLs are time-limited signed URLs from
# https://www.nuscenes.org/nuscenes#download (login required).
#
# usage:
#   1. cp scripts/nuscenes_urls_example.txt scripts/nuscenes_urls.txt
#   2. paste fresh URLs from nuScenes session into scripts/nuscenes_urls.txt
#   3. bash scripts/01-download_nuscenes.sh

HOST_DATA_ROOT="${HOST_DATA_ROOT:-/mnt/e/datasets}"
NUSCENES_DIR="${NUSCENES_DIR:-${HOST_DATA_ROOT}/nuscenes}"
DOWNLOAD_TMP="${DOWNLOAD_TMP:-${HOST_DATA_ROOT}/_tmp_nuscenes}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUSCENES_URLS_FILE="${NUSCENES_URLS_FILE:-${SCRIPT_DIR}/nuscenes_urls.txt}"

if [[ ! -f "${NUSCENES_URLS_FILE}" ]]; then
    cat >&2 <<EOF
ERROR: URL list not found at: ${NUSCENES_URLS_FILE}

create it from the example:
  cp ${SCRIPT_DIR}/nuscenes_urls_example.txt ${NUSCENES_URLS_FILE}

then paste fresh signed URLs from your nuScenes session
(https://www.nuscenes.org/nuscenes#download, login required).
required archives:
  - v1.0-mini.tgz
  - v1.0-trainval01_blobs.tgz ... v1.0-trainval10_blobs.tgz
  - v1.0-trainval_meta.tgz
  - can_bus.zip
  - nuScenes-map-expansion-v1.3.zip

URLs expire (typically 7 days). re-issue from nuScenes if download fails.
EOF
    exit 1
fi

mkdir -p "${NUSCENES_DIR}" "${DOWNLOAD_TMP}"

if command -v aria2c >/dev/null 2>&1; then
    aria2c \
        --dir="${DOWNLOAD_TMP}" \
        --input-file="${NUSCENES_URLS_FILE}" \
        --max-concurrent-downloads=4 \
        --split=8 \
        --max-connection-per-server=8 \
        --continue=true \
        --auto-file-renaming=false
else
    while IFS= read -r url; do
        [[ -z "${url}" || "${url}" =~ ^# ]] && continue
        wget --continue --directory-prefix="${DOWNLOAD_TMP}" "${url}"
    done < "${NUSCENES_URLS_FILE}"
fi

cd "${DOWNLOAD_TMP}"
shopt -s nullglob
for archive in *.tgz *.tar.gz; do
    echo "extracting ${archive} -> ${NUSCENES_DIR}/"
    tar -xzf "${archive}" -C "${NUSCENES_DIR}/"
done
for archive in *.zip; do
    echo "extracting ${archive} -> ${NUSCENES_DIR}/"
    unzip -o "${archive}" -d "${NUSCENES_DIR}/"
done
shopt -u nullglob

echo "done. layout:"
ls -la "${NUSCENES_DIR}"
echo "tmp archives are still in ${DOWNLOAD_TMP} — remove manually after verification."
