#!/bin/bash
# Download CIFAR-10-C and CIFAR-100-C onto a node that HAS network
# (Gadi login or copyq). GPU volta jobs cannot reach Zenodo.
#
# Usage:
#   CIFAR_C_ROOT=/scratch/gs14/sl9144/datasets bash download_cifar_c.sh
set -euo pipefail

DEST="${CIFAR_C_ROOT:-/scratch/gs14/sl9144/datasets}"
export CIFAR_C_ROOT="${DEST}"
mkdir -p "${DEST}"
cd "${DEST}"

C10_URL="${C10_URL:-https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1}"
C100_URL="${C100_URL:-https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1}"

CORRUPTIONS=(
  gaussian_noise shot_noise impulse_noise
  defocus_blur glass_blur motion_blur zoom_blur
  snow frost fog brightness
  contrast elastic_transform pixelate jpeg_compression
)

fetch() {
  local url="$1"
  local out="$2"
  if command -v wget >/dev/null 2>&1; then
    wget -c --tries=20 --timeout=60 -O "${out}" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --retry 20 --retry-delay 5 -C - -o "${out}" "${url}"
  else
    echo "ERROR: need wget or curl" >&2
    exit 1
  fi
}

complete_dir() {
  local dir="$1"
  [[ -f "${dir}/labels.npy" ]] || return 1
  local c
  for c in "${CORRUPTIONS[@]}"; do
    [[ -f "${dir}/${c}.npy" ]] || return 1
  done
  return 0
}

download_one() {
  local name="$1"
  local url="$2"
  local tar="${DEST}/${name}.tar"
  local dir="${DEST}/${name}"

  if complete_dir "${dir}"; then
    echo "[SKIP] ${dir} already complete"
    return 0
  fi

  echo "[FETCH] ${url}"
  fetch "${url}" "${tar}"
  echo "[UNTAR] ${tar}"
  tar -xf "${tar}" -C "${DEST}"
  if [[ ! -d "${dir}" ]]; then
    echo "ERROR: expected ${dir} after untar" >&2
    ls -la "${DEST}"
    exit 1
  fi
  if ! complete_dir "${dir}"; then
    echo "ERROR: ${dir} missing labels or a standard corruption npy" >&2
    ls -la "${dir}"
    exit 1
  fi
  rm -f "${tar}"
  echo "[OK] ${dir}"
}

echo "CIFAR_C_ROOT=${DEST}"
download_one "CIFAR-10-C" "${C10_URL}"
download_one "CIFAR-100-C" "${C100_URL}"

python3 - <<'PY'
import numpy as np
from pathlib import Path
import os
root = Path(os.environ.get("CIFAR_C_ROOT", "/scratch/gs14/sl9144/datasets"))
for name in ("CIFAR-10-C", "CIFAR-100-C"):
    d = root / name
    labels = np.load(d / "labels.npy", mmap_mode="r")
    noise = np.load(d / "gaussian_noise.npy", mmap_mode="r")
    print(f"{name}: labels={labels.shape} {labels.dtype} gaussian_noise={noise.shape} {noise.dtype}")
    assert labels.shape[0] == 50000, labels.shape
    assert noise.shape == (50000, 32, 32, 3), noise.shape
print("[VERIFY] shapes ok")
PY

echo "=== CIFAR-C ready under ${DEST} ==="
ls -ld "${DEST}/CIFAR-10-C" "${DEST}/CIFAR-100-C"
