#!/usr/bin/env bash
# FC3rev：mne_l2 reg_coeff 扫描 + rate_uniform 噪声注入
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

export FC3REV_EPOCHS="${FC3REV_EPOCHS:-50}"
export FC3REV_BATCH="${FC3REV_BATCH:-128}"
export FC3REV_NUM_WORKERS="${FC3REV_NUM_WORKERS:-4}"
export MNIST_ROOT="${MNIST_ROOT:-${HOME}/datasets}"

echo "[INFO] QCFS_simulation root: ${ROOT}"
echo "[INFO] python: $(command -v python)"
echo "[INFO] FC3REV_EPOCHS=${FC3REV_EPOCHS} FC3REV_BATCH=${FC3REV_BATCH} FC3REV_NUM_WORKERS=${FC3REV_NUM_WORKERS}"
echo "[INFO] MNIST_ROOT=${MNIST_ROOT}"

python -c 'import torch; print("[INFO] torch:", torch.__version__, "cuda:", torch.cuda.is_available())'

python -u "${SCRIPT_DIR}/run_fc3rev_mne_reg_coeff_scan_noise_sweep_rate_uniform_L16_T16.py" \
  --no-plot \
  "$@"

echo "[DONE] results under: ${ROOT}/../important_results/new_fc3/mne_reg_coeff_scan_acc01 (or --out-dir)"
