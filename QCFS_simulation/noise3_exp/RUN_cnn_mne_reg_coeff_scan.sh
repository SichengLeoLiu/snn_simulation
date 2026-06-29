#!/usr/bin/env bash
# MNIST CNN2：mne_l2 reg_coeff 扫描 + weight_decay 基线 + rate_uniform 噪声注入
# 默认 seeds=40..44（5 seed mean±std），与 strict-seed 实验一致
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

export CNN_EPOCHS="${CNN_EPOCHS:-100}"
export CNN_BATCH="${CNN_BATCH:-128}"
export CNN_NUM_WORKERS="${CNN_NUM_WORKERS:-8}"

echo "[INFO] QCFS_simulation root: ${ROOT}"
echo "[INFO] CNN_EPOCHS=${CNN_EPOCHS} CNN_BATCH=${CNN_BATCH} CNN_NUM_WORKERS=${CNN_NUM_WORKERS}"
python -c 'import torch; print("[INFO] torch:", torch.__version__, "cuda:", torch.cuda.is_available())'

python -u "${SCRIPT_DIR}/run_cnn_mne_reg_coeff_scan_noise_sweep_rate_uniform_L16_T16.py" "$@"

echo "[DONE] results: ${ROOT}/noise3_exp/cnn_mne_reg_coeff_scan_noise_sweep_rate_uniform_L16_T16/"
