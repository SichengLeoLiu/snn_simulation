#!/usr/bin/env bash
# CIFAR-100 VGG16：mne_l2 reg_coeff 扫描 + weight_decay 对比 + 噪声注入（方案 C）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

export CIFAR_ROOT="${CIFAR_ROOT:-${HOME}/datasets}"

echo "[INFO] QCFS_simulation root: ${ROOT}"
echo "[INFO] CIFAR_ROOT: ${CIFAR_ROOT}"
python -c 'import torch; print("[INFO] torch:", torch.__version__, "cuda:", torch.cuda.is_available())'

python "${SCRIPT_DIR}/run_cifar100_vgg16_mne_reg_coeff_scan_noise_sweep_rate_uniform_L16_T16.py"

echo "[DONE] results: ${ROOT}/noise3_exp/cifar100_vgg16_mne_reg_coeff_scan_noise_sweep_rate_uniform_L16_T16/"
