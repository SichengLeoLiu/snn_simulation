#!/usr/bin/env bash
# CIFAR-100 VGG16：mne_l2 vs weight_decay 单 seed 噪声注入（rate_uniform, L=16, T=16）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# 可选：指定 CIFAR 数据目录
export CIFAR_ROOT="${CIFAR_ROOT:-${HOME}/datasets}"

echo "[INFO] QCFS_simulation root: ${ROOT}"
echo "[INFO] CIFAR_ROOT: ${CIFAR_ROOT}"
python -c "import torch; print('[INFO] torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

python "${SCRIPT_DIR}/run_cifar100_vgg16_strict_seed_noise_sweep_rate_uniform_L16_T16.py"

echo "[DONE] results: ${ROOT}/noise3_exp/cifar100_vgg16_strict_seed_noise_sweep_rate_uniform_L16_T16/"
