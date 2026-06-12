#!/usr/bin/env bash
# CIFAR-100 VGG16：mne_l2 + weight_decay 组合对比 + 噪声注入
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

export CIFAR_ROOT="${CIFAR_ROOT:-${HOME}/datasets}"

python "${SCRIPT_DIR}/run_cifar100_vgg16_mne_l2_wd_combo_noise_sweep_rate_uniform_L16_T16.py"

echo "[DONE] results: ${ROOT}/noise3_exp/cifar100_vgg16_mne_l2_wd_combo_noise_sweep_rate_uniform_L16_T16/"
