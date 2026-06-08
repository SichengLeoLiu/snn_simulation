#!/usr/bin/env bash
# CIFAR-10 VGG16：wd vs mne_l2 vs mne_l2+wd (rc=1e-4, wd=1e-4) + 噪声注入
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

export CIFAR_ROOT="${CIFAR_ROOT:-${HOME}/datasets}"

python "${SCRIPT_DIR}/run_cifar10_vgg16_mne_l2_wd_combo_noise_sweep_rate_uniform_L16_T16.py"

echo "[DONE] results: ${ROOT}/noise3_exp/cifar10_vgg16_mne_l2_wd_combo_noise_sweep_rate_uniform_L16_T16/"
