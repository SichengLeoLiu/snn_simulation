#!/usr/bin/env bash
# CIFAR-10/100：normal 模式下三路方法精度对比（复用已有 checkpoint，仅测试）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

export CIFAR_ROOT="${CIFAR_ROOT:-${HOME}/datasets}"

python "${SCRIPT_DIR}/run_cifar10_cifar100_normal_mode_acc_compare_L16_T16.py"

echo "[DONE] results: ${ROOT}/noise3_exp/cifar10_cifar100_vgg16_normal_mode_acc_compare_L16_T16/"
