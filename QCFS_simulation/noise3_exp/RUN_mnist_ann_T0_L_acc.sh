#!/usr/bin/env bash
# MNIST ANN (T=0)：FC3 / CNN2 各规模在 L=2,4,8,16,32 下的 test acc（5 seeds mean±std）
#
# 用法：
#   bash noise3_exp/RUN_mnist_ann_T0_L_acc.sh
#   bash noise3_exp/RUN_mnist_ann_T0_L_acc.sh --model-type fc
#   bash noise3_exp/RUN_mnist_ann_T0_L_acc.sh --model-type cnn --force-test
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

python "${SCRIPT_DIR}/run_mnist_ann_T0_L_acc.py" "$@"

echo "[DONE] results: ${ROOT}/noise3_exp/mnist_ann_T0_L_acc/"
