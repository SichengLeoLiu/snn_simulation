#!/usr/bin/env bash
# MNIST CNN2 多 seed weight_decay L×T 精度表（c2c4 / c4c8 / c8c16 / c16c32）
#
# 用法：
#   bash noise3_exp/RUN_cnn_wd_strict_seed_L_T_acc.sh
#   bash noise3_exp/RUN_cnn_wd_strict_seed_L_T_acc.sh --latex-only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

python "${SCRIPT_DIR}/run_cnn_wd_strict_seed_L_T_acc.py" "$@"

echo "[DONE] LaTeX table: ${ROOT}/noise3_exp/cnn_wd_strict_seed_normal_L_T_acc/cnn_wd_strict_seed_normal_L_T_acc_table.tex"
