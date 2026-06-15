#!/usr/bin/env bash
# MNIST fc3 strict-seed：mne_l2+wd rate_uniform 噪声扫描 + 四路 mean±std 折线图
#
# 训练 checkpoint 与 normal 版共用；若已跑过 normal mne_l2+wd，本脚本主要做 rate_uniform 测试。
#
# 用法：
#   bash noise3_exp/RUN_fc3_strict_seed_mne_l2_wd_rate_uniform.sh
#   bash noise3_exp/RUN_fc3_strict_seed_mne_l2_wd_rate_uniform.sh --plot-only --copy-important
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

python "${SCRIPT_DIR}/run_fc3_strict_seed_mne_l2_wd_noise_sweep_rate_uniform_L16_T16.py" "$@"

echo "[DONE] plots: ${ROOT}/noise3_exp/ablation_mne_l2_vs_weight_decay_l16_fc3_h4_h8_h16_h32_h64_h128/strict_seed_train_rate_uniform_L16_T16/"
