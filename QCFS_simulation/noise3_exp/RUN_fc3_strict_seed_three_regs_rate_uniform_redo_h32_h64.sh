#!/usr/bin/env bash
# 重做 fc3 h32/h64 rate_uniform 三路多 seed 噪声实验 + 重画图（无紫线）
#
# 用法：
#   bash noise3_exp/RUN_fc3_strict_seed_three_regs_rate_uniform_redo_h32_h64.sh
#   bash noise3_exp/RUN_fc3_strict_seed_three_regs_rate_uniform_redo_h32_h64.sh test-only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

MODE="${1:-full}"
COMMON=(
  --h-list 32 64
  --replot
  --copy-important
  --plot-h-list 4 8 16 32 64 128
  --font-size 14
  --legend-font-size 12
)

if [[ "${MODE}" == "test-only" ]]; then
  echo "[INFO] 仅重跑 h32/h64 噪声扫描（保留 checkpoint）"
  python "${SCRIPT_DIR}/run_fc3_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16.py" \
    "${COMMON[@]}" --force-test
elif [[ "${MODE}" == "full" ]]; then
  echo "[INFO] 重训 + 重测 h32/h64（旧 checkpoint/CSV 会先备份到 strict_seed_train_rate_uniform_L16_T16_backups/）"
  echo "[WARN] 汇总 raw/agg CSV 仍会 upsert 更新；如需完全隔离请手动复制 OUT 目录后再跑"
  python "${SCRIPT_DIR}/run_fc3_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16.py" \
    "${COMMON[@]}" --retrain --force-test
else
  echo "用法: $0 [full|test-only]" >&2
  exit 1
fi

echo "[DONE] plots: ${ROOT}/noise3_exp/.../strict_seed_train_rate_uniform_L16_T16/"
echo "[DONE] important: ${ROOT}/../important results/"
echo "[DONE] derivative: ${ROOT}/../derivative results/"
