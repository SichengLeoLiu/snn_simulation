#!/bin/bash
# Gadi login 节点即可（CPU 大约几分钟；有 GPU 更快）。
#   cd /home/595/sl9144/codes/snn_simulation && git pull
#   export CIFAR_ROOT=/home/595/sl9144/datasets
#   bash QCFS_simulation/noise3_exp/RUN_visualize_post_if_noise_gadi.sh
set -euo pipefail

module purge
module use /g/data/dk92/apps/Modules/modulefiles
module load NCI-ai-ml/24.11

export CIFAR_ROOT="${CIFAR_ROOT:-/home/595/sl9144/datasets}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${HOME}/.cache/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO="$(cd "${ROOT}/.." && pwd)"
cd "${ROOT}"

OUT_DIR="${OUT_DIR:-${REPO}/important_results/post_if_noise_viz_seed42}"
mkdir -p "${OUT_DIR}"

find_ckpt() {
  local name="$1"
  local ds="$2"
  local candidates=(
    "${ROOT}/${ds}-checkpoints/${name}"
    "/scratch/gs14/sl9144/snn_ckpts/${ds}-checkpoints/${name}"
    "/home/595/sl9144/codes/snn_simulation/QCFS_simulation/${ds}-checkpoints/${name}"
  )
  local path
  for path in "${candidates[@]}"; do
    if [[ -f "${path}" ]]; then
      echo "${path}"
      return 0
    fi
  done
  return 1
}

run_one() {
  local dataset="$1"
  local tag="$2"
  local ckpt_name="$3"
  local ckpt
  if ! ckpt="$(find_ckpt "${ckpt_name}" "${dataset}")"; then
    echo "SKIP missing ckpt: ${dataset} ${tag} ${ckpt_name}"
    return 0
  fi
  echo "=== ${dataset} ${tag} ==="
  echo "CKPT=${ckpt}"
  python -u noise3_exp/visualize_post_if_noise_scale.py \
    --dataset "${dataset}" \
    --arch vgg16 \
    --ckpt "${ckpt}" \
    --tag "${tag}" \
    --L 16 \
    --T 16 \
    --mode rate_uniform \
    --batch-size 16 \
    --max-batches 8 \
    --sigmas 0 1 2 3 5 \
    --channel 0 \
    --vmin 0 \
    --vmax 2.5 \
    --out-dir "${OUT_DIR}"
}

echo "ROOT=${ROOT}"
echo "OUT_DIR=${OUT_DIR}"
echo "CIFAR_ROOT=${CIFAR_ROOT}"
echo "python=$(command -v python)"

# seed42，与 5-seed 曲线同一套 ckpt
run_one cifar10 l2wo "vgg16_L[16]_mneablate_cifar10_weight_decay_weights_only_rcnone_seed42_L16_trainT0.pth"
run_one cifar10 oldmne "vgg16_L[16]_mneablate_cifar10_old_detach_rc0p0001_seed42_L16_trainT0.pth"
run_one cifar10 l1 "vgg16_L[16]_mneablate_cifar10_l1_rc1em05_seed42_L16_trainT0.pth"
run_one cifar10 l1 "vgg16_L[16]_strict_seed42_schemeC_noout_l1_l16_vgg16_rc1em05.pth"

run_one cifar100 l2wo "vgg16_L[16]_mneablate_cifar100_weight_decay_weights_only_rcnone_seed42_L16_trainT0.pth"
run_one cifar100 oldmne "vgg16_L[16]_mneablate_cifar100_old_detach_rc0p0001_seed42_L16_trainT0.pth"
run_one cifar100 l1 "vgg16_L[16]_mneablate_cifar100_l1_rc1em05_seed42_L16_trainT0.pth"

echo "=== DONE ==="
ls -la "${OUT_DIR}"
