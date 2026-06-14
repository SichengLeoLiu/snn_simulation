#!/bin/bash
# Gadi：将 MNIST 缓存到 gs14 scratch，供无网 GPU 节点使用
#
# login 节点（有网）:
#   source scripts/setup_gadi_mnist.sh
#   python scripts/download_mnist.py --verify
#
# GPU 节点（无网）:
#   source scripts/setup_gadi_mnist.sh
#   python noise3_exp/run_cnn_wd_strict_seed_L_T_acc.py

export MNIST_ROOT=/scratch/gs14/sl9144/datasets
mkdir -p "${MNIST_ROOT}"

echo "[setup_gadi_mnist] MNIST_ROOT=${MNIST_ROOT}"
echo "[setup_gadi_mnist] 若尚未下载，请在 login 节点运行: python scripts/download_mnist.py --verify"
