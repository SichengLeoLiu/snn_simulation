#!/bin/bash
# Gadi login 节点：配置 ImageNet 缓存到 gs14 scratch（source 本文件即可）
#
#   source scripts/setup_gadi_imagenet.sh
#   python scripts/download_imagenet.py --splits validation --verify

export IMAGENET_HF_HOME=/scratch/gs14/sl9144/huggingface
export HF_HOME="${IMAGENET_HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

echo "[setup_gadi_imagenet] HF_HOME=${HF_HOME}"
echo "[setup_gadi_imagenet] HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
