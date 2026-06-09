#!/bin/bash
# Gadi login 节点：配置 ImageNet 缓存到 gs14 scratch（source 本文件即可）
#
#   source scripts/setup_gadi_imagenet.sh
#   python scripts/download_imagenet.py --splits validation --verify

export IMAGENET_HF_HOME=/scratch/gs14/sl9144/huggingface
export HF_HOME="${IMAGENET_HF_HOME}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TMPDIR=/scratch/gs14/sl9144/tmp

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TMPDIR}"

echo "[setup_gadi_imagenet] HF_HOME=${HF_HOME}"
echo "[setup_gadi_imagenet] HF_HUB_CACHE=${HF_HUB_CACHE}"
echo "[setup_gadi_imagenet] HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "[setup_gadi_imagenet] TMPDIR=${TMPDIR}"
