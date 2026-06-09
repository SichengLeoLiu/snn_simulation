"""ImageNet HuggingFace 缓存路径配置（Gadi gs14 scratch 默认）。"""
from __future__ import annotations

import os
from pathlib import Path

# Gadi: project gs14, user sl9144（lquota 显示 scratch 约 1TB 可用）
GADI_SCRATCH_IMAGENET = Path("/scratch/gs14/sl9144/huggingface")


def _on_gadi_gs14_scratch() -> bool:
    return GADI_SCRATCH_IMAGENET.parent.is_dir()


def resolve_imagenet_hf_home() -> Path:
    """解析 ImageNet 用的 HF_HOME（优先 scratch，避免占满 home 10GB）。"""
    explicit = os.environ.get("IMAGENET_HF_HOME", "").strip()
    if explicit:
        return Path(explicit).expanduser()

    if _on_gadi_gs14_scratch():
        return GADI_SCRATCH_IMAGENET

    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        return Path(hf_home).expanduser()

    return Path.home() / ".cache" / "huggingface"


def resolve_imagenet_datasets_cache() -> Path:
    datasets_cache = os.environ.get("HF_DATASETS_CACHE", "").strip()
    if datasets_cache:
        return Path(datasets_cache).expanduser()
    return resolve_imagenet_hf_home() / "datasets"


def configure_imagenet_hf_env(verbose: bool = True) -> Path:
    """
    为 ImageNet 下载/训练设置 HF_HOME 与 HF_DATASETS_CACHE。
    在 Gadi 上自动使用 /scratch/gs14/sl9144/huggingface。
    """
    hf_home = resolve_imagenet_hf_home()
    prev_hf = os.environ.get("HF_HOME")

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    hf_home.mkdir(parents=True, exist_ok=True)
    (hf_home / "datasets").mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[imagenet_hf_env] HF_HOME={hf_home}", flush=True)
        print(
            f"[imagenet_hf_env] HF_DATASETS_CACHE={os.environ['HF_DATASETS_CACHE']}",
            flush=True,
        )
        if prev_hf and Path(prev_hf).expanduser() != hf_home:
            print(
                f"[imagenet_hf_env] 注意: HF_HOME 已由 {prev_hf} 改为 {hf_home}",
                flush=True,
            )
    return hf_home
