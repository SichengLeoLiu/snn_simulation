"""ImageNet HuggingFace 缓存路径与按 split 下载（避免 load_dataset 误拉全量 train）。"""
from __future__ import annotations

import os
from pathlib import Path

# Gadi: project gs14, user sl9144（lquota 显示 scratch 约 1TB 可用）
GADI_SCRATCH_IMAGENET = Path("/scratch/gs14/sl9144/huggingface")
GADI_SCRATCH_ROOT = Path("/scratch/gs14/sl9144")
IMAGENET_REPO_ID = "imagenet-1k"


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


def list_split_parquet_files(repo_id: str, split: str) -> list[str]:
    """列出 Hub 上某一 split 的全部 parquet 文件名。"""
    from huggingface_hub import list_repo_files

    prefix = f"{split}-"
    files = list_repo_files(repo_id, repo_type="dataset")
    matches = [
        f
        for f in files
        if f.endswith(".parquet") and f.rsplit("/", 1)[-1].startswith(prefix)
    ]
    return sorted(matches)


def hf_parquet_urls(repo_id: str, split: str) -> list[str]:
    files = list_split_parquet_files(repo_id, split)
    return [f"hf://datasets/{repo_id}/{name}" for name in files]


def download_split_parquet_files(
    repo_id: str,
    split: str,
    verbose: bool = True,
) -> list[str]:
    """
    仅下载指定 split 的 parquet 到 HF hub 缓存。
    不使用 load_dataset('imagenet-1k')，因其会误拉其它 split（HF issue #6793）。
    """
    from huggingface_hub import hf_hub_download

    files = list_split_parquet_files(repo_id, split)
    if not files:
        raise FileNotFoundError(
            f"未在 {repo_id!r} 找到 split={split!r} 的 parquet 文件"
        )

    if verbose:
        print(
            f"[download] split={split!r} parquet_files={len(files)}",
            flush=True,
        )

    local_paths: list[str] = []
    for idx, remote_file in enumerate(files, 1):
        if verbose:
            print(f"  [{idx}/{len(files)}] {remote_file}", flush=True)
        local_paths.append(
            hf_hub_download(
                repo_id=repo_id,
                filename=remote_file,
                repo_type="dataset",
            )
        )
    return local_paths


def load_imagenet_split(
    split: str,
    repo_id: str = IMAGENET_REPO_ID,
    cache_dir: str | Path | None = None,
):
    """从已缓存/将按需下载的 parquet 加载单个 split。"""
    from datasets import load_dataset

    urls = hf_parquet_urls(repo_id, split)
    if not urls:
        raise FileNotFoundError(f"split={split!r} 无 parquet 文件")
    return load_dataset(
        "parquet",
        data_files={split: urls},
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
    )


def load_imagenet_dataset(
    repo_id: str = IMAGENET_REPO_ID,
    cache_dir: str | Path | None = None,
    splits: tuple[str, ...] = ("train", "validation"),
):
    """
    加载 ImageNet train/validation。
    仅引用各 split 的 parquet；未缓存的文件会在访问该 split 时下载。
    """
    from datasets import load_dataset

    data_files = {}
    for split in splits:
        urls = hf_parquet_urls(repo_id, split)
        if not urls:
            raise FileNotFoundError(f"split={split!r} 无 parquet 文件")
        data_files[split] = urls

    return load_dataset(
        "parquet",
        data_files=data_files,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
