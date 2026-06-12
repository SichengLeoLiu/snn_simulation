"""ImageNet HuggingFace 缓存路径与按 split 下载（避免 load_dataset 误拉全量 train）。"""
from __future__ import annotations

import os
import time
from pathlib import Path

# Gadi: project gs14, user sl9144（lquota 显示 scratch 约 1TB 可用）
GADI_SCRATCH_IMAGENET = Path("/scratch/gs14/sl9144/huggingface")
GADI_SCRATCH_ROOT = Path("/scratch/gs14/sl9144")
IMAGENET_REPO_ID = "imagenet-1k"
IMAGENET_ENV_VERSION = "2026-06-scratch-v3-offline"


def _on_gadi_gs14_scratch() -> bool:
    return GADI_SCRATCH_IMAGENET.parent.is_dir()


def resolve_imagenet_hf_home() -> Path:
    """解析 ImageNet 用的 HF_HOME（Gadi 上强制 scratch，忽略 home 里的 HF_HOME）。"""
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


def resolve_hub_cache_dir() -> Path:
    hub_cache = os.environ.get("HF_HUB_CACHE", "").strip()
    if hub_cache:
        return Path(hub_cache).expanduser()
    return resolve_imagenet_hf_home() / "hub"


def home_hf_cache_candidates() -> list[Path]:
    return [
        Path.home() / "datasets" / "huggingface",
        Path.home() / ".cache" / "huggingface",
    ]


def configure_imagenet_hf_env(verbose: bool = True) -> Path:
    """
    为 ImageNet 下载/训练设置全部 HuggingFace 缓存到 scratch。
    必须在本文件任何 huggingface_hub / datasets 导入之前调用。
    """
    hf_home = resolve_imagenet_hf_home()
    hub_cache = hf_home / "hub"
    datasets_cache = hf_home / "datasets"
    prev_hf = os.environ.get("HF_HOME")
    prev_hub = os.environ.get("HF_HUB_CACHE")

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)

    hf_home.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)

    if _on_gadi_gs14_scratch():
        tmp_dir = GADI_SCRATCH_ROOT / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(tmp_dir)

    if verbose:
        print(f"[imagenet_hf_env] version={IMAGENET_ENV_VERSION}", flush=True)
        print(f"[imagenet_hf_env] HF_HOME={hf_home}", flush=True)
        print(f"[imagenet_hf_env] HF_HUB_CACHE={hub_cache}", flush=True)
        print(f"[imagenet_hf_env] HF_DATASETS_CACHE={datasets_cache}", flush=True)
        if os.environ.get("TMPDIR"):
            print(f"[imagenet_hf_env] TMPDIR={os.environ['TMPDIR']}", flush=True)
        if prev_hf and Path(prev_hf).expanduser().resolve() != hf_home.resolve():
            print(
                f"[imagenet_hf_env] HF_HOME changed: {prev_hf} -> {hf_home}",
                flush=True,
            )
        if prev_hub and Path(prev_hub).expanduser().resolve() != hub_cache.resolve():
            print(
                f"[imagenet_hf_env] HF_HUB_CACHE changed: {prev_hub} -> {hub_cache}",
                flush=True,
            )
    return hf_home


def _repo_cache_dir_names(repo_id: str) -> list[str]:
    return [f"datasets--{repo_id.replace('/', '--')}"]


def _split_parquet_prefix(split: str) -> str:
    return f"{split}-"


def find_local_split_parquet_files(
    split: str,
    repo_id: str = IMAGENET_REPO_ID,
    hub_cache: Path | str | None = None,
) -> list[str]:
    """
    在 HF hub 缓存中查找已下载的 split parquet（无需联网）。
    Gadi GPU 计算节点无外网，训练时必须走此路径。
    """
    hub_cache = Path(hub_cache) if hub_cache is not None else resolve_hub_cache_dir()
    prefix = _split_parquet_prefix(split)
    grouped: dict[str, list[str]] = {}

    for repo_dir_name in _repo_cache_dir_names(repo_id):
        repo_dir = hub_cache / repo_dir_name
        if not repo_dir.is_dir():
            continue
        for parquet_path in repo_dir.rglob("*.parquet"):
            if not parquet_path.is_file():
                continue
            if not parquet_path.name.startswith(prefix):
                continue
            grouped.setdefault(str(parquet_path.parent.resolve()), []).append(
                str(parquet_path.resolve())
            )

    if not grouped:
        return []

    best_group = max(grouped.values(), key=len)
    return sorted(best_group, key=lambda p: Path(p).name)


def _offline_mode_requested() -> bool:
    return os.environ.get("IMAGENET_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def list_split_parquet_files(repo_id: str, split: str) -> list[str]:
    """列出 Hub 上某一 split 的全部 parquet 文件名（需联网）。"""
    from huggingface_hub import list_repo_files

    prefix = _split_parquet_prefix(split)
    files = list_repo_files(repo_id, repo_type="dataset")
    matches = [
        f
        for f in files
        if f.endswith(".parquet") and f.rsplit("/", 1)[-1].startswith(prefix)
    ]
    return sorted(matches)


def resolve_split_parquet_files(
    repo_id: str,
    split: str,
    verbose: bool = False,
) -> list[str]:
    """
    优先返回 scratch 上已有的 parquet；仅在本地缺失且未强制离线时才访问 Hub。
    """
    configure_imagenet_hf_env(verbose=False)
    hub_cache = resolve_hub_cache_dir()
    local_paths = find_local_split_parquet_files(
        split, repo_id=repo_id, hub_cache=hub_cache
    )
    if local_paths:
        if verbose:
            print(
                f"[imagenet_hf_env] offline cache hit split={split!r} "
                f"files={len(local_paths)} hub_cache={hub_cache}",
                flush=True,
            )
        return local_paths

    if _offline_mode_requested():
        raise FileNotFoundError(
            f"未在本地找到 split={split!r} 的 parquet（IMAGENET_OFFLINE=1）。\n"
            f"请在 Gadi login 节点运行:\n"
            f"  source scripts/setup_gadi_imagenet.sh\n"
            f"  python scripts/download_imagenet.py --splits {split} --verify\n"
            f"hub_cache={hub_cache}"
        )

    return download_split_parquet_files(repo_id, split, verbose=verbose)


def download_split_parquet_files(
    repo_id: str,
    split: str,
    verbose: bool = True,
    max_retries: int = 8,
    retry_wait_sec: float = 30.0,
) -> list[str]:
    """
    仅下载指定 split 的 parquet 到 HF hub 缓存（HF_HUB_CACHE）。
    已完成的文件会自动跳过；网络中断后重新运行本函数即可续传。
    """
    from huggingface_hub import hf_hub_download

    configure_imagenet_hf_env(verbose=False)
    hub_cache_dir = resolve_hub_cache_dir()
    hub_cache = str(hub_cache_dir)
    hf_home = resolve_imagenet_hf_home()

    local_paths = find_local_split_parquet_files(
        split, repo_id=repo_id, hub_cache=hub_cache_dir
    )
    if local_paths:
        if verbose:
            print(
                f"[download] split={split!r} 使用本地缓存 parquet_files={len(local_paths)}",
                flush=True,
            )
        return local_paths

    if _offline_mode_requested():
        raise FileNotFoundError(
            f"离线模式且 split={split!r} 无本地 parquet，hub_cache={hub_cache_dir}"
        )

    files = list_split_parquet_files(repo_id, split)
    if not files:
        raise FileNotFoundError(
            f"未在 {repo_id!r} 找到 split={split!r} 的 parquet 文件"
        )

    if verbose:
        print(
            f"[download] split={split!r} parquet_files={len(files)} "
            f"hub_cache={hub_cache}",
            flush=True,
        )
        print(
            "[download] 已下载的文件会自动跳过；中断后重新运行同一命令即可续传。",
            flush=True,
        )

    local_paths: list[str] = []
    for idx, remote_file in enumerate(files, 1):
        if verbose:
            print(f"  [{idx}/{len(files)}] {remote_file}", flush=True)

        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_file,
                    repo_type="dataset",
                    cache_dir=hub_cache,
                )
                break
            except Exception as exc:
                last_err = exc
                if attempt >= max_retries:
                    raise
                if verbose:
                    print(
                        f"    [retry {attempt}/{max_retries - 1}] {exc}",
                        flush=True,
                    )
                time.sleep(retry_wait_sec)
        else:
            if last_err is not None:
                raise last_err

        local_path = str(Path(local_path).resolve())
        if not local_path.startswith(str(hf_home.resolve())):
            raise RuntimeError(
                f"下载未写入 scratch/hf_home: {local_path}\n"
                f"期望前缀: {hf_home}"
            )
        local_paths.append(local_path)

    if verbose and local_paths:
        print(f"[download] 首个文件落盘: {local_paths[0]}", flush=True)
    return local_paths


def load_imagenet_split(
    split: str,
    repo_id: str = IMAGENET_REPO_ID,
    cache_dir: str | Path | None = None,
    parquet_paths: list[str] | None = None,
):
    """从本地 parquet 路径加载单个 split（不再使用 hf://，避免误拉 train）。"""
    from datasets import load_dataset

    paths = parquet_paths
    if not paths:
        paths = resolve_split_parquet_files(repo_id, split, verbose=False)
    if not paths:
        raise FileNotFoundError(f"split={split!r} 无 parquet 文件")

    return load_dataset(
        "parquet",
        data_files={split: paths},
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
    )


def load_imagenet_dataset(
    repo_id: str = IMAGENET_REPO_ID,
    cache_dir: str | Path | None = None,
    splits: tuple[str, ...] = ("train", "validation"),
):
    """加载 ImageNet；仅引用各 split 已有/按需下载的 parquet 本地路径。"""
    from datasets import load_dataset

    configure_imagenet_hf_env(verbose=False)
    data_files = {}
    for split in splits:
        data_files[split] = resolve_split_parquet_files(
            repo_id, split, verbose=False
        )

    return load_dataset(
        "parquet",
        data_files=data_files,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
