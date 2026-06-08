#!/usr/bin/env python3
"""
预下载 HuggingFace ImageNet-1k，供 GetImageNet / main_train 使用。

本项目 dataloader 使用:
    load_dataset("imagenet-1k")

前置条件:
  1. pip install datasets huggingface_hub
  2. 在 HuggingFace 接受 ImageNet 使用条款:
     https://huggingface.co/datasets/ILSVRC/imagenet-1k
  3. 提供 token（任选其一）:
       export HF_TOKEN=hf_xxxx
       huggingface-cli login

用法示例:
  # 下载 train + validation（默认缓存到 ~/.cache/huggingface）
  python scripts/download_imagenet.py

  # 指定缓存目录（Gadi 上建议放到大容量盘）
  export HF_HOME=$HOME/datasets/huggingface
  python scripts/download_imagenet.py --cache_dir $HF_HOME/datasets

  # 只下载 validation（约 6.3GB，用于快速验证环境）
  python scripts/download_imagenet.py --splits validation

  # 下载后做简单校验
  python scripts/download_imagenet.py --verify
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ID = "imagenet-1k"
DEFAULT_SPLITS = ("train", "validation")


def _ensure_deps() -> None:
    try:
        import datasets  # noqa: F401
        import huggingface_hub  # noqa: F401
    except ImportError as exc:
        print(
            "缺少依赖，请先安装:\n"
            "  pip install datasets huggingface_hub\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def _resolve_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token.strip()
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    return None


def _login(token: str | None) -> None:
    from huggingface_hub import login

    if token:
        login(token=token, add_to_git_credential=False)
        print("[OK] 已使用 HF token 登录")
        return

    try:
        from huggingface_hub import whoami

        whoami()
        print("[OK] 检测到已有 HuggingFace 登录状态")
    except Exception:
        print(
            "[WARN] 未检测到 HF token / 登录状态。\n"
            "       ImageNet-1k 通常需要先在网页接受条款，再设置 HF_TOKEN 或运行 huggingface-cli login。",
            flush=True,
        )


def _download_split(repo_id: str, split: str, cache_dir: str | None) -> int:
    from datasets import load_dataset

    print(f"\n[DOWNLOAD] repo={repo_id!r} split={split!r} cache_dir={cache_dir!r}", flush=True)
    ds = load_dataset(repo_id, split=split, cache_dir=cache_dir)
    n = len(ds)
    print(f"[DONE] split={split!r}  samples={n:,}", flush=True)
    return n


def _verify_sample(repo_id: str, cache_dir: str | None) -> None:
    from datasets import load_dataset

    ds = load_dataset(repo_id, split="validation", cache_dir=cache_dir)
    item = ds[0]
    image = item["image"]
    label = item["label"]
    print(
        f"[VERIFY] validation[0]: label={label}, "
        f"image_size={getattr(image, 'size', image.shape if hasattr(image, 'shape') else '?')}",
        flush=True,
    )


def _cache_hint(cache_dir: str | None) -> None:
    hf_home = os.environ.get("HF_HOME")
    datasets_cache = os.environ.get("HF_DATASETS_CACHE")
    default = Path.home() / ".cache" / "huggingface"
    print("\n缓存位置说明:", flush=True)
    if cache_dir:
        print(f"  --cache_dir = {cache_dir}", flush=True)
    if hf_home:
        print(f"  HF_HOME      = {hf_home}", flush=True)
    if datasets_cache:
        print(f"  HF_DATASETS_CACHE = {datasets_cache}", flush=True)
    if not cache_dir and not hf_home and not datasets_cache:
        print(f"  默认目录     ≈ {default}", flush=True)
    print(
        "\n磁盘占用（解压后大致）:\n"
        "  train       ~140 GB\n"
        "  validation  ~  6 GB\n"
        "  合计        ~150 GB（另加少量元数据缓存）",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="预下载 HuggingFace ImageNet-1k")
    p.add_argument(
        "--repo_id",
        default=REPO_ID,
        help=f"HuggingFace 数据集 id（默认 {REPO_ID}，与 GetImageNet 一致）",
    )
    p.add_argument(
        "--cache_dir",
        default=os.environ.get("IMAGENET_CACHE_DIR"),
        help="datasets 缓存目录；也可设环境变量 IMAGENET_CACHE_DIR / HF_HOME",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=["train", "validation"],
        help="要下载的 split（默认 train validation）",
    )
    p.add_argument(
        "--token",
        default=None,
        help="HuggingFace token；默认读取 HF_TOKEN 等环境变量",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="下载完成后读取 validation 第一条样本做校验",
    )
    p.add_argument(
        "--skip_login",
        action="store_true",
        help="跳过 huggingface_hub.login（已全局登录时可加）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_deps()

    print("=" * 60, flush=True)
    print("ImageNet-1k 预下载（HuggingFace datasets）", flush=True)
    print("=" * 60, flush=True)
    _cache_hint(args.cache_dir)

    token = _resolve_token(args.token)
    if not args.skip_login:
        _login(token)

    counts: dict[str, int] = {}
    for split in args.splits:
        try:
            counts[split] = _download_split(args.repo_id, split, args.cache_dir)
        except Exception as exc:
            print(f"\n[ERROR] 下载 split={split!r} 失败: {exc}", file=sys.stderr, flush=True)
            print(
                "\n常见原因:\n"
                "  1. 未在 HuggingFace 网页接受 ImageNet 条款\n"
                "     https://huggingface.co/datasets/ILSVRC/imagenet-1k\n"
                "  2. token 无效或未 export HF_TOKEN\n"
                "  3. 磁盘空间不足（完整数据约 150GB）\n"
                "  4. 计算节点无外网（请在 login 节点或有网络的节点先下载）",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(1) from exc

    if args.verify:
        _verify_sample(args.repo_id, args.cache_dir)

    print("\n" + "=" * 60, flush=True)
    print("全部完成。样本数:", flush=True)
    for split, n in counts.items():
        print(f"  {split:12s} {n:,}", flush=True)
    print("\n训练时无需额外配置；GetImageNet 会自动从 HuggingFace 缓存读取。", flush=True)
    if args.cache_dir:
        print(
            f"\n建议在同一 shell 中设置:\n"
            f"  export HF_DATASETS_CACHE={args.cache_dir}\n"
            f"  # 或 export HF_HOME={Path(args.cache_dir).parent}",
            flush=True,
        )


if __name__ == "__main__":
    main()
