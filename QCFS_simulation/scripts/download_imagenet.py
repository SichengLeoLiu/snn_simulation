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

Gadi 注意:
  /home 配额仅 ~10GB。在 gs14 上脚本会自动使用:
    /scratch/gs14/sl9144/huggingface
  也可手动:
    source scripts/setup_gadi_imagenet.sh

用法示例:
  # 查看 home 配额与 HF 缓存占用
  python scripts/download_imagenet.py --check_disk

  # 删除已下载的 ImageNet 缓存（保留 HF token）
  python scripts/download_imagenet.py --clean

  # 只下载 validation（约 6GB，不会误下 train）
  python scripts/download_imagenet.py --splits validation --verify

  # 完整 train + validation（约 150GB，需大容量盘）
  python scripts/download_imagenet.py --splits train validation
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ID = "imagenet-1k"
DEFAULT_SPLITS = ("train", "validation")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Preprocess.imagenet_hf_env import (  # noqa: E402
    GADI_SCRATCH_ROOT,
    IMAGENET_ENV_VERSION,
    configure_imagenet_hf_env,
    download_split_parquet_files,
    home_hf_cache_candidates,
    load_imagenet_split,
    resolve_imagenet_datasets_cache,
    resolve_imagenet_hf_home,
)

DOWNLOAD_SCRIPT_VERSION = IMAGENET_ENV_VERSION


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


def _hf_home() -> Path:
    return resolve_imagenet_hf_home()


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


def _run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.2f} TB"


def check_disk_usage() -> None:
    home = Path.home()
    hf_home = _hf_home()
    print("=" * 60, flush=True)
    print("磁盘 / 配额检查", flush=True)
    print("=" * 60, flush=True)

    quota = _run_cmd(["quota", "-s"]) or _run_cmd(["lfs", "quota", "-h", str(home)])
    if quota:
        print("\n[quota home]", flush=True)
        print(quota, flush=True)
        if "5274M" in quota or "10240M" in quota:
            print(
                "  → home 已用约 5GB / 上限 10GB，剩余约 5GB（token/代码够用）",
                flush=True,
            )
    else:
        df = _run_cmd(["df", "-h", str(home)])
        if df:
            print("\n[df -h $HOME]", flush=True)
            print(df, flush=True)

    lquota = _run_cmd(["lquota"])
    if lquota:
        print("\n[lquota 项目盘]", flush=True)
        for line in lquota.splitlines():
            if "gs14" in line or "scratch" in line or "----" in line or "fs" in line:
                print(line, flush=True)

    print("\n[目录大小]", flush=True)
    for label, path in (
        ("$HOME", home),
        ("$HF_HOME", hf_home),
        ("$HF_HOME/hub", hf_home / "hub"),
    ):
        print(f"  {label:16s} {_fmt_bytes(_dir_size(path))}", flush=True)

    if GADI_SCRATCH_ROOT.is_dir():
        print(
            f"  /scratch/gs14/sl9144  {_fmt_bytes(_dir_size(GADI_SCRATCH_ROOT))}",
            flush=True,
        )

    imagenet_dirs = find_imagenet_cache_dirs(hf_home)
    if imagenet_dirs:
        print("\n[ImageNet 相关缓存]", flush=True)
        total = 0
        for p in imagenet_dirs:
            sz = _dir_size(p)
            total += sz
            print(f"  {_fmt_bytes(sz):>10s}  {p}", flush=True)
        print(f"  {'合计':16s} {_fmt_bytes(total)}", flush=True)
    else:
        print("\n[ImageNet 相关缓存] 未发现", flush=True)

    print(
        "\n提示: Gadi /home 通常 ~10GB；ImageNet 完整集 ~150GB 请放到 scratch 或 /g/data。",
        flush=True,
    )


def find_imagenet_cache_dirs(
    hf_roots: Path | list[Path] | None = None,
) -> list[Path]:
    if hf_roots is None:
        roots = [_hf_home()]
    elif isinstance(hf_roots, Path):
        roots = [hf_roots]
    else:
        roots = list(hf_roots)
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue

        patterns = [
            root / "hub" / "datasets--imagenet-1k",
            root / "hub" / "datasets--ILSVRC--imagenet-1k",
            root / "datasets" / "imagenet-1k",
            root / "datasets" / "ILSVRC___imagenet-1k",
        ]
        for p in patterns:
            if p.exists():
                found.append(p)

        hub = root / "hub"
        if hub.exists():
            for p in hub.glob("datasets--*imagenet*"):
                if p not in found:
                    found.append(p)

        datasets_dir = root / "datasets"
        if datasets_dir.exists():
            for p in datasets_dir.glob("*imagenet*"):
                if p not in found:
                    found.append(p)

        lock_dir = hub / ".locks" if hub.exists() else root / "hub" / ".locks"
        if lock_dir.exists():
            for p in lock_dir.glob("*imagenet*"):
                if p not in found:
                    found.append(p)

    return sorted(set(found))


def clean_imagenet_cache(
    hf_home: Path | None = None,
    include_home_copies: bool = True,
) -> None:
    roots = [hf_home or _hf_home()]
    if include_home_copies:
        roots.extend(home_hf_cache_candidates())
    targets = find_imagenet_cache_dirs(roots)

    if not targets:
        print("[CLEAN] 未发现 ImageNet 缓存，无需删除。", flush=True)
        return

    print("[CLEAN] 将删除以下目录:", flush=True)
    total = 0
    for p in targets:
        sz = _dir_size(p)
        total += sz
        print(f"  {_fmt_bytes(sz):>10s}  {p}", flush=True)
    print(f"  预计释放: {_fmt_bytes(total)}", flush=True)

    for p in targets:
        if p.is_dir():
            shutil.rmtree(p)
        elif p.is_file():
            p.unlink()
        print(f"  [removed] {p}", flush=True)

    print("[CLEAN DONE] HF token 未删除（仍在 $HF_HOME/token）。", flush=True)


def _download_split(repo_id: str, split: str, cache_dir: str | None) -> int:
    print(
        f"\n[DOWNLOAD] repo={repo_id!r} split={split!r} "
        f"cache_dir={cache_dir!r}",
        flush=True,
    )
    print(
        "  说明: 不使用 load_dataset('imagenet-1k')，"
        "仅拉当前 split 的 parquet（HF issue #6793）。",
        flush=True,
    )

    local_paths = download_split_parquet_files(repo_id, split, verbose=True)
    ds = load_imagenet_split(
        split,
        repo_id=repo_id,
        cache_dir=cache_dir,
        parquet_paths=local_paths,
    )
    n = len(ds)
    print(f"[DONE] split={split!r}  samples={n:,}", flush=True)
    return n


def _verify_sample(repo_id: str, cache_dir: str | None) -> None:
    ds = load_imagenet_split("validation", repo_id=repo_id, cache_dir=cache_dir)
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
    p.add_argument(
        "--check_disk",
        action="store_true",
        help="查看 home 配额与 HF / ImageNet 缓存占用后退出",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="删除已下载的 ImageNet 缓存（保留 HF token）后退出",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 必须在 import datasets/huggingface_hub 之前设置缓存路径
    hf_home = configure_imagenet_hf_env(verbose=True)
    print(f"[download_imagenet] version={DOWNLOAD_SCRIPT_VERSION}", flush=True)

    _ensure_deps()
    cache_dir = args.cache_dir or str(resolve_imagenet_datasets_cache())

    if args.check_disk:
        check_disk_usage()
        return

    if args.clean:
        clean_imagenet_cache(hf_home)
        check_disk_usage()
        return

    print("=" * 60, flush=True)
    print("ImageNet-1k 预下载（HuggingFace datasets）", flush=True)
    print("=" * 60, flush=True)
    _cache_hint(cache_dir)
    check_disk_usage()

    if "train" in args.splits and hf_home.as_posix().startswith(str(Path.home())):
        home_sz = _dir_size(Path.home())
        if home_sz > 8 * 1024**3:
            print(
                "\n[WARN] $HOME 已用 "
                f"{_fmt_bytes(home_sz)}，完整 train 约 140GB，"
                "请把 HF_HOME 设到 scratch 或 /g/data 后再下载 train。",
                flush=True,
            )

    token = _resolve_token(args.token)
    if not args.skip_login:
        _login(token)

    counts: dict[str, int] = {}
    for split in args.splits:
        try:
            counts[split] = _download_split(args.repo_id, split, cache_dir)
        except Exception as exc:
            err = str(exc)
            print(f"\n[ERROR] 下载 split={split!r} 失败: {exc}", file=sys.stderr, flush=True)
            extra = [
                "  1. 未在 HuggingFace 网页接受 ImageNet 条款",
                "     https://huggingface.co/datasets/ILSVRC/imagenet-1k",
                "  2. token 无效或未 export HF_TOKEN",
                "  3. 磁盘配额不足（Gadi /home 通常 ~10GB）",
                "  4. 计算节点无外网（请在 login 节点下载）",
            ]
            if "quota exceeded" in err.lower() or "errno 122" in err.lower():
                extra.insert(
                    0,
                    "  0. 磁盘配额已满 — 先运行: python scripts/download_imagenet.py --clean",
                )
            if "incompleteread" in err.lower() or "connection broken" in err.lower():
                extra.insert(
                    0,
                    "  0. 网络中断 — 已下载文件在 scratch 缓存中，直接重新运行同一命令续传",
                )
            print("\n常见原因:\n" + "\n".join(extra), file=sys.stderr, flush=True)
            raise SystemExit(1) from exc

    if args.verify:
        _verify_sample(args.repo_id, cache_dir)

    print("\n" + "=" * 60, flush=True)
    print("全部完成。样本数:", flush=True)
    for split, n in counts.items():
        print(f"  {split:12s} {n:,}", flush=True)
    print("\n训练时无需额外配置；GetImageNet 会自动从上述 HF 缓存读取。", flush=True)
    if cache_dir:
        print(
            f"\n当前缓存:\n"
            f"  export HF_HOME={os.environ['HF_HOME']}\n"
            f"  export HF_DATASETS_CACHE={os.environ['HF_DATASETS_CACHE']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
