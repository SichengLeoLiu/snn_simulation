#!/usr/bin/env python3
"""
预下载 MNIST，供 Gadi 等无网 GPU 节点离线训练。

torchvision 会将数据存到 ``$MNIST_ROOT/MNIST/``（含 raw/ 与 processed/）。

Gadi 推荐流程（login 节点，有网）:
  source scripts/setup_gadi_mnist.sh
  python scripts/download_mnist.py --verify

GPU 节点（无网）:
  source scripts/setup_gadi_mnist.sh
  python noise3_exp/run_cnn_wd_strict_seed_L_T_acc.py

本地:
  python scripts/download_mnist.py
  python noise3_exp/run_cnn_wd_strict_seed_L_T_acc.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Preprocess.getdataloader import MNIST_ROOT, _mnist_available  # noqa: E402


def default_root() -> str:
    gadi = Path("/scratch/gs14/sl9144/datasets")
    if gadi.is_dir():
        return str(gadi)
    return MNIST_ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download MNIST for offline use")
    p.add_argument(
        "--root",
        default=os.environ.get("MNIST_ROOT", default_root()),
        help="MNIST 根目录（默认 MNIST_ROOT 或 Gadi scratch）",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="下载后校验样本数并打印路径",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = os.path.expanduser(args.root)
    os.makedirs(root, exist_ok=True)

    if _mnist_available(root):
        print(f"[download_mnist] already present under {root}/MNIST")
        if args.verify:
            _verify(root)
        return

    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise SystemExit("需要 torchvision: pip install torchvision") from exc

    transform = transforms.ToTensor()
    print(f"[download_mnist] downloading to {root}/MNIST ...")
    train = datasets.MNIST(root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root, train=False, download=True, transform=transform)
    print(f"[download_mnist] done: train={len(train)} test={len(test)}")

    if args.verify:
        _verify(root)


def _verify(root: str) -> None:
    if not _mnist_available(root):
        raise SystemExit(f"[download_mnist] verify failed: missing processed/*.pt in {root}/MNIST")
    processed = Path(root) / "MNIST" / "processed"
    train_pt = processed / "training.pt"
    test_pt = processed / "test.pt"
    print(f"[download_mnist] verify ok")
    print(f"  MNIST_ROOT={root}")
    print(f"  training.pt size={train_pt.stat().st_size / 1e6:.1f} MB")
    print(f"  test.pt size={test_pt.stat().st_size / 1e6:.1f} MB")
    print("  GPU 节点请先: source scripts/setup_gadi_mnist.sh")


if __name__ == "__main__":
    main()
