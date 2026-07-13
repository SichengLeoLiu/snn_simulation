#!/usr/bin/env python3
"""
预下载 Fashion-MNIST，供 Gadi 等无网 GPU 节点离线训练。

会下载到 ``$MNIST_ROOT/FashionMNIST/``（torchvision 默认结构）。

Gadi 推荐（login 节点，有网）:
  source scripts/setup_gadi_mnist.sh
  python scripts/download_fashion_mnist.py --verify

GPU 节点（无网）:
  source scripts/setup_gadi_mnist.sh
  python main_train.py -data fashion_mnist -arch cnn2_c8_c16 -L 16 --epochs 100
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def default_root() -> str:
    gadi = Path("/scratch/gs14/sl9144/datasets")
    if gadi.is_dir():
        return str(gadi)
    return os.path.expanduser(os.environ.get("MNIST_ROOT", "~/datasets"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Fashion-MNIST for offline use")
    p.add_argument(
        "--root",
        default=os.environ.get("MNIST_ROOT", default_root()),
        help="Fashion-MNIST 根目录（默认 MNIST_ROOT 或 Gadi scratch）",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="下载后校验样本数并打印路径",
    )
    return p.parse_args()


def _fashion_available(root: str) -> bool:
    fashion_root = Path(root) / "FashionMNIST"
    raw = fashion_root / "raw"
    names = (
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    )
    for name in names:
        plain = raw / name
        gz = raw / f"{name}.gz"
        if not (plain.is_file() or gz.is_file()):
            return False
    return True


def _verify(root: str) -> None:
    if not _fashion_available(root):
        raise SystemExit(
            f"[download_fashion_mnist] verify failed: missing FashionMNIST under {root}/FashionMNIST"
        )
    print("[download_fashion_mnist] verify ok")
    print(f"  MNIST_ROOT={root}")
    print("  expected train=60000 test=10000")
    print("  GPU 节点请先: source scripts/setup_gadi_mnist.sh")


def main() -> None:
    args = parse_args()
    root = os.path.expanduser(args.root)
    os.makedirs(root, exist_ok=True)

    if _fashion_available(root):
        print(f"[download_fashion_mnist] already present under {root}/FashionMNIST")
        if args.verify:
            _verify(root)
        return

    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise SystemExit("需要 torchvision: pip install torchvision") from exc

    transform = transforms.ToTensor()
    print(f"[download_fashion_mnist] downloading to {root}/FashionMNIST ...")
    train = datasets.FashionMNIST(root, train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(root, train=False, download=True, transform=transform)
    print(f"[download_fashion_mnist] done: train={len(train)} test={len(test)}")
    if args.verify:
        _verify(root)


if __name__ == "__main__":
    main()
