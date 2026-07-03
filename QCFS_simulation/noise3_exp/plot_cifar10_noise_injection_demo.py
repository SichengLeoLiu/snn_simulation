"""
CIFAR-10 噪声注入过程可视化：同一张图在 σ=0 / 0.5 / 1.0 下的效果。

与实验一致：在归一化输入张量上施加 x <- x + sigma * N(0,1)，再反归一化显示。

用法：
  python noise3_exp/plot_cifar10_noise_injection_demo.py --no-caption --copy-important
  python noise3_exp/plot_cifar10_noise_injection_demo.py --index 7 --seed 42 --font-size 18
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
IMPORTANT_RESULTS = ROOT.parent / "important results"
OUT_DIR = ROOT / "noise3_exp" / "cifar10_noise_injection_demo"
CIFAR_ROOT = os.path.expanduser(os.environ.get("CIFAR_ROOT", "~/datasets"))

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
DEFAULT_SIGMAS = [0.0, 0.5, 1.0]


def denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR_MEAN, dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD, dtype=x.dtype).view(3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def inject_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x.clone()
    return x + torch.randn_like(x) * sigma


def load_cifar10_sample(index: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    download = not (Path(CIFAR_ROOT) / "cifar-10-batches-py").is_dir()
    dataset = datasets.CIFAR10(
        root=CIFAR_ROOT,
        train=False,
        download=download,
        transform=transform,
    )
    image, _label = dataset[index]
    return image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CIFAR-10 噪声注入三面板示意图")
    p.add_argument("--index", type=int, default=7, help="CIFAR-10 test 样本索引")
    p.add_argument("--seed", type=int, default=42, help="噪声随机种子")
    p.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=DEFAULT_SIGMAS,
        help="噪声 sigma 列表（默认 0 0.5 1）",
    )
    p.add_argument("--font-size", type=float, default=18.0)
    p.add_argument("--copy-important", action="store_true")
    p.add_argument(
        "--no-caption",
        action="store_true",
        help="输出文件名带 _no_caption（仅保留各 panel 的 sigma 标签）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    image = load_cifar10_sample(args.index)
    suffix = "_no_caption" if args.no_caption else ""
    out_png = OUT_DIR / f"cifar10_noise_injection_demo{suffix}.png"

    torch.manual_seed(args.seed)
    panels = [
        denormalize(inject_gaussian_noise(image, sigma))
        for sigma in args.sigmas
    ]

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), dpi=180)
    if n == 1:
        axes = [axes]
    plt.rcParams.update({"font.size": args.font_size})
    for ax, sigma, img in zip(axes, args.sigmas, panels):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"$\sigma = {sigma:g}$")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] saved {out_png}", flush=True)

    if args.copy_important:
        IMPORTANT_RESULTS.mkdir(parents=True, exist_ok=True)
        dest = IMPORTANT_RESULTS / out_png.name
        shutil.copy2(out_png, dest)
        print(f"[PLOT] copied {dest}", flush=True)


if __name__ == "__main__":
    main()
