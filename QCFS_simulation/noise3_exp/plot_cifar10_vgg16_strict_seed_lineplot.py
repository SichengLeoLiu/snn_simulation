"""
从 agg CSV 重画 CIFAR-10/100 VGG16 strict-seed 三路噪声折线图（L2 / MNE L2 / No reg）。

用法：
  python noise3_exp/plot_cifar10_vgg16_strict_seed_lineplot.py --dataset cifar10
  python noise3_exp/plot_cifar10_vgg16_strict_seed_lineplot.py --dataset cifar100 --derivative
  python noise3_exp/plot_cifar10_vgg16_strict_seed_lineplot.py \\
      --dataset cifar10 --font-size 18 --legend-font-size 16 --copy-root
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent

PLOT_ORDER = [
    "weight_decay",
    "mne_l2 rc=1e-4",
    "no regularization",
]
LINE_STYLES = {
    "weight_decay": {"color": "#ff7f0e", "label": "L2"},
    "mne_l2 rc=1e-4": {"color": "#1f77b4", "label": "MNE L2"},
    "no regularization": {"color": "#2ca02c", "label": "No reg"},
}


def default_paths(dataset: str) -> tuple[Path, Path]:
    out = (
        ROOT
        / "noise3_exp"
        / f"{dataset}_vgg16_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16"
    )
    agg = out / f"{dataset}_vgg16_strict_seed_three_regs_noise_sweep_mean_std.csv"
    return out, agg


def load_agg_rows(agg_csv: Path) -> list[dict]:
    if not agg_csv.exists() or agg_csv.stat().st_size <= 80:
        raise FileNotFoundError(f"agg CSV 不存在或为空: {agg_csv}")
    with agg_csv.open(newline="") as f:
        return list(csv.DictReader(f))


def acc_derivative(sigma: np.ndarray, acc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """d(acc)/d(sigma)，单位：%/sigma。"""
    return sigma, np.gradient(acc, sigma)


def plot_derivative_results(
    dataset: str,
    agg_rows: list[dict],
    out_dir: Path,
    font_size: float,
    legend_font_size: float,
    no_caption_only: bool,
) -> None:
    plt.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": legend_font_size,
    })
    variants = [True] if no_caption_only else [False, True]
    for no_caption in variants:
        fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=180)
        all_y: list[float] = []
        for label in PLOT_ORDER:
            rr = [r for r in agg_rows if r["label"] == label]
            if not rr:
                continue
            rr.sort(key=lambda x: float(x["sigma"]))
            sigma = np.array([float(r["sigma"]) for r in rr])
            acc = np.array([float(r["acc_mean"]) for r in rr])
            x, y = acc_derivative(sigma, acc)
            all_y.extend(y.tolist())
            style = LINE_STYLES[label]
            ax.plot(
                x, y, marker="o", linewidth=2.4, markersize=6,
                color=style["color"], label=style["label"],
            )
        ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.6)
        ax.set_xlabel("Gaussian noise sigma")
        ax.set_ylabel(r"$d(\mathrm{Acc})/d\sigma$ (\%/sigma)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([round(i * 0.1, 1) for i in range(11)])
        if all_y:
            pad = max(0.5, 0.05 * (max(all_y) - min(all_y)))
            ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        suffix = "_no_caption" if no_caption else ""
        out_png = (
            out_dir
            / f"{dataset}_vgg16_strict_seed_three_regs_noise_sweep_derivative{suffix}.png"
        )
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] saved {out_png}", flush=True)


def plot_results(
    dataset: str,
    agg_rows: list[dict],
    out_dir: Path,
    font_size: float,
    legend_font_size: float,
    no_caption_only: bool,
) -> None:
    multi_seed = any(int(r["n_seeds"]) > 1 for r in agg_rows)
    plt.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": legend_font_size,
    })
    variants = [True] if no_caption_only else [False, True]
    for no_caption in variants:
        fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=180)
        all_y: list[float] = []
        for label in PLOT_ORDER:
            rr = [r for r in agg_rows if r["label"] == label]
            if not rr:
                continue
            rr.sort(key=lambda x: float(x["sigma"]))
            x = [float(r["sigma"]) for r in rr]
            y = [float(r["acc_mean"]) for r in rr]
            s = [float(r["acc_std"]) for r in rr]
            all_y.extend([yy - ss for yy, ss in zip(y, s)])
            all_y.extend([yy + ss for yy, ss in zip(y, s)])
            style = LINE_STYLES[label]
            ax.plot(
                x, y, marker="o", linewidth=2.4, markersize=6,
                color=style["color"], label=style["label"],
            )
            if multi_seed and any(ss > 0 for ss in s):
                ax.fill_between(
                    x,
                    [yy - ss for yy, ss in zip(y, s)],
                    [yy + ss for yy, ss in zip(y, s)],
                    color=style["color"], alpha=0.18, linewidth=0,
                )
        ax.set_xlabel("Gaussian noise sigma")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([round(i * 0.1, 1) for i in range(11)])
        if all_y:
            ax.set_ylim(min(all_y) - 1.0, max(all_y) + 1.0)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", frameon=False)
        fig.tight_layout()
        suffix = "_no_caption" if no_caption else ""
        out_png = (
            out_dir
            / f"{dataset}_vgg16_strict_seed_three_regs_noise_sweep_mean_std_lineplot{suffix}.png"
        )
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] saved {out_png}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="重画 CIFAR-10/100 VGG16 strict-seed 噪声图")
    p.add_argument(
        "--dataset",
        choices=["cifar10", "cifar100"],
        default="cifar10",
    )
    p.add_argument("--agg-csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--font-size", type=float, default=20.0)
    p.add_argument("--legend-font-size", type=float, default=18.0)
    p.add_argument("--no-caption-only", action="store_true")
    p.add_argument(
        "--derivative",
        action="store_true",
        help="绘制 acc 对 sigma 的一阶导数（acc 下降率）",
    )
    p.add_argument(
        "--copy-root",
        action="store_true",
        help="复制 no_caption 图到仓库根目录与 important results/",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset
    default_out, default_agg = default_paths(dataset)
    out_dir = args.out_dir or default_out
    agg_csv = args.agg_csv or default_agg
    if not agg_csv.exists() or agg_csv.stat().st_size <= 80:
        alt = REPO_ROOT / f"{dataset}_vgg16_strict_seed_three_regs_noise_sweep_mean_std.csv"
        if alt.exists() and alt.stat().st_size > 80:
            agg_csv = alt
    rows = load_agg_rows(agg_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.derivative:
        plot_derivative_results(
            dataset, rows, out_dir, args.font_size, args.legend_font_size, args.no_caption_only
        )
        png_name = f"{dataset}_vgg16_strict_seed_three_regs_noise_sweep_derivative_no_caption.png"
    else:
        plot_results(
            dataset, rows, out_dir, args.font_size, args.legend_font_size, args.no_caption_only
        )
        png_name = (
            f"{dataset}_vgg16_strict_seed_three_regs_noise_sweep_mean_std_lineplot_no_caption.png"
        )
    if args.copy_root:
        src = out_dir / png_name
        if src.exists():
            for dest in (
                REPO_ROOT / src.name,
                REPO_ROOT / "important results" / src.name,
            ):
                if dest.resolve() == src.resolve():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                print(f"[PLOT] copied {dest}", flush=True)


if __name__ == "__main__":
    main()
