"""
从 agg CSV 重画 CIFAR-10 VGG16 strict-seed 三路噪声折线图（可调字号）。

用法：
  python noise3_exp/plot_cifar10_vgg16_strict_seed_lineplot.py
  python noise3_exp/plot_cifar10_vgg16_strict_seed_lineplot.py \\
      --agg-csv ../cifar10_vgg16_strict_seed_three_regs_noise_sweep_mean_std.csv \\
      --font-size 18 --legend-font-size 16 --copy-root
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DEFAULT_OUT = (
    ROOT
    / "noise3_exp"
    / "cifar10_vgg16_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16"
)
DEFAULT_AGG = DEFAULT_OUT / "cifar10_vgg16_strict_seed_three_regs_noise_sweep_mean_std.csv"

PLOT_ORDER = ["weight_decay", "mne_l2 rc=1e-4", "mne_l2+wd rc=1e-4 wd=1e-4"]
LINE_STYLES = {
    "weight_decay": {"color": "#ff7f0e", "label": "L2"},
    "mne_l2 rc=1e-4": {"color": "#1f77b4", "label": "MNE L2"},
    "mne_l2+wd rc=1e-4 wd=1e-4": {
        "color": "#98df8a",
        "label": "MNE L2+L2",
    },
}


def load_agg_rows(agg_csv: Path) -> list[dict]:
    if not agg_csv.exists() or agg_csv.stat().st_size <= 80:
        raise FileNotFoundError(f"agg CSV 不存在或为空: {agg_csv}")
    with agg_csv.open(newline="") as f:
        return list(csv.DictReader(f))


def plot_results(
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
        out_png = out_dir / f"cifar10_vgg16_strict_seed_three_regs_noise_sweep_mean_std_lineplot{suffix}.png"
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] saved {out_png}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="重画 CIFAR-10 VGG16 strict-seed 噪声图")
    p.add_argument("--agg-csv", type=Path, default=DEFAULT_AGG)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--font-size", type=float, default=20.0)
    p.add_argument("--legend-font-size", type=float, default=18.0)
    p.add_argument("--no-caption-only", action="store_true")
    p.add_argument(
        "--copy-root",
        action="store_true",
        help="复制 no_caption 图到仓库根目录与 important results/",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    agg_csv = args.agg_csv
    if not agg_csv.exists() or agg_csv.stat().st_size <= 80:
        alt = REPO_ROOT / "cifar10_vgg16_strict_seed_three_regs_noise_sweep_mean_std.csv"
        if alt.exists() and alt.stat().st_size > 80:
            agg_csv = alt
    rows = load_agg_rows(agg_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_results(rows, args.out_dir, args.font_size, args.legend_font_size, args.no_caption_only)
    if args.copy_root:
        src = args.out_dir / "cifar10_vgg16_strict_seed_three_regs_noise_sweep_mean_std_lineplot_no_caption.png"
        if src.exists():
            for dest in (
                REPO_ROOT / src.name,
                REPO_ROOT / "important results" / src.name,
            ):
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                print(f"[PLOT] copied {dest}", flush=True)


if __name__ == "__main__":
    main()
