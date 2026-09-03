"""
从 mean_std CSV 绘制 CIFAR-10 VGG16 三路正则噪声折线图。

注意：agg CSV 中 sigma 可能被格式化为 %.1f（0.05/0.10/0.15 均显示 0.1），
本脚本按每个 method 内的行序恢复 sigma = 0, 0.05, ..., 1.0。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "cifar10_vgg16_strict_seed_three_regs_noise_sweep_mean_std.csv"
DEFAULT_OUT = ROOT / "plots" / "cifar10_vgg16"

PLOT_ORDER = [
    "mne_l2 rc=1e-4",
    "weight_decay",
    "no regularization",
]
LINE_STYLES = {
    "mne_l2 rc=1e-4": {"color": "#ff7f0e", "label": "MNE-L2"},
    "weight_decay": {"color": "#1f77b4", "label": "L2"},
    "no regularization": {"color": "#2ca02c", "label": "No Reg"},
}
SIGMA_STEP = 0.05


def load_agg_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty csv: {csv_path}")

    by_label: dict[str, list[dict]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    fixed: list[dict] = []
    for label in PLOT_ORDER:
        group = by_label.get(label, [])
        if not group:
            continue
        n = len(group)
        for i, row in enumerate(group):
            out = dict(row)
            if n >= 2:
                out["sigma"] = f"{i * SIGMA_STEP:.2f}".rstrip("0").rstrip(".") or "0"
            fixed.append(out)
    return fixed


def plot_results(
    agg_rows: list[dict],
    out_dir: Path,
    title: str | None,
    font_size: float,
    legend_font_size: float,
    stem: str,
) -> None:
    multi_seed = any(int(r["n_seeds"]) > 1 for r in agg_rows)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": legend_font_size,
        }
    )

    for no_caption in (False, True):
        fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=220)
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
                x,
                y,
                marker="o",
                linewidth=2.2,
                markersize=5,
                color=style["color"],
                label=style["label"],
            )
            if multi_seed and any(ss > 0 for ss in s):
                ax.fill_between(
                    x,
                    [yy - ss for yy, ss in zip(y, s)],
                    [yy + ss for yy, ss in zip(y, s)],
                    color=style["color"],
                    alpha=0.18,
                    linewidth=0,
                )

        ax.set_xlabel("Gaussian noise sigma")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([round(i * 0.1, 1) for i in range(11)])
        if all_y:
            ax.set_ylim(min(all_y) - 1.0, max(all_y) + 1.0)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", frameon=True)
        if title and not no_caption:
            ax.set_title(title)
        fig.tight_layout()

        suffix = "_no_caption" if no_caption else ""
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            out_path = out_dir / f"{stem}{suffix}.{ext}"
            fig.savefig(out_path, bbox_inches="tight")
            print(f"[PLOT] {out_path}")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot CIFAR-10 VGG16 three-regs noise sweep from mean_std CSV")
    p.add_argument("--agg-csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--stem", default="cifar10_vgg16_three_regs_noise_sweep_step0p05")
    p.add_argument("--title", default="CIFAR-10 VGG16: noise sweep (L=16, T=16, rate_uniform, drs_rerun)")
    p.add_argument("--no-title", action="store_true")
    p.add_argument("--font-size", type=float, default=18.0)
    p.add_argument("--legend-font-size", type=float, default=16.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_agg_rows(args.agg_csv)
    plot_results(
        rows,
        args.out_dir,
        None if args.no_title else args.title,
        args.font_size,
        args.legend_font_size,
        args.stem,
    )


if __name__ == "__main__":
    main()
