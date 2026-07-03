"""
从 ablation CNN 三路正则噪声 CSV 重画折线图（无 caption、可调字号）。

默认处理 c2c4 / c4c8 / c16c32 单 seed 实验目录下的
``noise_sweep_three_methods_long.csv``。

用法：
  python noise3_exp/plot_cnn_three_methods_noise_sweep.py
  python noise3_exp/plot_cnn_three_methods_noise_sweep.py --font-size 20 --legend-font-size 18
  python noise3_exp/plot_cnn_three_methods_noise_sweep.py --copy-important
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
IMPORTANT_RESULTS = ROOT.parent / "important results"

CNN_EXPERIMENTS = [
    {
        "label": "c2c4",
        "arch": "cnn2_c2_c4",
        "dir": ROOT / "noise3_exp" / "ablation_mne_l2_vs_weight_decay_l16" / "noise_sweep_sigma_0_1_T16_rate_uniform",
    },
    {
        "label": "c4c8",
        "arch": "cnn2_c4_c8",
        "dir": ROOT / "noise3_exp" / "ablation_mne_l2_vs_weight_decay_l16_c4_c8" / "noise_sweep_sigma_0_1_T16_rate_uniform",
    },
    {
        "label": "c16c32",
        "arch": "cnn2_c16_c32",
        "dir": ROOT / "noise3_exp" / "ablation_mne_l2_vs_weight_decay_l16_c16_c32" / "noise_sweep_sigma_0_1_T16_rate_uniform",
    },
]

METHOD_ORDER = ["weight_decay", "mne_l2", "no_regularization"]

LINE_STYLES = {
    "weight_decay": {"color": "#1f77b4", "label": "weight_decay"},
    "mne_l2": {"color": "#ff7f0e", "label": "mne_l2"},
    "no_regularization": {"color": "#2ca02c", "label": "no regularization"},
}


def load_long_csv(csv_path: Path) -> dict[str, list[tuple[float, float]]]:
    curves: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            curves[row["method"]].append((float(row["sigma"]), float(row["acc"])))
    for method in curves:
        curves[method].sort(key=lambda x: x[0])
    return dict(curves)


def important_plot_name(arch: str) -> str:
    return f"{arch}_rate_uniform_three_methods_noise_sweep_lineplot_no_caption.png"


def plot_one(
    exp_dir: Path,
    curves: dict[str, list[tuple[float, float]]],
    font_size: float,
    legend_font_size: float,
    copy_important: bool,
    arch: str,
) -> Path:
    plt.rcParams.update({"font.size": font_size, "legend.fontsize": legend_font_size})
    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=180)
    all_y: list[float] = []
    for method in METHOD_ORDER:
        pts = curves.get(method)
        if not pts:
            continue
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        all_y.extend(y)
        style = LINE_STYLES[method]
        ax.plot(
            x, y, marker="o", linewidth=2.4, markersize=5,
            color=style["color"], label=style["label"],
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
    out_png = exp_dir / "noise_sweep_three_methods_lineplot_no_caption.png"
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[PLOT] saved {out_png}", flush=True)
    if copy_important:
        IMPORTANT_RESULTS.mkdir(parents=True, exist_ok=True)
        dest = IMPORTANT_RESULTS / important_plot_name(arch)
        shutil.copy2(out_png, dest)
        print(f"[PLOT] copied {dest}", flush=True)
    return out_png


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="重画 CNN 三路正则噪声折线图（无 caption）")
    p.add_argument("--font-size", type=float, default=18.0)
    p.add_argument("--legend-font-size", type=float, default=16.0)
    p.add_argument("--copy-important", action="store_true")
    p.add_argument(
        "--labels",
        nargs="+",
        default=[e["label"] for e in CNN_EXPERIMENTS],
        help="c2c4 c4c8 c16c32（默认全部）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    label_set = set(args.labels)
    for exp in CNN_EXPERIMENTS:
        if exp["label"] not in label_set:
            continue
        exp_dir = exp["dir"]
        csv_path = exp_dir / "noise_sweep_three_methods_long.csv"
        if not csv_path.exists():
            print(f"[SKIP] missing {csv_path}", flush=True)
            continue
        curves = load_long_csv(csv_path)
        plot_one(
            exp_dir, curves, args.font_size, args.legend_font_size,
            args.copy_important, exp["arch"],
        )


if __name__ == "__main__":
    main()
