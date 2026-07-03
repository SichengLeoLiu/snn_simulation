"""
fc3 strict-seed rate_uniform：acc 对 sigma 的一阶导数图（三路，无 mne_l2+wd）。

输出到仓库根目录 derivative results/，便于说明 acc drop 速度。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AGG_CSV = (
    ROOT
    / "noise3_exp"
    / "ablation_mne_l2_vs_weight_decay_l16_fc3_h4_h8_h16_h32_h64_h128"
    / "strict_seed_train_rate_uniform_L16_T16"
    / "strict_seed_train_noise_sweep_fc3_h4_h8_h16_h32_h64_h128_mean_std.csv"
)
DERIVATIVE_RESULTS = ROOT.parent / "derivative results"

THREE_REGS = ["mne_l2", "weight_decay", "no_regularization"]
ALL_H_LIST = [4, 8, 16, 32, 64, 128]
IF_MODE = "rate_uniform"

LINE_STYLES = {
    "mne_l2": {"color": "#1f77b4", "label": "mne_l2 (mean)"},
    "weight_decay": {"color": "#ff7f0e", "label": "weight_decay (mean)"},
    "no_regularization": {"color": "#2ca02c", "label": "no regularization (mean)"},
}


def arch_for(h: int) -> str:
    return f"fc3_h{h}"


def load_agg_rows() -> list[dict]:
    with AGG_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def acc_derivative(sigma: np.ndarray, acc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """d(acc)/d(sigma)，单位：%/sigma。"""
    dacc = np.gradient(acc, sigma)
    return sigma, dacc


def plot_name(arch: str) -> str:
    return f"strict_seed_train_{arch}_rate_uniform_acc_derivative_no_caption.png"


def plot_derivatives(
    agg_rows: list[dict],
    h_list: list[int],
    font_size: float,
    legend_font_size: float,
) -> None:
    DERIVATIVE_RESULTS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": font_size, "legend.fontsize": legend_font_size})

    for h in h_list:
        arch = arch_for(h)
        rows_arch = [r for r in agg_rows if r["arch"] == arch]
        if not rows_arch:
            print(f"[PLOT] skip {arch}: no data", flush=True)
            continue

        fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=180)
        all_y: list[float] = []
        for reg in THREE_REGS:
            rr = [r for r in rows_arch if r["regularizer"] == reg]
            if not rr:
                continue
            rr.sort(key=lambda x: float(x["sigma"]))
            sigma = np.array([float(r["sigma"]) for r in rr])
            acc = np.array([float(r["acc_mean"]) for r in rr])
            x, y = acc_derivative(sigma, acc)
            all_y.extend(y.tolist())
            style = LINE_STYLES[reg]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.4,
                markersize=6,
                color=style["color"],
                label=style["label"],
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

        out_png = DERIVATIVE_RESULTS / plot_name(arch)
        fig.savefig(out_png)
        plt.close(fig)
        print(f"[PLOT] saved {out_png}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="fc3 rate_uniform acc derivative plots (three regs, no caption)"
    )
    p.add_argument(
        "--h-list",
        type=int,
        nargs="+",
        default=ALL_H_LIST,
        help=f"hidden sizes (default {' '.join(map(str, ALL_H_LIST))})",
    )
    p.add_argument("--font-size", type=float, default=14.0)
    p.add_argument("--legend-font-size", type=float, default=12.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not AGG_CSV.exists():
        raise SystemExit(f"mean_std CSV not found: {AGG_CSV}")
    plot_derivatives(
        load_agg_rows(),
        args.h_list,
        args.font_size,
        args.legend_font_size,
    )
    print(f"[DONE] output dir: {DERIVATIVE_RESULTS}", flush=True)


if __name__ == "__main__":
    main()
