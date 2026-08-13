"""Plot MNE stability ablation noise sweeps from mean/std CSV."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ORDER = [
    "weight_decay",
    "old_detach",
    "raw_w_lambda",
    "folded_w_lambda",
    "fanin_mean",
    "full_bn",
    "lref_only",
]

STYLES = {
    "weight_decay": {
        "label": "Weight decay",
        "color": "#0072B2",
        "linestyle": "--",
    },
    "old_detach": {
        "label": r"Old MNE (detach $\lambda$)",
        "color": "#E69F00",
        "linestyle": "-",
    },
    "raw_w_lambda": {
        "label": r"MNE raw $W$ ($\lambda$ trainable)",
        "color": "#000000",
        "linestyle": "-",
    },
    "folded_w_lambda": {
        "label": r"MNE folded $\widetilde{W}$ ($\lambda$ trainable)",
        "color": "#56B4E9",
        "linestyle": "-",
    },
    "fanin_mean": {
        "label": "Stable MNE (BN detached)",
        "color": "#009E73",
        "linestyle": "-",
    },
    "full_bn": {
        "label": "Stable MNE (BN affine on)",
        "color": "#D55E00",
        "linestyle": "-",
    },
    "lref_only": {
        "label": r"Stable MNE ($\ell_{\mathrm{ref}}$ only)",
        "color": "#CC79A7",
        "linestyle": "-",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot MNE stability ablation noise curves."
    )
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--title", default="")
    return parser.parse_args()


def load_grouped(csv_path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] in STYLES:
                grouped[row["variant"]].append(row)
    return grouped


def plot_curve(ax, variant: str, rows: list[dict], *, relative: bool) -> None:
    rows = sorted(rows, key=lambda row: float(row["sigma"]))
    sigma = [float(row["sigma"]) for row in rows]
    mean = [float(row["acc_mean"]) for row in rows]
    clean = mean[0] if mean else 1.0
    y = [value / clean * 100.0 if relative else value for value in mean]
    style = STYLES[variant]
    ax.plot(
        sigma,
        y,
        marker="o",
        linewidth=2.2,
        markersize=4.5,
        color=style["color"],
        linestyle=style["linestyle"],
        label=style["label"],
    )


def main() -> None:
    args = parse_args()
    out = args.out or args.csv.with_name("mne_stability_ablation_noise_sweep.png")
    grouped = load_grouped(args.csv)
    present = [variant for variant in ORDER if grouped[variant]]
    if not present:
        raise ValueError(f"No known variants in {args.csv}")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 11,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), dpi=220)

    ax = axes[0]
    for variant in present:
        plot_curve(ax, variant, grouped[variant], relative=False)
    ax.set_xlabel(r"Gaussian noise $\sigma$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([index / 10 for index in range(11)])
    ax.set_ylim(30, 93)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", frameon=True)
    ax.set_title("Absolute accuracy")

    ax = axes[1]
    for variant in present:
        plot_curve(ax, variant, grouped[variant], relative=True)
    ax.set_xlabel(r"Gaussian noise $\sigma$")
    ax.set_ylabel("Retention (% of clean)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([index / 10 for index in range(11)])
    ax.set_ylim(30, 103)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", frameon=True)
    ax.set_title("Relative retention")

    if args.title:
        fig.suptitle(args.title, fontsize=16, y=1.02)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {out}")

    zoom_out = out.with_name(out.stem + "_mne_zoom.png")
    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=220)
    for variant in present:
        if variant == "weight_decay":
            continue
        plot_curve(ax, variant, grouped[variant], relative=False)
    ax.set_xlabel(r"Gaussian noise $\sigma$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([index / 10 for index in range(11)])
    ax.set_ylim(85.4, 91.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", frameon=True)
    if args.title:
        ax.set_title(args.title + " (MNE zoom)")
    fig.tight_layout()
    fig.savefig(zoom_out, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {zoom_out}")


if __name__ == "__main__":
    main()
