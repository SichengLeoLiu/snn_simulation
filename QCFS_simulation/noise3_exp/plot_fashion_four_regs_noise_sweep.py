#!/usr/bin/env python3
"""Plot Fashion-MNIST four-method post-IF noise curves (CIFAR overlay style)."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ORDER = [
    "weight_decay",
    "weight_decay_weights_only",
    "old_detach",
    "calibrated_mne_a0p1",
]

STYLES = {
    "weight_decay": {
        "label": "L2-all (all params)",
        "color": "#0072B2",
        "linestyle": "--",
    },
    "weight_decay_weights_only": {
        "label": "L2-wo (weights only)",
        "color": "#009E73",
        "linestyle": "-",
    },
    "old_detach": {
        "label": r"Old MNE (detach $\lambda$)",
        "color": "#E69F00",
        "linestyle": "-",
    },
    "calibrated_mne_a0p1": {
        "label": r"Calibrated MNE ($\alpha = 0.1$)",
        "color": "#6A3D9A",
        "linestyle": "-",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--title",
        default=r"Fashion-MNIST cnn2 seed42 · post-IF noise $\sigma \in [0, 3]$",
    )
    parser.add_argument(
        "--panel-by-arch",
        action="store_true",
        help="Draw one subplot per architecture in the CSV.",
    )
    return parser.parse_args()


def load_grouped(csv_path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            method = row.get("method") or row.get("variant")
            if method in STYLES:
                grouped[method].append(row)
    return grouped


def _acc_key(rows: list[dict]) -> str:
    return "acc_mean" if rows and "acc_mean" in rows[0] else "acc"


def _plot_axes(ax, grouped: dict[str, list[dict]], present: list[str], acc_key: str) -> float:
    xmax = 0.0
    for method in present:
        rows = sorted(grouped[method], key=lambda row: float(row["sigma"]))
        sigma = [float(row["sigma"]) for row in rows]
        acc = [float(row[acc_key]) for row in rows]
        xmax = max(xmax, max(sigma) if sigma else 0.0)
        style = STYLES[method]
        if acc_key == "acc_mean" and rows and "acc_std" in rows[0]:
            std = [float(row["acc_std"]) for row in rows]
            if any(value > 0 for value in std):
                lo = [a - s for a, s in zip(acc, std)]
                hi = [a + s for a, s in zip(acc, std)]
                ax.fill_between(sigma, lo, hi, color=style["color"], alpha=0.18, linewidth=0)
        ax.plot(
            sigma,
            acc,
            marker="o",
            linewidth=2.2,
            markersize=5.0,
            color=style["color"],
            linestyle=style["linestyle"],
            label=style["label"],
        )
    return xmax


def _style_axes(ax, xmax: float, ymin: float, ymax: float, title: str, *, legend: bool) -> None:
    ax.set_xlabel(r"Gaussian noise $\sigma$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(-0.08, xmax + 0.08)
    ax.set_xticks([index * 0.5 for index in range(int(round(xmax / 0.5)) + 1)])
    ax.set_ylim(ymin, ymax)
    ax.grid(alpha=0.3)
    if legend:
        ax.legend(loc="lower left", frameon=True, fontsize=8)
    ax.set_title(title)


def main() -> None:
    args = parse_args()
    out = args.out or args.csv.with_name("fashion_four_regs_noise_sweep.png")
    with args.csv.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    if not all_rows:
        raise ValueError(f"Empty CSV: {args.csv}")

    acc_key = _acc_key(all_rows)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )

    if args.panel_by_arch and any(row.get("arch") for row in all_rows):
        arches = []
        for row in all_rows:
            arch = row.get("arch") or ""
            if arch and arch not in arches:
                arches.append(arch)
        n = len(arches)
        ncols = 2 if n > 1 else 1
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(8.2 * ncols, 5.2 * nrows), dpi=200, squeeze=False
        )
        for index, arch in enumerate(arches):
            ax = axes[index // ncols][index % ncols]
            grouped: dict[str, list[dict]] = defaultdict(list)
            for row in all_rows:
                if row.get("arch") == arch and (row.get("method") or row.get("variant")) in STYLES:
                    grouped[row.get("method") or row.get("variant")].append(row)
            present = [method for method in ORDER if grouped[method]]
            abs_vals = [
                float(row[acc_key]) for method in present for row in grouped[method]
            ]
            ymin = max(0.0, min(abs_vals) - 5.0)
            ymax = min(100.0, max(abs_vals) + 3.0)
            xmax = _plot_axes(ax, grouped, present, acc_key)
            _style_axes(ax, xmax, ymin, ymax, arch, legend=(index == 0))
        for index in range(n, nrows * ncols):
            axes[index // ncols][index % ncols].axis("off")
        fig.suptitle(args.title, fontsize=16, y=1.01)
        fig.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] {out}")
        return

    grouped = load_grouped(args.csv)
    present = [method for method in ORDER if grouped[method]]
    if not present:
        raise ValueError(f"No known methods in {args.csv}")
    abs_vals = [
        float(row[acc_key]) for method in present for row in grouped[method]
    ]
    ymin = max(0.0, min(abs_vals) - 5.0)
    ymax = min(100.0, max(abs_vals) + 3.0)

    fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=220)
    xmax = _plot_axes(ax, grouped, present, acc_key)
    _style_axes(ax, xmax, ymin, ymax, args.title, legend=True)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {out}")

    zoom_limit = 1.5
    zoom_vals = []
    zoom_series = []
    for method in present:
        rows = sorted(grouped[method], key=lambda row: float(row["sigma"]))
        rows = [row for row in rows if float(row["sigma"]) <= zoom_limit + 1e-9]
        if not rows:
            continue
        zoom_series.append((method, rows))
        zoom_vals.extend(float(row[acc_key]) for row in rows)
    if zoom_series and zoom_vals:
        zoom_out = out.with_name(out.stem + "_sigma0_1p5.png")
        fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=220)
        dummy = {method: rows for method, rows in zoom_series}
        xmax = _plot_axes(ax, dummy, [method for method, _ in zoom_series], acc_key)
        _style_axes(
            ax,
            min(xmax, zoom_limit),
            max(0.0, min(zoom_vals) - 4.0),
            min(100.0, max(zoom_vals) + 2.0),
            args.title.replace(r"$\sigma \in [0, 3]$", r"$\sigma \in [0, 1.5]$"),
            legend=True,
        )
        ax.set_xlim(-0.05, zoom_limit + 0.05)
        ax.set_xticks([index * 0.25 for index in range(int(round(zoom_limit / 0.25)) + 1)])
        fig.tight_layout()
        fig.savefig(zoom_out, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] {zoom_out}")


if __name__ == "__main__":
    main()
