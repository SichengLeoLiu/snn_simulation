"""Regenerate the CIFAR VGG16 robustness plots used in the paper.

The paper figure should use the original rate_uniform, L=16, T=16 CIFAR
noise-sweep CSVs under all_results_from_gadi. This script keeps that data fixed
and only standardizes the method order/colors for the manuscript.
"""
from __future__ import annotations

import csv
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PAPER_IMAGES = (
    ROOT.parent.parent
    / "Formatting_Instructions_for_ICLR_2026_Conference_Submissions1_2"
    / "images"
)

RAW_CSVS = {
    "cifar10": ROOT
    / "all_results_from_gadi"
    / "noise3_exp"
    / "cifar10_vgg16_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16"
    / "cifar10_vgg16_strict_seed_three_regs_noise_sweep_raw.csv",
    "cifar100": ROOT
    / "all_results_from_gadi"
    / "cifar100_vgg16_strict_seed_three_regs_noise_sweep_raw.csv",
}

OUTPUT_NAMES = {
    "cifar10": "cifar10_vgg16_strict_seed_three_regs_noise_sweep_mean_std_lineplot_no_caption.png",
    "cifar100": "cifar100_vgg16_strict_seed_three_regs_noise_sweep_mean_std_lineplot_no_caption.png",
}

METHODS = ("mne_l2", "weight_decay", "no_regularization")
LABELS = {
    "mne_l2": "MNE-L2 (Ours)",
    "weight_decay": "L2",
    "no_regularization": "No Reg",
}
COLORS = {
    "mne_l2": "#ff7f0e",
    "weight_decay": "#1f77b4",
    "no_regularization": "#2ca02c",
}


def aggregate(path: Path) -> dict[str, tuple[list[float], list[float], list[float]]]:
    buckets: dict[tuple[str, float], list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            method = row["method"]
            if method not in METHODS:
                continue
            sigma = round(float(row["sigma"]), 6)
            buckets[(method, sigma)].append(float(row["acc"]))

    series: dict[str, tuple[list[float], list[float], list[float]]] = {}
    for method in METHODS:
        sigmas = sorted(sigma for m, sigma in buckets if m == method)
        means: list[float] = []
        stds: list[float] = []
        for sigma in sigmas:
            values = buckets[(method, sigma)]
            means.append(statistics.mean(values))
            stds.append(statistics.stdev(values) if len(values) > 1 else 0.0)
        series[method] = (sigmas, means, stds)
    return series


def plot_dataset(dataset: str, raw_csv: Path, out_name: str) -> None:
    data = aggregate(raw_csv)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )

    fig, ax = plt.subplots(figsize=(7.1, 4.4), dpi=240)
    lower: list[float] = []
    upper: list[float] = []

    for method in METHODS:
        sigmas, means, stds = data[method]
        if not sigmas:
            continue
        color = COLORS[method]
        ax.plot(
            sigmas,
            means,
            marker="o",
            linewidth=2.0,
            markersize=4.0,
            color=color,
            label=LABELS[method],
        )
        if any(std > 0 for std in stds):
            low = [mean - std for mean, std in zip(means, stds)]
            high = [mean + std for mean, std in zip(means, stds)]
            ax.fill_between(sigmas, low, high, color=color, alpha=0.16, linewidth=0)
            lower.extend(low)
            upper.extend(high)
        else:
            lower.extend(means)
            upper.extend(means)

    ax.set_xlabel("Gaussian noise sigma")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([round(i * 0.1, 1) for i in range(11)])
    if lower and upper:
        ax.set_ylim(min(lower) - 1.0, max(upper) + 1.0)
    ax.grid(alpha=0.28)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()

    local_out = ROOT / out_name
    fig.savefig(local_out, bbox_inches="tight")
    plt.close(fig)

    PAPER_IMAGES.mkdir(parents=True, exist_ok=True)
    paper_out = PAPER_IMAGES / out_name
    shutil.copy2(local_out, paper_out)
    print(f"[{dataset}] wrote {local_out}")
    print(f"[{dataset}] copied {paper_out}")


def main() -> None:
    for dataset, raw_csv in RAW_CSVS.items():
        plot_dataset(dataset, raw_csv, OUTPUT_NAMES[dataset])


if __name__ == "__main__":
    main()
