#!/usr/bin/env python3
"""Plot whole-model BN gamma distributions from analyze_bn_gamma_stats output."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "L1-all": "#0072B2",
    "L1 (all params)": "#0072B2",
    "MNE-all": "#009E73",
    "MNE-L2 (all params)": "#009E73",
    "L2-all": "#D55E00",
    "L2 (all params)": "#D55E00",
    "MNE-standard": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to bn_gamma_values.csv")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to the CSV directory.",
    )
    return parser.parse_args()


def load_values(path: Path) -> dict[str, np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"method", "gamma"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Expected columns {sorted(required)} in {path}")
        for row in reader:
            grouped[row["method"]].append(float(row["gamma"]))
    if not grouped:
        raise ValueError(f"No gamma values found in {path}")
    return {method: np.asarray(values) for method, values in grouped.items()}


def main() -> None:
    args = parse_args()
    values_by_method = load_values(args.csv)
    out_dir = args.out_dir or args.csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    epsilon = 1e-6
    all_log_abs = np.concatenate(
        [np.log10(np.abs(values) + epsilon) for values in values_by_method.values()]
    )
    bins = np.linspace(all_log_abs.min(), all_log_abs.max(), 80)
    fallback_colors = plt.get_cmap("tab10").colors

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7))
    for method_index, (method, values) in enumerate(values_by_method.items()):
        color = COLORS.get(method, fallback_colors[method_index % len(fallback_colors)])
        abs_values = np.abs(values)

        axes[0].hist(
            np.log10(abs_values + epsilon),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=color,
            label=method,
        )

        sorted_abs = np.sort(abs_values)
        cdf = np.arange(1, sorted_abs.size + 1) / sorted_abs.size
        axes[1].step(
            sorted_abs,
            cdf,
            where="post",
            linewidth=2.2,
            color=color,
            label=method,
        )

    axes[0].set_xlabel(r"$\log_{10}(|\gamma| + 10^{-6})$")
    axes[0].set_ylabel("Probability density")
    axes[0].grid(True, alpha=0.25)

    axes[1].set_xscale("symlog", linthresh=1e-3)
    axes[1].set_xlabel(r"Absolute BN scale $|\gamma|$")
    axes[1].set_ylabel(r"Cumulative fraction $P(|\gamma_i| \leq x)$")
    axes[1].set_ylim(0.0, 1.01)
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=9)

    fig.tight_layout()
    stem = out_dir / "bn_gamma_whole_model_distribution"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    for method, values in values_by_method.items():
        print(
            f"{method:16s} n={values.size:4d} "
            f"median|gamma|={np.median(np.abs(values)):.6f}"
        )
    print(f"[DONE] {stem.with_suffix('.png')}")
    print(f"[DONE] {stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
