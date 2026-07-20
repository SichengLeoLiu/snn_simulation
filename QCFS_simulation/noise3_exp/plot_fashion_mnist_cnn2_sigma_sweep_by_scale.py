#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Fashion-MNIST CNN2 sigma-vs-accuracy with SEM for each scale."
    )
    parser.add_argument(
        "--input-root",
        default="../important_results/fashion_mnist_cnn2_three_regs/noise_sweep_rate_uniform_L8_T8",
        help="Root directory containing arch/reg/seed subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        default="../plots/fashion_mnist_cnn2_sigma_sweep_by_scale_L8_T8",
        help="Directory to save output figures.",
    )
    parser.add_argument("--L", type=int, default=8, help="L row to read from matrix CSV.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument(
        "--filename-suffix",
        default="_L8",
        help="Suffix inserted before .png, e.g. _L8 or empty string.",
    )
    parser.add_argument(
        "--use-errorbar-style",
        action="store_true",
        help="Use marker+errorbar style instead of line+band style.",
    )
    parser.add_argument(
        "--arch-list",
        nargs="+",
        default=["cnn2_c2_c4", "cnn2_c4_c8", "cnn2_c8_c16", "cnn2_c16_c32"],
    )
    return parser.parse_args()


def _find_matrix_csv(seed_dir: Path, seed: int) -> Path:
    pattern = f"noise_sweep_matrix_*_seed_{seed}.csv"
    matches = sorted(seed_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing matrix CSV in {seed_dir}")
    return matches[0]


def _read_sigma_acc(matrix_csv: Path, target_l: int) -> tuple[list[float], list[float]]:
    df = pd.read_csv(matrix_csv)
    l_col = str(df.columns[0])
    l_values = pd.to_numeric(df[l_col], errors="coerce")
    match = df[l_values == target_l]
    row = match.iloc[0] if not match.empty else df.iloc[0]
    sigma_vals = [float(c) for c in df.columns[1:]]
    acc_vals = [float(row[c]) for c in df.columns[1:]]
    return sigma_vals, acc_vals


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reg_order = ["mne_l2", "weight_decay", "no_regularization"]
    reg_label = {
        "mne_l2": "MNE-L2",
        "weight_decay": "L2",
        "no_regularization": "No Reg",
    }
    reg_color = {
        "mne_l2": "#ff7f0e",
        "weight_decay": "#1f77b4",
        "no_regularization": "#2ca02c",
    }

    for arch in args.arch_list:
        plt.figure(figsize=(7.0, 4.5))
        for reg in reg_order:
            curves: list[list[float]] = []
            sigma_vals: list[float] | None = None
            for seed in args.seeds:
                seed_dir = input_root / arch / reg / f"seed_{seed}"
                matrix_csv = _find_matrix_csv(seed_dir, seed)
                sigma, acc = _read_sigma_acc(matrix_csv, args.L)
                sigma_vals = sigma
                curves.append(acc)

            assert sigma_vals is not None
            n = len(curves)
            means = [sum(vals) / n for vals in zip(*curves)]
            sems = []
            for vals in zip(*curves):
                vals_list = list(vals)
                if n <= 1:
                    sems.append(0.0)
                else:
                    mu = sum(vals_list) / n
                    var = sum((x - mu) ** 2 for x in vals_list) / (n - 1)
                    sems.append(math.sqrt(var / n))

            if args.use_errorbar_style:
                plt.errorbar(
                    sigma_vals,
                    means,
                    yerr=sems,
                    marker="o",
                    linewidth=1.8,
                    capsize=3.0,
                    color=reg_color[reg],
                    label=reg_label[reg],
                )
            else:
                upper = [m + s for m, s in zip(means, sems)]
                lower = [m - s for m, s in zip(means, sems)]
                plt.plot(
                    sigma_vals,
                    means,
                    linewidth=2.0,
                    marker="o",
                    markersize=3.8,
                    color=reg_color[reg],
                    label=reg_label[reg],
                )
                plt.fill_between(
                    sigma_vals,
                    lower,
                    upper,
                    color=reg_color[reg],
                    alpha=0.18,
                    linewidth=0.0,
                )

        plt.xlabel("Gaussian noise sigma")
        plt.ylabel("Accuracy (%)")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend(frameon=False)
        plt.tight_layout()

        out_png = output_dir / f"{arch}_three_regs_sigma_vs_acc_with_sem{args.filename_suffix}.png"
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"[OK] saved {out_png}")


if __name__ == "__main__":
    main()
