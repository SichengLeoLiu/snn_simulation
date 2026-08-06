from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ORDERS = {
    "parameter_scope": [
        "weight_decay_weights_only",
        "manual_l2_w_bn",
        "manual_l2_w_if",
        "manual_l2_w_bn_if",
        "manual_l2_all",
    ],
    "bn_affine": [
        "weight_decay_weights_only",
        "manual_l2_w_bn_gamma",
        "manual_l2_w_bn_beta",
        "manual_l2_w_bn",
    ],
}

STYLES = {
    "weight_decay_weights_only": {
        "label": "Weights only",
        "color": "#0072B2",
    },
    "manual_l2_w_bn": {
        "label": "Weights + BN γ + β",
        "color": "#56B4E9",
    },
    "manual_l2_w_bn_gamma": {
        "label": "Weights + BN γ",
        "color": "#009E73",
    },
    "manual_l2_w_bn_beta": {
        "label": "Weights + BN β",
        "color": "#E69F00",
    },
    "manual_l2_w_if": {
        "label": "Weights + IF threshold",
        "color": "#E69F00",
    },
    "manual_l2_w_bn_if": {
        "label": "Weights + BN + IF",
        "color": "#CC79A7",
    },
    "manual_l2_all": {
        "label": "All parameters",
        "color": "#D55E00",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot L2 parameter-scope noise sweeps from ablation mean/std CSV."
    )
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--title", default="")
    parser.add_argument(
        "--scope-set",
        choices=("auto", *ORDERS),
        default="auto",
        help="Curves to plot; auto selects BN-affine curves when gamma/beta are present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or args.csv.with_name(
        f"{args.dataset}_l2_parameter_scope_noise_sweep.png"
    )

    grouped: dict[str, list[dict]] = defaultdict(list)
    with args.csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == args.dataset and row["variant"] in STYLES:
                grouped[row["variant"]].append(row)

    scope_set = args.scope_set
    if scope_set == "auto":
        scope_set = (
            "bn_affine"
            if grouped["manual_l2_w_bn_gamma"] or grouped["manual_l2_w_bn_beta"]
            else "parameter_scope"
        )
    order = ORDERS[scope_set]
    missing = [variant for variant in order if not grouped[variant]]
    if missing:
        raise ValueError(f"Missing variants in {args.csv}: {', '.join(missing)}")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )
    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=220)

    for variant in order:
        rows = sorted(grouped[variant], key=lambda row: float(row["sigma"]))
        sigma = [float(row["sigma"]) for row in rows]
        mean = [float(row["acc_mean"]) for row in rows]
        std = [float(row["acc_std"]) for row in rows]
        style = STYLES[variant]
        ax.plot(
            sigma,
            mean,
            marker="o",
            linewidth=2.2,
            markersize=4.5,
            color=style["color"],
            label=style["label"],
        )
        if any(value > 0 for value in std):
            ax.fill_between(
                sigma,
                [value - error for value, error in zip(mean, std)],
                [value + error for value, error in zip(mean, std)],
                color=style["color"],
                alpha=0.16,
                linewidth=0,
            )

    ax.set_xlabel("Gaussian noise sigma")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([index / 10 for index in range(11)])
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=True)
    if args.title:
        ax.set_title(args.title)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {out}")


if __name__ == "__main__":
    main()
