#!/usr/bin/env python3
"""Redraw retained-accuracy robustness-score bars used in the paper."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent

METHODS = ["mne_l2", "weight_decay", "no_regularization"]
METHOD_LABELS = {
    "mne_l2": "MNE-L2 (Ours)",
    "weight_decay": "L2",
    "no_regularization": "No Reg",
}
METHOD_COLORS = {
    "mne_l2": "#ff7f0e",
    "weight_decay": "#1f77b4",
    "no_regularization": "#2ca02c",
}


def setup_style(font_size: int = 16) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": font_size,
            "axes.labelsize": font_size + 2,
            "legend.fontsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
        }
    )


def tight_ylim(values: list[float], *, top: float = 1.02) -> tuple[float, float]:
    spread = max(values) - min(values)
    pad = max(0.006, 0.18 * spread)
    return max(0.0, min(values) - pad), min(top, max(values) + pad)


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def grouped_bar(
    rows: list[dict],
    arch_keys: list[str],
    arch_labels: list[str],
    *,
    arch_field: str,
    method_field: str,
    xlabel: str,
    out_base: Path,
) -> None:
    setup_style(16)
    x = np.arange(len(arch_keys))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(10.0, len(arch_keys) * 2.2), 6.8), dpi=220)
    vals: list[float] = []

    for idx, method in enumerate(METHODS):
        means, errs = [], []
        for arch in arch_keys:
            row = next(
                (r for r in rows if r[arch_field] == arch and r[method_field] == method),
                None,
            )
            mean = float(row["RS_mean"]) if row else np.nan
            err = float(row["RS_sem"]) if row else 0.0
            means.append(mean)
            errs.append(err)
            if not np.isnan(mean):
                vals.append(mean)

        xpos = x + (idx - 1) * width
        bars = ax.bar(
            xpos,
            means,
            width,
            yerr=errs,
            capsize=3,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
            label=METHOD_LABELS[method],
        )
        for b, v in zip(bars, means):
            if np.isnan(v):
                continue
            ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(arch_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Robustness Score")
    if vals:
        ax.set_ylim(*tight_ylim(vals))
    ax.grid(axis="y", alpha=0.25, linewidth=0.9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        out = out_base.with_suffix(ext)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        print(f"[SAVED] {out}")
    plt.close(fig)


def single_bar(rows: list[dict], *, out_base: Path) -> None:
    setup_style(18)
    method_map = {"MNE-L2": "mne_l2", "MNE-L2 (Ours)": "mne_l2", "L2": "weight_decay", "No Reg": "no_regularization"}
    x = np.arange(len(METHODS))
    means = []
    for method in METHODS:
        row = next((r for r in rows if method_map.get(r["method"], r["method"]) == method), None)
        means.append(float(row["RS"]) if row else np.nan)

    fig, ax = plt.subplots(figsize=(8.8, 6.8), dpi=220)
    bars = ax.bar(
        x,
        means,
        width=0.55,
        color=[METHOD_COLORS[m] for m in METHODS],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.92,
    )
    for b, v in zip(bars, means):
        if np.isnan(v):
            continue
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS])
    ax.set_ylabel("Robustness Score")
    vals = [v for v in means if not np.isnan(v)]
    if vals:
        ax.set_ylim(*tight_ylim(vals, top=1.08))
    ax.grid(axis="y", alpha=0.24, linewidth=0.9)
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        out = out_base.with_suffix(ext)
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        print(f"[SAVED] {out}")
    plt.close(fig)


def main() -> None:
    cnn_csv = ROOT / "all_results_from_gadi" / "cnn2_noise_sweep_step0p05_plots" / "cnn2_three_regs_robustness_score.csv"
    grouped_bar(
        load_rows(cnn_csv),
        ["cnn2_c2_c4", "cnn2_c4_c8", "cnn2_c8_c16", "cnn2_c16_c32"],
        ["c2c4", "c4c8", "c8c16", "c16c32"],
        arch_field="arch",
        method_field="method",
        xlabel="CNN2 model scale",
        out_base=cnn_csv.parent / "cnn2_three_regs_robustness_score_bar_with_sem",
    )

    fc_csv = ROOT / "important results" / "new_fc3" / "plots" / "fc3rev_three_regs_robustness_score.csv"
    grouped_bar(
        [r for r in load_rows(fc_csv) if int(r["hidden_size"]) in (8, 16, 32)],
        ["fc3rev_h8", "fc3rev_h16", "fc3rev_h32"],
        ["h8", "h16", "h32"],
        arch_field="arch",
        method_field="regularizer",
        xlabel="Hidden Size",
        out_base=fc_csv.parent / "fc3rev_three_regs_robustness_score_bar_with_sem",
    )

    imagenet_csv = ROOT / "imagenet_resnet18_three_regs_robustness_score.csv"
    single_bar(
        load_rows(imagenet_csv),
        out_base=ROOT / "imagenet_resnet18_three_regs_robustness_score_bar",
    )


if __name__ == "__main__":
    main()
