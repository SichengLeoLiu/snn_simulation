#!/usr/bin/env python3
"""Generate DRS bar charts for FC3rev, CNN2, CIFAR, and ImageNet."""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from robustness_metrics import derivative_robustness_score

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "drs_results" / "plots"

METHODS = ["weight_decay", "mne_l2", "no_regularization"]
METHOD_LABELS = {
    "weight_decay": "L2",
    "mne_l2": "MNE-L2",
    "no_regularization": "No Reg",
}
METHOD_COLORS = {
    "weight_decay": "#ff7f0e",
    "mne_l2": "#1f77b4",
    "no_regularization": "#2ca02c",
}

FC3REV_RAW = ROOT / "important results" / "new_fc3" / "fc3rev_h8_h256_three_regs_noise_sweep_raw.csv"
FC3REV_H_LIST = [8, 16, 32]

CNN2_ROOT = ROOT / "all_results_from_gadi" / "cnn2_noise_sweep_step0p05_full_extracted"
CNN2_ARCHES = ["cnn2_c2_c4", "cnn2_c4_c8", "cnn2_c8_c16", "cnn2_c16_c32"]
CNN2_LABELS = {
    "cnn2_c2_c4": "c2c4",
    "cnn2_c4_c8": "c4c8",
    "cnn2_c8_c16": "c8c16",
    "cnn2_c16_c32": "c16c32",
}

CIFAR_RAW = {
    "cifar10": ROOT
    / "all_results_from_gadi"
    / "noise3_exp"
    / "cifar10_vgg16_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16"
    / "cifar10_vgg16_strict_seed_three_regs_noise_sweep_raw.csv",
    "cifar100": ROOT / "all_results_from_gadi" / "cifar100_vgg16_strict_seed_three_regs_noise_sweep_raw.csv",
}

IMAGENET_L2_MNE = ROOT / "imagenet_resnet18_l2_vs_mnel2_combined.csv"
IMAGENET_NO_REG = ROOT / "imagenet_resnet18_no_reg_noise_sweep.csv"


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


def read_matrix_csv(path: Path) -> tuple[list[float], list[float]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)
    start = 2 if header[:2] == ["L", "T"] else 1
    return [float(x) for x in header[start:]], [float(x) for x in row[start:]]


def load_raw_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate_from_raw(
    raw_rows: list[dict],
    group_keys: tuple[str, ...],
    *,
    method_key: str = "regularizer",
) -> list[dict]:
    buckets: dict[tuple, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    meta: dict[tuple, dict] = {}
    for r in raw_rows:
        key = tuple(r[k] for k in group_keys)
        buckets[key][int(r["seed"])].append((float(r["sigma"]), float(r["acc"])))
        if key not in meta:
            meta[key] = {k: r[k] for k in group_keys}

    rows = []
    for key in sorted(buckets):
        vals = []
        for seed in sorted(buckets[key]):
            pairs = sorted(buckets[key][seed], key=lambda x: x[0])
            vals.append(derivative_robustness_score([p[0] for p in pairs], [p[1] for p in pairs]))
        n = len(vals)
        std = statistics.stdev(vals) if n > 1 else 0.0
        rows.append({
            **meta[key],
            "method": meta[key].get(method_key, meta[key].get("method", "")),
            "DRS_mean": statistics.mean(vals),
            "DRS_std": std,
            "DRS_sem": std / (n ** 0.5) if n else 0.0,
            "n_seeds": n,
        })
    return rows


def grouped_bar(
    arch_keys: list[str],
    arch_labels: list[str],
    rows: list[dict],
    out_dir: Path,
    *,
    title: str,
    xlabel: str,
    out_stem: str,
    with_sem: bool = True,
    method_field: str = "method",
) -> None:
    setup_style(16)
    x = np.arange(len(arch_keys))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(10.0, len(arch_keys) * 2.2), 6.8), dpi=220)
    all_means = []

    for idx, method in enumerate(METHODS):
        means, errs = [], []
        for arch in arch_keys:
            row = next(
                (r for r in rows if r.get("arch") == arch and r.get(method_field) == method),
                None,
            )
            if row is None:
                row = next(
                    (r for r in rows if r.get("arch_label") == arch and r.get(method_field) == method),
                    None,
                )
            m = row["DRS_mean"] if row else np.nan
            means.append(m)
            errs.append(row["DRS_sem"] if row and with_sem else 0.0)
            if not np.isnan(m):
                all_means.append(m)

        xpos = x + (idx - 1) * width
        bars = ax.bar(
            xpos, means, width,
            yerr=errs if with_sem else None,
            capsize=3 if with_sem else 0,
            color=METHOD_COLORS[method],
            edgecolor="black", linewidth=0.5, alpha=0.9,
            label=METHOD_LABELS[method],
        )
        for b, v in zip(bars, means):
            if np.isnan(v):
                continue
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(arch_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Derivative Robustness Score (DRS)")
    ax.set_title(title)
    if all_means:
        ax.set_ylim(max(0.0, min(all_means) - 0.1), min(1.05, max(all_means) + 0.1))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, frameon=False)
    fig.tight_layout()

    suffix = "_with_sem" if with_sem else ""
    for ext in (".png", ".pdf"):
        out = out_dir / f"{out_stem}{suffix}{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[SAVED] {out}")
    plt.close(fig)


def single_dataset_bar(
    rows: list[dict],
    out_dir: Path,
    *,
    title: str,
    out_stem: str,
    with_sem: bool = True,
    method_field: str = "method",
) -> None:
    setup_style(18)
    x = np.arange(len(METHODS))
    fig, ax = plt.subplots(figsize=(8.8, 6.8), dpi=220)
    means, errs, colors = [], [], []
    for method in METHODS:
        row = next((r for r in rows if r.get(method_field) == method), None)
        means.append(row["DRS_mean"] if row else np.nan)
        errs.append(row["DRS_sem"] if row and with_sem else 0.0)
        colors.append(METHOD_COLORS[method])

    bars = ax.bar(x, means, width=0.55, yerr=errs if with_sem else None,
                  capsize=3 if with_sem else 0, color=colors,
                  edgecolor="black", linewidth=0.5, alpha=0.92)
    for b, v, e in zip(bars, means, errs):
        if np.isnan(v):
            continue
        ax.text(b.get_x() + b.get_width() / 2, v + e + 0.015, f"{v:.3f}",
                ha="center", va="bottom", fontsize=13)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS])
    ax.set_ylabel("Derivative Robustness Score (DRS)")
    ax.set_title(title)
    valid = [m for m in means if not np.isnan(m)]
    if valid:
        ax.set_ylim(max(0.0, min(valid) - 0.12), min(1.08, max(valid) + 0.12))
    ax.grid(axis="y", alpha=0.24)
    fig.tight_layout()

    suffix = "_with_sem" if with_sem else ""
    for ext in (".png", ".pdf"):
        out = out_dir / f"{out_stem}{suffix}{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        print(f"[SAVED] {out}")
    plt.close(fig)


def plot_fc3rev_h8_h16_h32(out_dir: Path) -> list[dict]:
    raw = load_raw_csv(FC3REV_RAW)
    all_rows = aggregate_from_raw(raw, ("arch", "regularizer"))
    h_archs = [f"fc3rev_h{h}" for h in FC3REV_H_LIST]
    labels = [f"h{h}" for h in FC3REV_H_LIST]
    subset = [r for r in all_rows if r["arch"] in h_archs]
    grouped_bar(
        h_archs, labels, subset, out_dir,
        title="FC3rev (MNIST, 2h→h): DRS by hidden size",
        xlabel="Hidden size",
        out_stem="fc3rev_h8_h16_h32_drs_bar",
    )
    return subset


def plot_cnn2_all(out_dir: Path) -> list[dict]:
    rows = []
    for arch in CNN2_ARCHES:
        for method in METHODS:
            seed_files = sorted((CNN2_ROOT / arch / method).glob("seed_*/noise_sweep_matrix_*.csv"))
            vals = []
            for fp in seed_files:
                sigmas, accs = read_matrix_csv(fp)
                vals.append(derivative_robustness_score(sigmas, accs))
            if not vals:
                continue
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            rows.append({
                "arch": arch,
                "arch_label": CNN2_LABELS[arch],
                "method": method,
                "DRS_mean": statistics.mean(vals),
                "DRS_std": std,
                "DRS_sem": std / (len(vals) ** 0.5),
                "n_seeds": len(vals),
            })
    grouped_bar(
        CNN2_ARCHES, [CNN2_LABELS[a] for a in CNN2_ARCHES], rows, out_dir,
        title="CNN2 (MNIST): DRS by model scale",
        xlabel="CNN2 architecture",
        out_stem="cnn2_all_scales_drs_bar",
    )
    return rows


def plot_cifar(dataset: str, out_dir: Path) -> list[dict]:
    raw = load_raw_csv(CIFAR_RAW[dataset])
    rows = aggregate_from_raw(raw, ("method",), method_key="method")
    single_dataset_bar(
        rows, out_dir,
        title=f"{dataset.upper()} VGG16: DRS (three regularizers)",
        out_stem=f"{dataset}_vgg16_drs_bar",
    )
    return rows


def plot_imagenet(out_dir: Path) -> list[dict]:
    sigmas, l2, mne = [], [], []
    with IMAGENET_L2_MNE.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sigmas.append(float(r["sigma"]))
            l2.append(float(r["acc_l2_weight_decay"]))
            mne.append(float(r["acc_mne_l2_rc1e-4"]))
    sigmas_nr, no_reg = [], []
    with IMAGENET_NO_REG.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sigmas_nr.append(float(r["sigma"]))
            no_reg.append(float(r["acc"]))

    rows = [
        {"method": "weight_decay", "DRS_mean": derivative_robustness_score(sigmas, l2), "DRS_sem": 0.0},
        {"method": "mne_l2", "DRS_mean": derivative_robustness_score(sigmas, mne), "DRS_sem": 0.0},
        {"method": "no_regularization", "DRS_mean": derivative_robustness_score(sigmas_nr, no_reg), "DRS_sem": 0.0},
    ]
    single_dataset_bar(
        rows, out_dir,
        title="ImageNet ResNet18: DRS (three regularizers)",
        out_stem="imagenet_resnet18_drs_bar",
        with_sem=False,
    )
    return rows


def save_summary_csv(out_dir: Path, name: str, rows: list[dict], fields: list[str]) -> None:
    out = out_dir / f"{name}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6f}" if isinstance(r.get(k), float) else r.get(k, "")) for k in fields})
    print(f"[SAVED] {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all DRS bar charts")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fc3 = plot_fc3rev_h8_h16_h32(out_dir)
    cnn2 = plot_cnn2_all(out_dir)
    c10 = plot_cifar("cifar10", out_dir)
    c100 = plot_cifar("cifar100", out_dir)
    img = plot_imagenet(out_dir)

    save_summary_csv(out_dir, "fc3rev_h8_h16_h32_drs", fc3,
                     ["arch", "method", "DRS_mean", "DRS_std", "DRS_sem", "n_seeds"])
    save_summary_csv(out_dir, "cnn2_all_scales_drs", cnn2,
                     ["arch", "arch_label", "method", "DRS_mean", "DRS_std", "DRS_sem", "n_seeds"])
    save_summary_csv(out_dir, "cifar10_vgg16_drs", c10, ["method", "DRS_mean", "DRS_std", "DRS_sem", "n_seeds"])
    save_summary_csv(out_dir, "cifar100_vgg16_drs", c100, ["method", "DRS_mean", "DRS_std", "DRS_sem", "n_seeds"])
    save_summary_csv(out_dir, "imagenet_resnet18_drs", img, ["method", "DRS_mean"])


if __name__ == "__main__":
    main()
