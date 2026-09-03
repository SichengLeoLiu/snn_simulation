#!/usr/bin/env python3
"""Build the recent CIFAR-10 VGG16 robustness-mechanism figure set."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve()
QCFS_ROOT = HERE.parents[1]
REPO_ROOT = QCFS_ROOT.parent

METHOD_ORDER = [
    "MNE-standard",
    "Weights-only",
    "L1-all",
    "MNE-all",
    "L2-all",
    "Weights+BN-gamma",
]

COLORS = {
    "MNE-standard": "#0072B2",
    "Weights-only": "#009E73",
    "L1-all": "#E69F00",
    "MNE-all": "#CC79A7",
    "L2-all": "#D55E00",
    "Weights+BN-gamma": "#6F4E9C",
    "mne_l2": "#0072B2",
    "weight_decay": "#D55E00",
    "l1": "#E69F00",
}

DISPLAY = {
    "mne_l2": "MNE-L2",
    "weight_decay": "Weight decay",
    "l1": "L1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relative-log", type=Path, required=True)
    parser.add_argument("--layerwise-log", type=Path, required=True)
    parser.add_argument(
        "--pre-csv",
        type=Path,
        default=REPO_ROOT
        / "all_results_from_gadi"
        / "cifar10_vgg16_three_regs_l1_l2_mne_pre_input_if"
        / "cifar10_vgg16_strict_seed_three_regs_noise_sweep_mean_std.csv",
    )
    parser.add_argument(
        "--post-csv",
        type=Path,
        default=REPO_ROOT
        / "all_results_from_gadi"
        / "cifar10_vgg16_three_regs_l1_l2_mne_post_input_if"
        / "cifar10_vgg16_strict_seed_three_regs_noise_sweep_mean_std.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT
        / "important_results"
        / "recent_cifar10_mechanism_summary",
    )
    return parser.parse_args()


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {
                key: f"{value:.6f}" if isinstance(value, float) else value
                for key, value in row.items()
            }
            writer.writerow(formatted)


def parse_relative_log(path: Path) -> tuple[list[dict], dict[str, dict[str, float]]]:
    method = None
    scales: dict[str, dict[str, float]] = {}
    rows: list[dict] = []
    method_re = re.compile(r"^\[METHOD\]\s+(.+?)\s+<-")
    scale_re = re.compile(
        r"lambda_input=([0-9.eE+-]+)\s+RMS\(h_input\)=([0-9.eE+-]+)"
    )
    result_re = re.compile(
        r"mode=(\S+)\s+level=(\S+)\s+actual_.=([0-9.eE+-]+)\s+"
        r"repeat=(\d+)\s+acc=([0-9.eE+-]+)"
    )

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        match = method_re.match(line)
        if match:
            method = match.group(1)
            continue
        if method is None:
            continue
        match = scale_re.search(
            line.replace("lambda_input", "lambda_input").replace("\u03bb_input", "lambda_input")
        )
        if match:
            scales[method] = {
                "lambda_input": float(match.group(1)),
                "rms_input": float(match.group(2)),
            }
            continue
        match = result_re.search(line)
        if match:
            rows.append(
                {
                    "method": method,
                    "scale_mode": match.group(1),
                    "level_name": match.group(2),
                    "actual_sigma": float(match.group(3)),
                    "repeat": int(match.group(4)),
                    "accuracy": float(match.group(5)),
                }
            )
    if not rows or not scales:
        raise RuntimeError(f"Could not parse relative-noise log: {path}")
    return rows, scales


def aggregate_relative(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["scale_mode"], row["level_name"])].append(row)

    result = []
    for (method, mode, level), values in grouped.items():
        accs = [float(v["accuracy"]) for v in values]
        sigmas = [float(v["actual_sigma"]) for v in values]
        result.append(
            {
                "method": method,
                "scale_mode": mode,
                "level_name": level,
                "actual_sigma": statistics.mean(sigmas),
                "n_repeats": len(accs),
                "acc_mean": statistics.mean(accs),
                "acc_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            }
        )
    return result


def parse_layerwise_log(path: Path) -> list[dict]:
    method = None
    rows: list[dict] = []
    method_re = re.compile(r"^\[METHOD\]\s+(.+?)\s+<-")
    layer_re = re.compile(
        r"^\[(\d+)\]\s+(\S+)\s+lambda=([0-9.eE+-]+)\s+"
        r"RMS\(gamma\)/lambda=(\S+)\s+Gfro=(\S+)\s+Gspec=(\S+)\s+"
        r"actRMS=(\S+)\s+sigmaeff=(\S+)\s+d50=(\S+)\s+rho=(\S+)\s+"
        r"P\(E\)=(\S+)"
    )

    def value(text: str) -> float:
        return float("nan") if text.lower() == "nan" else float(text)

    replacements = {
        "\u03bb": "lambda",
        "\u03b3": "gamma",
        "\u03c3eff": "sigmaeff",
        "\u03c1": "rho",
    }
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        match = method_re.match(line)
        if match:
            method = match.group(1)
            continue
        if method is None:
            continue
        normalized = line
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        match = layer_re.match(normalized)
        if not match:
            continue
        rows.append(
            {
                "method": method,
                "layer_index": int(match.group(1)),
                "layer": match.group(2),
                "lambda": value(match.group(3)),
                "rms_gamma_over_lambda": value(match.group(4)),
                "g_fro": value(match.group(5)),
                "g_spec": value(match.group(6)),
                "activation_rms": value(match.group(7)),
                "sigma_eff": value(match.group(8)),
                "d50": value(match.group(9)),
                "rho": value(match.group(10)),
                # The source script uses clean.ne(noisy), so this is crossing probability.
                "p_cross": value(match.group(11)),
            }
        )
    if not rows:
        raise RuntimeError(f"Could not parse layerwise log: {path}")
    return rows


def plot_pre_post(pre_rows: list[dict], post_rows: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, rows, title in zip(
        axes,
        [pre_rows, post_rows],
        ["Pre-IF Gaussian noise", "Post-IF Gaussian noise"],
    ):
        for method in ["mne_l2", "weight_decay", "l1"]:
            pts = sorted(
                [row for row in rows if row["method"] == method],
                key=lambda row: float(row["sigma"]),
            )
            x = np.array([float(row["sigma"]) for row in pts])
            y = np.array([float(row["acc_mean"]) for row in pts])
            std = np.array([float(row["acc_std"]) for row in pts])
            ax.plot(x, y, marker="o", markersize=3, linewidth=2, color=COLORS[method], label=DISPLAY[method])
            ax.fill_between(x, y - std, y + std, color=COLORS[method], alpha=0.13, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Absolute Gaussian noise sigma")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlim(0, 1)
        ax.legend(frameon=False)
    axes[0].set_ylim(82, 91.2)
    axes[1].set_ylim(15, 92)
    fig.suptitle("Noise location changes the apparent robustness gap (5 seeds)", y=1.02)
    save_figure(fig, out_dir, "fig1_pre_vs_post_absolute_noise")


def lookup(aggregated: list[dict], method: str, mode: str, level: str) -> dict:
    for row in aggregated:
        if (
            row["method"] == method
            and row["scale_mode"] == mode
            and row["level_name"] == level
        ):
            return row
    raise KeyError((method, mode, level))


def plot_absolute_vs_snr(aggregated: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    for method in METHOD_ORDER:
        pts = sorted(
            [
                row
                for row in aggregated
                if row["method"] == method and row["scale_mode"] == "absolute"
            ],
            key=lambda row: float(row["level_name"].split("_")[1]),
        )
        x = [float(row["level_name"].split("_")[1]) for row in pts]
        y = [row["acc_mean"] for row in pts]
        std = [row["acc_std"] for row in pts]
        ax.errorbar(x, y, yerr=std, marker="o", markersize=4, linewidth=2, capsize=2, color=COLORS[method], label=method)
    ax.set_title("Fixed absolute noise")
    ax.set_xlabel("Absolute Gaussian noise sigma")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(0, 1)
    ax.set_ylim(15, 93)

    ax = axes[1]
    levels = ["clean", "snr_30dB", "snr_20dB", "snr_15dB", "snr_10dB", "snr_5dB", "snr_0dB"]
    labels = ["clean", "30", "20", "15", "10", "5", "0"]
    x = np.arange(len(levels))
    for method in METHOD_ORDER:
        pts = [lookup(aggregated, method, "snr", level) for level in levels]
        y = [row["acc_mean"] for row in pts]
        std = [row["acc_std"] for row in pts]
        ax.errorbar(x, y, yerr=std, marker="o", markersize=4, linewidth=2, capsize=2, color=COLORS[method], label=method)
    ax.set_title("Activation-SNR-matched noise")
    ax.set_xlabel("SNR (dB; lower means stronger relative noise)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(88, 92)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Scale matching removes the catastrophic deficit within the tested range (seed 42, 5 noise repeats)", y=1.02)
    fig.subplots_adjust(bottom=0.2)
    save_figure(fig, out_dir, "fig2_absolute_vs_snr_matched_noise")


def build_scale_summary(aggregated: list[dict], scales: dict[str, dict[str, float]]) -> list[dict]:
    rows = []
    for method in METHOD_ORDER:
        clean = lookup(aggregated, method, "snr", "clean")["acc_mean"]
        absolute = lookup(aggregated, method, "absolute", "sigma_1")["acc_mean"]
        snr_zero = lookup(aggregated, method, "snr", "snr_0dB")["acc_mean"]
        lam = scales[method]["lambda_input"]
        rms = scales[method]["rms_input"]
        rows.append(
            {
                "method": method,
                "lambda_input": lam,
                "rms_input": rms,
                "sigma1_effective_snr_db": 20.0 * math.log10(rms),
                "sigma1_over_lambda": 1.0 / lam,
                "clean_acc": clean,
                "absolute_sigma1_acc": absolute,
                "absolute_sigma1_drop": clean - absolute,
                "snr_0db_acc": snr_zero,
                "snr_0db_drop": clean - snr_zero,
            }
        )
    return rows


def plot_scale_relationship(summary: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    x = np.arange(len(summary))
    width = 0.36
    axes[0].bar(x - width / 2, [row["lambda_input"] for row in summary], width, label="Input IF threshold", color="#4C78A8")
    axes[0].bar(x + width / 2, [row["rms_input"] for row in summary], width, label="Post-IF activation RMS", color="#F2A541")
    axes[0].set_xticks(x, [row["method"] for row in summary], rotation=35, ha="right")
    axes[0].set_ylabel("Internal scale")
    axes[0].set_title("Injection-site scales")
    axes[0].legend(frameon=False)

    snr_offsets = {
        "MNE-standard": (-82, -18),
        "Weights-only": (-34, 12),
        "L1-all": (8, -20),
        "MNE-all": (8, 5),
        "L2-all": (8, 8),
        "Weights+BN-gamma": (8, -15),
    }
    threshold_offsets = {
        "MNE-standard": (-28, 14),
        "Weights-only": (8, -18),
        "L1-all": (6, -22),
        "MNE-all": (8, 5),
        "L2-all": (-56, 8),
        "Weights+BN-gamma": (8, -15),
    }
    for row in summary:
        axes[1].scatter(row["sigma1_effective_snr_db"], row["absolute_sigma1_drop"], s=62, color=COLORS[row["method"]], zorder=3)
        axes[1].annotate(row["method"], (row["sigma1_effective_snr_db"], row["absolute_sigma1_drop"]), xytext=snr_offsets[row["method"]], textcoords="offset points", fontsize=8)
    axes[1].set_xlabel("Effective SNR when absolute sigma = 1 (dB)")
    axes[1].set_ylabel("Accuracy drop (percentage points)")
    axes[1].set_title("Lower effective SNR, larger drop")

    axes[1].margins(x=0.12, y=0.12)

    for row in summary:
        axes[2].scatter(row["sigma1_over_lambda"], row["absolute_sigma1_drop"], s=62, color=COLORS[row["method"]], zorder=3)
        axes[2].annotate(row["method"], (row["sigma1_over_lambda"], row["absolute_sigma1_drop"]), xytext=threshold_offsets[row["method"]], textcoords="offset points", fontsize=8)
    axes[2].set_xlabel("Noise-to-threshold ratio (sigma / lambda)")
    axes[2].set_ylabel("Accuracy drop (percentage points)")
    axes[2].set_title("Threshold-relative severity")
    axes[2].margins(x=0.12, y=0.12)
    fig.suptitle("Fixed sigma compares different effective perturbation strengths", y=1.02)
    save_figure(fig, out_dir, "fig3_internal_scale_and_effective_severity")


def plot_layerwise(rows: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 7.0), sharex=True)
    layer_rows = sorted(
        [row for row in rows if row["method"] == METHOD_ORDER[0] and row["layer_index"] > 0],
        key=lambda row: row["layer_index"],
    )
    indices = [row["layer_index"] for row in layer_rows]
    labels = [row["layer"] for row in layer_rows]

    for method in METHOD_ORDER:
        pts = sorted(
            [row for row in rows if row["method"] == method and row["layer_index"] > 0],
            key=lambda row: row["layer_index"],
        )
        axes[0].plot(indices, [row["rho"] for row in pts], marker="o", markersize=3, linewidth=1.8, color=COLORS[method], label=method)
        axes[1].plot(indices, [row["p_cross"] for row in pts], marker="o", markersize=3, linewidth=1.8, color=COLORS[method], label=method)

    for ax in axes:
        ax.axvspan(7 - 0.35, 12 + 0.35, color="#B0B0B0", alpha=0.10, linewidth=0)
    axes[0].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Margin-to-noise ratio rho")
    axes[0].set_title("Late VGG layers lose normalized margin under L2-all and BN-gamma L2")
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    axes[1].set_ylabel("Empirical crossing probability P_cross")
    axes[1].set_xlabel("IF layer")
    axes[1].set_ylim(0, 0.8)
    axes[1].set_xticks(indices, labels, rotation=42, ha="right")
    axes[1].set_title("Low rho coincides with frequent clean/noisy representation changes")
    fig.suptitle("Layerwise response to post-input-IF absolute noise sigma = 1 (seed 42)", y=1.01)
    save_figure(fig, out_dir, "fig4_layerwise_margin_and_crossing")


def build_endpoint_summary(pre_rows: list[dict], post_rows: list[dict]) -> list[dict]:
    result = []
    for location, rows in [("pre_input_if", pre_rows), ("post_input_if", post_rows)]:
        for method in ["mne_l2", "weight_decay", "l1"]:
            clean = next(row for row in rows if row["method"] == method and float(row["sigma"]) == 0.0)
            noisy = next(row for row in rows if row["method"] == method and float(row["sigma"]) == 1.0)
            result.append(
                {
                    "noise_position": location,
                    "method": DISPLAY[method],
                    "n_seeds": int(clean["n_seeds"]),
                    "clean_acc_mean": float(clean["acc_mean"]),
                    "sigma1_acc_mean": float(noisy["acc_mean"]),
                    "sigma1_acc_std": float(noisy["acc_std"]),
                    "absolute_drop": float(clean["acc_mean"]) - float(noisy["acc_mean"]),
                }
            )
    return result


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    pre_rows = read_csv(args.pre_csv)
    post_rows = read_csv(args.post_csv)
    relative_raw, scales = parse_relative_log(args.relative_log)
    relative_mean = aggregate_relative(relative_raw)
    layerwise_rows = parse_layerwise_log(args.layerwise_log)
    scale_summary = build_scale_summary(relative_mean, scales)
    endpoint_summary = build_endpoint_summary(pre_rows, post_rows)

    write_csv(args.out_dir / "pre_post_sigma1_summary.csv", endpoint_summary)
    write_csv(args.out_dir / "relative_noise_mean_std.csv", relative_mean)
    write_csv(args.out_dir / "scale_severity_summary.csv", scale_summary)
    write_csv(args.out_dir / "layerwise_metrics.csv", layerwise_rows)

    plot_pre_post(pre_rows, post_rows, args.out_dir)
    plot_absolute_vs_snr(relative_mean, args.out_dir)
    plot_scale_relationship(scale_summary, args.out_dir)
    plot_layerwise(layerwise_rows, args.out_dir)

    print(f"[DONE] wrote summary figures and CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
