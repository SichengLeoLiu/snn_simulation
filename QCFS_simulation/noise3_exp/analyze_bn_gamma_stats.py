#!/usr/bin/env python3
"""Summarize BN gamma distributions from checkpoints (per method / per layer)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_CKPTS = {
    "L1-all": "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_l1_all_rc1em05_seed42_L16_trainT0.pth",
    "MNE-all": "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_mne_l2_all_rc0p0001_seed42_L16_trainT0.pth",
    "L2-all": "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_manual_l2_all_rc0p00025_seed42_L16_trainT0.pth",
    "MNE-standard": "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_old_detach_rc0p0001_seed42_L16_trainT0.pth",
}


def _load_state_dict(path: Path) -> dict:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported checkpoint format: {path}")


def _extract_bn_gammas(state_dict: dict) -> list[tuple[str, torch.Tensor]]:
    layers = []
    for key, value in state_dict.items():
        if not key.endswith(".weight"):
            continue
        prefix = key[: -len("weight")]
        if prefix + "running_var" not in state_dict:
            continue
        layers.append((key, value.detach().float().reshape(-1)))
    return layers


def _stats(values: torch.Tensor) -> dict:
    abs_g = values.abs()
    active = abs_g[abs_g > 0.1]
    return {
        "n": int(abs_g.numel()),
        "median_abs": float(abs_g.median().item()),
        "rms": float(values.pow(2).mean().sqrt().item()),
        "p90_abs": float(torch.quantile(abs_g, 0.90).item()),
        "max_abs": float(abs_g.max().item()),
        "frac_lt_0p001": float((abs_g < 0.001).float().mean().item()),
        "frac_lt_0p01": float((abs_g < 0.01).float().mean().item()),
        "frac_lt_0p1": float((abs_g < 0.1).float().mean().item()),
        "active_mean_abs_gt_0p1": (
            float(active.mean().item()) if active.numel() > 0 else float("nan")
        ),
        "n_active_gt_0p1": int(active.numel()),
    }


def _plot_distributions(
    method_values: dict[str, np.ndarray],
    layer_values: dict[str, list[tuple[str, np.ndarray]]],
    out_dir: Path,
) -> None:
    colors = {
        "L1-all": "#0072B2",
        "MNE-all": "#009E73",
        "L2-all": "#D55E00",
        "MNE-standard": "#CC79A7",
    }
    fallback_colors = plt.get_cmap("tab10").colors

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7))

    # ECDF preserves the point mass near zero and does not depend on histogram bins.
    for method_index, (method, values) in enumerate(method_values.items()):
        abs_values = np.abs(values)
        sorted_values = np.sort(abs_values)
        cdf = np.arange(1, sorted_values.size + 1) / sorted_values.size
        color = colors.get(method, fallback_colors[method_index % len(fallback_colors)])
        axes[0].step(sorted_values, cdf, where="post", linewidth=2.2, label=method, color=color)

    axes[0].set_xscale("symlog", linthresh=1e-3)
    axes[0].set_xlabel(r"Absolute BN scale $|\gamma|$")
    axes[0].set_ylabel("Cumulative fraction")
    axes[0].set_ylim(0.0, 1.01)
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    for method_index, (method, layers) in enumerate(layer_values.items()):
        medians = []
        q25 = []
        q75 = []
        for _, values in layers:
            abs_values = np.abs(values)
            medians.append(np.median(abs_values))
            q25.append(np.quantile(abs_values, 0.25))
            q75.append(np.quantile(abs_values, 0.75))
        layer_indices = np.arange(1, len(layers) + 1)
        color = colors.get(method, fallback_colors[method_index % len(fallback_colors)])
        axes[1].plot(layer_indices, medians, marker="o", linewidth=2.0, label=method, color=color)
        axes[1].fill_between(layer_indices, q25, q75, color=color, alpha=0.16)

    axes[1].set_yscale("symlog", linthresh=1e-3)
    axes[1].set_xlabel("BN layer index")
    axes[1].set_ylabel(r"Layerwise $|\gamma|$ (median and IQR)")
    axes[1].set_xticks(np.arange(1, max(len(v) for v in layer_values.values()) + 1))
    axes[1].grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"bn_gamma_distribution.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_whole_model_distributions(
    method_values: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    colors = {
        "L1 (all params)": "#0072B2",
        "L1-all": "#0072B2",
        "MNE-L2 (all params)": "#009E73",
        "MNE-all": "#009E73",
        "L2 (all params)": "#D55E00",
        "L2-all": "#D55E00",
        "MNE-standard": "#CC79A7",
    }
    fallback_colors = plt.get_cmap("tab10").colors
    epsilon = 1e-6

    all_log_abs = np.concatenate(
        [np.log10(np.abs(values) + epsilon) for values in method_values.values()]
    )
    bins = np.linspace(all_log_abs.min(), all_log_abs.max(), 80)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7))
    for method_index, (method, values) in enumerate(method_values.items()):
        color = colors.get(method, fallback_colors[method_index % len(fallback_colors)])
        log_abs = np.log10(np.abs(values) + epsilon)
        axes[0].hist(
            log_abs,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            label=method,
            color=color,
        )

        sorted_abs = np.sort(np.abs(values))
        cdf = np.arange(1, sorted_abs.size + 1) / sorted_abs.size
        axes[1].step(
            sorted_abs,
            cdf,
            where="post",
            linewidth=2.2,
            label=method,
            color=color,
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
    for suffix in ("png", "pdf"):
        fig.savefig(
            out_dir / f"bn_gamma_whole_model_distribution.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("../important_results/bn_gamma_stats_all_params_seed42"),
    )
    parser.add_argument(
        "--ckpt",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        help="Optional override: --ckpt 'Label' path/to.pth (repeatable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpts = dict(DEFAULT_CKPTS)
    if args.ckpt:
        ckpts = {label: path for label, path in args.ckpt}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    layer_rows = []
    method_rows = []
    value_rows = []
    method_values: dict[str, np.ndarray] = {}
    layer_values: dict[str, list[tuple[str, np.ndarray]]] = {}

    print(
        f"{'method':22s} {'layer':28s} n  median|γ|     RMS     p90|γ|   max|γ|  "
        f"<1e-3  <1e-2  <1e-1  active_mean"
    )
    for method, path_str in ckpts.items():
        path = Path(path_str)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)

        layers = _extract_bn_gammas(_load_state_dict(path))
        if not layers:
            raise RuntimeError(f"No BN gamma found in {path}")

        all_gamma = torch.cat([tensor for _, tensor in layers])
        method_values[method] = all_gamma.numpy()
        layer_values[method] = [(key, tensor.numpy()) for key, tensor in layers]
        method_stats = _stats(all_gamma)
        method_rows.append({"method": method, "layer": "ALL", **method_stats, "checkpoint": str(path)})
        print(
            f"{method:22s} {'ALL':28s} {method_stats['n']:4d} "
            f"{method_stats['median_abs']:9.6f} {method_stats['rms']:8.6f} "
            f"{method_stats['p90_abs']:9.6f} {method_stats['max_abs']:8.6f} "
            f"{method_stats['frac_lt_0p001']:6.3f} {method_stats['frac_lt_0p01']:6.3f} "
            f"{method_stats['frac_lt_0p1']:6.3f} {method_stats['active_mean_abs_gt_0p1']:11.6f}"
        )

        for index, (key, tensor) in enumerate(layers):
            stats = _stats(tensor)
            value_rows.extend(
                {
                    "method": method,
                    "layer_index": index,
                    "layer": key,
                    "channel_index": channel_index,
                    "gamma": float(gamma),
                    "abs_gamma": abs(float(gamma)),
                }
                for channel_index, gamma in enumerate(tensor.tolist())
            )
            layer_rows.append(
                {
                    "method": method,
                    "layer_index": index,
                    "layer": key,
                    **stats,
                    "checkpoint": str(path),
                }
            )
            print(
                f"{method:22s} {key:28s} {stats['n']:4d} "
                f"{stats['median_abs']:9.6f} {stats['rms']:8.6f} "
                f"{stats['p90_abs']:9.6f} {stats['max_abs']:8.6f} "
                f"{stats['frac_lt_0p001']:6.3f} {stats['frac_lt_0p01']:6.3f} "
                f"{stats['frac_lt_0p1']:6.3f} {stats['active_mean_abs_gt_0p1']:11.6f}"
            )

    fieldnames = [
        "method",
        "layer_index",
        "layer",
        "n",
        "median_abs",
        "rms",
        "p90_abs",
        "max_abs",
        "frac_lt_0p001",
        "frac_lt_0p01",
        "frac_lt_0p1",
        "active_mean_abs_gt_0p1",
        "n_active_gt_0p1",
        "checkpoint",
    ]
    layer_csv = args.out_dir / "bn_gamma_stats_by_layer.csv"
    method_csv = args.out_dir / "bn_gamma_stats_by_method.csv"
    values_csv = args.out_dir / "bn_gamma_values.csv"
    with layer_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(layer_rows)
    with method_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[name for name in fieldnames if name != "layer_index"],
        )
        writer.writeheader()
        writer.writerows(method_rows)
    with values_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "layer_index", "layer", "channel_index", "gamma", "abs_gamma"],
        )
        writer.writeheader()
        writer.writerows(value_rows)

    _plot_distributions(method_values, layer_values, args.out_dir)
    _plot_whole_model_distributions(method_values, args.out_dir)

    print(f"[DONE] method summary: {method_csv}")
    print(f"[DONE] layer summary:  {layer_csv}")
    print(f"[DONE] raw gamma values: {values_csv}")
    print(f"[DONE] distribution plot: {args.out_dir / 'bn_gamma_distribution.png'}")
    print(f"[DONE] vector plot:       {args.out_dir / 'bn_gamma_distribution.pdf'}")
    print(
        f"[DONE] whole-model plot:  "
        f"{args.out_dir / 'bn_gamma_whole_model_distribution.png'}"
    )
    print(
        f"[DONE] whole-model PDF:   "
        f"{args.out_dir / 'bn_gamma_whole_model_distribution.pdf'}"
    )


if __name__ == "__main__":
    main()
