from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = ROOT / "QCFS_simulation"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from Models import modelpool


METHOD_ORDER = [
    "wd_weight_only",
    "manual_l2_w_bn",
    "manual_l2_w_if",
    "manual_l2_w_bn_if",
    "manual_l2_all",
]
METHOD_LABELS = {
    "wd_weight_only": "Weights only",
    "manual_l2_w_bn": "Weights + BN",
    "manual_l2_w_if": "Weights + IF",
    "manual_l2_w_bn_if": "Weights + BN + IF",
    "manual_l2_all": "All parameters",
}
METHOD_COLORS = {
    "wd_weight_only": "#0072B2",
    "manual_l2_w_bn": "#56B4E9",
    "manual_l2_w_if": "#E69F00",
    "manual_l2_w_bn_if": "#CC79A7",
    "manual_l2_all": "#D55E00",
}
ARCHITECTURES = {
    "cnn6": {"label": "Narrow, early pool", "width": "narrow", "pooling": "early"},
    "cnn6_narrow_staged": {
        "label": "Narrow, staged pool",
        "width": "narrow",
        "pooling": "staged",
    },
    "cnn6_wide_early": {"label": "Wide, early pool", "width": "wide", "pooling": "early"},
    "cnn6_vgg": {"label": "Wide, staged pool", "width": "wide", "pooling": "staged"},
}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values):
    return statistics.mean(values)


def _std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _auc(points: list[tuple[float, float]]) -> float:
    points = sorted(points)
    width = points[-1][0] - points[0][0]
    if width <= 0:
        return points[0][1]
    area = sum(
        (x1 - x0) * (y0 + y1) / 2.0
        for (x0, y0), (x1, y1) in zip(points, points[1:])
    )
    return area / width


def _per_seed_curves(rows: list[dict]) -> dict:
    repeats = defaultdict(list)
    for row in rows:
        key = (row["method"], int(row["seed"]), float(row["sigma"]))
        repeats[key].append(float(row["accuracy"]))
    curves = defaultdict(dict)
    for (method, seed, sigma), values in repeats.items():
        curves[(method, seed)][sigma] = _mean(values)
    return curves


def _summarize_multiseed(rows: list[dict]):
    curves = _per_seed_curves(rows)
    grouped = defaultdict(list)
    for (method, _seed), points in curves.items():
        for sigma, accuracy in points.items():
            grouped[(method, sigma)].append(accuracy)

    curve_rows = []
    for (method, sigma), values in sorted(grouped.items()):
        curve_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "sigma": sigma,
                "n_training_seeds": len(values),
                "accuracy_mean": _mean(values),
                "accuracy_std": _std(values),
            }
        )

    seeds = sorted(
        set(seed for method, seed in curves if method == "wd_weight_only")
        & set(seed for method, seed in curves if method == "manual_l2_all")
    )
    delta_by_sigma = defaultdict(list)
    adjusted_by_sigma = defaultdict(list)
    auc_deltas = []
    for seed in seeds:
        weights = curves[("wd_weight_only", seed)]
        all_params = curves[("manual_l2_all", seed)]
        clean_delta = weights[0.0] - all_params[0.0]
        for sigma in sorted(set(weights) & set(all_params)):
            delta = weights[sigma] - all_params[sigma]
            delta_by_sigma[sigma].append(delta)
            adjusted_by_sigma[sigma].append(delta - clean_delta)
        auc_deltas.append(_auc(list(weights.items())) - _auc(list(all_params.items())))

    delta_rows = []
    for sigma in sorted(delta_by_sigma):
        delta_rows.append(
            {
                "sigma": sigma,
                "n_training_seeds": len(delta_by_sigma[sigma]),
                "paired_delta_mean": _mean(delta_by_sigma[sigma]),
                "paired_delta_std": _std(delta_by_sigma[sigma]),
                "clean_adjusted_delta_mean": _mean(adjusted_by_sigma[sigma]),
                "clean_adjusted_delta_std": _std(adjusted_by_sigma[sigma]),
            }
        )
    summary_rows = []
    for method in ("wd_weight_only", "manual_l2_all"):
        clean = grouped[(method, 0.0)]
        end_sigma = max(sigma for grouped_method, sigma in grouped if grouped_method == method)
        end = grouped[(method, end_sigma)]
        auc_values = [_auc(list(curves[(method, seed)].items())) for seed in seeds]
        drops = [
            curves[(method, seed)][0.0] - curves[(method, seed)][end_sigma]
            for seed in seeds
        ]
        summary_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_training_seeds": len(seeds),
                "clean_mean": _mean(clean),
                "clean_std": _std(clean),
                "end_sigma": end_sigma,
                "end_mean": _mean(end),
                "end_std": _std(end),
                "drop_mean": _mean(drops),
                "drop_std": _std(drops),
                "auc_mean": _mean(auc_values),
                "auc_std": _std(auc_values),
            }
        )
    paired_auc = {
        "n_training_seeds": len(auc_deltas),
        "paired_auc_delta_mean": _mean(auc_deltas),
        "paired_auc_delta_std": _std(auc_deltas),
    }
    return curve_rows, delta_rows, summary_rows, paired_auc


def _summarize_single_seed(rows: list[dict], extra: dict | None = None):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], float(row["sigma"]))].append(float(row["accuracy"]))
    curve_rows = []
    for (method, sigma), values in sorted(grouped.items()):
        curve_rows.append(
            {
                **(extra or {}),
                "method": method,
                "method_label": METHOD_LABELS[method],
                "sigma": sigma,
                "n_noise_draws": len(values),
                "accuracy_mean": _mean(values),
                "accuracy_std": _std(values),
            }
        )
    curves = defaultdict(list)
    for row in curve_rows:
        curves[row["method"]].append((row["sigma"], row["accuracy_mean"]))
    summary_rows = []
    for method, points in sorted(curves.items()):
        points = sorted(points)
        summary_rows.append(
            {
                **(extra or {}),
                "method": method,
                "method_label": METHOD_LABELS[method],
                "clean_accuracy": points[0][1],
                "end_sigma": points[-1][0],
                "end_accuracy": points[-1][1],
                "absolute_drop": points[0][1] - points[-1][1],
                "curve_auc": _auc(points),
            }
        )
    return curve_rows, summary_rows


def _checkpoint_path(model: str, method: str, seed: int = 42) -> Path:
    rc_tag = "none" if method == "wd_weight_only" else "0p00025"
    suffix = (
        f"fashion_spectral_mne_{model}_{method}_rc{rc_tag}_seed{seed}"
        f"_ep30_L16_trainT0"
    )
    return SIM_ROOT / "fashion_mnist-checkpoints" / f"{model}_L[16]_{suffix}.pth"


def _parameter_diagnostics(models: list[str], methods: list[str], seeds=(42,)):
    summary_rows = []
    layer_rows = []
    for model_name in models:
        for method in methods:
            for seed in seeds:
                checkpoint = _checkpoint_path(model_name, method, seed=seed)
                if not checkpoint.exists():
                    continue
                model = modelpool(model_name, "fashion_mnist")
                model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
                gains = []
                hidden_thresholds = []
                bn_gamma = []
                bn_beta = []
                for index in range(1, model.num_conv_layers + 1):
                    conv = getattr(model, f"conv{index}")
                    bn = getattr(model, f"bn{index}")
                    if_layer = getattr(model, f"if{index}")
                    weight = conv.weight.detach().double()
                    gamma = bn.weight.detach().double()
                    variance = bn.running_var.detach().double()
                    scale = gamma / torch.sqrt(variance + float(bn.eps))
                    folded = weight * scale.view(-1, 1, 1, 1)
                    filter_norm = folded.reshape(folded.shape[0], -1).norm(dim=1).mean()
                    threshold = float(if_layer.thresh.detach().item())
                    gain = float(filter_norm.item()) / max(abs(threshold), 1e-8)
                    gains.append(gain)
                    hidden_thresholds.append(threshold)
                    bn_gamma.extend(bn.weight.detach().abs().reshape(-1).tolist())
                    bn_beta.extend(bn.bias.detach().abs().reshape(-1).tolist())
                    layer_rows.append(
                        {
                            "model": model_name,
                            "architecture": ARCHITECTURES[model_name]["label"],
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "seed": seed,
                            "layer": index,
                            "channels": conv.out_channels,
                            "if_threshold": threshold,
                            "bn_gamma_abs_mean": float(bn.weight.detach().abs().mean().item()),
                            "bn_beta_abs_mean": float(bn.bias.detach().abs().mean().item()),
                            "folded_threshold_normalized_gain": gain,
                        }
                    )
                gain_product = math.prod(gains)
                gain_geomean = gain_product ** (1.0 / len(gains))
                summary_rows.append(
                    {
                        "model": model_name,
                        "architecture": ARCHITECTURES[model_name]["label"],
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "seed": seed,
                        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                        "input_if_threshold": float(model.input_if.thresh.detach().item()),
                        "hidden_if_threshold_mean": _mean(hidden_thresholds),
                        "hidden_if_threshold_min": min(hidden_thresholds),
                        "hidden_if_threshold_max": max(hidden_thresholds),
                        "bn_gamma_abs_mean": _mean(bn_gamma),
                        "bn_beta_abs_mean": _mean(bn_beta),
                        "folded_gain_geomean": gain_geomean,
                        "folded_gain_product": gain_product,
                        "checkpoint": str(checkpoint),
                    }
                )
    return summary_rows, layer_rows


def _aggregate_parameter_scales(rows: list[dict]):
    metrics = [
        "input_if_threshold",
        "hidden_if_threshold_mean",
        "bn_gamma_abs_mean",
        "bn_beta_abs_mean",
        "folded_gain_geomean",
        "folded_gain_product",
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    summary = []
    for method, values in sorted(grouped.items()):
        item = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "n_training_seeds": len(values),
        }
        for metric in metrics:
            metric_values = [float(row[metric]) for row in values]
            item[f"{metric}_mean"] = _mean(metric_values)
            item[f"{metric}_std"] = _std(metric_values)
        summary.append(item)
    return summary


def _aggregate_diagnostics(layer_rows: list[dict], margin_rows: list[dict]):
    layer_grouped = defaultdict(list)
    for row in layer_rows:
        key = (row["method"], float(row["sigma"]), int(row["layer_index"]), row["layer"])
        layer_grouped[key].append(row)
    layer_mean = []
    metrics = [
        "relative_l2",
        "layer_amplification",
        "cosine_similarity",
        "spike_mismatch_rate",
        "clean_boundary_margin_lt_0p05",
        "noisy_boundary_margin_lt_0p05",
        "clean_firing_rate",
        "noisy_firing_rate",
    ]
    for (method, sigma, layer_index, layer), values in sorted(layer_grouped.items()):
        row = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "sigma": sigma,
            "layer_index": layer_index,
            "layer": layer,
            "n_noise_draws": len(values),
        }
        for metric in metrics:
            present = [float(value[metric]) for value in values if value.get(metric) not in (None, "", "nan")]
            row[f"{metric}_mean"] = _mean(present) if present else float("nan")
            row[f"{metric}_std"] = _std(present) if present else float("nan")
        layer_mean.append(row)

    margin_grouped = defaultdict(list)
    for row in margin_rows:
        margin_grouped[(row["method"], float(row["sigma"]))].append(row)
    margin_mean = []
    margin_metrics = [
        "clean_accuracy",
        "noisy_accuracy",
        "accuracy_drop",
        "prediction_flip_rate",
        "clean_correct_to_wrong_rate",
        "clean_margin_mean",
        "noisy_margin_mean",
        "margin_change_mean",
        "clean_negative_margin_rate",
        "noisy_negative_margin_rate",
    ]
    for (method, sigma), values in sorted(margin_grouped.items()):
        row = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "sigma": sigma,
            "n_noise_draws": len(values),
            "n_samples_per_draw": int(values[0]["n_samples"]),
        }
        for metric in margin_metrics:
            metric_values = [float(value[metric]) for value in values]
            row[f"{metric}_mean"] = _mean(metric_values)
            row[f"{metric}_std"] = _std(metric_values)
        margin_mean.append(row)
    return layer_mean, margin_mean


def _plot_multiseed(output: Path, curve_rows: list[dict], delta_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    for method in ("wd_weight_only", "manual_l2_all"):
        points = sorted((row for row in curve_rows if row["method"] == method), key=lambda row: row["sigma"])
        axes[0].errorbar(
            [row["sigma"] for row in points],
            [row["accuracy_mean"] for row in points],
            yerr=[row["accuracy_std"] for row in points],
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            marker="o",
            linewidth=2.2,
            capsize=3,
        )
    axes[1].errorbar(
        [row["sigma"] for row in delta_rows],
        [row["paired_delta_mean"] for row in delta_rows],
        yerr=[row["paired_delta_std"] for row in delta_rows],
        color="#009E73",
        marker="s",
        linewidth=2.2,
        capsize=3,
        label="Raw paired delta",
    )
    axes[1].plot(
        [row["sigma"] for row in delta_rows],
        [row["clean_adjusted_delta_mean"] for row in delta_rows],
        color="#6A3D9A",
        marker="^",
        linewidth=2.0,
        label="Clean-adjusted delta",
    )
    axes[1].axhline(0, color="#777777", linestyle="--", linewidth=1)
    axes[0].set_ylabel("SNN accuracy (%)")
    axes[1].set_ylabel("Weights-only minus all-params (pp)")
    for axis in axes:
        axis.set_xlabel("Absolute post-input-IF noise sigma")
        axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    axes[0].set_title("Five training seeds")
    axes[1].set_title("Paired robustness gap")
    fig.tight_layout()
    fig.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_scope(output: Path, rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for method in METHOD_ORDER:
        points = sorted((row for row in rows if row["method"] == method), key=lambda row: row["sigma"])
        ax.errorbar(
            [row["sigma"] for row in points],
            [row["accuracy_mean"] for row in points],
            yerr=[row["accuracy_std"] for row in points],
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            marker="o",
            linewidth=2,
            capsize=2,
        )
    ax.set_xlabel("Absolute post-input-IF noise sigma")
    ax.set_ylabel("SNN accuracy (%)")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Which parameter family causes the robustness loss?")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_architecture(output: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), sharex=True, sharey=True)
    for axis, (model, spec) in zip(axes.flat, ARCHITECTURES.items()):
        for method in ("wd_weight_only", "manual_l2_all"):
            points = sorted(
                (row for row in rows if row["model"] == model and row["method"] == method),
                key=lambda row: row["sigma"],
            )
            axis.errorbar(
                [row["sigma"] for row in points],
                [row["accuracy_mean"] for row in points],
                yerr=[row["accuracy_std"] for row in points],
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                marker="o",
                linewidth=2,
                capsize=2,
            )
        axis.set_title(spec["label"])
        axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    for axis in axes[-1, :]:
        axis.set_xlabel("Absolute noise sigma")
    for axis in axes[:, 0]:
        axis.set_ylabel("SNN accuracy (%)")
    fig.suptitle("Six-layer 2 x 2 control: width and pooling position")
    fig.tight_layout()
    fig.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_relative(output: Path, rows: list[dict]) -> None:
    scales = ["absolute", "input_if_threshold", "post_input_if_rms"]
    titles = ["Absolute sigma", "Sigma / input threshold", "Sigma / clean activation RMS"]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))
    for axis, scale, title in zip(axes, scales, titles):
        for method in ("wd_weight_only", "manual_l2_all"):
            points = sorted(
                (row for row in rows if row["sigma_scale"] == scale and row["method"] == method),
                key=lambda row: row["sigma"],
            )
            axis.errorbar(
                [row["sigma"] for row in points],
                [row["accuracy_mean"] for row in points],
                yerr=[row["accuracy_std"] for row in points],
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                marker="o",
                linewidth=2,
                capsize=2,
            )
        axis.set_title(title)
        axis.set_xlabel("Noise level")
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("SNN accuracy (%)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_layerwise(output: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), sharey=False)
    for axis, sigma in zip(axes, (0.5, 1.0)):
        for method in ("wd_weight_only", "manual_l2_w_bn", "manual_l2_w_if", "manual_l2_all"):
            points = sorted(
                (row for row in rows if row["method"] == method and row["sigma"] == sigma),
                key=lambda row: row["layer_index"],
            )
            axis.errorbar(
                [row["layer_index"] for row in points],
                [row["relative_l2_mean"] for row in points],
                yerr=[row["relative_l2_std"] for row in points],
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                marker="o",
                linewidth=2,
                capsize=2,
            )
        axis.set_title(f"Absolute sigma = {sigma:g}")
        axis.set_xlabel("Layer (0 = post-input-IF, 1-6 = IF layers)")
        axis.set_ylabel("Relative representation perturbation")
        axis.set_xticks(range(7))
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the Fashion L2-scope causal follow-up.")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "important_results" / "fashion_l2_scope_followup",
    )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    multiseed_raw = _read_csv(args.root / "multiseed_absolute" / "internal_if_noise_raw.csv")
    multi_curve, multi_delta, multi_summary, paired_auc = _summarize_multiseed(multiseed_raw)
    _write_csv(args.root / "multiseed_curve.csv", multi_curve)
    _write_csv(args.root / "multiseed_paired_delta.csv", multi_delta)
    _write_csv(args.root / "multiseed_summary.csv", multi_summary)
    _write_csv(args.root / "multiseed_paired_auc.csv", [paired_auc])

    scope_raw = _read_csv(args.root / "parameter_scope_absolute" / "internal_if_noise_raw.csv")
    scope_curve, scope_summary = _summarize_single_seed(scope_raw)
    _write_csv(args.root / "parameter_scope_curve.csv", scope_curve)
    _write_csv(args.root / "parameter_scope_summary.csv", scope_summary)

    architecture_curve = []
    architecture_summary = []
    arch_paths = {
        "cnn6": ROOT / "important_results" / "fashion_deep_l2_scope_seed42" / "cnn6_post_input_if",
        "cnn6_vgg": ROOT / "important_results" / "fashion_vgglike_l2_scope_seed42" / "cnn6_vgg_post_input_if",
        "cnn6_narrow_staged": args.root / "architecture_2x2" / "cnn6_narrow_staged_post_input_if",
        "cnn6_wide_early": args.root / "architecture_2x2" / "cnn6_wide_early_post_input_if",
    }
    for model, path in arch_paths.items():
        raw = _read_csv(path / "internal_if_noise_raw.csv")
        extra = {"model": model, **ARCHITECTURES[model]}
        curve, summary = _summarize_single_seed(raw, extra=extra)
        architecture_curve.extend(curve)
        architecture_summary.extend(summary)
    _write_csv(args.root / "architecture_2x2_curve.csv", architecture_curve)
    _write_csv(args.root / "architecture_2x2_summary.csv", architecture_summary)

    relative_curve = []
    relative_summary = []
    absolute_curve, absolute_summary = _summarize_single_seed(
        [row for row in scope_raw if row["method"] in ("wd_weight_only", "manual_l2_all")],
        extra={"sigma_scale": "absolute"},
    )
    relative_curve.extend(absolute_curve)
    relative_summary.extend(absolute_summary)
    for scale, directory in (
        ("input_if_threshold", "relative_threshold"),
        ("post_input_if_rms", "relative_rms"),
    ):
        raw = _read_csv(args.root / directory / "internal_if_noise_raw.csv")
        curve, summary = _summarize_single_seed(raw, extra={"sigma_scale": scale})
        relative_curve.extend(curve)
        relative_summary.extend(summary)
    _write_csv(args.root / "absolute_vs_relative_curve.csv", relative_curve)
    _write_csv(args.root / "absolute_vs_relative_summary.csv", relative_summary)

    parameter_summary, parameter_layers = _parameter_diagnostics(
        list(ARCHITECTURES), METHOD_ORDER
    )
    _write_csv(args.root / "parameter_scale_summary.csv", parameter_summary)
    _write_csv(args.root / "parameter_scale_layers.csv", parameter_layers)
    multiseed_parameter_rows, _ = _parameter_diagnostics(
        ["cnn6_vgg"], ["wd_weight_only", "manual_l2_all"], seeds=(40, 41, 42, 43, 44)
    )
    _write_csv(args.root / "parameter_scale_multiseed_raw.csv", multiseed_parameter_rows)
    _write_csv(
        args.root / "parameter_scale_multiseed_summary.csv",
        _aggregate_parameter_scales(multiseed_parameter_rows),
    )

    diagnostic_layers = _read_csv(args.root / "layerwise_mechanism_raw.csv")
    diagnostic_margins = _read_csv(args.root / "output_margin_raw.csv")
    layer_mean, margin_mean = _aggregate_diagnostics(diagnostic_layers, diagnostic_margins)
    _write_csv(args.root / "layerwise_mechanism_mean_std.csv", layer_mean)
    _write_csv(args.root / "output_margin_mean_std.csv", margin_mean)

    _plot_multiseed(args.root / "figure_multiseed.png", multi_curve, multi_delta)
    _plot_scope(args.root / "figure_parameter_scope.png", scope_curve)
    _plot_architecture(args.root / "figure_architecture_2x2.png", architecture_curve)
    _plot_relative(args.root / "figure_absolute_vs_relative.png", relative_curve)
    _plot_layerwise(args.root / "figure_layerwise_mechanism.png", layer_mean)
    print(f"[DONE] Aggregated results in {args.root}")


if __name__ == "__main__":
    main()
