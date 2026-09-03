from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "important_results" / "fashion_vgglike_l2_scope_seed42"
INPUTS = {
    "Narrow CNN6": (
        ROOT
        / "important_results"
        / "fashion_deep_l2_scope_seed42"
        / "cnn6_post_input_if"
        / "internal_if_noise_raw.csv"
    ),
    "VGG-like CNN6": (
        DEFAULT_OUTPUT_DIR
        / "cnn6_vgg_post_input_if"
        / "internal_if_noise_raw.csv"
    ),
}
METHODS = ("wd_weight_only", "manual_l2_all")


def _load_summary() -> list[dict]:
    summary = []
    for architecture, path in INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")
        grouped = defaultdict(dict)
        with path.open("r", newline="") as handle:
            for row in csv.DictReader(handle):
                method = row["method"]
                if method not in METHODS:
                    continue
                key = (float(row["sigma"]), int(row["noise_repeat"]))
                grouped[key][method] = float(row["accuracy"])

        clean_delta = None
        paired = defaultdict(list)
        for (sigma, _repeat), values in grouped.items():
            missing = set(METHODS) - set(values)
            if missing:
                raise ValueError(
                    f"Unpaired methods for {architecture}, sigma={sigma}: {missing}"
                )
            paired[sigma].append(values)
            if sigma == 0:
                clean_delta = values["wd_weight_only"] - values["manual_l2_all"]
        if clean_delta is None:
            raise ValueError(f"No sigma=0 reference for {architecture}")

        for sigma, values in sorted(paired.items()):
            weights = [item["wd_weight_only"] for item in values]
            all_params = [item["manual_l2_all"] for item in values]
            deltas = [left - right for left, right in zip(weights, all_params)]
            summary.append(
                {
                    "architecture": architecture,
                    "sigma": sigma,
                    "n_noise_draws": len(values),
                    "weights_only_acc_mean": statistics.mean(weights),
                    "weights_only_acc_std": (
                        statistics.stdev(weights) if len(weights) > 1 else 0.0
                    ),
                    "all_params_acc_mean": statistics.mean(all_params),
                    "all_params_acc_std": (
                        statistics.stdev(all_params) if len(all_params) > 1 else 0.0
                    ),
                    "delta_acc_mean": statistics.mean(deltas),
                    "delta_acc_std": (
                        statistics.stdev(deltas) if len(deltas) > 1 else 0.0
                    ),
                    "clean_adjusted_delta": statistics.mean(deltas) - clean_delta,
                }
            )
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def _plot(path: Path, rows: list[dict]) -> None:
    colors = {"wd_weight_only": "#0072B2", "manual_l2_all": "#D55E00"}
    labels = {
        "wd_weight_only": "weights-only",
        "manual_l2_all": "all-parameters",
    }
    architecture_styles = {
        "Narrow CNN6": ("--", "o"),
        "VGG-like CNN6": ("-", "s"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    for architecture in INPUTS:
        points = sorted(
            (row for row in rows if row["architecture"] == architecture),
            key=lambda row: row["sigma"],
        )
        sigmas = [row["sigma"] for row in points]
        linestyle, marker = architecture_styles[architecture]
        for method in METHODS:
            mean_key = (
                "weights_only_acc_mean"
                if method == "wd_weight_only"
                else "all_params_acc_mean"
            )
            std_key = (
                "weights_only_acc_std"
                if method == "wd_weight_only"
                else "all_params_acc_std"
            )
            axes[0].errorbar(
                sigmas,
                [row[mean_key] for row in points],
                yerr=[row[std_key] for row in points],
                color=colors[method],
                linestyle=linestyle,
                marker=marker,
                linewidth=2.0,
                markersize=6,
                capsize=3,
                label=f"{architecture}, {labels[method]}",
            )

        axes[1].errorbar(
            sigmas,
            [row["delta_acc_mean"] for row in points],
            yerr=[row["delta_acc_std"] for row in points],
            color="#009E73" if architecture == "VGG-like CNN6" else "#555555",
            linestyle=linestyle,
            marker=marker,
            linewidth=2.2,
            markersize=6,
            capsize=3,
            label=architecture,
        )

    axes[0].set_title("Post-input-IF noise accuracy")
    axes[0].set_ylabel("SNN accuracy (%)")
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[1].set_title("L2 parameter-scope delta")
    axes[1].set_ylabel("weights-only - all-parameters (pp)")
    axes[1].axhline(0, color="#888888", linestyle="--", linewidth=1.1)
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.set_xlabel("Noise sigma")
        ax.set_xticks((0, 0.25, 0.5, 0.75, 1.0))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Fashion-MNIST: narrow versus VGG-like six-layer CNN", fontsize=14)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare L2 parameter scope for narrow and VGG-like CNN6."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default="cnn6_architecture_scope_comparison")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_summary()
    output = args.output_dir / args.output_stem
    _write_csv(output.with_suffix(".csv"), rows)
    _plot(output, rows)
    print(f"[DONE] {output.with_suffix('.csv')}")
    print(f"[DONE] {output.with_suffix('.png')}")
    print(f"[DONE] {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
