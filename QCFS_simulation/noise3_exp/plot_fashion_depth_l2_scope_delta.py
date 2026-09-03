from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = ROOT / "important_results" / "fashion_deep_l2_scope_seed42"
DEPTHS = (2, 4, 6, 8, 10)
METHODS = ("wd_weight_only", "manual_l2_all")


def _load_paired_deltas(result_dir: Path) -> list[dict]:
    rows = []
    for depth in DEPTHS:
        path = result_dir / f"cnn{depth}_post_input_if" / "internal_if_noise_raw.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing depth result: {path}")

        grouped = defaultdict(dict)
        with path.open("r", newline="") as handle:
            for row in csv.DictReader(handle):
                method = row["method"]
                if method not in METHODS:
                    continue
                key = (float(row["sigma"]), int(row["noise_repeat"]))
                grouped[key][method] = float(row["accuracy"])

        clean_delta = None
        depth_rows = []
        for (sigma, repeat), values in sorted(grouped.items()):
            missing = set(METHODS) - set(values)
            if missing:
                raise ValueError(
                    f"Unpaired methods for CNN{depth}, sigma={sigma}, repeat={repeat}: {missing}"
                )
            delta = values["wd_weight_only"] - values["manual_l2_all"]
            if sigma == 0:
                clean_delta = delta
            depth_rows.append(
                {
                    "depth": depth,
                    "sigma": sigma,
                    "noise_repeat": repeat,
                    "weights_only_acc": values["wd_weight_only"],
                    "all_params_acc": values["manual_l2_all"],
                    "delta_acc": delta,
                }
            )
        if clean_delta is None:
            raise ValueError(f"CNN{depth} has no sigma=0 reference")
        for row in depth_rows:
            row["clean_adjusted_delta"] = row["delta_acc"] - clean_delta
            rows.append(row)
    return rows


def _aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["depth"], row["sigma"])].append(row)

    summary = []
    for (depth, sigma), values in sorted(grouped.items()):
        deltas = [row["delta_acc"] for row in values]
        adjusted = [row["clean_adjusted_delta"] for row in values]
        summary.append(
            {
                "depth": depth,
                "sigma": sigma,
                "n_noise_draws": len(values),
                "weights_only_acc_mean": statistics.mean(
                    row["weights_only_acc"] for row in values
                ),
                "all_params_acc_mean": statistics.mean(
                    row["all_params_acc"] for row in values
                ),
                "delta_acc_mean": statistics.mean(deltas),
                "delta_acc_std": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "clean_adjusted_delta_mean": statistics.mean(adjusted),
                "clean_adjusted_delta_std": (
                    statistics.stdev(adjusted) if len(adjusted) > 1 else 0.0
                ),
            }
        )
    return summary


def _write_summary(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def _plot(path: Path, rows: list[dict]) -> None:
    colors = {
        0.0: "#222222",
        0.25: "#009E73",
        0.5: "#0072B2",
        0.75: "#E69F00",
        1.0: "#D55E00",
    }
    markers = {0.0: "o", 0.25: "s", 0.5: "^", 0.75: "D", 1.0: "v"}

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    sigmas = sorted({float(row["sigma"]) for row in rows})
    for sigma in sigmas:
        points = sorted(
            (row for row in rows if float(row["sigma"]) == sigma),
            key=lambda row: int(row["depth"]),
        )
        depths = [int(row["depth"]) for row in points]
        means = [float(row["delta_acc_mean"]) for row in points]
        errors = [float(row["delta_acc_std"]) for row in points]
        label = f"sigma = {sigma:g}"
        ax.errorbar(
            depths,
            means,
            yerr=errors,
            color=colors[sigma],
            marker=markers[sigma],
            linewidth=2.0,
            markersize=6.5,
            capsize=3,
            label=label,
        )

    ax.axhline(0, color="#777777", linewidth=1.2, linestyle="--", zorder=0)
    ax.set_xticks(DEPTHS)
    ax.set_xlabel("Number of Conv-BN-IF layers")
    ax.set_ylabel("Delta accuracy: weights-only - all-parameters (pp)")
    ax.set_title("L2 parameter-scope accuracy delta across network depth")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=3, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot paired L2-scope accuracy deltas across CNN depth."
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-stem", default="l2_scope_delta_by_depth")
    args = parser.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = _load_paired_deltas(args.result_dir)
    summary_rows = _aggregate(raw_rows)
    csv_path = args.result_dir / f"{args.output_stem}.csv"
    _write_summary(csv_path, summary_rows)
    _plot(args.result_dir / args.output_stem, summary_rows)
    print(f"[DONE] {csv_path}")
    print(f"[DONE] {csv_path.with_suffix('.png')}")
    print(f"[DONE] {csv_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
