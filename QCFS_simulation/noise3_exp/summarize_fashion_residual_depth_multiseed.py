from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = (
    "val_ann_acc",
    "test_ann_acc",
    "train_val_gap",
    "snn_sigma0_acc",
    "snn_sigma1_acc",
    "snn_drop",
    "lambda_mean",
    "gamma_abs_mean",
)


def parse_seed_roots(entries):
    roots = {}
    for entry in entries:
        seed_text, separator, path_text = entry.partition("=")
        if not separator:
            raise ValueError("--seed-root entries must use SEED=PATH")
        roots[int(seed_text)] = Path(path_text).expanduser().resolve()
    return roots


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate paired all-parameter minus weight-only depth results"
    )
    parser.add_argument("--seed-root", action="append", required=True)
    parser.add_argument("--depths", nargs="+", type=int, default=(20, 40, 60, 80))
    parser.add_argument("--checkpoint", default="best_val", choices=("best_val", "final"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.seed_roots = parse_seed_roots(args.seed_root)
    args.out_dir = args.out_dir.expanduser().resolve()
    return args


def read_methods(path, checkpoint):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["method"]: row
        for row in rows
        if row["checkpoint"] == checkpoint
    }


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_rows(args):
    rows = []
    for seed, root in sorted(args.seed_roots.items()):
        for depth in args.depths:
            summary = root / f"depth_{depth}" / "summary.csv"
            if not summary.exists():
                raise FileNotFoundError(summary)
            methods = read_methods(summary, args.checkpoint)
            weights = methods["weights_only"]
            all_params = methods["all_parameters"]
            row = {
                "seed": seed,
                "depth": depth,
                "n_lambda": int(float(weights["n_lambda"])),
            }
            for metric in METRICS:
                weight_value = float(weights[metric])
                all_value = float(all_params[metric])
                row[f"weights_{metric}"] = weight_value
                row[f"all_{metric}"] = all_value
                row[f"delta_{metric}"] = all_value - weight_value
            rows.append(row)
    return rows


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["depth"])].append(row)

    output = []
    for depth, depth_rows in sorted(grouped.items()):
        row = {
            "depth": depth,
            "n_lambda": depth_rows[0]["n_lambda"],
            "n_seeds": len(depth_rows),
        }
        for metric in METRICS:
            for prefix in ("weights", "all", "delta"):
                key = f"{prefix}_{metric}"
                values = [float(item[key]) for item in depth_rows]
                row[f"{key}_mean"] = statistics.mean(values)
                row[f"{key}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
        output.append(row)
    return output


def plot(rows, out_dir):
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / "matplotlib_cache"))
    import matplotlib.pyplot as plt

    depths = [row["depth"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    left_series = (
        ("delta_val_ann_acc", "ANN validation"),
        ("delta_test_ann_acc", "ANN test"),
        ("delta_snn_sigma0_acc", "SNN sigma=0"),
        ("delta_snn_sigma1_acc", "SNN sigma=1"),
    )
    for key, label in left_series:
        axes[0].errorbar(
            depths,
            [row[f"{key}_mean"] for row in rows],
            yerr=[row[f"{key}_std"] for row in rows],
            marker="o",
            capsize=3,
            label=label,
        )
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xlabel("Number of Conv-BN-IF layers")
    axes[0].set_ylabel("Accuracy difference (percentage points)\nAll parameters - weights only")
    axes[0].set_xticks(depths)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    right_series = (
        ("delta_train_val_gap", "Train-validation gap"),
        ("delta_snn_drop", "SNN sigma 0-to-1 drop"),
    )
    for key, label in right_series:
        axes[1].errorbar(
            depths,
            [row[f"{key}_mean"] for row in rows],
            yerr=[row[f"{key}_std"] for row in rows],
            marker="o",
            capsize=3,
            label=label,
        )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Number of Conv-BN-IF layers")
    axes[1].set_ylabel("Difference (percentage points)\nAll parameters - weights only")
    axes[1].set_xticks(depths)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    path = out_dir / "depth_accuracy_differences_two_seed_mean_std.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = paired_rows(args)
    mean_rows = aggregate(raw_rows)
    raw_path = args.out_dir / "depth_accuracy_differences_by_seed.csv"
    mean_path = args.out_dir / "depth_accuracy_differences_two_seed_mean_std.csv"
    write_csv(raw_path, raw_rows)
    write_csv(mean_path, mean_rows)
    plot_path = plot(mean_rows, args.out_dir)
    print(f"[DONE] raw={raw_path}")
    print(f"[DONE] mean_std={mean_path}")
    print(f"[DONE] plot={plot_path}")


if __name__ == "__main__":
    main()
