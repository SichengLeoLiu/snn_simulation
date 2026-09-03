from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "noise3_exp" / "run_fashion_deep_narrow_lambda_overfit.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fashion-MNIST deep-narrow all-parameter versus weight-only L2 sweep"
    )
    parser.add_argument(
        "--depths", nargs="+", type=int, default=(6, 9, 12, 15, 18, 24, 40, 60, 80)
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--noise-repeats", type=int, default=3)
    parser.add_argument("--reg-coeff", type=float, default=2.5e-4)
    parser.add_argument(
        "--architecture",
        choices=("staged", "residual_constant"),
        default="staged",
    )
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--evaluate-best-only", action="store_true")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "fashion_depth_l2_scope_seed42",
    )
    parser.add_argument(
        "--existing-depth12-dir",
        type=Path,
        default=ROOT.parent
        / "important_results"
        / "fashion_deep_narrow_lambda_overfit_seed42",
    )
    parser.add_argument(
        "--reuse-roots",
        type=Path,
        nargs="*",
        default=[ROOT.parent / "important_results" / "fashion_depth_l2_scope_seed42"],
        help="roots containing reusable depth_<n>/summary.csv results",
    )
    return parser.parse_args()


def summary_dir(depth, args):
    if (
        args.architecture == "staged"
        and depth == 12
        and (args.existing_depth12_dir / "summary.csv").exists()
    ):
        return args.existing_depth12_dir.resolve()
    for root in args.reuse_roots:
        candidate = root / f"depth_{depth}"
        if (candidate / "summary.csv").exists():
            return candidate.resolve()
    return (args.out_root / f"depth_{depth}").resolve()


def distribute_depth(depth):
    if depth < 3:
        raise ValueError(f"depth must be at least three, got {depth}")
    base, remainder = divmod(depth, 3)
    return tuple(base + (1 if index < remainder else 0) for index in range(3))


def run_depth(depth, args):
    if args.architecture == "residual_constant":
        if depth < 4:
            raise ValueError("residual_constant requires depth >= 4")
        stage_blocks = (1, 2, depth - 3)
        channels = (8, 8, 8)
    else:
        stage_blocks = distribute_depth(depth)
        channels = (8, 16, 24)
    out_dir = summary_dir(depth, args)
    summary = out_dir / "summary.csv"
    if summary.exists():
        print(f"[SKIP] depth={depth} existing={summary}", flush=True)
        return out_dir
    if args.skip_training:
        raise FileNotFoundError(f"missing depth={depth} summary: {summary}")

    command = [
        sys.executable,
        "-u",
        str(RUNNER),
        "--methods",
        "weights_only",
        "all_parameters",
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--threads",
        str(args.threads),
        "--device",
        args.device,
        "--reg-coeff",
        str(args.reg_coeff),
        "--channels",
        *(str(value) for value in channels),
        "--stage-blocks",
        *(str(value) for value in stage_blocks),
        "--L",
        "16",
        "--test-T",
        "16",
        "--if-mode",
        "rate_uniform",
        "--noise-repeats",
        str(args.noise_repeats),
        "--out-dir",
        str(out_dir),
    ]
    if args.architecture == "residual_constant":
        command.append("--residual-pairs")
    if args.evaluate_best_only:
        command.append("--evaluate-best-only")
    print("[RUN] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return out_dir


def read_best_rows(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["method"]: row
        for row in rows
        if row["checkpoint"] == "best_val"
    }


def value(row, key):
    return float(row[key])


def aggregate(depth_to_dir, args):
    rows = []
    for depth, directory in sorted(depth_to_dir.items()):
        methods = read_best_rows(directory / "summary.csv")
        weights = methods["weights_only"]
        all_params = methods["all_parameters"]
        rows.append(
            {
                "architecture": args.architecture,
                "depth": depth,
                "n_lambda": int(float(weights["n_lambda"])),
                "weights_val_acc": value(weights, "val_ann_acc"),
                "all_val_acc": value(all_params, "val_ann_acc"),
                "delta_val_acc": value(all_params, "val_ann_acc")
                - value(weights, "val_ann_acc"),
                "weights_test_acc": value(weights, "test_ann_acc"),
                "all_test_acc": value(all_params, "test_ann_acc"),
                "delta_test_acc": value(all_params, "test_ann_acc")
                - value(weights, "test_ann_acc"),
                "weights_train_val_gap": value(weights, "train_val_gap"),
                "all_train_val_gap": value(all_params, "train_val_gap"),
                "delta_train_val_gap": value(all_params, "train_val_gap")
                - value(weights, "train_val_gap"),
                "weights_snn_sigma0": value(weights, "snn_sigma0_acc"),
                "all_snn_sigma0": value(all_params, "snn_sigma0_acc"),
                "delta_snn_sigma0": value(all_params, "snn_sigma0_acc")
                - value(weights, "snn_sigma0_acc"),
                "weights_snn_sigma1": value(weights, "snn_sigma1_acc"),
                "all_snn_sigma1": value(all_params, "snn_sigma1_acc"),
                "delta_snn_sigma1": value(all_params, "snn_sigma1_acc")
                - value(weights, "snn_sigma1_acc"),
                "weights_snn_drop": value(weights, "snn_drop"),
                "all_snn_drop": value(all_params, "snn_drop"),
                "delta_snn_drop": value(all_params, "snn_drop")
                - value(weights, "snn_drop"),
                "weights_lambda_mean": value(weights, "lambda_mean"),
                "all_lambda_mean": value(all_params, "lambda_mean"),
                "delta_lambda_mean": value(all_params, "lambda_mean")
                - value(weights, "lambda_mean"),
                "weights_gamma_abs_mean": value(weights, "gamma_abs_mean"),
                "all_gamma_abs_mean": value(all_params, "gamma_abs_mean"),
            }
        )

    args.out_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_root / "depth_accuracy_differences.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows, csv_path


def plot(rows, args):
    os.environ.setdefault("MPLCONFIGDIR", str(args.out_root / "matplotlib_cache"))
    import matplotlib.pyplot as plt

    depths = [row["depth"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    accuracy_series = (
        ("delta_val_acc", "ANN validation"),
        ("delta_test_acc", "ANN test"),
        ("delta_snn_sigma0", "SNN sigma=0"),
        ("delta_snn_sigma1", "SNN sigma=1"),
    )
    for key, label in accuracy_series:
        axes[0].plot(depths, [row[key] for row in rows], marker="o", label=label)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xlabel("Number of Conv-BN-IF layers")
    axes[0].set_ylabel("Accuracy difference (percentage points)\nAll parameters - weights only")
    axes[0].set_xticks(depths)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(
        depths,
        [row["delta_train_val_gap"] for row in rows],
        marker="o",
        color="#8c564b",
        label="Train-validation gap",
    )
    axes[1].plot(
        depths,
        [row["delta_snn_drop"] for row in rows],
        marker="s",
        color="#9467bd",
        label="SNN sigma 0-to-1 drop",
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Number of Conv-BN-IF layers")
    axes[1].set_ylabel("Difference (percentage points)\nAll parameters - weights only")
    axes[1].set_xticks(depths)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    path = args.out_root / "depth_accuracy_differences.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    absolute_fig, absolute_axes = plt.subplots(1, 3, figsize=(14, 4.2))
    absolute_series = (
        ("weights_test_acc", "all_test_acc", "ANN test accuracy"),
        ("weights_snn_sigma0", "all_snn_sigma0", "SNN sigma=0 accuracy"),
        ("weights_snn_sigma1", "all_snn_sigma1", "SNN sigma=1 accuracy"),
    )
    for axis, (weights_key, all_key, title) in zip(absolute_axes, absolute_series):
        axis.plot(
            depths,
            [row[weights_key] for row in rows],
            marker="o",
            label="Weights only",
        )
        axis.plot(
            depths,
            [row[all_key] for row in rows],
            marker="s",
            label="All parameters",
        )
        axis.set_title(title)
        axis.set_xlabel("Number of Conv-BN-IF layers")
        axis.set_ylabel("Accuracy (%)")
        axis.set_xticks(depths)
        axis.grid(alpha=0.25)
    absolute_axes[0].legend(frameon=False)
    absolute_fig.tight_layout()
    absolute_path = args.out_root / "depth_absolute_accuracies.png"
    absolute_fig.savefig(absolute_path, dpi=220, bbox_inches="tight")
    plt.close(absolute_fig)
    return path, absolute_path


def main():
    args = parse_args()
    args.out_root = args.out_root.resolve()
    args.existing_depth12_dir = args.existing_depth12_dir.resolve()
    args.reuse_roots = [root.resolve() for root in args.reuse_roots]
    depth_to_dir = {depth: run_depth(depth, args) for depth in args.depths}
    rows, csv_path = aggregate(depth_to_dir, args)
    plot_path, absolute_plot_path = plot(rows, args)
    print(f"[DONE] table={csv_path}", flush=True)
    print(f"[DONE] plot={plot_path}", flush=True)
    print(f"[DONE] absolute_plot={absolute_plot_path}", flush=True)


if __name__ == "__main__":
    main()
