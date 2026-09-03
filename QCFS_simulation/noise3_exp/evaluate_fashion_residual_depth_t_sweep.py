from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from run_fashion_deep_narrow_lambda_overfit import (
    DeepNarrowFashionCNN,
    evaluate,
    make_loaders,
    resolve_device,
    scale_stats,
    seed_everything,
)


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("weights_only", "all_parameters")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained deep residual Fashion-MNIST models across T"
    )
    parser.add_argument("--depths", nargs="+", type=int, required=True)
    parser.add_argument("--time-steps", nargs="+", type=int, default=(2, 4, 8, 16))
    parser.add_argument("--checkpoint-root", action="append", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--val-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--noise-repeats", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.checkpoint_root = [path.expanduser().resolve() for path in args.checkpoint_root]
    args.out_dir = args.out_dir.expanduser().resolve()
    return args


def checkpoint_path(depth, method, roots):
    for root in roots:
        candidate = root / f"depth_{depth}" / method / "best_val.pt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing depth={depth} method={method} checkpoint")


def load_checkpoint(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved_args = payload["args"]
    model = DeepNarrowFashionCNN(
        channels=tuple(saved_args["channels"]),
        blocks_per_stage=int(saved_args["blocks_per_stage"]),
        stage_blocks=tuple(saved_args["stage_blocks"]),
        residual_pairs=bool(saved_args["residual_pairs"]),
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    return model, saved_args


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(model, saved_args, test_loader, criterion, args, device, depth, method):
    rows = []
    stats = scale_stats(model)
    for time_steps in sorted(set(args.time_steps)):
        model.set_T(time_steps)
        model.set_L(int(saved_args["L"]))
        model.set_mode(saved_args["if_mode"])
        sigma_accuracies = {}
        sigma_stds = {}
        for sigma in (0.0, 1.0):
            repeats = 1 if sigma == 0 else args.noise_repeats
            accuracies = []
            for repeat in range(repeats):
                seed_everything(args.seed + 10000 + repeat)
                model.set_first_layer_input_noise_sigma(sigma)
                accuracies.append(evaluate(model, test_loader, criterion, device)["acc"])
            sigma_accuracies[sigma] = float(np.mean(accuracies))
            sigma_stds[sigma] = (
                float(np.std(accuracies, ddof=1)) if repeats > 1 else 0.0
            )
        rows.append(
            {
                "depth": depth,
                "n_lambda": stats["n_lambda"],
                "method": method,
                "T": time_steps,
                "sigma0_acc": sigma_accuracies[0.0],
                "sigma1_acc": sigma_accuracies[1.0],
                "sigma1_noise_repeat_std": sigma_stds[1.0],
                "drop_sigma0_to_1": sigma_accuracies[0.0] - sigma_accuracies[1.0],
                "lambda_mean": stats["lambda_mean"],
                "gamma_abs_mean": stats["gamma_abs_mean"],
                "checkpoint": str(path_for_row(depth, method, args.checkpoint_root)),
            }
        )
        print(
            f"depth={depth:03d} method={method:<14} T={time_steps:02d} "
            f"sigma0={sigma_accuracies[0.0]:.2f} sigma1={sigma_accuracies[1.0]:.2f}",
            flush=True,
        )
    return rows


def path_for_row(depth, method, roots):
    return checkpoint_path(depth, method, roots)


def paired_deltas(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(int(row["depth"]), int(row["T"]))][row["method"]] = row
    output = []
    for (depth, time_steps), methods in sorted(grouped.items()):
        weights = methods["weights_only"]
        all_params = methods["all_parameters"]
        output.append(
            {
                "depth": depth,
                "T": time_steps,
                "weights_sigma0": weights["sigma0_acc"],
                "all_sigma0": all_params["sigma0_acc"],
                "delta_sigma0": all_params["sigma0_acc"] - weights["sigma0_acc"],
                "weights_sigma1": weights["sigma1_acc"],
                "all_sigma1": all_params["sigma1_acc"],
                "delta_sigma1": all_params["sigma1_acc"] - weights["sigma1_acc"],
                "weights_drop": weights["drop_sigma0_to_1"],
                "all_drop": all_params["drop_sigma0_to_1"],
                "delta_drop": all_params["drop_sigma0_to_1"]
                - weights["drop_sigma0_to_1"],
            }
        )
    return output


def plot(rows, deltas, args):
    os.environ.setdefault("MPLCONFIGDIR", str(args.out_dir / "matplotlib_cache"))
    import matplotlib.pyplot as plt

    depths = sorted({int(row["depth"]) for row in rows})
    time_steps = sorted({int(row["T"]) for row in rows})
    fig, axes = plt.subplots(2, len(depths), figsize=(4.2 * len(depths), 7.2), squeeze=False)
    for column, depth in enumerate(depths):
        for method, label, marker in (
            ("weights_only", "Weights only", "o"),
            ("all_parameters", "All parameters", "s"),
        ):
            selected = sorted(
                (row for row in rows if row["depth"] == depth and row["method"] == method),
                key=lambda row: row["T"],
            )
            axes[0, column].plot(
                [row["T"] for row in selected],
                [row["sigma0_acc"] for row in selected],
                marker=marker,
                label=label,
            )
            axes[1, column].plot(
                [row["T"] for row in selected],
                [row["sigma1_acc"] for row in selected],
                marker=marker,
                label=label,
            )
        axes[0, column].set_title(f"Depth {depth}, sigma=0")
        axes[1, column].set_title(f"Depth {depth}, sigma=1")
        for axis in axes[:, column]:
            axis.set_xlabel("Time steps T")
            axis.set_ylabel("Accuracy (%)")
            axis.set_xticks(time_steps)
            axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    accuracy_path = args.out_dir / "t_sweep_absolute_accuracies.png"
    fig.savefig(accuracy_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    delta_fig, delta_axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for axis, key, title in (
        (delta_axes[0], "delta_sigma0", "SNN sigma=0"),
        (delta_axes[1], "delta_sigma1", "SNN sigma=1"),
    ):
        for depth in depths:
            selected = sorted(
                (row for row in deltas if row["depth"] == depth),
                key=lambda row: row["T"],
            )
            axis.plot(
                [row["T"] for row in selected],
                [row[key] for row in selected],
                marker="o",
                label=f"Depth {depth}",
            )
        axis.axhline(0, color="black", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Time steps T")
        axis.set_ylabel("All parameters - weights only (pp)")
        axis.set_xticks(time_steps)
        axis.grid(alpha=0.25)
    delta_axes[0].legend(frameon=False, ncol=2)
    delta_fig.tight_layout()
    delta_path = args.out_dir / "t_sweep_accuracy_differences.png"
    delta_fig.savefig(delta_path, dpi=220, bbox_inches="tight")
    plt.close(delta_fig)
    return accuracy_path, delta_path


def main():
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    loader_args = SimpleNamespace(
        split_seed=args.split_seed,
        val_size=args.val_size,
        limit_train=0,
        limit_val=0,
        limit_test=0,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
    )
    test_loader = make_loaders(loader_args)[3]
    criterion = nn.CrossEntropyLoss()
    rows = []
    for depth in sorted(set(args.depths)):
        for method in METHODS:
            path = checkpoint_path(depth, method, args.checkpoint_root)
            model, saved_args = load_checkpoint(path, device)
            rows.extend(
                evaluate_model(
                    model, saved_args, test_loader, criterion, args, device, depth, method
                )
            )
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
    deltas = paired_deltas(rows)
    results_path = args.out_dir / "t_sweep_results.csv"
    deltas_path = args.out_dir / "t_sweep_paired_differences.csv"
    write_csv(results_path, rows)
    write_csv(deltas_path, deltas)
    accuracy_path, delta_path = plot(rows, deltas, args)
    print(f"[DONE] results={results_path}", flush=True)
    print(f"[DONE] deltas={deltas_path}", flush=True)
    print(f"[DONE] absolute_plot={accuracy_path}", flush=True)
    print(f"[DONE] delta_plot={delta_path}", flush=True)


if __name__ == "__main__":
    main()
