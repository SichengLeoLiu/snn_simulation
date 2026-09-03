from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
BRANCH_ORDER = (
    "A_l2_wo",
    "B_global_cmne",
    "C_layerwise_cmne",
    "D_global_cmne_frozen",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_endpoints(path: Path) -> tuple[float, float]:
    for row in _read_rows(path):
        if int(float(row["L"])) != 16:
            continue
        values = {
            float(key): float(value)
            for key, value in row.items()
            if key not in ("L", "T") and value not in (None, "")
        }
        return values[0.0], values[1.0]
    raise ValueError(f"No L=16 row in {path}")


def _matrix_path(out_root: Path, branch: str) -> Path:
    return (
        out_root
        / "fixed_epoch30"
        / branch
        / "noise_sweep_matrix_fashion_mnist_cnn2_c16_c32_T16_"
        "mode_rate_uniform_schedule_normal_seed_42.csv"
    )


def _plot(rows: list[dict], out_root: Path) -> None:
    labels = [row["branch"].split("_", 1)[0] for row in rows]
    x = list(range(len(rows)))
    width = 0.36
    metrics = (
        ("snn_sigma0", "SNN clean accuracy (%)"),
        ("snn_sigma1", "SNN accuracy at sigma=1 (%)"),
        ("drop", "Accuracy drop (points)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, (metric, title) in zip(axes, metrics):
        best = [row[f"best_{metric}"] for row in rows]
        fixed = [row[f"epoch30_{metric}"] for row in rows]
        ax.bar([value - width / 2 for value in x], best, width, label="Best ANN checkpoint")
        ax.bar([value + width / 2 for value in x], fixed, width, label="Fixed epoch 30")
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Accuracy / drop (points)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Fashion-MNIST c16-c32: checkpoint timing changes robustness")
    fig.tight_layout()
    fig.savefig(out_root / "best_vs_epoch30_endpoints.png", dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        type=Path,
        default=(
            ROOT.parent
            / "important_results"
            / "fashion_calibrated_mne_shared_fork_c16c32_seed42_ep30"
        ),
    )
    args = parser.parse_args()
    args.out_root = args.out_root.resolve()
    return args


def main() -> None:
    args = parse_args()
    best_by_branch = {
        row["branch"]: row for row in _read_rows(args.out_root / "summary.csv")
    }
    history = _read_rows(args.out_root / "training_history.csv")
    final_ann = {
        branch: float(
            [row for row in history if row["branch"] == branch][-1]["val_acc"]
        )
        for branch in BRANCH_ORDER
    }

    rows = []
    for branch in BRANCH_ORDER:
        best = best_by_branch[branch]
        epoch30_snn0, epoch30_snn1 = _read_endpoints(
            _matrix_path(args.out_root, branch)
        )
        best_snn0 = float(best["snn_sigma0"])
        best_snn1 = float(best["snn_sigma1"])
        rows.append(
            {
                "branch": branch,
                "best_epoch": int(float(best["best_epoch"])),
                "best_ann_acc": float(best["best_ann_acc"]),
                "best_snn_sigma0": best_snn0,
                "best_snn_sigma1": best_snn1,
                "best_drop": best_snn0 - best_snn1,
                "epoch30_ann_acc": final_ann[branch],
                "epoch30_snn_sigma0": epoch30_snn0,
                "epoch30_snn_sigma1": epoch30_snn1,
                "epoch30_drop": epoch30_snn0 - epoch30_snn1,
                "sigma1_change_epoch30_minus_best": epoch30_snn1 - best_snn1,
            }
        )

    output = args.out_root / "checkpoint_timing_summary.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, args.out_root)
    print(output)


if __name__ == "__main__":
    main()
