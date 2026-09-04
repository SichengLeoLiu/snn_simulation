#!/usr/bin/env python3
"""CIFAR-10/100 ResNet-18 5-seed test accuracy vs post-IF σ.

Reads test_sweep.csv under important_results/cifar_resnet18_four_regs_5seed,
writes four_regs_5seed_test_mean_std.csv, and plots mean ± std.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DEFAULT = ROOT / "important_results" / "cifar_resnet18_four_regs_5seed"
PLOT_DEFAULT = ROOT / "plots" / "cifar10_cifar100_resnet18_four_regs_5seed_test.png"
SEEDS = (40, 41, 42, 43, 44)

# Plot order: MNE variants first, then L2 / onesided baselines.
METHODS = (
    ("nodetach", "MNE-L2 no-detach", "#8c564b", "-", 2.5),
    ("mne", "MNE-L2 detach", "#D62728", "-", 2.2),
    ("l2wo", "L2-wo", "#009E73", "-", 2.0),
    ("onesided", "One-sided MNE", "#6F3FA0", "-", 2.0),
    ("l2all", "L2-all", "#0072B2", "--", 2.0),
)


def _mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var)


def _read_sweep(path: Path) -> list[tuple[float, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append((float(row["sigma"]), float(row["accuracy"])))
    return rows


def aggregate(data_root: Path) -> list[dict]:
    out: list[dict] = []
    missing: list[str] = []
    for ds in ("cifar10", "cifar100"):
        for method, label, *_ in METHODS:
            by_sigma: dict[float, list[float]] = defaultdict(list)
            for seed in SEEDS:
                path = data_root / ds / f"r18_{method}" / f"seed{seed}" / "test_sweep.csv"
                if not path.is_file():
                    missing.append(str(path))
                    continue
                for sigma, acc in _read_sweep(path):
                    by_sigma[sigma].append(acc)
            if not by_sigma:
                continue
            for sigma in sorted(by_sigma):
                mean, std = _mean_std(by_sigma[sigma])
                out.append(
                    {
                        "dataset": ds,
                        "method": method,
                        "label": label,
                        "sigma": sigma,
                        "n_seeds": len(by_sigma[sigma]),
                        "acc_mean": mean,
                        "acc_std": std,
                    }
                )
    if missing:
        raise SystemExit("missing sweeps:\n  " + "\n  ".join(missing))
    return out


def write_mean_std(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "method", "label", "sigma", "n_seeds", "acc_mean", "acc_std"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def _series(rows: list[dict], dataset: str, method: str) -> tuple[list[float], list[float], list[float]]:
    rr = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
    rr.sort(key=lambda r: float(r["sigma"]))
    xs = [float(r["sigma"]) for r in rr]
    ys = [float(r["acc_mean"]) for r in rr]
    ss = [float(r["acc_std"]) for r in rr]
    return xs, ys, ss


def plot(rows: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharex=True)
    panels = (
        (axes[0], "cifar10", "CIFAR-10 ResNet-18", (0, 100)),
        (axes[1], "cifar100", "CIFAR-100 ResNet-18", (0, 85)),
    )
    handles = []
    for ax, ds, title, ylim in panels:
        for method, label, color, ls, lw in METHODS:
            xs, ys, ss = _series(rows, ds, method)
            if not xs:
                continue
            (line,) = ax.plot(
                xs,
                ys,
                color=color,
                linestyle=ls,
                linewidth=lw,
                marker="o",
                markersize=4.0,
                label=label,
            )
            ax.fill_between(
                xs,
                [y - s for y, s in zip(ys, ss)],
                [y + s for y, s in zip(ys, ss)],
                color=color,
                alpha=0.16,
                linewidth=0,
            )
            if ax is axes[0]:
                handles.append(line)
        ax.set_title(title)
        ax.set_xlabel(r"per-timestep Gaussian $\sigma$")
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.set_xlim(0, 5)
        ax.set_ylim(*ylim)
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        r"CIFAR ResNet-18  ·  5 seeds (40–44), mean ± std  ·  "
        r"$T{=}16$ rate_uniform, post_input_if",
        fontsize=11,
    )
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="lower center",
        ncol=len(handles),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--out", type=Path, default=PLOT_DEFAULT)
    args = parser.parse_args()
    rows = aggregate(args.data_root)
    write_mean_std(args.data_root / "four_regs_5seed_test_mean_std.csv", rows)
    plot(rows, args.out)


if __name__ == "__main__":
    main()
