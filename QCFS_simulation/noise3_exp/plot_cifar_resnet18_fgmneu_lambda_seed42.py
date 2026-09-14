#!/usr/bin/env python3
"""Plot ResNet-18 detach (seed 42) / ∇λ-only (5-seed) / FG-MNE-U (5-seed)."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IR = ROOT / "important_results"
PLOT_ROOT = ROOT / "plots"
DETACH = IR / "cifar_mne_unmatched_head_l2_seed42"
FULL = IR / "cifar_mne_nodetach_unmatched_head_l2_seed42"
LAMBDA = IR / "cifar_resnet18_fgmneu_lambda_seed42"
SEEDS = (40, 41, 42, 43, 44)


def _read(path: Path) -> list[tuple[float, float]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append((float(row["sigma"]), float(row["accuracy"])))
    rows.sort()
    return rows


def _mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    mean = sum(xs) / n
    if n < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var)


def _agg(paths: list[Path]) -> tuple[list[float], list[float], list[float], int] | None:
    buckets: dict[float, list[float]] = defaultdict(list)
    n_files = 0
    for path in paths:
        rows = _read(path)
        if not rows:
            continue
        n_files += 1
        for sigma, acc in rows:
            buckets[round(sigma, 6)].append(acc)
    if not buckets:
        return None
    xs, ys, ss = [], [], []
    for sigma in sorted(buckets):
        mean, std = _mean_std(buckets[sigma])
        xs.append(sigma)
        ys.append(mean)
        ss.append(std)
    return xs, ys, ss, n_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-root", type=Path, default=LAMBDA)
    parser.add_argument(
        "--out",
        type=Path,
        default=PLOT_ROOT / "ablation_resnet18_fgmneu_lambda_5seed_test.png",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharex=True)
    handles = []
    for ax, ds, title, ylim in (
        (axes[0], "cifar10", "CIFAR-10 ResNet-18", (0, 100)),
        (axes[1], "cifar100", "CIFAR-100 ResNet-18", (0, 90)),
    ):
        series = (
            (
                r"detach  (seed 42)",
                "#D62728",
                "-.",
                2.0,
                [DETACH / ds / "resnet18_mne_head" / "seed42" / "test_sweep.csv"],
            ),
            (
                r"$\nabla\lambda$ only",
                "#9467BD",
                "-",
                2.2,
                [
                    args.lambda_root / ds / "resnet18_fgmneu_lambda" / f"seed{s}" / "test_sweep.csv"
                    for s in SEEDS
                ],
            ),
            (
                r"FG-MNE-U  ($\nabla\gamma,\nabla\lambda$)",
                "#0072B2",
                "-",
                2.5,
                [
                    FULL / ds / "resnet18_mne_head" / f"seed{s}" / "test_sweep.csv"
                    for s in SEEDS
                ],
            ),
        )
        for label, color, ls, lw, paths in series:
            packed = _agg(paths)
            if packed is None:
                print(f"missing {ds}/{label}")
                continue
            xs, ys, ss, n = packed
            shown = f"{label}  ({n}-seed ±std)" if n > 1 else label
            if n > 1 and any(v > 0 for v in ss):
                handle = ax.errorbar(
                    xs,
                    ys,
                    yerr=ss,
                    color=color,
                    linestyle=ls,
                    linewidth=lw,
                    marker="o",
                    markersize=4.0,
                    elinewidth=1.1,
                    capsize=2.8,
                    label=shown,
                )
            else:
                (handle,) = ax.plot(
                    xs, ys, color=color, linestyle=ls, linewidth=lw, marker="o", markersize=4.0, label=shown
                )
            if ax is axes[0]:
                handles.append(handle)
        ax.set_title(title)
        ax.set_xlabel(r"post-IF $\sigma$")
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.set_xlim(0, 5)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        r"ResNet-18  ·  unmatched L2 on  ·  $\nabla\lambda$ only vs FG-MNE-U",
        fontsize=11,
    )
    fig.tight_layout()
    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.02),
            fontsize=8.5,
        )
        fig.subplots_adjust(bottom=0.22)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
