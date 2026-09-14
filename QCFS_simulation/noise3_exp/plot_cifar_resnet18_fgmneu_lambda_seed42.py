#!/usr/bin/env python3
"""Plot ResNet-18 seed-42 detach / ∇λ-only / FG-MNE-U (all with unmatched L2)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IR = ROOT / "important_results"
PLOT_ROOT = ROOT / "plots"
DETACH = IR / "cifar_mne_unmatched_head_l2_seed42"
FULL = IR / "cifar_mne_nodetach_unmatched_head_l2_seed42"
LAMBDA = IR / "cifar_resnet18_fgmneu_lambda_seed42"

SPECS = (
    ("detach", r"detach ($\nabla\gamma{=}0,\nabla\lambda{=}0$)", "#D62728", "-.", 2.0,
     lambda ds: DETACH / ds / "resnet18_mne_head" / "seed42" / "test_sweep.csv"),
    ("lambda", r"$\nabla\lambda$ only", "#9467BD", "-", 2.2,
     lambda ds: LAMBDA / ds / "resnet18_fgmneu_lambda" / "seed42" / "test_sweep.csv"),
    ("full", r"FG-MNE-U ($\nabla\gamma,\nabla\lambda$)", "#0072B2", "-", 2.5,
     lambda ds: FULL / ds / "resnet18_mne_head" / "seed42" / "test_sweep.csv"),
)


def _read(path: Path) -> tuple[list[float], list[float]] | None:
    if not path.is_file():
        return None
    xs, ys = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            xs.append(float(row["sigma"]))
            ys.append(float(row["accuracy"]))
    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-root", type=Path, default=LAMBDA)
    parser.add_argument("--out", type=Path, default=PLOT_ROOT / "ablation_resnet18_fgmneu_lambda_seed42_test.png")
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
        for key, label, color, ls, lw, getter in SPECS:
            if key == "lambda":
                path = args.lambda_root / ds / "resnet18_fgmneu_lambda" / "seed42" / "test_sweep.csv"
            else:
                path = getter(ds)
            packed = _read(path)
            if packed is None:
                print(f"missing {ds}/{key}: {path}")
                continue
            xs, ys = packed
            line, = ax.plot(xs, ys, color=color, linestyle=ls, linewidth=lw, label=label)
            if ax is axes[0]:
                handles.append(line)
        ax.set_title(title + "  ·  seed 42")
        ax.set_xlabel(r"post-IF $\sigma$")
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.set_xlim(0, 5)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        r"ResNet-18  ·  unmatched L2 on  ·  $\nabla\lambda$ only vs FG-MNE-U  ·  seed 42",
        fontsize=11,
    )
    fig.tight_layout()
    if handles:
        fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.02))
        fig.subplots_adjust(bottom=0.22)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
