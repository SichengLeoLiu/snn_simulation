#!/usr/bin/env python3
"""Plot VGG-16 L2-wo/L2-all (existing) vs FG-MNE-U/L1-wo (new) at T=L."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
L2_ROOT = ROOT / "important_results" / "cifar_vgg16_l2_lscale_seed42"
NEW_ROOT = ROOT / "important_results" / "cifar_fgmneu_l1_lscale_seed42"
PLOT_ROOT = ROOT / "plots"

METHODS = (
    ("fgmneu", r"FG-MNE-U"),
    ("l1wo", "L1-wo"),
    ("l2wo", "L2-wo"),
    ("l2all", "L2-all"),
)
COLORS = {
    "fgmneu": "#0072B2",
    "l1wo": "#E69F00",
    "l2wo": "#009E73",
    "l2all": "#6F3FA0",
}
LS = {"fgmneu": "-", "l1wo": ":", "l2wo": "-.", "l2all": "--"}
LW = {"fgmneu": 2.5, "l1wo": 2.0, "l2wo": 2.0, "l2all": 2.0}


def _read_sweep(path: Path) -> tuple[list[float], list[float]] | None:
    if not path.is_file():
        return None
    xs, ys = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            xs.append(float(row["sigma"]))
            ys.append(float(row["accuracy"]))
    return xs, ys


def _series(dataset: str, method: str, L: int, new_root: Path) -> tuple[list[float], list[float]] | None:
    if method in ("fgmneu", "l1wo"):
        path = new_root / dataset / f"vgg16_{method}_L{L}" / "test_sweep.csv"
    else:
        path = L2_ROOT / dataset / f"l2l_{method}_L{L}" / "test_sweep.csv"
    return _read_sweep(path)


def plot_one(L: int, out: Path, new_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharex=True)
    for ax, ds, title, ylim in (
        (axes[0], "cifar10", "CIFAR-10 VGG-16 test", (0, 100)),
        (axes[1], "cifar100", "CIFAR-100 VGG-16 test", (0, 70)),
    ):
        for key, label in METHODS:
            packed = _series(ds, key, L, new_root)
            if packed is None:
                continue
            xs, ys = packed
            ax.plot(
                xs,
                ys,
                color=COLORS[key],
                linestyle=LS[key],
                linewidth=LW[key],
                label=label,
            )
        ax.set_title(title)
        ax.set_xlabel(r"post-IF $\sigma$")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlim(0, 5)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower left")
    fig.suptitle(
        f"L2 / L1 / FG-MNE-U · eval T=L={L}, seed 42, post-IF  ·  "
        r"locked $\eta$, no per-$L$ retune",
        fontsize=11,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, choices=(4, 8, 32), nargs="+", default=(4, 8, 32))
    parser.add_argument("--new-root", type=Path, default=NEW_ROOT)
    parser.add_argument("--out-root", type=Path, default=PLOT_ROOT)
    args = parser.parse_args()
    for L in args.L:
        plot_one(L, args.out_root / f"cifar10_cifar100_vgg16_fgmneu_l1_L{L}_seed42_test.png", args.new_root)


if __name__ == "__main__":
    main()
