#!/usr/bin/env python3
"""Plot L2-wo / L2-all vs MNE-L2 at fixed eval T=L (seed 42).

One figure per L. CIFAR-10 and CIFAR-100 side-by-side.
MNE uses suite=lscale noscale / noscale_nd (rc=0.0256).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
L2_ROOT = ROOT / "important_results" / "cifar_vgg16_l2_lscale_seed42"
MNE_ROOT = ROOT / "important_results" / "cifar_vgg16_mne_lscale_seed42"
PLOT_ROOT = ROOT / "plots"

METHODS = (
    ("nodetach", r"MNE-L2 no-detach"),
    ("mne", r"MNE-L2 detach"),
    ("l2wo", "L2-wo"),
    ("l2all", "L2-all"),
)
COLORS = {
    "nodetach": "#8c564b",
    "mne": "#D62728",
    "l2wo": "#009E73",
    "l2all": "#6F3FA0",
}
LS = {"nodetach": "-", "mne": "-", "l2wo": "-.", "l2all": "--"}
LW = {"nodetach": 2.5, "mne": 2.0, "l2wo": 2.0, "l2all": 2.0}


def _read_sweep(path: Path) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            xs.append(float(row["sigma"]))
            ys.append(float(row["accuracy"]))
    return xs, ys


def _series(dataset: str, method: str, L: int) -> tuple[list[float], list[float]] | None:
    if method == "nodetach":
        path = MNE_ROOT / dataset / f"lscale_noscale_nd_L{L}" / "test_sweep.csv"
    elif method == "mne":
        path = MNE_ROOT / dataset / f"lscale_noscale_L{L}" / "test_sweep.csv"
    else:
        path = L2_ROOT / dataset / f"l2l_{method}_L{L}" / "test_sweep.csv"
    if not path.is_file():
        return None
    return _read_sweep(path)


def plot_one(L: int, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharex=True)
    for ax, ds, title, ylim in (
        (axes[0], "cifar10", "CIFAR-10 VGG-16 test", (0, 100)),
        (axes[1], "cifar100", "CIFAR-100 VGG-16 test", (0, 70)),
    ):
        for key, label in METHODS:
            packed = _series(ds, key, L)
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
        f"L2 vs MNE-L2 · eval T=L={L}, seed 42, post-IF  ·  "
        "MNE noscale rc=0.0256 (no-detach / detach)",
        fontsize=11,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, choices=(4, 8, 16, 32), nargs="+", default=(8, 32))
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PLOT_ROOT,
        help="Directory for output PNGs",
    )
    args = parser.parse_args()
    for L in args.L:
        plot_one(L, args.out_root / f"cifar10_cifar100_vgg16_l2_vs_mne_L{L}_seed42_test.png")


if __name__ == "__main__":
    main()
