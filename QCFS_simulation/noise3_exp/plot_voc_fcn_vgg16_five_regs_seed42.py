#!/usr/bin/env python3
"""VOC 2012 FCN-32s five-regularizer val mIoU vs post-IF σ (seed 42)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "important_results" / "voc_fcn_vgg16_five_regs_seed42"
PLOT_DEFAULT = ROOT / "plots" / "voc2012_fcn_vgg16_five_regs_seed42_val.png"

METHODS = (
    ("nodetach", r"MNE-L2 no-detach", "#8c564b", "-", 2.5),
    ("mne", r"MNE-L2 detach", "#D62728", "-", 2.2),
    ("l2wo", "L2-wo", "#009E73", "-.", 2.0),
    ("l2all", "L2-all", "#6F3FA0", "--", 2.0),
    ("l1wo", "L1-wo", "#E69F00", ":", 2.0),
)


def _read_sweep(path: Path) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            xs.append(float(row["sigma"]))
            ys.append(float(row["mIoU"]))
    return xs, ys


def plot_val(data_root: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=200)
    missing = []
    for key, label, color, ls, lw in METHODS:
        path = data_root / key / "val_sweep.csv"
        if not path.is_file():
            missing.append(str(path))
            continue
        xs, ys = _read_sweep(path)
        ax.plot(
            xs,
            ys,
            color=color,
            linestyle=ls,
            linewidth=lw,
            marker="o",
            markersize=5.5,
            label=label,
        )
    if missing:
        raise SystemExit("missing sweeps:\n  " + "\n  ".join(missing))
    ax.set_title("VOC 2012 val  ·  FCN-32s VGG-16-BN-IF  ·  seed 42")
    ax.set_xlabel(r"post-IF $\sigma$")
    ax.set_ylabel("mIoU (%)")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 45)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, frameon=True)
    fig.suptitle(
        r"Segmentation transfer  ·  T=16 rate_uniform, ImageNet init, post-IF",
        fontsize=11,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out", type=Path, default=PLOT_DEFAULT)
    args = parser.parse_args()
    plot_val(args.data_root, args.out)


if __name__ == "__main__":
    main()
