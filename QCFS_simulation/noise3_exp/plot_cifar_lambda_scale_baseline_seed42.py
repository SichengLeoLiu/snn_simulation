#!/usr/bin/env python3
"""Plot L2-wo λ-scale / λ-grow vs λ-MNE-U (seed 42)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IR = ROOT / "important_results"
PLOT = ROOT / "plots"
SCALE = IR / "cifar_lambda_scale_baseline_seed42"
GROW = IR / "cifar_l2wo_lambda_grow_seed42"


def _read(path: Path):
    if not path.is_file():
        return None
    xs, ys = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            xs.append(float(row["sigma"]))
            ys.append(float(row["accuracy"]))
    return xs, ys


def _card(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale-root", type=Path, default=SCALE)
    parser.add_argument("--grow-root", type=Path, default=GROW)
    parser.add_argument("--arch", choices=("vgg16", "resnet18"), default="vgg16")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.out is None:
        args.out = PLOT / f"baseline_{args.arch}_lambda_scale_seed42_test.png"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0), sharex=True)
    handles = []
    for ax, ds, ylim in (
        (axes[0], "cifar10", (0, 100) if args.arch == "vgg16" else (40, 100)),
        (axes[1], "cifar100", (0, 85) if args.arch == "vgg16" else (15, 85)),
    ):
        summary = _card(args.scale_root / ds / f"{args.arch}_scale_summary.json")
        selected = summary.get("selected_tag") if summary else None
        specs = [
            ("1", r"L2-wo  ($\lambda\times 1$)", "#009E73", "-.", 2.0),
            ("match_mean", r"L2-wo  (match mean $\lambda$)", "#CC79A7", ":", 2.2),
            (selected or "1p5", r"L2-wo  (val-matched clean)", "#E69F00", "--", 2.3),
            ("lambda_mneu", r"$\lambda$-MNE-U", "#0072B2", "-", 2.5),
        ]
        drawn = set()
        for tag, label, color, ls, lw in specs:
            if tag in drawn or tag is None:
                continue
            path = (
                args.scale_root
                / ds
                / f"{args.arch}_scale_{tag}"
                / "seed42"
                / "test_sweep.csv"
            )
            packed = _read(path)
            if packed is None:
                print(f"missing {ds}/{tag}: {path}")
                continue
            xs, ys = packed
            line, = ax.plot(xs, ys, color=color, linestyle=ls, linewidth=lw, label=label)
            if ax is axes[0]:
                handles.append(line)
            drawn.add(tag)
        grow_dir = args.grow_root / ds
        if grow_dir.is_dir():
            for path in sorted(grow_dir.glob(f"{args.arch}_grow_c*/seed42/test_sweep.csv")):
                packed = _read(path)
                if packed is None:
                    continue
                tag = path.parents[1].name.split("_c", 1)[-1]
                xs, ys = packed
                line, = ax.plot(xs, ys, color="#8c564b", linestyle="-", linewidth=1.6, alpha=0.8, label=rf"$\lambda$-grow $\eta$={tag.replace('p', '.')}")
                if ax is axes[0]:
                    handles.append(line)
        card = _card(args.scale_root / ds / f"{args.arch}_scale_{selected or '1'}" / "seed42" / "scorecard.json")
        lam = _card(args.scale_root / ds / f"{args.arch}_scale_lambda_mneu" / "seed42" / "scorecard.json")
        ax.set_title(f"{ds} {args.arch}  ·  seed 42")
        ax.set_xlabel(r"post-IF Gaussian $\sigma$")
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.set_xlim(0, 5)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        note = []
        if card:
            note.append(
                rf"sel $\lambda$={card['if_thresh_mean']:.2f} fire={card['test_clean_fire']:.3f} "
                rf"E={card['test_clean_energy_mJ']:.3f} mJ"
            )
        if lam:
            note.append(
                rf"$\lambda$-MNE-U $\lambda$={lam['if_thresh_mean']:.2f} fire={lam['test_clean_fire']:.3f} "
                rf"E={lam['test_clean_energy_mJ']:.3f} mJ"
            )
        if note:
            ax.text(0.02, 0.02, "\n".join(note), transform=ax.transAxes, fontsize=7.5, va="bottom")
    fig.suptitle(
        rf"{args.arch}  ·  threshold-scaling baseline vs $\lambda$-MNE-U  ·  seed 42",
        fontsize=11,
    )
    fig.tight_layout()
    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=min(4, len(handles)),
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
