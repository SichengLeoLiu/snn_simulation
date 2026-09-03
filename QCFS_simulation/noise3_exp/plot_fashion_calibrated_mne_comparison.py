from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _read_matrix(path: Path) -> dict[float, float]:
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    return {
        float(key): float(value)
        for key, value in row.items()
        if key not in {"L", "T"} and value not in {None, ""}
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare weights-only L2 with L2-calibrated MNE noise sweeps."
    )
    parser.add_argument("--l2", required=True, type=Path)
    parser.add_argument("--mne", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    l2 = _read_matrix(args.l2)
    mne = _read_matrix(args.mne)
    sigmas = sorted(set(l2) & set(mne))
    if not sigmas:
        raise ValueError("The two sweep files have no shared sigma values")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "sigma": f"{sigma:g}",
            "l2_weights_only": f"{l2[sigma]:.6f}",
            "calibrated_mne_alpha_0p1": f"{mne[sigma]:.6f}",
            "mne_minus_l2": f"{mne[sigma] - l2[sigma]:.6f}",
        }
        for sigma in sigmas
    ]
    csv_path = args.out_dir / "full_sweep_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, (ax_acc, ax_delta) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_acc.plot(sigmas, [l2[s] for s in sigmas], marker="o", label="L2 weights-only")
    ax_acc.plot(
        sigmas,
        [mne[s] for s in sigmas],
        marker="o",
        label="L2-calibrated MNE (alpha=0.1)",
    )
    ax_acc.set_ylabel("SNN accuracy (%)")
    ax_acc.grid(alpha=0.25)
    ax_acc.legend()

    deltas = [mne[s] - l2[s] for s in sigmas]
    ax_delta.axhline(0.0, color="black", linewidth=1)
    ax_delta.plot(sigmas, deltas, marker="o", color="#c44e52")
    ax_delta.set_xlabel("Gaussian noise sigma (post-input-IF)")
    ax_delta.set_ylabel("Delta (pp)")
    ax_delta.grid(alpha=0.25)

    fig.suptitle("Fashion-MNIST cnn2_c8_c16, seed 42, L=T=16")
    fig.tight_layout()
    figure_path = args.out_dir / "full_sweep_comparison.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] {csv_path}")
    print(f"[DONE] {figure_path}")


if __name__ == "__main__":
    main()
