import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT_CSV = (
    ROOT
    / "QCFS_simulation"
    / "noise3_exp"
    / "fc3_wd_strict_seed_normal_T0_T16_L4_L16"
    / "fc3_wd_t0_t16_l4_l16_merged_table.csv"
)


def read_rows(path: Path):
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "arch": r["arch"],
                    "hidden_size": int(r["hidden_size"]),
                    "t0_l4": float(r["T0_L4_mean"]),
                    "t16_l4": float(r["T16_L4_mean"]),
                    "t0_l16": float(r["T0_L16_mean"]),
                    "t16_l16": float(r["T16_L16_mean"]),
                }
            )
    rows.sort(key=lambda x: x["hidden_size"])
    return rows


def add_labels(ax, bars):
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h + 0.08,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main():
    rows = [r for r in read_rows(INPUT_CSV) if 8 <= r["hidden_size"] <= 128]
    if not rows:
        raise RuntimeError(f"No h8~h128 rows found in {INPUT_CSV}")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )

    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.8), dpi=220, constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, row in zip(axes, rows):
        x = np.arange(2)
        width = 0.36
        ann = np.array([row["t0_l4"], row["t0_l16"]])
        snn = np.array([row["t16_l4"], row["t16_l16"]])

        bars_ann = ax.bar(
            x - width / 2,
            ann,
            width,
            label="ANN (T=0)",
            color="#4C78A8",
            edgecolor="black",
            linewidth=0.6,
            alpha=0.92,
            zorder=3,
        )
        bars_snn = ax.bar(
            x + width / 2,
            snn,
            width,
            label="SNN (T=16)",
            color="#F58518",
            edgecolor="black",
            linewidth=0.6,
            alpha=0.92,
            zorder=3,
        )
        add_labels(ax, bars_ann)
        add_labels(ax, bars_snn)

        ax.set_xticks(x)
        ax.set_xticklabels(["L=4", "L=16"])
        ax.set_title(f"fc3_h{row['hidden_size']}", fontsize=11, fontweight="bold")
        ax.set_ylim(40, 101)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))

    out_base = ROOT / "important results" / "figure_fc_h8_to_h128_t0_t16_l4_l16_overview_no_caption"
    fig.savefig(f"{out_base}.png", bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    print(f"[SAVED] {out_base}.png")
    print(f"[SAVED] {out_base}.pdf")


if __name__ == "__main__":
    main()
