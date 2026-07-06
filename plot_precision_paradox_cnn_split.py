import matplotlib.pyplot as plt
import numpy as np


def add_labels(ax, bars):
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h + 0.20,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_one(model_tag: str, ann_vals, snn_vals, out_base: str):
    x = np.arange(2)
    width = 0.36

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.6), dpi=220, constrained_layout=True)
    bars_ann = ax.bar(
        x - width / 2,
        ann_vals,
        width,
        label="ANN",
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.92,
        zorder=3,
    )
    bars_snn = ax.bar(
        x + width / 2,
        snn_vals,
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
    ax.set_xticklabels(["L=4", "L=16"], fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(88, 100.8)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False)

    fig.savefig(f"{out_base}.png", bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_base}.png")
    print(f"[SAVED] {out_base}.pdf")


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.labelsize": 11, "legend.fontsize": 10})

    # From figure1_precision_paradox_cnn_no_caption.png source data
    ann_small = [97.530, 97.822]  # c2c4, L=4/16
    snn_small = [96.910, 90.680]  # c2c4, L=4/16
    ann_large = [99.018, 99.080]  # c16c32, L=4/16
    snn_large = [98.756, 98.806]  # c16c32, L=4/16

    out_small = "important results/figure1_precision_paradox_cnn_small_no_caption"
    out_large = "important results/figure1_precision_paradox_cnn_large_no_caption"
    plot_one("small", ann_small, snn_small, out_small)
    plot_one("large", ann_large, snn_large, out_large)


if __name__ == "__main__":
    main()
