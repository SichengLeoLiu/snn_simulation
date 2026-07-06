import matplotlib.pyplot as plt
import numpy as np


def _add_value_labels(ax, bars, values):
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.22,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )


def plot_family(ax, groups, ann_mean, snn_mean, y_min):
    x = np.arange(len(groups))
    width = 0.36

    ann_bars = ax.bar(
        x - width / 2,
        ann_mean,
        width,
        label="ANN",
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.92,
        zorder=3,
    )
    snn_bars = ax.bar(
        x + width / 2,
        snn_mean,
        width,
        label="SNN (T=16)",
        color="#F58518",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.92,
        zorder=3,
    )

    _add_value_labels(ax, ann_bars, ann_mean)
    _add_value_labels(ax, snn_bars, snn_mean)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(y_min, 101)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
        }
    )

    groups_cnn = [
        "Small c2c4\nL=4",
        "Small c2c4\nL=16",
        "Large c16c32\nL=4",
        "Large c16c32\nL=16",
    ]
    ann_cnn = np.array([97.530, 97.822, 99.018, 99.080])
    snn_cnn = np.array([96.910, 90.680, 98.756, 98.806])

    groups_fc = [
        "Small h8\nL=4",
        "Small h8\nL=16",
        "Large h128\nL=4",
        "Large h128\nL=16",
    ]
    ann_fc = np.array([87.696, 77.412, 97.690, 97.812])
    snn_fc = np.array([88.704, 76.886, 97.842, 97.796])
    fig_cnn, ax_cnn = plt.subplots(1, 1, figsize=(6.8, 4.9), dpi=220, constrained_layout=True)
    plot_family(ax_cnn, groups_cnn, ann_cnn, snn_cnn, y_min=80)
    handles, labels = ax_cnn.get_legend_handles_labels()
    ax_cnn.legend(handles, labels, loc="lower left", frameon=False)
    out_cnn = "important results/figure1_precision_paradox_cnn_no_caption"
    fig_cnn.savefig(f"{out_cnn}.png", bbox_inches="tight")
    fig_cnn.savefig(f"{out_cnn}.pdf", bbox_inches="tight")
    plt.close(fig_cnn)
    print(f"[SAVED] {out_cnn}.png")
    print(f"[SAVED] {out_cnn}.pdf")

    fig_fc, ax_fc = plt.subplots(1, 1, figsize=(6.8, 4.9), dpi=220, constrained_layout=True)
    plot_family(ax_fc, groups_fc, ann_fc, snn_fc, y_min=90)
    handles, labels = ax_fc.get_legend_handles_labels()
    ax_fc.legend(handles, labels, loc="lower left", frameon=False)
    out_fc = "important results/figure1_precision_paradox_fc_no_caption"
    fig_fc.savefig(f"{out_fc}.png", bbox_inches="tight")
    fig_fc.savefig(f"{out_fc}.pdf", bbox_inches="tight")
    plt.close(fig_fc)
    print(f"[SAVED] {out_fc}.png")
    print(f"[SAVED] {out_fc}.pdf")


if __name__ == "__main__":
    main()
