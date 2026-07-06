import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
L2_MNE_CSV = ROOT / "imagenet_resnet18_l2_vs_mnel2_combined.csv"
NO_REG_CSV = ROOT / "imagenet_resnet18_no_reg_noise_sweep.csv"
SIGMA_STEP = 0.1

STYLES = {
    "L2": {"color": "#ff7f0e", "label": "L2"},
    "MNE-L2": {"color": "#1f77b4", "label": "MNE-L2"},
    "No Reg": {"color": "#2ca02c", "label": "No Reg"},
}


def load_l2_mne():
    sigmas, l2, mne = [], [], []
    with L2_MNE_CSV.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sigmas.append(float(r["sigma"]))
            l2.append(float(r["acc_l2_weight_decay"]))
            mne.append(float(r["acc_mne_l2_rc1e-4"]))
    return sigmas, l2, mne


def load_no_reg():
    sigmas, accs = [], []
    with NO_REG_CSV.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = round(float(r["sigma"]), 6)
            if abs(s / SIGMA_STEP - round(s / SIGMA_STEP)) > 1e-6:
                continue
            sigmas.append(s)
            accs.append(float(r["acc"]))
    return sigmas, accs


def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 18,
            "axes.labelsize": 20,
            "axes.titlesize": 20,
            "legend.fontsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )


def plot_line(ax, sigmas, values, label_key):
    style = STYLES[label_key]
    ax.plot(
        sigmas,
        values,
        marker="o",
        markersize=6.5,
        linewidth=2.8,
        color=style["color"],
        label=style["label"],
        markerfacecolor=style["color"],
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=3,
    )


def plot_combined(out_base: Path):
    setup_style()
    sigmas, l2, mne = load_l2_mne()
    sigmas_nr, no_reg = load_no_reg()

    fig, ax = plt.subplots(figsize=(10.8, 7.2), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plot_line(ax, sigmas, l2, "L2")
    plot_line(ax, sigmas, mne, "MNE-L2")
    plot_line(ax, sigmas_nr, no_reg, "No Reg")

    ax.set_xlabel("Gaussian noise sigma")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([i / 10 for i in range(11)])
    ax.margins(x=0)
    ax.set_ylim(0, 65)
    ax.grid(axis="both", alpha=0.24, linewidth=0.9, zorder=0)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    for ext in (".png", ".pdf"):
        out = out_base.with_suffix(ext)
        fig.savefig(out, facecolor="white", bbox_inches="tight")
        print(f"[SAVED] {out}")
    plt.close(fig)


def main():
    plot_combined(ROOT / "imagenet_resnet18_l2_vs_mnel2_combined")
    plot_combined(ROOT / "imagenet_resnet18_l2_vs_mnel2_combined_no_caption")


if __name__ == "__main__":
    main()
