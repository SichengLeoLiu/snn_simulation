import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
L2_MNE_CSV = ROOT / "imagenet_resnet18_l2_vs_mnel2_combined.csv"
NO_REG_CSV = ROOT / "imagenet_resnet18_no_reg_noise_sweep.csv"
OUT_CSV = ROOT / "imagenet_resnet18_three_regs_robustness_score.csv"
OUT_BASE = ROOT / "imagenet_resnet18_three_regs_robustness_score_bar"

METHODS = [
    ("L2", "acc_l2_weight_decay", "#ff7f0e"),
    ("MNE-L2", "acc_mne_l2_rc1e-4", "#1f77b4"),
    ("No Reg", None, "#2ca02c"),
]


def robust_score(sigmas, accs):
    a0 = accs[0]
    if a0 <= 0:
        return 0.0
    rs = 0.0
    for i in range(len(sigmas) - 1):
        ds = sigmas[i + 1] - sigmas[i]
        if ds <= 0:
            continue
        rs += 0.5 * (accs[i] / a0 + accs[i + 1] / a0) * ds
    return rs


def load_curves():
    sigmas, l2, mne = [], [], []
    with L2_MNE_CSV.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sigmas.append(float(r["sigma"]))
            l2.append(float(r["acc_l2_weight_decay"]))
            mne.append(float(r["acc_mne_l2_rc1e-4"]))

    sigmas_nr, no_reg = [], []
    with NO_REG_CSV.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sigmas_nr.append(float(r["sigma"]))
            no_reg.append(float(r["acc"]))

    return {
        "L2": (sigmas, l2),
        "MNE-L2": (sigmas, mne),
        "No Reg": (sigmas_nr, no_reg),
    }


def compute_rs_rows(curves):
    rows = []
    for label, _, _ in METHODS:
        sigmas, accs = curves[label]
        rs = robust_score(sigmas, accs)
        rows.append(
            {
                "model": "resnet18",
                "method": label,
                "A0": accs[0],
                "RS": rs,
            }
        )
    return rows


def save_csv(rows):
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "method", "A0", "RS"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "A0": f"{r['A0']:.6f}", "RS": f"{r['RS']:.6f}"})
    print(f"[SAVED] {OUT_CSV}")


def plot_bar(rows):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 18,
            "axes.labelsize": 20,
            "legend.fontsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )

    labels = [r["method"] for r in rows]
    values = [r["RS"] for r in rows]
    colors = [c for _, _, c in METHODS]

    fig, ax = plt.subplots(figsize=(8.8, 6.8), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        values,
        width=0.55,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.92,
    )
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            v + 0.012,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=14,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Robustness Score")
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis="y", alpha=0.24, linewidth=0.9)
    fig.tight_layout()

    for ext in (".png", ".pdf"):
        out = OUT_BASE.with_suffix(ext)
        fig.savefig(out, facecolor="white", bbox_inches="tight")
        print(f"[SAVED] {out}")
    plt.close(fig)


def main():
    curves = load_curves()
    rows = compute_rs_rows(curves)
    save_csv(rows)
    plot_bar(rows)

    print("\n=== ImageNet ResNet18 Robustness Score (sigma=0~1, step=0.1) ===")
    for r in rows:
        print(f"{r['method']:<8}  A(0)={r['A0']:.3f}%  RS={r['RS']:.6f}")


if __name__ == "__main__":
    main()
