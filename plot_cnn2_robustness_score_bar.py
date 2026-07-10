import argparse
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from robustness_metrics import derivative_robustness_score

ROOT = Path(__file__).resolve().parent
GADI = ROOT / "all_results_from_gadi"

ARCHES = ["cnn2_c2_c4", "cnn2_c4_c8", "cnn2_c8_c16", "cnn2_c16_c32"]
ARCH_LABELS = {
    "cnn2_c2_c4": "c2c4",
    "cnn2_c4_c8": "c4c8",
    "cnn2_c8_c16": "c8c16",
    "cnn2_c16_c32": "c16c32",
}
METHODS = ["weight_decay", "mne_l2", "no_regularization"]
METHOD_LABELS = {
    "weight_decay": "L2",
    "mne_l2": "MNE L2",
    "no_regularization": "No reg",
}
METHOD_COLORS = {
    "weight_decay": "#ff7f0e",
    "mne_l2": "#1f77b4",
    "no_regularization": "#2ca02c",
}


def read_matrix_csv(path: Path):
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)
    start = 2 if header[:2] == ["L", "T"] else 1
    sigmas = [float(x) for x in header[start:]]
    accs = [float(x) for x in row[start:]]
    return sigmas, accs


def compute_scores(data_root: Path):
    rows = []
    for arch in ARCHES:
        for method in METHODS:
            seed_files = sorted((data_root / arch / method).glob("seed_*/noise_sweep_matrix_*.csv"))
            vals = []
            for fp in seed_files:
                sigmas, accs = read_matrix_csv(fp)
                vals.append(derivative_robustness_score(sigmas, accs))
            if not vals:
                continue
            n = len(vals)
            drs_std = statistics.stdev(vals) if n > 1 else 0.0
            rows.append(
                {
                    "arch": arch,
                    "arch_label": ARCH_LABELS[arch],
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "DRS_mean": statistics.mean(vals),
                    "DRS_std": drs_std,
                    "DRS_sem": drs_std / (n ** 0.5) if n > 0 else 0.0,
                    "n_seeds": n,
                }
            )
    return rows


def data_root_for_mode(if_mode: str) -> Path:
    if if_mode == "normal":
        return GADI / "cnn2_noise_sweep_step0p05_full_extracted" / "normal"
    return GADI / "cnn2_noise_sweep_step0p05_full_extracted"


def out_dir_for_mode(if_mode: str) -> Path:
    if if_mode == "normal":
        return GADI / "cnn2_noise_sweep_step0p05_plots" / "normal"
    return GADI / "cnn2_noise_sweep_step0p05_plots"


def save_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "arch",
                "arch_label",
                "method",
                "method_label",
                "DRS_mean",
                "DRS_std",
                "DRS_sem",
                "n_seeds",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    **r,
                    "DRS_mean": f"{r['DRS_mean']:.6f}",
                    "DRS_std": f"{r['DRS_std']:.6f}",
                    "DRS_sem": f"{r['DRS_sem']:.6f}",
                }
            )


def plot_bar(rows, out_dir: Path, error_key: str | None, suffix: str):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.labelsize": 18,
            "legend.fontsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

    x = np.arange(len(ARCHES))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.8, 6.8), dpi=220)

    for idx, method in enumerate(METHODS):
        means = []
        errs = []
        for arch in ARCHES:
            r = next((v for v in rows if v["arch"] == arch and v["method"] == method), None)
            means.append(r["DRS_mean"] if r else np.nan)
            errs.append(r[error_key] if r and error_key else 0.0)
        xpos = x + (idx - 1) * width
        bars = ax.bar(
            xpos,
            means,
            width,
            yerr=errs if error_key else None,
            capsize=3 if error_key else 0,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
            label=METHOD_LABELS[method],
        )
        for b, v in zip(bars, means):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                v + 0.003,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHES])
    ax.set_xlabel("CNN2 model scale")
    ax.set_ylabel("Derivative Robustness Score (DRS)")
    all_vals = [r["DRS_mean"] for r in rows]
    if all_vals:
        ax.set_ylim(max(0.0, min(all_vals) - 0.08), min(1.02, max(all_vals) + 0.08))
    ax.grid(axis="y", alpha=0.25, linewidth=0.9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"cnn2_three_regs_drs_bar{suffix}.png"
    out_pdf = out_dir / f"cnn2_three_regs_drs_bar{suffix}.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[SAVED] {out_png}")
    print(f"[SAVED] {out_pdf}")


def main():
    p = argparse.ArgumentParser(description="Plot CNN2 three-regs robustness score bar chart")
    p.add_argument(
        "--if-mode",
        choices=["rate_uniform", "normal"],
        default="rate_uniform",
    )
    args = p.parse_args()
    data_root = data_root_for_mode(args.if_mode)
    out_dir = out_dir_for_mode(args.if_mode)
    if not data_root.exists():
        raise FileNotFoundError(f"Missing data root: {data_root}")

    rows = compute_scores(data_root)
    out_csv = out_dir / "cnn2_three_regs_drs.csv"
    save_csv(rows, out_csv)
    print(f"[SAVED] {out_csv}")
    plot_bar(rows, out_dir, error_key=None, suffix="")
    plot_bar(rows, out_dir, error_key="DRS_sem", suffix="_with_sem")


if __name__ == "__main__":
    main()
