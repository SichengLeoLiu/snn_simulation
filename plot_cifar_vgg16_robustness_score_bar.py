import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from robustness_metrics import derivative_robustness_score

ROOT = Path(__file__).resolve().parent
DATASETS = {
    "cifar10": ROOT
    / "all_results_from_gadi"
    / "noise3_exp"
    / "cifar10_vgg16_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16"
    / "cifar10_vgg16_strict_seed_three_regs_noise_sweep_raw.csv",
    "cifar100": ROOT / "all_results_from_gadi" / "cifar100_vgg16_strict_seed_three_regs_noise_sweep_raw.csv",
}
OUT_DIR = ROOT / "all_results_from_gadi" / "cifar_vgg16_robustness_score_plots"

METHODS = ["weight_decay", "mne_l2", "no_regularization"]
METHOD_LABELS = {
    "weight_decay": "L2",
    "mne_l2": "MNE-L2",
    "no_regularization": "No Reg",
}
METHOD_COLORS = {
    "weight_decay": "#ff7f0e",
    "mne_l2": "#1f77b4",
    "no_regularization": "#2ca02c",
}


def compute_drs_rows(raw_rows, dataset: str):
    by_method_seed = defaultdict(lambda: defaultdict(list))
    for r in raw_rows:
        if r["method"] not in METHODS:
            continue
        by_method_seed[r["method"]][int(r["seed"])].append((float(r["sigma"]), float(r["acc"])))

    rows = []
    for method in METHODS:
        vals = []
        for seed in sorted(by_method_seed[method].keys()):
            pairs = sorted(by_method_seed[method][seed], key=lambda x: x[0])
            sigmas = [p[0] for p in pairs]
            accs = [p[1] for p in pairs]
            vals.append(derivative_robustness_score(sigmas, accs))
        if not vals:
            continue
        n = len(vals)
        drs_std = statistics.stdev(vals) if n > 1 else 0.0
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "DRS_mean": statistics.mean(vals),
                "DRS_std": drs_std,
                "DRS_sem": drs_std / (n ** 0.5) if n > 0 else 0.0,
                "n_seeds": n,
            }
        )
    return rows


def load_raw(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))
    return rows


def save_csv(all_rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "method",
                "method_label",
                "DRS_mean",
                "DRS_std",
                "DRS_sem",
                "n_seeds",
            ],
        )
        w.writeheader()
        for r in all_rows:
            w.writerow(
                {
                    **{k: r[k] for k in ("dataset", "method", "method_label", "n_seeds")},
                    "DRS_mean": f"{r['DRS_mean']:.6f}",
                    "DRS_std": f"{r['DRS_std']:.6f}",
                    "DRS_sem": f"{r['DRS_sem']:.6f}",
                }
            )
    print(f"[SAVED] {path}")


def plot_bar(rows, dataset: str, with_sem: bool):
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

    x = np.arange(len(METHODS))
    width = 0.55
    fig, ax = plt.subplots(figsize=(8.8, 6.8), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, method in enumerate(METHODS):
        row = next((r for r in rows if r["method"] == method), None)
        mean = row["DRS_mean"] if row else np.nan
        err = row["DRS_sem"] if row and with_sem else 0.0
        bar = ax.bar(
            i,
            mean,
            width,
            yerr=err if with_sem else None,
            capsize=3 if with_sem else 0,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.92,
        )
        if row and not np.isnan(mean):
            ax.text(
                bar[0].get_x() + bar[0].get_width() / 2.0,
                mean + (err if with_sem else 0.012),
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=14,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS])
    ax.set_ylabel("Derivative Robustness Score (DRS)")
    vals = [r["DRS_mean"] for r in rows]
    if vals:
        ax.set_ylim(max(0.0, min(vals) - 0.08), min(1.08, max(vals) + 0.08))
    ax.grid(axis="y", alpha=0.24, linewidth=0.9)
    fig.tight_layout()

    suffix = "_with_sem" if with_sem else ""
    for ext in (".png", ".pdf"):
        out = OUT_DIR / f"{dataset}_vgg16_three_regs_drs_bar{suffix}{ext}"
        fig.savefig(out, facecolor="white", bbox_inches="tight")
        print(f"[SAVED] {out}")
    plt.close(fig)


def print_table(rows, dataset: str):
    print(f"\n=== {dataset.upper()} VGG16 DRS (mean ± std, n seeds) ===")
    for r in rows:
        print(
            f"{r['method_label']:<8}  DRS={r['DRS_mean']:.6f} ± {r['DRS_std']:.6f}  "
            f"(SEM={r['DRS_sem']:.6f}, n={r['n_seeds']})"
        )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for dataset, path in DATASETS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        rows = compute_drs_rows(load_raw(path), dataset)
        all_rows.extend(rows)
        print_table(rows, dataset)
        plot_bar(rows, dataset, with_sem=False)
        plot_bar(rows, dataset, with_sem=True)

    save_csv(all_rows, OUT_DIR / "cifar10_cifar100_vgg16_three_regs_drs.csv")


if __name__ == "__main__":
    main()
