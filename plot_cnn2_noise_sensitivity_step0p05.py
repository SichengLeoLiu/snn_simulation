import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "all_results_from_gadi" / "cnn2_noise_sweep_step0p05_full_extracted"
OUT_DIR = ROOT / "all_results_from_gadi" / "cnn2_noise_sweep_step0p05_plots"

ARCHES = ["cnn2_c2_c4", "cnn2_c4_c8", "cnn2_c8_c16", "cnn2_c16_c32"]
METHODS = ["weight_decay", "mne_l2", "no_regularization"]

LINE_STYLES = {
    "weight_decay": {"color": "#ff7f0e", "label": "L2"},
    "mne_l2": {"color": "#1f77b4", "label": "MNE L2"},
    "no_regularization": {"color": "#2ca02c", "label": "No reg"},
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


def finite_diff_ns(sigmas, accs):
    mids = []
    vals = []
    for i in range(len(sigmas) - 1):
        ds = sigmas[i + 1] - sigmas[i]
        if ds <= 0:
            continue
        mids.append((sigmas[i] + sigmas[i + 1]) / 2.0)
        vals.append((accs[i] - accs[i + 1]) / ds)
    return mids, vals


def collect_arch_method(arch: str, method: str):
    seed_files = sorted((DATA_ROOT / arch / method).glob("seed_*/noise_sweep_matrix_*.csv"))
    if not seed_files:
        return [], [], []

    bucket = defaultdict(list)
    for fp in seed_files:
        sigmas, accs = read_matrix_csv(fp)
        mids, ns_vals = finite_diff_ns(sigmas, accs)
        for x, y in zip(mids, ns_vals):
            bucket[round(x, 6)].append(y)

    xs = sorted(bucket.keys())
    ys = [statistics.mean(bucket[x]) for x in xs]
    return xs, ys


def plot_one_arch(arch: str):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 18,
            "axes.labelsize": 20,
            "axes.titlesize": 18,
            "legend.fontsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )
    fig, ax = plt.subplots(figsize=(10.6, 7.2), dpi=220)

    all_y = []
    for method in METHODS:
        xs, ys = collect_arch_method(arch, method)
        if not xs:
            continue
        style = LINE_STYLES[method]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=5.8,
            linewidth=2.8,
            color=style["color"],
            label=style["label"],
        )
        all_y.extend(ys)

    ax.set_xlabel("Gaussian noise sigma")
    ax.set_ylabel(r"$\mathrm{NS}$")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([i / 10 for i in range(11)])
    if all_y:
        ymin = min(all_y)
        ymax = max(all_y)
        pad = max(0.5, 0.08 * (ymax - ymin))
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(alpha=0.25, linewidth=0.9)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    out_png = OUT_DIR / f"{arch}_three_regs_noise_sensitivity_step0p05.png"
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for arch in ARCHES:
        plot_one_arch(arch)


if __name__ == "__main__":
    main()
