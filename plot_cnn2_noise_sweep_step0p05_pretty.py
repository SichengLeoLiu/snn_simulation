import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
GADI = ROOT / "all_results_from_gadi"
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


def paths_for_mode(if_mode: str):
    if if_mode == "normal":
        data_root = GADI / "cnn2_noise_sweep_step0p05_full_extracted" / "normal"
        out_dir = GADI / "cnn2_noise_sweep_step0p05_plots" / "normal"
    else:
        data_root = GADI / "cnn2_noise_sweep_step0p05_full_extracted"
        out_dir = GADI / "cnn2_noise_sweep_step0p05_plots"
    return data_root, out_dir


def collect_arch_method(data_root: Path, arch: str, method: str):
    seed_files = sorted((data_root / arch / method).glob("seed_*/noise_sweep_matrix_*.csv"))
    if not seed_files:
        return [], [], []

    bucket = defaultdict(list)
    for fp in seed_files:
        sigmas, accs = read_matrix_csv(fp)
        for s, a in zip(sigmas, accs):
            bucket[round(s, 6)].append(a)

    xs = sorted(bucket.keys())
    ys = [statistics.mean(bucket[x]) for x in xs]
    ysem = [
        (statistics.stdev(bucket[x]) / (len(bucket[x]) ** 0.5)) if len(bucket[x]) > 1 else 0.0
        for x in xs
    ]
    return xs, ys, ysem


def plot_one_arch(data_root: Path, out_dir: Path, arch: str):
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
    fig, ax = plt.subplots(figsize=(10.8, 7.2), dpi=220)

    all_y = []
    for method in METHODS:
        xs, ys, ysem = collect_arch_method(data_root, arch, method)
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
        ax.fill_between(
            xs,
            [y - s for y, s in zip(ys, ysem)],
            [y + s for y, s in zip(ys, ysem)],
            color=style["color"],
            alpha=0.14,
            linewidth=0,
        )
        all_y.extend([y - s for y, s in zip(ys, ysem)])
        all_y.extend([y + s for y, s in zip(ys, ysem)])

    ax.set_xlabel("Gaussian noise sigma")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([i / 10 for i in range(11)])
    ax.margins(x=0)
    if all_y:
        ymin = min(all_y)
        ymax = max(all_y)
        pad = max(0.8, 0.08 * (ymax - ymin))
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(alpha=0.24, linewidth=0.9)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{arch}_three_regs_noise_sweep_step0p05.png"
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def main():
    p = argparse.ArgumentParser(description="Plot CNN2 three-regs noise sweep curves")
    p.add_argument(
        "--if-mode",
        choices=["rate_uniform", "normal"],
        default="rate_uniform",
        help="IF mode subfolder under all_results_from_gadi (default: rate_uniform)",
    )
    args = p.parse_args()
    data_root, out_dir = paths_for_mode(args.if_mode)
    if not data_root.exists():
        raise FileNotFoundError(f"Missing data root: {data_root}")
    for arch in ARCHES:
        plot_one_arch(data_root, out_dir, arch)


if __name__ == "__main__":
    main()
