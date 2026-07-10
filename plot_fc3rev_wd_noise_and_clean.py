import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from robustness_metrics import derivative_robustness_score

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "important results" / "new_fc3"
OUT_DIR = DEFAULT_DATA_DIR / "plots"

LINE_COLOR = "#1f77b4"
BAR_COLOR = "#4C78A8"
SNN_COLOR = "#F58518"


def clean_merged_path(data_dir: Path, if_mode: str) -> Path:
    if if_mode == "normal":
        return data_dir / "fc3rev_h8_h256_wd_t0_t16_l4_l16_merged.csv"
    return data_dir / "fc3rev_h4_h256_wd_clean_acc_rate_uniform_merged.csv"


def clean_overview_stem(if_mode: str) -> str:
    if if_mode == "normal":
        return "fc3rev_h8_h256_wd_clean_acc_overview"
    return "fc3rev_h4_h256_wd_clean_acc_rate_uniform_overview"


def noise_raw_path(data_dir: Path) -> Path:
    return data_dir / "fc3rev_h8_h256_wd_noise_sweep_raw.csv"


def read_noise_raw(data_dir: Path):
    rows = []
    with noise_raw_path(data_dir).open(newline="") as f:
        rows.extend(csv.DictReader(f))
    return rows


def read_clean_merged(data_dir: Path, if_mode: str):
    merged_path = clean_merged_path(data_dir, if_mode)
    rows = []
    with merged_path.open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "arch": r["arch"],
                    "hidden_size": int(r["hidden_size"]),
                    "T0_L4_mean": float(r["T0_L4_mean"]),
                    "T0_L4_std": float(r["T0_L4_std"]),
                    "T16_L4_mean": float(r["T16_L4_mean"]),
                    "T16_L4_std": float(r["T16_L4_std"]),
                    "T0_L16_mean": float(r["T0_L16_mean"]),
                    "T0_L16_std": float(r["T0_L16_std"]),
                    "T16_L16_mean": float(r["T16_L16_mean"]),
                    "T16_L16_std": float(r["T16_L16_std"]),
                }
            )
    rows.sort(key=lambda x: x["hidden_size"])
    return rows


def arch_label(arch: str) -> str:
    return f"h{arch.split('_h')[1]}"


def collect_arch_curve(noise_rows, arch: str):
    bucket = defaultdict(list)
    for r in noise_rows:
        if r["arch"] != arch:
            continue
        bucket[round(float(r["sigma"]), 6)].append(float(r["acc"]))

    xs = sorted(bucket.keys())
    ys = [statistics.mean(bucket[x]) for x in xs]
    ysem = [
        (statistics.stdev(bucket[x]) / (len(bucket[x]) ** 0.5)) if len(bucket[x]) > 1 else 0.0
        for x in xs
    ]
    return xs, ys, ysem


def compute_drs_rows(noise_rows):
    by_arch_seed = defaultdict(lambda: defaultdict(list))
    for r in noise_rows:
        by_arch_seed[r["arch"]][int(r["seed"])].append((float(r["sigma"]), float(r["acc"])))

    rows = []
    for arch in sorted(by_arch_seed.keys(), key=lambda a: int(a.split("_h")[1])):
        vals = []
        for seed in sorted(by_arch_seed[arch].keys()):
            pairs = sorted(by_arch_seed[arch][seed], key=lambda x: x[0])
            sigmas = [p[0] for p in pairs]
            accs = [p[1] for p in pairs]
            vals.append(derivative_robustness_score(sigmas, accs))
        n = len(vals)
        drs_std = statistics.stdev(vals) if n > 1 else 0.0
        rows.append(
            {
                "arch": arch,
                "hidden_size": int(arch.split("_h")[1]),
                "DRS_mean": statistics.mean(vals),
                "DRS_std": drs_std,
                "DRS_sem": drs_std / (n ** 0.5) if n > 0 else 0.0,
                "n_seeds": n,
            }
        )
    return rows


def setup_style(font_size=16):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": font_size,
            "axes.labelsize": font_size + 2,
            "legend.fontsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
        }
    )


def plot_noise_lines(noise_rows, out_dir: Path = OUT_DIR):
    setup_style(18)
    out_dir.mkdir(parents=True, exist_ok=True)
    archs = sorted({r["arch"] for r in noise_rows}, key=lambda a: int(a.split("_h")[1]))

    all_y = []
    for arch in archs:
        xs, ys, ysem = collect_arch_curve(noise_rows, arch)
        if not xs:
            continue

        fig, ax = plt.subplots(figsize=(10.8, 7.2), dpi=220)
        ax.plot(xs, ys, marker="o", markersize=5.8, linewidth=2.8, color=LINE_COLOR, label="L2 (weight decay)")
        ax.fill_between(
            xs,
            [y - s for y, s in zip(ys, ysem)],
            [y + s for y, s in zip(ys, ysem)],
            color=LINE_COLOR,
            alpha=0.14,
            linewidth=0,
        )
        all_y.extend([y - s for y, s in zip(ys, ysem)])
        all_y.extend([y + s for y, s in zip(ys, ysem)])

        ax.set_xlabel("Gaussian noise sigma")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([i / 10 for i in range(11)])
        ax.grid(alpha=0.24, linewidth=0.9)
        ax.legend(loc="lower left", frameon=False)
        fig.tight_layout()

        label = arch_label(arch)
        out_png = out_dir / f"fc3rev_{label}_wd_noise_sweep_step0p05.png"
        out_pdf = out_dir / f"fc3rev_{label}_wd_noise_sweep_step0p05.pdf"
        fig.savefig(out_png)
        fig.savefig(out_pdf)
        plt.close(fig)
        print(f"[SAVED] {out_png}")

    if all_y:
        ymin = min(all_y)
        ymax = max(all_y)
        pad = max(0.8, 0.08 * (ymax - ymin))
        ylim = (ymin - pad, ymax + pad)
    else:
        ylim = (80, 100)

    for arch in archs:
        xs, ys, ysem = collect_arch_curve(noise_rows, arch)
        if not xs:
            continue
        fig, ax = plt.subplots(figsize=(10.8, 7.2), dpi=220)
        ax.plot(xs, ys, marker="o", markersize=5.8, linewidth=2.8, color=LINE_COLOR, label="L2 (weight decay)")
        ax.fill_between(
            xs,
            [y - s for y, s in zip(ys, ysem)],
            [y + s for y, s in zip(ys, ysem)],
            color=LINE_COLOR,
            alpha=0.14,
            linewidth=0,
        )
        ax.set_xlabel("Gaussian noise sigma")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([i / 10 for i in range(11)])
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.24, linewidth=0.9)
        ax.legend(loc="lower left", frameon=False)
        fig.tight_layout()
        label = arch_label(arch)
        out_png = out_dir / f"fc3rev_{label}_wd_noise_sweep_step0p05_shared_ylim.png"
        fig.savefig(out_png)
        plt.close(fig)
        print(f"[SAVED] {out_png}")


def plot_drs_bar(drs_rows, with_sem: bool, out_dir: Path = OUT_DIR):
    setup_style(16)
    x = np.arange(len(drs_rows))
    means = [r["DRS_mean"] for r in drs_rows]
    sems = [r["DRS_sem"] for r in drs_rows]
    labels = [f"h{r['hidden_size']}" for r in drs_rows]

    fig, ax = plt.subplots(figsize=(10.8, 6.8), dpi=220)
    bars = ax.bar(
        x,
        means,
        width=0.55,
        yerr=sems if with_sem else None,
        capsize=3 if with_sem else 0,
        color=BAR_COLOR,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
    )
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2.0, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("FC3rev model scale (2h→h)")
    ax.set_ylabel("Derivative Robustness Score (DRS)")
    if means:
        ax.set_ylim(max(0.0, min(means) - 0.08), min(1.02, max(means) + 0.08))
    ax.grid(axis="y", alpha=0.25, linewidth=0.9)
    fig.tight_layout()

    suffix = "_with_sem" if with_sem else ""
    out_png = out_dir / f"fc3rev_wd_drs_bar{suffix}.png"
    out_pdf = out_dir / f"fc3rev_wd_drs_bar{suffix}.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def _shared_clean_ylim(clean_rows):
    all_lows, all_highs = [], []
    for row in clean_rows:
        for mean_key, std_key in (
            ("T0_L4_mean", "T0_L4_std"),
            ("T0_L16_mean", "T0_L16_std"),
            ("T16_L4_mean", "T16_L4_std"),
            ("T16_L16_mean", "T16_L16_std"),
        ):
            m, s = row[mean_key], row[std_key]
            all_lows.append(m - s)
            all_highs.append(m + s)
    ymin, ymax = min(all_lows), max(all_highs)
    pad = max(2.0, 0.06 * (ymax - ymin))
    return max(0.0, ymin - pad), min(100.5, ymax + pad + 2.0)


def plot_clean_overview(clean_rows, out_dir: Path, if_mode: str):
    setup_style(11)
    n = len(clean_rows)
    shared_ylim = _shared_clean_ylim(clean_rows)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 5.2), dpi=220, constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, row in zip(axes, clean_rows):
        x = np.arange(2)
        width = 0.36
        ann = np.array([row["T0_L4_mean"], row["T0_L16_mean"]])
        snn = np.array([row["T16_L4_mean"], row["T16_L16_mean"]])
        ann_err = np.array([row["T0_L4_std"], row["T0_L16_std"]])
        snn_err = np.array([row["T16_L4_std"], row["T16_L16_std"]])

        bars_ann = ax.bar(
            x - width / 2,
            ann,
            width,
            yerr=ann_err,
            capsize=3,
            label="ANN (T=0)",
            color=BAR_COLOR,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.92,
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
        )
        bars_snn = ax.bar(
            x + width / 2,
            snn,
            width,
            yerr=snn_err,
            capsize=3,
            label="SNN (T=16)",
            color=SNN_COLOR,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.92,
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
        )
        for bars in (bars_ann, bars_snn):
            for b in bars:
                h = b.get_height()
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    h + 0.4,
                    f"{h:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(["L=4", "L=16"])
        ax.set_title(f"h{row['hidden_size']}", fontsize=11, fontweight="bold")
        ax.set_ylim(*shared_ylim)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)

    axes[0].set_ylabel("Accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))

    stem = clean_overview_stem(if_mode)
    out_png = out_dir / f"{stem}.png"
    out_pdf = out_dir / f"{stem}.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def save_drs_csv(drs_rows, out_dir: Path = OUT_DIR):
    out = out_dir / "fc3rev_wd_drs.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["arch", "hidden_size", "DRS_mean", "DRS_std", "DRS_sem", "n_seeds"],
        )
        w.writeheader()
        for r in drs_rows:
            w.writerow(
                {
                    **{k: r[k] for k in ("arch", "hidden_size", "n_seeds")},
                    "DRS_mean": f"{r['DRS_mean']:.6f}",
                    "DRS_std": f"{r['DRS_std']:.6f}",
                    "DRS_sem": f"{r['DRS_sem']:.6f}",
                }
            )
    print(f"[SAVED] {out}")


def print_clean_table(clean_rows, if_mode: str):
    print(f"\n=== FC3rev Clean Accuracy (weight_decay, {if_mode}, mean ± std, 5 seeds) ===")
    header = f"{'Model':<12} {'T=0 L=4':<18} {'T=16 L=4':<18} {'T=0 L=16':<18} {'T=16 L=16':<18}"
    print(header)
    print("-" * len(header))
    for r in clean_rows:
        h = r["hidden_size"]
        print(
            f"{'h'+str(h):<12} "
            f"{r['T0_L4_mean']:.3f}±{r['T0_L4_std']:.3f}     "
            f"{r['T16_L4_mean']:.3f}±{r['T16_L4_std']:.3f}     "
            f"{r['T0_L16_mean']:.3f}±{r['T0_L16_std']:.3f}     "
            f"{r['T16_L16_mean']:.3f}±{r['T16_L16_std']:.3f}"
        )


def main():
    p = argparse.ArgumentParser(description="Plot FC3rev L2 wd noise sweep and clean acc overview")
    p.add_argument("--only-clean", action="store_true", help="Only redraw clean acc overview")
    p.add_argument(
        "--if-mode",
        type=str,
        default="normal",
        choices=["normal", "rate_uniform"],
        help="Which clean acc CSV to read/plot",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing fc3rev wd CSV outputs",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Plot output directory (default: <data-dir>/plots)",
    )
    args = p.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir or (data_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_rows = read_clean_merged(data_dir, args.if_mode)
    if args.only_clean:
        plot_clean_overview(clean_rows, out_dir, args.if_mode)
        print_clean_table(clean_rows, args.if_mode)
        return

    noise_rows = read_noise_raw(data_dir)
    drs_rows = compute_drs_rows(noise_rows)

    plot_noise_lines(noise_rows, out_dir)
    save_drs_csv(drs_rows, out_dir)
    plot_drs_bar(drs_rows, with_sem=False, out_dir=out_dir)
    plot_drs_bar(drs_rows, with_sem=True, out_dir=out_dir)
    plot_clean_overview(clean_rows, out_dir, args.if_mode)
    print_clean_table(clean_rows, args.if_mode)


if __name__ == "__main__":
    main()
