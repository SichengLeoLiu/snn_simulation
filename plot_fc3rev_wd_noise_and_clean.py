import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "important results" / "new_fc3"
NOISE_RAW = DATA_DIR / "fc3rev_h8_h256_wd_noise_sweep_raw.csv"
CLEAN_MERGED = DATA_DIR / "fc3rev_h8_h256_wd_t0_t16_l4_l16_merged.csv"
OUT_DIR = DATA_DIR / "plots"

LINE_COLOR = "#1f77b4"
BAR_COLOR = "#4C78A8"
SNN_COLOR = "#F58518"


def read_noise_raw():
    rows = []
    with NOISE_RAW.open(newline="") as f:
        rows.extend(csv.DictReader(f))
    return rows


def read_clean_merged():
    rows = []
    with CLEAN_MERGED.open(newline="") as f:
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


def compute_rs_rows(noise_rows):
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
            vals.append(robust_score(sigmas, accs))
        n = len(vals)
        rs_std = statistics.stdev(vals) if n > 1 else 0.0
        rows.append(
            {
                "arch": arch,
                "hidden_size": int(arch.split("_h")[1]),
                "RS_mean": statistics.mean(vals),
                "RS_std": rs_std,
                "RS_sem": rs_std / (n ** 0.5) if n > 0 else 0.0,
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


def plot_noise_lines(noise_rows):
    setup_style(18)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
        out_png = OUT_DIR / f"fc3rev_{label}_wd_noise_sweep_step0p05.png"
        out_pdf = OUT_DIR / f"fc3rev_{label}_wd_noise_sweep_step0p05.pdf"
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
        out_png = OUT_DIR / f"fc3rev_{label}_wd_noise_sweep_step0p05_shared_ylim.png"
        fig.savefig(out_png)
        plt.close(fig)
        print(f"[SAVED] {out_png}")


def plot_rs_bar(rs_rows, with_sem: bool):
    setup_style(16)
    x = np.arange(len(rs_rows))
    means = [r["RS_mean"] for r in rs_rows]
    sems = [r["RS_sem"] for r in rs_rows]
    labels = [f"h{r['hidden_size']}" for r in rs_rows]

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
    ax.set_ylabel("Robustness Score")
    ax.set_ylim(0.55, 1.02)
    ax.grid(axis="y", alpha=0.25, linewidth=0.9)
    fig.tight_layout()

    suffix = "_with_sem" if with_sem else ""
    out_png = OUT_DIR / f"fc3rev_wd_robustness_score_bar{suffix}.png"
    out_pdf = OUT_DIR / f"fc3rev_wd_robustness_score_bar{suffix}.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def plot_clean_overview(clean_rows):
    setup_style(11)
    n = len(clean_rows)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 4.8), dpi=220, constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, row in zip(axes, clean_rows):
        x = np.arange(2)
        width = 0.36
        ann = np.array([row["T0_L4_mean"], row["T0_L16_mean"]])
        snn = np.array([row["T16_L4_mean"], row["T16_L16_mean"]])
        bars_ann = ax.bar(
            x - width / 2,
            ann,
            width,
            label="ANN (T=0)",
            color=BAR_COLOR,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.92,
        )
        bars_snn = ax.bar(
            x + width / 2,
            snn,
            width,
            label="SNN (T=16)",
            color=SNN_COLOR,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.92,
        )
        for bars in (bars_ann, bars_snn):
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2.0, h + 0.08, f"{h:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(["L=4", "L=16"])
        ax.set_title(f"fc3rev_h{row['hidden_size']}", fontsize=11, fontweight="bold")
        ax.set_ylim(85, 101)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)

    axes[0].set_ylabel("Accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))

    out_png = OUT_DIR / "fc3rev_h8_h256_wd_clean_acc_overview.png"
    out_pdf = OUT_DIR / "fc3rev_h8_h256_wd_clean_acc_overview.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def save_rs_csv(rs_rows):
    out = OUT_DIR / "fc3rev_wd_robustness_score.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["arch", "hidden_size", "RS_mean", "RS_std", "RS_sem", "n_seeds"],
        )
        w.writeheader()
        for r in rs_rows:
            w.writerow(
                {
                    **{k: r[k] for k in ("arch", "hidden_size", "n_seeds")},
                    "RS_mean": f"{r['RS_mean']:.6f}",
                    "RS_std": f"{r['RS_std']:.6f}",
                    "RS_sem": f"{r['RS_sem']:.6f}",
                }
            )
    print(f"[SAVED] {out}")


def print_clean_table(clean_rows):
    print("\n=== FC3rev Clean Accuracy (weight_decay, mean ± std, 5 seeds) ===")
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    noise_rows = read_noise_raw()
    clean_rows = read_clean_merged()
    rs_rows = compute_rs_rows(noise_rows)

    plot_noise_lines(noise_rows)
    save_rs_csv(rs_rows)
    plot_rs_bar(rs_rows, with_sem=False)
    plot_rs_bar(rs_rows, with_sem=True)
    plot_clean_overview(clean_rows)
    print_clean_table(clean_rows)


if __name__ == "__main__":
    main()
