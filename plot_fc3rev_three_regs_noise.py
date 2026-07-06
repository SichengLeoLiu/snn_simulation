import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "important results" / "new_fc3"
NOISE_RAW = DATA_DIR / "fc3rev_h8_h256_three_regs_noise_sweep_raw.csv"
OUT_DIR = DATA_DIR / "plots"

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


def read_noise_raw():
    if not NOISE_RAW.exists():
        legacy = DATA_DIR / "fc3rev_h8_h256_wd_noise_sweep_raw.csv"
        raise FileNotFoundError(
            f"Missing {NOISE_RAW}. Run three-regs experiment first. (legacy wd-only: {legacy})"
        )
    with NOISE_RAW.open(newline="") as f:
        return list(csv.DictReader(f))


def arch_label(arch: str) -> str:
    return f"h{arch.split('_h')[1]}"


def collect_curve(noise_rows, arch: str, reg: str):
    bucket = defaultdict(list)
    for r in noise_rows:
        if r["arch"] != arch or r["regularizer"] != reg:
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
    by_key = defaultdict(lambda: defaultdict(list))
    for r in noise_rows:
        key = (r["arch"], r["regularizer"])
        by_key[key][int(r["seed"])].append((float(r["sigma"]), float(r["acc"])))

    rows = []
    for (arch, reg) in sorted(by_key.keys(), key=lambda k: (int(k[0].split("_h")[1]), METHODS.index(k[1]))):
        vals = []
        for seed in sorted(by_key[(arch, reg)].keys()):
            pairs = sorted(by_key[(arch, reg)][seed], key=lambda x: x[0])
            vals.append(robust_score([p[0] for p in pairs], [p[1] for p in pairs]))
        n = len(vals)
        rs_std = statistics.stdev(vals) if n > 1 else 0.0
        rows.append(
            {
                "arch": arch,
                "hidden_size": int(arch.split("_h")[1]),
                "regularizer": reg,
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

    shared_ylim = None
    all_y = []
    for arch in archs:
        for reg in METHODS:
            xs, ys, ysem = collect_curve(noise_rows, arch, reg)
            if xs:
                all_y.extend([y - s for y, s in zip(ys, ysem)])
                all_y.extend([y + s for y, s in zip(ys, ysem)])
    if all_y:
        ymin, ymax = min(all_y), max(all_y)
        pad = max(0.8, 0.08 * (ymax - ymin))
        shared_ylim = (ymin - pad, ymax + pad)

    for arch in archs:
        fig, ax = plt.subplots(figsize=(10.8, 7.2), dpi=220)
        for reg in METHODS:
            xs, ys, ysem = collect_curve(noise_rows, arch, reg)
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=5.8,
                linewidth=2.8,
                color=METHOD_COLORS[reg],
                label=METHOD_LABELS[reg],
            )
            ax.fill_between(
                xs,
                [y - s for y, s in zip(ys, ysem)],
                [y + s for y, s in zip(ys, ysem)],
                color=METHOD_COLORS[reg],
                alpha=0.14,
                linewidth=0,
            )
        ax.set_xlabel("Gaussian noise sigma")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([i / 10 for i in range(11)])
        ax.grid(alpha=0.24, linewidth=0.9)
        ax.legend(loc="lower left", frameon=False)
        fig.tight_layout()
        label = arch_label(arch)
        out_png = OUT_DIR / f"fc3rev_{label}_three_regs_noise_sweep_step0p05.png"
        out_pdf = OUT_DIR / f"fc3rev_{label}_three_regs_noise_sweep_step0p05.pdf"
        fig.savefig(out_png)
        fig.savefig(out_pdf)
        plt.close(fig)
        print(f"[SAVED] {out_png}")

        if shared_ylim:
            fig, ax = plt.subplots(figsize=(10.8, 7.2), dpi=220)
            for reg in METHODS:
                xs, ys, ysem = collect_curve(noise_rows, arch, reg)
                if not xs:
                    continue
                ax.plot(xs, ys, marker="o", markersize=5.8, linewidth=2.8, color=METHOD_COLORS[reg], label=METHOD_LABELS[reg])
                ax.fill_between(
                    xs,
                    [y - s for y, s in zip(ys, ysem)],
                    [y + s for y, s in zip(ys, ysem)],
                    color=METHOD_COLORS[reg],
                    alpha=0.14,
                    linewidth=0,
                )
            ax.set_xlabel("Gaussian noise sigma")
            ax.set_ylabel("Accuracy (%)")
            ax.set_xlim(0.0, 1.0)
            ax.set_xticks([i / 10 for i in range(11)])
            ax.set_ylim(*shared_ylim)
            ax.grid(alpha=0.24, linewidth=0.9)
            ax.legend(loc="lower left", frameon=False)
            fig.tight_layout()
            out_png = OUT_DIR / f"fc3rev_{label}_three_regs_noise_sweep_step0p05_shared_ylim.png"
            fig.savefig(out_png)
            plt.close(fig)
            print(f"[SAVED] {out_png}")


def plot_rs_bar(rs_rows, with_sem: bool):
    setup_style(16)
    archs = sorted({r["arch"] for r in rs_rows}, key=lambda a: int(a.split("_h")[1]))
    x = np.arange(len(archs))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11.2, 6.8), dpi=220)
    for idx, reg in enumerate(METHODS):
        means, errs = [], []
        for arch in archs:
            row = next((v for v in rs_rows if v["arch"] == arch and v["regularizer"] == reg), None)
            means.append(row["RS_mean"] if row else np.nan)
            errs.append(row["RS_sem"] if row and with_sem else 0.0)
        xpos = x + (idx - 1) * width
        bars = ax.bar(
            xpos,
            means,
            width,
            yerr=errs if with_sem else None,
            capsize=3 if with_sem else 0,
            color=METHOD_COLORS[reg],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
            label=METHOD_LABELS[reg],
        )
        for b, v in zip(bars, means):
            if np.isnan(v):
                continue
            ax.text(b.get_x() + b.get_width() / 2.0, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([arch_label(a) for a in archs])
    ax.set_xlabel("FC3rev model scale (2h→h)")
    ax.set_ylabel("Robustness Score")
    ax.set_ylim(0.55, 1.02)
    ax.grid(axis="y", alpha=0.25, linewidth=0.9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
    fig.tight_layout()

    suffix = "_with_sem" if with_sem else ""
    out_png = OUT_DIR / f"fc3rev_three_regs_robustness_score_bar{suffix}.png"
    out_pdf = OUT_DIR / f"fc3rev_three_regs_robustness_score_bar{suffix}.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def save_rs_csv(rs_rows):
    out = OUT_DIR / "fc3rev_three_regs_robustness_score.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["arch", "hidden_size", "regularizer", "RS_mean", "RS_std", "RS_sem", "n_seeds"],
        )
        w.writeheader()
        for r in rs_rows:
            w.writerow(
                {
                    **{k: r[k] for k in ("arch", "hidden_size", "regularizer", "n_seeds")},
                    "RS_mean": f"{r['RS_mean']:.6f}",
                    "RS_std": f"{r['RS_std']:.6f}",
                    "RS_sem": f"{r['RS_sem']:.6f}",
                }
            )
    print(f"[SAVED] {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    noise_rows = read_noise_raw()
    rs_rows = compute_rs_rows(noise_rows)
    plot_noise_lines(noise_rows)
    save_rs_csv(rs_rows)
    plot_rs_bar(rs_rows, with_sem=False)
    plot_rs_bar(rs_rows, with_sem=True)


if __name__ == "__main__":
    main()
