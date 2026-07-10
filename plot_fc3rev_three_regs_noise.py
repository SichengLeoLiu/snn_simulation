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


def read_noise_raw(data_dir: Path):
    noise_raw = data_dir / "fc3rev_h8_h256_three_regs_noise_sweep_raw.csv"
    if not noise_raw.exists():
        legacy = data_dir / "fc3rev_h8_h256_wd_noise_sweep_raw.csv"
        raise FileNotFoundError(
            f"Missing {noise_raw}. Run three-regs experiment first. (legacy wd-only: {legacy})"
        )
    with noise_raw.open(newline="") as f:
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


def compute_drs_rows(noise_rows):
    by_key = defaultdict(lambda: defaultdict(list))
    for r in noise_rows:
        key = (r["arch"], r["regularizer"])
        by_key[key][int(r["seed"])].append((float(r["sigma"]), float(r["acc"])))

    rows = []
    for (arch, reg) in sorted(by_key.keys(), key=lambda k: (int(k[0].split("_h")[1]), METHODS.index(k[1]))):
        vals = []
        for seed in sorted(by_key[(arch, reg)].keys()):
            pairs = sorted(by_key[(arch, reg)][seed], key=lambda x: x[0])
            vals.append(derivative_robustness_score([p[0] for p in pairs], [p[1] for p in pairs]))
        n = len(vals)
        drs_std = statistics.stdev(vals) if n > 1 else 0.0
        rows.append(
            {
                "arch": arch,
                "hidden_size": int(arch.split("_h")[1]),
                "regularizer": reg,
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
        out_png = out_dir / f"fc3rev_{label}_three_regs_noise_sweep_step0p05.png"
        out_pdf = out_dir / f"fc3rev_{label}_three_regs_noise_sweep_step0p05.pdf"
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
            out_png = out_dir / f"fc3rev_{label}_three_regs_noise_sweep_step0p05_shared_ylim.png"
            fig.savefig(out_png)
            plt.close(fig)
            print(f"[SAVED] {out_png}")


def plot_drs_bar(
    drs_rows,
    with_sem: bool,
    h_list: list[int] | None = None,
    xlabel: str = "FC3rev model scale (2h→h)",
    out_stem: str = "fc3rev_three_regs_drs_bar",
    out_dir: Path = OUT_DIR,
):
    setup_style(16)
    archs = sorted({r["arch"] for r in drs_rows}, key=lambda a: int(a.split("_h")[1]))
    if h_list is not None:
        h_set = set(h_list)
        archs = [a for a in archs if int(a.split("_h")[1]) in h_set]
    x = np.arange(len(archs))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11.2, 6.8), dpi=220)
    all_means = []
    for idx, reg in enumerate(METHODS):
        means, errs = [], []
        for arch in archs:
            row = next((v for v in drs_rows if v["arch"] == arch and v["regularizer"] == reg), None)
            means.append(row["DRS_mean"] if row else np.nan)
            errs.append(row["DRS_sem"] if row and with_sem else 0.0)
        all_means.extend([m for m in means if not np.isnan(m)])
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
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Derivative Robustness Score (DRS)")
    if all_means:
        ymin = max(0.0, min(all_means) - 0.08)
        ymax = min(1.02, max(all_means) + 0.08)
        ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", alpha=0.25, linewidth=0.9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
    fig.tight_layout()

    suffix = "_with_sem" if with_sem else ""
    out_png = out_dir / f"{out_stem}{suffix}.png"
    out_pdf = out_dir / f"{out_stem}{suffix}.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def save_drs_csv(drs_rows, out_dir: Path = OUT_DIR):
    out = out_dir / "fc3rev_three_regs_drs.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["arch", "hidden_size", "regularizer", "DRS_mean", "DRS_std", "DRS_sem", "n_seeds"],
        )
        w.writeheader()
        for r in drs_rows:
            w.writerow(
                {
                    **{k: r[k] for k in ("arch", "hidden_size", "regularizer", "n_seeds")},
                    "DRS_mean": f"{r['DRS_mean']:.6f}",
                    "DRS_std": f"{r['DRS_std']:.6f}",
                    "DRS_sem": f"{r['DRS_sem']:.6f}",
                }
            )
    print(f"[SAVED] {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot FC3rev three-regs noise sweep and DRS bars")
    p.add_argument("--drs-bar-h-list", type=int, nargs="+", default=None, help="DRS bar: hidden sizes to include")
    p.add_argument("--rs-bar-h-list", type=int, nargs="+", default=None, help=argparse.SUPPRESS)
    p.add_argument("--drs-bar-xlabel", type=str, default="Hidden Size")
    p.add_argument("--rs-bar-xlabel", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--drs-bar-stem",
        type=str,
        default=None,
        help="DRS bar output filename stem (default auto from h-list)",
    )
    p.add_argument("--rs-bar-stem", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--only-drs-bar", action="store_true", help="Only redraw DRS bar chart(s)")
    p.add_argument("--only-rs-bar", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no-sem-bar", action="store_true", help="Skip DRS bar without SEM")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory with fc3rev three-regs CSV outputs",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Plot output directory (default: <data-dir>/plots)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    out_dir = args.out_dir or (data_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    noise_rows = read_noise_raw(data_dir)
    drs_rows = compute_drs_rows(noise_rows)

    h_list = args.drs_bar_h_list or args.rs_bar_h_list
    only_bar = args.only_drs_bar or args.only_rs_bar
    if h_list is None and only_bar:
        h_list = [8, 16, 32]
    stem = args.drs_bar_stem or args.rs_bar_stem
    if stem is None and h_list is not None:
        stem = "fc3rev_three_regs_drs_bar_" + "_".join(f"h{h}" for h in h_list)
    if stem is None:
        stem = "fc3rev_three_regs_drs_bar"
    xlabel = (args.drs_bar_xlabel if args.rs_bar_xlabel is None else args.rs_bar_xlabel)
    if h_list is None:
        xlabel = "FC3rev model scale (2h→h)"

    if only_bar:
        if not args.no_sem_bar:
            plot_drs_bar(drs_rows, with_sem=False, h_list=h_list, xlabel=xlabel, out_stem=stem, out_dir=out_dir)
        plot_drs_bar(drs_rows, with_sem=True, h_list=h_list, xlabel=xlabel, out_stem=stem, out_dir=out_dir)
        return

    plot_noise_lines(noise_rows, out_dir)
    save_drs_csv(drs_rows, out_dir)
    plot_drs_bar(drs_rows, with_sem=False, out_dir=out_dir)
    plot_drs_bar(drs_rows, with_sem=True, out_dir=out_dir)
    if h_list is not None:
        plot_drs_bar(
            drs_rows,
            with_sem=True,
            h_list=h_list,
            xlabel=args.drs_bar_xlabel,
            out_stem=stem,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
