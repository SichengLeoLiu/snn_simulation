"""
MNIST fc3 strict-seed：mne_l2+wd 多 seed 训练 + 噪声扫描，并合并到已有三路 mean±std 折线图。

方法：mne_l2+wd (rc=5e-2, wd=5e-4)，与现有 fc3 strict-seed 中 mne_l2 / wd 单用系数一致。
设置：L=16, T=16, IF mode=normal, sigma=0~1.0 step=0.1, seeds=40..44。

用法：
  python noise3_exp/run_fc3_strict_seed_mne_l2_wd_noise_sweep_normal_L16_T16.py
  python noise3_exp/run_fc3_strict_seed_mne_l2_wd_noise_sweep_normal_L16_T16.py --h-list 8 16 32 128 --seed 42
  python noise3_exp/run_fc3_strict_seed_mne_l2_wd_noise_sweep_normal_L16_T16.py --plot-only
  python noise3_exp/run_fc3_strict_seed_mne_l2_wd_noise_sweep_normal_L16_T16.py --copy-important

合并绘图会读取已有 raw CSV（mne_l2 / weight_decay / no_regularization），
追加 mne_l2_wd 后重画 strict_seed_train_fc3_h*_noise_sweep_mean_std_lineplot*.png。
"""
from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = (
    ROOT
    / "noise3_exp"
    / "ablation_mne_l2_vs_weight_decay_l16_fc3_h4_h8_h16_h32_h64_h128"
    / "strict_seed_train_normal_L16_T16"
)
IMPORTANT_RESULTS = ROOT.parent / "important results"

DEFAULT_SEEDS = [40, 41, 42, 43, 44]
DEFAULT_H_LIST = [8, 16, 32, 64, 128]
REG_MNE_L2_WD = "mne_l2_wd"
LVAL = 16
TVAL = 16
IF_MODE = "normal"
MNE_RC = 5e-2
WD_COMBO = 5e-4

ALL_REGS = ["mne_l2", "mne_l2_wd", "weight_decay", "no_regularization"]

LINE_STYLES = {
    "mne_l2": {"color": "#1f77b4", "label": "mne_l2 (mean)"},
    "mne_l2_wd": {"color": "#9467bd", "label": "mne_l2+wd (mean)"},
    "weight_decay": {"color": "#ff7f0e", "label": "weight_decay (mean)"},
    "no_regularization": {"color": "#2ca02c", "label": "no regularization (mean)"},
}

RAW_CSV = OUT / "strict_seed_train_noise_sweep_fc3_h4_h8_h16_h32_h64_h128_raw.csv"
AGG_CSV = OUT / "strict_seed_train_noise_sweep_fc3_h4_h8_h16_h32_h64_h128_mean_std.csv"

RAW_FIELDS = [
    "arch",
    "hidden_size",
    "regularizer",
    "seed",
    "L",
    "T",
    "if_mode",
    "sigma",
    "acc",
    "checkpoint",
    "matrix_csv",
]

AGG_FIELDS = [
    "arch",
    "hidden_size",
    "regularizer",
    "sigma",
    "acc_mean",
    "acc_std",
    "n_seeds",
]


def coeff_tag(v: float) -> str:
    return f"{v:.0e}".replace("-", "m").replace("+", "p")


def arch_for(h: int) -> str:
    return f"fc3_h{h}"


def build_suffix(arch: str, seed: int) -> str:
    return (
        f"strict_seed{seed}_ablation_mne_l2_wd_l{LVAL}_{arch}"
        f"_rc{coeff_tag(MNE_RC)}_wd{coeff_tag(WD_COMBO)}"
    )


def ckpt_path(arch: str, seed: int) -> Path:
    return ROOT / "mnist-checkpoints" / f"{arch}_L[{LVAL}]_{build_suffix(arch, seed)}.pth"


def train_one(arch: str, seed: int) -> Path:
    ckpt = ckpt_path(arch, seed)
    if ckpt.exists():
        print(f"[SKIP TRAIN] {ckpt.name}", flush=True)
        return ckpt

    cmd = [
        sys.executable,
        str(ROOT / "main_train.py"),
        "-data",
        "mnist",
        "-arch",
        arch,
        "-L",
        str(LVAL),
        "--epochs",
        "100",
        "-j",
        "0",
        "-b",
        "128",
        "--seed",
        str(seed),
        "--device",
        "auto",
        "--time",
        "0",
        "--spike_schedule",
        "normal",
        "--regularizer",
        "mne_l2",
        "--weight_decay",
        str(WD_COMBO),
        "--reg_coeff",
        str(MNE_RC),
        "--suffix",
        build_suffix(arch, seed),
    ]
    print(f"[TRAIN] {arch} {REG_MNE_L2_WD} seed={seed}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
    print(f"[TRAIN DONE] {ckpt.name}", flush=True)
    return ckpt


def test_noise_sweep(arch: str, seed: int, ckpt: Path) -> Path:
    out_dir = OUT / arch / REG_MNE_L2_WD / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = (
        out_dir
        / f"noise_sweep_matrix_mnist_{arch}_T{TVAL}_mode_{IF_MODE}_schedule_normal_seed_{seed}.csv"
    )
    if matrix.exists():
        print(f"[SKIP TEST] {arch} seed={seed}", flush=True)
        return matrix

    cmd = [
        sys.executable,
        str(ROOT / "main_test.py"),
        "-data",
        "mnist",
        "-arch",
        arch,
        "-L",
        str(LVAL),
        "-T",
        str(TVAL),
        "-j",
        "0",
        "-b",
        "128",
        "--seed",
        str(seed),
        "--device",
        "auto",
        "--mode",
        IF_MODE,
        "--spike_schedule",
        "normal",
        "--weights",
        str(ckpt),
        "--noise_sweep",
        "--noise_sigma_start",
        "0.0",
        "--noise_sigma_end",
        "1.0",
        "--noise_sigma_step",
        "0.1",
        "--noise_output_dir",
        str(out_dir),
    ]
    print(f"[TEST] {arch} seed={seed} mode={IF_MODE}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    if not matrix.exists():
        cands = sorted(
            out_dir.glob(
                f"noise_sweep_matrix_*_T{TVAL}_mode_{IF_MODE}_schedule_normal_seed_{seed}.csv"
            )
        )
        if not cands:
            raise FileNotFoundError(f"matrix missing: {out_dir}")
        matrix = cands[0]
    print(f"[TEST DONE] {matrix.name}", flush=True)
    return matrix


def read_matrix(mat: Path) -> list[tuple[float, float]]:
    with mat.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)
    start = 2 if header[:2] == ["L", "T"] else 1
    sigmas = [float(x) for x in header[start:]]
    accs = [float(x) for x in row[start:]]
    return list(zip(sigmas, accs))


def load_raw_rows() -> list[dict]:
    if not RAW_CSV.exists():
        return []
    with RAW_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def upsert_mne_l2_wd_rows(arch: str, h: int, seed: int, rows: list[dict]) -> None:
    kept = [
        r
        for r in load_raw_rows()
        if not (r["arch"] == arch and r["regularizer"] == REG_MNE_L2_WD and int(r["seed"]) == seed)
    ]
    kept.extend(rows)
    kept.sort(
        key=lambda r: (
            int(r["hidden_size"]),
            ALL_REGS.index(r["regularizer"]) if r["regularizer"] in ALL_REGS else 99,
            int(r["seed"]),
            float(r["sigma"]),
        )
    )
    OUT.mkdir(parents=True, exist_ok=True)
    with RAW_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(kept)


def aggregate_rows(raw_rows: list[dict]) -> list[dict]:
    bucket: dict[tuple[str, int, str, float], list[float]] = defaultdict(list)
    for row in raw_rows:
        bucket[(row["arch"], int(row["hidden_size"]), row["regularizer"], float(row["sigma"]))].append(
            float(row["acc"])
        )

    agg_rows = []
    for (arch, h, reg, sigma), vals in sorted(
        bucket.items(), key=lambda x: (x[0][1], ALL_REGS.index(x[0][2]), x[0][3])
    ):
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        agg_rows.append(
            {
                "arch": arch,
                "hidden_size": h,
                "regularizer": reg,
                "sigma": f"{sigma:.1f}",
                "acc_mean": f"{mean:.6f}",
                "acc_std": f"{std:.6f}",
                "n_seeds": len(vals),
            }
        )
    return agg_rows


def plot_results(agg_rows: list[dict], h_list: list[int], copy_important: bool) -> None:
    plt.rcParams.update({"font.size": 11, "legend.fontsize": 9})
    for h in h_list:
        arch = arch_for(h)
        rows_arch = [r for r in agg_rows if r["arch"] == arch]
        if not rows_arch:
            print(f"[PLOT] skip {arch}: no data", flush=True)
            continue

        for no_caption in (False, True):
            fig, ax = plt.subplots(figsize=(9.0, 5.8), dpi=180)
            all_y = []
            for reg in ALL_REGS:
                rr = [r for r in rows_arch if r["regularizer"] == reg]
                if not rr:
                    continue
                rr.sort(key=lambda x: float(x["sigma"]))
                x = [float(r["sigma"]) for r in rr]
                y = [float(r["acc_mean"]) for r in rr]
                s = [float(r["acc_std"]) for r in rr]
                all_y.extend([yy - ss for yy, ss in zip(y, s)])
                all_y.extend([yy + ss for yy, ss in zip(y, s)])
                style = LINE_STYLES[reg]
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=2.2,
                    markersize=5,
                    color=style["color"],
                    label=style["label"],
                )
                if any(ss > 0 for ss in s):
                    ax.fill_between(
                        x,
                        [yy - ss for yy, ss in zip(y, s)],
                        [yy + ss for yy, ss in zip(y, s)],
                        color=style["color"],
                        alpha=0.18,
                        linewidth=0,
                    )
            ax.set_xlabel("Gaussian noise sigma")
            ax.set_ylabel("Accuracy (%)")
            ax.set_xlim(-0.02, 1.02)
            ax.set_xticks([round(i * 0.1, 1) for i in range(11)])
            if all_y:
                ax.set_ylim(min(all_y) - 0.8, max(all_y) + 0.8)
            ax.grid(alpha=0.3)
            ax.legend(loc="lower left", frameon=False)
            if not no_caption:
                ax.set_title(
                    f"{arch} strict-seed noise sweep "
                    f"(L=16,T=16,{IF_MODE}, incl. mne_l2+wd)"
                )
            fig.tight_layout()
            suffix = "_no_caption" if no_caption else ""
            out_png = OUT / f"strict_seed_train_{arch}_noise_sweep_mean_std_lineplot{suffix}.png"
            fig.savefig(out_png)
            plt.close(fig)
            print(f"[PLOT] saved {out_png}", flush=True)
            if copy_important:
                IMPORTANT_RESULTS.mkdir(parents=True, exist_ok=True)
                dest = IMPORTANT_RESULTS / out_png.name
                dest.write_bytes(out_png.read_bytes())
                print(f"[PLOT] copied {dest}", flush=True)


def finalize(h_list: list[int], copy_important: bool) -> None:
    raw_rows = load_raw_rows()
    agg_rows = aggregate_rows(raw_rows)
    with AGG_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AGG_FIELDS)
        writer.writeheader()
        writer.writerows(agg_rows)
    plot_results(agg_rows, h_list, copy_important)
    print(f"[TABLE] raw: {RAW_CSV}", flush=True)
    print(f"[TABLE] agg: {AGG_CSV}", flush=True)


def run_one(arch: str, h: int, seed: int) -> list[dict]:
    ckpt = train_one(arch, seed)
    matrix = test_noise_sweep(arch, seed, ckpt)
    rows = []
    for sigma, acc in read_matrix(matrix):
        rows.append(
            {
                "arch": arch,
                "hidden_size": h,
                "regularizer": REG_MNE_L2_WD,
                "seed": seed,
                "L": LVAL,
                "T": TVAL,
                "if_mode": IF_MODE,
                "sigma": sigma,
                "acc": acc,
                "checkpoint": str(ckpt.relative_to(ROOT)),
                "matrix_csv": str(matrix.relative_to(ROOT)),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="fc3 strict-seed mne_l2+wd + 合并 mean±std 图")
    p.add_argument(
        "--h-list",
        type=int,
        nargs="+",
        default=DEFAULT_H_LIST,
        help=f"隐藏层规模（默认 {' '.join(map(str, DEFAULT_H_LIST))}）",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=f"随机种子（默认 {' '.join(map(str, DEFAULT_SEEDS))}）",
    )
    p.add_argument("--seed", type=int, default=None, help="只跑单个 seed")
    p.add_argument("--plot-only", action="store_true", help="仅合并 raw 并重绘图")
    p.add_argument(
        "--copy-important",
        action="store_true",
        help="同时将 PNG 复制到仓库根目录 important results/",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [args.seed] if args.seed is not None else args.seeds
    h_list = args.h_list

    if args.plot_only:
        finalize(h_list, args.copy_important)
        return

    print(f"\n=== fc3 strict-seed mne_l2+wd (rc={MNE_RC}, wd={WD_COMBO}) ===", flush=True)
    print(f"h_list={h_list} seeds={seeds}", flush=True)

    for h in h_list:
        arch = arch_for(h)
        for seed in seeds:
            rows = run_one(arch, h, seed)
            upsert_mne_l2_wd_rows(arch, h, seed, rows)

    finalize(h_list, args.copy_important)


if __name__ == "__main__":
    main()
