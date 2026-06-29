"""
MNIST CNN2：mne_l2 reg_coeff 扫描 + weight_decay 基线 + rate_uniform 噪声注入。

模型：cnn2_c2_c4 / cnn2_c4_c8 / cnn2_c8_c16 / cnn2_c16_c32（可 --arch-list 指定）
默认 rc ∈ {1e-4, 5e-4, 1e-3, 1e-2, 5e-2}；训练 L=16 T=0，测试 L=16 T=16 rate_uniform。
默认 seeds=40..44（5 seed mean±std），与 strict-seed 实验一致。

checkpoint 后缀含 rcscan，与 strict-seed 实验区分，不会覆盖已有权重。

用法：
  python noise3_exp/run_cnn_mne_reg_coeff_scan_noise_sweep_rate_uniform_L16_T16.py
  python noise3_exp/run_cnn_mne_reg_coeff_scan_noise_sweep_rate_uniform_L16_T16.py \\
      --arch-list c2c4 c4c8 --seed 42
  python noise3_exp/run_cnn_mne_reg_coeff_scan_noise_sweep_rate_uniform_L16_T16.py \\
      --rc-list 1e-4 1e-3 5e-2 --plot-only
  bash noise3_exp/RUN_cnn_mne_reg_coeff_scan.sh
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "noise3_exp" / "cnn_mne_reg_coeff_scan_noise_sweep_rate_uniform_L16_T16"
REPO_ROOT = ROOT.parent

DEFAULT_SEEDS = [40, 41, 42, 43, 44]
DEFAULT_RC_LIST = [1e-4, 5e-4, 1e-3, 1e-2, 5e-2]
CNN_VARIANTS = [(2, 4), (4, 8), (8, 16), (16, 32)]
LVAL = 16
TVAL = 16
IF_MODE = "rate_uniform"
WD = 5e-4
EPOCHS = int(os.environ.get("CNN_EPOCHS", "100"))
BATCH = int(os.environ.get("CNN_BATCH", "128"))
NUM_WORKERS = int(os.environ.get("CNN_NUM_WORKERS", "8"))
SCAN_TAG = "rcscan"

RAW_CSV = OUT / "cnn_mne_reg_coeff_scan_noise_sweep_raw.csv"
AGG_CSV = OUT / "cnn_mne_reg_coeff_scan_noise_sweep_mean_std.csv"
SUMMARY_CSV = OUT / "cnn_mne_reg_coeff_scan_best_test_summary.csv"
BEST_RC_CSV = OUT / "cnn_mne_reg_coeff_scan_best_rc_per_arch.csv"

RAW_FIELDS = [
    "arch", "method", "regularizer", "reg_coeff", "seed", "L", "T",
    "if_mode", "sigma", "acc", "checkpoint", "matrix_csv",
]
AGG_FIELDS = [
    "arch", "method", "regularizer", "reg_coeff", "sigma",
    "acc_mean", "acc_std", "n_seeds",
]
SUMMARY_FIELDS = [
    "arch", "method", "regularizer", "reg_coeff",
    "acc_sigma0_mean", "acc_sigma0_std",
    "acc_sigma1_mean", "acc_sigma1_std",
    "acc_drop_mean", "acc_drop_std", "n_seeds",
]

ARCH_ALIASES = {
    "c2c4": (2, 4),
    "c4c8": (4, 8),
    "c8c16": (8, 16),
    "c16c32": (16, 32),
}

RC_COLORS = {
    "1em04": "#1f77b4",
    "5em04": "#17becf",
    "1em03": "#2ca02c",
    "1em02": "#9467bd",
    "5em02": "#d62728",
}


def coeff_tag(v: float) -> str:
    return f"{v:.0e}".replace("-", "m").replace("+", "p")


def arch_name(c1: int, c2: int) -> str:
    return f"cnn2_c{c1}_c{c2}"


def resolve_arch_list(names: list[str] | None) -> list[tuple[int, int]]:
    if not names:
        return list(CNN_VARIANTS)
    out: list[tuple[int, int]] = []
    for name in names:
        key = name.strip().lower()
        if key in ARCH_ALIASES:
            pair = ARCH_ALIASES[key]
        elif key.startswith("cnn2_c") and "_c" in key:
            rest = key.replace("cnn2_c", "")
            c1_s, c2_s = rest.split("_c", 1)
            pair = (int(c1_s), int(c2_s))
        else:
            raise ValueError(f"未知 arch: {name!r}，可用 c2c4/c4c8/c8c16/c16c32")
        if pair not in out:
            out.append(pair)
    return out


def method_key(reg: str, reg_coeff: Optional[float] = None) -> str:
    if reg == "weight_decay":
        return "weight_decay"
    return f"mne_l2:{coeff_tag(reg_coeff)}"


def methods_for_run(rc_list: list[float], include_wd: bool) -> list[str]:
    keys = []
    if include_wd:
        keys.append("weight_decay")
    keys.extend(method_key("mne_l2", c) for c in rc_list)
    return keys


def build_suffix(arch: str, reg: str, seed: int, reg_coeff: Optional[float] = None) -> str:
    if reg == "weight_decay":
        return f"seed{seed}_{SCAN_TAG}_wd_l{LVAL}_{arch}"
    return f"seed{seed}_{SCAN_TAG}_mne_l2_l{LVAL}_{arch}_rc{coeff_tag(reg_coeff)}"


def ckpt_path(arch: str, reg: str, seed: int, reg_coeff: Optional[float] = None) -> Path:
    suffix = build_suffix(arch, reg, seed, reg_coeff)
    return ROOT / "mnist-checkpoints" / f"{arch}_L[{LVAL}]_{suffix}.pth"


def test_out_dir(arch: str, tag: str, seed: int) -> Path:
    safe = tag.replace(":", "_")
    return OUT / arch / safe / f"seed_{seed}"


def train_one(
    arch: str, reg: str, seed: int, reg_coeff: Optional[float] = None, retrain: bool = False
) -> Path:
    ckpt = ckpt_path(arch, reg, seed, reg_coeff)
    if retrain and ckpt.exists():
        ckpt.unlink()
        print(f"[RETRAIN] removed {ckpt.name}", flush=True)
    if ckpt.exists():
        print(f"[SKIP TRAIN] {ckpt.name}", flush=True)
        return ckpt

    if reg == "mne_l2":
        regularizer, wd, rc = "mne_l2", 0.0, reg_coeff
    else:
        regularizer, wd, rc = "weight_decay", WD, 1.0

    cmd = [
        sys.executable,
        str(ROOT / "main_train.py"),
        "-data", "mnist",
        "-arch", arch,
        "-L", str(LVAL),
        "--epochs", str(EPOCHS),
        "-j", str(NUM_WORKERS),
        "-b", str(BATCH),
        "--seed", str(seed),
        "--device", "auto",
        "--time", "0",
        "--spike_schedule", "normal",
        "--regularizer", regularizer,
        "--weight_decay", str(wd),
        "--reg_coeff", str(rc),
        "--suffix", build_suffix(arch, reg, seed, reg_coeff),
    ]
    label = method_key(reg, reg_coeff)
    print(f"[TRAIN] {arch} {label} seed={seed}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
    print(f"[TRAIN DONE] {ckpt.name}", flush=True)
    return ckpt


def test_noise_sweep(
    arch: str,
    tag: str,
    seed: int,
    ckpt: Path,
    force_test: bool = False,
) -> Path:
    out_dir = test_out_dir(arch, tag, seed)
    matrix = (
        out_dir
        / f"noise_sweep_matrix_mnist_{arch}_T{TVAL}_mode_{IF_MODE}_schedule_normal_seed_{seed}.csv"
    )
    if force_test and matrix.exists():
        matrix.unlink()
        combined = out_dir / "noise_sweep_combined_L_T.csv"
        if combined.exists():
            combined.unlink()
    if matrix.exists():
        print(f"[SKIP TEST] {arch} {tag} seed={seed}", flush=True)
        return matrix

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "main_test.py"),
        "-data", "mnist",
        "-arch", arch,
        "-L", str(LVAL),
        "-T", str(TVAL),
        "-j", str(NUM_WORKERS),
        "-b", str(BATCH),
        "--seed", str(seed),
        "--device", "auto",
        "--mode", IF_MODE,
        "--spike_schedule", "normal",
        "--weights", str(ckpt),
        "--noise_sweep",
        "--noise_sigma_start", "0.0",
        "--noise_sigma_end", "1.0",
        "--noise_sigma_step", "0.1",
        "--noise_output_dir", str(out_dir),
    ]
    print(f"[TEST] {arch} {tag} seed={seed} mode={IF_MODE}", flush=True)
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


def curve_acc(curve: list[tuple[float, float]], sigma: float) -> float:
    d = dict(curve)
    return float(d.get(sigma, d.get(int(sigma) if sigma == int(sigma) else sigma)))


def line_style_for(key: str, rc_list: list[float]) -> dict:
    if key == "weight_decay":
        return {"color": "#ff7f0e", "label": "weight_decay"}
    rc_tag = key.split(":", 1)[1]
    for coeff in rc_list:
        if coeff_tag(coeff) == rc_tag:
            return {
                "color": RC_COLORS.get(rc_tag, "#333333"),
                "label": f"mne_l2 rc={coeff:.0e}",
            }
    return {"color": "#333333", "label": key}


def load_raw_rows() -> list[dict]:
    if not RAW_CSV.exists():
        return []
    with RAW_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def upsert_run_rows(
    new_rows: list[dict],
    archs: set[str],
    methods: list[str],
    seeds: list[int],
) -> None:
    kept = [
        r
        for r in load_raw_rows()
        if not (
            r["arch"] in archs
            and r["method"] in methods
            and int(r["seed"]) in seeds
        )
    ]
    kept.extend(new_rows)
    kept.sort(
        key=lambda r: (
            r["arch"],
            r["method"],
            int(r["seed"]),
            float(r["sigma"]),
        )
    )
    OUT.mkdir(parents=True, exist_ok=True)
    with RAW_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writeheader()
        w.writerows(kept)


def aggregate_curve_rows(raw_rows: list[dict]) -> list[dict]:
    bucket: dict[tuple[str, str, str, str, float], list[float]] = defaultdict(list)
    for row in raw_rows:
        bucket[
            (
                row["arch"],
                row["method"],
                row["regularizer"],
                row["reg_coeff"],
                float(row["sigma"]),
            )
        ].append(float(row["acc"]))

    agg_rows = []
    for (arch, method, reg, rc_str, sigma), vals in sorted(
        bucket.items(),
        key=lambda x: (x[0][0], x[0][1], x[0][4]),
    ):
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        agg_rows.append(
            {
                "arch": arch,
                "method": method,
                "regularizer": reg,
                "reg_coeff": rc_str,
                "sigma": f"{sigma:.1f}",
                "acc_mean": f"{mean:.6f}",
                "acc_std": f"{std:.6f}",
                "n_seeds": len(vals),
            }
        )
    return agg_rows


def aggregate_summary_rows(raw_rows: list[dict]) -> list[dict]:
    bucket: dict[tuple[str, str, str, str], list[tuple[float, float, float]]] = defaultdict(list)
    by_seed: dict[tuple[str, str, str, str, int], list[tuple[float, float]]] = defaultdict(list)
    for row in raw_rows:
        key = (row["arch"], row["method"], row["regularizer"], row["reg_coeff"], int(row["seed"]))
        by_seed[key].append((float(row["sigma"]), float(row["acc"])))

    for key, pts in by_seed.items():
        arch, method, reg, rc_str, _seed = key
        curve = sorted(pts, key=lambda x: x[0])
        a0 = curve_acc(curve, 0.0)
        a1 = curve_acc(curve, 1.0)
        bucket[(arch, method, reg, rc_str)].append((a0, a1, a0 - a1))

    summary_rows = []
    for (arch, method, reg, rc_str), vals in sorted(bucket.items(), key=lambda x: (x[0][0], x[0][1])):
        a0s = [v[0] for v in vals]
        a1s = [v[1] for v in vals]
        drops = [v[2] for v in vals]
        n = len(vals)
        summary_rows.append(
            {
                "arch": arch,
                "method": method,
                "regularizer": reg,
                "reg_coeff": rc_str,
                "acc_sigma0_mean": f"{statistics.mean(a0s):.6f}",
                "acc_sigma0_std": f"{statistics.stdev(a0s) if n > 1 else 0.0:.6f}",
                "acc_sigma1_mean": f"{statistics.mean(a1s):.6f}",
                "acc_sigma1_std": f"{statistics.stdev(a1s) if n > 1 else 0.0:.6f}",
                "acc_drop_mean": f"{statistics.mean(drops):.6f}",
                "acc_drop_std": f"{statistics.stdev(drops) if n > 1 else 0.0:.6f}",
                "n_seeds": n,
            }
        )
    return summary_rows


def pick_best_rc_per_arch(summary_rows: list[dict]) -> list[dict]:
    by_arch: dict[str, list[dict]] = defaultdict(list)
    for row in summary_rows:
        if row["regularizer"] != "mne_l2":
            continue
        by_arch[row["arch"]].append(row)

    best_rows = []
    for arch in sorted(by_arch):
        cands = by_arch[arch]
        best = min(
            cands,
            key=lambda r: (
                float(r["acc_drop_mean"]),
                -float(r["acc_sigma0_mean"]),
            ),
        )
        best_rows.append(
            {
                "arch": arch,
                "best_reg_coeff": best["reg_coeff"],
                "acc_sigma0_mean": best["acc_sigma0_mean"],
                "acc_sigma0_std": best["acc_sigma0_std"],
                "acc_sigma1_mean": best["acc_sigma1_mean"],
                "acc_sigma1_std": best["acc_sigma1_std"],
                "acc_drop_mean": best["acc_drop_mean"],
                "acc_drop_std": best["acc_drop_std"],
                "n_seeds": best["n_seeds"],
            }
        )
    return best_rows


def plot_arch(
    arch: str,
    agg_rows: list[dict],
    rc_list: list[float],
    seeds: list[int],
    font_size: float,
    legend_font_size: float,
) -> None:
    plt.rcParams.update({"font.size": font_size, "legend.fontsize": legend_font_size})
    rows_arch = [r for r in agg_rows if r["arch"] == arch]
    if not rows_arch:
        return

    order = ["weight_decay"] + [method_key("mne_l2", c) for c in rc_list]
    seed_label = f"{min(seeds)}..{max(seeds)}" if len(seeds) > 1 else str(seeds[0])

    for no_caption in (False, True):
        fig, ax = plt.subplots(figsize=(10.0, 6.2), dpi=180)
        all_y: list[float] = []
        for key in order:
            rr = [r for r in rows_arch if r["method"] == key]
            if not rr:
                continue
            rr.sort(key=lambda x: float(x["sigma"]))
            x = [float(r["sigma"]) for r in rr]
            y = [float(r["acc_mean"]) for r in rr]
            s = [float(r["acc_std"]) for r in rr]
            all_y.extend([yy - ss for yy, ss in zip(y, s)])
            all_y.extend([yy + ss for yy, ss in zip(y, s)])
            style = line_style_for(key, rc_list)
            ax.plot(
                x, y, marker="o", linewidth=2.2, markersize=5,
                color=style["color"], label=style["label"],
            )
            if any(ss > 0 for ss in s):
                ax.fill_between(
                    x,
                    [yy - ss for yy, ss in zip(y, s)],
                    [yy + ss for yy, ss in zip(y, s)],
                    color=style["color"], alpha=0.18, linewidth=0,
                )
        ax.set_xlabel("Gaussian noise sigma")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([round(i * 0.1, 1) for i in range(11)])
        if all_y:
            ax.set_ylim(min(all_y) - 1.0, max(all_y) + 1.0)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", frameon=False, ncol=2)
        if not no_caption:
            ax.set_title(
                f"MNIST {arch} mne_l2 rc scan vs wd "
                f"(seeds={seed_label}, L={LVAL}, T={TVAL}, {IF_MODE})"
            )
        fig.tight_layout()
        suffix = "_no_caption" if no_caption else ""
        out_png = OUT / f"{arch}_mne_reg_coeff_scan_noise_sweep_mean_std{suffix}.png"
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] saved {out_png}", flush=True)


def print_arch_summary(arch: str, summary_rows: list[dict], seeds: list[int]) -> None:
    rows = [r for r in summary_rows if r["arch"] == arch]
    if not rows:
        return
    seed_label = f"{min(seeds)}..{max(seeds)}" if len(seeds) > 1 else str(seeds[0])
    n = rows[0]["n_seeds"]
    print(f"\n--- {arch} summary (seeds={seed_label}, n={n}) ---", flush=True)
    print(
        f"{'method':<22} {'acc@0':>16} {'acc@1':>16} {'drop':>16}",
        flush=True,
    )
    for row in rows:
        rc_str = row["reg_coeff"] if row["reg_coeff"] else "wd"
        print(
            f"{row['method']:<22} "
            f"{float(row['acc_sigma0_mean']):>7.3f}±{float(row['acc_sigma0_std']):<6.3f} "
            f"{float(row['acc_sigma1_mean']):>7.3f}±{float(row['acc_sigma1_std']):<6.3f} "
            f"{float(row['acc_drop_mean']):>7.3f}±{float(row['acc_drop_std']):<6.3f}  ({rc_str})",
            flush=True,
        )


def finalize_tables_and_plots(
    arch_pairs: list[tuple[int, int]],
    rc_list: list[float],
    seeds: list[int],
    font_size: float,
    legend_font_size: float,
    copy_root: bool,
) -> None:
    raw_rows = load_raw_rows()
    if not raw_rows:
        raise SystemExit("无可用结果，请先运行训练+测试或检查 OUT 目录")

    agg_rows = aggregate_curve_rows(raw_rows)
    summary_rows = aggregate_summary_rows(raw_rows)
    best_rc_rows = pick_best_rc_per_arch(summary_rows)

    with AGG_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AGG_FIELDS)
        w.writeheader()
        w.writerows(agg_rows)
    with SUMMARY_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summary_rows)
    with BEST_RC_CSV.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "arch", "best_reg_coeff",
                "acc_sigma0_mean", "acc_sigma0_std",
                "acc_sigma1_mean", "acc_sigma1_std",
                "acc_drop_mean", "acc_drop_std", "n_seeds",
            ],
        )
        w.writeheader()
        w.writerows(best_rc_rows)

    print(f"\n[TABLE] raw: {RAW_CSV}", flush=True)
    print(f"[TABLE] agg: {AGG_CSV}", flush=True)
    print(f"[TABLE] summary: {SUMMARY_CSV}", flush=True)
    print(f"[TABLE] best rc: {BEST_RC_CSV}", flush=True)

    for c1, c2 in arch_pairs:
        arch = arch_name(c1, c2)
        plot_arch(arch, agg_rows, rc_list, seeds, font_size, legend_font_size)
        print_arch_summary(arch, summary_rows, seeds)

    print("\n--- best mne_l2 rc per arch (min mean drop) ---", flush=True)
    for row in best_rc_rows:
        print(
            f"{row['arch']}: rc={row['best_reg_coeff']}  "
            f"drop={float(row['acc_drop_mean']):.3f}±{float(row['acc_drop_std']):.3f}  "
            f"acc@0={float(row['acc_sigma0_mean']):.3f}±{float(row['acc_sigma0_std']):.3f}",
            flush=True,
        )

    if copy_root:
        for c1, c2 in arch_pairs:
            arch = arch_name(c1, c2)
            src = OUT / f"{arch}_mne_reg_coeff_scan_noise_sweep_mean_std_no_caption.png"
            if src.exists():
                dest = REPO_ROOT / src.name
                if dest.resolve() != src.resolve():
                    shutil.copy2(src, dest)
                    print(f"[PLOT] copied {dest}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MNIST CNN mne_l2 rc scan + noise sweep (multi-seed)")
    p.add_argument(
        "--arch-list",
        nargs="+",
        default=None,
        help="c2c4 c4c8 c8c16 c16c32（默认四个全跑）",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--seed", type=int, default=None, help="只跑单个 seed（覆盖 --seeds）")
    p.add_argument(
        "--rc-list",
        type=float,
        nargs="+",
        default=None,
        help=f"reg_coeff 列表（默认 {' '.join(f'{x:.0e}' for x in DEFAULT_RC_LIST)}）",
    )
    p.add_argument("--skip-wd", action="store_true", help="跳过 weight_decay 基线")
    p.add_argument("--retrain", action="store_true", help="删除已有 checkpoint 后重训")
    p.add_argument("--force-test", action="store_true", help="删除已有 matrix 后重测")
    p.add_argument("--plot-only", action="store_true", help="仅从 raw CSV 重算 agg/summary 并出图")
    p.add_argument("--font-size", type=float, default=11.0)
    p.add_argument("--legend-font-size", type=float, default=9.0)
    p.add_argument(
        "--copy-root",
        action="store_true",
        help="复制各 arch 的 no_caption 图到仓库根目录",
    )
    return p.parse_args()


def run_one_config(
    arch: str,
    reg: str,
    seed: int,
    rc_list: list[float],
    reg_coeff: Optional[float],
    retrain: bool,
    force_test: bool,
) -> list[dict]:
    ckpt = train_one(arch, reg, seed, reg_coeff=reg_coeff, retrain=retrain)
    tag = method_key(reg, reg_coeff)
    mat = test_noise_sweep(arch, tag, seed, ckpt, force_test=force_test)
    rc_str = f"{reg_coeff:.0e}" if reg_coeff is not None else ""
    rows = []
    for sigma, acc in read_matrix(mat):
        rows.append(
            {
                "arch": arch,
                "method": tag,
                "regularizer": reg,
                "reg_coeff": rc_str,
                "seed": seed,
                "L": LVAL,
                "T": TVAL,
                "if_mode": IF_MODE,
                "sigma": sigma,
                "acc": acc,
                "checkpoint": str(ckpt.relative_to(ROOT)),
                "matrix_csv": str(mat.relative_to(ROOT)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    arch_pairs = resolve_arch_list(args.arch_list)
    rc_list = args.rc_list if args.rc_list is not None else list(DEFAULT_RC_LIST)
    seeds = [args.seed] if args.seed is not None else args.seeds
    include_wd = not args.skip_wd
    methods = methods_for_run(rc_list, include_wd)
    archs = {arch_name(c1, c2) for c1, c2 in arch_pairs}

    if args.plot_only:
        finalize_tables_and_plots(
            arch_pairs, rc_list, seeds, args.font_size, args.legend_font_size, args.copy_root
        )
        return

    new_rows: list[dict] = []
    for c1, c2 in arch_pairs:
        arch = arch_name(c1, c2)
        seed_label = f"{min(seeds)}..{max(seeds)}" if len(seeds) > 1 else str(seeds[0])
        print(f"\n=== {arch} rc scan seeds={seed_label} ===", flush=True)
        for seed in seeds:
            if include_wd:
                new_rows.extend(
                    run_one_config(
                        arch, "weight_decay", seed, rc_list, None,
                        args.retrain, args.force_test,
                    )
                )
            for coeff in rc_list:
                new_rows.extend(
                    run_one_config(
                        arch, "mne_l2", seed, rc_list, coeff,
                        args.retrain, args.force_test,
                    )
                )

    upsert_run_rows(new_rows, archs, methods, seeds)
    finalize_tables_and_plots(
        arch_pairs, rc_list, seeds, args.font_size, args.legend_font_size, args.copy_root
    )


if __name__ == "__main__":
    main()
