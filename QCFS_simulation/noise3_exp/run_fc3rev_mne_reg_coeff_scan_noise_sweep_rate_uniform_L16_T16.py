"""
FC3rev (2h->h)：mne_l2 reg_coeff 扫描 + weight_decay 基线 + rate_uniform 噪声注入。

默认 h ∈ {8, 128, 256}，rc ∈ {1e-3, 5e-3, 1e-2, 5e-2, 1e-1}，seeds=40..44。
仅扫描 MNE-L2 各 reg_coeff，按 RS 选最优 β（默认不含 L2 基线）。
训练 L=16 T=0 epochs=50；测试 L=16 T=16 rate_uniform sigma step=0.05。
checkpoint 后缀含 rcscan，不覆盖 strict-seed 权重。

Gadi:
  cd ~/codes/snn_simulation/QCFS_simulation
  export MNIST_ROOT=/scratch/gs14/sl9144/datasets
  python -u noise3_exp/run_fc3rev_mne_reg_coeff_scan_noise_sweep_rate_uniform_L16_T16.py
"""
from __future__ import annotations

import argparse
import csv
import os
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

DEFAULT_H_LIST = [8, 128, 256]
DEFAULT_SEEDS = [40, 41, 42, 43, 44]
DEFAULT_RC_LIST = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
LVAL = 16
TVAL = 16
IF_MODE = "rate_uniform"
WD = 5e-4
EPOCHS = int(os.environ.get("FC3REV_EPOCHS", "50"))
BATCH = int(os.environ.get("FC3REV_BATCH", "128"))
NUM_WORKERS = int(os.environ.get("FC3REV_NUM_WORKERS", "0"))
SCAN_TAG = "rcscan"

RC_COLORS = {
    "1em03": "#aec7e8",
    "5em03": "#17becf",
    "1em02": "#2ca02c",
    "5em02": "#9467bd",
    "1em01": "#d62728",
}

RAW_FIELDS = [
    "arch", "hidden_size", "method", "regularizer", "reg_coeff", "seed",
    "L", "T", "if_mode", "sigma", "acc", "checkpoint", "matrix_csv",
]
SUMMARY_FIELDS = [
    "arch", "hidden_size", "method", "regularizer", "reg_coeff",
    "acc_sigma0_mean", "acc_sigma0_std", "acc_sigma1_mean", "acc_sigma1_std",
    "acc_drop_mean", "acc_drop_std", "RS_mean", "RS_sem", "n_seeds",
]


def coeff_tag(v: float) -> str:
    return f"{v:.0e}".replace("-", "m").replace("+", "p")


def arch_for(h: int) -> str:
    return f"fc3rev_h{h}"


def method_key(reg: str, reg_coeff: Optional[float] = None) -> str:
    if reg == "weight_decay":
        return "weight_decay"
    return f"mne_l2:{coeff_tag(reg_coeff)}"


def build_suffix(arch: str, reg: str, seed: int, reg_coeff: Optional[float] = None) -> str:
    if reg == "weight_decay":
        return f"seed{seed}_{SCAN_TAG}_wd_l{LVAL}_{arch}"
    return f"seed{seed}_{SCAN_TAG}_mne_l2_l{LVAL}_{arch}_rc{coeff_tag(reg_coeff)}"


def ckpt_path(arch: str, reg: str, seed: int, reg_coeff: Optional[float] = None) -> Path:
    return ROOT / "mnist-checkpoints" / f"{arch}_L[{LVAL}]_{build_suffix(arch, reg, seed, reg_coeff)}.pth"


def test_out_dir(out_root: Path, arch: str, tag: str, seed: int) -> Path:
    safe = tag.replace(":", "_")
    return out_root / "noise_sweep_rate_uniform_L16_T16" / arch / safe / f"seed_{seed}"


def train_one(
    arch: str, reg: str, seed: int, reg_coeff: Optional[float], retrain: bool
) -> Path:
    ckpt = ckpt_path(arch, reg, seed, reg_coeff)
    if retrain and ckpt.exists():
        ckpt.unlink()
    if ckpt.exists():
        print(f"[SKIP TRAIN] {ckpt.name}", flush=True)
        return ckpt

    if reg == "mne_l2":
        regularizer, wd, rc = "mne_l2", 0.0, reg_coeff
    else:
        regularizer, wd, rc = "weight_decay", WD, 1.0

    cmd = [
        sys.executable, str(ROOT / "main_train.py"),
        "-data", "mnist", "-arch", arch, "-L", str(LVAL),
        "--epochs", str(EPOCHS), "-j", str(NUM_WORKERS), "-b", str(BATCH),
        "--seed", str(seed), "--device", "auto", "--time", "0",
        "--spike_schedule", "normal",
        "--regularizer", regularizer, "--weight_decay", str(wd),
        "--reg_coeff", str(rc),
        "--suffix", build_suffix(arch, reg, seed, reg_coeff),
    ]
    print(f"[TRAIN] {arch} {method_key(reg, reg_coeff)} seed={seed}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
    return ckpt


def test_noise_sweep(
    out_root: Path, arch: str, tag: str, seed: int, ckpt: Path, force_test: bool
) -> Path:
    out_dir = test_out_dir(out_root, arch, tag, seed)
    matrix = (
        out_dir
        / f"noise_sweep_matrix_mnist_{arch}_T{TVAL}_mode_{IF_MODE}_schedule_normal_seed_{seed}.csv"
    )
    if force_test and matrix.exists():
        matrix.unlink()
    if matrix.exists():
        print(f"[SKIP TEST] {arch} {tag} seed={seed}", flush=True)
        return matrix

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "main_test.py"),
        "-data", "mnist", "-arch", arch, "-L", str(LVAL), "-T", str(TVAL),
        "-j", str(NUM_WORKERS), "-b", str(BATCH), "--seed", str(seed),
        "--device", "auto", "--mode", IF_MODE, "--spike_schedule", "normal",
        "--weights", str(ckpt), "--noise_sweep",
        "--noise_sigma_start", "0.0", "--noise_sigma_end", "1.0",
        "--noise_sigma_step", "0.05", "--noise_output_dir", str(out_dir),
    ]
    print(f"[TEST] {arch} {tag} seed={seed}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    if not matrix.exists():
        cands = sorted(out_dir.glob(f"noise_sweep_matrix_*_seed_{seed}.csv"))
        if not cands:
            raise FileNotFoundError(f"matrix missing: {out_dir}")
        matrix = cands[0]
    return matrix


def read_matrix(mat: Path) -> list[tuple[float, float]]:
    with mat.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)
    start = 2 if header[:2] == ["L", "T"] else 1
    return list(zip([float(x) for x in header[start:]], [float(x) for x in row[start:]]))


def robust_score(sigmas: list[float], accs: list[float]) -> float:
    a0 = accs[0]
    if a0 <= 0:
        return 0.0
    rs = 0.0
    for i in range(len(sigmas) - 1):
        ds = sigmas[i + 1] - sigmas[i]
        if ds > 0:
            rs += 0.5 * (accs[i] / a0 + accs[i + 1] / a0) * ds
    return rs


def load_raw(raw_csv: Path) -> list[dict]:
    if not raw_csv.exists():
        return []
    with raw_csv.open(newline="") as f:
        return list(csv.DictReader(f))


def upsert_rows(raw_csv: Path, new_rows: list[dict], archs: set[str], methods: list[str], seeds: list[int]) -> None:
    kept = [
        r for r in load_raw(raw_csv)
        if not (r["arch"] in archs and r["method"] in methods and int(r["seed"]) in seeds)
    ]
    kept.extend(new_rows)
    kept.sort(key=lambda r: (int(r["hidden_size"]), r["method"], int(r["seed"]), float(r["sigma"])))
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    with raw_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writeheader()
        w.writerows(kept)


def aggregate_summary(raw_rows: list[dict]) -> list[dict]:
    by_key: dict[tuple, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for r in raw_rows:
        key = (r["arch"], int(r["hidden_size"]), r["method"], r["regularizer"], r["reg_coeff"])
        by_key[key][int(r["seed"])].append((float(r["sigma"]), float(r["acc"])))

    rows = []
    for (arch, h, method, reg, rc_str), seed_map in sorted(by_key.items(), key=lambda x: (x[0][1], x[0][2])):
        a0s, a1s, drops, rss = [], [], [], []
        for pairs in seed_map.values():
            pairs = sorted(pairs, key=lambda x: x[0])
            sigmas = [p[0] for p in pairs]
            accs = [p[1] for p in pairs]
            a0, a1 = accs[0], accs[-1]
            a0s.append(a0)
            a1s.append(a1)
            drops.append(a0 - a1)
            rss.append(robust_score(sigmas, accs))
        n = len(a0s)
        rs_std = statistics.stdev(rss) if n > 1 else 0.0
        rows.append({
            "arch": arch, "hidden_size": h, "method": method,
            "regularizer": reg, "reg_coeff": rc_str,
            "acc_sigma0_mean": f"{statistics.mean(a0s):.6f}",
            "acc_sigma0_std": f"{(statistics.stdev(a0s) if n > 1 else 0.0):.6f}",
            "acc_sigma1_mean": f"{statistics.mean(a1s):.6f}",
            "acc_sigma1_std": f"{(statistics.stdev(a1s) if n > 1 else 0.0):.6f}",
            "acc_drop_mean": f"{statistics.mean(drops):.6f}",
            "acc_drop_std": f"{(statistics.stdev(drops) if n > 1 else 0.0):.6f}",
            "RS_mean": f"{statistics.mean(rss):.6f}",
            "RS_sem": f"{(rs_std / (n ** 0.5) if n > 0 else 0.0):.6f}",
            "n_seeds": n,
        })
    return rows


def pick_best_rc(summary_rows: list[dict]) -> list[dict]:
    by_arch: dict[str, list[dict]] = defaultdict(list)
    for r in summary_rows:
        if r["regularizer"] == "mne_l2":
            by_arch[r["arch"]].append(r)
    best = []
    for arch in sorted(by_arch, key=lambda a: int(a.split("_h")[1])):
        cands = by_arch[arch]
        winner = max(cands, key=lambda r: (float(r["RS_mean"]), float(r["acc_sigma0_mean"])))
        best.append({
            "arch": arch,
            "hidden_size": winner["hidden_size"],
            "best_reg_coeff": winner["reg_coeff"],
            "RS_mean": winner["RS_mean"],
            "RS_sem": winner["RS_sem"],
            "acc_sigma0_mean": winner["acc_sigma0_mean"],
            "acc_drop_mean": winner["acc_drop_mean"],
            "n_seeds": winner["n_seeds"],
        })
    return best


def plot_arch(out_dir: Path, arch: str, raw_rows: list[dict], rc_list: list[float], include_wd: bool) -> None:
    rows = [r for r in raw_rows if r["arch"] == arch]
    if not rows:
        return
    order = (["weight_decay"] if include_wd else []) + [method_key("mne_l2", c) for c in rc_list]
    fig, ax = plt.subplots(figsize=(10.8, 7.2), dpi=220)
    plt.style.use("seaborn-v0_8-whitegrid")
    for key in order:
        rr = [r for r in rows if r["method"] == key]
        if not rr:
            continue
        bucket: dict[float, list[float]] = defaultdict(list)
        for r in rr:
            bucket[round(float(r["sigma"]), 6)].append(float(r["acc"]))
        xs = sorted(bucket)
        ys = [statistics.mean(bucket[x]) for x in xs]
        ysem = [
            statistics.stdev(bucket[x]) / (len(bucket[x]) ** 0.5) if len(bucket[x]) > 1 else 0.0
            for x in xs
        ]
        if key == "weight_decay":
            color, label = "#ff7f0e", "L2"
        else:
            rc_tag = key.split(":", 1)[1]
            color = RC_COLORS.get(rc_tag, "#333333")
            label = f"MNE-L2 rc={next(c for c in rc_list if coeff_tag(c) == rc_tag):.0e}"
        ax.plot(xs, ys, marker="o", linewidth=2.6, color=color, label=label)
        ax.fill_between(xs, [y - s for y, s in zip(ys, ysem)], [y + s for y, s in zip(ys, ysem)],
                        color=color, alpha=0.14, linewidth=0)
    ax.set_xlabel("Gaussian noise sigma")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([i / 10 for i in range(11)])
    ax.grid(alpha=0.24)
    ax.legend(loc="lower left", frameon=False, ncol=2)
    fig.tight_layout()
    h = arch.split("_h")[1]
    out_png = out_dir / f"fc3rev_h{h}_mne_reg_coeff_scan_noise_sweep.png"
    fig.savefig(out_png)
    fig.savefig(out_dir / f"fc3rev_h{h}_mne_reg_coeff_scan_noise_sweep.pdf")
    plt.close(fig)
    print(f"[PLOT] {out_png}", flush=True)


def run_config(
    out_root: Path, h: int, reg: str, seed: int, rc: Optional[float],
    retrain: bool, force_test: bool,
) -> list[dict]:
    arch = arch_for(h)
    ckpt = train_one(arch, reg, seed, rc, retrain)
    tag = method_key(reg, rc)
    mat = test_noise_sweep(out_root, arch, tag, seed, ckpt, force_test)
    rc_str = f"{rc:.0e}" if rc is not None else ""
    return [
        {
            "arch": arch, "hidden_size": h, "method": tag, "regularizer": reg,
            "reg_coeff": rc_str, "seed": seed, "L": LVAL, "T": TVAL,
            "if_mode": IF_MODE, "sigma": sigma, "acc": acc,
            "checkpoint": str(ckpt.relative_to(ROOT)),
            "matrix_csv": str(mat.relative_to(ROOT)),
        }
        for sigma, acc in read_matrix(mat)
    ]


def finalize(out_root: Path, h_list: list[int], rc_list: list[float], include_wd: bool) -> None:
    raw_csv = out_root / "fc3rev_mne_reg_coeff_scan_noise_sweep_raw.csv"
    summary_csv = out_root / "fc3rev_mne_reg_coeff_scan_summary.csv"
    best_csv = out_root / "fc3rev_mne_reg_coeff_scan_best_rc.csv"
    plot_dir = out_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = load_raw(raw_csv)
    if not raw_rows:
        raise SystemExit("无 raw 数据")

    summary = aggregate_summary(raw_rows)
    best = pick_best_rc(summary)
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summary)
    with best_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(best[0].keys()) if best else [])
        if best:
            w.writeheader()
            w.writerows(best)

    for h in h_list:
        plot_arch(plot_dir, arch_for(h), raw_rows, rc_list, include_wd)

    print(f"\n[DONE] raw: {raw_csv}", flush=True)
    print(f"[DONE] summary: {summary_csv}", flush=True)
    print(f"[DONE] best rc: {best_csv}", flush=True)
    print("\n--- best mne_l2 rc per arch (max RS) ---", flush=True)
    for row in best:
        print(
            f"{row['arch']}: rc={row['best_reg_coeff']}  "
            f"RS={float(row['RS_mean']):.4f}±{float(row['RS_sem']):.4f}  "
            f"acc@0={float(row['acc_sigma0_mean']):.3f}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FC3rev mne_l2 reg_coeff scan + noise sweep")
    p.add_argument("--h-list", type=int, nargs="+", default=DEFAULT_H_LIST)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--rc-list", type=float, nargs="+", default=None)
    p.add_argument(
        "--include-wd",
        action="store_true",
        help="额外训练 weight_decay 基线用于对比（默认只扫 mne_l2）",
    )
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--force-test", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument(
        "--out-dir",
        default=str(ROOT.parent / "important_results" / "new_fc3" / "mne_reg_coeff_scan"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    h_list = args.h_list
    rc_list = args.rc_list if args.rc_list is not None else list(DEFAULT_RC_LIST)
    seeds = [args.seed] if args.seed is not None else args.seeds
    include_wd = args.include_wd
    methods = (["weight_decay"] if include_wd else []) + [method_key("mne_l2", c) for c in rc_list]
    archs = {arch_for(h) for h in h_list}
    raw_csv = out_root / "fc3rev_mne_reg_coeff_scan_noise_sweep_raw.csv"

    if args.plot_only:
        finalize(out_root, h_list, rc_list, include_wd)
        return

    new_rows: list[dict] = []
    for h in h_list:
        arch = arch_for(h)
        print(f"\n=== {arch} rc scan seeds={seeds} ===", flush=True)
        for seed in seeds:
            if include_wd:
                new_rows.extend(run_config(out_root, h, "weight_decay", seed, None, args.retrain, args.force_test))
            for rc in rc_list:
                new_rows.extend(run_config(out_root, h, "mne_l2", seed, rc, args.retrain, args.force_test))

    upsert_rows(raw_csv, new_rows, archs, methods, seeds)
    finalize(out_root, h_list, rc_list, include_wd)


if __name__ == "__main__":
    main()
