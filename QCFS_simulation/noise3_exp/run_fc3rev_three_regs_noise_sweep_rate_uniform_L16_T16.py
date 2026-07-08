"""
FC3rev (2h->h) 三路正则 + rate_uniform 噪声扫描（仅 L=16 训练 + 噪声测试，不含 clean acc）。

方法：mne_l2 / weight_decay / no_regularization
测试：L=16, T=16, IF mode=rate_uniform, sigma=0~1 step=0.05, seeds=40..44

用法（Gadi）：
  cd ~/codes/snn_simulation/QCFS_simulation
  export MNIST_ROOT=/scratch/gs14/sl9144/datasets

  # 仅补跑 MNE-L2 和 No Reg（weight_decay 已有）
  python -u noise3_exp/run_fc3rev_three_regs_noise_sweep_rate_uniform_L16_T16.py \
    --h-list 8 16 32 64 128 256 \
    --seeds 40 41 42 43 44 \
    --regs mne_l2 no_regularization \
    --epochs 50 \
    --out-dir ../important_results/new_fc3

  # 三路全量重跑
  python -u noise3_exp/run_fc3rev_three_regs_noise_sweep_rate_uniform_L16_T16.py \
    --h-list 8 16 32 64 128 256 \
    --seeds 40 41 42 43 44 \
    --regs mne_l2 weight_decay no_regularization \
    --retrain --force-noise-test \
    --epochs 50 \
    --out-dir ../important_results/new_fc3
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LVAL = 16
TVAL = 16
IF_MODE = "rate_uniform"
DEFAULT_REGS = ["mne_l2", "weight_decay", "no_regularization"]
BATCH = int(os.environ.get("FC3REV_BATCH", "128"))
NUM_WORKERS = int(os.environ.get("FC3REV_NUM_WORKERS", "0"))


def _path_for_csv(path: Path) -> str:
    p = path.resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(p)


def coeff_tag(v: float) -> str:
    return f"{v:.0e}".replace("-", "m").replace("+", "p")


def arch_for(h: int) -> str:
    return f"fc3rev_h{h}"


def build_suffix(arch: str, reg: str, seed: int, mne_rc: float = 5e-2) -> str:
    if reg == "weight_decay":
        return f"strict_seed{seed}_ablation_wd_l{LVAL}_{arch}"
    if reg == "no_regularization":
        return f"strict_seed{seed}_ablation_none_l{LVAL}_{arch}"
    return f"strict_seed{seed}_ablation_mne_l2_l{LVAL}_{arch}_rc{coeff_tag(mne_rc)}"


def ckpt_path(arch: str, reg: str, seed: int, mne_rc: float = 5e-2) -> Path:
    suffix = build_suffix(arch, reg, seed, mne_rc)
    return PROJECT_ROOT / "mnist-checkpoints" / f"{arch}_L[{LVAL}]_{suffix}.pth"


def test_out_dir(noise_root: Path, arch: str, reg: str, seed: int) -> Path:
    return noise_root / arch / reg / f"seed_{seed}"


def legacy_wd_out_dir(noise_root: Path, arch: str, seed: int) -> Path:
    return noise_root / arch / f"seed_{seed}"


def matrix_path(out_dir: Path, arch: str, seed: int) -> Path:
    return (
        out_dir
        / f"noise_sweep_matrix_mnist_{arch}_T{TVAL}_mode_{IF_MODE}_schedule_normal_seed_{seed}.csv"
    )


def train_one(
    arch: str, reg: str, seed: int, epochs: int, retrain: bool, mne_rc: float = 5e-2
) -> Path:
    ckpt = ckpt_path(arch, reg, seed, mne_rc)
    if retrain and ckpt.exists():
        ckpt.unlink()
        print(f"[RETRAIN] removed {ckpt.name}", flush=True)
    if ckpt.exists():
        print(f"[SKIP TRAIN] {ckpt.name}", flush=True)
        return ckpt

    if reg == "mne_l2":
        regularizer, wd, rc = "mne_l2", 0.0, mne_rc
    elif reg == "weight_decay":
        regularizer, wd, rc = "weight_decay", 5e-4, 1.0
    else:
        regularizer, wd, rc = "weight_decay", 0.0, 1.0

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "main_train.py"),
        "-data",
        "mnist",
        "-arch",
        arch,
        "-L",
        str(LVAL),
        "--epochs",
        str(epochs),
        "-j",
        str(NUM_WORKERS),
        "-b",
        str(BATCH),
        "--seed",
        str(seed),
        "--device",
        "auto",
        "--time",
        "0",
        "--spike_schedule",
        "normal",
        "--regularizer",
        regularizer,
        "--weight_decay",
        str(wd),
        "--reg_coeff",
        str(rc),
        "--suffix",
        build_suffix(arch, reg, seed, mne_rc),
    ]
    print(f"[TRAIN] {arch} {reg} seed={seed} epochs={epochs}", flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
    return ckpt


def run_noise_sweep(
    arch: str,
    reg: str,
    seed: int,
    ckpt: Path,
    out_dir: Path,
    force_noise_test: bool,
) -> Path:
    mat = matrix_path(out_dir, arch, seed)
    if force_noise_test and mat.exists():
        mat.unlink()
    if mat.exists():
        print(f"[SKIP NOISE] {arch} {reg} seed={seed}", flush=True)
        return mat

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "main_test.py"),
        "-data",
        "mnist",
        "-arch",
        arch,
        "-L",
        str(LVAL),
        "-T",
        str(TVAL),
        "-j",
        str(NUM_WORKERS),
        "-b",
        str(BATCH),
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
        "0.05",
        "--noise_output_dir",
        str(out_dir),
    ]
    print(f"[NOISE] {arch} {reg} seed={seed} mode={IF_MODE} L={LVAL} T={TVAL}", flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    if not mat.exists():
        cands = sorted(out_dir.glob(f"noise_sweep_matrix_*_T{TVAL}_mode_{IF_MODE}_schedule_normal_seed_{seed}.csv"))
        if not cands:
            raise FileNotFoundError(f"noise matrix missing: {out_dir}")
        mat = cands[0]
    return mat


def read_matrix(mat: Path):
    with mat.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)
    start = 2 if header[:2] == ["L", "T"] else 1
    sigmas = [float(x) for x in header[start:]]
    accs = [float(x) for x in row[start:]]
    return list(zip(sigmas, accs))


def discover_matrix_files(noise_root: Path, arch: str, reg: str, seeds: list[int]) -> list[Path]:
    files: list[Path] = []
    for seed in seeds:
        new_dir = test_out_dir(noise_root, arch, reg, seed)
        mat = matrix_path(new_dir, arch, seed)
        if mat.exists():
            files.append(mat)
            continue
        if reg == "weight_decay":
            legacy_dir = legacy_wd_out_dir(noise_root, arch, seed)
            legacy_mat = matrix_path(legacy_dir, arch, seed)
            if legacy_mat.exists():
                files.append(legacy_mat)
    return files


def collect_all_rows(
    noise_root: Path,
    h_list: list[int],
    regs: list[str],
    seeds: list[int],
    mne_rc: float = 5e-2,
) -> list[dict]:
    rows: list[dict] = []
    for h in h_list:
        arch = arch_for(h)
        for reg in regs:
            ckpt_guess = ckpt_path(arch, reg, seeds[0], mne_rc) if seeds else None
            for seed in seeds:
                ckpt = ckpt_path(arch, reg, seed, mne_rc)
                for mat in discover_matrix_files(noise_root, arch, reg, [seed]):
                    for sigma, acc in read_matrix(mat):
                        rows.append(
                            {
                                "arch": arch,
                                "hidden_size": h,
                                "regularizer": reg,
                                "if_mode": IF_MODE,
                                "L": LVAL,
                                "T": TVAL,
                                "seed": seed,
                                "sigma": f"{sigma:.2f}",
                                "acc": f"{acc:.6f}",
                                "checkpoint": _path_for_csv(ckpt if ckpt.exists() else ckpt_guess or ckpt),
                                "matrix_csv": _path_for_csv(mat),
                            }
                        )
    return rows


def _load_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _merge_rows_by_arch(existing: list[dict], new_rows: list[dict], archs: set[str]) -> list[dict]:
    kept = [r for r in existing if r["arch"] not in archs]
    kept.extend(new_rows)
    kept.sort(
        key=lambda r: (
            int(r["hidden_size"]),
            r["regularizer"],
            int(r["seed"]),
            float(r["sigma"]),
        )
    )
    return kept


RAW_FIELDS = [
    "arch",
    "hidden_size",
    "regularizer",
    "if_mode",
    "L",
    "T",
    "seed",
    "sigma",
    "acc",
    "checkpoint",
    "matrix_csv",
]


def write_noise_tables(rows: list[dict], raw_csv: Path, mean_csv: Path) -> None:
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    with raw_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writeheader()
        w.writerows(rows)

    bucket = defaultdict(list)
    for r in rows:
        bucket[(r["arch"], int(r["hidden_size"]), r["regularizer"], float(r["sigma"]))].append(float(r["acc"]))

    mean_rows = []
    for (arch, h, reg, sigma), vals in sorted(bucket.items(), key=lambda x: (x[0][1], x[0][2], x[0][3])):
        mean_rows.append(
            {
                "arch": arch,
                "hidden_size": h,
                "regularizer": reg,
                "if_mode": IF_MODE,
                "L": LVAL,
                "T": TVAL,
                "sigma": f"{sigma:.2f}",
                "acc_mean": f"{statistics.mean(vals):.6f}",
                "acc_std": f"{(statistics.stdev(vals) if len(vals) > 1 else 0.0):.6f}",
                "n_seeds": len(vals),
            }
        )

    with mean_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "arch",
                "hidden_size",
                "regularizer",
                "if_mode",
                "L",
                "T",
                "sigma",
                "acc_mean",
                "acc_std",
                "n_seeds",
            ],
        )
        w.writeheader()
        w.writerows(mean_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FC3rev three-regs noise sweep (L16 train + rate_uniform test)")
    p.add_argument("--h-list", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    p.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    p.add_argument("--regs", nargs="+", default=DEFAULT_REGS, choices=DEFAULT_REGS)
    p.add_argument(
        "--mne-reg-coeff",
        type=float,
        default=5e-2,
        help="MNE-L2 reg_coeff (default 5e-2; use 1e-3 for tuned rerun)",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--force-noise-test", action="store_true")
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT.parent / "important_results" / "new_fc3"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    noise_root = out_dir / "noise_sweep_rate_uniform_L16_T16"

    raw_csv = out_dir / "fc3rev_h8_h256_three_regs_noise_sweep_raw.csv"
    mean_csv = out_dir / "fc3rev_h8_h256_three_regs_noise_sweep_mean_std.csv"

    print(f"[MNE-REG-COEFF] {args.mne_reg_coeff}", flush=True)
    print(f"[FC3REV_BATCH] {BATCH}", flush=True)
    print(f"[FC3REV_NUM_WORKERS] {NUM_WORKERS}", flush=True)

    for h in args.h_list:
        arch = arch_for(h)
        for reg in args.regs:
            for seed in args.seeds:
                ckpt = train_one(arch, reg, seed, args.epochs, args.retrain, args.mne_reg_coeff)
                out_seed_dir = test_out_dir(noise_root, arch, reg, seed)
                run_noise_sweep(arch, reg, seed, ckpt, out_seed_dir, args.force_noise_test)

    all_regs = sorted(set(DEFAULT_REGS))
    archs_run = {arch_for(h) for h in args.h_list}
    new_rows = collect_all_rows(
        noise_root, args.h_list, all_regs, args.seeds, args.mne_reg_coeff
    )
    if not new_rows:
        raise RuntimeError("No noise sweep rows collected; check output directories.")
    rows = _merge_rows_by_arch(_load_csv_rows(raw_csv), new_rows, archs_run)
    write_noise_tables(rows, raw_csv, mean_csv)

    print(f"[DONE] noise raw: {raw_csv}", flush=True)
    print(f"[DONE] noise mean: {mean_csv}", flush=True)


if __name__ == "__main__":
    main()
