"""
MNIST ANN (T=0) 测试精度：FC3 / CNN2 不同规模，L=2,4,8,16,32。

与 run_fc3_wd_strict_seed_L_T_acc.py / run_cnn_wd_strict_seed_L_T_acc.py
共用 checkpoint 命名（strict_seed + ablation_wd），训练阶段亦为 T=0 ANN。

模型：
  FC3:  fc3_h4 / h8 / h16 / h32 / h64 / h128
  CNN2: cnn2_c2_c4 / c4_c8 / c8_c16 / c16_c32

用法：
  python noise3_exp/run_mnist_ann_T0_L_acc.py
  python noise3_exp/run_mnist_ann_T0_L_acc.py --model-type fc
  python noise3_exp/run_mnist_ann_T0_L_acc.py --model-type cnn --force-test
  bash noise3_exp/RUN_mnist_ann_T0_L_acc.sh
"""
from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Models import modelpool
from Preprocess import datapool
from utils import get_torch_device, seed_all, val

OUT = ROOT / "noise3_exp" / "mnist_ann_T0_L_acc"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [40, 41, 42, 43, 44]
L_LIST = [2, 4, 8, 16, 32]
T_ANN = 0
IF_MODE = "normal"
SPIKE_SCHEDULE = "normal"
EPOCHS = 100
BATCH = 128

FC_H_LIST = [4, 8, 16, 32, 64, 128]
CNN_VARIANTS = [(2, 4), (4, 8), (8, 16), (16, 32)]

RAW_CSV = OUT / "mnist_ann_T0_L_acc_raw.csv"
MEAN_STD_CSV = OUT / "mnist_ann_T0_L_acc_mean_std.csv"
L2_L16_SUMMARY_CSV = OUT / "mnist_ann_T0_L_acc_L2_vs_L16_summary.csv"


def fc_arch(h: int) -> str:
    return f"fc3_h{h}"


def cnn_arch(c1: int, c2: int) -> str:
    return f"cnn2_c{c1}_c{c2}"


def cnn_label(c1: int, c2: int) -> str:
    return f"c{c1}c{c2}"


def build_suffix(arch: str, l_val: int, seed: int) -> str:
    return f"strict_seed{seed}_ablation_wd_l{l_val}_{arch}"


def ckpt_path(arch: str, l_val: int, seed: int) -> Path:
    suffix = build_suffix(arch, l_val, seed)
    return ROOT / "mnist-checkpoints" / f"{arch}_L[{l_val}]_{suffix}.pth"


def iter_jobs(model_type: str) -> list[dict]:
    jobs = []
    if model_type in ("fc", "all"):
        for h in FC_H_LIST:
            jobs.append(
                {
                    "family": "fc3",
                    "arch": fc_arch(h),
                    "size_label": f"h{h}",
                    "hidden_size": h,
                    "c1": "",
                    "c2": "",
                }
            )
    if model_type in ("cnn", "all"):
        for c1, c2 in CNN_VARIANTS:
            jobs.append(
                {
                    "family": "cnn2",
                    "arch": cnn_arch(c1, c2),
                    "size_label": cnn_label(c1, c2),
                    "hidden_size": "",
                    "c1": c1,
                    "c2": c2,
                }
            )
    return jobs


def normalize_row(row: dict) -> dict:
    out = {
        "family": row["family"],
        "arch": row["arch"],
        "size_label": row["size_label"],
        "regularizer": row["regularizer"],
        "L": int(row["L"]),
        "T": int(row["T"]),
        "seed": int(row["seed"]),
        "if_mode": row["if_mode"],
        "acc": row["acc"],
        "checkpoint": row["checkpoint"],
    }
    if row.get("hidden_size", "") != "":
        out["hidden_size"] = int(row["hidden_size"])
    else:
        out["hidden_size"] = ""
    if row.get("c1", "") != "":
        out["c1"] = int(row["c1"])
        out["c2"] = int(row["c2"])
    else:
        out["c1"] = ""
        out["c2"] = ""
    return out


def load_existing_raw() -> dict:
    if not RAW_CSV.exists():
        return {}
    rows = {}
    with RAW_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            norm = normalize_row(row)
            key = (norm["arch"], norm["L"], norm["seed"])
            rows[key] = norm
    return rows


def save_raw(rows: list[dict]) -> None:
    fieldnames = [
        "family", "arch", "size_label", "hidden_size", "c1", "c2",
        "regularizer", "L", "T", "seed", "if_mode", "acc", "checkpoint",
    ]
    norm_rows = [normalize_row(r) for r in rows]
    norm_rows.sort(
        key=lambda r: (r["family"], r["size_label"], r["L"], r["seed"])
    )
    with RAW_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(norm_rows)


def train_one(arch: str, l_val: int, seed: int, retrain: bool) -> Path:
    ckpt = ckpt_path(arch, l_val, seed)
    if retrain and ckpt.exists():
        ckpt.unlink()
        print(f"[RETRAIN] removed {ckpt.name}", flush=True)
    if ckpt.exists():
        print(f"[SKIP TRAIN] {ckpt.name}", flush=True)
        return ckpt

    suffix = build_suffix(arch, l_val, seed)
    cmd = [
        sys.executable,
        str(ROOT / "main_train.py"),
        "-data", "mnist",
        "-arch", arch,
        "-L", str(l_val),
        "--epochs", str(EPOCHS),
        "-j", "0",
        "-b", str(BATCH),
        "--seed", str(seed),
        "--device", "auto",
        "--time", "0",
        "--spike_schedule", SPIKE_SCHEDULE,
        "--regularizer", "weight_decay",
        "--weight_decay", "5e-4",
        "--reg_coeff", "1.0",
        "--suffix", suffix,
    ]
    print(f"[TRAIN] {arch} L={l_val} seed={seed} T=0(ANN)", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
    print(f"[TRAIN DONE] {ckpt.name}", flush=True)
    return ckpt


def eval_acc(
    arch: str, l_val: int, seed: int, ckpt: Path, device
) -> float:
    seed_all(seed)
    _, test_loader = datapool("mnist", BATCH, num_workers=0, pin_memory=False)
    model = modelpool(arch, "mnist")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    model.set_L(l_val)
    model.set_T(T_ANN)
    model.set_mode(IF_MODE)
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule(SPIKE_SCHEDULE)
    if hasattr(model, "set_first_layer_input_noise_sigma"):
        model.set_first_layer_input_noise_sigma(0.0)
    return float(val(model, test_loader, T=T_ANN, device=device, verbose=False))


def aggregate_rows(raw_rows: list[dict]) -> list[dict]:
    bucket: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for r in raw_rows:
        bucket[(r["arch"], r["size_label"], int(r["L"]))].append(float(r["acc"]))

    mean_rows = []
    for (arch, size_label, l_val), vals in sorted(bucket.items()):
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        sample = next(r for r in raw_rows if r["arch"] == arch and int(r["L"]) == l_val)
        mean_rows.append(
            {
                "family": sample["family"],
                "arch": arch,
                "size_label": size_label,
                "hidden_size": sample.get("hidden_size", ""),
                "c1": sample.get("c1", ""),
                "c2": sample.get("c2", ""),
                "regularizer": "weight_decay",
                "L": l_val,
                "T": T_ANN,
                "if_mode": IF_MODE,
                "acc_mean": f"{mean:.6f}",
                "acc_std": f"{std:.6f}",
                "n_seeds": len(vals),
            }
        )
    return mean_rows


def write_l2_l16_summary(mean_rows: list[dict]) -> None:
    by_arch: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in mean_rows:
        by_arch[r["arch"]][int(r["L"])] = r

    summary = []
    for arch in sorted(by_arch):
        l2 = by_arch[arch].get(2)
        l16 = by_arch[arch].get(16)
        if not l2 or not l16:
            continue
        diff = float(l16["acc_mean"]) - float(l2["acc_mean"])
        summary.append(
            {
                "family": l2["family"],
                "arch": arch,
                "size_label": l2["size_label"],
                "L2_acc_mean": l2["acc_mean"],
                "L2_acc_std": l2["acc_std"],
                "L16_acc_mean": l16["acc_mean"],
                "L16_acc_std": l16["acc_std"],
                "diff_L16_minus_L2_mean": f"{diff:.6f}",
            }
        )

    with L2_L16_SUMMARY_CSV.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "family", "arch", "size_label",
                "L2_acc_mean", "L2_acc_std",
                "L16_acc_mean", "L16_acc_std",
                "diff_L16_minus_L2_mean",
            ],
        )
        w.writeheader()
        w.writerows(summary)


def finalize(raw_rows: list[dict]) -> None:
    mean_rows = aggregate_rows(raw_rows)
    with MEAN_STD_CSV.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "family", "arch", "size_label", "hidden_size", "c1", "c2",
                "regularizer", "L", "T", "if_mode", "acc_mean", "acc_std", "n_seeds",
            ],
        )
        w.writeheader()
        w.writerows(mean_rows)
    write_l2_l16_summary(mean_rows)
    print(f"[DONE] raw: {RAW_CSV}", flush=True)
    print(f"[DONE] mean_std: {MEAN_STD_CSV}", flush=True)
    print(f"[DONE] L2 vs L16: {L2_L16_SUMMARY_CSV}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MNIST ANN (T=0) acc for FC3/CNN2 at L=2,4,8,16,32"
    )
    p.add_argument(
        "--model-type",
        choices=["fc", "cnn", "all"],
        default="all",
        help="fc | cnn | all（默认 all）",
    )
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="仅从已有 raw CSV 重算 mean±std",
    )
    p.add_argument(
        "--retrain",
        action="store_true",
        help="删除已有 checkpoint 后重新训练",
    )
    p.add_argument(
        "--force-test",
        action="store_true",
        help="忽略已有 raw 记录，重新评测 T=0",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    existing = load_existing_raw()
    raw_rows = list(existing.values())

    if args.aggregate_only:
        if not raw_rows:
            raise SystemExit(f"raw CSV 不存在或为空: {RAW_CSV}")
        finalize(raw_rows)
        return

    device = get_torch_device("auto")
    print(f"[DEVICE] {device}", flush=True)
    print(
        f"[CONFIG] T={T_ANN} (ANN), L={L_LIST}, seeds={SEEDS}, model_type={args.model_type}",
        flush=True,
    )

    for job in iter_jobs(args.model_type):
        arch = job["arch"]
        for l_val in L_LIST:
            for seed in SEEDS:
                ckpt = train_one(arch, l_val, seed, retrain=args.retrain)
                key = (arch, l_val, seed)
                if key in existing and not args.force_test:
                    print(
                        f"[SKIP TEST] {job['size_label']} L={l_val} T=0 seed={seed}",
                        flush=True,
                    )
                    continue
                acc = eval_acc(arch, l_val, seed, ckpt, device)
                row = {
                    "family": job["family"],
                    "arch": arch,
                    "size_label": job["size_label"],
                    "hidden_size": job["hidden_size"],
                    "c1": job["c1"],
                    "c2": job["c2"],
                    "regularizer": "weight_decay",
                    "L": l_val,
                    "T": T_ANN,
                    "seed": seed,
                    "if_mode": IF_MODE,
                    "acc": f"{acc:.6f}",
                    "checkpoint": str(ckpt.relative_to(ROOT)),
                }
                if key in existing:
                    raw_rows = [r for r in raw_rows if not (
                        r["arch"] == arch and int(r["L"]) == l_val and int(r["seed"]) == seed
                    )]
                raw_rows.append(row)
                existing[key] = row
                save_raw(raw_rows)
                print(
                    f"[TEST] {job['size_label']} L={l_val} T=0 seed={seed} acc={acc:.3f}",
                    flush=True,
                )

    finalize(raw_rows)


if __name__ == "__main__":
    main()
