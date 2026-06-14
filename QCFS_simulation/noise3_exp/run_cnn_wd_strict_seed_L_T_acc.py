"""
MNIST CNN2 多 seed weight_decay：扫 L×T 测试精度，输出 mean±std 与 LaTeX 表格。

模型：cnn2_c2_c4 / cnn2_c4_c8 / cnn2_c8_c16 / cnn2_c16_c32
设置：normal 脉冲发放，wd=5e-4，5 seeds (40–44)

用法：
  python noise3_exp/run_cnn_wd_strict_seed_L_T_acc.py
  python noise3_exp/run_cnn_wd_strict_seed_L_T_acc.py --latex-only
  bash noise3_exp/RUN_cnn_wd_strict_seed_L_T_acc.sh
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

OUT = ROOT / "noise3_exp" / "cnn_wd_strict_seed_normal_L_T_acc"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [40, 41, 42, 43, 44]
CNN_VARIANTS = [(2, 4), (4, 8), (8, 16), (16, 32)]
L_LIST = [2, 4, 8, 16, 32]
T_LIST = [2, 4, 8, 16, 32]
TABLE_L_LIST = [4, 16]
IF_MODE = "normal"
SPIKE_SCHEDULE = "normal"

RAW_CSV = OUT / "cnn_wd_strict_seed_normal_L_T_acc_raw.csv"
MEAN_STD_CSV = OUT / "cnn_wd_strict_seed_normal_L_T_acc_mean_std.csv"
LATEX_TEX = OUT / "cnn_wd_strict_seed_normal_L_T_acc_table.tex"


def arch_name(c1: int, c2: int) -> str:
    return f"cnn2_c{c1}_c{c2}"


def size_label(c1: int, c2: int) -> str:
    return f"c{c1}c{c2}"


def build_suffix(arch: str, l_val: int, seed: int) -> str:
    return f"strict_seed{seed}_ablation_wd_l{l_val}_{arch}"


def ckpt_path(arch: str, l_val: int, seed: int) -> Path:
    suffix = build_suffix(arch, l_val, seed)
    return ROOT / "mnist-checkpoints" / f"{arch}_L[{l_val}]_{suffix}.pth"


def normalize_row(row: dict) -> dict:
    c1, c2 = map(int, row["c1"]), int(row["c2"])
    return {
        "arch": row["arch"],
        "c1": c1,
        "c2": c2,
        "size_label": row["size_label"],
        "regularizer": row["regularizer"],
        "L": int(row["L"]),
        "T": int(row["T"]),
        "seed": int(row["seed"]),
        "if_mode": row["if_mode"],
        "acc": row["acc"],
        "checkpoint": row["checkpoint"],
    }


def load_existing_raw() -> dict:
    if not RAW_CSV.exists():
        return {}
    rows = {}
    with RAW_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            norm = normalize_row(row)
            key = (norm["arch"], norm["L"], norm["T"], norm["seed"])
            rows[key] = norm
    return rows


def save_raw(rows: list[dict]) -> None:
    fieldnames = [
        "arch", "c1", "c2", "size_label", "regularizer", "L", "T", "seed",
        "if_mode", "acc", "checkpoint",
    ]
    norm_rows = [normalize_row(r) for r in rows]
    norm_rows.sort(key=lambda r: (r["c1"], r["c2"], r["L"], r["T"], r["seed"]))
    with RAW_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(norm_rows)


def train_one(arch: str, l_val: int, seed: int) -> Path:
    ckpt = ckpt_path(arch, l_val, seed)
    if ckpt.exists():
        print(f"[SKIP TRAIN] {ckpt.name}", flush=True)
        return ckpt

    suffix = build_suffix(arch, l_val, seed)
    cmd = [
        sys.executable,
        str(ROOT / "main_train.py"),
        "-data",
        "mnist",
        "-arch",
        arch,
        "-L",
        str(l_val),
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
        SPIKE_SCHEDULE,
        "--regularizer",
        "weight_decay",
        "--weight_decay",
        "5e-4",
        "--reg_coeff",
        "1.0",
        "--suffix",
        suffix,
    ]
    print(f"[TRAIN] {arch} L={l_val} seed={seed}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
    print(f"[TRAIN DONE] {ckpt.name}", flush=True)
    return ckpt


def eval_acc(arch: str, l_val: int, t_val: int, seed: int, ckpt: Path, device) -> float:
    seed_all(seed)
    _, test_loader = datapool("mnist", 128, num_workers=0, pin_memory=False)
    model = modelpool(arch, "mnist")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    model.set_L(l_val)
    model.set_T(t_val)
    model.set_mode(IF_MODE)
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule(SPIKE_SCHEDULE)
    if hasattr(model, "set_first_layer_input_noise_sigma"):
        model.set_first_layer_input_noise_sigma(0.0)
    return float(val(model, test_loader, T=t_val, device=device, verbose=False))


def aggregate_rows(raw_rows: list[dict]) -> list[dict]:
    bucket: dict[tuple[str, int, int, int, int], list[float]] = defaultdict(list)
    for r in raw_rows:
        bucket[(r["arch"], r["c1"], r["c2"], int(r["L"]), int(r["T"]))].append(float(r["acc"]))

    mean_rows = []
    for (arch, c1, c2, l_val, t_val), vals in sorted(bucket.items()):
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        mean_rows.append(
            {
                "arch": arch,
                "c1": c1,
                "c2": c2,
                "size_label": size_label(c1, c2),
                "regularizer": "weight_decay",
                "L": l_val,
                "T": t_val,
                "if_mode": IF_MODE,
                "acc_mean": f"{mean:.6f}",
                "acc_std": f"{std:.6f}",
                "n_seeds": len(vals),
            }
        )
    return mean_rows


def cell(mean_rows: list[dict], c1: int, c2: int, l_val: int, t_val: int) -> str:
    label = size_label(c1, c2)
    for r in mean_rows:
        if (
            r["size_label"] == label
            and int(r["L"]) == l_val
            and int(r["T"]) == t_val
        ):
            return f"${float(r['acc_mean']):.2f} \\pm {float(r['acc_std']):.2f}$"
    return "--"


def write_latex_table(mean_rows: list[dict]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{MNIST CNN weight decay: test accuracy (\%, mean $\pm$ std, 5 seeds, normal mode).}",
        r"\label{tab:cnn_wd_L_T_acc}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"CNN size & Quantization ($L$) & $T=2$ & $T=4$ & $T=8$ & $T=16$ & $T=32$ \\",
        r"\midrule",
    ]
    for vi, (c1, c2) in enumerate(CNN_VARIANTS):
        for li, l_val in enumerate(TABLE_L_LIST):
            size_tex = f"${size_label(c1, c2)}$" if li == 0 else ""
            vals = " & ".join(cell(mean_rows, c1, c2, l_val, t) for t in T_LIST)
            lines.append(f"{size_tex} & {l_val} & {vals} \\\\")
        if vi < len(CNN_VARIANTS) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    LATEX_TEX.write_text("\n".join(lines) + "\n")
    print(f"[LATEX] saved {LATEX_TEX}", flush=True)


def finalize(raw_rows: list[dict]) -> None:
    mean_rows = aggregate_rows(raw_rows)
    with MEAN_STD_CSV.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "arch", "c1", "c2", "size_label", "regularizer", "L", "T",
                "if_mode", "acc_mean", "acc_std", "n_seeds",
            ],
        )
        w.writeheader()
        w.writerows(mean_rows)
    write_latex_table(mean_rows)
    print(f"[DONE] raw: {RAW_CSV}", flush=True)
    print(f"[DONE] mean_std: {MEAN_STD_CSV}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MNIST CNN wd strict-seed L×T accuracy")
    p.add_argument(
        "--latex-only",
        action="store_true",
        help="仅从已有 raw CSV 重算 mean±std 并写 LaTeX",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    existing = load_existing_raw()
    raw_rows = list(existing.values())

    if args.latex_only:
        if not raw_rows:
            raise SystemExit(f"raw CSV 不存在或为空: {RAW_CSV}")
        finalize(raw_rows)
        return

    device = get_torch_device("auto")
    print(f"[DEVICE] {device}", flush=True)

    for c1, c2 in CNN_VARIANTS:
        arch = arch_name(c1, c2)
        label = size_label(c1, c2)
        for l_val in L_LIST:
            for seed in SEEDS:
                ckpt = train_one(arch, l_val, seed)
                for t_val in T_LIST:
                    key = (arch, l_val, t_val, seed)
                    if key in existing:
                        print(
                            f"[SKIP TEST] {label} L={l_val} T={t_val} seed={seed}",
                            flush=True,
                        )
                        continue
                    acc = eval_acc(arch, l_val, t_val, seed, ckpt, device)
                    row = {
                        "arch": arch,
                        "c1": c1,
                        "c2": c2,
                        "size_label": label,
                        "regularizer": "weight_decay",
                        "L": l_val,
                        "T": t_val,
                        "seed": seed,
                        "if_mode": IF_MODE,
                        "acc": f"{acc:.6f}",
                        "checkpoint": str(ckpt.relative_to(ROOT)),
                    }
                    raw_rows.append(row)
                    existing[key] = row
                    save_raw(raw_rows)
                    print(
                        f"[TEST] {label} L={l_val} T={t_val} seed={seed} acc={acc:.3f}",
                        flush=True,
                    )

    finalize(raw_rows)


if __name__ == "__main__":
    main()
