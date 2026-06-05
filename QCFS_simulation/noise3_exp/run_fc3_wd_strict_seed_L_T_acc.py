import csv
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models import modelpool
from Preprocess import datapool
from utils import get_torch_device, seed_all, val

OUT = PROJECT_ROOT / "noise3_exp" / "fc3_wd_strict_seed_normal_L_T_acc"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [40, 41, 42, 43, 44]
H_LIST = [4, 8, 16, 32, 64, 128]
L_LIST = [2, 4, 8, 16, 32]
T_LIST = [2, 4, 8, 16, 32]
IF_MODE = "normal"
SPIKE_SCHEDULE = "normal"

RAW_CSV = OUT / "fc3_wd_strict_seed_normal_L_T_acc_raw.csv"
MEAN_STD_CSV = OUT / "fc3_wd_strict_seed_normal_L_T_acc_mean_std.csv"
L2_L16_CSV = OUT / "fc3_wd_strict_seed_normal_L2_vs_L16_acc_summary.csv"
L2_L16_DIFF_CSV = OUT / "fc3_wd_strict_seed_normal_L2_vs_L16_acc_diff.csv"


def build_suffix(arch: str, l_val: int, seed: int) -> str:
    return f"strict_seed{seed}_ablation_wd_l{l_val}_{arch}"


def ckpt_path(arch: str, l_val: int, seed: int) -> Path:
    suffix = build_suffix(arch, l_val, seed)
    return PROJECT_ROOT / "mnist-checkpoints" / f"{arch}_L[{l_val}]_{suffix}.pth"


def normalize_row(row: dict) -> dict:
    return {
        "arch": row["arch"],
        "hidden_size": int(row["hidden_size"]),
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
    with RAW_CSV.open() as f:
        for row in csv.DictReader(f):
            norm = normalize_row(row)
            key = (norm["arch"], norm["L"], norm["T"], norm["seed"])
            rows[key] = norm
    return rows


def save_raw(rows: list[dict]) -> None:
    fieldnames = [
        "arch", "hidden_size", "regularizer", "L", "T", "seed",
        "if_mode", "acc", "checkpoint",
    ]
    norm_rows = [normalize_row(r) for r in rows]
    norm_rows.sort(key=lambda r: (r["hidden_size"], r["L"], r["T"], r["seed"]))
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
        sys.executable, str(PROJECT_ROOT / "main_train.py"),
        "-data", "mnist",
        "-arch", arch,
        "-L", str(l_val),
        "--epochs", "100",
        "-j", "0",
        "-b", "128",
        "--seed", str(seed),
        "--device", "auto",
        "--time", "0",
        "--spike_schedule", SPIKE_SCHEDULE,
        "--regularizer", "weight_decay",
        "--weight_decay", "5e-4",
        "--reg_coeff", "1.0",
        "--suffix", suffix,
    ]
    print(f"[TRAIN] {arch} L={l_val} seed={seed}", flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
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
    acc = float(val(model, test_loader, T=t_val, device=device, verbose=False))
    return acc


def aggregate_and_summarize(raw_rows: list[dict]) -> None:
    bucket = defaultdict(list)
    for r in raw_rows:
        bucket[(r["arch"], int(r["hidden_size"]), int(r["L"]), int(r["T"]))].append(float(r["acc"]))

    mean_rows = []
    for (arch, h, l_val, t_val), vals in sorted(bucket.items()):
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        mean_rows.append({
            "arch": arch,
            "hidden_size": h,
            "regularizer": "weight_decay",
            "L": l_val,
            "T": t_val,
            "if_mode": IF_MODE,
            "acc_mean": f"{mean:.6f}",
            "acc_std": f"{std:.6f}",
            "n_seeds": len(vals),
        })

    with MEAN_STD_CSV.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "arch", "hidden_size", "regularizer", "L", "T", "if_mode",
                "acc_mean", "acc_std", "n_seeds",
            ],
        )
        w.writeheader()
        w.writerows(mean_rows)

    summary_rows = []
    diff_rows = []
    for h in H_LIST:
        arch = f"fc3_h{h}"
        for t_val in T_LIST:
            l2 = next((r for r in mean_rows if r["arch"] == arch and int(r["L"]) == 2 and int(r["T"]) == t_val), None)
            l16 = next((r for r in mean_rows if r["arch"] == arch and int(r["L"]) == 16 and int(r["T"]) == t_val), None)
            if not l2 or not l16:
                continue
            diff_mean = float(l16["acc_mean"]) - float(l2["acc_mean"])
            summary_rows.append({
                "arch": arch,
                "hidden_size": h,
                "T": t_val,
                "L2_acc_mean": l2["acc_mean"],
                "L2_acc_std": l2["acc_std"],
                "L16_acc_mean": l16["acc_mean"],
                "L16_acc_std": l16["acc_std"],
                "diff_L16_minus_L2_mean": f"{diff_mean:.6f}",
            })
            diff_rows.append({
                "arch": arch,
                "hidden_size": h,
                "T": t_val,
                "acc_diff_L16_minus_L2": f"{diff_mean:.6f}",
            })

    with L2_L16_CSV.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "arch", "hidden_size", "T",
                "L2_acc_mean", "L2_acc_std", "L16_acc_mean", "L16_acc_std",
                "diff_L16_minus_L2_mean",
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)

    with L2_L16_DIFF_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arch", "hidden_size", "T", "acc_diff_L16_minus_L2"])
        w.writeheader()
        w.writerows(diff_rows)


def main() -> None:
    existing = load_existing_raw()
    raw_rows = list(existing.values())
    device = get_torch_device("auto")
    print(f"[DEVICE] {device}", flush=True)

    for h in H_LIST:
        arch = f"fc3_h{h}"
        for l_val in L_LIST:
            for seed in SEEDS:
                ckpt = train_one(arch, l_val, seed)
                for t_val in T_LIST:
                    key = (arch, l_val, t_val, seed)
                    if key in existing:
                        print(f"[SKIP TEST] {arch} L={l_val} T={t_val} seed={seed}", flush=True)
                        continue
                    acc = eval_acc(arch, l_val, t_val, seed, ckpt, device)
                    row = {
                        "arch": arch,
                        "hidden_size": h,
                        "regularizer": "weight_decay",
                        "L": l_val,
                        "T": t_val,
                        "seed": seed,
                        "if_mode": IF_MODE,
                        "acc": f"{acc:.6f}",
                        "checkpoint": str(ckpt.relative_to(PROJECT_ROOT)),
                    }
                    raw_rows.append(row)
                    existing[key] = row
                    save_raw(raw_rows)
                    print(
                        f"[TEST] {arch} L={l_val} T={t_val} seed={seed} acc={acc:.3f}",
                        flush=True,
                    )

    aggregate_and_summarize(raw_rows)
    print(f"[DONE] raw: {RAW_CSV}", flush=True)
    print(f"[DONE] mean_std: {MEAN_STD_CSV}", flush=True)
    print(f"[DONE] L2 vs L16: {L2_L16_CSV}", flush=True)


if __name__ == "__main__":
    main()
