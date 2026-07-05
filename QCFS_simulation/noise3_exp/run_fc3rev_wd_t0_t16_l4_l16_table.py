import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models import modelpool
from Preprocess import datapool
from utils import get_torch_device, seed_all, val


def ckpt_path(arch: str, l_val: int, seed: int) -> Path:
    suffix = f"strict_seed{seed}_ablation_wd_l{l_val}_{arch}"
    return PROJECT_ROOT / "mnist-checkpoints" / f"{arch}_L[{l_val}]_{suffix}.pth"


def train_one(arch: str, l_val: int, seed: int, epochs: int, retrain: bool) -> Path:
    ckpt = ckpt_path(arch, l_val, seed)
    if retrain and ckpt.exists():
        ckpt.unlink()
        print(f"[RETRAIN] removed {ckpt.name}", flush=True)
    if ckpt.exists():
        print(f"[SKIP TRAIN] {ckpt.name}", flush=True)
        return ckpt

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "main_train.py"),
        "-data",
        "mnist",
        "-arch",
        arch,
        "-L",
        str(l_val),
        "--epochs",
        str(epochs),
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
        "weight_decay",
        "--weight_decay",
        "5e-4",
        "--reg_coeff",
        "1.0",
        "--suffix",
        f"strict_seed{seed}_ablation_wd_l{l_val}_{arch}",
    ]
    print(f"[TRAIN] {arch} L={l_val} seed={seed} epochs={epochs}", flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
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
    model.set_mode("normal")
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule("normal")
    if hasattr(model, "set_first_layer_input_noise_sigma"):
        model.set_first_layer_input_noise_sigma(0.0)
    return float(val(model, test_loader, T=t_val, device=device, verbose=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FC3rev (2h->h) h8 multi-seed eval for T0/T16 at L4/L16")
    p.add_argument("--h", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    p.add_argument("--l-list", type=int, nargs="+", default=[4, 16])
    p.add_argument("--t-list", type=int, nargs="+", default=[0, 16])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT.parent / "important results" / "new_fc3"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    arch = f"fc3rev_h{args.h}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = out_dir / f"{arch}_wd_t0_t16_l4_l16_raw.csv"
    mean_csv = out_dir / f"{arch}_wd_t0_t16_l4_l16_mean_std.csv"
    merged_csv = out_dir / f"{arch}_wd_t0_t16_l4_l16_merged.csv"

    device = get_torch_device(args.device)
    print(f"[DEVICE] {device}", flush=True)

    raw_rows = []
    for l_val in args.l_list:
        for seed in args.seeds:
            ckpt = train_one(arch, l_val, seed, args.epochs, args.retrain)
            for t_val in args.t_list:
                acc = eval_acc(arch, l_val, t_val, seed, ckpt, device)
                row = {
                    "arch": arch,
                    "hidden_size": args.h,
                    "regularizer": "weight_decay",
                    "if_mode": "normal",
                    "L": l_val,
                    "T": t_val,
                    "seed": seed,
                    "acc": f"{acc:.6f}",
                    "checkpoint": str(ckpt.relative_to(PROJECT_ROOT)),
                }
                raw_rows.append(row)
                print(f"[EVAL] {arch} L={l_val} T={t_val} seed={seed} acc={acc:.3f}", flush=True)

    with raw_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "arch",
                "hidden_size",
                "regularizer",
                "if_mode",
                "L",
                "T",
                "seed",
                "acc",
                "checkpoint",
            ],
        )
        w.writeheader()
        w.writerows(raw_rows)

    bucket = {}
    for r in raw_rows:
        key = (int(r["L"]), int(r["T"]))
        bucket.setdefault(key, []).append(float(r["acc"]))

    mean_rows = []
    for (l_val, t_val), vals in sorted(bucket.items()):
        mean_rows.append(
            {
                "arch": arch,
                "hidden_size": args.h,
                "L": l_val,
                "T": t_val,
                "acc_mean": f"{statistics.mean(vals):.6f}",
                "acc_std": f"{(statistics.stdev(vals) if len(vals) > 1 else 0.0):.6f}",
                "n_seeds": len(vals),
            }
        )
    with mean_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["arch", "hidden_size", "L", "T", "acc_mean", "acc_std", "n_seeds"],
        )
        w.writeheader()
        w.writerows(mean_rows)

    def pick(l_val: int, t_val: int):
        return next((r for r in mean_rows if int(r["L"]) == l_val and int(r["T"]) == t_val), None)

    r_t0_l4 = pick(4, 0)
    r_t16_l4 = pick(4, 16)
    r_t0_l16 = pick(16, 0)
    r_t16_l16 = pick(16, 16)
    merged = {
        "arch": arch,
        "hidden_size": args.h,
        "T0_L4_mean": r_t0_l4["acc_mean"],
        "T0_L4_std": r_t0_l4["acc_std"],
        "T16_L4_mean": r_t16_l4["acc_mean"],
        "T16_L4_std": r_t16_l4["acc_std"],
        "Gap_L4_T0_minus_T16": f"{(float(r_t0_l4['acc_mean']) - float(r_t16_l4['acc_mean'])):.6f}",
        "T0_L16_mean": r_t0_l16["acc_mean"],
        "T0_L16_std": r_t0_l16["acc_std"],
        "T16_L16_mean": r_t16_l16["acc_mean"],
        "T16_L16_std": r_t16_l16["acc_std"],
        "Gap_L16_T0_minus_T16": f"{(float(r_t0_l16['acc_mean']) - float(r_t16_l16['acc_mean'])):.6f}",
    }
    with merged_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "arch",
                "hidden_size",
                "T0_L4_mean",
                "T0_L4_std",
                "T16_L4_mean",
                "T16_L4_std",
                "Gap_L4_T0_minus_T16",
                "T0_L16_mean",
                "T0_L16_std",
                "T16_L16_mean",
                "T16_L16_std",
                "Gap_L16_T0_minus_T16",
            ],
        )
        w.writeheader()
        w.writerow(merged)

    print(f"[DONE] raw: {raw_csv}", flush=True)
    print(f"[DONE] mean_std: {mean_csv}", flush=True)
    print(f"[DONE] merged: {merged_csv}", flush=True)


if __name__ == "__main__":
    main()
