import argparse
import csv
import statistics
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
    p = argparse.ArgumentParser(
        description="FC3 weight_decay: evaluate T=0/16 at L=4/16 and export one merged table"
    )
    p.add_argument("--h-list", type=int, nargs="+", default=[16, 128])
    p.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    p.add_argument("--l-list", type=int, nargs="+", default=[4, 16])
    p.add_argument("--t-list", type=int, nargs="+", default=[0, 16])
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--out-dir",
        type=str,
        default="noise3_exp/fc3_wd_strict_seed_normal_T0_T16_L4_L16",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = out_dir / "fc3_wd_t0_t16_l4_l16_raw.csv"
    mean_csv = out_dir / "fc3_wd_t0_t16_l4_l16_mean_std.csv"
    merged_csv = out_dir / "fc3_wd_t0_t16_l4_l16_merged_table.csv"
    merged_md = out_dir / "fc3_wd_t0_t16_l4_l16_merged_table.md"

    device = get_torch_device(args.device)
    print(f"[DEVICE] {device}", flush=True)

    raw_rows = []
    for h in args.h_list:
        arch = f"fc3_h{h}"
        for l_val in args.l_list:
            for seed in args.seeds:
                ckpt = ckpt_path(arch, l_val, seed)
                if not ckpt.exists():
                    print(f"[MISS CKPT] {ckpt}", flush=True)
                    continue
                for t_val in args.t_list:
                    acc = eval_acc(arch, l_val, t_val, seed, ckpt, device)
                    row = {
                        "arch": arch,
                        "hidden_size": h,
                        "regularizer": "weight_decay",
                        "if_mode": "normal",
                        "L": l_val,
                        "T": t_val,
                        "seed": seed,
                        "acc": f"{acc:.6f}",
                        "checkpoint": str(ckpt.relative_to(PROJECT_ROOT)),
                    }
                    raw_rows.append(row)
                    print(
                        f"[EVAL] {arch} L={l_val} T={t_val} seed={seed} acc={acc:.3f}",
                        flush=True,
                    )

    if not raw_rows:
        raise RuntimeError("No rows evaluated. Check checkpoints and arguments.")

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
        key = (r["arch"], int(r["hidden_size"]), int(r["L"]), int(r["T"]))
        bucket.setdefault(key, []).append(float(r["acc"]))

    mean_rows = []
    for (arch, h, l_val, t_val), vals in sorted(bucket.items()):
        mean_rows.append(
            {
                "arch": arch,
                "hidden_size": h,
                "regularizer": "weight_decay",
                "if_mode": "normal",
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
            fieldnames=[
                "arch",
                "hidden_size",
                "regularizer",
                "if_mode",
                "L",
                "T",
                "acc_mean",
                "acc_std",
                "n_seeds",
            ],
        )
        w.writeheader()
        w.writerows(mean_rows)

    merged_rows = []
    for h in sorted(set(int(r["hidden_size"]) for r in mean_rows)):
        arch = f"fc3_h{h}"

        def pick(l_val: int, t_val: int):
            return next(
                (
                    r
                    for r in mean_rows
                    if r["arch"] == arch and int(r["L"]) == l_val and int(r["T"]) == t_val
                ),
                None,
            )

        r_t0_l4 = pick(4, 0)
        r_t16_l4 = pick(4, 16)
        r_t0_l16 = pick(16, 0)
        r_t16_l16 = pick(16, 16)
        if not (r_t0_l4 and r_t16_l4 and r_t0_l16 and r_t16_l16):
            continue

        gap_l4 = float(r_t0_l4["acc_mean"]) - float(r_t16_l4["acc_mean"])
        gap_l16 = float(r_t0_l16["acc_mean"]) - float(r_t16_l16["acc_mean"])
        merged_rows.append(
            {
                "arch": arch,
                "hidden_size": h,
                "T0_L4_mean": r_t0_l4["acc_mean"],
                "T0_L4_std": r_t0_l4["acc_std"],
                "T16_L4_mean": r_t16_l4["acc_mean"],
                "T16_L4_std": r_t16_l4["acc_std"],
                "Gap_L4_T0_minus_T16": f"{gap_l4:.6f}",
                "T0_L16_mean": r_t0_l16["acc_mean"],
                "T0_L16_std": r_t0_l16["acc_std"],
                "T16_L16_mean": r_t16_l16["acc_mean"],
                "T16_L16_std": r_t16_l16["acc_std"],
                "Gap_L16_T0_minus_T16": f"{gap_l16:.6f}",
            }
        )

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
        w.writerows(merged_rows)

    with merged_md.open("w") as f:
        f.write(
            "| arch | hidden_size | T0 L4 | T16 L4 | Gap L4 (T0-T16) | T0 L16 | T16 L16 | Gap L16 (T0-T16) |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in merged_rows:
            f.write(
                "| {arch} | {hidden_size} | {T0_L4_mean}±{T0_L4_std} | "
                "{T16_L4_mean}±{T16_L4_std} | {Gap_L4_T0_minus_T16} | "
                "{T0_L16_mean}±{T0_L16_std} | {T16_L16_mean}±{T16_L16_std} | "
                "{Gap_L16_T0_minus_T16} |\n".format(**r)
            )

    print(f"[DONE] raw: {raw_csv}", flush=True)
    print(f"[DONE] mean_std: {mean_csv}", flush=True)
    print(f"[DONE] merged csv: {merged_csv}", flush=True)
    print(f"[DONE] merged md: {merged_md}", flush=True)


if __name__ == "__main__":
    main()
