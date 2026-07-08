import argparse
import csv
import os
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

BATCH = int(os.environ.get("FC3REV_BATCH", "128"))
NUM_WORKERS = int(os.environ.get("FC3REV_NUM_WORKERS", "0"))


def _path_for_csv(path: Path) -> str:
    """Prefer project-relative path; fallback to absolute path."""
    p = path.resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(p)


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


def eval_acc(
    arch: str,
    l_val: int,
    t_val: int,
    seed: int,
    ckpt: Path,
    device,
    if_mode: str = "normal",
) -> float:
    seed_all(seed)
    use_cuda = device.type == "cuda"
    _, test_loader = datapool(
        "mnist", BATCH, num_workers=NUM_WORKERS, pin_memory=use_cuda
    )
    model = modelpool(arch, "mnist")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    model.set_L(l_val)
    model.set_T(t_val)
    model.set_mode(if_mode)
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule("normal")
    if hasattr(model, "set_first_layer_input_noise_sigma"):
        model.set_first_layer_input_noise_sigma(0.0)
    return float(val(model, test_loader, T=t_val, device=device, verbose=False))


def matrix_path(arch: str, seed: int, noise_out_dir: Path) -> Path:
    return (
        noise_out_dir
        / arch
        / f"seed_{seed}"
        / f"noise_sweep_matrix_mnist_{arch}_T16_mode_rate_uniform_schedule_normal_seed_{seed}.csv"
    )


def run_noise_sweep(arch: str, seed: int, ckpt_l16: Path, noise_out_dir: Path, force_noise_test: bool) -> Path:
    mat = matrix_path(arch, seed, noise_out_dir)
    if force_noise_test and mat.exists():
        mat.unlink()
    if mat.exists():
        print(f"[SKIP NOISE] {mat.name}", flush=True)
        return mat

    out_dir = noise_out_dir / arch / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "main_test.py"),
        "-data",
        "mnist",
        "-arch",
        arch,
        "-L",
        "16",
        "-T",
        "16",
        "-j",
        str(NUM_WORKERS),
        "-b",
        str(BATCH),
        "--seed",
        str(seed),
        "--device",
        "auto",
        "--mode",
        "rate_uniform",
        "--spike_schedule",
        "normal",
        "--weights",
        str(ckpt_l16),
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
    print(f"[NOISE] {arch} seed={seed} mode=rate_uniform L=16 T=16", flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    if not mat.exists():
        cands = sorted(out_dir.glob(f"noise_sweep_matrix_*_T16_mode_rate_uniform_schedule_normal_seed_{seed}.csv"))
        if not cands:
            raise FileNotFoundError(f"noise matrix missing: {out_dir}")
        mat = cands[0]
    return mat


def read_matrix(mat: Path):
    with mat.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)
    start = 2 if header[:2] == ["L", "T"] else 1
    sigmas = [float(x) for x in header[start:]]
    accs = [float(x) for x in row[start:]]
    return list(zip(sigmas, accs))


def _load_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _merge_rows_by_arch(existing: list[dict], new_rows: list[dict], archs: set[str]) -> list[dict]:
    kept = [r for r in existing if r["arch"] not in archs]
    kept.extend(new_rows)
    return kept


def _build_merged_rows(clean_mean_rows: list[dict]) -> list[dict]:
    merged_rows = []
    archs = sorted({r["arch"] for r in clean_mean_rows}, key=lambda a: int(a.split("_h")[1]))
    for arch in archs:
        h = int(arch.split("_h")[1])

        def pick(l_val: int, t_val: int):
            return next(
                (
                    r
                    for r in clean_mean_rows
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
        merged_rows.append(
            {
                "arch": arch,
                "hidden_size": h,
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
        )
    return merged_rows


def clean_csv_paths(out_dir: Path, if_mode: str) -> tuple[Path, Path, Path]:
    if if_mode == "normal":
        stem = "fc3rev_h8_h256_wd_t0_t16_l4_l16"
    else:
        stem = "fc3rev_h4_h256_wd_clean_acc_rate_uniform"
    return (
        out_dir / f"{stem}_raw.csv",
        out_dir / f"{stem}_mean_std.csv",
        out_dir / f"{stem}_merged.csv",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FC3rev(2h->h) h4~h256: multi-seed clean T0/T16 (L4/L16) + Gaussian noise sweep"
    )
    p.add_argument("--h-list", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    p.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    p.add_argument("--l-list", type=int, nargs="+", default=[4, 16])
    p.add_argument("--t-list", type=int, nargs="+", default=[0, 16])
    p.add_argument(
        "--if-mode",
        type=str,
        default="normal",
        choices=["normal", "rate_uniform"],
        help="IF mode for clean acc eval (noise sweep always rate_uniform L16 T16)",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--force-noise-test", action="store_true")
    p.add_argument("--skip-noise-sweep", action="store_true")
    p.add_argument("--skip-clean-eval", action="store_true", help="Skip clean T/L acc eval")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT.parent / "important results" / "new_fc3"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    noise_out_dir = out_dir / "noise_sweep_rate_uniform_L16_T16"

    clean_raw_csv, clean_mean_csv, clean_merged_csv = clean_csv_paths(out_dir, args.if_mode)
    noise_raw_csv = out_dir / "fc3rev_h8_h256_wd_noise_sweep_raw.csv"
    noise_mean_csv = out_dir / "fc3rev_h8_h256_wd_noise_sweep_mean_std.csv"

    device = get_torch_device(args.device)
    print(f"[DEVICE] {device}", flush=True)
    print(f"[FC3REV_BATCH] {BATCH}", flush=True)
    print(f"[FC3REV_NUM_WORKERS] {NUM_WORKERS}", flush=True)
    print(f"[IF-MODE] clean eval: {args.if_mode}", flush=True)

    clean_rows = []
    noise_rows = []
    for h in args.h_list:
        arch = f"fc3rev_h{h}"
        ckpt_l16_map = {}
        if not args.skip_clean_eval:
            for l_val in args.l_list:
                for seed in args.seeds:
                    ckpt = train_one(arch, l_val, seed, args.epochs, args.retrain)
                    if l_val == 16:
                        ckpt_l16_map[seed] = ckpt
                    for t_val in args.t_list:
                        acc = eval_acc(
                            arch, l_val, t_val, seed, ckpt, device, if_mode=args.if_mode
                        )
                        clean_rows.append(
                            {
                                "arch": arch,
                                "hidden_size": h,
                                "regularizer": "weight_decay",
                                "if_mode": args.if_mode,
                                "L": l_val,
                                "T": t_val,
                                "seed": seed,
                                "acc": f"{acc:.6f}",
                                "checkpoint": _path_for_csv(ckpt),
                            }
                        )
                        print(
                            f"[EVAL] {arch} mode={args.if_mode} L={l_val} T={t_val} "
                            f"seed={seed} acc={acc:.3f}",
                            flush=True,
                        )
        elif not args.skip_noise_sweep:
            for seed in args.seeds:
                ckpt = ckpt_path(arch, 16, seed)
                if ckpt.exists():
                    ckpt_l16_map[seed] = ckpt

        if not args.skip_noise_sweep:
            for seed in args.seeds:
                ckpt_l16 = ckpt_l16_map.get(seed)
                if ckpt_l16 is None:
                    raise FileNotFoundError(f"Missing L16 checkpoint for {arch} seed={seed}")
                mat = run_noise_sweep(arch, seed, ckpt_l16, noise_out_dir, args.force_noise_test)
                for sigma, acc in read_matrix(mat):
                    noise_rows.append(
                        {
                            "arch": arch,
                            "hidden_size": h,
                            "regularizer": "weight_decay",
                            "if_mode": "rate_uniform",
                            "L": 16,
                            "T": 16,
                            "seed": seed,
                            "sigma": f"{sigma:.2f}",
                            "acc": f"{acc:.6f}",
                            "checkpoint": _path_for_csv(ckpt_l16),
                            "matrix_csv": _path_for_csv(mat),
                        }
                    )

    archs_run = {f"fc3rev_h{h}" for h in args.h_list}
    if not args.skip_clean_eval:
        clean_rows = _merge_rows_by_arch(_load_csv_rows(clean_raw_csv), clean_rows, archs_run)
    else:
        clean_rows = _load_csv_rows(clean_raw_csv)
    if not args.skip_noise_sweep:
        noise_rows = _merge_rows_by_arch(_load_csv_rows(noise_raw_csv), noise_rows, archs_run)

    if not args.skip_clean_eval and clean_rows:
        with clean_raw_csv.open("w", newline="") as f:
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
            w.writerows(clean_rows)

        clean_bucket = defaultdict(list)
        for r in clean_rows:
            clean_bucket[(r["arch"], int(r["hidden_size"]), int(r["L"]), int(r["T"]))].append(
                float(r["acc"])
            )

        clean_mean_rows = []
        for (arch, h, l_val, t_val), vals in sorted(clean_bucket.items()):
            clean_mean_rows.append(
                {
                    "arch": arch,
                    "hidden_size": h,
                    "regularizer": "weight_decay",
                    "if_mode": args.if_mode,
                    "L": l_val,
                    "T": t_val,
                    "acc_mean": f"{statistics.mean(vals):.6f}",
                    "acc_std": f"{(statistics.stdev(vals) if len(vals) > 1 else 0.0):.6f}",
                    "n_seeds": len(vals),
                }
            )
        with clean_mean_csv.open("w", newline="") as f:
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
            w.writerows(clean_mean_rows)

        merged_rows = _build_merged_rows(clean_mean_rows)

        with clean_merged_csv.open("w", newline="") as f:
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

    if noise_rows:
        with noise_raw_csv.open("w", newline="") as f:
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
                    "sigma",
                    "acc",
                    "checkpoint",
                    "matrix_csv",
                ],
            )
            w.writeheader()
            w.writerows(noise_rows)

        noise_bucket = defaultdict(list)
        for r in noise_rows:
            noise_bucket[(r["arch"], int(r["hidden_size"]), float(r["sigma"]))].append(float(r["acc"]))
        noise_mean_rows = []
        for (arch, h, sigma), vals in sorted(noise_bucket.items(), key=lambda x: (x[0][1], x[0][2])):
            noise_mean_rows.append(
                {
                    "arch": arch,
                    "hidden_size": h,
                    "regularizer": "weight_decay",
                    "if_mode": "rate_uniform",
                    "L": 16,
                    "T": 16,
                    "sigma": f"{sigma:.2f}",
                    "acc_mean": f"{statistics.mean(vals):.6f}",
                    "acc_std": f"{(statistics.stdev(vals) if len(vals) > 1 else 0.0):.6f}",
                    "n_seeds": len(vals),
                }
            )
        with noise_mean_csv.open("w", newline="") as f:
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
            w.writerows(noise_mean_rows)

    print(f"[DONE] clean raw: {clean_raw_csv}", flush=True)
    print(f"[DONE] clean mean: {clean_mean_csv}", flush=True)
    print(f"[DONE] clean merged: {clean_merged_csv}", flush=True)
    if noise_rows:
        print(f"[DONE] noise raw: {noise_raw_csv}", flush=True)
        print(f"[DONE] noise mean: {noise_mean_csv}", flush=True)


if __name__ == "__main__":
    main()
