#!/usr/bin/env python3
"""
Fashion-MNIST: CNN2(c2c4/c4c8/c8c16/c16c32) 三路正则 + rate_uniform 噪声扫描（Python 启动器）。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Fashion-MNIST CNN2 three-reg noise sweep (rate_uniform)"
    )
    p.add_argument(
        "--arch-list",
        nargs="+",
        default=["cnn2_c2_c4", "cnn2_c4_c8", "cnn2_c8_c16", "cnn2_c16_c32"],
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    p.add_argument(
        "--regs",
        nargs="+",
        default=["mne_l2", "weight_decay", "no_regularization"],
        choices=["mne_l2", "weight_decay", "no_regularization"],
    )
    p.add_argument("--epochs", type=int, default=int(os.environ.get("FASHION_CNN2_EPOCHS", "100")))
    p.add_argument("--batch", type=int, default=int(os.environ.get("CNN_BATCH", "128")))
    p.add_argument("--workers", type=int, default=int(os.environ.get("CNN_NUM_WORKERS", "8")))
    p.add_argument("--L", type=int, default=int(os.environ.get("SNN_L", "16")))
    p.add_argument("--T", type=int, default=int(os.environ.get("SNN_T", "16")))
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--first-layer-noise-position",
        choices=["post_input_if", "pre_input_if", "input_image"],
        default="post_input_if",
    )
    p.add_argument("--ckpt-save-mode", choices=["best", "last"], default=os.environ.get("CKPT_SAVE_MODE", "best"))
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--force-test", action="store_true")
    p.add_argument("--out-root", default=os.environ.get("OUT_ROOT"))
    return p.parse_args()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    args = parse_args()
    mnist_root = os.environ.get("MNIST_ROOT", f"{Path.home()}/datasets")
    os.environ["MNIST_ROOT"] = mnist_root
    out_root = Path(
        args.out_root
        or f"../important_results/fashion_mnist_cnn2_three_regs/noise_sweep_rate_uniform_L{args.L}_T{args.T}"
    )

    print(f"[INFO] ROOT={ROOT}")
    print(f"[INFO] MNIST_ROOT={mnist_root}")
    print(f"[INFO] ARCHS={args.arch_list} SEEDS={args.seeds} REGS={args.regs}")
    print(f"[INFO] EPOCHS={args.epochs} BATCH={args.batch} WORKERS={args.workers} L={args.L} T={args.T}")
    print(f"[INFO] CKPT_SAVE_MODE={args.ckpt_save_mode} RETRAIN={args.retrain} FORCE_TEST={args.force_test}")

    for arch in args.arch_list:
        for reg in args.regs:
            for seed in args.seeds:
                if reg == "mne_l2":
                    regularizer, wd, rc = "mne_l2", "0.0", "5e-2"
                    suffix = f"fashion_strict_seed{seed}_ablation_mne_l2_l{args.L}_{arch}_rc5em02"
                elif reg == "weight_decay":
                    regularizer, wd, rc = "weight_decay", "5e-4", "1.0"
                    suffix = f"fashion_strict_seed{seed}_ablation_wd_l{args.L}_{arch}"
                else:
                    regularizer, wd, rc = "weight_decay", "0.0", "1.0"
                    suffix = f"fashion_strict_seed{seed}_ablation_none_l{args.L}_{arch}"

                ckpt = ROOT / "fashion_mnist-checkpoints" / f"{arch}_L[{args.L}]_{suffix}.pth"
                out_dir = out_root / arch / reg / f"seed_{seed}"

                if args.retrain and ckpt.exists():
                    ckpt.unlink()
                if args.force_test and out_dir.exists():
                    for p in out_dir.glob("noise_sweep_matrix_*.csv"):
                        p.unlink()
                    combined = out_dir / "noise_sweep_combined_L_T.csv"
                    if combined.exists():
                        combined.unlink()

                print(f"[TRAIN] {arch} {reg} seed={seed}")
                run(
                    [
                        sys.executable,
                        "-u",
                        str(ROOT / "main_train.py"),
                        "-data",
                        "fashion_mnist",
                        "-arch",
                        arch,
                        "-L",
                        str(args.L),
                        "--epochs",
                        str(args.epochs),
                        "-j",
                        str(args.workers),
                        "-b",
                        str(args.batch),
                        "--seed",
                        str(seed),
                        "--device",
                        args.device,
                        "--time",
                        "0",
                        "--spike_schedule",
                        "normal",
                        "--regularizer",
                        regularizer,
                        "--weight_decay",
                        wd,
                        "--reg_coeff",
                        rc,
                        "--ckpt-save-mode",
                        args.ckpt_save_mode,
                        "--suffix",
                        suffix,
                    ]
                )

                print(f"[TEST] {arch} {reg} seed={seed}")
                run(
                    [
                        sys.executable,
                        "-u",
                        str(ROOT / "main_test.py"),
                        "-data",
                        "fashion_mnist",
                        "-arch",
                        arch,
                        "-L",
                        str(args.L),
                        "-T",
                        str(args.T),
                        "-j",
                        str(args.workers),
                        "-b",
                        str(args.batch),
                        "--seed",
                        str(seed),
                        "--device",
                        args.device,
                        "--mode",
                        "rate_uniform",
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
                        "--first_layer_noise_position",
                        args.first_layer_noise_position,
                        "--noise_output_dir",
                        str(out_dir),
                    ]
                )

    print(f"[DONE] OUT_ROOT={out_root}")


if __name__ == "__main__":
    main()
