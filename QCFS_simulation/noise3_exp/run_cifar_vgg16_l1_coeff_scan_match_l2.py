#!/usr/bin/env python3
"""
CIFAR VGG16：扫描 L1 系数，按干净准确率（sigma=0）对齐 L2（wd=5e-4）。

流程：
  1) 对每个 --l1-rcs 训练/测试 L1（可复用已有同 suffix checkpoint）
  2) 可选同时跑一遍 L2 作为参考，或从 --l2-ref-mean-std 读取已有 L2 曲线
  3) 输出 summary CSV，并打印与 L2 干净准确率最接近的 L1 系数

用法：
  python noise3_exp/run_cifar_vgg16_l1_coeff_scan_match_l2.py \\
      --dataset cifar10 \\
      --l1-rcs 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 \\
      --first-layer-noise-position post_input_if \\
      --out-dir ../important_results/cifar10_vgg16_l1_coeff_scan_match_l2_post_input_if \\
      --l2-ref-mean-std ../important_results/cifar10_vgg16_three_regs_l1_l2_mne_post_input_if/cifar10_vgg16_strict_seed_three_regs_noise_sweep_mean_std.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "noise3_exp" / "run_cifar_vgg16_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16.py"
DEFAULT_L1_RCS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan L1 coeffs to match L2 clean accuracy")
    p.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    p.add_argument("--l1-rcs", type=float, nargs="+", default=DEFAULT_L1_RCS)
    p.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    p.add_argument(
        "--first-layer-noise-position",
        choices=["post_input_if", "pre_input_if", "input_image"],
        default="post_input_if",
    )
    p.add_argument("--out-dir", required=True, help="扫描结果总目录")
    p.add_argument(
        "--l2-ref-mean-std",
        default=None,
        help="已有 L2 mean_std CSV（含 method=weight_decay）；提供则不重跑 L2",
    )
    p.add_argument(
        "--also-run-l2",
        action="store_true",
        help="若未给 --l2-ref-mean-std，则在 out-dir 内跑一遍 L2 作参考",
    )
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--force-test", action="store_true")
    return p.parse_args()


def run_one_method(
    dataset: str,
    method: str,
    out_dir: Path,
    seeds: list[int],
    noise_pos: str,
    l1_rc: Optional[float] = None,
    retrain: bool = False,
    force_test: bool = False,
) -> Path:
    cmd = [
        sys.executable,
        "-u",
        str(RUNNER),
        "--dataset",
        dataset,
        "--method",
        method,
        "--seeds",
        *[str(s) for s in seeds],
        "--first-layer-noise-position",
        noise_pos,
        "--out-dir",
        str(out_dir),
    ]
    if l1_rc is not None:
        cmd.extend(["--l1-rc", str(l1_rc)])
    if retrain:
        cmd.append("--retrain")
    if force_test:
        cmd.append("--force-test")
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return out_dir / f"{dataset}_vgg16_strict_seed_three_regs_noise_sweep_mean_std.csv"


def read_sigma_stats(mean_std_csv: Path, method: str) -> tuple[float, float, float, float]:
    """Return (acc0_mean, acc0_std, acc1_mean, acc1_std)."""
    rows = []
    with mean_std_csv.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] == method:
                rows.append(r)
    if not rows:
        raise ValueError(f"method={method} not found in {mean_std_csv}")
    by_sigma = {float(r["sigma"]): r for r in rows}
    r0 = by_sigma[min(by_sigma)]
    r1 = by_sigma[max(by_sigma)]
    return (
        float(r0["acc_mean"]),
        float(r0["acc_std"]),
        float(r1["acc_mean"]),
        float(r1["acc_std"]),
    )


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # L2 reference
    if args.l2_ref_mean_std:
        l2_csv = Path(args.l2_ref_mean_std)
        if not l2_csv.exists():
            raise FileNotFoundError(l2_csv)
        print(f"[REF] L2 from {l2_csv}", flush=True)
    elif args.also_run_l2:
        l2_out = out_root / "ref_l2"
        l2_csv = run_one_method(
            args.dataset,
            "weight_decay",
            l2_out,
            args.seeds,
            args.first_layer_noise_position,
            retrain=args.retrain,
            force_test=args.force_test,
        )
    else:
        raise SystemExit("请提供 --l2-ref-mean-std，或加 --also-run-l2")

    l2_a0, l2_s0, l2_a1, l2_s1 = read_sigma_stats(l2_csv, "weight_decay")
    print(
        f"[L2] sigma0={l2_a0:.3f}±{l2_s0:.3f}  sigma1={l2_a1:.3f}±{l2_s1:.3f}",
        flush=True,
    )

    summary_rows = []
    best = None
    for rc in args.l1_rcs:
        tag = f"{rc:.0e}".replace("-", "m").replace("+", "p")
        l1_out = out_root / f"l1_rc_{tag}"
        mean_csv = run_one_method(
            args.dataset,
            "l1",
            l1_out,
            args.seeds,
            args.first_layer_noise_position,
            l1_rc=rc,
            retrain=args.retrain,
            force_test=args.force_test,
        )
        a0, s0, a1, s1 = read_sigma_stats(mean_csv, "l1")
        gap = abs(a0 - l2_a0)
        row = {
            "l1_rc": f"{rc:g}",
            "clean_acc_mean": f"{a0:.6f}",
            "clean_acc_std": f"{s0:.6f}",
            "sigma1_acc_mean": f"{a1:.6f}",
            "sigma1_acc_std": f"{s1:.6f}",
            "l2_clean_acc_mean": f"{l2_a0:.6f}",
            "abs_clean_gap_to_l2": f"{gap:.6f}",
            "out_dir": str(l1_out),
        }
        summary_rows.append(row)
        print(
            f"[L1 rc={rc:g}] clean={a0:.3f}±{s0:.3f}  sigma1={a1:.3f}±{s1:.3f}  "
            f"|clean-L2|={gap:.3f}",
            flush=True,
        )
        if best is None or gap < best[0]:
            best = (gap, rc, a0, a1)

    summary_csv = out_root / f"{args.dataset}_vgg16_l1_coeff_scan_match_l2_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    assert best is not None
    gap, rc, a0, a1 = best
    print("\n=== MATCH RESULT ===", flush=True)
    print(f"L2 clean target: {l2_a0:.3f}", flush=True)
    print(f"Best L1 rc: {rc:g}  (clean={a0:.3f}, |gap|={gap:.3f}, sigma1={a1:.3f})", flush=True)
    print(f"summary: {summary_csv}", flush=True)
    print(
        "\nNext: rerun three-regs with matched L1, e.g.\n"
        f"  python -u noise3_exp/run_cifar_vgg16_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16.py \\\n"
        f"    --dataset {args.dataset} --l1-rc {rc:g} \\\n"
        f"    --first-layer-noise-position {args.first_layer_noise_position} \\\n"
        f"    --out-dir ../important_results/{args.dataset}_vgg16_three_regs_l1_l2_mne_"
        f"{args.first_layer_noise_position}_l1matched\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
