import argparse
import csv
import os
import re
import subprocess
import sys
from typing import List, Tuple

import numpy as np


ACC_PATTERN = re.compile(r"Test acc = ([0-9.]+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="多 seed 运行单个 (L,T) 配置，提取 acc，并可计算相对基准均值的 p 值。"
    )
    p.add_argument("--dataset", default="mnist", type=str)
    p.add_argument("--arch", default="cnn2", type=str)
    p.add_argument("--device", default="mps", type=str)
    p.add_argument("--mode", default="rate_uniform", type=str)
    p.add_argument("--spike_schedule", default="normal", type=str)
    p.add_argument("--batch_size", default=128, type=int)
    p.add_argument("--workers", default=4, type=int)

    p.add_argument("--run_name", default="LxTy", type=str)
    p.add_argument("--L", required=True, type=int)
    p.add_argument("--T", required=True, type=int)

    p.add_argument("--num_runs", default=10, type=int, help="重复次数")
    p.add_argument("--seed_start", default=44, type=int, help="起始 seed")
    p.add_argument(
        "--num_permutations",
        default=10000,
        type=int,
        help="符号翻转检验次数（越大越稳定）",
    )
    p.add_argument(
        "--acc_null_mean",
        default=None,
        type=float,
        help="可选：设定基准均值（如 95.0），计算两侧 p 值；不设则仅统计 acc。",
    )
    p.add_argument(
        "--output_csv",
        default="Noise_exp/acc_pvalue_summary.csv",
        type=str,
        help="输出 CSV 路径（相对 QCFS_simulation 目录）",
    )
    p.add_argument(
        "--python_exec",
        default=sys.executable,
        type=str,
        help="用于调用 main_test.py 的 Python 解释器",
    )
    return p.parse_args()


def run_one(
    main_test_path: str,
    python_exec: str,
    dataset: str,
    arch: str,
    device: str,
    mode: str,
    spike_schedule: str,
    batch_size: int,
    workers: int,
    L: int,
    T: int,
    seed: int,
) -> float:
    cmd = [
        python_exec,
        main_test_path,
        "-data",
        dataset,
        "-arch",
        arch,
        "-dev",
        device,
        "-b",
        str(batch_size),
        "-j",
        str(workers),
        "-L",
        str(L),
        "-T",
        str(T),
        "--mode",
        mode,
        "--spike_schedule",
        spike_schedule,
        "--seed",
        str(seed),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    text = proc.stdout
    hits = ACC_PATTERN.findall(text)
    if not hits:
        raise RuntimeError(f"seed={seed}, L={L}, T={T} 未解析到 Test acc")
    return float(hits[-1])


def sign_flip_pvalue_against_mean(
    x: np.ndarray,
    null_mean: float,
    num_permutations: int,
    rng: np.random.Generator,
) -> float:
    centered = x - float(null_mean)
    observed = abs(centered.mean())
    count = 0
    for _ in range(num_permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=centered.shape[0])
        diff = abs((centered * signs).mean())
        if diff >= observed:
            count += 1
    return (count + 1) / (num_permutations + 1)


def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    m = float(arr.mean())
    s = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return m, s


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_test_path = os.path.join(script_dir, "main_test.py")
    if not os.path.exists(main_test_path):
        raise FileNotFoundError(f"找不到 main_test.py: {main_test_path}")

    out_csv = args.output_csv
    if not os.path.isabs(out_csv):
        out_csv = os.path.join(script_dir, out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    seeds = [args.seed_start + i for i in range(args.num_runs)]
    print(f"Seeds: {seeds}")
    print(f"{args.run_name}: L={args.L}, T={args.T}")

    acc_values: List[float] = []
    for idx, seed in enumerate(seeds, 1):
        print(f"[{idx}/{len(seeds)}] seed={seed} running {args.run_name}")
        acc = run_one(
            main_test_path=main_test_path,
            python_exec=args.python_exec,
            dataset=args.dataset,
            arch=args.arch,
            device=args.device,
            mode=args.mode,
            spike_schedule=args.spike_schedule,
            batch_size=args.batch_size,
            workers=args.workers,
            L=args.L,
            T=args.T,
            seed=seed,
        )
        print(f"  acc={acc:.3f}")
        acc_values.append(acc)

    arr = np.array(acc_values, dtype=np.float64)
    mean_acc, std_acc = mean_std(acc_values)
    pval = ""
    if args.acc_null_mean is not None:
        rng = np.random.default_rng(args.seed_start)
        pval = sign_flip_pvalue_against_mean(
            arr, args.acc_null_mean, args.num_permutations, rng
        )

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed",
                "acc",
            ]
        )
        for s, a in zip(seeds, acc_values):
            writer.writerow([s, a])

        writer.writerow([])
        writer.writerow(["metric", "value"])
        writer.writerow(["mean_acc", mean_acc])
        writer.writerow(["std_acc", std_acc])
        writer.writerow(["acc_null_mean", args.acc_null_mean if args.acc_null_mean is not None else ""])
        writer.writerow(["p_value_sign_flip_two_sided", pval])
        writer.writerow(["num_runs", args.num_runs])
        writer.writerow(["num_permutations", args.num_permutations])
        writer.writerow(["dataset", args.dataset])
        writer.writerow(["arch", args.arch])
        writer.writerow(["L", args.L])
        writer.writerow(["T", args.T])
        writer.writerow(["run_name", args.run_name])
        writer.writerow(["mode", args.mode])
        writer.writerow(["spike_schedule", args.spike_schedule])
        writer.writerow(["device", args.device])

    print("\n完成。")
    print(f"输出 CSV: {out_csv}")
    if args.acc_null_mean is None:
        print(f"mean_acc={mean_acc:.4f}, std_acc={std_acc:.4f} (未计算 p 值)")
    else:
        print(
            f"mean_acc={mean_acc:.4f}, std_acc={std_acc:.4f}, "
            f"p={float(pval):.6f} (vs null_mean={args.acc_null_mean})"
        )


if __name__ == "__main__":
    main()
