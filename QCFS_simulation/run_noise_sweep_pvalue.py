import argparse
import csv
import os
import re
import subprocess
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np


ACC_PATTERN = re.compile(r"noise_sweep mode=.* sigma=([0-9.]+) acc=([0-9.]+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="多次运行 noise_sweep，并计算两组配置的置换检验 p 值。"
    )
    p.add_argument("--dataset", default="mnist", type=str)
    p.add_argument("--arch", default="cnn2", type=str)
    p.add_argument("--device", default="mps", type=str)
    p.add_argument("--mode", default="rate_uniform", type=str)
    p.add_argument("--spike_schedule", default="normal", type=str)
    p.add_argument("--batch_size", default=128, type=int)
    p.add_argument("--workers", default=4, type=int)

    p.add_argument("--group_a_name", default="A", type=str)
    p.add_argument("--group_a_L", required=True, type=int)
    p.add_argument("--group_a_T", required=True, type=int)
    p.add_argument("--group_b_name", default="B", type=str)
    p.add_argument("--group_b_L", required=True, type=int)
    p.add_argument("--group_b_T", required=True, type=int)

    p.add_argument("--num_runs", default=10, type=int, help="重复次数")
    p.add_argument("--seed_start", default=44, type=int, help="起始 seed")

    p.add_argument("--noise_sigma_start", default=0.0, type=float)
    p.add_argument("--noise_sigma_end", default=1.0, type=float)
    p.add_argument("--noise_sigma_step", default=0.02, type=float)
    p.add_argument("--noise_target_acc", default=90.0, type=float)

    p.add_argument(
        "--noise_output_dir",
        default="Noise_exp/pvalue_runs",
        type=str,
        help="中间 noise_sweep CSV 输出目录",
    )
    p.add_argument(
        "--result_prefix",
        default="noise_sweep_pvalue",
        type=str,
        help="最终统计结果文件名前缀（输出到 noise_output_dir）",
    )
    p.add_argument(
        "--num_permutations",
        default=10000,
        type=int,
        help="置换检验次数（越大越稳定）",
    )
    p.add_argument(
        "--python_exec",
        default=sys.executable,
        type=str,
        help="用于调用 main_test.py 的 Python 解释器",
    )
    return p.parse_args()


def build_sigma_values(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("noise_sigma_step 必须 > 0")
    if end < start:
        raise ValueError("noise_sigma_end 必须 >= noise_sigma_start")
    vals = []
    cur = float(start)
    eps = 1e-12
    while cur <= end + eps:
        vals.append(round(cur, 10))
        cur += step
    return vals


def sigma_key(x: float) -> str:
    return ("%.6f" % float(x)).rstrip("0").rstrip(".") or "0"


def run_one_noise_sweep(
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
    noise_sigma_start: float,
    noise_sigma_end: float,
    noise_sigma_step: float,
    noise_target_acc: float,
    noise_output_dir: str,
) -> Tuple[List[Tuple[float, float]], str]:
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
        "--noise_sweep",
        "--noise_sigma_start",
        str(noise_sigma_start),
        "--noise_sigma_end",
        str(noise_sigma_end),
        "--noise_sigma_step",
        str(noise_sigma_step),
        "--noise_target_acc",
        str(noise_target_acc),
        "--noise_output_dir",
        noise_output_dir,
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    out_text = proc.stdout

    points: List[Tuple[float, float]] = []
    for line in out_text.splitlines():
        m = ACC_PATTERN.search(line)
        if m:
            points.append((float(m.group(1)), float(m.group(2))))

    if not points:
        raise RuntimeError("未在输出中解析到 noise_sweep 结果，请检查 main_test.py 输出。")
    return points, out_text


def points_to_vector(
    points: Sequence[Tuple[float, float]], sigma_values: Sequence[float]
) -> np.ndarray:
    mp: Dict[str, float] = {sigma_key(s): float(acc) for s, acc in points}
    return np.array([mp[sigma_key(s)] for s in sigma_values], dtype=np.float64)


def permutation_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    num_permutations: int,
    rng: np.random.Generator,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    observed = abs(x.mean() - y.mean())
    pooled = np.concatenate([x, y])
    nx = len(x)
    count = 0
    for _ in range(num_permutations):
        perm = rng.permutation(pooled)
        diff = abs(perm[:nx].mean() - perm[nx:].mean())
        if diff >= observed:
            count += 1
    return (count + 1) / (num_permutations + 1)


def auc_trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_test_path = os.path.join(script_dir, "main_test.py")
    if not os.path.exists(main_test_path):
        raise FileNotFoundError(f"找不到 main_test.py: {main_test_path}")

    os.makedirs(args.noise_output_dir, exist_ok=True)
    sigma_values = build_sigma_values(
        args.noise_sigma_start, args.noise_sigma_end, args.noise_sigma_step
    )
    sigma_arr = np.array(sigma_values, dtype=np.float64)

    seeds = [args.seed_start + i for i in range(args.num_runs)]
    print(f"Seeds: {seeds}")
    print(
        f"Group A={args.group_a_name}(L={args.group_a_L}, T={args.group_a_T}), "
        f"Group B={args.group_b_name}(L={args.group_b_L}, T={args.group_b_T})"
    )

    all_a: List[np.ndarray] = []
    all_b: List[np.ndarray] = []

    for idx, seed in enumerate(seeds, 1):
        print(f"\n[{idx}/{len(seeds)}] seed={seed} - running group A")
        pts_a, _ = run_one_noise_sweep(
            main_test_path=main_test_path,
            python_exec=args.python_exec,
            dataset=args.dataset,
            arch=args.arch,
            device=args.device,
            mode=args.mode,
            spike_schedule=args.spike_schedule,
            batch_size=args.batch_size,
            workers=args.workers,
            L=args.group_a_L,
            T=args.group_a_T,
            seed=seed,
            noise_sigma_start=args.noise_sigma_start,
            noise_sigma_end=args.noise_sigma_end,
            noise_sigma_step=args.noise_sigma_step,
            noise_target_acc=args.noise_target_acc,
            noise_output_dir=args.noise_output_dir,
        )
        vec_a = points_to_vector(pts_a, sigma_values)
        all_a.append(vec_a)

        print(f"[{idx}/{len(seeds)}] seed={seed} - running group B")
        pts_b, _ = run_one_noise_sweep(
            main_test_path=main_test_path,
            python_exec=args.python_exec,
            dataset=args.dataset,
            arch=args.arch,
            device=args.device,
            mode=args.mode,
            spike_schedule=args.spike_schedule,
            batch_size=args.batch_size,
            workers=args.workers,
            L=args.group_b_L,
            T=args.group_b_T,
            seed=seed,
            noise_sigma_start=args.noise_sigma_start,
            noise_sigma_end=args.noise_sigma_end,
            noise_sigma_step=args.noise_sigma_step,
            noise_target_acc=args.noise_target_acc,
            noise_output_dir=args.noise_output_dir,
        )
        vec_b = points_to_vector(pts_b, sigma_values)
        all_b.append(vec_b)

    arr_a = np.stack(all_a, axis=0)  # [runs, sigmas]
    arr_b = np.stack(all_b, axis=0)
    mean_a = arr_a.mean(axis=0)
    std_a = arr_a.std(axis=0, ddof=1)
    mean_b = arr_b.mean(axis=0)
    std_b = arr_b.std(axis=0, ddof=1)

    rng = np.random.default_rng(args.seed_start)
    pvals = []
    for j in range(arr_a.shape[1]):
        p = permutation_pvalue(
            arr_a[:, j], arr_b[:, j], args.num_permutations, rng
        )
        pvals.append(p)
    pvals = np.array(pvals, dtype=np.float64)

    auc_a = np.array([auc_trapz(r, sigma_arr) for r in arr_a], dtype=np.float64)
    auc_b = np.array([auc_trapz(r, sigma_arr) for r in arr_b], dtype=np.float64)
    auc_p = permutation_pvalue(auc_a, auc_b, args.num_permutations, rng)

    run_csv = os.path.join(args.noise_output_dir, f"{args.result_prefix}_runs.csv")
    with open(run_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "seed", "sigma", "acc"])
        for i, seed in enumerate(seeds):
            for j, sigma in enumerate(sigma_values):
                writer.writerow([args.group_a_name, seed, sigma, float(arr_a[i, j])])
                writer.writerow([args.group_b_name, seed, sigma, float(arr_b[i, j])])

    summary_csv = os.path.join(
        args.noise_output_dir, f"{args.result_prefix}_summary.csv"
    )
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sigma",
                f"mean_{args.group_a_name}",
                f"std_{args.group_a_name}",
                f"mean_{args.group_b_name}",
                f"std_{args.group_b_name}",
                "delta_mean_A_minus_B",
                "p_value_permutation",
            ]
        )
        for j, sigma in enumerate(sigma_values):
            writer.writerow(
                [
                    sigma,
                    float(mean_a[j]),
                    float(std_a[j]),
                    float(mean_b[j]),
                    float(std_b[j]),
                    float(mean_a[j] - mean_b[j]),
                    float(pvals[j]),
                ]
            )

    auc_csv = os.path.join(args.noise_output_dir, f"{args.result_prefix}_auc.csv")
    with open(auc_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "seed", "auc"])
        for seed, v in zip(seeds, auc_a):
            writer.writerow([args.group_a_name, seed, float(v)])
        for seed, v in zip(seeds, auc_b):
            writer.writerow([args.group_b_name, seed, float(v)])
        writer.writerow([])
        writer.writerow(["auc_mean_" + args.group_a_name, float(auc_a.mean())])
        writer.writerow(["auc_mean_" + args.group_b_name, float(auc_b.mean())])
        writer.writerow(["auc_delta_A_minus_B", float(auc_a.mean() - auc_b.mean())])
        writer.writerow(["auc_p_value_permutation", float(auc_p)])

    print("\n完成。输出文件：")
    print(f"- 每次运行明细: {run_csv}")
    print(f"- 每个 sigma 统计+p值: {summary_csv}")
    print(f"- AUC 与 p 值: {auc_csv}")


if __name__ == "__main__":
    main()
