import argparse
import csv
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import numpy as np


ACC_PATTERN = re.compile(r"Test acc = ([0-9.]+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="多 seed：每个 seed 先做 ANN(T=0)训练，再按多个 T 做 SNN 转换测试并汇总 CSV。"
    )
    p.add_argument("--dataset", default="mnist", type=str)
    p.add_argument("--arch", default="cnn2", type=str)
    p.add_argument("--device", default="mps", type=str)
    p.add_argument("--L", required=True, type=int)
    p.add_argument("--run_name", default="mnist_run", type=str)

    p.add_argument("--num_runs", default=10, type=int)
    p.add_argument("--seed_start", default=44, type=int)

    p.add_argument("--epochs", default=20, type=int)
    p.add_argument("--lr", default=0.01, type=float)
    p.add_argument("--weight_decay", default=0.0, type=float)
    p.add_argument("--batch_size", default=128, type=int)
    p.add_argument("--workers", default=4, type=int)

    p.add_argument(
        "--ann_train_spike_schedule",
        default="normal",
        type=str,
        help="ANN 训练阶段 spike schedule（T=0 时通常用 normal）",
    )
    p.add_argument(
        "--test_spike_schedule",
        default="normal",
        type=str,
        help="测试阶段 spike schedule（main_test.py 参数）",
    )
    p.add_argument(
        "--test_mode",
        default="normal",
        type=str,
        help="测试阶段 IF mode（main_test.py --mode）",
    )
    p.add_argument(
        "--test_T_values",
        default="2,4,8,16,32",
        type=str,
        help="测试时间步列表，逗号分隔；默认 2,4,8,16,32",
    )

    p.add_argument(
        "--output_csv",
        default="Noise_exp/train_test_multiseed_results.csv",
        type=str,
        help="输出 CSV（相对 QCFS_simulation 目录）",
    )
    p.add_argument(
        "--python_exec",
        default=sys.executable,
        type=str,
        help="用于运行 main_train.py/main_test.py 的解释器",
    )
    return p.parse_args()


def _resolved_model_name(dataset: str, model: str) -> str:
    m = model.lower()
    d = dataset.lower().replace("-", "").replace("_", "")
    if d in ("diff1d", "toydiff1d"):
        return "diff1d"
    if d != "mnist" and m in ("cnn2", "cnn2_mnist"):
        return "vgg16"
    return model


def _parse_test_t_values(raw: str) -> List[int]:
    parts = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not parts:
        raise ValueError("--test_T_values 不能为空")
    t_values: List[int] = []
    for p in parts:
        t = int(p)
        if t <= 0:
            raise ValueError("test T 必须 > 0（SNN 转换测试）")
        t_values.append(t)
    # 去重并保持原有顺序
    uniq: List[int] = []
    seen = set()
    for t in t_values:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _checkpoint_path(script_dir: str, dataset: str, arch: str, L: int, suffix: str) -> str:
    ds = dataset.lower()
    log_ds = "diff1d" if ds.replace("_", "").replace("-", "") in ("diff1d", "toydiff1d") else ds
    ckpt_dir = os.path.join(script_dir, f"{log_ds}-checkpoints")
    identifier = f"{arch}_L[{L}]"
    if suffix:
        identifier += f"_{suffix}"
    return os.path.join(ckpt_dir, f"{identifier}.pth")


def run_train(
    script_dir: str,
    args: argparse.Namespace,
    seed: int,
    suffix: str,
) -> Tuple[float, str]:
    cmd = [
        args.python_exec,
        os.path.join(script_dir, "main_train.py"),
        "-data",
        args.dataset,
        "-arch",
        args.arch,
        "-dev",
        args.device,
        "-L",
        str(args.L),
        "-T",
        "0",
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "-lr",
        str(args.lr),
        "-wd",
        str(args.weight_decay),
        "-b",
        str(args.batch_size),
        "-j",
        str(args.workers),
        "--spike_schedule",
        args.ann_train_spike_schedule,
        "-suffix",
        suffix,
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
            cwd=script_dir,
        )
    except subprocess.CalledProcessError as e:
        out = e.stdout or ""
        tail = "\n".join(out.splitlines()[-80:])
        raise RuntimeError(
            f"训练失败（seed={seed}）。命令: {' '.join(cmd)}\n"
            f"--- train stdout/stderr (tail) ---\n{tail}"
        ) from e
    elapsed = time.time() - t0
    return elapsed, proc.stdout


def run_test(
    script_dir: str,
    args: argparse.Namespace,
    seed: int,
    weight_path: str,
    test_T: int,
    test_mode: str,
    test_spike_schedule: str,
) -> Tuple[float, float, str]:
    cmd = [
        args.python_exec,
        os.path.join(script_dir, "main_test.py"),
        "-data",
        args.dataset,
        "-arch",
        args.arch,
        "-dev",
        args.device,
        "-L",
        str(args.L),
        "-T",
        str(test_T),
        "--seed",
        str(seed),
        "-b",
        str(args.batch_size),
        "-j",
        str(args.workers),
        "--mode",
        test_mode,
        "--spike_schedule",
        test_spike_schedule,
        "-w",
        weight_path,
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
            cwd=script_dir,
        )
    except subprocess.CalledProcessError as e:
        out = e.stdout or ""
        tail = "\n".join(out.splitlines()[-80:])
        raise RuntimeError(
            f"测试失败（seed={seed}）。命令: {' '.join(cmd)}\n"
            f"--- test stdout/stderr (tail) ---\n{tail}"
        ) from e
    elapsed = time.time() - t0
    text = proc.stdout
    hits = ACC_PATTERN.findall(text)
    if not hits:
        raise RuntimeError(f"seed={seed} 未从 main_test 输出中解析到 Test acc")
    return float(hits[-1]), elapsed, text


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_py = os.path.join(script_dir, "main_train.py")
    test_py = os.path.join(script_dir, "main_test.py")
    if not os.path.exists(train_py):
        raise FileNotFoundError(f"找不到 main_train.py: {train_py}")
    if not os.path.exists(test_py):
        raise FileNotFoundError(f"找不到 main_test.py: {test_py}")

    out_csv = args.output_csv
    if not os.path.isabs(out_csv):
        out_csv = os.path.join(script_dir, out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    seeds = [args.seed_start + i for i in range(args.num_runs)]
    test_t_values = _parse_test_t_values(args.test_T_values)
    arch_resolved = _resolved_model_name(args.dataset, args.arch)

    print(f"Seeds: {seeds}")
    print(
        f"Config: dataset={args.dataset}, arch={args.arch}, resolved_arch={arch_resolved}, "
        f"L={args.L}, train_T=0(ANN), test_T_values={test_t_values}, epochs={args.epochs}"
    )

    rows: List[dict] = []
    accs_by_t: Dict[int, List[float]] = {t: [] for t in test_t_values}
    for idx, seed in enumerate(seeds, 1):
        suffix = f"{args.run_name}_seed{seed}"
        ckpt = _checkpoint_path(script_dir, args.dataset, arch_resolved, args.L, suffix)
        print(f"\n[{idx}/{len(seeds)}] seed={seed} -> ANN training T=0 (suffix={suffix})")
        train_elapsed, _ = run_train(script_dir, args, seed, suffix)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"训练后未找到权重文件: {ckpt}")

        acc_seed: Dict[int, float] = {}
        test_time_seed: Dict[int, float] = {}
        for t in test_t_values:
            print(f"[{idx}/{len(seeds)}] seed={seed} -> SNN converted test T={t}")
            acc_t, elapsed_t, _ = run_test(
                script_dir,
                args,
                seed,
                ckpt,
                test_T=t,
                test_mode=args.test_mode,
                test_spike_schedule=args.test_spike_schedule,
            )
            acc_seed[t] = acc_t
            test_time_seed[t] = elapsed_t
            accs_by_t[t].append(acc_t)

        acc_summary = ", ".join([f"T{t}={acc_seed[t]:.3f}" for t in test_t_values])
        print(
            f"  done seed={seed}: {acc_summary}, "
            f"train_time={train_elapsed:.1f}s"
        )
        rows.append(
            {
                "seed": seed,
                "acc_by_t": acc_seed,
                "train_seconds": train_elapsed,
                "test_seconds_by_t": test_time_seed,
            }
        )

    mean_std_by_t: Dict[int, Tuple[float, float]] = {}
    for t in test_t_values:
        arr = np.array(accs_by_t[t], dtype=np.float64)
        mean_t = float(arr.mean())
        std_t = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        mean_std_by_t[t] = (mean_t, std_t)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        head = ["seed", "train_seconds"]
        for t in test_t_values:
            head.append(f"acc_T{t}")
            head.append(f"test_seconds_T{t}")
        writer.writerow(head)
        for r in rows:
            row = [r["seed"], f"{r['train_seconds']:.3f}"]
            for t in test_t_values:
                row.append(f"{r['acc_by_t'][t]:.6f}")
                row.append(f"{r['test_seconds_by_t'][t]:.3f}")
            writer.writerow(row)
        writer.writerow([])
        writer.writerow(["metric", "value"])
        for t in test_t_values:
            mean_t, std_t = mean_std_by_t[t]
            writer.writerow([f"mean_acc_T{t}", f"{mean_t:.6f}"])
            writer.writerow([f"std_acc_T{t}", f"{std_t:.6f}"])
        writer.writerow(["num_runs", args.num_runs])
        writer.writerow(["seed_start", args.seed_start])
        writer.writerow(["dataset", args.dataset])
        writer.writerow(["arch", args.arch])
        writer.writerow(["resolved_arch", arch_resolved])
        writer.writerow(["L", args.L])
        writer.writerow(["train_T", 0])
        writer.writerow(["test_T_values", ",".join(str(t) for t in test_t_values)])
        writer.writerow(["epochs", args.epochs])
        writer.writerow(["run_name", args.run_name])
        writer.writerow(["device", args.device])
        writer.writerow(["test_mode", args.test_mode])
        writer.writerow(["ann_train_spike_schedule", args.ann_train_spike_schedule])
        writer.writerow(["test_spike_schedule", args.test_spike_schedule])

    print("\n完成。")
    print(f"输出 CSV: {out_csv}")
    print(
        " | ".join(
            [
                f"T{t}: mean={mean_std_by_t[t][0]:.4f}, std={mean_std_by_t[t][1]:.4f}"
                for t in test_t_values
            ]
        )
    )


if __name__ == "__main__":
    main()
