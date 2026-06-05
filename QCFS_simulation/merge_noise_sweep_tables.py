import argparse
import csv
import glob
import os
import re
from typing import Dict, List, Tuple


T_PATTERN = re.compile(r"_T(\d+)_")


def sigma_sort_key(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return float("inf")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="合并 noise_sweep CSV 为单表，并按 L、T 升序排序。"
    )
    p.add_argument(
        "--input_dir",
        default="noise2_exp",
        type=str,
        help="输入目录（相对 QCFS_simulation 或绝对路径）",
    )
    p.add_argument(
        "--output_csv",
        default="noise2_exp/noise_sweep_combined_L_T.csv",
        type=str,
        help="输出 CSV（相对 QCFS_simulation 或绝对路径）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = args.input_dir
    if not os.path.isabs(input_dir):
        input_dir = os.path.join(script_dir, input_dir)
    output_csv = args.output_csv
    if not os.path.isabs(output_csv):
        output_csv = os.path.join(script_dir, output_csv)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "noise_sweep_matrix_*.csv")))
    if not files:
        raise FileNotFoundError(f"未找到 noise_sweep_matrix_*.csv: {input_dir}")

    rows: List[Tuple[int, int, Dict[str, str]]] = []
    all_sigmas = set()

    for path in files:
        name = os.path.basename(path)
        mt = T_PATTERN.search(name)
        if not mt:
            continue
        t_val = int(mt.group(1))

        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            sigma_cols = [c for c in reader.fieldnames if c not in ("L", "T")]
            for c in sigma_cols:
                all_sigmas.add(c)

            for r in reader:
                # 优先从 L 列读取；若不存在则尝试 T 列（兼容历史表头）
                l_raw = r.get("L", "")
                if str(l_raw).strip() == "" and "T" in r:
                    l_raw = r.get("T", "")
                if str(l_raw).strip() == "":
                    continue
                l_val = int(float(l_raw))
                sigma_to_acc = {}
                for s in sigma_cols:
                    v = r.get(s, "")
                    if v is not None and str(v).strip() != "":
                        sigma_to_acc[s] = v
                rows.append((l_val, t_val, sigma_to_acc))

    if not rows:
        raise RuntimeError("未解析到任何有效行，请检查输入 CSV 格式。")

    ordered_sigmas = sorted(all_sigmas, key=sigma_sort_key)
    rows.sort(key=lambda x: (x[0], x[1]))  # L asc, T asc

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["L", "T"] + ordered_sigmas)
        for l_val, t_val, sigma_to_acc in rows:
            row = [l_val, t_val]
            for s in ordered_sigmas:
                row.append(sigma_to_acc.get(s, ""))
            writer.writerow(row)

    print(f"已合并 {len(files)} 个文件，输出: {output_csv}")


if __name__ == "__main__":
    main()
