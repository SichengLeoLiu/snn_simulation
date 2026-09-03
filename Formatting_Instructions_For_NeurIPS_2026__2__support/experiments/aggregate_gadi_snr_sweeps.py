#!/usr/bin/env python3
"""Aggregate per-seed relative-SNR sweeps without mixing noise and training seeds."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing seed_*/relative_snr_noise_raw.csv outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to --root.",
    )
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    return parser.parse_args()


def read_rows(root: Path) -> list[dict[str, str]]:
    paths = sorted(root.glob("seed_*/relative_snr_noise_raw.csv"))
    if not paths:
        raise FileNotFoundError(f"No seed_*/relative_snr_noise_raw.csv files under {root}")

    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["source_csv"] = str(path)
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    per_seed_repeats = defaultdict(list)
    per_seed_sigma = defaultdict(list)
    for row in rows:
        key = (
            int(row["seed"]),
            row["method"],
            row["scale_mode"],
            row["level_name"],
            row["level_value"],
        )
        per_seed_repeats[key].append(float(row["accuracy"]))
        per_seed_sigma[key].append(float(row["actual_sigma"]))

    per_seed_rows = []
    for key, accuracies in sorted(per_seed_repeats.items()):
        seed, method, scale_mode, level_name, level_value = key
        per_seed_rows.append(
            {
                "seed": seed,
                "method": method,
                "scale_mode": scale_mode,
                "level_name": level_name,
                "level_value": level_value,
                "n_noise_repeats": len(accuracies),
                "acc_mean_over_noise": f"{statistics.fmean(accuracies):.6f}",
                "acc_std_over_noise": (
                    f"{statistics.stdev(accuracies):.6f}" if len(accuracies) > 1 else "0.000000"
                ),
                "actual_sigma": f"{statistics.fmean(per_seed_sigma[key]):.9g}",
            }
        )

    across_seeds = defaultdict(list)
    across_sigmas = defaultdict(list)
    for row in per_seed_rows:
        key = (
            row["method"],
            row["scale_mode"],
            row["level_name"],
            row["level_value"],
        )
        across_seeds[key].append(float(row["acc_mean_over_noise"]))
        across_sigmas[key].append(float(row["actual_sigma"]))

    aggregate_rows = []
    for key, accuracies in sorted(across_seeds.items()):
        method, scale_mode, level_name, level_value = key
        aggregate_rows.append(
            {
                "method": method,
                "scale_mode": scale_mode,
                "level_name": level_name,
                "level_value": level_value,
                "n_training_seeds": len(accuracies),
                "acc_mean": f"{statistics.fmean(accuracies):.6f}",
                "acc_std_across_training_seeds": (
                    f"{statistics.stdev(accuracies):.6f}" if len(accuracies) > 1 else "0.000000"
                ),
                "actual_sigma_mean_across_seeds": f"{statistics.fmean(across_sigmas[key]):.9g}",
                "actual_sigma_std_across_seeds": (
                    f"{statistics.stdev(across_sigmas[key]):.9g}"
                    if len(across_sigmas[key]) > 1
                    else "0"
                ),
            }
        )
    return per_seed_rows, aggregate_rows


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.root
    rows = read_rows(args.root)
    found_seeds = sorted({int(row["seed"]) for row in rows})
    missing = sorted(set(args.expected_seeds) - set(found_seeds))
    if missing:
        raise RuntimeError(f"Missing expected training seeds: {missing}; found {found_seeds}")

    per_seed_rows, aggregate_rows = aggregate(rows)
    write_csv(output_dir / "snr_per_training_seed.csv", per_seed_rows)
    write_csv(output_dir / "snr_multiseed_mean_std.csv", aggregate_rows)
    print(
        f"Aggregated {len(rows)} raw evaluations across seeds {found_seeds} "
        f"into {output_dir / 'snr_multiseed_mean_std.csv'}"
    )


if __name__ == "__main__":
    main()
