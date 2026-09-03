#!/usr/bin/env python3
"""Merge Fashion-MNIST latency-screen CSVs and compute paired MNE deltas."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


METHOD_ORDER = {"old_detach": 0, "orthogonal": 1, "no_reg": 2, "l2": 3}
T_CRITICAL_95 = {
    1: 12.706205,
    2: 4.302653,
    3: 3.182446,
    4: 2.776445,
    5: 2.570582,
    6: 2.446912,
    7: 2.364624,
    8: 2.306004,
    9: 2.262157,
}


def parse_run(value: str) -> tuple[str, int, Path]:
    try:
        run_key, directory = value.split("=", 1)
        if ":" in run_key:
            dataset, timestep = run_key.split(":", 1)
        else:
            dataset, timestep = "", run_key
        return dataset, int(timestep), Path(directory)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "runs must use [DATASET:]TIMESTEP=RESULT_DIRECTORY, "
            "for example mnist:4=results/mnist_T4"
        ) from exc


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def merge_runs(
    runs: list[tuple[str, int, Path]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    merged: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []

    for dataset, timestep, directory in sorted(runs):
        mean_rows = read_csv(directory / "internal_if_noise_mean_std.csv")
        summary_rows = read_csv(directory / "internal_if_noise_summary.csv")
        raw_rows = read_csv(directory / "internal_if_noise_raw.csv")

        mean_by_method_sigma = {
            (row["method"], float(row["sigma"])): row for row in mean_rows
        }
        for summary in summary_rows:
            method = summary["method"]
            clean = mean_by_method_sigma[(method, 0.0)]
            sigma_025 = mean_by_method_sigma[(method, 0.25)]
            sigma_05 = mean_by_method_sigma[(method, 0.5)]
            merged.append(
                {
                    "dataset": dataset,
                    "T": timestep,
                    "method": method,
                    "method_label": summary["method_label"],
                    "clean_acc": clean["acc_mean"],
                    "clean_std": clean["acc_std"],
                    "sigma_0p25_acc": sigma_025["acc_mean"],
                    "sigma_0p25_std": sigma_025["acc_std"],
                    "sigma_0p5_acc": sigma_05["acc_mean"],
                    "sigma_0p5_std": sigma_05["acc_std"],
                    "drop_to_0p5": summary["absolute_drop"],
                    "retention_at_0p5": summary["end_retention"],
                    "absolute_curve_auc": summary["curve_acc_auc"],
                    "retention_curve_auc": summary["curve_retention_auc"],
                }
            )

        repeat_averages: dict[tuple[str, int, float], list[float]] = defaultdict(list)
        labels: dict[str, str] = {}
        for row in raw_rows:
            method = row["method"]
            labels[method] = row["method_label"]
            repeat_averages[(method, int(row["seed"]), float(row["sigma"]))].append(
                float(row["accuracy"])
            )
        seed_means = {
            key: statistics.fmean(values) for key, values in repeat_averages.items()
        }

        baselines = sorted(
            {method for method, _, _ in seed_means if method != "old_detach"},
            key=lambda method: METHOD_ORDER.get(method, 99),
        )
        sigmas = sorted({sigma for _, _, sigma in seed_means})
        seeds = sorted({seed for _, seed, _ in seed_means})
        for baseline in baselines:
            for sigma in sigmas:
                differences = [
                    seed_means[("old_detach", seed, sigma)]
                    - seed_means[(baseline, seed, sigma)]
                    for seed in seeds
                ]
                difference_mean = statistics.fmean(differences)
                difference_std = statistics.stdev(differences)
                t_critical = T_CRITICAL_95.get(len(differences) - 1, 1.96)
                ci_margin = t_critical * difference_std / math.sqrt(len(differences))
                paired_rows.append(
                    {
                        "dataset": dataset,
                        "T": timestep,
                        "sigma": f"{sigma:g}",
                        "baseline": baseline,
                        "baseline_label": labels[baseline],
                        "n_seeds": len(differences),
                        "mne_minus_baseline_mean": f"{difference_mean:.6f}",
                        "mne_minus_baseline_std": f"{difference_std:.6f}",
                        "ci95_low": f"{difference_mean - ci_margin:.6f}",
                        "ci95_high": f"{difference_mean + ci_margin:.6f}",
                        "mne_wins": sum(value > 0 for value in differences),
                        "ties": sum(value == 0 for value in differences),
                    }
                )

    merged.sort(
        key=lambda row: (
            str(row["dataset"]),
            int(row["T"]),
            METHOD_ORDER.get(str(row["method"]), 99),
        )
    )
    return merged, paired_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, type=parse_run)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--paired-out", type=Path, required=True)
    args = parser.parse_args()

    summary_rows, paired_rows = merge_runs(args.runs)
    summary_fields = list(summary_rows[0])
    paired_fields = list(paired_rows[0])
    write_csv(args.summary_out, summary_rows, summary_fields)
    write_csv(args.paired_out, paired_rows, paired_fields)
    print(f"Wrote {args.summary_out} ({len(summary_rows)} rows)")
    print(f"Wrote {args.paired_out} ({len(paired_rows)} rows)")


if __name__ == "__main__":
    main()
