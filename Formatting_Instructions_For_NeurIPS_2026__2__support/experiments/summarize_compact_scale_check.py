#!/usr/bin/env python3
"""Aggregate the compact T=4 absolute-versus-SNR-matched noise check."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


SUPPORT_DIR = Path(__file__).resolve().parents[1]
PAPER_DIR = SUPPORT_DIR.with_name(SUPPORT_DIR.name.removesuffix("_support"))
RESULTS_DIR = SUPPORT_DIR / "experiments" / "results"
OUTPUT_DIR = RESULTS_DIR / "compact_scale_check"

DATASETS = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion",
}
METHODS = {
    "old_detach": "MNE-L2",
    "orthogonal": "Orthogonal",
    "no_reg": "No Reg",
    "l2": "L2",
}
PROTOCOLS = {
    "fixed": ("compact_absolute_T4_sigma0p5", 0.5, 3),
    "matched_6db": ("compact_snr_matched_T4_6db", 0.5, 3),
    "matched_0db": ("compact_snr_matched_T4", 1.0, 5),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) != 5:
        raise ValueError(f"Expected five training seeds, found {len(values)}")
    return statistics.fmean(values), statistics.stdev(values)


def per_seed_accuracy(
    rows: list[dict[str, str]], method: str, sigma: float, repeats: int
) -> dict[int, float]:
    selected = {}
    for row in rows:
        if row["method"] != method or not math.isclose(float(row["sigma"]), sigma):
            continue
        observed_repeats = int(row["n_noise_repeats"])
        expected_repeats = 1 if sigma == 0 else repeats
        if observed_repeats != expected_repeats:
            raise ValueError(
                f"{method} sigma={sigma}: expected {expected_repeats} repeats, "
                f"found {observed_repeats}"
            )
        selected[int(row["seed"])] = float(row["acc_mean"])
    if sorted(selected) != [40, 41, 42, 43, 44]:
        raise ValueError(f"Incomplete seed coverage for {method} sigma={sigma}")
    return selected


def per_seed_rms(dataset: str, method: str) -> dict[int, float]:
    path = (
        RESULTS_DIR
        / "compact_snr_matched_T4_6db"
        / dataset
        / "internal_if_noise_raw.csv"
    )
    rows = read_csv(path)
    values = {}
    for row in rows:
        if row["method"] == method:
            values[int(row["seed"])] = float(row["scale_reference"])
    if sorted(values) != [40, 41, 42, 43, 44]:
        raise ValueError(f"Incomplete RMS coverage for {dataset}/{method}")
    return values


def summarize() -> list[dict[str, float | int | str]]:
    output = []
    for dataset, dataset_label in DATASETS.items():
        protocol_rows = {}
        for protocol, (root, _sigma, _repeats) in PROTOCOLS.items():
            path = RESULTS_DIR / root / dataset / "internal_if_noise_repeat_mean_std.csv"
            protocol_rows[protocol] = read_csv(path)

        for method, method_label in METHODS.items():
            rms_by_seed = per_seed_rms(dataset, method)
            rms_mean, rms_std = mean_std(list(rms_by_seed.values()))
            fixed_snr = [20.0 * math.log10(value / 0.5) for value in rms_by_seed.values()]
            fixed_snr_mean, fixed_snr_std = mean_std(fixed_snr)

            item: dict[str, float | int | str] = {
                "dataset": dataset,
                "dataset_label": dataset_label,
                "method": method,
                "method_label": method_label,
                "n_training_seeds": 5,
                "post_input_if_rms_mean": rms_mean,
                "post_input_if_rms_std": rms_std,
                "fixed_sigma0p5_snr_db_mean": fixed_snr_mean,
                "fixed_sigma0p5_snr_db_std": fixed_snr_std,
            }

            for protocol, (_root, sigma, repeats) in PROTOCOLS.items():
                rows = protocol_rows[protocol]
                clean = per_seed_accuracy(rows, method, 0.0, repeats)
                noisy = per_seed_accuracy(rows, method, sigma, repeats)
                noisy_values = [noisy[seed] for seed in sorted(noisy)]
                drops = [clean[seed] - noisy[seed] for seed in sorted(clean)]
                noisy_mean, noisy_std = mean_std(noisy_values)
                drop_mean, drop_std = mean_std(drops)
                item[f"{protocol}_noisy_acc_mean"] = noisy_mean
                item[f"{protocol}_noisy_acc_std"] = noisy_std
                item[f"{protocol}_drop_mean"] = drop_mean
                item[f"{protocol}_drop_std"] = drop_std
            output.append(item)
    return output


def write_outputs(rows: list[dict[str, float | int | str]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "compact_scale_check_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tex_path = OUTPUT_DIR / "compact_scale_check_rows.tex"
    with tex_path.open("w") as handle:
        for index, row in enumerate(rows):
            if index == len(METHODS):
                handle.write("\\midrule\n")
            handle.write(
                f"{row['dataset_label']} & {row['method_label']} & "
                f"${row['post_input_if_rms_mean']:.3f}\\pm{row['post_input_if_rms_std']:.3f}$ & "
                f"${row['fixed_sigma0p5_snr_db_mean']:.2f}\\pm{row['fixed_sigma0p5_snr_db_std']:.2f}$ & "
                f"${row['fixed_drop_mean']:.2f}\\pm{row['fixed_drop_std']:.2f}$ & "
                f"${row['matched_6db_drop_mean']:.2f}\\pm{row['matched_6db_drop_std']:.2f}$ \\\\\n"
            )
        handle.write("\\bottomrule\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    write_outputs(summarize())
