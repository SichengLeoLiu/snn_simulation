#!/usr/bin/env python3
"""
Compute Derivative Robustness Score (DRS) across all noise-sweep experiments.

Outputs:
  drs_results/drs_all_models.csv          — master table
  drs_results/<experiment>/drs_*.csv      — per-experiment breakdown
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

from robustness_metrics import (
    aggregate_drs,
    aggregate_drs_from_raw_rows,
    auc_robustness_score,
    derivative_robustness_score,
)

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "drs_results"

MASTER_FIELDS = [
    "experiment", "dataset", "arch", "hidden_size", "method", "regularizer",
    "DRS_mean", "DRS_std", "DRS_sem", "AUC_RS_mean", "AUC_RS_std", "n_seeds",
]


def load_raw_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_matrix_csv(path: Path) -> tuple[list[float], list[float]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)
    start = 2 if header[:2] == ["L", "T"] else 1
    sigmas = [float(x) for x in header[start:]]
    accs = [float(x) for x in row[start:]]
    return sigmas, accs


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {}
            for k in fieldnames:
                v = r.get(k, "")
                if isinstance(v, float):
                    out[k] = f"{v:.6f}"
                else:
                    out[k] = v
            w.writerow(out)
    print(f"[SAVED] {path}")


def tag_row(experiment: str, dataset: str, **kwargs) -> dict:
    return {"experiment": experiment, "dataset": dataset, **kwargs}


# ---------------------------------------------------------------------------
# FC3rev three-regs (raw long CSV)
# ---------------------------------------------------------------------------
def compute_fc3rev_three_regs(name: str, raw_csv: Path) -> list[dict]:
    if not raw_csv.exists():
        print(f"[SKIP] missing {raw_csv}")
        return []
    raw = load_raw_csv(raw_csv)
    drs_rows = aggregate_drs_from_raw_rows(
        raw, group_keys=("arch", "regularizer")
    )
    out = []
    for r in drs_rows:
        h = int(r["arch"].split("_h")[1])
        out.append(tag_row(
            name, "mnist",
            arch=r["arch"], hidden_size=h, method=r["regularizer"],
            regularizer=r["regularizer"],
            DRS_mean=r["DRS_mean"], DRS_std=r["DRS_std"], DRS_sem=r["DRS_sem"],
            AUC_RS_mean=r["AUC_RS_mean"], AUC_RS_std=r["AUC_RS_std"],
            n_seeds=r["n_seeds"],
        ))
    write_csv(
        out,
        OUT_ROOT / name / "drs_fc3rev_three_regs.csv",
        ["arch", "regularizer", "DRS_mean", "DRS_std", "DRS_sem",
         "AUC_RS_mean", "AUC_RS_std", "n_seeds"],
    )
    return out


# ---------------------------------------------------------------------------
# FC3rev WD-only
# ---------------------------------------------------------------------------
def compute_fc3rev_wd(name: str, raw_csv: Path) -> list[dict]:
    if not raw_csv.exists():
        print(f"[SKIP] missing {raw_csv}")
        return []
    raw = load_raw_csv(raw_csv)
    by_arch_seed: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in raw:
        by_arch_seed[r["arch"]][int(r["seed"])].append(
            (float(r["sigma"]), float(r["acc"]))
        )
    out = []
    for arch in sorted(by_arch_seed, key=lambda a: int(a.split("_h")[1])):
        stats = aggregate_drs(by_arch_seed[arch])
        h = int(arch.split("_h")[1])
        out.append(tag_row(
            name, "mnist",
            arch=arch, hidden_size=h, method="weight_decay",
            regularizer="weight_decay", **stats,
        ))
    write_csv(
        out,
        OUT_ROOT / name / "drs_fc3rev_wd.csv",
        ["arch", "DRS_mean", "DRS_std", "DRS_sem", "AUC_RS_mean", "n_seeds"],
    )
    return out


# ---------------------------------------------------------------------------
# CNN2 — per-arch per-method matrix CSVs
# ---------------------------------------------------------------------------
CNN2_ARCHES = ["cnn2_c2_c4", "cnn2_c4_c8", "cnn2_c8_c16", "cnn2_c16_c32"]
CNN2_METHODS = ["weight_decay", "mne_l2", "no_regularization"]


def compute_cnn2(name: str, data_root: Path) -> list[dict]:
    if not data_root.exists():
        print(f"[SKIP] missing {data_root}")
        return []
    out = []
    for arch in CNN2_ARCHES:
        for method in CNN2_METHODS:
            seed_files = sorted(
                (data_root / arch / method).glob("seed_*/noise_sweep_matrix_*.csv")
            )
            curves = {}
            for fp in seed_files:
                seed = int(fp.parent.name.split("_")[1])
                sigmas, accs = read_matrix_csv(fp)
                curves[seed] = list(zip(sigmas, accs))
            if not curves:
                continue
            stats = aggregate_drs(curves)
            out.append(tag_row(
                name, "mnist",
                arch=arch, hidden_size="", method=method,
                regularizer=method, **stats,
            ))
    write_csv(
        out,
        OUT_ROOT / name / "drs_cnn2.csv",
        ["arch", "method", "DRS_mean", "DRS_std", "DRS_sem",
         "AUC_RS_mean", "AUC_RS_std", "n_seeds"],
    )
    return out


# ---------------------------------------------------------------------------
# CIFAR VGG16 — raw long CSV
# ---------------------------------------------------------------------------
def compute_cifar_vgg16(name: str, raw_csv: Path, dataset: str) -> list[dict]:
    if not raw_csv.exists():
        print(f"[SKIP] missing {raw_csv}")
        return []
    raw = load_raw_csv(raw_csv)
    by_method_seed: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in raw:
        if r.get("method") not in ("weight_decay", "mne_l2", "no_regularization"):
            continue
        by_method_seed[r["method"]][int(r["seed"])].append(
            (float(r["sigma"]), float(r["acc"]))
        )
    out = []
    for method in ("weight_decay", "mne_l2", "no_regularization"):
        if method not in by_method_seed:
            continue
        stats = aggregate_drs(by_method_seed[method])
        out.append(tag_row(
            name, dataset,
            arch="vgg16", hidden_size="", method=method,
            regularizer=method, **stats,
        ))
    write_csv(
        out,
        OUT_ROOT / name / f"drs_{dataset}_vgg16.csv",
        ["method", "DRS_mean", "DRS_std", "DRS_sem",
         "AUC_RS_mean", "AUC_RS_std", "n_seeds"],
    )
    return out


# ---------------------------------------------------------------------------
# ImageNet ResNet18 — combined CSVs (single curve per method, no seeds)
# ---------------------------------------------------------------------------
def compute_imagenet_resnet18(name: str) -> list[dict]:
    l2_mne = ROOT / "imagenet_resnet18_l2_vs_mnel2_combined.csv"
    no_reg = ROOT / "imagenet_resnet18_no_reg_noise_sweep.csv"
    if not l2_mne.exists() or not no_reg.exists():
        print(f"[SKIP] ImageNet CSVs missing")
        return []

    sigmas, l2, mne = [], [], []
    with l2_mne.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sigmas.append(float(r["sigma"]))
            l2.append(float(r["acc_l2_weight_decay"]))
            mne.append(float(r["acc_mne_l2_rc1e-4"]))

    sigmas_nr, no_reg_accs = [], []
    with no_reg.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sigmas_nr.append(float(r["sigma"]))
            no_reg_accs.append(float(r["acc"]))

    curves = {
        "weight_decay": (sigmas, l2),
        "mne_l2": (sigmas, mne),
        "no_regularization": (sigmas_nr, no_reg_accs),
    }
    out = []
    detail = []
    for method, (s, a) in curves.items():
        drs = derivative_robustness_score(s, a)
        auc = auc_robustness_score(s, a)
        detail.append({
            "method": method,
            "DRS": drs,
            "AUC_RS": auc,
            "A0": a[0] if a else 0.0,
        })
        out.append(tag_row(
            name, "imagenet",
            arch="resnet18", hidden_size="", method=method,
            regularizer=method,
            DRS_mean=drs, DRS_std=0.0, DRS_sem=0.0,
            AUC_RS_mean=auc, AUC_RS_std=0.0, n_seeds=1,
        ))
    write_csv(detail, OUT_ROOT / name / "drs_imagenet_resnet18.csv",
              ["method", "DRS", "AUC_RS", "A0"])
    return out


# ---------------------------------------------------------------------------
# FC3 strict-seed (h4-h128)
# ---------------------------------------------------------------------------
def compute_fc3_strict_seed(name: str, raw_csv: Path) -> list[dict]:
    if not raw_csv.exists():
        print(f"[SKIP] missing {raw_csv}")
        return []
    raw = load_raw_csv(raw_csv)
    drs_rows = aggregate_drs_from_raw_rows(
        raw, group_keys=("arch", "regularizer")
    )
    out = []
    for r in drs_rows:
        h = int(r["arch"].split("_h")[1]) if "_h" in r["arch"] else ""
        out.append(tag_row(
            name, "mnist",
            arch=r["arch"], hidden_size=h, method=r["regularizer"],
            regularizer=r["regularizer"],
            DRS_mean=r["DRS_mean"], DRS_std=r["DRS_std"], DRS_sem=r["DRS_sem"],
            AUC_RS_mean=r["AUC_RS_mean"], AUC_RS_std=r["AUC_RS_std"],
            n_seeds=r["n_seeds"],
        ))
    write_csv(
        out,
        OUT_ROOT / name / "drs_fc3_strict_seed.csv",
        ["arch", "regularizer", "DRS_mean", "DRS_std", "DRS_sem",
         "AUC_RS_mean", "n_seeds"],
    )
    return out


# ---------------------------------------------------------------------------
# MNE reg coeff scan
# ---------------------------------------------------------------------------
def compute_mne_reg_coeff_scan(name: str, raw_csv: Path) -> list[dict]:
    if not raw_csv.exists():
        print(f"[SKIP] missing {raw_csv}")
        return []
    raw = load_raw_csv(raw_csv)
    drs_rows = aggregate_drs_from_raw_rows(
        raw, group_keys=("arch", "method", "regularizer", "reg_coeff")
    )
    out = []
    for r in drs_rows:
        h = int(r["arch"].split("_h")[1])
        out.append(tag_row(
            name, "mnist",
            arch=r["arch"], hidden_size=h,
            method=r["method"], regularizer=r["regularizer"],
            reg_coeff=r.get("reg_coeff", ""),
            DRS_mean=r["DRS_mean"], DRS_std=r["DRS_std"], DRS_sem=r["DRS_sem"],
            AUC_RS_mean=r["AUC_RS_mean"], AUC_RS_std=r["AUC_RS_std"],
            n_seeds=r["n_seeds"],
        ))
    write_csv(
        out,
        OUT_ROOT / name / "drs_mne_reg_coeff_scan.csv",
        ["arch", "method", "regularizer", "reg_coeff",
         "DRS_mean", "DRS_std", "DRS_sem", "AUC_RS_mean", "n_seeds"],
    )
    return out


def main() -> None:
    all_rows: list[dict] = []

    experiments = [
        ("new_fc3_three_regs",
         lambda: compute_fc3rev_three_regs(
             "new_fc3_three_regs",
             ROOT / "important results" / "new_fc3" / "fc3rev_h8_h256_three_regs_noise_sweep_raw.csv",
         )),
        ("new_fc3.1_three_regs",
         lambda: compute_fc3rev_three_regs(
             "new_fc3.1_three_regs",
             ROOT / "important results" / "new_fc3.1" / "new_fc3.1" / "fc3rev_h8_h256_three_regs_noise_sweep_raw.csv",
         )),
        ("new_fc3.2_three_regs",
         lambda: compute_fc3rev_three_regs(
             "new_fc3.2_three_regs",
             ROOT / "important results" / "new_fc3.2" / "fc3rev_h8_h256_three_regs_noise_sweep_raw.csv",
         )),
        ("new_fc3_wd",
         lambda: compute_fc3rev_wd(
             "new_fc3_wd",
             ROOT / "important results" / "new_fc3" / "fc3rev_h8_h256_wd_noise_sweep_raw.csv",
         )),
        ("cnn2_rate_uniform",
         lambda: compute_cnn2(
             "cnn2_rate_uniform",
             ROOT / "all_results_from_gadi" / "cnn2_noise_sweep_step0p05_full_extracted",
         )),
        ("cnn2_normal",
         lambda: compute_cnn2(
             "cnn2_normal",
             ROOT / "all_results_from_gadi" / "cnn2_noise_sweep_step0p05_full_extracted" / "normal",
         )),
        ("cifar10_vgg16",
         lambda: compute_cifar_vgg16(
             "cifar10_vgg16",
             ROOT / "all_results_from_gadi" / "noise3_exp"
             / "cifar10_vgg16_strict_seed_three_regs_noise_sweep_rate_uniform_L16_T16"
             / "cifar10_vgg16_strict_seed_three_regs_noise_sweep_raw.csv",
             "cifar10",
         )),
        ("cifar100_vgg16",
         lambda: compute_cifar_vgg16(
             "cifar100_vgg16",
             ROOT / "all_results_from_gadi" / "cifar100_vgg16_strict_seed_three_regs_noise_sweep_raw.csv",
             "cifar100",
         )),
        ("imagenet_resnet18", lambda: compute_imagenet_resnet18("imagenet_resnet18")),
        ("fc3_strict_seed_rate_uniform",
         lambda: compute_fc3_strict_seed(
             "fc3_strict_seed_rate_uniform",
             ROOT / "QCFS_simulation" / "noise3_exp"
             / "ablation_mne_l2_vs_weight_decay_l16_fc3_h4_h8_h16_h32_h64_h128"
             / "strict_seed_train_rate_uniform_L16_T16"
             / "strict_seed_train_noise_sweep_fc3_h4_h8_h16_h32_h64_h128_raw.csv",
         )),
    ]

    # Gadi scan result if synced locally
    scan_csv = ROOT / "important results" / "new_fc3" / "mne_reg_coeff_scan_acc01" / "fc3rev_mne_reg_coeff_scan_noise_sweep_raw.csv"
    if scan_csv.exists():
        experiments.append(
            ("mne_reg_coeff_scan_acc01",
             lambda: compute_mne_reg_coeff_scan("mne_reg_coeff_scan_acc01", scan_csv))
        )

    for exp_name, fn in experiments:
        print(f"\n=== {exp_name} ===")
        try:
            rows = fn()
            all_rows.extend(rows)
        except Exception as e:
            print(f"[ERROR] {exp_name}: {e}")

    write_csv(all_rows, OUT_ROOT / "drs_all_models.csv", MASTER_FIELDS)

    # Summary table to stdout
    print("\n" + "=" * 72)
    print("DRS SUMMARY (mean across seeds)")
    print("=" * 72)
    for exp in sorted({r["experiment"] for r in all_rows}):
        sub = [r for r in all_rows if r["experiment"] == exp]
        print(f"\n[{exp}]")
        for r in sorted(sub, key=lambda x: (str(x.get("arch", "")), str(x.get("method", "")))):
            arch = r.get("arch", "")
            method = r.get("method", r.get("regularizer", ""))
            drs = r["DRS_mean"]
            auc = r.get("AUC_RS_mean", float("nan"))
            print(f"  {arch:<20} {method:<22} DRS={drs:.4f}  (old AUC-RS={auc:.4f})  n={r['n_seeds']}")


if __name__ == "__main__":
    main()
