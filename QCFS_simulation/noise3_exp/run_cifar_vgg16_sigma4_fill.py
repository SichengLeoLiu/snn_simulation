#!/usr/bin/env python3
"""Insert the missing post-IF sigma=4 point into existing CIFAR sweeps.

Does not retrain and does not repeat sigma in {0,1,2,3,5}. Cells that already
contain sigma=4 are skipped.

Default (vgg16, cifar10) also fills TA-MNE-U at L=16 T=4/8 and at L=T in {4,8,32}.
Other arch/dataset pairs fill L2-all, L2-wo, and L1-wo only:
  L=16, T=16, seeds 40-44
  L=16, T=4 and T=8, seeds 40-44
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_cifar_l16_t48_evalseed0 import load_snn as load_snn_l16  # noqa: E402
from run_cifar_unified_baseline_evalseed0 import (  # noqa: E402
    EVAL_SEED,
    get_torch_device,
    resolve_ckpt,
    sweep,
    test_loader,
    val_loader,
)
from run_cifar_vgg16_lt_grid_cifar10 import (  # noqa: E402
    load_snn as load_snn_vgg,
    resolve_checkpoint,
)

SCRATCH = Path("/scratch/gs14/sl9144/snn_results")
SIGMA = 4.0
SEEDS = (40, 41, 42, 43, 44)


def three_reg_cells(arch: str, dataset: str) -> list[tuple[str, int, int, int, Path]]:
    rows = []
    for method in ("l2all", "l2wo", "l1wo"):
        for seed in SEEDS:
            rows.append(
                (
                    method,
                    16,
                    16,
                    seed,
                    SCRATCH
                    / "cifar_unified_baseline_evalseed0"
                    / dataset
                    / f"{arch}_{method}"
                    / f"seed{seed}",
                )
            )
            for test_t in (4, 8):
                rows.append(
                    (
                        method,
                        16,
                        test_t,
                        seed,
                        SCRATCH
                        / "cifar_l16_t48_evalseed0"
                        / dataset
                        / f"{arch}_{method}"
                        / f"seed{seed}"
                        / f"T{test_t}",
                    )
                )
    return rows


def cells(arch: str, dataset: str) -> list[tuple[str, int, int, int, Path]]:
    if not (arch == "vgg16" and dataset == "cifar10"):
        return three_reg_cells(arch, dataset)
    rows = []
    for method in ("l2all", "l2wo", "l1wo"):
        for seed in SEEDS:
            rows.append(
                (
                    method,
                    16,
                    16,
                    seed,
                    SCRATCH
                    / "cifar_unified_baseline_evalseed0"
                    / "cifar10"
                    / f"vgg16_{method}"
                    / f"seed{seed}",
                )
            )
    for method in ("l2all", "l2wo", "l1wo", "lambda"):
        for seed in SEEDS:
            for test_t in (4, 8):
                rows.append(
                    (
                        method,
                        16,
                        test_t,
                        seed,
                        SCRATCH
                        / "cifar_l16_t48_evalseed0"
                        / "cifar10"
                        / f"vgg16_{method}"
                        / f"seed{seed}"
                        / f"T{test_t}",
                    )
                )
    for quant_l in (4, 8, 32):
        rows.append(
            (
                "lambda",
                quant_l,
                quant_l,
                42,
                SCRATCH
                / "cifar_fgmneu_lambda_lscale_seed42"
                / "cifar10"
                / f"vgg16_lambda_L{quant_l}"
                / "seed42",
            )
        )
    return rows


def has_sigma(path: Path, sigma: float = SIGMA) -> bool:
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if abs(float(row["sigma"]) - sigma) < 1e-6:
                return True
    return False


def insert_sigma(path: Path, new_row: dict) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields:
        raise RuntimeError(f"empty sweep: {path}")
    extra = {key: "" for key in fields}
    for key, value in new_row.items():
        if key in extra:
            extra[key] = value
    rows.append(extra)
    rows.sort(key=lambda row: float(row["sigma"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="vgg16", choices=("vgg16", "resnet18"))
    parser.add_argument("--dataset", default="cifar10", choices=("cifar10", "cifar100"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    parser.add_argument("--dry-resolve", action="store_true")
    args = parser.parse_args()
    device = None
    pin = False
    for method, quant_l, test_t, seed, out in cells(args.arch, args.dataset):
        test_csv = out / "test_sweep.csv"
        val_csv = out / "val_sweep.csv"
        if has_sigma(test_csv) and has_sigma(val_csv):
            print(f"[SKIP] {test_csv}", flush=True)
            continue
        if not test_csv.is_file() or not val_csv.is_file():
            print(f"[MISSING SWEEP] {out}", flush=True)
            continue
        if args.arch == "vgg16" and method == "lambda":
            checkpoint = resolve_checkpoint(args.dataset, method, quant_l, seed)
        else:
            checkpoint = resolve_ckpt(args.arch, args.dataset, method, seed)
        print(
            f"[SIGMA4] {args.arch} {args.dataset} {method} L={quant_l} T={test_t} seed{seed}\n"
            f"       {checkpoint}",
            flush=True,
        )
        if args.dry_resolve:
            continue
        if device is None:
            device = get_torch_device(args.device)
            pin = device.type == "cuda"
        if args.arch == "vgg16" and method == "lambda":
            model = load_snn_vgg(checkpoint, device, args.dataset, quant_l, test_t)
        else:
            model = load_snn_l16(checkpoint, device, args.arch, args.dataset, test_t)
        val_rows = sweep(model, val_loader(args, pin), device, "val", args.eval_seed, [SIGMA])
        test_rows = sweep(model, test_loader(args, pin), device, "test", args.eval_seed, [SIGMA])
        insert_sigma(val_csv, val_rows[0])
        insert_sigma(test_csv, test_rows[0])
        card_path = out / "scorecard.json"
        if card_path.is_file():
            card = json.loads(card_path.read_text())
            card["test_sigma4"] = float(test_rows[0]["accuracy"])
            card["val_sigma4"] = float(val_rows[0]["accuracy"])
            card_path.write_text(json.dumps(card, indent=2) + "\n")
        print(
            f"[DONE] {method} L={quant_l} T={test_t} seed{seed} "
            f"s4={float(test_rows[0]['accuracy']):.2f}",
            flush=True,
        )
        del model


if __name__ == "__main__":
    main()
