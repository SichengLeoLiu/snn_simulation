#!/usr/bin/env python3
"""Insert the missing post-IF sigma=4 point into existing VGG-16 CIFAR-10 sweeps.

Does not retrain and does not repeat sigma in {0,1,2,3,5}. Cells that already
contain sigma=4 (the 0.25-step L=T scans, and TA-MNE-U at L=T=16) are skipped.

Fills:
  L=16, T=16, seeds 40-44: L2-all, L2-wo, L1-wo
  L=16, T=4 and T=8, seeds 40-44: L2-all, L2-wo, L1-wo, TA-MNE-U
  L=T in {4,8,32}, seed 42: TA-MNE-U
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

from run_cifar_unified_baseline_evalseed0 import (  # noqa: E402
    EVAL_SEED,
    get_torch_device,
    sweep,
    test_loader,
    val_loader,
)
from run_cifar_vgg16_lt_grid_cifar10 import (  # noqa: E402
    resolve_checkpoint,
    load_snn,
)

SCRATCH = Path("/scratch/gs14/sl9144/snn_results")
SIGMA = 4.0
SEEDS = (40, 41, 42, 43, 44)


def cells() -> list[tuple[str, int, int, int, Path]]:
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
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    parser.add_argument("--dry-resolve", action="store_true")
    args = parser.parse_args()
    args.dataset = "cifar10"
    device = None
    pin = False
    for method, quant_l, test_t, seed, out in cells():
        test_csv = out / "test_sweep.csv"
        val_csv = out / "val_sweep.csv"
        if has_sigma(test_csv) and has_sigma(val_csv):
            print(f"[SKIP] {test_csv}", flush=True)
            continue
        if not test_csv.is_file() or not val_csv.is_file():
            print(f"[MISSING SWEEP] {out}", flush=True)
            continue
        checkpoint = resolve_checkpoint("cifar10", method, quant_l, seed)
        print(
            f"[SIGMA4] {method} L={quant_l} T={test_t} seed{seed}\n       {checkpoint}",
            flush=True,
        )
        if args.dry_resolve:
            continue
        if device is None:
            device = get_torch_device(args.device)
            pin = device.type == "cuda"
        model = load_snn(checkpoint, device, "cifar10", quant_l, test_t)
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
