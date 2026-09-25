#!/usr/bin/env python3
"""Insert sigma=4 into existing T=0 ANN noise sweeps. Eval only.

Covers the four panels of the ANN-vs-SNN figure:
  VGG-16 and ResNet-18, CIFAR-10 and CIFAR-100,
  L2-all, L2-wo, L1-wo, TA-MNE-U, seeds 40-44.
Does not retrain and does not repeat sigma in {0,1,2,3,5}.
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

from run_cifar_ann_t0_evalseed0 import load_ann, resolve_any  # noqa: E402
from run_cifar_unified_baseline_evalseed0 import (  # noqa: E402
    EVAL_SEED,
    get_torch_device,
    sweep,
    test_loader,
    val_loader,
)

SCRATCH = Path("/scratch/gs14/sl9144/snn_results")
SIGMA = 4.0
ARCHS = ("vgg16", "resnet18")
DATASETS = ("cifar10", "cifar100")
METHODS = ("l2all", "l2wo", "l1wo", "lambda")
SEEDS = (40, 41, 42, 43, 44)


def cells():
    rows = []
    for dataset in DATASETS:
        for arch in ARCHS:
            for method in METHODS:
                for seed in SEEDS:
                    rows.append(
                        (
                            arch,
                            dataset,
                            method,
                            seed,
                            SCRATCH
                            / "cifar_ann_t0_evalseed0"
                            / dataset
                            / f"{arch}_{method}"
                            / f"seed{seed}",
                        )
                    )
    return rows


def has_sigma(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if abs(float(row["sigma"]) - SIGMA) < 1e-6:
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
    device = None
    pin = False
    for arch, dataset, method, seed, out in cells():
        args.dataset = dataset
        test_csv = out / "test_sweep.csv"
        val_csv = out / "val_sweep.csv"
        if has_sigma(test_csv) and has_sigma(val_csv):
            print(f"[SKIP] {test_csv}", flush=True)
            continue
        if not test_csv.is_file() or not val_csv.is_file():
            print(f"[MISSING SWEEP] {out}", flush=True)
            continue
        checkpoint = resolve_any(arch, dataset, method, seed)
        print(f"[SIGMA4 ANN] {arch} {dataset} {method} seed{seed}\n       {checkpoint}", flush=True)
        if args.dry_resolve:
            continue
        if device is None:
            device = get_torch_device(args.device)
            pin = device.type == "cuda"
        model = load_ann(checkpoint, device, arch, dataset)
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
            f"[DONE] {arch} {dataset} {method} seed{seed} "
            f"s4={float(test_rows[0]['accuracy']):.2f}",
            flush=True,
        )
        del model


if __name__ == "__main__":
    main()
