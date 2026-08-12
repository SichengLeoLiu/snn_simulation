#!/usr/bin/env python3
"""
CIFAR VGG16: run explicit L2 (manual_l2) and MNE-L2 with CE/reg grad probes
at early / mid / late epochs.

Writes:
  <out-dir>/<method>/..._grad_probe.csv  (via main_train --grad_probe)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARCH = "vgg16"
LVAL = 16
EPOCHS = int(os.environ.get("CIFAR_EPOCHS", "300"))
LR = 0.1
BATCH = int(os.environ.get("CIFAR_BATCH", "128"))
NUM_WORKERS = int(os.environ.get("CIFAR_NUM_WORKERS", "8"))

METHOD_SPECS = {
    "manual_l2": {
        "regularizer": "manual_l2",
        "reg_coeff": 2.5e-4,  # matches SGD wd=5e-4 gradient scale on weights
        "weight_decay": 0.0,
        "extra": [],
        "label": "explicit-L2-weights",
    },
    "mne_l2": {
        "regularizer": "mne_l2",
        "reg_coeff": 1e-4,
        "weight_decay": 0.0,
        "extra": ["--mne_detach_lambda"],
        "label": "MNE-L2-standard",
    },
}


def _run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def train_one(
    dataset: str,
    method: str,
    seed: int,
    out_dir: Path,
    epochs: int,
    probe_epochs: str,
) -> None:
    spec = METHOD_SPECS[method]
    suffix = f"gradprobe_{spec['label']}_seed{seed}_L{LVAL}"
    ckpt_dir = ROOT / f"{dataset}-checkpoints"
    cmd = [
        sys.executable,
        "-u",
        "main_train.py",
        "-data",
        dataset,
        "-arch",
        ARCH,
        "-T",
        "0",
        "-L",
        str(LVAL),
        "--epochs",
        str(epochs),
        "-lr",
        str(LR),
        "-b",
        str(BATCH),
        "-j",
        str(NUM_WORKERS),
        "--seed",
        str(seed),
        "--regularizer",
        spec["regularizer"],
        "--reg_coeff",
        str(spec["reg_coeff"]),
        "--weight_decay",
        str(spec["weight_decay"]),
        "-suffix",
        suffix,
        "--ckpt-save-mode",
        "last",
        "--grad_probe",
        "--grad_probe_epochs",
        probe_epochs,
        "--grad_probe_csv",
        str(out_dir / method / f"seed{seed}_grad_probe.csv"),
    ] + list(spec["extra"])
    (out_dir / method).mkdir(parents=True, exist_ok=True)
    _run(cmd, cwd=ROOT)
    # Also copy default-named probe if any leftover
    for p in ckpt_dir.glob(f"*{suffix}*_grad_probe.csv"):
        dest = out_dir / method / p.name
        if not dest.exists():
            dest.write_bytes(p.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["manual_l2", "mne_l2"],
        choices=sorted(METHOD_SPECS.keys()),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--grad-probe-epochs",
        default="auto",
        help="Passed to main_train --grad_probe_epochs (auto=early/mid/late)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for probe CSVs",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = (
            ROOT.parent
            / "important_results"
            / f"{args.dataset}_{ARCH}_grad_probe_l2_mne_seed{'_'.join(map(str, args.seeds))}"
        )
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        for method in args.methods:
            train_one(
                args.dataset,
                method,
                seed,
                out_dir,
                args.epochs,
                args.grad_probe_epochs,
            )
    print(f"done. probes under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
