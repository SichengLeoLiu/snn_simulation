#!/usr/bin/env python3
"""Frozen onesided MNE q-assignment ablation (no test-curve retuning).

Frozen recipe
-------------
    q_OS = 1 + α max(r̂ − τ, 0),  α=4, τ=0.5, r_max=8, β=5e-4
    warmup: alpha starts at epoch 30, 50-epoch linear ramp
    ANN train T=0; SNN eval T=16, rate_uniform, post_input_if

Four assignments share the same β and the same onesided q_OS formula:

    identity  L2-wo               q_{l,c} = 1
    strength  Strength-only       q_{l,c} = q̄_OS  (param-weighted mean over IF channels)
    shuffle   Shuffled One-sided  permute q within each layer, keep the q values
    risk      One-sided MNE       true risk-to-channel map (green line)

Every epoch logs q_mean/std/min/max, P(r̂>τ), ||∇R||, and ANN train/test.
SNN clean / AUC[0,5] / AUC[3,5] are measured once after training on the
selected checkpoint. Per-epoch SNN sweeps are ~10 min each and would
exceed a 24h walltime; pass --snn-eval-every N only if you explicitly
want mid-training snapshots.

Selection/logging uses a 5k train-holdout val sweep. Test is recorded only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from Models import modelpool  # noqa: E402
from Models.VGG import remap_legacy_vgg_state_dict  # noqa: E402
from utils import get_torch_device, seed_all  # noqa: E402
import measure_vgg16_horowitz_energy as energy  # noqa: E402


ARCH = "vgg16"
LVAL = 16
TEST_T = 16
EPOCHS = 300
LR = 0.1
VAL_SIZE = 5000
VAL_SPLIT_SEED = 0
SIGMAS = [i / 4 for i in range(0, 21)]  # 0, 0.25, ..., 5
HIGH_NOISE_MIN = 3.0
ALPHA = 4.0
TAU = 0.5
BETA = 5e-4
RISK_MIN = 0.5
RISK_MAX = 8.0
ALPHA_START = 30
ALPHA_WARMUP = 50

ASSIGNMENTS = {
    "identity": "L2-wo",
    "strength": "Strength-only",
    "shuffle": "Shuffled One-sided",
    "risk": "One-sided MNE",
}

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = [n / 255.0 for n in [129.3, 124.1, 112.4]]
CIFAR100_STD = [n / 255.0 for n in [68.2, 65.4, 70.4]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assignment",
        choices=tuple(ASSIGNMENTS),
        default=None,
        help="identity | strength | shuffle | risk",
    )
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent
        / "important_results"
        / "cifar_vgg16_onesided_q_assignment_ablation_seed42",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    if not args.summarize and args.assignment is None:
        parser.error("specify --assignment or --summarize")
    return args


def config_name(assignment: str) -> str:
    return f"qabl_{assignment}_a4_tau0.5_b0.0005_rmax8"


def suffix(args) -> str:
    return (
        f"{config_name(args.assignment)}_seed{args.seed}_L{LVAL}_trainT0"
    )


def ckpt_filename(args) -> str:
    return f"{ARCH}_L[{LVAL}]_{suffix(args)}.pth"


def cfg_dir(args) -> Path:
    return args.out_root / args.dataset / config_name(args.assignment)


def ckpt_path(args) -> Path:
    return cfg_dir(args) / "checkpoints" / ckpt_filename(args)


def trapz(xs: list[float], ys: list[float]) -> float:
    area = 0.0
    for i in range(1, len(xs)):
        area += 0.5 * (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1])
    return area


def auc_range(rows: list[dict], lo: float, hi: float) -> float:
    xs = []
    ys = []
    for row in rows:
        sigma = float(row["sigma"])
        if lo - 1e-12 <= sigma <= hi + 1e-12:
            xs.append(sigma)
            ys.append(float(row["accuracy"]))
    if len(xs) < 2:
        raise ValueError(f"need ≥2 points in [{lo}, {hi}], got {xs}")
    return trapz(xs, ys)


def acc_at(rows: list[dict], sigma: float) -> float:
    for row in rows:
        if abs(float(row["sigma"]) - sigma) < 1e-9:
            return float(row["accuracy"])
    raise KeyError(sigma)


def eval_dataset(dataset: str, *, train: bool):
    root = os.path.expanduser(os.environ.get("CIFAR_ROOT", "~/datasets"))
    if dataset == "cifar10":
        mean, std, cls = CIFAR10_MEAN, CIFAR10_STD, datasets.CIFAR10
    else:
        mean, std, cls = CIFAR100_MEAN, CIFAR100_STD, datasets.CIFAR100
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return cls(root, train=train, transform=transform, download=False)


def val_loader(args, pin_memory: bool) -> DataLoader:
    train_ds = eval_dataset(args.dataset, train=True)
    g = torch.Generator().manual_seed(VAL_SPLIT_SEED)
    perm = torch.randperm(len(train_ds), generator=g).tolist()
    subset = Subset(train_ds, perm[:VAL_SIZE])
    return DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )


def test_loader(args, pin_memory: bool) -> DataLoader:
    return DataLoader(
        eval_dataset(args.dataset, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )


def load_model(ckpt: Path, device, dataset: str):
    model = modelpool(ARCH, dataset)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(remap_legacy_vgg_state_dict(state), strict=True)
    model.set_L(LVAL)
    model.set_T(TEST_T)
    model.set_mode("rate_uniform")
    model.set_spike_schedule("normal")
    if hasattr(model, "set_first_layer_input_noise_position"):
        model.set_first_layer_input_noise_position("post_input_if")
    if hasattr(model, "set_first_layer_input_noise_type"):
        model.set_first_layer_input_noise_type("gaussian")
    return model.to(device).eval()


def sweep(model, loader, device, split: str, seed: int) -> list[dict]:
    rows = []
    for sigma in SIGMAS:
        seed_all(seed)
        model.set_first_layer_input_noise_sigma(float(sigma))
        result = energy.evaluate(model, loader, device, TEST_T, 0)
        use_energy = "yes" if abs(sigma) < 1e-12 else "no"
        row = {
            "split": split,
            "sigma": f"{sigma:g}",
            "accuracy": f"{result['accuracy']:.6f}",
            "if_firing_density": f"{result['if_firing_density']:.6f}",
            "energy_mJ": f"{result['energy_mJ']:.8f}",
            "use_for_energy": use_energy,
            "n_samples": result["n_samples"],
        }
        rows.append(row)
        extra = "" if use_energy == "yes" else " [diag energy]"
        print(
            f"{split:<5} sigma={sigma:g} acc={result['accuracy']:.2f} "
            f"fire={result['if_firing_density']:.4f} "
            f"E={result['energy_mJ']:.4f} mJ{extra}",
            flush=True,
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def snn_metrics(rows: list[dict], prefix: str) -> dict:
    return {
        f"{prefix}_clean": acc_at(rows, 0.0),
        f"{prefix}_sigma5": acc_at(rows, 5.0),
        f"{prefix}_auc_full": auc_range(rows, 0.0, 5.0),
        f"{prefix}_auc_high": auc_range(rows, HIGH_NOISE_MIN, 5.0),
        f"{prefix}_clean_energy_mJ": float(rows[0]["energy_mJ"]),
        f"{prefix}_clean_fire": float(rows[0]["if_firing_density"]),
    }


def train(args) -> Path:
    out = cfg_dir(args)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_path(args)
    epoch_log = out / "epoch_log.csv"
    if ckpt.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {ckpt}", flush=True)
        return ckpt
    if args.test_only:
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        return ckpt
    cmd = [
        sys.executable,
        str(ROOT / "main_train.py"),
        "-data", args.dataset,
        "-arch", ARCH,
        "-L", str(LVAL),
        "-T", "0",
        "--epochs", str(args.epochs),
        "-lr", str(LR),
        "-b", str(args.batch_size),
        "-j", str(args.workers),
        "--seed", str(args.seed),
        "--device", args.device,
        "--spike_schedule", "normal",
        "--ckpt-save-mode", "best",
        "--ckpt-dir", str(ckpt_dir),
        "-suffix", suffix(args),
        "--regularizer", "calibrated_mne_l2",
        "--weight_decay", "0",
        "--reg_coeff", str(BETA),
        "--calibrated_mne_alpha", str(ALPHA),
        "--calibrated_mne_onesided",
        "--calibrated_mne_tau", str(TAU),
        "--calibrated_mne_risk_min", str(RISK_MIN),
        "--calibrated_mne_risk_max", str(RISK_MAX),
        "--calibrated_mne_alpha_start_epoch", str(ALPHA_START),
        "--calibrated_mne_alpha_warmup_epochs", str(ALPHA_WARMUP),
        "--calibrated_mne_q_assignment", args.assignment,
        "--epoch_log_csv", str(epoch_log),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"training finished but missing {ckpt}")
    return ckpt


def scorecard(val_rows, test_rows, args, ckpt: Path) -> dict:
    card = {
        "config": config_name(args.assignment),
        "label": ASSIGNMENTS[args.assignment],
        "assignment": args.assignment,
        "dataset": args.dataset,
        "seed": args.seed,
        "alpha": ALPHA,
        "tau": TAU,
        "beta": BETA,
        "risk_max": RISK_MAX,
        "checkpoint": str(ckpt),
        "epoch_log": str(cfg_dir(args) / "epoch_log.csv"),
    }
    card.update(snn_metrics(val_rows, "val"))
    card.update(snn_metrics(test_rows, "test"))
    return card


def summarize(out_root: Path) -> None:
    cards = []
    for path in sorted(out_root.glob("*/*/scorecard.json")):
        cards.append(json.loads(path.read_text()))
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    print(
        f"{'dataset':<10} {'assign':<10} {'label':<20} "
        f"{'val0':>7} {'valAUC':>8} {'valHi':>8} "
        f"{'test0':>7} {'testAUC':>8} {'testHi':>8}"
    )
    for card in cards:
        print(
            f"{card['dataset']:<10} {card['assignment']:<10} {card['label']:<20} "
            f"{card['val_clean']:7.2f} {card['val_auc_full']:8.2f} {card['val_auc_high']:8.2f} "
            f"{card['test_clean']:7.2f} {card['test_auc_full']:8.2f} {card['test_auc_high']:8.2f}"
        )
    write_csv(out_root / "ablation_ranking.csv", cards)
    print(f"Wrote {out_root / 'ablation_ranking.csv'}")


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(args.out_root)
        return

    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] {args.dataset} {ASSIGNMENTS[args.assignment]} "
        f"({args.assignment}) seed={args.seed}  "
        f"frozen α={ALPHA} τ={TAU} rmax={RISK_MAX} β={BETA}",
        flush=True,
    )
    print(
        "[INFO] epoch_log: q stats / P(r>τ) / ||∇R|| / ANN acc. "
        "SNN clean+AUC after training on the selected ckpt.",
        flush=True,
    )
    ckpt = train(args)
    copied = out / ckpt.name
    if Path(ckpt).resolve() != copied.resolve():
        shutil.copy2(ckpt, copied)
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = load_model(ckpt, device, args.dataset)
    val_rows = sweep(model, val_loader(args, pin), device, "val", args.seed)
    write_csv(out / "val_sweep.csv", val_rows)
    test_rows = sweep(model, test_loader(args, pin), device, "test", args.seed)
    write_csv(out / "test_sweep.csv", test_rows)
    card = scorecard(val_rows, test_rows, args, ckpt)
    (out / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
