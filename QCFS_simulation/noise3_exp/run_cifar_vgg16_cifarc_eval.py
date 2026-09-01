#!/usr/bin/env python3
"""Eval existing CIFAR VGG-16 checkpoints on CIFAR-10-C / CIFAR-100-C.

Methods (reuse trained ckpts, do not retrain):
    mne       Old MNE-L2, detach λ, rc=1e-4
    l2wo      optimizer weights-only WD=5e-4
    l2all     optimizer WD=5e-4 (all params)
    onesided  frozen One-sided, α=4, τ=0.5, β=5e-4, r_max=8

Protocol matches the Gaussian-noise paper runs except the perturbation
is Hendrycks corruption on the raw image, not post-IF Gaussian:
    SNN T=16, rate_uniform, spike_schedule=normal, sigma=0.

CIFAR-C must already sit on disk. Gadi gpuvolta has no network.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Models import modelpool  # noqa: E402
from Models.VGG import remap_legacy_vgg_state_dict  # noqa: E402
from run_cifar_vgg16_onesided_q_assignment_ablation import (  # noqa: E402
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
    LVAL,
    TEST_T,
    write_csv,
)
from utils import get_torch_device  # noqa: E402

ARCH = "vgg16"
SEVERITIES = (1, 2, 3, 4, 5)
SAMPLES_PER_SEVERITY = 10000
CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)
METHODS = ("mne", "l2wo", "l2all", "onesided")
METHOD_LABEL = {
    "mne": "MNE-L2",
    "l2wo": "L2-wo",
    "l2all": "L2-all",
    "onesided": "One-sided",
}
CKPT_ROOT_DEFAULT = Path("/home/595/sl9144/codes/snn_simulation/QCFS_simulation")
ONESIDED_ROOT_DEFAULT = Path(
    "/scratch/gs14/sl9144/snn_results/cifar_vgg16_onesided_q_assignment_ablation_seed42"
)
CIFAR_C_ROOT_DEFAULT = Path("/scratch/gs14/sl9144/datasets")
FIVE_REGS_SCRATCH = Path("/scratch/gs14/sl9144/snn_results")


class CIFARCorruption(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index])
        return self.transform(image), int(self.labels[index])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar_vgg16_cifarc_seed42",
    )
    parser.add_argument("--ckpt-root", type=Path, default=CKPT_ROOT_DEFAULT)
    parser.add_argument("--onesided-root", type=Path, default=ONESIDED_ROOT_DEFAULT)
    parser.add_argument(
        "--cifar-c-root",
        type=Path,
        default=Path(os.environ.get("CIFAR_C_ROOT", str(CIFAR_C_ROOT_DEFAULT))),
    )
    parser.add_argument(
        "--cifar-root",
        type=Path,
        default=Path(os.path.expanduser(os.environ.get("CIFAR_ROOT", "~/datasets"))),
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    if not args.summarize and not args.check_data and args.method is None:
        parser.error("specify --method, --summarize, or --check-data")
    return args


def mean_std(dataset: str):
    if dataset == "cifar10":
        return CIFAR10_MEAN, CIFAR10_STD
    return CIFAR100_MEAN, CIFAR100_STD


def eval_transform(dataset: str):
    mean, std = mean_std(dataset)
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def cifarc_dir(root: Path, dataset: str) -> Path:
    name = "CIFAR-10-C" if dataset == "cifar10" else "CIFAR-100-C"
    candidates = [
        root / name,
        root / name.lower(),
        root,
    ]
    for path in candidates:
        if (path / "labels.npy").is_file() and (path / f"{CORRUPTIONS[0]}.npy").is_file():
            return path
    raise FileNotFoundError(
        f"CIFAR-C for {dataset} not found under {root}. "
        "Download on login/copyq first: bash noise3_exp/download_cifar_c.sh"
    )


def verify_cifarc(root: Path, dataset: str) -> Path:
    path = cifarc_dir(root, dataset)
    labels = np.load(path / "labels.npy", mmap_mode="r")
    if labels.shape[0] != SEVERITIES[-1] * SAMPLES_PER_SEVERITY:
        raise ValueError(f"{path}/labels.npy has shape {labels.shape}")
    for name in CORRUPTIONS:
        arr = np.load(path / f"{name}.npy", mmap_mode="r")
        if arr.shape != (labels.shape[0], 32, 32, 3):
            raise ValueError(f"{path}/{name}.npy has shape {arr.shape}")
    print(f"[OK] {dataset} CIFAR-C at {path}", flush=True)
    return path


def cfg_dir(args) -> Path:
    return args.out_root / args.dataset / args.method


def mneablate_names(dataset: str, variant: str, seed: int) -> list[str]:
    rc = "rc0p0001" if variant == "old_detach" else "rcnone"
    base = f"{ARCH}_L[{LVAL}]_mneablate_{dataset}_{variant}_{rc}_seed{seed}_L{LVAL}"
    return [f"{base}_trainT0.pth", f"{base}.pth"]


def existing(path: Path) -> Path | None:
    return path if path.is_file() else None


def find_mneablate(args, variant: str) -> Path | None:
    names = mneablate_names(args.dataset, variant, args.seed)
    roots = [
        args.ckpt_root / f"{args.dataset}-checkpoints",
        ROOT / f"{args.dataset}-checkpoints",
        FIVE_REGS_SCRATCH / f"{args.dataset}_{ARCH}_five_regs_sigma0_5_5seed",
        FIVE_REGS_SCRATCH / f"{args.dataset}_{ARCH}_four_regs_sigma0_3_5seed",
    ]
    for root in roots:
        for name in names:
            hit = existing(root / name)
            if hit is not None:
                return hit
    return None


def find_onesided(args) -> Path | None:
    folder = "qabl_risk_a4_tau0.5_b0.0005_rmax8"
    name = f"{ARCH}_L[{LVAL}]_{folder}_seed{args.seed}_L{LVAL}_trainT0.pth"
    candidates = [
        args.onesided_root / args.dataset / folder / "checkpoints" / name,
        args.onesided_root / args.dataset / folder / name,
        ROOT.parent
        / "important_results"
        / "cifar_vgg16_onesided_q_assignment_ablation_seed42"
        / args.dataset
        / folder
        / "checkpoints"
        / name,
        FIVE_REGS_SCRATCH
        / "cifar_ga_mne_seed42"
        / ARCH
        / args.dataset
        / f"gadiag_{ARCH}_onesided"
        / "checkpoints"
        / f"{ARCH}_L[{LVAL}]_gadiag_{ARCH}_onesided_seed{args.seed}_L{LVAL}_trainT0.pth",
    ]
    for path in candidates:
        hit = existing(path)
        if hit is not None:
            return hit
    return None


def resolve_ckpt(args) -> Path:
    mapping = {
        "mne": lambda: find_mneablate(args, "old_detach"),
        "l2wo": lambda: find_mneablate(args, "weight_decay_weights_only"),
        "l2all": lambda: find_mneablate(args, "weight_decay"),
        "onesided": lambda: find_onesided(args),
    }
    ckpt = mapping[args.method]()
    if ckpt is None:
        raise FileNotFoundError(
            f"missing {METHOD_LABEL[args.method]} ckpt for {args.dataset} seed={args.seed}"
        )
    return ckpt


def load_model(ckpt: Path, device, dataset: str):
    model = modelpool(ARCH, dataset)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(remap_legacy_vgg_state_dict(state), strict=True)
    model.set_L(LVAL)
    model.set_T(TEST_T)
    model.set_mode("rate_uniform")
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule("normal")
    if hasattr(model, "set_first_layer_input_noise_position"):
        model.set_first_layer_input_noise_position("post_input_if")
    if hasattr(model, "set_first_layer_input_noise_type"):
        model.set_first_layer_input_noise_type("gaussian")
    if hasattr(model, "set_first_layer_input_noise_sigma"):
        model.set_first_layer_input_noise_sigma(0.0)
    return model.to(device).eval()


def evaluate_acc(model, loader, device) -> float:
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            if outputs.dim() == 3:
                outputs = outputs.mean(0)
            correct += int(outputs.argmax(dim=1).eq(labels).sum().item())
            total += int(labels.numel())
    return 100.0 * correct / total


def make_loader(dataset, args, pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )


def clean_loader(args, pin_memory: bool) -> DataLoader:
    cls = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
    data = cls(
        str(args.cifar_root),
        train=False,
        transform=eval_transform(args.dataset),
        download=False,
    )
    return make_loader(data, args, pin_memory)


def corruption_loader(cdir: Path, corruption: str, severity: int, args, pin_memory: bool):
    images = np.load(cdir / f"{corruption}.npy", mmap_mode="r")
    labels = np.load(cdir / "labels.npy", mmap_mode="r")
    start = (severity - 1) * SAMPLES_PER_SEVERITY
    stop = start + SAMPLES_PER_SEVERITY
    ds = CIFARCorruption(
        np.array(images[start:stop]),
        np.array(labels[start:stop]),
        eval_transform(args.dataset),
    )
    return make_loader(ds, args, pin_memory)


def load_done(path: Path) -> dict[tuple[str, int], dict]:
    if not path.is_file():
        return {}
    done = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            done[(row["corruption"], int(row["severity"]))] = row
    return done


def row_mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def scorecard(args, ckpt: Path, rows: list[dict]) -> dict:
    corrupt_rows = [row for row in rows if row["corruption"] != "clean"]
    clean = next(row for row in rows if row["corruption"] == "clean")
    by_sev = {}
    for severity in SEVERITIES:
        subset = [row for row in corrupt_rows if int(row["severity"]) == severity]
        by_sev[f"severity_{severity}"] = row_mean(subset, "accuracy")
    by_c = {}
    for name in CORRUPTIONS:
        subset = [row for row in corrupt_rows if row["corruption"] == name]
        by_c[name] = row_mean(subset, "accuracy")
    return {
        "dataset": args.dataset,
        "method": args.method,
        "label": METHOD_LABEL[args.method],
        "arch": ARCH,
        "seed": args.seed,
        "checkpoint": str(ckpt),
        "clean": float(clean["accuracy"]),
        "mCA": row_mean(corrupt_rows, "accuracy"),
        "n_corrupt_evals": len(corrupt_rows),
        **by_sev,
        "per_corruption": by_c,
    }


def summarize(out_root: Path) -> None:
    cards = []
    for path in sorted(out_root.glob("*/*/scorecard.json")):
        cards.append(json.loads(path.read_text()))
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    print(
        f"{'dataset':<10} {'method':<10} {'clean':>8} {'mCA':>8} "
        f"{'s1':>8} {'s3':>8} {'s5':>8}"
    )
    table = []
    for card in cards:
        print(
            f"{card['dataset']:<10} {card['method']:<10} "
            f"{card['clean']:8.2f} {card['mCA']:8.2f} "
            f"{card['severity_1']:8.2f} {card['severity_3']:8.2f} "
            f"{card['severity_5']:8.2f}"
        )
        flat = {
            "dataset": card["dataset"],
            "method": card["method"],
            "label": card["label"],
            "seed": card["seed"],
            "clean": card["clean"],
            "mCA": card["mCA"],
            "severity_1": card["severity_1"],
            "severity_2": card["severity_2"],
            "severity_3": card["severity_3"],
            "severity_4": card["severity_4"],
            "severity_5": card["severity_5"],
            "checkpoint": card["checkpoint"],
        }
        for name, value in card.get("per_corruption", {}).items():
            flat[name] = value
        table.append(flat)
    write_csv(out_root / "cifarc_ranking.csv", table)
    print(f"Wrote {out_root / 'cifarc_ranking.csv'}")


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(args.out_root)
        return
    if args.check_data:
        verify_cifarc(args.cifar_c_root, args.dataset)
        return

    cdir = verify_cifarc(args.cifar_c_root, args.dataset)
    ckpt = resolve_ckpt(args)
    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "cifarc_raw.csv"
    done = load_done(csv_path)
    expected = [("clean", 0)] + [(c, s) for c in CORRUPTIONS for s in SEVERITIES]
    print(
        f"[INFO] {args.dataset} {METHOD_LABEL[args.method]} seed={args.seed} "
        f"T={TEST_T} rate_uniform  ckpt={ckpt}",
        flush=True,
    )

    def write_outputs(row_map: dict) -> None:
        ordered = [row_map[key] for key in expected]
        write_csv(csv_path, ordered)
        card = scorecard(args, ckpt, ordered)
        (out / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
        print(
            json.dumps(
                {k: card[k] for k in ("dataset", "method", "clean", "mCA")},
                indent=2,
            ),
            flush=True,
        )
        print(f"Wrote {out}", flush=True)

    if all(key in done for key in expected):
        print("[SKIP] all CIFAR-C cells already measured", flush=True)
        write_outputs(done)
        return

    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = load_model(ckpt, device, args.dataset)
    rows = list(done.values())

    if ("clean", 0) not in done:
        acc = evaluate_acc(model, clean_loader(args, pin), device)
        row = {
            "dataset": args.dataset,
            "method": args.method,
            "seed": str(args.seed),
            "corruption": "clean",
            "severity": "0",
            "accuracy": f"{acc:.6f}",
            "n_samples": str(SAMPLES_PER_SEVERITY),
        }
        rows.append(row)
        done[("clean", 0)] = row
        write_csv(csv_path, rows)
        print(f"clean acc={acc:.2f}", flush=True)
    else:
        print(f"[SKIP] clean acc={float(done[('clean', 0)]['accuracy']):.2f}", flush=True)

    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            key = (corruption, severity)
            if key in done:
                print(
                    f"[SKIP] {corruption} s{severity} "
                    f"acc={float(done[key]['accuracy']):.2f}",
                    flush=True,
                )
                continue
            loader = corruption_loader(cdir, corruption, severity, args, pin)
            acc = evaluate_acc(model, loader, device)
            row = {
                "dataset": args.dataset,
                "method": args.method,
                "seed": str(args.seed),
                "corruption": corruption,
                "severity": str(severity),
                "accuracy": f"{acc:.6f}",
                "n_samples": str(SAMPLES_PER_SEVERITY),
            }
            rows.append(row)
            done[key] = row
            write_csv(csv_path, rows)
            print(f"{corruption} s{severity} acc={acc:.2f}", flush=True)

    write_outputs(done)


if __name__ == "__main__":
    main()
