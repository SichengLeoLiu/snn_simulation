#!/usr/bin/env python3
"""Matched-variance noise-family eval for FG-MNE-U CIFAR checkpoints.

Reuse existing CIFAR VGG-16 / ResNet-18 checkpoints. Do not retrain.
Do not write into training directories. Do not retune η from test.

Families share the same target standard deviation σ:

    Gaussian  N(0, σ²)
    Laplace   b = σ/√2   (Var = 2b²)
    Uniform   half-width a = σ√3   (Var = a²/3)

Protocol: T=L=16, rate_uniform, post_input_if, EVAL_SEED=0.
Report official test only. σ grid default 0,1,2,3,5.

ResNet detach is same-map seed42 only. Do not invent 5-seed detach.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Models.layer import (  # noqa: E402
    MATCHED_VARIANCE_NOISE_TYPES,
    sample_matched_variance_noise,
)

LVAL = 16
TEST_T = 16
EVAL_SEED = 0
DEFAULT_SIGMAS = (0.0, 1.0, 2.0, 3.0, 5.0)
DEFAULT_SEEDS = (40, 41, 42, 43, 44)
SCRATCH = Path(os.environ.get("SNN_RESULTS", "/scratch/gs14/sl9144/snn_results"))
HOME_QCFS = Path(
    os.environ.get(
        "HOME_QCFS",
        "/home/595/sl9144/codes/snn_simulation/QCFS_simulation",
    )
)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = [n / 255.0 for n in [129.3, 124.1, 112.4]]
CIFAR100_STD = [n / 255.0 for n in [68.2, 65.4, 70.4]]

METHODS = {
    "l2wo": "L2-wo",
    "detach": "MNE-L2 detach",
    "nodetach": "No-detach MNE",
    "fgmneu": "FG-MNE-U",
}
ARCH_REMAP_VGG = {"vgg16": True, "resnet18": False}


def get_torch_device(device_str: str = "auto") -> torch.device:
    s = (device_str or "auto").strip().lower()
    if s in ("auto", "", "0"):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    return torch.device(s)


def seed_all(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("vgg16", "resnet18"), default=None)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default=None)
    parser.add_argument(
        "--method",
        choices=tuple(METHODS) + ("all",),
        default="all",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--noise-types",
        nargs="+",
        default=list(MATCHED_VARIANCE_NOISE_TYPES),
        choices=list(MATCHED_VARIANCE_NOISE_TYPES),
    )
    parser.add_argument("--sigmas", nargs="+", type=float, default=list(DEFAULT_SIGMAS))
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ckpt", type=Path, default=None, help="Override one checkpoint.")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=SCRATCH / "cifar_fgmneu_noise_family",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-resolve", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    args.sigmas = sorted({float(s) for s in args.sigmas})
    args.noise_types = [str(nt).strip().lower() for nt in args.noise_types]
    if not args.self_check and (args.arch is None or args.dataset is None):
        parser.error("--arch and --dataset are required unless --self-check")
    return args


def seed_list(arch: str, method: str, seeds: list[int]) -> list[int]:
    if arch == "resnet18" and method == "detach":
        keep = [s for s in seeds if s == 42]
        if 42 not in seeds:
            keep = [42]
        extra = [s for s in seeds if s != 42]
        if extra:
            print(
                f"[NOTE] ResNet detach is seed42 only; skip seeds {extra}",
                flush=True,
            )
        return keep or [42]
    return list(seeds)


def method_list(method: str) -> list[str]:
    if method == "all":
        return list(METHODS)
    return [method]


def mneablate_name(dataset: str, variant: str, seed: int) -> str:
    rc = "rcnone" if variant == "weight_decay_weights_only" else "rc0p0001"
    return (
        f"vgg16_L[16]_mneablate_{dataset}_{variant}_{rc}_"
        f"seed{seed}_L16_trainT0.pth"
    )


def ckpt_candidates(arch: str, dataset: str, method: str, seed: int) -> list[Path]:
    paths: list[Path] = []
    extra = os.environ.get("CIFAR_CKPT_ROOT")
    roots = [ROOT, HOME_QCFS]
    if extra:
        roots.insert(0, Path(extra))

    if arch == "vgg16":
        if method == "l2wo":
            name = mneablate_name(dataset, "weight_decay_weights_only", seed)
            for root in roots:
                paths.append(root / f"{dataset}-checkpoints" / name)
            if seed == 42:
                paths.append(
                    SCRATCH
                    / "cifar_vgg16_mne_component_ablation_seed42"
                    / dataset
                    / "comp_l2wo_fixed"
                    / "checkpoints"
                    / "vgg16_L[16]_comp_l2wo_fixed_seed42_L16_trainT0.pth"
                )
        elif method == "detach":
            name = mneablate_name(dataset, "old_detach", seed)
            for root in roots:
                paths.append(root / f"{dataset}-checkpoints" / name)
            if seed == 42:
                paths.append(
                    SCRATCH
                    / "cifar_vgg16_mne_component_ablation_seed42"
                    / dataset
                    / "comp_mne_fixed"
                    / "checkpoints"
                    / "vgg16_L[16]_comp_mne_fixed_seed42_L16_trainT0.pth"
                )
        elif method == "nodetach":
            paths.append(
                SCRATCH
                / "cifar_vgg16_nodetach_5seed"
                / dataset
                / "v16_nodetach"
                / f"seed{seed}"
                / "checkpoints"
                / f"vgg16_L[16]_v16_nodetach_seed{seed}_L16_trainT0.pth"
            )
            if seed == 42:
                paths.append(
                    SCRATCH
                    / "cifar_vgg16_mne_component_ablation_seed42"
                    / dataset
                    / "comp_nodetach_fixed"
                    / "checkpoints"
                    / "vgg16_L[16]_comp_nodetach_fixed_seed42_L16_trainT0.pth"
                )
        elif method == "fgmneu":
            paths.append(
                SCRATCH
                / "cifar_mne_nodetach_unmatched_head_l2_seed42"
                / dataset
                / "vgg16_mne_head"
                / f"seed{seed}"
                / "checkpoints"
                / f"vgg16_L[16]_vgg16_mne_head_seed{seed}_L16_trainT0.pth"
            )
    elif arch == "resnet18":
        if method == "l2wo":
            paths.append(
                SCRATCH
                / "cifar_resnet18_four_regs_5seed"
                / dataset
                / "r18_l2wo"
                / f"seed{seed}"
                / "checkpoints"
                / f"resnet18_L[16]_r18_l2wo_seed{seed}_L16_trainT0.pth"
            )
        elif method == "detach":
            paths.append(
                SCRATCH
                / "cifar_resnet18_fair_mne_detach"
                / dataset
                / "r18_mne_resnet_rc1e-4"
                / f"seed{seed}"
                / "checkpoints"
                / f"resnet18_L[16]_r18_mne_resnet_rc1e-4_seed{seed}_L16_trainT0.pth"
            )
        elif method == "nodetach":
            paths.append(
                SCRATCH
                / "cifar_resnet18_nodetach_resnetmap_seed42"
                / dataset
                / "resnet18_mne_body"
                / f"seed{seed}"
                / "checkpoints"
                / f"resnet18_L[16]_resnet18_mne_body_seed{seed}_L16_trainT0.pth"
            )
        elif method == "fgmneu":
            paths.append(
                SCRATCH
                / "cifar_mne_nodetach_unmatched_head_l2_seed42"
                / dataset
                / "resnet18_mne_head"
                / f"seed{seed}"
                / "checkpoints"
                / f"resnet18_L[16]_resnet18_mne_head_seed{seed}_L16_trainT0.pth"
            )
    # Unique, preserve order.
    seen = set()
    out = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def resolve_ckpt(arch: str, dataset: str, method: str, seed: int, override: Path | None) -> Path:
    if override is not None:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"--ckpt missing: {path}")
        return path
    tried = ckpt_candidates(arch, dataset, method, seed)
    for path in tried:
        if path.is_file():
            return path
    lines = "\n".join(f"  {p}" for p in tried) or "  (no candidates)"
    raise FileNotFoundError(
        f"checkpoint not found for {arch} {dataset} {method} seed{seed}:\n{lines}"
    )


def out_dir(args, method: str, seed: int) -> Path:
    return args.out_root / args.dataset / args.arch / method / f"seed{seed}"


def test_loader(args, pin_memory: bool) -> DataLoader:
    root = os.path.expanduser(os.environ.get("CIFAR_ROOT", "~/datasets"))
    if args.dataset == "cifar10":
        mean, std, cls = CIFAR10_MEAN, CIFAR10_STD, datasets.CIFAR10
    else:
        mean, std, cls = CIFAR100_MEAN, CIFAR100_STD, datasets.CIFAR100
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    data = cls(root, train=False, transform=transform, download=False)
    return DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )


def load_model(ckpt: Path, device, arch: str, dataset: str):
    from Models import modelpool
    from Models.VGG import remap_legacy_vgg_state_dict

    model = modelpool(arch, dataset)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if ARCH_REMAP_VGG[arch]:
        state = remap_legacy_vgg_state_dict(state)
    model.load_state_dict(state, strict=True)
    model.set_L(LVAL)
    model.set_T(TEST_T)
    model.set_mode("rate_uniform")
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule("normal")
    model.set_first_layer_input_noise_position("post_input_if")
    model.set_first_layer_input_noise_type("gaussian")
    model.set_first_layer_input_noise_sigma(0.0)
    return model.to(device).eval()


@torch.inference_mode()
def accuracy(model, loader, device) -> tuple[float, int]:
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        if logits.dim() == 3:
            logits = logits.mean(0)
        correct += int(logits.argmax(dim=1).eq(labels).sum().item())
        total += int(labels.numel())
    return 100.0 * correct / max(total, 1), total


def trapz(xs: list[float], ys: list[float]) -> float:
    area = 0.0
    for i in range(1, len(xs)):
        area += 0.5 * (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1])
    return area


def acc_at(rows: list[dict], noise_type: str, sigma: float) -> float:
    for row in rows:
        if row["noise_type"] == noise_type and abs(float(row["sigma"]) - sigma) < 1e-9:
            return float(row["accuracy"])
    raise KeyError((noise_type, sigma))


def auc_of(rows: list[dict], noise_type: str, lo: float, hi: float) -> float:
    xs, ys = [], []
    for row in rows:
        if row["noise_type"] != noise_type:
            continue
        sigma = float(row["sigma"])
        if lo - 1e-12 <= sigma <= hi + 1e-12:
            xs.append(sigma)
            ys.append(float(row["accuracy"]))
    if len(xs) < 2:
        return float("nan")
    return trapz(xs, ys)


def expected_keys(noise_types: list[str], sigmas: list[float]) -> set[tuple[str, float]]:
    return {(nt, float(s)) for nt in noise_types for s in sigmas}


def load_existing(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def self_check() -> int:
    torch.manual_seed(0)
    x = torch.zeros(400_000)
    failed = 0
    for nt in MATCHED_VARIANCE_NOISE_TYPES:
        for sigma in (1.0, 3.0, 5.0):
            torch.manual_seed(0)
            noise = sample_matched_variance_noise(x, sigma, nt)
            mean = float(noise.mean().item())
            var = float(noise.var(unbiased=False).item())
            rel = abs(var - sigma * sigma) / (sigma * sigma)
            ok = abs(mean) < 0.03 and rel < 0.03
            status = "OK" if ok else "FAIL"
            if not ok:
                failed += 1
            print(
                f"[{status}] {nt:8s} σ={sigma:g} mean={mean:+.5f} "
                f"var={var:.5f} target={sigma * sigma:.5f} rel={rel:.4f}",
                flush=True,
            )
    from Models.VGG import _inject_noise_tensor

    x = torch.zeros(8, 3, 8, 8)
    torch.manual_seed(0)
    injected = _inject_noise_tensor(x, 2.0, "gaussian", 0)
    torch.manual_seed(0)
    baseline = x + torch.randn_like(x) * 2.0
    if not torch.allclose(injected, baseline):
        failed += 1
        print("[FAIL] VGG gaussian inject drifted from randn * σ", flush=True)
    else:
        print("[OK] VGG gaussian inject == randn * σ", flush=True)
    if failed:
        print(f"SELF-CHECK FAILED ({failed})", flush=True)
        return 1
    print("SELF-CHECK PASSED", flush=True)
    return 0


def scorecard_from_rows(args, method: str, seed: int, ckpt: Path, rows: list[dict]) -> dict:
    card = {
        "arch": args.arch,
        "dataset": args.dataset,
        "method": method,
        "label": METHODS[method],
        "seed": seed,
        "eval_seed": args.eval_seed,
        "T": TEST_T,
        "L": LVAL,
        "mode": "rate_uniform",
        "noise_position": "post_input_if",
        "checkpoint": str(ckpt),
        "sigmas": [float(s) for s in args.sigmas],
        "noise_types": list(args.noise_types),
    }
    for nt in args.noise_types:
        for sigma in args.sigmas:
            key = "clean" if abs(sigma) < 1e-12 else f"sigma{sigma:g}".replace(".", "p")
            card[f"{nt}_{key}"] = acc_at(rows, nt, sigma)
        card[f"{nt}_auc_full"] = auc_of(rows, nt, min(args.sigmas), max(args.sigmas))
        if min(args.sigmas) <= 3.0 <= max(args.sigmas):
            card[f"{nt}_auc_high"] = auc_of(rows, nt, 3.0, max(args.sigmas))
    return card


def eval_one(args, method: str, seed: int, loader, device) -> dict:
    dest = out_dir(args, method, seed)
    sweep_path = dest / "test_sweep.csv"
    ckpt = resolve_ckpt(args.arch, args.dataset, method, seed, args.ckpt)
    want = expected_keys(args.noise_types, args.sigmas)
    existing = [] if args.force else load_existing(sweep_path)
    have = {
        (row["noise_type"], float(row["sigma"]))
        for row in existing
        if row.get("noise_type") in args.noise_types
    }
    if want <= have:
        print(f"[SKIP] complete {sweep_path}", flush=True)
        rows = [row for row in existing if (row["noise_type"], float(row["sigma"])) in want]
        card = scorecard_from_rows(args, method, seed, ckpt, rows)
        (dest / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
        return card

    print(
        f"[LOAD] {args.arch} {args.dataset} {method} seed{seed}\n       {ckpt}",
        flush=True,
    )
    model = load_model(ckpt, device, args.arch, args.dataset)
    rows_by_key = {
        (row["noise_type"], float(row["sigma"])): row
        for row in existing
        if row.get("noise_type") in args.noise_types
    }
    clean_acc = None
    clean_n = None
    if any(abs(s) < 1e-12 for s in args.sigmas) and not args.force:
        for nt in args.noise_types:
            prev = rows_by_key.get((nt, 0.0))
            if prev is not None:
                clean_acc = float(prev["accuracy"])
                clean_n = int(prev["n_samples"])
                break

    for nt in args.noise_types:
        model.set_first_layer_input_noise_type(nt)
        for sigma in args.sigmas:
            key = (nt, float(sigma))
            if key in rows_by_key and not args.force:
                print(
                    f"[SKIP] {nt} σ={sigma:g} acc={float(rows_by_key[key]['accuracy']):.2f}",
                    flush=True,
                )
                continue
            if abs(sigma) < 1e-12:
                if clean_acc is None:
                    seed_all(args.eval_seed)
                    model.set_first_layer_input_noise_sigma(0.0)
                    clean_acc, clean_n = accuracy(model, loader, device)
                acc, n_samples = clean_acc, clean_n
            else:
                seed_all(args.eval_seed)
                model.set_first_layer_input_noise_sigma(float(sigma))
                acc, n_samples = accuracy(model, loader, device)
            row = {
                "arch": args.arch,
                "dataset": args.dataset,
                "method": method,
                "label": METHODS[method],
                "seed": str(seed),
                "noise_type": nt,
                "sigma": f"{sigma:g}",
                "accuracy": f"{acc:.6f}",
                "n_samples": str(n_samples),
                "eval_seed": str(args.eval_seed),
                "checkpoint": str(ckpt),
            }
            rows_by_key[key] = row
            print(
                f"{args.arch} {args.dataset} {method} seed{seed} "
                f"{nt} σ={sigma:g} acc={acc:.2f}",
                flush=True,
            )

    model.set_first_layer_input_noise_sigma(0.0)
    model.set_first_layer_input_noise_type("gaussian")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows = [rows_by_key[key] for key in sorted(rows_by_key, key=lambda x: (x[0], x[1]))]
    write_csv(sweep_path, rows)
    card = scorecard_from_rows(args, method, seed, ckpt, rows)
    (dest / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
    return card


def summarize(args, cards: list[dict]) -> None:
    if not cards:
        return
    path = args.out_root / f"{args.arch}_{args.dataset}_noise_family_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cards, indent=2) + "\n")
    print(f"[SUMMARY] {path}", flush=True)


def main() -> int:
    args = parse_args()
    if args.self_check:
        return self_check()

    methods = method_list(args.method)
    if args.dry_resolve:
        missing = 0
        for method in methods:
            for seed in seed_list(args.arch, method, args.seeds):
                try:
                    ckpt = resolve_ckpt(args.arch, args.dataset, method, seed, args.ckpt)
                except FileNotFoundError as exc:
                    missing += 1
                    print(f"[MISSING] {args.arch} {args.dataset} {method} seed{seed}\n{exc}", flush=True)
                    continue
                print(
                    f"[OK] {args.arch} {args.dataset} {method} seed{seed}\n     {ckpt}",
                    flush=True,
                )
        return 1 if missing else 0

    pin = str(args.device).startswith("cuda") or args.device == "auto"
    device = get_torch_device(args.device)
    loader = test_loader(args, pin_memory=device.type == "cuda" and pin)
    print(
        f"arch={args.arch} dataset={args.dataset} methods={methods} "
        f"seeds={args.seeds} types={args.noise_types} sigmas={args.sigmas} "
        f"eval_seed={args.eval_seed} device={device} out={args.out_root}",
        flush=True,
    )
    cards = []
    for method in methods:
        for seed in seed_list(args.arch, method, args.seeds):
            cards.append(eval_one(args, method, seed, loader, device))
    summarize(args, cards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
