#!/usr/bin/env python3
"""CIFAR seed-42 threshold-scaling baseline vs λ-MNE-U.

Two arms, neither retunes η/α/τ from the test curve:

  scale  Eval-only. Take the existing L2-wo checkpoint and multiply every
         IF threshold by a fixed coefficient. Select the scale whose val
         clean accuracy is closest to λ-MNE-U. Test is recorded only.

  grow   Train L2-wo + a simple λ-only regularizer
             R_λ = -mean_l log(λ_l)
         with a pre-specified η_λ grid. Optimizer WD stays at L2-wo 5e-4
         on Conv/Linear weights. Select η_λ on val clean vs λ-MNE-U.

Protocol: T=L=16, rate_uniform, post_input_if, EVAL_SEED=0, 5k val split.
Reports IF thresholds, clean firing density, and Horowitz arithmetic energy.

Do not overwrite L2-wo / FG-MNE-U / λ-MNE-U / ImageNet trees.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Models.layer import IF  # noqa: E402
from lambda_grow import compute_lambda_grow_regularization  # noqa: E402

EPOCHS = 300
LR = 0.1
LVAL = 16
TEST_T = 16

SCRATCH = Path("/scratch/gs14/sl9144/snn_results")
IR = ROOT.parent / "important_results"
SEED = 42
EVAL_NOISE_SEED = 0
L2_WD = 5e-4
DEFAULT_SCALES = (0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
DEFAULT_GROW = (1e-2, 3e-2, 1e-1)
LAYER_MAP = {"vgg16": "legacy", "resnet18": "resnet"}

L2WO_CKPT = {
    "vgg16": (
        SCRATCH
        / "cifar_vgg16_mne_component_ablation_seed42"
        / "{dataset}/comp_l2wo_fixed/checkpoints"
        / "vgg16_L[16]_comp_l2wo_fixed_seed42_L16_trainT0.pth",
        IR
        / "cifar_vgg16_mne_component_ablation_seed42"
        / "{dataset}/comp_l2wo_fixed/checkpoints"
        / "vgg16_L[16]_comp_l2wo_fixed_seed42_L16_trainT0.pth",
    ),
    "resnet18": (
        SCRATCH
        / "cifar_resnet18_four_regs_5seed"
        / "{dataset}/r18_l2wo/seed42/checkpoints"
        / "resnet18_L[16]_r18_l2wo_seed42_L16_trainT0.pth",
        IR
        / "cifar_resnet18_four_regs_5seed"
        / "{dataset}/r18_l2wo/seed42/checkpoints"
        / "resnet18_L[16]_r18_l2wo_seed42_L16_trainT0.pth",
    ),
}
LAMBDA_CKPT = {
    "vgg16": (
        SCRATCH
        / "cifar_fgmneu_grad_ablation_seed42"
        / "{dataset}/vgg16_fgmneu_lambda/seed42/checkpoints"
        / "vgg16_L[16]_vgg16_fgmneu_lambda_seed42_L16_trainT0.pth",
        IR
        / "cifar_fgmneu_grad_ablation_seed42"
        / "{dataset}/vgg16_fgmneu_lambda/seed42/checkpoints"
        / "vgg16_L[16]_vgg16_fgmneu_lambda_seed42_L16_trainT0.pth",
    ),
    "resnet18": (
        SCRATCH
        / "cifar_resnet18_fgmneu_lambda_seed42"
        / "{dataset}/resnet18_fgmneu_lambda/seed42/checkpoints"
        / "resnet18_L[16]_resnet18_fgmneu_lambda_seed42_L16_trainT0.pth",
        IR
        / "cifar_resnet18_fgmneu_lambda_seed42"
        / "{dataset}/resnet18_fgmneu_lambda/seed42/checkpoints"
        / "resnet18_L[16]_resnet18_fgmneu_lambda_seed42_L16_trainT0.pth",
    ),
}
LAMBDA_CARD = {
    "vgg16": (
        SCRATCH / "cifar_fgmneu_grad_ablation_seed42/{dataset}/vgg16_fgmneu_lambda/seed42/scorecard.json",
        IR / "cifar_fgmneu_grad_ablation_seed42/{dataset}/vgg16_fgmneu_lambda/seed42/scorecard.json",
    ),
    "resnet18": (
        SCRATCH / "cifar_resnet18_fgmneu_lambda_seed42/{dataset}/resnet18_fgmneu_lambda/seed42/scorecard.json",
        IR / "cifar_resnet18_fgmneu_lambda_seed42/{dataset}/resnet18_fgmneu_lambda/seed42/scorecard.json",
    ),
}
FORBIDDEN_OUT = (
    "cifar_resnet18_four_regs_5seed",
    "cifar_vgg16_mne_component_ablation_seed42",
    "cifar_mne_nodetach_unmatched_head_l2_seed42",
    "cifar_mne_unmatched_head_l2_seed42",
    "cifar_fgmneu_grad_ablation_seed42",
    "cifar_resnet18_fgmneu_lambda_seed42",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("scale", "grow"), default="scale")
    parser.add_argument("--arch", choices=("vgg16", "resnet18"), default="vgg16")
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval-seed", type=int, default=EVAL_NOISE_SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scales", default=",".join(str(x) for x in DEFAULT_SCALES))
    parser.add_argument("--grow-coeff", type=float, default=None)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-resolve", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args()
    args.scales = tuple(float(x) for x in str(args.scales).split(",") if x.strip())
    if args.out_root is None:
        name = (
            "cifar_lambda_scale_baseline_seed42"
            if args.arm == "scale"
            else "cifar_l2wo_lambda_grow_seed42"
        )
        args.out_root = ROOT.parent / "important_results" / name
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    return args


def _fmt(path: Path, dataset: str) -> Path:
    return Path(str(path).format(dataset=dataset))


def first_existing(paths: tuple[Path, ...], dataset: str) -> Path | None:
    for path in paths:
        resolved = _fmt(path, dataset)
        if resolved.is_file():
            return resolved
    return _fmt(paths[0], dataset) if paths else None


def require_file(path: Path | None, label: str) -> Path:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def scale_tag(scale: float) -> str:
    return f"{scale:g}".replace(".", "p")


def coeff_tag(coeff: float) -> str:
    return f"{coeff:g}".replace(".", "p")


def arm_dir(args, extra: str) -> Path:
    return args.out_root / args.dataset / f"{args.arch}_{args.arm}_{extra}" / f"seed{args.seed}"


def grow_ckpt_path(args, coeff: float) -> Path:
    tag = coeff_tag(coeff)
    suffix = f"{args.arch}_l2wo_lamgrow_c{tag}_seed{args.seed}_L{LVAL}_trainT0"
    return arm_dir(args, f"c{tag}") / "checkpoints" / f"{args.arch}_L[{LVAL}]_{suffix}.pth"


def backup_thresholds(model) -> dict[int, torch.Tensor]:
    return {
        id(module): module.thresh.detach().clone()
        for module in model.modules()
        if isinstance(module, IF) and getattr(module, "thresh", None) is not None
    }


def restore_thresholds(model, backup: dict[int, torch.Tensor]) -> None:
    for module in model.modules():
        if isinstance(module, IF) and id(module) in backup:
            module.thresh.data.copy_(backup[id(module)])


def apply_scale(model, backup: dict[int, torch.Tensor], scale: float) -> None:
    for module in model.modules():
        if isinstance(module, IF) and id(module) in backup:
            module.thresh.data.copy_(backup[id(module)] * scale)


def _runtime():
    from run_cifar_mne_unmatched_head_l2_seed42 import load_model
    from run_cifar_vgg16_onesided_q_assignment_ablation import (
        snn_metrics,
        sweep,
        test_loader,
        val_loader,
        write_csv,
    )
    from utils import get_torch_device

    return load_model, snn_metrics, sweep, test_loader, val_loader, write_csv, get_torch_device


def thresh_card(model, layer_map: str) -> dict:
    del layer_map
    values = [
        float(module.thresh.detach().float().mean())
        for module in model.modules()
        if isinstance(module, IF) and getattr(module, "thresh", None) is not None
    ]
    if not values:
        return {
            "if_n": 0,
            "if_thresh_mean": float("nan"),
            "if_thresh_median": float("nan"),
            "if_thresh_min": float("nan"),
            "if_thresh_max": float("nan"),
        }
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    return {
        "if_n": len(values),
        "if_thresh_mean": sum(values) / len(values),
        "if_thresh_median": median,
        "if_thresh_min": min(values),
        "if_thresh_max": max(values),
    }


def mean_thresh(model) -> float:
    values = [
        float(module.thresh.detach().float().mean())
        for module in model.modules()
        if isinstance(module, IF) and getattr(module, "thresh", None) is not None
    ]
    if not values:
        raise RuntimeError("model has no IF thresholds")
    return sum(values) / len(values)


def lambda_target(dataset: str, arch: str) -> dict | None:
    for path in LAMBDA_CARD[arch]:
        resolved = _fmt(path, dataset)
        if resolved.is_file():
            return json.loads(resolved.read_text())
    return None


def evaluate_model(model, args, pin: bool, device, extra: dict) -> dict:
    _, snn_metrics, sweep, test_loader, val_loader, _, _ = _runtime()
    layer_map = LAYER_MAP[args.arch]
    thresh = thresh_card(model, layer_map)
    val_rows = sweep(model, val_loader(args, pin), device, "val", args.eval_seed)
    test_rows = sweep(model, test_loader(args, pin), device, "test", args.eval_seed)
    card = {
        "arch": args.arch,
        "dataset": args.dataset,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "selection_uses_test": False,
        "protocol": {
            "T": TEST_T,
            "L": LVAL,
            "mode": "rate_uniform",
            "noise": "post_input_if gaussian",
            "eta_locked": True,
        },
        **thresh,
        **snn_metrics(val_rows, "val"),
        **snn_metrics(test_rows, "test"),
        **extra,
    }
    return card, val_rows, test_rows


def write_cell(out: Path, card: dict, val_rows, test_rows) -> None:
    _, _, _, _, _, write_csv, _ = _runtime()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "val_sweep.csv", val_rows)
    write_csv(out / "test_sweep.csv", test_rows)
    (out / "scorecard.json").write_text(json.dumps(card, indent=2, default=str) + "\n")
    print(json.dumps(card, indent=2, default=str), flush=True)
    print(f"Wrote {out}", flush=True)


def pick_closest(cards: list[dict], target: float, key: str = "val_clean") -> dict:
    def score(card: dict):
        scale = float(card.get("scale", 1.0))
        return (abs(float(card[key]) - target), abs(math.log(max(scale, 1e-8))))

    return min(cards, key=score)


def train_grow_cmd(args, coeff: float) -> list[str]:
    checkpoint = grow_ckpt_path(args, coeff)
    tag = coeff_tag(coeff)
    suffix = f"{args.arch}_l2wo_lamgrow_c{tag}_seed{args.seed}_L{LVAL}_trainT0"
    out = arm_dir(args, f"c{tag}")
    return [
        sys.executable,
        str(ROOT / "main_train.py"),
        "-data", args.dataset,
        "-arch", args.arch,
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
        "--ckpt-dir", str(checkpoint.parent),
        "-suffix", suffix,
        "--regularizer", "l2wo_lambda_grow",
        "--weight_decay", str(L2_WD),
        "--reg_coeff", str(coeff),
        "--epoch_log_csv", str(out / "epoch_log.csv"),
    ]


def train_grow(args, coeff: float) -> Path:
    checkpoint = grow_ckpt_path(args, coeff)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {checkpoint}", flush=True)
        return checkpoint
    if args.test_only:
        raise FileNotFoundError(checkpoint)
    cmd = train_grow_cmd(args, coeff)
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return checkpoint
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def guard_out_root(out_root: Path) -> None:
    text = str(out_root)
    for name in FORBIDDEN_OUT:
        if name in text.rstrip("/").split("/"):
            raise SystemExit(f"refusing to write into {name}: {out_root}")


def self_check() -> None:
    ns = argparse.Namespace(
        arm="grow",
        arch="vgg16",
        dataset="cifar10",
        seed=SEED,
        epochs=EPOCHS,
        batch_size=128,
        workers=8,
        device="cpu",
        out_root=ROOT.parent / "important_results" / "cifar_l2wo_lambda_grow_seed42",
        retrain=False,
        test_only=False,
        dry_run=True,
    )
    cmd = train_grow_cmd(ns, 0.01)
    if cmd[cmd.index("--regularizer") + 1] != "l2wo_lambda_grow":
        raise AssertionError("grow arm must use l2wo_lambda_grow")
    if cmd[cmd.index("--weight_decay") + 1] != str(L2_WD):
        raise AssertionError("WD must stay locked to L2-wo 5e-4")
    if cmd[cmd.index("--reg_coeff") + 1] != "0.01":
        raise AssertionError("η_λ must be the pre-specified grow coeff")
    if "--mne_detach_lambda" in cmd or "--mne_no_detach_bn_affine" in cmd:
        raise AssertionError("λ-grow must not set MNE flags")
    if "mne_l2" in cmd:
        raise AssertionError("λ-grow must not use MNE")
    model = torch.nn.Sequential(IF(thresh=8.0), IF(thresh=4.0))
    loss_small = float(compute_lambda_grow_regularization(model).detach())
    model[0].thresh.data.fill_(16.0)
    model[1].thresh.data.fill_(8.0)
    loss_large = float(compute_lambda_grow_regularization(model).detach())
    if not (loss_large < loss_small):
        raise AssertionError("R=-mean(log λ) must decrease when λ increases")
    dummy = torch.nn.Sequential(IF(thresh=2.0))
    backup = backup_thresholds(dummy)
    apply_scale(dummy, backup, 3.0)
    if abs(float(dummy[0].thresh.detach()) - 6.0) > 1e-6:
        raise AssertionError("scale must multiply IF thresholds")
    restore_thresholds(dummy, backup)
    if abs(float(dummy[0].thresh.detach()) - 2.0) > 1e-6:
        raise AssertionError("restore must recover the L2-wo thresholds")
    guard_out_root(ROOT.parent / "important_results" / "cifar_lambda_scale_baseline_seed42")
    try:
        guard_out_root(SCRATCH / "cifar_resnet18_four_regs_5seed")
    except SystemExit:
        pass
    else:
        raise AssertionError("must refuse to overwrite four-regs")
    print("[self-check] threshold-scaling baseline flags ok", flush=True)


def summarize(out_root: Path) -> None:
    cards = [
        json.loads(path.read_text())
        for path in sorted(out_root.glob("*/*/seed*/scorecard.json"))
    ]
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    print(
        f"{'dataset':<10} {'arch':<9} {'arm':<7} {'tag':<12} "
        f"{'val0':>7} {'test0':>7} {'s5':>7} "
        f"{'λ':>7} {'fire':>7} {'E_mJ':>8}"
    )
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for card in cards:
        tag = card.get("scale_tag") or card.get("coeff_tag") or card.get("method", "")
        print(
            f"{card.get('dataset',''):<10} {card.get('arch',''):<9} "
            f"{card.get('arm',''):<7} {str(tag):<12} "
            f"{card.get('val_clean', float('nan')):7.2f} "
            f"{card.get('test_clean', float('nan')):7.2f} "
            f"{card.get('test_sigma5', float('nan')):7.2f} "
            f"{card.get('if_thresh_mean', float('nan')):7.3f} "
            f"{card.get('test_clean_fire', float('nan')):7.4f} "
            f"{card.get('test_clean_energy_mJ', float('nan')):8.4f}"
        )
        grouped.setdefault(
            (str(card.get("dataset")), str(card.get("arch")), str(card.get("arm"))),
            [],
        ).append(card)
    print("\n[val-matched vs λ-MNE-U]")
    for (dataset, arch, arm), rows in grouped.items():
        if arm not in ("scale", "grow"):
            continue
        target = lambda_target(dataset, arch)
        if not target or "val_clean" not in target:
            print(f"  {dataset} {arch} {arm}: λ-MNE-U val_clean missing")
            continue
        chosen = pick_closest(rows, float(target["val_clean"]))
        tag = chosen.get("scale_tag") or chosen.get("coeff_tag")
        print(
            f"  {dataset} {arch} {arm}: pick {tag}  "
            f"val={chosen['val_clean']:.2f} (target {target['val_clean']:.2f})  "
            f"test={chosen['test_clean']:.2f}/{chosen['test_sigma5']:.2f}  "
            f"λ={chosen['if_thresh_mean']:.3f} fire={chosen['test_clean_fire']:.4f} "
            f"E={chosen['test_clean_energy_mJ']:.4f} mJ"
        )


def run_scale(args) -> None:
    load_model, _, _, _, _, _, get_torch_device = _runtime()
    l2_ckpt = require_file(first_existing(L2WO_CKPT[args.arch], args.dataset), "L2-wo checkpoint")
    lam_ckpt = first_existing(LAMBDA_CKPT[args.arch], args.dataset)
    target = lambda_target(args.dataset, args.arch)
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    print(f"[REUSE L2-wo] {l2_ckpt}", flush=True)
    model = load_model(l2_ckpt, device, args.arch, args.dataset)
    backup = backup_thresholds(model)
    l2_mean = mean_thresh(model)
    scales = list(args.scales)
    lam_mean = None
    if lam_ckpt is not None and lam_ckpt.is_file():
        ref = load_model(lam_ckpt, device, args.arch, args.dataset)
        lam_mean = mean_thresh(ref)
        match = lam_mean / max(l2_mean, 1e-8)
        if all(abs(match - s) > 1e-6 for s in scales):
            scales.append(match)
        print(f"[REUSE λ-MNE-U] {lam_ckpt}  meanλ={lam_mean:.4f}  match_scale={match:.4f}", flush=True)
    elif target and target.get("if_thresh_mean"):
        lam_mean = float(target["if_thresh_mean"])
        match = lam_mean / max(l2_mean, 1e-8)
        if all(abs(match - s) > 1e-6 for s in scales):
            scales.append(match)
        print(f"[REF λ-MNE-U scorecard] meanλ={lam_mean:.4f}  match_scale={match:.4f}", flush=True)
    else:
        print("[WARN] λ-MNE-U checkpoint/scorecard missing; still evaluate the scale grid", flush=True)

    cards = []
    for scale in scales:
        apply_scale(model, backup, scale)
        tag = scale_tag(scale)
        is_match = lam_mean is not None and abs(scale - lam_mean / max(l2_mean, 1e-8)) < 1e-8
        extra = {
            "arm": "scale",
            "method": "l2wo_lambda_scale",
            "label": rf"L2-wo  $\lambda\times{scale:g}$",
            "scale": scale,
            "scale_tag": "match_mean" if is_match and abs(scale - 1.0) > 1e-8 else tag,
            "l2wo_checkpoint": str(l2_ckpt),
            "l2wo_if_thresh_mean": l2_mean,
            "selected_on": "val_clean vs λ-MNE-U",
        }
        card, val_rows, test_rows = evaluate_model(model, args, pin, device, extra)
        write_cell(arm_dir(args, extra["scale_tag"]), card, val_rows, test_rows)
        cards.append(card)
        restore_thresholds(model, backup)

    if lam_ckpt is not None and lam_ckpt.is_file():
        ref = load_model(lam_ckpt, device, args.arch, args.dataset)
        extra = {
            "arm": "lambda",
            "method": "lambda_mneu",
            "label": r"$\lambda$-MNE-U",
            "scale": 1.0,
            "scale_tag": "lambda_mneu",
            "checkpoint": str(lam_ckpt),
        }
        card, val_rows, test_rows = evaluate_model(ref, args, pin, device, extra)
        write_cell(arm_dir(args, "lambda_mneu"), card, val_rows, test_rows)
        cards.append(card)

    target_clean = None
    if target and "val_clean" in target:
        target_clean = float(target["val_clean"])
    lam_card = next((c for c in cards if c.get("method") == "lambda_mneu"), None)
    if lam_card is not None:
        target_clean = float(lam_card["val_clean"])
    summary = {
        "dataset": args.dataset,
        "arch": args.arch,
        "arm": "scale",
        "l2wo_checkpoint": str(l2_ckpt),
        "lambda_checkpoint": str(lam_ckpt) if lam_ckpt and lam_ckpt.is_file() else None,
        "target_val_clean": target_clean,
        "grid": [
            {
                "scale": c["scale"],
                "scale_tag": c["scale_tag"],
                "val_clean": c["val_clean"],
                "test_clean": c["test_clean"],
                "test_sigma5": c["test_sigma5"],
                "if_thresh_mean": c["if_thresh_mean"],
                "test_clean_fire": c["test_clean_fire"],
                "test_clean_energy_mJ": c["test_clean_energy_mJ"],
            }
            for c in cards
        ],
    }
    scale_cards = [c for c in cards if c.get("method") == "l2wo_lambda_scale"]
    if target_clean is not None and scale_cards:
        chosen = pick_closest(scale_cards, target_clean)
        summary["selected_scale"] = chosen["scale"]
        summary["selected_tag"] = chosen["scale_tag"]
        summary["selected_val_clean"] = chosen["val_clean"]
        summary["selected_test_clean"] = chosen["test_clean"]
        summary["selected_test_sigma5"] = chosen["test_sigma5"]
        print(
            f"[SELECT val_clean] scale={chosen['scale']:g} "
            f"val={chosen['val_clean']:.2f} target={target_clean:.2f} "
            f"test={chosen['test_clean']:.2f}/{chosen['test_sigma5']:.2f} "
            f"λ={chosen['if_thresh_mean']:.3f} fire={chosen['test_clean_fire']:.4f} "
            f"E={chosen['test_clean_energy_mJ']:.4f} mJ",
            flush=True,
        )
    out = args.out_root / args.dataset / f"{args.arch}_scale_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"Wrote {out}", flush=True)


def run_grow(args) -> None:
    if args.grow_coeff is None:
        raise SystemExit("grow arm needs --grow-coeff")
    coeff = float(args.grow_coeff)
    if args.dry_resolve or args.dry_run:
        print(" ".join(train_grow_cmd(args, coeff)), flush=True)
        if args.dry_run and not args.test_only:
            return
    load_model, _, _, _, _, _, get_torch_device = _runtime()
    checkpoint = train_grow(args, coeff)
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = load_model(checkpoint, device, args.arch, args.dataset)
    extra = {
        "arm": "grow",
        "method": "l2wo_lambda_grow",
        "label": rf"L2-wo + $\lambda$-grow $\eta_\lambda={coeff:g}$",
        "coeff": coeff,
        "coeff_tag": coeff_tag(coeff),
        "regularizer": "l2wo_lambda_grow",
        "weight_decay": L2_WD,
        "reg_coeff": coeff,
        "checkpoint": str(checkpoint),
        "selected_on": "val_clean vs λ-MNE-U (after the η grid finishes)",
    }
    card, val_rows, test_rows = evaluate_model(model, args, pin, device, extra)
    write_cell(arm_dir(args, f"c{coeff_tag(coeff)}"), card, val_rows, test_rows)


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return
    if args.summarize:
        summarize(args.out_root)
        return
    print(
        f"[INFO] threshold-scaling  arm={args.arm} {args.arch} {args.dataset} "
        f"seed={args.seed} eval_seed={args.eval_seed} out={args.out_root}",
        flush=True,
    )
    if args.dry_resolve or args.dry_run:
        print("L2-wo", first_existing(L2WO_CKPT[args.arch], args.dataset))
        print("λ-MNE-U", first_existing(LAMBDA_CKPT[args.arch], args.dataset))
        if args.arm == "grow" and args.grow_coeff is not None:
            print("grow ckpt", grow_ckpt_path(args, args.grow_coeff))
            print(" ".join(train_grow_cmd(args, args.grow_coeff)))
        return
    guard_out_root(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.arm == "scale":
        run_scale(args)
        return
    run_grow(args)


if __name__ == "__main__":
    main()
