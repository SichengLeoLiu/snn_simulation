#!/usr/bin/env python3
"""CIFAR seed-42 FG-MNE-U / L1-wo at L=T in {4, 8, 32}.

Locked coefficients, no per-L retune from test:
  FG-MNE-U  mne_l2_unmatched, η_MNE=1e-4, η_U=5e-4
            no-detach (grads into λ and BN γ), default L^2 scaling
            VGG sequential/legacy map; ResNet residual-aware map
  L1-wo     regularizer=l1, rc=1e-5, Conv/Linear weights only, WD=0

Protocol matches the existing VGG L2 lscale cells:
  ANN train T=0, 300 epochs, lr=0.1
  SNN eval T=L, rate_uniform, post_input_if, Gaussian σ=0…5
  Noise seed = training seed (42), same as L2 lscale (not EVAL_SEED=0)

L=16 already exists (FG-MNE-U 5-seed / VGG five-regs L1). This runner
refuses to retrain L=16 unless --allow-L16. Optional reuse-only eval
copies those checkpoints into the new directory.

Writes a new scratch tree. Do not overwrite l2_lscale, mne_lscale,
four-regs, FG-MNE-U L=16, or ImageNet.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Models import modelpool  # noqa: E402
from Models.VGG import remap_legacy_vgg_state_dict  # noqa: E402
from run_cifar_vgg16_mne_component_ablation_seed42 import sweep  # noqa: E402
from run_cifar_vgg16_onesided_q_assignment_ablation import (  # noqa: E402
    EPOCHS,
    LR,
    snn_metrics,
    test_loader,
    val_loader,
    write_csv,
)
from utils import dump_mne_mapping_report, get_torch_device  # noqa: E402

SCRATCH = Path("/scratch/gs14/sl9144/snn_results")
SEED = 42
QUANT_LS = (4, 8, 16, 32)
TRAIN_LS = (4, 8, 32)
MNE_RC = 1e-4
L2_WD = 5e-4
L1_RC = 1e-5

METHODS = {
    "fgmneu": {
        "label": "FG-MNE-U",
        "regularizer": "mne_l2_unmatched",
        "weight_decay": 0.0,
        "reg_coeff": MNE_RC,
        "unmatched_l2_coeff": L2_WD,
        "unmatched_scope": "all",
    },
    "l1wo": {
        "label": "L1-wo",
        "regularizer": "l1",
        "weight_decay": 0.0,
        "reg_coeff": L1_RC,
        "unmatched_l2_coeff": None,
        "unmatched_scope": None,
    },
}

ARCH_SPECS = {
    "vgg16": {"layer_map": "legacy", "remap_vgg": True},
    "resnet18": {"layer_map": "resnet", "remap_vgg": False},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=tuple(ARCH_SPECS), default="vgg16")
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--method", choices=tuple(METHODS), default=None)
    parser.add_argument("--quant-L", type=int, choices=QUANT_LS, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--allow-L16", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar_fgmneu_l1_lscale_seed42",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    if args.summarize:
        return args
    if args.method is None:
        parser.error("specify --method or --summarize")
    if int(args.quant_L) == 16 and not args.allow_L16 and not args.test_only:
        parser.error("L=16 already exists; pass --test-only --allow-L16 to reuse, not retrain")
    return args


def config_name(args) -> str:
    return f"{args.arch}_{args.method}_L{int(args.quant_L)}"


def suffix(args) -> str:
    return f"{config_name(args)}_seed{args.seed}_L{int(args.quant_L)}_trainT0"


def cfg_dir(args) -> Path:
    return args.out_root / args.dataset / config_name(args)


def ckpt_path(args) -> Path:
    return cfg_dir(args) / "checkpoints" / f"{args.arch}_L[{int(args.quant_L)}]_{suffix(args)}.pth"


def _existing(path: Path) -> Path | None:
    return path if path.is_file() else None


def reuse_l16(args) -> Path | None:
    if int(args.quant_L) != 16:
        return None
    ds = args.dataset
    seed = args.seed
    if args.method == "fgmneu":
        if args.arch == "vgg16":
            name = f"vgg16_L[16]_vgg16_mne_head_seed{seed}_L16_trainT0.pth"
            roots = [
                SCRATCH / "cifar_mne_nodetach_unmatched_head_l2_seed42" / ds / "vgg16_mne_head" / f"seed{seed}" / "checkpoints" / name,
                ROOT.parent / "important_results" / "cifar_mne_nodetach_unmatched_head_l2_seed42" / ds / "vgg16_mne_head" / f"seed{seed}" / "checkpoints" / name,
            ]
        else:
            name = f"resnet18_L[16]_resnet18_mne_head_seed{seed}_L16_trainT0.pth"
            roots = [
                SCRATCH / "cifar_mne_nodetach_unmatched_head_l2_seed42" / ds / "resnet18_mne_head" / f"seed{seed}" / "checkpoints" / name,
            ]
        for path in roots:
            hit = _existing(path)
            if hit is not None:
                return hit
        return None
    names = [
        f"{args.arch}_L[16]_mneablate_{ds}_l1_rc1em05_seed{seed}_L16_trainT0.pth",
        f"{args.arch}_L[16]_mneablate_{ds}_l1_rc1em05_seed{seed}_L16.pth",
    ]
    roots = [
        Path(os.environ.get("CKPT_ROOT", "/home/595/sl9144/codes/snn_simulation/QCFS_simulation")) / f"{ds}-checkpoints",
        ROOT / f"{ds}-checkpoints",
        SCRATCH / f"{ds}_{args.arch}_five_regs_sigma0_5_5seed",
        Path(f"/home/595/sl9144/codes/snn_simulation/QCFS_simulation/{ds}-checkpoints"),
    ]
    for root in roots:
        for name in names:
            hit = _existing(root / name)
            if hit is not None:
                return hit
            for match in root.glob(f"**/{name}"):
                return match
    return None


def train_cmd(args) -> list[str]:
    spec = METHODS[args.method]
    arch = ARCH_SPECS[args.arch]
    quant_l = int(args.quant_L)
    cmd = [
        sys.executable,
        str(ROOT / "main_train.py"),
        "-data", args.dataset,
        "-arch", args.arch,
        "-L", str(quant_l),
        "-T", "0",
        "--epochs", str(args.epochs),
        "-lr", str(LR),
        "-b", str(args.batch_size),
        "-j", str(args.workers),
        "--seed", str(args.seed),
        "--device", args.device,
        "--spike_schedule", "normal",
        "--ckpt-save-mode", "best",
        "--ckpt-dir", str(cfg_dir(args) / "checkpoints"),
        "-suffix", suffix(args),
        "--regularizer", spec["regularizer"],
        "--weight_decay", str(spec["weight_decay"]),
        "--reg_coeff", str(spec["reg_coeff"]),
    ]
    if args.method == "fgmneu":
        cmd += [
            "--mne_layer_map", arch["layer_map"],
            "--mne_no_detach_bn_affine",
            "--unmatched_l2_coeff", str(spec["unmatched_l2_coeff"]),
            "--mne_unmatched_scope", spec["unmatched_scope"],
            "--epoch_log_csv", str(cfg_dir(args) / "epoch_log.csv"),
            "--mapping_diag_dir", str(cfg_dir(args) / "mapping_init"),
        ]
    return cmd


def train(args) -> tuple[Path, bool]:
    out = cfg_dir(args)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_path(args)
    if ckpt.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {ckpt}", flush=True)
        return ckpt, False
    if int(args.quant_L) == 16:
        reused = reuse_l16(args)
        if reused is None:
            raise FileNotFoundError("L=16 reuse checkpoint missing; will not retrain L=16")
        if args.retrain:
            raise RuntimeError("refusing --retrain at L=16")
        shutil.copy2(reused, ckpt)
        print(f"[REUSE L16] {reused} -> {ckpt}", flush=True)
        return ckpt, True
    if args.test_only:
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        return ckpt, False
    cmd = train_cmd(args)
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return ckpt, False
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"training finished but missing {ckpt}")
    return ckpt, False


def load_model(ckpt: Path, device, args):
    spec = ARCH_SPECS[args.arch]
    quant_l = int(args.quant_L)
    model = modelpool(args.arch, args.dataset)
    model._mne_layer_map = spec["layer_map"]
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if spec["remap_vgg"]:
        state = remap_legacy_vgg_state_dict(state)
    model.load_state_dict(state, strict=True)
    model.set_L(quant_l)
    model.set_T(quant_l)
    model.set_mode("rate_uniform")
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule("normal")
    if hasattr(model, "set_first_layer_input_noise_position"):
        model.set_first_layer_input_noise_position("post_input_if")
    if hasattr(model, "set_first_layer_input_noise_type"):
        model.set_first_layer_input_noise_type("gaussian")
    return model.to(device).eval()


def summarize(out_root: Path) -> None:
    cards = []
    for path in sorted(out_root.glob("*/*/scorecard.json")):
        cards.append(json.loads(path.read_text()))
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    print(
        f"{'dataset':<10} {'arch':<10} {'method':<8} {'L':>3} "
        f"{'test0':>7} {'test5':>7} {'AUC':>8}"
    )
    rows = []
    for card in cards:
        print(
            f"{card['dataset']:<10} {card['arch']:<10} {card['method']:<8} "
            f"{int(card['quant_level']):3d} "
            f"{card['test_clean']:7.2f} {card['test_sigma5']:7.2f} "
            f"{card['test_auc_full']:8.1f}"
        )
        rows.append(card)
    if rows:
        write_csv(out_root / "fgmneu_l1_lscale_summary.csv", rows)
        print(f"Wrote {out_root / 'fgmneu_l1_lscale_summary.csv'}")


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(args.out_root)
        return

    spec = METHODS[args.method]
    arch = ARCH_SPECS[args.arch]
    quant_l = int(args.quant_L)
    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] {args.dataset} {args.arch} {spec['label']} "
        f"L=T={quant_l} mapped={arch['layer_map']} seed={args.seed} "
        f"η_MNE={spec['reg_coeff']} η_U={spec['unmatched_l2_coeff']}",
        flush=True,
    )
    if args.dry_run and quant_l != 16:
        print("[DRY RUN]", " ".join(train_cmd(args)), flush=True)
        return
    ckpt, reused = train(args)
    if args.dry_run:
        return
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = load_model(ckpt, device, args)
    if args.method == "fgmneu":
        dump_mne_mapping_report(
            model,
            out,
            layer_map=arch["layer_map"],
            quant_level=quant_l,
        )
    eval_args = argparse.Namespace(
        dataset=args.dataset,
        batch_size=args.batch_size,
        workers=0,
    )
    val_rows = sweep(model, val_loader(eval_args, pin), device, "val", args.seed, quant_l)
    write_csv(out / "val_sweep.csv", val_rows)
    test_rows = sweep(model, test_loader(eval_args, pin), device, "test", args.seed, quant_l)
    write_csv(out / "test_sweep.csv", test_rows)
    card = {
        "config": config_name(args),
        "label": spec["label"],
        "method": args.method,
        "dataset": args.dataset,
        "arch": args.arch,
        "seed": args.seed,
        "eval_seed": args.seed,
        "quant_level": quant_l,
        "eval_T": quant_l,
        "regularizer": spec["regularizer"],
        "weight_decay": spec["weight_decay"],
        "reg_coeff": spec["reg_coeff"],
        "unmatched_l2_coeff": spec["unmatched_l2_coeff"],
        "layer_map": arch["layer_map"],
        "detach_lambda": False if args.method == "fgmneu" else None,
        "l16_reused": reused,
        "checkpoint": str(ckpt),
        "selection_uses_test": False,
        "protocol": {
            "T": quant_l,
            "L": quant_l,
            "mode": "rate_uniform",
            "noise": "post_input_if gaussian",
            "noise_seed": args.seed,
            "eta_locked": True,
        },
        **snn_metrics(val_rows, "val"),
        **snn_metrics(test_rows, "test"),
    }
    (out / "scorecard.json").write_text(json.dumps(card, indent=2, default=str) + "\n")
    print(json.dumps(card, indent=2, default=str), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
