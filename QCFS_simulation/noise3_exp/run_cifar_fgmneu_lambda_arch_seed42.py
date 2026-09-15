#!/usr/bin/env python3
"""CIFAR ∇λ-only + unmatched L2 on other VGG / ResNet sizes.

Architectures: vgg11, vgg13, vgg19, resnet34.
Do not use this runner for vgg16 or resnet18; those trees already exist.

Same locked hybrid as VGG-16 A4 / ResNet-18 λ-only:
  unmatched L2 on (η_U=5e-4, scope=head)
  MNE gradients into IF-λ, BN-γ still detached
  VGG: legacy map; ResNet-34: residual-aware map
  η_MNE=1e-4, T=L=16, EVAL_SEED=0, post_input_if

One job = one (arch, dataset, seed). Default seed 42. Do not retune.
Do not overwrite VGG-16 / ResNet-18 / ImageNet trees.
"""
from __future__ import annotations

import argparse
import json
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

from Models import modelpool  # noqa: E402
from Models.VGG import remap_legacy_vgg_state_dict  # noqa: E402
from run_cifar_mne_unmatched_head_l2_seed42 import (  # noqa: E402
    EVAL_NOISE_SEED,
    L2_WD,
    LVAL,
    MNE_RC,
    dump_mne_mapping_report,
    snn_metrics,
    sweep,
    test_loader,
    unmatched_scope_card,
    val_loader,
    write_csv,
)
from run_cifar_vgg16_onesided_q_assignment_ablation import EPOCHS, LR, TEST_T  # noqa: E402
from utils import get_torch_device  # noqa: E402

ARCHS = ("vgg11", "vgg13", "vgg19", "resnet34")
BLOCKED = ("vgg16", "resnet18")
SEED = 42
METHOD = "lambda"
LAYER_MAP = {
    "vgg11": "legacy",
    "vgg13": "legacy",
    "vgg19": "legacy",
    "resnet34": "resnet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=ARCHS, default=None)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval-seed", type=int, default=EVAL_NOISE_SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar_fgmneu_lambda_arch_seed42",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    if args.self_check or args.summarize:
        return args
    if args.arch is None:
        parser.error("--arch is required unless --self-check/--summarize")
    if args.arch in BLOCKED:
        parser.error(f"{args.arch} already has a λ-only tree; do not use this runner")
    return args


def config_name(arch: str) -> str:
    return f"{arch}_fgmneu_{METHOD}"


def cfg_dir(args) -> Path:
    return args.out_root / args.dataset / config_name(args.arch) / f"seed{args.seed}"


def suffix(args) -> str:
    return f"{config_name(args.arch)}_seed{args.seed}_L{LVAL}_trainT0"


def ckpt_path(args) -> Path:
    return cfg_dir(args) / "checkpoints" / f"{args.arch}_L[{LVAL}]_{suffix(args)}.pth"


def layer_map(arch: str) -> str:
    return LAYER_MAP[arch]


def train_cmd(args) -> list[str]:
    out = cfg_dir(args)
    cmd = [
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
        "--ckpt-dir", str(ckpt_path(args).parent),
        "-suffix", suffix(args),
        "--regularizer", "mne_l2_unmatched",
        "--weight_decay", "0",
        "--reg_coeff", str(MNE_RC),
        "--unmatched_l2_coeff", str(L2_WD),
        "--mne_unmatched_scope", "head",
        "--mne_layer_map", layer_map(args.arch),
        "--mapping_diag_dir", str(out / "mapping_init"),
        "--epoch_log_csv", str(out / "epoch_log.csv"),
    ]
    return cmd


def train(args) -> Path:
    checkpoint = ckpt_path(args)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {checkpoint}", flush=True)
        return checkpoint
    if args.test_only:
        raise FileNotFoundError(checkpoint)
    cmd = train_cmd(args)
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return checkpoint
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def load_trained(ckpt: Path, device, arch: str, dataset: str):
    model = modelpool(arch, dataset)
    model._mne_layer_map = layer_map(arch)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if arch.startswith("vgg"):
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


def _ns(**kwargs):
    base = dict(
        dataset="cifar10",
        seed=SEED,
        epochs=EPOCHS,
        batch_size=128,
        workers=8,
        device="cpu",
        out_root=ROOT.parent / "important_results" / "cifar_fgmneu_lambda_arch_seed42",
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def self_check() -> None:
    for arch, expected_map in LAYER_MAP.items():
        cmd = train_cmd(_ns(arch=arch))
        joined = " ".join(cmd)
        if cmd[cmd.index("-arch") + 1] != arch:
            raise AssertionError(f"{arch}: -arch mismatch")
        if cmd[cmd.index("--regularizer") + 1] != "mne_l2_unmatched":
            raise AssertionError(f"{arch}: must keep unmatched L2")
        if cmd[cmd.index("--mne_layer_map") + 1] != expected_map:
            raise AssertionError(f"{arch}: layer map must be {expected_map}")
        if cmd[cmd.index("--unmatched_l2_coeff") + 1] != str(L2_WD):
            raise AssertionError(f"{arch}: η_U must stay locked to L2-wo WD")
        if cmd[cmd.index("--mne_unmatched_scope") + 1] != "head":
            raise AssertionError(f"{arch}: unmatched scope must be head")
        if "--mne_detach_lambda" in cmd:
            raise AssertionError(f"{arch}: λ-only must not detach λ")
        if "--mne_no_detach_bn_affine" in cmd:
            raise AssertionError(f"{arch}: λ-only must keep γ detached")
        if "vgg16" in joined or "resnet18" in joined:
            raise AssertionError("must not write into vgg16/resnet18 trees")
        ckpt = ckpt_path(_ns(arch=arch, dataset="cifar100", seed=40))
        if f"{arch}_fgmneu_lambda" not in str(ckpt) or "seed40" not in str(ckpt):
            raise AssertionError(f"{arch}: output path drifted: {ckpt}")
    print("[self-check] CIFAR size-sweep ∇λ-only flags ok", flush=True)


def summarize(out_root: Path) -> None:
    cards = [
        json.loads(path.read_text())
        for path in sorted(out_root.glob("*/*/seed*/scorecard.json"))
    ]
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    print(f"{'dataset':<10} {'arch':<9} {'seed':>5} {'clean':>7} {'s5':>7} {'aucH':>8}")
    for card in cards:
        print(
            f"{card['dataset']:<10} {card.get('arch', '?'):<9} {card.get('seed', '?'):>5} "
            f"{card['test_clean']:7.2f} {card['test_sigma5']:7.2f} "
            f"{card['test_auc_high']:8.2f}"
        )


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.self_check:
        self_check()
        return
    if args.summarize:
        summarize(args.out_root)
        return

    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] {args.arch} ∇λ-only + unmatched L2  {args.dataset} "
        f"seed={args.seed} eval_seed={args.eval_seed} "
        f"η_MNE={MNE_RC} η_U={L2_WD} map={layer_map(args.arch)}",
        flush=True,
    )
    if args.dry_run:
        print("[DRY RUN]", " ".join(train_cmd(args)), flush=True)
        return
    checkpoint = train(args)
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = load_trained(checkpoint, device, args.arch, args.dataset)
    scope = unmatched_scope_card(model, layer_map(args.arch))
    (out / "unmatched_scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    dump_mne_mapping_report(
        model, out / "mapping_eval", layer_map=layer_map(args.arch), quant_level=LVAL
    )
    val_rows = sweep(model, val_loader(args, pin), device, "val", args.eval_seed)
    write_csv(out / "val_sweep.csv", val_rows)
    test_rows = sweep(model, test_loader(args, pin), device, "test", args.eval_seed)
    write_csv(out / "test_sweep.csv", test_rows)
    card = {
        "config": config_name(args.arch),
        "label": r"∇λ only + unmatched L2",
        "method": METHOD,
        "arch": args.arch,
        "dataset": args.dataset,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "regularizer": "mne_l2_unmatched",
        "layer_map": layer_map(args.arch),
        "unmatched_scope": "head",
        "nabla_gamma": False,
        "nabla_lambda": True,
        "reg_coeff": MNE_RC,
        "unmatched_l2_coeff": L2_WD,
        "checkpoint": str(checkpoint),
        "selection_uses_test": False,
        "protocol": {
            "T": TEST_T,
            "L": LVAL,
            "mode": "rate_uniform",
            "noise": "post_input_if gaussian",
            "eta_locked": True,
        },
        **scope,
        **snn_metrics(val_rows, "val"),
        **snn_metrics(test_rows, "test"),
    }
    (out / "scorecard.json").write_text(json.dumps(card, indent=2, default=str) + "\n")
    print(json.dumps(card, indent=2, default=str), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
