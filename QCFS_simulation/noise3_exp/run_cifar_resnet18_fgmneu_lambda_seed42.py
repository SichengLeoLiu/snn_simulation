#!/usr/bin/env python3
"""CIFAR ResNet-18 seed-42 ∇λ-only FG-MNE-U screen.

Same locked hybrid as VGG A4 λ-only:
  unmatched L2 on (η_U=5e-4, scope=head)
  MNE gradients into IF-λ, BN-γ still detached
  residual-aware map, η_MNE=1e-4
  T=L=16, EVAL_SEED=0, post_input_if

Do not retune. Do not overwrite FG-MNE-U / detach+head / ImageNet trees.
Compare against existing seed-42:
  detach+U   cifar_mne_unmatched_head_l2_seed42
  FG-MNE-U   cifar_mne_nodetach_unmatched_head_l2_seed42
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_cifar_fgmneu_grad_ablation_seed42 import (  # noqa: E402
    GRAD_FLAGS,
)
from run_cifar_mne_unmatched_head_l2_seed42 import (  # noqa: E402
    EVAL_NOISE_SEED,
    L2_WD,
    LVAL,
    MNE_RC,
    dump_mne_mapping_report,
    load_model,
    snn_metrics,
    sweep,
    test_loader,
    unmatched_scope_card,
    val_loader,
    write_csv,
)
from run_cifar_vgg16_onesided_q_assignment_ablation import EPOCHS, LR, TEST_T  # noqa: E402
from utils import get_torch_device  # noqa: E402

ARCH = "resnet18"
LAYER_MAP = "resnet"
SEED = 42
METHOD = "lambda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=ROOT.parent / "important_results" / "cifar_resnet18_fgmneu_lambda_seed42",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    return args


def config_name() -> str:
    return f"{ARCH}_fgmneu_{METHOD}"


def cfg_dir(args) -> Path:
    return args.out_root / args.dataset / config_name() / f"seed{args.seed}"


def suffix(args) -> str:
    return f"{config_name()}_seed{args.seed}_L{LVAL}_trainT0"


def ckpt_path(args) -> Path:
    return cfg_dir(args) / "checkpoints" / f"{ARCH}_L[{LVAL}]_{suffix(args)}.pth"


def train_cmd(args) -> list[str]:
    out = cfg_dir(args)
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
        "--ckpt-dir", str(ckpt_path(args).parent),
        "-suffix", suffix(args),
        "--regularizer", "mne_l2_unmatched",
        "--weight_decay", "0",
        "--reg_coeff", str(MNE_RC),
        "--unmatched_l2_coeff", str(L2_WD),
        "--mne_unmatched_scope", "head",
        "--mne_layer_map", LAYER_MAP,
        "--mapping_diag_dir", str(out / "mapping_init"),
        "--epoch_log_csv", str(out / "epoch_log.csv"),
    ]
    cmd += list(GRAD_FLAGS[METHOD])
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


def self_check() -> None:
    ns = argparse.Namespace(
        dataset="cifar10",
        seed=SEED,
        epochs=EPOCHS,
        batch_size=128,
        workers=8,
        device="cpu",
        out_root=ROOT.parent / "important_results" / "cifar_resnet18_fgmneu_lambda_seed42",
    )
    cmd = train_cmd(ns)
    if cmd[cmd.index("--regularizer") + 1] != "mne_l2_unmatched":
        raise AssertionError("must keep unmatched L2")
    if cmd[cmd.index("--mne_layer_map") + 1] != "resnet":
        raise AssertionError("ResNet must use residual-aware map")
    if cmd[cmd.index("--unmatched_l2_coeff") + 1] != str(L2_WD):
        raise AssertionError("η_U must stay locked to L2-wo WD")
    if cmd[cmd.index("--mne_unmatched_scope") + 1] != "head":
        raise AssertionError("scope must match ResNet FG-MNE-U (head)")
    if "--mne_detach_lambda" in cmd:
        raise AssertionError("λ-only must not detach λ")
    if "--mne_no_detach_bn_affine" in cmd:
        raise AssertionError("λ-only must keep γ detached")
    if GRAD_FLAGS[METHOD]:
        raise AssertionError("λ-only extra flags must be empty")
    print("[self-check] ResNet-18 ∇λ-only flags ok", flush=True)


def summarize(out_root: Path) -> None:
    cards = [
        json.loads(path.read_text())
        for path in sorted(out_root.glob("*/*/seed*/scorecard.json"))
    ]
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    print(f"{'dataset':<10} {'clean':>7} {'s5':>7} {'aucH':>8}")
    for card in cards:
        print(
            f"{card['dataset']:<10} {card['test_clean']:7.2f} "
            f"{card['test_sigma5']:7.2f} {card['test_auc_high']:8.2f}"
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
        f"[INFO] ResNet-18 ∇λ-only + unmatched L2  {args.dataset} "
        f"seed={args.seed} eval_seed={args.eval_seed} "
        f"η_MNE={MNE_RC} η_U={L2_WD} map={LAYER_MAP}",
        flush=True,
    )
    if args.dry_run:
        print("[DRY RUN]", " ".join(train_cmd(args)), flush=True)
        return
    checkpoint = train(args)
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = load_model(checkpoint, device, ARCH, args.dataset)
    scope = unmatched_scope_card(model, LAYER_MAP)
    (out / "unmatched_scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    dump_mne_mapping_report(model, out / "mapping_eval", layer_map=LAYER_MAP, quant_level=LVAL)
    val_rows = sweep(model, val_loader(args, pin), device, "val", args.eval_seed)
    write_csv(out / "val_sweep.csv", val_rows)
    test_rows = sweep(model, test_loader(args, pin), device, "test", args.eval_seed)
    write_csv(out / "test_sweep.csv", test_rows)
    card = {
        "config": config_name(),
        "label": r"∇λ only + unmatched L2",
        "method": "lambda",
        "arch": ARCH,
        "dataset": args.dataset,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "regularizer": "mne_l2_unmatched",
        "layer_map": LAYER_MAP,
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
        **snn_metrics(val_rows, "val"),
        **snn_metrics(test_rows, "test"),
    }
    (out / "scorecard.json").write_text(json.dumps(card, indent=2, default=str) + "\n")
    print(json.dumps(card, indent=2, default=str), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
