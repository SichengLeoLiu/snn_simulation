#!/usr/bin/env python3
"""CIFAR ∇λ-only at L=T in {4, 8, 32}, seed 42.

Same locked hybrid as the L=16 λ-only trees:
  unmatched L2 on (η_U=5e-4, scope=head)
  MNE gradients into IF-λ, BN-γ still detached
  VGG legacy map; ResNet residual-aware map
  η_MNE=1e-4, EVAL_SEED=0, post_input_if Gaussian

Train ANN T=0; eval SNN T=L. Do not retune. Do not train L=16.
Do not overwrite L=16 λ-only, L2 lscale, FG-MNE-U, or ImageNet.

Writes /scratch/.../cifar_fgmneu_lambda_lscale_seed42
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

from run_cifar_unified_baseline_evalseed0 import (  # noqa: E402
    EVAL_SEED,
    get_torch_device,
    snn_metrics,
    sweep,
    test_loader,
    val_loader,
    write_csv,
)

ARCHS = ("vgg16", "resnet18")
DATASETS = ("cifar10", "cifar100")
TRAIN_LS = (4, 8, 32)
SEED = 42
EPOCHS = 300
LR = 0.1
MNE_RC = 1e-4
L2_WD = 5e-4
LAYER_MAP = {"vgg16": "legacy", "resnet18": "resnet"}
BLOCKED_TREES = (
    "cifar_fgmneu_grad_ablation_seed42",
    "cifar_resnet18_fgmneu_lambda_seed42",
    "cifar_fgmneu_lambda_arch_seed42",
    "cifar_vgg16_l2_lscale_seed42",
    "cifar_vgg16_mne_lscale_seed42",
    "cifar_fgmneu_l1_lscale_seed42",
    "cifar_unified_baseline_evalseed0",
    "cifar_ann_t0_evalseed0",
    "imagenet_resnet18_fgmneu_seed42",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=ARCHS, default=None)
    parser.add_argument("--dataset", choices=DATASETS, default=None)
    parser.add_argument("--quant-L", type=int, choices=TRAIN_LS, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar_fgmneu_lambda_lscale_seed42",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    if args.self_check or args.summarize:
        return args
    if args.arch is None or args.dataset is None or args.quant_L is None:
        parser.error("--arch --dataset --quant-L are required unless --self-check/--summarize")
    if int(args.quant_L) == 16:
        parser.error("L=16 λ-only already exists; this runner will not train it")
    return args


def config_name(arch: str, quant_l: int) -> str:
    return f"{arch}_lambda_L{int(quant_l)}"


def cfg_dir(args) -> Path:
    return args.out_root / args.dataset / config_name(args.arch, args.quant_L) / f"seed{args.seed}"


def suffix(args) -> str:
    quant_l = int(args.quant_L)
    return f"{config_name(args.arch, quant_l)}_seed{args.seed}_L{quant_l}_trainT0"


def ckpt_path(args) -> Path:
    quant_l = int(args.quant_L)
    return cfg_dir(args) / "checkpoints" / f"{args.arch}_L[{quant_l}]_{suffix(args)}.pth"


def train_cmd(args) -> list[str]:
    quant_l = int(args.quant_L)
    out = cfg_dir(args)
    return [
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
        "--ckpt-dir", str(ckpt_path(args).parent),
        "-suffix", suffix(args),
        "--regularizer", "mne_l2_unmatched",
        "--weight_decay", "0",
        "--reg_coeff", str(MNE_RC),
        "--unmatched_l2_coeff", str(L2_WD),
        "--mne_unmatched_scope", "head",
        "--mne_layer_map", LAYER_MAP[args.arch],
        "--mapping_diag_dir", str(out / "mapping_init"),
        "--epoch_log_csv", str(out / "epoch_log.csv"),
    ]


def train(args) -> Path:
    checkpoint = ckpt_path(args)
    if any(name in str(checkpoint) for name in BLOCKED_TREES):
        raise RuntimeError(f"refusing to write into blocked tree: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {checkpoint}", flush=True)
        return checkpoint
    if args.test_only:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        return checkpoint
    cmd = train_cmd(args)
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return checkpoint
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def load_snn(ckpt: Path, device, arch: str, dataset: str, quant_l: int):
    from Models import modelpool
    from Models.VGG import remap_legacy_vgg_state_dict
    import torch

    model = modelpool(arch, dataset)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if arch.startswith("vgg"):
        state = remap_legacy_vgg_state_dict(state)
    model.load_state_dict(state, strict=True)
    model.set_L(int(quant_l))
    model.set_T(int(quant_l))
    model.set_mode("rate_uniform")
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule("normal")
    model.set_first_layer_input_noise_position("post_input_if")
    model.set_first_layer_input_noise_type("gaussian")
    model.set_first_layer_input_noise_sigma(0.0)
    return model.to(device).eval()


def _ns(**kwargs):
    base = dict(
        arch="vgg16",
        dataset="cifar10",
        quant_L=4,
        seed=SEED,
        epochs=EPOCHS,
        batch_size=128,
        workers=8,
        device="cpu",
        out_root=ROOT.parent / "important_results" / "cifar_fgmneu_lambda_lscale_seed42",
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def self_check() -> None:
    cmd = train_cmd(_ns())
    joined = " ".join(cmd)
    if cmd[cmd.index("-L") + 1] != "4":
        raise AssertionError("L-scale must train at the requested L")
    if cmd[cmd.index("-T") + 1] != "0":
        raise AssertionError("ANN train stays T=0")
    if cmd[cmd.index("--regularizer") + 1] != "mne_l2_unmatched":
        raise AssertionError("must keep unmatched L2")
    if cmd[cmd.index("--unmatched_l2_coeff") + 1] != str(L2_WD):
        raise AssertionError("η_U must stay 5e-4")
    if cmd[cmd.index("--mne_unmatched_scope") + 1] != "head":
        raise AssertionError("unmatched scope must be head")
    if cmd[cmd.index("--mne_layer_map") + 1] != "legacy":
        raise AssertionError("VGG must use legacy map")
    if "--mne_detach_lambda" in cmd:
        raise AssertionError("λ-only must not detach λ")
    if "--mne_no_detach_bn_affine" in cmd:
        raise AssertionError("λ-only must keep γ detached")
    r18 = train_cmd(_ns(arch="resnet18", quant_L=32, dataset="cifar100"))
    if r18[r18.index("--mne_layer_map") + 1] != "resnet":
        raise AssertionError("ResNet must use residual-aware map")
    if "L32" not in r18[r18.index("-suffix") + 1]:
        raise AssertionError("suffix must record L")
    if any(name in joined for name in ("cifar_fgmneu_grad_ablation", "imagenet_")):
        raise AssertionError("must not write L=16 or ImageNet trees")
    ckpt = str(ckpt_path(_ns(quant_L=8)))
    if "cifar_fgmneu_lambda_lscale_seed42" not in ckpt:
        raise AssertionError(f"new tree required: {ckpt}")
    print("[self-check] λ-only L-scale flags ok", flush=True)


def summarize(out_root: Path) -> None:
    cards = [
        json.loads(path.read_text())
        for path in sorted(out_root.glob("*/*/seed*/scorecard.json"))
    ]
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    print(
        f"{'dataset':<10} {'arch':<9} {'L':>3} {'T':>3} "
        f"{'clean':>7} {'s5':>7}"
    )
    for card in cards:
        proto = card.get("protocol", {})
        print(
            f"{card['dataset']:<10} {card.get('arch', '?'):<9} "
            f"{proto.get('L', '?'):>3} {proto.get('T', '?'):>3} "
            f"{card['test_clean']:7.2f} {card['test_sigma5']:7.2f}"
        )


def eval_trained(args, checkpoint: Path) -> None:
    out = cfg_dir(args)
    card_path = out / "scorecard.json"
    if card_path.is_file() and not args.force and not args.retrain:
        print(f"[SKIP EVAL] {card_path}", flush=True)
        return
    if args.dry_run:
        print("[DRY EVAL]", checkpoint, flush=True)
        return
    quant_l = int(args.quant_L)
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = load_snn(checkpoint, device, args.arch, args.dataset, quant_l)
    if int(getattr(model, "T", -1)) != quant_l or int(getattr(model, "L", -1)) != quant_l:
        raise RuntimeError(f"expected T=L={quant_l}, got T={getattr(model,'T',None)} L={getattr(model,'L',None)}")
    val_rows = sweep(
        model,
        val_loader(args, pin),
        device,
        "val",
        args.eval_seed,
        [0.0, 1.0, 2.0, 3.0, 5.0],
    )
    write_csv(out / "val_sweep.csv", val_rows)
    test_rows = sweep(
        model,
        test_loader(args, pin),
        device,
        "test",
        args.eval_seed,
        [0.0, 1.0, 2.0, 3.0, 5.0],
    )
    write_csv(out / "test_sweep.csv", test_rows)
    card = {
        "config": config_name(args.arch, quant_l),
        "label": "nabla-lambda only",
        "method": "lambda",
        "arch": args.arch,
        "dataset": args.dataset,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "regularizer": "mne_l2_unmatched",
        "layer_map": LAYER_MAP[args.arch],
        "unmatched_scope": "head",
        "nabla_gamma": False,
        "nabla_lambda": True,
        "reg_coeff": MNE_RC,
        "unmatched_l2_coeff": L2_WD,
        "checkpoint": str(checkpoint),
        "protocol": {
            "T": quant_l,
            "L": quant_l,
            "mode": "rate_uniform",
            "noise": "post_input_if gaussian",
            "eval_seed_locked": True,
            "eta_locked": True,
        },
        **snn_metrics(val_rows, "val"),
        **snn_metrics(test_rows, "test"),
    }
    card_path.write_text(json.dumps(card, indent=2, default=str) + "\n")
    print(json.dumps(card, indent=2, default=str), flush=True)
    print(f"Wrote {out}", flush=True)


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return
    if args.summarize:
        summarize(args.out_root)
        return
    args.out_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] λ-only L-scale {args.arch} {args.dataset} L=T={args.quant_L} "
        f"seed={args.seed} eval_seed={args.eval_seed}",
        flush=True,
    )
    checkpoint = train(args)
    eval_trained(args, checkpoint)


if __name__ == "__main__":
    main()
