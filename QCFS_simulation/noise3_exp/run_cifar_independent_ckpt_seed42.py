#!/usr/bin/env python3
"""CIFAR independent checkpoint selection, seed 42.

Retrain the main-table methods with a fixed 45k/5k split so epoch choice
does not use official test ANN accuracy. Existing test-best checkpoints
are not last-epoch weights, so they cannot be re-evaluated.

Protocol
--------
  train on perm[5000:] of CIFAR train (aug), VAL_SPLIT_SEED=0
  select on perm[:5000] (eval transform; same 5k as SNN val_loader)
  official 10k test is logged / SNN-swept after training, never used to save
  locked η_MNE=1e-4, η_U=5e-4; do not retune
  T=L=16, EVAL_SEED=0, rate_uniform, post_input_if

Methods for the VGG independent-ckpt table
-----------------------------------------
  l2wo     optimizer weights-only WD=5e-4
  l1wo     L1 on Conv/Linear weights, rc=1e-5
  lambda   MNE + unmatched-head L2, ∇λ=✓ ∇γ=0   (paper method)

``detach`` / ``fgmneu`` remain in the CLI only so already-queued jobs
do not crash; do not submit them. ResNet is not part of this table.

Writes a new scratch tree. Do not overwrite existing λ-only / FG-MNE-U /
ImageNet trees.
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

ARCHS = ("vgg16", "resnet18")
SUBMIT_ARCHS = ("vgg16",)
SUBMIT_METHODS = ("l2wo", "l1wo", "lambda")
METHODS = ("l2wo", "l1wo", "lambda", "detach", "fgmneu")
DATASETS = ("cifar10", "cifar100")
SEED = 42
EVAL_NOISE_SEED = 0
EPOCHS = 300
LR = 0.1
LVAL = 16
TEST_T = 16
MNE_RC = 1e-4
L2_WD = 5e-4
L1_RC = 1e-5
VAL_SIZE = 5000
VAL_SPLIT_SEED = 0
LAYER_MAP = {"vgg16": "legacy", "resnet18": "resnet"}
GRAD_FLAGS = {
    "detach": ["--mne_detach_lambda"],
    "lambda": [],
    "fgmneu": ["--mne_no_detach_bn_affine"],
}
LABELS = {
    "l2wo": "L2-wo",
    "l1wo": "L1-wo",
    "lambda": r"∇λ only + unmatched L2",
    "detach": "detach+U",
    "fgmneu": "FG-MNE-U",
}
NABLA = {
    "l2wo": (False, False),
    "l1wo": (False, False),
    "lambda": (False, True),
    "detach": (False, False),
    "fgmneu": (True, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="cifar10")
    parser.add_argument("--arch", choices=ARCHS, default="vgg16")
    parser.add_argument("--method", choices=METHODS, default="lambda")
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
        default=ROOT.parent / "important_results" / "cifar_independent_ckpt_seed42",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    return args


def config_name(arch: str, method: str) -> str:
    return f"{arch}_{method}"


def cfg_dir(args) -> Path:
    return args.out_root / args.dataset / config_name(args.arch, args.method) / f"seed{args.seed}"


def suffix(args) -> str:
    return f"{config_name(args.arch, args.method)}_seed{args.seed}_L{LVAL}_trainT0"


def ckpt_path(args) -> Path:
    return cfg_dir(args) / "checkpoints" / f"{args.arch}_L[{LVAL}]_{suffix(args)}.pth"


def selection_path(args) -> Path:
    ckpt = ckpt_path(args)
    return ckpt.with_name(ckpt.stem + "_selection.json")


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
        "--ckpt-select-split", "val",
        "--val-holdout", str(VAL_SIZE),
        "--val-split-seed", str(VAL_SPLIT_SEED),
        "--ckpt-dir", str(ckpt_path(args).parent),
        "-suffix", suffix(args),
        "--epoch_log_csv", str(out / "epoch_log.csv"),
    ]
    if args.method == "l2wo":
        cmd += [
            "--regularizer", "weight_decay_weights_only",
            "--weight_decay", str(L2_WD),
        ]
        return cmd
    if args.method == "l1wo":
        cmd += [
            "--regularizer", "l1",
            "--weight_decay", "0",
            "--reg_coeff", str(L1_RC),
        ]
        return cmd
    cmd += [
        "--regularizer", "mne_l2_unmatched",
        "--weight_decay", "0",
        "--reg_coeff", str(MNE_RC),
        "--unmatched_l2_coeff", str(L2_WD),
        "--mne_unmatched_scope", "head",
        "--mne_layer_map", LAYER_MAP[args.arch],
        "--mapping_diag_dir", str(out / "mapping_init"),
    ]
    cmd += list(GRAD_FLAGS[args.method])
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


def _ns(**kwargs):
    base = dict(
        dataset="cifar10",
        arch="vgg16",
        method="lambda",
        seed=SEED,
        epochs=EPOCHS,
        batch_size=128,
        workers=8,
        device="cpu",
        out_root=ROOT.parent / "important_results" / "cifar_independent_ckpt_seed42",
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def self_check() -> None:
    import torch
    from Preprocess.getdataloader import cifar_holdout_indices

    val_idx, train_idx = cifar_holdout_indices(50000, VAL_SIZE, VAL_SPLIT_SEED)
    perm = torch.randperm(50000, generator=torch.Generator().manual_seed(VAL_SPLIT_SEED)).tolist()
    if val_idx != perm[:VAL_SIZE] or train_idx != perm[VAL_SIZE:]:
        raise AssertionError("holdout indices must match SNN val_loader")
    if len(train_idx) != 45000:
        raise AssertionError("train split must be 45k")

    blocked = (
        "cifar_fgmneu_grad_ablation",
        "cifar_mne_unmatched_head",
        "cifar_mne_nodetach_unmatched_head",
        "cifar_resnet18_fgmneu_lambda",
        "cifar_fgmneu_lambda_arch",
    )
    for arch in ARCHS:
        for method in METHODS:
            for dataset in DATASETS:
                ns = _ns(arch=arch, method=method, dataset=dataset)
                cmd = train_cmd(ns)
                joined = " ".join(cmd)
                ckpt = str(ckpt_path(ns))
                if "cifar_independent_ckpt_seed42" not in ckpt:
                    raise AssertionError(f"must write a new tree: {ckpt}")
                if any(name in ckpt for name in blocked):
                    raise AssertionError(f"must not overwrite {ckpt}")
                if cmd[cmd.index("--ckpt-select-split") + 1] != "val":
                    raise AssertionError(f"{method}: must select on val")
                if cmd[cmd.index("--val-holdout") + 1] != str(VAL_SIZE):
                    raise AssertionError("val holdout must stay 5000")
                if cmd[cmd.index("--val-split-seed") + 1] != str(VAL_SPLIT_SEED):
                    raise AssertionError("split seed must stay 0")
                if cmd[cmd.index("--ckpt-save-mode") + 1] != "best":
                    raise AssertionError("save mode must be best-on-val")
                if method == "l2wo":
                    if cmd[cmd.index("--regularizer") + 1] != "weight_decay_weights_only":
                        raise AssertionError("l2wo regularizer")
                    if cmd[cmd.index("--weight_decay") + 1] != str(L2_WD):
                        raise AssertionError("l2wo WD must stay 5e-4")
                    if "--mne_detach_lambda" in cmd or "--mne_no_detach_bn_affine" in cmd:
                        raise AssertionError("l2wo must not pass MNE grad flags")
                    if "mne_l2_unmatched" in joined:
                        raise AssertionError("l2wo must not use unmatched MNE")
                    continue
                if method == "l1wo":
                    if cmd[cmd.index("--regularizer") + 1] != "l1":
                        raise AssertionError("l1wo regularizer")
                    if cmd[cmd.index("--reg_coeff") + 1] != str(L1_RC):
                        raise AssertionError("l1wo rc must stay 1e-5")
                    if cmd[cmd.index("--weight_decay") + 1] != "0":
                        raise AssertionError("l1wo must not use optimizer WD")
                    if "mne_l2_unmatched" in joined:
                        raise AssertionError("l1wo must not use unmatched MNE")
                    continue
                if cmd[cmd.index("--regularizer") + 1] != "mne_l2_unmatched":
                    raise AssertionError(f"{method}: keep unmatched L2")
                if cmd[cmd.index("--unmatched_l2_coeff") + 1] != str(L2_WD):
                    raise AssertionError("η_U must stay locked to L2-wo WD")
                if cmd[cmd.index("--reg_coeff") + 1] != str(MNE_RC):
                    raise AssertionError("η_MNE must stay 1e-4")
                if cmd[cmd.index("--mne_layer_map") + 1] != LAYER_MAP[arch]:
                    raise AssertionError(f"{arch} layer map")
                extra = GRAD_FLAGS[method]
                for flag in extra:
                    if flag not in cmd:
                        raise AssertionError(f"{method} missing {flag}")
                if method == "lambda":
                    if extra:
                        raise AssertionError("λ-only extra flags must be empty")
                    if "--mne_detach_lambda" in cmd:
                        raise AssertionError("λ-only must not detach λ")
                    if "--mne_no_detach_bn_affine" in cmd:
                        raise AssertionError("λ-only must keep γ detached")
                if method == "detach" and "--mne_detach_lambda" not in cmd:
                    raise AssertionError("detach+U must detach λ")
                if method == "fgmneu":
                    if "--mne_no_detach_bn_affine" not in cmd:
                        raise AssertionError("FG-MNE-U must update γ")
                    if "--mne_detach_lambda" in cmd:
                        raise AssertionError("FG-MNE-U must not detach λ")
    for arch in SUBMIT_ARCHS:
        for method in SUBMIT_METHODS:
            ns = _ns(arch=arch, method=method)
            if "vgg16_" not in str(ckpt_path(ns)):
                raise AssertionError("submit table must stay on vgg16")
    print("[self-check] independent-ckpt 45k/5k flags ok", flush=True)


def load_selection(args) -> dict:
    path = selection_path(args)
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def summarize(out_root: Path) -> None:
    cards = [
        json.loads(path.read_text())
        for path in sorted(out_root.glob("*/*/seed*/scorecard.json"))
    ]
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    print(
        f"{'dataset':<10} {'arch':<9} {'method':<8} {'sel_ep':>6} "
        f"{'clean':>7} {'s5':>7} {'aucH':>8}"
    )
    for card in cards:
        print(
            f"{card['dataset']:<10} {card.get('arch', '?'):<9} "
            f"{card.get('method', '?'):<8} {card.get('selected_epoch', '?'):>6} "
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

    from Models import modelpool
    from Models.VGG import remap_legacy_vgg_state_dict
    from run_cifar_mne_unmatched_head_l2_seed42 import (
        dump_mne_mapping_report,
        snn_metrics,
        sweep,
        test_loader,
        unmatched_scope_card,
        val_loader,
        write_csv,
    )
    import torch
    from utils import get_torch_device

    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    nabla_gamma, nabla_lambda = NABLA[args.method]
    print(
        f"[INFO] independent ckpt  {args.arch} {args.method} {args.dataset} "
        f"seed={args.seed} eval_seed={args.eval_seed} "
        f"select=val({VAL_SIZE}/{VAL_SPLIT_SEED}) "
        f"η_MNE={MNE_RC} η_U={L2_WD} map={LAYER_MAP[args.arch]}",
        flush=True,
    )
    if args.dry_run:
        print("[DRY RUN]", " ".join(train_cmd(args)), flush=True)
        return
    checkpoint = train(args)
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = modelpool(args.arch, args.dataset)
    model._mne_layer_map = LAYER_MAP[args.arch]
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if args.arch.startswith("vgg"):
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
    model = model.to(device).eval()

    scope = unmatched_scope_card(model, LAYER_MAP[args.arch])
    (out / "unmatched_scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    dump_mne_mapping_report(
        model, out / "mapping_eval", layer_map=LAYER_MAP[args.arch], quant_level=LVAL
    )
    val_rows = sweep(model, val_loader(args, pin), device, "val", args.eval_seed)
    write_csv(out / "val_sweep.csv", val_rows)
    test_rows = sweep(model, test_loader(args, pin), device, "test", args.eval_seed)
    write_csv(out / "test_sweep.csv", test_rows)
    selected = load_selection(args)
    card = {
        "config": config_name(args.arch, args.method),
        "label": LABELS[args.method],
        "method": args.method,
        "arch": args.arch,
        "dataset": args.dataset,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "regularizer": {
            "l2wo": "weight_decay_weights_only",
            "l1wo": "l1",
        }.get(args.method, "mne_l2_unmatched"),
        "layer_map": LAYER_MAP[args.arch],
        "unmatched_scope": None if args.method in ("l2wo", "l1wo") else "head",
        "nabla_gamma": nabla_gamma,
        "nabla_lambda": nabla_lambda,
        "reg_coeff": { "l2wo": None, "l1wo": L1_RC }.get(args.method, MNE_RC),
        "unmatched_l2_coeff": 0.0 if args.method in ("l2wo", "l1wo") else L2_WD,
        "weight_decay": L2_WD if args.method == "l2wo" else 0.0,
        "checkpoint": str(checkpoint),
        "selection_uses_test": False,
        "ckpt_select_split": "val",
        "val_holdout": VAL_SIZE,
        "val_split_seed": VAL_SPLIT_SEED,
        "n_train": selected.get("n_train", 45000),
        "selected_epoch": selected.get("epoch"),
        "ann_val_selected": selected.get("select_acc"),
        "ann_test_at_selected": selected.get("ann_test_acc"),
        "protocol": {
            "T": TEST_T,
            "L": LVAL,
            "mode": "rate_uniform",
            "noise": "post_input_if gaussian",
            "eta_locked": True,
            "ckpt_select": "cifar 45k/5k holdout ANN acc",
        },
        **snn_metrics(val_rows, "val"),
        **snn_metrics(test_rows, "test"),
    }
    (out / "scorecard.json").write_text(json.dumps(card, indent=2, default=str) + "\n")
    print(json.dumps(card, indent=2, default=str), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
