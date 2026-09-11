#!/usr/bin/env python3
"""CIFAR VGG-16 seed-42 FG-MNE-U gradient ablation (P1).

FG-MNE-U keeps unmatched-weight L2 (η_U=5e-4) on all four cells and only
changes whether MNE gradients enter BN-γ and/or IF-λ:

    detach   ∇γ=0, ∇λ=0   reuse detach+head
    gamma    ∇γ=✓, ∇λ=0   train
    lambda   ∇γ=0, ∇λ=✓   train
    full     ∇γ=✓, ∇λ=✓   reuse no-detach+head (FG-MNE-U)

VGG legacy map. Do not retune η. Do not overwrite the reused trees.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Models import modelpool  # noqa: E402
from Models.layer import IF  # noqa: E402
from run_cifar_mne_unmatched_head_l2_seed42 import (  # noqa: E402
    EVAL_NOISE_SEED,
    L2_WD,
    LVAL,
    MNE_RC,
    classifier_norms,
    dump_mne_mapping_report,
    flatten_margin,
    load_model,
    run_margin_diag,
    snn_metrics,
    sweep,
    test_loader,
    unmatched_scope_card,
    val_loader,
    write_csv,
)
from run_cifar_vgg16_onesided_q_assignment_ablation import EPOCHS, LR, TEST_T  # noqa: E402
from utils import (  # noqa: E402
    collect_weight_layer_matches,
    get_torch_device,
    unmatched_weight_rows,
)

ARCH = "vgg16"
SEED = 42
LAYER_MAP = "legacy"
GRADS = ("detach", "gamma", "lambda", "full")
SCRATCH = Path("/scratch/gs14/sl9144/snn_results")
REUSE = {
    "detach": SCRATCH / "cifar_mne_unmatched_head_l2_seed42",
    "full": SCRATCH / "cifar_mne_nodetach_unmatched_head_l2_seed42",
}
REUSE_LOCAL = {
    "detach": ROOT.parent / "important_results" / "cifar_mne_unmatched_head_l2_seed42",
    "full": ROOT.parent / "important_results" / "cifar_mne_nodetach_unmatched_head_l2_seed42",
}
GRAD_FLAGS = {
    "detach": ["--mne_detach_lambda"],
    "gamma": ["--mne_detach_lambda", "--mne_no_detach_bn_affine"],
    "lambda": [],
    "full": ["--mne_no_detach_bn_affine"],
}
GRAD_NABLA = {
    "detach": (False, False),
    "gamma": (True, False),
    "lambda": (False, True),
    "full": (True, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--grad", choices=GRADS, default="gamma")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval-seed", type=int, default=EVAL_NOISE_SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--skip-diag", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--n-streams", type=int, default=8)
    parser.add_argument("--sigmas", default="1,3,5")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar_fgmneu_grad_ablation_seed42",
    )
    args = parser.parse_args()
    args.sigmas = tuple(float(x) for x in str(args.sigmas).split(",") if x.strip())
    args.skip_diag = bool(args.skip_diag or os.environ.get("SKIP_DIAG", "0") == "1")
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    return args


def config_name(grad: str) -> str:
    return f"{ARCH}_fgmneu_{grad}"


def cfg_dir(args) -> Path:
    return args.out_root / args.dataset / config_name(args.grad) / f"seed{args.seed}"


def ckpt_path(args) -> Path:
    suffix = f"{config_name(args.grad)}_seed{args.seed}_L{LVAL}_trainT0"
    return cfg_dir(args) / "checkpoints" / f"{ARCH}_L[{LVAL}]_{suffix}.pth"


def reuse_dir(grad: str, dataset: str) -> Path | None:
    for root in (REUSE[grad], REUSE_LOCAL[grad]):
        path = root / dataset / f"{ARCH}_mne_head" / f"seed{SEED}"
        if (path / "scorecard.json").is_file() or list(path.glob("checkpoints/*.pth")):
            return path
    return REUSE[grad] / dataset / f"{ARCH}_mne_head" / f"seed{SEED}"


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _geomean(values: list[float]) -> float:
    positive = [float(v) for v in values if v > 0.0]
    if not positive:
        return float("nan")
    return float(math.exp(sum(math.log(v) for v in positive) / len(positive)))


def layer_scale_rows(model, layer_map: str) -> list[dict]:
    rows = []
    for match in collect_weight_layer_matches(model, layer_map=layer_map):
        weight = match["weight"].detach().float()
        bn = match["bn"]
        if_mod = match["if_mod"]
        thresh = (
            float(if_mod.thresh.detach().float().abs().reshape(-1)[0].clamp(min=1e-8))
            if if_mod is not None and getattr(if_mod, "thresh", None) is not None
            else float("nan")
        )
        gamma_mean = float("nan")
        folded_scale_mean = float("nan")
        folded_gain = float("nan")
        if bn is not None and getattr(bn, "weight", None) is not None:
            gamma = bn.weight.detach().float()
            var = bn.running_var.detach().float()
            eps = float(getattr(bn, "eps", 1e-5))
            scale = gamma / torch.sqrt(var + eps)
            gamma_mean = float(gamma.mean())
            folded_scale_mean = float(scale.abs().mean())
            folded = weight * scale.reshape([-1] + [1] * (weight.ndim - 1))
            filter_norm = float(folded.reshape(folded.shape[0], -1).norm(dim=1).mean())
            if math.isfinite(thresh):
                folded_gain = filter_norm / max(abs(thresh), 1e-8)
        rows.append(
            {
                "name": match["name"],
                "if_name": match["if_name"],
                "bn_name": match["bn_name"],
                "matched": int(bool(match["matched"])),
                "is_head": int(bool(match["is_head"])),
                "if_thresh": thresh,
                "bn_gamma_mean": gamma_mean,
                "folded_scale_mean": folded_scale_mean,
                "folded_gain": folded_gain,
                "weight_frobenius": float(torch.linalg.vector_norm(weight)),
            }
        )
    return rows


def scale_stats(model, layer_map: str) -> dict:
    thresh = [
        float(module.thresh.detach().float().mean())
        for module in model.modules()
        if isinstance(module, IF) and getattr(module, "thresh", None) is not None
    ]
    gamma = [
        float(module.weight.detach().float().mean())
        for module in model.modules()
        if isinstance(module, nn.BatchNorm2d) and getattr(module, "weight", None) is not None
    ]
    rows = layer_scale_rows(model, layer_map)
    matched = [row for row in rows if row["matched"]]
    return {
        "if_n": len(thresh),
        "if_thresh_mean": _mean(thresh),
        "if_thresh_median": _median(thresh),
        "if_thresh_min": min(thresh) if thresh else float("nan"),
        "if_thresh_max": max(thresh) if thresh else float("nan"),
        "bn_n": len(gamma),
        "bn_gamma_mean": _mean(gamma),
        "bn_gamma_median": _median(gamma),
        "folded_scale_mean": _mean([row["folded_scale_mean"] for row in matched]),
        "folded_gain_geomean": _geomean(
            [row["folded_gain"] for row in matched if math.isfinite(row["folded_gain"])]
        ),
    }


def ann_from_log(path: Path) -> dict:
    if path is None or not path.is_file():
        return {
            "ann_test_last": float("nan"),
            "ann_test_best": float("nan"),
        }
    values = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("ann_test_acc", "")
            if raw in (None, ""):
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
    return {
        "ann_test_last": values[-1] if values else float("nan"),
        "ann_test_best": max(values) if values else float("nan"),
    }


def train_cmd(args) -> list[str]:
    out = cfg_dir(args)
    checkpoint = ckpt_path(args)
    suffix = f"{config_name(args.grad)}_seed{args.seed}_L{LVAL}_trainT0"
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
        "--ckpt-dir", str(checkpoint.parent),
        "-suffix", suffix,
        "--regularizer", "mne_l2_unmatched",
        "--weight_decay", "0",
        "--reg_coeff", str(MNE_RC),
        "--unmatched_l2_coeff", str(L2_WD),
        "--mne_unmatched_scope", "head",
        "--mne_layer_map", LAYER_MAP,
        "--mapping_diag_dir", str(out / "mapping_init"),
        "--epoch_log_csv", str(out / "epoch_log.csv"),
    ]
    cmd += list(GRAD_FLAGS[args.grad])
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
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def self_check() -> None:
    model = modelpool(ARCH, "cifar10")
    names = [row["name"] for row in unmatched_weight_rows(model, "head", layer_map=LAYER_MAP)]
    if names != ["classifier.7"]:
        raise AssertionError(f"expected unmatched head ['classifier.7'], got {names}")
    ns = argparse.Namespace(
        dataset="cifar10",
        grad="gamma",
        seed=SEED,
        epochs=EPOCHS,
        batch_size=128,
        workers=8,
        device="cpu",
        out_root=ROOT.parent / "important_results" / "cifar_fgmneu_grad_ablation_seed42",
    )
    gamma = train_cmd(ns)
    if "--mne_detach_lambda" not in gamma or "--mne_no_detach_bn_affine" not in gamma:
        raise AssertionError(f"gamma flags wrong: {gamma}")
    if gamma[gamma.index("--regularizer") + 1] != "mne_l2_unmatched":
        raise AssertionError("ablation must keep unmatched-weight L2")
    if gamma[gamma.index("--unmatched_l2_coeff") + 1] != str(L2_WD):
        raise AssertionError("η_U must stay locked to L2-wo WD")
    if gamma[gamma.index("--mne_layer_map") + 1] != LAYER_MAP:
        raise AssertionError("VGG ablation must use legacy map")
    ns.grad = "lambda"
    lam = train_cmd(ns)
    if "--mne_detach_lambda" in lam:
        raise AssertionError("lambda-only must not detach λ")
    if "--mne_no_detach_bn_affine" in lam:
        raise AssertionError("lambda-only must keep γ detached")
    ns.grad = "full"
    full = train_cmd(ns)
    if "--mne_no_detach_bn_affine" not in full or "--mne_detach_lambda" in full:
        raise AssertionError(f"full flags wrong: {full}")
    ns.grad = "detach"
    det = train_cmd(ns)
    if "--mne_detach_lambda" not in det or "--mne_no_detach_bn_affine" in det:
        raise AssertionError(f"detach flags wrong: {det}")
    print("[self-check] FG-MNE-U grad flags ok", flush=True)


def _load_reuse_card(grad: str, dataset: str) -> dict | None:
    src = reuse_dir(grad, dataset)
    path = src / "scorecard.json" if src is not None else None
    if path is None or not path.is_file():
        return None
    card = json.loads(path.read_text())
    nabla_gamma, nabla_lambda = GRAD_NABLA[grad]
    card.update(
        {
            "grad": grad,
            "nabla_gamma": nabla_gamma,
            "nabla_lambda": nabla_lambda,
            "dataset": dataset,
            "historical_checkpoint_reused": True,
            "reuse_source": str(path),
            "method_name": "FG-MNE-U",
        }
    )
    return card


def summarize(out_root: Path) -> None:
    cards = [
        json.loads(path.read_text())
        for path in sorted(out_root.glob("*/*/seed*/scorecard.json"))
    ]
    seen = {(card.get("dataset"), card.get("grad")) for card in cards}
    for dataset in ("cifar10", "cifar100"):
        for grad in ("detach", "full"):
            if (dataset, grad) in seen:
                continue
            reused = _load_reuse_card(grad, dataset)
            if reused is not None:
                cards.append(reused)
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    order = {name: i for i, name in enumerate(GRADS)}
    cards.sort(key=lambda c: (c.get("dataset", ""), order.get(c.get("grad", ""), 99)))
    print(
        f"{'dataset':<9} {'grad':<8} {'dγ':>3} {'dλ':>3} "
        f"{'clean':>7} {'s5':>7} {'aucH':>8} {'λmean':>7} {'γmean':>7} "
        f"{'gain':>7} {'||W||F':>8} {'gap':>7} {'ρ5':>7}"
    )
    for card in cards:
        print(
            f"{card.get('dataset', ''):<9} {card.get('grad', ''):<8} "
            f"{int(card.get('nabla_gamma', 0)):3d} {int(card.get('nabla_lambda', 0)):3d} "
            f"{float(card.get('test_clean', float('nan'))):7.2f} "
            f"{float(card.get('test_sigma5', float('nan'))):7.2f} "
            f"{float(card.get('test_auc_high', float('nan'))):8.2f} "
            f"{float(card.get('if_thresh_mean', float('nan'))):7.3f} "
            f"{float(card.get('bn_gamma_mean', float('nan'))):7.3f} "
            f"{float(card.get('folded_gain_geomean', float('nan'))):7.3f} "
            f"{float(card.get('classifier_frobenius', float('nan'))):8.3f} "
            f"{float(card.get('ann_snn_gap', float('nan'))):7.2f} "
            f"{float(card.get('rho_true_sigma5_median', float('nan'))):7.3f}"
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

    nabla_gamma, nabla_lambda = GRAD_NABLA[args.grad]
    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] FG-MNE-U grad={args.grad} ∇γ={nabla_gamma} ∇λ={nabla_lambda} "
        f"{args.dataset} seed={args.seed} η_MNE={MNE_RC} η_U={L2_WD} map={LAYER_MAP}",
        flush=True,
    )

    reused = False
    checkpoint = ckpt_path(args)
    if args.grad in ("detach", "full") and not args.retrain:
        src = reuse_dir(args.grad, args.dataset)
        src_ckpts = sorted(src.glob("checkpoints/*.pth")) if src is not None else []
        if src_ckpts:
            checkpoint = src_ckpts[0]
            reused = True
            print(f"[REUSE CKPT] {checkpoint}", flush=True)
        elif (src / "scorecard.json").is_file():
            print(f"[REUSE CARD] {src / 'scorecard.json'}", flush=True)
            card = json.loads((src / "scorecard.json").read_text())
            card.update(
                {
                    "grad": args.grad,
                    "nabla_gamma": nabla_gamma,
                    "nabla_lambda": nabla_lambda,
                    "historical_checkpoint_reused": True,
                    "reuse_source": str(src / "scorecard.json"),
                    "method_name": "FG-MNE-U",
                }
            )
            (out / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
            for name in ("test_sweep.csv", "val_sweep.csv", "margin_summary.json"):
                if (src / name).is_file():
                    (out / name).write_bytes((src / name).read_bytes())
            print(json.dumps(card, indent=2), flush=True)
            return
        else:
            raise FileNotFoundError(f"reuse missing for {args.grad}: {src}")
    else:
        checkpoint = train(args)

    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = load_model(checkpoint, device, ARCH, args.dataset)
    scope = unmatched_scope_card(model, LAYER_MAP)
    (out / "unmatched_scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    dump_mne_mapping_report(model, out / "mapping_eval", layer_map=LAYER_MAP, quant_level=LVAL)
    norms = classifier_norms(model)
    scales = scale_stats(model, LAYER_MAP)
    write_csv(out / "layer_scale.csv", layer_scale_rows(model, LAYER_MAP))
    val_rows = sweep(model, val_loader(args, pin), device, "val", args.eval_seed)
    write_csv(out / "val_sweep.csv", val_rows)
    test_rows = sweep(model, test_loader(args, pin), device, "test", args.eval_seed)
    write_csv(out / "test_sweep.csv", test_rows)
    log = out / "epoch_log.csv"
    if reused:
        src = reuse_dir(args.grad, args.dataset)
        src_log = src / "epoch_log.csv" if src is not None else None
        if src_log is not None and src_log.is_file() and not log.is_file():
            log.write_bytes(src_log.read_bytes())
    ann = ann_from_log(log)
    margin = {}
    if not args.skip_diag:
        margin = run_margin_diag(
            model, test_loader(args, pin), device, args, log if log.is_file() else Path("")
        )
        (out / "margin_summary.json").write_text(json.dumps(margin, indent=2) + "\n")
        if margin.get("layer_scale"):
            write_csv(out / "if_probe.csv", margin["layer_scale"])
    card = {
        "config": config_name(args.grad),
        "method_name": "FG-MNE-U",
        "grad": args.grad,
        "nabla_gamma": nabla_gamma,
        "nabla_lambda": nabla_lambda,
        "dataset": args.dataset,
        "arch": ARCH,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "regularizer": "mne_l2_unmatched",
        "layer_map": LAYER_MAP,
        "eta_mne": MNE_RC,
        "eta_u": L2_WD,
        "historical_checkpoint_reused": reused,
        "checkpoint": str(checkpoint),
        "selection_uses_test": False,
        "T": TEST_T,
        "L": LVAL,
        **scope,
        **norms,
        **scales,
        **ann,
        **flatten_margin(margin),
    }
    card.update(snn_metrics(val_rows, "val"))
    card.update(snn_metrics(test_rows, "test"))
    test_clean = float(card.get("test_clean", float("nan")))
    card["ann_snn_gap"] = float(card["ann_test_last"] - test_clean) if math.isfinite(
        card["ann_test_last"]
    ) and math.isfinite(test_clean) else float("nan")
    (out / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2), flush=True)


if __name__ == "__main__":
    main()
