#!/usr/bin/env python3
"""CIFAR ResNet-18 seed-42 Old MNE-L2 layer-role ablation.

Do not retune α/τ. Same Old MNE recipe: detach λ, rc=1e-4.

Question: after ResNet-aware matching adds 11 layers (8 residual
terminals + 3 shortcuts), which of those extras change high-noise
behaviour relative to legacy (stem + residual_preact, 9/21)?

Variants
--------
    legacy           9 layers, Sequential Conv-BN-IF (reuse)
    aware            20 layers, unmatched strength (reuse)
    aware_gmatch     20 layers, ||∇R|| matched to legacy (reuse)
    no_shortcut      aware minus 3 shortcuts
    no_terminal      aware minus 8 residual terminals
    terminals_only   8 residual_function.3 only
    shortcuts_only   3 projection shortcuts only

New trains: no_shortcut, no_terminal, terminals_only, shortcuts_only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Models import modelpool  # noqa: E402
from run_cifar_resnet18_four_regs_5seed import ARCH  # noqa: E402
from run_cifar_vgg16_onesided_q_assignment_ablation import (  # noqa: E402
    EPOCHS,
    LR,
    LVAL,
    TEST_T,
    snn_metrics,
    sweep,
    test_loader,
    val_loader,
    write_csv,
)
from utils import (  # noqa: E402
    ce_vs_reg_grad_ratio,
    collect_weight_layer_matches,
    compute_mne_l2_regularization,
    dump_mne_mapping_report,
    get_torch_device,
    parse_mne_include_roles,
    summarize_weight_layer_matches,
)

SEED = 42
MNE_RC = 1e-4
MAPPING_REUSE_DEFAULT = Path(
    "/scratch/gs14/sl9144/snn_results/cifar_resnet18_mapping_diag_seed42"
)
INTENSITY_REUSE_DEFAULT = Path(
    "/scratch/gs14/sl9144/snn_results/cifar_resnet18_intensity_diag_seed42"
)
FOUR_REGS_DEFAULT = Path(
    "/scratch/gs14/sl9144/snn_results/cifar_resnet18_four_regs_5seed"
)

STEM_PREACT = "stem,residual_preact"
NO_SHORTCUT = "stem,residual_preact,residual_terminal"
NO_TERMINAL = "stem,residual_preact,shortcut"

VARIANTS = {
    "legacy": {
        "label": "Old MNE legacy (stem+preact)",
        "layer_map": "legacy",
        "include_roles": "",
        "grad_match": "",
        "reuse": (("mapping", "mapdiag_mne_legacy"), ("four_regs", "r18_mne")),
    },
    "aware": {
        "label": "Old MNE ResNet-aware, unmatched",
        "layer_map": "resnet",
        "include_roles": "",
        "grad_match": "",
        "reuse": (("mapping", "mapdiag_mne_resnet"),),
    },
    "aware_gmatch": {
        "label": "Old MNE ResNet-aware, ||∇R|| match legacy",
        "layer_map": "resnet",
        "include_roles": "",
        "grad_match": "legacy",
        "reuse": (("intensity", "intdiag_mne_aware_gmatch"),),
    },
    "no_shortcut": {
        "label": "Aware minus shortcuts",
        "layer_map": "resnet",
        "include_roles": NO_SHORTCUT,
        "grad_match": "",
        "reuse": None,
    },
    "no_terminal": {
        "label": "Aware minus residual terminals",
        "layer_map": "resnet",
        "include_roles": NO_TERMINAL,
        "grad_match": "",
        "reuse": None,
    },
    "terminals_only": {
        "label": "Residual terminals only",
        "layer_map": "resnet",
        "include_roles": "residual_terminal",
        "grad_match": "",
        "reuse": None,
    },
    "shortcuts_only": {
        "label": "Shortcuts only",
        "layer_map": "resnet",
        "include_roles": "shortcut",
        "grad_match": "",
        "reuse": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default=None)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument(
        "--reuse-root",
        type=Path,
        default=Path(os.environ.get("REUSE_ROOT", str(MAPPING_REUSE_DEFAULT))),
    )
    parser.add_argument(
        "--intensity-root",
        type=Path,
        default=Path(os.environ.get("INTENSITY_ROOT", str(INTENSITY_REUSE_DEFAULT))),
    )
    parser.add_argument(
        "--four-regs-root",
        type=Path,
        default=Path(os.environ.get("FOUR_REGS_ROOT", str(FOUR_REGS_DEFAULT))),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar_resnet18_mne_role_ablation_seed42",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    if args.variant is None:
        parser.error("specify --variant")
    return args


def config_name(variant: str) -> str:
    return f"mneabl_{variant}"


def suffix(args) -> str:
    return f"{config_name(args.variant)}_seed{args.seed}_L{LVAL}_trainT0"


def ckpt_filename(args) -> str:
    return f"{ARCH}_L[{LVAL}]_{suffix(args)}.pth"


def cfg_dir(args) -> Path:
    return args.out_root / args.dataset / config_name(args.variant)


def ckpt_path(args) -> Path:
    return cfg_dir(args) / "checkpoints" / ckpt_filename(args)


def reuse_ckpt(args) -> Path | None:
    spec = VARIANTS[args.variant]
    if spec["reuse"] is None:
        return None
    candidates = []
    for kind, folder in spec["reuse"]:
        if kind == "mapping":
            name = f"{ARCH}_L[{LVAL}]_{folder}_seed{args.seed}_L{LVAL}_trainT0.pth"
            candidates.append(
                args.reuse_root / args.dataset / folder / "checkpoints" / name
            )
        elif kind == "intensity":
            name = f"{ARCH}_L[{LVAL}]_{folder}_seed{args.seed}_L{LVAL}_trainT0.pth"
            candidates.append(
                args.intensity_root / args.dataset / folder / "checkpoints" / name
            )
        elif kind == "four_regs":
            name = f"{ARCH}_L[{LVAL}]_{folder}_seed{args.seed}_L{LVAL}_trainT0.pth"
            candidates.append(
                args.four_regs_root
                / args.dataset
                / folder
                / f"seed{args.seed}"
                / "checkpoints"
                / name
            )
        else:
            raise ValueError(f"unknown reuse kind {kind!r}")
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_model(ckpt: Path, device, dataset: str, spec):
    model = modelpool(ARCH, dataset)
    model._mne_layer_map = spec["layer_map"]
    model._mne_grad_match_layer_map = spec["grad_match"] or None
    model._mne_include_roles = spec["include_roles"] or None
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.set_L(LVAL)
    model.set_T(TEST_T)
    model.set_mode("rate_uniform")
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule("normal")
    if hasattr(model, "set_first_layer_input_noise_position"):
        model.set_first_layer_input_noise_position("post_input_if")
    if hasattr(model, "set_first_layer_input_noise_type"):
        model.set_first_layer_input_noise_type("gaussian")
    return model.to(device).eval()


def make_reg_fn(spec):
    return lambda m, t, q: compute_mne_l2_regularization(
        m,
        quant_level=LVAL,
        detach_lambda=True,
        grad_match_layer_map=spec["grad_match"] or None,
        include_roles=spec["include_roles"] or None,
    )


def included_layer_stats(model, spec) -> dict:
    roles = parse_mne_include_roles(spec["include_roles"])
    rows = collect_weight_layer_matches(model, spec["layer_map"])
    kept = [
        row
        for row in rows
        if row["matched"] and (roles is None or row["role"] in roles)
    ]
    counts = Counter(row["role"] for row in kept)
    n_params = sum(int(row["n_params"]) for row in kept)
    n_all = sum(int(row["n_params"]) for row in rows if not row["is_head"])
    return {
        "n_included_layers": len(kept),
        "n_matched_layers": sum(1 for row in rows if row["matched"]),
        "included_roles": ",".join(roles) if roles else "all",
        "included_role_counts": dict(counts),
        "included_param_ratio": (n_params / n_all) if n_all else 0.0,
    }


def train(args) -> Path:
    spec = VARIANTS[args.variant]
    out = cfg_dir(args)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_path(args)
    if ckpt.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {ckpt}", flush=True)
        return ckpt
    reused = None if args.retrain else reuse_ckpt(args)
    if reused is not None and not args.test_only:
        shutil.copy2(reused, ckpt)
        print(f"[REUSE CKPT] {reused} -> {ckpt}", flush=True)
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
        "--regularizer", "mne_l2",
        "--weight_decay", "0.0",
        "--reg_coeff", str(MNE_RC),
        "--mne_detach_lambda",
        "--mne_layer_map", spec["layer_map"],
        "--epoch_log_csv", str(out / "epoch_log.csv"),
        "--mapping_diag_dir", str(out / "mapping_init"),
    ]
    if spec["include_roles"]:
        cmd += ["--mne_include_roles", spec["include_roles"]]
    if spec["grad_match"]:
        cmd += ["--mne_grad_match_layer_map", spec["grad_match"]]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"training finished but missing {ckpt}")
    return ckpt


def main() -> None:
    args = parse_args()
    spec = VARIANTS[args.variant]
    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] {args.dataset} ResNet-18 {spec['label']} seed={args.seed} "
        f"layer_map={spec['layer_map']} roles={spec['include_roles'] or 'all'} "
        f"grad_match={spec['grad_match'] or 'none'}",
        flush=True,
    )
    ckpt = train(args)
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = load_model(ckpt, device, args.dataset, spec)
    role_stats = included_layer_stats(model, spec)
    report = dump_mne_mapping_report(
        model,
        out,
        layer_map=spec["layer_map"],
        quant_level=LVAL,
        extra={
            "legacy_match": summarize_weight_layer_matches(
                collect_weight_layer_matches(model, "legacy")
            ),
            "resnet_match": summarize_weight_layer_matches(
                collect_weight_layer_matches(model, "resnet")
            ),
            **role_stats,
        },
    )
    eval_args = argparse.Namespace(
        dataset=args.dataset,
        batch_size=args.batch_size,
        workers=0,
    )
    val_rows = sweep(model, val_loader(eval_args, pin), device, "val", args.seed)
    write_csv(out / "val_sweep.csv", val_rows)
    test_rows = sweep(model, test_loader(eval_args, pin), device, "test", args.seed)
    write_csv(out / "test_sweep.csv", test_rows)
    grads = {
        "ce_grad_norm": float("nan"),
        "reg_grad_norm": float("nan"),
        "reg_coeff_grad_norm": float("nan"),
        "reg_ce_ratio": float("nan"),
        "ce_loss": float("nan"),
        "reg_loss": float("nan"),
    }
    try:
        images, labels = next(iter(val_loader(eval_args, pin)))
        grads = ce_vs_reg_grad_ratio(
            model,
            images,
            labels,
            nn.CrossEntropyLoss(),
            make_reg_fn(spec),
            0,
            LVAL,
            MNE_RC,
        )
    except Exception as exc:
        print(f"[WARN] ce/reg grad ratio failed after sweep: {exc}", flush=True)
    match_stats = getattr(model, "_mne_grad_match_stats", {}) or {}
    card = {
        "config": config_name(args.variant),
        "label": spec["label"],
        "variant": args.variant,
        "dataset": args.dataset,
        "arch": ARCH,
        "seed": args.seed,
        "layer_map": spec["layer_map"],
        "include_roles": spec["include_roles"] or "all",
        "grad_match_layer_map": spec["grad_match"] or None,
        "regularizer": "mne_l2",
        "reg_coeff": MNE_RC,
        "checkpoint": str(ckpt),
        "matched_layers_over_total": report["matched_layers_over_total"],
        **role_stats,
        "mne_grad_match_scale": match_stats.get("scale"),
        **grads,
        **snn_metrics(val_rows, "val"),
        **snn_metrics(test_rows, "test"),
    }
    (out / "scorecard.json").write_text(json.dumps(card, indent=2, default=str) + "\n")
    print(json.dumps(card, indent=2, default=str), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
