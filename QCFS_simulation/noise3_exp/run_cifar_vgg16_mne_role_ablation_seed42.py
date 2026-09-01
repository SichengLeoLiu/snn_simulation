#!/usr/bin/env python3
"""CIFAR VGG-16 seed-42 Old MNE-L2 block-role ablation.

Do not retune α/τ. Same Old MNE recipe: detach λ, rc=1e-4.

VGG has no residual_terminal / shortcut. The analogue of the ResNet
role split is the five Sequential blocks (layer1–layer5):

    layer1  2 convs   stem / early
    layer2  2 convs
    layer3  3 convs
    layer4  3 convs
    layer5  3 convs   last conv block (terminal analogue)

Variants
--------
    all          all matched layers (reuse Old MNE seed-42)
    early        layer1+layer2
    late         layer4+layer5
    no_late      layer1–layer4
    no_early     layer2–layer5
    late_only    layer5 only
    early_only   layer1 only

New trains: every variant except all.
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
from Models.VGG import remap_legacy_vgg_state_dict  # noqa: E402
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

ARCH = "vgg16"
SEED = 42
MNE_RC = 1e-4
CKPT_ROOT_DEFAULT = Path("/home/595/sl9144/codes/snn_simulation/QCFS_simulation")

EARLY = "layer1,layer2"
LATE = "layer4,layer5"
NO_LATE = "layer1,layer2,layer3,layer4"
NO_EARLY = "layer2,layer3,layer4,layer5"

VARIANTS = {
    "all": {
        "label": "Old MNE all matched",
        "include_roles": "",
        "reuse": True,
    },
    "early": {
        "label": "layer1+layer2",
        "include_roles": EARLY,
        "reuse": False,
    },
    "late": {
        "label": "layer4+layer5",
        "include_roles": LATE,
        "reuse": False,
    },
    "no_late": {
        "label": "layer1-4, drop layer5",
        "include_roles": NO_LATE,
        "reuse": False,
    },
    "no_early": {
        "label": "layer2-5, drop layer1",
        "include_roles": NO_EARLY,
        "reuse": False,
    },
    "late_only": {
        "label": "layer5 only",
        "include_roles": "layer5",
        "reuse": False,
    },
    "early_only": {
        "label": "layer1 only",
        "include_roles": "layer1",
        "reuse": False,
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
        "--ckpt-root",
        type=Path,
        default=Path(os.environ.get("CKPT_ROOT", str(CKPT_ROOT_DEFAULT))),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar_vgg16_mne_role_ablation_seed42",
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
    if not VARIANTS[args.variant]["reuse"]:
        return None
    names = [
        f"{ARCH}_L[{LVAL}]_mneablate_{args.dataset}_old_detach_rc0p0001_seed{args.seed}_L{LVAL}_trainT0.pth",
        f"{ARCH}_L[{LVAL}]_mneablate_{args.dataset}_old_detach_rc0p0001_seed{args.seed}_L{LVAL}.pth",
    ]
    roots = [
        args.ckpt_root / f"{args.dataset}-checkpoints",
        ROOT / f"{args.dataset}-checkpoints",
        Path(f"/scratch/gs14/sl9144/snn_results/{args.dataset}_{ARCH}_five_regs_sigma0_5_5seed"),
    ]
    for root in roots:
        for name in names:
            path = root / name
            if path.is_file():
                return path
    return None


def load_model(ckpt: Path, device, dataset: str, spec):
    model = modelpool(ARCH, dataset)
    model._mne_layer_map = "legacy"
    model._mne_include_roles = spec["include_roles"] or None
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(remap_legacy_vgg_state_dict(state), strict=True)
    model.set_L(LVAL)
    model.set_T(TEST_T)
    model.set_mode("rate_uniform")
    model.set_spike_schedule("normal")
    model.set_first_layer_input_noise_position("post_input_if")
    model.set_first_layer_input_noise_type("gaussian")
    return model.to(device).eval()


def make_reg_fn(spec):
    return lambda m, t, q: compute_mne_l2_regularization(
        m,
        quant_level=LVAL,
        detach_lambda=True,
        include_roles=spec["include_roles"] or None,
    )


def included_layer_stats(model, spec) -> dict:
    roles = parse_mne_include_roles(spec["include_roles"])
    rows = collect_weight_layer_matches(model, "legacy")
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
        "--mne_layer_map", "legacy",
        "--epoch_log_csv", str(out / "epoch_log.csv"),
        "--mapping_diag_dir", str(out / "mapping_init"),
    ]
    if spec["include_roles"]:
        cmd += ["--mne_include_roles", spec["include_roles"]]
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
        f"[INFO] {args.dataset} VGG-16 {spec['label']} seed={args.seed} "
        f"roles={spec['include_roles'] or 'all'}",
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
        layer_map="legacy",
        quant_level=LVAL,
        extra={
            "legacy_match": summarize_weight_layer_matches(
                collect_weight_layer_matches(model, "legacy")
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
    card = {
        "config": config_name(args.variant),
        "label": spec["label"],
        "variant": args.variant,
        "dataset": args.dataset,
        "arch": ARCH,
        "seed": args.seed,
        "layer_map": "legacy",
        "include_roles": spec["include_roles"] or "all",
        "regularizer": "mne_l2",
        "reg_coeff": MNE_RC,
        "checkpoint": str(ckpt),
        "matched_layers_over_total": report["matched_layers_over_total"],
        **role_stats,
        **grads,
        **snn_metrics(val_rows, "val"),
        **snn_metrics(test_rows, "test"),
    }
    (out / "scorecard.json").write_text(json.dumps(card, indent=2, default=str) + "\n")
    print(json.dumps(card, indent=2, default=str), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
