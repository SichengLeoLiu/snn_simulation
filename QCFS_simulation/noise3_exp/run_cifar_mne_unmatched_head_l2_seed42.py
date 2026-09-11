#!/usr/bin/env python3
"""P0/P1 single-seed MNE-L2 + unmatched-head L2 screen.

Hybrid regularizer (do not use mne_l2_all):

    L = L_task + η_MNE R_MNE + η_head R_unmatched

    R_MNE        = sum_{l in S_IF} L^2 M_eff,l / (λ_l^2 + eps)
    R_unmatched  = (1/2) sum_{j not in S_IF} ||W_j||_F^2

BN-γ/β, bias and IF thresholds are never regularized. η_head is locked to
the L2-wo baseline WD=5e-4 so the head constraint is not test-tuned.

Methods
-------
    l2wo            reuse L2-wo (eval only)
    mne             reuse current MNE-L2 detach (eval only)
    mne_head        train: MNE on IF body, ordinary L2 on unmatched head
    mne_unmatched   train: MNE on IF body, ordinary L2 on all unmatched
                    Conv/Linear. On ResNet-18/resnet-map and VGG-16/legacy
                    this is the same layer as mne_head (final classifier).

Do not retrain the reused L2-wo / MNE directories. Do not pick η from the
test curve. Shared eval noise: EVAL_SEED=0, T=L=16, rate_uniform, post-IF.

``--body nodetach`` uses grads into λ and BN γ (``--mne_no_detach_bn_affine``,
no ``--mne_detach_lambda``) and reuses existing no-detach checkpoints.
ResNet hybrid training still uses the resnet layer map so it matches the
detach+head screen; reused ResNet four-regs no-detach was trained with the
default legacy map.
"""
from __future__ import annotations

import argparse
import json
import os
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
from Models.VGG import remap_legacy_vgg_state_dict  # noqa: E402
from run_cifar_resnet18_margin_noise_diag_seed42 import (  # noqa: E402
    DEFAULT_STREAMS,
    IfScaleProbe,
    collect_logits,
    epoch_log_ratio,
    error_breakdown,
    fine_to_coarse_map,
    load_coarse,
    mean_breakdown,
    summarize_vec,
    top12_margin,
    true_margin,
)
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
    collect_weight_layer_matches,
    dump_mne_mapping_report,
    get_torch_device,
    unmatched_weight_rows,
)

SEED = 42
EVAL_NOISE_SEED = 0
MNE_RC = 1e-4
L2_WD = 5e-4
SM_FLOOR = 1e-8
SCRATCH = Path("/scratch/gs14/sl9144/snn_results")
DEFAULT_DIAG_SIGMAS = (1.0, 3.0, 5.0)

NODETACH_REUSE = {
    "resnet18": {
        "mne": (
            SCRATCH
            / "cifar_resnet18_four_regs_5seed"
            / "{dataset}/r18_nodetach/seed{seed}/checkpoints"
            / "resnet18_L[16]_r18_nodetach_seed{seed}_L16_trainT0.pth"
        ),
        "log": SCRATCH / "cifar_resnet18_four_regs_5seed/{dataset}/r18_nodetach/seed{seed}/epoch_log.csv",
        "trained_layer_map": "legacy",
    },
    "vgg16": {
        "mne": (
            SCRATCH
            / "cifar_vgg16_mne_component_ablation_seed42"
            / "{dataset}/comp_nodetach_fixed/checkpoints"
            / "vgg16_L[16]_comp_nodetach_fixed_seed42_L16_trainT0.pth"
        ),
        "log": (
            SCRATCH
            / "cifar_vgg16_mne_component_ablation_seed42/{dataset}/comp_nodetach_fixed/epoch_log.csv"
        ),
        "trained_layer_map": "legacy",
    },
}

ARCH_SPECS = {
    "resnet18": {
        "layer_map": "resnet",
        "remap_vgg": False,
        "reuse": {
            "l2wo": (
                SCRATCH
                / "cifar_resnet18_four_regs_5seed"
                / "{dataset}/r18_l2wo/seed{seed}/checkpoints"
                / "resnet18_L[16]_r18_l2wo_seed{seed}_L16_trainT0.pth"
            ),
            "mne": (
                SCRATCH
                / "cifar_resnet18_fair_mne_detach"
                / "{dataset}/r18_mne_resnet_rc1e-4/seed{seed}/checkpoints"
                / "resnet18_L[16]_r18_mne_resnet_rc1e-4_seed{seed}_L16_trainT0.pth"
            ),
        },
        "log": {
            "l2wo": SCRATCH / "cifar_resnet18_four_regs_5seed/{dataset}/r18_l2wo/seed{seed}/epoch_log.csv",
            "mne": SCRATCH
            / "cifar_resnet18_fair_mne_detach/{dataset}/r18_mne_resnet_rc1e-4/seed{seed}/epoch_log.csv",
        },
    },
    "vgg16": {
        "layer_map": "legacy",
        "remap_vgg": True,
        "reuse": {
            "l2wo": (
                SCRATCH
                / "cifar_vgg16_mne_component_ablation_seed42"
                / "{dataset}/comp_l2wo_fixed/checkpoints"
                / "vgg16_L[16]_comp_l2wo_fixed_seed42_L16_trainT0.pth"
            ),
            "mne": (
                SCRATCH
                / "cifar_vgg16_mne_component_ablation_seed42"
                / "{dataset}/comp_mne_fixed/checkpoints"
                / "vgg16_L[16]_comp_mne_fixed_seed42_L16_trainT0.pth"
            ),
        },
        "log": {
            "l2wo": SCRATCH
            / "cifar_vgg16_mne_component_ablation_seed42/{dataset}/comp_l2wo_fixed/epoch_log.csv",
            "mne": SCRATCH
            / "cifar_vgg16_mne_component_ablation_seed42/{dataset}/comp_mne_fixed/epoch_log.csv",
        },
    },
}

METHODS = {
    "l2wo": {
        "label": "L2-wo",
        "train": False,
        "reuse_key": "l2wo",
        "regularizer": "weight_decay_weights_only",
        "scope": None,
    },
    "mne": {
        "label": "MNE-L2",
        "train": False,
        "reuse_key": "mne",
        "regularizer": "mne_l2",
        "scope": None,
    },
    "mne_head": {
        "label": "MNE-L2 + classifier-head L2",
        "train": True,
        "reuse_key": None,
        "regularizer": "mne_l2_unmatched",
        "scope": "head",
    },
    "mne_unmatched": {
        "label": "MNE-L2 + all-unmatched-weight L2",
        "train": True,
        "reuse_key": None,
        "regularizer": "mne_l2_unmatched",
        "scope": "all",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=tuple(ARCH_SPECS), default="resnet18")
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar100")
    parser.add_argument("--method", choices=tuple(METHODS), default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=int(os.environ.get("EVAL_SEED", str(EVAL_NOISE_SEED))),
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--skip-diag", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--body",
        choices=("detach", "nodetach"),
        default=os.environ.get("BODY", "detach"),
        help="IF-body MNE recipe. nodetach = grads into λ and BN γ.",
    )
    parser.add_argument("--n-streams", type=int, default=int(os.environ.get("N_STREAMS", str(DEFAULT_STREAMS))))
    parser.add_argument(
        "--sigmas",
        default=os.environ.get("DIAG_SIGMAS", "1,3,5"),
        help="comma-separated sigmas for s_m / rho",
    )
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args()
    args.eval_seed = int(args.eval_seed)
    args.n_streams = int(args.n_streams)
    args.body = str(args.body).strip().lower()
    args.sigmas = tuple(float(x) for x in str(args.sigmas).split(",") if x.strip())
    args.skip_diag = bool(args.skip_diag or os.environ.get("SKIP_DIAG", "0") == "1")
    if args.out_root is None:
        folder = (
            "cifar_mne_nodetach_unmatched_head_l2_seed42"
            if args.body == "nodetach"
            else "cifar_mne_unmatched_head_l2_seed42"
        )
        args.out_root = ROOT.parent / "important_results" / folder
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    if args.summarize or args.self_check:
        return args
    if args.method is None:
        parser.error("--method is required unless --summarize/--self-check")
    if args.n_streams < 2 and not args.skip_diag:
        parser.error("--n-streams must be >= 2")
    return args


def config_name(arch: str, method: str) -> str:
    return f"{arch}_{method}"


def suffix(args) -> str:
    return f"{config_name(args.arch, args.method)}_seed{args.seed}_L{LVAL}_trainT0"


def cfg_dir(args) -> Path:
    return args.out_root / args.dataset / config_name(args.arch, args.method) / f"seed{args.seed}"


def ckpt_path(args) -> Path:
    return cfg_dir(args) / "checkpoints" / f"{args.arch}_L[{LVAL}]_{suffix(args)}.pth"


def method_label(args) -> str:
    spec = METHODS[args.method]
    if args.method == "l2wo":
        return spec["label"]
    body = "no-detach" if args.body == "nodetach" else "detach"
    if args.method == "mne":
        return f"MNE-L2 {body}"
    if args.method == "mne_head":
        return f"MNE-L2 {body} + classifier-head L2"
    if args.method == "mne_unmatched":
        return f"MNE-L2 {body} + all-unmatched-weight L2"
    return spec["label"]


def _format_path(path: Path, dataset: str, seed: int) -> Path:
    return Path(str(path).format(dataset=dataset, seed=seed))


def reuse_ckpt(args) -> Path:
    spec = METHODS[args.method]
    key = spec["reuse_key"]
    if key is None:
        raise ValueError(f"{args.method} is not a reuse arm")
    if args.body == "nodetach" and key == "mne":
        return _format_path(NODETACH_REUSE[args.arch]["mne"], args.dataset, args.seed)
    return _format_path(ARCH_SPECS[args.arch]["reuse"][key], args.dataset, args.seed)


def reuse_log(args) -> Path | None:
    spec = METHODS[args.method]
    key = spec["reuse_key"]
    if key is None:
        log = cfg_dir(args) / "epoch_log.csv"
        return log if log.is_file() else None
    if args.body == "nodetach" and key == "mne":
        path = _format_path(NODETACH_REUSE[args.arch]["log"], args.dataset, args.seed)
    else:
        path = _format_path(ARCH_SPECS[args.arch]["log"][key], args.dataset, args.seed)
    return path if path.is_file() else None


def unmatched_scope_card(model, layer_map: str) -> dict:
    rows = collect_weight_layer_matches(model, layer_map)
    unmatched = [row for row in rows if not row["matched"]]
    head = [row for row in unmatched if row["is_head"]]
    all_names = [row["name"] for row in unmatched]
    head_names = [row["name"] for row in head]
    return {
        "layer_map": layer_map,
        "matched": [row["name"] for row in rows if row["matched"]],
        "unmatched_all": all_names,
        "unmatched_head": head_names,
        "unmatched_body": [row["name"] for row in unmatched if not row["is_head"]],
        "scopes_identical": all_names == head_names,
        "n_matched": sum(1 for row in rows if row["matched"]),
        "n_unmatched": len(unmatched),
        "n_unmatched_head": len(head),
        "n_unmatched_body": len(unmatched) - len(head),
    }


def classifier_norms(model) -> dict:
    last = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and getattr(module, "weight", None) is not None:
            last = (name, module.weight.detach().float())
    if last is None:
        return {
            "classifier_name": "",
            "classifier_frobenius": float("nan"),
            "classifier_spectral": float("nan"),
            "classifier_n_params": 0,
        }
    name, weight = last
    return {
        "classifier_name": name,
        "classifier_frobenius": float(torch.linalg.matrix_norm(weight, ord="fro")),
        "classifier_spectral": float(torch.linalg.matrix_norm(weight, ord=2)),
        "classifier_n_params": int(weight.numel()),
    }


def load_model(ckpt: Path, device, arch: str, dataset: str):
    spec = ARCH_SPECS[arch]
    model = modelpool(arch, dataset)
    model._mne_layer_map = spec["layer_map"]
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if spec["remap_vgg"]:
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


def train(args) -> Path:
    spec = METHODS[args.method]
    if not spec["train"]:
        raise ValueError(f"{args.method} is eval-only reuse")
    out = cfg_dir(args)
    checkpoint = ckpt_path(args)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {checkpoint}", flush=True)
        return checkpoint
    if args.test_only:
        raise FileNotFoundError(checkpoint)
    layer_map = ARCH_SPECS[args.arch]["layer_map"]
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
        "--ckpt-dir", str(checkpoint.parent),
        "-suffix", suffix(args),
        "--regularizer", "mne_l2_unmatched",
        "--weight_decay", "0",
        "--reg_coeff", str(MNE_RC),
        "--unmatched_l2_coeff", str(L2_WD),
        "--mne_unmatched_scope", spec["scope"],
        "--mne_layer_map", layer_map,
        "--mapping_diag_dir", str(out / "mapping_init"),
        "--epoch_log_csv", str(out / "epoch_log.csv"),
    ]
    if args.body == "nodetach":
        cmd.append("--mne_no_detach_bn_affine")
    else:
        cmd.append("--mne_detach_lambda")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not checkpoint.exists():
        raise FileNotFoundError(f"training finished but checkpoint is missing: {checkpoint}")
    return checkpoint


def resolve_checkpoint(args) -> tuple[Path, bool]:
    spec = METHODS[args.method]
    if spec["train"]:
        return train(args), False
    reused = reuse_ckpt(args)
    if not reused.is_file():
        raise FileNotFoundError(f"reuse checkpoint missing: {reused}")
    print(f"[REUSE] {reused}", flush=True)
    return reused, True


def run_margin_diag(model, loader, device, args, log_path: Path | None) -> dict:
    probe = IfScaleProbe(model)
    clean_logits, labels = collect_logits(
        model, loader, device, 0.0, args.eval_seed, 0, probe=probe
    )
    m_true = true_margin(clean_logits, labels)
    m_top12 = top12_margin(clean_logits)
    coarse = load_coarse(args.dataset, labels.numel())
    mapping = fine_to_coarse_map(labels, coarse) if coarse is not None else None
    clean_break = error_breakdown(clean_logits, labels, coarse, mapping)
    sigma_block = {}
    for sigma in args.sigmas:
        noisy_true = []
        breaks = []
        for stream in range(args.n_streams):
            stream_seed = int(args.eval_seed + stream)
            logits, _ = collect_logits(model, loader, device, sigma, stream_seed, 0)
            mt = true_margin(logits, labels)
            noisy_true.append(mt)
            breaks.append(error_breakdown(logits, labels, coarse, mapping))
            print(
                f"sigma={sigma:g} stream={stream} seed={stream_seed} "
                f"top1={breaks[-1]['top1_acc']:.2f}",
                flush=True,
            )
        stacked = torch.stack(noisy_true, dim=0)
        s_true = (stacked - m_true.unsqueeze(0)).std(dim=0, unbiased=True)
        rho_true = m_true / s_true.clamp(min=SM_FLOOR)
        sigma_block[f"{sigma:g}"] = {
            "sigma": float(sigma),
            "s_m_true": summarize_vec(s_true),
            "rho_true": summarize_vec(rho_true),
            "noisy": mean_breakdown(breaks),
        }
    layer_rows = probe.rows(args.sigmas)
    return {
        "clean": {
            "m_true": summarize_vec(m_true),
            "m_top12": summarize_vec(m_top12),
            **clean_break,
        },
        "noise": sigma_block,
        "layer_scale": layer_rows,
        "grad_ratio": epoch_log_ratio(log_path) if log_path is not None else epoch_log_ratio(Path("")),
        "n_samples": int(labels.numel()),
        "n_streams": int(args.n_streams),
        "sigmas": [float(sigma) for sigma in args.sigmas],
    }


def flatten_margin(margin: dict) -> dict:
    if not margin:
        return {}
    clean = margin.get("clean", {})
    noise = margin.get("noise", {})
    s5 = noise.get("5", noise.get("5.0", {}))
    out = {
        "clean_m_true_median": clean.get("m_true", {}).get("median", float("nan")),
        "clean_m_true_p10": clean.get("m_true", {}).get("p10", float("nan")),
        "clean_m_true_mean": clean.get("m_true", {}).get("mean", float("nan")),
        "s_m_true_sigma5_median": s5.get("s_m_true", {}).get("median", float("nan")),
        "rho_true_sigma5_median": s5.get("rho_true", {}).get("median", float("nan")),
        "noisy_top1_sigma5_mean": s5.get("noisy", {}).get("top1_acc", float("nan")),
    }
    return out


def scorecard(val_rows, test_rows, args, checkpoint: Path, reused: bool, extra: dict) -> dict:
    spec = METHODS[args.method]
    arch = ARCH_SPECS[args.arch]
    detach = args.body != "nodetach"
    card = {
        "config": config_name(args.arch, args.method),
        "label": method_label(args),
        "method": args.method,
        "body": args.body,
        "dataset": args.dataset,
        "arch": args.arch,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "regularizer": spec["regularizer"],
        "layer_map": arch["layer_map"],
        "detach_lambda": bool(detach) and (spec["train"] or args.method == "mne"),
        "detach_bn_affine": bool(detach),
        "unmatched_scope": spec["scope"],
        "reg_coeff": MNE_RC if spec["regularizer"] != "weight_decay_weights_only" else None,
        "unmatched_l2_coeff": L2_WD if spec["train"] else None,
        "eta_mne": MNE_RC if spec["regularizer"] != "weight_decay_weights_only" else None,
        "eta_head": L2_WD if spec["train"] else (L2_WD if args.method == "l2wo" else 0.0),
        "weight_decay": L2_WD if args.method == "l2wo" else 0.0,
        "historical_checkpoint_reused": reused,
        "checkpoint": str(checkpoint),
        "selection_uses_test": False,
        "protocol": {
            "T": TEST_T,
            "L": LVAL,
            "mode": "rate_uniform",
            "noise": "post_input_if gaussian",
            "eta_head_locked_to_l2wo": True,
            "body": args.body,
        },
    }
    if reused and args.body == "nodetach" and args.method == "mne":
        card["reused_checkpoint_trained_layer_map"] = NODETACH_REUSE[args.arch]["trained_layer_map"]
        if args.arch == "resnet18":
            card["resnet_nodetach_reuse_note"] = (
                "four-regs no-detach used default legacy map; hybrid trains with resnet map"
            )
    card.update(snn_metrics(val_rows, "val"))
    card.update(snn_metrics(test_rows, "test"))
    card.update(extra)
    return card


def self_check() -> None:
    for arch, spec in ARCH_SPECS.items():
        for dataset in ("cifar10", "cifar100"):
            model = modelpool(arch, dataset)
            model._mne_layer_map = spec["layer_map"]
            card = unmatched_scope_card(model, spec["layer_map"])
            head_rows = unmatched_weight_rows(model, "head")
            all_rows = unmatched_weight_rows(model, "all")
            print(
                f"{arch} {dataset} map={spec['layer_map']} "
                f"matched={card['n_matched']} unmatched_all={card['unmatched_all']} "
                f"unmatched_head={card['unmatched_head']} "
                f"identical={card['scopes_identical']}",
                flush=True,
            )
            if [row["name"] for row in head_rows] != card["unmatched_head"]:
                raise AssertionError("head unmatched rows drifted")
            if [row["name"] for row in all_rows] != card["unmatched_all"]:
                raise AssertionError("all unmatched rows drifted")
            if card["n_unmatched_body"] != 0:
                print(f"[WARN] unmatched body layers: {card['unmatched_body']}", flush=True)
    print("self-check ok", flush=True)


def summarize(out_root: Path) -> None:
    cards = [
        json.loads(path.read_text())
        for path in sorted(out_root.glob("*/*/seed*/scorecard.json"))
    ]
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    print(
        f"{'dataset':<9} {'arch':<9} {'method':<14} "
        f"{'clean':>7} {'s5':>7} {'aucH':>8} "
        f"{'||W||F':>8} {'||W||2':>8} "
        f"{'m_p50':>7} {'m_p10':>7} {'sm5':>7} {'rho5':>7} ident"
    )
    for card in cards:
        print(
            f"{card['dataset']:<9} {card['arch']:<9} {card['method']:<14} "
            f"{float(card.get('test_clean', float('nan'))):7.2f} "
            f"{float(card.get('test_sigma5', float('nan'))):7.2f} "
            f"{float(card.get('test_auc_high', float('nan'))):8.2f} "
            f"{float(card.get('classifier_frobenius', float('nan'))):8.3f} "
            f"{float(card.get('classifier_spectral', float('nan'))):8.3f} "
            f"{float(card.get('clean_m_true_median', float('nan'))):7.3f} "
            f"{float(card.get('clean_m_true_p10', float('nan'))):7.3f} "
            f"{float(card.get('s_m_true_sigma5_median', float('nan'))):7.3f} "
            f"{float(card.get('rho_true_sigma5_median', float('nan'))):7.3f} "
            f"{str(card.get('scopes_identical', ''))}"
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
    layer_map = ARCH_SPECS[args.arch]["layer_map"]
    print(
        f"[INFO] {args.arch} {args.dataset} {args.method} body={args.body} "
        f"layer_map={layer_map} seed={args.seed} eval_seed={args.eval_seed} "
        f"η_MNE={MNE_RC} η_head={L2_WD}",
        flush=True,
    )
    checkpoint, reused = resolve_checkpoint(args)
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = load_model(checkpoint, device, args.arch, args.dataset)
    scope = unmatched_scope_card(model, layer_map)
    (out / "unmatched_scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    dump_mne_mapping_report(model, out / "mapping_eval", layer_map=layer_map, quant_level=LVAL)
    print(
        f"[SCOPE] unmatched_head={scope['unmatched_head']} "
        f"unmatched_all={scope['unmatched_all']} identical={scope['scopes_identical']}",
        flush=True,
    )
    if args.method == "mne_unmatched" and scope["scopes_identical"]:
        print(
            "[NOTE] unmatched-all == unmatched-head on this model; "
            "mne_unmatched is the same regularizer as mne_head.",
            flush=True,
        )
    norms = classifier_norms(model)
    val_rows = sweep(model, val_loader(args, pin), device, "val", args.eval_seed)
    write_csv(out / "val_sweep.csv", val_rows)
    test_rows = sweep(model, test_loader(args, pin), device, "test", args.eval_seed)
    write_csv(out / "test_sweep.csv", test_rows)
    margin = {}
    if not args.skip_diag:
        margin = run_margin_diag(model, test_loader(args, pin), device, args, reuse_log(args))
        (out / "margin_summary.json").write_text(json.dumps(margin, indent=2) + "\n")
        if margin.get("layer_scale"):
            write_csv(out / "layer_scale.csv", margin["layer_scale"])
    extra = {
        **norms,
        **scope,
        **flatten_margin(margin),
    }
    card = scorecard(val_rows, test_rows, args, checkpoint, reused, extra)
    (out / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2), flush=True)


if __name__ == "__main__":
    main()
