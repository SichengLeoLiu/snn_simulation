#!/usr/bin/env python3
"""CIFAR ResNet-18 seed-42 MNE → One-sided mathematical bridge.

Same code path, ResNet-aware map, layer-mean 1/C_l, detached λ/γ/v,
BN fold, unmatched head skipped, rc=1e-4, no α warmup.

    R(q) = Σ_l (1/C_l) Σ_c q_{l,c} ||W_{l,c}||_F^2

    uniform     q=1
    raw         q=r                         (must reproduce Old MNE)
    normalized  q=r / r̄
    clipped     Renorm(clip(r/r̄, 0.5, 8))
    onesided    Renorm(1+α[r̃-τ]_+), α=4, τ=0.5
    mne_ref     existing Old MNE ResNet-aware checkpoint (eval only)

r̄ and Renorm use equal layer weight. Non-raw variants freeze
s=||∇R_raw||/||∇R_k|| at the first regularized epoch.

Do not retune α/τ from test. First stage: CIFAR-10 only.
If raw_vs_mne_relerr is not ~0, stop before the later transforms.
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
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Models import modelpool  # noqa: E402
from run_cifar_resnet18_four_regs_5seed import ARCH  # noqa: E402
from run_cifar_vgg16_onesided_q_assignment_ablation import (  # noqa: E402
    ALPHA,
    EPOCHS,
    LR,
    LVAL,
    RISK_MAX,
    RISK_MIN,
    TAU,
    TEST_T,
    acc_at,
    auc_range,
    snn_metrics,
    test_loader,
    val_loader,
    write_csv,
)
import measure_vgg16_horowitz_energy as energy  # noqa: E402
from utils import (  # noqa: E402
    ce_vs_reg_grad_ratio,
    collect_weight_layer_matches,
    compute_mne_l2_regularization,
    compute_mne_os_bridge_regularization,
    get_torch_device,
    seed_all,
    summarize_weight_layer_matches,
)

SEED = 42
MNE_RC = 1e-4
SIGMAS = [i / 2 for i in range(0, 11)]  # 0, 0.5, ..., 5
MAPPING_REUSE_DEFAULT = Path(
    "/scratch/gs14/sl9144/snn_results/cifar_resnet18_mapping_diag_seed42"
)

VARIANTS = {
    "uniform": {
        "label": "Uniform q=1",
        "transform": "uniform",
        "reuse": None,
    },
    "raw": {
        "label": "Raw q=r (Old MNE bridge)",
        "transform": "raw",
        "reuse": None,
    },
    "normalized": {
        "label": "Normalized q=r/r̄",
        "transform": "normalized",
        "reuse": None,
    },
    "clipped": {
        "label": "Clipped Renorm(clip(r/r̄,a,b))",
        "transform": "clipped",
        "reuse": None,
    },
    "onesided": {
        "label": "One-sided Renorm(1+α[r̃-τ]_+)",
        "transform": "onesided",
        "reuse": None,
    },
    "mne_ref": {
        "label": "Old MNE ResNet-aware (reference)",
        "transform": None,
        "reuse": "mapdiag_mne_resnet",
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe-raw", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--reuse-root",
        type=Path,
        default=Path(os.environ.get("REUSE_ROOT", str(MAPPING_REUSE_DEFAULT))),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar_resnet18_mne_os_bridge_seed42",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    if args.summarize or args.probe_raw:
        return args
    if args.variant is None:
        parser.error("specify --variant")
    return args


def config_name(variant: str) -> str:
    return f"bridge_{variant}"


def suffix(args) -> str:
    return f"{config_name(args.variant)}_seed{args.seed}_L{LVAL}_trainT0"


def ckpt_filename(args) -> str:
    return f"{ARCH}_L[{LVAL}]_{suffix(args)}.pth"


def cfg_dir(args) -> Path:
    return args.out_root / args.dataset / config_name(args.variant)


def ckpt_path(args) -> Path:
    return cfg_dir(args) / "checkpoints" / ckpt_filename(args)


def reuse_ckpt(args) -> Path | None:
    folder = VARIANTS[args.variant]["reuse"]
    if folder is None:
        return None
    name = f"{ARCH}_L[{LVAL}]_{folder}_seed{args.seed}_L{LVAL}_trainT0.pth"
    path = args.reuse_root / args.dataset / folder / "checkpoints" / name
    return path if path.is_file() else None


def load_model(ckpt: Path, device, dataset: str):
    model = modelpool(ARCH, dataset)
    model._mne_layer_map = "resnet"
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
    if spec["transform"] is None:
        return lambda m, t, q: compute_mne_l2_regularization(
            m, quant_level=LVAL, detach_lambda=True
        )
    return lambda m, t, q: compute_mne_os_bridge_regularization(
        m,
        quant_level=LVAL,
        risk_transform=spec["transform"],
        clip_min=RISK_MIN,
        clip_max=RISK_MAX,
        alpha=ALPHA,
        tau=TAU,
    )


def train_cmd(args) -> list[str]:
    spec = VARIANTS[args.variant]
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
        "--ckpt-dir", str(out / "checkpoints"),
        "-suffix", suffix(args),
        "--regularizer", "mne_os_bridge",
        "--weight_decay", "0",
        "--reg_coeff", str(MNE_RC),
        "--mne_layer_map", "resnet",
        "--bridge_risk_transform", spec["transform"],
        "--bridge_clip_min", str(RISK_MIN),
        "--bridge_clip_max", str(RISK_MAX),
        "--bridge_alpha", str(ALPHA),
        "--bridge_tau", str(TAU),
        "--epoch_log_csv", str(out / "epoch_log.csv"),
        "--mapping_diag_dir", str(out / "mapping_init"),
    ]
    if spec["transform"] != "raw":
        cmd.append("--bridge_match_raw_grad")
    return cmd


def train(args) -> Path:
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
    if VARIANTS[args.variant]["transform"] is None:
        raise FileNotFoundError(
            f"mne_ref requires the mapping-diag Old MNE checkpoint under {args.reuse_root}"
        )
    if args.test_only:
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        return ckpt
    cmd = train_cmd(args)
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return ckpt
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"training finished but missing {ckpt}")
    return ckpt


def sweep(model, loader, device, split: str, seed: int) -> list[dict]:
    rows = []
    for sigma in SIGMAS:
        seed_all(seed)
        model.set_first_layer_input_noise_sigma(float(sigma))
        result = energy.evaluate(model, loader, device, TEST_T, 0)
        use_energy = "yes" if abs(sigma) < 1e-12 else "no"
        rows.append(
            {
                "split": split,
                "sigma": f"{sigma:g}",
                "accuracy": f"{result['accuracy']:.6f}",
                "if_firing_density": f"{result['if_firing_density']:.6f}",
                "energy_mJ": f"{result['energy_mJ']:.8f}",
                "use_for_energy": use_energy,
                "n_samples": result["n_samples"],
                "time_steps": TEST_T,
            }
        )
        extra = "" if use_energy == "yes" else " [diag energy]"
        print(
            f"{split:<5} sigma={sigma:g} acc={result['accuracy']:.2f} "
            f"fire={result['if_firing_density']:.4f} "
            f"E={result['energy_mJ']:.4f} mJ{extra}",
            flush=True,
        )
    return rows


def extra_metrics(rows: list[dict], prefix: str) -> dict:
    card = snn_metrics(rows, prefix)
    card[f"{prefix}_sigma1"] = acc_at(rows, 1.0)
    card[f"{prefix}_sigma3"] = acc_at(rows, 3.0)
    card[f"{prefix}_auc_full"] = auc_range(rows, 0.0, 5.0)
    card[f"{prefix}_auc_high"] = auc_range(rows, 3.0, 5.0)
    return card


def write_layer_csv(path: Path, layers: list[dict]) -> None:
    if not layers:
        return
    write_csv(path, layers)


def probe_raw(dataset: str, seed: int) -> dict:
    seed_all(seed)
    model = modelpool(ARCH, dataset)
    model._mne_layer_map = "resnet"
    model.set_L(LVAL)
    raw = compute_mne_os_bridge_regularization(
        model, quant_level=LVAL, risk_transform="raw"
    )
    mne = compute_mne_l2_regularization(
        model, quant_level=LVAL, detach_lambda=True
    )
    relerr = abs(float(raw.detach()) - float(mne.detach())) / max(
        abs(float(mne.detach())), 1e-12
    )
    match = summarize_weight_layer_matches(
        collect_weight_layer_matches(model, "resnet")
    )
    card = {
        "dataset": dataset,
        "raw_R": float(raw.detach()),
        "mne_R": float(mne.detach()),
        "raw_vs_mne_relerr": relerr,
        **match,
    }
    print(json.dumps(card, indent=2), flush=True)
    return card


def summarize(out_root: Path) -> None:
    cards = []
    for path in sorted(out_root.glob("*/*/scorecard.json")):
        cards.append(json.loads(path.read_text()))
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    print(
        f"{'dataset':<10} {'variant':<12} {'relerr':>9} "
        f"{'test0':>7} {'test1':>7} {'test3':>7} {'test5':>7} {'p>τ':>6}"
    )
    for card in cards:
        print(
            f"{card['dataset']:<10} {card['variant']:<12} "
            f"{float(card.get('raw_vs_mne_relerr', float('nan'))):9.2e} "
            f"{card.get('test_clean', float('nan')):7.2f} "
            f"{card.get('test_sigma1', float('nan')):7.2f} "
            f"{card.get('test_sigma3', float('nan')):7.2f} "
            f"{card.get('test_sigma5', float('nan')):7.2f} "
            f"{float(card.get('p_gt_tau', float('nan'))):6.3f}"
        )
    write_csv(out_root / "bridge_seed42_summary.csv", cards)
    print(f"Wrote {out_root / 'bridge_seed42_summary.csv'}")


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(args.out_root)
        return
    if args.probe_raw:
        probe_raw(args.dataset, args.seed)
        return

    spec = VARIANTS[args.variant]
    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] {args.dataset} ResNet-18 {spec['label']} seed={args.seed} "
        f"transform={spec['transform']} rc={MNE_RC} layer_map=resnet",
        flush=True,
    )
    if args.dry_run:
        if spec["transform"] is None:
            print("[DRY RUN] mne_ref eval only", "reuse", reuse_ckpt(args), flush=True)
        else:
            print("[DRY RUN]", " ".join(train_cmd(args)), flush=True)
        return

    ckpt = train(args)
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = load_model(ckpt, device, args.dataset)
    eval_args = argparse.Namespace(
        dataset=args.dataset,
        batch_size=args.batch_size,
        workers=0,
    )
    val_rows = sweep(model, val_loader(eval_args, pin), device, "val", args.seed)
    write_csv(out / "val_sweep.csv", val_rows)
    test_rows = sweep(model, test_loader(eval_args, pin), device, "test", args.seed)
    write_csv(out / "test_sweep.csv", test_rows)

    init_card = {}
    init_path = out / "mapping_init" / "bridge_init.json"
    if init_path.is_file():
        init_card = json.loads(init_path.read_text())
    reg_fn = make_reg_fn(spec)
    try:
        _ = reg_fn(model, 0, LVAL)
    except Exception as exc:
        print(f"[WARN] post-hoc bridge stats failed: {exc}", flush=True)
    stats = getattr(model, "_bridge_stats", {}) or {}
    if stats.get("layers"):
        write_layer_csv(out / "layer_q.csv", stats["layers"])
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
            reg_fn,
            0,
            LVAL,
            MNE_RC,
        )
    except Exception as exc:
        print(f"[WARN] ce/reg grad ratio failed after sweep: {exc}", flush=True)
    match = summarize_weight_layer_matches(
        collect_weight_layer_matches(model, "resnet")
    )
    card = {
        "config": config_name(args.variant),
        "label": spec["label"],
        "variant": args.variant,
        "dataset": args.dataset,
        "arch": ARCH,
        "seed": args.seed,
        "layer_map": "resnet",
        "risk_transform": spec["transform"],
        "regularizer": "mne_l2" if spec["transform"] is None else "mne_os_bridge",
        "reg_coeff": MNE_RC,
        "alpha": ALPHA,
        "tau": TAU,
        "clip_min": RISK_MIN,
        "clip_max": RISK_MAX,
        "checkpoint": str(ckpt),
        "matched_layers_over_total": match.get("matched_layers_over_total"),
        "body_param_ratio": match.get("body_param_ratio"),
        "q_mean": stats.get("q_mean"),
        "q_std": stats.get("q_std"),
        "q_min": stats.get("q_min"),
        "q_max": stats.get("q_max"),
        "p_gt_tau": stats.get("p_gt_tau"),
        "r_bar": stats.get("r_bar"),
        "bridge_grad_scale": stats.get("bridge_grad_scale", init_card.get("bridge_grad_scale")),
        "raw_vs_mne_relerr": init_card.get("raw_vs_mne_relerr"),
        "g_raw": init_card.get("g_raw"),
        "g_k": init_card.get("g_k"),
        **grads,
        **extra_metrics(val_rows, "val"),
        **extra_metrics(test_rows, "test"),
    }
    (out / "scorecard.json").write_text(json.dumps(card, indent=2, default=str) + "\n")
    print(json.dumps(card, indent=2, default=str), flush=True)
    print(f"Wrote {out}", flush=True)
    if args.variant == "raw" and init_card:
        relerr = float(init_card.get("raw_vs_mne_relerr", float("nan")))
        if relerr != relerr or relerr > 1e-5:
            raise SystemExit(
                f"Raw does not match Old MNE at init (relerr={relerr:.3e}). "
                "Do not submit normalized/clipped/onesided until this is fixed."
            )


if __name__ == "__main__":
    main()
