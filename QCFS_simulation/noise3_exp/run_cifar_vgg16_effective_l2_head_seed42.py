#!/usr/bin/env python3
"""VGG-16 seed-42 Effective L2 + unmatched-head L2 (missing component cell).

The five-row same-map component table is:

    L2-wo
    Effective L2 + unmatched L2          <-- this runner trains this cell
    MNE-L2 detach + unmatched L2         reuse detach+head
    MNE-L2, gradient through λ,γ + unmatched L2   reuse FG-MNE-U
    MNE-L2, gradient through λ,γ (no unmatched L2) reuse component nodetach

Old component ``effective`` is sum_l M_eff,l on IF-matched layers only.
This cell keeps that body term and adds ordinary L2 on unmatched Conv/Linear:

    L = L_task
      + η_MNE sum_{ℓ in S_IF} M_eff,ℓ
      + (η_U / 2) sum_{j in U} ||W_j||_F^2

No 1/λ^2, no L^2, gradients do not enter λ or BN-γ.
η_MNE=1e-4, η_U=5e-4, legacy map. Do not retune from the test curve.
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

from Models import modelpool  # noqa: E402
from run_cifar_mne_unmatched_head_l2_seed42 import (  # noqa: E402
    EVAL_NOISE_SEED,
    L2_WD,
    LVAL,
    MNE_RC,
    classifier_norms,
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
from utils import get_torch_device, unmatched_weight_rows  # noqa: E402

ARCH = "vgg16"
SEED = 42
LAYER_MAP = "legacy"
CONFIG = "vgg16_effective_head"
LABEL = r"Effective L2 + unmatched L2"
SCRATCH = Path("/scratch/gs14/sl9144/snn_results")
LOCAL = ROOT.parent / "important_results"

FIVE_ROW = (
    (
        "l2wo",
        "L2-wo",
        True,
        False,
        False,
        LOCAL / "cifar_vgg16_mne_component_ablation_seed42/{dataset}/comp_l2wo_fixed/scorecard.json",
        SCRATCH / "cifar_vgg16_mne_component_ablation_seed42/{dataset}/comp_l2wo_fixed/scorecard.json",
    ),
    (
        "effective_head",
        "Effective L2 + unmatched L2",
        True,
        False,
        False,
        LOCAL / "cifar_vgg16_effective_l2_head_seed42/{dataset}/vgg16_effective_head/seed42/scorecard.json",
        SCRATCH / "cifar_vgg16_effective_l2_head_seed42/{dataset}/vgg16_effective_head/seed42/scorecard.json",
    ),
    (
        "detach_head",
        r"MNE-L2 detach + unmatched L2",
        True,
        True,
        False,
        LOCAL / "cifar_mne_unmatched_head_l2_seed42/{dataset}/vgg16_mne_head/seed42/scorecard.json",
        SCRATCH / "cifar_mne_unmatched_head_l2_seed42/{dataset}/vgg16_mne_head/seed42/scorecard.json",
    ),
    (
        "fgmneu",
        r"MNE-L2, gradient through $\lambda,\gamma$ + unmatched L2",
        True,
        True,
        True,
        LOCAL / "cifar_mne_nodetach_unmatched_head_l2_seed42/{dataset}/vgg16_mne_head/seed42/scorecard.json",
        SCRATCH / "cifar_mne_nodetach_unmatched_head_l2_seed42/{dataset}/vgg16_mne_head/seed42/scorecard.json",
    ),
    (
        "nodetach_nohead",
        r"MNE-L2, gradient through $\lambda,\gamma$ (no unmatched L2)",
        False,
        True,
        True,
        LOCAL / "cifar_vgg16_mne_component_ablation_seed42/{dataset}/comp_nodetach_fixed/scorecard.json",
        SCRATCH / "cifar_vgg16_mne_component_ablation_seed42/{dataset}/comp_nodetach_fixed/scorecard.json",
    ),
)


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
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar_vgg16_effective_l2_head_seed42",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    return args


def cfg_dir(args) -> Path:
    return args.out_root / args.dataset / CONFIG / f"seed{args.seed}"


def ckpt_path(args) -> Path:
    suffix = f"{CONFIG}_seed{args.seed}_L{LVAL}_trainT0"
    return cfg_dir(args) / "checkpoints" / f"{ARCH}_L[{LVAL}]_{suffix}.pth"


def train_cmd(args) -> list[str]:
    out = cfg_dir(args)
    checkpoint = ckpt_path(args)
    suffix = f"{CONFIG}_seed{args.seed}_L{LVAL}_trainT0"
    return [
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
        "--mne_detach_lambda",
        "--mne_no_lambda",
        "--mne_no_l_scale",
        "--mapping_diag_dir", str(out / "mapping_init"),
        "--epoch_log_csv", str(out / "epoch_log.csv"),
    ]


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
        seed=SEED,
        epochs=EPOCHS,
        batch_size=128,
        workers=8,
        device="cpu",
        out_root=ROOT.parent / "important_results" / "cifar_vgg16_effective_l2_head_seed42",
    )
    cmd = train_cmd(ns)
    joined = " ".join(cmd)
    if cmd[cmd.index("--regularizer") + 1] != "mne_l2_unmatched":
        raise AssertionError("must use unmatched-weight fallback, not mne_l2_all")
    if "--mne_no_lambda" not in cmd or "--mne_no_l_scale" not in cmd:
        raise AssertionError(f"Effective L2 body flags missing: {joined}")
    if "--mne_detach_lambda" not in cmd:
        raise AssertionError("Effective L2 must detach λ")
    if "--mne_no_detach_bn_affine" in cmd:
        raise AssertionError("Effective L2 must keep BN-γ detached")
    if cmd[cmd.index("--unmatched_l2_coeff") + 1] != str(L2_WD):
        raise AssertionError("η_U must stay locked to L2-wo WD")
    if cmd[cmd.index("--mne_layer_map") + 1] != LAYER_MAP:
        raise AssertionError("VGG must use legacy map")
    print("[self-check] Effective L2 + unmatched L2 flags ok", flush=True)


def _read_card(dataset: str, local: Path, scratch: Path) -> dict | None:
    for path in (Path(str(local).format(dataset=dataset)), Path(str(scratch).format(dataset=dataset))):
        if path.is_file():
            card = json.loads(path.read_text())
            card["_scorecard_path"] = str(path)
            return card
    return None


def summarize(out_root: Path) -> None:
    print(
        f"{'dataset':<9} {'method':<52} {'U':>3} {'1/λ²':>4} {'∇λγ':>4} "
        f"{'clean':>7} {'s5':>7} {'aucH':>8} {'||W||F':>8}"
    )
    missing = []
    for dataset in ("cifar10", "cifar100"):
        for key, label, unmatched, inv_lambda, grad_lg, local, scratch in FIVE_ROW:
            if key == "effective_head":
                trained = out_root / dataset / CONFIG / f"seed{SEED}" / "scorecard.json"
                card = json.loads(trained.read_text()) if trained.is_file() else _read_card(
                    dataset, local, scratch
                )
            else:
                card = _read_card(dataset, local, scratch)
            if card is None:
                missing.append(f"{dataset}/{key}")
                print(f"{dataset:<9} {label:<52} {'—':>3} {'—':>4} {'—':>4} {'miss':>7}")
                continue
            print(
                f"{dataset:<9} {label:<52} "
                f"{'Y' if unmatched else 'N':>3} "
                f"{'Y' if inv_lambda else 'N':>4} "
                f"{'Y' if grad_lg else 'N':>4} "
                f"{float(card.get('test_clean', float('nan'))):7.2f} "
                f"{float(card.get('test_sigma5', float('nan'))):7.2f} "
                f"{float(card.get('test_auc_high', float('nan'))):8.2f} "
                f"{float(card.get('classifier_frobenius', float('nan'))):8.3f}"
            )
    if missing:
        print("missing:", ", ".join(missing))


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
        f"[INFO] {LABEL} {args.dataset} seed={args.seed} "
        f"η_MNE={MNE_RC} η_U={L2_WD} map={LAYER_MAP} "
        f"no 1/λ^2, no L^2, detach λ and BN-γ",
        flush=True,
    )
    checkpoint = train(args)
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = load_model(checkpoint, device, ARCH, args.dataset)
    scope = unmatched_scope_card(model, LAYER_MAP)
    (out / "unmatched_scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    dump_mne_mapping_report(model, out / "mapping_eval", layer_map=LAYER_MAP, quant_level=LVAL)
    norms = classifier_norms(model)
    val_rows = sweep(model, val_loader(args, pin), device, "val", args.eval_seed)
    write_csv(out / "val_sweep.csv", val_rows)
    test_rows = sweep(model, test_loader(args, pin), device, "test", args.eval_seed)
    write_csv(out / "test_sweep.csv", test_rows)
    card = {
        "config": CONFIG,
        "method": "effective_head",
        "label": LABEL,
        "paper_header": r"Effective L2 + unmatched L2",
        "unmatched_l2": True,
        "divide_by_lambda": False,
        "scale_by_l": False,
        "gradient_through_lambda_gamma": False,
        "dataset": args.dataset,
        "arch": ARCH,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "regularizer": "mne_l2_unmatched",
        "layer_map": LAYER_MAP,
        "eta_mne": MNE_RC,
        "eta_u": L2_WD,
        "detach_lambda": True,
        "detach_bn_affine": True,
        "checkpoint": str(checkpoint),
        "selection_uses_test": False,
        "T": TEST_T,
        "L": LVAL,
        **scope,
        **norms,
    }
    card.update(snn_metrics(val_rows, "val"))
    card.update(snn_metrics(test_rows, "test"))
    (out / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2), flush=True)


if __name__ == "__main__":
    main()
