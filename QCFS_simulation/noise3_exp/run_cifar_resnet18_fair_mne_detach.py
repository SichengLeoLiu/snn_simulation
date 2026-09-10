#!/usr/bin/env python3
"""Strictly fair ResNet-18 MNE-L2 detach baseline (P0).

Historical four-regs / mapping-diag MNE checkpoints are not controls:
they used the VGG rc=1e-4 and, unless explicitly remapped, the legacy
Conv-IF layer map. This runner retrains Old MNE-L2 detach with:

  * --mne_layer_map resnet
  * the same numerical regularizer budget grid as L2-wo / Task-Cov (around β=5e-4)
  * 5k train-holdout val for checkpointing and rc selection
  * a shared test noise stream (eval seed 0)

Do not pick rc from the test curve. After the seed-42 grid finishes:

  python noise3_exp/run_cifar_resnet18_fair_mne_detach.py --select \\
    --out-root important_results/cifar_resnet18_fair_mne_detach

Locked rc is 1e-4. The 5-seed follow-up trains seeds 40–44 at that rc
with eval seed 0. Extra noise streams go under eval_seed{N}/ and must not
replace the canonical scorecard.
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
from utils import dump_mne_mapping_report, get_torch_device  # noqa: E402

ARCH = "resnet18"
SEED = 42
SEEDS = (40, 41, 42, 43, 44)
EVAL_NOISE_SEED = 0
LAYER_MAP = "resnet"
L2_WD = 5e-4
CLEAN_TOLERANCE = 0.5
LOCKED_RC = "1e-4"
RC_CHOICES = ("1e-4", "3e-4", "5e-4", "1e-3")


def parse_seed_list(raw: str) -> list[int]:
    seeds = []
    for part in (raw or "").replace(":", ",").replace(" ", ",").split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar100")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval-seed", type=int, default=int(os.environ.get("EVAL_SEED", str(EVAL_NOISE_SEED))))
    parser.add_argument(
        "--extra-eval-seeds",
        default=os.environ.get("EXTRA_EVAL_SEEDS", ""),
        help="Comma-separated extra noise seeds written under eval_seed{N}/; typical 1,2,3,4",
    )
    parser.add_argument("--reg-coeff", default=os.environ.get("REG_COEFF", LOCKED_RC), choices=RC_CHOICES)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar_resnet18_fair_mne_detach",
    )
    args = parser.parse_args()
    args.reg_coeff = str(args.reg_coeff)
    args.eval_seed = int(args.eval_seed)
    args.extra_eval_seeds = parse_seed_list(args.extra_eval_seeds)
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    return args


def config_name(reg_coeff: str) -> str:
    return f"r18_mne_resnet_rc{reg_coeff}"


def suffix(args) -> str:
    return f"{config_name(args.reg_coeff)}_seed{args.seed}_L{LVAL}_trainT0"


def cfg_dir(args) -> Path:
    return args.out_root / args.dataset / config_name(args.reg_coeff) / f"seed{args.seed}"


def ckpt_path(args) -> Path:
    return cfg_dir(args) / "checkpoints" / f"{ARCH}_L[{LVAL}]_{suffix(args)}.pth"


def load_model(ckpt: Path, device, dataset: str):
    model = modelpool(ARCH, dataset)
    model._mne_layer_map = LAYER_MAP
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.set_L(LVAL)
    model.set_T(TEST_T)
    model.set_mode("rate_uniform")
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule("normal")
    model.set_first_layer_input_noise_position("post_input_if")
    model.set_first_layer_input_noise_type("gaussian")
    return model.to(device).eval()


def train(args) -> Path:
    out = cfg_dir(args)
    checkpoint = ckpt_path(args)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {checkpoint}", flush=True)
        return checkpoint
    if args.test_only:
        raise FileNotFoundError(checkpoint)
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
        "-suffix", suffix(args),
        "--regularizer", "mne_l2",
        "--weight_decay", "0",
        "--reg_coeff", args.reg_coeff,
        "--mne_detach_lambda",
        "--mne_layer_map", LAYER_MAP,
        "--mapping_diag_dir", str(out / "mapping_init"),
        "--epoch_log_csv", str(out / "epoch_log.csv"),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not checkpoint.exists():
        raise FileNotFoundError(f"training finished but checkpoint is missing: {checkpoint}")
    return checkpoint


def eval_dest(out: Path, eval_seed: int, canonical_seed: int) -> Path:
    if eval_seed == canonical_seed:
        return out
    return out / f"eval_seed{eval_seed}"


def write_eval(model, args, checkpoint: Path, device, eval_seed: int, dest: Path, dump_mapping: bool) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    pin = device.type == "cuda"
    if dump_mapping:
        dump_mne_mapping_report(model, dest / "mapping_eval", layer_map=LAYER_MAP, quant_level=LVAL)
    val_rows = sweep(model, val_loader(args, pin), device, "val", eval_seed)
    write_csv(dest / "val_sweep.csv", val_rows)
    test_rows = sweep(model, test_loader(args, pin), device, "test", eval_seed)
    write_csv(dest / "test_sweep.csv", test_rows)
    saved = args.eval_seed
    args.eval_seed = eval_seed
    card = scorecard(val_rows, test_rows, args, checkpoint)
    args.eval_seed = saved
    (dest / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2), flush=True)
    return card


def scorecard(val_rows, test_rows, args, checkpoint: Path) -> dict:
    card = {
        "config": config_name(args.reg_coeff),
        "label": f"MNE-L2 detach · resnet map · rc={args.reg_coeff}",
        "method": "mne_detach",
        "dataset": args.dataset,
        "arch": ARCH,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "regularizer": "mne_l2",
        "layer_map": LAYER_MAP,
        "detach_lambda": True,
        "reg_coeff": float(args.reg_coeff),
        "reg_coeff_tag": args.reg_coeff,
        "l2_wo_budget": L2_WD,
        "weight_decay": 0.0,
        "historical_checkpoint_reused": False,
        "checkpoint": str(checkpoint),
    }
    card.update(snn_metrics(val_rows, "val"))
    card.update(snn_metrics(test_rows, "test"))
    return card


def select(out_root: Path) -> dict:
    chosen = {}
    rows = []
    for dataset in ("cifar10", "cifar100"):
        cards = []
        for tag in RC_CHOICES:
            path = out_root / dataset / config_name(tag) / f"seed{SEED}" / "scorecard.json"
            if path.is_file():
                cards.append(json.loads(path.read_text()))
        if not cards:
            print(f"{dataset}: no scorecards")
            continue
        best_clean = max(float(card["val_clean"]) for card in cards)
        floor = best_clean - CLEAN_TOLERANCE
        eligible = [card for card in cards if float(card["val_clean"]) + 1e-12 >= floor]
        pick = max(eligible, key=lambda card: (float(card["val_auc_high"]), float(card["val_clean"]), -float(card["reg_coeff"])))
        chosen[dataset] = {
            "reg_coeff": pick["reg_coeff_tag"],
            "val_clean": pick["val_clean"],
            "val_auc_high": pick["val_auc_high"],
            "clean_floor": floor,
            "n_candidates": len(cards),
            "n_eligible": len(eligible),
            "config": pick["config"],
        }
        print(
            f"{dataset}: pick rc={pick['reg_coeff_tag']} "
            f"val_clean={pick['val_clean']:.2f} val_auc_high={pick['val_auc_high']:.2f} "
            f"(floor {floor:.2f}, {len(eligible)}/{len(cards)} eligible)"
        )
        for card in cards:
            rows.append(
                {
                    "dataset": dataset,
                    "reg_coeff": card["reg_coeff_tag"],
                    "selected": card["reg_coeff_tag"] == pick["reg_coeff_tag"],
                    "val_clean": card["val_clean"],
                    "val_auc_high": card["val_auc_high"],
                    "test_clean": card["test_clean"],
                    "test_sigma5": card["test_sigma5"],
                    "test_auc_high": card["test_auc_high"],
                }
            )
    payload = {
        "protocol": "max val_auc_high among rc with val_clean >= best_val_clean - 0.5",
        "selection_uses_test": False,
        "layer_map": LAYER_MAP,
        "eval_seed": EVAL_NOISE_SEED,
        "seed": SEED,
        "selected": chosen,
        "candidates": rows,
    }
    if rows:
        write_csv(out_root / "rc_grid.csv", rows)
    (out_root / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out_root / 'selection.json'}")
    return payload


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, var ** 0.5


def _acc_at_csv(path: Path, sigma: float) -> float | None:
    if not path.is_file():
        return None
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if abs(float(row["sigma"]) - sigma) < 1e-12:
                return float(row["accuracy"])
    return None


def summarize(out_root: Path) -> None:
    print(f"{'dataset':<10} {'eval':>4} {'n':>3} {'test0':>16} {'test3':>16} {'test5':>16} {'AUC3-5':>16}")
    rows = []
    for dataset in ("cifar10", "cifar100"):
        grouped: dict[int, list[dict]] = {}
        sweeps: dict[int, list[float]] = {}
        for seed in SEEDS:
            seed_dir = out_root / dataset / config_name(LOCKED_RC) / f"seed{seed}"
            canonical = seed_dir / "scorecard.json"
            extras = sorted(seed_dir.glob("eval_seed*/scorecard.json"))
            paths = ([canonical] if canonical.is_file() else []) + extras
            if not paths:
                print(f"{dataset:<10} seed{seed} MISSING")
            for path in paths:
                card = json.loads(path.read_text())
                eval_seed = int(card["eval_seed"])
                grouped.setdefault(eval_seed, []).append(card)
                sigma3 = _acc_at_csv(path.parent / "test_sweep.csv", 3.0)
                if sigma3 is not None:
                    sweeps.setdefault(eval_seed, []).append(sigma3)
        for eval_seed in sorted(grouped):
            group = grouped[eval_seed]
            t0m, t0s = _mean_std([float(card["test_clean"]) for card in group])
            t5m, t5s = _mean_std([float(card["test_sigma5"]) for card in group])
            hm, hs = _mean_std([float(card["test_auc_high"]) for card in group])
            s3 = sweeps.get(eval_seed, [])
            t3m, t3s = _mean_std(s3) if s3 else (float("nan"), float("nan"))
            print(
                f"{dataset:<10} {eval_seed:4d} {len(group):3d} "
                f"{t0m:7.2f}±{t0s:<6.2f} {t3m:7.2f}±{t3s:<6.2f} "
                f"{t5m:7.2f}±{t5s:<6.2f} {hm:7.1f}±{hs:<6.1f}"
            )
            rows.append(
                {
                    "dataset": dataset,
                    "reg_coeff": LOCKED_RC,
                    "eval_seed": eval_seed,
                    "n_seeds": len(group),
                    "test_clean_mean": t0m,
                    "test_clean_std": t0s,
                    "test_sigma3_mean": t3m,
                    "test_sigma3_std": t3s,
                    "test_sigma5_mean": t5m,
                    "test_sigma5_std": t5s,
                    "test_auc_high_mean": hm,
                    "test_auc_high_std": hs,
                }
            )
    if rows:
        write_csv(out_root / "fair_mne_5seed_summary.csv", rows)
        print(f"Wrote {out_root / 'fair_mne_5seed_summary.csv'}")


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.select:
        select(args.out_root)
        return
    if args.summarize:
        summarize(args.out_root)
        return
    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] {args.dataset} ResNet-18 fair MNE-detach rc={args.reg_coeff} "
        f"layer_map={LAYER_MAP} seed={args.seed} eval_seed={args.eval_seed} "
        f"extra_eval_seeds={args.extra_eval_seeds}",
        flush=True,
    )
    checkpoint = train(args)
    device = get_torch_device(args.device)
    model = load_model(checkpoint, device, args.dataset)
    canonical = out / "scorecard.json"
    if canonical.is_file() and not args.retrain:
        print(f"[SKIP CANONICAL EVAL] {canonical}", flush=True)
    else:
        write_eval(
            model,
            args,
            checkpoint,
            device,
            args.eval_seed,
            eval_dest(out, args.eval_seed, args.eval_seed),
            dump_mapping=not (out / "mapping_eval").exists(),
        )
    for extra in args.extra_eval_seeds:
        if extra == args.eval_seed:
            continue
        dest = eval_dest(out, extra, args.eval_seed)
        extra_card = dest / "scorecard.json"
        if extra_card.is_file() and not args.retrain:
            print(f"[SKIP EXTRA EVAL] {extra_card}", flush=True)
            continue
        print(f"[EXTRA EVAL] eval_seed={extra} -> {dest}", flush=True)
        write_eval(model, args, checkpoint, device, extra, dest, dump_mapping=False)


if __name__ == "__main__":
    main()
