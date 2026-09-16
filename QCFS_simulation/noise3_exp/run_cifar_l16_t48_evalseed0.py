#!/usr/bin/env python3
"""L=16 conversion eval at T=4 and T=8, EVAL_SEED=0.

Reuse already-trained L=16 checkpoints (L2-all / L2-wo / L1-wo / ∇λ-only).
Jobs only test. Do not retrain. Do not write T=0 / T=16 trees, λ-only
training trees, FG-MNE-U, or ImageNet.

T=0 is a separate runner. T=16 already exists. This fills T=4/8 only.

Writes /scratch/.../cifar_l16_t48_evalseed0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_cifar_ann_t0_evalseed0 import (  # noqa: E402
    BLOCKED_TREES,
    LABELS,
    METHODS,
    resolve_any,
)
from run_cifar_unified_baseline_evalseed0 import (  # noqa: E402
    ARCHS,
    DATASETS,
    DEFAULT_SEEDS,
    DEFAULT_SIGMAS,
    EVAL_SEED,
    LVAL,
    get_torch_device,
    snn_metrics,
    sweep,
    test_loader,
    val_loader,
    write_csv,
)

TEST_TS = (4, 8)
OUT_NAME = "cifar_l16_t48_evalseed0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=ARCHS, default=None)
    parser.add_argument("--dataset", choices=DATASETS, default=None)
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--times", nargs="+", type=int, default=list(TEST_TS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sigmas", nargs="+", type=float, default=list(DEFAULT_SIGMAS))
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-resolve", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / OUT_NAME,
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    args.sigmas = sorted({float(s) for s in args.sigmas})
    args.times = [int(t) for t in args.times]
    if args.self_check or args.summarize:
        return args
    if args.seed is not None:
        args.seeds = [int(args.seed)]
    else:
        args.seeds = [int(s) for s in args.seeds]
    if args.arch is None or args.dataset is None:
        parser.error("--arch and --dataset are required unless --self-check/--summarize")
    if any(t in (0, 16) for t in args.times):
        parser.error("this runner is T=4/8 only; T=0 and T=16 have their own trees")
    if any(t <= 0 for t in args.times):
        parser.error("SNN T must be positive")
    return args


def cfg_dir(args, seed: int, method: str, test_t: int) -> Path:
    return args.out_root / args.dataset / f"{args.arch}_{method}" / f"seed{seed}" / f"T{test_t}"


def load_snn(ckpt: Path, device, arch: str, dataset: str, test_t: int):
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
    model.set_L(LVAL)
    model.set_T(int(test_t))
    model.set_mode("rate_uniform")
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule("normal")
    model.set_first_layer_input_noise_position("post_input_if")
    model.set_first_layer_input_noise_type("gaussian")
    model.set_first_layer_input_noise_sigma(0.0)
    return model.to(device).eval()


def self_check() -> None:
    if TEST_TS != (4, 8):
        raise AssertionError("default T grid must be 4 and 8")
    if LVAL != 16:
        raise AssertionError("T=4/8 eval stays on L=16 weights")
    if OUT_NAME in BLOCKED_TREES:
        raise AssertionError("new T=4/8 tree must not be blocked")
    if LVAL != 16:
        raise AssertionError("T=4/8 eval stays on L=16 weights")
    if OUT_NAME in BLOCKED_TREES:
        raise AssertionError("new T=4/8 tree must not be blocked")
    print("[self-check] L=16 T=4/8 eval-only flags ok", flush=True)


def summarize(out_root: Path) -> None:
    cards = [
        json.loads(path.read_text())
        for path in sorted(out_root.glob("*/*/seed*/T*/scorecard.json"))
    ]
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    print(
        f"{'dataset':<10} {'arch':<9} {'method':<9} {'seed':>5} {'T':>3} "
        f"{'clean':>7} {'s1':>7} {'s5':>7}"
    )
    for card in cards:
        print(
            f"{card['dataset']:<10} {card.get('arch', '?'):<9} "
            f"{card.get('method', '?'):<9} {card.get('seed', '?'):>5} "
            f"{card.get('protocol', {}).get('T', '?'):>3} "
            f"{card['test_clean']:7.2f} {card.get('test_sigma1', float('nan')):7.2f} "
            f"{card['test_sigma5']:7.2f}"
        )


def eval_one(args, seed: int, method: str, test_t: int, checkpoint: Path) -> None:
    out = cfg_dir(args, seed, method, test_t)
    extra_blocked = BLOCKED_TREES + ("cifar_ann_t0_evalseed0", "cifar_unified_baseline_evalseed0")
    if any(name in str(out) for name in extra_blocked):
        raise RuntimeError(f"refusing to write into blocked tree: {out}")
    out.mkdir(parents=True, exist_ok=True)
    card_path = out / "scorecard.json"
    if card_path.is_file() and not args.force:
        print(f"[SKIP EVAL] {card_path}", flush=True)
        return
    print(f"[REUSE L=16 T={test_t}] {checkpoint}", flush=True)
    if args.dry_run or args.dry_resolve:
        print("[DRY]", checkpoint, flush=True)
        return
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = load_snn(checkpoint, device, args.arch, args.dataset, test_t)
    if int(getattr(model, "T", -1)) != int(test_t):
        raise RuntimeError(f"model.T is {getattr(model, 'T', None)}, expected {test_t}")
    val_rows = sweep(model, val_loader(args, pin), device, "val", args.eval_seed, args.sigmas)
    write_csv(out / "val_sweep.csv", val_rows)
    test_rows = sweep(model, test_loader(args, pin), device, "test", args.eval_seed, args.sigmas)
    write_csv(out / "test_sweep.csv", test_rows)
    card = {
        "config": f"{args.arch}_{method}_T{test_t}",
        "label": LABELS[method],
        "method": method,
        "arch": args.arch,
        "dataset": args.dataset,
        "seed": seed,
        "eval_seed": args.eval_seed,
        "reused_checkpoint": True,
        "checkpoint": str(checkpoint),
        "protocol": {
            "T": int(test_t),
            "L": LVAL,
            "forward": "snn",
            "mode": "rate_uniform",
            "noise": "post_input_if gaussian",
            "eval_seed_locked": True,
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
    methods = METHODS if args.method is None else (args.method,)
    print(
        f"[INFO] L=16 T={args.times} {args.arch} {args.dataset} "
        f"methods={list(methods)} seeds={args.seeds} eval_seed={args.eval_seed}",
        flush=True,
    )
    for method in methods:
        args.method = method
        for seed in args.seeds:
            try:
                checkpoint = resolve_any(args.arch, args.dataset, method, seed)
            except FileNotFoundError as exc:
                if args.dry_run or args.dry_resolve:
                    print(f"[MISSING] {exc}", flush=True)
                    continue
                raise
            for test_t in args.times:
                eval_one(args, seed, method, test_t, checkpoint)


if __name__ == "__main__":
    main()
