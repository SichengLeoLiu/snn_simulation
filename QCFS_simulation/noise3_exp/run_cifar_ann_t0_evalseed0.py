#!/usr/bin/env python3
"""T=0 ANN re-eval of already-trained CIFAR checkpoints.

Same weights as the T=16 conversion figure, but the forward is QCFS ANN
(T=0, L=16). Jobs only test. Do not retrain. Do not write into the T=16
unified tree, λ-only trees, FG-MNE-U, ImageNet, or historical five-regs.

Locked eval: EVAL_SEED=0, post_input_if Gaussian, σ ∈ {0,1,2,3,5}.
σ=0 is the ANN number for tables. The σ sweep is the ANN robustness curve.

Writes /scratch/.../cifar_ann_t0_evalseed0. Gadi default is 4 GPUs:
one job per (dataset, arch), looping all four methods and seeds 40-44.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_cifar_unified_baseline_evalseed0 import (  # noqa: E402
    ARCHS,
    DATASETS,
    DEFAULT_SEEDS,
    DEFAULT_SIGMAS,
    EVAL_SEED,
    LVAL,
    LOCAL_IR,
    SCRATCH,
    get_torch_device,
    resolve_ckpt,
    snn_metrics,
    sweep,
    test_loader,
    val_loader,
    write_csv,
)

TEST_T = 0
METHODS = ("l2all", "l2wo", "l1wo", "lambda")
LABELS = {
    "l2all": "L2-all",
    "l2wo": "L2-wo",
    "l1wo": "L1-wo",
    "lambda": "nabla-lambda only",
}
BLOCKED_TREES = (
    "cifar_unified_baseline_evalseed0",
    "cifar_vgg16_five_regs",
    "cifar_resnet18_four_regs_5seed",
    "cifar_resnet18_l1wo_5seed",
    "cifar_fgmneu_noise_family",
    "cifar_independent_ckpt_seed42",
    "cifar_fgmneu_grad_ablation_seed42",
    "cifar_resnet18_fgmneu_lambda_seed42",
)
LAMBDA_TREES = {
    "vgg16": "cifar_fgmneu_grad_ablation_seed42",
    "resnet18": "cifar_resnet18_fgmneu_lambda_seed42",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=ARCHS, default=None)
    parser.add_argument("--dataset", choices=DATASETS, default=None)
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument(
        "--all",
        action="store_true",
        help="One process: both datasets, both archs, all four methods. Prefer 4 GPU panels.",
    )
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
        default=ROOT.parent / "important_results" / "cifar_ann_t0_evalseed0",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    args.sigmas = sorted({float(s) for s in args.sigmas})
    if args.self_check or args.summarize:
        return args
    if args.seed is not None:
        args.seeds = [int(args.seed)]
    else:
        args.seeds = [int(s) for s in args.seeds]
    if args.all:
        if args.arch is not None or args.dataset is not None or args.method is not None:
            parser.error("--all cannot be combined with --arch/--dataset/--method")
        return args
    if args.arch is None or args.dataset is None:
        parser.error("--arch and --dataset are required unless --all/--self-check/--summarize")
    return args


def lambda_ckpt_candidates(arch: str, dataset: str, seed: int) -> list[Path]:
    tree = LAMBDA_TREES[arch]
    folder = f"{arch}_fgmneu_lambda"
    name = f"{arch}_L[{LVAL}]_{folder}_seed{seed}_L{LVAL}_trainT0.pth"
    rel = Path(dataset) / folder / f"seed{seed}" / "checkpoints" / name
    return [
        SCRATCH / tree / rel,
        LOCAL_IR / tree / rel,
        Path("/scratch/gs14/sl9144/snn_results") / tree / rel,
    ]


def resolve_lambda_ckpt(arch: str, dataset: str, seed: int) -> Path:
    tried = lambda_ckpt_candidates(arch, dataset, seed)
    for path in tried:
        if path.is_file():
            return path
    lines = "\n".join(f"  {p}" for p in tried)
    raise FileNotFoundError(f"lambda checkpoint not found for {arch} {dataset} seed{seed}:\n{lines}")


def resolve_any(arch: str, dataset: str, method: str, seed: int) -> Path:
    if method == "lambda":
        return resolve_lambda_ckpt(arch, dataset, seed)
    return resolve_ckpt(arch, dataset, method, seed)


def cfg_dir(args, seed: int) -> Path:
    return args.out_root / args.dataset / f"{args.arch}_{args.method}" / f"seed{seed}"


def load_ann(ckpt: Path, device, arch: str, dataset: str):
    from Models import modelpool
    from Models.VGG import remap_legacy_vgg_state_dict

    model = modelpool(arch, dataset)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if arch.startswith("vgg"):
        state = remap_legacy_vgg_state_dict(state)
    model.load_state_dict(state, strict=True)
    model.set_L(LVAL)
    model.set_T(TEST_T)
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule("normal")
    model.set_first_layer_input_noise_position("post_input_if")
    model.set_first_layer_input_noise_type("gaussian")
    model.set_first_layer_input_noise_sigma(0.0)
    return model.to(device).eval()


def self_check() -> None:
    if TEST_T != 0:
        raise AssertionError("this runner must evaluate T=0 ANN")
    out = str(Path("/scratch/gs14/sl9144/snn_results/cifar_ann_t0_evalseed0"))
    if "cifar_ann_t0_evalseed0" not in out:
        raise AssertionError("new T=0 tree required")
    if any(name == "cifar_ann_t0_evalseed0" for name in BLOCKED_TREES):
        raise AssertionError("T=0 tree must not be in the blocked list")
    vgg_l2 = str(resolve_ckpt.__name__)
    if vgg_l2 != "resolve_ckpt":
        raise AssertionError("L2/L1 must reuse the unified ckpt resolver")
    lam = str(lambda_ckpt_candidates("vgg16", "cifar10", 42)[0])
    if "cifar_fgmneu_grad_ablation_seed42" not in lam:
        raise AssertionError("VGG λ-only must reuse the existing lambda tree")
    if "vgg16_fgmneu_lambda" not in lam:
        raise AssertionError("VGG λ-only filename mismatch")
    r18 = str(lambda_ckpt_candidates("resnet18", "cifar100", 40)[0])
    if "cifar_resnet18_fgmneu_lambda_seed42" not in r18:
        raise AssertionError("ResNet λ-only must reuse the existing lambda tree")
    if TEST_T == 16:
        raise AssertionError("must not silently fall back to SNN T=16")
    n_panels = len(DATASETS) * len(ARCHS)
    if n_panels != 4:
        raise AssertionError(f"expected 4 GPU panels, got {n_panels}")
    if len(METHODS) != 4:
        raise AssertionError("each panel must loop four methods")
    print("[self-check] ANN T=0 eval-only flags ok", flush=True)


def summarize(out_root: Path) -> None:
    cards = [
        json.loads(path.read_text())
        for path in sorted(out_root.glob("*/*/seed*/scorecard.json"))
    ]
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    print(
        f"{'dataset':<10} {'arch':<9} {'method':<9} {'seed':>5} "
        f"{'ann0':>7} {'s1':>7} {'s5':>7}"
    )
    for card in cards:
        print(
            f"{card['dataset']:<10} {card.get('arch', '?'):<9} "
            f"{card.get('method', '?'):<9} {card.get('seed', '?'):>5} "
            f"{card['test_clean']:7.2f} {card.get('test_sigma1', float('nan')):7.2f} "
            f"{card['test_sigma5']:7.2f}"
        )


def eval_one(args, seed: int) -> None:
    out = cfg_dir(args, seed)
    if any(name in str(out) for name in BLOCKED_TREES):
        raise RuntimeError(f"refusing to write into blocked tree: {out}")
    out.mkdir(parents=True, exist_ok=True)
    card_path = out / "scorecard.json"
    if card_path.is_file() and not args.force:
        print(f"[SKIP EVAL] {card_path}", flush=True)
        return
    try:
        checkpoint = resolve_any(args.arch, args.dataset, args.method, seed)
    except FileNotFoundError as exc:
        if args.dry_run or args.dry_resolve:
            print(f"[MISSING] {exc}", flush=True)
            return
        raise
    print(f"[REUSE T=0] {checkpoint}", flush=True)
    if args.dry_run or args.dry_resolve:
        print("[DRY]", checkpoint, flush=True)
        return
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = load_ann(checkpoint, device, args.arch, args.dataset)
    if int(getattr(model, "T", -1)) != 0:
        raise RuntimeError(f"model.T is {getattr(model, 'T', None)}, expected 0")
    val_rows = sweep(model, val_loader(args, pin), device, "val", args.eval_seed, args.sigmas)
    write_csv(out / "val_sweep.csv", val_rows)
    test_rows = sweep(model, test_loader(args, pin), device, "test", args.eval_seed, args.sigmas)
    write_csv(out / "test_sweep.csv", test_rows)
    card = {
        "config": f"{args.arch}_{args.method}",
        "label": LABELS[args.method],
        "method": args.method,
        "arch": args.arch,
        "dataset": args.dataset,
        "seed": seed,
        "eval_seed": args.eval_seed,
        "reused_checkpoint": True,
        "checkpoint": str(checkpoint),
        "train_noise_sigma": 0.0,
        "protocol": {
            "T": TEST_T,
            "L": LVAL,
            "forward": "ann",
            "mode": "unused_at_T0",
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
    if args.all:
        combos = [
            (dataset, arch, method)
            for dataset in DATASETS
            for arch in ARCHS
            for method in METHODS
        ]
    else:
        methods = METHODS if args.method is None else (args.method,)
        combos = [(args.dataset, args.arch, method) for method in methods]
    print(
        f"[INFO] ANN T=0 combos={len(combos)} seeds={args.seeds} "
        f"eval_seed={args.eval_seed} sigmas={args.sigmas}",
        flush=True,
    )
    for dataset, arch, method in combos:
        args.dataset = dataset
        args.arch = arch
        args.method = method
        print(
            f"[INFO] ANN T=0 {args.arch} {args.method} {args.dataset} "
            f"seeds={args.seeds}",
            flush=True,
        )
        for seed in args.seeds:
            eval_one(args, seed)


if __name__ == "__main__":
    main()
