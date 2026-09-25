#!/usr/bin/env python3
"""Fill missing VGG-16 CIFAR-10 (L, T) noise cells. Eval only.

Existing diagonals stay put (T=L). L=16 T=4/8 already exist.
This runner only evaluates the unchecked cells:

  L=4  seed 42     T in {8,16,32}
  L=8  seed 42     T in {4,16,32}
  L=16 seeds 40-44 T=32
  L=32 seed 42     T in {4,8,16}

Methods: L2-all, L2-wo, L1-wo, TA-MNE-U.
Protocol matches the L=16 T=4/8 jobs: rate_uniform, post-IF Gaussian,
EVAL_SEED=0, sigma in {0,1,2,3,5}. Do not retrain.
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

from run_cifar_ann_t0_evalseed0 import resolve_any  # noqa: E402
from run_cifar_unified_baseline_evalseed0 import (  # noqa: E402
    EVAL_SEED,
    get_torch_device,
    snn_metrics,
    sweep,
    test_loader,
    val_loader,
    write_csv,
)

SCRATCH = Path("/scratch/gs14/sl9144/snn_results")
METHODS = ("l2all", "l2wo", "l1wo", "lambda")
LABELS = {
    "l2all": "L2-all",
    "l2wo": "L2-wo",
    "l1wo": "L1-wo",
    "lambda": "TA-MNE-U",
}
# quant L -> eval timesteps that are still missing
MISSING = {
    4: (8, 16, 32),
    8: (4, 16, 32),
    16: (32,),
    32: (4, 8, 16),
}
SIGMAS = (0.0, 1.0, 2.0, 3.0, 5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cifar10", choices=("cifar10",))
    parser.add_argument("--arch", default="vgg16", choices=("vgg16",))
    parser.add_argument("--quant-L", type=int, choices=tuple(MISSING), required=False)
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--times", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-resolve", action="store_true")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=SCRATCH / "cifar_vgg16_lt_grid_cifar10",
    )
    args = parser.parse_args()
    if args.quant_L is None:
        parser.error("--quant-L is required")
    args.quant_L = int(args.quant_L)
    default_times = list(MISSING[args.quant_L])
    args.times = [int(t) for t in (args.times or default_times)]
    illegal = [t for t in args.times if t == args.quant_L or t not in (4, 8, 16, 32)]
    if illegal:
        parser.error(f"refusing T={illegal}; only missing cells are allowed")
    if args.quant_L == 16 and any(t in (4, 8) for t in args.times):
        parser.error("L=16 T=4/8 already exist")
    if args.seeds is None:
        args.seeds = [40, 41, 42, 43, 44] if args.quant_L == 16 else [42]
    else:
        args.seeds = [int(s) for s in args.seeds]
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    return args


def _first_existing(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.is_file():
            return path
    lines = "\n".join(f"  {path}" for path in candidates)
    raise FileNotFoundError(f"checkpoint missing:\n{lines}")


def resolve_lscale(dataset: str, method: str, quant_l: int, seed: int) -> Path:
    if seed != 42:
        raise FileNotFoundError(f"L={quant_l} only has seed 42")
    if method in ("l2wo", "l2all"):
        folder = f"l2l_{method}_L{quant_l}"
        name = f"vgg16_L[{quant_l}]_{folder}_seed42_L{quant_l}_trainT0.pth"
        relative = Path(dataset) / folder / "checkpoints" / name
        tree = "cifar_vgg16_l2_lscale_seed42"
    elif method == "l1wo":
        folder = f"vgg16_l1wo_L{quant_l}"
        name = f"vgg16_L[{quant_l}]_{folder}_seed42_L{quant_l}_trainT0.pth"
        relative = Path(dataset) / folder / "checkpoints" / name
        tree = "cifar_fgmneu_l1_lscale_seed42"
    elif method == "lambda":
        folder = f"vgg16_lambda_L{quant_l}"
        name = f"vgg16_L[{quant_l}]_{folder}_seed42_L{quant_l}_trainT0.pth"
        relative = Path(dataset) / folder / "seed42" / "checkpoints" / name
        tree = "cifar_fgmneu_lambda_lscale_seed42"
    else:
        raise ValueError(method)
    return _first_existing(
        [
            SCRATCH / tree / relative,
            ROOT.parent / "important_results" / tree / relative,
        ]
    )


def resolve_checkpoint(dataset: str, method: str, quant_l: int, seed: int) -> Path:
    if quant_l == 16:
        return resolve_any("vgg16", dataset, method, seed)
    return resolve_lscale(dataset, method, quant_l, seed)


def cfg_dir(args, method: str, seed: int, test_t: int) -> Path:
    return (
        args.out_root
        / args.dataset
        / f"vgg16_{method}_L{args.quant_L}"
        / f"seed{seed}"
        / f"T{test_t}"
    )


def load_snn(ckpt: Path, device, dataset: str, quant_l: int, test_t: int):
    from Models import modelpool
    from Models.VGG import remap_legacy_vgg_state_dict
    from Models.layer import IF
    import torch

    model = modelpool("vgg16", dataset)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(remap_legacy_vgg_state_dict(state), strict=True)
    model.set_L(quant_l)
    model.set_T(test_t)
    model.set_mode("rate_uniform")
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule("normal")
    model.set_first_layer_input_noise_position("post_input_if")
    model.set_first_layer_input_noise_type("gaussian")
    model.set_first_layer_input_noise_sigma(0.0)
    model = model.to(device).eval()
    if_ls = [int(m.L) for m in model.modules() if isinstance(m, IF)]
    if int(getattr(model, "T", -1)) != test_t or not if_ls or any(L != quant_l for L in if_ls):
        raise RuntimeError(
            f"expected T={test_t} L={quant_l}, got T={getattr(model, 'T', None)} "
            f"IF.L={sorted(set(if_ls)) if if_ls else None}"
        )
    return model


def eval_one(args, method: str, seed: int, test_t: int, checkpoint: Path) -> None:
    out = cfg_dir(args, method, seed, test_t)
    card_path = out / "scorecard.json"
    if card_path.is_file() and not args.force:
        print(f"[SKIP] {card_path}", flush=True)
        return
    print(f"[EVAL] L={args.quant_L} T={test_t} {method} seed{seed}\n       {checkpoint}", flush=True)
    if args.dry_resolve:
        return
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = load_snn(checkpoint, device, args.dataset, args.quant_L, test_t)
    val_rows = sweep(model, val_loader(args, pin), device, "val", args.eval_seed, list(SIGMAS))
    test_rows = sweep(model, test_loader(args, pin), device, "test", args.eval_seed, list(SIGMAS))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "val_sweep.csv", val_rows)
    write_csv(out / "test_sweep.csv", test_rows)
    card = {
        "config": f"vgg16_{method}_L{args.quant_L}_T{test_t}",
        "label": LABELS[method],
        "method": method,
        "arch": "vgg16",
        "dataset": args.dataset,
        "seed": seed,
        "eval_seed": args.eval_seed,
        "reused_checkpoint": True,
        "checkpoint": str(checkpoint),
        "protocol": {
            "T": int(test_t),
            "L": int(args.quant_L),
            "forward": "snn",
            "mode": "rate_uniform",
            "noise": "post_input_if gaussian",
            "eval_seed_locked": True,
        },
        **snn_metrics(val_rows, "val"),
        **snn_metrics(test_rows, "test"),
    }
    card_path.write_text(json.dumps(card, indent=2) + "\n")
    print(
        f"[DONE] {method} L={args.quant_L} T={test_t} seed{seed} "
        f"clean={card['test_clean']:.2f} s5={card['test_sigma5']:.2f}",
        flush=True,
    )
    del model


def main() -> None:
    args = parse_args()
    methods = [args.method] if args.method else list(METHODS)
    args.out_root.mkdir(parents=True, exist_ok=True)
    for method in methods:
        for seed in args.seeds:
            checkpoint = resolve_checkpoint(args.dataset, method, args.quant_L, seed)
            for test_t in args.times:
                eval_one(args, method, seed, test_t, checkpoint)


if __name__ == "__main__":
    main()
