#!/usr/bin/env python3
"""CIFAR-10 VGG-16 layer diagnostic. Eval only. Do not retrain.

Reuses analyze_vgg_layerwise_noise_theory on the same images and the same
noise draws (noise seed locked to 0). L=T=16, rate_uniform, post_input_if
Gaussian. Methods: SSSR, L1-wo, L2-wo, L2-all.

Per IF layer:
    S = M_eff / Delta^2, Delta = lambda / L
    p_cross = fraction of quantised states that change
    output_perturbation_rms = RMS of the pre-activation change
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Preprocess import datapool  # noqa: E402
from analyze_vgg_layerwise_noise_theory import analyze_one, _write_csv  # noqa: E402
from run_cifar_ann_t0_evalseed0 import resolve_any  # noqa: E402
from utils import get_torch_device, seed_all  # noqa: E402

NOISE_SEED = 0
METHODS = (
    ("SSSR", "lambda"),
    ("L1-wo", "l1wo"),
    ("L2-wo", "l2wo"),
    ("L2-all", "l2all"),
)
SIGMAS = (0.5, 1.0, 5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--arch", default="vgg16")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--sigmas", type=float, nargs="+", default=list(SIGMAS))
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--d-max-samples", type=int, default=2_000_000)
    parser.add_argument("--power-iters", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/scratch/gs14/sl9144/snn_results/cifar_vgg16_pcross_l16_cifar10"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_torch_device(args.device)
    _, loader = datapool(args.dataset, batch_size=args.batch_size, num_workers=args.workers)
    probe_args = argparse.Namespace(
        arch=args.arch,
        dataset=args.dataset,
        L=args.L,
        T=args.T,
        mode="rate_uniform",
        max_batches=args.max_batches,
        d_max_samples=args.d_max_samples,
        power_iters=args.power_iters,
        seed=NOISE_SEED,
        noise_sigma=0.0,
    )
    rows = []
    for train_seed in args.seeds:
        for sigma in args.sigmas:
            for label, key in METHODS:
                ckpt = resolve_any(args.arch, args.dataset, key, train_seed)
                if ckpt is None or not Path(ckpt).is_file():
                    raise FileNotFoundError(f"missing {label} seed {train_seed}: {ckpt}")
                probe_args.noise_sigma = float(sigma)
                seed_all(NOISE_SEED)
                got = analyze_one(label, Path(ckpt), loader, probe_args, device)
                for row in got:
                    row["train_seed"] = train_seed
                    row["noise_seed"] = NOISE_SEED
                    rows.append(row)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    out = args.out_root / args.dataset / f"seed{'-'.join(str(s) for s in args.seeds)}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "layerwise_S_pcross.csv"
    _write_csv(path, rows)
    print(f"[DONE] {path} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
