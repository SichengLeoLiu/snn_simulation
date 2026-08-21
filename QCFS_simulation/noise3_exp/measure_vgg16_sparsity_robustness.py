#!/usr/bin/env python3
"""CIFAR VGG16: clean sparsity/energy vs post-IF σ=1 robustness.

Reuses the five-reg VGG16 checkpoints. Does not retrain.

Clean inference (σ=0) is the only source of firing-rate / SynOps / Horowitz
energy. Robustness is a separate post-IF evaluation at σ=1; noisy firing
rates are stored as diagnostics and must not be used as deployment energy.

Statistical unit is the training seed:
  1. within each seed, reduce the full 10,000-image test set to
     mean / median / p95 over samples;
  2. across seeds 40–44, report mean ± sample standard deviation.

Main-text methods: L2-all vs MNE-L2.
Appendix methods: L2-wo, L1.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Models.layer import IF  # noqa: E402
from Preprocess import datapool  # noqa: E402
from utils import get_torch_device, seed_all  # noqa: E402
from measure_vgg16_horowitz_energy import (  # noqa: E402
    E_AC_PJ,
    E_MAC_PJ,
    METHODS,
    PJ_TO_MJ,
    load_model,
    resolve_checkpoint,
)


MAIN_METHODS = ("l2", "mne_l2")
APPENDIX_METHODS = ("l2_wo", "l1")
SEED_METRICS = (
    "accuracy",
    "firing_density_mean",
    "firing_density_median",
    "firing_density_p95",
    "event_synops_mean",
    "event_synops_median",
    "event_synops_p95",
    "rest_acs_mean",
    "rest_acs_median",
    "rest_acs_p95",
    "energy_mJ_mean",
    "energy_mJ_median",
    "energy_mJ_p95",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100"])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS),
        default=["l2", "l2_wo", "l1", "mne_l2"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--time-steps", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--arch", default="vgg16")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=None,
        help="Single post-IF robustness sigma. Ignored if --noise-sigmas is set.",
    )
    parser.add_argument(
        "--noise-sigmas",
        nargs="+",
        type=float,
        default=None,
        help="Post-IF robustness sigmas. Clean energy always uses sigma=0.",
    )
    parser.add_argument(
        "--noise-position",
        default="post_input_if",
        choices=["post_input_if", "pre_input_if", "pre_first_conv"],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT.parent
        / "important_results"
        / "cifar_vgg16_sparsity_robustness_T4_8_16_5seed",
    )
    return parser.parse_args()


def _split_time_batch(x: torch.Tensor, time_steps: int) -> torch.Tensor:
    if x.shape[0] % time_steps != 0:
        raise ValueError(
            f"leading dim {x.shape[0]} is not divisible by T={time_steps}"
        )
    batch = x.shape[0] // time_steps
    return x.view(time_steps, batch, *x.shape[1:])


def _per_sample_sum(x: torch.Tensor, time_steps: int) -> torch.Tensor:
    xt = _split_time_batch(x, time_steps)
    return xt.reshape(time_steps, xt.shape[1], -1).sum(dim=(0, 2))


def _event_map(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        return x.ne(0).to(dtype=torch.float32) * float(module.out_features)
    kernel_h, kernel_w = module.kernel_size
    active = x.ne(0).to(dtype=torch.float32)
    kernel = torch.ones(
        module.out_channels,
        module.in_channels // module.groups,
        kernel_h,
        kernel_w,
        device=active.device,
        dtype=active.dtype,
    )
    return F.conv2d(
        active,
        kernel,
        bias=None,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
    )


def _reduce_samples(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


class SampleMeter:
    """Accumulate per-sample IF firing, event ACs, and first-layer MACs."""

    def __init__(self, model: nn.Module, time_steps: int):
        self.time_steps = int(time_steps)
        self.first_conv = next(
            module for module in model.modules() if isinstance(module, nn.Conv2d)
        )
        self.if_nonzero: list[torch.Tensor] = []
        self.if_elements: list[torch.Tensor] = []
        self.first_macs: list[torch.Tensor] = []
        self.rest_acs: list[torch.Tensor] = []
        self.event_synops: list[torch.Tensor] = []
        self._batch_if_nonzero: torch.Tensor | None = None
        self._batch_if_elements: torch.Tensor | None = None
        self._batch_first_macs: torch.Tensor | None = None
        self._batch_rest_acs: torch.Tensor | None = None
        self._batch_event_synops: torch.Tensor | None = None
        self.handles = []
        for module in model.modules():
            if isinstance(module, IF):
                self.handles.append(module.register_forward_hook(self._if_hook))
            elif isinstance(module, (nn.Conv2d, nn.Linear)):
                self.handles.append(module.register_forward_hook(self._op_hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def _zeros(self, batch: int, ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros(batch, device=ref.device, dtype=torch.float64)

    def _ensure_batch(self, batch: int, ref: torch.Tensor) -> None:
        if self._batch_if_nonzero is not None:
            return
        self._batch_if_nonzero = self._zeros(batch, ref)
        self._batch_if_elements = self._zeros(batch, ref)
        self._batch_first_macs = self._zeros(batch, ref)
        self._batch_rest_acs = self._zeros(batch, ref)
        self._batch_event_synops = self._zeros(batch, ref)

    def start_batch(self) -> None:
        self._batch_if_nonzero = None
        self._batch_if_elements = None
        self._batch_first_macs = None
        self._batch_rest_acs = None
        self._batch_event_synops = None

    def finish_batch(self) -> None:
        if self._batch_if_nonzero is None:
            return
        self.if_nonzero.append(self._batch_if_nonzero.cpu())
        self.if_elements.append(self._batch_if_elements.cpu())
        self.first_macs.append(self._batch_first_macs.cpu())
        self.rest_acs.append(self._batch_rest_acs.cpu())
        self.event_synops.append(self._batch_event_synops.cpu())
        self.start_batch()

    def _if_hook(self, _module, _inputs, output) -> None:
        if not torch.is_tensor(output):
            return
        xt = _split_time_batch(output.detach(), self.time_steps)
        batch = xt.shape[1]
        self._ensure_batch(batch, xt)
        self._batch_if_nonzero += xt.ne(0).reshape(self.time_steps, batch, -1).sum(
            dim=(0, 2)
        ).to(dtype=torch.float64)
        self._batch_if_elements += float(self.time_steps * xt[0, 0].numel())

    def _op_hook(self, module, inputs, output) -> None:
        if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
            return
        x = inputs[0].detach()
        events = _per_sample_sum(_event_map(module, x), self.time_steps).to(
            dtype=torch.float64
        )
        dense = _per_sample_sum(
            torch.ones_like(output, dtype=torch.float32) * (
                float(module.in_features)
                if isinstance(module, nn.Linear)
                else float(
                    (module.in_channels // module.groups)
                    * module.kernel_size[0]
                    * module.kernel_size[1]
                )
            ),
            self.time_steps,
        ).to(dtype=torch.float64)
        batch = events.shape[0]
        self._ensure_batch(batch, events)
        self._batch_event_synops += events
        if module is self.first_conv:
            self._batch_first_macs += dense
        else:
            self._batch_rest_acs += events

    def arrays(self) -> dict[str, np.ndarray]:
        if_nonzero = torch.cat(self.if_nonzero).numpy()
        if_elements = torch.cat(self.if_elements).numpy()
        first_macs = torch.cat(self.first_macs).numpy()
        rest_acs = torch.cat(self.rest_acs).numpy()
        event_synops = torch.cat(self.event_synops).numpy()
        firing = if_nonzero / np.maximum(if_elements, 1.0)
        energy = (first_macs * E_MAC_PJ + rest_acs * E_AC_PJ) * PJ_TO_MJ
        return {
            "firing_density": firing,
            "event_synops": event_synops,
            "rest_acs": rest_acs,
            "first_macs": first_macs,
            "energy_mJ": energy,
        }


@torch.inference_mode()
def evaluate(model, loader, device, time_steps: int, sigma: float, position: str) -> dict:
    model.set_T(time_steps)
    model.set_mode("rate_uniform")
    model.set_spike_schedule("normal")
    model.set_first_layer_input_noise_position(position)
    model.set_first_layer_input_noise_sigma(float(sigma))
    meter = SampleMeter(model, time_steps)
    correct = []
    try:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            meter.start_batch()
            logits = model(images)
            if time_steps > 0:
                logits = logits.mean(0)
            pred = logits.argmax(dim=1)
            meter.finish_batch()
            correct.append(pred.eq(labels).to(dtype=torch.float64).cpu())
    finally:
        meter.close()

    correct_np = torch.cat(correct).numpy()
    stats = meter.arrays()
    n = int(correct_np.size)
    if n != 10000:
        raise RuntimeError(f"expected 10000 test images, got {n}")
    fire = _reduce_samples(stats["firing_density"])
    synops = _reduce_samples(stats["event_synops"])
    rest = _reduce_samples(stats["rest_acs"])
    energy = _reduce_samples(stats["energy_mJ"])
    return {
        "n_samples": n,
        "accuracy": float(100.0 * correct_np.mean()),
        "firing_density_mean": fire["mean"],
        "firing_density_median": fire["median"],
        "firing_density_p95": fire["p95"],
        "event_synops_mean": synops["mean"],
        "event_synops_median": synops["median"],
        "event_synops_p95": synops["p95"],
        "rest_acs_mean": rest["mean"],
        "rest_acs_median": rest["median"],
        "rest_acs_p95": rest["p95"],
        "first_macs_per_sample": float(stats["first_macs"].mean()),
        "energy_mJ_mean": energy["mean"],
        "energy_mJ_median": energy["median"],
        "energy_mJ_p95": energy["p95"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    method_order = {name: i for i, name in enumerate(METHODS)}
    for row in rows:
        key = (row["dataset"], row["method"], int(row["T"]), float(row["sigma"]), row["condition"])
        grouped[key].append(row)
    output = []
    for (dataset, method, time_steps, sigma, condition), values in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            method_order.get(item[0][1], 99),
            item[0][2],
            item[0][3],
        ),
    ):
        item = {
            "dataset": dataset,
            "arch": values[0]["arch"],
            "method": method,
            "method_label": values[0]["method_label"],
            "T": time_steps,
            "sigma": f"{sigma:.6g}",
            "condition": condition,
            "use_for_energy": values[0]["use_for_energy"],
            "n_seeds": len(values),
            "n_samples": values[0]["n_samples"],
            "noise_position": values[0]["noise_position"],
        }
        for metric in SEED_METRICS:
            samples = [float(row[metric]) for row in values]
            item[f"{metric}_mean"] = statistics.fmean(samples)
            item[f"{metric}_std"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
        output.append(item)
    return output


def _fmt(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def print_tables(rows: list[dict]) -> None:
    by_key = {
        (
            row["dataset"],
            row["method"],
            int(row["T"]),
            row["condition"],
            f"{float(row['sigma']):.6g}",
        ): row
        for row in rows
    }

    def grab(dataset, method, time_steps, condition, sigma="0"):
        return by_key[(dataset, method, time_steps, condition, f"{float(sigma):.6g}")]

    print("\n=== Clean sparsity / energy (σ=0 only; unit = training seed) ===")
    print(
        f"{'Dataset':<10} {'Method':<8} {'T':>3} "
        f"{'Acc':>14} {'Fire mean':>16} {'Fire med':>16} {'Fire p95':>16} "
        f"{'Energy mJ':>16}"
    )
    for dataset in ("cifar10", "cifar100"):
        for method in ("l2", "mne_l2", "l2_wo", "l1"):
            for time_steps in (4, 8, 16):
                try:
                    row = grab(dataset, method, time_steps, "clean", 0)
                except KeyError:
                    continue
                print(
                    f"{dataset:<10} {row['method_label']:<8} {time_steps:>3} "
                    f"{_fmt(row['accuracy_mean'], row['accuracy_std'], 2):>14} "
                    f"{_fmt(row['firing_density_mean_mean'], row['firing_density_mean_std'], 4):>16} "
                    f"{_fmt(row['firing_density_median_mean'], row['firing_density_median_std'], 4):>16} "
                    f"{_fmt(row['firing_density_p95_mean'], row['firing_density_p95_std'], 4):>16} "
                    f"{_fmt(row['energy_mJ_mean_mean'], row['energy_mJ_mean_std'], 4):>16}"
                )

    noise_sigmas = sorted(
        {
            float(row["sigma"])
            for row in rows
            if row["condition"] == "noise"
        }
    )
    print(
        "\n=== Robustness post-IF (accuracy only; do not use noisy firing as energy) ==="
    )
    header = f"{'Dataset':<10} {'Method':<8} {'T':>3}" + "".join(
        f"{'Acc σ=' + str(int(s) if s == int(s) else s):>14}" for s in noise_sigmas
    )
    print(header)
    for dataset in ("cifar10", "cifar100"):
        for method in ("l2", "mne_l2", "l2_wo", "l1"):
            for time_steps in (4, 8, 16):
                cells = []
                missing = False
                for sigma in noise_sigmas:
                    try:
                        row = grab(dataset, method, time_steps, "noise", sigma)
                    except KeyError:
                        missing = True
                        break
                    cells.append(_fmt(row["accuracy_mean"], row["accuracy_std"], 2))
                if missing:
                    continue
                label = grab(dataset, method, time_steps, "noise", noise_sigmas[0])[
                    "method_label"
                ]
                print(f"{dataset:<10} {label:<8} {time_steps:>3}" + "".join(f"{c:>14}" for c in cells))


def main() -> None:
    args = parse_args()
    if not args.out_dir.is_absolute():
        args.out_dir = (ROOT / args.out_dir).resolve()
    device = get_torch_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.noise_sigmas:
        noise_sigmas = [float(s) for s in args.noise_sigmas]
    elif args.noise_sigma is not None:
        noise_sigmas = [float(args.noise_sigma)]
    else:
        noise_sigmas = [1.0]
    eval_plan = [(0.0, "clean", "yes")] + [
        (sigma, "noise", "no") for sigma in noise_sigmas
    ]
    raw_rows = []
    print(
        "[INFO] clean σ=0 → sparsity/energy; "
        f"σ={noise_sigmas} {args.noise_position} → robustness only",
        flush=True,
    )
    print(
        "[INFO] within seed: sample mean/median/p95; across seeds: mean±sample std",
        flush=True,
    )

    for dataset in args.datasets:
        _, loader = datapool(
            dataset,
            args.batch_size,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
        )
        for method in args.methods:
            label = METHODS[method]["label"]
            for seed in args.seeds:
                ckpt = resolve_checkpoint(dataset, method, seed, args)
                model = load_model(dataset, ckpt, device, args)
                model.set_L(args.L)
                for time_steps in args.time_steps:
                    for sigma, condition, use_energy in eval_plan:
                        seed_all(seed)
                        result = evaluate(
                            model,
                            loader,
                            device,
                            time_steps,
                            sigma,
                            args.noise_position,
                        )
                        row = {
                            "dataset": dataset,
                            "arch": args.arch,
                            "method": method,
                            "method_label": label,
                            "seed": seed,
                            "L": args.L,
                            "T": time_steps,
                            "sigma": f"{sigma:.6g}",
                            "condition": condition,
                            "use_for_energy": use_energy,
                            "noise_position": args.noise_position,
                            **result,
                            "checkpoint": str(ckpt),
                        }
                        raw_rows.append(row)
                        print(
                            f"{dataset:10s} {label:7s} seed={seed} T={time_steps:2d} "
                            f"{condition:5s} σ={sigma:g} acc={result['accuracy']:.2f} "
                            f"fire={result['firing_density_mean']:.4f}/"
                            f"{result['firing_density_median']:.4f}/"
                            f"{result['firing_density_p95']:.4f} "
                            f"E={result['energy_mJ_mean']:.4f} mJ"
                            + (
                                ""
                                if use_energy == "yes"
                                else " [diag only]"
                            ),
                            flush=True,
                        )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    summary = aggregate(raw_rows)
    write_csv(args.out_dir / "sparsity_robustness_seed.csv", raw_rows)
    write_csv(args.out_dir / "sparsity_robustness_summary.csv", summary)
    print_tables(summary)
    print(f"Wrote {args.out_dir / 'sparsity_robustness_seed.csv'}")
    print(f"Wrote {args.out_dir / 'sparsity_robustness_summary.csv'}")
    print("Main-text methods: L2-all, MNE-L2. Appendix methods: L2-wo, L1.")


if __name__ == "__main__":
    main()
