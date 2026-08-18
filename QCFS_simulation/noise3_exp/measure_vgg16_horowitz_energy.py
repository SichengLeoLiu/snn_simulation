#!/usr/bin/env python3
"""Horowitz 45nm energy table for CIFAR VGG16.

Follows Rathi & Roy 2023 / Yao et al. 2023 / TPP:

  E_MAC = 4.6 pJ,  E_AC = 0.9 pJ  (32-bit, 45nm CMOS, Horowitz 2014)

  ANN (T=0 QCFS):  E = N_MAC * E_MAC
  SNN:  first conv sees analog pixels, so T * N_MAC_conv1 * E_MAC;
        later conv/linear use event ACs (nonzero presynaptic events
        times fan-out) * E_AC.

This is an algorithmic estimate, not chip energy. Reuses the five-reg
CIFAR VGG16 checkpoints (test-only, no retraining).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Models import modelpool  # noqa: E402
from Models.layer import IF  # noqa: E402
from Models.VGG import remap_legacy_vgg_state_dict  # noqa: E402
from Preprocess import datapool  # noqa: E402
from utils import get_torch_device, seed_all  # noqa: E402
import run_mne_stability_ablation as ablation  # noqa: E402


E_MAC_PJ = 4.6
E_AC_PJ = 0.9
PJ_TO_MJ = 1e-9

METHODS = {
    "l2": {
        "variant": "weight_decay",
        "rc": None,
        "label": "L2",
    },
    "l2_wo": {
        "variant": "weight_decay_weights_only",
        "rc": None,
        "label": "L2-wo",
    },
    "l1": {
        "variant": "l1",
        "rc": 1e-5,
        "label": "L1",
    },
    "mne_l2": {
        "variant": "old_detach",
        "rc": 1e-4,
        "label": "MNE-L2",
    },
}


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
        "--max-samples",
        type=int,
        default=0,
        help="0 uses the full test set.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar_vgg16_three_regs_horowitz_energy",
    )
    return parser.parse_args()


def resolve_checkpoint(dataset: str, method: str, seed: int, args) -> Path:
    spec = METHODS[method]
    ckpt_args = SimpleNamespace(arch=args.arch, L=args.L, train_T=0)
    ckpt = ablation._resolve_checkpoint(
        dataset, spec["variant"], spec["rc"], seed, None, ckpt_args
    )
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    return ckpt


def _dense_ops(module: nn.Module, output: torch.Tensor) -> float:
    if isinstance(module, nn.Linear):
        return float(output.numel() * module.in_features)
    kernel_h, kernel_w = module.kernel_size
    fan_in = (module.in_channels // module.groups) * kernel_h * kernel_w
    return float(output.numel() * fan_in)


def _event_synops(module: nn.Module, x: torch.Tensor) -> float:
    if isinstance(module, nn.Linear):
        return float(torch.count_nonzero(x).item() * module.out_features)
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
    synops = F.conv2d(
        active,
        kernel,
        bias=None,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
    )
    return float(synops.sum().item())


class EnergyMeter:
    """Count ANN MACs, first-layer SNN MACs, and remaining event ACs."""

    def __init__(self, model: nn.Module):
        self.first_conv = next(
            module for module in model.modules() if isinstance(module, nn.Conv2d)
        )
        self.if_nonzero = 0
        self.if_elements = 0
        self.first_dense_ops = 0.0
        self.rest_dense_ops = 0.0
        self.rest_event_synops = 0.0
        self.all_event_synops = 0.0
        self.handles = []
        for module in model.modules():
            if isinstance(module, IF):
                self.handles.append(module.register_forward_hook(self._if_hook))
            elif isinstance(module, (nn.Conv2d, nn.Linear)):
                self.handles.append(module.register_forward_hook(self._op_hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def _if_hook(self, _module, _inputs, output) -> None:
        if not torch.is_tensor(output):
            return
        detached = output.detach()
        self.if_nonzero += int(torch.count_nonzero(detached).item())
        self.if_elements += detached.numel()

    def _op_hook(self, module, inputs, output) -> None:
        if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
            return
        x = inputs[0].detach()
        dense = _dense_ops(module, output)
        events = _event_synops(module, x)
        self.all_event_synops += events
        if module is self.first_conv:
            self.first_dense_ops += dense
        else:
            self.rest_dense_ops += dense
            self.rest_event_synops += events


def evaluate(model, loader, device, time_steps: int, max_samples: int) -> dict:
    meter = EnergyMeter(model)
    correct = 0
    samples = 0
    try:
        with torch.inference_mode():
            for images, labels in loader:
                if max_samples > 0 and samples >= max_samples:
                    break
                if max_samples > 0 and samples + images.shape[0] > max_samples:
                    keep = max_samples - samples
                    images = images[:keep]
                    labels = labels[:keep]
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                if time_steps > 0:
                    logits = logits.mean(0)
                correct += int(logits.argmax(dim=1).eq(labels).sum().item())
                samples += labels.numel()
    finally:
        meter.close()

    t = float(max(time_steps, 1))
    mode = "ANN" if time_steps <= 0 else "SNN"
    ann_macs = (meter.first_dense_ops + meter.rest_dense_ops) / (t * samples)
    first_macs = meter.first_dense_ops / samples
    rest_acs = meter.rest_event_synops / samples
    event_synops = meter.all_event_synops / samples
    ann_energy_mj = ann_macs * E_MAC_PJ * PJ_TO_MJ
    snn_energy_mj = (first_macs * E_MAC_PJ + rest_acs * E_AC_PJ) * PJ_TO_MJ
    event_only_mj = event_synops * E_AC_PJ * PJ_TO_MJ
    energy_mj = ann_energy_mj if mode == "ANN" else snn_energy_mj
    return {
        "n_samples": samples,
        "mode": mode,
        "accuracy": 100.0 * correct / samples,
        "if_firing_density": (
            meter.if_nonzero / meter.if_elements if meter.if_elements else 0.0
        ),
        "ann_macs_per_sample": ann_macs,
        "snn_first_macs_per_sample": first_macs,
        "snn_rest_acs_per_sample": rest_acs,
        "event_synops_per_sample": event_synops,
        "ann_energy_mJ": ann_energy_mj,
        "snn_energy_mJ": snn_energy_mj,
        "snn_event_only_energy_mJ": event_only_mj,
        "energy_mJ": energy_mj,
        "energy_ratio_ann_over_snn": (
            1.0
            if mode == "ANN"
            else (ann_energy_mj / snn_energy_mj if snn_energy_mj > 0 else float("nan"))
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    method_order = {name: i for i, name in enumerate(METHODS)}
    for row in rows:
        grouped[(row["dataset"], row["method"], row["mode"], int(row["T"]))].append(row)
    metrics = [
        "accuracy",
        "if_firing_density",
        "energy_mJ",
        "ann_energy_mJ",
        "snn_energy_mJ",
        "snn_event_only_energy_mJ",
        "energy_ratio_ann_over_snn",
        "event_synops_per_sample",
    ]
    output = []
    for (dataset, method, mode, time_steps), values in sorted(
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
            "mode": mode,
            "T": time_steps,
            "n_seeds": len(values),
            "ann_macs_per_sample": values[0]["ann_macs_per_sample"],
        }
        for metric in metrics:
            samples = [float(row[metric]) for row in values]
            item[f"{metric}_mean"] = statistics.fmean(samples)
            item[f"{metric}_std"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
        output.append(item)
    return output


def print_table(rows: list[dict]) -> None:
    print("\nTable. VGG-16 energy (Horowitz 45nm: MAC=4.6 pJ, AC=0.9 pJ)")
    print(
        f"{'Dataset':<12} {'Method':<8} {'Mode':<4} {'T':>3} "
        f"{'Acc.(%)':>14} {'Energy(mJ)':>16}"
    )
    for row in rows:
        print(
            f"{row['dataset']:<12} {row['method_label']:<8} {row['mode']:<4} {row['T']:>3} "
            f"{row['accuracy_mean']:6.2f}±{row['accuracy_std']:<5.2f} "
            f"{row['energy_mJ_mean']:8.4f}±{row['energy_mJ_std']:<6.4f}"
        )


def load_model(dataset: str, ckpt: Path, device: torch.device, args) -> nn.Module:
    model = modelpool(args.arch, dataset)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(remap_legacy_vgg_state_dict(state), strict=True)
    return model.to(device).eval()


def main() -> None:
    args = parse_args()
    device = get_torch_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = []
    loaders = {}

    print(
        f"[INFO] Horowitz energy: E_MAC={E_MAC_PJ} pJ, E_AC={E_AC_PJ} pJ, "
        f"first-layer MAC + remaining event AC",
        flush=True,
    )

    for dataset in args.datasets:
        _, loaders[dataset] = datapool(
            dataset,
            args.batch_size,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
        )
        for method in args.methods:
            label = METHODS[method]["label"]
            for seed in args.seeds:
                ckpt = resolve_checkpoint(dataset, method, seed, args)
                seed_all(seed)
                model = load_model(dataset, ckpt, device, args)
                eval_steps = [0, *args.time_steps]
                for time_steps in eval_steps:
                    model.set_T(time_steps)
                    model.set_L(args.L)
                    model.set_mode("normal" if time_steps <= 0 else "rate_uniform")
                    model.set_spike_schedule("normal")
                    model.set_first_layer_input_noise_sigma(0.0)
                    result = evaluate(
                        model,
                        loaders[dataset],
                        device,
                        time_steps,
                        args.max_samples,
                    )
                    row = {
                        "dataset": dataset,
                        "arch": args.arch,
                        "method": method,
                        "method_label": label,
                        "seed": seed,
                        "L": args.L,
                        "T": time_steps,
                        **result,
                        "checkpoint": str(ckpt),
                    }
                    raw_rows.append(row)
                    print(
                        f"{dataset:10s} {label:7s} seed={seed} "
                        f"{result['mode']:3s} T={time_steps:2d} "
                        f"acc={result['accuracy']:.2f} "
                        f"E={result['energy_mJ']:.4f} mJ",
                        flush=True,
                    )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    summary = aggregate(raw_rows)
    write_csv(args.out_dir / "vgg16_horowitz_energy_raw.csv", raw_rows)
    write_csv(args.out_dir / "vgg16_horowitz_energy_summary.csv", summary)
    print_table(summary)
    print(f"Wrote {args.out_dir / 'vgg16_horowitz_energy_raw.csv'}")
    print(f"Wrote {args.out_dir / 'vgg16_horowitz_energy_summary.csv'}")


if __name__ == "__main__":
    main()
