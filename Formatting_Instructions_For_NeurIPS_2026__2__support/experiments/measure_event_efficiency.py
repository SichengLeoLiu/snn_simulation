#!/usr/bin/env python3
"""Measure event activity and operation proxies for converted CNN2 models.

The reported SynOps count is an algorithmic event-driven proxy: every nonzero
presynaptic spike is charged once per downstream synaptic connection. It is not
a hardware energy or wall-clock latency measurement.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORT_DIR = Path(__file__).resolve().parents[1]
PAPER_DIR = SUPPORT_DIR.with_name(SUPPORT_DIR.name.removesuffix("_support"))
REPO_DIR = PAPER_DIR.parent
QCFS_DIR = REPO_DIR / "QCFS_simulation"
if str(QCFS_DIR) not in sys.path:
    sys.path.insert(0, str(QCFS_DIR))

from Models import IF, modelpool  # noqa: E402
from Preprocess import datapool  # noqa: E402
from utils import get_torch_device, seed_all  # noqa: E402


METHODS = {
    "mne": ("old_detach", "0p0001", "MNE-L2"),
    "orthogonal": ("orthogonal", "0p0001", "Orthogonal"),
    "no_reg": ("no_reg", "none", "No regularization"),
    "l2": ("l2", "none", "L2"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=["mnist", "fashion_mnist"]
    )
    parser.add_argument(
        "--methods", nargs="+", choices=sorted(METHODS), default=list(METHODS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--time-steps", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1024,
        help="Maximum test examples per checkpoint; 0 uses the full test set.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out-dir", type=Path, default=SUPPORT_DIR / "experiments" / "results"
    )
    return parser.parse_args()


def checkpoint_path(dataset: str, method: str, seed: int, args: argparse.Namespace) -> Path:
    method_tag, rc_tag, _ = METHODS[method]
    dataset_tag = "fashion" if dataset == "fashion_mnist" else dataset
    suffix = (
        f"{dataset_tag}_spectral_mne_cnn2_{method_tag}_rc{rc_tag}_seed{seed}"
        f"_ep{args.epochs}_L{args.L}_trainT0"
    )
    return (
        QCFS_DIR
        / f"{dataset}-checkpoints"
        / f"cnn2_L[{args.L}]_{suffix}.pth"
    )


class ActivityMeter:
    def __init__(self, model: nn.Module, time_steps: int):
        self.time_steps = time_steps
        self.if_nonzero = 0
        self.if_elements = 0
        self.dense_snn_ops = 0.0
        self.event_synops = 0.0
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
        if isinstance(module, nn.Linear):
            fan_in = module.in_features
            self.dense_snn_ops += float(output.numel() * fan_in)
            self.event_synops += float(torch.count_nonzero(x).item() * module.out_features)
            return

        kernel_h, kernel_w = module.kernel_size
        fan_in = (module.in_channels // module.groups) * kernel_h * kernel_w
        self.dense_snn_ops += float(output.numel() * fan_in)

        # Exact connection count, including padding and stride boundary effects.
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
        self.event_synops += float(synops.sum().item())


def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    time_steps: int,
    max_samples: int,
) -> dict[str, float]:
    meter = ActivityMeter(model, time_steps)
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
                logits = model(images).mean(0)
                correct += int(logits.argmax(dim=1).eq(labels).sum().item())
                samples += labels.numel()
    finally:
        meter.close()

    ann_dense_macs = meter.dense_snn_ops / float(time_steps)
    return {
        "n_samples": samples,
        "accuracy": 100.0 * correct / samples,
        "if_firing_density": meter.if_nonzero / meter.if_elements,
        "ann_dense_macs_per_sample": ann_dense_macs / samples,
        "snn_dense_ops_per_sample": meter.dense_snn_ops / samples,
        "event_synops_per_sample": meter.event_synops / samples,
        "event_synops_per_ann_mac": meter.event_synops / ann_dense_macs,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"], int(row["T"]))].append(row)

    metrics = [
        "accuracy",
        "if_firing_density",
        "event_synops_per_sample",
        "event_synops_per_ann_mac",
    ]
    output = []
    for (dataset, method, time_steps), values in sorted(grouped.items()):
        item = {
            "dataset": dataset,
            "method": method,
            "method_label": values[0]["method_label"],
            "T": time_steps,
            "n_seeds": len(values),
            "parameters": values[0]["parameters"],
            "fp32_param_kib": values[0]["fp32_param_kib"],
            "ann_dense_macs_per_sample": values[0]["ann_dense_macs_per_sample"],
        }
        for metric in metrics:
            samples = [float(row[metric]) for row in values]
            item[f"{metric}_mean"] = statistics.fmean(samples)
            item[f"{metric}_std"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
        output.append(item)
    return output


def main() -> None:
    args = parse_args()
    device = get_torch_device(args.device)
    raw_rows = []
    loaders = {}

    for dataset in args.datasets:
        _, loaders[dataset] = datapool(
            dataset,
            args.batch_size,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
        )
        for method in args.methods:
            _, _, method_label = METHODS[method]
            for seed in args.seeds:
                path = checkpoint_path(dataset, method, seed, args)
                if not path.exists():
                    raise FileNotFoundError(path)
                seed_all(seed)
                model = modelpool("cnn2", dataset)
                model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
                model.to(device).eval()
                parameters = sum(parameter.numel() for parameter in model.parameters())
                for time_steps in args.time_steps:
                    model.set_T(time_steps)
                    model.set_L(args.L)
                    model.set_mode("rate_uniform")
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
                        "method": method,
                        "method_label": method_label,
                        "seed": seed,
                        "L": args.L,
                        "T": time_steps,
                        "parameters": parameters,
                        "fp32_param_kib": parameters * 4.0 / 1024.0,
                        **result,
                        "checkpoint": str(path),
                    }
                    raw_rows.append(row)
                    print(
                        f"{dataset:13s} {method:10s} seed={seed} T={time_steps:2d} "
                        f"acc={result['accuracy']:.2f} density={result['if_firing_density']:.4f} "
                        f"SynOps/MAC={result['event_synops_per_ann_mac']:.3f}",
                        flush=True,
                    )
                del model

    summary_rows = aggregate(raw_rows)
    write_csv(args.out_dir / "event_efficiency_raw.csv", raw_rows)
    write_csv(args.out_dir / "event_efficiency_summary.csv", summary_rows)
    print(f"Wrote {args.out_dir / 'event_efficiency_raw.csv'}")
    print(f"Wrote {args.out_dir / 'event_efficiency_summary.csv'}")


if __name__ == "__main__":
    main()
