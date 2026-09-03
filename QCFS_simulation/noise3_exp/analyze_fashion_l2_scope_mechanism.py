from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Models import IF, modelpool
from Preprocess import datapool
from utils import get_torch_device, seed_all


METHOD_SPECS = {
    "wd_weight_only": ("none", "weights only"),
    "manual_l2_w_bn": ("0p00025", "weights + BN"),
    "manual_l2_w_if": ("0p00025", "weights + IF"),
    "manual_l2_w_bn_if": ("0p00025", "weights + BN + IF"),
    "manual_l2_all": ("0p00025", "all parameters"),
}


def _checkpoint_path(model: str, method: str, seed: int, epochs: int, level: int) -> Path:
    rc_tag = METHOD_SPECS[method][0]
    suffix = (
        f"fashion_spectral_mne_{model}_{method}_rc{rc_tag}_seed{seed}"
        f"_ep{epochs}_L{level}_trainT0"
    )
    return ROOT / "fashion_mnist-checkpoints" / f"{model}_L[{level}]_{suffix}.pth"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _quantization_boundary_margin(ratio: torch.Tensor, time_steps: int) -> torch.Tensor:
    nearest = torch.round(ratio - 0.5) + 0.5
    nearest = nearest.clamp(min=0.5, max=float(time_steps) - 0.5)
    return (ratio - nearest).abs()


class ActivationCapture:
    def __init__(self, model, time_steps: int):
        self.model = model
        self.time_steps = int(time_steps)
        self.batch_size = 0
        self.handles = []
        self.rate_maps = {}
        self.spikes = {}
        self.layer_stats = {}

        self.handles.append(model.conv1.register_forward_pre_hook(self._capture_input))
        for index in range(1, model.num_conv_layers + 1):
            module = getattr(model, f"if{index}")
            self.handles.append(
                module.register_forward_pre_hook(self._capture_if_input(f"if{index}"))
            )
            self.handles.append(
                module.register_forward_hook(self._capture_if_output(f"if{index}"))
            )

    def reset(self, batch_size: int) -> None:
        self.batch_size = int(batch_size)
        self.rate_maps = {}
        self.spikes = {}
        self.layer_stats = defaultdict(dict)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def _time_view(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().view(self.time_steps, self.batch_size, *tensor.shape[1:])

    def _capture_input(self, _module, inputs) -> None:
        x = self._time_view(inputs[0])
        self.rate_maps["post_input_if"] = x.mean(dim=0).float().cpu()
        self.layer_stats["post_input_if"]["rms"] = float(
            x.float().pow(2).mean().sqrt().item()
        )

    def _capture_if_input(self, name: str):
        def hook(module: IF, inputs) -> None:
            x = self._time_view(inputs[0]).float()
            threshold = module.thresh.detach().float().clamp(min=1e-6)
            ratio = x.sum(dim=0) / threshold
            margin = _quantization_boundary_margin(ratio, self.time_steps)
            self.layer_stats[name].update(
                {
                    "boundary_margin_mean": float(margin.mean().item()),
                    "boundary_margin_lt_0p05": float((margin < 0.05).float().mean().item()),
                    "boundary_margin_lt_0p10": float((margin < 0.10).float().mean().item()),
                }
            )

        return hook

    def _capture_if_output(self, name: str):
        def hook(_module: IF, _inputs, output) -> None:
            x = self._time_view(output)
            spike = x.ne(0)
            count = spike.sum(dim=0)
            self.rate_maps[name] = x.float().mean(dim=0).cpu()
            self.spikes[name] = spike.cpu()
            self.layer_stats[name].update(
                {
                    "rms": float(x.float().pow(2).mean().sqrt().item()),
                    "firing_rate": float(spike.float().mean().item()),
                    "zero_ratio": float(count.eq(0).float().mean().item()),
                    "saturation_ratio": float(
                        count.eq(self.time_steps).float().mean().item()
                    ),
                }
            )

        return hook


def _classification_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], float("-inf"))
    return true_logits - masked.max(dim=1).values


def _mean_or_nan(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def _median_or_nan(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _evaluate_pair(
    model,
    loader,
    capture: ActivationCapture,
    device: torch.device,
    sigma: float,
    noise_seed: int,
    max_samples: int,
):
    layer_totals = defaultdict(
        lambda: {
            "diff_sq": 0.0,
            "clean_sq": 0.0,
            "noisy_sq": 0.0,
            "dot": 0.0,
            "spike_mismatch": 0,
            "spike_total": 0,
            "batches": 0,
        }
    )
    scalar_stats = defaultdict(lambda: defaultdict(list))
    clean_margins = []
    noisy_margins = []
    margin_deltas = []
    clean_correct = 0
    noisy_correct = 0
    clean_correct_to_wrong = 0
    prediction_flips = 0
    total = 0

    for batch_index, (images, labels) in enumerate(loader):
        if total >= max_samples:
            break
        keep = min(images.shape[0], max_samples - total)
        images = images[:keep].to(device)
        labels = labels[:keep].to(device)
        capture.reset(keep)

        model.set_first_layer_input_noise_sigma(0.0)
        with torch.no_grad():
            clean_logits = model(images.clone()).mean(dim=0)
        clean_rates = {name: value.clone() for name, value in capture.rate_maps.items()}
        clean_spikes = {name: value.clone() for name, value in capture.spikes.items()}
        clean_stats = {
            name: dict(values) for name, values in capture.layer_stats.items()
        }

        seed_all(noise_seed + batch_index)
        capture.reset(keep)
        model.set_first_layer_input_noise_sigma(sigma)
        with torch.no_grad():
            noisy_logits = model(images.clone()).mean(dim=0)
        noisy_rates = capture.rate_maps
        noisy_spikes = capture.spikes
        noisy_stats = capture.layer_stats

        for layer_name, clean in clean_rates.items():
            noisy = noisy_rates[layer_name]
            clean_flat = clean.reshape(-1).double()
            noisy_flat = noisy.reshape(-1).double()
            totals = layer_totals[layer_name]
            totals["diff_sq"] += float((noisy_flat - clean_flat).pow(2).sum().item())
            totals["clean_sq"] += float(clean_flat.pow(2).sum().item())
            totals["noisy_sq"] += float(noisy_flat.pow(2).sum().item())
            totals["dot"] += float((clean_flat * noisy_flat).sum().item())
            totals["batches"] += 1
            if layer_name in clean_spikes:
                totals["spike_mismatch"] += int(
                    clean_spikes[layer_name].ne(noisy_spikes[layer_name]).sum().item()
                )
                totals["spike_total"] += clean_spikes[layer_name].numel()

            for key, value in clean_stats[layer_name].items():
                scalar_stats[layer_name][f"clean_{key}"].append(value)
            for key, value in noisy_stats[layer_name].items():
                scalar_stats[layer_name][f"noisy_{key}"].append(value)

        clean_pred = clean_logits.argmax(dim=1)
        noisy_pred = noisy_logits.argmax(dim=1)
        clean_ok = clean_pred.eq(labels)
        noisy_ok = noisy_pred.eq(labels)
        clean_correct += int(clean_ok.sum().item())
        noisy_correct += int(noisy_ok.sum().item())
        clean_correct_to_wrong += int((clean_ok & ~noisy_ok).sum().item())
        prediction_flips += int(clean_pred.ne(noisy_pred).sum().item())

        clean_margin = _classification_margin(clean_logits, labels)
        noisy_margin = _classification_margin(noisy_logits, labels)
        clean_margins.extend(clean_margin.cpu().tolist())
        noisy_margins.extend(noisy_margin.cpu().tolist())
        margin_deltas.extend((noisy_margin - clean_margin).cpu().tolist())
        total += keep

    model.set_first_layer_input_noise_sigma(0.0)
    layer_rows = []
    previous_relative = None
    for layer_index, layer_name in enumerate(
        ["post_input_if"] + [f"if{i}" for i in range(1, model.num_conv_layers + 1)]
    ):
        totals = layer_totals[layer_name]
        relative_l2 = math.sqrt(totals["diff_sq"] / max(totals["clean_sq"], 1e-18))
        cosine = totals["dot"] / math.sqrt(
            max(totals["clean_sq"] * totals["noisy_sq"], 1e-18)
        )
        amplification = (
            relative_l2 / max(previous_relative, 1e-12)
            if previous_relative is not None
            else 1.0
        )
        previous_relative = relative_l2
        stats = scalar_stats[layer_name]
        layer_rows.append(
            {
                "layer_index": layer_index,
                "layer": layer_name,
                "relative_l2": relative_l2,
                "layer_amplification": amplification,
                "cosine_similarity": cosine,
                "spike_mismatch_rate": (
                    totals["spike_mismatch"] / totals["spike_total"]
                    if totals["spike_total"]
                    else float("nan")
                ),
                **{key: _mean_or_nan(values) for key, values in stats.items()},
            }
        )

    output_row = {
        "n_samples": total,
        "clean_accuracy": 100.0 * clean_correct / total,
        "noisy_accuracy": 100.0 * noisy_correct / total,
        "accuracy_drop": 100.0 * (clean_correct - noisy_correct) / total,
        "prediction_flip_rate": prediction_flips / total,
        "clean_correct_to_wrong_rate": (
            clean_correct_to_wrong / clean_correct if clean_correct else float("nan")
        ),
        "clean_margin_mean": _mean_or_nan(clean_margins),
        "clean_margin_median": _median_or_nan(clean_margins),
        "noisy_margin_mean": _mean_or_nan(noisy_margins),
        "noisy_margin_median": _median_or_nan(noisy_margins),
        "margin_change_mean": _mean_or_nan(margin_deltas),
        "margin_change_median": _median_or_nan(margin_deltas),
        "clean_negative_margin_rate": sum(value < 0 for value in clean_margins) / total,
        "noisy_negative_margin_rate": sum(value < 0 for value in noisy_margins) / total,
    }
    return layer_rows, output_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired layerwise and classification-margin diagnostics for Fashion CNN6."
    )
    parser.add_argument("--model", default="cnn6_vgg")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_SPECS),
        choices=sorted(METHOD_SPECS),
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--sigmas", nargs="+", default=[0.5, 1.0], type=float)
    parser.add_argument("--noise-repeats", default=3, type=int)
    parser.add_argument("--noise-seed", default=20260803, type=int)
    parser.add_argument("--max-samples", default=2048, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--L", default=16, type=int)
    parser.add_argument("--T", default=16, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT.parent / "important_results" / "fashion_l2_scope_followup",
    )
    args = parser.parse_args()
    if args.T <= 0:
        raise ValueError("--T must be positive for SNN diagnostics")
    if args.noise_repeats < 1 or args.max_samples < 1:
        raise ValueError("--noise-repeats and --max-samples must be positive")
    if any(sigma <= 0 for sigma in args.sigmas):
        raise ValueError("diagnostic sigmas must be positive")
    return args


def main() -> None:
    args = parse_args()
    device = get_torch_device(args.device)
    _, test_loader = datapool(
        "fashion_mnist",
        args.batch_size,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    layer_rows = []
    output_rows = []

    for method in args.methods:
        checkpoint = _checkpoint_path(
            args.model, method, args.seed, args.epochs, args.L
        )
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        model = modelpool(args.model, "fashion_mnist")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
        model.to(device)
        model.set_T(args.T)
        model.set_L(args.L)
        model.set_mode("rate_uniform")
        model.set_spike_schedule("normal")
        model.eval()
        capture = ActivationCapture(model, args.T)
        try:
            for sigma_index, sigma in enumerate(sorted(set(args.sigmas))):
                for noise_repeat in range(args.noise_repeats):
                    noise_seed = (
                        args.noise_seed + sigma_index * 10000 + noise_repeat * 1000
                    )
                    print(
                        f"[PAIR] method={method} sigma={sigma:g} repeat={noise_repeat}",
                        flush=True,
                    )
                    pair_layers, pair_output = _evaluate_pair(
                        model,
                        test_loader,
                        capture,
                        device,
                        sigma,
                        noise_seed,
                        args.max_samples,
                    )
                    common = {
                        "model": args.model,
                        "method": method,
                        "method_label": METHOD_SPECS[method][1],
                        "seed": args.seed,
                        "sigma": f"{sigma:.6g}",
                        "noise_repeat": noise_repeat,
                        "noise_seed": noise_seed,
                        "L": args.L,
                        "T": args.T,
                        "noise_position": "post_input_if",
                        "checkpoint": str(checkpoint),
                    }
                    layer_rows.extend({**common, **row} for row in pair_layers)
                    output_rows.append({**common, **pair_output})
        finally:
            capture.close()

    _write_csv(args.out_dir / "layerwise_mechanism_raw.csv", layer_rows)
    _write_csv(args.out_dir / "output_margin_raw.csv", output_rows)
    print(f"[DONE] {args.out_dir / 'layerwise_mechanism_raw.csv'}")
    print(f"[DONE] {args.out_dir / 'output_margin_raw.csv'}")


if __name__ == "__main__":
    main()
