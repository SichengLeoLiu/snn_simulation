from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from Models.cnn_mnist import remap_legacy_cnn2_state_dict
from Models.layer import IF, MergeTemporalDim, ExpandTemporalDim, add_dimention
from Models.spike_temporal_adjust import SPIKE_SCHEDULE_MODES
from Models.spike_temporal_adjust import (
    temporal_rearrange_after_first_if,
    first_conv_with_weight_sign_schedule,
)
from Preprocess import datapool
from utils import get_torch_device, seed_all


def parse_lt_from_name(stem: str) -> Tuple[Optional[int], int]:
    m_l = re.search(r"L\[(\d+)\]", stem)
    m_t = re.search(r"T\[(\d+)\]", stem)
    l_val = int(m_l.group(1)) if m_l else None
    t_val = int(m_t.group(1)) if m_t else 0
    return l_val, t_val


def detect_schedule_from_name(stem: str) -> str:
    for mode in sorted(SPIKE_SCHEDULE_MODES, key=len, reverse=True):
        if mode != "normal" and mode in stem:
            return mode
    return "normal"


def extract_activation(logits, feat_if1, feat_if2, layer_name: str) -> torch.Tensor:
    if layer_name == "if1":
        return feat_if1
    if layer_name == "if2":
        return feat_if2
    if layer_name == "logits":
        if logits.dim() == 3:
            return logits.mean(0)
        return logits
    raise ValueError(f"Unsupported activation layer: {layer_name}")


class RunningMoments:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0

    def update(self, x: torch.Tensor) -> None:
        v = x.detach().double().reshape(-1)
        self.count += int(v.numel())
        self.sum += float(v.sum().item())
        self.sumsq += float((v * v).sum().item())

    def variance(self) -> float:
        if self.count == 0:
            return float("nan")
        mean = self.sum / self.count
        var = self.sumsq / self.count - mean * mean
        return max(var, 0.0)

    def rms(self) -> float:
        if self.count == 0:
            return float("nan")
        return math.sqrt(max(self.sumsq / self.count, 0.0))


def iter_checkpoints(root: Path, dirs: Iterable[str]) -> Iterable[Path]:
    for d in dirs:
        folder = root / d
        if not folder.exists():
            print(f"[WARN] Missing directory, skip: {folder}")
            continue
        for p in sorted(folder.glob("*.pth")):
            yield p


class CNN2MNISTFlexible(nn.Module):
    """可按 checkpoint 自动适配通道数的 CNN2（保持与原模型接口一致）。"""

    def __init__(self, c1: int = 2, c2: int = 4, num_classes: int = 10):
        super().__init__()
        self.T = 0
        self.merge = MergeTemporalDim(0)
        self.expand = ExpandTemporalDim(0)
        self.spike_schedule = "normal"
        self.first_layer_input_noise_sigma = 0.0
        self.first_layer_input_noise_type = "gaussian"

        self.input_if = IF()
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.if1 = IF()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.if2 = IF()
        self.pool2 = nn.MaxPool2d(2)

        self.classifier = nn.Linear(c2 * 7 * 7, num_classes)

    def set_spike_schedule(self, mode: str):
        if mode not in SPIKE_SCHEDULE_MODES:
            raise ValueError(f"spike_schedule must be in {sorted(SPIKE_SCHEDULE_MODES)}")
        self.spike_schedule = mode

    def set_T(self, T):
        self.T = T
        for module in self.modules():
            if isinstance(module, (IF, ExpandTemporalDim)):
                module.T = T
                if T > 0:
                    module.spike_counts = [0] * T
                    module.total_elements = [0] * T

    def set_L(self, L):
        for module in self.modules():
            if isinstance(module, IF):
                module.L = L

    def set_mode(self, mode="normal"):
        for module in self.modules():
            if isinstance(module, IF):
                module.mode = mode

    def set_first_layer_input_noise_sigma(self, sigma=0.0):
        self.first_layer_input_noise_sigma = max(0.0, float(sigma))

    def set_first_layer_input_noise_type(self, noise_type="gaussian"):
        nt = str(noise_type).strip().lower()
        if nt not in ("gaussian", "pink"):
            raise ValueError("noise_type must be gaussian or pink")
        self.first_layer_input_noise_type = nt

    def _inject_first_layer_input_noise(self, x):
        sigma = self.first_layer_input_noise_sigma
        if sigma <= 0:
            return x
        if self.first_layer_input_noise_type == "pink":
            # 复用简化版：此脚本默认 gaussian；pink 退化为 white 近似。
            noise = torch.randn_like(x)
        else:
            noise = torch.randn_like(x)
        return x + noise * sigma

    @staticmethod
    def _if_out_to_firing_map(x_tb, if_layer, T):
        th = if_layer.thresh.data.clamp(min=1e-8)
        if T and T > 0:
            tb, c, h, w = x_tb.shape
            b = tb // T
            s = x_tb.view(T, b, c, h, w).sum(dim=0)
            return s / th
        return x_tb / th

    def forward_with_if_features(self, x):
        T = self.T
        if T > 0:
            x = x.clone()
            x = add_dimention(x, T)
            x = self.merge(x)

        x = self.input_if(x)
        x = self._inject_first_layer_input_noise(x)

        if T > 0:
            sch = self.spike_schedule
            if sch in ("weight_sign_pos_front", "weight_sign_neg_front"):
                x = first_conv_with_weight_sign_schedule(x, T, self.conv1, sch)
            else:
                x = temporal_rearrange_after_first_if(x, T, sch)
                x = self.conv1(x)
        else:
            x = self.conv1(x)

        x = self.bn1(x)
        x = self.if1(x)
        feat_if1 = self._if_out_to_firing_map(x, self.if1, T)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.if2(x)
        feat_if2 = self._if_out_to_firing_map(x, self.if2, T)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if T > 0:
            x = self.expand(x)
        return x, feat_if1, feat_if2


def load_mnist_model_for_checkpoint(
    ckpt_path: Path,
    device: torch.device,
    l_value: int,
    t_value: int,
    schedule: str,
    base_arch: str,
) -> torch.nn.Module:
    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict, _ = remap_legacy_cnn2_state_dict(state_dict)

    if "conv1.weight" not in state_dict or "conv2.weight" not in state_dict:
        raise KeyError("checkpoint missing conv1.weight/conv2.weight")

    c1 = int(state_dict["conv1.weight"].shape[0])
    c2 = int(state_dict["conv2.weight"].shape[0])
    if base_arch.lower() not in ("cnn2", "cnn2_mnist"):
        raise ValueError(f"Unsupported arch for this script: {base_arch}")
    model = CNN2MNISTFlexible(c1=c1, c2=c2, num_classes=10)

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    model.set_T(t_value)
    model.set_L(l_value)
    model.set_mode("normal")
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule(schedule)
    return model


@torch.no_grad()
def estimate_sigma_eff_for_model(
    model,
    test_loader,
    device: torch.device,
    activation_layer: str,
    noise_sigma: float,
    repeats: int,
    max_batches: int,
) -> Dict[str, float]:
    if not hasattr(model, "set_first_layer_input_noise_sigma"):
        raise RuntimeError("Model does not support first-layer input noise injection.")
    if not hasattr(model, "forward_with_if_features"):
        raise RuntimeError("Model does not support forward_with_if_features.")

    moments = RunningMoments()

    model.set_first_layer_input_noise_sigma(0.0)
    for batch_idx, (images, _) in enumerate(test_loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(device)
        logits_c, feat1_c, feat2_c = model.forward_with_if_features(images)
        clean = extract_activation(logits_c, feat1_c, feat2_c, activation_layer)

        for _ in range(repeats):
            model.set_first_layer_input_noise_sigma(noise_sigma)
            logits_n, feat1_n, feat2_n = model.forward_with_if_features(images)
            noisy = extract_activation(logits_n, feat1_n, feat2_n, activation_layer)
            delta = noisy - clean
            moments.update(delta)

    model.set_first_layer_input_noise_sigma(0.0)
    var_eff = moments.variance()
    sigma_eff = math.sqrt(var_eff) if not math.isnan(var_eff) else float("nan")
    rms_delta = moments.rms()

    return {
        "sigma_eff": sigma_eff,
        "var_eff": var_eff,
        "rms_delta": rms_delta,
        "n_delta_elements": moments.count,
    }


def aggregate_by_lt(rows):
    bucket = defaultdict(list)
    for r in rows:
        key = (int(r["L"]), int(r["T"]))
        bucket[key].append(r)

    out = []
    for (l_val, t_val), items in sorted(bucket.items(), key=lambda x: (x[0][0], x[0][1])):
        sigmas = [float(x["sigma_eff"]) for x in items]
        vars_ = [float(x["var_eff"]) for x in items]
        out.append(
            {
                "L": l_val,
                "T": t_val,
                "num_checkpoints": len(items),
                "sigma_eff_mean": sum(sigmas) / len(sigmas),
                "sigma_eff_min": min(sigmas),
                "sigma_eff_max": max(sigmas),
                "var_eff_mean": sum(vars_) / len(vars_),
            }
        )
    return out


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate sigma_eff by clean/noisy activation differences."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root directory containing checkpoint folders.",
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["mnist-checkpoints", "mnist-checkpoints_c4_c8", "mnist-checkpoints_c16_c32"],
        help="Checkpoint folders (relative to --root).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="Dataset name for datapool/modelpool.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="cnn2",
        help="Model arch for modelpool.",
    )
    parser.add_argument(
        "--activation-layer",
        type=str,
        default="if1",
        choices=["if1", "if2", "logits"],
        help="Activation tensor used for delta = noisy - clean.",
    )
    parser.add_argument(
        "--noise-variance",
        type=float,
        default=0.5,
        help="Injected Gaussian variance at first-layer input.",
    )
    parser.add_argument(
        "--noise-type",
        type=str,
        default="gaussian",
        choices=["gaussian", "pink"],
        help="First-layer noise type.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=8,
        help="Noisy repeats per batch.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=8,
        help="Maximum test batches per checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Test loader batch size.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | mps | cuda | cpu",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=44,
        help="Random seed.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("noise3_exp/sigma_eff_by_checkpoint.csv"),
        help="Per-checkpoint output CSV path.",
    )
    parser.add_argument(
        "--out-lt",
        type=Path,
        default=Path("noise3_exp/sigma_eff_by_L_T.csv"),
        help="Aggregated (L,T) output CSV path.",
    )
    args = parser.parse_args()

    if args.dataset.lower().replace("-", "").replace("_", "") != "mnist":
        raise ValueError("Current script is focused on MNIST CNN2 checkpoints.")

    if args.noise_variance < 0:
        raise ValueError("noise_variance must be >= 0.")
    noise_sigma = math.sqrt(args.noise_variance)

    device = get_torch_device(args.device)
    print(f"[INFO] device={device}")
    print(
        "[INFO] activation_layer=%s, noise_type=%s, noise_variance=%.6f, noise_sigma=%.6f"
        % (args.activation_layer, args.noise_type, args.noise_variance, noise_sigma)
    )

    seed_all(args.seed)
    _, test_loader = datapool(
        "mnist",
        args.batch_size,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    rows = []
    for ckpt in iter_checkpoints(args.root, args.dirs):
        stem = ckpt.stem
        l_val, t_val = parse_lt_from_name(stem)
        if l_val is None:
            print(f"[WARN] Cannot parse L from checkpoint name, skip: {ckpt}")
            continue
        schedule = detect_schedule_from_name(stem)

        try:
            model = load_mnist_model_for_checkpoint(
                ckpt_path=ckpt,
                device=device,
                l_value=l_val,
                t_value=t_val,
                schedule=schedule,
                base_arch=args.arch,
            )
            if hasattr(model, "set_first_layer_input_noise_type"):
                model.set_first_layer_input_noise_type(args.noise_type)
            stats = estimate_sigma_eff_for_model(
                model=model,
                test_loader=test_loader,
                device=device,
                activation_layer=args.activation_layer,
                noise_sigma=noise_sigma,
                repeats=args.repeats,
                max_batches=args.max_batches,
            )

            row = {
                "checkpoint": str(ckpt.relative_to(args.root)),
                "L": l_val,
                "T": t_val,
                "schedule": schedule,
                "activation_layer": args.activation_layer,
                "noise_type": args.noise_type,
                "noise_variance": args.noise_variance,
                "noise_sigma": noise_sigma,
                "sigma_eff": stats["sigma_eff"],
                "var_eff": stats["var_eff"],
                "rms_delta": stats["rms_delta"],
                "n_delta_elements": stats["n_delta_elements"],
                "repeats": args.repeats,
                "max_batches": args.max_batches,
            }
            rows.append(row)
            print(
                "[OK] %s | L=%d T=%d sch=%s | sigma_eff=%.6e var_eff=%.6e"
                % (
                    ckpt.name,
                    l_val,
                    t_val,
                    schedule,
                    row["sigma_eff"],
                    row["var_eff"],
                )
            )
        except Exception as exc:
            print(f"[ERR] {ckpt}: {exc}")

    rows.sort(key=lambda x: (x["checkpoint"]))
    write_csv(
        args.out,
        rows,
        fieldnames=[
            "checkpoint",
            "L",
            "T",
            "schedule",
            "activation_layer",
            "noise_type",
            "noise_variance",
            "noise_sigma",
            "sigma_eff",
            "var_eff",
            "rms_delta",
            "n_delta_elements",
            "repeats",
            "max_batches",
        ],
    )
    lt_rows = aggregate_by_lt(rows)
    write_csv(
        args.out_lt,
        lt_rows,
        fieldnames=[
            "L",
            "T",
            "num_checkpoints",
            "sigma_eff_mean",
            "sigma_eff_min",
            "sigma_eff_max",
            "var_eff_mean",
        ],
    )
    print(f"\nSaved {len(rows)} rows to: {args.out}")
    print(f"Saved {len(lt_rows)} rows to: {args.out_lt}")


if __name__ == "__main__":
    main()
