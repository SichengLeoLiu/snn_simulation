#!/usr/bin/env python3
"""Measure clean post-IF feature-map scale and visualize Gaussian noise σ=0..5.

Matches the CIFAR VGG noise-sweep protocol: T=16, rate_uniform, noise added on
the first IF output (post_input_if).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Models import modelpool
from Models.VGG import remap_legacy_vgg_state_dict
from Models.layer import IF
from Preprocess import datapool
from utils import get_torch_device, seed_all
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--arch", default="vgg16")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "cifar10-checkpoints" / "vgg16_L[16].pth",
    )
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--mode", default="rate_uniform")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Fixed post-IF channel to visualize (default 0). Use -1 to pick max-energy.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="", help="Optional suffix for output filenames.")
    parser.add_argument(
        "--sigmas",
        nargs="+",
        type=float,
        default=[0.0, 1.0, 2.0, 3.0, 5.0],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT.parent / "plots",
    )
    return parser.parse_args()


def _first_if(model) -> IF:
    for module in model.modules():
        if isinstance(module, IF):
            return module
    raise RuntimeError("No IF layer found")


def _load_model(path: Path, args, device):
    model = modelpool(args.arch, args.dataset)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = remap_legacy_vgg_state_dict(state)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    model.set_L(args.L)
    model.set_T(args.T)
    model.set_mode(args.mode)
    model.set_first_layer_input_noise_type("gaussian")
    model.set_first_layer_input_noise_position("post_input_if")
    model.set_first_layer_input_noise_sigma(0.0)
    return model


def _capture_post_if(model, images):
    captured = {}

    def hook(_module, _inputs, output):
        captured["z"] = output.detach()

    handle = _first_if(model).register_forward_hook(hook)
    try:
        model.set_first_layer_input_noise_sigma(0.0)
        with torch.no_grad():
            model(images.clone())
    finally:
        handle.remove()
    if "z" not in captured:
        raise RuntimeError("Failed to capture first-IF output")
    return captured["z"]


def _to_tbchw(z: torch.Tensor, time_steps: int) -> torch.Tensor:
    if time_steps <= 1 or z.shape[0] % time_steps != 0:
        return z.unsqueeze(0)
    batch = z.shape[0] // time_steps
    return z.view(time_steps, batch, *z.shape[1:])


def _unnormalize(image_chw: torch.Tensor) -> np.ndarray:
    array = image_chw.detach().cpu().float().numpy()
    rgb = np.transpose(array, (1, 2, 0))
    rgb = np.clip(rgb * CIFAR_STD + CIFAR_MEAN, 0.0, 1.0)
    return rgb


def _accumulate_stats(model, loader, device, args) -> dict[str, float]:
    total_sum = 0.0
    total_abs = 0.0
    total_sq = 0.0
    total_count = 0
    vmin = math.inf
    vmax = -math.inf
    with torch.no_grad():
        for batch_index, (images, _labels) in enumerate(loader):
            if batch_index >= args.max_batches:
                break
            z = _capture_post_if(model, images.to(device))
            flat = z.float()
            total_sum += float(flat.sum().item())
            total_abs += float(flat.abs().sum().item())
            total_sq += float(flat.pow(2).sum().item())
            total_count += int(flat.numel())
            vmin = min(vmin, float(flat.min().item()))
            vmax = max(vmax, float(flat.max().item()))
    mean = total_sum / total_count
    mean_abs = total_abs / total_count
    rms = math.sqrt(total_sq / total_count)
    var = max(0.0, total_sq / total_count - mean * mean)
    return {
        "count": total_count,
        "mean": mean,
        "mean_abs": mean_abs,
        "std": math.sqrt(var),
        "rms": rms,
        "min": vmin,
        "max": vmax,
    }


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = get_torch_device("auto")
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _train_loader, test_loader = datapool(
        args.dataset,
        args.batch_size,
        num_workers=0,
        pin_memory=False,
    )
    model = _load_model(args.ckpt, args, device)
    lam = float(_first_if(model).thresh.detach().float().view(-1)[0].item())
    stats = _accumulate_stats(model, test_loader, device, args)

    images, _labels = next(iter(test_loader))
    vis_image = images[args.image_index].clone()
    images = images.to(device)
    clean = _capture_post_if(model, images.clone())
    tbchw = _to_tbchw(clean, args.T)
    time_steps, batch, channels, height, width = tbchw.shape
    sample = tbchw[:, args.image_index]
    rng = torch.Generator(device="cpu")
    rng.manual_seed(args.seed)

    channel_energy = sample.mean(dim=0).pow(2).mean(dim=(1, 2))
    if int(args.channel) < 0:
        channel = int(torch.argmax(channel_energy).item())
    else:
        channel = int(args.channel) % int(channels)

    sigmas = [float(value) for value in args.sigmas]
    noisy_rate_maps = []
    noisy_channel_maps = []
    for sigma in sigmas:
        if sigma <= 0:
            noisy = sample
        else:
            noise = torch.randn(sample.shape, generator=rng, device="cpu").to(sample.device)
            noisy = sample + sigma * noise
        noisy_rate_maps.append(noisy.mean(dim=0).mean(dim=0).cpu().numpy())
        noisy_channel_maps.append(noisy.mean(dim=0)[channel].cpu().numpy())

    vmin = float(np.percentile(noisy_rate_maps[0], 1))
    vmax = float(np.percentile(noisy_rate_maps[0], 99))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    ch_vmin = float(np.percentile(noisy_channel_maps[0], 1))
    ch_vmax = float(np.percentile(noisy_channel_maps[0], 99))
    if ch_vmax <= ch_vmin:
        ch_vmax = ch_vmin + 1e-6

    n_sigma = len(sigmas)
    fig, axes = plt.subplots(2, n_sigma + 1, figsize=(2.8 * (n_sigma + 1), 5.6), dpi=180)
    rgb = _unnormalize(vis_image)
    axes[0][0].imshow(rgb)
    axes[0][0].set_title("input image", fontsize=10)
    axes[0][0].axis("off")
    axes[1][0].axis("off")
    for index, sigma in enumerate(sigmas):
        ax_mean = axes[0][index + 1]
        ax_ch = axes[1][index + 1]
        im0 = ax_mean.imshow(noisy_rate_maps[index], cmap="viridis", vmin=vmin, vmax=vmax)
        ax_mean.set_title(rf"mean ch  $\sigma={sigma:g}$", fontsize=10)
        ax_mean.axis("off")
        ax_ch.imshow(noisy_channel_maps[index], cmap="viridis", vmin=ch_vmin, vmax=ch_vmax)
        ax_ch.set_title(rf"ch {channel}  $\sigma={sigma:g}$", fontsize=10)
        ax_ch.axis("off")
        if index == n_sigma - 1:
            fig.colorbar(im0, ax=ax_mean, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"{args.dataset} {args.arch} T={args.T} {args.mode}  |  "
        f"clean mean={stats['mean']:.3f}, rms={stats['rms']:.3f}, λ={lam:.3f}",
        fontsize=12,
    )
    tag = str(args.tag).strip()
    suffix = f"_{tag}" if tag else ""
    fig.tight_layout()
    png = out_dir / f"{args.dataset}_{args.arch}_post_if_noise_sigma0_5{suffix}.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for sigma in sigmas:
        snr = float("inf") if sigma <= 0 else stats["rms"] / sigma
        snr_db = float("inf") if sigma <= 0 else 20.0 * math.log10(max(snr, 1e-12))
        rows.append(
            {
                "tag": tag,
                "checkpoint": str(args.ckpt),
                "sigma": sigma,
                "lambda": lam,
                "clean_mean": stats["mean"],
                "clean_mean_abs": stats["mean_abs"],
                "clean_std": stats["std"],
                "clean_rms": stats["rms"],
                "clean_min": stats["min"],
                "clean_max": stats["max"],
                "sigma_over_rms": 0.0 if stats["rms"] == 0 else sigma / stats["rms"],
                "sigma_over_lambda": 0.0 if lam == 0 else sigma / abs(lam),
                "snr_linear": snr,
                "snr_db": snr_db,
            }
        )
    csv_path = out_dir / f"{args.dataset}_{args.arch}_post_if_noise_scale{suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"ckpt: {args.ckpt}")
    print(f"device: {device}")
    print(
        f"post-IF clean: mean={stats['mean']:.4f}  |x|={stats['mean_abs']:.4f}  "
        f"std={stats['std']:.4f}  rms={stats['rms']:.4f}  "
        f"min={stats['min']:.4f}  max={stats['max']:.4f}  lambda={lam:.4f}"
    )
    print(
        f"shape captured as time-averaged maps: "
        f"T={time_steps} B={batch} C={channels} {height}x{width}  viz_channel={channel}"
    )
    for row in rows:
        if row["sigma"] <= 0:
            print(f"  sigma={row['sigma']:g}: clean reference")
        else:
            print(
                f"  sigma={row['sigma']:g}: sigma/rms={row['sigma_over_rms']:.2f}  "
                f"sigma/lambda={row['sigma_over_lambda']:.2f}  SNR={row['snr_db']:.1f} dB"
            )
    print(f"[PLOT] {png}")
    print(f"[CSV]  {csv_path}")


if __name__ == "__main__":
    main()
