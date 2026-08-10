#!/usr/bin/env python3
"""
Priority-1 relative / SNR-matched post-input-IF noise control for CIFAR-10 VGG16.

Reuses the six seed-42 checkpoints (no retraining). For each checkpoint we
calibrate the clean post-input-IF activation RMS (and input-IF threshold λ),
then inject Gaussian noise through the existing absolute-sigma API:

  absolute:              σ = σ_abs
  lambda_relative:       σ = α · λ_input
  activation_relative:   σ = α · RMS(h_input)
  snr:                   σ = RMS(h_input) / 10^(SNR_dB / 20)

Recommended default is SNR-matched with
  SNR ∈ {30, 20, 15, 10, 5, 0} dB
and 5 noise repeats per (method, level).
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Models import modelpool
from Models.VGG import remap_legacy_vgg_state_dict
from Models.layer import IF
from Preprocess import datapool
from utils import get_torch_device, seed_all, val


DEFAULT_CKPTS = {
    "L1-all": "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_l1_all_rc1em05_seed42_L16_trainT0.pth",
    "MNE-all": "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_mne_l2_all_rc0p0001_seed42_L16_trainT0.pth",
    "L2-all": "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_manual_l2_all_rc0p00025_seed42_L16_trainT0.pth",
    "Weights-only": "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_weight_decay_weights_only_rcnone_seed42_L16_trainT0.pth",
    "Weights+BN-gamma": "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_manual_l2_w_bn_gamma_rc0p00025_seed42_L16_trainT0.pth",
    "MNE-standard": "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_old_detach_rc0p0001_seed42_L16_trainT0.pth",
}

SCALE_MODES = ("absolute", "lambda_relative", "activation_relative", "snr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--arch", default="vgg16")
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--mode", default="rate_uniform", choices=["rate_uniform", "normal"])
    parser.add_argument("--spike-schedule", default="normal")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42, help="Checkpoint seed / dataloader seed.")
    parser.add_argument("--noise-seed", type=int, default=20260809)
    parser.add_argument("--noise-repeats", type=int, default=5)
    parser.add_argument(
        "--noise-position",
        default="post_input_if",
        choices=["post_input_if", "pre_input_if"],
        help="Where to inject first-layer Gaussian noise relative to the input IF.",
    )
    parser.add_argument("--rms-calibration-batches", type=int, default=20)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_CKPTS),
        choices=sorted(DEFAULT_CKPTS),
    )
    parser.add_argument(
        "--ckpt",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        help="Optional override: --ckpt Label path.pth",
    )
    parser.add_argument(
        "--scale-modes",
        nargs="+",
        default=["snr"],
        choices=list(SCALE_MODES),
        help="Noise scaling modes to evaluate. Prefer 'snr'.",
    )
    parser.add_argument(
        "--snr-db",
        nargs="+",
        type=float,
        default=[30, 20, 15, 10, 5, 0],
        help="SNR grid (dB) for --scale-modes snr. Clean (no noise) is always included.",
    )
    parser.add_argument(
        "--sigmas",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Absolute sigma grid for --scale-modes absolute.",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
        help="Multiplier grid for lambda_relative / activation_relative.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT.parent
        / "important_results"
        / "cifar10_vgg16_relative_snr_noise_seed42",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.noise_repeats < 1:
        raise ValueError("--noise-repeats must be >= 1")
    if args.rms_calibration_batches < 1:
        raise ValueError("--rms-calibration-batches must be >= 1")
    if any(x < 0 for x in args.sigmas):
        raise ValueError("--sigmas must be non-negative")
    if any(x < 0 for x in args.alphas):
        raise ValueError("--alphas must be non-negative")
    return args


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_ckpt(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _load_model(path: Path, args, device):
    model = modelpool(args.arch, args.dataset)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = remap_legacy_vgg_state_dict(state)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    model.set_L(args.L)
    model.set_T(args.T)
    model.set_mode(args.mode)
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule(args.spike_schedule)
    if hasattr(model, "set_first_layer_input_noise_type"):
        model.set_first_layer_input_noise_type("gaussian")
    if hasattr(model, "set_first_layer_input_noise_position"):
        model.set_first_layer_input_noise_position(args.noise_position)
    model.set_first_layer_input_noise_sigma(0.0)
    return model


def _first_input_if(model) -> tuple[str, IF]:
    for name, module in model.named_modules():
        if isinstance(module, IF):
            return name, module
    raise RuntimeError("No IF layer found for input-IF calibration")


def _estimate_injection_site_rms(
    model, loader, device, max_batches: int, noise_position: str
) -> float:
    """
    RMS of the clean tensor that receives noise:
      post_input_if -> first IF output
      pre_input_if  -> first IF input (BN output)
    """
    _, input_if = _first_input_if(model)
    total_sq = 0.0
    total_count = 0

    def _accumulate(tensor):
        nonlocal total_sq, total_count
        detached = tensor.detach().float()
        total_sq += float(detached.pow(2).sum().item())
        total_count += int(detached.numel())

    if noise_position == "pre_input_if":
        def capture(_module, inputs):
            _accumulate(inputs[0])

        handle = input_if.register_forward_pre_hook(capture)
    else:
        def capture(_module, _inputs, output):
            _accumulate(output)

        handle = input_if.register_forward_hook(capture)
    try:
        model.set_first_layer_input_noise_sigma(0.0)
        with torch.no_grad():
            for batch_index, (images, _labels) in enumerate(loader):
                if batch_index >= max_batches:
                    break
                model(images.to(device).clone())
    finally:
        handle.remove()
    if total_count <= 0:
        raise RuntimeError(f"Could not estimate RMS at {noise_position}")
    return math.sqrt(total_sq / total_count)


def _calibrate_scales(model, loader, device, args) -> dict[str, float]:
    _, input_if = _first_input_if(model)
    lam = float(input_if.thresh.detach().float().abs().clamp(min=1e-8).view(-1)[0].item())
    rms = _estimate_injection_site_rms(
        model,
        loader,
        device,
        args.rms_calibration_batches,
        args.noise_position,
    )
    return {"lambda_input": lam, "rms_input": rms}


def _level_grid(scale_mode: str, args) -> list[tuple[str, float]]:
    """
    Return ordered (level_name, level_value) pairs.
    Clean / no-noise is always first when not already present.
    """
    if scale_mode == "snr":
        levels = [("clean", float("inf"))]
        for snr in args.snr_db:
            levels.append((f"snr_{snr:g}dB", float(snr)))
        return levels
    if scale_mode == "absolute":
        values = sorted(set(float(x) for x in args.sigmas))
        if 0.0 not in values:
            values = [0.0] + values
        return [(f"sigma_{v:g}", v) for v in values]
    # lambda_relative / activation_relative
    values = sorted(set(float(x) for x in args.alphas))
    if 0.0 not in values:
        values = [0.0] + values
    return [(f"alpha_{v:g}", v) for v in values]


def _actual_sigma(scale_mode: str, level_value: float, scales: dict[str, float]) -> float:
    if scale_mode == "absolute":
        return max(0.0, float(level_value))
    if scale_mode == "lambda_relative":
        return max(0.0, float(level_value) * scales["lambda_input"])
    if scale_mode == "activation_relative":
        return max(0.0, float(level_value) * scales["rms_input"])
    if scale_mode == "snr":
        if not math.isfinite(level_value):
            return 0.0
        return float(scales["rms_input"] / (10.0 ** (float(level_value) / 20.0)))
    raise ValueError(f"Unknown scale mode: {scale_mode}")


def _aggregate(raw_rows: list[dict]):
    grouped = defaultdict(list)
    for row in raw_rows:
        key = (
            row["method"],
            row["scale_mode"],
            row["level_name"],
            float(row["level_value"]) if math.isfinite(float(row["level_value"])) else float("inf"),
        )
        grouped[key].append(float(row["accuracy"]))

    mean_rows = []
    for (method, scale_mode, level_name, level_value), values in sorted(
        grouped.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][3] if math.isfinite(kv[0][3]) else 1e9)
    ):
        mean_rows.append(
            {
                "method": method,
                "scale_mode": scale_mode,
                "level_name": level_name,
                "level_value": f"{level_value:.9g}" if math.isfinite(level_value) else "inf",
                "n_repeats": len(values),
                "acc_mean": f"{statistics.mean(values):.6f}",
                "acc_std": (
                    f"{statistics.stdev(values):.6f}" if len(values) > 1 else "0.000000"
                ),
                "actual_sigma_mean": "",  # filled below from raw
            }
        )

    # Attach mean actual_sigma from raw rows.
    sigma_by_key = defaultdict(list)
    scale_ref_by_key = {}
    for row in raw_rows:
        key = (row["method"], row["scale_mode"], row["level_name"])
        sigma_by_key[key].append(float(row["actual_sigma"]))
        scale_ref_by_key[key] = (
            float(row["lambda_input"]),
            float(row["rms_input"]),
        )
    for row in mean_rows:
        key = (row["method"], row["scale_mode"], row["level_name"])
        sigmas = sigma_by_key[key]
        row["actual_sigma_mean"] = f"{statistics.mean(sigmas):.9g}"
        lam, rms = scale_ref_by_key[key]
        row["lambda_input"] = f"{lam:.9g}"
        row["rms_input"] = f"{rms:.9g}"

    # Per-method summary at the noisiest level + retention vs clean.
    summary_rows = []
    by_method_mode = defaultdict(list)
    for row in mean_rows:
        by_method_mode[(row["method"], row["scale_mode"])].append(row)
    for (method, scale_mode), rows in sorted(by_method_mode.items()):
        def sort_key(r):
            lv = float("inf") if r["level_value"] == "inf" else float(r["level_value"])
            # For SNR, larger dB is cleaner; put clean/inf first, then descending SNR.
            if scale_mode == "snr":
                return (0 if r["level_value"] == "inf" else 1, -lv if math.isfinite(lv) else 0)
            return (lv,)

        rows = sorted(rows, key=sort_key)
        clean = float(rows[0]["acc_mean"])
        end = float(rows[-1]["acc_mean"])
        summary_rows.append(
            {
                "method": method,
                "scale_mode": scale_mode,
                "clean_acc": f"{clean:.6f}",
                "end_level": rows[-1]["level_name"],
                "end_level_value": rows[-1]["level_value"],
                "end_actual_sigma": rows[-1]["actual_sigma_mean"],
                "end_acc": f"{end:.6f}",
                "end_acc_std": rows[-1]["acc_std"],
                "absolute_drop": f"{clean - end:.6f}",
                "end_retention": f"{end / clean:.6f}" if clean > 0 else "nan",
                "lambda_input": rows[0]["lambda_input"],
                "rms_input": rows[0]["rms_input"],
            }
        )
    return mean_rows, summary_rows


def _plot(mean_rows: list[dict], out_dir: Path, noise_position: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib unavailable; skip plots", flush=True)
        return

    methods_order = [
        "MNE-standard",
        "Weights-only",
        "L1-all",
        "MNE-all",
        "L2-all",
        "Weights+BN-gamma",
    ]
    colors = {
        "MNE-standard": "#1f77b4",
        "Weights-only": "#2ca02c",
        "L1-all": "#17becf",
        "MNE-all": "#ff7f0e",
        "L2-all": "#d62728",
        "Weights+BN-gamma": "#9467bd",
    }

    by_mode = defaultdict(list)
    for row in mean_rows:
        by_mode[row["scale_mode"]].append(row)

    for scale_mode, rows in by_mode.items():
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        snr_tick_labels = None
        snr_tick_xs = None

        def sort_key(r):
            lv = float("inf") if r["level_value"] == "inf" else float(r["level_value"])
            if scale_mode == "snr":
                return (0 if r["level_value"] == "inf" else 1, -lv if math.isfinite(lv) else 0)
            return (lv,)

        for method in methods_order:
            pts = [r for r in rows if r["method"] == method]
            if not pts:
                continue
            pts = sorted(pts, key=sort_key)
            if scale_mode == "snr":
                xs = list(range(len(pts)))
                snr_tick_xs = xs
                snr_tick_labels = [
                    "clean" if p["level_value"] == "inf" else f'{float(p["level_value"]):g}'
                    for p in pts
                ]
            else:
                xs = [float(p["level_value"]) for p in pts]
            ys = [float(p["acc_mean"]) for p in pts]
            yerr = [float(p["acc_std"]) for p in pts]
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker="o",
                linewidth=2,
                label=method,
                color=colors.get(method),
            )
        ax.set_ylabel("Accuracy (%)")
        if scale_mode == "snr":
            if snr_tick_xs is not None and snr_tick_labels is not None:
                ax.set_xticks(snr_tick_xs)
                ax.set_xticklabels(snr_tick_labels)
            ax.set_xlabel("SNR (dB)  [left = cleaner]")
            ax.set_title(f"CIFAR-10 VGG16 {noise_position} (SNR-matched)")
        elif scale_mode == "absolute":
            ax.set_xlabel(r"Absolute $\sigma$")
            ax.set_title(f"CIFAR-10 VGG16 {noise_position} (absolute)")
        elif scale_mode == "lambda_relative":
            ax.set_xlabel(r"$\alpha$  ($\sigma=\alpha\lambda_{\mathrm{input}}$)")
            ax.set_title(f"CIFAR-10 VGG16 {noise_position} (λ-relative)")
        else:
            ax.set_xlabel(r"$\alpha$  ($\sigma=\alpha\,\mathrm{RMS}(h)$)")
            ax.set_title(f"CIFAR-10 VGG16 {noise_position} (activation-relative)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = out_dir / f"curve_{scale_mode}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"[DONE] plot: {out}", flush=True)


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = get_torch_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = {name: DEFAULT_CKPTS[name] for name in args.methods}
    if args.ckpt:
        for label, path in args.ckpt:
            ckpts[label] = path

    _, test_loader = datapool(
        args.dataset,
        args.batch_size,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    raw_rows: list[dict] = []
    method_list = list(ckpts.items())
    for method_index, (method, path_str) in enumerate(method_list):
        path = _resolve_ckpt(path_str)
        if not path.exists():
            print(f"[SKIP] missing checkpoint for {method}: {path}", flush=True)
            continue
        print(f"[METHOD] {method} <- {path}", flush=True)
        model = _load_model(path, args, device)
        scales = _calibrate_scales(model, test_loader, device, args)
        print(
            f"  noise_position={args.noise_position}  "
            f"λ_input={scales['lambda_input']:.6g}  "
            f"RMS(h)={scales['rms_input']:.6g}",
            flush=True,
        )

        for scale_mode_index, scale_mode in enumerate(args.scale_modes):
            levels = _level_grid(scale_mode, args)
            for level_index, (level_name, level_value) in enumerate(levels):
                actual_sigma = _actual_sigma(scale_mode, level_value, scales)
                is_clean = actual_sigma <= 0.0
                repeats = 1 if is_clean else args.noise_repeats
                for noise_repeat in range(repeats):
                    eval_noise_seed = (
                        args.noise_seed
                        + args.seed * 10_000
                        + method_index * 1_000
                        + scale_mode_index * 200
                        + level_index * 20
                        + noise_repeat
                    )
                    seed_all(eval_noise_seed)
                    model.set_first_layer_input_noise_sigma(actual_sigma)
                    acc = val(model, test_loader, args.T, device, verbose=False)
                    level_value_str = (
                        "inf" if not math.isfinite(level_value) else f"{level_value:.9g}"
                    )
                    print(
                        f"  mode={scale_mode} level={level_name} "
                        f"actual_σ={actual_sigma:.6g} repeat={noise_repeat} "
                        f"acc={acc:.3f}",
                        flush=True,
                    )
                    raw_rows.append(
                        {
                            "dataset": args.dataset,
                            "arch": args.arch,
                            "method": method,
                            "seed": args.seed,
                            "L": args.L,
                            "T": args.T,
                            "mode": args.mode,
                            "noise_position": args.noise_position,
                            "noise_type": "gaussian",
                            "scale_mode": scale_mode,
                            "level_name": level_name,
                            "level_value": level_value_str,
                            "lambda_input": f"{scales['lambda_input']:.9g}",
                            "rms_input": f"{scales['rms_input']:.9g}",
                            "actual_sigma": f"{actual_sigma:.9g}",
                            "noise_repeat": noise_repeat,
                            "noise_seed": eval_noise_seed,
                            "accuracy": f"{acc:.6f}",
                            "checkpoint": str(path),
                        }
                    )
            model.set_first_layer_input_noise_sigma(0.0)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_rows, summary_rows = _aggregate(raw_rows)
    _write_csv(args.out_dir / "relative_snr_noise_raw.csv", raw_rows)
    _write_csv(args.out_dir / "relative_snr_noise_mean_std.csv", mean_rows)
    _write_csv(args.out_dir / "relative_snr_noise_summary.csv", summary_rows)
    print(f"[DONE] raw:     {args.out_dir / 'relative_snr_noise_raw.csv'}", flush=True)
    print(f"[DONE] mean:    {args.out_dir / 'relative_snr_noise_mean_std.csv'}", flush=True)
    print(f"[DONE] summary: {args.out_dir / 'relative_snr_noise_summary.csv'}", flush=True)

    if not args.no_plot:
        _plot(mean_rows, args.out_dir, args.noise_position)


if __name__ == "__main__":
    main()
