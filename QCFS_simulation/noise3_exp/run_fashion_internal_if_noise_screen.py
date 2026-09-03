from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Models import modelpool
from Preprocess import datapool
from utils import get_torch_device, seed_all, val


METHOD_SPECS = {
    "old_detach": {"rc_tag": "0p0001", "label": "MNE-L2"},
    "old_detach_current": {"rc_tag": "0p0001", "label": "MNE-L2"},
    "mne_scale_trainable": {
        "rc_tag": "0p0001",
        "label": "MNE-L2 (BN/IF trainable)",
    },
    "mne_global_wd_all": {
        "rc_tag": "0p0001",
        "label": "MNE-L2 + WD (all params)",
    },
    "l2_sp": {"rc_tag": "0p01", "label": "L2-SP"},
    "l2": {"rc_tag": "none", "label": "L2"},
    "wd_all_current": {"rc_tag": "none", "label": "WD (all params)"},
    "manual_l2_all": {"rc_tag": "0p00025", "label": "Manual L2 (all params)"},
    "manual_l2_w_bn": {"rc_tag": "0p00025", "label": "L2 (weights + BN)"},
    "manual_l2_w_if": {"rc_tag": "0p00025", "label": "L2 (weights + IF)"},
    "manual_l2_w_bn_if": {"rc_tag": "0p00025", "label": "L2 (weights + BN + IF)"},
    "manual_l2": {"rc_tag": "0p00025", "label": "Manual L2 (weights only)"},
    "wd_weight_only": {"rc_tag": "none", "label": "WD (weights only)"},
    "no_reg": {"rc_tag": "none", "label": "No Reg"},
    "no_reg_current": {"rc_tag": "none", "label": "No Reg"},
    "orthogonal": {"rc_tag": "0p0001", "label": "Orthogonal"},
    "threshold_l2": {"rc_tag": "0p0001", "label": "Threshold-L2"},
    "l1_all": {"rc_tag": "1em05", "label": "L1 (all params)"},
    "elastic_net_all": {"rc_tag": "0p00025", "label": "Elastic Net (all params)"},
    "scale_l2": {"rc_tag": "0p00025", "label": "Scale-L2 diagnostic"},
}

SITE_MODULES = {
    "post_input_if": ("input_if",),
    "post_if1": ("if1",),
    "both": ("input_if", "if1"),
}


def _checkpoint_path(args, method: str, seed: int) -> Path:
    rc_tag = METHOD_SPECS[method]["rc_tag"]
    dataset_tag = "fashion" if args.dataset == "fashion_mnist" else args.dataset
    suffix = (
        f"{dataset_tag}_spectral_mne_{args.model}_{method}_rc{rc_tag}_seed{seed}"
        f"_ep{args.epochs}_L{args.L}_trainT0"
    )
    return (
        ROOT
        / f"{args.dataset}-checkpoints"
        / f"{args.model}_L[{args.L}]_{suffix}.pth"
    )


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_model(args, method: str, seed: int, device: torch.device):
    checkpoint = _checkpoint_path(args, method, seed)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    model = modelpool(args.model, args.dataset)
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.set_T(args.T)
    model.set_L(args.L)
    model.set_mode(args.if_mode)
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule(args.spike_schedule)
    if hasattr(model, "set_first_layer_input_noise_sigma"):
        model.set_first_layer_input_noise_sigma(0.0)
    model.eval()
    return model, checkpoint


def _register_noise_hooks(model, site: str, sigma_ref: dict[str, float]):
    handles = []

    def add_noise(_module, _inputs, output):
        sigma = sigma_ref["value"]
        if sigma <= 0:
            return output
        if not torch.is_tensor(output):
            raise TypeError("IF noise hook expected a tensor output")
        return output + sigma * torch.randn_like(output)

    for module_name in SITE_MODULES[site]:
        module = getattr(model, module_name, None)
        if module is None:
            raise ValueError(f"Model {type(model).__name__} has no module {module_name!r}")
        handles.append(module.register_forward_hook(add_noise))
    return handles


def _estimate_post_input_if_rms(model, loader, device, max_batches: int) -> float:
    total_sq = 0.0
    total_count = 0

    def capture(_module, _inputs, output):
        nonlocal total_sq, total_count
        detached = output.detach().float()
        total_sq += float(detached.pow(2).sum().item())
        total_count += detached.numel()

    handle = model.input_if.register_forward_hook(capture)
    try:
        model.set_first_layer_input_noise_sigma(0.0)
        with torch.no_grad():
            for batch_index, (images, _labels) in enumerate(loader):
                if batch_index >= max_batches:
                    break
                model(images.to(device).clone())
    finally:
        handle.remove()
    if total_count == 0:
        raise RuntimeError("Could not estimate post-input-IF RMS")
    return (total_sq / total_count) ** 0.5


def _noise_scale_reference(args, model, test_loader, device) -> float:
    if args.sigma_scale == "absolute":
        return 1.0
    if args.sigma_scale == "input_if_threshold":
        return float(model.input_if.thresh.detach().abs().clamp(min=1e-8).item())
    if args.sigma_scale == "post_input_if_rms":
        return _estimate_post_input_if_rms(
            model, test_loader, device, args.rms_calibration_batches
        )
    raise ValueError(f"Unsupported sigma scale: {args.sigma_scale}")


def _trapezoid_mean(points: list[tuple[float, float]]) -> float:
    points = sorted(points)
    if len(points) == 1:
        return points[0][1]
    width = points[-1][0] - points[0][0]
    if width <= 0:
        return points[0][1]
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area / width


def _aggregate(raw_rows: list[dict]):
    per_seed = defaultdict(list)
    for row in raw_rows:
        per_seed[
            (
                row["site"],
                row["method"],
                row["sigma_scale"],
                float(row["sigma"]),
                int(row["seed"]),
            )
        ].append(float(row["accuracy"]))

    repeat_rows = []
    for (site, method, sigma_scale, sigma, seed), values in sorted(per_seed.items()):
        repeat_rows.append(
            {
                "site": site,
                "method": method,
                "method_label": METHOD_SPECS[method]["label"],
                "sigma_scale": sigma_scale,
                "sigma": f"{sigma:.6g}",
                "seed": seed,
                "n_noise_repeats": len(values),
                "acc_mean": f"{statistics.mean(values):.6f}",
                "acc_std": (
                    f"{statistics.stdev(values):.6f}"
                    if len(values) > 1
                    else "0.000000"
                ),
            }
        )

    grouped = defaultdict(list)
    for (site, method, sigma_scale, sigma, _seed), values in per_seed.items():
        grouped[(site, method, sigma_scale, sigma)].append(statistics.mean(values))

    mean_rows = []
    for (site, method, sigma_scale, sigma), values in sorted(grouped.items()):
        mean_rows.append(
            {
                "site": site,
                "method": method,
                "method_label": METHOD_SPECS[method]["label"],
                "sigma_scale": sigma_scale,
                "sigma": f"{sigma:.6g}",
                "n": len(values),
                "acc_mean": f"{statistics.mean(values):.6f}",
                "acc_std": (
                    f"{statistics.stdev(values):.6f}"
                    if len(values) > 1
                    else "0.000000"
                ),
            }
        )

    curves = defaultdict(list)
    for row in mean_rows:
        curves[(row["site"], row["method"], row["method_label"], row["sigma_scale"])].append(
            (float(row["sigma"]), float(row["acc_mean"]))
        )

    summary_rows = []
    for (site, method, label, sigma_scale), points in sorted(curves.items()):
        points = sorted(points)
        clean = points[0][1]
        end = points[-1][1]
        auc_mean = _trapezoid_mean(points)
        summary_rows.append(
            {
                "site": site,
                "method": method,
                "method_label": label,
                "sigma_scale": sigma_scale,
                "clean_acc": f"{clean:.6f}",
                "end_sigma": f"{points[-1][0]:.6g}",
                "end_acc": f"{end:.6f}",
                "absolute_drop": f"{clean - end:.6f}",
                "end_retention": f"{end / clean:.6f}" if clean > 0 else "nan",
                "curve_acc_auc": f"{auc_mean:.6f}",
                "curve_retention_auc": (
                    f"{auc_mean / clean:.6f}" if clean > 0 else "nan"
                ),
            }
        )
    return repeat_rows, mean_rows, summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation-only layerwise IF-noise screen for MNIST-family CNN2."
    )
    parser.add_argument(
        "--dataset",
        default="fashion_mnist",
        choices=["fashion_mnist", "mnist"],
    )
    parser.add_argument("--model", default="cnn2_c8_c16")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["old_detach", "l2_sp", "l2"],
        choices=sorted(METHOD_SPECS),
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        default=["post_input_if", "post_if1", "both"],
        choices=sorted(SITE_MODULES),
    )
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[40])
    parser.add_argument("--noise-seed", type=int, default=20260801)
    parser.add_argument("--noise-repeats", type=int, default=1)
    parser.add_argument(
        "--sigma-scale",
        choices=["absolute", "input_if_threshold", "post_input_if_rms"],
        default="absolute",
        help="Interpret --sigmas as absolute values or multipliers of a model scale.",
    )
    parser.add_argument("--rms-calibration-batches", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--if-mode", default="rate_uniform")
    parser.add_argument("--spike-schedule", default="normal")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT.parent / "important_results" / "fashion_internal_if_noise_screen",
    )
    args = parser.parse_args()
    args.sigmas = sorted(set(args.sigmas))
    if not args.sigmas or args.sigmas[0] != 0:
        raise ValueError("--sigmas must include 0 for the clean reference")
    if any(sigma < 0 for sigma in args.sigmas):
        raise ValueError("--sigmas must be non-negative")
    if args.noise_repeats < 1:
        raise ValueError("--noise-repeats must be at least 1")
    if args.rms_calibration_batches < 1:
        raise ValueError("--rms-calibration-batches must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    device = get_torch_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _, test_loader = datapool(
        args.dataset,
        args.batch_size,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    raw_rows = []
    for seed in args.seeds:
        for method in args.methods:
            seed_all(seed)
            model, checkpoint = _load_model(args, method, seed, device)
            scale_reference = _noise_scale_reference(
                args, model, test_loader, device
            )
            print(
                f"method={method} seed={seed} sigma_scale={args.sigma_scale} "
                f"reference={scale_reference:.6g}",
                flush=True,
            )
            for site in args.sites:
                sigma_ref = {"value": 0.0}
                handles = _register_noise_hooks(model, site, sigma_ref)
                try:
                    for sigma_index, sigma in enumerate(args.sigmas):
                        repeats = 1 if sigma == 0 else args.noise_repeats
                        for noise_repeat in range(repeats):
                            # Paired random streams make method differences less noisy.
                            eval_noise_seed = (
                                args.noise_seed
                                + seed * 1000
                                + sigma_index * 100
                                + noise_repeat
                            )
                            seed_all(eval_noise_seed)
                            actual_sigma = sigma * scale_reference
                            sigma_ref["value"] = actual_sigma
                            accuracy = val(
                                model, test_loader, args.T, device, verbose=False
                            )
                            print(
                                f"site={site} method={method} seed={seed} "
                                f"T={args.T} sigma={sigma:.4g} "
                                f"actual_sigma={actual_sigma:.4g} "
                                f"repeat={noise_repeat} acc={accuracy:.3f}",
                                flush=True,
                            )
                            raw_rows.append(
                                {
                                    "dataset": args.dataset,
                                    "model": args.model,
                                    "method": method,
                                    "method_label": METHOD_SPECS[method]["label"],
                                    "seed": seed,
                                    "L": args.L,
                                    "T": args.T,
                                    "site": site,
                                    "sigma": f"{sigma:.6g}",
                                    "sigma_scale": args.sigma_scale,
                                    "scale_reference": f"{scale_reference:.9g}",
                                    "actual_sigma": f"{actual_sigma:.9g}",
                                    "noise_repeat": noise_repeat,
                                    "noise_seed": eval_noise_seed,
                                    "accuracy": f"{accuracy:.6f}",
                                    "checkpoint": str(checkpoint),
                                }
                            )
                finally:
                    for handle in handles:
                        handle.remove()

    repeat_rows, mean_rows, summary_rows = _aggregate(raw_rows)
    _write_csv(
        args.out_dir / "internal_if_noise_raw.csv",
        raw_rows,
        [
            "dataset",
            "model",
            "method",
            "method_label",
            "seed",
            "L",
            "T",
            "site",
            "sigma",
            "sigma_scale",
            "scale_reference",
            "actual_sigma",
            "noise_repeat",
            "noise_seed",
            "accuracy",
            "checkpoint",
        ],
    )
    _write_csv(
        args.out_dir / "internal_if_noise_repeat_mean_std.csv",
        repeat_rows,
        [
            "site",
            "method",
            "method_label",
            "sigma_scale",
            "sigma",
            "seed",
            "n_noise_repeats",
            "acc_mean",
            "acc_std",
        ],
    )
    _write_csv(
        args.out_dir / "internal_if_noise_mean_std.csv",
        mean_rows,
        ["site", "method", "method_label", "sigma_scale", "sigma", "n", "acc_mean", "acc_std"],
    )
    _write_csv(
        args.out_dir / "internal_if_noise_summary.csv",
        summary_rows,
        [
            "site",
            "method",
            "method_label",
            "sigma_scale",
            "clean_acc",
            "end_sigma",
            "end_acc",
            "absolute_drop",
            "end_retention",
            "curve_acc_auc",
            "curve_retention_auc",
        ],
    )
    print(f"[DONE] {args.out_dir / 'internal_if_noise_summary.csv'}")


if __name__ == "__main__":
    main()
