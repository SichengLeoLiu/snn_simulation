#!/usr/bin/env python3
"""Summarize BN gamma distributions from checkpoints (per method / per layer)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


DEFAULT_CKPTS = {
    "L1 (all params)": "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_l1_all_rc1em05_seed42_L16_trainT0.pth",
    "MNE-L2 (all params)": "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_mne_l2_all_rc0p0001_seed42_L16_trainT0.pth",
    "L2 (all params)": "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_manual_l2_all_rc0p00025_seed42_L16_trainT0.pth",
}


def _load_state_dict(path: Path) -> dict:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported checkpoint format: {path}")


def _extract_bn_gammas(state_dict: dict) -> list[tuple[str, torch.Tensor]]:
    layers = []
    for key, value in state_dict.items():
        if not key.endswith(".weight"):
            continue
        prefix = key[: -len("weight")]
        if prefix + "running_var" not in state_dict:
            continue
        layers.append((key, value.detach().float().reshape(-1)))
    return layers


def _stats(values: torch.Tensor) -> dict:
    abs_g = values.abs()
    active = abs_g[abs_g > 0.1]
    return {
        "n": int(abs_g.numel()),
        "median_abs": float(abs_g.median().item()),
        "rms": float(values.pow(2).mean().sqrt().item()),
        "p90_abs": float(torch.quantile(abs_g, 0.90).item()),
        "max_abs": float(abs_g.max().item()),
        "frac_lt_0p001": float((abs_g < 0.001).float().mean().item()),
        "frac_lt_0p01": float((abs_g < 0.01).float().mean().item()),
        "frac_lt_0p1": float((abs_g < 0.1).float().mean().item()),
        "active_mean_abs_gt_0p1": (
            float(active.mean().item()) if active.numel() > 0 else float("nan")
        ),
        "n_active_gt_0p1": int(active.numel()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("../important_results/bn_gamma_stats_all_params_seed42"),
    )
    parser.add_argument(
        "--ckpt",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        help="Optional override: --ckpt 'Label' path/to.pth (repeatable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpts = dict(DEFAULT_CKPTS)
    if args.ckpt:
        ckpts = {label: path for label, path in args.ckpt}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    layer_rows = []
    method_rows = []

    print(
        f"{'method':22s} {'layer':28s} n  median|γ|     RMS     p90|γ|   max|γ|  "
        f"<1e-3  <1e-2  <1e-1  active_mean"
    )
    for method, path_str in ckpts.items():
        path = Path(path_str)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)

        layers = _extract_bn_gammas(_load_state_dict(path))
        if not layers:
            raise RuntimeError(f"No BN gamma found in {path}")

        all_gamma = torch.cat([tensor for _, tensor in layers])
        method_stats = _stats(all_gamma)
        method_rows.append({"method": method, "layer": "ALL", **method_stats, "checkpoint": str(path)})
        print(
            f"{method:22s} {'ALL':28s} {method_stats['n']:4d} "
            f"{method_stats['median_abs']:9.6f} {method_stats['rms']:8.6f} "
            f"{method_stats['p90_abs']:9.6f} {method_stats['max_abs']:8.6f} "
            f"{method_stats['frac_lt_0p001']:6.3f} {method_stats['frac_lt_0p01']:6.3f} "
            f"{method_stats['frac_lt_0p1']:6.3f} {method_stats['active_mean_abs_gt_0p1']:11.6f}"
        )

        for index, (key, tensor) in enumerate(layers):
            stats = _stats(tensor)
            layer_rows.append(
                {
                    "method": method,
                    "layer_index": index,
                    "layer": key,
                    **stats,
                    "checkpoint": str(path),
                }
            )
            print(
                f"{method:22s} {key:28s} {stats['n']:4d} "
                f"{stats['median_abs']:9.6f} {stats['rms']:8.6f} "
                f"{stats['p90_abs']:9.6f} {stats['max_abs']:8.6f} "
                f"{stats['frac_lt_0p001']:6.3f} {stats['frac_lt_0p01']:6.3f} "
                f"{stats['frac_lt_0p1']:6.3f} {stats['active_mean_abs_gt_0p1']:11.6f}"
            )

    fieldnames = [
        "method",
        "layer_index",
        "layer",
        "n",
        "median_abs",
        "rms",
        "p90_abs",
        "max_abs",
        "frac_lt_0p001",
        "frac_lt_0p01",
        "frac_lt_0p1",
        "active_mean_abs_gt_0p1",
        "n_active_gt_0p1",
        "checkpoint",
    ]
    layer_csv = args.out_dir / "bn_gamma_stats_by_layer.csv"
    method_csv = args.out_dir / "bn_gamma_stats_by_method.csv"
    with layer_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(layer_rows)
    with method_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[name for name in fieldnames if name != "layer_index"],
        )
        writer.writeheader()
        writer.writerows(method_rows)

    print(f"[DONE] method summary: {method_csv}")
    print(f"[DONE] layer summary:  {layer_csv}")


if __name__ == "__main__":
    main()
