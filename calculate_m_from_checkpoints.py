from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch


def extract_state_dict(obj: object) -> Dict[str, torch.Tensor]:
    """Normalize checkpoint payloads into a state_dict-like mapping."""
    if isinstance(obj, OrderedDict):
        return obj
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], (dict, OrderedDict)):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], (dict, OrderedDict)):
            return obj["model_state_dict"]
        return obj
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    raise TypeError(f"Unsupported checkpoint payload type: {type(obj)}")


def iter_weight_layers(state_dict: Dict[str, torch.Tensor]) -> Iterable[Tuple[str, torch.Tensor]]:
    """
    Yield only conv/fc weights:
    - Conv: 4D [C_out, C_in, k, k]
    - FC:   2D [C_out, C_in]
    """
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if not name.endswith("weight"):
            continue
        if tensor.ndim not in (2, 4):
            continue
        yield name, tensor.detach().float().cpu()


def compute_m_metrics(state_dict: Dict[str, torch.Tensor]) -> dict:
    layer_stats: List[dict] = []

    total_sq_sum = 0.0
    total_numel = 0

    for layer_name, w in iter_weight_layers(state_dict):
        sq = w.pow(2)
        layer_mean = sq.mean().item()

        if sq.ndim == 4:
            per_out_mean = sq.view(sq.shape[0], -1).mean(dim=1)
            strict_upper = per_out_mean.max().item()
            layer_type = "conv"
        else:
            per_out_mean = sq.mean(dim=1)
            strict_upper = per_out_mean.max().item()
            layer_type = "fc"

        layer_stats.append(
            {
                "layer": layer_name,
                "type": layer_type,
                "shape": tuple(w.shape),
                "m_layer_mean": layer_mean,
                "m_layer_strict": strict_upper,
                "numel": int(w.numel()),
            }
        )

        total_sq_sum += sq.sum().item()
        total_numel += int(w.numel())

    if not layer_stats:
        raise ValueError("No conv/fc weight tensors found in checkpoint.")

    m_global_mean = total_sq_sum / total_numel
    m_global_strict = max(x["m_layer_strict"] for x in layer_stats)

    return {
        "m_global_mean": m_global_mean,
        "m_global_strict": m_global_strict,
        "layers_used": len(layer_stats),
        "weights_used": total_numel,
        "layer_stats": layer_stats,
    }


def collect_checkpoints(root: Path, target_dirs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for d in target_dirs:
        full_dir = root / d
        if not full_dir.exists():
            print(f"[WARN] Directory not found, skipped: {full_dir}")
            continue
        files = sorted(full_dir.glob("*.pth"))
        paths.extend(files)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate M from trained checkpoints.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root directory that contains the checkpoint folders.",
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["mnist-checkpoints", "mnist-checkpoints_c4_c8", "mnist-checkpoints_c16_c32"],
        help="Checkpoint directories relative to --root.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("m_values_mnist_checkpoints.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-layer",
        type=Path,
        default=None,
        help="Optional per-layer CSV path. Defaults to '<out>_layers.csv'.",
    )
    args = parser.parse_args()

    checkpoints = collect_checkpoints(args.root, args.dirs)
    if not checkpoints:
        raise FileNotFoundError("No checkpoint (.pth) files found in target directories.")

    rows = []
    layer_rows = []
    for ckpt in checkpoints:
        try:
            payload = torch.load(ckpt, map_location="cpu")
            state = extract_state_dict(payload)
            metrics = compute_m_metrics(state)
            rows.append(
                {
                    "checkpoint": str(ckpt.relative_to(args.root)),
                    "m_global_mean": metrics["m_global_mean"],
                    "m_global_strict": metrics["m_global_strict"],
                    "layers_used": metrics["layers_used"],
                    "weights_used": metrics["weights_used"],
                }
            )
            for layer in metrics["layer_stats"]:
                layer_rows.append(
                    {
                        "checkpoint": str(ckpt.relative_to(args.root)),
                        "layer": layer["layer"],
                        "type": layer["type"],
                        "shape": str(layer["shape"]),
                        "m_layer_mean": layer["m_layer_mean"],
                        "m_layer_strict": layer["m_layer_strict"],
                        "numel": layer["numel"],
                    }
                )
            print(
                f"[OK] {ckpt.name}: "
                f"M_mean={metrics['m_global_mean']:.6e}, "
                f"M_strict={metrics['m_global_strict']:.6e}, "
                f"layers={metrics['layers_used']}"
            )
        except Exception as exc:  # Keep batch processing robust
            print(f"[ERR] {ckpt}: {exc}")

    rows.sort(key=lambda x: x["checkpoint"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["checkpoint", "m_global_mean", "m_global_strict", "layers_used", "weights_used"],
        )
        writer.writeheader()
        writer.writerows(rows)

    out_layer = args.out_layer
    if out_layer is None:
        out_layer = args.out.with_name(f"{args.out.stem}_layers.csv")
    out_layer.parent.mkdir(parents=True, exist_ok=True)
    layer_rows.sort(key=lambda x: (x["checkpoint"], x["layer"]))
    with out_layer.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["checkpoint", "layer", "type", "shape", "m_layer_mean", "m_layer_strict", "numel"],
        )
        writer.writeheader()
        writer.writerows(layer_rows)

    print(f"\nSaved {len(rows)} rows to: {args.out}")
    print(f"Saved {len(layer_rows)} rows to: {out_layer}")


if __name__ == "__main__":
    main()
