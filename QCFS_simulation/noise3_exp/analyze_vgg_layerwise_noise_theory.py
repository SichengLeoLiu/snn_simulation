#!/usr/bin/env python3
"""
Per-layer VGG16 diagnostics for post-IF noise theory:

  λ_l, RMS(γ_l)/λ_l, BN-folded Frobenius/spectral gains,
  clean activation RMS, σ_eff,l, d_l percentiles, ρ_l=d_l/σ_eff,
  and empirical/Gaussian P(E_l).

Default evaluation matches the CIFAR noise-sweep setting:
ANN-trained checkpoint tested as SNN with T=L, mode=rate_uniform,
noise injected at post_input_if.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from Models import modelpool
from Models.VGG import remap_legacy_vgg_state_dict
from Models.layer import IF
from Preprocess import datapool
from utils import (
    _approx_largest_singular_value,
    _resolve_bn_if_for_layer,
    get_torch_device,
    seed_all,
)


DEFAULT_CKPTS = {
    "L1-all": "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_l1_all_rc1em05_seed42_L16_trainT0.pth",
    "MNE-all": "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_mne_l2_all_rc0p0001_seed42_L16_trainT0.pth",
    "L2-all": "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_manual_l2_all_rc0p00025_seed42_L16_trainT0.pth",
    "Weights-only": "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_weight_decay_weights_only_rcnone_seed42_L16_trainT0.pth",
    "Weights+BN-gamma": "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_manual_l2_w_bn_gamma_rc0p00025_seed42_L16_trainT0.pth",
    "MNE-standard": "cifar10-checkpoints/vgg16_L[16]_mneablate_cifar10_old_detach_rc0p0001_seed42_L16_trainT0.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--arch", default="vgg16")
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--mode", default="rate_uniform", choices=["rate_uniform", "normal"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--noise-sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--power-iters", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("../important_results/vgg_layerwise_noise_theory_seed42"),
    )
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
        help="Optional extra/override checkpoint: --ckpt Label path.pth",
    )
    return parser.parse_args()


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
    if hasattr(model, "set_first_layer_input_noise_type"):
        model.set_first_layer_input_noise_type("gaussian")
    if hasattr(model, "set_first_layer_input_noise_position"):
        model.set_first_layer_input_noise_position("post_input_if")
    return model


def _enumerate_if_layers(model) -> list[tuple[str, IF]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, IF)]


def _matched_weight_layers(model) -> list[dict]:
    module_map = dict(model.named_modules())
    rows = []
    for lname, layer in model.named_modules():
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            continue
        if getattr(layer, "weight", None) is None:
            continue
        bn_mod, if_mod = _resolve_bn_if_for_layer(lname, module_map)
        if if_mod is None:
            continue
        if_name = None
        for name, module in module_map.items():
            if module is if_mod:
                if_name = name
                break
        bn_name = None
        for name, module in module_map.items():
            if module is bn_mod:
                bn_name = name
                break
        rows.append(
            {
                "weight_name": lname,
                "bn_name": bn_name,
                "if_name": if_name,
                "layer": layer,
                "bn": bn_mod,
                "if_mod": if_mod,
            }
        )
    return rows


def _bn_fold_weight(weight: torch.Tensor, bn_mod, eps: float = 1e-6) -> torch.Tensor:
    if bn_mod is None:
        return weight
    bn_eps = float(getattr(bn_mod, "eps", eps))
    gamma = bn_mod.weight.detach().to(device=weight.device, dtype=weight.dtype)
    var = bn_mod.running_var.detach().to(device=weight.device, dtype=weight.dtype).clamp(min=bn_eps)
    scale = gamma / torch.sqrt(var + bn_eps)
    view_shape = [scale.shape[0]] + [1] * (weight.dim() - 1)
    return weight * scale.view(*view_shape)


def _weight_metrics(spec: dict, L: int, power_iters: int) -> dict:
    w = spec["layer"].weight.detach()
    w_eff = _bn_fold_weight(w, spec["bn"])
    w_flat = w_eff.reshape(w_eff.shape[0], -1)
    m_eff = (w_flat * w_flat).sum(dim=1).mean()
    fro_norm = torch.linalg.vector_norm(w_flat)
    # Mean per-out Frobenius energy gain used by hinge-MNE:
    #   G_fro = L * sqrt(M_eff) / λ
    # Spectral gain:
    #   G_spec = L * σ_max(W_eff) / λ
    sigma_max = _approx_largest_singular_value(w_flat, power_iters=power_iters)
    lam = float(spec["if_mod"].thresh.detach().float().clamp(min=1e-6).view(-1)[0].item())
    if spec["bn"] is not None and spec["bn"].weight is not None:
        gamma = spec["bn"].weight.detach().float()
        rms_gamma = float(gamma.pow(2).mean().sqrt().item())
        mean_abs_gamma = float(gamma.abs().mean().item())
    else:
        rms_gamma = float("nan")
        mean_abs_gamma = float("nan")
    return {
        "lambda": lam,
        "rms_gamma": rms_gamma,
        "mean_abs_gamma": mean_abs_gamma,
        "rms_gamma_over_lambda": (rms_gamma / lam) if math.isfinite(rms_gamma) else float("nan"),
        "m_eff": float(m_eff.item()),
        "frobenius_norm_folded": float(fro_norm.item()),
        "frobenius_gain": float(L) * math.sqrt(float(m_eff.item()) + 1e-12) / lam,
        "spectral_sigma_max": float(sigma_max.item()),
        "spectral_gain": float(L) * float(sigma_max.item()) / lam,
        "is_input_if": spec["if_name"] == "layer1.2",
    }


def _boundary_margin_ratio(ratio: torch.Tensor, time_steps: int) -> torch.Tensor:
    # Decision boundaries of round(ratio) are at k + 0.5.
    nearest = torch.round(ratio - 0.5) + 0.5
    nearest = nearest.clamp(min=0.5, max=float(time_steps) - 0.5)
    return (ratio - nearest).abs()


class LayerActivationProbe:
    """Capture per-IF summed pre-activations z=sum_t x_t under clean/noisy passes."""

    def __init__(self, model, time_steps: int):
        self.model = model
        self.T = int(time_steps)
        self.batch_size = 0
        self.handles = []
        self.z_maps: dict[str, torch.Tensor] = {}
        self.post_input_if_z: torch.Tensor | None = None
        for name, module in _enumerate_if_layers(model):
            self.handles.append(module.register_forward_pre_hook(self._make_pre_hook(name)))
        # For post_input_if, noise is added after layer1.2 and before layer1.3 (Dropout).
        # Hooking layer1.3 input therefore sees the actual injected noise.
        if "layer1.3" in dict(model.named_modules()):
            self.handles.append(
                dict(model.named_modules())["layer1.3"].register_forward_pre_hook(
                    self._make_injection_site_hook()
                )
            )

    def _sum_over_time(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().float()
        if self.T > 0:
            x = x.view(self.T, self.batch_size, *x.shape[1:])
            return x.sum(dim=0)
        return x

    def _make_pre_hook(self, name: str):
        def hook(_module, inputs):
            self.z_maps[name] = self._sum_over_time(inputs[0]).cpu()

        return hook

    def _make_injection_site_hook(self):
        def hook(_module, inputs):
            self.post_input_if_z = self._sum_over_time(inputs[0]).cpu()

        return hook

    def reset(self, batch_size: int) -> None:
        self.batch_size = int(batch_size)
        self.z_maps = {}
        self.post_input_if_z = None

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _percentile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(torch.quantile(values, q).item())


def _accumulate_activation_stats(
    model,
    loader,
    device,
    args,
) -> dict[str, dict]:
    probe = LayerActivationProbe(model, args.T)
    totals = defaultdict(
        lambda: {
            "clean_sq": 0.0,
            "diff_sq": 0.0,
            "n": 0,
            "d_values": [],
            "event_hits": 0,
            "event_total": 0,
        }
    )
    inject = {"clean_sq": 0.0, "diff_sq": 0.0, "n": 0}
    try:
        for batch_index, (images, _) in enumerate(loader):
            if batch_index >= args.max_batches:
                break
            images = images.to(device)
            keep = images.shape[0]

            probe.reset(keep)
            model.set_first_layer_input_noise_sigma(0.0)
            with torch.no_grad():
                model(images.clone())
            clean = {k: v.clone() for k, v in probe.z_maps.items()}
            clean_inject = None if probe.post_input_if_z is None else probe.post_input_if_z.clone()

            seed_all(args.seed + 10007 + batch_index)
            probe.reset(keep)
            model.set_first_layer_input_noise_sigma(args.noise_sigma)
            with torch.no_grad():
                model(images.clone())
            noisy = probe.z_maps
            noisy_inject = probe.post_input_if_z
            model.set_first_layer_input_noise_sigma(0.0)

            if clean_inject is not None and noisy_inject is not None:
                # For post_input_if, noise is added on the first IF *output*.
                inject["clean_sq"] += float(clean_inject.float().pow(2).sum().item())
                inject["diff_sq"] += float((noisy_inject - clean_inject).float().pow(2).sum().item())
                inject["n"] += int(clean_inject.numel())

            for name, z_clean in clean.items():
                z_noisy = noisy[name]
                module = dict(model.named_modules())[name]
                lam = float(module.thresh.detach().float().clamp(min=1e-6).view(-1)[0].item())
                ratio = z_clean / lam
                if args.T > 0:
                    d_ratio = _boundary_margin_ratio(ratio, args.T)
                    n_clean = torch.round(ratio).long().clamp(0, args.T)
                    n_noisy = torch.round(z_noisy / lam).long().clamp(0, args.T)
                else:
                    # T=0 QCFS: boundaries of round(L*clamp(u/λ)) / L.
                    q = ratio.clamp(0, 1)
                    nearest = torch.round(q * args.L - 0.5) + 0.5
                    nearest = nearest.clamp(min=0.5, max=float(args.L) - 0.5)
                    d_ratio = (q * args.L - nearest).abs() / float(args.L)
                    n_clean = torch.round(q * args.L).long().clamp(0, args.L)
                    n_noisy = torch.round((z_noisy / lam).clamp(0, 1) * args.L).long().clamp(0, args.L)

                d_physical = d_ratio * lam
                diff = (z_noisy - z_clean).float()
                bucket = totals[name]
                bucket["clean_sq"] += float(z_clean.float().pow(2).sum().item())
                bucket["diff_sq"] += float(diff.pow(2).sum().item())
                bucket["n"] += int(z_clean.numel())
                bucket["d_values"].append(d_physical.reshape(-1).cpu())
                bucket["event_hits"] += int(n_clean.ne(n_noisy).sum().item())
                bucket["event_total"] += int(n_clean.numel())
    finally:
        probe.close()
        model.set_first_layer_input_noise_sigma(0.0)

    out = {}
    inject_sigma = math.sqrt(inject["diff_sq"] / max(inject["n"], 1)) if inject["n"] else float("nan")
    inject_rms = math.sqrt(inject["clean_sq"] / max(inject["n"], 1)) if inject["n"] else float("nan")
    for name, bucket in totals.items():
        d_all = torch.cat(bucket["d_values"]) if bucket["d_values"] else torch.tensor([])
        sigma_eff = math.sqrt(bucket["diff_sq"] / max(bucket["n"], 1))
        clean_rms = math.sqrt(bucket["clean_sq"] / max(bucket["n"], 1))
        d_p01 = _percentile(d_all, 0.01)
        d_p05 = _percentile(d_all, 0.05)
        d_p50 = _percentile(d_all, 0.50)
        # Gaussian approximation using median margin:
        # P(|N(0,σ)| > d) = erfc(d / (σ√2))
        if sigma_eff > 0 and math.isfinite(d_p50):
            p_e_gauss = math.erfc(d_p50 / (sigma_eff * math.sqrt(2.0)))
        else:
            p_e_gauss = float("nan")
        rho = (d_p50 / sigma_eff) if sigma_eff > 0 else float("nan")
        out[name] = {
            "clean_activation_rms": clean_rms,
            "sigma_eff": sigma_eff,
            "d_p01": d_p01,
            "d_p05": d_p05,
            "d_p50": d_p50,
            "rho_median": rho,
            "p_e_empirical": (
                bucket["event_hits"] / bucket["event_total"]
                if bucket["event_total"]
                else float("nan")
            ),
            "p_e_gaussian": p_e_gauss,
            "n_units": bucket["n"],
            "injection_site_sigma_eff": inject_sigma,
            "injection_site_clean_rms": inject_rms,
        }
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_one(method: str, ckpt: Path, loader, args, device) -> list[dict]:
    print(f"[METHOD] {method} <- {ckpt}", flush=True)
    model = _load_model(ckpt, args, device)
    weight_rows = _matched_weight_layers(model)
    weight_by_if = {row["if_name"]: row for row in weight_rows}
    act_stats = _accumulate_activation_stats(model, loader, device, args)

    rows = []
    # Ensure input IF appears first when present.
    if_names = [name for name, _ in _enumerate_if_layers(model)]
    for layer_index, if_name in enumerate(if_names):
        row = {
            "method": method,
            "layer_index": layer_index,
            "if_name": if_name,
            "is_input_if": if_name == "layer1.2",
            "checkpoint": str(ckpt),
            "L": args.L,
            "T": args.T,
            "mode": args.mode,
            "noise_sigma": args.noise_sigma,
            "max_batches": args.max_batches,
        }
        if if_name in weight_by_if:
            row.update(_weight_metrics(weight_by_if[if_name], args.L, args.power_iters))
            row["weight_name"] = weight_by_if[if_name]["weight_name"]
            row["bn_name"] = weight_by_if[if_name]["bn_name"]
        else:
            module = dict(model.named_modules())[if_name]
            row.update(
                {
                    "lambda": float(module.thresh.detach().float().clamp(min=1e-6).view(-1)[0].item()),
                    "rms_gamma": float("nan"),
                    "mean_abs_gamma": float("nan"),
                    "rms_gamma_over_lambda": float("nan"),
                    "m_eff": float("nan"),
                    "frobenius_norm_folded": float("nan"),
                    "frobenius_gain": float("nan"),
                    "spectral_sigma_max": float("nan"),
                    "spectral_gain": float("nan"),
                    "weight_name": "",
                    "bn_name": "",
                }
            )
        row.update(act_stats.get(if_name, {}))
        rows.append(row)
        print(
            f"  [{layer_index:02d}] {if_name:16s} λ={row['lambda']:.4f} "
            f"RMS(γ)/λ={row.get('rms_gamma_over_lambda', float('nan')):.4f} "
            f"Gfro={row.get('frobenius_gain', float('nan')):.4f} "
            f"Gspec={row.get('spectral_gain', float('nan')):.4f} "
            f"actRMS={row.get('clean_activation_rms', float('nan')):.4f} "
            f"σeff={row.get('sigma_eff', float('nan')):.4f} "
            f"d50={row.get('d_p50', float('nan')):.4f} "
            f"ρ={row.get('rho_median', float('nan')):.4f} "
            f"P(E)={row.get('p_e_empirical', float('nan')):.4f}",
            flush=True,
        )
    return rows


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = get_torch_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = {name: DEFAULT_CKPTS[name] for name in args.methods}
    if args.ckpt:
        for label, path in args.ckpt:
            ckpts[label] = path

    _, test_loader = datapool(args.dataset, batch_size=args.batch_size, num_workers=args.workers)
    all_rows = []
    for method, path_str in ckpts.items():
        path = Path(path_str)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            print(f"[SKIP] missing checkpoint for {method}: {path}", flush=True)
            continue
        all_rows.extend(analyze_one(method, path, test_loader, args, device))

    layer_csv = args.out_dir / "layerwise_noise_theory.csv"
    _write_csv(layer_csv, all_rows)

    # Compact method-level summary focused on input IF + mean over layers.
    summary_rows = []
    by_method = defaultdict(list)
    for row in all_rows:
        by_method[row["method"]].append(row)
    for method, rows in by_method.items():
        input_rows = [row for row in rows if row.get("is_input_if")]
        input_row = input_rows[0] if input_rows else rows[0]

        def _mean(key):
            vals = [row[key] for row in rows if key in row and math.isfinite(float(row[key]))]
            return float(sum(vals) / len(vals)) if vals else float("nan")

        summary_rows.append(
            {
                "method": method,
                "input_lambda": input_row.get("lambda"),
                "input_rms_gamma_over_lambda": input_row.get("rms_gamma_over_lambda"),
                "input_frobenius_gain": input_row.get("frobenius_gain"),
                "input_spectral_gain": input_row.get("spectral_gain"),
                "input_clean_activation_rms": input_row.get("clean_activation_rms"),
                "input_sigma_eff": input_row.get("sigma_eff"),
                "input_d_p50": input_row.get("d_p50"),
                "input_rho_median": input_row.get("rho_median"),
                "input_p_e_empirical": input_row.get("p_e_empirical"),
                "mean_lambda": _mean("lambda"),
                "mean_rms_gamma_over_lambda": _mean("rms_gamma_over_lambda"),
                "mean_frobenius_gain": _mean("frobenius_gain"),
                "mean_spectral_gain": _mean("spectral_gain"),
                "mean_sigma_eff": _mean("sigma_eff"),
                "mean_d_p50": _mean("d_p50"),
                "mean_rho_median": _mean("rho_median"),
                "mean_p_e_empirical": _mean("p_e_empirical"),
            }
        )

    summary_csv = args.out_dir / "layerwise_noise_theory_summary.csv"
    _write_csv(summary_csv, summary_rows)
    print(f"[DONE] layer csv:   {layer_csv}", flush=True)
    print(f"[DONE] summary csv: {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
