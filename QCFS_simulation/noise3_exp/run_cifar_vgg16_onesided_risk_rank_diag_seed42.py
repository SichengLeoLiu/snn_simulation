#!/usr/bin/env python3
"""VGG-16 One-sided risk-ranking diagnostic (observational). Seed-42 pilot.

Does r / r||W||^2 / q actually rank channels that change under post-IF
noise and that affect the classifier? Freeze checkpoints. Do not retrain.
Do not retune α/τ. Do not use σ=5 as the main correlation setting.

Protocol
--------
    L=T=16, rate_uniform, post_input_if Gaussian
    2000 images from the 5k train-holdout val split
    K=5 noise draws, σ ∈ {0.25, 0.5, 1.0}
    Skip the input IF when counting crossings

Outputs: channel_stats.csv, layer_stats.csv, spearman_summary.json,
paired_metrics.json, channel_groups.json (top/mid/bot/shuffle 20% by r).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Models import modelpool  # noqa: E402
from Models.VGG import remap_legacy_vgg_state_dict  # noqa: E402
from Models.layer import IF  # noqa: E402
from analyze_vgg_layerwise_noise_theory import (  # noqa: E402
    LayerActivationProbe,
    _is_input_if_name,
)
from run_cifar_vgg16_onesided_q_assignment_ablation import (  # noqa: E402
    ALPHA,
    LVAL,
    RISK_MAX,
    RISK_MIN,
    TAU,
    TEST_T,
    VAL_SIZE,
    VAL_SPLIT_SEED,
    eval_dataset,
)
from utils import (  # noqa: E402
    _raw_channel_risk,
    collect_weight_layer_matches,
    get_torch_device,
    seed_all,
)

ARCH = "vgg16"
SEED = 42
N_IMAGES = 2000
N_NOISE = 5
SIGMAS = (0.25, 0.5, 1.0)
N_PERM = 200
N_BOOT = 500
EPS = 1e-6
METHODS = ("onesided", "mne", "l2wo")
ONESIDED_ROOT_DEFAULT = Path(
    "/scratch/gs14/sl9144/snn_results/cifar_vgg16_onesided_q_assignment_ablation_seed42"
)
COMP_ROOT_DEFAULT = Path(
    "/scratch/gs14/sl9144/snn_results/cifar_vgg16_mne_component_ablation_seed42"
)
FIVE_REGS_DEFAULT = Path("/scratch/gs14/sl9144/snn_results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="onesided")
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-images", type=int, default=N_IMAGES)
    parser.add_argument("--n-noise", type=int, default=N_NOISE)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "64")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--sigmas",
        default="0.25,0.5,1.0",
        help="Comma-separated post-IF σ values. Avoid σ=5 for ranking.",
    )
    parser.add_argument("--onesided-root", type=Path, default=ONESIDED_ROOT_DEFAULT)
    parser.add_argument("--comp-root", type=Path, default=COMP_ROOT_DEFAULT)
    parser.add_argument("--ckpt-root", type=Path, default=Path(os.environ.get("CKPT_ROOT", str(ROOT))))
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar_vgg16_onesided_risk_rank_seed42",
    )
    args = parser.parse_args()
    args.sigma_list = tuple(float(x) for x in str(args.sigmas).split(",") if x.strip())
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    return args


def _exists(path: Path) -> Path | None:
    return path if path.is_file() else None


def resolve_ckpt(args) -> Path:
    ds, seed = args.dataset, args.seed
    if args.method == "onesided":
        folder = "qabl_risk_a4_tau0.5_b0.0005_rmax8"
        name = f"{ARCH}_L[{LVAL}]_{folder}_seed{seed}_L{LVAL}_trainT0.pth"
        candidates = [
            args.onesided_root / ds / folder / "checkpoints" / name,
            ROOT.parent
            / "important_results"
            / "cifar_vgg16_onesided_q_assignment_ablation_seed42"
            / ds
            / folder
            / "checkpoints"
            / name,
            FIVE_REGS_DEFAULT
            / "cifar_ga_mne_seed42"
            / ARCH
            / ds
            / f"gadiag_{ARCH}_onesided"
            / "checkpoints"
            / f"{ARCH}_L[{LVAL}]_gadiag_{ARCH}_onesided_seed{seed}_L{LVAL}_trainT0.pth",
        ]
    else:
        variant = "mne" if args.method == "mne" else "l2wo"
        folder = f"comp_{variant}_fixed"
        name = f"{ARCH}_L[{LVAL}]_{folder}_seed{seed}_L{LVAL}_trainT0.pth"
        mneablate = (
            f"{ARCH}_L[{LVAL}]_mneablate_{ds}_"
            f"{'old_detach_rc0p0001' if args.method == 'mne' else 'weight_decay_weights_only_rcnone'}"
            f"_seed{seed}_L{LVAL}_trainT0.pth"
        )
        candidates = [
            args.comp_root / ds / folder / "checkpoints" / name,
            ROOT.parent
            / "important_results"
            / "cifar_vgg16_mne_component_ablation_seed42"
            / ds
            / folder
            / "checkpoints"
            / name,
            args.ckpt_root / f"{ds}-checkpoints" / mneablate,
            ROOT / f"{ds}-checkpoints" / mneablate,
        ]
    for path in candidates:
        hit = _exists(path)
        if hit is not None:
            return hit
    raise FileNotFoundError(f"missing {args.method} ckpt for {ds} seed={seed}")


def load_model(ckpt: Path, dataset: str, device):
    model = modelpool(ARCH, dataset)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(remap_legacy_vgg_state_dict(state), strict=True)
    model.set_L(LVAL)
    model.set_T(TEST_T)
    model.set_mode("rate_uniform")
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule("normal")
    model.set_first_layer_input_noise_position("post_input_if")
    model.set_first_layer_input_noise_type("gaussian")
    return model.to(device).eval()


def val_subset(dataset: str, n_images: int, batch_size: int, workers: int, pin: bool):
    train_ds = eval_dataset(dataset, train=True)
    g = torch.Generator().manual_seed(VAL_SPLIT_SEED)
    perm = torch.randperm(len(train_ds), generator=g).tolist()
    subset = Subset(train_ds, perm[: min(n_images, VAL_SIZE)])
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
    )


def channel_predictors(model) -> list[dict]:
    """Per matched Conv/Linear channel: r, r||W||^2, q. Input IF is flagged."""
    rows = collect_weight_layer_matches(model, layer_map="legacy")
    matched = [row for row in rows if row["matched"]]
    risks = [_raw_channel_risk(row, LVAL, eps=EPS, fold_bn=True) for row in matched]
    total = sum(int(row["weight"][0].numel()) * risk.numel() for row, risk in zip(matched, risks))
    global_mean = sum(
        int(row["weight"][0].numel()) * risk.sum() for row, risk in zip(matched, risks)
    ) / float(max(total, 1))
    global_mean = global_mean.clamp(min=EPS)
    clipped = [(risk / global_mean).clamp(min=RISK_MIN, max=RISK_MAX) for risk in risks]
    clip_mean = sum(
        int(row["weight"][0].numel()) * c.sum() for row, c in zip(matched, clipped)
    ) / float(max(total, 1))
    clip_mean = clip_mean.clamp(min=EPS)
    rhat = [c / clip_mean for c in clipped]
    q_vals = [1.0 + float(ALPHA) * torch.relu(normed - float(TAU)) for normed in rhat]
    out = []
    for row, risk, normed, q in zip(matched, risks, rhat, q_vals):
        weight = row["weight"].detach()
        w2 = weight.reshape(weight.shape[0], -1).pow(2).sum(dim=1)
        skip = _is_input_if_name(row["if_name"] or "")
        out.append(
            {
                "weight_name": row["name"],
                "if_name": row["if_name"],
                "role": row["role"],
                "skip_input_if": skip,
                "lambda": float(row["if_mod"].thresh.detach().clamp(min=1e-3).view(-1)[0]),
                "r": risk.detach().cpu().numpy().astype(np.float64),
                "rhat": normed.detach().cpu().numpy().astype(np.float64),
                "q": q.detach().cpu().numpy().astype(np.float64),
                "w2": w2.detach().cpu().numpy().astype(np.float64),
                "m": (risk.detach() * w2).cpu().numpy().astype(np.float64),
            }
        )
    return out


def _gauss_q(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.erfc(z / math.sqrt(2.0))


def _spike_count(z: torch.Tensor, lam: float, time_steps: int) -> torch.Tensor:
    return torch.round(z / lam).long().clamp(0, time_steps)


def _bound_prob(z: torch.Tensor, lam: float, sigma: torch.Tensor, time_steps: int) -> torch.Tensor:
    """Gaussian-approximation crossing prob. sigma broadcasts to z."""
    k = torch.round(z / lam).clamp(0, time_steps)
    d_plus = lam * (k + 0.5) - z
    d_minus = z - lam * (k - 0.5)
    sig = sigma.clamp(min=EPS)
    term = torch.zeros_like(z)
    up = k < time_steps
    down = k > 0
    term = term + torch.where(up, _gauss_q((d_plus / sig).clamp(-20, 20)), torch.zeros_like(z))
    term = term + torch.where(down, _gauss_q((d_minus / sig).clamp(-20, 20)), torch.zeros_like(z))
    return term.clamp(0.0, 1.0)


def _sum_over_spatial(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 4:
        return tensor.flatten(2).sum(dim=(0, 2))
    return tensor.sum(dim=0)


def _new_acc(channels: int) -> dict:
    return {
        "n": 0.0,
        "diff_sq": np.zeros(channels, dtype=np.float64),
        "cross": np.zeros(channels, dtype=np.float64),
        "p_bound": np.zeros(channels, dtype=np.float64),
        "d_energy": np.zeros(channels, dtype=np.float64),
        "d_n": 0.0,
    }


def _spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _mae(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(x[mask] - y[mask])))


def _bootstrap_ci(x, y, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 8:
        return float("nan"), float("nan")
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, x.size, size=x.size)
        vals.append(_spearman(x[idx], y[idx]))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _perm_null(x, y, n_perm: int, rng: np.random.Generator) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return float("nan")
    vals = [_spearman(rng.permutation(x), y) for _ in range(n_perm)]
    return float(np.nanmedian(vals))


def _group_rows(rows: list[dict]):
    by_layer = defaultdict(list)
    for row in rows:
        by_layer[row["if_name"]].append(row)
    return by_layer


def _layer_means(rows: list[dict], key: str) -> list[float]:
    by_layer = _group_rows(rows)
    out = []
    for _, group in sorted(by_layer.items()):
        vals = np.array([g[key] for g in group], dtype=float)
        if np.isfinite(vals).any():
            out.append(float(np.nanmean(vals)))
    return out


def _within_layer(rows: list[dict], pred: str, outcome: str) -> list[float]:
    rhos = []
    for _, group in _group_rows(rows).items():
        if len(group) < 8:
            continue
        rhos.append(_spearman([g[pred] for g in group], [g[outcome] for g in group]))
    return rhos


def _rank_norm(rows: list[dict], key: str) -> np.ndarray:
    by_layer = _group_rows(rows)
    out = np.full(len(rows), np.nan)
    index = {id(row): i for i, row in enumerate(rows)}
    for group in by_layer.values():
        vals = np.array([g[key] for g in group], dtype=float)
        order = np.argsort(np.argsort(vals))
        denom = max(len(group) - 1, 1)
        for item, rank in zip(group, order):
            out[index[id(item)]] = float(rank) / float(denom)
    return out


def _corr_bundle(rows: list[dict], pred: str, outcome: str, rng: np.random.Generator) -> dict:
    x = np.array([r[pred] for r in rows], dtype=float)
    y = np.array([r[outcome] for r in rows], dtype=float)
    layer_x = _layer_means(rows, pred)
    layer_y = _layer_means(rows, outcome)
    within = _within_layer(rows, pred, outcome)
    rx = _rank_norm(rows, pred)
    ry = _rank_norm(rows, outcome)
    lo, hi = _bootstrap_ci(x, y, N_BOOT, rng)
    return {
        "predictor": pred,
        "outcome": outcome,
        "global_rho": _spearman(x, y),
        "global_mae": _mae(x, y) if pred.startswith("p_bound") else float("nan"),
        "global_ci95": [lo, hi],
        "layer_rho": _spearman(layer_x, layer_y),
        "within_layer_rho_mean": float(np.nanmean(within)) if within else float("nan"),
        "within_layer_rho_median": float(np.nanmedian(within)) if within else float("nan"),
        "ranknorm_rho": _spearman(rx, ry),
        "perm_null_median": _perm_null(x, y, N_PERM, rng),
        "n_channels": int(np.isfinite(x).sum()),
        "n_layers": len(_group_rows(rows)),
    }


def classification_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    correct = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels.view(-1, 1), -1e9)
    return correct - masked.max(dim=1).values


def accumulate_sigma(
    model,
    loader,
    device,
    sigma: float,
    n_noise: int,
    seed: int,
    predictors: list[dict],
) -> tuple[dict[str, dict], dict]:
    probe = LayerActivationProbe(model, TEST_T)
    if_names = [
        p["if_name"]
        for p in predictors
        if p["if_name"] and not p["skip_input_if"]
    ]
    acc = {}
    paired = {
        "n": 0.0,
        "logit_sq": 0.0,
        "margin_drop": 0.0,
        "clean_correct": 0.0,
        "noisy_correct": 0.0,
    }
    modules = dict(model.named_modules())
    try:
        for batch_i, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            keep = images.shape[0]
            probe.reset(keep)
            model.set_first_layer_input_noise_sigma(0.0)
            with torch.no_grad():
                logits_clean = model(images.clone())
                if logits_clean.dim() == 3:
                    logits_clean = logits_clean.mean(0)
            clean_z = {k: v.detach() for k, v in probe.z_maps.items()}
            m_clean = classification_margin(logits_clean, labels)
            pred_clean = logits_clean.argmax(1)
            for draw in range(n_noise):
                seed_all(seed + 10007 + batch_i * 64 + draw)
                probe.reset(keep)
                model.set_first_layer_input_noise_sigma(sigma)
                with torch.no_grad():
                    logits_noisy = model(images.clone())
                    if logits_noisy.dim() == 3:
                        logits_noisy = logits_noisy.mean(0)
                noisy_z = {k: v.detach() for k, v in probe.z_maps.items()}
                model.set_first_layer_input_noise_sigma(0.0)
                paired["n"] += float(keep)
                paired["logit_sq"] += float((logits_noisy - logits_clean).pow(2).sum().item())
                paired["margin_drop"] += float((m_clean - classification_margin(logits_noisy, labels)).sum().item())
                paired["clean_correct"] += float(pred_clean.eq(labels).sum().item())
                paired["noisy_correct"] += float(logits_noisy.argmax(1).eq(labels).sum().item())
                for if_name in if_names:
                    z_c = clean_z[if_name].float()
                    z_n = noisy_z[if_name].float()
                    lam = float(modules[if_name].thresh.detach().clamp(min=1e-3).view(-1)[0])
                    k_c = _spike_count(z_c, lam, TEST_T)
                    k_n = _spike_count(z_n, lam, TEST_T)
                    ch = z_c.shape[1]
                    bucket = acc.setdefault(if_name, _new_acc(ch))
                    spatial = int(np.prod(z_c.shape[2:])) if z_c.dim() == 4 else 1
                    bucket["n"] += float(z_c.shape[0] * spatial)
                    bucket["diff_sq"] += _sum_over_spatial((z_n - z_c).pow(2)).cpu().numpy()
                    bucket["cross"] += _sum_over_spatial(k_c.ne(k_n).float()).cpu().numpy()
            del clean_z, noisy_z
    finally:
        probe.close()
        model.set_first_layer_input_noise_sigma(0.0)
    return acc, paired


def accumulate_pbound_and_d(
    model,
    loader,
    device,
    acc: dict[str, dict],
    predictors: list[dict],
    criterion,
):
    """Gaussian-approx P_bound on SNN z; D from T=0 QCFS last_pre_quant.

    Probe hooks must be removed before the T=0 pass: they reshape the IF
    input as (T, B, ...), which is invalid when T=0.
    D is E_b[||∂L/∂z_c||_F^2] on the QCFS surrogate, matching
    update_graph_margin_stats (SNN rate_uniform is not used for D).
    """
    del predictors
    probe = LayerActivationProbe(model, TEST_T)
    modules = dict(model.named_modules())
    try:
        for images, _labels in loader:
            images = images.to(device)
            keep = images.shape[0]
            probe.reset(keep)
            model.set_first_layer_input_noise_sigma(0.0)
            model.set_T(TEST_T)
            with torch.no_grad():
                model(images.clone())
            clean_z = {k: v.detach() for k, v in probe.z_maps.items()}
            for if_name, bucket in acc.items():
                if if_name not in clean_z:
                    continue
                z = clean_z[if_name].float()
                lam = float(modules[if_name].thresh.detach().clamp(min=1e-3).view(-1)[0])
                sigma_eff = np.sqrt(bucket["diff_sq"] / max(bucket["n"], 1.0))
                sig = torch.as_tensor(sigma_eff, device=z.device, dtype=z.dtype)
                view = [1, -1] + [1] * (z.dim() - 2)
                pmap = _bound_prob(z, lam, sig.view(*view), TEST_T)
                bucket["p_bound"] += _sum_over_spatial(pmap).cpu().numpy()
    finally:
        probe.close()
        model.set_T(TEST_T)
        model.eval()

    model.set_first_layer_input_noise_sigma(0.0)
    model.set_T(0)
    model.eval()
    try:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            keep = images.shape[0]
            with torch.enable_grad():
                logits = model(images)
                ce = criterion(logits, labels)
                acts, names = [], []
                for name, module in modules.items():
                    if not isinstance(module, IF) or _is_input_if_name(name):
                        continue
                    z = getattr(module, "last_pre_quant", None)
                    if z is None or not z.requires_grad:
                        continue
                    names.append(name)
                    acts.append(z)
                if acts:
                    grads = torch.autograd.grad(ce, acts, retain_graph=False, allow_unused=True)
                    for name, grad in zip(names, grads):
                        if grad is None or name not in acc:
                            continue
                        # Sum over spatial (and batch), then divide by n_images later:
                        # D_c = E_b[ ||∂L/∂z_{·,c,·,·}||_F^2 ].
                        acc[name]["d_energy"] += _sum_over_spatial(grad.detach().pow(2)).cpu().numpy()
                        acc[name]["d_n"] += float(keep)
            model.zero_grad(set_to_none=True)
    finally:
        model.set_T(TEST_T)
        model.eval()


def percentile_groups(values: np.ndarray, frac: float = 0.2) -> dict:
    n = int(values.size)
    k = max(1, int(round(n * frac)))
    order = np.argsort(values)
    rng = np.random.default_rng(SEED)
    shuffle = rng.choice(n, size=k, replace=False)
    mid_start = max((n - k) // 2, 0)
    return {
        "top": order[-k:].tolist(),
        "bottom": order[:k].tolist(),
        "middle": order[mid_start: mid_start + k].tolist(),
        "shuffle": shuffle.tolist(),
        "k": k,
        "n": n,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = get_torch_device(args.device)
    ckpt = resolve_ckpt(args)
    out = args.out_root / args.dataset / args.method
    out.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] {args.dataset} {args.method} ckpt={ckpt}", flush=True)
    model = load_model(ckpt, args.dataset, device)
    predictors = channel_predictors(model)
    pin = device.type == "cuda"
    loader = val_subset(args.dataset, args.n_images, args.batch_size, args.workers, pin)
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)

    pairs = [
        ("r", "A"),
        ("r", "p_cross"),
        ("m", "p_cross"),
        ("q", "p_cross"),
        ("r", "D"),
        ("m", "I_output"),
        ("p_bound", "p_cross"),
        ("p_bound_D", "I_output"),
    ]
    all_channel_rows = []
    all_layer_rows = []
    summaries = []
    paired_out = {}

    for sigma in args.sigma_list:
        print(f"[INFO] sigma={sigma} K={args.n_noise} n={args.n_images}", flush=True)
        acc, paired = accumulate_sigma(
            model, loader, device, sigma, args.n_noise, args.seed, predictors
        )
        accumulate_pbound_and_d(model, loader, device, acc, predictors, criterion)
        n_pair = max(paired["n"], 1.0)
        logit_rms = math.sqrt(paired["logit_sq"] / n_pair)
        paired_out[str(sigma)] = {
            "delta_logit_rms": logit_rms,
            "delta_logit_rms_over_sqrtK": logit_rms / math.sqrt(max(args.n_noise, 1)),
            "delta_margin": paired["margin_drop"] / n_pair,
            "clean_acc": 100.0 * paired["clean_correct"] / n_pair,
            "noisy_acc": 100.0 * paired["noisy_correct"] / n_pair,
            "n": paired["n"],
        }
        rows = []
        for spec in predictors:
            if spec["skip_input_if"] or spec["if_name"] not in acc:
                continue
            bucket = acc[spec["if_name"]]
            n = max(bucket["n"], 1.0)
            sigma_eff = np.sqrt(bucket["diff_sq"] / n)
            p_cross = bucket["cross"] / n
            p_bound = bucket["p_bound"] / n
            d_n = max(bucket["d_n"], 1.0)
            d_c = bucket["d_energy"] / d_n
            a_c = sigma_eff / (spec["lambda"] + EPS)
            i_out = d_c * (sigma_eff ** 2)
            for c in range(spec["r"].size):
                rows.append(
                    {
                        "dataset": args.dataset,
                        "method": args.method,
                        "sigma": sigma,
                        "if_name": spec["if_name"],
                        "weight_name": spec["weight_name"],
                        "role": spec["role"],
                        "channel": c,
                        "lambda": spec["lambda"],
                        "r": float(spec["r"][c]),
                        "rhat": float(spec["rhat"][c]),
                        "m": float(spec["m"][c]),
                        "q": float(spec["q"][c]),
                        "w2": float(spec["w2"][c]),
                        "sigma_eff": float(sigma_eff[c]),
                        "A": float(a_c[c]),
                        "p_cross": float(p_cross[c]),
                        "p_bound": float(p_bound[c]),
                        "D": float(d_c[c]),
                        "S": float(p_cross[c] * d_c[c]),
                        "I_output": float(i_out[c]),
                        "p_bound_D": float(p_bound[c] * d_c[c]),
                    }
                )
        all_channel_rows.extend(rows)
        for pred, outcome in pairs:
            bundle = _corr_bundle(rows, pred, outcome, rng)
            bundle.update({"dataset": args.dataset, "method": args.method, "sigma": sigma})
            summaries.append(bundle)
            print(
                f"  {pred:>10} vs {outcome:<10}  layer={bundle['layer_rho']:.3f}  "
                f"within={bundle['within_layer_rho_mean']:.3f}  "
                f"ranknorm={bundle['ranknorm_rho']:.3f}  "
                f"perm={bundle['perm_null_median']:.3f}",
                flush=True,
            )
        for if_name, group in _group_rows(rows).items():
            all_layer_rows.append(
                {
                    "dataset": args.dataset,
                    "method": args.method,
                    "sigma": sigma,
                    "if_name": if_name,
                    "n_channels": len(group),
                    "r_mean": float(np.mean([g["r"] for g in group])),
                    "m_mean": float(np.mean([g["m"] for g in group])),
                    "q_mean": float(np.mean([g["q"] for g in group])),
                    "A_mean": float(np.mean([g["A"] for g in group])),
                    "p_cross_mean": float(np.mean([g["p_cross"] for g in group])),
                    "p_bound_mean": float(np.mean([g["p_bound"] for g in group])),
                    "D_mean": float(np.mean([g["D"] for g in group])),
                    "S_mean": float(np.mean([g["S"] for g in group])),
                }
            )

    groups = {}
    for spec in predictors:
        if spec["skip_input_if"]:
            continue
        groups[spec["if_name"]] = {
            "weight_name": spec["weight_name"],
            "by_r": percentile_groups(spec["r"]),
            "by_m": percentile_groups(spec["m"]),
            "by_q": percentile_groups(spec["q"]),
        }

    write_csv(out / "channel_stats.csv", all_channel_rows)
    write_csv(out / "layer_stats.csv", all_layer_rows)
    (out / "spearman_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (out / "paired_metrics.json").write_text(json.dumps(paired_out, indent=2) + "\n")
    (out / "channel_groups.json").write_text(json.dumps(groups, indent=2) + "\n")
    card = {
        "dataset": args.dataset,
        "method": args.method,
        "seed": args.seed,
        "checkpoint": str(ckpt),
        "n_images": args.n_images,
        "n_noise": args.n_noise,
        "sigmas": list(args.sigma_list),
        "n_scored_channels": len([p for p in predictors if not p["skip_input_if"]]),
        "paired": paired_out,
        "spearman": summaries,
    }
    (out / "scorecard.json").write_text(json.dumps(card, indent=2, default=str) + "\n")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
