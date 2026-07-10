"""
Robustness metrics for noise-sweep accuracy curves.

DRS (Derivative Robustness Score) — ICLR 2026 definition:
  R(σ) = A(σ) / A(σ₀)
  g_i = [(R(σ_i) - R(σ_{i+1})) / (σ_{i+1} - σ_i)]₊
  DRS = (1 + (1/(σ_K-σ₀)) Σ g_i² (σ_{i+1}-σ_i))⁻¹

Legacy AUC-RS (relative accuracy retention integral) kept for comparison only.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Iterable, Sequence


def _sorted_curve(sigmas: Sequence[float], accs: Sequence[float]) -> tuple[list[float], list[float]]:
    pairs = sorted(zip(sigmas, accs), key=lambda x: x[0])
    if not pairs:
        return [], []
    return [p[0] for p in pairs], [p[1] for p in pairs]


def acc_at_sigma(sigmas: Sequence[float], accs: Sequence[float], target: float) -> float:
    for s, a in zip(sigmas, accs):
        if abs(s - target) < 1e-6:
            return float(a)
    raise KeyError(f"sigma={target} missing from curve")


def derivative_robustness_score(sigmas: Sequence[float], accs: Sequence[float]) -> float:
    sigmas, accs = _sorted_curve(sigmas, accs)
    if len(sigmas) < 2:
        return 1.0 if (accs and accs[0] > 0) else 0.0

    a0 = acc_at_sigma(sigmas, accs, sigmas[0])
    if a0 <= 0:
        return 0.0

    sigma_0, sigma_k = sigmas[0], sigmas[-1]
    span = sigma_k - sigma_0
    if span <= 0:
        return 1.0

    retention = [a / a0 for a in accs]
    weighted_sq = 0.0
    for i in range(len(sigmas) - 1):
        ds = sigmas[i + 1] - sigmas[i]
        if ds <= 0:
            continue
        g = max(0.0, (retention[i] - retention[i + 1]) / ds)
        weighted_sq += g * g * ds

    return 1.0 / (1.0 + weighted_sq / span)


def auc_robustness_score(sigmas: Sequence[float], accs: Sequence[float]) -> float:
    """Legacy AUC-type robustness score (integral of R(σ) over [σ₀, σ_K])."""
    sigmas, accs = _sorted_curve(sigmas, accs)
    if not sigmas:
        return 0.0

    a0 = acc_at_sigma(sigmas, accs, sigmas[0])
    if a0 <= 0:
        return 0.0

    rs = 0.0
    for i in range(len(sigmas) - 1):
        ds = sigmas[i + 1] - sigmas[i]
        if ds <= 0:
            continue
        rs += 0.5 * (accs[i] / a0 + accs[i + 1] / a0) * ds
    return rs


def aggregate_drs(
    seed_curves: dict[int, Iterable[tuple[float, float]]],
) -> dict[str, float]:
    drs_vals, auc_vals = [], []
    for seed in sorted(seed_curves):
        pairs = sorted(seed_curves[seed], key=lambda x: x[0])
        sigmas = [p[0] for p in pairs]
        accs = [p[1] for p in pairs]
        drs_vals.append(derivative_robustness_score(sigmas, accs))
        auc_vals.append(auc_robustness_score(sigmas, accs))

    n = len(drs_vals)
    drs_std = statistics.stdev(drs_vals) if n > 1 else 0.0
    auc_std = statistics.stdev(auc_vals) if n > 1 else 0.0
    return {
        "DRS_mean": statistics.mean(drs_vals) if n else float("nan"),
        "DRS_std": drs_std,
        "DRS_sem": drs_std / (n ** 0.5) if n else 0.0,
        "AUC_RS_mean": statistics.mean(auc_vals) if n else float("nan"),
        "AUC_RS_std": auc_std,
        "AUC_RS_sem": auc_std / (n ** 0.5) if n else 0.0,
        "n_seeds": n,
    }


def aggregate_drs_from_raw_rows(
    raw_rows: list[dict],
    *,
    group_keys: tuple[str, ...],
    sigma_key: str = "sigma",
    acc_key: str = "acc",
    seed_key: str = "seed",
) -> list[dict]:
    """Group raw long-format noise rows and compute per-group DRS stats."""
    buckets: dict[tuple, dict[int, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    meta: dict[tuple, dict] = {}
    for r in raw_rows:
        key = tuple(r[k] for k in group_keys)
        buckets[key][int(r[seed_key])].append((float(r[sigma_key]), float(r[acc_key])))
        if key not in meta:
            meta[key] = {k: r[k] for k in group_keys}

    rows = []
    for key in sorted(buckets):
        stats = aggregate_drs(buckets[key])
        rows.append({**meta[key], **stats})
    return rows
