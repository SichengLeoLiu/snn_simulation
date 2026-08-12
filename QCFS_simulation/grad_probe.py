"""
Diagnose CE vs explicit-regularizer gradients on IF thresholds (lambda)
and BN affine gamma at selected training stages.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn

from Models.layer import IF


PROBE_CSV_FIELDS = [
    "stage",
    "epoch",
    "epochs",
    "regularizer",
    "reg_coeff",
    "param_kind",
    "layer",
    "numel",
    "value_mean",
    "value_rms",
    "value_l2",
    "grad_ce_l2",
    "grad_reg_l2",
    "grad_total_l2",
    "cos_ce_reg",
    "dot_ce_reg",
    "sign_agree_frac",
    "grad_ce_mean",
    "grad_reg_mean",
    "ce_loss",
    "reg_loss",
]


def default_probe_epochs(num_epochs: int) -> list[int]:
    """Early / mid / late epoch indices (0-based), unique and sorted."""
    n = max(int(num_epochs), 1)
    early = 0
    mid = max(0, (n - 1) // 2)
    late = n - 1
    return sorted({early, mid, late})


def parse_probe_epochs(spec: Optional[str], num_epochs: int) -> list[int]:
    """
    Parse --grad_probe_epochs.

    - None / empty / 'auto' -> early, mid, late
    - comma list of ints (0-based epoch indices)
    """
    if spec is None or str(spec).strip() == "" or str(spec).strip().lower() == "auto":
        return default_probe_epochs(num_epochs)
    epochs = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        ep = int(part)
        if ep < 0:
            ep = num_epochs + ep
        epochs.append(max(0, min(ep, num_epochs - 1)))
    return sorted(set(epochs)) if epochs else default_probe_epochs(num_epochs)


def _collect_scale_params(model: nn.Module) -> list[tuple[str, str, nn.Parameter]]:
    """Return (kind, name, parameter) for IF.thresh and BN.weight (gamma)."""
    rows: list[tuple[str, str, nn.Parameter]] = []
    for name, module in model.named_modules():
        if isinstance(module, IF) and getattr(module, "thresh", None) is not None:
            rows.append(("lambda", name, module.thresh))
        elif isinstance(
            module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        ) and getattr(module, "weight", None) is not None:
            rows.append(("bn_gamma", name, module.weight))
    return rows


def _tensor_stats(t: torch.Tensor) -> tuple[float, float, float]:
    t = t.detach().float().reshape(-1)
    if t.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(t.mean().item())
    rms = float(t.pow(2).mean().sqrt().item())
    l2 = float(t.norm(p=2).item())
    return mean, rms, l2


def _grad_pair_metrics(
    g_ce: Optional[torch.Tensor], g_reg: Optional[torch.Tensor]
) -> dict:
    if g_ce is None:
        g_ce = torch.zeros(1)
    if g_reg is None:
        g_reg = torch.zeros_like(g_ce)
    g_ce = g_ce.detach().float().reshape(-1)
    g_reg = g_reg.detach().float().reshape(-1)
    g_tot = g_ce + g_reg
    ce_l2 = float(g_ce.norm(p=2).item())
    reg_l2 = float(g_reg.norm(p=2).item())
    tot_l2 = float(g_tot.norm(p=2).item())
    dot = float(torch.dot(g_ce, g_reg).item())
    if ce_l2 > 0.0 and reg_l2 > 0.0:
        cos = dot / (ce_l2 * reg_l2)
    else:
        cos = float("nan")
    # Channel-wise sign agreement (ignore near-zero CE coords)
    mask = g_ce.abs() > 1e-12
    if bool(mask.any()):
        agree = float((torch.sign(g_ce[mask]) == torch.sign(g_reg[mask])).float().mean().item())
    else:
        agree = float("nan")
    return {
        "grad_ce_l2": ce_l2,
        "grad_reg_l2": reg_l2,
        "grad_total_l2": tot_l2,
        "cos_ce_reg": cos,
        "dot_ce_reg": dot,
        "sign_agree_frac": agree,
        "grad_ce_mean": float(g_ce.mean().item()),
        "grad_reg_mean": float(g_reg.mean().item()),
    }


def probe_ce_vs_reg_grads(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion,
    reg_loss_fn,
    reg_coeff: float,
    T: int = 0,
    quant_level=None,
    stage: str = "",
    epoch: int = 0,
    epochs: int = 0,
    regularizer: str = "",
) -> list[dict]:
    """
    One-batch diagnostic: separate CE and (coeff * R) grads on lambda / BN-gamma.

    Does not call optimizer.step(); clears parameter .grad afterwards.
    """
    if reg_loss_fn is None:
        raise ValueError(
            "grad probe requires an explicit regularizer (e.g. manual_l2 / mne_l2); "
            "optimizer-only weight_decay has no separate reg loss."
        )

    was_training = model.training
    model.train()
    targets = _collect_scale_params(model)
    params = [p for _, _, p in targets]
    device = images.device

    labels = labels.to(device)
    images = images.to(device)
    if T > 0:
        outputs = model(images).mean(0)
    else:
        outputs = model(images)
    ce = criterion(outputs, labels)
    reg_raw = reg_loss_fn(model, T, quant_level)
    reg = float(reg_coeff) * reg_raw

    # CE grads
    grads_ce = torch.autograd.grad(
        ce, params, retain_graph=True, allow_unused=True
    )
    # Reg grads (scaled by coeff, matching training objective)
    grads_reg = torch.autograd.grad(
        reg, params, retain_graph=False, allow_unused=True
    )

    ce_val = float(ce.detach().item())
    reg_val = float(reg.detach().item())
    rows: list[dict] = []
    for (kind, name, param), g_ce, g_reg in zip(targets, grads_ce, grads_reg):
        v_mean, v_rms, v_l2 = _tensor_stats(param.data)
        metrics = _grad_pair_metrics(g_ce, g_reg)
        rows.append(
            {
                "stage": stage,
                "epoch": int(epoch),
                "epochs": int(epochs),
                "regularizer": regularizer,
                "reg_coeff": float(reg_coeff),
                "param_kind": kind,
                "layer": name,
                "numel": int(param.numel()),
                "value_mean": v_mean,
                "value_rms": v_rms,
                "value_l2": v_l2,
                "ce_loss": ce_val,
                "reg_loss": reg_val,
                **metrics,
            }
        )

    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    return rows


def append_probe_csv(path: Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROBE_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in PROBE_CSV_FIELDS})


def summarize_probe_rows(rows: list[dict], max_layers: int = 8) -> list[str]:
    """Short log lines: lambda always, then a few BN gamma layers by |grad_reg|."""
    lines = []
    if not rows:
        return lines
    ce = rows[0].get("ce_loss", float("nan"))
    reg = rows[0].get("reg_loss", float("nan"))
    lines.append(f"grad_probe: ce={ce:.5g} reg={reg:.5g} n_params={len(rows)}")

    for row in rows:
        if row["param_kind"] != "lambda":
            continue
        cos = row["cos_ce_reg"]
        cos_s = "nan" if (isinstance(cos, float) and math.isnan(cos)) else f"{cos:.4f}"
        lines.append(
            "  lambda[{layer}]: val={value_mean:.4g} "
            "|g_ce|={grad_ce_l2:.4g} |g_reg|={grad_reg_l2:.4g} cos={cos}".format(
                layer=row["layer"],
                value_mean=row["value_mean"],
                grad_ce_l2=row["grad_ce_l2"],
                grad_reg_l2=row["grad_reg_l2"],
                cos=cos_s,
            )
        )

    bn_rows = [r for r in rows if r["param_kind"] == "bn_gamma"]
    bn_rows.sort(key=lambda r: float(r["grad_reg_l2"]), reverse=True)
    for row in bn_rows[:max_layers]:
        cos = row["cos_ce_reg"]
        cos_s = "nan" if (isinstance(cos, float) and math.isnan(cos)) else f"{cos:.4f}"
        lines.append(
            "  gamma[{layer}]: rms={value_rms:.4g} "
            "|g_ce|={grad_ce_l2:.4g} |g_reg|={grad_reg_l2:.4g} cos={cos} "
            "sign_agree={sign_agree_frac}".format(
                layer=row["layer"],
                value_rms=row["value_rms"],
                grad_ce_l2=row["grad_ce_l2"],
                grad_reg_l2=row["grad_reg_l2"],
                cos=cos_s,
                sign_agree_frac=(
                    "nan"
                    if (
                        isinstance(row["sign_agree_frac"], float)
                        and math.isnan(row["sign_agree_frac"])
                    )
                    else f"{row['sign_agree_frac']:.3f}"
                ),
            )
        )
    return lines


def stage_name_for_epoch(epoch: int, probe_epochs: list[int]) -> str:
    """Map epoch index to early/mid/late (or epoch_k if custom)."""
    if not probe_epochs:
        return f"epoch_{epoch}"
    ordered = list(probe_epochs)
    labels = ["early", "mid", "late"]
    if len(ordered) == 3:
        mapping = {ep: lab for ep, lab in zip(ordered, labels)}
        return mapping.get(epoch, f"epoch_{epoch}")
    if len(ordered) == 1:
        return "only"
    if len(ordered) == 2:
        mapping = {ordered[0]: "early", ordered[1]: "late"}
        return mapping.get(epoch, f"epoch_{epoch}")
    # >3 custom points
    try:
        idx = ordered.index(epoch)
    except ValueError:
        return f"epoch_{epoch}"
    if idx == 0:
        return "early"
    if idx == len(ordered) - 1:
        return "late"
    if idx == len(ordered) // 2:
        return "mid"
    return f"stage_{idx}"
