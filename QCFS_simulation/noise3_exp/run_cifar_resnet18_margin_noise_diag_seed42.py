#!/usr/bin/env python3
"""Checkpoint-only CIFAR ResNet-18 margin / noise-scale diagnostic.

No training. Loads existing seed-42 checkpoints and records:

  1. Clean top-1 (true-class) and top-2 margins: median and 10th percentile.
  2. Margin noise s_m = Std_streams(m̃ − m) over independent noise streams.
  3. Robustness score ρ = m / s_m.
  4. Per-IF σ/λ_l and σ/RMS(z_l) on a clean test pass (z = time-summed IF input).
  5. Training ||∇R|| / ||∇L_CE|| from the sibling epoch_log (no extra backward).
  6. CIFAR-100 Top-5 and coarse-superclass swaps among Top-1 errors.

Default arms (do not mix in historical four-regs MNE):
  l2wo  four_regs L2-wo seed 42
  mne   fair MNE-L2 detach, resnet map, val-locked rc=1e-4, seed 42

Eval protocol matches the locked sweeps: T=16 rate_uniform, post_input_if
Gaussian, test shuffle=False. Stream k uses seed EVAL_SEED+k.

Do not retune rc / β from these measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Models import modelpool  # noqa: E402
from Models.layer import IF  # noqa: E402
from run_cifar_vgg16_onesided_q_assignment_ablation import (  # noqa: E402
    LVAL,
    TEST_T,
    test_loader,
    write_csv,
)
from utils import get_torch_device, seed_all  # noqa: E402

ARCH = "resnet18"
SEED = 42
EVAL_NOISE_SEED = 0
DEFAULT_STREAMS = 8
DEFAULT_SIGMAS = (1.0, 3.0, 5.0)
SM_FLOOR = 1e-8
LAST_RATIO_EPOCHS = 50

SCRATCH = Path("/scratch/gs14/sl9144/snn_results")
METHODS = {
    "l2wo": {
        "label": "L2-wo",
        "root_env": "L2WO_ROOT",
        "default_root": SCRATCH / "cifar_resnet18_four_regs_5seed",
        "ckpt_rel": "{dataset}/r18_l2wo/seed{seed}/checkpoints/resnet18_L[16]_r18_l2wo_seed{seed}_L16_trainT0.pth",
        "log_rel": "{dataset}/r18_l2wo/seed{seed}/epoch_log.csv",
    },
    "mne": {
        "label": "MNE-L2 detach rc=1e-4 (val-locked)",
        "root_env": "MNE_ROOT",
        "default_root": SCRATCH / "cifar_resnet18_fair_mne_detach",
        "ckpt_rel": "{dataset}/r18_mne_resnet_rc1e-4/seed{seed}/checkpoints/resnet18_L[16]_r18_mne_resnet_rc1e-4_seed{seed}_L16_trainT0.pth",
        "log_rel": "{dataset}/r18_mne_resnet_rc1e-4/seed{seed}/epoch_log.csv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHODS), default=None)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=int(os.environ.get("EVAL_SEED", str(EVAL_NOISE_SEED))),
    )
    parser.add_argument("--n-streams", type=int, default=int(os.environ.get("N_STREAMS", str(DEFAULT_STREAMS))))
    parser.add_argument(
        "--sigmas",
        default=os.environ.get("DIAG_SIGMAS", "1,3,5"),
        help="comma-separated noise sigmas for s_m",
    )
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=int(os.environ.get("MAX_SAMPLES", "0")))
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar_resnet18_margin_noise_diag_seed42",
    )
    args = parser.parse_args()
    args.eval_seed = int(args.eval_seed)
    args.n_streams = int(args.n_streams)
    args.sigmas = tuple(float(x) for x in str(args.sigmas).split(",") if x.strip())
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    if args.summarize:
        return args
    if args.method is None:
        parser.error("--method is required unless --summarize")
    if args.n_streams < 2:
        parser.error("--n-streams must be >= 2")
    if not args.sigmas:
        parser.error("need at least one sigma")
    return args


def cfg_dir(args) -> Path:
    return args.out_root / args.dataset / args.method / f"seed{args.seed}"


def _format_rel(rel: str, dataset: str, seed: int) -> str:
    return rel.format(dataset=dataset, seed=seed)


def method_paths(method: str, dataset: str, seed: int) -> tuple[Path, Path]:
    spec = METHODS[method]
    root = Path(os.environ.get(spec["root_env"], str(spec["default_root"])))
    ckpt = root / _format_rel(spec["ckpt_rel"], dataset, seed)
    log = root / _format_rel(spec["log_rel"], dataset, seed)
    return ckpt, log


def load_model(ckpt: Path, device, dataset: str):
    model = modelpool(ARCH, dataset)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.set_L(LVAL)
    model.set_T(TEST_T)
    model.set_mode("rate_uniform")
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule("normal")
    model.set_first_layer_input_noise_position("post_input_if")
    model.set_first_layer_input_noise_type("gaussian")
    model.set_first_layer_input_noise_sigma(0.0)
    return model.to(device).eval()


def analog_z(module: IF, x: torch.Tensor) -> torch.Tensor:
    """Time-summed analog IF input. Matches rate_uniform z = sum_t x_t."""
    steps = int(getattr(module, "T", 0) or 0)
    if steps > 1 and x.dim() >= 1 and x.shape[0] % steps == 0:
        return x.reshape(steps, x.shape[0] // steps, *x.shape[1:]).sum(0)
    return x


def qtile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(torch.quantile(values.detach().float().reshape(-1).cpu(), q))


def summarize_vec(values: torch.Tensor) -> dict:
    flat = values.detach().float().reshape(-1).cpu()
    if flat.numel() == 0:
        return {"mean": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "mean": float(flat.mean()),
        "median": qtile(flat, 0.5),
        "p10": qtile(flat, 0.1),
        "p90": qtile(flat, 0.9),
    }


def true_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    row = torch.arange(logits.shape[0], device=logits.device)
    target = logits[row, labels]
    rival = logits.clone()
    rival[row, labels] = float("-inf")
    return target - rival.max(dim=1).values


def top12_margin(logits: torch.Tensor) -> torch.Tensor:
    top2 = logits.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def mean_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.mean(0) if logits.dim() == 3 else logits


class IfScaleProbe:
    def __init__(self, model: torch.nn.Module):
        self.modules = [(name, module) for name, module in model.named_modules() if isinstance(module, IF)]
        self.sq = {name: 0.0 for name, _ in self.modules}
        self.count = {name: 0 for name, _ in self.modules}
        self.lam = {
            name: float(module.thresh.detach().abs().reshape(-1)[0].clamp(min=1e-8).cpu())
            for name, module in self.modules
        }
        self.handles = []

    def attach(self) -> None:
        for name, module in self.modules:
            def _hook(_mod, inputs, _out, layer=name):
                z = analog_z(_mod, inputs[0].detach())
                self.sq[layer] += float(z.float().pow(2).sum().cpu())
                self.count[layer] += int(z.numel())

            self.handles.append(module.register_forward_hook(_hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def rows(self, sigmas: tuple[float, ...]) -> list[dict]:
        rows = []
        for name, _ in self.modules:
            rms = (self.sq[name] / max(self.count[name], 1)) ** 0.5
            row = {
                "layer": name,
                "lambda": self.lam[name],
                "rms_z": rms,
                "n": self.count[name],
            }
            for sigma in sigmas:
                row[f"sigma_{sigma:g}_over_lambda"] = float(sigma) / max(self.lam[name], 1e-8)
                row[f"sigma_{sigma:g}_over_rms_z"] = float(sigma) / max(rms, 1e-8)
            rows.append(row)
        return rows


@torch.inference_mode()
def collect_logits(model, loader, device, sigma: float, seed: int, max_samples: int, probe=None):
    seed_all(seed)
    model.set_first_layer_input_noise_sigma(float(sigma))
    chunks = []
    labels = []
    seen = 0
    if probe is not None:
        probe.attach()
    try:
        for images, targets in loader:
            if max_samples > 0 and seen >= max_samples:
                break
            if max_samples > 0 and seen + images.shape[0] > max_samples:
                keep = max_samples - seen
                images = images[:keep]
                targets = targets[:keep]
            images = images.to(device, non_blocking=True)
            logits = mean_logits(model(images)).float().cpu()
            chunks.append(logits)
            labels.append(targets.cpu())
            seen += int(targets.shape[0])
    finally:
        if probe is not None:
            probe.close()
        model.set_first_layer_input_noise_sigma(0.0)
    if not chunks:
        raise RuntimeError("no samples collected")
    return torch.cat(chunks, dim=0), torch.cat(labels, dim=0)


def load_coarse(dataset: str, n_labels: int) -> torch.Tensor | None:
    if dataset != "cifar100":
        return None
    root = Path(os.path.expanduser(os.environ.get("CIFAR_ROOT", "~/datasets")))
    path = root / "cifar-100-python" / "test"
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    coarse = torch.tensor(payload["coarse_labels"], dtype=torch.long)
    if coarse.numel() < n_labels:
        raise RuntimeError(f"CIFAR-100 coarse labels shorter than test set: {coarse.numel()} < {n_labels}")
    return coarse[:n_labels]


def fine_to_coarse_map(labels: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
    mapping = torch.full((100,), -1, dtype=torch.long)
    mapping[labels] = coarse
    return mapping


def _floats(rows: list[dict], key: str) -> list[float]:
    out = []
    for row in rows:
        raw = row.get(key, "")
        if raw in (None, "", "nan"):
            continue
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def epoch_log_ratio(path: Path) -> dict:
    payload = {
        "epoch_log": str(path),
        "exists": path.is_file(),
        "n_epochs": 0,
        "has_ce_grad_norm": False,
        "reg_ce_ratio_last": float("nan"),
        "reg_ce_ratio_last50_mean": float("nan"),
        "reg_ce_ratio_last50_median": float("nan"),
        "reg_grad_norm_last": float("nan"),
        "ce_grad_norm_last": float("nan"),
    }
    if not path.is_file():
        return payload
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    payload["n_epochs"] = len(rows)
    payload["has_ce_grad_norm"] = bool(rows) and "ce_grad_norm" in rows[0]
    ratios = _floats(rows, "reg_ce_ratio")
    regs = _floats(rows, "reg_grad_norm")
    ces = _floats(rows, "ce_grad_norm")
    if ratios:
        tail = ratios[-LAST_RATIO_EPOCHS:]
        payload["reg_ce_ratio_last"] = ratios[-1]
        payload["reg_ce_ratio_last50_mean"] = float(statistics.fmean(tail))
        payload["reg_ce_ratio_last50_median"] = float(statistics.median(tail))
    if regs:
        payload["reg_grad_norm_last"] = regs[-1]
    if ces:
        payload["ce_grad_norm_last"] = ces[-1]
    return payload


def error_breakdown(logits: torch.Tensor, labels: torch.Tensor, coarse: torch.Tensor | None, mapping: torch.Tensor | None) -> dict:
    pred = logits.argmax(dim=1)
    top5 = logits.topk(5, dim=1).indices
    top1_ok = pred.eq(labels)
    top5_ok = top5.eq(labels.view(-1, 1)).any(dim=1)
    n = max(int(labels.numel()), 1)
    n_err = int((~top1_ok).sum())
    near = int((~top1_ok & top5_ok).sum())
    row = {
        "top1_acc": 100.0 * float(top1_ok.float().mean()),
        "top5_acc": 100.0 * float(top5_ok.float().mean()),
        "n_top1_error": n_err,
        "frac_top1_error_still_top5": (near / n_err) if n_err else float("nan"),
        "frac_true_in_top5_given_top1_error": (near / n_err) if n_err else float("nan"),
        "n": n,
    }
    if coarse is not None and mapping is not None:
        pred_coarse = mapping[pred]
        same_coarse = pred_coarse.ge(0) & pred_coarse.eq(coarse)
        coarse_err = int((~top1_ok & same_coarse).sum())
        row["coarse_acc"] = 100.0 * float(same_coarse.float().mean())
        row["frac_top1_error_same_superclass"] = (coarse_err / n_err) if n_err else float("nan")
    return row


def mean_breakdown(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = [key for key in rows[0] if key != "n"]
    out = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) == row.get(key)]
        out[key] = float(statistics.fmean(values)) if values else float("nan")
    out["n"] = rows[0]["n"]
    out["n_streams"] = len(rows)
    return out


def run_diag(args) -> dict:
    spec = METHODS[args.method]
    ckpt, log_path = method_paths(args.method, args.dataset, args.seed)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    loader = test_loader(args, pin)
    print(
        f"[INFO] {args.dataset} {args.method} seed={args.seed} eval_seed={args.eval_seed} "
        f"streams={args.n_streams} sigmas={list(args.sigmas)} ckpt={ckpt}",
        flush=True,
    )
    model = load_model(ckpt, device, args.dataset)
    probe = IfScaleProbe(model)
    clean_logits, labels = collect_logits(
        model, loader, device, 0.0, args.eval_seed, args.max_samples, probe=probe
    )
    m_true = true_margin(clean_logits, labels)
    m_top12 = top12_margin(clean_logits)
    coarse = load_coarse(args.dataset, labels.numel())
    mapping = fine_to_coarse_map(labels, coarse) if coarse is not None else None
    clean_break = error_breakdown(clean_logits, labels, coarse, mapping)

    sigma_block = {}
    sample_cols = {
        "index": torch.arange(labels.numel()),
        "label": labels,
        "m_true": m_true,
        "m_top12": m_top12,
        "pred_clean": clean_logits.argmax(dim=1),
    }
    if coarse is not None:
        sample_cols["coarse"] = coarse

    for sigma in args.sigmas:
        noisy_true = []
        noisy_top12 = []
        breaks = []
        for stream in range(args.n_streams):
            stream_seed = int(args.eval_seed + stream)
            logits, _ = collect_logits(model, loader, device, sigma, stream_seed, args.max_samples)
            if logits.shape[0] != labels.numel():
                raise RuntimeError(f"sample count changed at sigma={sigma} stream={stream}")
            mt = true_margin(logits, labels)
            m12 = top12_margin(logits)
            noisy_true.append(mt)
            noisy_top12.append(m12)
            breaks.append(error_breakdown(logits, labels, coarse, mapping))
            print(
                f"sigma={sigma:g} stream={stream} seed={stream_seed} "
                f"top1={breaks[-1]['top1_acc']:.2f} top5={breaks[-1]['top5_acc']:.2f}",
                flush=True,
            )
        stacked_true = torch.stack(noisy_true, dim=0)
        stacked_12 = torch.stack(noisy_top12, dim=0)
        s_true = (stacked_true - m_true.unsqueeze(0)).std(dim=0, unbiased=True)
        s_12 = (stacked_12 - m_top12.unsqueeze(0)).std(dim=0, unbiased=True)
        rho_true = m_true / s_true.clamp(min=SM_FLOOR)
        rho_12 = m_top12 / s_12.clamp(min=SM_FLOOR)
        tag = f"s{sigma:g}".replace(".", "p")
        sample_cols[f"s_m_true_{tag}"] = s_true
        sample_cols[f"s_m_top12_{tag}"] = s_12
        sample_cols[f"rho_true_{tag}"] = rho_true
        sample_cols[f"rho_top12_{tag}"] = rho_12
        sigma_block[f"{sigma:g}"] = {
            "sigma": float(sigma),
            "s_m_true": summarize_vec(s_true),
            "s_m_top12": summarize_vec(s_12),
            "rho_true": summarize_vec(rho_true),
            "rho_top12": summarize_vec(rho_12),
            "noisy": mean_breakdown(breaks),
        }

    layer_rows = probe.rows(args.sigmas)
    write_csv(out / "layer_scale.csv", layer_rows)
    sample_rows = []
    n = labels.numel()
    for i in range(n):
        row = {key: (int(val[i]) if val.dtype in (torch.int64, torch.int32) else float(val[i])) for key, val in sample_cols.items()}
        sample_rows.append(row)
    write_csv(out / "per_sample.csv", sample_rows)

    card = {
        "dataset": args.dataset,
        "method": args.method,
        "label": spec["label"],
        "arch": ARCH,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "n_streams": args.n_streams,
        "sigmas": [float(s) for s in args.sigmas],
        "n_samples": n,
        "checkpoint": str(ckpt),
        "layer_map": "resnet" if args.method == "mne" else "n/a",
        "clean": {
            "m_true": summarize_vec(m_true),
            "m_top12": summarize_vec(m_top12),
            **clean_break,
        },
        "noise": sigma_block,
        "layer_scale": layer_rows,
        "grad_ratio": epoch_log_ratio(log_path),
        "protocol": {
            "T": TEST_T,
            "L": LVAL,
            "mode": "rate_uniform",
            "noise": "post_input_if gaussian",
            "s_m": "unbiased std over streams of (noisy_margin - clean_margin)",
            "m_true": "z_y - max_{k!=y} z_k",
            "m_top12": "top1 logit - top2 logit",
            "rho": "m / max(s_m, 1e-8)",
            "selection_uses_test": False,
        },
    }
    (out / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps({k: card[k] for k in ("dataset", "method", "clean", "grad_ratio")}, indent=2), flush=True)
    return card


def summarize(out_root: Path) -> None:
    cards = [json.loads(path.read_text()) for path in sorted(out_root.glob("*/*/seed*/scorecard.json"))]
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    print(
        f"{'dataset':<10} {'method':<6} {'m_p50':>8} {'m_p10':>8} "
        f"{'s1_p50':>8} {'rho1_p50':>9} {'s3_p50':>8} {'rho3_p50':>9} "
        f"{'top1':>7} {'top5':>7} {'near5':>7} {'ratio50':>9}"
    )
    for card in cards:
        clean = card["clean"]
        noise = card.get("noise", {})
        s1 = noise.get("1", noise.get("1.0", {}))
        s3 = noise.get("3", noise.get("3.0", {}))
        grad = card.get("grad_ratio", {})
        print(
            f"{card['dataset']:<10} {card['method']:<6} "
            f"{clean['m_true']['median']:8.3f} {clean['m_true']['p10']:8.3f} "
            f"{s1.get('s_m_true', {}).get('median', float('nan')):8.3f} "
            f"{s1.get('rho_true', {}).get('median', float('nan')):9.3f} "
            f"{s3.get('s_m_true', {}).get('median', float('nan')):8.3f} "
            f"{s3.get('rho_true', {}).get('median', float('nan')):9.3f} "
            f"{clean['top1_acc']:7.2f} {clean.get('top5_acc', float('nan')):7.2f} "
            f"{s3.get('noisy', {}).get('frac_top1_error_still_top5', float('nan')):7.3f} "
            f"{grad.get('reg_ce_ratio_last50_mean', float('nan')):9.5f}"
        )


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(args.out_root)
        return
    run_diag(args)


if __name__ == "__main__":
    main()
