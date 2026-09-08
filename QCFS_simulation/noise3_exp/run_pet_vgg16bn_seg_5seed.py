#!/usr/bin/env python3
"""Pet foreground segmentation: 5 training seeds, shared test noise, encoder/decoder probe.

Only the right-hand task from the cls/seg pairing. Methods: L2-wo, MNE detach,
MNE no-detach. Training seeds 40–44. Every eval uses the same post-IF noise
stream (PET_EVAL_SEED=0), independent of the training seed.

Reports clean fg-IoU / mIoU, σ=3 and σ=5, and high-noise AUC [3,5]. After the
sweep, records encoder vs decoder IF crossing and margin-to-noise at σ∈{0,3,5}
to test whether the MNE gain lives in the serial decoder's later stages.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_pet_vgg16bn_cls_seg_seed42 as base  # noqa: E402
from Models.PetVGG import (  # noqa: E402
    IGNORE_INDEX,
    decoder_if_modules,
    encoder_if_modules,
)
from Models.layer import IF  # noqa: E402
from pet import pet_is_ready  # noqa: E402
from utils import get_torch_device, seed_all  # noqa: E402

SEEDS = (40, 41, 42, 43, 44)
EVAL_NOISE_SEED = 0
PROBE_SIGMAS = (0.0, 3.0, 5.0)
HIGH_NOISE_MIN = 3.0
METHODS = ("l2wo", "mne", "nodetach")
SCORE_KEYS = (
    "clean_fg_iou",
    "clean_miou",
    "fg_iou_s3",
    "fg_iou_s5",
    "miou_s3",
    "miou_s5",
    "auc_high_fg",
    "auc_high_miou",
)

_ORIG_CFG_DIR = base.cfg_dir
_ORIG_CKPT_PATH = base.ckpt_path


def cfg_dir(args) -> Path:
    return args.out_root / args.method / f"seed{args.seed}"


def ckpt_path(args) -> Path:
    head = "headif" if args.head_if else "linear"
    return (
        cfg_dir(args)
        / "checkpoints"
        / f"{base.ARCH}_seg_{head}_L[{base.LVAL}]_{args.method}_seed{args.seed}_L{base.LVAL}_trainT{base.TRAIN_T}.pth"
    )


def install_output_layout() -> None:
    base.cfg_dir = cfg_dir
    base.ckpt_path = ckpt_path


def restore_output_layout() -> None:
    base.cfg_dir = _ORIG_CFG_DIR
    base.ckpt_path = _ORIG_CKPT_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("PET_EPOCHS", str(base.EPOCHS))))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("PET_BATCH", "8")))
    parser.add_argument("--eval-batch-size", type=int, default=int(os.environ.get("PET_EVAL_BATCH", "2")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("PET_NUM_WORKERS", "4")))
    parser.add_argument("--lr", type=float, default=base.LR)
    parser.add_argument("--size", type=int, default=int(os.environ.get("PET_SIZE", "224")))
    parser.add_argument("--eval-t", type=int, default=int(os.environ.get("PET_EVAL_T", str(base.TEST_T))))
    parser.add_argument("--eval-seed", type=int, default=int(os.environ.get("PET_EVAL_SEED", str(EVAL_NOISE_SEED))))
    parser.add_argument("--max-probe-images", type=int, default=int(os.environ.get("PET_PROBE_IMAGES", "400")))
    parser.add_argument("--skip-probe", action="store_true", default=os.environ.get("PET_SKIP_PROBE", "0") == "1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--pet-root",
        type=Path,
        default=Path(os.environ.get("PET_ROOT", os.environ.get("CIFAR_ROOT", "~/datasets"))),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "pet_vgg16bn_seg_5seed",
    )
    args = parser.parse_args()
    args.task = "seg"
    args.head_if = False
    args.max_eval_images = 0
    args.pet_root = Path(os.path.expanduser(str(args.pet_root)))
    args.eval_t = max(0, int(args.eval_t))
    args.eval_seed = int(args.eval_seed)
    args.max_probe_images = max(0, int(args.max_probe_images))
    if args.max_probe_images == 0:
        args.skip_probe = True
    if args.batch_size <= 0:
        args.batch_size = 8
    if args.eval_batch_size <= 0:
        args.eval_batch_size = 2
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    if args.summarize or args.self_check:
        return args
    if args.method is None:
        parser.error("--method is required unless --summarize/--self-check")
    return args


class Meter:
    def __init__(self):
        self.total = 0.0
        self.weight = 0.0

    def add(self, value, weight=1.0):
        if value is None or not math.isfinite(float(value)) or weight <= 0:
            return
        self.total += float(value) * float(weight)
        self.weight += float(weight)

    def mean(self):
        return self.total / self.weight if self.weight else float("nan")


def spike_map(pre: torch.Tensor, module: IF) -> torch.Tensor:
    thresh = float(module.thresh.detach().clamp(min=1e-3).item())
    steps = int(module.T)
    if steps <= 0:
        levels = int(module.L)
        return torch.floor((pre / thresh).clamp(0, 1) * levels + 0.5).clamp(0, levels)
    inner = pre.shape[0] // steps
    timed = pre.view(steps, inner, *pre.shape[1:])
    return torch.round(timed.sum(0) / thresh).clamp(0, steps)


def crossing_rate(clean_pre: torch.Tensor, noisy_pre: torch.Tensor, module: IF) -> float:
    return float((spike_map(clean_pre, module) != spike_map(noisy_pre, module)).float().mean().item())


def binary_margin(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    gt = mask.clamp(0, 1).unsqueeze(1)
    gathered = logits.gather(1, gt)
    rival = logits.gather(1, 1 - gt)
    return (gathered - rival).squeeze(1)


def time_mean(tensor: torch.Tensor, steps: int) -> torch.Tensor:
    if steps <= 0:
        return tensor
    batch = tensor.shape[0] // steps
    return tensor.view(steps, batch, *tensor.shape[1:]).mean(0)


def spatial_rms(delta: torch.Tensor) -> torch.Tensor:
    return delta.float().square().mean(1, keepdim=True).sqrt()


def _install_pre_hooks(modules):
    handles = []

    def _hook(module, inputs, _output):
        module._pre = inputs[0].detach()

    for module in modules:
        handles.append(module.register_forward_hook(_hook))
    return handles


class _FeatTap:
    def __init__(self):
        self.value = None

    def hook(self, _module, _inputs, output):
        self.value = output.detach()


def _hw_map(tensor: torch.Tensor, size) -> torch.Tensor:
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    mapped = F.interpolate(tensor.float(), size=size, mode="bilinear", align_corners=False)
    return mapped.squeeze(0).squeeze(0)


def _mean_valid(tensor: torch.Tensor, valid: torch.Tensor) -> float:
    mapped = tensor if tuple(tensor.shape[-2:]) == tuple(valid.shape[-2:]) else _hw_map(tensor, valid.shape[-2:])
    picked = mapped[valid]
    if picked.numel() == 0:
        return float("nan")
    return float(picked.mean().item())


def _rms_map(clean_feat: torch.Tensor, noisy_feat: torch.Tensor, size) -> torch.Tensor:
    return _hw_map(spatial_rms(noisy_feat - clean_feat), size)


@torch.no_grad()
def probe_encoder_decoder(args, model, loader, device, sigma: float) -> list[dict]:
    model.eval()
    model.set_T(args.eval_t)
    model.set_mode("rate_uniform" if args.eval_t > 0 else "normal")
    enc_ifs = encoder_if_modules(model)
    dec_ifs = decoder_if_modules(model)
    ifs = [module for _name, module in enc_ifs] + [module for _i, _n, module in dec_ifs]
    pre_handles = _install_pre_hooks(ifs)
    enc_tap = _FeatTap()
    dec_tap = _FeatTap()
    stage_taps = [_FeatTap() for _ in model.decoder]
    feat_handles = [
        model.encoder.stage5.register_forward_hook(enc_tap.hook),
        model.decoder.register_forward_hook(dec_tap.hook),
    ]
    feat_handles.extend(block.register_forward_hook(tap.hook) for block, tap in zip(model.decoder, stage_taps))
    meters = defaultdict(lambda: {key: Meter() for key in ("cross", "rms", "margin", "drop", "mtn")})
    n_seen = 0
    n_used = 0
    pix_used = 0
    t0 = time.time()
    try:
        for images, masks, _ids in loader:
            for image, mask in zip(images, masks):
                image = image.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                pair_seed = int(args.eval_seed) * 1_000_003 + n_seen + 17
                seed_all(pair_seed)
                model.set_first_layer_input_noise_sigma(0.0)
                clean_logits = model(image.unsqueeze(0))
                clean_pre = {id(module): module._pre for module in ifs}
                clean_enc = time_mean(enc_tap.value, args.eval_t)
                clean_dec = time_mean(dec_tap.value, args.eval_t)
                clean_stages = [time_mean(tap.value, args.eval_t) for tap in stage_taps]
                seed_all(pair_seed)
                model.set_first_layer_input_noise_sigma(float(sigma))
                noisy_logits = model(image.unsqueeze(0))
                noisy_enc = time_mean(enc_tap.value, args.eval_t)
                noisy_dec = time_mean(dec_tap.value, args.eval_t)
                noisy_stages = [time_mean(tap.value, args.eval_t) for tap in stage_taps]
                n_seen += 1
                valid = mask != IGNORE_INDEX
                if not valid.any():
                    if args.max_probe_images and n_seen >= args.max_probe_images:
                        break
                    continue
                hw = mask.shape[-2:]
                clean_m = binary_margin(clean_logits, mask.unsqueeze(0)).squeeze(0)
                drop = clean_m - binary_margin(noisy_logits, mask.unsqueeze(0)).squeeze(0)
                groups = [
                    ("encoder", [module for _n, module in enc_ifs], _rms_map(clean_enc, noisy_enc, hw)),
                    ("decoder", [module for _i, _n, module in dec_ifs], _rms_map(clean_dec, noisy_dec, hw)),
                ]
                by_stage = defaultdict(list)
                for index, _name, module in dec_ifs:
                    by_stage[index].append(module)
                for index, mods in sorted(by_stage.items()):
                    groups.append(
                        (f"decoder.{index}", mods, _rms_map(clean_stages[index], noisy_stages[index], hw))
                    )
                pix = int(valid.sum().item())
                for name, mods, rms_map in groups:
                    cross = sum(crossing_rate(clean_pre[id(module)], module._pre, module) for module in mods) / max(
                        1, len(mods)
                    )
                    rms = _mean_valid(rms_map, valid)
                    margin = _mean_valid(clean_m, valid)
                    drop_v = _mean_valid(drop, valid)
                    mtn = margin / rms if rms is not None and math.isfinite(rms) and rms > 1e-8 else float("nan")
                    bucket = meters[name]
                    bucket["cross"].add(cross)
                    bucket["rms"].add(rms)
                    bucket["margin"].add(margin)
                    bucket["drop"].add(drop_v)
                    bucket["mtn"].add(mtn)
                n_used += 1
                pix_used += pix
                if args.max_probe_images and n_seen >= args.max_probe_images:
                    break
            if args.max_probe_images and n_seen >= args.max_probe_images:
                break
    finally:
        for handle in pre_handles + feat_handles:
            handle.remove()
        model.set_first_layer_input_noise_sigma(0.0)
    rows = []
    for name in sorted(meters):
        bucket = meters[name]
        rows.append(
            {
                "method": args.method,
                "seed": args.seed,
                "eval_seed": args.eval_seed,
                "sigma": f"{sigma:g}",
                "group": name,
                "n_images": n_used,
                "n_pixels": pix_used,
                "p_cross": f"{bucket['cross'].mean():.6f}",
                "feat_rms": f"{bucket['rms'].mean():.8g}",
                "margin_clean": f"{bucket['margin'].mean():.6f}",
                "margin_drop": f"{bucket['drop'].mean():.6f}",
                "margin_to_noise": f"{bucket['mtn'].mean():.8g}",
                "seconds": f"{time.time() - t0:.1f}",
            }
        )
    return rows


def evaluate_ckpt(args, spec: dict, device, ckpt: Path) -> dict:
    pin = device.type == "cuda"
    _train, test_loader, n_train, n_test = base.pet_loaders(args, pin)
    model, _ = base.make_model("seg", args.seed, device, load_imagenet=False, head_if=False)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        if state.get("reg_coeff") is not None:
            spec["reg_coeff"] = state["reg_coeff"]
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    ann = base.evaluate_task(args, model, test_loader, device, 0.0, 0, "normal")
    print(json.dumps({"ann_T0": ann}), flush=True)
    (out / "ann_test.json").write_text(json.dumps(ann, indent=2) + "\n")
    rows = []
    for sigma in base.SIGMAS:
        row = base.evaluate_task(args, model, test_loader, device, float(sigma), args.eval_t, "rate_uniform")
        rows.append(row)
        print(json.dumps(row), flush=True)
    base.write_csv(out / "test_sweep.csv", rows)
    probe_rows = []
    if not args.skip_probe:
        for sigma in PROBE_SIGMAS:
            print(json.dumps({"phase": "probe", "sigma": sigma}), flush=True)
            probe_rows.extend(probe_encoder_decoder(args, model, test_loader, device, float(sigma)))
        if probe_rows:
            base.write_csv(out / "encoder_decoder_probe.csv", probe_rows)
    card = {
        "method": args.method,
        "label": spec["label"],
        "task": "seg",
        "arch": base.ARCH,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "quant_level": base.LVAL,
        "eval_T": args.eval_t,
        "readout": "linear",
        "regularizer": spec["regularizer"],
        "weight_decay": spec["weight_decay"],
        "reg_coeff": spec["reg_coeff"],
        "noise_position": "post_input_if",
        "if_mode": "rate_uniform",
        "n_train": n_train,
        "n_test": n_test,
        "n_probe": 0 if args.skip_probe else args.max_probe_images,
        "checkpoint": str(ckpt),
        "ann_fg_iou": float(ann["fg_iou"]),
        "ann_miou": float(ann["mIoU"]),
        "clean_fg_iou": base.metric_at(rows, 0.0, "fg_iou"),
        "clean_miou": base.metric_at(rows, 0.0, "mIoU"),
        "fg_iou_s3": base.metric_at(rows, 3.0, "fg_iou"),
        "fg_iou_s5": base.metric_at(rows, 5.0, "fg_iou"),
        "miou_s3": base.metric_at(rows, 3.0, "mIoU"),
        "miou_s5": base.metric_at(rows, 5.0, "mIoU"),
        "auc_high_fg": base.auc_range(rows, HIGH_NOISE_MIN, 5.0, "fg_iou"),
        "auc_high_miou": base.auc_range(rows, HIGH_NOISE_MIN, 5.0, "mIoU"),
        "test_auc_high": base.auc_range(rows, HIGH_NOISE_MIN, 5.0, "fg_iou"),
        "conversion_gap_fg": float(ann["fg_iou"]) - base.metric_at(rows, 0.0, "fg_iou"),
        **base.mapping_card(model),
    }
    (out / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2), flush=True)
    return card


def _mean_std(values: list[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan"), float("nan")
    mean = sum(finite) / len(finite)
    if len(finite) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return mean, math.sqrt(var)


def _fmt(mean: float, std: float) -> str:
    if not math.isfinite(mean):
        return "nan"
    return f"{mean:6.2f}±{std:.2f}"


def summarize_probe(out_root: Path) -> None:
    grouped = defaultdict(list)
    for method in METHODS:
        for path in sorted((out_root / method).glob("seed*/encoder_decoder_probe.csv")):
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    key = (row["method"], row["group"], f"{float(row['sigma']):g}")
                    grouped[key].append(row)
    if not grouped:
        return
    rows = []
    print(f"{'method':<10} {'group':<10} {'σ':>3} {'n':>3} {'p_cross':>12} {'m2n':>12} {'rms':>12}")
    for key in sorted(grouped, key=lambda item: (item[0], float(item[2]), item[1])):
        method, group, sigma = key
        items = grouped[key]
        p_mean, p_std = _mean_std([row["p_cross"] for row in items])
        m_mean, m_std = _mean_std([row["margin_to_noise"] for row in items])
        r_mean, r_std = _mean_std([row["feat_rms"] for row in items])
        rows.append(
            {
                "method": method,
                "group": group,
                "sigma": sigma,
                "n_seeds": len(items),
                "p_cross_mean": p_mean,
                "p_cross_std": p_std,
                "margin_to_noise_mean": m_mean,
                "margin_to_noise_std": m_std,
                "feat_rms_mean": r_mean,
                "feat_rms_std": r_std,
            }
        )
        print(
            f"{method:<10} {group:<10} {sigma:>3} {len(items):3d} "
            f"{_fmt(100.0 * p_mean, 100.0 * p_std)} {_fmt(m_mean, m_std)} {_fmt(r_mean, r_std)}"
        )
    base.write_csv(out_root / "probe_mean_std.csv", rows)
    print(f"Wrote {out_root / 'probe_mean_std.csv'}")


def summarize(out_root: Path) -> None:
    rows = []
    print(f"{'method':<10} {'n':>3} {'fg0':>12} {'mIoU0':>12} {'fg3':>12} {'fg5':>12} {'AUCh':>12}")
    for method in METHODS:
        cards = [json.loads(path.read_text()) for path in sorted((out_root / method).glob("seed*/scorecard.json"))]
        if not cards:
            continue
        stats = {key: _mean_std([float(card[key]) for card in cards]) for key in SCORE_KEYS}
        row = {"method": method, "n_seeds": len(cards)}
        for key, (mean, std) in stats.items():
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
        rows.append(row)
        print(
            f"{method:<10} {len(cards):3d} "
            f"{_fmt(*stats['clean_fg_iou'])} {_fmt(*stats['clean_miou'])} "
            f"{_fmt(*stats['fg_iou_s3'])} {_fmt(*stats['fg_iou_s5'])} "
            f"{_fmt(*stats['auc_high_fg'])}"
        )
    if rows:
        base.write_csv(out_root / "summary_mean_std.csv", rows)
        print(f"Wrote {out_root / 'summary_mean_std.csv'}")
    summarize_probe(out_root)


def self_check(device) -> dict:
    model, _ = base.make_model("seg", 0, device, load_imagenet=False, head_if=False)
    model.eval()
    enc = encoder_if_modules(model)
    dec = decoder_if_modules(model)
    dummy = torch.zeros(1, 3, 64, 64, device=device)
    logits = model(dummy)
    margin = binary_margin(logits, torch.zeros(1, 64, 64, dtype=torch.long, device=device))
    module = enc[0][1]
    module.T = 2
    pre = torch.zeros(2, 1, 4, 4)
    pre[0, 0, 0, 0] = float(module.thresh.detach())
    same = crossing_rate(pre, pre.clone(), module)
    shifted = pre.clone()
    shifted[0, 0, 0, 0] = 0.0
    changed = crossing_rate(pre, shifted, module)
    card = {
        "n_encoder_if": len(enc),
        "n_decoder_if": len(dec),
        "decoder_stages": sorted({index for index, _n, _m in dec}),
        "logits_shape": list(logits.shape),
        "margin_shape": list(margin.shape),
        "identical_crossing": same,
        "shifted_crossing": changed,
        "layout": str(cfg_dir(argparse.Namespace(out_root=Path("/tmp"), method="mne", seed=40))),
    }
    assert card["n_encoder_if"] == 13, card
    assert card["n_decoder_if"] == 5, card
    assert card["decoder_stages"] == [0, 1, 2, 3, 4], card
    assert same == 0.0, card
    assert changed > 0.0, card
    assert card["layout"].endswith("mne/seed40"), card
    print(json.dumps(card, indent=2), flush=True)
    return card


def main() -> None:
    args = parse_args()
    device = get_torch_device(args.device)
    if args.self_check:
        self_check(device)
        return
    if args.summarize:
        summarize(args.out_root)
        return
    if not pet_is_ready(args.pet_root):
        raise SystemExit(
            f"Oxford-IIIT Pet missing under {args.pet_root}. "
            "Download once to /scratch/gs14/sl9144/datasets."
        )
    install_output_layout()
    spec = base.method_spec(args.method)
    args.out_root.mkdir(parents=True, exist_ok=True)
    ckpt = base.train_one(args, spec, device)
    evaluate_ckpt(args, spec, device, ckpt)
    print(f"Wrote {cfg_dir(args)}", flush=True)


if __name__ == "__main__":
    main()
