#!/usr/bin/env python3
"""DeepLabV3-ResNet50 AvgPool probes: residual splice, ASPP restore, pixel groups.

Loads the already-trained AvgPool QCFS checkpoints. No retraining.
Paired clean/noisy forwards at T=16 rate_uniform, post_input_if.

    residual   ||ΔF||², ||ΔS||², 2⟨ΔF,ΔS⟩, post-add IF crossing;
               splice cached clean F or S and re-score mIoU / margin.
    aspp       per-branch energy; restore one clean branch at a time.
    pixel      crossing, feature RMS, logit-margin drop on boundary /
               interior, class, and clean-margin bins.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_voc_deeplabv3_resnet50_mne_seed42 as runner  # noqa: E402
from Models.FCN import IGNORE_INDEX, NUM_CLASSES, VOC_SEG_CLASSES  # noqa: E402
from Models.layer import IF  # noqa: E402
from voc_seg import (  # noqa: E402
    confusion_update,
    pad_to_stride,
    scores_from_confusion,
    voc2012_seg_is_ready,
)
from utils import get_torch_device, seed_all  # noqa: E402

METHODS = ("original", "l2wo", "mne", "nodetach")
ASPP_NAMES = ("b1x1", "d12", "d24", "d36", "pool")
MARGIN_BINS = (("low", 0.0, 1.0), ("mid", 1.0, 3.0), ("high", 3.0, math.inf))
DEFAULT_SIGMAS = (0.5, 1.0, 1.5)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--seed", type=int, default=runner.SEED)
    parser.add_argument("--stem-pool", choices=("max", "avg"), default=os.environ.get("VOC_STEM_POOL", "avg"))
    parser.add_argument("--eval-t", type=int, default=int(os.environ.get("VOC_EVAL_T", "16")))
    parser.add_argument("--max-images", type=int, default=int(os.environ.get("VOC_PROBE_IMAGES", "200")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("VOC_NUM_WORKERS", "4")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--sigma-min", type=float, default=runner._env_float("VOC_SIGMA_MIN"))
    parser.add_argument("--sigma-max", type=float, default=runner._env_float("VOC_SIGMA_MAX"))
    parser.add_argument("--sigma-step", type=float, default=runner._env_float("VOC_SIGMA_STEP"))
    parser.add_argument("--sigmas", default=os.environ.get("VOC_SIGMAS", ""))
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=Path(os.environ.get("CKPT_ROOT", str(ROOT.parent / "important_results" / "voc_deeplabv3_resnet50_avgpool_mne_seed42"))),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(os.environ.get("OUT_DIR", str(ROOT.parent / "important_results" / "voc_deeplabv3_resnet50_avgpool_probe_seed42"))),
    )
    parser.add_argument(
        "--voc-root",
        type=Path,
        default=Path(os.environ.get("VOC_ROOT", os.environ.get("CIFAR_ROOT", "~/datasets"))),
    )
    args = parser.parse_args()
    args.voc_root = Path(os.path.expanduser(str(args.voc_root)))
    args.eval_t = max(0, int(args.eval_t))
    args.max_images = max(0, int(args.max_images))
    args.stem_pool = str(args.stem_pool).strip().lower()
    if args.sigma_step is not None:
        args.sigmas = runner.sigma_grid(
            0.5 if args.sigma_min is None else float(args.sigma_min),
            1.5 if args.sigma_max is None else float(args.sigma_max),
            float(args.sigma_step),
        )
    elif str(args.sigmas).strip():
        args.sigmas = runner.parse_sigmas(args.sigmas)
    else:
        args.sigmas = DEFAULT_SIGMAS
    for key in ("ckpt_root", "out_root"):
        path = getattr(args, key)
        if not path.is_absolute():
            setattr(args, key, (ROOT / path).resolve())
    if args.self_check:
        return args
    if args.method is None:
        parser.error("--method is required unless --self-check")
    return args


def mean_sq(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().square().mean().item())


def cross_term(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((2.0 * left.detach().float() * right.detach().float()).mean().item())


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
    clean = spike_map(clean_pre, module)
    noisy = spike_map(noisy_pre, module)
    return float((clean != noisy).float().mean().item())


def boundary_mask(mask: torch.Tensor) -> torch.Tensor:
    valid = mask != IGNORE_INDEX
    padded = F.pad(mask.view(1, 1, *mask.shape).float(), (1, 1, 1, 1), value=float(IGNORE_INDEX))
    center = padded[:, :, 1:-1, 1:-1]
    neigh = torch.cat(
        (
            padded[:, :, :-2, 1:-1],
            padded[:, :, 2:, 1:-1],
            padded[:, :, 1:-1, :-2],
            padded[:, :, 1:-1, 2:],
        ),
        dim=1,
    )
    edge = (neigh != center).any(1, keepdim=True) & (center != float(IGNORE_INDEX))
    kernel = torch.ones(1, 1, 3, 3, device=mask.device)
    dilated = F.conv2d(edge.float(), kernel, padding=1) > 0
    return dilated.squeeze(0).squeeze(0) & valid


def logit_margin(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    gt = mask.clamp(0, NUM_CLASSES - 1)
    gathered = logits.gather(1, gt.unsqueeze(0).unsqueeze(0)).squeeze(1)
    fill = torch.full_like(logits, -1e9)
    fill.scatter_(1, gt.unsqueeze(0).unsqueeze(0), -1e9)
    rival = fill.max(1).values
    return gathered.squeeze(0) - rival.squeeze(0)


def upsample_map(tensor: torch.Tensor, size, mode: str = "bilinear") -> torch.Tensor:
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[-2:] != size:
        kwargs = {} if mode == "nearest" else {"align_corners": False}
        tensor = F.interpolate(tensor.float(), size=size, mode=mode, **kwargs)
    while tensor.dim() > 2:
        tensor = tensor.squeeze(0)
    return tensor


def _branch_if(branch: nn.Module) -> IF:
    ifs = [module for module in branch.modules() if isinstance(module, IF)]
    if not ifs:
        raise RuntimeError("ASPP branch has no IF")
    return ifs[-1]


def _project_if(aspp: nn.Module) -> IF:
    ifs = [module for module in aspp.project.modules() if isinstance(module, IF)]
    if not ifs:
        raise RuntimeError("ASPP project has no IF")
    return ifs[-1]


def iter_bottlenecks(model):
    rows = []
    for name, module in model.backbone.named_modules():
        if hasattr(module, "if3") and hasattr(module, "conv3") and hasattr(module, "bn3"):
            rows.append((name, module))
    return rows


def aspp_module(model):
    head = model.classifier[0]
    if not hasattr(head, "convs") or not hasattr(head, "project"):
        raise RuntimeError("DeepLab classifier[0] is not ASPP")
    return head


def wrap_bottleneck(block: nn.Module) -> None:
    def forward(self, x):
        identity = x
        out = self.if1(self.bn1(self.conv1(x)))
        out = self.if2(self.bn2(self.conv2(out)))
        residual = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        if getattr(self, "_use_clean_F", False):
            residual = self._clean_F
        if getattr(self, "_use_clean_S", False):
            identity = self._clean_S
        post = residual + identity
        y = self.if3(post)
        if getattr(self, "_record", False):
            self._cache = {
                "F": residual.detach(),
                "S": identity.detach(),
                "post": post.detach(),
            }
        return y

    block.forward = types.MethodType(forward, block)
    block._record = False
    block._use_clean_F = False
    block._use_clean_S = False


def wrap_aspp(module: nn.Module) -> None:
    def _capture_pre(if_module, inputs, _output):
        if_module._pre = inputs[0].detach()

    for conv in module.convs:
        _branch_if(conv).register_forward_hook(_capture_pre)
    _project_if(module).register_forward_hook(_capture_pre)

    def forward(self, x):
        branches = []
        restore = getattr(self, "_use_clean_branch", None)
        for index, conv in enumerate(self.convs):
            if restore is not None and index == restore:
                branches.append(self._clean_branches[index])
                continue
            branches.append(conv(x))
        cat = torch.cat(branches, dim=1)
        y = self.project(cat)
        if getattr(self, "_record", False):
            self._cache = {
                "branches": [item.detach() for item in branches],
                "out": y.detach(),
            }
        return y

    module.forward = types.MethodType(forward, module)
    module._record = False
    module._use_clean_branch = None


def set_recording(blocks, aspp, enabled: bool) -> None:
    for _name, block in blocks:
        block._record = enabled
    aspp._record = enabled


def clear_splice(blocks, aspp) -> None:
    for _name, block in blocks:
        block._use_clean_F = False
        block._use_clean_S = False
    aspp._use_clean_branch = None


def store_clean(blocks, aspp) -> None:
    for _name, block in blocks:
        block._clean_F = block._cache["F"]
        block._clean_S = block._cache["S"]
    aspp._clean_branches = list(aspp._cache["branches"])


def free_clean(blocks, aspp) -> None:
    for _name, block in blocks:
        for key in ("_clean_F", "_clean_S", "_cache"):
            if hasattr(block, key):
                delattr(block, key)
    for key in ("_clean_branches", "_cache"):
        if hasattr(aspp, key):
            delattr(aspp, key)


def load_probe_model(args, device):
    holder = argparse.Namespace(
        method=args.method,
        seed=args.seed,
        out_root=args.ckpt_root,
        stem_pool=args.stem_pool,
        retrain=False,
        test_only=True,
    )
    ckpt = runner.ckpt_path(holder)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    model = runner.make_model(args.seed, device, load_coco=False, stem_pool=args.stem_pool)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        saved = str(state.get("stem_pool", "max"))
        if saved != args.stem_pool:
            raise ValueError(f"checkpoint stem_pool={saved} != {args.stem_pool}")
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    model.set_T(args.eval_t)
    model.set_mode("rate_uniform" if args.eval_t > 0 else "normal")
    model.set_first_layer_input_noise_position("post_input_if")
    blocks = iter_bottlenecks(model)
    if len(blocks) != 16:
        raise RuntimeError(f"expected 16 bottlenecks, got {len(blocks)}")
    for _name, block in blocks:
        wrap_bottleneck(block)
    aspp = aspp_module(model)
    wrap_aspp(aspp)
    return model, ckpt, blocks, aspp


def forward_logits(model, image, sigma: float):
    model.set_first_layer_input_noise_sigma(float(sigma))
    padded, height, width = pad_to_stride(image)
    logits = model(padded.unsqueeze(0))
    return logits[:, :, :height, :width], height, width


def aligned_logits(logits, mask):
    if logits.shape[-2:] != mask.shape[-2:]:
        logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def scores_of(logits, mask, device):
    pred = logits.argmax(1)
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    confusion_update(conf, pred, mask.unsqueeze(0))
    scores = scores_from_confusion(conf)
    margin = logit_margin(logits, mask)
    valid = mask != IGNORE_INDEX
    mean_margin = float(margin[valid].mean().item()) if valid.any() else float("nan")
    return scores, mean_margin, pred, margin, valid


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def probe_one_sigma(args, model, loader, device, blocks, aspp, sigma: float) -> dict:
    residual_energy = {name: {k: Meter() for k in ("dF2", "dS2", "cross", "post2", "p_cross")} for name, _ in blocks}
    residual_conf = {(name, which): torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device) for name, _ in blocks for which in ("F", "S")}
    residual_margin = {(name, which): Meter() for name, _ in blocks for which in ("F", "S")}
    aspp_energy = {name: {k: Meter() for k in ("d2", "p_cross")} for name in ASPP_NAMES}
    aspp_proj = {"d2": Meter(), "p_cross": Meter()}
    aspp_conf = {name: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device) for name in ASPP_NAMES}
    aspp_margin = {name: Meter() for name in ASPP_NAMES}
    base_conf = {key: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device) for key in ("clean", "noisy")}
    base_margin = {key: Meter() for key in ("clean", "noisy")}
    pixel = defaultdict(lambda: {k: Meter() for k in ("p_cross", "feat_rms", "margin_clean", "margin_drop", "acc")})
    proj_if = _project_if(aspp)
    n_images = 0
    t0 = time.time()
    for batch in loader:
        for image, mask, image_id, _hw in batch:
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            pair_seed = args.seed * 100_003 + n_images + 17

            seed_all(pair_seed)
            set_recording(blocks, aspp, True)
            clear_splice(blocks, aspp)
            model.set_first_layer_input_noise_sigma(0.0)
            clean_logits, _, _ = forward_logits(model, image, 0.0)
            store_clean(blocks, aspp)
            clean_aspp = aspp._cache["out"]
            clean_post = {name: block._cache["post"] for name, block in blocks}
            clean_branches = list(aspp._cache["branches"])
            clean_branch_pre = [_branch_if(aspp.convs[i])._pre for i in range(len(ASPP_NAMES))]
            clean_proj_pre = proj_if._pre

            seed_all(pair_seed)
            noisy_logits, _, _ = forward_logits(model, image, sigma)
            noisy_aspp = aspp._cache["out"]
            noisy_post = {name: block._cache["post"] for name, block in blocks}
            noisy_branches = list(aspp._cache["branches"])
            noisy_branch_pre = [_branch_if(aspp.convs[i])._pre for i in range(len(ASPP_NAMES))]
            noisy_proj_pre = proj_if._pre
            set_recording(blocks, aspp, False)

            clean_logits = aligned_logits(clean_logits, mask)
            noisy_logits = aligned_logits(noisy_logits, mask)
            clean_scores, clean_m, _pred_c, clean_margin, valid = scores_of(clean_logits, mask, device)
            noisy_scores, noisy_m, pred_n, noisy_margin, _valid = scores_of(noisy_logits, mask, device)
            confusion_update(base_conf["clean"], clean_logits.argmax(1), mask.unsqueeze(0))
            confusion_update(base_conf["noisy"], pred_n, mask.unsqueeze(0))
            base_margin["clean"].add(clean_m, int(valid.sum()))
            base_margin["noisy"].add(noisy_m, int(valid.sum()))

            for name, block in blocks:
                dF = block._cache["F"] - block._clean_F
                dS = block._cache["S"] - block._clean_S
                residual_energy[name]["dF2"].add(mean_sq(dF))
                residual_energy[name]["dS2"].add(mean_sq(dS))
                residual_energy[name]["cross"].add(cross_term(dF, dS))
                residual_energy[name]["post2"].add(mean_sq(dF + dS))
                residual_energy[name]["p_cross"].add(crossing_rate(clean_post[name], noisy_post[name], block.if3))

            for index, name in enumerate(ASPP_NAMES):
                delta = noisy_branches[index] - clean_branches[index]
                aspp_energy[name]["d2"].add(mean_sq(delta))
                aspp_energy[name]["p_cross"].add(
                    crossing_rate(clean_branch_pre[index], noisy_branch_pre[index], _branch_if(aspp.convs[index]))
                )
            aspp_proj["d2"].add(mean_sq(noisy_aspp - clean_aspp))
            aspp_proj["p_cross"].add(crossing_rate(clean_proj_pre, noisy_proj_pre, proj_if))

            feat_rms = (noisy_aspp - clean_aspp).float().square().mean(1, keepdim=True).sqrt()
            feat_rms = upsample_map(feat_rms, mask.shape[-2:], mode="bilinear")
            last_cross = (spike_map(clean_proj_pre, proj_if) != spike_map(noisy_proj_pre, proj_if)).float().mean(1, keepdim=True)
            last_cross = upsample_map(last_cross, mask.shape[-2:], mode="nearest")
            drop = clean_margin - noisy_margin
            acc = (pred_n.squeeze(0) == mask) & valid
            edge = boundary_mask(mask)
            interior = valid & ~edge
            _add_pixel(pixel, "region", "boundary", edge, last_cross, feat_rms, clean_margin, drop, acc)
            _add_pixel(pixel, "region", "interior", interior, last_cross, feat_rms, clean_margin, drop, acc)
            for cls in range(NUM_CLASSES):
                sel = valid & (mask == cls)
                _add_pixel(pixel, "class", VOC_SEG_CLASSES[cls], sel, last_cross, feat_rms, clean_margin, drop, acc)
            for bin_name, lo, hi in MARGIN_BINS:
                sel = valid & (clean_margin >= lo) & (clean_margin < hi)
                _add_pixel(pixel, "clean_margin", bin_name, sel, last_cross, feat_rms, clean_margin, drop, acc)

            for name, block in blocks:
                for which, flag in (("F", "_use_clean_F"), ("S", "_use_clean_S")):
                    clear_splice(blocks, aspp)
                    setattr(block, flag, True)
                    seed_all(pair_seed)
                    logits, _, _ = forward_logits(model, image, sigma)
                    logits = aligned_logits(logits, mask)
                    scores, mean_m, pred, _margin, _v = scores_of(logits, mask, device)
                    confusion_update(residual_conf[(name, which)], pred, mask.unsqueeze(0))
                    residual_margin[(name, which)].add(mean_m, int(valid.sum()))
            for index, name in enumerate(ASPP_NAMES):
                clear_splice(blocks, aspp)
                aspp._use_clean_branch = index
                seed_all(pair_seed)
                logits, _, _ = forward_logits(model, image, sigma)
                logits = aligned_logits(logits, mask)
                scores, mean_m, pred, _margin, _v = scores_of(logits, mask, device)
                confusion_update(aspp_conf[name], pred, mask.unsqueeze(0))
                aspp_margin[name].add(mean_m, int(valid.sum()))

            clear_splice(blocks, aspp)
            free_clean(blocks, aspp)
            n_images += 1
            if n_images % 10 == 0:
                print(json.dumps({"sigma": sigma, "n_images": n_images, "image_id": image_id}), flush=True)
            if args.max_images and n_images >= args.max_images:
                break
        if args.max_images and n_images >= args.max_images:
            break

    clean = scores_from_confusion(base_conf["clean"])
    noisy = scores_from_confusion(base_conf["noisy"])
    residual_rows, residual_iv = [], []
    for name, _block in blocks:
        row = {
            "method": args.method,
            "sigma": sigma,
            "block": name,
            "n_images": n_images,
            "dF2": f"{residual_energy[name]['dF2'].mean():.8g}",
            "dS2": f"{residual_energy[name]['dS2'].mean():.8g}",
            "cross": f"{residual_energy[name]['cross'].mean():.8g}",
            "post2": f"{residual_energy[name]['post2'].mean():.8g}",
            "p_cross_if3": f"{residual_energy[name]['p_cross'].mean():.6f}",
        }
        residual_rows.append(row)
        for which in ("F", "S"):
            scores = scores_from_confusion(residual_conf[(name, which)])
            residual_iv.append(
                {
                    "method": args.method,
                    "sigma": sigma,
                    "block": name,
                    "replace": which,
                    "n_images": n_images,
                    "mIoU": f"{scores['mIoU']:.6f}",
                    "pixel_acc": f"{scores['pixel_acc']:.6f}",
                    "margin": f"{residual_margin[(name, which)].mean():.6f}",
                    "clean_mIoU": f"{clean['mIoU']:.6f}",
                    "noisy_mIoU": f"{noisy['mIoU']:.6f}",
                    "clean_margin": f"{base_margin['clean'].mean():.6f}",
                    "noisy_margin": f"{base_margin['noisy'].mean():.6f}",
                }
            )
    aspp_rows, aspp_iv = [], []
    for name in ASPP_NAMES:
        aspp_rows.append(
            {
                "method": args.method,
                "sigma": sigma,
                "branch": name,
                "n_images": n_images,
                "d2": f"{aspp_energy[name]['d2'].mean():.8g}",
                "p_cross": f"{aspp_energy[name]['p_cross'].mean():.6f}",
            }
        )
        scores = scores_from_confusion(aspp_conf[name])
        aspp_iv.append(
            {
                "method": args.method,
                "sigma": sigma,
                "restore": name,
                "n_images": n_images,
                "mIoU": f"{scores['mIoU']:.6f}",
                "pixel_acc": f"{scores['pixel_acc']:.6f}",
                "margin": f"{aspp_margin[name].mean():.6f}",
                "clean_mIoU": f"{clean['mIoU']:.6f}",
                "noisy_mIoU": f"{noisy['mIoU']:.6f}",
                "clean_margin": f"{base_margin['clean'].mean():.6f}",
                "noisy_margin": f"{base_margin['noisy'].mean():.6f}",
            }
        )
    aspp_rows.append(
        {
            "method": args.method,
            "sigma": sigma,
            "branch": "project",
            "n_images": n_images,
            "d2": f"{aspp_proj['d2'].mean():.8g}",
            "p_cross": f"{aspp_proj['p_cross'].mean():.6f}",
        }
    )
    pixel_rows = []
    for (kind, group), meters in sorted(pixel.items()):
        pixel_rows.append(
            {
                "method": args.method,
                "sigma": sigma,
                "group_type": kind,
                "group": group,
                "n_images": n_images,
                "n_pixels": int(meters["p_cross"].weight),
                "p_cross": f"{meters['p_cross'].mean():.6f}",
                "feat_rms": f"{meters['feat_rms'].mean():.8g}",
                "margin_clean": f"{meters['margin_clean'].mean():.6f}",
                "margin_drop": f"{meters['margin_drop'].mean():.6f}",
                "acc": f"{meters['acc'].mean():.6f}",
            }
        )
    summary = {
        "method": args.method,
        "sigma": sigma,
        "n_images": n_images,
        "seconds": round(time.time() - t0, 1),
        "clean_mIoU": clean["mIoU"],
        "noisy_mIoU": noisy["mIoU"],
        "clean_margin": base_margin["clean"].mean(),
        "noisy_margin": base_margin["noisy"].mean(),
    }
    return {
        "residual_energy": residual_rows,
        "residual_intervene": residual_iv,
        "aspp_energy": aspp_rows,
        "aspp_intervene": aspp_iv,
        "pixel_groups": pixel_rows,
        "summary": summary,
    }


def _add_pixel(store, kind, group, select, cross, feat_rms, margin_clean, drop, acc):
    if select.dtype != torch.bool:
        select = select.bool()
    weight = int(select.sum().item())
    if weight <= 0:
        return
    meters = store[(kind, group)]
    meters["p_cross"].add(float(cross[select].mean().item()), weight)
    meters["feat_rms"].add(float(feat_rms[select].mean().item()), weight)
    meters["margin_clean"].add(float(margin_clean[select].mean().item()), weight)
    meters["margin_drop"].add(float(drop[select].mean().item()), weight)
    meters["acc"].add(float(acc[select].float().mean().item()), weight)


def self_check(device) -> dict:
    model = runner.make_model(0, device, load_coco=False, stem_pool="avg")
    model.eval()
    model.set_T(0)
    blocks = iter_bottlenecks(model)
    assert len(blocks) == 16, len(blocks)
    for _name, block in blocks:
        wrap_bottleneck(block)
    aspp = aspp_module(model)
    wrap_aspp(aspp)
    set_recording(blocks, aspp, True)
    dummy = torch.zeros(1, 3, 64, 64, device=device)
    logits = model(dummy)
    assert logits.shape[1] == 21, tuple(logits.shape)
    name, block = blocks[0]
    cache = block._cache
    assert cache["F"].shape == cache["S"].shape
    dF = torch.randn_like(cache["F"])
    dS = torch.randn_like(cache["S"])
    post = mean_sq(dF + dS)
    parts = mean_sq(dF) + mean_sq(dS) + cross_term(dF, dS)
    assert abs(post - parts) < 1e-5, (post, parts)
    assert len(aspp._cache["branches"]) == 5
    n_maxpool = runner.count_maxpool2d(model)
    assert n_maxpool == 0
    card = {
        "n_bottlenecks": len(blocks),
        "n_aspp_branches": len(aspp._cache["branches"]),
        "logits_shape": list(logits.shape),
        "energy_identity_ok": True,
        "n_maxpool": n_maxpool,
        "first_block": name,
    }
    print(json.dumps(card, indent=2), flush=True)
    return card


def main() -> None:
    args = parse_args()
    device = get_torch_device(args.device)
    if args.self_check:
        self_check(device)
        return
    if not voc2012_seg_is_ready(args.voc_root):
        raise SystemExit(f"VOC 2012 segmentation missing under {args.voc_root}")
    holder = argparse.Namespace(
        voc_root=args.voc_root,
        batch_size=1,
        workers=args.workers,
        crop=512,
        seed=args.seed,
    )
    _train, val_loader, split, n_train, n_val = runner.voc_loaders(holder, device.type == "cuda")
    model, ckpt, blocks, aspp = load_probe_model(args, device)
    out = args.out_root / args.method
    out.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "method": args.method,
                "ckpt": str(ckpt),
                "stem_pool": args.stem_pool,
                "sigmas": list(args.sigmas),
                "max_images": args.max_images,
                "eval_T": args.eval_t,
                "split": split,
                "n_val": n_val,
                "n_train": n_train,
            }
        ),
        flush=True,
    )
    buckets = defaultdict(list)
    summaries = []
    for sigma in args.sigmas:
        print(json.dumps({"phase": "sigma", "sigma": sigma}), flush=True)
        result = probe_one_sigma(args, model, val_loader, device, blocks, aspp, float(sigma))
        for key in ("residual_energy", "residual_intervene", "aspp_energy", "aspp_intervene", "pixel_groups"):
            buckets[key].extend(result[key])
        summaries.append(result["summary"])
        print(json.dumps(result["summary"]), flush=True)
    for key, rows in buckets.items():
        write_csv(out / f"{key}.csv", rows)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
