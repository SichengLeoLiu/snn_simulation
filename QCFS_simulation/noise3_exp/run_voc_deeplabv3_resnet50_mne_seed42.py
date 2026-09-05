#!/usr/bin/env python3
"""VOC2012 val DeepLabV3-ResNet50: original / L2-wo / MNE detach / MNE no-detach.

Protocol
--------
    original   TorchVision COCO-VOC weights, no weight FT. IF λ from 99.9%
               activations, then the same QCFS eval as the other arms.
    l2wo       10-epoch QCFS FT, SGD WD=5e-4 on Conv weights.
    mne        same FT, MNE-L2 detach λ/γ. β matches ||∇W L2-wo|| at init.
    nodetach   same FT and same frozen β, grads into λ.

MNE is a conversion-time weight preparation, not a change to Bu 2025's
training-free converter. Eval is VOC2012 val (Bu Table 3 DeepLab is COCO).
Train T=0, L=16; eval T=16 rate_uniform, post_input_if. Seed 42.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Models.DeepLab import (  # noqa: E402
    COCO_WEIGHT_NAME,
    build_deeplabv3_resnet50_if,
    cached_deeplab_weight_path,
)
from Models.FCN import IGNORE_INDEX, NUM_CLASSES, init_if_thresholds_percentile  # noqa: E402
from Models.layer import IF  # noqa: E402
from voc_seg import (  # noqa: E402
    CROP,
    VOCSegSet,
    collate_train,
    collate_val,
    confusion_update,
    download_voc,
    pad_to_stride,
    resolve_train_split,
    scores_from_confusion,
    voc2012_seg_is_ready,
)
from voc_ssd import voc_is_ready  # noqa: E402
from utils import (  # noqa: E402
    collect_weight_layer_matches,
    compute_mne_l2_regularization,
    dump_mne_mapping_report,
    get_torch_device,
    seed_all,
    summarize_weight_layer_matches,
)

ARCH = "deeplabv3_resnet50"
SEED = 42
LVAL = 16
TRAIN_T = 0
TEST_T = 16
L2_WD = 5e-4
EPOCHS = 10
LR = 1e-4
MILESTONES = (6, 8)
SIGMAS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)
HIGH_NOISE_MIN = 3.0
METHODS = ("original", "l2wo", "mne", "nodetach")
PERCENTILE = 99.9
PERCENTILE_IMAGES = 64
LAYER_MAP = "resnet"

DETACH = dict(
    detach_lambda=True,
    detach_bn_stats=True,
    detach_bn_affine=True,
    divide_by_lambda=True,
    scale_by_l=True,
    l_ref=None,
    fold_bn=True,
    layer_map=LAYER_MAP,
)
NODETACH = dict(DETACH, detach_lambda=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("VOC_EPOCHS", str(EPOCHS))))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("VOC_BATCH", "1")))
    parser.add_argument("--accum", type=int, default=int(os.environ.get("VOC_ACCUM", "8")))
    parser.add_argument("--eval-batch-size", type=int, default=int(os.environ.get("VOC_EVAL_BATCH", "1")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("VOC_NUM_WORKERS", "4")))
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--crop", type=int, default=CROP)
    parser.add_argument("--eval-t", type=int, default=int(os.environ.get("VOC_EVAL_T", str(TEST_T))))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--download-voc", action="store_true")
    parser.add_argument("--max-eval-images", type=int, default=0)
    parser.add_argument(
        "--voc-root",
        type=Path,
        default=Path(os.environ.get("VOC_ROOT", os.environ.get("CIFAR_ROOT", "~/datasets"))),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "voc_deeplabv3_resnet50_mne_seed42",
    )
    args = parser.parse_args()
    args.voc_root = Path(os.path.expanduser(str(args.voc_root)))
    args.accum = max(1, int(args.accum))
    args.eval_t = max(0, int(args.eval_t))
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    if args.summarize or args.self_check or args.download_voc or args.dry_run:
        return args
    if args.method is None:
        parser.error("--method is required unless --summarize/--self-check/--download-voc/--dry-run")
    return args


def method_spec(method: str) -> dict:
    if method == "original":
        return {
            "label": "original COCO-VOC, no FT",
            "regularizer": None,
            "weight_decay": 0.0,
            "reg_coeff": None,
            "mne_kw": None,
        }
    if method == "l2wo":
        return {
            "label": "L2-wo",
            "regularizer": "weight_decay_weights_only",
            "weight_decay": L2_WD,
            "reg_coeff": None,
            "mne_kw": None,
        }
    table = {
        "mne": ("MNE-L2 detach λ", DETACH),
        "nodetach": ("MNE-L2 no-detach λ", NODETACH),
    }
    label, kw = table[method]
    return {
        "label": label,
        "regularizer": "mne_l2",
        "weight_decay": 0.0,
        "reg_coeff": None,
        "mne_kw": dict(kw),
    }


def cfg_dir(args) -> Path:
    return args.out_root / args.method


def ckpt_path(args) -> Path:
    return (
        cfg_dir(args)
        / "checkpoints"
        / f"{ARCH}_L[{LVAL}]_{args.method}_seed{args.seed}_L{LVAL}_trainT{TRAIN_T}.pth"
    )


def make_model(seed: int, device, load_coco: bool = True):
    seed_all(seed)
    model = build_deeplabv3_resnet50_if(load_coco=load_coco)
    model._mne_layer_map = LAYER_MAP
    model.set_L(LVAL)
    model.set_T(TRAIN_T)
    model.set_mode("normal")
    model.set_first_layer_input_noise_position("post_input_if")
    model.set_first_layer_input_noise_type("gaussian")
    model.set_first_layer_input_noise_sigma(0.0)
    return model.to(device)


def matched_weight_beta(model, wd: float = L2_WD) -> dict:
    rows = [row for row in collect_weight_layer_matches(model, LAYER_MAP) if row["matched"]]
    weights = [row["weight"] for row in rows]
    if not weights:
        raise ValueError("No IF-matched Conv layers for β matching")
    penalty = compute_mne_l2_regularization(model, quant_level=LVAL, **DETACH)
    grads = torch.autograd.grad(penalty, weights, allow_unused=False)
    mne_norm = math.sqrt(sum(float(grad.detach().square().sum()) for grad in grads))
    l2_norm = float(wd) * math.sqrt(sum(float(weight.detach().square().sum()) for weight in weights))
    if not math.isfinite(mne_norm) or mne_norm <= 0 or not math.isfinite(l2_norm):
        raise ValueError(f"Invalid matched-layer gradient norms: MNE={mne_norm} L2={l2_norm}")
    return {
        "beta_match": l2_norm / mne_norm,
        "unit_mne_grad_norm": mne_norm,
        "reference_l2_grad_norm": l2_norm,
        "n_matched": len(rows),
        "weight_decay": wd,
    }


def optimizer_for(model, spec: dict, lr: float):
    matches = collect_weight_layer_matches(model, LAYER_MAP)
    if spec["regularizer"] == "weight_decay_weights_only":
        decay_ids = {id(row["weight"]) for row in matches}
        decay, no_decay = [], []
        for parameter in model.parameters():
            (decay if id(parameter) in decay_ids else no_decay).append(parameter)
        params = [
            {"params": decay, "weight_decay": spec["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ]
    elif spec["regularizer"] == "mne_l2":
        head_ids = {id(row["weight"]) for row in matches if not row["matched"]}
        decay, no_decay = [], []
        for parameter in model.parameters():
            (decay if id(parameter) in head_ids else no_decay).append(parameter)
        params = [
            {"params": decay, "weight_decay": L2_WD},
            {"params": no_decay, "weight_decay": 0.0},
        ]
    else:
        return None
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0)


def reg_loss(model, spec: dict):
    if spec["regularizer"] == "mne_l2":
        return compute_mne_l2_regularization(model, quant_level=LVAL, **spec["mne_kw"])
    return None


def trapz(xs, ys) -> float:
    total = 0.0
    for i in range(1, len(xs)):
        total += 0.5 * (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1])
    return total


def auc_range(rows: list[dict], lo: float, hi: float, key="mIoU") -> float:
    pts = sorted(((float(row["sigma"]), float(row[key])) for row in rows), key=lambda item: item[0])
    xs = [x for x, _ in pts if lo - 1e-9 <= x <= hi + 1e-9]
    ys = [y for x, y in pts if lo - 1e-9 <= x <= hi + 1e-9]
    if len(xs) < 2:
        raise ValueError(f"need ≥2 points in [{lo}, {hi}], got {xs}")
    return trapz(xs, ys)


def metric_at(rows: list[dict], sigma: float, key="mIoU") -> float:
    for row in rows:
        if abs(float(row["sigma"]) - sigma) < 1e-9:
            return float(row[key])
    raise KeyError(sigma)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def voc_loaders(args, pin: bool):
    split = resolve_train_split(args.voc_root)
    print(f"[DATA] train split={split}", flush=True)
    train_set = VOCSegSet(args.voc_root, split, train=True, crop=args.crop)
    val_set = VOCSegSet(args.voc_root, "val", train=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_train,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, min(args.workers, 4)),
        collate_fn=collate_val,
        pin_memory=pin,
        persistent_workers=False,
    )
    return train_loader, val_loader, split, len(train_set), len(val_set)


@torch.no_grad()
def evaluate_miou(
    model, loader, device, sigma: float, seed: int, max_images: int = 0, eval_t: int = TEST_T, eval_mode: str = "rate_uniform"
):
    seed_all(seed)
    model.eval()
    model.set_T(eval_t)
    model.set_mode(eval_mode if eval_t > 0 else "normal")
    model.set_first_layer_input_noise_sigma(float(sigma))
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    fires = []

    def _fire_hook(_module, _inp, output):
        fires.append(float((output.detach() != 0).float().mean().cpu()))

    handles = [module.register_forward_hook(_fire_hook) for module in model.modules() if isinstance(module, IF)]
    n = 0
    t0 = time.time()
    try:
        for batch in loader:
            for image, mask, _image_id, _hw in batch:
                image = image.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                padded, height, width = pad_to_stride(image)
                logits = model(padded.unsqueeze(0))
                logits = logits[:, :, :height, :width]
                if logits.shape[-2:] != mask.shape[-2:]:
                    logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
                pred = logits.argmax(1)
                confusion_update(conf, pred, mask.unsqueeze(0))
                n += 1
                if max_images and n >= max_images:
                    break
            if max_images and n >= max_images:
                break
    finally:
        for handle in handles:
            handle.remove()
    scores = scores_from_confusion(conf)
    fire = float(sum(fires) / len(fires)) if fires else float("nan")
    return {
        "sigma": f"{sigma:g}",
        "mIoU": f"{scores['mIoU']:.6f}",
        "pixel_acc": f"{scores['pixel_acc']:.6f}",
        "n_images": n,
        "seconds": f"{time.time() - t0:.1f}",
        "if_firing_density": f"{fire:.6f}",
        "per_class_iou": scores["per_class_iou"],
    }


def mapping_card(model) -> dict:
    rows = collect_weight_layer_matches(model, LAYER_MAP)
    summary = summarize_weight_layer_matches(rows)
    n_if = sum(isinstance(module, IF) for module in model.modules())
    return {"n_if": n_if, "n_params": int(sum(p.numel() for p in model.parameters())), **summary}


def self_check(device) -> dict:
    model = make_model(SEED, device, load_coco=False)
    model.eval()
    dummy = torch.zeros(1, 3, 128, 128, device=device)
    logits = model(dummy)
    card = {"logits_shape": list(logits.shape), **mapping_card(model)}
    assert logits.shape == (1, 21, 128, 128), card
    assert card["n_if"] == 56, card
    assert card["n_matched"] == 60, card
    assert card["n_unmatched"] == 1, card
    assert card["unmatched_body"] == [], card
    model.set_T(2)
    model.set_mode("rate_uniform")
    logits_t = model(dummy)
    assert logits_t.shape == logits.shape, (logits_t.shape, logits.shape)
    report = matched_weight_beta(model)
    assert report["beta_match"] > 0 and math.isfinite(report["beta_match"]), report
    model.zero_grad(set_to_none=True)
    compute_mne_l2_regularization(model, quant_level=LVAL, **DETACH).backward()
    thresh = next(module.thresh for module in model.modules() if isinstance(module, IF))
    assert thresh.grad is None
    model.zero_grad(set_to_none=True)
    compute_mne_l2_regularization(model, quant_level=LVAL, **NODETACH).backward()
    assert thresh.grad is not None
    card["beta_match"] = report["beta_match"]
    print(json.dumps(card, indent=2), flush=True)
    return card


def dry_run(args, device) -> None:
    if cached_deeplab_weight_path() is None:
        raise SystemExit(
            f"DeepLab COCO-VOC cache missing ({COCO_WEIGHT_NAME}). "
            "Cache it on a login node before dry-run."
        )
    pin = device.type == "cuda"
    train_loader, val_loader, split, n_train, n_val = voc_loaders(args, pin)
    model = make_model(args.seed, device, load_coco=True)
    print(f"[DRY] split={split} n_train={n_train} n_val={n_val} {mapping_card(model)}", flush=True)
    init_if_thresholds_percentile(model, train_loader, device, q=PERCENTILE, max_images=min(8, n_train))
    spec = method_spec("mne")
    spec["reg_coeff"] = matched_weight_beta(model)["beta_match"]
    opt = optimizer_for(model, spec, args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    images, masks, _ids = next(iter(train_loader))
    opt.zero_grad(set_to_none=True)
    loss = criterion(model(images.to(device)), masks.to(device))
    extra = reg_loss(model, spec)
    loss = loss + float(spec["reg_coeff"]) * extra
    loss.backward()
    opt.step()
    mem = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    print(f"[DRY] train step ok loss={float(loss.detach()):.4f} peak_gb={mem:.2f}", flush=True)
    row = evaluate_miou(model, val_loader, device, 0.0, args.seed, max_images=2, eval_t=min(2, args.eval_t))
    print(json.dumps({k: v for k, v in row.items() if k != "per_class_iou"}), flush=True)


def train_one(args, spec: dict, device) -> Path:
    out = cfg_dir(args)
    ckpt = ckpt_path(args)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if ckpt.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {ckpt}", flush=True)
        return ckpt
    if args.test_only:
        raise FileNotFoundError(ckpt)

    pin = device.type == "cuda"
    train_loader, _val_loader, split, n_train, n_val = voc_loaders(args, pin)
    model = make_model(args.seed, device, load_coco=True)
    print(f"[INIT] COCO-VOC; train={n_train} val={n_val} split={split}", flush=True)
    thresh = init_if_thresholds_percentile(
        model, train_loader, device, q=PERCENTILE, max_images=PERCENTILE_IMAGES
    )
    (out / "if_thresh_init.json").write_text(json.dumps(thresh, indent=2) + "\n")
    extra = {"method": args.method, "spec": {k: v for k, v in spec.items() if k != "mne_kw"}}
    if spec["regularizer"] == "mne_l2":
        report = matched_weight_beta(model)
        spec["reg_coeff"] = report["beta_match"]
        extra["strength"] = report
        (out / "beta_match.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"beta_match": report["beta_match"]}), flush=True)
    dump_mne_mapping_report(model, out / "mapping_init", layer_map=LAYER_MAP, quant_level=LVAL, extra=extra)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    if spec["regularizer"] is None:
        torch.save({"state_dict": model.state_dict(), "epoch": 0, "method": args.method}, ckpt)
        return ckpt

    opt = optimizer_for(model, spec, args.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(MILESTONES), gamma=0.1)
    epoch_log = out / "epoch_log.csv"
    fields = ["epoch", "lr", "loss", "ce_loss", "reg_loss", "seconds"]
    if epoch_log.exists() and args.retrain:
        epoch_log.unlink()

    model.train()
    model.set_T(TRAIN_T)
    model.set_mode("normal")
    accum = int(args.accum)
    for epoch in range(1, args.epochs + 1):
        seed_all(args.seed + epoch)
        t0 = time.time()
        running = {k: 0.0 for k in ("loss", "ce_loss", "reg_loss")}
        n_batch = 0
        opt.zero_grad(set_to_none=True)
        for step, (images, masks, _ids) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            ce = criterion(logits, masks)
            extra_loss = reg_loss(model, spec)
            loss = ce
            reg_value = 0.0
            if extra_loss is not None and spec["reg_coeff"] is not None:
                loss = loss + float(spec["reg_coeff"]) * extra_loss
                reg_value = float(extra_loss.detach())
            (loss / accum).backward()
            if step % accum == 0 or step == len(train_loader):
                opt.step()
                opt.zero_grad(set_to_none=True)
            running["loss"] += float(loss.detach())
            running["ce_loss"] += float(ce.detach())
            running["reg_loss"] += reg_value
            n_batch += 1
        sched.step()
        row = {
            "epoch": epoch,
            "lr": f"{sched.get_last_lr()[0]:.6g}",
            "loss": f"{running['loss'] / max(1, n_batch):.6f}",
            "ce_loss": f"{running['ce_loss'] / max(1, n_batch):.6f}",
            "reg_loss": f"{running['reg_loss'] / max(1, n_batch):.6f}",
            "seconds": f"{time.time() - t0:.1f}",
        }
        write_header = not epoch_log.exists()
        with epoch_log.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row), flush=True)
        torch.save({"state_dict": model.state_dict(), "epoch": epoch, "method": args.method, "reg_coeff": spec["reg_coeff"]}, ckpt)
    return ckpt


def evaluate_ckpt(args, spec: dict, device, ckpt: Path) -> dict:
    pin = device.type == "cuda"
    _train, val_loader, split, n_train, n_val = voc_loaders(args, pin)
    model = make_model(args.seed, device, load_coco=False)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        if state.get("reg_coeff") is not None:
            spec["reg_coeff"] = state["reg_coeff"]
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    ann = evaluate_miou(
        model, val_loader, device, 0.0, args.seed, max_images=args.max_eval_images, eval_t=0, eval_mode="normal"
    )
    print(json.dumps({"ann_T0": {k: v for k, v in ann.items() if k != "per_class_iou"}}), flush=True)
    (out / "ann_val.json").write_text(
        json.dumps({**{k: v for k, v in ann.items() if k != "per_class_iou"}, "eval_T": 0}, indent=2) + "\n"
    )
    rows = []
    for sigma in SIGMAS:
        row = evaluate_miou(
            model,
            val_loader,
            device,
            sigma,
            args.seed,
            max_images=args.max_eval_images,
            eval_t=args.eval_t,
        )
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "per_class_iou"}), flush=True)
    csv_rows = [{k: v for k, v in row.items() if k != "per_class_iou"} for row in rows]
    write_csv(out / "val_sweep.csv", csv_rows)
    (out / "val_per_class_iou.json").write_text(
        json.dumps({row["sigma"]: row["per_class_iou"] for row in rows}, indent=2) + "\n"
    )
    match = mapping_card(model)
    card = {
        "method": args.method,
        "label": spec["label"],
        "arch": ARCH,
        "seed": args.seed,
        "quant_level": LVAL,
        "eval_T": args.eval_t,
        "regularizer": spec["regularizer"],
        "weight_decay": spec["weight_decay"],
        "reg_coeff": spec["reg_coeff"],
        "noise_position": "post_input_if",
        "if_mode": "rate_uniform",
        "train_split": split,
        "n_train": n_train,
        "n_val": n_val,
        "checkpoint": str(ckpt),
        "matched_layers": match.get("n_matched"),
        "unmatched_layers": match.get("n_unmatched"),
        "unmatched_body": match.get("unmatched_body"),
        "val_ann": float(ann["mIoU"]),
        "val_ann_pixacc": float(ann["pixel_acc"]),
        "val_clean": metric_at(csv_rows, 0.0),
        "val_sigma0p5": metric_at(csv_rows, 0.5),
        "val_sigma1": metric_at(csv_rows, 1.0),
        "val_sigma2": metric_at(csv_rows, 2.0),
        "val_sigma3": metric_at(csv_rows, 3.0),
        "val_sigma5": metric_at(csv_rows, 5.0),
        "val_auc_full": auc_range(csv_rows, 0.0, 5.0),
        "val_auc_high": auc_range(csv_rows, HIGH_NOISE_MIN, 5.0),
        "val_clean_pixacc": metric_at(csv_rows, 0.0, "pixel_acc"),
        "val_sigma5_pixacc": metric_at(csv_rows, 5.0, "pixel_acc"),
        "val_clean_fire": next(
            float(r["if_firing_density"]) for r in csv_rows if abs(float(r["sigma"]) - 0.0) < 1e-9
        ),
    }
    (out / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2), flush=True)
    return card


def summarize(out_root: Path) -> None:
    cards = []
    for path in sorted(out_root.glob("*/scorecard.json")):
        cards.append(json.loads(path.read_text()))
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    print(f"{'method':<18} {'ann':>8} {'clean':>8} {'s5':>8} {'AUC':>8} {'matched':>8}")
    for card in cards:
        print(
            f"{card['method']:<18} {card['val_ann']:8.2f} {card['val_clean']:8.2f} "
            f"{card['val_sigma5']:8.2f} {card['val_auc_full']:8.2f} "
            f"{str(card.get('matched_layers')):>8}"
        )


def main() -> None:
    args = parse_args()
    if args.download_voc:
        print(download_voc(args.voc_root), flush=True)
        if not voc2012_seg_is_ready(args.voc_root):
            raise SystemExit(f"VOC downloaded but segmentation split missing under {args.voc_root}")
        return
    device = get_torch_device(args.device)
    if args.self_check:
        self_check(device)
        return
    if args.dry_run:
        if not voc2012_seg_is_ready(args.voc_root):
            raise SystemExit(f"VOC 2012 segmentation missing under {args.voc_root}")
        dry_run(args, device)
        return
    if args.summarize:
        summarize(args.out_root)
        return
    if not voc2012_seg_is_ready(args.voc_root):
        extra = ""
        if not voc_is_ready(args.voc_root):
            extra = " Also run: python noise3_exp/voc_ssd.py --download --voc-root DIR"
        raise SystemExit(f"VOC 2012 segmentation missing under {args.voc_root}.{extra}")
    if cached_deeplab_weight_path() is None and not args.test_only:
        raise SystemExit(
            f"DeepLab COCO-VOC cache missing ({COCO_WEIGHT_NAME}) under TORCH_HOME. "
            "Cache it on a login node; compute nodes have no internet."
        )
    spec = method_spec(args.method)
    args.out_root.mkdir(parents=True, exist_ok=True)
    ckpt = train_one(args, spec, device)
    evaluate_ckpt(args, spec, device, ckpt)
    print(f"Wrote {cfg_dir(args)}", flush=True)


if __name__ == "__main__":
    main()
