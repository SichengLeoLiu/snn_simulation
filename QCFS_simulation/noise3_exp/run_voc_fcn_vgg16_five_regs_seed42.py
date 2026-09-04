#!/usr/bin/env python3
"""VOC 2012 FCN-VGG16-BN-IF seed-42 segmentation noise transfer.

Question: does MNE-L2 / no-detach transfer from classification to a discrete
dense prediction task (per-pixel argmax over spike counts)? No box regression.

Methods (one PBS job each)
--------------------------
    l2wo              optimizer WD on Conv/Linear weights, wd=5e-4
    mne               published Old MNE-L2, detach λ and BN γ, rc=1e-4
    nodetach_lambda   grads into λ only (BN γ still detached)
    nodetach_gamma    grads into BN γ only (λ still detached)
    nodetach          grads into λ and BN γ

Protocol
--------
    Train T=0 ANN, L=16, ImageNet VGG-16 BN encoder init.
    Eval T=16, rate_uniform, post_input_if Gaussian.
    Data: VOC 2012 train / val (no SBD trainaug in this feasibility run).
    Metrics: mIoU, pixel accuracy, robust-mIoU AUC. Seed 42 only.
    Do not retune coefficients. Do not select on test noise.
"""
from __future__ import annotations

import argparse
import csv
import json
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

from Models.FCN import FCNVGG16, IGNORE_INDEX, NUM_CLASSES, load_vgg16_bn_into_fcn  # noqa: E402
from Models.layer import IF  # noqa: E402
from voc_seg import (  # noqa: E402
    CROP,
    VOCSegSet,
    collate_train,
    collate_val,
    confusion_update,
    download_voc,
    pad_to_stride,
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

ARCH = "fcn_vgg16"
SEED = 42
LVAL = 16
TRAIN_T = 0
TEST_T = 16
MNE_RC = 1e-4
L2_WD = 5e-4
EPOCHS = 50
LR = 1e-3
MILESTONES = (30, 40)
SIGMAS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)
HIGH_NOISE_MIN = 3.0
METHODS = ("l2wo", "mne", "nodetach_lambda", "nodetach_gamma", "nodetach")

DETACH_BOTH = dict(
    detach_lambda=True,
    detach_bn_stats=True,
    detach_bn_affine=True,
    divide_by_lambda=True,
    scale_by_l=True,
    l_ref=None,
    fold_bn=True,
)
NODETACH_LAMBDA = dict(DETACH_BOTH, detach_lambda=False, detach_bn_affine=True)
NODETACH_GAMMA = dict(DETACH_BOTH, detach_lambda=True, detach_bn_affine=False)
NODETACH_BOTH = dict(DETACH_BOTH, detach_lambda=False, detach_bn_affine=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("VOC_EPOCHS", str(EPOCHS))))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("VOC_BATCH", "8")))
    parser.add_argument("--eval-batch-size", type=int, default=int(os.environ.get("VOC_EVAL_BATCH", "1")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("VOC_NUM_WORKERS", "8")))
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--crop", type=int, default=CROP)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--self-check", action="store_true")
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
        default=ROOT.parent / "important_results" / "voc_fcn_vgg16_five_regs_seed42",
    )
    args = parser.parse_args()
    args.voc_root = Path(os.path.expanduser(str(args.voc_root)))
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    if args.summarize or args.self_check or args.download_voc:
        return args
    if args.method is None:
        parser.error("--method is required unless --summarize/--self-check/--download-voc")
    return args


def method_spec(method: str) -> dict:
    if method == "l2wo":
        return {
            "label": "L2-wo",
            "regularizer": "weight_decay_weights_only",
            "weight_decay": L2_WD,
            "reg_coeff": None,
            "mne_kw": None,
        }
    table = {
        "mne": ("MNE-L2 detach", DETACH_BOTH),
        "nodetach_lambda": (r"MNE-L2 λ no-detach", NODETACH_LAMBDA),
        "nodetach_gamma": (r"MNE-L2 γ no-detach", NODETACH_GAMMA),
        "nodetach": ("MNE-L2 no-detach", NODETACH_BOTH),
    }
    label, kw = table[method]
    return {
        "label": label,
        "regularizer": "mne_l2",
        "weight_decay": 0.0,
        "reg_coeff": MNE_RC,
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


def make_model(seed: int, device, load_imagenet: bool = True):
    seed_all(seed)
    model = FCNVGG16()
    model._mne_layer_map = "legacy"
    n_copied = 0
    if load_imagenet:
        n_copied = load_vgg16_bn_into_fcn(model)
    model.set_L(LVAL)
    model.set_T(TRAIN_T)
    model.set_mode("normal")
    model.set_spike_schedule("normal")
    model.set_first_layer_input_noise_position("post_input_if")
    model.set_first_layer_input_noise_type("gaussian")
    model.set_first_layer_input_noise_sigma(0.0)
    return model.to(device), n_copied


def optimizer_for(model, spec: dict, lr: float):
    if spec["regularizer"] == "weight_decay_weights_only":
        decay_ids = {
            id(module.weight)
            for module in model.modules()
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear))
            and getattr(module, "weight", None) is not None
        }
        decay, no_decay = [], []
        for parameter in model.parameters():
            (decay if id(parameter) in decay_ids else no_decay).append(parameter)
        params = [
            {"params": decay, "weight_decay": spec["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        wd = 0.0
    else:
        params = model.parameters()
        wd = float(spec["weight_decay"])
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)


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
    xs, ys = [], []
    for row in rows:
        sigma = float(row["sigma"])
        if lo - 1e-12 <= sigma <= hi + 1e-12:
            xs.append(sigma)
            ys.append(float(row[key]))
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
    train_set = VOCSegSet(args.voc_root, "train", train=True, crop=args.crop)
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
    return train_loader, val_loader


@torch.no_grad()
def evaluate_miou(model, loader, device, sigma: float, seed: int, max_images: int = 0):
    seed_all(seed)
    model.eval()
    model.set_T(TEST_T)
    model.set_mode("rate_uniform")
    model.set_first_layer_input_noise_sigma(float(sigma))
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    fires = []

    def _fire_hook(_module, _inp, output):
        fires.append(float((output.detach() != 0).float().mean().cpu()))

    handles = [
        module.register_forward_hook(_fire_hook)
        for module in model.modules()
        if isinstance(module, IF)
    ]
    n = 0
    t0 = time.time()
    try:
        for batch in loader:
            for image, mask, _image_id, hw in batch:
                image = image.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                padded, height, width = pad_to_stride(image)
                logits = model(padded.unsqueeze(0))
                logits = logits[:, :, :height, :width]
                if logits.shape[-2:] != mask.shape[-2:]:
                    logits = F.interpolate(
                        logits, size=mask.shape[-2:], mode="bilinear", align_corners=False
                    )
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


def self_check(device) -> dict:
    model, n_copied = make_model(SEED, device, load_imagenet=False)
    dummy = torch.zeros(2, 3, 512, 512, device=device)
    logits = model(dummy)
    rows = collect_weight_layer_matches(model, layer_map="legacy")
    summary = summarize_weight_layer_matches(rows)
    card = {
        "n_imagenet_copied": n_copied,
        "logits_shape": list(logits.shape),
        "n_if": sum(isinstance(m, IF) for m in model.modules()),
        **summary,
    }
    assert logits.shape == (2, 21, 512, 512), card
    assert summary["n_matched"] == 15, card
    assert summary["n_unmatched"] == 0, card
    model.set_T(2)
    model.set_mode("rate_uniform")
    logits_t = model(dummy)
    assert logits_t.shape == logits.shape, (logits_t.shape, logits.shape)
    model.set_T(TRAIN_T)
    model.set_mode("normal")
    print(json.dumps(card, indent=2), flush=True)
    return card


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
    train_loader, _ = voc_loaders(args, pin)
    model, n_copied = make_model(args.seed, device, load_imagenet=True)
    print(f"[INIT] copied {n_copied} ImageNet VGG-16 BN conv/BN pairs", flush=True)
    dump_mne_mapping_report(
        model,
        out / "mapping_init",
        layer_map="legacy",
        quant_level=LVAL,
        extra={"method": args.method, "spec": {k: v for k, v in spec.items() if k != "mne_kw"}},
    )
    opt = optimizer_for(model, spec, args.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(MILESTONES), gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    epoch_log = out / "epoch_log.csv"
    fields = ["epoch", "lr", "loss", "ce_loss", "reg_loss", "seconds"]
    if epoch_log.exists() and args.retrain:
        epoch_log.unlink()

    model.train()
    model.set_T(TRAIN_T)
    model.set_mode("normal")
    for epoch in range(1, args.epochs + 1):
        seed_all(args.seed + epoch)
        t0 = time.time()
        running = {k: 0.0 for k in ("loss", "ce_loss", "reg_loss")}
        n_batch = 0
        for images, masks, _ids in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(images)
            ce = criterion(logits, masks)
            extra = reg_loss(model, spec)
            loss = ce
            reg_value = 0.0
            if extra is not None and spec["reg_coeff"] is not None:
                loss = loss + float(spec["reg_coeff"]) * extra
                reg_value = float(extra.detach())
            loss.backward()
            opt.step()
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
        torch.save({"state_dict": model.state_dict(), "epoch": epoch, "method": args.method}, ckpt)
    return ckpt


def evaluate_ckpt(args, spec: dict, device, ckpt: Path) -> dict:
    pin = device.type == "cuda"
    _, val_loader = voc_loaders(args, pin)
    model, _ = make_model(args.seed, device, load_imagenet=False)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    rows = []
    for sigma in SIGMAS:
        row = evaluate_miou(
            model, val_loader, device, sigma, args.seed, max_images=args.max_eval_images
        )
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "per_class_iou"}), flush=True)
    out = cfg_dir(args)
    csv_rows = [{k: v for k, v in row.items() if k != "per_class_iou"} for row in rows]
    write_csv(out / "val_sweep.csv", csv_rows)
    (out / "val_per_class_iou.json").write_text(
        json.dumps({row["sigma"]: row["per_class_iou"] for row in rows}, indent=2) + "\n"
    )
    match = summarize_weight_layer_matches(collect_weight_layer_matches(model, "legacy"))
    card = {
        "method": args.method,
        "label": spec["label"],
        "arch": ARCH,
        "seed": args.seed,
        "quant_level": LVAL,
        "eval_T": TEST_T,
        "regularizer": spec["regularizer"],
        "weight_decay": spec["weight_decay"],
        "reg_coeff": spec["reg_coeff"],
        "noise_position": "post_input_if",
        "if_mode": "rate_uniform",
        "checkpoint": str(ckpt),
        "matched_layers": match.get("n_matched"),
        "unmatched_layers": match.get("n_unmatched"),
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
        "val_clean_fire": next(float(r["if_firing_density"]) for r in csv_rows if abs(float(r["sigma"]) - 0.0) < 1e-9),
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
    print(f"{'method':<18} {'clean':>8} {'s5':>8} {'AUC':>8} {'pix0':>8} {'matched':>8}")
    for card in cards:
        print(
            f"{card['method']:<18} {card['val_clean']:8.2f} {card['val_sigma5']:8.2f} "
            f"{card['val_auc_full']:8.2f} {card['val_clean_pixacc']:8.2f} "
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
    if args.summarize:
        summarize(args.out_root)
        return
    if not voc2012_seg_is_ready(args.voc_root):
        extra = ""
        if not voc_is_ready(args.voc_root):
            extra = " Also run: python noise3_exp/voc_ssd.py --download --voc-root DIR"
        raise SystemExit(
            f"VOC 2012 segmentation missing under {args.voc_root}.{extra}"
        )
    spec = method_spec(args.method)
    args.out_root.mkdir(parents=True, exist_ok=True)
    ckpt = train_one(args, spec, device)
    evaluate_ckpt(args, spec, device, ckpt)
    print(f"Wrote {cfg_dir(args)}", flush=True)


if __name__ == "__main__":
    main()
