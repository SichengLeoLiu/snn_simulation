#!/usr/bin/env python3
"""VOC SSD300-VGG16-BN-IF seed-42 five-regularizer noise transfer.

Question: does MNE-L2 / no-detach transfer from classification to detection
on a serial VGG backbone? Do not retune coefficients.

Methods (one PBS job each)
--------------------------
    mne        published Old MNE-L2, detach λ, rc=1e-4
    nodetach   same formula, grads into λ and BN γ, rc=1e-4
    l2wo       optimizer WD on Conv/Linear weights, wd=5e-4
    l2all      optimizer WD on all params, wd=5e-4
    l1wo       explicit L1 on Conv/Linear weights, rc=1e-5

Protocol
--------
    Train T=0 ANN, L=16, ImageNet VGG-16 BN init.
    Eval T=16, rate_uniform, post_input_if Gaussian.
    Data: VOC 2007+2012 trainval, VOC 2007 test, mAP@0.5 (VOC07 11-point).
    Seed 42 only (feasibility). Do not select on test noise.
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Models.SSD import SSD300VGG16, load_vgg16_bn_into_ssd  # noqa: E402
from Models.layer import IF  # noqa: E402
from voc_ssd import (  # noqa: E402
    VOCDetectionSet,
    collate_voc,
    detect,
    download_voc,
    multibox_loss,
    voc_is_ready,
    voc_map,
    voc_root_from,
)
from utils import (  # noqa: E402
    compute_l1_regularization,
    compute_mne_l2_regularization,
    dump_mne_mapping_report,
    get_torch_device,
    seed_all,
    summarize_weight_layer_matches,
    collect_weight_layer_matches,
)

ARCH = "ssd300_vgg16"
SEED = 42
LVAL = 16
TRAIN_T = 0
TEST_T = 16
MNE_RC = 1e-4
L1_RC = 1e-5
L2_WD = 5e-4
EPOCHS = 80
LR = 1e-3
MILESTONES = (50, 70)
SIGMAS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)
HIGH_NOISE_MIN = 3.0
METHODS = ("mne", "nodetach", "l2wo", "l2all", "l1wo")

MNE_KW = dict(
    detach_lambda=True,
    detach_bn_stats=True,
    detach_bn_affine=True,
    divide_by_lambda=True,
    scale_by_l=True,
    l_ref=None,
    fold_bn=True,
)
NODETACH_KW = dict(
    detach_lambda=False,
    detach_bn_stats=True,
    detach_bn_affine=False,
    divide_by_lambda=True,
    scale_by_l=True,
    l_ref=None,
    fold_bn=True,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("VOC_EPOCHS", str(EPOCHS))))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("VOC_BATCH", "16")))
    parser.add_argument("--eval-batch-size", type=int, default=int(os.environ.get("VOC_EVAL_BATCH", "8")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("VOC_NUM_WORKERS", "8")))
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
        default=ROOT.parent / "important_results" / "voc_ssd300_five_regs_seed42",
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
    if method == "mne":
        return {
            "label": "MNE-L2 detach",
            "regularizer": "mne_l2",
            "weight_decay": 0.0,
            "reg_coeff": MNE_RC,
            "mne_kw": dict(MNE_KW),
        }
    if method == "nodetach":
        return {
            "label": "MNE-L2 no-detach",
            "regularizer": "mne_l2",
            "weight_decay": 0.0,
            "reg_coeff": MNE_RC,
            "mne_kw": dict(NODETACH_KW),
        }
    if method == "l2wo":
        return {
            "label": "L2-wo",
            "regularizer": "weight_decay_weights_only",
            "weight_decay": L2_WD,
            "reg_coeff": None,
            "mne_kw": None,
        }
    if method == "l2all":
        return {
            "label": "L2-all",
            "regularizer": "weight_decay",
            "weight_decay": L2_WD,
            "reg_coeff": None,
            "mne_kw": None,
        }
    return {
        "label": "L1-wo",
        "regularizer": "l1",
        "weight_decay": 0.0,
        "reg_coeff": L1_RC,
        "mne_kw": None,
    }


def cfg_dir(args) -> Path:
    return args.out_root / args.method


def ckpt_path(args) -> Path:
    return cfg_dir(args) / "checkpoints" / f"{ARCH}_L[{LVAL}]_{args.method}_seed{args.seed}_L{LVAL}_trainT{TRAIN_T}.pth"


def make_model(seed: int, device, load_imagenet: bool = True):
    seed_all(seed)
    model = SSD300VGG16()
    model._mne_layer_map = "legacy"
    n_copied = 0
    if load_imagenet:
        n_copied = load_vgg16_bn_into_ssd(model)
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
    if spec["regularizer"] == "l1":
        return compute_l1_regularization(model)
    return None


def trapz(xs, ys) -> float:
    total = 0.0
    for i in range(1, len(xs)):
        total += 0.5 * (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1])
    return total


def auc_range(rows: list[dict], lo: float, hi: float, key="mAP") -> float:
    xs, ys = [], []
    for row in rows:
        sigma = float(row["sigma"])
        if lo - 1e-12 <= sigma <= hi + 1e-12:
            xs.append(sigma)
            ys.append(float(row[key]))
    if len(xs) < 2:
        raise ValueError(f"need ≥2 points in [{lo}, {hi}], got {xs}")
    return trapz(xs, ys)


def map_at(rows: list[dict], sigma: float) -> float:
    for row in rows:
        if abs(float(row["sigma"]) - sigma) < 1e-9:
            return float(row["mAP"])
    raise KeyError(sigma)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def voc_loaders(args, pin: bool):
    root = voc_root_from(args.voc_root)
    train_set = VOCDetectionSet(root, "trainval", train=True)
    test_set = VOCDetectionSet(root, "test", train=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_voc,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=max(0, min(args.workers, 4)),
        collate_fn=collate_voc,
        pin_memory=pin,
        persistent_workers=False,
    )
    return train_loader, test_loader, root


@torch.no_grad()
def evaluate_map(model, loader, device, sigma: float, seed: int, max_images: int = 0):
    seed_all(seed)
    model.eval()
    model.set_T(TEST_T)
    model.set_mode("rate_uniform")
    model.set_first_layer_input_noise_sigma(float(sigma))
    pred_by_image = {}
    gt_by_image = {}
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
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            loc, conf = model(images)
            outputs = detect(loc, conf, model.priors)
            for target, output in zip(targets, outputs):
                image_id = target["id"]
                gt_by_image[image_id] = {
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu(),
                }
                pred_by_image[image_id] = {
                    "boxes": output["boxes"].cpu(),
                    "scores": output["scores"].cpu(),
                    "labels": output["labels"].cpu(),
                }
                n += 1
                if max_images and n >= max_images:
                    break
            if max_images and n >= max_images:
                break
    finally:
        for handle in handles:
            handle.remove()
    mean_ap, aps = voc_map(pred_by_image, gt_by_image)
    fire = float(sum(fires) / len(fires)) if fires else float("nan")
    return {
        "sigma": f"{sigma:g}",
        "mAP": f"{100.0 * mean_ap:.6f}",
        "n_images": n,
        "seconds": f"{time.time() - t0:.1f}",
        "if_firing_density": f"{fire:.6f}",
        "per_class_ap": {k: (None if v != v else round(100.0 * v, 4)) for k, v in aps.items()},
    }


def self_check(device) -> dict:
    model, n_copied = make_model(SEED, device, load_imagenet=False)
    dummy = torch.zeros(2, 3, 300, 300, device=device)
    loc, conf = model(dummy)
    sources = model._sources(dummy)
    shapes = [tuple(src.shape[-2:]) for src in sources]
    rows = collect_weight_layer_matches(model, layer_map="legacy")
    summary = summarize_weight_layer_matches(rows)
    n_priors = int(model.priors.shape[0])
    card = {
        "n_imagenet_copied": n_copied,
        "n_priors": n_priors,
        "loc_shape": list(loc.shape),
        "conf_shape": list(conf.shape),
        "source_hw": shapes,
        "expected_hw": [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
        **summary,
    }
    assert n_priors == 8732, card
    assert loc.shape == (2, 8732, 4), card
    assert conf.shape == (2, 8732, 21), card
    assert shapes == card["expected_hw"], card
    assert summary["n_matched"] == 23, card
    assert summary["n_unmatched"] == 12, card
    assert summary["n_head_layers"] == 12, card
    model.set_T(2)
    model.set_mode("rate_uniform")
    loc_t, conf_t = model(dummy)
    assert loc_t.shape == loc.shape, (loc_t.shape, loc.shape)
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
    train_loader, _, voc_root = voc_loaders(args, pin)
    model, n_copied = make_model(args.seed, device, load_imagenet=True)
    print(f"[INIT] copied {n_copied} ImageNet VGG-16 BN conv/BN pairs; voc={voc_root}", flush=True)
    dump_mne_mapping_report(
        model,
        out / "mapping_init",
        layer_map="legacy",
        quant_level=LVAL,
        extra={"method": args.method, "spec": {k: v for k, v in spec.items() if k != "mne_kw"}},
    )
    opt = optimizer_for(model, spec, args.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(MILESTONES), gamma=0.1)
    epoch_log = out / "epoch_log.csv"
    fields = [
        "epoch",
        "lr",
        "loss",
        "loc_loss",
        "conf_loss",
        "reg_loss",
        "n_pos",
        "seconds",
    ]
    if epoch_log.exists() and args.retrain:
        epoch_log.unlink()

    model.train()
    model.set_T(TRAIN_T)
    model.set_mode("normal")
    for epoch in range(1, args.epochs + 1):
        seed_all(args.seed + epoch)
        t0 = time.time()
        running = {k: 0.0 for k in ("loss", "loc_loss", "conf_loss", "reg_loss", "n_pos")}
        n_batches = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loc, conf = model(images)
            det_loss, loc_loss, conf_loss, n_pos = multibox_loss(loc, conf, targets, model.priors)
            extra = reg_loss(model, spec)
            if extra is not None and spec["reg_coeff"] is not None:
                total = det_loss + float(spec["reg_coeff"]) * extra
                extra_v = float(extra.detach())
            else:
                total = det_loss
                extra_v = 0.0
            total.backward()
            opt.step()
            running["loss"] += float(total.detach())
            running["loc_loss"] += float(loc_loss)
            running["conf_loss"] += float(conf_loss)
            running["reg_loss"] += extra_v
            running["n_pos"] += n_pos
            n_batches += 1
        sched.step()
        row = {
            "epoch": epoch,
            "lr": f"{sched.get_last_lr()[0]:.6g}",
            "loss": f"{running['loss'] / max(1, n_batches):.6f}",
            "loc_loss": f"{running['loc_loss'] / max(1, n_batches):.6f}",
            "conf_loss": f"{running['conf_loss'] / max(1, n_batches):.6f}",
            "reg_loss": f"{running['reg_loss'] / max(1, n_batches):.6f}",
            "n_pos": f"{running['n_pos'] / max(1, n_batches):.2f}",
            "seconds": f"{time.time() - t0:.1f}",
        }
        write_header = not epoch_log.exists()
        with epoch_log.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} lr={row['lr']} loss={row['loss']} "
            f"loc={row['loc_loss']} conf={row['conf_loss']} reg={row['reg_loss']} "
            f"n_pos={row['n_pos']} {row['seconds']}s",
            flush=True,
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "method": args.method,
                "seed": args.seed,
            },
            ckpt,
        )
    return ckpt


def load_trained(ckpt: Path, device):
    model, _ = make_model(SEED, device, load_imagenet=False)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def summarize(out_root: Path) -> None:
    cards = []
    for method in METHODS:
        path = out_root / method / "scorecard.json"
        if path.is_file():
            cards.append(json.loads(path.read_text()))
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    print(
        f"{'method':<10} {'clean':>7} {'s0.5':>7} {'s1':>7} {'s2':>7} "
        f"{'s3':>7} {'s5':>7} {'AUC':>8} {'AUChi':>8}"
    )
    for card in cards:
        print(
            f"{card['method']:<10} {card['test_clean']:7.2f} "
            f"{card.get('test_sigma0p5', float('nan')):7.2f} "
            f"{card.get('test_sigma1', float('nan')):7.2f} "
            f"{card.get('test_sigma2', float('nan')):7.2f} "
            f"{card['test_sigma3']:7.2f} {card['test_sigma5']:7.2f} "
            f"{card['test_auc_full']:8.2f} {card['test_auc_high']:8.2f}"
        )


def sigma_lookup(rows, sigma: float) -> float:
    try:
        return map_at(rows, sigma)
    except KeyError:
        return float("nan")


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.download_voc:
        print(download_voc(args.voc_root), flush=True)
        return
    if args.summarize:
        summarize(args.out_root)
        return

    device = get_torch_device(args.device)
    if args.self_check or args.dry_run:
        card = self_check(device)
        (args.out_root / "self_check.json").write_text(json.dumps(card, indent=2) + "\n")
        if args.self_check or args.method is None:
            return

    if not voc_is_ready(args.voc_root):
        raise FileNotFoundError(
            f"VOC 2007+2012 missing under VOC_ROOT={args.voc_root}. "
            "On a login node: python noise3_exp/voc_ssd.py --download "
            f"--voc-root {args.voc_root}"
        )

    spec = method_spec(args.method)
    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "method": args.method,
                "spec": {k: v for k, v in spec.items() if k != "mne_kw"},
                "seed": args.seed,
                "voc_root": str(args.voc_root),
                "out": str(out),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        print("[DRY RUN] skip train/eval", flush=True)
        return

    ckpt = train_one(args, spec, device)
    model = load_trained(ckpt, device)
    dump_mne_mapping_report(
        model,
        out / "mapping_final",
        layer_map="legacy",
        quant_level=LVAL,
        extra={"method": args.method, "checkpoint": str(ckpt)},
    )
    _, test_loader, _ = voc_loaders(args, device.type == "cuda")
    rows = []
    per_class = {}
    for sigma in SIGMAS:
        result = evaluate_map(
            model,
            test_loader,
            device,
            sigma,
            args.seed,
            max_images=args.max_eval_images,
        )
        per_class[f"{sigma:g}"] = result.pop("per_class_ap")
        rows.append(result)
        print(
            f"test T={TEST_T} sigma={sigma:g} mAP={result['mAP']} "
            f"n={result['n_images']} {result['seconds']}s",
            flush=True,
        )
    write_csv(out / "test_sweep.csv", rows)
    (out / "per_class_ap.json").write_text(json.dumps(per_class, indent=2) + "\n")
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
        "test_clean": map_at(rows, 0.0),
        "test_sigma0p5": sigma_lookup(rows, 0.5),
        "test_sigma1": sigma_lookup(rows, 1.0),
        "test_sigma2": sigma_lookup(rows, 2.0),
        "test_sigma3": map_at(rows, 3.0),
        "test_sigma5": map_at(rows, 5.0),
        "test_auc_full": auc_range(rows, 0.0, 5.0),
        "test_auc_high": auc_range(rows, HIGH_NOISE_MIN, 5.0),
        "test_clean_fire": float(rows[0]["if_firing_density"]),
    }
    (out / "scorecard.json").write_text(json.dumps(card, indent=2, default=str) + "\n")
    print(json.dumps(card, indent=2, default=str), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
