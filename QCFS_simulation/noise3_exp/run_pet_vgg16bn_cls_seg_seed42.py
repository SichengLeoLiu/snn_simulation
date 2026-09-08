#!/usr/bin/env python3
"""Oxford-IIIT Pet VGG16-BN classification vs foreground segmentation.

Question
--------
With the same images, VGG16-BN encoder (Conv-BN-IF, AvgPool, no skip/ASPP),
ImageNet encoder init, L=T=16, and linear final readout, does MNE's gain over
L2-wo shrink when the task changes from 37-way classification to binary
foreground segmentation?

Methods (one PBS job each)
--------------------------
    l2wo       SGD WD=5e-4 on Conv/Linear weights.
    mne        MNE-L2 detach λ/γ. β matches ||∇W L2-wo|| at init, then frozen.
    nodetach   same frozen β, gradients into λ.

Protocol
--------
    Official trainval / test. Both tasks use the same ids, 224² inputs, and
    encoder. Uncertain trimap pixels (value 3) are ignore index 255.
    Train T=0; eval T=0 (ANN) then T=16 rate_uniform post_input_if.
    Seed 42. Do not retune coefficients from the test noise curve.
    Default readout is linear; --head-if is a separate ablation.
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Models.FCN import init_if_thresholds_percentile  # noqa: E402
from Models.layer import IF  # noqa: E402
from Models.PetVGG import (  # noqa: E402
    IGNORE_INDEX,
    NUM_CLS,
    NUM_SEG,
    PetVGGClassifier,
    PetVGGSegmentor,
    count_if,
    count_maxpool2d,
    load_vgg16_bn_into_encoder,
)
from pet import (  # noqa: E402
    SIZE,
    PetSet,
    download_pet,
    pet_is_ready,
    scores_from_binary_confusion,
)
from voc_seg import confusion_update  # noqa: E402
from utils import (  # noqa: E402
    collect_weight_layer_matches,
    compute_mne_l2_regularization,
    dump_mne_mapping_report,
    get_torch_device,
    seed_all,
    summarize_weight_layer_matches,
)

ARCH = "pet_vgg16bn"
SEED = 42
LVAL = 16
TRAIN_T = 0
TEST_T = 16
L2_WD = 5e-4
EPOCHS = 40
LR = 1e-3
MILESTONES = (25, 35)
SIGMAS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)
HIGH_NOISE_MIN = 3.0
TASKS = ("cls", "seg")
METHODS = ("l2wo", "mne", "nodetach")
PERCENTILE = 99.9
PERCENTILE_IMAGES = 64
LAYER_MAP = "legacy"

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
    _task = os.environ.get("TASK", "").strip()
    parser.add_argument("--task", choices=TASKS, default=_task if _task in TASKS else None)
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("PET_EPOCHS", str(EPOCHS))))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("PET_BATCH", "0")))
    parser.add_argument("--eval-batch-size", type=int, default=int(os.environ.get("PET_EVAL_BATCH", "0")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("PET_NUM_WORKERS", "4")))
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--size", type=int, default=int(os.environ.get("PET_SIZE", str(SIZE))))
    parser.add_argument("--eval-t", type=int, default=int(os.environ.get("PET_EVAL_T", str(TEST_T))))
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="RNG seed for the test noise stream. Default: the training seed.",
    )
    parser.add_argument("--head-if", action="store_true", default=os.environ.get("PET_HEAD_IF", "0") == "1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--download-pet", action="store_true")
    parser.add_argument("--max-eval-images", type=int, default=0)
    parser.add_argument(
        "--pet-root",
        type=Path,
        default=Path(os.environ.get("PET_ROOT", os.environ.get("CIFAR_ROOT", "~/datasets"))),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "pet_vgg16bn_cls_seg_seed42",
    )
    args = parser.parse_args()
    args.pet_root = Path(os.path.expanduser(str(args.pet_root)))
    args.eval_t = max(0, int(args.eval_t))
    args.size = max(32, int(args.size))
    if args.eval_seed is None:
        env_eval = os.environ.get("PET_EVAL_SEED", "").strip()
        args.eval_seed = int(env_eval) if env_eval else args.seed
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    if args.summarize or args.self_check or args.download_pet:
        return args
    if args.task is None or args.method is None:
        parser.error("--task and --method are required unless --summarize/--self-check/--download-pet")
    if args.batch_size <= 0:
        args.batch_size = 16 if args.task == "cls" else 8
    if args.eval_batch_size <= 0:
        args.eval_batch_size = 4 if args.task == "cls" else 2
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
        "mne": ("MNE-L2 detach", DETACH),
        "nodetach": ("MNE-L2 no-detach", NODETACH),
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
    return args.out_root / args.task / args.method


def ckpt_path(args) -> Path:
    head = "headif" if args.head_if else "linear"
    return (
        cfg_dir(args)
        / "checkpoints"
        / f"{ARCH}_{args.task}_{head}_L[{LVAL}]_{args.method}_seed{args.seed}_L{LVAL}_trainT{TRAIN_T}.pth"
    )


def make_model(task: str, seed: int, device, load_imagenet: bool = True, head_if: bool = False):
    seed_all(seed)
    if task == "cls":
        model = PetVGGClassifier(head_if=head_if)
    elif task == "seg":
        model = PetVGGSegmentor(head_if=head_if)
    else:
        raise ValueError(task)
    model._mne_layer_map = LAYER_MAP
    n_copied = 0
    if load_imagenet:
        n_copied = load_vgg16_bn_into_encoder(model.encoder)
    model.set_L(LVAL)
    model.set_T(TRAIN_T)
    model.set_mode("normal")
    model.set_spike_schedule("normal")
    model.set_first_layer_input_noise_position("post_input_if")
    model.set_first_layer_input_noise_type("gaussian")
    model.set_first_layer_input_noise_sigma(0.0)
    return model.to(device), n_copied


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
        raise ValueError(spec["regularizer"])
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


def auc_range(rows: list[dict], lo: float, hi: float, key: str) -> float:
    pts = sorted(((float(row["sigma"]), float(row[key])) for row in rows), key=lambda item: item[0])
    xs = [x for x, _ in pts if lo - 1e-9 <= x <= hi + 1e-9]
    ys = [y for x, y in pts if lo - 1e-9 <= x <= hi + 1e-9]
    if len(xs) < 2:
        raise ValueError(f"need ≥2 points in [{lo}, {hi}], got {xs}")
    return trapz(xs, ys)


def metric_at(rows: list[dict], sigma: float, key: str) -> float:
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


def pet_loaders(args, pin: bool):
    train_set = PetSet(args.pet_root, "trainval", args.task, train=True, size=args.size)
    test_set = PetSet(args.pet_root, "test", args.task, train=False, size=args.size)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=max(0, min(args.workers, 4)),
        pin_memory=pin,
        persistent_workers=False,
    )
    return train_loader, test_loader, len(train_set), len(test_set)


def _fire_handles(model):
    fires = []

    def _hook(_module, _inp, output):
        fires.append(float((output.detach() != 0).float().mean().cpu()))

    handles = [module.register_forward_hook(_hook) for module in model.modules() if isinstance(module, IF)]
    return fires, handles


@torch.no_grad()
def evaluate_task(args, model, loader, device, sigma: float, eval_t: int, eval_mode: str) -> dict:
    seed_all(int(getattr(args, "eval_seed", args.seed)))
    model.eval()
    model.set_T(eval_t)
    model.set_mode(eval_mode if eval_t > 0 else "normal")
    model.set_first_layer_input_noise_sigma(float(sigma))
    fires, handles = _fire_handles(model)
    n = 0
    t0 = time.time()
    try:
        if args.task == "cls":
            correct = 0
            total = 0
            for images, labels, _ids in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                pred = model(images).argmax(1)
                correct += int((pred == labels).sum().item())
                total += int(labels.numel())
                n += int(images.shape[0])
                if args.max_eval_images and n >= args.max_eval_images:
                    break
            acc = 100.0 * correct / max(1, total)
            row = {"acc": f"{acc:.6f}", "n_correct": correct, "n_total": total}
        else:
            conf = torch.zeros(NUM_SEG, NUM_SEG, dtype=torch.long, device=device)
            for images, masks, _ids in loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                pred = model(images).argmax(1)
                confusion_update(conf, pred, masks)
                n += int(images.shape[0])
                if args.max_eval_images and n >= args.max_eval_images:
                    break
            scores = scores_from_binary_confusion(conf)
            row = {key: f"{scores[key]:.6f}" for key in ("fg_iou", "dice", "mIoU", "pixel_acc", "bg_iou")}
    finally:
        for handle in handles:
            handle.remove()
    fire = float(sum(fires) / len(fires)) if fires else float("nan")
    row.update(
        {
            "sigma": f"{sigma:g}",
            "n_images": n,
            "seconds": f"{time.time() - t0:.1f}",
            "if_firing_density": f"{fire:.6f}",
            "eval_T": eval_t,
            "eval_seed": int(getattr(args, "eval_seed", args.seed)),
        }
    )
    return row


def mapping_card(model) -> dict:
    rows = collect_weight_layer_matches(model, LAYER_MAP)
    summary = summarize_weight_layer_matches(rows)
    return {
        "n_if": count_if(model),
        "n_maxpool": count_maxpool2d(model),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        **summary,
    }


def self_check(device) -> dict:
    cards = {}
    dummy = torch.zeros(2, 3, 224, 224, device=device)
    for task, expected_if, expected_matched, expected_unmatched in (
        ("cls", 13, 13, 1),
        ("seg", 18, 18, 1),
    ):
        model, n_copied = make_model(task, 0, device, load_imagenet=False, head_if=False)
        model.eval()
        logits = model(dummy)
        card = {"n_imagenet_copied": n_copied, "logits_shape": list(logits.shape), **mapping_card(model)}
        if task == "cls":
            assert tuple(logits.shape) == (2, NUM_CLS), card
        else:
            assert tuple(logits.shape) == (2, NUM_SEG, 224, 224), card
        assert card["n_if"] == expected_if, card
        assert card["n_matched"] == expected_matched, card
        assert card["n_unmatched"] == expected_unmatched, card
        assert card["n_maxpool"] == 0, card
        assert card["unmatched_body"] == [], card
        model.set_T(2)
        model.set_mode("rate_uniform")
        logits_t = model(dummy)
        assert tuple(logits_t.shape) == tuple(logits.shape), (logits_t.shape, logits.shape)
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
        cards[task] = card
    head_seg, _ = make_model("seg", 0, device, load_imagenet=False, head_if=True)
    cards["seg_head_if_n_if"] = count_if(head_seg)
    assert cards["seg_head_if_n_if"] == 19, cards
    print(json.dumps(cards, indent=2), flush=True)
    return cards


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
    train_loader, _test_loader, n_train, n_test = pet_loaders(args, pin)
    model, n_copied = make_model(args.task, args.seed, device, load_imagenet=True, head_if=args.head_if)
    print(
        json.dumps(
            {
                "init": "imagenet_vgg16_bn_encoder",
                "n_copied": n_copied,
                "task": args.task,
                "n_train": n_train,
                "n_test": n_test,
                "head_if": args.head_if,
                **mapping_card(model),
            }
        ),
        flush=True,
    )
    thresh = init_if_thresholds_percentile(
        model, train_loader, device, q=PERCENTILE, max_images=PERCENTILE_IMAGES
    )
    (out / "if_thresh_init.json").write_text(json.dumps(thresh, indent=2) + "\n")
    extra = {
        "method": args.method,
        "task": args.task,
        "head_if": args.head_if,
        "spec": {k: v for k, v in spec.items() if k != "mne_kw"},
    }
    if spec["regularizer"] == "mne_l2":
        report = matched_weight_beta(model)
        spec["reg_coeff"] = report["beta_match"]
        extra["strength"] = report
        (out / "beta_match.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"beta_match": report["beta_match"]}), flush=True)
    dump_mne_mapping_report(model, out / "mapping_init", layer_map=LAYER_MAP, quant_level=LVAL, extra=extra)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    opt = optimizer_for(model, spec, args.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(MILESTONES), gamma=0.1)
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
        for batch in train_loader:
            images = batch[0].to(device, non_blocking=True)
            target = batch[1].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(images)
            ce = criterion(logits, target)
            extra_loss = reg_loss(model, spec)
            loss = ce
            reg_value = 0.0
            if extra_loss is not None and spec["reg_coeff"] is not None:
                loss = loss + float(spec["reg_coeff"]) * extra_loss
                reg_value = float(extra_loss.detach())
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
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "method": args.method,
                "task": args.task,
                "head_if": args.head_if,
                "reg_coeff": spec["reg_coeff"],
            },
            ckpt,
        )
    return ckpt


def evaluate_ckpt(args, spec: dict, device, ckpt: Path) -> dict:
    pin = device.type == "cuda"
    _train, test_loader, n_train, n_test = pet_loaders(args, pin)
    model, _ = make_model(args.task, args.seed, device, load_imagenet=False, head_if=args.head_if)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        if bool(state.get("head_if", args.head_if)) != bool(args.head_if):
            raise ValueError(f"checkpoint head_if={state.get('head_if')} != --head-if {args.head_if}")
        if state.get("reg_coeff") is not None:
            spec["reg_coeff"] = state["reg_coeff"]
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    primary = "acc" if args.task == "cls" else "fg_iou"
    ann = evaluate_task(args, model, test_loader, device, 0.0, 0, "normal")
    print(json.dumps({"ann_T0": ann}), flush=True)
    (out / "ann_test.json").write_text(json.dumps(ann, indent=2) + "\n")
    rows = []
    for sigma in SIGMAS:
        row = evaluate_task(args, model, test_loader, device, float(sigma), args.eval_t, "rate_uniform")
        rows.append(row)
        print(json.dumps(row), flush=True)
    write_csv(out / "test_sweep.csv", rows)
    ann_metric = float(ann[primary])
    snn_clean = metric_at(rows, 0.0, primary)
    card = {
        "method": args.method,
        "label": spec["label"],
        "task": args.task,
        "arch": ARCH,
        "seed": args.seed,
        "eval_seed": int(getattr(args, "eval_seed", args.seed)),
        "quant_level": LVAL,
        "eval_T": args.eval_t,
        "head_if": args.head_if,
        "readout": "if" if args.head_if else "linear",
        "regularizer": spec["regularizer"],
        "weight_decay": spec["weight_decay"],
        "reg_coeff": spec["reg_coeff"],
        "noise_position": "post_input_if",
        "if_mode": "rate_uniform",
        "n_train": n_train,
        "n_test": n_test,
        "checkpoint": str(ckpt),
        "primary": primary,
        "ann_T0": ann_metric,
        "snn_clean": snn_clean,
        "conversion_gap": ann_metric - snn_clean,
        "test_sigma0p5": metric_at(rows, 0.5, primary),
        "test_sigma1": metric_at(rows, 1.0, primary),
        "test_sigma2": metric_at(rows, 2.0, primary),
        "test_sigma3": metric_at(rows, 3.0, primary),
        "test_sigma5": metric_at(rows, 5.0, primary),
        "test_auc_full": auc_range(rows, 0.0, 5.0, primary),
        "test_auc_high": auc_range(rows, HIGH_NOISE_MIN, 5.0, primary),
        **mapping_card(model),
    }
    if args.task == "seg":
        card["snn_clean_dice"] = metric_at(rows, 0.0, "dice")
        card["ann_dice"] = float(ann["dice"])
    (out / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2), flush=True)
    return card


def summarize(out_root: Path) -> None:
    cards = []
    for path in sorted(out_root.glob("*/*/scorecard.json")):
        cards.append(json.loads(path.read_text()))
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    print(f"{'task':<4} {'method':<10} {'ann':>8} {'snn0':>8} {'gap':>8} {'s1':>8} {'s5':>8} {'AUC':>8}")
    for card in cards:
        print(
            f"{card['task']:<4} {card['method']:<10} {card['ann_T0']:8.2f} {card['snn_clean']:8.2f} "
            f"{card['conversion_gap']:8.2f} {card['test_sigma1']:8.2f} {card['test_sigma5']:8.2f} "
            f"{card['test_auc_full']:8.2f}"
        )


def main() -> None:
    args = parse_args()
    if args.download_pet:
        print(download_pet(args.pet_root), flush=True)
        return
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
            "On a login node: python noise3_exp/run_pet_vgg16bn_cls_seg_seed42.py --download-pet --pet-root DIR"
        )
    spec = method_spec(args.method)
    args.out_root.mkdir(parents=True, exist_ok=True)
    ckpt = train_one(args, spec, device)
    evaluate_ckpt(args, spec, device, ckpt)
    print(f"Wrote {cfg_dir(args)}", flush=True)


if __name__ == "__main__":
    main()
