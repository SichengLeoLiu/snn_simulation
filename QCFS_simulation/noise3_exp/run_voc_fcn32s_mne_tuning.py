#!/usr/bin/env python3
"""Strength-matched FCN-32s tuning, separate from the original five-reg run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
import run_voc_fcn32s_vgg16_nobn_five_regs_seed42 as base
from voc_seg import VOCSegSet, collate_train, collate_val

METHODS = ("l2wo", "mne", "nodetach")
STRENGTHS = (0.1, 1.0, 10.0)
REFERENCE_WD = 5e-4
EPS = 1e-6


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Existing configuration differs: {path}; use a new output root")
    else:
        save_json(path, value)


def split_ids(train_ids, val_ids, n_tune, seed):
    train, val = sorted(set(train_ids)), sorted(set(val_ids))
    if len(train) != len(train_ids) or len(val) != len(val_ids):
        raise ValueError("Duplicate image IDs in dataset split")
    if set(train) & set(val):
        raise ValueError("Training split overlaps official validation; fix trainaug.txt first")
    if not 0 < n_tune < len(train):
        raise ValueError("Tuning size must leave nonempty fit and tuning sets")
    tune = sorted(random.Random(seed).sample(train, n_tune))
    return {"fit": sorted(set(train) - set(tune)), "tune": tune, "val": val}


def prepare(args):
    train = VOCSegSet(args.voc_root, args.train_split, train=False)
    val = VOCSegSet(args.voc_root, "val", train=False)
    value = {"train_split": args.train_split, "split_seed": args.split_seed,
             **split_ids(train.ids, val.ids, args.tune_size, args.split_seed)}
    immutable_json(args.split_file, value)
    print(json.dumps({key: len(value[key]) for key in ("fit", "tune", "val")}), flush=True)


def load_manifest(path):
    data = json.loads(Path(path).read_text())
    parts = [set(data[key]) for key in ("fit", "tune", "val")]
    if any(not part for part in parts) or any(parts[i] & parts[j] for i in range(3) for j in range(i)):
        raise ValueError("Split manifest must contain three nonempty, disjoint sets")
    if any(len(data[key]) != len(set(data[key])) for key in ("fit", "tune", "val")):
        raise ValueError("Duplicate IDs in split manifest")
    return data


@torch.no_grad()
def strength_report(model):
    """Analytic weight gradients for the existing no-BN, mean-channel MNE."""
    rows = []
    for row in base.collect_weight_layer_matches(model, layer_map="legacy"):
        if not row["matched"]:
            continue
        if row["bn"] is not None:
            raise ValueError("This matcher is for no-BN FCN-32s only")
        raw_lam = float(row["if_mod"].thresh.detach().reshape(-1)[0])
        if not math.isfinite(raw_lam):
            raise ValueError(f"Nonfinite threshold in {row['name']}")
        lam = max(1e-3, raw_lam)
        w2 = float(row["weight"].detach().square().sum())
        k = 2 * base.LVAL**2 / (row["out_channels"] * (lam**2 + EPS))
        rows.append({"layer": row["name"], "lambda": lam, "raw_lambda": raw_lam,
                     "out_channels": row["out_channels"], "weight_norm_sq": w2,
                     "unit_mne_weight_gradient_coeff": k,
                     "legacy_rc_to_l2_gradient_ratio": base.MNE_RC * k / REFERENCE_WD})
    raw_norm = math.sqrt(sum(r["unit_mne_weight_gradient_coeff"]**2 * r["weight_norm_sq"] for r in rows))
    ref_norm = REFERENCE_WD * math.sqrt(sum(r["weight_norm_sq"] for r in rows))
    if not rows or not math.isfinite(raw_norm) or raw_norm <= 0 or not math.isfinite(ref_norm):
        raise ValueError("Invalid matched-layer gradient norm")
    return {"beta_match": ref_norm / raw_norm, "unit_mne_grad_norm": raw_norm,
            "reference_l2_grad_norm": ref_norm, "layers": rows}


def optimizer_for(model, method, strength, lr):
    matches = base.collect_weight_layer_matches(model, layer_map="legacy")
    matched = {id(r["weight"]) for r in matches if r["matched"]}
    unmatched = {id(r["weight"]) for r in matches if not r["matched"]}
    groups = {"body": [], "head": [], "other": []}
    for p in model.parameters():
        groups["body" if id(p) in matched else "head" if id(p) in unmatched else "other"].append(p)
    return torch.optim.SGD([
        {"params": groups["body"], "weight_decay": REFERENCE_WD * strength if method == "l2wo" else 0},
        {"params": groups["head"], "weight_decay": REFERENCE_WD},
        {"params": groups["other"], "weight_decay": 0},
    ], lr=lr, momentum=0.9)


def datasets(args, manifest, confirm):
    train = VOCSegSet(args.voc_root, manifest["train_split"], train=True, crop=args.crop)
    official = VOCSegSet(args.voc_root, "val", train=False)
    if set(train.ids) != set(manifest["fit"] + manifest["tune"]) or set(official.ids) != set(manifest["val"]):
        raise ValueError("Dataset IDs have changed since preparing the split")
    train.ids = sorted(manifest["fit"] + (manifest["tune"] if confirm else []))
    evaluation = official if confirm else VOCSegSet(args.voc_root, manifest["train_split"], train=False)
    evaluation.ids = list(manifest["val"] if confirm else manifest["tune"])
    # Official validation must use official masks, never an augmented substitute.
    if confirm:
        evaluation.mask_dir = evaluation.root / "VOC2012" / "SegmentationClass"
        for image_id in evaluation.ids:
            if not (evaluation.mask_dir / f"{image_id}.png").is_file():
                raise FileNotFoundError(f"Official validation mask missing: {image_id}")
    for dataset in (train, evaluation):
        for image_id in dataset.ids:
            if not (dataset.img_dir / f"{image_id}.jpg").is_file():
                raise FileNotFoundError(f"Image missing: {image_id}")
            dataset._mask_path(image_id)
    return train, evaluation


def accumulation_divisor(step, n_batches, accum):
    return min(accum, n_batches - (step // accum) * accum)


def skip_train_batch(masks, loss):
    if int((masks != base.IGNORE_INDEX).sum()) == 0:
        return "all_ignore"
    if not torch.isfinite(loss):
        return "nonfinite"
    return None


def protocol(args, manifest):
    source_files = [Path(__file__), Path(base.__file__), base.ROOT / "utils.py",
                    base.ROOT / "Models/FCN.py", base.ROOT / "Models/layer.py", base.ROOT / "Models/VGG.py",
                    EXP / "voc_seg.py", EXP / "voc_ssd.py"]
    return {"split_hash": digest(manifest), "epochs": args.epochs, "lr": args.lr,
            "crop": args.crop, "accum": args.accum, "batch_size": 1,
            "workers": args.workers, "L": base.LVAL, "T": base.TEST_T,
            "sigmas": list(base.SIGMAS), "reference_wd": REFERENCE_WD,
            "matching": "initial_weight_gradient_norm_frozen_beta",
            "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}


def run_dir(root, stage, method, strength, seed):
    return Path(root) / stage / f"{method}_s{strength:g}_seed{seed}"


def run(args):
    manifest = load_manifest(args.split_file)
    selected = None
    confirm = args.stage == "confirm"
    if confirm:
        selected = json.loads(args.selection.read_text())
        pick = selected["selected"][args.method]
        if pick is None:
            raise ValueError("No eligible tuning configuration for this method")
        args.strength = pick["strength"]
        if selected["protocol"] != protocol(args, manifest):
            raise ValueError("Confirmation recipe differs from the locked tuning protocol")
    if args.strength is None:
        raise ValueError("--strength is required for tuning")
    config = {"protocol": protocol(args, manifest), "method": args.method, "seed": args.seed,
              "strength": args.strength, "stage": args.stage, "smoke": args.smoke,
              "selection_hash": digest(selected) if selected else None}
    stage = "smoke" if args.smoke else args.stage
    out = run_dir(args.out_root, stage, args.method, args.strength, args.seed)
    locked = out / "config.json"
    if locked.exists():
        existing = json.loads(locked.read_text())
        for key in ("method", "seed", "strength", "stage", "smoke"):
            if existing.get(key) != config.get(key):
                raise ValueError(f"Existing {locked} has {key}={existing.get(key)!r}, not {config.get(key)!r}")
        config = existing
    else:
        immutable_json(locked, config)
    if (out / "result.json").exists():
        print(f"Already complete: {out}", flush=True)
        return
    train, evaluation = datasets(args, manifest, confirm)
    if args.smoke:
        train.ids, evaluation.ids = train.ids[:2], evaluation.ids[:2]
    device = base.get_torch_device(args.device)
    train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=args.workers,
                              collate_fn=collate_train, pin_memory=device.type == "cuda")
    eval_loader = DataLoader(evaluation, batch_size=1, shuffle=False, num_workers=args.workers,
                             collate_fn=collate_val, pin_memory=device.type == "cuda")
    checkpoint = out / "last.pth"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True) if checkpoint.exists() else None
    model, copied = base.make_model(args.seed, device, load_imagenet=saved is None)
    spec = base.method_spec(args.method)
    opt = optimizer_for(model, args.method, args.strength, args.lr)
    milestones = sorted({max(1, int(args.epochs * frac)) for frac in (0.6, 0.8)})
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)
    if saved is not None:
        if saved["config"] != config:
            raise ValueError("Checkpoint configuration mismatch")
        model.load_state_dict(saved["state_dict"])
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        start, beta, history = saved["epoch"], saved["beta"], saved["history"]
        del saved
    else:
        thresholds = base.init_if_thresholds_percentile(model, train_loader, device,
                         q=base.PERCENTILE, max_images=min(base.PERCENTILE_IMAGES, len(train)))
        report = strength_report(model)
        beta = args.strength * report["beta_match"]
        save_json(out / "initial_strength.json", {**report, "beta": beta,
                  "thresholds": thresholds, "imagenet_init": copied})
        start, history = 0, []
    spec["reg_coeff"] = beta if args.method != "l2wo" else None
    criterion = torch.nn.CrossEntropyLoss(ignore_index=base.IGNORE_INDEX)
    epochs = 1 if args.smoke else args.epochs
    print(json.dumps({"output": str(out), "beta": spec["reg_coeff"],
                      "n_train": len(train), "n_eval": len(evaluation)}), flush=True)
    for epoch in range(start + 1, epochs + 1):
        base.seed_all(args.seed + epoch)
        model.train()
        model.set_T(0)
        model.set_mode("normal")
        model.set_first_layer_input_noise_sigma(0)
        opt.zero_grad(set_to_none=True)
        t0, ce_sum, reg_sum = time.time(), 0.0, 0.0
        for step, (images, masks, ids) in enumerate(train_loader):
            masks = masks.to(device)
            ce = criterion(model(images.to(device)), masks)
            extra = base.reg_loss(model, spec)
            penalty = beta * extra if extra is not None else ce.new_zeros(())
            loss = ce + penalty
            reason = skip_train_batch(masks, loss)
            if reason:
                print(f"skip {reason} epoch={epoch} step={step} ids={list(ids)}", flush=True)
                continue
            (loss / accumulation_divisor(step, len(train_loader), args.accum)).backward()
            if (step + 1) % args.accum == 0 or step + 1 == len(train_loader):
                opt.step()
                opt.zero_grad(set_to_none=True)
            ce_sum += float(ce.detach())
            reg_sum += float(penalty.detach())
            if (step + 1) % 100 == 0:
                print(f"epoch={epoch}/{epochs} batch={step + 1}/{len(train_loader)} ce={float(ce.detach()):.4f}", flush=True)
        stats = strength_report(model)
        history.append({"epoch": epoch, "lr": opt.param_groups[0]["lr"],
                        "ce": ce_sum / len(train_loader), "weighted_reg": reg_sum / len(train_loader),
                        "mne_to_reference_weight_grad_norm": beta / stats["beta_match"] if args.method != "l2wo" else None,
                        "seconds": time.time() - t0})
        sched.step()
        tmp = checkpoint.with_suffix(".tmp")
        torch.save({"config": config, "state_dict": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch, "beta": beta, "history": history}, tmp)
        tmp.replace(checkpoint)
        base.write_csv(out / "epoch_log.csv", history)
        print(json.dumps(history[-1]), flush=True)
    save_json(out / "final_strength.json", strength_report(model))
    ann = base.evaluate_miou(model, eval_loader, device, 0.0, args.seed, eval_t=0, eval_mode="normal")
    rows = []
    for sigma in base.SIGMAS:
        row = base.evaluate_miou(model, eval_loader, device, sigma, args.seed)
        rows.append(row)
        save_json(out / "evaluation_progress.json", {"ann": ann, "snn": rows})
        print(json.dumps({k: v for k, v in row.items() if k != "per_class_iou"}), flush=True)
    score = base.auc_range(rows, 0, 5) / 5
    save_json(out / "result.json", {"config": config, "beta": spec["reg_coeff"], "ann": ann,
              "snn": rows, "auc_mean_miou": score, "clean_miou": base.metric_at(rows, 0),
              "sigma5_miou": base.metric_at(rows, 5), "n_train": len(train), "n_eval": len(evaluation)})


def rank_candidates(rows, clean_tolerance):
    if clean_tolerance < 0 or not math.isfinite(clean_tolerance):
        raise ValueError("Clean tolerance must be finite and nonnegative")
    floor = max(r["clean_miou"] for r in rows if r["method"] == "l2wo") - clean_tolerance
    selected = {}
    for method in METHODS:
        valid = [r for r in rows if r["method"] == method and r["clean_miou"] >= floor]
        selected[method] = max(valid, key=lambda r: (r["auc_mean_miou"], r["clean_miou"], -r["strength"])) if valid else None
    return floor, selected


def select(args):
    manifest = load_manifest(args.split_file)
    rows, shared = [], None
    for method in METHODS:
        for strength in STRENGTHS:
            path = run_dir(args.out_root, "tune", method, strength, args.seed) / "result.json"
            result = json.loads(path.read_text())
            cfg = result["config"]
            if (cfg["stage"] != "tune" or cfg["smoke"] or cfg["method"] != method
                    or cfg["strength"] != strength or cfg["seed"] != args.seed):
                raise ValueError(f"Invalid tuning result: {path}")
            shared = shared or cfg["protocol"]
            if shared != cfg["protocol"] or shared["split_hash"] != digest(manifest):
                raise ValueError("Cannot compare runs with different protocols")
            if result["n_eval"] != len(manifest["tune"]) or any(int(r["n_images"]) != len(manifest["tune"]) for r in result["snn"]):
                raise ValueError("Incomplete tuning-set evaluation")
            for key in ("clean_miou", "auc_mean_miou", "sigma5_miou"):
                if not math.isfinite(result[key]):
                    raise ValueError(f"Invalid metric: {key}")
            rows.append({"method": method, "strength": strength, "beta": result["beta"],
                         **{k: result[k] for k in ("clean_miou", "auc_mean_miou", "sigma5_miou")}})
    floor, selected = rank_candidates(rows, args.clean_tolerance)
    value = {"protocol": shared, "selection_seed": args.seed, "clean_floor": floor,
             "clean_tolerance_pp": args.clean_tolerance, "objective": "trapezoid_mean_miou_sigma_0_to_5",
             "selected": selected, "all_candidates": rows}
    immutable_json(args.selection, value)
    base.write_csv(args.out_root / "tuning_summary.csv", rows)
    print(json.dumps(value, indent=2), flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "run", "select"))
    p.add_argument("--voc-root", type=Path, default=Path.home() / "datasets")
    p.add_argument("--out-root", type=Path, default=base.ROOT.parent / "important_results/voc_fcn32s_mne_tuning")
    p.add_argument("--split-file", type=Path)
    p.add_argument("--selection", type=Path)
    p.add_argument("--train-split", choices=("train", "trainaug"), default="train")
    p.add_argument("--tune-size", type=int, default=200)
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--stage", choices=("tune", "confirm"), default="tune")
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--strength", type=float, choices=STRENGTHS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--clean-tolerance", type=float, default=1.0)
    p.add_argument("--smoke", action="store_true", help="Two training/evaluation images, excluded from selection")
    args = p.parse_args(argv)
    args.voc_root, args.out_root = args.voc_root.expanduser().resolve(), args.out_root.expanduser().resolve()
    args.split_file = args.split_file or args.out_root / "split.json"
    args.selection = args.selection or args.out_root / "selection.json"
    if args.action == "run" and args.method is None:
        p.error("run requires --method")
    if args.epochs < 2 or args.accum < 1 or args.workers < 0 or args.crop < 32 or args.lr <= 0 or not math.isfinite(args.lr):
        p.error("Require epochs>=2, accum>=1, workers>=0, crop>=32 and finite lr>0")
    return args


if __name__ == "__main__":
    args = parse_args()
    {"prepare": prepare, "run": run, "select": select}[args.action](args)
