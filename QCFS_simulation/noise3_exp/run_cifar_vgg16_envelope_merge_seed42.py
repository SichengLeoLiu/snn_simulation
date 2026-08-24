#!/usr/bin/env python3
"""CIFAR-100 VGG16 seed-42 merge probes (single seed, not 5-seed).

Methods
-------
interp_dual : checkpoint lerp + oracle dual-expert (no training)
twostage    : 240 ep L2-wo/Calibrated, then 60 ep Old MNE with cosine β1;
              also soup the L2-dominant / mid / MNE-dominant snapshots
distill     : noise-aware two-teacher KL (Cal σ≤2, MNE σ>2)
gradnorm    : adaptive β_MNE ∝ ||∇CE|| / ||∇MNE|| plus small L2-all

Selection remains val-holdout; test is logged only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from Models import modelpool  # noqa: E402
from Models.VGG import remap_legacy_vgg_state_dict  # noqa: E402
from Preprocess import datapool  # noqa: E402
from utils import (  # noqa: E402
    compute_mne_l2_regularization,
    get_torch_device,
    seed_all,
    val,
)
import run_cifar_vgg16_hybrid_envelope_screen as screen  # noqa: E402
import run_mne_stability_ablation as ablation  # noqa: E402


SEED = 42
LVAL = 16
DATASET = "cifar100"
ARCH = "vgg16"
ETAS = [i / 10 for i in range(11)]
SWITCH_SIGMA = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=["interp_dual", "twostage", "distill", "gradnorm"],
    )
    parser.add_argument("--stage1", choices=["l2_wo", "calibrated"], default="calibrated")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="ignore existing distill/gradnorm checkpoints and train again",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar100_vgg16_envelope_merge_seed42",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    args.out_root.mkdir(parents=True, exist_ok=True)
    return args


def five_reg_ckpt(variant: str, rc, seed: int) -> Path:
    dummy = SimpleNamespace(arch=ARCH, L=LVAL, train_T=0, reg_warmup_epochs=0)
    path = ablation._resolve_checkpoint(DATASET, variant, rc, seed, None, dummy)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_state(path: Path) -> dict:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return remap_legacy_vgg_state_dict(state)


def build_model(state: dict, device):
    model = modelpool(ARCH, DATASET)
    model.load_state_dict(state, strict=True)
    model.set_L(LVAL)
    model.set_T(screen.TEST_T)
    model.set_mode("rate_uniform")
    model.set_spike_schedule("normal")
    model.set_first_layer_input_noise_position("post_input_if")
    model.set_first_layer_input_noise_type("gaussian")
    return model.to(device).eval()


def recalibrate_bn(model, loader, device) -> None:
    was_t = getattr(model, "T", 0)
    model.set_T(0)
    model.train()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()
            module.momentum = None
    with torch.no_grad():
        for images, _ in loader:
            model(images.to(device))
    model.eval()
    model.set_T(was_t)


def lerp_state(a: dict, b: dict, eta: float, *, keep_mne_thresh: bool) -> dict:
    out = {}
    for key, tensor_a in a.items():
        tensor_b = b[key]
        mixed = (1.0 - eta) * tensor_a.float() + eta * tensor_b.float()
        out[key] = mixed.to(dtype=tensor_a.dtype)
        if keep_mne_thresh and key.endswith("thresh"):
            out[key] = tensor_b.clone()
    return out


def dump_sweep(model, args, cfg_dir: Path, tag: str, device) -> dict:
    pin = device.type == "cuda"
    val_rows = screen.sweep(model, screen.val_loader(args, pin), device, "val")
    test_rows = screen.sweep(model, screen.test_loader(args, pin), device, "test")
    screen.write_csv(cfg_dir / f"{tag}_val_sweep.csv", val_rows)
    screen.write_csv(cfg_dir / f"{tag}_test_sweep.csv", test_rows)
    fake = SimpleNamespace(
        family=tag,
        beta0=None,
        beta1=None,
        alpha=None,
        tau=None,
        beta=None,
        seed=args.seed,
    )
    card = screen.scorecard(val_rows, test_rows, fake, Path(tag))
    card["config"] = tag
    (cfg_dir / f"{tag}_scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
    return card


def run_interp_dual(args, device) -> None:
    cal_path = five_reg_ckpt("calibrated_mne_a0p1", 5e-4, args.seed)
    mne_path = five_reg_ckpt("old_detach", 1e-4, args.seed)
    print(f"[CKPT] cal={cal_path}", flush=True)
    print(f"[CKPT] mne={mne_path}", flush=True)
    cal = load_state(cal_path)
    mne = load_state(mne_path)
    pin = device.type == "cuda"
    bn_loader = screen.val_loader(args, pin)
    cfg_dir = args.out_root / "interp_dual"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for keep_thresh in (False, True):
        mode = "keep_mne_thresh" if keep_thresh else "lerp_all"
        for eta in ETAS:
            tag = f"interp_{mode}_eta{eta:.1f}"
            print(f"[INTERP] {tag}", flush=True)
            state = lerp_state(cal, mne, eta, keep_mne_thresh=keep_thresh)
            model = build_model(state, device)
            recalibrate_bn(model, bn_loader, device)
            card = dump_sweep(model, args, cfg_dir, tag, device)
            card["eta"] = eta
            card["thresh_mode"] = mode
            cards.append(card)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("[DUAL] oracle switch at σ=2", flush=True)
    cal_model = build_model(cal, device)
    mne_model = build_model(mne, device)
    dual_rows = {"val": [], "test": []}
    for split, loader in (
        ("val", screen.val_loader(args, pin)),
        ("test", screen.test_loader(args, pin)),
    ):
        for sigma in screen.SIGMAS:
            model = cal_model if sigma < SWITCH_SIGMA else mne_model
            seed_all(args.seed)
            model.set_first_layer_input_noise_sigma(float(sigma))
            result = __import__(
                "measure_vgg16_horowitz_energy", fromlist=["evaluate"]
            ).evaluate(model, loader, device, screen.TEST_T, 0)
            dual_rows[split].append(
                {
                    "split": split,
                    "sigma": f"{sigma:g}",
                    "accuracy": f"{result['accuracy']:.6f}",
                    "if_firing_density": f"{result['if_firing_density']:.6f}",
                    "energy_mJ": f"{result['energy_mJ']:.8f}",
                    "use_for_energy": "yes" if abs(sigma) < 1e-12 else "no",
                    "n_samples": result["n_samples"],
                    "expert": "cal" if sigma < SWITCH_SIGMA else "mne",
                }
            )
            print(
                f"dual {split} σ={sigma:g} expert={dual_rows[split][-1]['expert']} "
                f"acc={result['accuracy']:.2f}",
                flush=True,
            )
    screen.write_csv(cfg_dir / "dual_oracle_val_sweep.csv", dual_rows["val"])
    screen.write_csv(cfg_dir / "dual_oracle_test_sweep.csv", dual_rows["test"])
    fake = SimpleNamespace(
        family="dual_oracle",
        beta0=None,
        beta1=None,
        alpha=None,
        tau=None,
        beta=None,
        seed=args.seed,
    )
    card = screen.scorecard(dual_rows["val"], dual_rows["test"], fake, Path("dual_oracle"))
    card["config"] = "dual_oracle_sigma2"
    cards.append(card)
    (cfg_dir / "dual_oracle_scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
    screen.write_csv(cfg_dir / "interp_dual_ranking.csv", cards)
    print(f"Wrote {cfg_dir}", flush=True)


def _train_cmd(suffix: str, extra: list[str], args) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "main_train.py"),
        "-data", DATASET,
        "-arch", ARCH,
        "-L", str(LVAL),
        "-T", "0",
        "-lr", "0.1",
        "-b", str(args.batch_size),
        "-j", str(args.workers),
        "--seed", str(args.seed),
        "--device", args.device,
        "--spike_schedule", "normal",
        "-suffix", suffix,
        *extra,
    ]


def run_twostage(args, device) -> None:
    name = f"twostage_{args.stage1}"
    cfg_dir = args.out_root / name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    s1_suffix = f"envmerge_{name}_s1_seed{args.seed}_L{LVAL}_trainT0"
    s2_suffix = f"envmerge_{name}_s2_seed{args.seed}_L{LVAL}_trainT0"
    s1_ckpt = ROOT / f"{DATASET}-checkpoints" / f"{ARCH}_L[{LVAL}]_{s1_suffix}.pth"
    s2_ckpt = ROOT / f"{DATASET}-checkpoints" / f"{ARCH}_L[{LVAL}]_{s2_suffix}.pth"
    mid_ckpt = ROOT / f"{DATASET}-checkpoints" / f"{ARCH}_L[{LVAL}]_{s2_suffix}_ep29.pth"

    if not s1_ckpt.exists():
        extra = [
            "--epochs", "240",
            "--ckpt-save-mode", "last",
            "--lr-cosine-t-max", "300",
        ]
        if args.stage1 == "l2_wo":
            extra += [
                "--regularizer", "weight_decay_weights_only",
                "--weight_decay", "5e-4",
            ]
        else:
            extra += [
                "--regularizer", "calibrated_mne_l2",
                "--weight_decay", "0",
                "--reg_coeff", "5e-4",
                "--calibrated_mne_alpha", "0.1",
                "--calibrated_mne_risk_min", "0.5",
                "--calibrated_mne_risk_max", "2.0",
                "--calibrated_mne_alpha_start_epoch", "30",
                "--calibrated_mne_alpha_warmup_epochs", "50",
            ]
        cmd = _train_cmd(s1_suffix, extra, args)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
    else:
        print(f"[SKIP] {s1_ckpt}", flush=True)

    if not s2_ckpt.exists():
        extra = [
            "--epochs", "60",
            "-lr", "0.01",
            "--ckpt-save-mode", "last",
            "--init-from", str(s1_ckpt),
            "--regularizer", "mne_l2",
            "--mne_detach_lambda",
            "--weight_decay", "1e-4",
            "--reg_coeff", "1e-4",
            "--reg_warmup_epochs", "60",
            "--reg_warmup_schedule", "cosine",
            "--save-epochs", "29",
        ]
        cmd = _train_cmd(s2_suffix, extra, args)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
    else:
        print(f"[SKIP] {s2_ckpt}", flush=True)

    pin = device.type == "cuda"
    bn_loader = screen.val_loader(args, pin)
    snapshots = {
        "l2_dominant": s1_ckpt,
        "balanced": mid_ckpt if mid_ckpt.exists() else s2_ckpt,
        "mne_dominant": s2_ckpt,
    }
    states = {key: load_state(path) for key, path in snapshots.items()}
    keys = list(next(iter(states.values())).keys())
    soup = {
        key: torch.stack([states[name][key].float() for name in snapshots]).mean(0)
        for key in keys
    }
    soup = {key: soup[key].to(dtype=states["mne_dominant"][key].dtype) for key in keys}

    for tag, state in {**states, "soup_bn": soup}.items():
        print(f"[TWOSTAGE EVAL] {tag}", flush=True)
        model = build_model(state, device)
        if tag == "soup_bn":
            recalibrate_bn(model, bn_loader, device)
        dump_sweep(model, args, cfg_dir, f"{name}_{tag}", device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"Wrote {cfg_dir}", flush=True)


def _gamma(sigma: float) -> float:
    if sigma < SWITCH_SIGMA:
        return 1.0
    if abs(sigma - SWITCH_SIGMA) < 1e-9:
        return 0.5
    return 0.0


def run_distill(args, device) -> None:
    cfg_dir = args.out_root / "distill"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"envmerge_distill_seed{args.seed}_L{LVAL}_trainT0"
    ckpt = ROOT / f"{DATASET}-checkpoints" / f"{ARCH}_L[{LVAL}]_{suffix}.pth"
    if ckpt.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {ckpt}", flush=True)
        student = build_model(load_state(ckpt), device)
        dump_sweep(student.eval(), args, cfg_dir, "distill", device)
        return
    train_loader, test_loader = datapool(
        DATASET,
        args.batch_size,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    cal = build_model(load_state(five_reg_ckpt("calibrated_mne_a0p1", 5e-4, args.seed)), device)
    mne = build_model(load_state(five_reg_ckpt("old_detach", 1e-4, args.seed)), device)
    cal.set_T(0)
    mne.set_T(0)
    cal.eval()
    mne.eval()
    student = modelpool(ARCH, DATASET)
    student.set_L(LVAL)
    student.set_T(0)
    student.set_spike_schedule("normal")
    student.set_first_layer_input_noise_position("post_input_if")
    student.set_first_layer_input_noise_type("gaussian")
    student.to(device)
    decay_ids = {
        id(module.weight)
        for module in student.modules()
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear))
        and getattr(module, "weight", None) is not None
    }
    decay, no_decay = [], []
    for parameter in student.parameters():
        (decay if id(parameter) in decay_ids else no_decay).append(parameter)
    optimizer = torch.optim.SGD(
        [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc = -1.0
    for epoch in range(args.epochs):
        student.train()
        running = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            sigma = float(torch.empty(1).uniform_(0.0, 5.0).item())
            gamma = _gamma(sigma)
            for model in (student, cal, mne):
                model.set_first_layer_input_noise_sigma(sigma)
            optimizer.zero_grad()
            logits = student(images)
            log_s = F.log_softmax(logits, dim=1)
            with torch.no_grad():
                p_cal = F.softmax(cal(images), dim=1)
                p_mne = F.softmax(mne(images), dim=1)
            loss = criterion(logits, labels)
            loss = loss + gamma * F.kl_div(log_s, p_cal, reduction="batchmean")
            loss = loss + (1.0 - gamma) * F.kl_div(log_s, p_mne, reduction="batchmean")
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            correct += int(logits.argmax(1).eq(labels).sum().item())
            total += labels.numel()
        scheduler.step()
        student.set_first_layer_input_noise_sigma(0.0)
        acc = val(student, test_loader, T=0, device=device, verbose=False)
        print(
            f"distill epoch {epoch}/{args.epochs} loss={running:.3f} "
            f"train={100 * correct / total:.2f} test={acc:.2f}",
            flush=True,
        )
        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), ckpt)
            print(f"Saving model to {ckpt}", flush=True)
    student.load_state_dict(load_state(ckpt))
    student.set_T(screen.TEST_T)
    student.set_mode("rate_uniform")
    dump_sweep(student.eval(), args, cfg_dir, "distill", device)


def run_gradnorm(args, device) -> None:
    cfg_dir = args.out_root / "gradnorm"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"envmerge_gradnorm_seed{args.seed}_L{LVAL}_trainT0"
    ckpt = ROOT / f"{DATASET}-checkpoints" / f"{ARCH}_L[{LVAL}]_{suffix}.pth"
    if ckpt.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {ckpt}", flush=True)
        model = build_model(load_state(ckpt), device)
        dump_sweep(model.eval(), args, cfg_dir, "gradnorm", device)
        return
    train_loader, test_loader = datapool(
        DATASET,
        args.batch_size,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    model = modelpool(ARCH, DATASET)
    model.set_L(LVAL)
    model.set_T(0)
    model.set_spike_schedule("normal")
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    best_acc = -1.0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        correct = 0
        total = 0
        betas = []
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            ce = criterion(logits, labels)
            mne = compute_mne_l2_regularization(model, LVAL, detach_lambda=True)
            grads_ce = torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True)
            grads_mne = torch.autograd.grad(mne, params, retain_graph=True, allow_unused=True)
            n_ce = math.sqrt(
                sum(float(g.detach().pow(2).sum()) for g in grads_ce if g is not None)
            )
            n_mne = math.sqrt(
                sum(float(g.detach().pow(2).sum()) for g in grads_mne if g is not None)
            )
            beta = 1e-4 * n_ce / (n_mne + 1e-12)
            beta = min(3e-4, max(3e-5, beta))
            betas.append(beta)
            loss = ce + beta * mne
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            correct += int(logits.argmax(1).eq(labels).sum().item())
            total += labels.numel()
        scheduler.step()
        acc = val(model, test_loader, T=0, device=device, verbose=False)
        mean_beta = sum(betas) / max(len(betas), 1)
        print(
            f"gradnorm epoch {epoch}/{args.epochs} loss={running:.3f} "
            f"beta={mean_beta:.3e} train={100 * correct / total:.2f} test={acc:.2f}",
            flush=True,
        )
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), ckpt)
            print(f"Saving model to {ckpt}", flush=True)
    model.load_state_dict(load_state(ckpt))
    model.set_T(screen.TEST_T)
    model.set_mode("rate_uniform")
    model.set_first_layer_input_noise_position("post_input_if")
    model.set_first_layer_input_noise_type("gaussian")
    dump_sweep(model.eval(), args, cfg_dir, "gradnorm", device)


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = get_torch_device(args.device)
    print(f"[INFO] method={args.method} stage1={args.stage1} device={device}", flush=True)
    if args.method == "interp_dual":
        run_interp_dual(args, device)
    elif args.method == "twostage":
        run_twostage(args, device)
    elif args.method == "distill":
        run_distill(args, device)
    else:
        run_gradnorm(args, device)
    print(f"=== ALL DONE {args.method} ===", flush=True)


if __name__ == "__main__":
    main()
