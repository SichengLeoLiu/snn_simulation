#!/usr/bin/env python3
"""CIFAR VGG-16 seed-42 MNE-L2 component ablation (paper main-text table).

This is NOT the block-role / scope ablation. Fixed L=T=16 unless --suite lscale.

Component variants (suite=component)
------------------------------------
    l2wo       sum ||W||_F^2 via optimizer WD on Conv/Linear weights
    effective  sum_l M_eff,l          (BN-folded, no 1/λ^2, no L^2)
    mne        sum_l L^2 M_eff,l / λ_l^2   (published Old MNE-L2)
    nodetach   same as mne, gradients into λ and BN γ

Intensity (suite=component)
---------------------------
    fixed   same knob as the published recipe: L2-wo WD=5e-4, MNE-family rc=1e-4
    gmatch  match init ||rc ∇_W R||_2 (or WD||W|| for L2-wo) to published MNE-L2

At init, ∇_W of mne and nodetach are identical, so nodetach/gmatch == nodetach/fixed
and mne/gmatch == mne/fixed. Those cells reuse the corresponding fixed checkpoint.

L^2 scaling (suite=lscale)
--------------------------
    R_scale    = (L / 16)^2 * sum M_eff / λ^2
    R_noscale  = sum M_eff / λ^2
    rc         = 1e-4 * 16^2 = 0.0256   so L=16 equals published MNE-L2
    train/eval L=T in {4, 8, 16}, same rc, no per-L retune.

Do not retune α/τ. One-sided q-assignment is a separate already-run experiment.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Models import modelpool  # noqa: E402
from Models.VGG import remap_legacy_vgg_state_dict  # noqa: E402
from run_cifar_vgg16_onesided_q_assignment_ablation import (  # noqa: E402
    EPOCHS,
    LR,
    LVAL,
    SIGMAS,
    TEST_T,
    snn_metrics,
    test_loader,
    val_loader,
    write_csv,
)
import measure_vgg16_horowitz_energy as energy  # noqa: E402
from utils import (  # noqa: E402
    _mne_covered_weight_ids,
    _parameter_grad_norm,
    ce_vs_reg_grad_ratio,
    compute_mne_l2_regularization,
    dump_mne_mapping_report,
    get_torch_device,
    seed_all,
)

ARCH = "vgg16"
SEED = 42
MNE_RC = 1e-4
L2_WD = 5e-4
L_REF = 16.0
LSCALE_RC = MNE_RC * (L_REF ** 2)  # 0.0256; L=16 matches published MNE
CKPT_ROOT_DEFAULT = Path("/home/595/sl9144/codes/snn_simulation/QCFS_simulation")

COMPONENT_VARIANTS = ("l2wo", "effective", "mne", "nodetach")
INTENSITIES = ("fixed", "gmatch")
LSCALE_MODES = ("scale", "noscale")
QUANT_LS = (4, 8, 16)

MNE_TRAIN_KW = dict(
    detach_lambda=True,
    detach_bn_stats=True,
    detach_bn_affine=True,
    divide_by_lambda=True,
    scale_by_l=True,
    l_ref=None,
    fold_bn=True,
)
EFFECTIVE_TRAIN_KW = dict(
    detach_lambda=True,
    detach_bn_stats=True,
    detach_bn_affine=True,
    divide_by_lambda=False,
    scale_by_l=False,
    l_ref=None,
    fold_bn=True,
)
NODETACH_TRAIN_KW = dict(
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
    parser.add_argument("--suite", choices=("component", "lscale", "l2l"), default="component")
    parser.add_argument("--variant", choices=(*COMPONENT_VARIANTS, "l2all"), default=None)
    parser.add_argument("--intensity", choices=INTENSITIES, default="fixed")
    parser.add_argument("--lscale-mode", choices=LSCALE_MODES, default=None)
    parser.add_argument("--quant-L", type=int, choices=QUANT_LS, default=16)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--probe-gmatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=Path(os.environ.get("CKPT_ROOT", str(CKPT_ROOT_DEFAULT))),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar_vgg16_mne_component_ablation_seed42",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    if args.summarize:
        return args
    if args.suite == "component" and args.variant is None:
        parser.error("suite=component requires --variant")
    if args.suite == "lscale" and args.lscale_mode is None:
        parser.error("suite=lscale requires --lscale-mode")
    if args.suite == "l2l" and args.variant not in ("l2wo", "l2all"):
        parser.error("suite=l2l requires --variant l2wo|l2all")
    return args


def config_name(args) -> str:
    if args.suite == "lscale":
        return f"lscale_{args.lscale_mode}_L{int(args.quant_L)}"
    if args.suite == "l2l":
        return f"l2l_{args.variant}_L{int(args.quant_L)}"
    return f"comp_{args.variant}_{args.intensity}"


def train_L(args) -> int:
    return int(args.quant_L) if args.suite in ("lscale", "l2l") else LVAL


def eval_T(args) -> int:
    return train_L(args) if args.suite in ("lscale", "l2l") else TEST_T


def suffix(args) -> str:
    return f"{config_name(args)}_seed{args.seed}_L{train_L(args)}_trainT0"


def ckpt_filename(args) -> str:
    return f"{ARCH}_L[{train_L(args)}]_{suffix(args)}.pth"


def cfg_dir(args) -> Path:
    return args.out_root / args.dataset / config_name(args)


def ckpt_path(args) -> Path:
    return cfg_dir(args) / "checkpoints" / ckpt_filename(args)


def gmatch_is_identity(args) -> bool:
    return args.suite == "component" and args.intensity == "gmatch" and args.variant in (
        "mne",
        "nodetach",
    )


def find_five_regs(args, method: str) -> Path | None:
    rc = "rc0p0001" if method == "old_detach" else "rcnone"
    names = [
        f"{ARCH}_L[{LVAL}]_mneablate_{args.dataset}_{method}_{rc}_seed{args.seed}_L{LVAL}_trainT0.pth",
        f"{ARCH}_L[{LVAL}]_mneablate_{args.dataset}_{method}_{rc}_seed{args.seed}_L{LVAL}.pth",
    ]
    roots = [
        args.ckpt_root / f"{args.dataset}-checkpoints",
        ROOT / f"{args.dataset}-checkpoints",
        Path(f"/scratch/gs14/sl9144/snn_results/{args.dataset}_{ARCH}_five_regs_sigma0_5_5seed"),
    ]
    for root in roots:
        for name in names:
            path = root / name
            if path.is_file():
                return path
    return None


def reuse_ckpt(args) -> Path | None:
    if args.retrain:
        return None
    if args.suite == "component":
        if args.variant == "l2wo" and args.intensity == "fixed":
            return find_five_regs(args, "weight_decay_weights_only")
        if args.variant == "mne" and args.intensity in ("fixed", "gmatch"):
            return find_five_regs(args, "old_detach")
        if gmatch_is_identity(args) and args.variant == "nodetach":
            alias = argparse.Namespace(**vars(args))
            alias.intensity = "fixed"
            path = ckpt_path(alias)
            if path.is_file():
                return path
        return None
    if args.suite == "lscale" and train_L(args) == LVAL:
        return find_five_regs(args, "old_detach")
    if args.suite == "l2l" and train_L(args) == LVAL:
        if args.variant == "l2wo":
            return find_five_regs(args, "weight_decay_weights_only")
        if args.variant == "l2all":
            return find_five_regs(args, "weight_decay")
    return None


def mne_kwargs_for(args) -> dict:
    if args.suite == "lscale":
        kw = dict(MNE_TRAIN_KW)
        if args.lscale_mode == "scale":
            kw["l_ref"] = L_REF
            kw["scale_by_l"] = True
        else:
            kw["l_ref"] = None
            kw["scale_by_l"] = False
        return kw
    if args.variant == "effective":
        return dict(EFFECTIVE_TRAIN_KW)
    if args.variant == "nodetach":
        return dict(NODETACH_TRAIN_KW)
    return dict(MNE_TRAIN_KW)


def extra_train_flags(args) -> list[str]:
    if args.suite == "l2l":
        return []
    flags: list[str] = ["--mne_layer_map", "legacy"]
    if args.suite == "lscale":
        flags.append("--mne_detach_lambda")
        if args.lscale_mode == "scale":
            flags += ["--mne_l_ref", str(int(L_REF))]
        else:
            flags.append("--mne_no_l_scale")
        return flags
    if args.variant == "effective":
        flags += ["--mne_detach_lambda", "--mne_no_lambda", "--mne_no_l_scale"]
        return flags
    if args.variant == "nodetach":
        flags += ["--mne_no_detach_bn_affine"]
        return flags
    flags.append("--mne_detach_lambda")
    return flags


def init_model(dataset: str, seed: int, quant_level: int):
    seed_all(seed)
    model = modelpool(ARCH, dataset)
    model._mne_layer_map = "legacy"
    model.set_L(quant_level)
    model.set_T(0)
    return model


def weight_tensors(model, mne_covered_only: bool):
    if mne_covered_only:
        covered = _mne_covered_weight_ids(model)
        return [parameter for parameter in model.parameters() if id(parameter) in covered]
    weights = []
    for module in model.modules():
        if not isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            continue
        weight = getattr(module, "weight", None)
        if weight is not None and weight.requires_grad:
            weights.append(weight)
    return weights


def concat_l2(tensors) -> float:
    total = None
    for tensor in tensors:
        term = tensor.detach().pow(2).sum()
        total = term if total is None else total + term
    if total is None:
        return 0.0
    return float(total.sqrt().item())


def mne_weight_grad_norm(model, quant_level: int, kwargs: dict) -> float:
    penalty = compute_mne_l2_regularization(model, quant_level=quant_level, **kwargs)
    params = weight_tensors(model, mne_covered_only=True)
    return float(_parameter_grad_norm(penalty, params, retain_graph=False).item())


def probe_gmatch(args) -> dict:
    quant_level = train_L(args)
    model = init_model(args.dataset, args.seed, quant_level)
    ref_norm = mne_weight_grad_norm(model, quant_level, MNE_TRAIN_KW)
    target = MNE_RC * ref_norm
    card = {
        "dataset": args.dataset,
        "seed": args.seed,
        "quant_level": quant_level,
        "mne_rc": MNE_RC,
        "mne_weight_grad_norm": ref_norm,
        "target_coeff_grad_norm": target,
    }
    if args.suite == "lscale" or args.variant in ("effective", "mne", "nodetach"):
        var_kw = mne_kwargs_for(args)
        var_norm = mne_weight_grad_norm(model, quant_level, var_kw)
        rc = target / max(var_norm, 1e-30)
        card.update(
            {
                "variant_weight_grad_norm": var_norm,
                "gmatch_reg_coeff": rc,
                "gmatch_scale_vs_fixed_mne_rc": rc / MNE_RC,
            }
        )
    if args.suite == "component" and args.variant == "l2wo":
        w_norm = concat_l2(weight_tensors(model, mne_covered_only=False))
        wd = target / max(w_norm, 1e-30)
        card.update(
            {
                "l2wo_weight_l2": w_norm,
                "fixed_weight_decay": L2_WD,
                "gmatch_weight_decay": wd,
                "gmatch_scale_vs_fixed_wd": wd / L2_WD,
            }
        )
    return card


def chosen_coeff(args, gmatch_card: dict) -> dict:
    if args.suite == "l2l":
        regularizer = (
            "weight_decay" if args.variant == "l2all" else "weight_decay_weights_only"
        )
        return {"regularizer": regularizer, "weight_decay": L2_WD, "reg_coeff": None}
    if args.suite == "lscale":
        return {"regularizer": "mne_l2", "weight_decay": 0.0, "reg_coeff": LSCALE_RC}
    if args.variant == "l2wo":
        wd = L2_WD
        if args.intensity == "gmatch":
            wd = float(gmatch_card["gmatch_weight_decay"])
        return {"regularizer": "weight_decay_weights_only", "weight_decay": wd, "reg_coeff": None}
    rc = MNE_RC
    if args.intensity == "gmatch" and not gmatch_is_identity(args):
        rc = float(gmatch_card["gmatch_reg_coeff"])
    return {"regularizer": "mne_l2", "weight_decay": 0.0, "reg_coeff": rc}


def train_cmd(args, coeff: dict) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "main_train.py"),
        "-data", args.dataset,
        "-arch", ARCH,
        "-L", str(train_L(args)),
        "-T", "0",
        "--epochs", str(args.epochs),
        "-lr", str(LR),
        "-b", str(args.batch_size),
        "-j", str(args.workers),
        "--seed", str(args.seed),
        "--device", args.device,
        "--spike_schedule", "normal",
        "--ckpt-save-mode", "best",
        "--ckpt-dir", str(cfg_dir(args) / "checkpoints"),
        "-suffix", suffix(args),
        "--regularizer", coeff["regularizer"],
        "--weight_decay", str(coeff["weight_decay"]),
    ]
    if coeff["reg_coeff"] is not None:
        cmd += ["--reg_coeff", str(coeff["reg_coeff"])]
    if coeff["regularizer"] == "mne_l2":
        cmd += extra_train_flags(args)
        cmd += [
            "--epoch_log_csv", str(cfg_dir(args) / "epoch_log.csv"),
            "--mapping_diag_dir", str(cfg_dir(args) / "mapping_init"),
        ]
    return cmd


def train(args, coeff: dict) -> Path:
    out = cfg_dir(args)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_path(args)
    if ckpt.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {ckpt}", flush=True)
        return ckpt
    reused = None if args.retrain else reuse_ckpt(args)
    if reused is not None and not args.test_only:
        shutil.copy2(reused, ckpt)
        print(f"[REUSE CKPT] {reused} -> {ckpt}", flush=True)
        return ckpt
    if args.test_only:
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        return ckpt
    if args.dry_run:
        print("[DRY RUN]", " ".join(train_cmd(args, coeff)), flush=True)
        return ckpt
    cmd = train_cmd(args, coeff)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"training finished but missing {ckpt}")
    return ckpt


def load_model(ckpt: Path, device, dataset: str, quant_level: int, time_steps: int):
    model = modelpool(ARCH, dataset)
    model._mne_layer_map = "legacy"
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(remap_legacy_vgg_state_dict(state), strict=True)
    model.set_L(quant_level)
    model.set_T(time_steps)
    model.set_mode("rate_uniform")
    model.set_spike_schedule("normal")
    model.set_first_layer_input_noise_position("post_input_if")
    model.set_first_layer_input_noise_type("gaussian")
    return model.to(device).eval()


def sweep(model, loader, device, split: str, seed: int, time_steps: int) -> list[dict]:
    rows = []
    for sigma in SIGMAS:
        seed_all(seed)
        model.set_first_layer_input_noise_sigma(float(sigma))
        result = energy.evaluate(model, loader, device, time_steps, 0)
        use_energy = "yes" if abs(sigma) < 1e-12 else "no"
        row = {
            "split": split,
            "sigma": f"{sigma:g}",
            "accuracy": f"{result['accuracy']:.6f}",
            "if_firing_density": f"{result['if_firing_density']:.6f}",
            "energy_mJ": f"{result['energy_mJ']:.8f}",
            "use_for_energy": use_energy,
            "n_samples": result["n_samples"],
            "time_steps": time_steps,
        }
        rows.append(row)
        extra = "" if use_energy == "yes" else " [diag energy]"
        print(
            f"{split:<5} T={time_steps} sigma={sigma:g} acc={result['accuracy']:.2f} "
            f"fire={result['if_firing_density']:.4f} "
            f"E={result['energy_mJ']:.4f} mJ{extra}",
            flush=True,
        )
    return rows


def make_reg_fn(args):
    if args.suite == "l2l" or (args.suite == "component" and args.variant == "l2wo"):
        return None
    kw = mne_kwargs_for(args)
    level = train_L(args)
    return lambda m, t, q: compute_mne_l2_regularization(m, quant_level=level, **kw)


def summarize(out_root: Path) -> None:
    cards = []
    for path in sorted(out_root.glob("*/*/scorecard.json")):
        cards.append(json.loads(path.read_text()))
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    print(
        f"{'dataset':<10} {'config':<28} {'rc/wd':>10} "
        f"{'test0':>7} {'test5':>7} {'testAUC':>8} {'testHi':>8}"
    )
    for card in cards:
        knob = card.get("reg_coeff")
        if knob is None:
            knob = card.get("weight_decay")
        print(
            f"{card['dataset']:<10} {card['config']:<28} {float(knob):10.4g} "
            f"{card['test_clean']:7.2f} {card['test_sigma5']:7.2f} "
            f"{card['test_auc_full']:8.2f} {card['test_auc_high']:8.2f}"
        )
    write_csv(out_root / "component_ablation_ranking.csv", cards)
    print(f"Wrote {out_root / 'component_ablation_ranking.csv'}")


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(args.out_root)
        return

    gmatch_card = probe_gmatch(args)
    print(json.dumps({"gmatch_probe": gmatch_card}, indent=2), flush=True)
    if args.probe_gmatch:
        return

    coeff = chosen_coeff(args, gmatch_card)
    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gmatch.json").write_text(json.dumps(gmatch_card, indent=2) + "\n")
    print(
        f"[INFO] {args.dataset} {config_name(args)} seed={args.seed} "
        f"L={train_L(args)} T_eval={eval_T(args)} coeff={coeff}",
        flush=True,
    )
    ckpt = train(args, coeff)
    if args.dry_run:
        return
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = load_model(ckpt, device, args.dataset, train_L(args), eval_T(args))
    dump_mne_mapping_report(
        model,
        out,
        layer_map="legacy",
        quant_level=train_L(args),
        extra={"gmatch": gmatch_card, "coeff": coeff},
    )
    eval_args = argparse.Namespace(
        dataset=args.dataset,
        batch_size=args.batch_size,
        workers=0,
    )
    val_rows = sweep(model, val_loader(eval_args, pin), device, "val", args.seed, eval_T(args))
    write_csv(out / "val_sweep.csv", val_rows)
    test_rows = sweep(model, test_loader(eval_args, pin), device, "test", args.seed, eval_T(args))
    write_csv(out / "test_sweep.csv", test_rows)
    grads = {
        "ce_grad_norm": float("nan"),
        "reg_grad_norm": float("nan"),
        "reg_coeff_grad_norm": float("nan"),
        "reg_ce_ratio": float("nan"),
        "ce_loss": float("nan"),
        "reg_loss": float("nan"),
    }
    reg_fn = make_reg_fn(args)
    if reg_fn is not None and coeff["reg_coeff"] is not None:
        try:
            images, labels = next(iter(val_loader(eval_args, pin)))
            grads = ce_vs_reg_grad_ratio(
                model,
                images,
                labels,
                nn.CrossEntropyLoss(),
                reg_fn,
                0,
                train_L(args),
                coeff["reg_coeff"],
            )
        except Exception as exc:
            print(f"[WARN] ce/reg grad ratio failed after sweep: {exc}", flush=True)
    card = {
        "config": config_name(args),
        "suite": args.suite,
        "variant": args.variant,
        "intensity": args.intensity if args.suite == "component" else None,
        "lscale_mode": args.lscale_mode if args.suite == "lscale" else None,
        "dataset": args.dataset,
        "arch": ARCH,
        "seed": args.seed,
        "quant_level": train_L(args),
        "eval_T": eval_T(args),
        "regularizer": coeff["regularizer"],
        "weight_decay": coeff["weight_decay"],
        "reg_coeff": coeff["reg_coeff"],
        "gmatch_is_identity": gmatch_is_identity(args),
        "checkpoint": str(ckpt),
        **grads,
        **snn_metrics(val_rows, "val"),
        **snn_metrics(test_rows, "test"),
    }
    (out / "scorecard.json").write_text(json.dumps(card, indent=2, default=str) + "\n")
    print(json.dumps(card, indent=2, default=str), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
