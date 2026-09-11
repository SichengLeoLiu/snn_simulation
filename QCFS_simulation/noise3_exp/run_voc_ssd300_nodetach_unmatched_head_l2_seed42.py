#!/usr/bin/env python3
"""VOC SSD300-VGG16-BN-IF seed-42: no-detach MNE + unmatched detection-head L2.

SSD has 12 loc/conf Convs with no following IF. Ordinary MNE-L2 skips them;
``mne_l2_unmatched`` applies (1/2)||W||_F^2 to those heads at η_head=5e-4,
the L2-wo WD. On this model scope=head and scope=all are the same 12 layers.

Methods
-------
    l2wo       reuse five-regs L2-wo (no retrain)
    nodetach   reuse five-regs no-detach MNE-L2 (no retrain)
    mne_head   train: no-detach MNE on IF body + L2 on unmatched heads

Do not retune η. Do not submit mne_unmatched (duplicate of mne_head here).
Do not overwrite voc_ssd300_five_regs_seed42.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_voc_ssd300_five_regs_seed42 import (  # noqa: E402
    ARCH,
    HIGH_NOISE_MIN,
    L2_WD,
    LVAL,
    MNE_RC,
    NODETACH_KW,
    SEED,
    SIGMAS,
    TEST_T,
    TRAIN_T,
    auc_range,
    evaluate_map,
    load_trained,
    make_model,
    map_at,
    sigma_lookup,
    train_one,
    voc_is_ready,
    voc_loaders,
    write_csv,
)
from utils import (  # noqa: E402
    collect_weight_layer_matches,
    compute_mne_l2_unmatched_regularization,
    dump_mne_mapping_report,
    get_torch_device,
    unmatched_weight_rows,
)
from voc_ssd import voc_root_from  # noqa: E402

SCRATCH = Path("/scratch/gs14/sl9144/snn_results")
REUSE_DEFAULT = SCRATCH / "voc_ssd300_five_regs_seed42"
REUSE_LOCAL = ROOT.parent / "important_results" / "voc_ssd300_five_regs_seed42"
REUSE_FILES = (
    "scorecard.json",
    "test_sweep.csv",
    "per_class_ap.json",
    "epoch_log.csv",
)
METHODS = ("l2wo", "nodetach", "mne_head")
REUSE_METHODS = ("l2wo", "nodetach")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("VOC_EPOCHS", "80")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("VOC_BATCH", "16")))
    parser.add_argument("--eval-batch-size", type=int, default=int(os.environ.get("VOC_EVAL_BATCH", "8")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("VOC_NUM_WORKERS", "8")))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--import-reuse", action="store_true")
    parser.add_argument("--max-eval-images", type=int, default=0)
    parser.add_argument(
        "--voc-root",
        type=Path,
        default=Path(os.environ.get("VOC_ROOT", os.environ.get("CIFAR_ROOT", "~/datasets"))),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "voc_ssd300_nodetach_unmatched_head_l2_seed42",
    )
    parser.add_argument(
        "--reuse-root",
        type=Path,
        default=Path(os.environ.get("REUSE_ROOT", str(REUSE_DEFAULT))),
    )
    args = parser.parse_args()
    args.voc_root = Path(os.path.expanduser(str(args.voc_root)))
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    if not args.reuse_root.is_absolute():
        args.reuse_root = (ROOT / args.reuse_root).resolve()
    if args.summarize or args.self_check or args.import_reuse:
        return args
    if args.method is None:
        parser.error("--method is required unless --summarize/--self-check/--import-reuse")
    return args


def method_spec(method: str) -> dict:
    if method == "l2wo":
        return {
            "label": "L2-wo",
            "regularizer": "weight_decay_weights_only",
            "weight_decay": L2_WD,
            "reg_coeff": None,
            "unmatched_scope": None,
            "unmatched_coeff": L2_WD,
            "mne_kw": None,
            "train": False,
        }
    if method == "nodetach":
        return {
            "label": "MNE-L2 no-detach",
            "regularizer": "mne_l2",
            "weight_decay": 0.0,
            "reg_coeff": MNE_RC,
            "unmatched_scope": None,
            "unmatched_coeff": 0.0,
            "mne_kw": dict(NODETACH_KW),
            "train": False,
        }
    return {
        "label": "MNE-L2 no-detach + unmatched-head L2",
        "regularizer": "mne_l2_unmatched",
        "weight_decay": 0.0,
        "reg_coeff": MNE_RC,
        "unmatched_scope": "head",
        "unmatched_coeff": L2_WD,
        "mne_kw": dict(NODETACH_KW),
        "train": True,
    }


def cfg_dir(args) -> Path:
    return args.out_root / args.method


def ckpt_path(args) -> Path:
    return cfg_dir(args) / "checkpoints" / (
        f"{ARCH}_L[{LVAL}]_{args.method}_seed{args.seed}_L{LVAL}_trainT{TRAIN_T}.pth"
    )


def resolve_reuse_root(args) -> Path | None:
    for root in (args.reuse_root, REUSE_LOCAL, REUSE_DEFAULT):
        if (root / "l2wo" / "scorecard.json").is_file() or (root / "nodetach" / "scorecard.json").is_file():
            return root
    return args.reuse_root if args.reuse_root.exists() else None


def unmatched_scope_card(model) -> dict:
    rows = collect_weight_layer_matches(model, layer_map="legacy")
    unmatched = [row for row in rows if not row["matched"]]
    head = [row for row in unmatched if row["is_head"]]
    all_names = [row["name"] for row in unmatched]
    head_names = [row["name"] for row in head]
    return {
        "layer_map": "legacy",
        "matched": [row["name"] for row in rows if row["matched"]],
        "unmatched_all": all_names,
        "unmatched_head": head_names,
        "unmatched_body": [row["name"] for row in unmatched if not row["is_head"]],
        "scopes_identical": all_names == head_names,
        "n_matched": sum(1 for row in rows if row["matched"]),
        "n_unmatched": len(unmatched),
        "n_unmatched_head": len(head),
        "n_unmatched_body": len(unmatched) - len(head),
    }


def unmatched_norms(model) -> dict:
    rows = unmatched_weight_rows(model, scope="head", layer_map="legacy")
    fro = []
    for row in rows:
        weight = row["weight"].detach().float()
        fro.append(float(weight.pow(2).sum().sqrt()))
    return {
        "unmatched_head_names": [row["name"] for row in rows],
        "unmatched_head_frobenius": fro,
        "unmatched_head_frobenius_sum": float(sum(fro)),
        "unmatched_head_n_params": int(sum(int(row["n_params"]) for row in rows)),
    }


def import_reuse(args) -> list[Path]:
    src_root = resolve_reuse_root(args)
    copied = []
    if src_root is None:
        print("[WARN] no five-regs reuse root with scorecards", flush=True)
        return copied
    for method in REUSE_METHODS:
        src = src_root / method
        dst = args.out_root / method
        src_card = src / "scorecard.json"
        dst_card = dst / "scorecard.json"
        if dst_card.is_file():
            print(f"[REUSE SKIP] {dst_card}", flush=True)
            continue
        if not src_card.is_file():
            print(f"[REUSE MISSING] {src_card}", flush=True)
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for name in REUSE_FILES:
            src_path = src / name
            if src_path.is_file():
                shutil.copy2(src_path, dst / name)
        mapping = src / "mapping_final" / "mapping_summary.json"
        if mapping.is_file():
            (dst / "mapping_final").mkdir(parents=True, exist_ok=True)
            shutil.copy2(mapping, dst / "mapping_final" / "mapping_summary.json")
        card = json.loads(dst_card.read_text())
        card["historical_checkpoint_reused"] = True
        card["reuse_source"] = str(src_card)
        dst_card.write_text(json.dumps(card, indent=2, default=str) + "\n")
        copied.append(dst_card)
        print(f"[REUSE COPY] {src_card} -> {dst_card}", flush=True)
    return copied


def self_check(device) -> dict:
    model, _ = make_model(SEED, device, load_imagenet=False)
    scope = unmatched_scope_card(model)
    head_rows = unmatched_weight_rows(model, "head")
    all_rows = unmatched_weight_rows(model, "all")
    expected = [f"loc_heads.{i}" for i in range(6)] + [f"conf_heads.{i}" for i in range(6)]
    extra = compute_mne_l2_unmatched_regularization(
        model,
        quant_level=LVAL,
        unmatched_scope="head",
        unmatched_coeff=L2_WD,
        mne_coeff=MNE_RC,
        **NODETACH_KW,
    )
    stats = model._mne_unmatched_stats
    card = {
        **scope,
        "expected_heads": expected,
        "unmatched_reg_layers": stats["layers"],
        "unmatched_n_layers": stats["n_layers"],
        "unmatched_l2": stats["unmatched_l2"],
        "hybrid_reg": float(extra.detach().cpu()),
        "eta_mne": MNE_RC,
        "eta_head": L2_WD,
    }
    assert scope["n_matched"] == 23, card
    assert scope["n_unmatched"] == 12, card
    assert scope["n_unmatched_head"] == 12, card
    assert scope["n_unmatched_body"] == 0, card
    assert scope["scopes_identical"], card
    assert [row["name"] for row in head_rows] == expected, card
    assert [row["name"] for row in all_rows] == expected, card
    assert stats["layers"] == expected, card
    assert stats["n_layers"] == 12, card
    assert float(extra.detach()) > 0, card
    print(json.dumps(card, indent=2), flush=True)
    return card


def evaluate_trained(args, spec: dict, ckpt: Path, device, reused: bool) -> dict:
    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    model = load_trained(ckpt, device)
    scope = unmatched_scope_card(model)
    (out / "unmatched_scope.json").write_text(json.dumps(scope, indent=2) + "\n")
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
    norms = unmatched_norms(model)
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
        "unmatched_scope": spec["unmatched_scope"],
        "eta_mne": MNE_RC if spec["regularizer"] != "weight_decay_weights_only" else None,
        "eta_head": spec["unmatched_coeff"],
        "layer_map": "legacy",
        "detach_lambda": False,
        "detach_bn_affine": False,
        "noise_position": "post_input_if",
        "if_mode": "rate_uniform",
        "checkpoint": str(ckpt),
        "historical_checkpoint_reused": reused,
        "selection_uses_test": False,
        "scopes_identical": scope["scopes_identical"],
        "n_unmatched_head": scope["n_unmatched_head"],
        "test_clean": map_at(rows, 0.0),
        "test_sigma0p5": sigma_lookup(rows, 0.5),
        "test_sigma1": sigma_lookup(rows, 1.0),
        "test_sigma2": sigma_lookup(rows, 2.0),
        "test_sigma3": map_at(rows, 3.0),
        "test_sigma5": map_at(rows, 5.0),
        "test_auc_full": auc_range(rows, 0.0, 5.0),
        "test_auc_high": auc_range(rows, HIGH_NOISE_MIN, 5.0),
        "test_clean_fire": float(rows[0]["if_firing_density"]),
        **scope,
        **norms,
    }
    (out / "scorecard.json").write_text(json.dumps(card, indent=2, default=str) + "\n")
    print(json.dumps(card, indent=2, default=str), flush=True)
    return card


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
        f"{'method':<12} {'clean':>7} {'s3':>7} {'s5':>7} "
        f"{'AUC':>8} {'AUChi':>8} {'||W||F':>8} ident"
    )
    for card in cards:
        print(
            f"{card['method']:<12} {float(card['test_clean']):7.2f} "
            f"{float(card['test_sigma3']):7.2f} {float(card['test_sigma5']):7.2f} "
            f"{float(card['test_auc_full']):8.2f} {float(card['test_auc_high']):8.2f} "
            f"{float(card.get('unmatched_head_frobenius_sum', float('nan'))):8.2f} "
            f"{card.get('scopes_identical', '')}"
        )


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    device = get_torch_device(args.device)
    if args.self_check or args.dry_run:
        card = self_check(device)
        (args.out_root / "self_check.json").write_text(json.dumps(card, indent=2) + "\n")
        if args.self_check or args.method is None:
            return
    if args.summarize:
        summarize(args.out_root)
        return
    if args.import_reuse or args.method in REUSE_METHODS:
        import_reuse(args)
        if args.import_reuse and args.method is None:
            return
        if args.method in REUSE_METHODS:
            print("[INFO] l2wo/nodetach are reuse-only; not training.", flush=True)
            return

    spec = method_spec(args.method)
    if not spec["train"]:
        raise ValueError(f"{args.method} is reuse-only")
    if not voc_is_ready(args.voc_root):
        raise FileNotFoundError(
            f"VOC 2007+2012 missing under VOC_ROOT={args.voc_root}. "
            f"On a login node: python noise3_exp/voc_ssd.py --download --voc-root {args.voc_root}"
        )
    import_reuse(args)
    out = cfg_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    model, _ = make_model(args.seed, device, load_imagenet=False)
    scope = unmatched_scope_card(model)
    (out / "unmatched_scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    print(
        json.dumps(
            {
                "method": args.method,
                "label": spec["label"],
                "seed": args.seed,
                "eta_mne": MNE_RC,
                "eta_head": L2_WD,
                "unmatched_scope": spec["unmatched_scope"],
                "scopes_identical": scope["scopes_identical"],
                "unmatched_head": scope["unmatched_head"],
                "voc_root": str(voc_root_from(args.voc_root)),
                "out": str(out),
            },
            indent=2,
        ),
        flush=True,
    )
    if not scope["scopes_identical"]:
        raise AssertionError(f"SSD unmatched-all != unmatched-head: {scope}")
    if args.dry_run:
        print("[DRY RUN] skip train/eval", flush=True)
        return

    # train_one reads args.method / cfg via five-regs helpers; alias paths.
    args_five = argparse.Namespace(**vars(args))
    args_five.out_root = args.out_root
    ckpt = train_one(args_five, spec, device)
    evaluate_trained(args, spec, ckpt, device, reused=False)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
