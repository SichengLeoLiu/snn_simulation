#!/usr/bin/env python3
"""CIFAR VGG16 seed-42 screen: approach the noise-sweep upper envelope.

Goal: a *single* checkpoint whose validation curve stays close to
Calibrated MNE at low noise and Old MNE at high noise. Do not claim it
is best at every sigma.

Families
--------
hybrid : optimizer L2-all (β0) + Old MNE (detach λ, β1)
    R = WD(β0) + β1 * Σ_l L^2 ||W̃_l||_F^2 / (λ_l^2+ε)

onesided : weights-only L2 with one-sided risk reweighting
    q = 1 + α max(r̂-τ, 0),  R = (β/2) Σ q W^2

Selection uses a 5k train-holdout val sweep (not the test set).
Test sweeps are recorded but must not pick the model.
Clean Horowitz energy is the deployment number; noisy firing is diagnostic.

Reference bars (VGG16 5-seed post-IF, T=16):
  CIFAR-100 Calibrated clean = 63.91%  (need val clean ≥ 63.41)
  CIFAR-100 Old MNE σ=5      = 41.14%  (need val σ=5 ≥ 39.14)
  CIFAR-10  Calibrated clean = 91.13%  (need val clean ≥ 90.63)
  CIFAR-10  Old MNE σ=5      = 72.06%  (need val σ=5 ≥ 70.06)
  Rank eligible configs by val AUC on σ∈[0,5].
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from Models import modelpool  # noqa: E402
from Models.VGG import remap_legacy_vgg_state_dict  # noqa: E402
from utils import get_torch_device, seed_all  # noqa: E402
import measure_vgg16_horowitz_energy as energy  # noqa: E402


DEFAULT_DATASET = "cifar100"
ARCH = "vgg16"
SEED = 42
LVAL = 16
TEST_T = 16
EPOCHS = 300
LR = 0.1
VAL_SIZE = 5000
VAL_SPLIT_SEED = 0
SIGMAS = [i / 4 for i in range(0, 21)]  # 0, 0.25, ..., 5

CLEAN_TOL = 0.5
SIGMA5_TOL = 2.0
REFS = {
    "cifar100": {"calibrated_clean": 63.91, "old_mne_sigma5": 41.14},
    "cifar10": {"calibrated_clean": 91.128, "old_mne_sigma5": 72.056},
}

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = [n / 255.0 for n in [129.3, 124.1, 112.4]]
CIFAR100_STD = [n / 255.0 for n in [68.2, 65.4, 70.4]]


def _fmt(v: float) -> str:
    return f"{float(v):.6g}".replace("+", "").replace("-", "m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["hybrid", "onesided"], default=None)
    parser.add_argument("--beta0", type=float, default=None, help="hybrid optimizer WD")
    parser.add_argument("--beta1", type=float, default=None, help="hybrid Old MNE coeff")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=5e-4, help="onesided L2 coeff")
    parser.add_argument(
        "--risk-max",
        type=float,
        default=2.0,
        help="onesided/calibrated mean-one risk clip upper bound",
    )
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent
        / "important_results"
        / "cifar100_vgg16_hybrid_envelope_screen_seed42",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    if not args.summarize and args.family is None:
        parser.error("specify --family or --summarize")
    if args.family == "hybrid" and (args.beta0 is None or args.beta1 is None):
        parser.error("hybrid requires --beta0 and --beta1")
    return args


def config_name(args) -> str:
    family = getattr(args, "family", None)
    if family == "hybrid":
        return f"hybrid_b0_{_fmt(args.beta0)}_b1_{_fmt(args.beta1)}"
    if family == "onesided":
        name = (
            f"onesided_a{_fmt(args.alpha)}_tau{_fmt(args.tau)}_b{_fmt(args.beta)}"
        )
        risk_max = float(getattr(args, "risk_max", 2.0))
        if abs(risk_max - 2.0) > 1e-12:
            name += f"_rmax{_fmt(risk_max)}"
        return name
    return str(family or "unknown")


def suffix(args) -> str:
    return f"hybenv_{config_name(args)}_seed{args.seed}_L{LVAL}_trainT0"


def dataset_refs(dataset: str) -> dict:
    return REFS[dataset]


def ckpt_path(args) -> Path:
    return ROOT / f"{args.dataset}-checkpoints" / f"{ARCH}_L[{LVAL}]_{suffix(args)}.pth"


def train(args) -> Path:
    ckpt = ckpt_path(args)
    if ckpt.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {ckpt}", flush=True)
        return ckpt
    if args.test_only:
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        return ckpt
    cmd = [
        sys.executable,
        str(ROOT / "main_train.py"),
        "-data", args.dataset,
        "-arch", ARCH,
        "-L", str(LVAL),
        "-T", "0",
        "--epochs", str(args.epochs),
        "-lr", str(LR),
        "-b", str(args.batch_size),
        "-j", str(args.workers),
        "--seed", str(args.seed),
        "--device", args.device,
        "--spike_schedule", "normal",
        "--ckpt-save-mode", "best",
        "-suffix", suffix(args),
    ]
    if args.family == "hybrid":
        cmd += [
            "--regularizer", "mne_l2",
            "--mne_detach_lambda",
            "--weight_decay", str(args.beta0),
            "--reg_coeff", str(args.beta1),
        ]
    else:
        cmd += [
            "--regularizer", "calibrated_mne_l2",
            "--weight_decay", "0",
            "--reg_coeff", str(args.beta),
            "--calibrated_mne_alpha", str(args.alpha),
            "--calibrated_mne_onesided",
            "--calibrated_mne_tau", str(args.tau),
            "--calibrated_mne_risk_min", "0.5",
            "--calibrated_mne_risk_max", str(getattr(args, "risk_max", 2.0)),
            "--calibrated_mne_alpha_start_epoch", "30",
            "--calibrated_mne_alpha_warmup_epochs", "50",
        ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"training finished but missing {ckpt}")
    return ckpt


def eval_dataset(dataset: str, *, train: bool):
    root = os.path.expanduser(os.environ.get("CIFAR_ROOT", "~/datasets"))
    if dataset == "cifar10":
        mean, std, cls = CIFAR10_MEAN, CIFAR10_STD, datasets.CIFAR10
    else:
        mean, std, cls = CIFAR100_MEAN, CIFAR100_STD, datasets.CIFAR100
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return cls(root, train=train, transform=transform, download=False)


def val_loader(args, pin_memory: bool) -> DataLoader:
    train_ds = eval_dataset(args.dataset, train=True)
    g = torch.Generator().manual_seed(VAL_SPLIT_SEED)
    perm = torch.randperm(len(train_ds), generator=g).tolist()
    subset = Subset(train_ds, perm[:VAL_SIZE])
    return DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )


def test_loader(args, pin_memory: bool) -> DataLoader:
    return DataLoader(
        eval_dataset(args.dataset, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )


def load_model(ckpt: Path, device, dataset: str):
    model = modelpool(ARCH, dataset)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(remap_legacy_vgg_state_dict(state), strict=True)
    model.set_L(LVAL)
    model.set_T(TEST_T)
    model.set_mode("rate_uniform")
    model.set_spike_schedule("normal")
    if hasattr(model, "set_first_layer_input_noise_position"):
        model.set_first_layer_input_noise_position("post_input_if")
    if hasattr(model, "set_first_layer_input_noise_type"):
        model.set_first_layer_input_noise_type("gaussian")
    return model.to(device).eval()


def trapz(xs: list[float], ys: list[float]) -> float:
    area = 0.0
    for i in range(1, len(xs)):
        area += 0.5 * (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1])
    return area


def sweep(model, loader, device, split: str) -> list[dict]:
    rows = []
    for sigma in SIGMAS:
        seed_all(SEED)
        model.set_first_layer_input_noise_sigma(float(sigma))
        result = energy.evaluate(model, loader, device, TEST_T, 0)
        use_energy = "yes" if abs(sigma) < 1e-12 else "no"
        row = {
            "split": split,
            "sigma": f"{sigma:g}",
            "accuracy": f"{result['accuracy']:.6f}",
            "if_firing_density": f"{result['if_firing_density']:.6f}",
            "energy_mJ": f"{result['energy_mJ']:.8f}",
            "use_for_energy": use_energy,
            "n_samples": result["n_samples"],
        }
        rows.append(row)
        extra = "" if use_energy == "yes" else " [diag energy]"
        print(
            f"{split:<5} sigma={sigma:g} acc={result['accuracy']:.2f} "
            f"fire={result['if_firing_density']:.4f} "
            f"E={result['energy_mJ']:.4f} mJ{extra}",
            flush=True,
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def scorecard(val_rows: list[dict], test_rows: list[dict], args, ckpt: Path) -> dict:
    def acc_at(rows, sigma):
        for row in rows:
            if abs(float(row["sigma"]) - sigma) < 1e-9:
                return float(row["accuracy"])
        raise KeyError(sigma)

    val_xs = [float(r["sigma"]) for r in val_rows]
    val_ys = [float(r["accuracy"]) for r in val_rows]
    test_xs = [float(r["sigma"]) for r in test_rows]
    test_ys = [float(r["accuracy"]) for r in test_rows]
    val_clean = acc_at(val_rows, 0.0)
    val_s5 = acc_at(val_rows, 5.0)
    refs = dataset_refs(args.dataset)
    clean_ok = val_clean >= refs["calibrated_clean"] - CLEAN_TOL
    s5_ok = val_s5 >= refs["old_mne_sigma5"] - SIGMA5_TOL
    card = {
        "config": config_name(args),
        "dataset": args.dataset,
        "family": args.family,
        "beta0": args.beta0,
        "beta1": args.beta1,
        "alpha": args.alpha if args.family == "onesided" else None,
        "tau": args.tau if args.family == "onesided" else None,
        "beta": args.beta if args.family == "onesided" else None,
        "seed": args.seed,
        "checkpoint": str(ckpt),
        "val_clean": val_clean,
        "val_sigma5": val_s5,
        "val_auc": trapz(val_xs, val_ys),
        "test_clean": acc_at(test_rows, 0.0),
        "test_sigma5": acc_at(test_rows, 5.0),
        "test_auc": trapz(test_xs, test_ys),
        "clean_ok": clean_ok,
        "sigma5_ok": s5_ok,
        "eligible": bool(clean_ok and s5_ok),
        "ref_calibrated_clean": refs["calibrated_clean"],
        "ref_old_mne_sigma5": refs["old_mne_sigma5"],
        "clean_energy_mJ": float(val_rows[0]["energy_mJ"]),
        "clean_fire": float(val_rows[0]["if_firing_density"]),
    }
    return card


def summarize(out_root: Path) -> None:
    cards = []
    for path in sorted(out_root.glob("*/scorecard.json")):
        cards.append(json.loads(path.read_text()))
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    cards.sort(key=lambda c: (-int(c["eligible"]), -float(c["val_auc"])))
    refs = dataset_refs(str(cards[0].get("dataset", DEFAULT_DATASET)))
    print(
        "\nSelection is VAL-only. Test numbers are logged, not used to pick.\n"
        f"Need val clean ≥ {refs['calibrated_clean'] - CLEAN_TOL:.2f} "
        f"and val σ=5 ≥ {refs['old_mne_sigma5'] - SIGMA5_TOL:.2f}.\n"
    )
    print(
        f"{'config':<36} {'elig':>4} {'val0':>7} {'val5':>7} {'valAUC':>8} "
        f"{'test0':>7} {'test5':>7}"
    )
    for card in cards:
        print(
            f"{card['config']:<36} {str(card['eligible']):>4} "
            f"{card['val_clean']:7.2f} {card['val_sigma5']:7.2f} "
            f"{card['val_auc']:8.2f} {card['test_clean']:7.2f} "
            f"{card['test_sigma5']:7.2f}"
        )
    eligible = [c for c in cards if c["eligible"]]
    chosen = (eligible or cards)[:2]
    print("\nBest two for 5-seed confirmation (do not switch by test σ):")
    for card in chosen:
        print(f"  {card['config']}  eligible={card['eligible']}  valAUC={card['val_auc']:.2f}")
    write_csv(out_root / "screen_ranking.csv", cards)
    print(f"Wrote {out_root / 'screen_ranking.csv'}")


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(args.out_root)
        return

    name = config_name(args)
    cfg_dir = args.out_root / name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] {args.dataset} {name}: val 5k holdout for selection; "
        "test 10k recorded only; clean energy for deployment",
        flush=True,
    )
    ckpt = train(args)
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = load_model(ckpt, device, args.dataset)
    val_rows = sweep(model, val_loader(args, pin), device, "val")
    write_csv(cfg_dir / "val_sweep.csv", val_rows)
    test_rows = sweep(model, test_loader(args, pin), device, "test")
    write_csv(cfg_dir / "test_sweep.csv", test_rows)
    card = scorecard(val_rows, test_rows, args, ckpt)
    (cfg_dir / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2), flush=True)
    print(f"Wrote {cfg_dir}", flush=True)


if __name__ == "__main__":
    main()
