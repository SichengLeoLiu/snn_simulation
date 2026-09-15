#!/usr/bin/env python3
"""Unify CIFAR baseline eval onto EVAL_SEED=0, plus ANN noise-injection.

Re-evaluate existing L2-all / L2-wo / L1-wo checkpoints with the same
fixed evaluation stream used by L2-wo (newer) and TA-MNE-U:
  T=L=16, rate_uniform, post_input_if, Gaussian, EVAL_SEED=0
  main-table grid σ ∈ {0,1,2,3,5}

Those three methods are eval-only. Do not retrain them. Do not write
into five-regs / four-regs / noise-family training trees.

ANN noise-injection is the one new training arm:
  L2-wo recipe (WD=5e-4, 300 ep, lr=0.1, T=0)
  Gaussian σ_train=1.0 at post_input_if during the train forward
  checkpoint still selected on clean ANN acc (same as L2-all/L2-wo)
  then SNN-eval with EVAL_SEED=0
  default seed 42; do not retune σ_train or WD

Writes /scratch/.../cifar_unified_baseline_evalseed0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

LVAL = 16
TEST_T = 16
EVAL_SEED = 0
EPOCHS = 300
LR = 0.1
L2_WD = 5e-4
TRAIN_SIGMA = 1.0
VAL_SIZE = 5000
VAL_SPLIT_SEED = 0
DEFAULT_SIGMAS = (0.0, 1.0, 2.0, 3.0, 5.0)
DEFAULT_SEEDS = (40, 41, 42, 43, 44)
SCRATCH = Path(os.environ.get("SNN_RESULTS", "/scratch/gs14/sl9144/snn_results"))
HOME_QCFS = Path(
    os.environ.get(
        "HOME_QCFS",
        "/home/595/sl9144/codes/snn_simulation/QCFS_simulation",
    )
)
LOCAL_IR = ROOT.parent / "important_results"
EVAL_METHODS = ("l2all", "l2wo", "l1wo")
TRAIN_METHODS = ("noiseinj",)
METHODS = EVAL_METHODS + TRAIN_METHODS
LABELS = {
    "l2all": "L2-all",
    "l2wo": "L2-wo",
    "l1wo": "L1-wo",
    "noiseinj": "ANN noise-injection (L2-wo, σ_train=1)",
}
ARCHS = ("vgg16", "resnet18")
DATASETS = ("cifar10", "cifar100")
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = [n / 255.0 for n in [129.3, 124.1, 112.4]]
CIFAR100_STD = [n / 255.0 for n in [68.2, 65.4, 70.4]]
BLOCKED_TREES = (
    "cifar_vgg16_five_regs",
    "cifar_resnet18_four_regs_5seed",
    "cifar_resnet18_l1wo_5seed",
    "cifar_fgmneu_noise_family",
    "cifar_independent_ckpt_seed42",
)


def get_torch_device(device_str: str = "auto") -> torch.device:
    s = (device_str or "auto").strip().lower()
    if s in ("auto", "", "0"):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    return torch.device(s)


def seed_all(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=ARCHS, default=None)
    parser.add_argument("--dataset", choices=DATASETS, default=None)
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sigmas", nargs="+", type=float, default=list(DEFAULT_SIGMAS))
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    parser.add_argument("--train-sigma", type=float, default=TRAIN_SIGMA)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("CIFAR_BATCH", "128")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CIFAR_NUM_WORKERS", "8")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-resolve", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "cifar_unified_baseline_evalseed0",
    )
    args = parser.parse_args()
    if not args.out_root.is_absolute():
        args.out_root = (ROOT / args.out_root).resolve()
    args.sigmas = sorted({float(s) for s in args.sigmas})
    if args.self_check or args.summarize:
        return args
    if args.arch is None or args.dataset is None or args.method is None:
        parser.error("--arch --dataset --method are required unless --self-check/--summarize")
    if args.method == "noiseinj" and args.seed is None:
        args.seeds = [42]
    elif args.seed is not None:
        args.seeds = [int(args.seed)]
    else:
        args.seeds = [int(s) for s in args.seeds]
    return args


def ckpt_roots() -> list[Path]:
    env_root = Path(os.environ["CIFAR_CKPT_ROOT"]) if os.environ.get("CIFAR_CKPT_ROOT") else None
    return [
        path
        for path in (
            env_root,
            Path("/scratch/gs14/sl9144/snn_ckpts"),
            Path("/scratch/gs14/sl9144/qcfs_checkpoints"),
            HOME_QCFS,
            ROOT,
            LOCAL_IR,
            SCRATCH,
        )
        if path is not None
    ]


def mneablate_names(dataset: str, variant: str, seed: int) -> list[str]:
    rc = {
        "weight_decay": "rcnone",
        "weight_decay_weights_only": "rcnone",
        "l1": "rc1em05",
    }[variant]
    stem = f"vgg16_L[16]_mneablate_{dataset}_{variant}_{rc}_seed{seed}_L16"
    return [f"{stem}_trainT0.pth", f"{stem}.pth"]


def ckpt_candidates(arch: str, dataset: str, method: str, seed: int) -> list[Path]:
    paths: list[Path] = []
    roots = ckpt_roots()
    if arch == "vgg16":
        variant = {
            "l2all": "weight_decay",
            "l2wo": "weight_decay_weights_only",
            "l1wo": "l1",
        }[method]
        for root in roots:
            for name in mneablate_names(dataset, variant, seed):
                paths.append(root / f"{dataset}-checkpoints" / name)
                paths.append(root / dataset / name)
        if method == "l2wo" and seed == 42:
            paths.append(
                SCRATCH
                / "cifar_vgg16_mne_component_ablation_seed42"
                / dataset
                / "comp_l2wo_fixed"
                / "checkpoints"
                / "vgg16_L[16]_comp_l2wo_fixed_seed42_L16_trainT0.pth"
            )
            paths.append(
                LOCAL_IR
                / "cifar_vgg16_mne_component_ablation_seed42"
                / dataset
                / "comp_l2wo_fixed"
                / "checkpoints"
                / "vgg16_L[16]_comp_l2wo_fixed_seed42_L16_trainT0.pth"
            )
        return _unique(paths)
    if method == "l1wo":
        rel = Path("cifar_resnet18_l1wo_5seed") / dataset / "r18_l1wo" / f"seed{seed}" / "checkpoints" / f"resnet18_L[16]_r18_l1wo_seed{seed}_L16_trainT0.pth"
        paths.append(SCRATCH / rel)
        paths.append(LOCAL_IR / rel)
    else:
        key = {"l2all": "l2all", "l2wo": "l2wo"}[method]
        rel = (
            Path("cifar_resnet18_four_regs_5seed")
            / dataset
            / f"r18_{key}"
            / f"seed{seed}"
            / "checkpoints"
            / f"resnet18_L[16]_r18_{key}_seed{seed}_L16_trainT0.pth"
        )
        paths.append(SCRATCH / rel)
        paths.append(LOCAL_IR / rel)
    return _unique(paths)


def _unique(paths: list[Path]) -> list[Path]:
    seen = set()
    out = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def resolve_ckpt(arch: str, dataset: str, method: str, seed: int) -> Path:
    tried = ckpt_candidates(arch, dataset, method, seed)
    for path in tried:
        if path.is_file():
            return path
    lines = "\n".join(f"  {p}" for p in tried) or "  (no candidates)"
    raise FileNotFoundError(
        f"checkpoint not found for {arch} {dataset} {method} seed{seed}:\n{lines}"
    )


def cfg_dir(args, seed: int) -> Path:
    return args.out_root / args.dataset / f"{args.arch}_{args.method}" / f"seed{seed}"


def noiseinj_suffix(args, seed: int) -> str:
    tag = str(args.train_sigma).replace(".", "p")
    return f"{args.arch}_noiseinj_s{tag}_seed{seed}_L{LVAL}_trainT0"


def noiseinj_ckpt(args, seed: int) -> Path:
    return cfg_dir(args, seed) / "checkpoints" / f"{args.arch}_L[{LVAL}]_{noiseinj_suffix(args, seed)}.pth"


def train_cmd(args, seed: int) -> list[str]:
    out = cfg_dir(args, seed)
    return [
        sys.executable,
        str(ROOT / "main_train.py"),
        "-data", args.dataset,
        "-arch", args.arch,
        "-L", str(LVAL),
        "-T", "0",
        "--epochs", str(args.epochs),
        "-lr", str(LR),
        "-b", str(args.batch_size),
        "-j", str(args.workers),
        "--seed", str(seed),
        "--device", args.device,
        "--spike_schedule", "normal",
        "--ckpt-save-mode", "best",
        "--ckpt-dir", str(noiseinj_ckpt(args, seed).parent),
        "-suffix", noiseinj_suffix(args, seed),
        "--regularizer", "weight_decay_weights_only",
        "--weight_decay", str(L2_WD),
        "--train-noise-sigma", str(args.train_sigma),
        "--train-noise-type", "gaussian",
        "--train-noise-position", "post_input_if",
        "--epoch_log_csv", str(out / "epoch_log.csv"),
    ]


def train_noiseinj(args, seed: int) -> Path:
    checkpoint = noiseinj_ckpt(args, seed)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {checkpoint}", flush=True)
        return checkpoint
    if args.test_only:
        raise FileNotFoundError(checkpoint)
    cmd = train_cmd(args, seed)
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return checkpoint
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def eval_dataset(dataset: str, *, train: bool):
    root = os.path.expanduser(os.environ.get("CIFAR_ROOT", "~/datasets"))
    if dataset == "cifar10":
        mean, std, cls = CIFAR10_MEAN, CIFAR10_STD, datasets.CIFAR10
    else:
        mean, std, cls = CIFAR100_MEAN, CIFAR100_STD, datasets.CIFAR100
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    return cls(root, train=train, transform=transform, download=False)


def val_loader(args, pin_memory: bool) -> DataLoader:
    train_ds = eval_dataset(args.dataset, train=True)
    g = torch.Generator().manual_seed(VAL_SPLIT_SEED)
    perm = torch.randperm(len(train_ds), generator=g).tolist()
    return DataLoader(
        Subset(train_ds, perm[:VAL_SIZE]),
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


def load_model(ckpt: Path, device, arch: str, dataset: str):
    from Models import modelpool
    from Models.VGG import remap_legacy_vgg_state_dict

    model = modelpool(arch, dataset)
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if arch.startswith("vgg"):
        state = remap_legacy_vgg_state_dict(state)
    model.load_state_dict(state, strict=True)
    model.set_L(LVAL)
    model.set_T(TEST_T)
    model.set_mode("rate_uniform")
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule("normal")
    model.set_first_layer_input_noise_position("post_input_if")
    model.set_first_layer_input_noise_type("gaussian")
    model.set_first_layer_input_noise_sigma(0.0)
    return model.to(device).eval()


@torch.inference_mode()
def accuracy(model, loader, device) -> tuple[float, int]:
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        if logits.dim() == 3:
            logits = logits.mean(0)
        correct += int(logits.argmax(dim=1).eq(labels).sum().item())
        total += int(labels.numel())
    return 100.0 * correct / max(total, 1), total


def trapz(xs: list[float], ys: list[float]) -> float:
    area = 0.0
    for i in range(1, len(xs)):
        area += 0.5 * (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1])
    return area


def acc_at(rows: list[dict], sigma: float) -> float:
    for row in rows:
        if abs(float(row["sigma"]) - sigma) < 1e-9:
            return float(row["accuracy"])
    raise KeyError(sigma)


def auc_range(rows: list[dict], lo: float, hi: float) -> float:
    xs, ys = [], []
    for row in rows:
        sigma = float(row["sigma"])
        if lo - 1e-12 <= sigma <= hi + 1e-12:
            xs.append(sigma)
            ys.append(float(row["accuracy"]))
    if len(xs) < 2:
        return float("nan")
    return trapz(xs, ys)


def snn_metrics(rows: list[dict], prefix: str) -> dict:
    return {
        f"{prefix}_clean": acc_at(rows, 0.0),
        f"{prefix}_sigma1": acc_at(rows, 1.0) if any(abs(float(r["sigma"]) - 1.0) < 1e-9 for r in rows) else float("nan"),
        f"{prefix}_sigma5": acc_at(rows, 5.0),
        f"{prefix}_auc_full": auc_range(rows, 0.0, 5.0),
        f"{prefix}_auc_high": auc_range(rows, 3.0, 5.0),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sweep(model, loader, device, split: str, eval_seed: int, sigmas: list[float]) -> list[dict]:
    rows = []
    for sigma in sigmas:
        seed_all(eval_seed)
        model.set_first_layer_input_noise_type("gaussian")
        model.set_first_layer_input_noise_position("post_input_if")
        model.set_first_layer_input_noise_sigma(float(sigma))
        acc, n_samples = accuracy(model, loader, device)
        rows.append(
            {
                "split": split,
                "sigma": f"{sigma:g}",
                "accuracy": f"{acc:.6f}",
                "n_samples": n_samples,
                "eval_seed": eval_seed,
            }
        )
        print(f"{split:<5} sigma={sigma:g} acc={acc:.2f}", flush=True)
    model.set_first_layer_input_noise_sigma(0.0)
    return rows


def _ns(**kwargs):
    base = dict(
        dataset="cifar10",
        arch="vgg16",
        method="l2all",
        seeds=list(DEFAULT_SEEDS),
        seed=None,
        sigmas=list(DEFAULT_SIGMAS),
        eval_seed=EVAL_SEED,
        train_sigma=TRAIN_SIGMA,
        epochs=EPOCHS,
        batch_size=128,
        workers=8,
        device="cpu",
        retrain=False,
        test_only=False,
        force=False,
        dry_run=True,
        out_root=ROOT.parent / "important_results" / "cifar_unified_baseline_evalseed0",
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def self_check() -> None:
    ns = _ns(method="noiseinj", arch="resnet18", dataset="cifar100", seed=42)
    cmd = train_cmd(ns, 42)
    joined = " ".join(cmd)
    if cmd[cmd.index("--regularizer") + 1] != "weight_decay_weights_only":
        raise AssertionError("noiseinj must sit on L2-wo")
    if cmd[cmd.index("--weight_decay") + 1] != str(L2_WD):
        raise AssertionError("noiseinj WD must stay 5e-4")
    if cmd[cmd.index("--train-noise-sigma") + 1] != str(TRAIN_SIGMA):
        raise AssertionError("σ_train must stay locked at 1.0")
    if "--ckpt-select-split" in cmd:
        raise AssertionError("noiseinj must match historical clean-test selection")
    if "cifar_independent_ckpt" in joined:
        raise AssertionError("must not write the independent-ckpt tree")
    ckpt = str(noiseinj_ckpt(ns, 42))
    if "cifar_unified_baseline_evalseed0" not in ckpt:
        raise AssertionError(f"new tree required: {ckpt}")
    if any(name in ckpt for name in BLOCKED_TREES):
        raise AssertionError(f"must not overwrite {ckpt}")
    vgg_l2 = " ".join(str(p) for p in ckpt_candidates("vgg16", "cifar10", "l2all", 42))
    if "weight_decay_rcnone" not in vgg_l2:
        raise AssertionError("VGG L2-all must resolve five-regs weight_decay ckpts")
    vgg_l1 = " ".join(str(p) for p in ckpt_candidates("vgg16", "cifar100", "l1wo", 40))
    if "l1_rc1em05" not in vgg_l1:
        raise AssertionError("VGG L1-wo must resolve five-regs l1 ckpts")
    r18_l1 = str(ckpt_candidates("resnet18", "cifar10", "l1wo", 42)[0])
    if "cifar_resnet18_l1wo_5seed" not in r18_l1:
        raise AssertionError("ResNet L1-wo lives in cifar_resnet18_l1wo_5seed")
    r18_l2 = str(ckpt_candidates("resnet18", "cifar10", "l2all", 42)[0])
    if "r18_l2all" not in r18_l2:
        raise AssertionError("ResNet L2-all lives in four_regs")
    print("[self-check] unified baseline eval-seed-0 flags ok", flush=True)


def summarize(out_root: Path) -> None:
    cards = [
        json.loads(path.read_text())
        for path in sorted(out_root.glob("*/*/seed*/scorecard.json"))
    ]
    if not cards:
        print(f"No scorecards in {out_root}")
        return
    print(
        f"{'dataset':<10} {'arch':<9} {'method':<9} {'seed':>5} "
        f"{'clean':>7} {'s1':>7} {'s5':>7}"
    )
    for card in cards:
        print(
            f"{card['dataset']:<10} {card.get('arch', '?'):<9} "
            f"{card.get('method', '?'):<9} {card.get('seed', '?'):>5} "
            f"{card['test_clean']:7.2f} {card.get('test_sigma1', float('nan')):7.2f} "
            f"{card['test_sigma5']:7.2f}"
        )


def eval_one(args, seed: int) -> None:
    out = cfg_dir(args, seed)
    out.mkdir(parents=True, exist_ok=True)
    card_path = out / "scorecard.json"
    if card_path.is_file() and not args.force and not args.retrain:
        print(f"[SKIP EVAL] {card_path}", flush=True)
        return
    if args.method == "noiseinj":
        checkpoint = train_noiseinj(args, seed)
        reused = False
    else:
        checkpoint = resolve_ckpt(args.arch, args.dataset, args.method, seed)
        reused = True
        print(f"[REUSE] {checkpoint}", flush=True)
    if args.dry_run or args.dry_resolve:
        print("[DRY]", checkpoint, flush=True)
        return
    device = get_torch_device(args.device)
    pin = device.type == "cuda"
    model = load_model(checkpoint, device, args.arch, args.dataset)
    val_rows = sweep(model, val_loader(args, pin), device, "val", args.eval_seed, args.sigmas)
    write_csv(out / "val_sweep.csv", val_rows)
    test_rows = sweep(model, test_loader(args, pin), device, "test", args.eval_seed, args.sigmas)
    write_csv(out / "test_sweep.csv", test_rows)
    card = {
        "config": f"{args.arch}_{args.method}",
        "label": LABELS[args.method],
        "method": args.method,
        "arch": args.arch,
        "dataset": args.dataset,
        "seed": seed,
        "eval_seed": args.eval_seed,
        "reused_checkpoint": reused,
        "checkpoint": str(checkpoint),
        "selection_uses_test": True,
        "train_noise_sigma": args.train_sigma if args.method == "noiseinj" else 0.0,
        "protocol": {
            "T": TEST_T,
            "L": LVAL,
            "mode": "rate_uniform",
            "noise": "post_input_if gaussian",
            "eval_seed_locked": True,
            "eta_locked": True,
        },
        **snn_metrics(val_rows, "val"),
        **snn_metrics(test_rows, "test"),
    }
    card_path.write_text(json.dumps(card, indent=2, default=str) + "\n")
    print(json.dumps(card, indent=2, default=str), flush=True)
    print(f"Wrote {out}", flush=True)


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return
    if args.summarize:
        summarize(args.out_root)
        return
    args.out_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] unified baseline {args.arch} {args.method} {args.dataset} "
        f"seeds={args.seeds} eval_seed={args.eval_seed} sigmas={args.sigmas}",
        flush=True,
    )
    for seed in args.seeds:
        eval_one(args, seed)


if __name__ == "__main__":
    main()
