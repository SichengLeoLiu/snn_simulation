from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Models import modelpool  # noqa: E402
from Preprocess import datapool  # noqa: E402
from utils import get_torch_device, val  # noqa: E402


def _fmt(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def _run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _checkpoint(model: str, level: int, suffix: str) -> Path:
    return ROOT / "fashion_mnist-checkpoints" / f"{model}_L[{level}]_{suffix}.pth"


def _read_noise_matrix(path: Path, level: int) -> dict[float, float]:
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(float(row["L"])) != int(level):
                continue
            return {
                float(key): float(value)
                for key, value in row.items()
                if key not in ("L", "T") and value not in (None, "")
            }
    raise ValueError(f"No L={level} row in {path}")


def _ann_accuracy(checkpoint: Path, args, test_loader, device) -> float:
    model = modelpool(args.model, "fashion_mnist")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.set_L(args.L)
    model.set_T(0)
    model.set_mode("normal")
    model.to(device)
    return float(val(model, test_loader, T=0, device=device, verbose=False))


def _train(method: str, alpha: float | None, suffix: str, args) -> Path:
    checkpoint = _checkpoint(args.model, args.L, suffix)
    if checkpoint.exists() and not args.retrain:
        print(f"[SKIP TRAIN] {checkpoint}", flush=True)
        return checkpoint

    command = [
        sys.executable,
        str(ROOT / "main_train.py"),
        "-data",
        "fashion_mnist",
        "-arch",
        args.model,
        "-L",
        str(args.L),
        "-T",
        "0",
        "--epochs",
        str(args.epochs),
        "-lr",
        str(args.lr),
        "-b",
        str(args.batch_size),
        "-j",
        str(args.workers),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--regularizer",
        method,
        "-suffix",
        suffix,
        "--ckpt-save-mode",
        "best",
    ]
    if method == "weight_decay_weights_only":
        command += ["-wd", str(args.weight_decay)]
    else:
        command += [
            "-wd",
            "0",
            "--reg_coeff",
            str(args.weight_decay),
            "--calibrated_mne_alpha",
            str(alpha),
            "--calibrated_mne_risk_min",
            str(args.risk_min),
            "--calibrated_mne_risk_max",
            str(args.risk_max),
            "--calibrated_mne_alpha_start_epoch",
            str(args.alpha_start_epoch),
            "--calibrated_mne_alpha_warmup_epochs",
            str(args.alpha_warmup_epochs),
        ]
    _run(command)
    return checkpoint


def _snn_noise_accuracy(
    checkpoint: Path, method_label: str, suffix: str, args
) -> dict[float, float]:
    output_dir = args.out_root / method_label / "noise"
    matrix = output_dir / (
        f"noise_sweep_matrix_fashion_mnist_{args.model}_T{args.T}"
        f"_mode_rate_uniform_schedule_normal_seed_{args.seed}.csv"
    )
    if not matrix.exists() or args.retest:
        command = [
            sys.executable,
            str(ROOT / "main_test.py"),
            "-data",
            "fashion_mnist",
            "-arch",
            args.model,
            "-L",
            str(args.L),
            "-T",
            str(args.T),
            "--mode",
            "rate_uniform",
            "--spike_schedule",
            "normal",
            "--noise_sweep",
            "--noise_sigma_start",
            "0",
            "--noise_sigma_end",
            "1",
            "--noise_sigma_step",
            "1",
            "--noise_output_dir",
            str(output_dir),
            "--first_layer_noise_position",
            "post_input_if",
            "--first_layer_noise_type",
            "gaussian",
            "-w",
            str(checkpoint),
            "-suffix",
            suffix,
            "-b",
            str(args.batch_size),
            "-j",
            str(args.workers),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
        ]
        _run(command)
    return _read_noise_matrix(matrix, args.L)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fashion-MNIST L2-calibrated MNE single-seed pilot."
    )
    parser.add_argument("--model", default="cnn2_c8_c16")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--L", default=16, type=int)
    parser.add_argument("--T", default=16, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--alphas", nargs="+", default=[0.0, 0.05, 0.1, 0.25, 0.5], type=float)
    parser.add_argument("--risk-min", default=0.5, type=float)
    parser.add_argument("--risk-max", default=2.0, type=float)
    parser.add_argument("--alpha-start-epoch", default=5, type=int)
    parser.add_argument("--alpha-warmup-epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT.parent / "important_results" / "fashion_calibrated_mne_pilot",
    )
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--retest", action="store_true")
    args = parser.parse_args()
    args.out_root = args.out_root.resolve()
    return args


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    device = get_torch_device(args.device)
    _, test_loader = datapool(
        "fashion_mnist",
        args.batch_size,
        num_workers=args.workers,
        pin_memory=False,
    )

    jobs = [("l2_wo", "weight_decay_weights_only", None)] + [
        (f"cmne_a{_fmt(alpha)}", "calibrated_mne_l2", alpha)
        for alpha in args.alphas
    ]
    rows = []
    for label, method, alpha in jobs:
        suffix = (
            f"fashion_calibrated_mne_{args.model}_{label}_seed{args.seed}"
            f"_ep{args.epochs}_L{args.L}"
        )
        checkpoint = _train(method, alpha, suffix, args)
        ann_acc = _ann_accuracy(checkpoint, args, test_loader, device)
        snn = _snn_noise_accuracy(checkpoint, label, suffix, args)
        clean = snn[0.0]
        noisy = snn[1.0]
        row = {
            "method": label,
            "alpha": "" if alpha is None else f"{alpha:.6g}",
            "ann_acc": f"{ann_acc:.6f}",
            "snn_sigma0": f"{clean:.6f}",
            "snn_sigma1": f"{noisy:.6f}",
            "drop": f"{clean - noisy:.6f}",
            "checkpoint": str(checkpoint),
        }
        rows.append(row)
        print(f"[RESULT] {row}", flush=True)
        if device.type == "mps":
            torch.mps.empty_cache()

    output = args.out_root / "summary.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] {output}", flush=True)


if __name__ == "__main__":
    main()
