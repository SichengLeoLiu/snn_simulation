from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Models.layer import ExpandTemporalDim, IF
from Preprocess.getdataloader import MNIST_ROOT


METHODS = ("weights_only", "all_parameters")


class DeepNarrowFashionCNN(nn.Module):
    """Deep narrow Conv-BN-IF network for Fashion-MNIST.

    The same number of convolutions is used in each of three stages. Spatial
    pooling follows stages one and two. Every convolution has a trainable IF
    threshold, with one additional threshold in ``input_if``.
    """

    def __init__(
        self,
        channels=(8, 16, 24),
        blocks_per_stage=4,
        stage_blocks=None,
        residual_pairs=False,
        num_classes=10,
    ):
        super().__init__()
        if len(channels) != 3:
            raise ValueError("expected exactly three channel stages")
        if stage_blocks is None:
            stage_blocks = (blocks_per_stage,) * 3
        stage_blocks = tuple(int(value) for value in stage_blocks)
        if len(stage_blocks) != 3 or any(value < 1 for value in stage_blocks):
            raise ValueError("stage_blocks must contain three positive integers")

        self.T = 0
        self.expand = ExpandTemporalDim(0)
        self.input_if = IF()
        self.first_layer_input_noise_sigma = 0.0

        stage_channels = []
        for channel, block_count in zip(channels, stage_blocks):
            stage_channels.extend([int(channel)] * block_count)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.ifs = nn.ModuleList()
        in_channels = 1
        for out_channels in stage_channels:
            self.convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            self.bns.append(nn.BatchNorm2d(out_channels))
            self.ifs.append(IF())
            in_channels = out_channels

        self.stage_blocks = stage_blocks
        self.pool_after = (stage_blocks[0], stage_blocks[0] + stage_blocks[1])
        self.residual_pairs = bool(residual_pairs)
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Linear(stage_channels[-1] * 7 * 7, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    @property
    def num_conv_layers(self):
        return len(self.convs)

    def set_T(self, T):
        self.T = int(T)
        for module in self.modules():
            if isinstance(module, (IF, ExpandTemporalDim)):
                module.T = self.T

    def set_L(self, L):
        for module in self.modules():
            if isinstance(module, IF):
                module.L = int(L)

    def set_mode(self, mode):
        for module in self.modules():
            if isinstance(module, IF):
                module.mode = mode

    def set_first_layer_input_noise_sigma(self, sigma):
        self.first_layer_input_noise_sigma = max(0.0, float(sigma))

    def forward(self, x):
        if self.T > 0:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            x = x.flatten(0, 1).contiguous()

        x = self.input_if(x)
        if self.first_layer_input_noise_sigma > 0:
            x = x + self.first_layer_input_noise_sigma * torch.randn_like(x)

        stage_start = 1
        pending_residual = None
        for index, (conv, bn, if_layer) in enumerate(
            zip(self.convs, self.bns, self.ifs), start=1
        ):
            pre_activation = bn(conv(x))
            stage_position = index - stage_start + 1
            if self.residual_pairs and stage_position % 2 == 1:
                pending_residual = x if x.shape == pre_activation.shape else None
                x = if_layer(pre_activation)
            elif self.residual_pairs and pending_residual is not None:
                x = if_layer(pre_activation + pending_residual)
                pending_residual = None
            else:
                x = if_layer(pre_activation)
            if index in self.pool_after:
                x = self.pool(x)
                stage_start = index + 1
                pending_residual = None

        x = self.classifier(torch.flatten(x, 1))
        if self.T > 0:
            x = self.expand(x)
        return x


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested):
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        print("[WARN] MPS unavailable; falling back to CPU", flush=True)
        return torch.device("cpu")
    return device


def make_loaders(args):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
    )
    root = str(Path(MNIST_ROOT).expanduser())
    try:
        full_train = datasets.FashionMNIST(
            root, train=True, download=False, transform=transform
        )
        test_data = datasets.FashionMNIST(
            root, train=False, download=False, transform=transform
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"Fashion-MNIST is not available under {root}; run "
            "scripts/download_fashion_mnist.py first"
        ) from error

    generator = torch.Generator().manual_seed(args.split_seed)
    permutation = torch.randperm(len(full_train), generator=generator).tolist()
    val_indices = permutation[: args.val_size]
    train_indices = permutation[args.val_size :]
    if args.limit_train:
        train_indices = train_indices[: args.limit_train]
    if args.limit_val:
        val_indices = val_indices[: args.limit_val]
    if args.limit_test:
        test_data = Subset(test_data, range(min(args.limit_test, len(test_data))))

    train_data = Subset(full_train, train_indices)
    val_data = Subset(full_train, val_indices)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": False,
    }
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **common,
    )
    train_eval_loader = DataLoader(train_data, shuffle=False, **common)
    val_loader = DataLoader(val_data, shuffle=False, **common)
    test_loader = DataLoader(test_data, shuffle=False, **common)
    return train_loader, train_eval_loader, val_loader, test_loader


def l2_penalty(model, method):
    if method == "all_parameters":
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    elif method == "weights_only":
        parameters = [
            module.weight
            for module in model.modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]
    else:
        raise ValueError(f"unknown method: {method}")
    return torch.stack([parameter.pow(2).sum() for parameter in parameters]).sum()


def decode_logits(outputs):
    return outputs.mean(0) if outputs.dim() == 3 else outputs


def train_epoch(model, loader, optimizer, criterion, method, reg_coeff, device):
    model.train()
    ce_sum = 0.0
    objective_sum = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = decode_logits(model(images))
        ce = criterion(logits, labels)
        objective = ce + reg_coeff * l2_penalty(model, method)
        objective.backward()
        optimizer.step()

        batch_size = labels.size(0)
        ce_sum += float(ce.detach()) * batch_size
        objective_sum += float(objective.detach()) * batch_size
        correct += int(logits.argmax(1).eq(labels).sum())
        count += batch_size
    return {
        "ce": ce_sum / count,
        "objective": objective_sum / count,
        "acc": 100.0 * correct / count,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = decode_logits(model(images))
        batch_size = labels.size(0)
        loss_sum += float(criterion(logits, labels)) * batch_size
        correct += int(logits.argmax(1).eq(labels).sum())
        count += batch_size
    return {"ce": loss_sum / count, "acc": 100.0 * correct / count}


def scale_stats(model):
    lambdas = torch.cat(
        [module.thresh.detach().float().cpu().reshape(-1) for module in model.modules() if isinstance(module, IF)]
    )
    gammas = torch.cat(
        [module.weight.detach().float().cpu().reshape(-1) for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    )
    return {
        "n_lambda": int(lambdas.numel()),
        "lambda_mean": float(lambdas.mean()),
        "lambda_min": float(lambdas.min()),
        "lambda_max": float(lambdas.max()),
        "gamma_abs_mean": float(gammas.abs().mean()),
        "gamma_rms": float(gammas.pow(2).mean().sqrt()),
        "gamma_abs_median": float(gammas.abs().median()),
    }


def layer_scale_rows(model, method, checkpoint_kind):
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, IF):
            rows.append(
                {
                    "method": method,
                    "checkpoint": checkpoint_kind,
                    "parameter_type": "lambda",
                    "layer": name,
                    "value": float(module.thresh.detach().cpu()),
                }
            )
        elif isinstance(module, nn.BatchNorm2d):
            gamma = module.weight.detach().float().cpu()
            rows.append(
                {
                    "method": method,
                    "checkpoint": checkpoint_kind,
                    "parameter_type": "gamma_rms",
                    "layer": name,
                    "value": float(gamma.pow(2).mean().sqrt()),
                }
            )
    return rows


def load_state(model, state, device):
    model.load_state_dict(state, strict=True)
    model.to(device)


def evaluate_checkpoint(
    model,
    state,
    method,
    checkpoint_kind,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    args,
    device,
):
    load_state(model, state, device)
    model.set_T(0)
    model.set_L(args.L)
    model.set_mode("normal")
    model.set_first_layer_input_noise_sigma(0.0)
    train_metrics = evaluate(model, train_loader, criterion, device)
    val_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)

    row = {
        "method": method,
        "checkpoint": checkpoint_kind,
        "train_ann_acc": train_metrics["acc"],
        "val_ann_acc": val_metrics["acc"],
        "test_ann_acc": test_metrics["acc"],
        "train_val_gap": train_metrics["acc"] - val_metrics["acc"],
    }
    row.update(scale_stats(model))

    model.set_T(args.test_T)
    model.set_L(args.L)
    model.set_mode(args.if_mode)
    for sigma in (0.0, 1.0):
        repeats = 1 if sigma == 0 else args.noise_repeats
        accuracies = []
        for repeat in range(repeats):
            seed_everything(args.seed + 10000 + repeat)
            model.set_first_layer_input_noise_sigma(sigma)
            accuracies.append(evaluate(model, test_loader, criterion, device)["acc"])
        row[f"snn_sigma{int(sigma)}_acc"] = float(np.mean(accuracies))
        row[f"snn_sigma{int(sigma)}_std"] = float(np.std(accuracies, ddof=1)) if repeats > 1 else 0.0
    row["snn_drop"] = row["snn_sigma0_acc"] - row["snn_sigma1_acc"]
    return row, layer_scale_rows(model, method, checkpoint_kind)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_method(method, loaders, args, device):
    seed_everything(args.seed)
    model = DeepNarrowFashionCNN(
        channels=tuple(args.channels),
        blocks_per_stage=args.blocks_per_stage,
        stage_blocks=args.stage_blocks,
        residual_pairs=args.residual_pairs,
    ).to(device)
    model.set_T(0)
    model.set_L(args.L)
    initial_state = copy.deepcopy(model.state_dict())

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss()
    train_loader, train_eval_loader, val_loader, test_loader = loaders

    best_val_acc = -1.0
    best_epoch = -1
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            method,
            args.reg_coeff,
            device,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        stats = scale_stats(model)
        history.append(
            {
                "method": method,
                "epoch": epoch,
                "lr": scheduler.get_last_lr()[0],
                "train_ce": train_metrics["ce"],
                "train_objective": train_metrics["objective"],
                "train_online_acc": train_metrics["acc"],
                "val_ce": val_metrics["ce"],
                "val_acc": val_metrics["acc"],
                **stats,
            }
        )
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        scheduler.step()
        print(
            f"[{method}] epoch={epoch:02d}/{args.epochs} "
            f"train={train_metrics['acc']:.2f} val={val_metrics['acc']:.2f} "
            f"lambda={stats['lambda_mean']:.3f}",
            flush=True,
        )

    final_state = copy.deepcopy(model.state_dict())
    method_dir = args.out_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "initial_state": initial_state,
            "best_epoch": best_epoch,
            "args": vars(args),
        },
        method_dir / "best_val.pt",
    )
    torch.save(
        {"model_state": final_state, "best_epoch": best_epoch, "args": vars(args)},
        method_dir / "final.pt",
    )

    summary_rows = []
    layer_rows = []
    checkpoints = [("best_val", best_state)]
    if not args.evaluate_best_only:
        checkpoints.append(("final", final_state))
    for checkpoint_kind, state in checkpoints:
        row, scales = evaluate_checkpoint(
            model,
            state,
            method,
            checkpoint_kind,
            train_eval_loader,
            val_loader,
            test_loader,
            criterion,
            args,
            device,
        )
        row["best_epoch"] = best_epoch
        summary_rows.append(row)
        layer_rows.extend(scales)
    return history, summary_rows, layer_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deep-narrow Fashion-MNIST L2-scope and lambda-overfit pilot"
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--val-size", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--reg-coeff", type=float, default=2.5e-4)
    parser.add_argument("--channels", type=int, nargs=3, default=(8, 16, 24))
    parser.add_argument("--blocks-per-stage", type=int, default=4)
    parser.add_argument(
        "--stage-blocks",
        type=int,
        nargs=3,
        default=None,
        help="optional per-stage block counts; overrides --blocks-per-stage",
    )
    parser.add_argument(
        "--residual-pairs",
        action="store_true",
        help="add an identity shortcut around each same-shape pair of Conv-BN-IF layers",
    )
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--test-T", type=int, default=16)
    parser.add_argument("--if-mode", default="rate_uniform")
    parser.add_argument("--noise-repeats", type=int, default=3)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument(
        "--evaluate-best-only",
        action="store_true",
        help="skip the unused final-checkpoint evaluation",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT.parent / "important_results" / "fashion_deep_narrow_lambda_overfit_seed42",
    )
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    return args


def main():
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] device={device} out={args.out_dir}", flush=True)
    loaders = make_loaders(args)

    all_history = []
    all_summary = []
    all_layers = []
    for method in args.methods:
        history, summary, layers = run_method(method, loaders, args, device)
        all_history.extend(history)
        all_summary.extend(summary)
        all_layers.extend(layers)

    write_csv(args.out_dir / "training_history.csv", all_history)
    write_csv(args.out_dir / "summary.csv", all_summary)
    write_csv(args.out_dir / "layer_scales.csv", all_layers)
    with (args.out_dir / "config.json").open("w") as handle:
        json.dump({**vars(args), "out_dir": str(args.out_dir)}, handle, indent=2)
    print(f"[DONE] summary={args.out_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
