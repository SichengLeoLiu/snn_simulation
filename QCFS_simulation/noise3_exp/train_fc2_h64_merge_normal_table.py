import csv
import math
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models import modelpool
from Models.layer import add_dimention
from Preprocess import datapool
from utils import get_torch_device, seed_all, val


def coeff_tag(coeff: float) -> str:
    return f"{coeff:.0e}".replace("-", "m").replace("+", "p")


def build_ckpt_name(arch: str, l_val: int, regularizer: str, reg_coeff: float) -> tuple[str, str]:
    if regularizer == "weight_decay":
        suffix = f"ablation_wd_l{l_val}_{arch}"
        name = f"{arch}_L[{l_val}]_{suffix}.pth"
    elif regularizer == "none":
        suffix = f"ablation_none_l{l_val}_{arch}"
        name = f"{arch}_L[{l_val}]_{suffix}.pth"
    elif regularizer == "mne_l2":
        suffix = f"ablation_mne_l2_l{l_val}_{arch}_rc{coeff_tag(reg_coeff)}"
        name = f"{arch}_L[{l_val}]_{suffix}.pth"
    else:
        raise ValueError(f"unsupported regularizer: {regularizer}")
    return name, suffix


def train_one(arch: str, l_val: int, regularizer: str, reg_coeff: float) -> Path:
    ckpt_name, suffix = build_ckpt_name(arch, l_val, regularizer, reg_coeff)
    ckpt_path = PROJECT_ROOT / "mnist-checkpoints" / ckpt_name
    if ckpt_path.exists():
        print(f"[SKIP TRAIN] exists: {ckpt_path}")
        return ckpt_path

    if regularizer == "weight_decay":
        wd = 5e-4
        reg = "weight_decay"
        coeff = 1.0
    elif regularizer == "none":
        wd = 0.0
        reg = "weight_decay"
        coeff = 1.0
    else:
        wd = 0.0
        reg = "mne_l2"
        coeff = reg_coeff

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "main_train.py"),
        "-data",
        "mnist",
        "-arch",
        arch,
        "-L",
        str(l_val),
        "--epochs",
        "100",
        "-j",
        "0",
        "-b",
        "128",
        "--seed",
        "42",
        "--device",
        "auto",
        "--time",
        "0",
        "--spike_schedule",
        "normal",
        "--regularizer",
        reg,
        "--weight_decay",
        str(wd),
        "--reg_coeff",
        str(coeff),
        "--suffix",
        suffix,
    ]
    print("[TRAIN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint missing after train: {ckpt_path}")
    print(f"[TRAIN DONE] {ckpt_path}")
    return ckpt_path


def fc1_pre_if1_mean_over_time(model, images):
    x = images.clone()
    t_val = int(getattr(model, "T", 0))
    if t_val > 0:
        x = add_dimention(x, t_val)
        x = model.merge(x)
    x = torch.flatten(x, 1)
    x = model.input_if(x)
    x = model._inject_first_layer_input_noise(x)
    x = model.fc1(x)
    if t_val > 0:
        bsz = images.shape[0]
        hidden = x.shape[1]
        x = x.view(t_val, bsz, hidden).mean(0)
    return x


def eval_stats(model, test_loader, device, l_val: int, t_val: int):
    noise_variance = 0.5
    noise_sigma = math.sqrt(noise_variance)
    repeats = 6
    max_batches = 4

    model.set_L(int(l_val))
    model.set_T(int(t_val))
    model.set_mode("normal")
    if hasattr(model, "set_first_layer_input_noise_type"):
        model.set_first_layer_input_noise_type("gaussian")

    lam = float(model.if1.thresh.detach().view(-1)[0].item())
    h_tol = lam / (2.0 * float(l_val))

    cnt = 0
    eq = 0
    s = 0.0
    ss = 0.0
    model.set_first_layer_input_noise_sigma(0.0)

    with torch.no_grad():
        for bi, (images, _) in enumerate(test_loader):
            if bi >= max_batches:
                break
            images = images.to(device)
            clean = fc1_pre_if1_mean_over_time(model, images)
            for _ in range(repeats):
                model.set_first_layer_input_noise_sigma(noise_sigma)
                noisy = fc1_pre_if1_mean_over_time(model, images)
                delta = (noisy - clean).reshape(-1).float().cpu()
                eq += int((delta.abs() < h_tol).sum().item())
                cnt += int(delta.numel())
                s += float(delta.sum().item())
                ss += float((delta * delta).sum().item())
            model.set_first_layer_input_noise_sigma(0.0)

    mean = s / max(cnt, 1)
    var_eff = max(ss / max(cnt, 1) - mean * mean, 0.0)
    sigma_eff = math.sqrt(var_eff)
    rms_delta = math.sqrt(max(ss / max(cnt, 1), 0.0))
    p_e = eq / max(cnt, 1)
    acc = float(val(model, test_loader, T=int(t_val), device=device, verbose=False))

    return {
        "sigma_eff": sigma_eff,
        "var_eff": var_eff,
        "rms_delta": rms_delta,
        "p_e": p_e,
        "equal_count": eq,
        "total_count": cnt,
        "lambda_if1": lam,
        "h_tolerance": h_tol,
        "noise_variance": noise_variance,
        "noise_sigma": noise_sigma,
        "repeats": repeats,
        "max_batches": max_batches,
        "acc": acc,
    }


def key_for_row(r: dict) -> tuple:
    return (
        r.get("checkpoint_name", ""),
        r.get("L", ""),
        r.get("T", ""),
        r.get("if_mode", ""),
        r.get("metric_position", ""),
        r.get("time_reduce", ""),
        r.get("noise_variance", ""),
        r.get("noise_type", ""),
        r.get("repeats", ""),
        r.get("max_batches", ""),
    )


def main():
    in_table = PROJECT_ROOT / "noise3_exp/combined_sigmaeff_and_P_tolerant_same_preif1_position_normal_fc2_one_table.csv"
    out_table = PROJECT_ROOT / "noise3_exp/combined_sigmaeff_and_P_tolerant_same_preif1_position_normal_fc2_one_table_with_h64.csv"

    with in_table.open("r", newline="") as f:
        old_rows = list(csv.DictReader(f))
        fieldnames = old_rows[0].keys()

    arch = "fc2_h64"
    l_list = [2, 4, 8, 16, 32]
    t_list = [2, 4, 8, 16, 32]
    reg_list = ["weight_decay", "none", "mne_l2"]
    mne_coeff = 1e-3

    seed_all(44)
    device = get_torch_device("mps")
    _, test_loader = datapool("mnist", 128, num_workers=0, pin_memory=False)
    print(f"[DEVICE] {device}")

    # train / load checkpoints for h64
    ckpt_map = {}
    for l_val in l_list:
        for reg in reg_list:
            ckpt = train_one(arch=arch, l_val=l_val, regularizer=reg, reg_coeff=mne_coeff)
            ckpt_map[(l_val, reg)] = ckpt

    # evaluate rows for h64
    new_rows = []
    model_cache = {}
    for l_val in l_list:
        for reg in reg_list:
            ckpt = ckpt_map[(l_val, reg)]
            if ckpt not in model_cache:
                model = modelpool(arch, "mnist")
                state = torch.load(ckpt, map_location="cpu")
                model.load_state_dict(state, strict=True)
                model.to(device)
                model.eval()
                model_cache[ckpt] = model
            model = model_cache[ckpt]

            for t_val in t_list:
                stats = eval_stats(model, test_loader, device, l_val=l_val, t_val=t_val)
                row = {
                    "folder": "mnist-checkpoints",
                    "checkpoint": str(Path("mnist-checkpoints") / ckpt.name),
                    "checkpoint_name": ckpt.name,
                    "L": str(l_val),
                    "T": str(t_val),
                    "if_mode": "normal",
                    "metric_position": "fc1_output_pre_if1",
                    "time_reduce": "mean_over_time",
                    "sigma_eff": f"{stats['sigma_eff']:.15g}",
                    "var_eff": f"{stats['var_eff']:.15g}",
                    "rms_delta": f"{stats['rms_delta']:.15g}",
                    "p_metric": "P_tolerant_same_position",
                    "p_rule": "|x_noisy-x_clean| < lambda/(2L)",
                    "p_e": f"{stats['p_e']:.15g}",
                    "equal_count": str(stats["equal_count"]),
                    "total_count": str(stats["total_count"]),
                    "lambda_if1": f"{stats['lambda_if1']:.15g}",
                    "h_tolerance": f"{stats['h_tolerance']:.15g}",
                    "noise_variance": str(stats["noise_variance"]),
                    "noise_sigma": f"{stats['noise_sigma']:.16g}",
                    "noise_type": "gaussian",
                    "repeats": str(stats["repeats"]),
                    "max_batches": str(stats["max_batches"]),
                    "acc": f"{stats['acc']:.6f}",
                }
                new_rows.append(row)
                print(
                    f"[EVAL] {ckpt.name} T={t_val} p_e={stats['p_e']:.4f} acc={stats['acc']:.3f}"
                )

    merged = {key_for_row(r): r for r in old_rows}
    for r in new_rows:
        merged[key_for_row(r)] = r

    out_rows = list(merged.values())
    out_rows.sort(key=lambda r: (r["checkpoint_name"], int(r["L"]), int(r["T"])))

    with out_table.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(out_rows)

    print(f"[DONE] wrote merged table: {out_table} rows={len(out_rows)}")


if __name__ == "__main__":
    main()
