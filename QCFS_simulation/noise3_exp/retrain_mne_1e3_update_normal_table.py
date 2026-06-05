import csv
import math
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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


def infer_arch(ckpt_name: str) -> str:
    if ckpt_name.startswith("fc2_h128_"):
        return "fc2_h128"
    if ckpt_name.startswith("fc2_h512_"):
        return "fc2_h512"
    return "fc2"


def run_train_for_mne(arch: str, l_val: int, reg_coeff: float) -> Path:
    suffix = f"ablation_mne_l2_l{l_val}_{arch}_rc{coeff_tag(reg_coeff)}"
    ckpt_name = f"{arch}_L[{l_val}]_{suffix}.pth"
    ckpt = PROJECT_ROOT / "mnist-checkpoints" / ckpt_name
    if ckpt.exists():
        print(f"[SKIP TRAIN] exists: {ckpt}")
        return ckpt

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
        "mne_l2",
        "--weight_decay",
        "0.0",
        "--reg_coeff",
        str(reg_coeff),
        "--suffix",
        suffix,
    ]
    print("[TRAIN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not produced: {ckpt}")
    print(f"[TRAIN DONE] {ckpt}")
    return ckpt


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


def eval_one(model, test_loader, device, l_val: int, t_val: int):
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


def plot_mne_lines(rows, out_dir: Path):
    def size_from_name(name: str) -> int:
        if name.startswith("fc2_h128_"):
            return 128
        if name.startswith("fc2_h512_"):
            return 512
        return 256

    for t_val in (8, 16):
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        for size in (128, 256, 512):
            pts = [
                r
                for r in rows
                if int(r["T"]) == t_val
                and "ablation_mne_l2" in r["checkpoint_name"]
                and size_from_name(r["checkpoint_name"]) == size
            ]
            pts.sort(key=lambda z: int(z["L"]))
            if not pts:
                continue
            ax.plot(
                [int(p["L"]) for p in pts],
                [float(p["p_e"]) for p in pts],
                marker="o",
                linewidth=2,
                label=f"h{size}",
            )
        ax.set_title(f"P_e vs L (normal, mne_l2@1e-3, T={t_val})")
        ax.set_xlabel("L")
        ax.set_ylabel("P_e")
        ax.set_xticks([2, 4, 8, 16, 32])
        ax.grid(alpha=0.25)
        ax.legend(loc="best", title="hidden size")
        fig.tight_layout()
        out_png = out_dir / f"pe_vs_L_fc2_sizes_normal_mne_l2_rc1e3_T{t_val}.png"
        fig.savefig(out_png, dpi=220)
        plt.close(fig)
        print(f"[DONE] wrote plot: {out_png}")


def main():
    table = PROJECT_ROOT / "noise3_exp/combined_sigmaeff_and_P_tolerant_same_preif1_position_normal_fc2_one_table.csv"
    out_dir = table.parent
    reg_coeff = 1e-3

    with table.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())

    # unique L/T pairs by model for current mne rows
    mne_targets = {}
    for r in rows:
        if "ablation_mne_l2" not in r["checkpoint_name"]:
            continue
        arch = infer_arch(r["checkpoint_name"])
        l_val = int(r["L"])
        mne_targets[(arch, l_val)] = True
    targets = sorted(mne_targets.keys(), key=lambda x: (x[0], x[1]))
    print(f"[INFO] targets={len(targets)} (arch,L): {targets}")

    seed_all(44)
    device = get_torch_device("mps")
    _, test_loader = datapool("mnist", 128, num_workers=0, pin_memory=False)

    # train/load new checkpoints
    new_ckpt_map = {}
    for arch, l_val in targets:
        ckpt = run_train_for_mne(arch, l_val, reg_coeff)
        new_ckpt_map[(arch, l_val)] = ckpt

    # replace mne rows in table with re-evaluated stats from rc1e-3 ckpts
    updated_rows = []
    model_cache = {}
    for r in rows:
        if "ablation_mne_l2" not in r["checkpoint_name"]:
            updated_rows.append(r)
            continue
        arch = infer_arch(r["checkpoint_name"])
        l_val = int(r["L"])
        t_val = int(r["T"])
        ckpt = new_ckpt_map[(arch, l_val)]
        if arch not in model_cache or model_cache[arch][0] != ckpt:
            model = modelpool(arch, "mnist")
            sd = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(sd, strict=True)
            model.to(device)
            model.eval()
            model_cache[arch] = (ckpt, model)
        model = model_cache[arch][1]
        stats = eval_one(model, test_loader, device, l_val=l_val, t_val=t_val)

        suffix = f"ablation_mne_l2_l{l_val}_{arch}_rc{coeff_tag(reg_coeff)}"
        ckpt_name = f"{arch}_L[{l_val}]_{suffix}.pth"
        new_r = dict(r)
        new_r["folder"] = "mnist-checkpoints"
        new_r["checkpoint"] = str(Path("mnist-checkpoints") / ckpt_name)
        new_r["checkpoint_name"] = ckpt_name
        new_r["if_mode"] = "normal"
        new_r["metric_position"] = "fc1_output_pre_if1"
        new_r["time_reduce"] = "mean_over_time"
        new_r["sigma_eff"] = f"{stats['sigma_eff']:.15g}"
        new_r["var_eff"] = f"{stats['var_eff']:.15g}"
        new_r["rms_delta"] = f"{stats['rms_delta']:.15g}"
        new_r["p_metric"] = "P_tolerant_same_position"
        new_r["p_rule"] = "|x_noisy-x_clean| < lambda/(2L)"
        new_r["p_e"] = f"{stats['p_e']:.15g}"
        new_r["equal_count"] = str(stats["equal_count"])
        new_r["total_count"] = str(stats["total_count"])
        new_r["lambda_if1"] = f"{stats['lambda_if1']:.15g}"
        new_r["h_tolerance"] = f"{stats['h_tolerance']:.15g}"
        new_r["noise_variance"] = str(stats["noise_variance"])
        new_r["noise_sigma"] = f"{stats['noise_sigma']:.16g}"
        new_r["noise_type"] = "gaussian"
        new_r["repeats"] = str(stats["repeats"])
        new_r["max_batches"] = str(stats["max_batches"])
        new_r["acc"] = f"{stats['acc']:.6f}"
        updated_rows.append(new_r)
        print(
            f"[UPD] {ckpt_name} T={t_val} p_e={stats['p_e']:.4f} acc={stats['acc']:.3f}"
        )

    updated_rows.sort(key=lambda x: (x["checkpoint_name"], int(x["L"]), int(x["T"])))
    with table.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(updated_rows)
    print(f"[DONE] updated table: {table} rows={len(updated_rows)}")

    plot_mne_lines(updated_rows, out_dir)


if __name__ == "__main__":
    main()
