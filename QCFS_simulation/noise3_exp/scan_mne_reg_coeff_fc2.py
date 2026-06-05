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


def coeff_to_tag(coeff: float) -> str:
    s = f"{coeff:.0e}"
    return s.replace("-", "m").replace("+", "p")


def ckpt_name_for(arch: str, l_val: int, coeff: float) -> str:
    suffix = f"ablation_mne_l2_l{l_val}_{arch}_rc{coeff_to_tag(coeff)}"
    return f"{arch}_L[{l_val}]_{suffix}.pth", suffix


def run_train(
    project_root: Path,
    arch: str,
    l_val: int,
    coeff: float,
    seed: int = 42,
    epochs: int = 100,
) -> Path:
    ckpt_file, suffix = ckpt_name_for(arch, l_val, coeff)
    ckpt_path = project_root / "mnist-checkpoints" / ckpt_file
    if ckpt_path.exists():
        print(f"[SKIP TRAIN] exists: {ckpt_path}")
        return ckpt_path

    cmd = [
        sys.executable,
        str(project_root / "main_train.py"),
        "-data",
        "mnist",
        "-arch",
        arch,
        "-L",
        str(l_val),
        "--epochs",
        str(epochs),
        "-j",
        "0",
        "-b",
        "128",
        "--seed",
        str(seed),
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
        str(coeff),
        "--suffix",
        suffix,
    ]
    print("[TRAIN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(project_root), check=True)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"expected checkpoint not found: {ckpt_path}")
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


def evaluate_pe_acc(
    model,
    test_loader,
    device: torch.device,
    l_val: int,
    t_val: int,
    noise_sigma: float = math.sqrt(0.5),
    repeats: int = 6,
    max_batches: int = 4,
):
    model.set_L(int(l_val))
    model.set_T(int(t_val))
    model.set_mode("normal")
    if hasattr(model, "set_first_layer_input_noise_type"):
        model.set_first_layer_input_noise_type("gaussian")

    lam = float(model.if1.thresh.detach().view(-1)[0].item())
    h_tol = lam / (2.0 * float(l_val))
    eq = 0
    cnt = 0

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

    p_e = eq / max(cnt, 1)
    mean = s / max(cnt, 1)
    var_eff = max(ss / max(cnt, 1) - mean * mean, 0.0)
    sigma_eff = math.sqrt(var_eff)
    acc = float(val(model, test_loader, T=int(t_val), device=device, verbose=False))
    return p_e, acc, sigma_eff


def load_model_from_ckpt(ckpt_path: Path, arch: str, device: torch.device):
    model = modelpool(arch, "mnist")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def main():
    project_root = PROJECT_ROOT
    out_dir = Path(__file__).resolve().parent
    out_csv = out_dir / "mne_l2_reg_coeff_scan_fc2_normal.csv"
    coeffs = [1e-4, 3e-4, 5e-4, 1e-3]
    archs = [("fc2_h128", 128), ("fc2", 256), ("fc2_h512", 512)]
    l_val = 16
    t_values = [8, 16]

    seed_all(44)
    device = get_torch_device("mps")
    _, test_loader = datapool("mnist", 128, num_workers=0, pin_memory=False)
    print(f"[DEVICE] {device}")

    rows = []
    for arch, hidden in archs:
        for coeff in coeffs:
            ckpt = run_train(project_root, arch=arch, l_val=l_val, coeff=coeff)
            model = load_model_from_ckpt(ckpt, arch=arch, device=device)
            for t_val in t_values:
                p_e, acc, sigma_eff = evaluate_pe_acc(
                    model, test_loader, device, l_val=l_val, t_val=t_val
                )
                rows.append(
                    {
                        "arch": arch,
                        "hidden_size": hidden,
                        "L": l_val,
                        "T": t_val,
                        "reg_coeff": coeff,
                        "checkpoint": str(ckpt.relative_to(project_root)),
                        "p_e": p_e,
                        "acc": acc,
                        "sigma_eff": sigma_eff,
                    }
                )
                print(
                    f"[EVAL] arch={arch} coeff={coeff:.1e} T={t_val} "
                    f"p_e={p_e:.4f} acc={acc:.3f} sigma_eff={sigma_eff:.4f}"
                )

    rows.sort(key=lambda r: (r["T"], r["hidden_size"], r["reg_coeff"]))
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "arch",
                "hidden_size",
                "L",
                "T",
                "reg_coeff",
                "checkpoint",
                "p_e",
                "acc",
                "sigma_eff",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] wrote csv: {out_csv}")

    colors = {128: "#1f77b4", 256: "#ff7f0e", 512: "#2ca02c"}
    for t_val in t_values:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        for _, hidden in archs:
            pts = [r for r in rows if r["T"] == t_val and r["hidden_size"] == hidden]
            pts = sorted(pts, key=lambda x: x["reg_coeff"])
            ax.plot(
                [p["acc"] for p in pts],
                [p["p_e"] for p in pts],
                marker="o",
                linewidth=1.8,
                color=colors[hidden],
                label=f"h{hidden}",
            )
            for p in pts:
                ax.annotate(
                    f"{p['reg_coeff']:.0e}",
                    (p["acc"], p["p_e"]),
                    textcoords="offset points",
                    xytext=(3, 4),
                    fontsize=8,
                    color=colors[hidden],
                )
        ax.set_xlabel("Accuracy (%)")
        ax.set_ylabel("P_e")
        ax.set_title(f"MNE-L2 reg_coeff scan (normal), L={l_val}, T={t_val}")
        ax.grid(alpha=0.25)
        ax.legend(title="hidden size")
        fig.tight_layout()
        out_png = out_dir / f"mne_l2_reg_coeff_scan_pe_acc_normal_L{l_val}_T{t_val}.png"
        fig.savefig(out_png, dpi=220)
        plt.close(fig)
        print(f"[DONE] wrote plot: {out_png}")


if __name__ == "__main__":
    main()
