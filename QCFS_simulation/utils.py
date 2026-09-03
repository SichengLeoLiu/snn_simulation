import time
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import random
import os
import logging
import re
from Models import IF


def _mps_available():
    return getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()


def get_torch_device(device_str: str = "auto") -> torch.device:
    """
    选择训练/推理设备。
    - auto: cuda（若可用）> mps（若可用）> cpu
    - cpu / mps / cuda / cuda:N：按名称强制使用（不可用时抛错）
    """
    s = (device_str or "auto").strip().lower()
    if s in ("auto", "", "0"):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if s == "cpu":
        return torch.device("cpu")
    if s in ("mps", "metal"):
        if not _mps_available():
            raise RuntimeError("指定了 mps，但当前环境不可用（需 Apple Silicon 且 PyTorch 支持 MPS）")
        return torch.device("mps")
    if s.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 cuda，但当前环境不可用")
        return torch.device(s if ":" in s else "cuda:0")
    return torch.device(device_str)


def configure_cuda_fast(device: torch.device) -> None:
    """H200/A100 吞吐：TF32 + cuDNN benchmark。由环境变量 QCFS_CUDA_FAST=1 打开。"""
    enabled = os.environ.get("QCFS_CUDA_FAST", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled or device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    cap = torch.cuda.get_device_capability(device)
    print(
        "QCFS_CUDA_FAST: name=%s capability=%s cuda=%s tf32=1 cudnn.benchmark=1"
        % (torch.cuda.get_device_name(device), cap, torch.version.cuda),
        flush=True,
    )


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if _mps_available() and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def train(
    model,
    device,
    train_loader,
    criterion,
    optimizer,
    T,
    quant_level=None,
    reg_loss_fn=None,
    reg_coeff=1.0,
):
    running_loss = 0
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    epoch_acc = None
    n_reg_stats = 0
    for i, (images, labels) in enumerate((train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)
        ga_mode = str(getattr(model, "_ga_mne_mode", "") or "")
        gm_cfg = _gm_os_cfg(model)
        ga_on = ga_mode in ("nocov", "full") or bool(gm_cfg and gm_cfg.get("use_m"))
        if ga_on:
            set_basicblock_ga_cache(model, True)
        if T > 0:
            outputs = model(images).mean(0)
        else:
            outputs = model(images)
        loss = criterion(outputs, labels)
        if gm_cfg is not None:
            update_graph_margin_stats(model, loss)
        if reg_loss_fn is not None:
            reg = reg_loss_fn(model, T, quant_level)
            loss = loss + float(reg_coeff) * reg
            stats = getattr(model, "_calibrated_mne_stats", None)
            if stats is not None:
                if epoch_acc is None:
                    epoch_acc = {
                        "q_mean": 0.0,
                        "q_std": 0.0,
                        "q_min": float("inf"),
                        "q_max": float("-inf"),
                        "q_os_mean": 0.0,
                        "q_os_std": 0.0,
                        "p_gt_tau": 0.0,
                        "p_at_qmax": 0.0,
                    }
                n_reg_stats += 1
                for key in (
                    "q_mean",
                    "q_std",
                    "q_os_mean",
                    "q_os_std",
                    "p_gt_tau",
                    "p_at_qmax",
                ):
                    epoch_acc[key] += float(stats.get(key, 0.0))
                epoch_acc["q_min"] = min(
                    epoch_acc["q_min"], float(stats.get("q_min", 1.0))
                )
                epoch_acc["q_max"] = max(
                    epoch_acc["q_max"], float(stats.get("q_max", 1.0))
                )
        if ga_on:
            set_basicblock_ga_cache(model, False)
        running_loss += loss.item()
        loss.backward()
        if gm_cfg is not None:
            clear_if_pre_quant(model)
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    if epoch_acc is not None and n_reg_stats > 0:
        averaged = {
            key: epoch_acc[key] / float(n_reg_stats)
            for key in (
                "q_mean",
                "q_std",
                "q_os_mean",
                "q_os_std",
                "p_gt_tau",
                "p_at_qmax",
            )
        }
        averaged["q_min"] = epoch_acc["q_min"]
        averaged["q_max"] = epoch_acc["q_max"]
        model._calibrated_mne_epoch_stats = averaged
    return running_loss, 100 * correct / total


def train_reg(
    model,
    device,
    train_loader,
    criterion,
    optimizer,
    T,
    quant_level=None,
    reg_loss_fn=None,
    reg_coeff=1.0,
):
    """回归任务：MSE；返回 (loss 累加和, 训练集 MAE)。"""
    running_loss = 0.0
    model.train()
    total_abs = 0.0
    total_n = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        labels = labels.to(device, dtype=torch.float32)
        images = images.to(device)
        if T > 0:
            outputs = model(images).mean(0)
        else:
            outputs = model(images)
        loss = criterion(outputs.view(-1), labels.view(-1))
        if reg_loss_fn is not None:
            reg = reg_loss_fn(model, T, quant_level)
            loss = loss + float(reg_coeff) * reg
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_abs += (outputs.view(-1) - labels.view(-1)).abs().sum().item()
        total_n += labels.numel()
    return running_loss, total_abs / max(total_n, 1)


def _layer_map_from_model(model, layer_map=None) -> str:
    if layer_map is not None:
        value = str(layer_map).strip().lower()
    else:
        value = str(getattr(model, "_mne_layer_map", "legacy")).strip().lower()
    if value not in ("legacy", "resnet"):
        raise ValueError(f"layer_map must be legacy or resnet, got {value!r}.")
    return value


def _is_classifier_head(layer_name, module) -> bool:
    parts = str(layer_name).split(".")
    last = parts[-1]
    if not isinstance(module, nn.Linear):
        return False
    return last in ("fc", "classifier") or parts[0] == "classifier"


MNE_LAYER_ROLES = (
    "stem",
    "residual_preact",
    "residual_terminal",
    "shortcut",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
    "layer5",
    "classifier_head",
    "other",
)


def parse_mne_include_roles(value) -> tuple[str, ...] | None:
    if value in (None, "", (), []):
        return None
    if isinstance(value, str):
        roles = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        roles = tuple(str(part).strip() for part in value if str(part).strip())
    unknown = [role for role in roles if role not in MNE_LAYER_ROLES]
    if unknown:
        raise ValueError(
            "mne include_roles must be in %s, got %s"
            % (MNE_LAYER_ROLES, unknown)
        )
    return roles


def _weight_layer_role(layer_name: str) -> str:
    parts = str(layer_name).split(".")
    last = parts[-1]
    if last in ("fc", "classifier") or parts[0] == "classifier":
        return "classifier_head"
    if "shortcut" in parts:
        return "shortcut"
    if "residual_function" in parts and last.isdigit():
        return "residual_terminal" if int(last) >= 3 else "residual_preact"
    if parts[0] in ("layer1", "layer2", "layer3", "layer4", "layer5") and last.isdigit():
        return parts[0]
    if parts[0].startswith("conv1") or layer_name.startswith("conv1"):
        return "stem"
    return "other"


def _module_name(module_map, module) -> str:
    if module is None:
        return ""
    for name, candidate in module_map.items():
        if candidate is module:
            return name
    return ""


def _resolve_bn_if_for_layer(layer_name, module_map, layer_map="legacy"):
    """Match BN / IF to a Conv or Linear weight layer.

    legacy: same-Sequential Conv→BN→IF (VGG) or conv/fc name heuristics.
    resnet: also pair the terminal residual conv and shortcut conv with the
    post-add IF at ``BasicBlock.act``. Stem Conv-BN-IF still uses legacy.
    """
    parts = layer_name.split(".")
    token = parts[-1]
    parent = ".".join(parts[:-1])

    def _full(n):
        return f"{parent}.{n}" if parent else n

    bn_mod = None
    if_mod = None

    if token.isdigit():
        i = int(token)
        next1 = module_map.get(_full(str(i + 1)))
        next2 = module_map.get(_full(str(i + 2)))
        if isinstance(next1, nn.modules.batchnorm._BatchNorm):
            bn_mod = next1
            if isinstance(next2, IF):
                if_mod = next2
        elif isinstance(next1, IF):
            if_mod = next1

    if bn_mod is None and if_mod is None:
        bn_names = []
        if_names = []

        if token.startswith("conv"):
            bn_names += [_full(token.replace("conv", "bn", 1))]
            if_names += [_full(token.replace("conv", "if", 1))]
        elif token.startswith("fc"):
            bn_names += [_full(token.replace("fc", "bn", 1))]
            if_names += [_full(token.replace("fc", "if", 1))]
        elif token.startswith("classifier"):
            bn_names += [_full(token.replace("classifier", "bn", 1))]
            if_names += [_full(token.replace("classifier", "if", 1))]

        m = re.search(r"(\d+)$", token)
        if m:
            idx = m.group(1)
            bn_names.append(_full(f"bn{idx}"))
            if_names.append(_full(f"if{idx}"))

        for n in bn_names:
            mod = module_map.get(n, None)
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                bn_mod = mod
                break

        for n in if_names:
            mod = module_map.get(n, None)
            if isinstance(mod, IF):
                if_mod = mod
                break

    if if_mod is None and str(layer_map).strip().lower() == "resnet":
        parent_kind = parts[-2] if len(parts) >= 2 else ""
        if parent_kind in ("residual_function", "shortcut"):
            if bn_mod is None and token.isdigit():
                next1 = module_map.get(_full(str(int(token) + 1)))
                if isinstance(next1, nn.modules.batchnorm._BatchNorm):
                    bn_mod = next1
            block_name = ".".join(parts[:-2])
            act = module_map.get(f"{block_name}.act" if block_name else "act")
            if isinstance(act, IF):
                if_mod = act

    return bn_mod, if_mod


def collect_weight_layer_matches(model, layer_map=None) -> list[dict]:
    """One row per trainable Conv/Linear weight, with BN/IF match metadata."""
    layer_map = _layer_map_from_model(model, layer_map)
    module_map = dict(model.named_modules())
    rows = []
    for lname, layer in model.named_modules():
        if not isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            continue
        weight = getattr(layer, "weight", None)
        if weight is None or not weight.requires_grad:
            continue
        bn_mod, if_mod = _resolve_bn_if_for_layer(
            lname, module_map, layer_map=layer_map
        )
        rows.append(
            {
                "name": lname,
                "role": _weight_layer_role(lname),
                "is_head": _is_classifier_head(lname, layer),
                "matched": if_mod is not None,
                "n_params": int(weight.numel()),
                "out_channels": int(weight.shape[0]),
                "weight": weight,
                "bn": bn_mod,
                "if_mod": if_mod,
                "bn_name": _module_name(module_map, bn_mod),
                "if_name": _module_name(module_map, if_mod),
            }
        )
    return rows


def summarize_weight_layer_matches(rows: list[dict]) -> dict:
    n_total = len(rows)
    n_matched = sum(1 for row in rows if row["matched"])
    body = [row for row in rows if not row["is_head"]]
    head = [row for row in rows if row["is_head"]]
    body_params = sum(row["n_params"] for row in body)
    matched_body_params = sum(row["n_params"] for row in body if row["matched"])
    head_params = sum(row["n_params"] for row in head)
    return {
        "n_layers": n_total,
        "n_matched": n_matched,
        "n_unmatched": n_total - n_matched,
        "n_body_layers": len(body),
        "n_body_matched": sum(1 for row in body if row["matched"]),
        "n_head_layers": len(head),
        "matched_layers_over_total": f"{n_matched}/{n_total}",
        "body_param_ratio": (
            float(matched_body_params) / float(body_params) if body_params else 0.0
        ),
        "head_params": int(head_params),
        "body_params": int(body_params),
        "unmatched_body": [
            row["name"] for row in body if not row["matched"]
        ],
    }


def _is_basic_block(module) -> bool:
    return (
        hasattr(module, "residual_function")
        and hasattr(module, "shortcut")
        and hasattr(module, "act")
    )


def _iter_basic_blocks(model):
    for name, module in model.named_modules():
        if _is_basic_block(module):
            yield name, module


def set_basicblock_ga_cache(model, enabled: bool) -> None:
    for _, module in _iter_basic_blocks(model):
        module._ga_cache = bool(enabled)
        if not enabled:
            for key in ("_ga_x", "_ga_h_res", "_ga_h_sc"):
                if hasattr(module, key):
                    delattr(module, key)


def _ga_branch_for_layer(layer_name: str):
    parts = str(layer_name).split(".")
    if "residual_function" in parts:
        idx = parts.index("residual_function")
        last = parts[-1]
        if last.isdigit() and int(last) >= 3:
            return ".".join(parts[:idx]), "residual"
        return None
    if "shortcut" in parts:
        idx = parts.index("shortcut")
        return ".".join(parts[:idx]), "shortcut"
    return None


def _channel_moment(left, right):
    if left.dim() == 4:
        return (left * right).mean(dim=(0, 2, 3))
    return (left * right).mean(dim=0)


def _forward_branches_frozen(block, x):
    saved = []
    for module in list(block.residual_function.modules()) + list(block.shortcut.modules()):
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            saved.append((module, module.track_running_stats))
            module.track_running_stats = False
    try:
        h_res = block.residual_function(x)
        h_sc = block.shortcut(x)
        return h_res, h_sc
    finally:
        for module, flag in saved:
            module.track_running_stats = flag


def estimate_ga_branch_stats(
    model,
    include_cov: bool = True,
    probe_sigma: float = 0.1,
    eps: float = 1e-6,
) -> dict:
    """Per-block φ/κ from a probe at each additive merge.

    Uses cached block inputs from the last CE forward when present.
    Identity shortcuts still enter κ even though they have no weights.
    """
    stats = {}
    sigma = float(probe_sigma)
    for name, block in _iter_basic_blocks(model):
        x = getattr(block, "_ga_x", None)
        if x is None:
            continue
        x = x.detach()
        noise = sigma * torch.randn_like(x)
        with torch.no_grad():
            h_res, h_sc = _forward_branches_frozen(block, x)
            h_res_p, h_sc_p = _forward_branches_frozen(block, x + noise)
        dh_res = h_res_p - h_res
        dh_sc = h_sc_p - h_sc
        var_res = _channel_moment(dh_res, dh_res)
        var_sc = _channel_moment(dh_sc, dh_sc)
        cov = _channel_moment(dh_res, dh_sc)
        if include_cov:
            phi_res = var_res + cov
            phi_sc = var_sc + cov
        else:
            phi_res = var_res
            phi_sc = var_sc
        n_branch = 2
        pos_res = torch.relu(phi_res)
        pos_sc = torch.relu(phi_sc)
        denom = pos_res + pos_sc + float(eps)
        kappa_res = n_branch * pos_res / denom
        kappa_sc = n_branch * pos_sc / denom
        delta_u = dh_res + dh_sc
        stats[name] = {
            "n_branch": n_branch,
            "phi_res": phi_res,
            "phi_sc": phi_sc,
            "kappa_res": kappa_res,
            "kappa_sc": kappa_sc,
            "var_res": var_res,
            "var_sc": var_sc,
            "cov": cov,
            "delta_u_rms": delta_u.pow(2).mean(dim=(0, 2, 3)).sqrt()
            if delta_u.dim() == 4
            else delta_u.pow(2).mean(dim=0).sqrt(),
            "has_shortcut_conv": any(
                isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
                for child in block.shortcut.modules()
            ),
        }
    return stats


def _kappa_for_layer(layer_name: str, ga_stats: dict, channels: int, device, dtype):
    spec = _ga_branch_for_layer(layer_name)
    if spec is None or not ga_stats:
        return torch.ones((channels,), device=device, dtype=dtype)
    block_name, branch = spec
    block = ga_stats.get(block_name)
    if block is None:
        return torch.ones((channels,), device=device, dtype=dtype)
    key = "kappa_res" if branch == "residual" else "kappa_sc"
    kappa = block[key].to(device=device, dtype=dtype)
    if kappa.numel() != channels:
        return torch.ones((channels,), device=device, dtype=dtype)
    return kappa


def _gm_os_cfg(model) -> dict | None:
    cfg = getattr(model, "_gm_os_cfg", None)
    return cfg if isinstance(cfg, dict) and cfg else None


def _gm_os_enabled(model) -> bool:
    return _gm_os_cfg(model) is not None


def _channel_mean(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 4:
        return tensor.mean(dim=(0, 2, 3))
    if tensor.dim() == 2:
        return tensor.mean(dim=0)
    return tensor.reshape(-1)


def _gaussian_q(z: torch.Tensor) -> torch.Tensor:
    return _standard_normal_cdf(-z)


def _if_margin_crossing_prob(
    if_mod,
    quant_level: int,
    sigma_mode: str = "act",
    sigma_noise: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor | None:
    """Per-channel p = E[2 Q(d / (σ_eff+ε))] on QCFS-normalized activations."""
    z = getattr(if_mod, "last_pre_quant", None)
    if z is None:
        return None
    z = z.detach()
    lam_min = max(float(eps), 1e-3)
    lam = if_mod.thresh.detach().to(device=z.device, dtype=z.dtype).clamp(min=lam_min).view(-1)[0]
    r = float(quant_level) * z / (lam + eps)
    d_left, d_right, use_left, use_right = _activation_boundary_distances(
        r, quant_level, eps=eps
    )
    d_near = torch.where(
        use_left & use_right,
        torch.minimum(d_left, d_right),
        torch.where(use_left, d_left, d_right),
    )
    mode = str(sigma_mode).strip().lower()
    if mode == "act":
        if z.dim() == 4:
            std_z = z.flatten(2).std(dim=(0, 2), unbiased=False)
            view = (1, -1, 1, 1)
        elif z.dim() == 2:
            std_z = z.std(dim=0, unbiased=False)
            view = (1, -1)
        else:
            std_z = z.std().reshape(1)
            view = (1,)
        sigma_r = (float(quant_level) / (lam + eps)) * std_z
        sigma = torch.sqrt(sigma_r.pow(2) + float(sigma_noise) ** 2).view(*view)
    else:
        sigma = z.new_full((), max(float(sigma_noise), eps))
    zscore = (d_near / (sigma + eps)).clamp(min=0.0, max=20.0)
    p_map = (2.0 * _gaussian_q(zscore)).clamp(0.0, 1.0)
    return _channel_mean(p_map)


def _if_downstream_energy(grad: torch.Tensor | None) -> torch.Tensor | None:
    if grad is None:
        return None
    return _channel_mean(grad.detach().pow(2))


def _allocate_edge_mass(
    layer_name: str,
    channels: int,
    ga_stats: dict,
    device,
    dtype,
    eps: float,
) -> torch.Tensor:
    ones = torch.ones((channels,), device=device, dtype=dtype)
    spec = _ga_branch_for_layer(layer_name)
    if spec is None or not ga_stats:
        return ones
    block_name, branch = spec
    block = ga_stats.get(block_name)
    if block is None or not block.get("has_shortcut_conv", False):
        return ones
    pos_res = torch.relu(block["phi_res"]).to(device=device, dtype=dtype)
    pos_sc = torch.relu(block["phi_sc"]).to(device=device, dtype=dtype)
    denom = pos_res + pos_sc + float(eps)
    mass = pos_res / denom if branch == "residual" else pos_sc / denom
    if mass.numel() != channels:
        return ones
    return mass


def _ema_inplace(store: dict, key: str, value: torch.Tensor, rho: float) -> torch.Tensor:
    prev = store.get(key)
    if prev is None or prev.shape != value.shape or prev.device != value.device:
        store[key] = value.clone()
    else:
        store[key] = (1.0 - float(rho)) * prev + float(rho) * value
    return store[key]


def clear_if_pre_quant(model) -> None:
    for module in model.modules():
        if isinstance(module, IF):
            module.last_pre_quant = None


def update_graph_margin_stats(model, ce_loss) -> dict:
    """Refresh detached node/edge risks from the current CE graph.

    r_n = p_n D_n,  r_e = r_n m_{e→n}. Risks are EMA-smoothed and never
    backpropagated into p, D, or m.
    """
    cfg = _gm_os_cfg(model)
    if cfg is None:
        return {}
    eps = float(cfg.get("eps", 1e-6))
    quant_level = int(cfg.get("quant_level", 16))
    rho = float(cfg.get("ema_rho", 0.1))
    use_p = bool(cfg.get("use_p", True))
    use_d = bool(cfg.get("use_d", False))
    use_m = bool(cfg.get("use_m", False))
    module_map = dict(model.named_modules())
    p_by_if = {}
    d_by_if = {}
    if_entries = []
    for name, module in module_map.items():
        if not isinstance(module, IF):
            continue
        z = getattr(module, "last_pre_quant", None)
        if z is None:
            continue
        if_entries.append((name, module, z))
        if use_p:
            p_c = _if_margin_crossing_prob(
                module,
                quant_level=quant_level,
                sigma_mode=str(cfg.get("sigma_mode", "act")),
                sigma_noise=float(cfg.get("sigma_noise", 1.0)),
                eps=eps,
            )
            if p_c is not None:
                p_by_if[name] = p_c
    if use_d and torch.is_tensor(ce_loss) and bool(ce_loss.requires_grad):
        acts = []
        names = []
        for name, module, z in if_entries:
            if z.requires_grad:
                names.append(name)
                acts.append(z)
        if acts:
            grads = torch.autograd.grad(
                ce_loss, acts, retain_graph=True, allow_unused=True
            )
            for name, grad in zip(names, grads):
                energy = _if_downstream_energy(grad)
                if energy is not None:
                    d_by_if[name] = energy
    ga_stats = {}
    if use_m:
        ga_stats = estimate_ga_branch_stats(
            model,
            include_cov=bool(cfg.get("include_cov", False)),
            probe_sigma=float(cfg.get("probe_sigma", 0.1)),
            eps=eps,
        )
        model._ga_mne_block_stats = ga_stats

    rows = collect_weight_layer_matches(model)
    edge_store = getattr(model, "_gm_os_edge_ema", None)
    if not isinstance(edge_store, dict):
        edge_store = {}
        model._gm_os_edge_ema = edge_store
    edge_risk = {}
    p_vals, d_vals, m_vals = [], [], []
    for row in rows:
        if not row["matched"]:
            continue
        weight = row["weight"]
        channels = int(weight.shape[0])
        if_name = row["if_name"]
        device, dtype = weight.device, weight.dtype
        p_c = p_by_if.get(if_name)
        if p_c is None or p_c.numel() != channels:
            p_c = torch.ones((channels,), device=device, dtype=dtype)
        else:
            p_c = p_c.to(device=device, dtype=dtype)
        d_c = d_by_if.get(if_name)
        if d_c is None or d_c.numel() != channels:
            d_c = torch.ones((channels,), device=device, dtype=dtype)
            d_raw = d_c
        else:
            d_c = d_c.to(device=device, dtype=dtype)
            d_raw = d_c
            scale = d_c.mean().clamp(min=eps)
            d_c = torch.ones_like(d_c) if float(d_c.max()) <= eps else d_c / scale
        r_n = p_c * d_c
        m_c = _allocate_edge_mass(
            row["name"], channels, ga_stats, device, dtype, eps
        ) if use_m else torch.ones((channels,), device=device, dtype=dtype)
        r_e = (r_n * m_c).detach()
        edge_risk[row["name"]] = _ema_inplace(edge_store, row["name"], r_e, rho)
        p_vals.append(p_c.detach())
        d_vals.append(d_raw.detach())
        m_vals.append(m_c.detach())
    model._gm_os_edge_risk = edge_risk
    stats = {
        "gm_n_edges": len(edge_risk),
        "gm_p_mean": float(torch.cat(p_vals).mean()) if p_vals else 1.0,
        "gm_d_mean": float(torch.cat(d_vals).mean()) if d_vals else 1.0,
        "gm_m_mean": float(torch.cat(m_vals).mean()) if m_vals else 1.0,
        "gm_r_mean": float(torch.cat(list(edge_risk.values())).mean()) if edge_risk else 1.0,
        **_summarize_ga_stats(ga_stats),
    }
    model._gm_os_stats = stats
    return stats


def _summarize_ga_stats(ga_stats: dict) -> dict:
    if not ga_stats:
        return {
            "ga_n_blocks": 0,
            "ga_kappa_res_mean": 1.0,
            "ga_kappa_sc_mean": 1.0,
            "ga_frac_phi_neg_res": 0.0,
            "ga_frac_phi_neg_sc": 0.0,
            "ga_cov_mean": 0.0,
        }
    kappa_res = torch.cat([row["kappa_res"].reshape(-1) for row in ga_stats.values()])
    kappa_sc = torch.cat([row["kappa_sc"].reshape(-1) for row in ga_stats.values()])
    phi_res = torch.cat([row["phi_res"].reshape(-1) for row in ga_stats.values()])
    phi_sc = torch.cat([row["phi_sc"].reshape(-1) for row in ga_stats.values()])
    cov = torch.cat([row["cov"].reshape(-1) for row in ga_stats.values()])
    return {
        "ga_n_blocks": len(ga_stats),
        "ga_kappa_res_mean": float(kappa_res.mean()),
        "ga_kappa_sc_mean": float(kappa_sc.mean()),
        "ga_frac_phi_neg_res": float((phi_res < 0).to(phi_res.dtype).mean()),
        "ga_frac_phi_neg_sc": float((phi_sc < 0).to(phi_sc.dtype).mean()),
        "ga_cov_mean": float(cov.mean()),
    }


def _raw_channel_risk(row: dict, quant_level: int, eps: float = 1e-6, fold_bn: bool = True):
    weight = row["weight"]
    if_mod = row["if_mod"]
    if if_mod is None:
        return None
    lam_min = max(float(eps), 1e-3)
    lam = if_mod.thresh.detach().to(
        device=weight.device, dtype=weight.dtype
    ).clamp(min=lam_min).view(-1)[0]
    risk = torch.ones(
        (weight.shape[0],), device=weight.device, dtype=weight.dtype
    ) * (float(quant_level) ** 2 / (lam.pow(2) + eps))
    bn_mod = row["bn"]
    if fold_bn and bn_mod is not None:
        bn_eps = float(getattr(bn_mod, "eps", eps))
        gamma = bn_mod.weight.detach().to(device=weight.device, dtype=weight.dtype)
        var = bn_mod.running_var.detach().to(
            device=weight.device, dtype=weight.dtype
        ).clamp(min=bn_eps)
        risk = risk * gamma.pow(2) / (var + bn_eps)
    return risk


def calibrated_q_by_layer(
    model,
    quant_level: int,
    alpha: float,
    tau: float = 0.5,
    risk_min: float = 0.5,
    risk_max: float = 8.0,
    fold_bn: bool = True,
    onesided: bool = True,
    q_assignment: str = "risk",
    layer_map=None,
    eps: float = 1e-6,
    mean_normalize_q: bool = False,
    ga_mode: str = "off",
    ga_probe_sigma: float = 0.1,
) -> list[dict]:
    """Per-layer q / risk table. Unmatched layers keep q=1 and empty risk."""
    rows = collect_weight_layer_matches(model, layer_map=layer_map)
    matched = [row for row in rows if row["matched"]]
    ga_mode = str(ga_mode or getattr(model, "_ga_mne_mode", "off") or "off").strip().lower()
    ga_stats = {}
    if ga_mode in ("nocov", "full"):
        ga_stats = estimate_ga_branch_stats(
            model,
            include_cov=(ga_mode == "full"),
            probe_sigma=float(
                getattr(model, "_ga_mne_probe_sigma", ga_probe_sigma)
            ),
            eps=eps,
        )
        model._ga_mne_block_stats = ga_stats
    risks = []
    for row in matched:
        risk = _raw_channel_risk(row, quant_level, eps=eps, fold_bn=fold_bn)
        if ga_stats:
            kappa = _kappa_for_layer(
                row["name"], ga_stats, risk.numel(), risk.device, risk.dtype
            )
            row["kappa_mean"] = float(kappa.mean().detach())
            risk = risk * kappa
        else:
            row["kappa_mean"] = 1.0
        row["_risk"] = risk
        risks.append(risk)
        row["n_per_output"] = int(row["weight"][0].numel())

    if matched:
        total_covered = sum(row["n_per_output"] * row["_risk"].numel() for row in matched)
        global_mean = sum(
            row["n_per_output"] * row["_risk"].sum() for row in matched
        ) / float(total_covered)
        global_mean = global_mean.clamp(min=eps)
        clipped = [
            (row["_risk"] / global_mean).clamp(min=risk_min, max=risk_max)
            for row in matched
        ]
        clipped_mean = sum(
            row["n_per_output"] * c.sum() for row, c in zip(matched, clipped)
        ) / float(total_covered)
        clipped_mean = clipped_mean.clamp(min=eps)
        normalized = [c / clipped_mean for c in clipped]
        if onesided:
            true_q = [
                1.0 + float(alpha) * torch.relu(normed - float(tau))
                for normed in normalized
            ]
        else:
            true_q = [
                (1.0 - float(alpha)) + float(alpha) * normed
                for normed in normalized
            ]
        q_os_mean = sum(
            row["n_per_output"] * q.sum() for row, q in zip(matched, true_q)
        ) / float(total_covered)
        if q_assignment == "identity":
            q_values = [torch.ones_like(q) for q in true_q]
        elif q_assignment == "strength":
            q_values = [torch.full_like(q, float(q_os_mean.detach())) for q in true_q]
        elif q_assignment == "layer_mean":
            q_values = [torch.full_like(q, float(q.mean())) for q in true_q]
        else:
            q_values = true_q
        q_mean_pre_norm = sum(
            row["n_per_output"] * q.sum() for row, q in zip(matched, q_values)
        ) / float(total_covered)
        if mean_normalize_q:
            q_bar = q_mean_pre_norm.clamp(min=eps).detach()
            q_values = [q / q_bar for q in q_values]
        for row, normed, q, q_true in zip(matched, normalized, q_values, true_q):
            row["risk_mean"] = float(normed.mean().detach())
            row["risk_max"] = float(normed.max().detach())
            row["p_gt_tau"] = float((normed > float(tau)).float().mean().detach())
            row["q_mean"] = float(q.mean().detach())
            row["q_max"] = float(q.max().detach())
            row["q_os_mean"] = float(q_true.mean().detach())
            row["q_mean_pre_norm"] = float(q_mean_pre_norm.detach())
            row.pop("_risk", None)
            row.pop("n_per_output", None)

    for row in rows:
        row.setdefault("risk_mean", float("nan"))
        row.setdefault("risk_max", float("nan"))
        row.setdefault("p_gt_tau", float("nan"))
        row.setdefault("q_mean", 1.0)
        row.setdefault("q_max", 1.0)
        row.setdefault("q_os_mean", 1.0)
        row.setdefault("q_mean_pre_norm", row.get("q_mean", 1.0))
        row.setdefault("kappa_mean", 1.0)
        for key in ("weight", "bn", "if_mod"):
            row.pop(key, None)
    return rows


def terminal_residual_shortcut_risks(layer_rows: list[dict]) -> list[dict]:
    """Pair residual_terminal vs shortcut that share the same post-add IF."""
    by_if = {}
    for row in layer_rows:
        if not row.get("matched") or not row.get("if_name"):
            continue
        if row["role"] not in ("residual_terminal", "shortcut"):
            continue
        by_if.setdefault(row["if_name"], {})[row["role"]] = row
    pairs = []
    for if_name, group in sorted(by_if.items()):
        terminal = group.get("residual_terminal")
        shortcut = group.get("shortcut")
        if terminal is None:
            continue
        pairs.append(
            {
                "if_name": if_name,
                "residual_name": terminal["name"],
                "residual_risk_mean": terminal.get("risk_mean"),
                "residual_q_mean": terminal.get("q_mean"),
                "residual_q_max": terminal.get("q_max"),
                "has_shortcut_conv": shortcut is not None,
                "shortcut_name": "" if shortcut is None else shortcut["name"],
                "shortcut_risk_mean": (
                    None if shortcut is None else shortcut.get("risk_mean")
                ),
                "shortcut_q_mean": (
                    None if shortcut is None else shortcut.get("q_mean")
                ),
                "shortcut_q_max": (
                    None if shortcut is None else shortcut.get("q_max")
                ),
                "residual_kappa_mean": terminal.get("kappa_mean"),
                "shortcut_kappa_mean": (
                    None if shortcut is None else shortcut.get("kappa_mean")
                ),
            }
        )
    return pairs


def ce_vs_reg_grad_ratio(
    model,
    images,
    labels,
    criterion,
    reg_loss_fn,
    T: int,
    quant_level,
    reg_coeff: float,
) -> dict:
    """||∇R|| / ||∇CE|| on all trainable parameters, one batch."""
    was_training = model.training
    model.train()
    device = next(model.parameters()).device
    images = images.to(device)
    labels = labels.to(device)
    ga_on = str(getattr(model, "_ga_mne_mode", "") or "") in ("nocov", "full")
    gm_cfg = _gm_os_cfg(model)
    if gm_cfg and gm_cfg.get("use_m"):
        ga_on = True
    if ga_on:
        set_basicblock_ga_cache(model, True)
    if T > 0:
        outputs = model(images).mean(0)
    else:
        outputs = model(images)
    ce = criterion(outputs, labels)
    if gm_cfg is not None:
        update_graph_margin_stats(model, ce)
    params = [p for p in model.parameters() if p.requires_grad]
    grads_ce = torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True)
    ce_sq = 0.0
    for grad in grads_ce:
        if grad is not None:
            ce_sq += float(grad.detach().pow(2).sum().item())
    ce_norm = ce_sq ** 0.5
    if reg_loss_fn is None:
        model.zero_grad(set_to_none=True)
        if ga_on:
            set_basicblock_ga_cache(model, False)
        if not was_training:
            model.eval()
        return {
            "ce_grad_norm": ce_norm,
            "reg_grad_norm": 0.0,
            "reg_coeff_grad_norm": 0.0,
            "reg_ce_ratio": float("nan"),
            "ce_loss": float(ce.detach()),
            "reg_loss": 0.0,
        }
    reg = reg_loss_fn(model, T, quant_level)
    grads_reg = torch.autograd.grad(reg, params, allow_unused=True)
    reg_sq = 0.0
    for grad in grads_reg:
        if grad is not None:
            reg_sq += float(grad.detach().pow(2).sum().item())
    reg_norm = reg_sq ** 0.5
    model.zero_grad(set_to_none=True)
    if ga_on:
        set_basicblock_ga_cache(model, False)
    if not was_training:
        model.eval()
    ratio = float("nan") if ce_norm <= 0.0 else (reg_norm * float(reg_coeff)) / ce_norm
    return {
        "ce_grad_norm": ce_norm,
        "reg_grad_norm": reg_norm,
        "reg_coeff_grad_norm": reg_norm * float(reg_coeff),
        "reg_ce_ratio": ratio,
        "ce_loss": float(ce.detach()),
        "reg_loss": float(reg.detach()) if torch.is_tensor(reg) else float(reg),
    }


def dump_mne_mapping_report(
    model,
    out_dir,
    layer_map=None,
    quant_level: int = 16,
    alpha: float = 4.0,
    tau: float = 0.5,
    risk_min: float = 0.5,
    risk_max: float = 8.0,
    q_assignment: str = "risk",
    onesided: bool = True,
    mean_normalize_q: bool = False,
    ga_mode: str = "off",
    ga_probe_sigma: float = 0.1,
    extra=None,
) -> dict:
    """Write mapping_summary.json, layer_q.csv, residual_shortcut_risk.csv."""
    import csv
    import json
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    layer_map = _layer_map_from_model(model, layer_map)
    model._mne_layer_map = layer_map
    match_rows = collect_weight_layer_matches(model, layer_map=layer_map)
    summary = summarize_weight_layer_matches(match_rows)
    q_rows = calibrated_q_by_layer(
        model,
        quant_level=quant_level,
        alpha=alpha,
        tau=tau,
        risk_min=risk_min,
        risk_max=risk_max,
        onesided=onesided,
        q_assignment=q_assignment,
        layer_map=layer_map,
        mean_normalize_q=mean_normalize_q,
        ga_mode=ga_mode,
        ga_probe_sigma=ga_probe_sigma,
    )
    pairs = terminal_residual_shortcut_risks(q_rows)
    with torch.no_grad():
        identity_pen = compute_l2_calibrated_mne_regularization(
            model,
            quant_level=quant_level,
            alpha=alpha,
            risk_min=risk_min,
            risk_max=risk_max,
            onesided=onesided,
            tau=tau,
            q_assignment="identity",
        )
        l2_pen = None
        for row in match_rows:
            term = 0.5 * row["weight"].pow(2).sum()
            l2_pen = term if l2_pen is None else l2_pen + term
    identity_vs_l2 = float("nan")
    if l2_pen is not None and float(l2_pen) > 0:
        identity_vs_l2 = float((identity_pen - l2_pen).abs() / l2_pen)
    payload = {
        "layer_map": layer_map,
        "quant_level": int(quant_level),
        "alpha": float(alpha),
        "tau": float(tau),
        "q_assignment": q_assignment,
        "mean_normalize_q": bool(mean_normalize_q),
        "ga_mode": str(ga_mode or "off"),
        "identity_vs_l2wo_relerr": identity_vs_l2,
        **summary,
        "p_gt_tau": (
            sum(
                row["out_channels"] * row["p_gt_tau"]
                for row in q_rows
                if row["matched"] and row["p_gt_tau"] == row["p_gt_tau"]
            )
            / max(1, sum(row["out_channels"] for row in q_rows if row["matched"]))
        ),
        "q_mean": (
            sum(
                row["n_params"] * row["q_mean"]
                for row in q_rows
                if row["matched"]
            )
            / max(1, sum(row["n_params"] for row in q_rows if row["matched"]))
        ),
        "q_mean_pre_norm": next(
            (
                row.get("q_mean_pre_norm", row["q_mean"])
                for row in q_rows
                if row["matched"]
            ),
            1.0,
        ),
        "residual_shortcut_pairs": pairs,
    }
    if extra:
        payload.update(extra)
    (out / "mapping_summary.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    csv_fields = [
        "name",
        "role",
        "is_head",
        "matched",
        "n_params",
        "out_channels",
        "bn_name",
        "if_name",
        "risk_mean",
        "risk_max",
        "p_gt_tau",
        "q_mean",
        "q_max",
        "q_os_mean",
        "q_mean_pre_norm",
        "kappa_mean",
    ]
    with (out / "layer_q.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in q_rows:
            writer.writerow({key: row.get(key, "") for key in csv_fields})
    if pairs:
        with (out / "residual_shortcut_risk.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pairs[0].keys()))
            writer.writeheader()
            writer.writerows(pairs)
    ga_stats = getattr(model, "_ga_mne_block_stats", None) or {}
    if ga_stats:
        ga_rows = []
        for block_name, block in sorted(ga_stats.items()):
            ga_rows.append(
                {
                    "block": block_name,
                    "has_shortcut_conv": int(bool(block["has_shortcut_conv"])),
                    "kappa_res_mean": float(block["kappa_res"].mean()),
                    "kappa_sc_mean": float(block["kappa_sc"].mean()),
                    "phi_res_mean": float(block["phi_res"].mean()),
                    "phi_sc_mean": float(block["phi_sc"].mean()),
                    "cov_mean": float(block["cov"].mean()),
                    "frac_phi_neg_res": float(
                        (block["phi_res"] < 0).to(block["phi_res"].dtype).mean()
                    ),
                    "frac_phi_neg_sc": float(
                        (block["phi_sc"] < 0).to(block["phi_sc"].dtype).mean()
                    ),
                    "delta_u_rms_mean": float(block["delta_u_rms"].mean()),
                }
            )
        with (out / "ga_block_stats.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ga_rows[0].keys()))
            writer.writeheader()
            writer.writerows(ga_rows)
        payload.update(_summarize_ga_stats(ga_stats))
        (out / "mapping_summary.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n"
        )
    return payload


def compute_l1_regularization(model, T=None, quant_level=None):
    """
    Standard L1 weight penalty: sum_i |W_i| over Conv/Linear weights (bias excluded).
    """
    reg = None
    for layer in model.modules():
        if not isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            continue
        if getattr(layer, "weight", None) is None:
            continue
        term = layer.weight.abs().sum()
        reg = term if reg is None else (reg + term)
    if reg is None:
        p = next(model.parameters(), None)
        if p is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=p.device, dtype=p.dtype)
    return reg


def compute_l1_all_regularization(model, T=None, quant_level=None):
    """Explicit L1 penalty over every trainable parameter."""
    terms = [parameter.abs().sum() for parameter in model.parameters() if parameter.requires_grad]
    if not terms:
        p = next(model.parameters(), None)
        if p is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=p.device, dtype=p.dtype)
    return torch.stack([term.reshape(()) for term in terms]).sum()


def compute_manual_l2_regularization(model, T=None, quant_level=None):
    """
    Explicit L2 penalty over Conv/Linear weights (bias excluded):

      R = sum_l ||W_l||_F^2

    Unlike optimizer weight decay, this value is added directly to the loss.
    """
    reg = None
    for layer in model.modules():
        if not isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            continue
        if getattr(layer, "weight", None) is None:
            continue
        term = layer.weight.pow(2).sum()
        reg = term if reg is None else (reg + term)
    if reg is None:
        p = next(model.parameters(), None)
        if p is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=p.device, dtype=p.dtype)
    return reg


def compute_manual_l2_all_regularization(model, T=None, quant_level=None):
    """
    Explicit L2 penalty over every trainable parameter:

      R = sum_p ||p||_2^2

    With ``reg_coeff = weight_decay / 2``, this has the same gradient as
    coupled optimizer weight decay over the same parameters.
    """
    reg = None
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        term = parameter.pow(2).sum()
        reg = term if reg is None else (reg + term)
    if reg is None:
        return torch.tensor(0.0)
    return reg


def compute_selective_l2_regularization(
    model,
    T=None,
    quant_level=None,
    *,
    include_bn: bool = False,
    include_bn_weight: bool = False,
    include_bn_bias: bool = False,
    include_if: bool = False,
):
    """L2 over Conv/Linear weights plus selected scale-parameter families."""
    parameters = []
    seen = set()

    def _append(parameter):
        if parameter is None or not parameter.requires_grad or id(parameter) in seen:
            return
        seen.add(id(parameter))
        parameters.append(parameter)

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            _append(getattr(module, "weight", None))
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            if include_bn or include_bn_weight:
                _append(module.weight)
            if include_bn or include_bn_bias:
                _append(module.bias)
        elif include_if and isinstance(module, IF):
            _append(module.thresh)

    if not parameters:
        p = next(model.parameters(), None)
        if p is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=p.device, dtype=p.dtype)
    return torch.stack([parameter.pow(2).sum().reshape(()) for parameter in parameters]).sum()


def compute_elastic_net_all_regularization(
    model,
    T=None,
    quant_level=None,
    l1_ratio: float = 0.04,
):
    """
    Elastic Net over every trainable parameter:

      R = sum_p ||p||_2^2 + l1_ratio * sum_p ||p||_1

    The global ``reg_coeff`` controls the L2 coefficient; its product with
    ``l1_ratio`` is the effective L1 coefficient.
    """
    if l1_ratio < 0:
        raise ValueError(f"l1_ratio must be non-negative, got {l1_ratio}.")
    l2 = compute_manual_l2_all_regularization(model, T=T, quant_level=quant_level)
    l1 = compute_l1_all_regularization(model, T=T, quant_level=quant_level)
    return l2 + float(l1_ratio) * l1


def compute_scale_l2_regularization(model, T=None, quant_level=None):
    """
    Mechanism-only L2 diagnostic over BN affine parameters and IF thresholds.

    This isolates the scale parameters that all-parameter weight decay reaches
    in addition to Conv/Linear weights. It is not intended as a general-purpose
    regularization baseline.
    """
    terms = []
    for module in model.modules():
        if isinstance(module, IF):
            if module.thresh.requires_grad:
                terms.append(module.thresh.pow(2).sum())
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            for parameter in (module.weight, module.bias):
                if parameter is not None and parameter.requires_grad:
                    terms.append(parameter.pow(2).sum())
    if not terms:
        p = next(model.parameters(), None)
        if p is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=p.device, dtype=p.dtype)
    return torch.stack([term.reshape(()) for term in terms]).sum()


def compute_group_lasso_regularization(model, T=None, quant_level=None, eps: float = 1e-12):
    """
    Filter-wise group Lasso for CNNs:

      R = sum_conv sum_output_filter ||W_filter||_2

    Each output filter (the first weight dimension) is one group. Linear layers
    are intentionally excluded because this baseline targets convolutional
    filter/channel sparsity.
    """
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}.")

    reg = None
    for layer in model.modules():
        if not isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            continue
        if getattr(layer, "weight", None) is None:
            continue
        filters = layer.weight.reshape(layer.weight.shape[0], -1)
        term = torch.sqrt((filters * filters).sum(dim=1) + eps).sum()
        reg = term if reg is None else (reg + term)
    if reg is None:
        p = next(model.parameters(), None)
        if p is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=p.device, dtype=p.dtype)
    return reg


def compute_spectral_norm_regularization(
    model,
    T=None,
    quant_level=None,
    power_iters: int = 3,
    eps: float = 1e-12,
):
    """
    Sum of approximate largest singular values over Conv/Linear weights.

      R = sum_l sigma_max(W_l)

    Convolution kernels are flattened to [out_channels, fan_in]. Singular
    vectors are estimated with detached power iteration, while the final
    Rayleigh quotient remains differentiable with respect to the weights.
    This avoids a full SVD for every layer and training batch.
    """
    if power_iters < 1:
        raise ValueError(f"power_iters must be >= 1, got {power_iters}.")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}.")

    reg = None
    for layer in model.modules():
        if not isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            continue
        if getattr(layer, "weight", None) is None:
            continue

        matrix = layer.weight.reshape(layer.weight.shape[0], -1)
        detached = matrix.detach()
        # A deterministic start keeps runs reproducible without storing extra
        # optimizer state. Alternating signs reduce accidental orthogonality.
        v = torch.ones(
            detached.shape[1], device=detached.device, dtype=detached.dtype
        )
        v[1::2] = -1
        v = v / (torch.linalg.vector_norm(v) + eps)
        with torch.no_grad():
            for _ in range(power_iters):
                u = detached.mv(v)
                u = u / (torch.linalg.vector_norm(u) + eps)
                v = detached.t().mv(u)
                v = v / (torch.linalg.vector_norm(v) + eps)

        sigma = torch.dot(u, matrix.mv(v)).abs()
        reg = sigma if reg is None else (reg + sigma)
    if reg is None:
        p = next(model.parameters(), None)
        if p is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=p.device, dtype=p.dtype)
    return reg


def compute_orthogonal_regularization(model, T=None, quant_level=None):
    """
    Soft orthogonal regularization over Conv/Linear weights.

      W = reshape(weight, [out_features, fan_in])
      R = sum_l ||G_l - I||_F^2

    where G_l is the smaller feasible Gram matrix:
      - W W^T when out_features <= fan_in
      - W^T W otherwise

    This keeps the target feasible for both tall and wide layers and avoids
    constructing the much larger infeasible Gram matrix.
    """
    reg = None
    for layer in model.modules():
        if not isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            continue
        if getattr(layer, "weight", None) is None:
            continue

        matrix = layer.weight.reshape(layer.weight.shape[0], -1)
        if matrix.shape[0] <= matrix.shape[1]:
            gram = matrix @ matrix.t()
            eye = torch.eye(
                matrix.shape[0], device=matrix.device, dtype=matrix.dtype
            )
        else:
            gram = matrix.t() @ matrix
            eye = torch.eye(
                matrix.shape[1], device=matrix.device, dtype=matrix.dtype
            )
        term = (gram - eye).pow(2).sum()
        reg = term if reg is None else (reg + term)

    if reg is None:
        p = next(model.parameters(), None)
        if p is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=p.device, dtype=p.dtype)
    return reg


def compute_effective_l2_regularization(
    model,
    T=None,
    quant_level=None,
    eps: float = 1e-6,
    detach_bn_stats: bool = True,
    detach_bn_affine: bool = True,
    normalize_by_fan_in: bool = True,
    layer_reduction: str = "mean",
):
    """
    BN-folded Effective-L2:

      W_eff = gamma / sqrt(var + eps) * W
      R = mean_l mean_o ||W_eff,l,o||_2^2

    This is a decomposed L2-family baseline that keeps BN folding from MNE-L2
    but removes threshold normalization and quantization-level scaling.
    """
    if layer_reduction not in ("sum", "mean"):
        raise ValueError(
            f"Unsupported layer_reduction={layer_reduction!r}; expected 'sum' or 'mean'."
        )

    module_map = dict(model.named_modules())
    terms = []

    for lname, layer in model.named_modules():
        if not isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            continue
        if getattr(layer, "weight", None) is None:
            continue

        w = layer.weight
        w_eff = w
        bn_mod, _ = _resolve_bn_if_for_layer(
            lname, module_map, layer_map=_layer_map_from_model(model)
        )
        if bn_mod is not None:
            bn_eps = float(getattr(bn_mod, "eps", eps))
            gamma = bn_mod.weight.to(device=w.device, dtype=w.dtype)
            var = bn_mod.running_var.to(device=w.device, dtype=w.dtype)
            if detach_bn_stats:
                var = var.detach()
            if detach_bn_affine:
                gamma = gamma.detach()
            var = var.clamp(min=bn_eps)
            scale = gamma / torch.sqrt(var + bn_eps)
            view_shape = [scale.shape[0]] + [1] * (w.dim() - 1)
            w_eff = w * scale.view(*view_shape)

        w_flat = w_eff.reshape(w_eff.shape[0], -1)
        term = (w_flat * w_flat).sum(dim=1).mean()
        if normalize_by_fan_in:
            term = term / max(1, w_flat.shape[1])
        terms.append(term)

    if not terms:
        p = next(model.parameters(), None)
        if p is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=p.device, dtype=p.dtype)
    stacked = torch.stack([term.reshape(()) for term in terms])
    return stacked.mean() if layer_reduction == "mean" else stacked.sum()


def compute_threshold_normalized_l2_regularization(
    model,
    T=None,
    quant_level=None,
    eps: float = 1e-6,
    use_max: bool = False,
    detach_lambda: bool = True,
    normalize_by_fan_in: bool = True,
    layer_reduction: str = "mean",
):
    """
    Threshold-normalized L2:

      R = mean_l M_raw,l / (lambda_l^2 + eps)

    where M_raw,l is the raw per-layer weight energy (without BN folding). This
    isolates the IF-threshold normalization effect from full MNE-L2.
    """
    if layer_reduction not in ("sum", "mean"):
        raise ValueError(
            f"Unsupported layer_reduction={layer_reduction!r}; expected 'sum' or 'mean'."
        )

    module_map = dict(model.named_modules())
    terms = []

    for lname, layer in model.named_modules():
        if not isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            continue
        if getattr(layer, "weight", None) is None:
            continue

        _, if_mod = _resolve_bn_if_for_layer(
            lname, module_map, layer_map=_layer_map_from_model(model)
        )
        if if_mod is None:
            continue

        w_flat = layer.weight.reshape(layer.weight.shape[0], -1)
        per_out_norm_sq = (w_flat * w_flat).sum(dim=1)
        m_raw = per_out_norm_sq.max() if use_max else per_out_norm_sq.mean()
        if normalize_by_fan_in:
            m_raw = m_raw / max(1, w_flat.shape[1])

        lam_min = max(eps, 1e-3)
        lam = if_mod.thresh.to(device=w_flat.device, dtype=w_flat.dtype).clamp(min=lam_min).view(-1)[0]
        if detach_lambda:
            lam = lam.detach()

        terms.append(m_raw / (lam.pow(2) + eps))

    if not terms:
        p = next(model.parameters(), None)
        if p is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=p.device, dtype=p.dtype)
    stacked = torch.stack([term.reshape(()) for term in terms])
    return stacked.mean() if layer_reduction == "mean" else stacked.sum()


def compute_l2_sp_regularization(
    model,
    reference_weights: dict[str, torch.Tensor],
    T=None,
    quant_level=None,
    layer_reduction: str = "mean",
):
    """
    L2-SP baseline:

      R = mean_l ||W_l - W_l^(0)||_F^2

    The reference weights are captured at initialization time. Using per-layer
    means keeps the coefficient scale more stable across model sizes.
    """
    if layer_reduction not in ("sum", "mean"):
        raise ValueError(
            f"Unsupported layer_reduction={layer_reduction!r}; expected 'sum' or 'mean'."
        )

    terms = []
    for lname, layer in model.named_modules():
        if not isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            continue
        if getattr(layer, "weight", None) is None:
            continue

        key = f"{lname}.weight"
        ref = reference_weights.get(key)
        if ref is None:
            continue
        terms.append((layer.weight - ref).pow(2).mean())

    if not terms:
        p = next(model.parameters(), None)
        if p is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=p.device, dtype=p.dtype)
    stacked = torch.stack([term.reshape(()) for term in terms])
    return stacked.mean() if layer_reduction == "mean" else stacked.sum()


def _approx_largest_singular_value(matrix, power_iters: int = 3, eps: float = 1e-12):
    if power_iters < 1:
        raise ValueError(f"power_iters must be >= 1, got {power_iters}.")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}.")

    detached = matrix.detach()
    v = torch.ones(detached.shape[1], device=detached.device, dtype=detached.dtype)
    v[1::2] = -1
    v = v / (torch.linalg.vector_norm(v) + eps)
    with torch.no_grad():
        for _ in range(power_iters):
            u = detached.mv(v)
            u = u / (torch.linalg.vector_norm(u) + eps)
            v = detached.t().mv(u)
            v = v / (torch.linalg.vector_norm(v) + eps)
    return torch.dot(u, matrix.mv(v)).abs()


def compute_spectral_mne_regularization(
    model,
    quant_level: int,
    eps: float = 1e-6,
    power_iters: int = 3,
    detach_lambda: bool = True,
    detach_bn_stats: bool = True,
    detach_bn_affine: bool = True,
    layer_reduction: str = "sum",
):
    """
    Spectral-MNE: replace MNE-L2's average effective weight energy with the
    largest singular direction of the BN-folded effective weight matrix.

      R = sum_l (L^2 * sigma_max(W_eff,l)^2) / (lambda_l^2 + eps)

    Layers without a matched IF threshold are skipped, matching MNE-L2.
    """
    if layer_reduction not in ("sum", "mean"):
        raise ValueError(f"Unsupported layer_reduction={layer_reduction!r}; expected 'sum' or 'mean'.")

    module_map = dict(model.named_modules())
    terms = []

    for lname, layer in model.named_modules():
        if not isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            continue
        if getattr(layer, "weight", None) is None:
            continue

        w = layer.weight
        w_eff = w

        bn_mod, if_mod = _resolve_bn_if_for_layer(
            lname, module_map, layer_map=_layer_map_from_model(model)
        )
        if if_mod is None:
            continue
        if bn_mod is not None:
            bn_eps = float(getattr(bn_mod, "eps", eps))
            gamma = bn_mod.weight.to(device=w.device, dtype=w.dtype)
            var = bn_mod.running_var.to(device=w.device, dtype=w.dtype)
            if detach_bn_stats:
                var = var.detach()
            if detach_bn_affine:
                gamma = gamma.detach()
            var = var.clamp(min=bn_eps)
            scale = gamma / torch.sqrt(var + bn_eps)
            view_shape = [scale.shape[0]] + [1] * (w.dim() - 1)
            w_eff = w * scale.view(*view_shape)

        matrix = w_eff.reshape(w_eff.shape[0], -1)
        sigma = _approx_largest_singular_value(
            matrix, power_iters=power_iters, eps=max(eps, 1e-12)
        )

        lam_min = max(eps, 1e-3)
        lam = if_mod.thresh.to(device=w.device, dtype=w.dtype).clamp(min=lam_min).view(-1)[0]
        if detach_lambda:
            lam = lam.detach()

        terms.append((float(quant_level) ** 2) * sigma.pow(2) / (lam.pow(2) + eps))

    if not terms:
        p = next(model.parameters(), None)
        if p is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=p.device, dtype=p.dtype)
    stacked = torch.stack([term.reshape(()) for term in terms])
    return stacked.mean() if layer_reduction == "mean" else stacked.sum()


def compute_l2_calibrated_mne_regularization(
    model,
    quant_level: int,
    alpha: float = 0.25,
    eps: float = 1e-6,
    risk_min: float = 0.5,
    risk_max: float = 2.0,
    fold_bn: bool = True,
    normalization: str = "global",
    onesided: bool = False,
    tau: float = 1.0,
    q_assignment: str = "risk",
    shuffle_seed: int = 0,
    q_max: float = 0.0,
    mean_normalize_q: bool = False,
    ga_mode: str = "off",
    ga_probe_sigma: float = 0.1,
    risk_source: str = "bn",
    budget_norm: bool = False,
):
    """Weights-only L2 with a detached, mean-one MNE risk reweighting.

    The function returns half of the weighted squared norm, so using
    ``reg_coeff=weight_decay`` matches coupled optimizer weight decay when
    ``alpha=0``. Conv/Linear weights without a matched IF (for example the
    classifier head) retain a plain L2 coefficient of one.
    """
    if float(alpha) < 0.0:
        raise ValueError(f"alpha must be >= 0, got {alpha}.")
    if (not onesided) and float(alpha) > 1.0:
        raise ValueError(f"alpha must be in [0, 1] unless onesided, got {alpha}.")
    if risk_min <= 0 or risk_max < risk_min:
        raise ValueError(
            f"Expected 0 < risk_min <= risk_max, got {risk_min}, {risk_max}."
        )
    normalization = str(normalization).strip().lower()
    if normalization not in ("global", "layerwise"):
        raise ValueError(
            "normalization must be 'global' or 'layerwise', "
            f"got {normalization!r}."
        )
    q_assignment = str(q_assignment).strip().lower()
    if q_assignment not in (
        "risk",
        "identity",
        "strength",
        "shuffle",
        "layer_mean",
    ):
        raise ValueError(
            "q_assignment must be risk, identity, strength, shuffle, "
            f"or layer_mean, got {q_assignment!r}."
        )
    q_cap = float(q_max)
    use_q_cap = math.isfinite(q_cap) and q_cap > 0.0
    gm_cfg = _gm_os_cfg(model)
    risk_source = str(
        risk_source or (gm_cfg and "gm") or getattr(model, "_calibrated_mne_risk_source", "bn")
        or "bn"
    ).strip().lower()
    if gm_cfg is not None:
        risk_source = "gm"
        budget_norm = bool(gm_cfg.get("budget_norm", budget_norm))
    ga_mode = str(ga_mode or getattr(model, "_ga_mne_mode", "off") or "off").strip().lower()
    ga_stats = {}
    if risk_source != "gm" and ga_mode in ("nocov", "full"):
        ga_stats = estimate_ga_branch_stats(
            model,
            include_cov=(ga_mode == "full"),
            probe_sigma=float(
                getattr(model, "_ga_mne_probe_sigma", ga_probe_sigma)
            ),
            eps=eps,
        )
        model._ga_mne_block_stats = ga_stats

    weighted_layers = []
    plain_weights = []
    module_map = dict(model.named_modules())
    gm_risks = getattr(model, "_gm_os_edge_risk", {}) if risk_source == "gm" else {}

    for lname, layer in model.named_modules():
        if not isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            continue
        weight = getattr(layer, "weight", None)
        if weight is None or not weight.requires_grad:
            continue

        bn_mod, if_mod = _resolve_bn_if_for_layer(
            lname, module_map, layer_map=_layer_map_from_model(model)
        )
        if if_mod is None:
            plain_weights.append(weight)
            continue

        with torch.no_grad():
            if risk_source == "gm":
                cached = gm_risks.get(lname) if isinstance(gm_risks, dict) else None
                if cached is not None and cached.numel() == weight.shape[0]:
                    risk = cached.detach().to(device=weight.device, dtype=weight.dtype)
                else:
                    risk = torch.ones(
                        (weight.shape[0],), device=weight.device, dtype=weight.dtype
                    )
            else:
                lam_min = max(float(eps), 1e-3)
                lam = if_mod.thresh.detach().to(
                    device=weight.device, dtype=weight.dtype
                ).clamp(min=lam_min).view(-1)[0]
                base_risk = float(quant_level) ** 2 / (lam.pow(2) + eps)
                risk = torch.ones(
                    (weight.shape[0],), device=weight.device, dtype=weight.dtype
                ) * base_risk

                if fold_bn and bn_mod is not None:
                    bn_eps = float(getattr(bn_mod, "eps", eps))
                    gamma = bn_mod.weight.detach().to(
                        device=weight.device, dtype=weight.dtype
                    )
                    var = bn_mod.running_var.detach().to(
                        device=weight.device, dtype=weight.dtype
                    ).clamp(min=bn_eps)
                    risk = risk * gamma.pow(2) / (var + bn_eps)
                if ga_stats:
                    kappa = _kappa_for_layer(
                        lname, ga_stats, risk.numel(), risk.device, risk.dtype
                    )
                    risk = risk * kappa

        n_per_output = weight[0].numel()
        weighted_layers.append((weight, risk, n_per_output))

    all_weights = [entry[0] for entry in weighted_layers] + plain_weights
    if not all_weights:
        parameter = next(model.parameters(), None)
        if parameter is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=parameter.device, dtype=parameter.dtype)

    # Unmatched-head-only models keep a plain L2 penalty.
    if not weighted_layers:
        penalty = sum(weight.pow(2).sum() for weight in all_weights)
        model._calibrated_mne_stats = {
            "alpha": float(alpha),
            "normalization": normalization,
            "onesided": bool(onesided),
            "tau": float(tau),
            "risk_mean": 1.0,
            "risk_min": 1.0,
            "risk_max": 1.0,
            "q_assignment": q_assignment,
            "q_cap": q_cap if use_q_cap else 0.0,
            "q_mean": 1.0,
            "q_std": 0.0,
            "q_min": 1.0,
            "q_max": 1.0,
            "q_os_mean": 1.0,
            "q_os_std": 0.0,
            "p_gt_tau": 0.0,
            "p_at_qmax": 0.0,
            "layer_q_mean_min": 1.0,
            "layer_q_mean_max": 1.0,
            "mean_normalize_q": bool(mean_normalize_q),
            "q_mean_pre_norm": 1.0,
            "q_tilde_mean": 1.0,
        }
        return 0.5 * penalty

    with torch.no_grad():
        total_covered_params = sum(
            n_per_output * risk.numel()
            for _, risk, n_per_output in weighted_layers
        )
        global_risk_mean = sum(
            n_per_output * risk.sum()
            for _, risk, n_per_output in weighted_layers
        ) / float(total_covered_params)
        global_risk_mean = global_risk_mean.clamp(min=eps)

        if normalization == "global":
            clipped_risks = [
                (risk / global_risk_mean).clamp(min=risk_min, max=risk_max)
                for _, risk, _ in weighted_layers
            ]
            clipped_mean = sum(
                n_per_output * clipped.sum()
                for clipped, (_, _, n_per_output) in zip(
                    clipped_risks, weighted_layers
                )
            ) / float(total_covered_params)
            clipped_mean = clipped_mean.clamp(min=eps)
            normalized_risks = [
                clipped / clipped_mean for clipped in clipped_risks
            ]
        else:
            normalized_risks = []
            for _, risk, _ in weighted_layers:
                layer_mean = risk.mean().clamp(min=eps)
                clipped = (risk / layer_mean).clamp(
                    min=risk_min, max=risk_max
                )
                normalized_risks.append(
                    clipped / clipped.mean().clamp(min=eps)
                )

        if onesided:
            # q = 1 + α max(r̂ − τ, 0): strengthen high-risk channels only.
            excess = [
                torch.relu(normalized - float(tau))
                for normalized in normalized_risks
            ]
            if budget_norm:
                total_excess = sum(
                    n_per_output * extra.sum()
                    for extra, (_, _, n_per_output) in zip(excess, weighted_layers)
                ) / float(total_covered_params)
                total_excess = total_excess.clamp(min=eps)
                true_q = [
                    1.0 + float(alpha) * extra / total_excess
                    for extra in excess
                ]
            else:
                true_q = [
                    1.0 + float(alpha) * extra for extra in excess
                ]
        else:
            true_q = [
                (1.0 - float(alpha)) + float(alpha) * normalized
                for normalized in normalized_risks
            ]

        all_raw_q = torch.cat(true_q)
        if use_q_cap:
            p_at_qmax = (all_raw_q >= q_cap).to(all_raw_q.dtype).mean()
            true_q = [q.clamp(max=q_cap) for q in true_q]
        else:
            p_at_qmax = all_raw_q.new_zeros(())

        q_os_mean = sum(
            n_per_output * q.sum()
            for q, (_, _, n_per_output) in zip(true_q, weighted_layers)
        ) / float(total_covered_params)
        all_true_q = torch.cat(true_q)
        all_normalized = torch.cat(normalized_risks)
        p_gt_tau = (all_normalized > float(tau)).to(all_normalized.dtype).mean()

        if q_assignment == "identity":
            q_values = [torch.ones_like(q) for q in true_q]
        elif q_assignment == "strength":
            q_bar = q_os_mean.detach()
            q_values = [torch.full_like(q, float(q_bar)) for q in true_q]
        elif q_assignment == "shuffle":
            generator = torch.Generator()
            generator.manual_seed(int(shuffle_seed) & 0x7FFFFFFF)
            q_values = []
            for q in true_q:
                if q.numel() <= 1:
                    q_values.append(q)
                    continue
                perm = torch.randperm(q.numel(), generator=generator)
                q_values.append(q.reshape(-1)[perm.to(q.device)].reshape_as(q))
        elif q_assignment == "layer_mean":
            # q_l = (1/C_l) Σ_c q_{l,c}^{OS}; every channel in the layer shares q_l.
            q_values = [
                torch.full_like(q, float(q.mean())) for q in true_q
            ]
        else:
            q_values = true_q

        q_mean_pre_norm = sum(
            n_per_output * q.sum()
            for q, (_, _, n_per_output) in zip(q_values, weighted_layers)
        ) / float(total_covered_params)
        if mean_normalize_q:
            q_bar = q_mean_pre_norm.clamp(min=eps).detach()
            q_values = [q / q_bar for q in q_values]
        q_mean = sum(
            n_per_output * q.sum()
            for q, (_, _, n_per_output) in zip(q_values, weighted_layers)
        ) / float(total_covered_params)
        all_q = torch.cat(q_values)
        layer_q_means = torch.stack([q.mean() for q in q_values])
        model._calibrated_mne_stats = {
            "alpha": float(alpha),
            "normalization": normalization,
            "onesided": bool(onesided),
            "tau": float(tau),
            "q_assignment": q_assignment,
            "q_cap": q_cap if use_q_cap else 0.0,
            "risk_mean": global_risk_mean.detach(),
            "risk_min": all_normalized.min().detach(),
            "risk_max": all_normalized.max().detach(),
            "q_mean": q_mean.detach(),
            "q_std": all_q.std(unbiased=False).detach(),
            "q_min": all_q.min().detach(),
            "q_max": all_q.max().detach(),
            "q_os_mean": q_os_mean.detach(),
            "q_os_std": all_true_q.std(unbiased=False).detach(),
            "p_gt_tau": p_gt_tau.detach(),
            "p_at_qmax": p_at_qmax.detach(),
            "layer_q_mean_min": layer_q_means.min().detach(),
            "layer_q_mean_max": layer_q_means.max().detach(),
            "mean_normalize_q": bool(mean_normalize_q),
            "q_mean_pre_norm": q_mean_pre_norm.detach(),
            "q_tilde_mean": q_mean.detach(),
            **_summarize_ga_stats(ga_stats),
            "ga_mode": ga_mode,
            "risk_source": risk_source,
            "budget_norm": bool(budget_norm),
            **(getattr(model, "_gm_os_stats", {}) or {}),
        }

    penalty = None
    for q, (weight, _, _) in zip(q_values, weighted_layers):
        flat = weight.reshape(weight.shape[0], -1)
        term = (q.view(-1, 1) * flat.pow(2)).sum()
        penalty = term if penalty is None else penalty + term
    for weight in plain_weights:
        term = weight.pow(2).sum()
        penalty = term if penalty is None else penalty + term
    return 0.5 * penalty


def _mne_l2_penalty(
    model,
    quant_level: int,
    eps: float,
    use_max: bool,
    detach_lambda: bool,
    detach_bn_stats: bool,
    detach_bn_affine,
    normalize_by_fan_in: bool,
    layer_reduction: str,
    l_ref,
    fold_bn: bool,
    full_frobenius: bool,
    layer_map: str,
    include_roles=None,
    divide_by_lambda: bool = True,
    scale_by_l: bool = True,
):
    module_map = dict(model.named_modules())
    terms = []
    if not scale_by_l:
        level_factor = 1.0
    elif l_ref is not None:
        level_factor = (float(quant_level) / float(l_ref)) ** 2
    else:
        level_factor = float(quant_level) ** 2

    for lname, layer in model.named_modules():
        if not isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            continue
        if getattr(layer, "weight", None) is None:
            continue

        w = layer.weight
        w_eff = w

        bn_mod, if_mod = _resolve_bn_if_for_layer(
            lname, module_map, layer_map=layer_map
        )
        # 方案 C：无匹配 IF 的层（如 VGG classifier.7 输出头）不参与 MNE-L2。
        if if_mod is None:
            continue
        if include_roles is not None and _weight_layer_role(lname) not in include_roles:
            continue
        if fold_bn and bn_mod is not None:
            bn_eps = float(getattr(bn_mod, "eps", eps))
            gamma = bn_mod.weight.to(device=w.device, dtype=w.dtype)
            var = bn_mod.running_var.to(device=w.device, dtype=w.dtype)
            if detach_bn_stats:
                var = var.detach()
            if detach_bn_affine:
                gamma = gamma.detach()
            var = var.clamp(min=bn_eps)
            scale = gamma / torch.sqrt(var + bn_eps)
            view_shape = [scale.shape[0]] + [1] * (w.dim() - 1)
            w_eff = w * scale.view(*view_shape)

        w_flat = w_eff.view(w_eff.shape[0], -1)
        per_out_norm_sq = (w_flat * w_flat).sum(dim=1)
        if full_frobenius:
            m_eff = per_out_norm_sq.sum()
        elif use_max:
            m_eff = per_out_norm_sq.max()
        else:
            m_eff = per_out_norm_sq.mean()
        if normalize_by_fan_in and not full_frobenius:
            m_eff = m_eff / max(1, w_flat.shape[1])

        lam_min = max(eps, 1e-3)
        lam = if_mod.thresh.to(device=w.device, dtype=w.dtype).clamp(min=lam_min).view(-1)[0]
        if detach_lambda:
            lam = lam.detach()

        term = level_factor * m_eff
        if divide_by_lambda:
            term = term / (lam.pow(2) + eps)
        terms.append(term)

    if not terms:
        p = next(model.parameters(), None)
        if p is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=p.device, dtype=p.dtype)
    stacked = torch.stack([term.reshape(()) for term in terms])
    return stacked.mean() if layer_reduction == "mean" else stacked.sum()


def _parameter_grad_norm(loss, params, retain_graph: bool):
    if not torch.is_tensor(loss) or not bool(loss.requires_grad):
        return loss.new_zeros(()) if torch.is_tensor(loss) else torch.zeros(())
    grads = torch.autograd.grad(
        loss, params, retain_graph=retain_graph, allow_unused=True
    )
    total = None
    for grad in grads:
        if grad is None:
            continue
        term = grad.pow(2).sum()
        total = term if total is None else total + term
    if total is None:
        return loss.new_zeros(())
    return total.sqrt()


def compute_mne_l2_regularization(
    model,
    quant_level: int,
    eps: float = 1e-6,
    use_max: bool = False,
    detach_lambda: bool = False,
    detach_bn_stats: bool = True,
    detach_bn_affine=None,
    normalize_by_fan_in: bool = False,
    layer_reduction: str = "sum",
    l_ref=None,
    fold_bn: bool = True,
    full_frobenius: bool = False,
    layer_map=None,
    grad_match_layer_map=None,
    include_roles=None,
    divide_by_lambda: bool = True,
    scale_by_l: bool = True,
):
    """
    Margin-Normalized Effective L2 (MNE-L2):

      R_rho = sum_l  (L^2 * M_eff,l) / (lambda_l^2 + eps)

    Component-ablation switches (defaults preserve the published formula):
      divide_by_lambda=False → drop 1/λ^2
      scale_by_l=False → drop L^2 (or (L/l_ref)^2)
      l_ref=K → replace L^2 with (L/K)^2

    默认 BN-folded effective weight:
      W_tilde = gamma / sqrt(var + eps) * W
    若无 BN，或 fold_bn=False，则 W_tilde = W（γ 不进入正则）。

    M_eff,l:
      - mean 版本: mean_o ||W_tilde_{l,o}||_F^2
      - max  版本: max_o  ||W_tilde_{l,o}||_F^2
      - frobenius 版本: ||W_tilde_l||_F^2，与 weights-only L2 的逐层项相同

    If ``grad_match_layer_map`` differs from the active map, scale R so
    ||∇R|| matches that reference map. The reference is never role-filtered.
    If ``include_roles`` is set, only matched Conv/Linear layers whose
    role is in that set receive an MNE term.
    """
    if layer_reduction not in ("sum", "mean"):
        raise ValueError(f"Unsupported layer_reduction={layer_reduction!r}; expected 'sum' or 'mean'.")
    if full_frobenius and use_max:
        raise ValueError("full_frobenius and use_max cannot be set together.")
    if detach_bn_affine is None:
        detach_bn_affine = detach_bn_stats
    if l_ref is not None and l_ref <= 0:
        raise ValueError(f"l_ref must be positive, got {l_ref}.")

    active_map = _layer_map_from_model(model, layer_map)
    if include_roles is None:
        include_roles = getattr(model, "_mne_include_roles", None)
    include_roles = parse_mne_include_roles(include_roles)
    kwargs = dict(
        model=model,
        quant_level=quant_level,
        eps=eps,
        use_max=use_max,
        detach_lambda=detach_lambda,
        detach_bn_stats=detach_bn_stats,
        detach_bn_affine=detach_bn_affine,
        normalize_by_fan_in=normalize_by_fan_in,
        layer_reduction=layer_reduction,
        l_ref=l_ref,
        fold_bn=fold_bn,
        full_frobenius=full_frobenius,
        divide_by_lambda=divide_by_lambda,
        scale_by_l=scale_by_l,
    )
    penalty = _mne_l2_penalty(
        layer_map=active_map, include_roles=include_roles, **kwargs
    )
    match_to = grad_match_layer_map
    if match_to is None:
        match_to = getattr(model, "_mne_grad_match_layer_map", None)
    if match_to in (None, "", active_map):
        return penalty

    match_to = _layer_map_from_model(model, match_to)
    if match_to == active_map and include_roles is None:
        return penalty
    ref = _mne_l2_penalty(layer_map=match_to, include_roles=None, **kwargs)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    ref_norm = _parameter_grad_norm(ref, params, retain_graph=False)
    raw_norm = _parameter_grad_norm(penalty, params, retain_graph=True)
    scale = (ref_norm / raw_norm.clamp(min=eps)).detach()
    model._mne_grad_match_stats = {
        "scale": float(scale.detach()),
        "reg_grad_norm": float(raw_norm.detach()),
        "ref_grad_norm": float(ref_norm.detach()),
        "matched_grad_norm": float((scale * raw_norm).detach()),
        "layer_map": active_map,
        "grad_match_layer_map": match_to,
    }
    return scale * penalty


BRIDGE_TRANSFORMS = ("uniform", "raw", "normalized", "clipped", "onesided")


def _bridge_layer_mean(values) -> torch.Tensor:
    return torch.stack([value.mean() for value in values]).mean()


def _bridge_renorm(values, eps: float):
    scale = _bridge_layer_mean(values).clamp(min=eps)
    return [value / scale for value in values], scale


def _collect_bridge_layers(
    model,
    quant_level: int,
    eps: float,
    fold_bn: bool,
    detach_lambda: bool,
    detach_bn_stats: bool,
    detach_bn_affine: bool,
):
    module_map = dict(model.named_modules())
    layer_map = _layer_map_from_model(model)
    layers = []
    for lname, layer in model.named_modules():
        if not isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            continue
        weight = getattr(layer, "weight", None)
        if weight is None or not weight.requires_grad:
            continue
        bn_mod, if_mod = _resolve_bn_if_for_layer(
            lname, module_map, layer_map=layer_map
        )
        if if_mod is None:
            continue
        lam_min = max(float(eps), 1e-3)
        lam = if_mod.thresh.to(device=weight.device, dtype=weight.dtype).clamp(
            min=lam_min
        ).view(-1)[0]
        if detach_lambda:
            lam = lam.detach()
        risk = torch.ones(
            (weight.shape[0],), device=weight.device, dtype=weight.dtype
        ) * (float(quant_level) ** 2 / (lam.pow(2) + eps))
        if fold_bn and bn_mod is not None:
            bn_eps = float(getattr(bn_mod, "eps", eps))
            gamma = bn_mod.weight.to(device=weight.device, dtype=weight.dtype)
            var = bn_mod.running_var.to(device=weight.device, dtype=weight.dtype)
            if detach_bn_stats:
                var = var.detach()
            if detach_bn_affine:
                gamma = gamma.detach()
            var = var.clamp(min=bn_eps)
            risk = risk * gamma.pow(2) / (var + bn_eps)
        layers.append(
            {
                "name": lname,
                "weight": weight,
                "risk": risk.detach(),
                "n_out": int(weight.shape[0]),
            }
        )
    return layers


def compute_mne_os_bridge_regularization(
    model,
    quant_level: int,
    risk_transform: str = "raw",
    eps: float = 1e-6,
    clip_min: float = 0.5,
    clip_max: float = 8.0,
    alpha: float = 4.0,
    tau: float = 0.5,
    fold_bn: bool = True,
    detach_lambda: bool = True,
    detach_bn_stats: bool = True,
    detach_bn_affine: bool = True,
    grad_scale: float = 1.0,
):
    """Unified MNE → One-sided bridge with layer-mean reduction.

    r_{l,c} = L^2 γ_{l,c}^2 / (λ_l^2 (v_{l,c}+ε))   (detached λ,γ,v)
    R(q)    = Σ_l (1/C_l) Σ_c q_{l,c} ||W_{l,c}||_F^2

    This matches published Old MNE-L2 when q=r (no 1/2, no unmatched head).
    ``risk_transform``:
      uniform     q=1
      raw         q=r
      normalized  q=r / r̄
      clipped     Renorm(clip(r/r̄, a, b))
      onesided    Renorm(1 + α[ r̃ − τ ]_+)
    r̄ and Renorm use equal layer weight: (1/N_L) Σ_l (1/C_l) Σ_c ·
    """
    transform = str(risk_transform).strip().lower()
    if transform not in BRIDGE_TRANSFORMS:
        raise ValueError(
            f"risk_transform must be one of {BRIDGE_TRANSFORMS}, got {transform!r}."
        )
    layers = _collect_bridge_layers(
        model,
        quant_level=quant_level,
        eps=eps,
        fold_bn=fold_bn,
        detach_lambda=detach_lambda,
        detach_bn_stats=detach_bn_stats,
        detach_bn_affine=detach_bn_affine,
    )
    if not layers:
        parameter = next(model.parameters(), None)
        if parameter is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=parameter.device, dtype=parameter.dtype)

    risks = [layer["risk"] for layer in layers]
    r_bar = _bridge_layer_mean(risks).clamp(min=eps)
    if transform == "uniform":
        q_values = [torch.ones_like(risk) for risk in risks]
    elif transform == "raw":
        q_values = list(risks)
    elif transform == "normalized":
        q_values = [risk / r_bar for risk in risks]
    else:
        clipped = [(risk / r_bar).clamp(min=clip_min, max=clip_max) for risk in risks]
        r_tilde, _ = _bridge_renorm(clipped, eps)
        if transform == "clipped":
            q_values = r_tilde
        else:
            onesided = [
                1.0 + float(alpha) * torch.relu(value - float(tau)) for value in r_tilde
            ]
            q_values, _ = _bridge_renorm(onesided, eps)

    penalty = None
    contribs = []
    for layer, q_value in zip(layers, q_values):
        flat = layer["weight"].reshape(layer["weight"].shape[0], -1)
        term = (q_value.view(-1) * flat.pow(2).sum(dim=1)).mean()
        penalty = term if penalty is None else penalty + term
        contribs.append(float(term.detach()))
    all_q = torch.cat(q_values)
    all_r = torch.cat(risks)
    r_tilde_cat = all_r / r_bar
    p_gt_tau = (r_tilde_cat > float(tau)).to(all_q.dtype).mean()
    if transform in ("clipped", "onesided"):
        clipped = [(risk / r_bar).clamp(min=clip_min, max=clip_max) for risk in risks]
        r_tilde, _ = _bridge_renorm(clipped, eps)
        p_gt_tau = (torch.cat(r_tilde) > float(tau)).to(all_q.dtype).mean()
    scale = float(getattr(model, "_bridge_grad_scale", grad_scale) or grad_scale)
    stats = {
        "risk_transform": transform,
        "r_bar": float(r_bar.detach()),
        "q_mean": float(_bridge_layer_mean(q_values).detach()),
        "q_std": float(all_q.std(unbiased=False).detach()),
        "q_min": float(all_q.min().detach()),
        "q_max": float(all_q.max().detach()),
        "p_gt_tau": float(p_gt_tau.detach()),
        "n_layers": len(layers),
        "bridge_grad_scale": scale,
        "layers": [
            {
                "name": layer["name"],
                "n_out": layer["n_out"],
                "r_mean": float(layer["risk"].mean().detach()),
                "q_mean": float(q_value.mean().detach()),
                "reg_contrib": contrib,
            }
            for layer, q_value, contrib in zip(layers, q_values, contribs)
        ],
    }
    model._bridge_stats = stats
    model._calibrated_mne_stats = {
        "alpha": float(alpha),
        "tau": float(tau),
        "q_assignment": transform,
        "q_mean": stats["q_mean"],
        "q_std": stats["q_std"],
        "q_min": stats["q_min"],
        "q_max": stats["q_max"],
        "p_gt_tau": stats["p_gt_tau"],
        "q_mean_pre_norm": stats["q_mean"],
        "mean_normalize_q": transform != "raw",
    }
    return penalty * scale


def init_mne_os_bridge_grad_scale(
    model,
    quant_level: int,
    risk_transform: str,
    eps: float = 1e-6,
    clip_min: float = 0.5,
    clip_max: float = 8.0,
    alpha: float = 4.0,
    tau: float = 0.5,
    match_raw_grad: bool = True,
) -> dict:
    """One-shot s_k = ||∇R_raw|| / ||∇R_k|| at the first regularized epoch. Frozen after."""
    kwargs = dict(
        model=model,
        quant_level=quant_level,
        eps=eps,
        clip_min=clip_min,
        clip_max=clip_max,
        alpha=alpha,
        tau=tau,
        grad_scale=1.0,
    )
    model._bridge_grad_scale = 1.0
    raw = compute_mne_os_bridge_regularization(risk_transform="raw", **kwargs)
    mne = compute_mne_l2_regularization(
        model,
        quant_level=quant_level,
        detach_lambda=True,
        detach_bn_stats=True,
        detach_bn_affine=True,
        fold_bn=True,
    )
    raw_val = float(raw.detach())
    mne_val = float(mne.detach())
    relerr = abs(raw_val - mne_val) / max(abs(mne_val), eps)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    g_raw = float(_parameter_grad_norm(raw, params, retain_graph=False).detach())
    if str(risk_transform) == "raw" or not match_raw_grad:
        scale = 1.0
        g_k = g_raw
    else:
        other = compute_mne_os_bridge_regularization(
            risk_transform=risk_transform, **kwargs
        )
        g_k = float(_parameter_grad_norm(other, params, retain_graph=False).detach())
        scale = g_raw / max(g_k, eps)
    model._bridge_grad_scale = float(scale)
    card = {
        "risk_transform": str(risk_transform),
        "raw_R": raw_val,
        "mne_R": mne_val,
        "raw_vs_mne_relerr": relerr,
        "g_raw": g_raw,
        "g_k": g_k,
        "bridge_grad_scale": float(scale),
    }
    model._bridge_init_card = card
    return card


def _mne_covered_weight_ids(model) -> set[int]:
    """Conv/Linear weight ids that receive an MNE-L2 term (matched IF required)."""
    module_map = dict(model.named_modules())
    covered = set()
    for lname, layer in model.named_modules():
        if not isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            continue
        weight = getattr(layer, "weight", None)
        if weight is None or not weight.requires_grad:
            continue
        _, if_mod = _resolve_bn_if_for_layer(
            lname, module_map, layer_map=_layer_map_from_model(model)
        )
        if if_mod is None:
            continue
        covered.add(id(weight))
    return covered


def compute_mne_l2_all_regularization(
    model,
    quant_level: int,
    eps: float = 1e-6,
    use_max: bool = False,
    detach_lambda: bool = False,
    detach_bn_stats: bool = True,
    detach_bn_affine=None,
    normalize_by_fan_in: bool = False,
    layer_reduction: str = "sum",
    l_ref=None,
    fold_bn: bool = True,
    full_frobenius: bool = False,
):
    """
    All-parameter coverage built on MNE-L2:

      R = MNE-L2(matched Conv/Linear weights)
        + sum_{p not in matched weights} ||p||_2^2

    Matched weights keep margin-normalized effective-L2 treatment; remaining
    trainable parameters (BN affine, IF thresholds, biases, unmatched heads)
    receive plain L2 so the regularizer covers every trainable parameter once.
    """
    mne = compute_mne_l2_regularization(
        model,
        quant_level=quant_level,
        eps=eps,
        use_max=use_max,
        detach_lambda=detach_lambda,
        detach_bn_stats=detach_bn_stats,
        detach_bn_affine=detach_bn_affine,
        normalize_by_fan_in=normalize_by_fan_in,
        layer_reduction=layer_reduction,
        l_ref=l_ref,
        fold_bn=fold_bn,
        full_frobenius=full_frobenius,
    )
    covered = _mne_covered_weight_ids(model)
    residual = None
    for parameter in model.parameters():
        if not parameter.requires_grad or id(parameter) in covered:
            continue
        term = parameter.pow(2).sum()
        residual = term if residual is None else (residual + term)
    if residual is None:
        return mne
    return mne + residual


def compute_stable_mne_l2_regularization(
    model,
    quant_level: int,
    eps: float = 1e-6,
    use_max: bool = False,
    detach_lambda: bool = True,
    detach_bn_running_stats: bool = True,
    detach_bn_affine: bool = False,
    normalize_by_fan_in: bool = True,
    layer_reduction: str = "mean",
    l_ref: float = 16.0,
):
    """Less coefficient-sensitive MNE-L2 variant for ablation."""
    return compute_mne_l2_regularization(
        model,
        quant_level=quant_level,
        eps=eps,
        use_max=use_max,
        detach_lambda=detach_lambda,
        detach_bn_stats=detach_bn_running_stats,
        detach_bn_affine=detach_bn_affine,
        normalize_by_fan_in=normalize_by_fan_in,
        layer_reduction=layer_reduction,
        l_ref=l_ref,
    )


def compute_hinge_mne_regularization(
    model,
    quant_level: int,
    eps: float = 1e-6,
    use_max: bool = False,
    detach_lambda: bool = True,
    detach_bn_stats: bool = True,
    detach_bn_affine: bool = True,
    tau: float = 1.0,
    use_log: bool = True,
    normalize_by_fan_in: bool = False,
    layer_reduction: str = "mean",
):
    """
    Hinge-MNE: only penalize layers whose effective gain exceeds tau.

      gain_l = L * sqrt(M_eff,l) / lambda_l
      R = mean_l relu(log(gain_l / tau))^2     if use_log
      R = mean_l relu(gain_l - tau)^2          otherwise

    The default detaches lambda and BN statistics/affine parameters so the
    penalty mainly shapes weights instead of moving thresholds or BN gamma.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}.")
    if layer_reduction not in ("sum", "mean"):
        raise ValueError(f"Unsupported layer_reduction={layer_reduction!r}; expected 'sum' or 'mean'.")

    module_map = dict(model.named_modules())
    terms = []

    for lname, layer in model.named_modules():
        if not isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            continue
        if getattr(layer, "weight", None) is None:
            continue

        w = layer.weight
        w_eff = w

        bn_mod, if_mod = _resolve_bn_if_for_layer(
            lname, module_map, layer_map=_layer_map_from_model(model)
        )
        if if_mod is None:
            continue
        if bn_mod is not None:
            bn_eps = float(getattr(bn_mod, "eps", eps))
            gamma = bn_mod.weight.to(device=w.device, dtype=w.dtype)
            var = bn_mod.running_var.to(device=w.device, dtype=w.dtype)
            if detach_bn_stats:
                var = var.detach()
            if detach_bn_affine:
                gamma = gamma.detach()
            var = var.clamp(min=bn_eps)
            scale = gamma / torch.sqrt(var + bn_eps)
            view_shape = [scale.shape[0]] + [1] * (w.dim() - 1)
            w_eff = w * scale.view(*view_shape)

        w_flat = w_eff.view(w_eff.shape[0], -1)
        per_out_norm_sq = (w_flat * w_flat).sum(dim=1)
        m_eff = per_out_norm_sq.max() if use_max else per_out_norm_sq.mean()
        if normalize_by_fan_in:
            m_eff = m_eff / max(1, w_flat.shape[1])

        lam_min = max(eps, 1e-3)
        lam = if_mod.thresh.to(device=w.device, dtype=w.dtype).clamp(min=lam_min).view(-1)[0]
        if detach_lambda:
            lam = lam.detach()

        gain = float(quant_level) * torch.sqrt(m_eff + eps) / (lam + eps)
        tau_t = torch.tensor(float(tau), device=w.device, dtype=w.dtype)
        if use_log:
            violation = F.relu(torch.log((gain + eps) / (tau_t + eps)))
        else:
            violation = F.relu(gain - tau_t)
        terms.append(violation.pow(2))

    if not terms:
        p = next(model.parameters(), None)
        if p is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=p.device, dtype=p.dtype)
    stacked = torch.stack([term.reshape(()) for term in terms])
    return stacked.mean() if layer_reduction == "mean" else stacked.sum()


def _standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _first_matched_conv_bn_if(model):
    """Return (weight_name, conv/linear, bn_or_None, if_mod) for the first matched layer."""
    module_map = dict(model.named_modules())
    for lname, layer in model.named_modules():
        if not isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            continue
        if getattr(layer, "weight", None) is None:
            continue
        bn_mod, if_mod = _resolve_bn_if_for_layer(
            lname, module_map, layer_map=_layer_map_from_model(model)
        )
        if if_mod is None:
            continue
        return lname, layer, bn_mod, if_mod
    return None, None, None, None


def _bn_fold_per_out_l2(weight: torch.Tensor, bn_mod, eps: float = 1e-6) -> torch.Tensor:
    """Per-output-channel ||γ W / sqrt(v)||_2 for Conv/Linear weights."""
    w_eff = weight
    if bn_mod is not None:
        bn_eps = float(getattr(bn_mod, "eps", eps))
        gamma = bn_mod.weight.to(device=weight.device, dtype=weight.dtype)
        var = bn_mod.running_var.detach().to(device=weight.device, dtype=weight.dtype).clamp(min=bn_eps)
        scale = gamma / torch.sqrt(var + bn_eps)
        view_shape = [scale.shape[0]] + [1] * (weight.dim() - 1)
        w_eff = weight * scale.view(*view_shape)
    w_flat = w_eff.reshape(w_eff.shape[0], -1)
    return torch.linalg.vector_norm(w_flat, ord=2, dim=1)


def _noise_scale_factor(protocol: str, quant_level: int, eval_T: int) -> float:
    """
    Scale a in s = a * σ / λ * ||W̃||:
      ann_qcfs: a = L
      snn_indep: a = sqrt(T)   (independent noise each timestep, then accumulate)
      snn_shared: a = T       (same noise shared across timesteps)
    """
    p = str(protocol).strip().lower()
    if p in ("ann", "ann_qcfs", "l"):
        return float(quant_level)
    if p in ("snn_indep", "indep", "sqrt_t", "pre_first_conv"):
        return math.sqrt(max(float(eval_T), 1.0))
    if p in ("snn_shared", "shared", "t"):
        return float(max(eval_T, 1))
    raise ValueError(f"Unknown pc_mne noise protocol={protocol!r}")


def _pc_mne_channel_noise_std(
    model,
    quant_level: int,
    noise_sigma: float,
    protocol: str = "snn_indep",
    eval_T: int = 16,
    eps: float = 1e-6,
    detach_lambda: bool = False,
):
    """
    First-layer per-channel theoretical noise std on QCFS-normalized scale r=Lz/λ:
      s_c = (a σ / λ) * ||W̃_c||_2
    Returns (s_per_channel [C], if_mod, lam, stats_dict) or (None, ...).
    """
    _, layer, bn_mod, if_mod = _first_matched_conv_bn_if(model)
    if layer is None or if_mod is None:
        return None, None, None, {}
    if getattr(if_mod, "last_pre_quant", None) is None:
        return None, if_mod, None, {"warn": "missing_last_pre_quant"}

    per_out = _bn_fold_per_out_l2(layer.weight, bn_mod, eps=eps)
    lam_min = max(eps, 1e-3)
    lam = if_mod.thresh.to(device=per_out.device, dtype=per_out.dtype).clamp(min=lam_min).view(-1)[0]
    if detach_lambda:
        lam = lam.detach()
    a = _noise_scale_factor(protocol, quant_level, eval_T)
    # On r-scale: r = L z / λ, so noise on r has std (L/λ)*σ_z with σ_z = σ*||W̃||
    # User writes s = a σ / λ * ||W̃|| with a=L for ANN, so s is already on the r = Lz/λ scale
    # when a=L: s = L σ /λ ||W|| = std of (L/λ * δz). Good.
    s = (float(a) * float(noise_sigma) / (lam + eps)) * per_out
    stats = {
        "lambda": float(lam.detach().item()),
        "a": float(a),
        "noise_sigma": float(noise_sigma),
        "mean_s": float(s.detach().mean().item()),
    }
    quant_stats = getattr(if_mod, "last_quant_stats", None)
    if isinstance(quant_stats, dict):
        stats.update(quant_stats)
    return s, if_mod, lam, stats


def _activation_boundary_distances(r: torch.Tensor, quant_level: int, eps: float = 1e-6):
    """
    For normalized activation r = L z / λ:
      boundaries b_k = k + 1/2, k = 0..L-1
    Returns (d_left, d_right, mask_left, mask_right) with detached boundary indices.
    """
    L = float(quant_level)
    # Interior left/right half-integer boundaries around r.
    b_left = torch.floor(r + 0.5) - 0.5
    b_right = b_left + 1.0
    b_left = b_left.detach()
    b_right = b_right.detach()

    d_left = (r - b_left).clamp(min=0.0)
    d_right = (b_right - r).clamp(min=0.0)

    # Zero / below: only upward crossing at 0.5 can change clamp(round,0,L).
    # Saturate / above L: only downward crossing at L-0.5.
    below = r <= 0.0
    above = r >= L
    interior = ~(below | above)

    d_left_eff = torch.where(below, torch.zeros_like(d_left), d_left)
    d_right_eff = torch.where(above, torch.zeros_like(d_right), d_right)
    # For below: distance to +0.5
    d_right_eff = torch.where(below, (0.5 - r).clamp(min=0.0), d_right_eff)
    # For above: distance to L-0.5
    d_left_eff = torch.where(above, (r - (L - 0.5)).clamp(min=0.0), d_left_eff)

    use_left = interior | above
    use_right = interior | below
    return d_left_eff, d_right_eff, use_left, use_right


def compute_pc_mne_regularization(
    model,
    quant_level: int,
    noise_sigma: float = 1.0,
    protocol: str = "snn_indep",
    eval_T: int = 16,
    eps: float = 1e-6,
    detach_lambda: bool = False,
    lambda_log_coeff: float = 0.0,
    lambda_ref: float = 1.0,
    first_layer_only: bool = True,
):
    """
    PC-MNE: mean Gaussian crossing probability on first-layer QCFS activations.

      R = mean_i [ Φ(-d_i^- / s_i) + Φ(-d_i^+ / s_i) ]

    Uses IF.last_pre_quant from the just-finished forward. Boundary indices are
    detached; distances and weights keep gradients. BN γ keeps grad via BN-fold;
    BN running_var is detached. No ordinary L2 on BN β / IF λ.
    """
    if not first_layer_only:
        raise NotImplementedError("PC-MNE v1 only supports first_layer_only=True")
    s_c, if_mod, lam, stats = _pc_mne_channel_noise_std(
        model,
        quant_level=quant_level,
        noise_sigma=noise_sigma,
        protocol=protocol,
        eval_T=eval_T,
        eps=eps,
        detach_lambda=detach_lambda,
    )
    if s_c is None or if_mod is None or getattr(if_mod, "last_pre_quant", None) is None:
        p = next(model.parameters(), None)
        return torch.zeros((), device=p.device if p is not None else "cpu")

    z = if_mod.last_pre_quant
    # r = L z / λ  (QCFS-normalized)
    r = float(quant_level) * z / (lam + eps)
    d_left, d_right, use_left, use_right = _activation_boundary_distances(r, quant_level, eps=eps)

    # Broadcast per-channel s to activation shape.
    if z.dim() == 4:
        # [B,C,H,W]
        s = s_c.view(1, -1, 1, 1).to(device=z.device, dtype=z.dtype)
    elif z.dim() == 2:
        s = s_c.view(1, -1).to(device=z.device, dtype=z.dtype)
    else:
        # Fallback: scalar mean s
        s = s_c.mean().to(device=z.device, dtype=z.dtype)

    s_safe = s + eps
    term = torch.zeros_like(r)
    term = term + torch.where(
        use_left,
        _standard_normal_cdf(-(d_left) / s_safe),
        torch.zeros_like(r),
    )
    term = term + torch.where(
        use_right,
        _standard_normal_cdf(-(d_right) / s_safe),
        torch.zeros_like(r),
    )
    loss = term.mean()

    if lambda_log_coeff > 0:
        lam_ref = max(float(lambda_ref), eps)
        loss = loss + float(lambda_log_coeff) * (torch.log(lam + eps) - math.log(lam_ref)).pow(2)

    model._pc_mne_stats = {
        **stats,
        "pc_mne": float(loss.detach().item()),
        "mean_d": float(torch.minimum(d_left, d_right).detach().mean().item()),
    }
    return loss


def compute_margin_mne_regularization(
    model,
    quant_level: int,
    noise_sigma: float = 1.0,
    protocol: str = "snn_indep",
    eval_T: int = 16,
    tau: float = 2.0,
    eps: float = 1e-6,
    detach_lambda: bool = False,
    lambda_log_coeff: float = 0.0,
    lambda_ref: float = 1.0,
    first_layer_only: bool = True,
):
    """
    Margin-MNE hinge on per-activation ρ = d / s:

      R = mean_i relu(τ - d_i / (s_i+eps))^2

    where d_i is distance to the nearest effective QCFS boundary.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if not first_layer_only:
        raise NotImplementedError("Margin-MNE v1 only supports first_layer_only=True")
    s_c, if_mod, lam, stats = _pc_mne_channel_noise_std(
        model,
        quant_level=quant_level,
        noise_sigma=noise_sigma,
        protocol=protocol,
        eval_T=eval_T,
        eps=eps,
        detach_lambda=detach_lambda,
    )
    if s_c is None or if_mod is None or getattr(if_mod, "last_pre_quant", None) is None:
        p = next(model.parameters(), None)
        return torch.zeros((), device=p.device if p is not None else "cpu")

    z = if_mod.last_pre_quant
    r = float(quant_level) * z / (lam + eps)
    d_left, d_right, use_left, use_right = _activation_boundary_distances(r, quant_level, eps=eps)
    # Nearest effective boundary distance.
    d_near = torch.where(
        use_left & use_right,
        torch.minimum(d_left, d_right),
        torch.where(use_left, d_left, d_right),
    )

    if z.dim() == 4:
        s = s_c.view(1, -1, 1, 1).to(device=z.device, dtype=z.dtype)
    elif z.dim() == 2:
        s = s_c.view(1, -1).to(device=z.device, dtype=z.dtype)
    else:
        s = s_c.mean().to(device=z.device, dtype=z.dtype)

    rho = d_near / (s + eps)
    loss = F.relu(float(tau) - rho).pow(2).mean()

    if lambda_log_coeff > 0:
        lam_ref = max(float(lambda_ref), eps)
        loss = loss + float(lambda_log_coeff) * (torch.log(lam + eps) - math.log(lam_ref)).pow(2)

    model._pc_mne_stats = {
        **stats,
        "margin_mne": float(loss.detach().item()),
        "mean_rho": float(rho.detach().mean().item()),
        "mean_d": float(d_near.detach().mean().item()),
        "tau": float(tau),
    }
    return loss


def compute_conv_mne_l2_regularization(
    model,
    quant_level: int,
    eps: float = 1e-6,
    use_max: bool = False,
    detach_lambda: bool = True,
):
    """
    Conv-MNE-L2 (CNN-aware MNE-L2):

      R_conv_mne = sum_l  (L^2 * M_conv,l) / (lambda_l^2 + eps)

    其中 M_conv,l 仅基于卷积层 fan-in 能量：
      M_conv,l,o = sum_{i,r} W_tilde_{l,o,i,r}^2
      M_conv,l   = mean_o(M_conv,l,o)  或 max_o(M_conv,l,o)

    W_tilde 为可选 BN-folded 权重；lambda 默认 stop-gradient（detach）。
    """
    module_map = dict(model.named_modules())
    reg = None

    for lname, layer in model.named_modules():
        if not isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            continue
        if getattr(layer, "weight", None) is None:
            continue

        w = layer.weight
        w_eff = w

        bn_mod, if_mod = _resolve_bn_if_for_layer(
            lname, module_map, layer_map=_layer_map_from_model(model)
        )
        if bn_mod is not None:
            gamma = bn_mod.weight.to(device=w.device, dtype=w.dtype)
            var = bn_mod.running_var.to(device=w.device, dtype=w.dtype)
            bn_eps = float(getattr(bn_mod, "eps", eps))
            scale = gamma / torch.sqrt(var + bn_eps)
            view_shape = [scale.shape[0]] + [1] * (w.dim() - 1)
            w_eff = w * scale.view(*view_shape)

        # 每个输出通道的 fan-in energy: sum_{i,r} W^2
        w_flat = w_eff.view(w_eff.shape[0], -1)
        per_out_fanin_energy = (w_flat * w_flat).sum(dim=1)

        if if_mod is not None and hasattr(if_mod, "thresh"):
            lam = if_mod.thresh.to(device=w.device, dtype=w.dtype).clamp(min=eps).view(-1)
            if detach_lambda:
                lam = lam.detach()
            if lam.numel() == per_out_fanin_energy.numel():
                per_out_term = per_out_fanin_energy / (lam.pow(2) + eps)
                term_base = per_out_term.max() if use_max else per_out_term.mean()
            else:
                lam_scalar = lam.mean()
                m_conv = (
                    per_out_fanin_energy.max()
                    if use_max
                    else per_out_fanin_energy.mean()
                )
                term_base = m_conv / (lam_scalar.pow(2) + eps)
        else:
            term_base = (
                per_out_fanin_energy.max()
                if use_max
                else per_out_fanin_energy.mean()
            )

        term = (float(quant_level) ** 2) * term_base
        reg = term if reg is None else (reg + term)

    if reg is None:
        p = next(model.parameters(), None)
        if p is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=p.device, dtype=p.dtype)
    return reg


def val_reg(model, test_loader, T, device, sample_iter=None, verbose=True):
    """回归：返回 RMSE（越小越好）。"""
    model.eval()
    total_se = 0.0
    total_n = 0
    start_time = time.time()
    if sample_iter is None:
        sample_iter = len(test_loader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device, dtype=torch.float32)
            outputs = model(inputs)
            if outputs.dim() == 3:
                outputs = outputs.mean(0)
            pred = outputs.view(-1)
            t = targets.view(-1)
            total_se += ((pred - t) ** 2).sum().item()
            total_n += t.numel()
            if verbose and (batch_idx + 1) % 20 == 0:
                rmse = (total_se / max(total_n, 1)) ** 0.5
                print(
                    f"batch idx={batch_idx + 1}: current RMSE: {rmse:.6f}"
                )
            if batch_idx == sample_iter:
                break
    rmse = (total_se / max(total_n, 1)) ** 0.5
    elapsed = time.time() - start_time
    if verbose:
        print(f"validate_model elapsed time: {elapsed} seconds")
    return rmse


def val(model, test_loader, T, device, sample_iter=None, verbose=True):
    start_time = time.time()  # Start the timer

    ### sample acc
    if sample_iter is None:
        sample_iter = len(test_loader)
        if verbose:
            print(f"sample_iter of the whole test data loader: {sample_iter}")

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # print(f"batch_idx: {batch_idx}")
            ### get batch size
            batch_size = inputs.size(0)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # 自动检测是否需要解码：如果输出是3维的[T, B, num_classes]，则进行解码
            if outputs.dim() == 3:
                outputs = outputs.mean(0)
            _, predicted = outputs.max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())

            ### Print accuracy every 20 mini-batches
            if verbose and (batch_idx + 1) % 20 == 0:
                current_acc = 100 * correct / total
                print(f"batch idx={batch_idx + 1}, batch size={batch_size}: current accuracy: {current_acc:.3f}%")

            if batch_idx == sample_iter:
                break

        final_acc = 100 * correct / total

    end_time = time.time() # End the timer
    elapsed_time = end_time - start_time # Calculate elapsed time
    if verbose:
        print(f"validate_model elapsed time: {elapsed_time} seconds") # Print the elapsed time

    return final_acc


def calibrate_thresholds(model, calib_loader, device, epochs=5, lr=0.01, verbose=True):
    """
    阈值校准函数：优化SNN网络中每个IF层的thresh参数，以最小化不均匀误差。
    
    使用'rate_uniform'模式作为教师（Teacher），'normal'模式作为学生（Student）。
    通过匹配累积膜电位/累积脉冲计数来减少突发性和长时间静默。
    
    Args:
        model: 预训练的 SNN 模型（含 IF 层）
        calib_loader: 校准数据集的数据加载器
        device: 计算设备（cuda / mps / cpu）
        epochs: 校准轮数（默认5）
        lr: 学习率（默认0.01）
        verbose: 是否打印详细信息（默认True）
    
    Returns:
        model: 校准后的模型（原地修改）
    """
    # 3. 确保模型已设置T（时间步数）
    if model.T == 0:
        raise ValueError("模型的时间步数T未设置，请先调用model.set_T(T)")
    
    T = model.T
    
    # 1. 创建教师模型（完全冻结，使用rate_uniform模式）
    teacher_model = type(model)(*model.__init__.__code__.co_varnames[:model.__init__.__code__.co_argcount-1])
    # 更简单的方法：深拷贝模型
    import copy
    teacher_model = copy.deepcopy(model)
    teacher_model.to(device)
    
    # 冻结教师模型的所有参数
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    teacher_model.set_mode('rate_uniform')
    
    # 2. 学生模型就是原始模型，只优化thresh参数
    student_model = model
    student_model.eval()  # 设置为评估模式，但允许thresh的梯度
    
    # 冻结学生模型的所有参数，除了thresh
    if_layers = []
    for name, module in student_model.named_modules():
        if isinstance(module, IF):
            # 冻结该模块的所有其他参数
            for param_name, param in module.named_parameters():
                if param_name == 'thresh':
                    param.requires_grad = True
                    if_layers.append((name, module))
                else:
                    param.requires_grad = False
    
    # 冻结模型的其他所有参数（Conv, Linear, BN等）
    for name, param in student_model.named_parameters():
        if 'thresh' not in name:
            param.requires_grad = False
    
    # 创建优化器，只优化thresh参数
    thresh_params = []
    for name, module in student_model.named_modules():
        if isinstance(module, IF):
            thresh_params.append(module.thresh)
    
    if len(thresh_params) == 0:
        print("警告：未找到任何IF层，跳过校准")
        return model
    
    optimizer = torch.optim.Adam(thresh_params, lr=lr)
    student_model.set_mode('normal')
    
    if verbose:
        print(f"开始阈值校准：")
        print(f"  - IF层数量: {len(if_layers)}")
        print(f"  - 时间步数T: {T}")
        print(f"  - 校准轮数: {epochs}")
        print(f"  - 学习率: {lr}")
        print(f"  - 使用独立的教师模型和学生模型")
    
    # 4. 校准循环
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, _) in enumerate(calib_loader):
            images = images.to(device)
            batch_size = images.size(0)
            
            optimizer.zero_grad()
            
            # 4.1 教师模型：使用rate_uniform生成理想输出
            with torch.no_grad():
                teacher_output = teacher_model(images)  # Shape: [T, batch, ...]
            
            # 4.2 学生模型：使用normal模式
            student_output = student_model(images)  # Shape: [T, batch, ...]
            
            # 4.3 计算损失：累积和的MSE
            if teacher_output.dim() < 2:
                raise ValueError(f"教师输出维度不正确: {teacher_output.shape}, 期望至少2维 [T, batch, ...]")
            if student_output.dim() < 2:
                raise ValueError(f"学生输出维度不正确: {student_output.shape}, 期望至少2维 [T, batch, ...]")
            
            teacher_cumsum = torch.cumsum(teacher_output, dim=0)  # [T, batch, ...]
            student_cumsum = torch.cumsum(student_output, dim=0)  # [T, batch, ...]
            
            # 计算MSE损失
            loss = F.mse_loss(student_cumsum, teacher_cumsum)
            
            # 4.4 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 确保thresh保持正值
            for module in student_model.modules():
                if isinstance(module, IF):
                    with torch.no_grad():
                        module.thresh.data = torch.clamp(module.thresh.data, min=0.1)
            
            total_loss += loss.item()
            num_batches += 1
            
            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}], Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        if verbose:
            print(f"Epoch [{epoch+1}/{epochs}] 完成, 平均损失: {avg_loss:.6f}")
    
    # 5. 恢复模型为normal模式
    student_model.set_mode('normal')
    
    if verbose:
        print("阈值校准完成！")
        print("校准后的thresh值：")
        for name, module in student_model.named_modules():
            if isinstance(module, IF):
                print(f"  {name}: {module.thresh.data.item():.4f}")
    
    return student_model
