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
    for i, (images, labels) in enumerate((train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)
        if T > 0:
            outputs = model(images).mean(0)
        else:
            outputs = model(images)
        loss = criterion(outputs, labels)
        if reg_loss_fn is not None:
            reg = reg_loss_fn(model, T, quant_level)
            loss = loss + float(reg_coeff) * reg
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
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


def _resolve_bn_if_for_layer(layer_name, module_map):
    """
    根据层名匹配该层后续的 BN 与 IF 层（用于 MNE-L2）。

    1) VGG/nn.Sequential 数字索引：layer1.0(Conv)->layer1.1(BN)->layer1.2(IF)
    2) 命名启发式：conv1->bn1/if1, fc1->if1 等（MNIST FC/CNN）
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

    return bn_mod, if_mod


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
        bn_mod, _ = _resolve_bn_if_for_layer(lname, module_map)
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

        _, if_mod = _resolve_bn_if_for_layer(lname, module_map)
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

        bn_mod, if_mod = _resolve_bn_if_for_layer(lname, module_map)
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
):
    """Weights-only L2 with a detached, mean-one MNE risk reweighting.

    The function returns half of the weighted squared norm, so using
    ``reg_coeff=weight_decay`` matches coupled optimizer weight decay when
    ``alpha=0``. Conv/Linear weights without a matched IF (for example the
    classifier head) retain a plain L2 coefficient of one.
    """
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    if risk_min <= 0 or risk_max < risk_min:
        raise ValueError(
            f"Expected 0 < risk_min <= risk_max, got {risk_min}, {risk_max}."
        )

    weighted_layers = []
    plain_weights = []
    module_map = dict(model.named_modules())

    for lname, layer in model.named_modules():
        if not isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            continue
        weight = getattr(layer, "weight", None)
        if weight is None or not weight.requires_grad:
            continue

        bn_mod, if_mod = _resolve_bn_if_for_layer(lname, module_map)
        if if_mod is None:
            plain_weights.append(weight)
            continue

        with torch.no_grad():
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

        n_per_output = weight[0].numel()
        weighted_layers.append((weight, risk, n_per_output))

    all_weights = [entry[0] for entry in weighted_layers] + plain_weights
    if not all_weights:
        parameter = next(model.parameters(), None)
        if parameter is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=parameter.device, dtype=parameter.dtype)

    # This branch is deliberately independent of lambda/BN values. It is the
    # exact explicit-loss counterpart of weights-only optimizer weight decay.
    if float(alpha) == 0.0 or not weighted_layers:
        penalty = sum(weight.pow(2).sum() for weight in all_weights)
        model._calibrated_mne_stats = {
            "alpha": float(alpha),
            "risk_mean": 1.0,
            "risk_min": 1.0,
            "risk_max": 1.0,
            "q_mean": 1.0,
            "q_min": 1.0,
            "q_max": 1.0,
        }
        return 0.5 * penalty

    with torch.no_grad():
        total_covered_params = sum(
            n_per_output * risk.numel()
            for _, risk, n_per_output in weighted_layers
        )
        risk_mean = sum(
            n_per_output * risk.sum()
            for _, risk, n_per_output in weighted_layers
        ) / float(total_covered_params)
        risk_mean = risk_mean.clamp(min=eps)

        clipped_risks = [
            (risk / risk_mean).clamp(min=risk_min, max=risk_max)
            for _, risk, _ in weighted_layers
        ]
        clipped_mean = sum(
            n_per_output * clipped.sum()
            for clipped, (_, _, n_per_output) in zip(
                clipped_risks, weighted_layers
            )
        ) / float(total_covered_params)
        clipped_mean = clipped_mean.clamp(min=eps)
        normalized_risks = [clipped / clipped_mean for clipped in clipped_risks]
        q_values = [
            (1.0 - float(alpha)) + float(alpha) * normalized
            for normalized in normalized_risks
        ]

        q_mean = sum(
            n_per_output * q.sum()
            for q, (_, _, n_per_output) in zip(q_values, weighted_layers)
        ) / float(total_covered_params)
        all_normalized = torch.cat(normalized_risks)
        all_q = torch.cat(q_values)
        model._calibrated_mne_stats = {
            "alpha": float(alpha),
            "risk_mean": risk_mean.detach(),
            "risk_min": all_normalized.min().detach(),
            "risk_max": all_normalized.max().detach(),
            "q_mean": q_mean.detach(),
            "q_min": all_q.min().detach(),
            "q_max": all_q.max().detach(),
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
):
    """
    Margin-Normalized Effective L2 (MNE-L2):

      R_rho = sum_l  (L^2 * M_eff,l) / (lambda_l^2 + eps)

    默认 BN-folded effective weight:
      W_tilde = gamma / sqrt(var + eps) * W
    若无 BN，或 fold_bn=False，则 W_tilde = W（γ 不进入正则）。

    M_eff,l:
      - mean 版本: mean_o ||W_tilde_{l,o}||_F^2
      - max  版本: max_o  ||W_tilde_{l,o}||_F^2
      - frobenius 版本: ||W_tilde_l||_F^2，与 weights-only L2 的逐层项相同
    """
    if layer_reduction not in ("sum", "mean"):
        raise ValueError(f"Unsupported layer_reduction={layer_reduction!r}; expected 'sum' or 'mean'.")
    if full_frobenius and use_max:
        raise ValueError("full_frobenius and use_max cannot be set together.")
    if detach_bn_affine is None:
        detach_bn_affine = detach_bn_stats
    if l_ref is not None and l_ref <= 0:
        raise ValueError(f"l_ref must be positive, got {l_ref}.")

    module_map = dict(model.named_modules())
    terms = []
    level_factor = (float(quant_level) / float(l_ref)) ** 2 if l_ref is not None else float(quant_level) ** 2

    for lname, layer in model.named_modules():
        if not isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            continue
        if getattr(layer, "weight", None) is None:
            continue

        w = layer.weight
        w_eff = w

        bn_mod, if_mod = _resolve_bn_if_for_layer(lname, module_map)
        # 方案 C：无匹配 IF 的层（如 VGG classifier.7 输出头）不参与 MNE-L2。
        if if_mod is None:
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

        terms.append(level_factor * m_eff / (lam.pow(2) + eps))

    if not terms:
        p = next(model.parameters(), None)
        if p is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=p.device, dtype=p.dtype)
    stacked = torch.stack([term.reshape(()) for term in terms])
    return stacked.mean() if layer_reduction == "mean" else stacked.sum()


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
        _, if_mod = _resolve_bn_if_for_layer(lname, module_map)
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

        bn_mod, if_mod = _resolve_bn_if_for_layer(lname, module_map)
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
        bn_mod, if_mod = _resolve_bn_if_for_layer(lname, module_map)
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

        bn_mod, if_mod = _resolve_bn_if_for_layer(lname, module_map)
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
