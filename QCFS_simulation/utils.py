import time
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


def compute_mne_l2_regularization(
    model,
    quant_level: int,
    eps: float = 1e-6,
    use_max: bool = False,
    detach_lambda: bool = False,
    detach_bn_stats: bool = True,
):
    """
    Margin-Normalized Effective L2 (MNE-L2):

      R_rho = sum_l  (L^2 * M_eff,l) / (lambda_l^2 + eps)

    其中 BN-folded effective weight:
      W_tilde = gamma / sqrt(var + eps) * W
    若无 BN，则 W_tilde = W。

    M_eff,l:
      - mean 版本: mean_o ||W_tilde_{l,o}||_F^2
      - max  版本: max_o  ||W_tilde_{l,o}||_F^2
    """
    module_map = dict(model.named_modules())
    reg = None

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
        if bn_mod is not None:
            bn_eps = float(getattr(bn_mod, "eps", eps))
            gamma = bn_mod.weight.to(device=w.device, dtype=w.dtype)
            var = bn_mod.running_var.to(device=w.device, dtype=w.dtype)
            if detach_bn_stats:
                gamma = gamma.detach()
                var = var.detach()
            var = var.clamp(min=bn_eps)
            scale = gamma / torch.sqrt(var + bn_eps)
            view_shape = [scale.shape[0]] + [1] * (w.dim() - 1)
            w_eff = w * scale.view(*view_shape)

        w_flat = w_eff.view(w_eff.shape[0], -1)
        per_out_norm_sq = (w_flat * w_flat).sum(dim=1)
        m_eff = per_out_norm_sq.max() if use_max else per_out_norm_sq.mean()

        lam_min = max(eps, 1e-3)
        lam = if_mod.thresh.to(device=w.device, dtype=w.dtype).clamp(min=lam_min).view(-1)[0]
        if detach_lambda:
            lam = lam.detach()

        term = (float(quant_level) ** 2) * m_eff / (lam.pow(2) + eps)
        reg = term if reg is None else (reg + term)

    if reg is None:
        p = next(model.parameters(), None)
        if p is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=p.device, dtype=p.dtype)
    return reg


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