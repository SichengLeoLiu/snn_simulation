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

def train(model, device, train_loader, criterion, optimizer, T):
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
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total


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