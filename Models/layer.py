import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class MergeTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()

class ExpandTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        y_shape = [self.T, int(x_seq.shape[0]/self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

class GradFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

myfloor = GradFloor.apply

# 全局共享的 rate_table
_SHARED_RATE_TABLE = {}
# 全局共享的非均匀模式表（用于rate_orthogonal_nonuniform）
_SHARED_NONUNIFORM_PATTERNS = {}

def get_rate_table(T, device):
    """
    预计算包含所有相位偏移的 Lookup Table，并进行全局共享。
    Shape: [T+1, T, T] -> [num_spikes, phase_offset, time_step]
    """
    key = (T, device)
    if key in _SHARED_RATE_TABLE:
        return _SHARED_RATE_TABLE[key]
    
    # 1. 基础模式：生成 k 个脉冲在 T 时间内的均匀分布
    # base_patterns shape: [T+1, T]
    base_patterns = torch.zeros(T + 1, T, device=device)
    for k in range(1, T + 1):
        interval = T / k
        indices = torch.floor(torch.arange(k, device=device) * interval).long()
        # 限制索引在 [0, T-1]
        indices = torch.clamp(indices, 0, T - 1) 
        base_patterns[k, indices] = 1.0

    # 2. 生成所有相位 (Phase) 的变体
    # 我们需要生成 shape [T+1, T, T] 的表
    # table[k, p, :] 表示发放 k 个脉冲，且向右循环位移 p 步后的序列
    full_table = []
    for p in range(T):
        # torch.roll 实现循环位移 (Cyclic Shift)
        shifted = torch.roll(base_patterns, shifts=p, dims=1)
        full_table.append(shifted)
    
    # Stack 起来: dim 0=spikes, dim 1=phase, dim 2=time
    rate_table = torch.stack(full_table, dim=1)

    _SHARED_RATE_TABLE[key] = rate_table
    return rate_table

class IF(nn.Module):
    def __init__(
            self,
            T=0,
            L=8,
            thresh=8.0,
            tau=1.,
            gama=1.0,
            scaling_factor=1.0,
            extra_time_steps=0):  # 新增参数：额外的空时间步数量
        """
        T: the number of timesteps, controlled by the function set_T in each model file
        L: controlled by the function set_L in each model file
        scaling_factor: controlled by the function set_scaling_factor in each model file
        extra_time_steps: 额外的空时间步数量，用于释放残留膜电位
        """
        super(IF, self).__init__()
        self.act = ZIF.apply
        self.thresh = nn.Parameter(torch.tensor([thresh]), requires_grad=True)
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.L = L
        self.T = T
        self.extra_time_steps = extra_time_steps  # 保存额外时间步数
        self.loss = 0
        self.scaling_factor = scaling_factor
        self.mode = 'normal'
        self.pattern_table = None

    def forward(self, x, return_mem=False):
        """
        Args:
            x: 输入张量
            return_mem: 如果为True，返回(output, mem)，其中mem是最后一个时间步的膜电位（发放前）
                        如果为False，只返回output（保持向后兼容）
        """
        mem_final = None  # 用于保存最后一个时间步的膜电位（发放前）
        
        if self.T > 0:
            # 1. 常规正脉冲模式
            if self.mode == 'normal':
                thre = self.thresh.data
                x = self.expand(x)
                mem = 0.5 * thre
                spike_pot = []
                for t in range(self.T):
                    mem = mem + x[t, ...]
                    if return_mem and t == self.T - 1:
                        # 保存最后一个时间步的膜电位（发放前）
                        mem_final = mem.clone()
                    spike = self.act(mem - thre, self.gama) * thre
                    mem = mem - spike
                    spike_pot.append(spike)
                x = torch.stack(spike_pot, dim=0)
                x = self.merge(x)
            

        else:
            # T=0：thresh 可学习，除法前 clamp 正下界，避免 lr 大时 thresh→0 出现 NaN
            eps = 1e-3
            th = self.thresh.clamp(min=eps)
            x = x / th
            x = torch.clamp(x, 0, 1)
            x = myfloor(x * self.L + 0.5) / self.L
            x = x * th
            if return_mem:
                # T=0 模式下，膜电位就是输入本身
                mem_final = x.clone()

        if return_mem:
            return x, mem_final
        else:
            return x

 
 
def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(T, 1, 1, 1, 1)
    return x
