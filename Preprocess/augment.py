"""CIFAR：Cutout（tensor）；无 torchvision AutoAugment 时的弱回退增强。"""
import random
import numpy as np
import torch


class Cutout:
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), dtype=np.float32)
        for _ in range(self.n_holes):
            y, x = random.randint(0, h - 1), random.randint(0, w - 1)
            y1 = int(np.clip(y - self.length / 2, 0, h))
            y2 = int(np.clip(y + self.length / 2, 0, h))
            x1 = int(np.clip(x - self.length / 2, 0, w))
            x2 = int(np.clip(x + self.length / 2, 0, w))
            mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask).to(img.device)
        return img * mask.expand_as(img)


class CIFAR10Policy:
    """若无 AutoAugment API，仅用轻微随机裁剪缩放作占位（建议升级 torchvision）。"""

    def __call__(self, img):
        return img
