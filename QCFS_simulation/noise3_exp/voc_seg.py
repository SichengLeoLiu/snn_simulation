"""VOC 2012 semantic segmentation dataset, mIoU, and download check."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from Models.FCN import IGNORE_INDEX, NUM_CLASSES, STRIDE
from voc_ssd import IMAGENET_MEAN, IMAGENET_STD, download_voc

CROP = 512


def voc2012_root_from(path: Path) -> Path:
    path = Path(path)
    for candidate in (path, path / "VOCdevkit"):
        if (candidate / "VOC2012" / "ImageSets" / "Segmentation" / "train.txt").is_file():
            return candidate
    raise FileNotFoundError(
        f"VOC 2012 segmentation not found under {path}. "
        "Expected VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt."
    )


def voc2012_seg_is_ready(path: Path) -> bool:
    try:
        root = voc2012_root_from(Path(path))
    except FileNotFoundError:
        return False
    mask_dir = root / "VOC2012" / "SegmentationClass"
    id_file = root / "VOC2012" / "ImageSets" / "Segmentation" / "train.txt"
    ids = [line.strip() for line in id_file.read_text().splitlines() if line.strip()]
    return bool(ids) and (mask_dir / f"{ids[0]}.png").is_file()


class VOCSegSet(Dataset):
    def __init__(self, voc_root: Path, split: str, train: bool, crop: int = CROP):
        self.root = voc2012_root_from(Path(voc_root))
        self.train = bool(train)
        self.crop = int(crop)
        id_file = self.root / "VOC2012" / "ImageSets" / "Segmentation" / f"{split}.txt"
        if not id_file.is_file():
            raise FileNotFoundError(id_file)
        self.ids = [line.strip() for line in id_file.read_text().splitlines() if line.strip()]
        self.img_dir = self.root / "VOC2012" / "JPEGImages"
        self.mask_dir = self.root / "VOC2012" / "SegmentationClass"
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        image = Image.open(self.img_dir / f"{image_id}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{image_id}.png")
        if self.train:
            image, mask = _train_pair(image, mask, self.crop)
        image_t = self.normalize(transforms.functional.to_tensor(image))
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image_t, mask_t, image_id, (mask_t.shape[-2], mask_t.shape[-1])


def _train_pair(image: Image.Image, mask: Image.Image, crop: int):
    if torch.rand(1).item() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    scale = float(torch.empty(1).uniform_(0.5, 1.5).item())
    new_w = max(crop, int(round(image.size[0] * scale)))
    new_h = max(crop, int(round(image.size[1] * scale)))
    image = image.resize((new_w, new_h), Image.BILINEAR)
    mask = mask.resize((new_w, new_h), Image.NEAREST)
    left = int(torch.randint(0, new_w - crop + 1, (1,)).item()) if new_w > crop else 0
    top = int(torch.randint(0, new_h - crop + 1, (1,)).item()) if new_h > crop else 0
    image = image.crop((left, top, left + crop, top + crop))
    mask = mask.crop((left, top, left + crop, top + crop))
    return image, mask


def pad_to_stride(image: torch.Tensor, stride: int = STRIDE):
    _, height, width = image.shape
    pad_h = (stride - height % stride) % stride
    pad_w = (stride - width % stride) % stride
    if pad_h or pad_w:
        image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), value=0.0)
    return image, height, width


def collate_train(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack([item[1] for item in batch], dim=0)
    ids = [item[2] for item in batch]
    return images, masks, ids


def collate_val(batch):
    return batch


def confusion_update(conf: torch.Tensor, pred: torch.Tensor, target: torch.Tensor):
    valid = target != IGNORE_INDEX
    if not valid.any():
        return
    pred = pred[valid].reshape(-1)
    target = target[valid].reshape(-1)
    idx = target * conf.shape[0] + pred
    conf += torch.bincount(idx, minlength=conf.numel()).reshape(conf.shape)


def scores_from_confusion(conf: torch.Tensor) -> dict:
    conf = conf.cpu().to(torch.float64)
    tp = torch.diag(conf)
    support = conf.sum(1)
    pred_sum = conf.sum(0)
    union = support + pred_sum - tp
    iou = torch.where(union > 0, tp / union.clamp(min=1e-12), torch.full_like(tp, float("nan")))
    valid = support > 0
    miou = float(iou[valid].mean().item()) if valid.any() else float("nan")
    pixel_acc = float(tp.sum().item() / conf.sum().clamp(min=1.0).item())
    return {
        "mIoU": 100.0 * miou,
        "pixel_acc": 100.0 * pixel_acc,
        "per_class_iou": [None if x != x else round(100.0 * float(x), 4) for x in iou.tolist()],
    }
