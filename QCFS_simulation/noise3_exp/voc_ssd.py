"""VOC data, MultiBox loss, and VOC07 mAP@0.5 for SSD300."""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from Models.SSD import VOC_CLASSES, decode_boxes, encode_boxes, nms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_TO_ID = {name: i + 1 for i, name in enumerate(VOC_CLASSES)}


def voc_root_from(path: Path) -> Path:
    path = Path(path)
    if (path / "VOC2007").is_dir() and (path / "VOC2012").is_dir():
        return path
    if (path / "VOCdevkit" / "VOC2007").is_dir():
        return path / "VOCdevkit"
    raise FileNotFoundError(
        f"VOC 2007+2012 not found under {path}. Expected VOCdevkit/VOC2007 and VOC2012."
    )


class VOCDetectionSet(Dataset):
    def __init__(self, voc_root: Path, image_set: str, train: bool):
        self.root = voc_root_from(Path(voc_root))
        self.train = bool(train)
        self.samples = []
        years = ("2007", "2012") if image_set != "test" else ("2007",)
        for year in years:
            split = "test" if image_set == "test" else image_set
            id_file = self.root / f"VOC{year}" / "ImageSets" / "Main" / f"{split}.txt"
            if not id_file.is_file():
                raise FileNotFoundError(id_file)
            ids = [line.strip() for line in id_file.read_text().splitlines() if line.strip()]
            img_dir = self.root / f"VOC{year}" / "JPEGImages"
            ann_dir = self.root / f"VOC{year}" / "Annotations"
            for image_id in ids:
                self.samples.append((img_dir / f"{image_id}.jpg", ann_dir / f"{image_id}.xml"))
        self.tf = transforms.Compose(
            [
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, ann_path = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        boxes, labels = _parse_voc_xml(ann_path, width, height)
        if self.train and boxes.numel() and torch.rand(1).item() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = boxes.clone()
            boxes[:, [0, 2]] = 1.0 - boxes[:, [2, 0]]
        tensor = self.tf(image)
        return tensor, {"boxes": boxes, "labels": labels, "id": img_path.stem}


def _parse_voc_xml(path: Path, width: int, height: int):
    root = ET.parse(path).getroot()
    boxes = []
    labels = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in CLASS_TO_ID:
            continue
        difficult = obj.findtext("difficult") or "0"
        if int(difficult) == 1:
            continue
        bnd = obj.find("bndbox")
        xmin = (float(bnd.findtext("xmin")) - 1.0) / width
        ymin = (float(bnd.findtext("ymin")) - 1.0) / height
        xmax = (float(bnd.findtext("xmax")) - 1.0) / width
        ymax = (float(bnd.findtext("ymax")) - 1.0) / height
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append((xmin, ymin, xmax, ymax))
        labels.append(CLASS_TO_ID[name])
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
    return torch.tensor(boxes, dtype=torch.float32).clamp(0, 1), torch.tensor(labels, dtype=torch.long)


def collate_voc(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


def box_iou(a, b):
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


def match_priors(priors, boxes, labels, iou_thresh=0.5):
    n_priors = priors.shape[0]
    loc_t = priors.new_zeros((n_priors, 4))
    conf_t = torch.zeros((n_priors,), dtype=torch.long, device=priors.device)
    if boxes.numel() == 0:
        return loc_t, conf_t
    priors_xyxy = torch.stack(
        (
            priors[:, 0] - priors[:, 2] / 2,
            priors[:, 1] - priors[:, 3] / 2,
            priors[:, 0] + priors[:, 2] / 2,
            priors[:, 1] + priors[:, 3] / 2,
        ),
        dim=1,
    ).clamp(0, 1)
    ious = box_iou(priors_xyxy, boxes)
    best_gt_iou, best_gt = ious.max(dim=1)
    best_prior_iou, best_prior = ious.max(dim=0)
    best_gt = best_gt.clone()
    best_gt_iou = best_gt_iou.clone()
    for gt_idx, prior_idx in enumerate(best_prior.tolist()):
        best_gt[int(prior_idx)] = int(gt_idx)
        best_gt_iou[int(prior_idx)] = 1.0
    matched = boxes[best_gt]
    loc_t = encode_boxes(matched, priors)
    conf_t = labels[best_gt]
    conf_t[best_gt_iou < iou_thresh] = 0
    conf_t[best_prior] = labels
    return loc_t, conf_t


def multibox_loss(loc_pred, conf_pred, targets, priors, neg_pos=3.0):
    loc_t = []
    conf_t = []
    for target in targets:
        loc_i, conf_i = match_priors(
            priors, target["boxes"].to(priors.device), target["labels"].to(priors.device)
        )
        loc_t.append(loc_i)
        conf_t.append(conf_i)
    loc_t = torch.stack(loc_t, dim=0)
    conf_t = torch.stack(conf_t, dim=0)
    pos = conf_t > 0
    n_pos = pos.sum().clamp(min=1)
    loc_loss = F.smooth_l1_loss(loc_pred[pos], loc_t[pos], reduction="sum") / n_pos
    conf_flat = conf_pred.reshape(-1, conf_pred.shape[-1])
    conf_target = conf_t.reshape(-1)
    loss_c = F.cross_entropy(conf_flat, conf_target, reduction="none").reshape(conf_t.shape)
    loss_c[pos] = 0
    n_neg = (neg_pos * pos.sum(1)).long().clamp(min=1)
    neg_mask = torch.zeros_like(pos)
    for b, k in enumerate(n_neg):
        _, idx = loss_c[b].sort(descending=True)
        neg_mask[b, idx[: int(k)]] = True
    conf_keep = pos | neg_mask
    conf_loss = F.cross_entropy(conf_pred[conf_keep], conf_t[conf_keep], reduction="sum") / n_pos
    return loc_loss + conf_loss, loc_loss.detach(), conf_loss.detach(), int(n_pos)


@torch.no_grad()
def detect(loc, conf, priors, score_thresh=0.01, iou_thresh=0.45, top_k=200, max_per_image=200):
    scores = torch.softmax(conf, dim=-1)
    decoded = decode_boxes(loc, priors)
    outputs = []
    for b in range(loc.shape[0]):
        boxes_b = []
        scores_b = []
        labels_b = []
        for cls in range(1, scores.shape[-1]):
            cls_scores = scores[b, :, cls]
            keep = cls_scores > score_thresh
            if keep.sum() == 0:
                continue
            kept_boxes = decoded[b][keep]
            kept_scores = cls_scores[keep]
            keep_nms = nms(kept_boxes, kept_scores, iou_thresh=iou_thresh, top_k=top_k)
            boxes_b.append(kept_boxes[keep_nms])
            scores_b.append(kept_scores[keep_nms])
            labels_b.append(torch.full((keep_nms.numel(),), cls, device=loc.device, dtype=torch.long))
        if boxes_b:
            boxes_cat = torch.cat(boxes_b, dim=0)
            scores_cat = torch.cat(scores_b, dim=0)
            labels_cat = torch.cat(labels_b, dim=0)
            if boxes_cat.shape[0] > max_per_image:
                _, idx = scores_cat.topk(max_per_image)
                boxes_cat, scores_cat, labels_cat = boxes_cat[idx], scores_cat[idx], labels_cat[idx]
            outputs.append({"boxes": boxes_cat, "scores": scores_cat, "labels": labels_cat})
        else:
            outputs.append(
                {
                    "boxes": loc.new_zeros((0, 4)),
                    "scores": loc.new_zeros((0,)),
                    "labels": torch.zeros((0,), dtype=torch.long, device=loc.device),
                }
            )
    return outputs


def voc07_ap(rec, prec):
    ap = 0.0
    for t in [k / 10.0 for k in range(11)]:
        mask = rec >= t
        ap += float(prec[mask].max()) if mask.any() else 0.0
    return ap / 11.0


def voc_map(pred_by_image, gt_by_image, num_classes=21, iou_thresh=0.5):
    aps = {}
    for cls in range(1, num_classes):
        records = []
        n_gt = 0
        for image_id, gt in gt_by_image.items():
            mask = gt["labels"] == cls
            gt_boxes = gt["boxes"][mask]
            n_gt += int(gt_boxes.shape[0])
            pred = pred_by_image.get(image_id, {})
            pmask = pred.get("labels", gt["labels"].new_zeros((0,))) == cls
            pboxes = pred.get("boxes", gt["boxes"].new_zeros((0, 4)))[pmask]
            pscores = pred.get("scores", gt["boxes"].new_zeros((0,)))[pmask]
            if pboxes.numel() == 0:
                continue
            order = pscores.argsort(descending=True)
            matched = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool)
            for idx in order:
                box = pboxes[idx]
                score = float(pscores[idx])
                if gt_boxes.numel() == 0:
                    records.append((score, 0))
                    continue
                ious = box_iou(box[None], gt_boxes)[0]
                j = int(ious.argmax())
                if float(ious[j]) >= iou_thresh and not bool(matched[j]):
                    matched[j] = True
                    records.append((score, 1))
                else:
                    records.append((score, 0))
        if n_gt == 0:
            aps[VOC_CLASSES[cls - 1]] = float("nan")
            continue
        if not records:
            aps[VOC_CLASSES[cls - 1]] = 0.0
            continue
        records.sort(key=lambda item: item[0], reverse=True)
        tp = torch.tensor([item[1] for item in records], dtype=torch.float32)
        fp = 1.0 - tp
        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        rec = tp_cum / n_gt
        prec = tp_cum / (tp_cum + fp_cum + 1e-6)
        aps[VOC_CLASSES[cls - 1]] = voc07_ap(rec, prec)
    values = [v for v in aps.values() if v == v]
    mean_ap = sum(values) / max(1, len(values))
    return mean_ap, aps


VOC_ARCHIVES = (
    (
        "VOCtrainval_06-Nov-2007.tar",
        (
            "https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
            "https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar",
        ),
    ),
    (
        "VOCtest_06-Nov-2007.tar",
        (
            "https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
            "https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar",
        ),
    ),
    (
        "VOCtrainval_11-May-2012.tar",
        (
            "https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
            "https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar",
        ),
    ),
)
VOC_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def voc_is_ready(path: Path) -> bool:
    try:
        root = voc_root_from(Path(path))
    except FileNotFoundError:
        return False
    needed = (
        root / "VOC2007" / "ImageSets" / "Main" / "trainval.txt",
        root / "VOC2007" / "ImageSets" / "Main" / "test.txt",
        root / "VOC2012" / "ImageSets" / "Main" / "trainval.txt",
    )
    return all(item.is_file() for item in needed)


def _download_one(urls, tar_path: Path) -> None:
    import shutil
    import subprocess
    import urllib.request

    if tar_path.is_file() and tar_path.stat().st_size > 1_000_000:
        print(f"[VOC] using cached {tar_path}", flush=True)
        return
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    wget = shutil.which("wget")
    curl = shutil.which("curl")
    for url in urls:
        tmp = tar_path.with_suffix(tar_path.suffix + ".part")
        print(f"[VOC] downloading {url}", flush=True)
        try:
            if wget:
                subprocess.run(
                    [
                        wget,
                        "-c",
                        "--tries=3",
                        f"--user-agent={VOC_USER_AGENT}",
                        "-O",
                        str(tmp),
                        url,
                    ],
                    check=True,
                )
            elif curl:
                subprocess.run(
                    [
                        curl,
                        "-L",
                        "--retry",
                        "3",
                        "-A",
                        VOC_USER_AGENT,
                        "-o",
                        str(tmp),
                        url,
                    ],
                    check=True,
                )
            else:
                request = urllib.request.Request(url, headers={"User-Agent": VOC_USER_AGENT})
                with urllib.request.urlopen(request, timeout=60) as src, tmp.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            if tmp.is_file() and tmp.stat().st_size > 1_000_000:
                tmp.replace(tar_path)
                return
            errors.append(f"{url}: downloaded file too small")
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            if tmp.exists():
                tmp.unlink()
    raise RuntimeError("VOC download failed:\n  " + "\n  ".join(errors))


def download_voc(dest: Path) -> Path:
    """Download VOC 2007+2012 into dest/VOCdevkit. Login-node only."""
    import tarfile

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if voc_is_ready(dest):
        return voc_root_from(dest)
    cache = dest / "_voc_tarballs"
    cache.mkdir(parents=True, exist_ok=True)
    for name, urls in VOC_ARCHIVES:
        tar_path = cache / name
        _download_one(urls, tar_path)
        print(f"[VOC] extracting {tar_path}", flush=True)
        with tarfile.open(tar_path, "r") as handle:
            handle.extractall(path=dest)
    root = voc_root_from(dest)
    if not voc_is_ready(root):
        raise FileNotFoundError(f"VOC download finished but files missing under {dest}")
    return root


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Pascal VOC 2007+2012")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--voc-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.download:
        raise SystemExit("pass --download --voc-root DIR")
    print(download_voc(args.voc_root))
