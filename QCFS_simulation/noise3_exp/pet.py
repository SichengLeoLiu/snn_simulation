"""Oxford-IIIT Pet classification + trimap foreground segmentation."""
from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from Models.PetVGG import IGNORE_INDEX, NUM_CLS
from voc_ssd import IMAGENET_MEAN, IMAGENET_STD

SIZE = 224
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
# thor first: the vgg.ox.ac.uk URLs 301-redirect, and wget -O then exits 3
# even after a complete download.
ARCHIVES = (
    (
        "images.tar.gz",
        (
            "https://thor.robots.ox.ac.uk/pets/images.tar.gz",
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
        ),
        700_000_000,
    ),
    (
        "annotations.tar.gz",
        (
            "https://thor.robots.ox.ac.uk/pets/annotations.tar.gz",
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
        ),
        10_000_000,
    ),
)


def pet_root_from(path: Path) -> Path:
    path = Path(path).expanduser()
    for candidate in (path, path / "oxford-iiit-pet", path / "OxfordIIITPet"):
        if (candidate / "annotations" / "trainval.txt").is_file() and (candidate / "images").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Oxford-IIIT Pet not found under {path}. "
        "Expected oxford-iiit-pet/annotations/trainval.txt and images/."
    )


def pet_is_ready(path: Path) -> bool:
    try:
        root = pet_root_from(Path(path))
    except FileNotFoundError:
        return False
    trainval = root / "annotations" / "trainval.txt"
    test = root / "annotations" / "test.txt"
    if not trainval.is_file() or not test.is_file():
        return False
    first = next((line.split()[0] for line in trainval.read_text().splitlines() if line.strip()), "")
    if not first:
        return False
    return _image_path(root, first) is not None and _trimap_path(root, first) is not None


def _image_path(root: Path, image_id: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png"):
        path = root / "images" / f"{image_id}{suffix}"
        if path.is_file():
            return path
    return None


def _trimap_path(root: Path, image_id: str) -> Path | None:
    path = root / "annotations" / "trimaps" / f"{image_id}.png"
    return path if path.is_file() else None


def parse_split_ids(root: Path, split: str) -> list[tuple[str, int]]:
    path = root / "annotations" / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        image_id, class_id = parts[0], int(parts[1])
        if not (1 <= class_id <= NUM_CLS):
            raise ValueError(f"bad class id {class_id} for {image_id}")
        rows.append((image_id, class_id - 1))
    if not rows:
        raise RuntimeError(f"empty Pet split {path}")
    return rows


def trimap_to_mask(trimap: np.ndarray) -> torch.Tensor:
    """1=foreground, 2=background, 3=uncertain → {1, 0, IGNORE_INDEX}."""
    mask = np.full(trimap.shape, IGNORE_INDEX, dtype=np.int64)
    mask[trimap == 1] = 1
    mask[trimap == 2] = 0
    return torch.from_numpy(mask)


def _complete_download(path: Path, min_bytes: int) -> bool:
    return path.is_file() and path.stat().st_size >= int(min_bytes)


def _download_one(urls, tar_path: Path, min_bytes: int) -> None:
    import shutil
    import subprocess
    import urllib.request

    if _complete_download(tar_path, min_bytes):
        print(f"[PET] using cached {tar_path}", flush=True)
        return
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tar_path.with_suffix(tar_path.suffix + ".part")
    if _complete_download(tmp, min_bytes):
        print(f"[PET] keeping complete partial {tmp}", flush=True)
        tmp.replace(tar_path)
        return
    errors = []
    wget = shutil.which("wget")
    curl = shutil.which("curl")
    for url in urls:
        print(f"[PET] downloading {url}", flush=True)
        try:
            code = 0
            if wget:
                code = subprocess.run(
                    ["wget", "-c", "--tries=3", f"--user-agent={USER_AGENT}", "-O", str(tmp), url],
                    check=False,
                ).returncode
            elif curl:
                code = subprocess.run(
                    ["curl", "-L", "--retry", "3", "-A", USER_AGENT, "-o", str(tmp), url],
                    check=False,
                ).returncode
            else:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=60) as src, tmp.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            if _complete_download(tmp, min_bytes):
                if code:
                    print(f"[PET] wget/curl exit {code} but {tmp.name} is complete", flush=True)
                tmp.replace(tar_path)
                return
            size = tmp.stat().st_size if tmp.is_file() else 0
            errors.append(f"{url}: exit={code} size={size}")
            if code == 3:
                raise RuntimeError(
                    "Pet download stopped with wget exit 3 (usually disk quota). "
                    f"Incomplete file {tmp} is {size} bytes. "
                    "Delete _pet_tarballs and download to scratch, e.g. "
                    "/scratch/gs14/sl9144/datasets, not $HOME."
                )
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            if tmp.exists() and not _complete_download(tmp, min_bytes):
                tmp.unlink()
    raise RuntimeError("Pet download failed:\n  " + "\n  ".join(errors))


def download_pet(dest: Path) -> Path:
    """Download Oxford-IIIT Pet into dest/oxford-iiit-pet. Login-node only."""
    dest = Path(dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    if pet_is_ready(dest):
        return pet_root_from(dest)
    out = dest / "oxford-iiit-pet"
    out.mkdir(parents=True, exist_ok=True)
    cache = dest / "_pet_tarballs"
    cache.mkdir(parents=True, exist_ok=True)
    for name, urls, min_bytes in ARCHIVES:
        tar_path = cache / name
        _download_one(urls, tar_path, min_bytes)
        print(f"[PET] extracting {tar_path}", flush=True)
        with tarfile.open(tar_path, "r") as handle:
            handle.extractall(path=out)
        tar_path.unlink()
        print(f"[PET] removed tarball {tar_path.name} after extract", flush=True)
    root = pet_root_from(dest)
    if not pet_is_ready(root):
        raise FileNotFoundError(f"Pet download finished but files missing under {dest}")
    return root


class PetSet(Dataset):
    """Official trainval / test. Classification and segmentation share the same ids."""

    def __init__(self, pet_root: Path, split: str, task: str, train: bool, size: int = SIZE):
        self.root = pet_root_from(Path(pet_root))
        self.task = str(task).strip().lower()
        if self.task not in ("cls", "seg"):
            raise ValueError(f"task must be cls or seg, got {task}")
        self.train = bool(train)
        self.size = int(size)
        self.ids = parse_split_ids(self.root, split)
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.ids)

    def _open_pair(self, image_id: str):
        img_path = _image_path(self.root, image_id)
        if img_path is None:
            raise FileNotFoundError(f"image missing for {image_id}")
        image = Image.open(img_path).convert("RGB")
        tri_path = _trimap_path(self.root, image_id)
        if tri_path is None:
            raise FileNotFoundError(f"trimap missing for {image_id}")
        trimap = Image.open(tri_path)
        return image, trimap

    def _geometry(self, image: Image.Image, trimap: Image.Image):
        if self.train:
            extra = 32
            resized = self.size + extra
            image = image.resize((resized, resized), Image.BILINEAR)
            trimap = trimap.resize((resized, resized), Image.NEAREST)
            left = int(torch.randint(0, extra + 1, (1,)).item())
            top = int(torch.randint(0, extra + 1, (1,)).item())
            image = image.crop((left, top, left + self.size, top + self.size))
            trimap = trimap.crop((left, top, left + self.size, top + self.size))
            if torch.rand(1).item() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                trimap = trimap.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            image = image.resize((self.size, self.size), Image.BILINEAR)
            trimap = trimap.resize((self.size, self.size), Image.NEAREST)
        return image, trimap

    def __getitem__(self, index):
        image_id, class_id = self.ids[index]
        image, trimap = self._open_pair(image_id)
        image, trimap = self._geometry(image, trimap)
        image_t = self.normalize(transforms.functional.to_tensor(image))
        if self.task == "cls":
            return image_t, class_id, image_id
        mask = trimap_to_mask(np.array(trimap, dtype=np.int16))
        return image_t, mask, image_id


def scores_from_binary_confusion(conf: torch.Tensor) -> dict:
    conf = conf.cpu().to(torch.float64)
    tp = conf[1, 1]
    fp = conf[0, 1]
    fn = conf[1, 0]
    tn = conf[0, 0]
    union_fg = tp + fp + fn
    union_bg = tn + fp + fn
    fg_iou = float((tp / union_fg.clamp(min=1e-12)).item()) if float(union_fg) > 0 else float("nan")
    bg_iou = float((tn / union_bg.clamp(min=1e-12)).item()) if float(union_bg) > 0 else float("nan")
    dice_den = 2.0 * tp + fp + fn
    dice = float((2.0 * tp / dice_den.clamp(min=1e-12)).item()) if float(dice_den) > 0 else float("nan")
    ious = [x for x in (bg_iou, fg_iou) if x == x]
    miou = float(sum(ious) / len(ious)) if ious else float("nan")
    total = conf.sum().clamp(min=1.0)
    pixel_acc = float(((tn + tp) / total).item())
    return {
        "fg_iou": 100.0 * fg_iou,
        "dice": 100.0 * dice,
        "mIoU": 100.0 * miou,
        "pixel_acc": 100.0 * pixel_acc,
        "bg_iou": 100.0 * bg_iou,
    }
