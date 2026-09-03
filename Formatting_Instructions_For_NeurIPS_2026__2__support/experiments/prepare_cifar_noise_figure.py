#!/usr/bin/env python3
"""Prepare the supplied CIFAR noise sweeps for the paper.

The transformation is deliberately pixel-preserving except for removing the
in-plot title, recoloring Old MNE from gold to red, and enlarging label regions
for a two-column-width side-by-side layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


SUPPORT_DIR = Path(__file__).resolve().parents[1]
PAPER_DIR = SUPPORT_DIR.with_name(SUPPORT_DIR.name.removesuffix("_support"))
FIGURE_DIR = PAPER_DIR / "images"
SOURCE_DIR = FIGURE_DIR / "source"

OLD_MNE_GOLD = np.array([230.0, 159.0, 0.0])
OLD_MNE_RED = np.array([214.0, 39.0, 40.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cifar10",
        type=Path,
        default=SOURCE_DIR / "cifar10_vgg16_5seed_post_if.png",
    )
    parser.add_argument(
        "--cifar100",
        type=Path,
        default=SOURCE_DIR / "cifar100_vgg16_5seed_post_if.png",
    )
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument(
        "--crop-top",
        type=int,
        default=64,
        help="Pixels removed above the plotting frame to drop the in-plot title.",
    )
    return parser.parse_args()


def recolor_old_mne(image: Image.Image) -> Image.Image:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)

    # Identify antialiased gold pixels and translucent gold bands as blends of
    # the source color over a near-white plotting background.
    source_delta = 255.0 - OLD_MNE_GOLD
    channel_alpha = (255.0 - pixels) / source_delta
    alpha = np.median(channel_alpha, axis=2)
    reconstructed = 255.0 - alpha[:, :, None] * source_delta
    error = np.max(np.abs(pixels - reconstructed), axis=2)
    mask = (alpha >= 0.025) & (alpha <= 1.025) & (error <= 8.0)

    clipped_alpha = np.clip(alpha, 0.0, 1.0)
    replacement = 255.0 - clipped_alpha[:, :, None] * (255.0 - OLD_MNE_RED)
    pixels[mask] = replacement[mask]
    return Image.fromarray(np.clip(np.rint(pixels), 0, 255).astype(np.uint8), "RGB")


def _enlarge_region(
    image: Image.Image,
    box: tuple[int, int, int, int],
    scale: float,
    *,
    clear: bool,
) -> None:
    region = image.crop(box)
    width = round(region.width * scale)
    height = round(region.height * scale)
    enlarged = region.resize((width, height), Image.Resampling.LANCZOS)
    center_x = (box[0] + box[2]) // 2
    center_y = (box[1] + box[3]) // 2
    left = min(max(0, center_x - width // 2), image.width - width)
    top = min(max(0, center_y - height // 2), image.height - height)
    if clear:
        image.paste("white", box)
    image.paste(enlarged, (left, top))


def enlarge_labels(image: Image.Image) -> Image.Image:
    enlarged = image.copy()
    _enlarge_region(enlarged, (760, 1120, 1150, 1170), 1.20, clear=True)
    _enlarge_region(enlarged, (15, 425, 70, 725), 1.18, clear=True)

    # The legend already masks the underlying curve region. Enlarging the
    # complete rendered box preserves that behavior while improving legibility.
    _enlarge_region(enlarged, (140, 850, 570, 1065), 1.12, clear=True)
    return enlarged


def prepare_panel(path: Path, crop_top: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if crop_top <= 0 or crop_top >= image.height:
        raise ValueError(f"Invalid crop_top={crop_top} for {path} with height {image.height}")
    image = enlarge_labels(recolor_old_mne(image))
    image = image.crop((0, crop_top, image.width, image.height))
    return image


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        prepare_panel(args.cifar10, args.crop_top),
        prepare_panel(args.cifar100, args.crop_top),
    ]
    panel_names = ["cifar10_absolute_noise_panel.png", "cifar100_absolute_noise_panel.png"]
    for image, name in zip(panels, panel_names):
        image.save(args.output_dir / name, optimize=True)

    gutter = 24
    width = sum(image.width for image in panels) + gutter
    height = max(image.height for image in panels)
    combined = Image.new("RGB", (width, height), "white")
    combined.paste(panels[0], (0, 0))
    combined.paste(panels[1], (panels[0].width + gutter, 0))

    png_path = args.output_dir / "cifar10_cifar100_absolute_noise.png"
    pdf_path = args.output_dir / "cifar10_cifar100_absolute_noise.pdf"
    combined.save(png_path, optimize=True)
    combined.save(pdf_path, "PDF", resolution=300.0, quality=95)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
