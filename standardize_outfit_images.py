"""
Quick start: pip install pillow rembg numpy
"""
import argparse
import io
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance
from rembg import remove

# Global configuration for polishing behavior
_POLISH_TARGET_SIZE = 800
_POLISH_ALLOW_UPSCALE = False


def polish_image(img: Image.Image) -> Image.Image:
    """Lightly enhance sharpness/contrast and optionally upscale small assets.

    The function preserves alpha where present by working in RGBA.
    """
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    if img.mode == "RGB":
        img = img.convert("RGBA")

    width, height = img.size
    min_side = min(width, height)
    if min_side < _POLISH_TARGET_SIZE and _POLISH_ALLOW_UPSCALE:
        scale = _POLISH_TARGET_SIZE / float(min_side)
        new_size = (int(round(width * scale)), int(round(height * scale)))
        img = img.resize(new_size, Image.LANCZOS)

    img = ImageEnhance.Sharpness(img).enhance(1.1)
    img = ImageEnhance.Contrast(img).enhance(1.1)
    return img


def remove_background(img: Image.Image) -> Image.Image:
    """Remove background using rembg; return RGBA image.

    Falls back to the original polished image if rembg fails.
    """
    try:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        result_bytes = remove(buffer.getvalue())
        result = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
        return result
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] rembg failed; using original image. Error: {exc}", file=sys.stderr)
        return img.convert("RGBA")


def autocrop_rgba(img: Image.Image) -> Image.Image:
    """Crop transparent margins based on alpha channel.

    Returns the original image if no non-transparent pixels are found.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    alpha = np.array(img.split()[-1])
    non_empty = np.argwhere(alpha > 10)
    if non_empty.size == 0:
        return img

    (ymin, xmin), (ymax, xmax) = non_empty.min(axis=0), non_empty.max(axis=0)
    # add 1 to max bounds because slicing is exclusive
    cropped = img.crop((xmin, ymin, xmax + 1, ymax + 1))
    return cropped


def standardize_tile(img: Image.Image, size: int, padding: int, allow_upscale: bool) -> Image.Image:
    """Fit the product into a square transparent canvas with consistent padding."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    available_w = max(size - 2 * padding, 1)
    available_h = max(size - 2 * padding, 1)
    img_w, img_h = img.size

    scale = min(available_w / img_w, available_h / img_h)
    if not allow_upscale:
        scale = min(scale, 1.0)

    new_size = (max(1, int(round(img_w * scale))), max(1, int(round(img_h * scale))))
    resized = img.resize(new_size, Image.LANCZOS)

    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    offset_x = (size - new_size[0]) // 2
    offset_y = (size - new_size[1]) // 2
    canvas.paste(resized, (offset_x, offset_y), resized)
    return canvas


def iter_image_files(input_dir: Path) -> Iterable[Path]:
    """Yield image-like files from directory."""
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            yield path


def process_folder(
    input_dir: Path,
    output_dir: Path,
    size: int,
    padding: int,
    allow_upscale: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    global _POLISH_TARGET_SIZE, _POLISH_ALLOW_UPSCALE
    _POLISH_TARGET_SIZE = size
    _POLISH_ALLOW_UPSCALE = allow_upscale

    for img_path in iter_image_files(input_dir):
        try:
            with Image.open(img_path) as img:
                polished = polish_image(img)
                cutout = remove_background(polished)
                cropped = autocrop_rgba(cutout)
                standardized = standardize_tile(cropped, size=size, padding=padding, allow_upscale=allow_upscale)

                output_name = f"{img_path.stem}_std.png"
                output_path = output_dir / output_name
                standardized.save(output_path, format="PNG")
                print(f"[info] saved {output_path}")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] skipping {img_path.name}: {exc}", file=sys.stderr)


def build_collage(output_dir: Path, size: int, columns: int):
    tiles = sorted(output_dir.glob("*_std.png"))
    if not tiles:
        print("[warn] no standardized images found for collage", file=sys.stderr)
        return

    gap = 20
    rows = math.ceil(len(tiles) / float(columns))
    width = columns * size + (columns + 1) * gap
    height = rows * size + (rows + 1) * gap

    collage = Image.new("RGB", (width, height), (255, 255, 255))

    for idx, tile_path in enumerate(tiles):
        with Image.open(tile_path) as tile:
            tile_rgba = tile.convert("RGBA")
        row, col = divmod(idx, columns)
        x = gap + col * (size + gap)
        y = gap + row * (size + gap)
        collage.paste(tile_rgba, (x, y), tile_rgba)

    collage_path = output_dir / "outfit_collage.png"
    collage.save(collage_path, format="PNG")
    print(f"[info] collage saved to {collage_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardize product images for outfit collages.")
    parser.add_argument("--input", required=True, type=Path, help="Folder with original images")
    parser.add_argument("--output", required=True, type=Path, help="Folder to write standardized PNGs")
    parser.add_argument("--size", type=int, default=800, help="Target square size in pixels (default: 800)")
    parser.add_argument("--padding", type=int, default=40, help="Padding inside the square (default: 40)")
    parser.add_argument("--allow-upscale", action="store_true", help="Allow upscaling images to fill the box")
    parser.add_argument("--make-collage", action="store_true", help="Build a contact-sheet collage after processing")
    parser.add_argument("--columns", type=int, default=4, help="Number of columns for the collage grid (default: 4)")
    return parser.parse_args()


def main():
    args = parse_args()
    process_folder(
        input_dir=args.input,
        output_dir=args.output,
        size=args.size,
        padding=args.padding,
        allow_upscale=args.allow_upscale,
    )

    if args.make_collage:
        build_collage(output_dir=args.output, size=args.size, columns=args.columns)


if __name__ == "__main__":
    main()
