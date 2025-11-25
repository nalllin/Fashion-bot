"""Outfit generation utilities for GlamUp LLM workflows.

This module bundles the Stable Diffusion XL inpainting + IP-Adapter pipeline,
collage-driven product retrieval, and card assembly helpers. It exposes a CLI
compatible entrypoint mirroring the old `outfit_generator.py` script while
keeping the generator logic centralized.
"""
from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from diffusers import AutoPipelineForInpainting, AutoencoderKL
from PIL import Image, ImageDraw, ImageOps

try:  # Optional, used for automatic mask generation
    from rembg import remove as rembg_remove
except Exception:  # pragma: no cover - optional dependency
    rembg_remove = None

LOGGER = logging.getLogger(__name__)


def parse_crop_boxes(raw_boxes: Optional[Sequence[str]]) -> List[Tuple[int, int, int, int]]:
    """Parse a list of crop box strings formatted as "x1,y1,x2,y2"."""
    boxes: List[Tuple[int, int, int, int]] = []
    for raw in raw_boxes or []:
        parts = raw.split(",")
        if len(parts) != 4:
            raise ValueError(f"Invalid crop box '{raw}'. Expected format x1,y1,x2,y2")
        try:
            box = tuple(int(p) for p in parts)  # type: ignore[assignment]
        except ValueError as exc:  # pragma: no cover - invalid input path
            raise ValueError(f"Crop box values must be integers: {raw}") from exc
        boxes.append(box)
    return boxes


def crop_products_from_collage(
    collage_path: Path, crop_boxes: Sequence[Tuple[int, int, int, int]], temp_dir: Path
) -> List[Path]:
    """Crop product photos from a collage image into temporary files."""
    collage = Image.open(collage_path).convert("RGB")
    results: List[Path] = []
    for idx, box in enumerate(crop_boxes):
        crop = collage.crop(box)
        out_path = temp_dir / f"product_{idx}.png"
        crop.save(out_path)
        results.append(out_path)
        LOGGER.info("Saved cropped product %s to %s", idx, out_path)
    return results


def gather_product_images(
    product_images: Optional[Sequence[str]],
    collage_path: Optional[str],
    crop_boxes: Optional[Sequence[str]],
    temp_dir: Path,
) -> List[Path]:
    """Unify direct item paths and collage crops into a single product list."""
    product_paths: List[Path] = []

    for path in product_images or []:
        as_path = Path(path)
        if not as_path.exists():
            raise FileNotFoundError(f"Product image not found: {as_path}")
        product_paths.append(as_path)

    if collage_path:
        collage_file = Path(collage_path)
        if not collage_file.exists():
            raise FileNotFoundError(f"Collage image not found: {collage_file}")
        if not crop_boxes:
            raise ValueError("--collage requires --crop-boxes to know where to crop items")
        parsed_boxes = parse_crop_boxes(crop_boxes)
        product_paths.extend(crop_products_from_collage(collage_file, parsed_boxes, temp_dir))

    if not product_paths:
        raise ValueError("No product images provided. Use --items or --collage with --crop-boxes.")

    return product_paths


def load_pipeline(device: str, dtype: torch.dtype, offload_cpu: bool) -> AutoPipelineForInpainting:
    """Load the SDXL inpainting pipeline with the IP-Adapter attached."""
    LOGGER.info("Loading VAE and inpainting pipeline to %s", device)
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype
    )
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        vae=vae,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.bin",
        low_cpu_mem_usage=True,
    )
    if device == "cuda":
        pipe = pipe.to(device)
    elif offload_cpu:
        pipe.enable_model_cpu_offload()
    return pipe


def choose_device(force_cpu: bool) -> tuple[str, torch.dtype, bool]:
    """Pick the execution device and dtype."""
    if torch.cuda.is_available() and not force_cpu:
        return "cuda", torch.float16, False
    LOGGER.warning("CUDA unavailable. Falling back to CPU; generation will be slower.")
    return "cpu", torch.float32, True


def load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def generate_mask(
    base_image: Image.Image,
    provided_mask: Optional[Path] = None,
    preferred_size: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    """Generate an inpainting mask.

    Priority:
    1) user-supplied mask path
    2) automatic segmentation via rembg (if installed)
    3) default full-image mask
    """
    target_size = preferred_size or base_image.size
    if provided_mask:
        mask = Image.open(provided_mask).convert("L").resize(target_size)
        return mask

    if rembg_remove is not None:
        LOGGER.info("Generating mask with rembg")
        cutout = rembg_remove(base_image)
        if cutout.mode != "RGBA":
            cutout = cutout.convert("RGBA")
        mask = cutout.getchannel("A")
        mask = ImageOps.invert(mask)
        return mask

    LOGGER.warning("rembg not installed; using full-image mask.")
    return Image.new("L", target_size, color=255)


def apply_item_to_model(
    pipe: AutoPipelineForInpainting,
    model_image: Image.Image,
    mask_image: Image.Image,
    item_image: Image.Image,
    num_inference_steps: int,
) -> Image.Image:
    """Run the inpainting pipeline to transfer a garment onto the model."""
    result = pipe(
        prompt="",  # no text prompt needed thanks to IP-Adapter guidance
        image=model_image,
        mask_image=mask_image,
        ip_adapter_image=item_image,
        num_inference_steps=num_inference_steps,
    ).images[0]
    return result


def build_outfit_card(
    composed_image: Image.Image,
    item_images: Sequence[Image.Image],
    output_path: Path,
    title: str = "Outfit Card",
) -> None:
    """Assemble the final outfit card with thumbnails on the side."""
    padding = 32
    thumb_size = (160, 160)
    column_width = thumb_size[0] + padding * 2

    card_width = composed_image.width + column_width + padding
    card_height = max(
        composed_image.height + padding * 2,
        (thumb_size[1] + padding) * len(item_images) + padding,
    )

    canvas = Image.new("RGB", (card_width, card_height), color=(247, 246, 244))
    draw = ImageDraw.Draw(canvas)

    title_pos = (padding, padding // 2)
    draw.text(title_pos, title, fill=(33, 33, 33))

    composed_top = (card_height - composed_image.height) // 2
    canvas.paste(composed_image, (column_width, composed_top))

    y = padding
    for idx, img in enumerate(item_images):
        thumb = ImageOps.fit(img, thumb_size)
        frame = Image.new("RGB", (thumb_size[0], thumb_size[1]), color=(255, 255, 255))
        frame.paste(thumb, (0, 0))
        canvas.paste(frame, (padding, y))
        draw.text((padding, y + thumb_size[1] + 8), f"Item {idx + 1}", fill=(80, 80, 80))
        y += thumb_size[1] + padding

    canvas.save(output_path)
    LOGGER.info("Saved outfit card to %s", output_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Create an outfit card via SDXL inpainting + IP-Adapter")
    parser.add_argument("--base", dest="base_model_image_path", required=True, help="Path to the base model image")
    parser.add_argument("--items", nargs="*", dest="product_images", help="Paths to cropped product photos")
    parser.add_argument("--collage", dest="collage_path", help="Optional collage containing all product items")
    parser.add_argument(
        "--crop-boxes",
        nargs="*",
        dest="crop_boxes",
        help="Crop boxes (x1,y1,x2,y2) for extracting products from the collage",
    )
    parser.add_argument(
        "--masks",
        nargs="*",
        dest="mask_paths",
        help="Optional mask images matching the product list order",
    )
    parser.add_argument("--output", dest="output_path", required=True, help="Output path for the final outfit card")
    parser.add_argument("--num-inference-steps", type=int, default=40, help="Diffusion inference steps")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution even if CUDA is available")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    device, dtype, offload_cpu = choose_device(args.cpu)
    base_model_image_path = Path(args.base)
    output_path = Path(args.output)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        product_paths = gather_product_images(
            product_images=args.product_images,
            collage_path=args.collage_path,
            crop_boxes=args.crop_boxes,
            temp_dir=temp_dir,
        )

        if args.mask_paths and len(args.mask_paths) not in (1, len(product_paths)):
            raise ValueError("--masks must be either a single mask or match the number of products")
        mask_paths = [Path(p) for p in args.mask_paths] if args.mask_paths else []

        base_image = load_image(base_model_image_path)
        pipe = load_pipeline(device, dtype, offload_cpu)

        composed = base_image
        applied_items: List[Image.Image] = []
        for idx, item_path in enumerate(product_paths):
            LOGGER.info("Applying item %s from %s", idx + 1, item_path)
            item_img = load_image(item_path)
            applied_items.append(item_img)

            mask_override = None
            if mask_paths:
                mask_override = mask_paths[min(idx, len(mask_paths) - 1)]
            mask = generate_mask(composed, provided_mask=mask_override, preferred_size=composed.size)

            composed = apply_item_to_model(
                pipe=pipe,
                model_image=composed,
                mask_image=mask,
                item_image=item_img,
                num_inference_steps=args.num_inference_steps,
            )

        build_outfit_card(composed, applied_items, output_path)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
