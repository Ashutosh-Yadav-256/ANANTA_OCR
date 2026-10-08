"""Transform bounding boxes into optimal crops for the recognizer model.

Preserves geometry rules separate from PIL so geometry can run headless without image libraries.
Uses area-based clamping to ensure model token efficiency.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .types import Block, CropConfig
from ..templates.contract import DROP_TYPES, is_transcribed

if TYPE_CHECKING:
    from PIL.Image import Image


def area_clamp(image: Image, cfg: CropConfig | None = None) -> Image:
    """Scale a crop so its pixel area falls within [cfg.min_px, cfg.max_px], preserving aspect ratio.

    Clamping on area rather than width/height prevents token explosion on extreme aspect ratios
    (e.g., long horizontal divider rules or thin vertical borders).
    """
    from PIL import Image as PILImage

    cfg = cfg or CropConfig()
    width, height = image.size
    pixels = width * height
    if pixels <= 0:
        return image

    if pixels < cfg.min_px:
        scale = math.sqrt(cfg.min_px / pixels)
    elif pixels > cfg.max_px:
        scale = math.sqrt(cfg.max_px / pixels)
    else:
        return image

    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    return image.resize((new_w, new_h), PILImage.LANCZOS)


def crop_for(block: Block, page: Image, cfg: CropConfig | None = None) -> Image | None:
    """Extract and pad the image crop for a given block, or None if skipped/invalid."""
    if block.type in DROP_TYPES or not is_transcribed(block.label):
        return None

    cfg = cfg or CropConfig()
    width, height = page.size
    x0, y0, x1, y1 = (round(v) for v in block.bbox_xyxy)
    if cfg.pad_px:
        x0, y0 = x0 - cfg.pad_px, y0 - cfg.pad_px
        x1, y1 = x1 + cfg.pad_px, y1 + cfg.pad_px
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)

    if x1 <= x0 or y1 <= y0:
        return None
    return page.crop((x0, y0, x1, y1)).convert("RGB")

