"""Unified offline execution pipeline for ANANTA OCR: AnantaDocLayout, AnantaBlockOCR, and AnantaOCR.

Handles both stages:
  - Stage 1: AnantaDocLayout (page image -> PageResult with blocks and reading order)
  - Stage 2: AnantaBlockOCR (page image + layout -> PageResult with transcribed text and Markdown)
  - AnantaOCR: End-to-end pipeline running both stages
"""

from __future__ import annotations

import json
import os

from .blocks import resolve_nested_equations
from .reconstruct import reconstruct
from .types import (
    CropConfig,
    DedupConfig,
    LayoutConfig,
    PageResult,
    RecognizerConfig,
)
from ..templates.contract import is_transcribed


DEFAULT_MAX_IMAGE_PIXELS = int(os.environ.get("ANANTA_MAX_IMAGE_PIXELS", "100000000"))
DEFAULT_MAX_IMAGE_BYTES = int(os.environ.get("ANANTA_MAX_IMAGE_BYTES", str(100 * 1024 * 1024)))

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = DEFAULT_MAX_IMAGE_PIXELS
except ImportError:
    pass


def _open(
    image_path: str,
    max_pixels: int | None = None,
    max_bytes: int | None = None,
):
    from PIL import Image
    from pathlib import Path

    p = Path(image_path)
    byte_limit = max_bytes if max_bytes is not None else DEFAULT_MAX_IMAGE_BYTES
    if byte_limit and p.is_file() and p.stat().st_size > byte_limit:
        raise ValueError(
            f"Image file size ({p.stat().st_size} bytes) exceeds safety limit of {byte_limit} bytes."
        )

    limit = max_pixels if max_pixels is not None else DEFAULT_MAX_IMAGE_PIXELS
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            if limit is not None and (w * h) > limit:
                raise ValueError(
                    f"Image dimensions ({w}x{h} = {w * h} pixels) exceed safety limit of {limit} pixels. "
                    "Set ANANTA_MAX_IMAGE_PIXELS environment variable to adjust."
                )
            return img.convert("RGB")
    except Image.DecompressionBombError as err:
        raise ValueError(
            f"Image exceeds safety limit of {limit} pixels: {err}. "
            "Set ANANTA_MAX_IMAGE_PIXELS environment variable to adjust."
        ) from err


def _as_page(layout: PageResult | dict | str) -> PageResult:
    if isinstance(layout, PageResult):
        return layout
    if isinstance(layout, str):
        with open(layout, encoding="utf-8") as fh:
            layout = json.load(fh)
    return PageResult.from_record(layout)


class AnantaDocLayout:
    """Stage 1: Document layout analysis, bounding box detection, and reading order ordering."""

    def __init__(
        self,
        ckpt: str | None = None,
        config: LayoutConfig | None = None,
        dedup: DedupConfig | None = None,
        backend=None,
    ) -> None:
        if backend is None:
            from .layout import AnantaDocLayoutBackend

            backend = AnantaDocLayoutBackend(ckpt, config, dedup)
        self.backend = backend

    def detect(self, image_path: str) -> PageResult:
        image = _open(image_path)
        return PageResult(
            image=os.path.basename(image_path),
            width=image.width,
            height=image.height,
            blocks=self.backend.detect(image),
        )

    def close(self) -> None:
        if self.backend:
            self.backend.close()

    def __enter__(self) -> AnantaDocLayout:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class AnantaBlockOCR:
    """Stage 2: Per-block transcription with specialized prompts for text, equations, and tables."""

    def __init__(
        self,
        ckpt: str | None = None,
        config: RecognizerConfig | None = None,
        dedup: DedupConfig | None = None,
        crop: CropConfig | None = None,
        backend=None,
    ) -> None:
        self.config = config or RecognizerConfig()
        self.dedup = dedup or DedupConfig()
        self.crop = crop or CropConfig()

        if backend is None:
            backend = self._init_default_backend(ckpt)
        self.backend = backend

    def _init_default_backend(self, ckpt: str | None):
        try:
            import torch
            if torch.cuda.is_available():
                try:
                    from .recognizer_vllm import VllmRecognizer
                    return VllmRecognizer(ckpt, self.config)
                except (ImportError, Exception):
                    pass
        except ImportError:
            pass

        from .recognizer import HfRecognizer
        device = "cuda" if _has_cuda() else "cpu"
        return HfRecognizer(ckpt, self.config, device=device)

    def run(self, image_path: str, layout: PageResult | dict | str) -> PageResult:
        from .recognizer import build_requests

        image = _open(image_path)
        page = _as_page(layout)
        blocks = [b.copy() for b in page.blocks]

        eligible = resolve_nested_equations(
            [b for b in blocks if is_transcribed(b.label)], self.dedup
        )
        requests, orders = build_requests(eligible, image, self.crop, self.config.table_format)
        texts = self.backend.transcribe(requests)

        if len(texts) != len(orders):
            raise RuntimeError(
                f"Recognizer returned {len(texts)} transcriptions for {len(orders)} crops; "
                "results would be misaligned."
            )

        by_order = dict(zip(orders, texts, strict=True))
        for block in blocks:
            block.text = (by_order.get(block.order) or "").strip()

        return PageResult(
            image=page.image,
            width=page.width,
            height=page.height,
            blocks=blocks,
            markdown=reconstruct(blocks),
        )

    def close(self) -> None:
        if self.backend:
            self.backend.close()

    def __enter__(self) -> AnantaBlockOCR:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class AnantaOCR:
    """Complete 2-stage ANANTA OCR pipeline combining AnantaDocLayout and AnantaBlockOCR."""

    def __init__(
        self,
        layout_ckpt: str | None = None,
        recognizer_ckpt: str | None = None,
        layout_config: LayoutConfig | None = None,
        recognizer_config: RecognizerConfig | None = None,
        dedup: DedupConfig | None = None,
        crop: CropConfig | None = None,
        layout_backend=None,
        recognizer_backend=None,
    ) -> None:
        self.ocr = AnantaBlockOCR(
            ckpt=recognizer_ckpt,
            config=recognizer_config,
            dedup=dedup,
            crop=crop,
            backend=recognizer_backend,
        )
        self.layout = AnantaDocLayout(
            ckpt=layout_ckpt,
            config=layout_config,
            dedup=dedup,
            backend=layout_backend,
        )

    def detect(self, image_path: str) -> PageResult:
        """Run Stage 1 only."""
        return self.layout.detect(image_path)

    def parse(self, image_path: str) -> PageResult:
        """Run Stage 1 and Stage 2 -> PageResult."""
        layout_res = self.layout.detect(image_path)
        return self.ocr.run(image_path, layout_res)

    def __call__(self, image_path: str) -> str:
        """Convenience method returning markdown text directly."""
        return self.parse(image_path).markdown or ""

    def close(self) -> None:
        self.ocr.close()
        self.layout.close()

    def __enter__(self) -> AnantaOCR:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


# Backward compatibility aliases
IndicDocLayout = AnantaDocLayout
IndicBlockOCR = AnantaBlockOCR
IndicOCR = AnantaOCR
