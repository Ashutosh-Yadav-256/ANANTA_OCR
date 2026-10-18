"""Layout backends for ANANTA OCR: Page image -> cleaned, reading-ordered blocks."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .blocks import clamp_to_page, clean_layout
from .types import Block, DedupConfig, LayoutConfig, PageResult
from ..templates.contract import map_label

if TYPE_CHECKING:
    from PIL.Image import Image


@runtime_checkable
class LayoutBackend(Protocol):
    """Layout detector interface. detect() must return blocks densely ordered with ranks 0..N-1."""

    def detect(self, image: Image) -> list[Block]: ...

    def close(self) -> None: ...


def _densify(blocks: list[Block]) -> list[Block]:
    """Sort blocks by reading order and renumber to a gap-free 0-based rank."""
    ordered = sorted(blocks, key=lambda b: b.order)
    for rank, block in enumerate(ordered):
        block.order = rank
    return ordered


class AnantaDocLayoutBackend:
    """Stage 1: PP-DocLayoutV3 with 37-class document taxonomy and reading-order head."""

    def __init__(
        self,
        ckpt: str | None = None,
        config: LayoutConfig | None = None,
        dedup: DedupConfig | None = None,
    ) -> None:
        from .checkpoints import resolve_ckpt
        from ..layout.infer import get_model

        self.config = config or LayoutConfig()
        self.dedup = dedup or DedupConfig()
        self.ckpt = resolve_ckpt("layout", ckpt)
        self.model = get_model(self.ckpt, device=self.config.device)

    def detect(self, image: Image) -> list[Block]:
        from ..layout.infer import infer

        width, height = image.size
        detections = infer(
            self.model,
            image,
            conf=self.config.conf,
            img_size=self.config.img_size,
            device=self.config.device,
        )

        blocks = []
        for det in detections:
            y0, x0, y1, x1 = det["bbox"]
            bbox = [x0 / 1000 * width, y0 / 1000 * height, x1 / 1000 * width, y1 / 1000 * height]
            label = str(det["label"])
            blocks.append(
                Block(
                    order=det["reading_order"],
                    label=label,
                    type=map_label(label),
                    bbox_xyxy=[round(v, 1) for v in clamp_to_page(bbox, width, height)],
                    conf=round(float(det.get("score", 1.0)), 3),
                )
            )

        return _densify(clean_layout(blocks, self.dedup))

    def close(self) -> None:
        self.model = None


# Backward compatibility alias
IndicDocLayoutBackend = AnantaDocLayoutBackend


class JsonLayoutBackend:
    """Replay an externally produced or pre-computed layout JSON file."""

    def __init__(self, layout: str | dict | PageResult) -> None:
        if isinstance(layout, PageResult):
            self.page = layout
        else:
            if isinstance(layout, str):
                with open(layout, encoding="utf-8") as fh:
                    layout = json.load(fh)
            self.page = PageResult.from_record(layout)

    def detect(self, image: Image) -> list[Block]:
        return _densify([b.copy() for b in self.page.blocks])

    def close(self) -> None:
        return None
