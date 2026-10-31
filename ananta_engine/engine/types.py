"""Plain-data types and configuration classes for ANANTA OCR.

All pipeline tuning parameters reside in typed dataclasses rather than environment variables,
ensuring determinism, programmatic configurability, and testability.

Stdlib-only -- zero dependencies, importable without PyTorch/CUDA/PIL.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from ..templates.contract import TableFormat, map_label


@dataclass
class Block:
    """A detected document region.
    
    Attributes:
        order: Reading order rank (0 = first).
        label: Fine-grained class label from the 37-class layout detector.
        type: Coarse category used to route prompts (Text, Equation, Table, etc.).
        bbox_xyxy: Pixel bounding box [x0, y0, x1, y1] clamped to image dimensions.
        conf: Detector confidence score [0.0, 1.0].
        text: Transcribed content, None before OCR, or '' for non-transcribed blocks.
    """

    order: int
    label: str
    type: str = ""
    bbox_xyxy: list[float] = field(default_factory=list)
    conf: float = 1.0
    text: str | None = None

    def __post_init__(self):
        if not self.type and self.label:
            self.type = map_label(self.label)

    def as_record(self) -> dict:
        """Serialize block to a stable dictionary format with rounded floats."""
        record: dict = {
            "order": self.order,
            "label": self.label,
            "type": self.type,
            "bbox_xyxy": [round(float(v), 1) for v in self.bbox_xyxy],
            "conf": round(float(self.conf), 3),
        }
        if self.text is not None:
            record["text"] = self.text
        return record

    @classmethod
    def from_record(cls, record: dict) -> Block:
        """Construct a Block from a serialized dictionary record."""
        label = record.get("label", "")
        return cls(
            order=int(record["order"]),
            label=str(label),
            type=str(record.get("type") or map_label(label)),
            bbox_xyxy=[float(v) for v in record["bbox_xyxy"]],
            conf=float(record.get("conf", 1.0)),
            text=record.get("text"),
        )

    def copy(self) -> Block:
        return dataclasses.replace(self, bbox_xyxy=list(self.bbox_xyxy))


@dataclass
class PageResult:
    """Document page parsing result containing all blocks and assembled markdown."""

    image: str = ""
    width: int = 0
    height: int = 0
    blocks: list[Block] = field(default_factory=list)
    markdown: str | None = None
    reading_order: list[int] | None = None

    def __init__(
        self,
        image: str | None = None,
        width: int = 0,
        height: int = 0,
        blocks: list[Block] | None = None,
        markdown: str | None = None,
        image_path: str | None = None,
        image_size: tuple[int, int] | list[int] | None = None,
        reading_order: list[int] | None = None,
    ) -> None:
        self.image = image if image is not None else (image_path if image_path is not None else "")
        if image_size is not None and len(image_size) >= 2:
            self.width = int(image_size[0])
            self.height = int(image_size[1])
        else:
            self.width = int(width)
            self.height = int(height)
        self.blocks = blocks if blocks is not None else []
        self.markdown = markdown
        self.reading_order = reading_order

    @property
    def image_path(self) -> str:
        return self.image

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def as_record(self) -> dict:
        return {
            "image": self.image,
            "width": self.width,
            "height": self.height,
            "blocks": [b.as_record() for b in self.blocks],
        }

    @classmethod
    def from_record(cls, record: dict) -> PageResult:
        return cls(
            image=str(record.get("image", "")),
            width=int(record.get("width", 0)),
            height=int(record.get("height", 0)),
            blocks=[Block.from_record(b) for b in record.get("blocks", [])],
        )


@dataclass(frozen=True)
class LayoutConfig:
    """Configuration for Stage 1 AnantaDocLayout detection."""

    conf: float = 0.5  # Detection confidence threshold
    img_size: int = 1024  # Square input resolution for RT-DETR
    device: str = "cuda"  # "cuda" or "cpu"


@dataclass(frozen=True)
class CropConfig:
    """Area-based clamping configuration for block crops passed to the recognizer.
    
    Clamping is based on total area rather than dimension to avoid exploding tokens
    on narrow lines/rules.
    """

    min_px_side: int = 256  # Minimum equivalent square dimension (upscaling floor)
    max_px_side: int = 1536  # Maximum equivalent square dimension (token ceiling)
    pad_px: int = 0  # Context padding around bounding box

    @property
    def min_px(self) -> int:
        return self.min_px_side**2

    @property
    def max_px(self) -> int:
        return self.max_px_side**2


#: Supported deduplication modes
DEDUP_MODES = ("both", "text_only", "eq_only")


@dataclass(frozen=True)
class DedupConfig:
    """Configuration for geometric deduplication and nested-box suppression."""

    nest: bool = True  # Suppress equations nested inside paragraphs
    mode: str = "both"  # "both", "text_only", or "eq_only"
    contain: float = 0.90  # Containment threshold for duplicate suppression
    wrap: float = 0.50  # Header/footer wrapping threshold
    nested: float = 0.70  # Threshold for nested equation absorption

    def __post_init__(self) -> None:
        if self.mode not in DEDUP_MODES:
            raise ValueError(f"DedupConfig.mode must be one of {DEDUP_MODES}, got {self.mode!r}")


@dataclass(frozen=True)
class RecognizerConfig:
    """Configuration for Stage 2 AnantaBlockOCR."""

    max_model_len: int = 8192
    max_tokens: int = 2048
    temperature: float = 0.0  # Greedy decoding for determinism
    gpu_memory_utilization: float = 0.80
    dtype: str = "bfloat16"
    batch_size: int = 2048  # Sub-batch limit to avoid scheduler stalls
    enforce_eager: bool = True  # Skips torch.compile overhead on 0.8B model
    table_format: TableFormat = TableFormat.HTML

    def merged(self, **overrides) -> RecognizerConfig:
        """Return a new RecognizerConfig with non-None overrides applied."""
        known = {f.name for f in dataclasses.fields(self)}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise TypeError(f"Unknown RecognizerConfig field(s): {', '.join(unknown)}")
        return dataclasses.replace(self, **{k: v for k, v in overrides.items() if v is not None})
