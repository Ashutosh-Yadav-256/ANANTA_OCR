"""Engine subpackage for ANANTA OCR."""

from .types import (
    Block,
    PageResult,
    LayoutConfig,
    CropConfig,
    DedupConfig,
    RecognizerConfig,
    TableFormat,
    DEDUP_MODES,
)
from .blocks import clean_layout, resolve_nested_equations, clamp_to_page, area, contained_frac
from .crops import crop_for, area_clamp
from .reconstruct import reconstruct, repair_math, dehyphenate
from .layout import LayoutBackend, AnantaDocLayoutBackend, JsonLayoutBackend, IndicDocLayoutBackend
from .recognizer import CropRequest, RecognizerBackend, HfRecognizer, build_requests
from .recognizer_vllm import VllmRecognizer
from .checkpoints import resolve_ckpt
from .offline import (
    AnantaDocLayout,
    AnantaBlockOCR,
    AnantaOCR,
    IndicDocLayout,
    IndicBlockOCR,
    IndicOCR,
)

__all__ = [
    "Block",
    "PageResult",
    "LayoutConfig",
    "CropConfig",
    "DedupConfig",
    "RecognizerConfig",
    "TableFormat",
    "DEDUP_MODES",
    "clean_layout",
    "resolve_nested_equations",
    "clamp_to_page",
    "area",
    "contained_frac",
    "crop_for",
    "area_clamp",
    "reconstruct",
    "repair_math",
    "dehyphenate",
    "LayoutBackend",
    "AnantaDocLayoutBackend",
    "IndicDocLayoutBackend",
    "JsonLayoutBackend",
    "CropRequest",
    "RecognizerBackend",
    "HfRecognizer",
    "build_requests",
    "VllmRecognizer",
    "resolve_ckpt",
    "AnantaDocLayout",
    "AnantaBlockOCR",
    "AnantaOCR",
    "IndicDocLayout",
    "IndicBlockOCR",
    "IndicOCR",
]
