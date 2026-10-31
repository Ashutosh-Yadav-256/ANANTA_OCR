"""ANANTA OCR: Advanced Unified Visual Document Intelligence.

Combines:
  1. Ananta Multi-Page VLM: Unlimited-horizon multi-page parsing (100+ pages) with 16x spatial compression.
  2. Ananta 2-Stage Block Engine: High-precision 37-class layout analysis (AnantaDocLayout) + multilingual block transcription (AnantaBlockOCR).
  3. Ananta Hybrid Engine: Reading-order guided layout segmentation + intelligent multi-engine transcription.
"""

from .ananta_engine import (
    Block,
    PageResult,
    LayoutConfig,
    CropConfig,
    DedupConfig,
    RecognizerConfig,
    TableFormat,
    AnantaDocLayout,
    AnantaBlockOCR,
    AnantaOCR,
    AnantaClient,
    HfRecognizer,
    VllmRecognizer,
    CLASSES,
    LABEL2ID,
    ID2LABEL,
    NUM_CLASSES,
)
from .unified_pipeline import (
    UnifiedOCR,
    RegionResult,
    classify_hybrid_block,
    HYBRID_BLOCK_CLASSES,
    HYBRID_NARRATIVE_CLASSES,
)

__all__ = [
    "AnantaOCR",
    "AnantaDocLayout",
    "AnantaBlockOCR",
    "AnantaClient",
    "UnifiedOCR",
    "RegionResult",
    "classify_hybrid_block",
    "HYBRID_BLOCK_CLASSES",
    "HYBRID_NARRATIVE_CLASSES",
    "Block",
    "PageResult",
    "LayoutConfig",
    "CropConfig",
    "DedupConfig",
    "RecognizerConfig",
    "TableFormat",
    "HfRecognizer",
    "VllmRecognizer",
    "CLASSES",
    "LABEL2ID",
    "ID2LABEL",
    "NUM_CLASSES",
]

