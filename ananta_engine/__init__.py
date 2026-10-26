"""ANANTA OCR Engine: Unified Layout Analysis & Per-Block Recognition Subsystem.

Provides:
  1. AnantaDocLayout: PP-DocLayoutV3 with 37-class document taxonomy & reading-order head.
  2. AnantaBlockOCR: Qwen3.5-0.8B VLM with 262k Indic & multilingual tokenizer.
  3. AnantaOCR: Combined 2-stage offline pipeline.
  4. AnantaClient: Distributed HTTP serving client.

Usage:
    from ananta_engine import AnantaOCR

    parser = AnantaOCR()
    result = parser.parse("page.png")
    print(result.markdown)
"""

from .engine.types import (
    Block,
    PageResult,
    LayoutConfig,
    CropConfig,
    DedupConfig,
    RecognizerConfig,
    TableFormat,
    DEDUP_MODES,
)
from .engine.offline import (
    AnantaDocLayout,
    AnantaBlockOCR,
    AnantaOCR,
    IndicDocLayout,
    IndicBlockOCR,
    IndicOCR,
)
from .engine.recognizer import HfRecognizer
from .engine.recognizer_vllm import VllmRecognizer
from .serving.client import AnantaClient, OCRClient
from .templates.contract import (
    map_label,
    prompt_for,
    is_transcribed,
    KEPT_BLOCK_TYPES,
    DROP_TYPES,
    OCR_SKIP_LABELS,
)
from .layout.labels import (
    CLASSES,
    LABEL2ID,
    ID2LABEL,
    NUM_CLASSES,
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
    "AnantaDocLayout",
    "AnantaBlockOCR",
    "AnantaOCR",
    "AnantaClient",
    "IndicDocLayout",
    "IndicBlockOCR",
    "IndicOCR",
    "OCRClient",
    "HfRecognizer",
    "VllmRecognizer",
    "map_label",
    "prompt_for",
    "is_transcribed",
    "KEPT_BLOCK_TYPES",
    "DROP_TYPES",
    "OCR_SKIP_LABELS",
    "CLASSES",
    "LABEL2ID",
    "ID2LABEL",
    "NUM_CLASSES",
]
