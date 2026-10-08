"""Templates and Contract definitions for ANANTA OCR."""

from .contract import (
    TableFormat,
    TEXT_PROMPT,
    EQUATION_PROMPT,
    TABLE_PROMPTS,
    LABEL_TO_TYPE,
    KEPT_BLOCK_TYPES,
    DROP_TYPES,
    MARGINALIA,
    HEAD_FOOT,
    OCR_SKIP_LABELS,
    OUTPUT_BLOCK_SCHEMA,
    OUTPUT_PAGE_SCHEMA,
    prompt_for,
    map_label,
    is_transcribed,
)

__all__ = [
    "TableFormat",
    "TEXT_PROMPT",
    "EQUATION_PROMPT",
    "TABLE_PROMPTS",
    "LABEL_TO_TYPE",
    "KEPT_BLOCK_TYPES",
    "DROP_TYPES",
    "MARGINALIA",
    "HEAD_FOOT",
    "OCR_SKIP_LABELS",
    "OUTPUT_BLOCK_SCHEMA",
    "OUTPUT_PAGE_SCHEMA",
    "prompt_for",
    "map_label",
    "is_transcribed",
]
