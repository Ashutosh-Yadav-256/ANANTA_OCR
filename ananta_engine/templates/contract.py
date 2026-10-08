"""ANANTA OCR Contract: Prompts, Block Taxonomy, and Output Schema.

Defines the interface between AnantaDocLayout and AnantaBlockOCR:
  - Labels: Raw 37-class document taxonomy.
  - Types: Coarse functional categories used for prompt selection.
  - Prompts: Specialized prompt templates for Text, Equation, and Table blocks.

Stdlib-only -- zero dependencies, importable without PyTorch/CUDA/PIL.
"""

from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        pass


class TableFormat(StrEnum):
    """HTML is default: colspan/rowspan and cell breaks are preserved.
    Markdown table cannot express merged cells or multi-line cell content.
    """
    HTML = "html"
    MARKDOWN = "markdown"


TEXT_PROMPT = (
    "Transcribe the text in this image. Write any mathematical expressions in LaTeX, "
    "using $...$ for inline math and $$...$$ for display equations."
)

EQUATION_PROMPT = "Output only the LaTeX for this equation image."

TABLE_PROMPTS = {
    TableFormat.HTML: (
        "Convert this table image to HTML. Preserve the structure exactly, using colspan and "
        "rowspan for merged cells and <br/> for line breaks within a cell. "
        "Output only the HTML table."
    ),
    TableFormat.MARKDOWN: (
        "Convert this table image to a GitHub-flavored markdown table. Output only the table."
    ),
}


def prompt_for(block_type: str, table_format: TableFormat = TableFormat.HTML) -> str:
    """Select prompt based on block type and table formatting preference."""
    if block_type == "Table":
        return TABLE_PROMPTS[TableFormat(table_format)]
    if block_type == "Equation":
        return EQUATION_PROMPT
    return TEXT_PROMPT


# Label mapping from 37-class layout taxonomy to coarse pipeline types
LABEL_TO_TYPE = {
    "table": "Table",
    "table-caption": "Caption",
    "equation": "Equation",
    "expression": "Equation",
    "diagram": "Figure",
    "chart": "Figure",
    "image": "Picture",
    "image-caption": "Caption",
    "title": "Title",
    "chapter-title": "Title",
    "section-title": "SectionHeader",
    "sub-section-title": "SectionHeader",
    "sub-sub-section-title": "SectionHeader",
    "header": "PageHeader",
    "footer": "PageFooter",
    "page-number": "PageNumber",
    "folio": "PageNumber",
    "footnote": "Footnote",
}


def map_label(label: str) -> str:
    """Map raw detector label -> pipeline block type. Unknown labels default to 'Text'."""
    return LABEL_TO_TYPE.get(str(label).strip().lower(), "Text")


KEPT_BLOCK_TYPES = (
    "Text",
    "Title",
    "SectionHeader",
    "Table",
    "Equation",
    "Caption",
    "Footnote",
    "PageHeader",
    "PageFooter",
    "PageNumber",
)

# Regions never cropped or transcribed (purely visual elements to avoid hallucination)
DROP_TYPES = frozenset({"Figure", "Picture"})

# Cleaned as their own group so a page-spanning paragraph does not swallow page numbers
MARGINALIA = frozenset({"Header", "Footer", "Page-number", "Folio"})
HEAD_FOOT = ("Header", "Footer")

# Preserved in output JSON with text="" for layout fidelity, but skipped during OCR
OCR_SKIP_LABELS = frozenset({"header", "footer", "diagram", "image", "chart", "advertisement"})


def is_transcribed(label: str) -> bool:
    """Return True if this label should be passed to the recognizer model."""
    return str(label).strip().lower() not in OCR_SKIP_LABELS


OUTPUT_BLOCK_SCHEMA = {
    "order": "int    -- reading-order rank (0 = first)",
    "label": "str    -- raw AnantaDocLayout label",
    "type": "str    -- pipeline type (see KEPT_BLOCK_TYPES)",
    "bbox_xyxy": "[float x4] -- pixel [x0, y0, x1, y1], clamped to page",
    "conf": "float  -- detection confidence score [0.0, 1.0]",
    "text": "str    -- transcribed text; '' if not transcribed",
}

OUTPUT_PAGE_SCHEMA = {
    "image": "str   -- source image filename",
    "width": "int   -- page width in pixels",
    "height": "int   -- page height in pixels",
    "blocks": "[BLOCK] -- detected blocks sorted in reading order",
}
