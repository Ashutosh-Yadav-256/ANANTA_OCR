"""Layout cleanup: Geometry filtering, duplicate suppression, nested-equation resolution.

Pure geometric operations on Block objects -- no PIL, PyTorch, or GPU requirements.
Keeps layout post-processing testable and fast in any Python environment.
"""

from __future__ import annotations

from .types import Block, DedupConfig
from ..templates.contract import HEAD_FOOT, MARGINALIA, is_transcribed

TEXTLIKE = frozenset({"Text", "Title", "SectionHeader", "Caption", "Footnote"})


def area(bbox: list[float]) -> float:
    """Compute width * height area of [x0, y0, x1, y1] bbox."""
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def contained_frac(small: list[float], big: list[float]) -> float:
    """Fraction of 'small' bbox area that lies inside 'big' bbox."""
    ix0, iy0 = max(small[0], big[0]), max(small[1], big[1])
    ix1, iy1 = min(small[2], big[2]), min(small[3], big[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    a = area(small)
    return inter / a if a > 0 else 0.0


def clamp_to_page(bbox: list[float], width: int, height: int) -> list[float]:
    """Clamp bounding box coordinates to image dimensions [0, width, 0, height]."""
    return [
        max(0.0, bbox[0]),
        max(0.0, bbox[1]),
        min(float(width), bbox[2]),
        min(float(height), bbox[3]),
    ]


def clean_layout(blocks: list[Block], cfg: DedupConfig | None = None) -> list[Block]:
    """Drop duplicate and spurious boxes while preserving reading order.

    Rules applied:
      1. Nested duplicates: A box >= cfg.contain inside a larger box of the same class group is removed.
      2. Headers/Footers: Only the largest header and largest footer survive.
      3. Empty frames: Header/footer boxes that wrap no actual content are dropped.
    """
    cfg = cfg or DedupConfig()
    n = len(blocks)
    box = lambda i: blocks[i].bbox_xyxy  # noqa: E731
    drop: set[int] = set()

    # Rule 1: Non-marginalia vs marginalia groupings
    groups = (
        lambda label: label not in MARGINALIA,
        lambda label: label in MARGINALIA and label not in HEAD_FOOT,
    )
    for in_group in groups:
        idxs = sorted(
            (i for i in range(n) if in_group(blocks[i].label)),
            key=lambda i: area(box(i)),
            reverse=True,
        )
        kept: list[int] = []
        for i in idxs:
            has_text = is_transcribed(blocks[i].label)
            # A block is never absorbed into a container that will not itself be transcribed
            if any(
                contained_frac(box(i), box(j)) >= cfg.contain
                and (is_transcribed(blocks[j].label) or not has_text)
                for j in kept
            ):
                drop.add(i)
            else:
                kept.append(i)

    # Rule 2 & 3: Header and Footer containment and deduplication
    for label in HEAD_FOOT:
        group = [i for i in range(n) if blocks[i].label == label and i not in drop]
        if not group:
            continue
        biggest = max(group, key=lambda i: area(box(i)))
        drop.update(i for i in group if i != biggest)
        
        # Check if header/footer encloses content
        wraps = any(
            blocks[k].label != label and contained_frac(box(k), box(biggest)) >= cfg.wrap
            for k in range(n)
        )
        if not wraps:
            drop.add(biggest)

    return [blocks[i] for i in range(n) if i not in drop]


def _nested_in(inner: list[float], outer: list[float], thresh: float) -> bool:
    inner_area = area(inner)
    if inner_area <= 0:
        return False
    return contained_frac(inner, outer) >= thresh and inner_area < 0.95 * area(outer)


def _absorbs_equation(container: Block, mode: str) -> bool:
    if container.type == "Equation":
        return mode != "text_only"
    if container.type in TEXTLIKE:
        return mode != "eq_only"
    return False


def resolve_nested_equations(blocks: list[Block], cfg: DedupConfig | None = None) -> list[Block]:
    """Drop equation boxes nested in containers (paragraphs/display math) that already cover them.
    
    Prevents duplicate transcription where the text prompt outputs inline math and the
    detector also spawned an isolated equation box.
    """
    cfg = cfg or DedupConfig()
    if not cfg.nest:
        return list(blocks)

    drop: set[int] = set()
    for i, inner in enumerate(blocks):
        if inner.type != "Equation":
            continue
        for j, container in enumerate(blocks):
            if (
                i != j
                and _nested_in(inner.bbox_xyxy, container.bbox_xyxy, cfg.nested)
                and _absorbs_equation(container, cfg.mode)
            ):
                drop.add(i)
                break
    return [b for k, b in enumerate(blocks) if k not in drop]

