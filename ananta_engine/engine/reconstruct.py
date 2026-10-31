"""Assemble transcribed blocks into reading-ordered, LaTeX-compliant Markdown.

Includes specialized repairs for Indic script characters in math mode and line-break dehyphenation.
Stdlib-only -- zero dependencies.
"""

from __future__ import annotations

import re

from .types import Block
from ..templates.contract import DROP_TYPES

# Dashes as codepoints: 002D HYPHEN-MINUS | 2010-2014 HYPHEN..EM DASH | 2212 MINUS SIGN
_HYPHEN_BREAK = re.compile(r"(\w)[-\u2010-\u2014\u2212]\n[ \t]*(\w)")
_MATH_DELIMITERS = ("$", "\\[", "\\(")

# A math span: $$...$$ (display) or $...$ (inline)
_MATH_SPAN = re.compile(r"\$\$(.+?)\$\$|(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)", re.S)

# Comprehensive Unicode coverage for all 22 official Indian languages and scripts:
# - Arabic & Perso-Arabic (Urdu, Kashmiri, Sindhi): U+0600-06FF, U+0750-077F, U+08A0-08FF, U+FB50-FDFF, U+FE70-FEFF
# - Indic Major Scripts (Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam): U+0900-0D7F
# - Vedic Extensions: U+1CD0-1CFF
# - Devanagari Extended: U+A8E0-A8FF
# - Ol Chiki (Santali): U+1C50-1C7F
# - Meetei Mayek (Manipuri): U+ABC0-ABFF, U+AAE0-AAFF
# - Tirhuta (Maithili): U+11480-114DF
# - Grantha / Sharada: U+11300-1137F, U+11180-111DF
_INDIC_RANGES = (
    r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff"
    r"\u0900-\u0d7f\u1cd0-\u1cff\ua8e0-\ua8ff\u1c50-\u1c7f\uabc0-\uabff\uaae0-\uaaff"
    r"\U00011480-\U000114df\U00011180-\U000111df\U00011300-\U0001137f]"
)
_NON_LATIN_RUN = re.compile(
    rf"{_INDIC_RANGES}(?:(?:{_INDIC_RANGES}|[ \u200c\u200d])*{_INDIC_RANGES})?"
)
_TEXT_CMD = re.compile(r"\\text\{[^{}]*\}")


def dehyphenate(text: str) -> str:
    """Rejoin words split by a hyphen across line breaks."""
    previous = None
    while previous != text:
        previous = text
        text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return text


def _repair_expression(tex: str, display: bool) -> str:
    """Make transcribed mathematical expressions valid in LaTeX/KaTeX/MathJax.

    Repairs:
      1. Indic scripts inside math mode: Wraps raw Indic text in \\text{...}.
      2. Bare newlines inside display math: Replaces raw newlines with \\\\ line breaks.
    """
    protected: list[str] = []

    def stash(match: re.Match) -> str:
        protected.append(match.group(0))
        return f"\x00{len(protected) - 1}\x00"

    tex = _TEXT_CMD.sub(stash, tex)
    tex = _NON_LATIN_RUN.sub(lambda m: f"\\text{{{m.group(0)}}}", tex)
    tex = re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], tex)

    if display:
        tex = re.sub(r"\s*\n\s*", r" \\\\ ", tex.strip())
    return tex


def repair_math(text: str) -> str:
    """Repair every math span within markdown prose while leaving normal text intact."""

    def fix(match: re.Match) -> str:
        display = match.group(1) is not None
        inner = _repair_expression(match.group(1) or match.group(2), display)
        return f"$${inner}$$" if display else f"${inner}$"

    return _MATH_SPAN.sub(fix, text)


def reconstruct(blocks: list[Block], repair: bool = True) -> str:
    """Reconstruct coherent Markdown from transcribed blocks sorted by reading order."""
    kept = sorted((b for b in blocks if b.type not in DROP_TYPES), key=lambda b: b.order)

    parts = []
    for block in kept:
        text = (block.text or "").strip()
        if not text:
            continue
        # Ensure equation blocks are properly delimited in LaTeX if not already wrapped
        if block.type == "Equation" and "$" not in text and not text.startswith(_MATH_DELIMITERS):
            text = f"$${text}$$"
        parts.append(text)

    markdown = dehyphenate("\n\n".join(parts))
    return repair_math(markdown) if repair else markdown

