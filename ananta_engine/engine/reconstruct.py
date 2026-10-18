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

# Runs of Indic and Perso-Arabic scripts:
# Arabic (Urdu, Kashmiri, Sindhi), Devanagari..Malayalam, and Ol Chiki (Santali).
_NON_LATIN_RUN = re.compile(
    r"[\u0600-\u06ff\u0900-\u0d7f\u1c50-\u1c7f]"
    r"(?:[\u0600-\u06ff\u0900-\u0d7f\u1c50-\u1c7f \u200c\u200d]*"
    r"[\u0600-\u06ff\u0900-\u0d7f\u1c50-\u1c7f])?"
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

