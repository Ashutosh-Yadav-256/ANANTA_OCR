"""Model checkpoint discovery and Hugging Face Hub resolution for ANANTA OCR.

Locates weights for:
  - "layout": AnantaDocLayout (PP-DocLayoutV3 with 37 classes and reading-order head)
  - "recognizer": AnantaBlockOCR (Qwen3.5-0.8B VLM with 262k Indic & multilingual tokenizer)

Resolution priority:
  1. Explicit path passed as argument
  2. Environment variables:
     - Layout: ANANTA_LAYOUT_CKPT or BODHAN_OCR_LAYOUT_CKPT
     - Recognizer: ANANTA_RECOGNIZER_CKPT or BODHAN_OCR_RECOGNIZER_CKPT
  3. Local directory: ./weights/{stage} or ananta_ocr/weights/{stage}
  4. Automatic Hugging Face Hub snapshot download (bodhan-ai/indic-ocr)
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_HF_REPO = "bodhan-ai/indic-ocr"
HF_REPO_ENV = "ANANTA_OCR_HF_REPO"

SUBDIRS = {"layout": "layout", "recognizer": "ocr"}
ENV_VARS = {
    "layout": ["ANANTA_LAYOUT_CKPT", "BODHAN_OCR_LAYOUT_CKPT"],
    "recognizer": ["ANANTA_RECOGNIZER_CKPT", "BODHAN_OCR_RECOGNIZER_CKPT"],
}


def resolve_ckpt(stage: str, explicit: str | None = None) -> str:
    """Resolve weight directory path for a stage ('layout' or 'recognizer').
    
    Downloads nothing until explicitly called, ensuring lightweight module imports and fast CLI help.
    """
    if stage not in SUBDIRS:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {sorted(SUBDIRS)}")
    if explicit:
        if os.path.exists(explicit):
            return explicit
        return explicit

    sub = SUBDIRS[stage]

    # 1. Check environment variable overrides
    for env_var in ENV_VARS[stage]:
        override = os.environ.get(env_var)
        if override and os.path.isdir(override):
            return override

    # 2. Check local directories
    search_paths = [
        Path.cwd() / "weights" / sub,
        Path(__file__).resolve().parents[2] / "weights" / sub,
        Path(__file__).resolve().parents[1] / "weights" / sub,
    ]
    for p in search_paths:
        if p.is_dir() and any(p.iterdir()):
            return str(p)

    # 3. Download from Hugging Face Hub
    repo = os.environ.get(HF_REPO_ENV, os.environ.get("BODHAN_OCR_HF_REPO", DEFAULT_HF_REPO))
    from huggingface_hub import snapshot_download

    root = snapshot_download(repo, allow_patterns=[f"weights/{sub}/*"])
    stage_path = os.path.join(root, "weights", sub)
    if os.path.isdir(stage_path):
        return stage_path
    return root
