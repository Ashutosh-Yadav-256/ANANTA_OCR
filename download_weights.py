#!/usr/bin/env python3
"""Download model weights for ANANTA OCR from Hugging Face.

Usage:
    python download_weights.py --model vlm       # Download Ananta VLM weights (baidu/Unlimited-OCR-2)
    python download_weights.py --model block     # Download Ananta Block & Layout weights (bodhan-ai/indic-ocr)
    python download_weights.py --all             # Download weights for all Ananta OCR engines
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

BASE_DIR = Path(__file__).resolve().parent

VLM_REPO = "baidu/Unlimited-OCR-2"
VLM_WEIGHT_FILES = [
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "tokenizer.json",
    "generation_config.json",
    "special_tokens_map.json",
]

BLOCK_REPO = "bodhan-ai/indic-ocr"


def download_vlm(target_dir: Path):
    """Download Ananta VLM weights into target directory."""
    print("=" * 65)
    print("DOWNLOADING ANANTA OCR VLM WEIGHTS")
    print(f"Source Repo: {VLM_REPO}")
    print(f"Destination: {target_dir}")
    print("=" * 65)

    for filename in VLM_WEIGHT_FILES:
        dest_file = target_dir / filename
        if dest_file.exists() and dest_file.stat().st_size > 1000:
            sz_mb = dest_file.stat().st_size / (1024 * 1024)
            print(f"[OK] {filename} already exists ({sz_mb:.1f} MB). Skipping.")
            continue

        print(f"[FETCH] Downloading {filename} from {VLM_REPO}...")
        try:
            downloaded = hf_hub_download(
                repo_id=VLM_REPO,
                filename=filename,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )
            sz_mb = Path(downloaded).stat().st_size / (1024 * 1024)
            print(f"[SUCCESS] Saved {filename} ({sz_mb:.1f} MB).")
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}")
            sys.exit(1)

    print("\n[DONE] Ananta VLM weights verified.\n")


def download_block(target_dir: Path):
    """Download Ananta Block & Layout weights."""
    print("=" * 65)
    print("DOWNLOADING ANANTA OCR BLOCK & LAYOUT WEIGHTS")
    print(f"Source Repo: {BLOCK_REPO}")
    print(f"Destination: {target_dir / 'weights'}")
    print("=" * 65)

    weights_dir = target_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("[FETCH] Downloading AnantaDocLayout weights...")
    try:
        layout_path = snapshot_download(
            repo_id=BLOCK_REPO,
            allow_patterns=["weights/layout/*"],
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        print(f"[SUCCESS] AnantaDocLayout downloaded to: {layout_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download AnantaDocLayout: {e}")
        sys.exit(1)

    print("\n[FETCH] Downloading AnantaBlockOCR weights...")
    try:
        ocr_path = snapshot_download(
            repo_id=BLOCK_REPO,
            allow_patterns=["weights/ocr/*"],
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        print(f"[SUCCESS] AnantaBlockOCR downloaded to: {ocr_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download AnantaBlockOCR: {e}")
        sys.exit(1)

    print("\n[DONE] Ananta Block & Layout weights verified.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download model weights for ANANTA OCR"
    )
    parser.add_argument(
        "--model",
        choices=["vlm", "block", "all", "ananta", "indic"],
        default="vlm",
        help="Model weights to download ('vlm' / 'block' / 'all')",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download weights for all ANANTA OCR engines",
    )
    parser.add_argument(
        "--target-dir",
        default=str(BASE_DIR),
        help="Local directory to store weights",
    )
    args = parser.parse_args()

    target = Path(args.target_dir)

    model_choice = args.model.lower()
    if args.all or model_choice == "all":
        download_vlm(target)
        download_block(target)
    elif model_choice in ("vlm", "ananta"):
        download_vlm(target)
    elif model_choice in ("block", "indic"):
        download_block(target)


if __name__ == "__main__":
    main()
