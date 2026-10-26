#!/usr/bin/env python3
"""ANANTA OCR: Turnkey Document Intelligence CLI.

A unified document understanding system combining:
  - Ananta VLM Engine ('vlm' / 'ananta'): Unlimited-horizon multi-page parsing for 100+ pages with 273 tokens/page compression.
  - Ananta Block Engine ('block' / 'indic'): 2-stage layout detection (AnantaDocLayout) & block recognition (AnantaBlockOCR) for 22 Indian languages.
  - Ananta Hybrid Engine ('hybrid'): Layout detection + targeted block transcription.

Usage Examples:
    # 1. Ananta VLM Mode (Multi-page document understanding)
    python run_inference.py --engine vlm --images page1.jpg page2.jpg --output-dir output/

    # 2. Ananta Block Mode (2-stage layout analysis + block transcription)
    python run_inference.py --engine block --images doc_hindi.png --output-dir output/

    # 3. Ananta Stage 1 only (Layout & Reading Order)
    python run_inference.py --engine block --images doc_hindi.png --layout-only --output-dir output/

    # 4. Ananta Stage 2 only (Transcribe existing layout JSON)
    python run_inference.py --engine block --ocr-only output/doc_hindi.layout.json --output-dir output/

    # 5. Ananta Hybrid mode
    python run_inference.py --engine hybrid --images doc.jpg --output-dir output/

    # 6. Display ANANTA OCR prompt templates and 37-class taxonomy
    python run_inference.py show-contract
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def check_environment():
    """Diagnostic check for CUDA, PyTorch, and accelerator availability."""
    import torch

    print("=" * 65)
    print("ANANTA OCR: SYSTEM DIAGNOSTIC")
    print("=" * 65)
    print(f"Python Version : {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")

    cuda_ok = torch.cuda.is_available()
    print(f"CUDA Available : {cuda_ok}")
    if cuda_ok:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Device     : {device_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("[NOTICE] No CUDA GPU detected.")
        print("         - Ananta Block Engine can run on CPU with Hugging Face backend.")
        print("         - Ananta VLM Engine requires NVIDIA GPU + bfloat16 for fast execution.")
    print("=" * 65 + "\n")
    return cuda_ok


def check_vlm_weights(model_dir: str) -> bool:
    """Verify Ananta VLM weight files exist locally."""
    required_files = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "config.json",
    ]
    missing = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        print(f"[ERROR] Missing required Ananta VLM model files: {missing}")
        print("Run 'python download_weights.py --model vlm' to download weights.")
        return False
    return True


def run_vlm(args):
    """Execute Ananta VLM multi-page pipeline."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not check_vlm_weights(args.model_dir):
        sys.exit(1)

    print(f"[ANANTA OCR] Loading VLM tokenizer and model from: {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModel.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=dtype,
    ).eval().to(device)

    print(f"[ANANTA OCR] Running multi-page parsing on {len(args.images)} page(s)...")
    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.perf_counter()
    raw_text, num_tokens = model.infer_multi(
        tokenizer,
        image_files=args.images,
        output_path=args.output_dir,
        save_results=True,
    )
    elapsed = time.perf_counter() - t0

    try:
        from utils.process_anantaocr import post_process_document
        markdown = post_process_document(raw_text)
        md_path = os.path.join(args.output_dir, "output.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)
    except Exception as e:
        print(f"[Notice] Post-processing fallback: {e}")
        markdown = raw_text

    print(f"\n[DONE] Generated {num_tokens} tokens in {elapsed:.2f}s ({num_tokens/max(elapsed, 0.001):.1f} tok/s).")
    print(f"Output saved to directory: '{args.output_dir}'")
    print("-" * 65)
    print(markdown[:600] + ("\n... [truncated]" if len(markdown) > 600 else ""))
    print("-" * 65)


def run_block(args):
    """Execute Ananta 2-stage block pipeline or sub-stages."""
    from ananta_engine import (
        AnantaDocLayout,
        AnantaBlockOCR,
        AnantaOCR,
        LayoutConfig,
        CropConfig,
        DedupConfig,
        RecognizerConfig,
        TableFormat,
    )
    from ananta_engine.inference.common import collect_images, collect_layouts, out_path, write_json

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Submode 1: Layout only
    if args.layout_only:
        print(f"[ANANTA OCR] Stage 1 (AnantaDocLayout) on {len(args.images)} image(s)...")
        layout_cfg = LayoutConfig(conf=args.conf, img_size=args.img_size, device=args.device)
        dedup = DedupConfig(nest=args.dedup_nest, mode=args.dedup_mode, contain=args.contain)
        stage = AnantaDocLayout(ckpt=args.layout_ckpt, config=layout_cfg, dedup=dedup)

        for img in collect_images(args.images):
            page = stage.detect(str(img))
            p = out_path(img.stem, str(out_dir), ".layout.json", img.parent)
            write_json(p, page.as_record())
            print(f"[layout] {img.name}: {len(page.blocks)} blocks -> {p}")
        return

    # Submode 2: OCR only against pre-computed layout
    if args.ocr_only:
        import json
        from ananta_engine import PageResult

        layout_files = [args.ocr_only] if os.path.isfile(args.ocr_only) else [str(p) for p in collect_layouts([args.ocr_only])]
        print(f"[ANANTA OCR] Stage 2 (AnantaBlockOCR) on {len(layout_files)} layout file(s)...")

        rec_cfg = RecognizerConfig(
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            table_format=TableFormat(args.table_format),
        )
        dedup = DedupConfig(nest=args.dedup_nest, mode=args.dedup_mode, contain=args.contain)
        crop = CropConfig(min_px_side=args.min_px_side, max_px_side=args.max_px_side, pad_px=args.pad_px)
        stage = AnantaBlockOCR(ckpt=args.ocr_ckpt, config=rec_cfg, dedup=dedup, crop=crop)

        for lay_path in layout_files:
            lp = Path(lay_path)
            with open(lp, encoding="utf-8") as fh:
                page = PageResult.from_record(json.load(fh))

            img_path = args.image_override or (lp.parent / page.image)
            if not os.path.exists(img_path):
                print(f"[ERROR] Image {page.image!r} not found beside {lp.name}. Pass --image-override.")
                continue

            result = stage.run(str(img_path), page)
            stem = lp.stem.replace(".layout", "")
            md_p = out_dir / f"{stem}.md"
            js_p = out_dir / f"{stem}.json"
            md_p.write_text(result.markdown or "", encoding="utf-8")
            write_json(js_p, result.as_record())
            transcribed = sum(1 for b in result.blocks if b.text)
            print(f"[ocr] {page.image}: {transcribed}/{len(result.blocks)} blocks transcribed -> {md_p}, {js_p}")
        return

    # Submode 3: Full 2-stage parse
    images = collect_images(args.images)
    print(f"[ANANTA OCR] Running full 2-stage block parse on {len(images)} image(s)...")

    layout_cfg = LayoutConfig(conf=args.conf, img_size=args.img_size, device=args.device)
    rec_cfg = RecognizerConfig(
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        table_format=TableFormat(args.table_format),
    )
    dedup = DedupConfig(nest=args.dedup_nest, mode=args.dedup_mode, contain=args.contain)
    crop = CropConfig(min_px_side=args.min_px_side, max_px_side=args.max_px_side, pad_px=args.pad_px)

    pipeline = AnantaOCR(
        layout_ckpt=args.layout_ckpt,
        recognizer_ckpt=args.ocr_ckpt,
        layout_config=layout_cfg,
        recognizer_config=rec_cfg,
        dedup=dedup,
        crop=crop,
    )

    t0 = time.perf_counter()
    for img in images:
        page_layout = pipeline.detect(str(img))
        if args.save_layout:
            lp = out_path(img.stem, str(out_dir), ".layout.json", img.parent)
            write_json(lp, page_layout.as_record())

        res = pipeline.ocr.run(str(img), page_layout)
        md_p = out_dir / f"{img.stem}.md"
        js_p = out_dir / f"{img.stem}.json"
        md_p.write_text(res.markdown or "", encoding="utf-8")
        write_json(js_p, res.as_record())

        transcribed = sum(1 for b in res.blocks if b.text)
        print(f"[ocr] {img.name}: {transcribed}/{len(res.blocks)} blocks transcribed -> {md_p}, {js_p}")

    print(f"\n[DONE] Processed {len(images)} page(s) in {time.perf_counter() - t0:.2f}s.")


def run_hybrid(args):
    """Execute Ananta Hybrid mode."""
    from unified_pipeline import UnifiedOCR
    from ananta_engine.inference.common import collect_images

    images = collect_images(args.images)
    print(f"[ANANTA OCR] Hybrid parsing on {len(images)} page(s)...")

    ocr = UnifiedOCR(
        engine="hybrid",
        model_dir=args.model_dir,
        layout_ckpt=args.layout_ckpt,
        recognizer_ckpt=args.ocr_ckpt,
        table_format=args.table_format,
        device=args.device,
        conf=args.conf,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )

    result = ocr.process(images, output_dir=args.output_dir, save_results=True)
    print(f"\n[DONE] Hybrid parsing completed in {result.elapsed_seconds}s.")
    print(f"Saved results to: '{args.output_dir}'")
    print("-" * 65)
    print(result.markdown[:600] + ("\n... [truncated]" if len(result.markdown) > 600 else ""))
    print("-" * 65)


def show_contract():
    """Print ANANTA OCR prompt contracts, 37-class layout taxonomy, and schema definitions."""
    from ananta_engine.templates import contract as C
    from ananta_engine.layout.labels import CLASSES

    print("=" * 65)
    print("ANANTA OCR SPECIFICATION: PROMPTS, TAXONOMY, & SCHEMAS")
    print("=" * 65)
    print("\n1. PROMPT TEMPLATES:")
    print(f"  Text / Headers : {C.TEXT_PROMPT}")
    print(f"  Equations      : {C.EQUATION_PROMPT}")
    for fmt, prompt in C.TABLE_PROMPTS.items():
        print(f"  Tables [{fmt}]: {prompt}")

    print(f"\n2. LAYOUT TAXONOMY ({len(CLASSES)} Classes):")
    for i, c in enumerate(CLASSES):
        print(f"  [{i:2d}] {c:<22}", end="\n" if (i + 1) % 3 == 0 else "")
    if len(CLASSES) % 3 != 0:
        print()

    print("\n3. FUNCTIONAL MAPPINGS:")
    print(f"  Transcribed types : {', '.join(C.KEPT_BLOCK_TYPES)}")
    print(f"  Never cropped     : {', '.join(sorted(C.DROP_TYPES))}")
    print(f"  Never transcribed : {', '.join(sorted(C.OCR_SKIP_LABELS))} (kept in JSON with text='')")

    print("\n4. OUTPUT SCHEMAS:")
    for k, v in C.OUTPUT_BLOCK_SCHEMA.items():
        print(f"  {k:<12} {v}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="ANANTA OCR: Turnkey Visual Document Intelligence CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="subcommand")
    subparsers.add_parser("show-contract", help="Display prompt templates, taxonomy, and schemas")

    # Main arguments
    parser.add_argument(
        "--engine",
        choices=["vlm", "block", "hybrid", "ananta", "indic"],
        default="vlm",
        help="ANANTA OCR Engine: 'vlm' (multi-page VLM), 'block' (2-stage layout & block OCR), or 'hybrid'",
    )
    parser.add_argument("--images", nargs="+", help="Paths to page image files or directories")
    parser.add_argument("--output-dir", default="output", help="Directory to save output files")
    parser.add_argument("--model-dir", default=".", help="Directory containing Ananta VLM model weights")

    # Ananta Block & Layout Engine options
    block_group = parser.add_argument_group("Ananta Block & Layout Options")
    block_group.add_argument("--layout-only", action="store_true", help="Run Stage 1 layout detection only")
    block_group.add_argument("--ocr-only", default=None, help="Run Stage 2 OCR only on a layout JSON file")
    block_group.add_argument("--image-override", default=None, help="Override image path when using --ocr-only")
    block_group.add_argument("--save-layout", action="store_true", help="Save intermediate .layout.json")
    block_group.add_argument("--layout-ckpt", default=None, help="AnantaDocLayout checkpoint path or HF repo")
    block_group.add_argument("--ocr-ckpt", default=None, help="AnantaBlockOCR checkpoint path or HF repo")
    block_group.add_argument("--table-format", choices=["html", "markdown"], default="html")
    block_group.add_argument("--conf", type=float, default=0.5, help="Layout detection confidence threshold")
    block_group.add_argument("--img-size", type=int, default=1024, help="Layout model input resolution")
    block_group.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")
    block_group.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per block crop")
    block_group.add_argument("--batch-size", type=int, default=2048, help="Batch size for recognizer")
    block_group.add_argument("--min-px-side", type=int, default=256)
    block_group.add_argument("--max-px-side", type=int, default=1536)
    block_group.add_argument("--pad-px", type=int, default=0)
    block_group.add_argument("--dedup-nest", action=argparse.BooleanOptionalAction, default=True)
    block_group.add_argument("--dedup-mode", choices=["both", "text_only", "eq_only"], default="both")
    block_group.add_argument("--contain", type=float, default=0.90)

    args = parser.parse_args()

    if args.subcommand == "show-contract":
        show_contract()
        return

    # Check for inputs
    if not args.images and not args.ocr_only:
        print("[NOTICE] No input images specified. Provide images via: --images page1.jpg page2.jpg")
        print("Run with --help for full usage details, or 'show-contract' to view taxonomy.")
        return

    check_environment()

    # Route to engine (handling aliases)
    engine = args.engine.lower()
    if engine in ("vlm", "ananta"):
        run_vlm(args)
    elif engine in ("block", "indic"):
        run_block(args)
    elif engine == "hybrid":
        run_hybrid(args)


if __name__ == "__main__":
    main()
