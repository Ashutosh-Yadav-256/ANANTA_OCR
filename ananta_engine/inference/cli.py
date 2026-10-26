"""ANANTA OCR CLI: Layout detection, block OCR, full parsing, and contract inspection.

Subcommands:
  layout IMG...  -o DIR   Stage 1: Detect layout -> DIR/<name>.layout.json
  ocr    LAY...  -o DIR   Stage 2: Transcribe layout blocks -> DIR/<name>.md + <name>.json
  parse  IMG...  -o DIR   Run both stages in one pipeline
  show-contract           Display prompt templates, taxonomy, and output schemas
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .common import (
    apply_yaml_defaults,
    collect_images,
    collect_layouts,
    out_path,
    write_json,
)


def _configs(args):
    from ..engine.types import (
        CropConfig,
        DedupConfig,
        LayoutConfig,
        RecognizerConfig,
    )

    return (
        LayoutConfig(conf=args.conf, img_size=args.img_size, device=args.device),
        RecognizerConfig(
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            table_format=args.table_format,
        ),
        DedupConfig(nest=args.dedup_nest, mode=args.dedup_mode, contain=args.contain),
        CropConfig(min_px_side=args.min_px_side, max_px_side=args.max_px_side, pad_px=args.pad_px),
    )


def _write_layout(page, out_dir, fallback) -> Path:
    path = out_path(Path(page.image).stem, out_dir, ".layout.json", fallback)
    write_json(path, page.as_record())
    print(f"[layout] {page.image}: {len(page.blocks)} blocks -> {path}", file=sys.stderr)
    return path


def _write_ocr(page, out_dir, fallback) -> None:
    stem = Path(page.image).stem
    md = out_path(stem, out_dir, ".md", fallback)
    js = out_path(stem, out_dir, ".json", fallback)
    md.write_text(page.markdown or "", encoding="utf-8")
    write_json(js, page.as_record())
    transcribed = sum(1 for b in page.blocks if b.text)
    print(
        f"[ocr] {page.image}: {transcribed}/{len(page.blocks)} transcribed -> {md}, {js}",
        file=sys.stderr,
    )


def cmd_layout(args) -> None:
    from ..engine.offline import AnantaDocLayout

    layout_cfg, _, dedup, _ = _configs(args)
    stage = AnantaDocLayout(ckpt=args.layout_ckpt, config=layout_cfg, dedup=dedup)
    for image in collect_images(args.images):
        _write_layout(stage.detect(str(image)), args.out_dir, image.parent)


def cmd_ocr(args) -> None:
    from ..engine.offline import AnantaBlockOCR
    from ..engine.types import PageResult

    _, rec_cfg, dedup, crop = _configs(args)
    stage = AnantaBlockOCR(ckpt=args.ocr_ckpt, config=rec_cfg, dedup=dedup, crop=crop)
    for layout_path in collect_layouts(args.layouts):
        import json

        with open(layout_path, encoding="utf-8") as fh:
            page = PageResult.from_record(json.load(fh))
        image = args.image or _find_image(layout_path, page.image)
        _write_ocr(stage.run(image, page), args.out_dir, layout_path.parent)


def _find_image(layout_path: Path, image_name: str) -> str:
    candidate = layout_path.parent / image_name
    if candidate.is_file():
        return str(candidate)
    sys.exit(f"Image {image_name!r} for {layout_path.name} not found; pass --image <file>")


def cmd_parse(args) -> None:
    from ..engine.offline import AnantaOCR

    layout_cfg, rec_cfg, dedup, crop = _configs(args)
    parser = AnantaOCR(
        layout_ckpt=args.layout_ckpt,
        recognizer_ckpt=args.ocr_ckpt,
        layout_config=layout_cfg,
        recognizer_config=rec_cfg,
        dedup=dedup,
        crop=crop,
    )
    for image in collect_images(args.images):
        page = parser.detect(str(image))
        if args.save_layout:
            _write_layout(page, args.out_dir, image.parent)
        _write_ocr(parser.ocr.run(str(image), page), args.out_dir, image.parent)


def cmd_show_contract(_args) -> None:
    from ..templates import contract as C
    from ..layout.labels import CLASSES

    print("=" * 65)
    print("ANANTA OCR SPECIFICATION: PROMPTS, TAXONOMY, & SCHEMAS")
    print("=" * 65)
    print("PROMPTS:")
    print(f"  text/headings : {C.TEXT_PROMPT}")
    print(f"  equations     : {C.EQUATION_PROMPT}")
    for fmt, prompt in C.TABLE_PROMPTS.items():
        print(f"  tables [{fmt}]: {prompt}")
    print(f"\nLAYOUT TAXONOMY ({len(CLASSES)} Classes):")
    for i, c in enumerate(CLASSES):
        print(f"  [{i:2d}] {c:<22}", end="\n" if (i + 1) % 3 == 0 else "")
    if len(CLASSES) % 3 != 0:
        print()
    print("\nBLOCK TYPES KEPT :", ", ".join(C.KEPT_BLOCK_TYPES))
    print("NEVER CROPPED    :", ", ".join(sorted(C.DROP_TYPES)))
    print("NEVER TRANSCRIBED:", ", ".join(sorted(C.OCR_SKIP_LABELS)), "(preserved with empty text)")
    print("\nOUTPUT PER-BLOCK SCHEMA:")
    for key, meaning in C.OUTPUT_BLOCK_SCHEMA.items():
        print(f"  {key:<12} {meaning}")
    print("=" * 65)


def build_parser() -> argparse.ArgumentParser:
    from ..engine.types import DEDUP_MODES

    p = argparse.ArgumentParser(
        prog="python -m ananta_ocr.ananta_engine.inference.cli",
        description="ANANTA OCR: Layout analysis and per-block OCR engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd")

    def common(sp):
        sp.add_argument("--config", help="YAML config supplying defaults for flags")
        sp.add_argument("-o", "--out-dir", default=None, help="Directory to save results")
        sp.add_argument("--layout-ckpt", default=None, help="AnantaDocLayout checkpoint path or repo")
        sp.add_argument("--ocr-ckpt", default=None, help="AnantaBlockOCR checkpoint path or repo")

        layout = sp.add_argument_group("layout options")
        layout.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
        layout.add_argument("--img-size", type=int, default=1024, help="Model image resolution")
        layout.add_argument("--device", default="cuda", help="Execution device (cuda/cpu)")

        dedup = sp.add_argument_group("dedup options")
        dedup.add_argument("--dedup-nest", action=argparse.BooleanOptionalAction, default=True)
        dedup.add_argument("--dedup-mode", choices=DEDUP_MODES, default="both")
        dedup.add_argument("--contain", type=float, default=0.90)

        crop = sp.add_argument_group("crop options")
        crop.add_argument("--min-px-side", type=int, default=256)
        crop.add_argument("--max-px-side", type=int, default=1536)
        crop.add_argument("--pad-px", type=int, default=0)

        rec = sp.add_argument_group("recognizer options")
        rec.add_argument("--gpu-memory-utilization", type=float, default=0.80)
        rec.add_argument("--max-tokens", type=int, default=2048)
        rec.add_argument("--batch-size", type=int, default=2048)
        rec.add_argument("--table-format", choices=["html", "markdown"], default="html")

    pl = sub.add_parser("layout", help="Stage 1: Detect layout -> <name>.layout.json")
    pl.add_argument("images", nargs="+", help="Image files or directories")
    common(pl)
    pl.set_defaults(func=cmd_layout)

    po = sub.add_parser("ocr", help="Stage 2: Transcribe layout -> <name>.md + <name>.json")
    po.add_argument("layouts", nargs="+", help="Layout JSON files or directories")
    po.add_argument("--image", default=None, help="Override image filename")
    common(po)
    po.set_defaults(func=cmd_ocr)

    pp = sub.add_parser("parse", help="Run both stages end-to-end")
    pp.add_argument("images", nargs="+", help="Image files or directories")
    pp.add_argument("--save-layout", action="store_true", help="Also save intermediate .layout.json")
    common(pp)
    pp.set_defaults(func=cmd_parse)

    sub.add_parser(
        "show-contract", help="Display prompt templates, taxonomy, and output schemas"
    ).set_defaults(func=cmd_show_contract)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
