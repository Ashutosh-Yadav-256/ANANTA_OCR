"""ANANTA OCR Unified Pipeline: Multi-Page VLM & 2-Stage Layout-Block Intelligence.

Provides a unified interface (`UnifiedOCR`) uniting all ANANTA OCR capabilities:
  1. Ananta VLM Engine ('vlm' / 'ananta'): Unlimited-horizon multi-page document VLM (DeepSeek-V2 MoE + SAM-ViT-B + CLIP-L + DASH attention)
  2. Ananta Block Engine ('block' / 'indic'): High-precision 2-stage layout and block OCR (AnantaDocLayout + AnantaBlockOCR)
  3. Ananta Hybrid Engine ('hybrid'): Combines AnantaDocLayout block detection with Ananta VLM and Block OCR

Example:
    from ananta_ocr.unified_pipeline import UnifiedOCR

    # Using Ananta VLM engine (multi-page document mode)
    ocr = UnifiedOCR(engine="vlm", model_dir="./")
    result = ocr.process(["page1.png", "page2.png"])
    print(result.markdown)

    # Using Ananta Block engine (2-stage layout + block OCR)
    ocr = UnifiedOCR(engine="block")
    result = ocr.process("doc_page.png")
    print(result.markdown)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .ananta_engine import (
    Block,
    PageResult,
    AnantaOCR,
    AnantaDocLayout,
    AnantaBlockOCR,
    TableFormat,
    LayoutConfig,
    CropConfig,
    DedupConfig,
    RecognizerConfig,
)


@dataclass
class UnifiedOCRResult:
    """Unified result container for ANANTA OCR."""

    markdown: str
    engine: str
    pages: list[dict[str, Any]] = field(default_factory=list)
    raw_output: str | None = None
    num_tokens: int = 0
    elapsed_seconds: float = 0.0

    def save(self, output_dir: str | Path, base_name: str = "document") -> dict[str, Path]:
        """Save results to markdown and JSON files."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        md_file = out_path / f"{base_name}.md"
        md_file.write_text(self.markdown, encoding="utf-8")

        saved = {"markdown": md_file}

        if self.pages:
            import json
            json_file = out_path / f"{base_name}.json"
            with open(json_file, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "engine": self.engine,
                        "num_pages": len(self.pages),
                        "pages": self.pages,
                    },
                    fh,
                    ensure_ascii=False,
                    indent=2,
                )
            saved["json"] = json_file

        return saved


class UnifiedOCR:
    """Unified entry point for ANANTA OCR document intelligence."""

    def __init__(
        self,
        engine: Literal["vlm", "block", "hybrid", "ananta", "indic"] = "vlm",
        model_dir: str = ".",
        layout_ckpt: str | None = None,
        recognizer_ckpt: str | None = None,
        table_format: str = "html",
        device: str = "auto",
        conf: float = 0.5,
        max_tokens: int = 2048,
        batch_size: int = 2048,
    ) -> None:
        raw_engine = engine.lower()
        # Canonical engine names
        if raw_engine in ("vlm", "ananta", "multipage"):
            self.engine_type = "vlm"
        elif raw_engine in ("block", "indic", "layout_ocr"):
            self.engine_type = "block"
        elif raw_engine == "hybrid":
            self.engine_type = "hybrid"
        else:
            raise ValueError(
                f"Unknown engine {engine!r}. Must be 'vlm' (or 'ananta'), 'block' (or 'indic'), or 'hybrid'."
            )

        self.model_dir = model_dir
        self.layout_ckpt = layout_ckpt
        self.recognizer_ckpt = recognizer_ckpt
        self.table_format = TableFormat(table_format)
        self.conf = conf
        self.max_tokens = max_tokens
        self.batch_size = batch_size

        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self._ananta_model = None
        self._ananta_tokenizer = None
        self._block_pipeline = None
        self._block_layout = None
        self._block_recognizer = None

    # -- Engine loaders ---------------------------------------------------- #

    def _get_vlm(self):
        """Lazy load Ananta VLM model and tokenizer."""
        if self._ananta_model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            print(f"[ANANTA OCR] Loading VLM tokenizer and model from: {self.model_dir}")
            self._ananta_tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, trust_remote_code=True
            )
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self._ananta_model = AutoModel.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=dtype,
            ).eval()
            if self.device != "cpu":
                self._ananta_model = self._ananta_model.to(self.device)
        return self._ananta_model, self._ananta_tokenizer

    def _get_block_pipeline(self) -> AnantaOCR:
        """Lazy load Ananta 2-stage block pipeline."""
        if self._block_pipeline is None:
            layout_cfg = LayoutConfig(conf=self.conf, device=self.device)
            rec_cfg = RecognizerConfig(
                max_tokens=self.max_tokens,
                batch_size=self.batch_size,
                table_format=self.table_format,
            )
            self._block_pipeline = AnantaOCR(
                layout_ckpt=self.layout_ckpt,
                recognizer_ckpt=self.recognizer_ckpt,
                layout_config=layout_cfg,
                recognizer_config=rec_cfg,
            )
        return self._block_pipeline

    def _get_block_layout(self) -> AnantaDocLayout:
        """Lazy load AnantaDocLayout alone."""
        if self._block_layout is None:
            layout_cfg = LayoutConfig(conf=self.conf, device=self.device)
            self._block_layout = AnantaDocLayout(ckpt=self.layout_ckpt, config=layout_cfg)
        return self._block_layout

    def _get_block_recognizer(self) -> AnantaBlockOCR:
        """Lazy load AnantaBlockOCR alone."""
        if self._block_recognizer is None:
            rec_cfg = RecognizerConfig(
                max_tokens=self.max_tokens,
                batch_size=self.batch_size,
                table_format=self.table_format,
            )
            self._block_recognizer = AnantaBlockOCR(ckpt=self.recognizer_ckpt, config=rec_cfg)
        return self._block_recognizer

    # -- Processing methods ------------------------------------------------ #

    def process(
        self,
        images: str | Path | list[str | Path],
        output_dir: str | Path | None = None,
        save_results: bool = True,
    ) -> UnifiedOCRResult:
        """Process document image(s) using the selected ANANTA OCR engine."""
        if isinstance(images, (str, Path)):
            image_list = [str(images)]
        else:
            image_list = [str(p) for p in images]

        t0 = time.perf_counter()

        if self.engine_type == "vlm":
            result = self._process_vlm(image_list, output_dir, save_results)
        elif self.engine_type == "block":
            result = self._process_block(image_list, output_dir, save_results)
        elif self.engine_type == "hybrid":
            result = self._process_hybrid(image_list, output_dir, save_results)
        else:
            raise ValueError(f"Unknown engine {self.engine_type!r}.")

        result.elapsed_seconds = round(time.perf_counter() - t0, 3)
        return result

    def _process_vlm(
        self,
        image_list: list[str],
        output_dir: str | Path | None,
        save_results: bool,
    ) -> UnifiedOCRResult:
        """Execute multi-page inference with Ananta VLM."""
        model, tokenizer = self._get_vlm()
        save_dir = str(output_dir) if output_dir else "output"
        os.makedirs(save_dir, exist_ok=True)

        raw_text, num_tokens = model.infer_multi(
            tokenizer,
            image_files=image_list,
            output_path=save_dir,
            save_results=save_results,
        )

        try:
            from utils.process_anantaocr import post_process_document
            markdown = post_process_document(raw_text)
        except Exception:
            markdown = raw_text

        pages = [{"page_index": i + 1, "image": img} for i, img in enumerate(image_list)]

        return UnifiedOCRResult(
            markdown=markdown,
            engine="vlm",
            pages=pages,
            raw_output=raw_text,
            num_tokens=num_tokens,
        )

    def _process_block(
        self,
        image_list: list[str],
        output_dir: str | Path | None,
        save_results: bool,
    ) -> UnifiedOCRResult:
        """Execute 2-stage layout + block transcription with Ananta Block Engine."""
        pipeline = self._get_block_pipeline()
        page_results: list[dict[str, Any]] = []
        md_parts: list[str] = []

        save_path = Path(output_dir) if output_dir else None
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)

        for img_path in image_list:
            res = pipeline.parse(img_path)
            page_dict = res.as_record()
            page_dict["markdown"] = res.markdown
            page_results.append(page_dict)

            if res.markdown:
                md_parts.append(res.markdown)

            if save_path and save_results:
                stem = Path(img_path).stem
                (save_path / f"{stem}.md").write_text(res.markdown or "", encoding="utf-8")
                import json
                with open(save_path / f"{stem}.json", "w", encoding="utf-8") as fh:
                    json.dump(res.as_record(), fh, ensure_ascii=False, indent=2)

        combined_md = "\n\n---\n\n".join(md_parts)
        return UnifiedOCRResult(
            markdown=combined_md,
            engine="block",
            pages=page_results,
            raw_output=None,
            num_tokens=0,
        )

    def _process_hybrid(
        self,
        image_list: list[str],
        output_dir: str | Path | None,
        save_results: bool,
    ) -> UnifiedOCRResult:
        """Hybrid execution: AnantaDocLayout block segmentation + AnantaBlockOCR transcription."""
        layout_stage = self._get_block_layout()
        ocr_stage = self._get_block_recognizer()

        page_results: list[dict[str, Any]] = []
        md_parts: list[str] = []

        save_path = Path(output_dir) if output_dir else None
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)

        for img_path in image_list:
            page_layout = layout_stage.detect(img_path)

            if save_path and save_results:
                import json
                stem = Path(img_path).stem
                with open(save_path / f"{stem}.layout.json", "w", encoding="utf-8") as fh:
                    json.dump(page_layout.as_record(), fh, ensure_ascii=False, indent=2)

            res = ocr_stage.run(img_path, page_layout)
            page_dict = res.as_record()
            page_dict["markdown"] = res.markdown
            page_results.append(page_dict)

            if res.markdown:
                md_parts.append(res.markdown)

            if save_path and save_results:
                (save_path / f"{stem}.md").write_text(res.markdown or "", encoding="utf-8")
                with open(save_path / f"{stem}.json", "w", encoding="utf-8") as fh:
                    json.dump(res.as_record(), fh, ensure_ascii=False, indent=2)

        combined_md = "\n\n---\n\n".join(md_parts)
        return UnifiedOCRResult(
            markdown=combined_md,
            engine="hybrid",
            pages=page_results,
            raw_output=None,
            num_tokens=0,
        )
