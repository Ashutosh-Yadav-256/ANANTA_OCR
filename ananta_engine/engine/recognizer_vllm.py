"""In-process vLLM recognizer backend for high-throughput batch inference."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

from .recognizer import CropRequest
from .types import RecognizerConfig

logger = logging.getLogger("ananta_ocr.engine.recognizer_vllm")


def _flashinfer_cache_is_usable(root: Path) -> bool:
    if not root.exists():
        return True
    if not os.access(root, os.W_OK | os.X_OK):
        return False
    return all(os.access(child, os.W_OK | os.X_OK) for child in root.iterdir() if child.is_dir())


def _prepare_runtime() -> None:
    """Prepare environment for vLLM and flashinfer initialization."""
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    cache_dir = Path.home() / ".cache" / "flashinfer"
    if not os.environ.get("FLASHINFER_WORKSPACE_BASE") and not _flashinfer_cache_is_usable(cache_dir):
        fallback = Path(tempfile.gettempdir()) / "flashinfer-cache"
        fallback.mkdir(parents=True, exist_ok=True)
        os.environ["FLASHINFER_WORKSPACE_BASE"] = str(fallback)


class VllmRecognizer:
    """In-process vLLM recognizer engine with continuous batching."""

    def __init__(self, ckpt: str | None = None, config: RecognizerConfig | None = None) -> None:
        _prepare_runtime()

        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        from .checkpoints import resolve_ckpt

        self.config = config or RecognizerConfig()
        self.ckpt = resolve_ckpt("recognizer", ckpt)
        self.processor = AutoProcessor.from_pretrained(self.ckpt)
        self.llm = LLM(
            model=self.ckpt,
            trust_remote_code=True,
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            dtype=self.config.dtype,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=self.config.enforce_eager,
        )
        self.sampling = SamplingParams(
            temperature=self.config.temperature, max_tokens=self.config.max_tokens
        )

    def _chat(self, prompt: str) -> str:
        return self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            add_generation_prompt=True,
            tokenize=False,
        )

    def transcribe(self, requests: list[CropRequest]) -> list[str]:
        if not requests:
            return []
        payload = [
            {"prompt": self._chat(r.prompt), "multi_modal_data": {"image": r.image}}
            for r in requests
        ]

        texts: list[str] = []
        for i in range(0, len(payload), self.config.batch_size):
            outputs = self.llm.generate(payload[i : i + self.config.batch_size], self.sampling)
            texts.extend(out.outputs[0].text for out in outputs)
        return texts

    def close(self) -> None:
        self.llm = None
        self.processor = None
