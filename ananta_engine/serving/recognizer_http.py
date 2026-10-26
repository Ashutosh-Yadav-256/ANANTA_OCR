"""HTTP-based recognizer backend connecting to an OpenAI-compatible vLLM server."""

from __future__ import annotations

import base64
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from ..engine.types import RecognizerConfig

if TYPE_CHECKING:
    from PIL.Image import Image
    from ..engine.recognizer import CropRequest

logger = logging.getLogger("ananta_ocr.serving.recognizer_http")

DEFAULT_MODEL = "ananta_ocr"


def encode_crop(image: Image) -> str:
    """Encode PIL image as base64 PNG data URI."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_messages(image: Image, prompt: str) -> list[dict[str, Any]]:
    """Build a chat completion user turn message containing crop image and prompt."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": encode_crop(image)}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


class HttpRecognizer:
    """Transcribe block crops concurrently via an OpenAI-compatible vLLM endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str = "",
        timeout: float = 300.0,
        config: RecognizerConfig | None = None,
        num_workers: int = 32,
        strict: bool = True,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.config = config or RecognizerConfig()
        self.num_workers = max(1, num_workers)
        self.strict = strict
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI

            key = api_key or os.environ.get("OCR_API_KEY") or "EMPTY"
            self._client = OpenAI(base_url=base_url, api_key=key, timeout=timeout)

    def _one(self, request: CropRequest) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=build_messages(request.image, request.prompt),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    def transcribe(self, requests: list[CropRequest]) -> list[str]:
        if not requests:
            return []

        texts: list[str] = [""] * len(requests)
        failures: list[str] = []

        def run(index: int) -> None:
            try:
                texts[index] = self._one(requests[index])
            except Exception as exc:
                failures.append(f"crop {index}: {type(exc).__name__}: {exc}")

        workers = min(self.num_workers, len(requests))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, range(len(requests))))

        if failures:
            summary = f"{len(failures)}/{len(requests)} crops failed; first: {failures[0]}"
            if self.strict:
                raise RuntimeError(summary)
            logger.warning("%s -- those blocks carry empty text", summary)

        return texts

    def close(self) -> None:
        self._client = None

