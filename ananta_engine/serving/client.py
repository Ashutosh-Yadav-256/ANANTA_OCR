"""Client for distributed ANANTA OCR: Local lightweight layout analysis + remote vLLM recognizer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..engine.types import (
    CropConfig,
    DedupConfig,
    LayoutConfig,
    RecognizerConfig,
)
from .recognizer_http import DEFAULT_MODEL, HttpRecognizer

if TYPE_CHECKING:
    from ..engine.types import PageResult

logger = logging.getLogger("ananta_ocr.serving.client")


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class AnantaClient:
    """Client for running ANANTA OCR with Stage 1 (layout) locally and Stage 2 (OCR) on a remote vLLM server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        *,
        model: str = DEFAULT_MODEL,
        api_key: str = "EMPTY",
        timeout: float = 300.0,
        layout_ckpt: str | None = None,
        layout_config: LayoutConfig | None = None,
        recognizer_config: RecognizerConfig | None = None,
        dedup: DedupConfig | None = None,
        crop: CropConfig | None = None,
        num_workers: int = 32,
        strict: bool = True,
        client: Any | None = None,
    ) -> None:
        from ..engine.offline import AnantaBlockOCR, AnantaDocLayout

        self.recognizer_config = recognizer_config or RecognizerConfig()
        self.backend = HttpRecognizer(
            base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            config=self.recognizer_config,
            num_workers=num_workers,
            strict=strict,
            client=client,
        )
        if layout_config is None:
            layout_config = LayoutConfig(device=resolve_device())
        self.layout = AnantaDocLayout(ckpt=layout_ckpt, config=layout_config, dedup=dedup)
        self.ocr = AnantaBlockOCR(
            backend=self.backend,
            config=self.recognizer_config,
            dedup=dedup,
            crop=crop,
        )

    def detect(self, image_path: str) -> PageResult:
        """Run Stage 1 locally without touching remote recognizer."""
        return self.layout.detect(image_path)

    def parse(self, image_path: str) -> PageResult:
        return self.ocr.run(image_path, self.layout.detect(image_path))

    def parse_batch(self, image_paths: list[str]) -> list[PageResult]:
        return [self.parse(p) for p in image_paths]

    def health(self) -> bool:
        try:
            models = self.backend._client.models.list()
        except Exception as exc:
            logger.warning("Health check failed: %s", exc)
            return False
        return any(getattr(m, "id", None) == self.backend.model for m in models.data)

    def close(self) -> None:
        self.ocr.close()
        self.layout.close()

    def __enter__(self) -> AnantaClient:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


# Backward compatibility alias
OCRClient = AnantaClient
