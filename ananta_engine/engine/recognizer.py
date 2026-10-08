"""AnantaBlockOCR recognizer backends and crop request building.

Heavy imports (torch, transformers) are kept inside methods to preserve instant CLI responsiveness.
Includes upstream fix for Qwen3.5 generation_config EOS token bug (248044 -> 262146 <|im_end|>).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

from .types import CropConfig, RecognizerConfig

if TYPE_CHECKING:
    from PIL.Image import Image

CORRECT_EOS = 262146  # <|im_end|>
BROKEN_EOS = 248044


class CropRequest(NamedTuple):
    image: Image
    prompt: str


@runtime_checkable
class RecognizerBackend(Protocol):
    """Recognizer backend protocol: transcribe accepts crop requests and returns texts."""

    def transcribe(self, requests: list[CropRequest]) -> list[str]: ...

    def close(self) -> None: ...


def build_requests(blocks, page, crop_cfg: CropConfig, table_format) -> tuple[list[CropRequest], list[int]]:
    """Crop each block and pair it with its specialized prompt.

    Returns:
      (requests, orders): The crop requests and corresponding reading order ranks.
    """
    from .crops import area_clamp, crop_for
    from ..templates.contract import prompt_for

    requests, orders = [], []
    for block in blocks:
        crop = crop_for(block, page, crop_cfg)
        if crop is None:
            continue
        requests.append(
            CropRequest(area_clamp(crop, crop_cfg), prompt_for(block.type, table_format))
        )
        orders.append(block.order)
    return requests, orders


class HfRecognizer:
    """Reference recognizer running on Hugging Face transformers.

    Lightweight quickstart path that runs on CPU or GPU without requiring vLLM.
    Ideal for single-page exploration and environments without Triton/vLLM.
    """

    def __init__(
        self,
        ckpt: str | None = None,
        config: RecognizerConfig | None = None,
        device: str = "auto",
        attn_implementation: str = "sdpa",
        batch_size: int = 8,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from .checkpoints import resolve_ckpt

        self._torch = torch
        self.config = config or RecognizerConfig()
        self.batch_size = batch_size
        self.ckpt = resolve_ckpt("recognizer", ckpt)

        self.processor = AutoProcessor.from_pretrained(self.ckpt)
        tokenizer = self.processor.tokenizer
        tokenizer.padding_side = "left"

        if device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            target_device = device

        dtype = getattr(torch, self.config.dtype) if target_device == "cuda" else torch.float32

        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": target_device if target_device != "cpu" else None,
        }
        if target_device == "cuda":
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.ckpt,
            **model_kwargs
        )
        if target_device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()

        # Fix upstream bug: Shipped checkpoint generation_config carries eos_token_id 248044,
        # which is a wordpiece, not a turn terminator. Enforce correct token 262146 (<|im_end|>).
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos in (None, BROKEN_EOS):
            eos = CORRECT_EOS
        self.eos_token_id = eos
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None) or self.eos_token_id

    def _prompt(self, text: str) -> str:
        return self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
            add_generation_prompt=True,
            tokenize=False,
        )

    def transcribe(self, requests: list[CropRequest]) -> list[str]:
        if not requests:
            return []
        texts: list[str] = []
        for i in range(0, len(requests), self.batch_size):
            chunk = requests[i : i + self.batch_size]
            inputs = self.processor(
                text=[self._prompt(r.prompt) for r in chunk],
                images=[r.image for r in chunk],
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            prompt_len = inputs["input_ids"].shape[-1]

            with self._torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=self.eos_token_id,
                    pad_token_id=self.pad_token_id,
                )
            texts.extend(self.processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True))
        return [t.strip() for t in texts]

    def close(self) -> None:
        self.model = None
        self.processor = None
