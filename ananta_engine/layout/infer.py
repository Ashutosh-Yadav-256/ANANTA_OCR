"""Single-image inference for IndicDocLayout -> boxes, labels, and reading order.

Emits detected blocks in the normalized schema:
  [{bbox: [y0, x0, y1, x1] (0-1000), label: str, reading_order: int, score: float}]
"""

from __future__ import annotations

import numpy as np
import torch

from .labels import ID2LABEL
from .order_loss import decode_order, pairwise_scores

_CACHE = {}


def cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    """Center-form boxes [cx, cy, w, h] -> corner form [x0, y0, x1, y1]."""
    c = b.clone()
    c[..., 0], c[..., 1] = b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2
    c[..., 2], c[..., 3] = b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2
    return c


def get_model(ckpt: str, device: str = "cuda"):
    """Load and cache an IndicDocLayout model checkpoint."""
    key = (ckpt, device)
    if key not in _CACHE:
        from .modeling_ppdoc import PPDocLayoutV3Trainable

        _CACHE[key] = PPDocLayoutV3Trainable.from_pretrained(ckpt).to(device).eval()
    return _CACHE[key]


@torch.no_grad()
def infer(model, pil_img, conf: float = 0.5, img_size: int = 1024, device: str = "cuda") -> list[dict]:
    """Perform layout detection and reading order decoding on a single PIL image."""
    im = pil_img.convert("RGB").resize((img_size, img_size))
    x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255.0)
    out = model(pixel_values=x[None].to(device))
    scores, labels = out.logits.sigmoid().max(-1)
    boxes = cxcywh_to_xyxy(out.pred_boxes)[0]

    if getattr(out, "order_logits", None) is not None:
        order_scores = out.order_logits[0]
    else:
        nq = model.config.num_queries
        order_scores = pairwise_scores(out.last_hidden_state[:, -nq:], model.ro_q, model.ro_k)[0]

    keep = (scores[0] > conf).nonzero().squeeze(-1)
    if keep.numel() == 0:
        return []
    kept_boxes, kept_scores, kept_labels = boxes[keep], scores[0][keep], labels[0][keep]

    # Reading order is decoded over kept boxes only
    sub = order_scores[keep][:, keep].cpu().float()
    sequence = decode_order(sub).tolist()
    rank = [0] * len(sequence)
    for r, position in enumerate(sequence):
        rank[position] = r + 1

    content = []
    for j in range(keep.numel()):
        x0, y0, x1, y1 = kept_boxes[j].tolist()
        content.append(
            {
                "bbox": [
                    round(y0 * 1000, 1),
                    round(x0 * 1000, 1),
                    round(y1 * 1000, 1),
                    round(x1 * 1000, 1),
                ],
                "label": ID2LABEL[int(kept_labels[j])],
                "reading_order": rank[j],
                "score": round(float(kept_scores[j]), 3),
            }
        )
    return content

