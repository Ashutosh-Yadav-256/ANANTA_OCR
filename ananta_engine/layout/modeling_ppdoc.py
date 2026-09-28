"""Trainable and inferable PP-DocLayoutV3 with integrated reading-order head.

Extends Hugging Face's PPDocLayoutV3ForObjectDetection with:
  - Reading order head outputs (pairwise precedence logits)
  - Custom loss computation combining RT-DETR detection loss and locality-weighted GCE order loss
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

try:
    from transformers import PPDocLayoutV3Config, PPDocLayoutV3ForObjectDetection
    from transformers.loss.loss_rt_detr import RTDetrHungarianMatcher
    from transformers.utils import ModelOutput
    _TRANSFORMERS_PPDOC_OK = True
except ImportError:
    _TRANSFORMERS_PPDOC_OK = False
    PPDocLayoutV3Config = object
    PPDocLayoutV3ForObjectDetection = nn.Module
    RTDetrHungarianMatcher = object
    from dataclasses import dataclass
    ModelOutput = object

from .order_loss import locality_gce


@dataclass
class PPDocOutput(ModelOutput if _TRANSFORMERS_PPDOC_OK else object):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    pred_boxes: torch.FloatTensor | None = None
    order_logits: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None


class PPDocLayoutV3Trainable(PPDocLayoutV3ForObjectDetection):
    def __init__(self, config):
        if not _TRANSFORMERS_PPDOC_OK:
            raise ImportError(
                "PPDocLayoutV3 requires transformers with PP-DocLayoutV3 support (transformers>=4.48). "
                "Please upgrade via: pip install 'transformers>=4.48'"
            )
        super().__init__(config)
        self.lambda_order = getattr(config, "lambda_order", 5.0)
        self._matcher = RTDetrHungarianMatcher(config)
        self.loss_type = "RTDetrForObjectDetection"

    @classmethod
    def build(cls, ckpt, num_labels, id2label, label2id, lambda_order=5.0):
        config = PPDocLayoutV3Config.from_pretrained(
            ckpt, num_labels=num_labels, id2label=id2label, label2id=label2id
        )
        config.lambda_order = lambda_order
        config.loss_type = "RTDetrForObjectDetection"
        config.num_denoising = 0
        defaults = {
            "use_focal_loss": True,
            "auxiliary_loss": True,
            "weight_loss_vfl": 1.0,
            "weight_loss_bbox": 5.0,
            "weight_loss_giou": 2.0,
            "matcher_class_cost": 2.0,
            "matcher_bbox_cost": 5.0,
            "matcher_giou_cost": 2.0,
            "matcher_alpha": 0.25,
            "matcher_gamma": 2.0,
            "focal_loss_alpha": 0.25,
            "focal_loss_gamma": 2.0,
            "eos_coefficient": 1e-4,
        }
        for k, v in defaults.items():
            if not hasattr(config, k):
                setattr(config, k, v)
        model = cls.from_pretrained(ckpt, config=config, ignore_mismatched_sizes=True)
        for m in model.modules():
            if (isinstance(m, nn.Linear) and m.out_features == num_labels) or (
                isinstance(m, nn.Embedding) and m.num_embeddings == num_labels + 1
            ):
                m.reset_parameters()
        return model

    def _order_loss(self, order_logits, logits, pred_boxes, labels):
        idx = self._matcher({"logits": logits, "pred_boxes": pred_boxes}, labels)
        tot, n = 0.0, 0
        for b, (src, tgt) in enumerate(idx):
            if src.numel() < 2:
                continue
            order = labels[b]["reading_order"][tgt]
            S = order_logits[b][src][:, src]
            tot = tot + locality_gce(S, order)
            n += 1
        return tot / max(n, 1) if n else order_logits.sum() * 0.0

    def forward(self, pixel_values, pixel_mask=None, labels=None, **kwargs):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        dn = outputs.denoising_meta_values if self.training else None
        outputs_class = outputs.intermediate_logits
        outputs_coord = outputs.intermediate_reference_points
        logits, pred_boxes = outputs_class[:, -1], outputs_coord[:, -1]
        order_logits = outputs.out_order_logits[:, -1]
        loss = None
        if labels is not None:
            loss, _, _ = self.loss_function(
                logits,
                labels,
                self.device,
                pred_boxes,
                self.config,
                outputs_class,
                outputs_coord,
                enc_topk_logits=outputs.enc_topk_logits,
                enc_topk_bboxes=outputs.enc_topk_bboxes,
                denoising_meta_values=dn,
            )
            loss = loss + self.lambda_order * self._order_loss(
                order_logits, logits, pred_boxes, labels
            )
        return PPDocOutput(
            loss=loss,
            logits=logits,
            pred_boxes=pred_boxes,
            order_logits=order_logits,
            last_hidden_state=outputs.last_hidden_state,
        )

