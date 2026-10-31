# Third-Party Notices and Component Provenance

This document provides legal attribution, component lineage, and origin notices for software libraries, model architectures, and pretrained checkpoints incorporated in or adapted by ANANTA OCR.

---

## 1. Architectural Component Lineage

### PP-DocLayoutV3 (PaddleOCR)
- **Original Authors**: Baidu Inc.
- **Repository**: https://github.com/PaddlePaddle/PaddleOCR
- **License**: Apache License 2.0
- **Incorporated In**: `ananta_engine/layout/`
- **Modifications**: 
  - Implementation of antisymmetric pairwise reading-order head:
    $$S_{ij} = \frac{(W_q q_i)^T (W_k q_j) - (W_q q_j)^T (W_k q_i)}{\sqrt{d_r}}$$
  - Supervised with Locality-Weighted Generalized Cross-Entropy (GCE) loss.
  - Hungarian bipartite matching alignment for query-to-ground-truth ordering supervision.

### Bodhan AI (Indic Document Intelligence)
- **Original Authors**: Bodhan AI Contributors
- **Repository**: https://github.com/Bodhan-AI/bodhan_genai
- **License**: Apache License 2.0
- **Incorporated In**: `ananta_engine/engine/`, `ananta_engine/templates/`, `ananta_engine/serving/`
- **Modifications**:
  - Modular two-stage block recognition pipeline.
  - Area-clamped aspect-preserving crop scaling ($256^2 \le A \le 1536^2$).
  - Post-processing math engine with Unicode Indic script protection (`\text{...}`).
  - Structured HTML table parsing with colspan/rowspan preservation.
  - Turnkey distributed serving client (`AnantaClient`) and HTTP backend.

### DeepSeek-V2 MoE Architecture
- **Original Authors**: DeepSeek-AI
- **Repository**: https://github.com/deepseek-ai/DeepSeek-V2
- **License**: MIT License
- **Incorporated In**: `modeling_deepseekv2.py`, `configuration_deepseek_v2.py`, `modeling_anantaocr.py`
- **Modifications**:
  - Mixture-of-Experts decoder with 64 routed experts and 2 shared experts (Top-6 activated).
  - Custom autoregressive generation loop (`_uocr_generate`) with DynamicCache and page-count guards.
  - Direct tensor device mapping replacing hardcoded CUDA assumptions.

### Dual Vision Backbone (DeepEncoder)
- **Segment Anything Model (SAM-ViT-B)**: Meta AI Research (Apache License 2.0)
- **CLIP-L/14@224**: OpenAI (MIT License)
- **Incorporated In**: `deepencoder.py`
- **Modifications**:
  - Concatenation of high-frequency geometric boundary features (SAM) with dense semantic character tokens (CLIP).
  - Linear feature projection down to 1280 hidden dimensions, yielding 273 visual tokens per standard page.

### Sparse Reference Sliding Window Attention (DASH) & Triton Kernels
- **Original Authors**: DeepSeek-AI / OpenAI Triton Contributors
- **License**: MIT License
- **Incorporated In**: `dash_attention.py`, `sparse_gather_triton.py`, `sparse_gather_e2e_triton.py`, `_sparse_base_impl.py`
- **Modifications**:
  - 1.5-entmax differentiable block router with learned log-prior bias.
  - Custom forward and backward Triton gather kernels for end-to-end gradient propagation.
  - PyTorch reference fallbacks for environments without Triton GPU support.

---

## 2. Checkpoint Provenance and Licensing

Pretrained weights used by ANANTA OCR are downloaded dynamically via `download_weights.py` and are subject to their respective upstream licenses:

| Component | Upstream Checkpoint Source | License | Format |
| :--- | :--- | :--- | :--- |
| Stage 1 Layout Detector | `bodhan-ai/indic-ocr` (layout) | Apache 2.0 | PyTorch / SafeTensors |
| Stage 2 Block Recognizer | `bodhan-ai/indic-ocr` (recognizer) | Apache 2.0 | SafeTensors (262k Vocab) |
| Multi-Page VLM Backbone | `baidu/Unlimited-OCR-2` | MIT License | SafeTensors (~3B MoE) |

The multi-page VLM backbone weights are fetched from `baidu/Unlimited-OCR-2` on Hugging Face, licensed under the MIT License by Baidu Inc. ANANTA OCR adapts this foundational multimodal backbone with SR-SWA DASH sparse attention routing, custom autoregressive generation loops, and specialized prompt engineering.

Pretrained model weights must be downloaded and used in accordance with their respective model cards and license agreements.

---

## 3. License Texts

### Apache License, Version 2.0
```text
Copyright 2026 Baidu Inc.
Copyright 2026 Bodhan AI Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### MIT License
```text
Copyright (c) 2026 DeepSeek-AI
Copyright (c) 2026 OpenAI
Copyright (c) 2026 AnantaOCR Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

