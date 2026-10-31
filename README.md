# ANANTA OCR: Advanced Visual Document Intelligence

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/Triton-Hardware--Accelerated-00599C.svg)](https://github.com/openai/triton)
[![Architecture: MoE + 2-Stage VLM](https://img.shields.io/badge/Architecture-MoE%20%2B%202--Stage%20VLM-purple.svg)](#architecture-specifications)
[![Attention: SR-SWA](https://img.shields.io/badge/Attention-SR--SWA%20%2F%20DASH-green.svg)](#subsystem-1-ananta-vlm-engine)
[![Languages: 22+ Indic & English](https://img.shields.io/badge/Languages-22%2B%20Indic%20%2B%20English-orange.svg)](#supported-languages-scripts--syntaxes)

A unified visual document parsing architecture combining unlimited-horizon multi-page document understanding across 100+ pages with fine-grained 37-class layout analysis and multilingual block recognition.

[Overview](#system-overview) • [Supported Languages](#supported-languages-scripts--syntaxes) • [Architecture Flows](#system-architecture-flows) • [Technical Specifications](#architecture-specifications) • [Feature Matrix](#feature-comparison-matrix) • [37-Class Taxonomy](#37-class-layout-taxonomy-reference) • [CLI Reference](#cli-reference) • [Python API](#python-api-reference) • [Verification Suite](#verification-suite) • [License](#license-and-attribution)

---

### [Open Interactive Architecture Visualizer](assets/architecture.html)

</div>

---

## System Overview

ANANTA OCR unifies two document understanding paradigms into a single engine:

1. **Ananta VLM Engine (`vlm`)**: A multi-page Vision-Language Model engineered for ultra-long documents (100+ pages). Utilizes dual-backbone visual compression (SAM-ViT-B + CLIP-L) to compress standard $1024 \times 1024$ pages into 273 visual tokens (16x spatial downsampling). Employs Sparse Reference Sliding Window Attention (SR-SWA / DASH) with 1.5-entmax routing to ensure strictly bounded $O(1)$ decoding memory and latency per step over a DeepSeek-V2 Mixture-of-Experts (MoE) decoder.
2. **Ananta Block Engine (`block`)**: A high-precision two-stage pipeline. Stage 1 employs AnantaDocLayout (PP-DocLayoutV3 RT-DETR, 33M parameters) to detect a 37-class document taxonomy with an antisymmetric reading-order head. Stage 2 executes AnantaBlockOCR (Qwen3.5-0.8B VLM) with a 262,144-token multilingual vocabulary covering all 22 official Indian languages and English, utilizing specialized prompts for prose, display equations, and merged HTML tables.
3. **Ananta Hybrid Engine (`hybrid`)**: Combines AnantaDocLayout topological segmentation with dynamic multi-engine routing, allocating complex visual regions to AnantaBlockOCR while preserving global document coherence through Ananta VLM.

---

## Supported Languages, Scripts & Syntaxes

ANANTA OCR provides comprehensive document recognition across all 22 official scheduled Indian languages, English, global languages, and specialized scientific/technical notation systems. Powered by a 262,144-token multilingual vocabulary and hybrid vision backbones, the model natively transcribes complex conjunct consonants, matras, Perso-Arabic ligatures, mixed-script code-switching, and mathematical environments without character truncation or unk degradation.

### 1. The 22 Scheduled Official Languages of India

ANANTA OCR natively supports every language recognized under the Eighth Schedule of the Constitution of India:

| # | Language | ISO 639-1 / 639-3 | Script | Language Family | Unicode Range | Native Numeral System |
| :- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Assamese | `as` / `asm` | Eastern Nagari (Assamese) | Indo-Aryan | `U+0980` - `U+09FF` | `০ ১ ২ ৩ ৪ ৫ ৬ ৭ ৮ ৯` |
| 2 | Bengali | `bn` / `ben` | Bengali | Indo-Aryan | `U+0980` - `U+09FF` | `০ ১ ২ ৩ ৪ ৫ ৬ ৭ ৮ ৯` |
| 3 | Bodo | `brx` | Devanagari | Sino-Tibetan | `U+0900` - `U+097F` | `० १ २ ३ ४ ५ ६ ७ ८ ९` |
| 4 | Dogri | `doi` | Devanagari | Indo-Aryan | `U+0900` - `U+097F` | `० १ २ ३ ४ ५ ६ ७ ८ ९` |
| 5 | Gujarati | `gu` / `guj` | Gujarati | Indo-Aryan | `U+0A80` - `U+0AFF` | `૦ ૧ ૨ ૩ ૪ ૫ ૬ ૭ ૮ ૯` |
| 6 | Hindi | `hi` / `hin` | Devanagari | Indo-Aryan | `U+0900` - `U+097F` | `० १ २ ३ ४ ५ ६ ७ ८ ९` |
| 7 | Kannada | `kn` / `kan` | Kannada | Dravidian | `U+0C80` - `U+0CFF` | `೦ ೧ ೨ ೩ ೪ ೫ ೬ ೭ ೮ ೯` |
| 8 | Kashmiri | `ks` / `kas` | Perso-Arabic (Nastaliq) / Devanagari | Indo-Aryan | `U+0600` - `U+06FF` / `U+0900` - `U+097F` | `۰ ۱ ۲ ۳ ۴ ۵ ۶ ۷ ۸ ۹` / `०-९` |
| 9 | Konkani | `kok` | Devanagari / Latin / Kannada | Indo-Aryan | `U+0900` - `U+097F` / `U+0020` - `U+007F` | `० १ २ ३ ४ ५ ६ ७ ८ ९` / `0-9` |
| 10 | Maithili | `mai` | Devanagari / Tirhuta | Indo-Aryan | `U+0900` - `U+097F` / `U+11480` - `U+114DF` | `० १ २ ३ ४ ५ ६ ७ ८ ९` |
| 11 | Malayalam | `ml` / `mal` | Malayalam | Dravidian | `U+0D00` - `U+0D7F` | `൦ ൧ ൨ ൩ ൪ ൫ ൬ ൭ ൮ ൯` |
| 12 | Manipuri (Meitei) | `mni` | Meetei Mayek / Eastern Nagari | Sino-Tibetan | `U+ABC0` - `U+ABFF` / `U+0980` - `U+09FF` | `꯰ ꯱ ꯲ ꯳ ꯴ ꯵ ꯶ ꯷ ꯸ ꯹` |
| 13 | Marathi | `mr` / `mar` | Devanagari (Balbodh) | Indo-Aryan | `U+0900` - `U+097F` | `० १ २ ३ ४ ५ ६ ७ ८ ९` |
| 14 | Nepali | `ne` / `nep` | Devanagari | Indo-Aryan | `U+0900` - `U+097F` | `० १ २ ३ ४ ५ ६ ७ ८ ९` |
| 15 | Odia | `or` / `ori` | Odia | Indo-Aryan | `U+0B00` - `U+0B7F` | `୦ ୧ ୨ ୩ ୪ ୫ ୬ ୭ ୮ ୯` |
| 16 | Punjabi | `pa` / `pan` | Gurmukhi | Indo-Aryan | `U+0A00` - `U+0A7F` | `੦ ੧ ੨ ੩ ੪ ੫ ੬ ੭ ੮ ੯` |
| 17 | Sanskrit | `sa` / `san` | Devanagari (Classical & Vedic Accents) | Indo-Aryan | `U+0900` - `U+097F`, `U+1CD0` - `U+1CFF` | `० १ २ ३ ४ ५ ६ ७ ८ ९` |
| 18 | Santali | `sat` | Ol Chiki | Austroasiatic | `U+1C50` - `U+1C7F` | `᱐ ᱑ ᱒ ᱓ ᱔ ᱕ ᱖ ᱗ ᱘ ᱙` |
| 19 | Sindhi | `sd` / `snd` | Perso-Arabic / Devanagari | Indo-Aryan | `U+0600` - `U+06FF` / `U+0900` - `U+097F` | `۰ ۱ ۲ ۳ ۴ ۵ ۶ ۷ ۸ ۹` / `०-९` |
| 20 | Tamil | `ta` / `tam` | Tamil | Dravidian | `U+0B80` - `U+0BFF` | `௧ ௨ ௩ ௪ ௫ ௬ ௭ ௮ ௯ ௰` |
| 21 | Telugu | `te` / `tel` | Telugu | Dravidian | `U+0C00` - `U+0C7F` | `౦ ౧ ౨ ౩ ౪ ౫ ౬ ౭ ౮ ౯` |
| 22 | Urdu | `ur` / `urd` | Perso-Arabic (Nastaliq & Naskh) | Indo-Aryan | `U+0600` - `U+06FF`, `U+FB50` - `U+FDFF` | `۰ ۱ ۲ ۳ ۴ ۵ ۶ ۷ ۸ ۹` |

### 2. Global & International Languages

| Language | Code | Script | Typical Application & Document Domains |
| :--- | :--- | :--- | :--- |
| English | `en` | Latin | International research publications, legal contracts, business filings, software manuals |
| Chinese (Simplified & Traditional) | `zh` / `cmn` | CJK Unified Ideographs (`U+4E00` - `U+9FFF`) | Technical literature, trade documentations, multi-page regulatory forms |
| Western European Languages | `es`, `fr`, `de`, `pt`, `it` | Latin (with extended diacritics) | Academic articles, international reports, multi-lingual forms |

### 3. Mixed-Script Code-Switching

Indian educational, governmental, judicial, and media documents frequently interleave multiple scripts within a single line or paragraph. ANANTA OCR provides seamless transcription for:

- **Hinglish & Vernacular-English Hybrids**: Hindi-English, Bengali-English, Tamil-English, Telugu-English mixtures common in school textbooks, entrance examinations (CBSE, JEE, NEET, UPSC), and judicial pleadings.
- **Bi-Scriptual Government Gazettes**: Bilingual official notices with parallel or interspersed English and vernacular text blocks.
- **Latin Technical Terms in Vernacular Prose**: Preserves English chemical symbols, algebraic variables, medical terminology, and URLs embedded in Indic sentences without phonological distortion.

### 4. Mathematical, Scientific & Technical Syntaxes

ANANTA OCR transcribes domain-specific notations into machine-readable standards:

| Syntax Domain | Target Format | Representation & Capabilities |
| :--- | :--- | :--- |
| Inline Mathematics | LaTeX (`$...$`) | Embedded algebraic variables, Greek symbols (`\alpha`, `\beta`, `\theta`), superscripts, subscripts, limits |
| Display Mathematics | LaTeX (`$$...$$`, `align`, `gather`) | Multi-line derivations, fractions (`\frac`), radicals (`\sqrt`), integral/summation calculus, linear algebra matrices |
| Indic Math Safeguard | LaTeX `\text{...}` Wrappers | Automatic post-processing wraps any Unicode Indic glyphs inside mathematical spans in `\text{...}`, eliminating KaTeX/MathJax compilation breakage |
| Complex Tables | HTML `<table>` or GFM Markdown | Full preservation of multi-column spans (`colspan`), multi-row spans (`rowspan`), multi-level header hierarchies, and empty cell alignments |
| Source Code & Algorithms | Markdown Code Blocks (````...````) | Indentation-accurate transcription of Python, C/C++, Java, SQL, JSON, YAML, pseudocode, and terminal listings |
| Chemical Formulations | LaTeX / Text | Molecular formulas, state indicators (aq, s, g), reaction arrows, bond representations |
| Examination Layouts | Standardized Markdown | Question stems, answer option grids `(A), (B), (C), (D)`, solved problem steps, multi-part exercise hierarchies |

### 5. Multi-System Numeral Recognition

ANANTA OCR transcribes both international Arabic numerals and native regional numeral scripts:

- **International Standard**: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`
- **Devanagari (Hindi, Marathi, Sanskrit, Nepali)**: `०, १, २, ३, ४, ५, ६, ७, ८, ९`
- **Bengali / Assamese**: `০, ১, ২, ৩, ৪, ৫, ৬, ৭, ৮, ৯`
- **Gurmukhi (Punjabi)**: `੦, ੧, ੨, ੩, ੪, ੫, ੬, ੭, ੮, ੯`
- **Gujarati**: `૦, ૧, ૨, ૩, ૪, ૫, ૬, ૭, ૮, ૯`
- **Odia**: `୦, ୧, ୨, ୩, ୪, ୫, ୬, ୭, ୮, ୯`
- **Tamil**: `௧, ௨, ௩, ௪, ௫, ௬, ௭, ௮, ௯, ௰`
- **Telugu**: `౦, ౧, ౨, ౩, ౪, ౫, ౬, ౭, ౮, ౯`
- **Kannada**: `೦, ೧, ೨, ೩, ೪, ೫, ೬, ೭, ೮, ೯`
- **Malayalam**: `൦, ൧, ൨, ൩, ൪, ൫, ൬, ൭, ൮, ൯`
- **Perso-Arabic (Urdu, Sindhi, Kashmiri)**: `۰, ۱, ۲, ۳, ۴, ۵, ۶, ۷, ۸, ۹`
- **Ol Chiki (Santali)**: `᱐, ᱑, ᱒, ᱓, ᱔, ᱕, ᱖, ᱗, ᱘, ᱙`

### 6. Architectural Tokenizer Specifications

- **Vocabulary Dimension**: 262,144 subword units.
- **Coverage**: Byte-Pair Encoding (BPE) trained across massive multilingual and multi-script corpora.
- **Conjunct Consonant Preservation**: Native multi-character subwords for complex Indic conjuncts (samyuktaksharas like क्ष, त्र, ज्ञ, श्र, द्ध, ष्ट्र) preventing excessive token fragmentation.
- **Zero-Width Joiner Support**: Proper boundary encoding for Zero-Width Joiner (`U+200D`) and Zero-Width Non-Joiner (`U+200C`) required for correct Malayalam, Bengali, and Perso-Arabic rendering.

---

## System Architecture Flows

### 1. Macro Orchestration Flow

```mermaid
flowchart TD
    DOC["Input Document (PDF / Multi-Page Images)"] --> ROUTE{"Engine Dispatcher"}

    subgraph ENGINE_VLM ["Ananta VLM Engine (--engine vlm)"]
        ROUTE -->|"vlm"| V_ENC["DeepEncoder: Dual Feature Extraction"]
        V_ENC --> V_SAM["SAM-ViT-B (Edge & Geometric Layout)"]
        V_ENC --> V_CLIP["CLIP-L/14@224 (Dense Semantics)"]
        V_SAM & V_CLIP --> V_PROJ["Linear Projector (2048 -> 1280)"]
        V_PROJ --> V_TOK["Visual Tokens (273 tokens/page)"]
        V_TOK --> V_ATTN["SR-SWA DASH Attention (1.5-entmax Routing)"]
        V_ATTN --> V_MOE["DeepSeek-V2 MoE (12 Layers, Top-6 / 64 Experts)"]
        V_MOE --> V_OUT["Markdown with Grounded Coordinate Tags"]
    end

    subgraph ENGINE_BLOCK ["Ananta Block Engine (--engine block)"]
        ROUTE -->|"block"| B_LAY["Stage 1: AnantaDocLayout (RT-DETR 33M)"]
        B_LAY --> B_TAX["37-Class Boundary Box Extraction"]
        B_TAX --> B_RO["Antisymmetric Reading Order Head (Locality GCE)"]
        B_RO --> B_DEDUP["Geometric Deduplication & Nested Box Absorption"]
        B_DEDUP --> B_CROP["Area-Clamped Cropping (256^2 to 1536^2 pixels)"]
        B_CROP --> B_OCR["Stage 2: AnantaBlockOCR (Qwen3.5-0.8B + 262k Vocab)"]
        B_OCR --> B_PROMPT["Targeted Prompt Execution (Text / LaTeX / HTML Table)"]
        B_PROMPT --> B_REPAIR["Math Engine: Indic Script Protection & Dehyphenation"]
        B_REPAIR --> B_OUT["Structured PageResult JSON & Validated Markdown"]
    end

    subgraph ENGINE_HYBRID ["Ananta Hybrid Engine (--engine hybrid)"]
        ROUTE -->|"hybrid"| H_SEG["AnantaDocLayout Structural Partitioning"]
        H_SEG --> H_DISP["Content-Type Router"]
        H_DISP -->|"Indic / Math / Tables"| B_OCR
        H_DISP -->|"Global Narrative"| V_MOE
        H_DISP --> H_MERGE["Topological Document Synthesis"]
        H_MERGE --> H_OUT["Unified Document Presentation"]
    end
```

### 2. Ananta VLM Decoding & Sparse Attention Lifecycle

```mermaid
flowchart LR
    subgraph INPUT_PAGES ["Page Sequences"]
        P1["Page 1 (273 tok)"]
        P2["Page 2 (273 tok)"]
        PN["Page N (273 tok)"]
    end

    subgraph ATTENTION_ROUTING ["SR-SWA / DASH Sparse Routing"]
        P1 & P2 & PN --> BLK_SUM["Block Summary Generation (32-tok KV Blocks)"]
        BLK_SUM --> ROUTER["1.5-entmax Differentiable Top-12 Block Router"]
        ROUTER --> REMOTE_KV["12 Selected Remote Reference Blocks"]
        LOCAL_WND["Local Sliding Window (128 Tokens)"]
        REMOTE_KV & LOCAL_WND --> FUSED_KERN["Custom Fused Triton Attention Kernel"]
    end

    subgraph MOE_DECODER ["Mixture-of-Experts Decoder"]
        FUSED_KERN --> GATING["Affinity Router"]
        GATING --> EX_SHARED["2 Shared Experts"]
        GATING --> EX_ROUTED["64 Routed Experts (Top-6 Activated)"]
        EX_SHARED & EX_ROUTED --> LINEAR["Output Projection & Generation"]
    end
```

### 3. Ananta Block Engine Two-Stage Detail

```mermaid
flowchart TD
    PAGE_IMG["Raw Document Page Image"] --> RT_DETR["PP-DocLayoutV3 RT-DETR Backbone"]
    
    subgraph STAGE_1 ["Stage 1: Layout & Reading Order"]
        RT_DETR --> DET_BOXES["37-Class Bounding Coordinates (0-1000 Normalized)"]
        RT_DETR --> ORDER_HEAD["Query Projections (ro_q, ro_k)"]
        ORDER_HEAD --> PAIRWISE["Antisymmetric Precedence Matrix S[i,j]"]
        PAIRWISE --> TOPOSORT["Topological Order Sequence Decoding"]
        DET_BOXES & TOPOSORT --> MERGE_DET["Raw Ordered Blocks"]
        MERGE_DET --> GEOM_FILTER["Containment Filtering (contain=0.90)"]
        GEOM_FILTER --> NEST_EQ["Nested Equation Resolution in Paragraphs"]
    end

    subgraph STAGE_2 ["Stage 2: Specialized Crop Transcription"]
        NEST_EQ --> CROP_CLAMP["Aspect-Preserving Area Clamping"]
        CROP_CLAMP --> PROMPT_SEL{"Block Type Selector"}
        PROMPT_SEL -->|"Text"| P_TXT["Prose & Inline LaTeX Prompt"]
        PROMPT_SEL -->|"Equation"| P_EQ["Pure LaTeX Display Formula Prompt"]
        PROMPT_SEL -->|"Table"| P_TBL["HTML Table Grid (colspan/rowspan) Prompt"]
        P_TXT & P_EQ & P_TBL --> QWEN["Qwen3.5-0.8B VLM (Sarvam-30B Tokenizer)"]
        QWEN --> RAW_TEXT["Raw Token Streams"]
        RAW_TEXT --> LATEX_FIX["LaTeX Indic Protection: \\text{Unicode}"]
        LATEX_FIX --> DEHYPH["Hyphenated Word Rejoining across Breaks"]
    end

    DEHYPH --> FINAL_DOC["Final Markdown (.md) and PageResult (.json)"]
```

### 4. End-to-End Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant CLI as Unified CLI (run_inference.py)
    participant Pipe as Unified Pipeline
    participant Lay as AnantaDocLayout (Stage 1)
    participant Blk as AnantaBlockOCR (Stage 2)
    participant VLM as Ananta VLM Engine
    participant Post as Post-Processing Engine

    User->>CLI: Execute run_inference.py (--engine, --images)
    CLI->>Pipe: Initialize UnifiedOCR(engine)
    
    alt Engine is 'vlm'
        Pipe->>VLM: Ingest document images
        VLM->>VLM: Extract dual visual features (SAM + CLIP)
        VLM->>VLM: Apply DASH sparse attention over MoE layers
        VLM-->>Pipe: Return raw tagged tokens
        Pipe->>Post: Execute post_process_document()
        Post-->>Pipe: Emit structured Markdown
    else Engine is 'block'
        Pipe->>Lay: Detect document layout
        Lay->>Lay: Extract 37-class boxes & decode reading order
        Lay-->>Pipe: Return ordered Block sequence
        Pipe->>Blk: Dispatch area-clamped crops with type prompts
        Blk->>Blk: Transcribe via Qwen3.5-0.8B (greedy decoding)
        Blk-->>Pipe: Return transcribed block content
        Pipe->>Post: Execute repair_math() & dehyphenate()
        Post-->>Pipe: Emit PageResult JSON & Markdown
    else Engine is 'hybrid'
        Pipe->>Lay: Execute layout segmentation
        Lay-->>Pipe: Structured partitions
        Pipe->>Blk: Route complex table & equation regions
        Pipe->>VLM: Route continuous prose
        Pipe->>Post: Reconstruct unified document representation
        Post-->>Pipe: Emit consolidated output
    end

    Pipe-->>CLI: Return UnifiedOCRResult
    CLI-->>User: Save outputs to disk (.md, .json)
```

---

## Architecture Specifications

### Dual Vision Backbone (DeepEncoder)

Standard ViT backbones generate unwieldy token sequences for high-resolution images ($64 \times 64 = 4096$ tokens per image). ANANTA OCR resolves this bottleneck through a dual-encoder architecture:

- **SAM-ViT-B**: Segment Anything Model image encoder providing high-frequency boundary representation, stroke sharpness, and layout geometry.
- **CLIP-L/14@224**: Vision Transformer capturing dense semantic character representations.
- **Linear Feature Projector**: Fuses concatenated 2048-dimensional embeddings ($768 + 1024$) down to 1280 dimensions, achieving an exact allocation of **273 visual tokens per page** (16x spatial compression relative to conventional baselines).

### SR-SWA (DASH) Attention Engine

To eliminate quadratic complexity $O(N^2)$ across multi-page contexts, ANANTA OCR implements Sparse Reference Sliding Window Attention:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K_{\text{active}}^T}{\sqrt{d}} + B\right) V_{\text{active}}$$

Where:
- $K_{\text{active}}$ consists of 12 global visual reference blocks dynamically selected via differentiable 1.5-entmax routing over precomputed block summaries, plus a local sliding window of 128 dynamic text tokens.
- Constant $O(1)$ decoding memory and step latency are preserved uniformly from Page 1 to Page 100+.

### DeepSeek-V2 Lineage MoE Decoder

- **Layer Count**: 12 Transformer decoder layers.
- **Hidden Dimension**: 1280.
- **Expert Architecture**: 64 routed experts + 2 shared experts.
- **Routing Mechanism**: Dynamic affinity gating activating the Top-6 routed experts per token.
- **Active Parameters**: Approximately 500M active parameters out of 3B total parameters.

### Reading-Order Head & Locality GCE

Stage 1 layout detection utilizes an antisymmetric bilinear formulation:

$$S_{ij} = \frac{(W_q q_i)^T (W_k q_j) - (W_q q_j)^T (W_k q_i)}{\sqrt{d_r}}$$

Trained via Locality-Weighted Generalized Cross-Entropy:

$$\mathcal{L}_{\text{GCE}} = \frac{1 - p_{ij}^q}{q} \cdot \exp\left(-\frac{|pos_i - pos_j|}{\tau}\right)$$

This penalizes inversion of adjacent blocks heavily while allowing robust global topological sequence recovery.

### Multilingual Tokenization & Math Repair

- **Tokenizer**: Sarvam-30B BPE tokenizer with 262,144 entries, providing native subword representations for all 22 official Indian languages and English.
- **LaTeX Math Engine**: Scans display and inline math spans for Unicode Indic runs (`\u0600-\u06ff`, `\u0900-\u0d7f`, `\u1c50-\u1c7f`) and wraps them in `\text{...}`, preventing KaTeX and MathJax compilation failures. Replaces bare newlines inside display environments with LaTeX line breaks (`\\`).

---

## Feature Comparison Matrix

| Dimension | Ananta VLM Engine (`vlm`) | Ananta Block Engine (`block`) | Ananta Hybrid Engine (`hybrid`) |
| :--- | :--- | :--- | :--- |
| **Primary Domain** | Multi-page PDFs (100+ pages), dense technical literature | Complex Indic scripts, educational exams, formulas | Mixed document portfolios with varied visual elements |
| **Model Structure** | DeepSeek-V2 MoE (~3B total / ~500M active) | RT-DETR (33M) + Qwen3.5-0.8B VLM | Dual architecture cooperative pipeline |
| **Token Cost / Page** | Strictly constant (273 tokens/page) | Crop adaptive ($256^2$ to $1536^2$ pixels) | Dynamically partitioned |
| **Attention Kernel** | SR-SWA 1.5-entmax DASH Sparse Attention | Dense SDPA / FlashAttention / vLLM | Hybrid routing |
| **Language Coverage** | Multilingual Latin & General | 22 Official Indian Languages + English | Complete cross-lingual coverage |
| **Layout Topology** | Grounded coordinate token sequence | 37-class taxonomy + pairwise reading order matrix | Topological block segmentation |
| **Table Formatting** | Markdown / HTML | HTML with colspan/rowspan & GFM Markdown | HTML with full merge preservation |
| **Formula Handling** | Inline and display LaTeX | Display LaTeX with automated Unicode protection | Multi-stage math verification |
| **Serving Options** | PyTorch / SGLang Prefix Cache | PyTorch / vLLM Continuous Batching / HTTP | End-to-end configurable |
| **Execution Hardware** | NVIDIA GPU (bfloat16 + Triton) | GPU (vLLM) or CPU (Hugging Face backend) | GPU recommended |

---

## 37-Class Layout Taxonomy Reference

The layout detection model identifies the following document components:

| Category | Classes Detected |
| :--- | :--- |
| **Headings & Structure** | `Title`, `Chapter-title`, `Section-title`, `Sub-section-title`, `Sub-sub-section-title`, `Chapter-end-section` |
| **Prose Elements** | `Paragraph`, `Footnote`, `Placeholder-text`, `Code` |
| **Educational Components**| `Question`, `Answer`, `MCQ`, `Solved-example`, `List`, `Infobox` |
| **Formulas & Mathematics**| `Equation`, `Expression` |
| **Tables & Visual Assets** | `Table`, `Table-caption`, `Image`, `Image-caption`, `Diagram`, `Chart` |
| **Publication Metadata** | `Header`, `Footer`, `Page-number`, `Folio`, `Table-of-contents`, `Index`, `Reference` |
| **Administrative & Media** | `Author`, `Dateline`, `Contact-info`, `Website-link`, `Advertisement`, `Flag` |

---

## Installation & Environment Setup

### 1. Pinned Dependencies

```bash
git clone https://github.com/Ashutosh-Yadav-256/ANANTA_OCR.git
cd ANANTA_OCR
pip install -r requirements.txt
```

### 2. Pretrained Weight Download

Download the required checkpoints from Hugging Face:

```bash
python download_weights.py --model vlm

python download_weights.py --model block

python download_weights.py --all
```

---

## CLI Reference

The unified command-line tool [`run_inference.py`](run_inference.py) provides access to all engines:

### Ananta VLM Mode (Multi-Page Ingestion)

```bash
python run_inference.py \
    --engine vlm \
    --images sample_docs/page1.jpg sample_docs/page2.jpg \
    --output-dir output/
```

### Ananta Block Mode (Two-Stage Layout & Block Recognition)

```bash
python run_inference.py \
    --engine block \
    --images exam_paper.png \
    --table-format html \
    --output-dir output/
```

### Ananta Stage 1 Only (Layout Bounding & Reading Order)

```bash
python run_inference.py \
    --engine block \
    --images document.png \
    --layout-only \
    --output-dir output/
```

### Ananta Stage 2 Only (Transcription from Precomputed Layout)

```bash
python run_inference.py \
    --engine block \
    --ocr-only output/document.layout.json \
    --output-dir output/
```

### Ananta Hybrid Mode

```bash
python run_inference.py \
    --engine hybrid \
    --images complex_page.png \
    --output-dir output/
```

### Contract & Schema Inspection

```bash
python run_inference.py show-contract
```

---

## Python API Reference

### Unified High-Level Pipeline

```python
from ananta_ocr import UnifiedOCR

vlm_pipeline = UnifiedOCR(engine="vlm", model_dir="./")
vlm_result = vlm_pipeline.process(["page1.png", "page2.png"])
print(vlm_result.markdown)
vlm_result.save("output/", base_name="vlm_document")

block_pipeline = UnifiedOCR(engine="block", table_format="html", device="cuda")
block_result = block_pipeline.process("indic_sample.png")
print(block_result.markdown)
block_result.save("output/", base_name="block_document")
```

### Direct Subsystem Access

```python
from ananta_ocr import AnantaDocLayout, AnantaBlockOCR

layout_detector = AnantaDocLayout()
page_layout = layout_detector.detect("page.png")

for block in page_layout.blocks:
    print(f"Rank {block.order}: [{block.label}] bbox={block.bbox_xyxy} conf={block.conf}")

recognizer = AnantaBlockOCR()
transcription = recognizer.run("page.png", page_layout)
print(transcription.markdown)
```

### High-Throughput Remote Serving

```python
from ananta_ocr import AnantaClient

client = AnantaClient(base_url="http://localhost:8000/v1")
result = client.parse("sample_newspaper.png")
print(result.markdown)
```

---

## Verification Suite

ANANTA OCR includes an automated unit test suite verifying geometric filtering, taxonomy parsing, prompt generation, math sanitization, and pipeline initialization:

```bash
python -m unittest discover -s tests -v
```

All 15 automated test cases validate deterministic behavior across all supported subsystems.

---

## Repository Structure

```text
ANANTA_OCR/
|-- assets/
|   `-- architecture.html
|-- wheel/
|   `-- sglang-*.whl
|-- utils/
|   `-- process_anantaocr.py
|-- ananta_engine/
|   |-- engine/
|   |   |-- types.py
|   |   |-- blocks.py
|   |   |-- crops.py
|   |   |-- reconstruct.py
|   |   |-- layout.py
|   |   |-- recognizer.py
|   |   |-- recognizer_vllm.py
|   |   |-- checkpoints.py
|   |   `-- offline.py
|   |-- layout/
|   |   |-- labels.py
|   |   |-- order_loss.py
|   |   |-- modeling_ppdoc.py
|   |   `-- infer.py
|   |-- templates/
|   |   `-- contract.py
|   |-- serving/
|   |   |-- client.py
|   |   `-- recognizer_http.py
|   `-- inference/
|       |-- common.py
|       `-- cli.py
|-- tests/
|   `-- test_ananta_ocr.py
|-- unified_pipeline.py
|-- run_inference.py
|-- download_weights.py
|-- dash_attention.py
|-- deepencoder.py
|-- modeling_anantaocr.py
|-- modeling_deepseekv2.py
|-- sparse_gather_e2e_triton.py
|-- sparse_gather_triton.py
|-- config.json
|-- tokenizer.json
|-- requirements.txt
|-- NOTICES.md
\-- README.md
```

---

## License and Attribution

ANANTA OCR is distributed under open-source licenses:

- **ANANTA OCR Core Engine & Triton Kernels**: Released under the **MIT License**. Copyright (c) 2026 AnantaOCR Contributors. Portions Copyright (c) 2026 Baidu.
- **ANANTA OCR Block & Layout Components**: Derived from foundational open-source research from Bodhan AI, licensed under the **Apache License 2.0**. Copyright (c) 2026 Contributors.
- **Third-Party Notices & Component Provenance**: See [NOTICES.md](NOTICES.md) for complete file-level attribution and upstream checkpoint licenses.

```text
MIT License

Copyright (c) 2026 AnantaOCR Contributors
Portions Copyright (c) 2026 Baidu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

```text
Apache License, Version 2.0

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
