"""Automated unit test suite for ANANTA OCR.

Run with:
    python -m unittest discover -s tests
"""

import sys
from pathlib import Path

# Ensure ananta_ocr and parent directory are on sys.path
TEST_DIR = Path(__file__).resolve().parent
REPO_DIR = TEST_DIR.parent
WORKSPACE_DIR = REPO_DIR.parent

for p in [str(REPO_DIR), str(WORKSPACE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import unittest

# Try importing from ananta_ocr or local directory
try:
    from ananta_ocr import (
        Block,
        PageResult,
        LayoutConfig,
        CropConfig,
        DedupConfig,
        RecognizerConfig,
        TableFormat,
        UnifiedOCR,
        CLASSES,
        NUM_CLASSES,
    )
    from ananta_ocr.ananta_engine.templates.contract import (
        prompt_for,
        map_label,
        is_transcribed,
        TEXT_PROMPT,
        EQUATION_PROMPT,
    )
    from ananta_ocr.ananta_engine.engine.blocks import (
        area,
        contained_frac,
        clamp_to_page,
        clean_layout,
        resolve_nested_equations,
    )
    from ananta_ocr.ananta_engine.engine.reconstruct import (
        dehyphenate,
        repair_math,
        reconstruct,
    )
    from ananta_ocr.ananta_engine.engine.recognizer import CORRECT_EOS, BROKEN_EOS
except ImportError:
    from ananta_engine import (
        Block,
        PageResult,
        LayoutConfig,
        CropConfig,
        DedupConfig,
        RecognizerConfig,
        TableFormat,
        CLASSES,
        NUM_CLASSES,
    )
    from unified_pipeline import UnifiedOCR
    from ananta_engine.templates.contract import (
        prompt_for,
        map_label,
        is_transcribed,
        TEXT_PROMPT,
        EQUATION_PROMPT,
    )
    from ananta_engine.engine.blocks import (
        area,
        contained_frac,
        clamp_to_page,
        clean_layout,
        resolve_nested_equations,
    )
    from ananta_engine.engine.reconstruct import (
        dehyphenate,
        repair_math,
        reconstruct,
    )
    from ananta_engine.engine.recognizer import CORRECT_EOS, BROKEN_EOS


class TestAnantaContract(unittest.TestCase):
    def test_taxonomy_classes(self):
        self.assertEqual(len(CLASSES), 37)
        self.assertEqual(NUM_CLASSES, 37)
        self.assertIn("Question", CLASSES)
        self.assertIn("Paragraph", CLASSES)
        self.assertIn("Equation", CLASSES)
        self.assertIn("Table", CLASSES)

    def test_prompt_routing(self):
        self.assertEqual(prompt_for("Text"), TEXT_PROMPT)
        self.assertEqual(prompt_for("Equation"), EQUATION_PROMPT)
        html_prompt = prompt_for("Table", TableFormat.HTML)
        self.assertIn("HTML", html_prompt)
        md_prompt = prompt_for("Table", TableFormat.MARKDOWN)
        self.assertIn("markdown", md_prompt)

    def test_label_mapping(self):
        self.assertEqual(map_label("question"), "Text")
        self.assertEqual(map_label("equation"), "Equation")
        self.assertEqual(map_label("table"), "Table")
        self.assertEqual(map_label("header"), "PageHeader")
        self.assertEqual(map_label("footer"), "PageFooter")

    def test_is_transcribed(self):
        self.assertTrue(is_transcribed("Paragraph"))
        self.assertTrue(is_transcribed("Equation"))
        self.assertFalse(is_transcribed("header"))
        self.assertFalse(is_transcribed("diagram"))
        self.assertFalse(is_transcribed("image"))


class TestAnantaBlocks(unittest.TestCase):
    def test_area(self):
        self.assertEqual(area([0, 0, 100, 200]), 20000.0)
        self.assertEqual(area([100, 200, 50, 100]), 0.0)

    def test_contained_frac(self):
        small = [10, 10, 20, 20]
        big = [0, 0, 50, 50]
        self.assertEqual(contained_frac(small, big), 1.0)
        outside = [60, 60, 70, 70]
        self.assertEqual(contained_frac(outside, big), 0.0)

    def test_clamp_to_page(self):
        clamped = clamp_to_page([-10, -5, 1050, 1100], 1000, 1000)
        self.assertEqual(clamped, [0.0, 0.0, 1000.0, 1000.0])

    def test_clean_layout(self):
        b1 = Block(order=0, label="Paragraph", type="Text", bbox_xyxy=[0, 0, 100, 100], conf=0.9)
        b2 = Block(order=1, label="Paragraph", type="Text", bbox_xyxy=[10, 10, 90, 90], conf=0.8)
        cleaned = clean_layout([b1, b2], DedupConfig(contain=0.9))
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0].order, 0)


class TestAnantaReconstruct(unittest.TestCase):
    def test_dehyphenate(self):
        text = "trans-\ncription of docu-\nment"
        self.assertEqual(dehyphenate(text), "transcription of document")

    def test_repair_math_indic(self):
        raw = "$$\u09ac\u09be\u0982\u09b2\u09be = 10$$"
        repaired = repair_math(raw)
        self.assertIn(r"\text{", repaired)

    def test_repair_math_newlines(self):
        raw = "$$x = 1\ny = 2$$"
        repaired = repair_math(raw)
        self.assertIn(r"\\", repaired)

    def test_reconstruct(self):
        b1 = Block(order=0, label="Paragraph", type="Text", bbox_xyxy=[0, 0, 100, 50], conf=0.9, text="Heading")
        b2 = Block(order=1, label="Paragraph", type="Text", bbox_xyxy=[0, 60, 100, 100], conf=0.9, text="Body text")
        md = reconstruct([b1, b2])
        self.assertEqual(md, "Heading\n\nBody text")


class TestAnantaPipeline(unittest.TestCase):
    def test_eos_token_constant(self):
        self.assertEqual(CORRECT_EOS, 262146)
        self.assertEqual(BROKEN_EOS, 248044)

    def test_unified_ocr_init(self):
        ocr_vlm = UnifiedOCR(engine="vlm")
        self.assertEqual(ocr_vlm.engine_type, "vlm")

        ocr_block = UnifiedOCR(engine="block")
        self.assertEqual(ocr_block.engine_type, "block")

        ocr_hybrid = UnifiedOCR(engine="hybrid")
        self.assertEqual(ocr_hybrid.engine_type, "hybrid")

        # Test aliases
        ocr_ananta = UnifiedOCR(engine="ananta")
        self.assertEqual(ocr_ananta.engine_type, "vlm")

        ocr_indic = UnifiedOCR(engine="indic")
        self.assertEqual(ocr_indic.engine_type, "block")

    def test_page_result_serialization(self):
        b = Block(order=0, label="Paragraph", type="Text", bbox_xyxy=[10.12, 20.34, 100.56, 200.78], conf=0.954, text="Sample")
        page = PageResult(image="test.png", width=1000, height=1000, blocks=[b], markdown="Sample")
        record = page.as_record()
        self.assertEqual(record["image"], "test.png")
        self.assertEqual(len(record["blocks"]), 1)
        self.assertEqual(record["blocks"][0]["bbox_xyxy"], [10.1, 20.3, 100.6, 200.8])

        reconstructed_page = PageResult.from_record(record)
        self.assertEqual(reconstructed_page.image, "test.png")
        self.assertEqual(len(reconstructed_page.blocks), 1)
        self.assertEqual(reconstructed_page.blocks[0].text, "Sample")

    def test_hybrid_routing_classification(self):
        try:
            from ananta_ocr import classify_hybrid_block
        except ImportError:
            from unified_pipeline import classify_hybrid_block
        self.assertEqual(classify_hybrid_block("Table"), "block")
        self.assertEqual(classify_hybrid_block("Equation"), "block")
        self.assertEqual(classify_hybrid_block("MCQ"), "block")
        self.assertEqual(classify_hybrid_block("Code"), "block")
        self.assertEqual(classify_hybrid_block("Paragraph"), "vlm")
        self.assertEqual(classify_hybrid_block("Title"), "vlm")

    def test_image_safety_limits(self):
        try:
            from ananta_ocr.ananta_engine.engine.offline import _open
        except ImportError:
            from ananta_engine.engine.offline import _open
        import tempfile
        from PIL import Image

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            img = Image.new("RGB", (200, 200), color="white")
            img.save(temp_path)

        with self.assertRaises(ValueError):
            _open(temp_path, max_pixels=10000)

        safe_img = _open(temp_path, max_pixels=50000)
        self.assertEqual(safe_img.size, (200, 200))
        Path(temp_path).unlink(missing_ok=True)

    def test_page_result_flexible_schema(self):
        p1 = PageResult(image="doc1.png", width=800, height=600)
        self.assertEqual(p1.image, "doc1.png")
        self.assertEqual(p1.image_path, "doc1.png")
        self.assertEqual(p1.image_size, (800, 600))

        p2 = PageResult(image_path="doc2.png", image_size=(1200, 900))
        self.assertEqual(p2.image, "doc2.png")
        self.assertEqual(p2.width, 1200)
        self.assertEqual(p2.height, 900)
        self.assertEqual(p2.image_path, "doc2.png")
        self.assertEqual(p2.image_size, (1200, 900))

    def test_block_auto_type_resolution(self):
        b = Block(order=1, label="Equation", bbox_xyxy=[0, 0, 50, 50], conf=0.92)
        self.assertEqual(b.type, "Equation")
        b2 = Block(order=2, label="Paragraph", bbox_xyxy=[0, 60, 50, 100], conf=0.88)
        self.assertEqual(b2.type, "Text")

    def test_hybrid_region_synthesis_no_duplication(self):
        try:
            from ananta_ocr import RegionResult
        except ImportError:
            from unified_pipeline import RegionResult

        r1 = RegionResult(order=0, label="Title", type="Text", bbox_xyxy=[0, 0, 100, 30], engine="vlm", text="Document Title")
        r2 = RegionResult(order=1, label="Paragraph", type="Text", bbox_xyxy=[0, 40, 100, 80], engine="vlm", text="First paragraph text.")
        r3 = RegionResult(order=2, label="Equation", type="Equation", bbox_xyxy=[0, 90, 100, 120], engine="block", text="$$E = mc^2$$")
        r4 = RegionResult(order=3, label="Paragraph", type="Text", bbox_xyxy=[0, 130, 100, 170], engine="vlm", text="Second paragraph text.")

        blocks = [r.to_block() for r in [r1, r2, r3, r4]]
        self.assertEqual(blocks[0].text, "Document Title")
        self.assertEqual(blocks[1].text, "First paragraph text.")
        self.assertEqual(blocks[2].text, "$$E = mc^2$$")
        self.assertEqual(blocks[3].text, "Second paragraph text.")

        md = reconstruct(blocks)
        self.assertIn("Document Title", md)
        self.assertIn("First paragraph text.", md)
        self.assertIn("$$E = mc^2$$", md)
        self.assertIn("Second paragraph text.", md)
        self.assertEqual(md.count("Document Title"), 1)
        self.assertEqual(md.count("First paragraph text."), 1)

    def test_meetei_mayek_and_extended_indic_math_repair(self):
        raw_math = "$x = ꯄ + y$"
        repaired = repair_math(raw_math)
        self.assertIn(r"\text{ꯄ}", repaired)

        raw_hindi = "$$बल = द्रव्यमान \\times त्वरण$$"
        repaired_hindi = repair_math(raw_hindi)
        self.assertIn(r"\text{बल}", repaired_hindi)
        self.assertIn(r"\text{द्रव्यमान}", repaired_hindi)


if __name__ == "__main__":
    unittest.main()
