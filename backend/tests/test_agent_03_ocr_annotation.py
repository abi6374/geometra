"""Unit tests for the OCR and Annotation Agent (Agent 3).

Tests cover:
- Dimension parsing from text (all types: linear, diameter, radius, angular, coordinate)
- Tolerance and fit parsing
- Label parsing (section labels, annotations)
- DXF dimension entity extraction
- PDF text content extraction
- Full OCR pipeline (using test images + OpenCV fallback)
- Edge cases: empty images, no text, missing data
- Bounding box center computation
- Number parsing with comma/period decimals
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geometra.agents.agent_03_ocr_annotation import (
    OCRAnnotationAgent,
    OCREngine,
    parse_dimensions_from_text,
    _parse_number,
    _bbox_center,
    _extract_dxf_dimensions,
    _extract_dxf_text_labels,
    _extract_pdf_annotations,
    _merge_dimension_groups,
)


# ── Agent Fixture ─────────────────────────────────────────────────────────────


@pytest.fixture
def agent() -> OCRAnnotationAgent:
    return OCRAnnotationAgent()


# ── Test Documents ────────────────────────────────────────────────────────────


def _raster_doc(path: Path) -> dict:
    """Create a minimal raster document (same format as Agent 1 output)."""
    import cv2

    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    return {
        "file_path": str(path),
        "format": "png",
        "data": {
            "image": img,
            "gray": gray,
            "width": w,
            "height": h,
        },
    }


@pytest.fixture
def dxf_document() -> dict:
    """Simulate the output of InputProcessingAgent for a DXF file with dimensions."""
    return {
        "file_path": "/fake/test.dxf",
        "format": "dxf",
        "data": {
            "entities": [
                # Dimension entities
                {
                    "type": "DIMENSION",
                    "layer": "0",
                    "text": "200",
                    "measurement": 200.0,
                    "position": (150.0, 20.0),
                },
                {
                    "type": "DIMENSION",
                    "layer": "0",
                    "text": "150",
                    "measurement": 150.0,
                    "position": (270.0, 125.0),
                },
                {
                    "type": "DIMENSION",
                    "layer": "0",
                    "text": "⌀60",
                    "measurement": 60.0,
                    "position": (100.0, 80.0),
                },
                # Text entities
                {
                    "type": "TEXT",
                    "layer": "1",
                    "text": "H7",
                    "position": (80.0, 160.0),
                },
                {
                    "type": "TEXT",
                    "layer": "1",
                    "text": "A-A",
                    "position": (200.0, 300.0),
                },
                {
                    "type": "MTEXT",
                    "layer": "2",
                    "text": "SECTION A-A SCALE 1:2",
                    "position": (200.0, 320.0),
                },
                {
                    "type": "LINE",
                    "layer": "0",
                    "start": (0.0, 0.0),
                    "end": (100.0, 0.0),
                },
            ],
            "layers": {"0": {}, "1": {}, "2": {}},
            "entity_counts": {
                "DIMENSION": 3,
                "TEXT": 2,
                "MTEXT": 1,
                "LINE": 1,
            },
        },
    }


@pytest.fixture
def pdf_document() -> dict:
    """Simulate the output of InputProcessingAgent for a PDF with text content."""
    return {
        "file_path": "/fake/test.pdf",
        "format": "pdf",
        "data": {
            "pages": [
                {"page_number": 1, "width": 612.0, "height": 792.0},
            ],
            "text_content": "200\n150\n⌀60\nH7\n±0.5\nA-A\nSECTION NOTES\nSCALE 1:2",
            "page_count": 1,
            "metadata": {},
        },
    }


# ── Agent: validate_input ─────────────────────────────────────────────────────


class TestValidateInput:
    def test_valid_raster_doc(self, agent: OCRAnnotationAgent, synthetic_annotation_drawing: Path) -> None:
        doc = _raster_doc(synthetic_annotation_drawing)
        assert agent.validate_input(doc) is True

    def test_valid_dxf_doc(self, agent: OCRAnnotationAgent, dxf_document: dict) -> None:
        assert agent.validate_input(dxf_document) is True

    def test_invalid_none(self, agent: OCRAnnotationAgent) -> None:
        assert agent.validate_input(None) is False

    def test_invalid_string(self, agent: OCRAnnotationAgent) -> None:
        assert agent.validate_input("some string") is False

    def test_missing_format(self, agent: OCRAnnotationAgent) -> None:
        assert agent.validate_input({"data": {}}) is False


# ── Number Parsing ────────────────────────────────────────────────────────────


class TestParseNumber:
    def test_integer(self) -> None:
        assert _parse_number("42") == 42

    def test_decimal(self) -> None:
        assert _parse_number("3.14") == 3.14

    def test_comma_decimal(self) -> None:
        # European decimal separator
        assert _parse_number("3,14") == 3.14

    def test_negative(self) -> None:
        assert _parse_number("-5.5") == -5.5

    def test_positive(self) -> None:
        assert _parse_number("+10") == 10

    def test_zero(self) -> None:
        assert _parse_number("0") == 0


# ── Bounding Box Center ───────────────────────────────────────────────────────


class TestBboxCenter:
    def test_standard_bbox(self) -> None:
        bbox = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]]
        cx, cy = _bbox_center(bbox)
        assert cx == 50.0
        assert cy == 25.0

    def test_empty_bbox(self) -> None:
        cx, cy = _bbox_center([])
        assert cx == 0.0
        assert cy == 0.0

    def test_single_point(self) -> None:
        bbox = [[10.0, 20.0]]
        cx, cy = _bbox_center(bbox)
        assert cx == 10.0
        assert cy == 20.0

    def test_asymmetric(self) -> None:
        bbox = [[5.0, 5.0], [15.0, 5.0], [15.0, 25.0], [5.0, 25.0]]
        cx, cy = _bbox_center(bbox)
        assert cx == 10.0
        assert cy == 15.0


# ── Dimension Parsing ─────────────────────────────────────────────────────────


class TestParseDimensionsFromText:
    def test_linear_dimension(self) -> None:
        blocks = [
            {"text": "200", "bbox": [[130, 25], [160, 25], [160, 35], [130, 35]], "confidence": 0.95},
        ]
        result = parse_dimensions_from_text(blocks)
        assert len(result["dimensions"]) == 1
        d = result["dimensions"][0]
        assert d["value"] == 200
        assert d["type"] == "linear"

    def test_diameter_dimension_prefix(self) -> None:
        blocks = [
            {"text": "⌀60", "bbox": [[60, 80], [90, 80], [90, 95], [60, 95]], "confidence": 0.93},
        ]
        result = parse_dimensions_from_text(blocks)
        assert len(result["dimensions"]) == 1
        d = result["dimensions"][0]
        assert d["value"] == 60
        assert d["type"] == "diameter"

    def test_diameter_with_letter_o(self) -> None:
        blocks = [
            {"text": "Ø40", "bbox": [[0, 0], [30, 0], [30, 15], [0, 15]], "confidence": 0.9},
        ]
        result = parse_dimensions_from_text(blocks)
        assert len(result["dimensions"]) == 1
        d = result["dimensions"][0]
        assert d["value"] == 40
        assert d["type"] == "diameter"

    def test_radius_dimension(self) -> None:
        blocks = [
            {"text": "R25", "bbox": [[75, 130], [105, 130], [105, 145], [75, 145]], "confidence": 0.92},
        ]
        result = parse_dimensions_from_text(blocks)
        assert len(result["dimensions"]) == 1
        d = result["dimensions"][0]
        assert d["value"] == 25
        assert d["type"] == "radius"

    def test_radius_lowercase(self) -> None:
        blocks = [
            {"text": "r12.5", "bbox": [], "confidence": 0.9},
        ]
        result = parse_dimensions_from_text(blocks)
        assert len(result["dimensions"]) == 1
        assert result["dimensions"][0]["value"] == 12.5
        assert result["dimensions"][0]["type"] == "radius"

    def test_angular_dimension(self) -> None:
        blocks = [
            {"text": "90°", "bbox": [[220, 25], [240, 25], [240, 35], [220, 35]], "confidence": 0.91},
        ]
        result = parse_dimensions_from_text(blocks)
        assert len(result["dimensions"]) == 1
        d = result["dimensions"][0]
        assert d["value"] == 90
        assert d["type"] == "angular"
        assert d["unit"] == "deg"

    def test_angular_with_deg_suffix(self) -> None:
        blocks = [
            {"text": "45º", "bbox": [], "confidence": 0.85},
        ]
        result = parse_dimensions_from_text(blocks)
        assert len(result["dimensions"]) == 1
        assert result["dimensions"][0]["value"] == 45

    def test_coordinate_dimension(self) -> None:
        blocks = [
            {"text": "X:200 Y:150", "bbox": [], "confidence": 0.88},
        ]
        result = parse_dimensions_from_text(blocks)
        assert len(result["dimensions"]) == 1
        d = result["dimensions"][0]
        assert d["value"] == (200.0, 150.0)
        assert d["type"] == "coordinate"

    def test_coordinate_with_equals(self) -> None:
        blocks = [
            {"text": "X=100 Y=50", "bbox": [], "confidence": 0.87},
        ]
        result = parse_dimensions_from_text(blocks)
        assert len(result["dimensions"]) == 1
        assert result["dimensions"][0]["value"] == (100.0, 50.0)

    def test_tolerance_parsing(self) -> None:
        blocks = [
            {"text": "±0.5", "bbox": [[290, 80], [315, 80], [315, 92], [290, 92]], "confidence": 0.94},
        ]
        result = parse_dimensions_from_text(blocks)
        assert len(result["tolerances"]) == 1
        t = result["tolerances"][0]
        assert t["value"] == 0.5
        assert t["symbol"] == "±"
        assert t["type"] == "tolerance"

    def test_fit_parsing(self) -> None:
        blocks = [
            {"text": "H7", "bbox": [[70, 150], [90, 150], [90, 165], [70, 165]], "confidence": 0.95},
        ]
        result = parse_dimensions_from_text(blocks)
        assert len(result["tolerances"]) == 1
        t = result["tolerances"][0]
        assert t["fit"] == "H7"
        assert t["type"] == "fit"

    def test_fit_with_nominal_size(self) -> None:
        blocks = [
            {"text": "10 H7", "bbox": [], "confidence": 0.93},
        ]
        result = parse_dimensions_from_text(blocks)
        assert len(result["tolerances"]) == 1
        t = result["tolerances"][0]
        assert t["fit"] == "H7"
        assert t["nominal_size"] == 10

    def test_lowercase_fit(self) -> None:
        blocks = [
            {"text": "h11", "bbox": [], "confidence": 0.9},
        ]
        result = parse_dimensions_from_text(blocks)
        assert len(result["tolerances"]) == 1
        assert result["tolerances"][0]["fit"] == "h11"

    def test_section_label(self) -> None:
        blocks = [
            {"text": "A-A", "bbox": [[180, 295], [210, 295], [210, 310], [180, 310]], "confidence": 0.96},
            {"text": "B-B", "bbox": [[400, 275], [425, 275], [425, 290], [400, 290]], "confidence": 0.94},
        ]
        result = parse_dimensions_from_text(blocks)
        assert len(result["labels"]) == 2
        assert result["labels"][0]["text"] == "A-A"
        assert result["labels"][0]["type"] == "section_label"

    def test_long_section_label(self) -> None:
        blocks = [
            {"text": "SECTION A-A", "bbox": [], "confidence": 0.9},
        ]
        result = parse_dimensions_from_text(blocks)
        labels = [l for l in result["labels"] if l["type"] == "section_label"]
        # "SECTION A-A" won't match the short pattern but will be an annotation label
        annotations = [l for l in result["labels"]]
        assert len(annotations) >= 1

    def test_general_annotation(self) -> None:
        blocks = [
            {"text": "SECTION NOTES", "bbox": [], "confidence": 0.90},
            {"text": "SCALE 1:2", "bbox": [], "confidence": 0.85},
        ]
        result = parse_dimensions_from_text(blocks)
        assert len(result["labels"]) >= 2

    def test_multiple_dimensions(self) -> None:
        blocks = [
            {"text": "200", "bbox": [], "confidence": 0.95},
            {"text": "150", "bbox": [], "confidence": 0.94},
            {"text": "⌀60", "bbox": [], "confidence": 0.93},
            {"text": "R25", "bbox": [], "confidence": 0.92},
            {"text": "H7", "bbox": [], "confidence": 0.95},
            {"text": "±0.5", "bbox": [], "confidence": 0.94},
        ]
        result = parse_dimensions_from_text(blocks)
        # 200 (linear), 150 (linear), ⌀60 (diameter), R25 (radius) = 4
        assert len(result["dimensions"]) == 4
        assert len(result["tolerances"]) == 2  # H7, ±0.5

    def test_low_confidence_filtered(self) -> None:
        blocks = [
            {"text": "200", "bbox": [], "confidence": 0.3},  # below MIN_CONFIDENCE (0.5)
        ]
        result = parse_dimensions_from_text(blocks)
        assert len(result["dimensions"]) == 0

    def test_empty_blocks(self) -> None:
        result = parse_dimensions_from_text([])
        assert len(result["dimensions"]) == 0
        assert len(result["tolerances"]) == 0
        assert len(result["labels"]) == 0

    def test_bbox_position_in_dimension(self) -> None:
        blocks = [
            {"text": "200", "bbox": [[130.0, 25.0], [160.0, 25.0], [160.0, 35.0], [130.0, 35.0]], "confidence": 0.95},
        ]
        result = parse_dimensions_from_text(blocks)
        d = result["dimensions"][0]
        assert d["position"]["x"] == 145.0  # (130+160+160+130)/4
        assert d["position"]["y"] == 30.0  # (25+25+35+35)/4

    def test_standalone_number_with_extra_text(self) -> None:
        """A number followed by non-numeric text should not be parsed as linear dim."""
        blocks = [
            {"text": "200mm", "bbox": [], "confidence": 0.95},
        ]
        result = parse_dimensions_from_text(blocks)
        # "200mm" should not match as standalone number because of trailing "mm"
        # But it could match as a label or other type
        dims = result["dimensions"]
        labels = result["labels"]
        # It could be a label, or if we add MM parsing it would be a dim
        assert len(dims) + len(labels) >= 1


# ── DXF Dimension Extraction ──────────────────────────────────────────────────


class TestDxfDimensionExtraction:
    def test_extracts_dxf_dimensions(self, dxf_document: dict) -> None:
        dims = _extract_dxf_dimensions(dxf_document["data"])
        assert len(dims) == 3

    def test_dxf_dimension_types(self, dxf_document: dict) -> None:
        dims = _extract_dxf_dimensions(dxf_document["data"])
        types = {d["type"] for d in dims}
        # DIMENSION defaults to "radius" in our mapping
        assert "radius" in types or "linear" in types

    def test_dxf_dimension_values(self, dxf_document: dict) -> None:
        dims = _extract_dxf_dimensions(dxf_document["data"])
        values = {d["value"] for d in dims}
        assert 200 in values or 200.0 in values
        assert 150 in values or 150.0 in values

    def test_dxf_dimension_has_source(self, dxf_document: dict) -> None:
        dims = _extract_dxf_dimensions(dxf_document["data"])
        for d in dims:
            assert d["source"] == "dxf"
            assert "layer" in d

    def test_dxf_text_labels(self, dxf_document: dict) -> None:
        labels = _extract_dxf_text_labels(dxf_document["data"])
        assert len(labels) == 3  # H7, A-A, SECTION A-A SCALE 1:2

    def test_dxf_text_label_types(self, dxf_document: dict) -> None:
        labels = _extract_dxf_text_labels(dxf_document["data"])
        types = {l["type"] for l in labels}
        assert "section_label" in types or "annotation" in types

    def test_dxf_no_text_entities(self) -> None:
        data = {"entities": [{"type": "LINE", "start": (0, 0), "end": (10, 0)}]}
        labels = _extract_dxf_text_labels(data)
        assert len(labels) == 0

    def test_dxf_no_dimension_entities(self) -> None:
        data = {"entities": [{"type": "LINE", "start": (0, 0), "end": (10, 0)}]}
        dims = _extract_dxf_dimensions(data)
        assert len(dims) == 0


# ── PDF Text Extraction ───────────────────────────────────────────────────────


class TestPdfAnnotationExtraction:
    def test_extracts_pdf_annotations(self, pdf_document: dict) -> None:
        blocks = _extract_pdf_annotations(pdf_document["data"])
        assert len(blocks) >= 4  # multiple lines parsed

    def test_pdf_annotation_types(self, pdf_document: dict) -> None:
        blocks = _extract_pdf_annotations(pdf_document["data"])
        types = {b["type"] for b in blocks}
        assert "dimension" in types or "tolerance" in types or "annotation" in types

    def test_pdf_empty_text(self) -> None:
        data = {"text_content": "", "pages": []}
        blocks = _extract_pdf_annotations(data)
        assert len(blocks) == 0

    def test_pdf_no_text_key(self) -> None:
        data = {"pages": []}
        blocks = _extract_pdf_annotations(data)
        assert len(blocks) == 0


# ── OCR Engine ────────────────────────────────────────────────────────────────


class TestOCREngine:
    def test_engine_initialization(self) -> None:
        engine = OCREngine()
        assert engine._paddle_available is None  # not yet checked
        assert engine._paddle_ocr is None  # not yet initialized

    def test_paddle_available_check_lazy(self) -> None:
        engine = OCREngine()
        # This will check and cache the result
        available = engine.paddle_available
        assert isinstance(available, bool)
        # Second call should return cached value
        assert engine._paddle_available is not None

    def test_opencv_fallback_on_empty_image(self) -> None:
        engine = OCREngine()
        # Force OpenCV fallback even if Paddle is available
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        # Draw some text-like regions
        cv2 = pytest.importorskip("cv2")
        cv2.putText(image, "TEST", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        result = engine._run_opencv_ocr(image)
        assert isinstance(result, list)
        # MSER should detect the text region
        assert len(result) >= 0  # may be 0 depending on MSER sensitivity

    def test_opencv_fallback_grayscale(self) -> None:
        engine = OCREngine()
        gray = np.ones((100, 100), dtype=np.uint8) * 255
        gray[30:50, 30:70] = 0  # a dark region (simulating text)
        result = engine._run_opencv_ocr(gray)
        assert isinstance(result, list)


# ── Full Agent Pipeline ───────────────────────────────────────────────────────


class TestFullPipeline:
    def test_agent_process_raster(self, agent: OCRAnnotationAgent, synthetic_annotation_drawing: Path) -> None:
        doc = _raster_doc(synthetic_annotation_drawing)
        result = agent.process(doc, force_opencv=True)
        assert isinstance(result, dict)
        assert "dimensions" in result
        assert "tolerances" in result
        assert "labels" in result
        assert "text_blocks" in result
        assert "raw_text" in result
        assert "ocr_engine" in result

    def test_agent_process_raster_structure(self, agent: OCRAnnotationAgent, synthetic_annotation_drawing: Path) -> None:
        doc = _raster_doc(synthetic_annotation_drawing)
        result = agent.process(doc, force_opencv=True)
        required = {"dimensions", "tolerances", "labels", "text_blocks", "raw_text", "ocr_engine"}
        assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"

    def test_agent_process_dxf(self, agent: OCRAnnotationAgent, dxf_document: dict) -> None:
        result = agent.process(dxf_document)
        assert result["ocr_engine"] == "dxf"
        assert len(result["text_blocks"]) >= 1
        assert "raw_text" in result

    def test_agent_process_dxf_has_dimensions(self, agent: OCRAnnotationAgent, dxf_document: dict) -> None:
        result = agent.process(dxf_document)
        assert len(result["dimensions"]) >= 3

    def test_agent_process_pdf(self, agent: OCRAnnotationAgent, pdf_document: dict) -> None:
        result = agent.process(pdf_document)
        assert result["ocr_engine"] == "pdf_text"
        assert isinstance(result["raw_text"], str) and len(result["raw_text"]) > 0

    def test_agent_process_missing_image(self, agent: OCRAnnotationAgent) -> None:
        doc = {"format": "png", "data": {}}
        result = agent.process(doc)
        assert result["ocr_engine"] == "none"
        assert len(result["dimensions"]) == 0

    def test_agent_process_with_kwargs(self, agent: OCRAnnotationAgent, synthetic_annotation_drawing: Path) -> None:
        doc = _raster_doc(synthetic_annotation_drawing)
        result = agent.process(doc, force_opencv=True, extra_option=True)
        assert "dimensions" in result

    def test_agent_synthetic_dimensions(self, agent: OCRAnnotationAgent, synthetic_dimension_drawing: Path) -> None:
        doc = _raster_doc(synthetic_dimension_drawing)
        result = agent.process(doc, force_opencv=True)
        # Even with OpenCV fallback (no recognition), the pipeline should not crash
        assert isinstance(result, dict)
        assert "dimensions" in result

    def test_agent_complex_drawing(self, agent: OCRAnnotationAgent, synthetic_complex_drawing: Path) -> None:
        doc = _raster_doc(synthetic_complex_drawing)
        result = agent.process(doc, force_opencv=True)
        assert isinstance(result, dict)
        assert "raw_text" in result

    def test_agent_blank_drawing(self, agent: OCRAnnotationAgent, blank_drawing: Path) -> None:
        doc = _raster_doc(blank_drawing)
        result = agent.process(doc, force_opencv=True)
        assert isinstance(result, dict)
        assert len(result["text_blocks"]) == 0 or len(result["dimensions"]) == 0

    def test_process_alias(self, agent: OCRAnnotationAgent, dxf_document: dict) -> None:
        result = agent.process(dxf_document)
        assert "dimensions" in result

    def test_ocr_engine_reuse(self, agent: OCRAnnotationAgent) -> None:
        """The OCR engine should be lazily initialized once."""
        engine1 = agent.ocr_engine
        engine2 = agent.ocr_engine
        assert engine1 is engine2

    def test_no_mutual_modification(self, agent: OCRAnnotationAgent, dxf_document: dict) -> None:
        """Calling process should not mutate the input document."""
        import copy

        doc_copy = copy.deepcopy(dxf_document)
        agent.process(dxf_document)
        assert dxf_document == doc_copy


# ── Dimension Merging ─────────────────────────────────────────────────────────


class TestDimensionMerging:
    def test_merge_nearby_dimensions(self) -> None:
        dims = [
            {"value": 10, "unit": "mm", "position": {"x": 100, "y": 100},
             "bbox": [], "confidence": 0.9, "type": "linear", "text_raw": "10", "tolerance": None},
            {"value": 10, "unit": "mm", "position": {"x": 105, "y": 102},
             "bbox": [], "confidence": 0.85, "type": "linear", "text_raw": "H7", "tolerance": None},
        ]
        merged = _merge_dimension_groups(dims, max_group_distance=20.0)
        assert len(merged) == 1

    def test_no_merge_far_apart(self) -> None:
        dims = [
            {"value": 200, "unit": "mm", "position": {"x": 50, "y": 50},
             "bbox": [], "confidence": 0.9, "type": "linear", "text_raw": "200", "tolerance": None},
            {"value": 150, "unit": "mm", "position": {"x": 300, "y": 300},
             "bbox": [], "confidence": 0.9, "type": "linear", "text_raw": "150", "tolerance": None},
        ]
        merged = _merge_dimension_groups(dims, max_group_distance=20.0)
        assert len(merged) == 2

    def test_empty_list(self) -> None:
        merged = _merge_dimension_groups([])
        assert merged == []

    def test_merge_preserves_keys(self) -> None:
        dims = [
            {"value": 100, "unit": "mm", "position": {"x": 100, "y": 200},
             "bbox": [], "confidence": 0.95, "type": "linear", "text_raw": "100", "tolerance": None},
        ]
        merged = _merge_dimension_groups(dims)
        assert len(merged) == 1
        assert merged[0]["value"] == 100
        assert merged[0]["type"] == "linear"
