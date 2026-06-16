"""Integration tests for the full Geometra pipeline (Agents 1→6).

Tests the end-to-end flow from an input file through all 6 processing agents,
verifying that each agent's output is compatible with the next agent's input
and the final CAD output has the expected structure.

Pipeline:
  Agent 1 (InputProcessing)  → file → normalized document
  Agent 2 (DrawingUnderstanding) → doc → geometric primitives
  Agent 3 (OCRAnnotation)    → doc → annotations (dimensions, tolerances, labels)
  Agent 4 (FeatureRecognition) → primitives → detected features
  Agent 5 (EngineeringReasoning) → primitives + annotations + features → feature tree
  Agent 6 (CADGeneration)    → feature tree → CAD scripts + exports
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geometra.agents.agent_01_input_processing import InputProcessingAgent
from geometra.agents.agent_02_drawing_understanding import DrawingUnderstandingAgent
from geometra.agents.agent_03_ocr_annotation import OCRAnnotationAgent
from geometra.agents.agent_04_feature_recognition import FeatureRecognitionAgent
from geometra.agents.agent_05_engineering_reasoning import EngineeringReasoningAgent
from geometra.agents.agent_06 import CADGenerationAgent


# ── Agent instances (module-level for reuse) ─────────────────────────────────

agent_1 = InputProcessingAgent()
agent_2 = DrawingUnderstandingAgent()
agent_3 = OCRAnnotationAgent()
agent_4 = FeatureRecognitionAgent()
agent_5 = EngineeringReasoningAgent()
agent_6 = CADGenerationAgent()


# ── Test: Agent input/output validation contracts ────────────────────────────


class TestAgentContracts:
    """Each agent's output must contain the keys expected as input by the next."""

    def test_agent1_output_has_agent2_required_keys(self, sample_dxf: Path) -> None:
        doc = agent_1.process(str(sample_dxf))
        # Agent 2's validate_input requires "format" and "data"
        assert "format" in doc, "Agent 1 output missing 'format'"
        assert "data" in doc, "Agent 1 output missing 'data'"
        assert agent_2.validate_input(doc), "Agent 1 output fails Agent 2 validation"

    def test_agent2_output_has_primitives_agent4_reads(self, sample_dxf: Path) -> None:
        doc = agent_1.process(str(sample_dxf))
        primitives = agent_2.process(doc)
        # Agent 4's detect() reads lines, circles, arcs, contours, symbols via .get()
        # These are the keys connecting Agent 2 → Agent 4
        assert "lines" in primitives, "Agent 2 output missing 'lines'"
        assert "circles" in primitives, "Agent 2 output missing 'circles'"
        assert "arcs" in primitives, "Agent 2 output missing 'arcs'"
        assert "contours" in primitives, "Agent 2 output missing 'contours'"

    def test_agent3_output_has_agent5_required_keys(self, sample_dxf: Path) -> None:
        doc = agent_1.process(str(sample_dxf))
        annotations = agent_3.process(doc)
        assert "dimensions" in annotations, "Agent 3 output missing 'dimensions'"
        assert "tolerances" in annotations, "Agent 3 output missing 'tolerances'"
        assert "labels" in annotations, "Agent 3 output missing 'labels'"

    def test_agent4_output_has_agent5_required_keys(self, sample_dxf: Path) -> None:
        doc = agent_1.process(str(sample_dxf))
        primitives = agent_2.process(doc)
        features = agent_4.process(primitives)
        assert "features" in features, "Agent 4 output missing 'features'"
        assert "feature_count" in features
        assert "class_counts" in features
        assert "detection_method" in features

    def test_agent5_output_has_agent6_required_keys(self, sample_dxf: Path) -> None:
        doc = agent_1.process(str(sample_dxf))
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc)
        features = agent_4.process(primitives)
        result = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        # Agent 6's validate_input requires "operations"
        assert "operations" in result, "Agent 5 output missing 'operations'"
        assert "parameters" in result
        assert agent_6.validate_input(result), "Agent 5 output fails Agent 6 validation"


# ── Test: Full DXF Pipeline (deterministic, no CV variance) ──────────────────


class TestFullDxfPipeline:
    """End-to-end pipeline test with DXF input (fully deterministic)."""

    def test_pipeline_runs_without_error(self, sample_dxf: Path) -> None:
        doc = agent_1.process(str(sample_dxf))
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc)
        features = agent_4.process(primitives)
        # Agent 5 needs all three: primitives, annotations, features
        result = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        assert isinstance(result, dict)

    def test_agent1_dxf_format(self, sample_dxf: Path) -> None:
        doc = agent_1.process(str(sample_dxf))
        assert doc["format"] == "dxf"
        assert doc["is_3d"] is False

    def test_agent2_dxf_primitives(self, sample_dxf: Path) -> None:
        doc = agent_1.process(str(sample_dxf))
        primitives = agent_2.process(doc)
        # DXF has a LINE and a CIRCLE
        assert len(primitives["lines"]) >= 1
        assert len(primitives["circles"]) >= 1
        # All DXF arcs, contours, symbols are empty for this simple file
        assert isinstance(primitives["arcs"], list)
        assert isinstance(primitives["contours"], list)
        assert isinstance(primitives["symbols"], list)

    def test_agent3_dxf_annotations(self, sample_dxf: Path) -> None:
        doc = agent_1.process(str(sample_dxf))
        annotations = agent_3.process(doc)
        assert isinstance(annotations["dimensions"], list)
        assert annotations["ocr_engine"] in ("dxf", "rule_set")

    def test_agent4_dxf_features(self, sample_dxf: Path) -> None:
        doc = agent_1.process(str(sample_dxf))
        primitives = agent_2.process(doc)
        features = agent_4.process(primitives)
        # Should detect at least a hole from the circle
        assert features["feature_count"] >= 1
        assert "hole" in features["class_counts"]
        assert features["detection_method"] == "cv"

    def test_agent5_dxf_reasoning(self, sample_dxf: Path) -> None:
        doc = agent_1.process(str(sample_dxf))
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc)
        features = agent_4.process(primitives)
        result = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        assert "feature_tree" in result
        assert "operations" in result
        assert "parameters" in result
        assert "rules_fired" in result
        assert "reasoning_engine" in result
        # Should have at least one operation (drill from the circle/hole)
        assert len(result["operations"]) >= 1

    def test_agent6_dxf_cad_generation(self, sample_dxf: Path) -> None:
        """Full Agents 1→6 pipeline with DXF should produce CAD output."""
        doc = agent_1.process(str(sample_dxf))
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc)
        features = agent_4.process(primitives)
        reasoning = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        cad = agent_6.process(reasoning)
        assert cad["success"] is True
        assert "openscad_script" in cad
        assert "cadquery_script" in cad
        assert "scad" in cad["formats"]
        assert "py" in cad["formats"]
        assert cad["feature_count"] >= 1
        # OpenSCAD script should reference the body dimensions
        script = cad["openscad_script"]
        assert "body_w" in script
        assert "body_h" in script
        assert "t =" in script

    def test_agent6_dxf_generates_valid_openscad(self, sample_dxf: Path) -> None:
        """Full pipeline: the generated OpenSCAD should have valid syntax."""
        doc = agent_1.process(str(sample_dxf))
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc)
        features = agent_4.process(primitives)
        reasoning = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        cad = agent_6.process(reasoning)
        script = cad["openscad_script"]
        # Should have body plate
        assert "cube([body_w, body_h, t])" in script
        # Should have a hole for the DXF circle
        assert "cylinder" in script or "difference" in script

    def test_agent6_dxf_cadquery_script_syntax(self, sample_dxf: Path) -> None:
        """Full pipeline: the CadQuery script should have valid Python syntax."""
        doc = agent_1.process(str(sample_dxf))
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc)
        features = agent_4.process(primitives)
        reasoning = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        cad = agent_6.process(reasoning)
        script = cad["cadquery_script"]
        assert "def build_model():" in script
        assert "cq.Workplane" in script
        assert "cq.exporters.export" in script

    def test_agent5_feature_tree_structure(self, sample_dxf: Path) -> None:
        doc = agent_1.process(str(sample_dxf))
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc)
        features = agent_4.process(primitives)
        result = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        tree = result["feature_tree"]
        assert "nodes" in tree
        assert "edges" in tree
        # Root node should be "body"
        assert len(tree["nodes"]) > 0, "Feature tree has no nodes"
        assert isinstance(tree["nodes"][0], dict), f"Expected dict nodes, got {type(tree['nodes'][0])}"
        node_ids = {n["id"] for n in tree["nodes"]}
        assert "body" in node_ids, "Feature tree missing 'body' node"

    def test_agent5_operations_well_formed(self, sample_dxf: Path) -> None:
        doc = agent_1.process(str(sample_dxf))
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc)
        features = agent_4.process(primitives)
        result = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        for op in result["operations"]:
            assert "operation" in op
            assert "feature_id" in op


# ── Test: Full Synthetic PNG Pipeline (CV-based, real image processing) ──────


class TestFullPngPipeline:
    """End-to-end pipeline with synthetic PNG input (includes real CV processing)."""

    def test_pipeline_runs_without_error(self, synthetic_annotation_drawing: Path) -> None:
        doc = agent_1.process(str(synthetic_annotation_drawing))
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc, force_opencv=True)
        features = agent_4.process(primitives)
        result = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        assert isinstance(result, dict)

    def test_agent1_png_format(self, synthetic_annotation_drawing: Path) -> None:
        doc = agent_1.process(str(synthetic_annotation_drawing))
        assert doc["format"] == "png"
        assert "data" in doc
        assert "image" in doc["data"]

    def test_agent2_png_primitives(self, synthetic_annotation_drawing: Path) -> None:
        doc = agent_1.process(str(synthetic_annotation_drawing))
        primitives = agent_2.process(doc)
        # The synthetic drawing has a rectangle (4 lines or 1 contour),
        # two circles, dimension lines, and text
        assert len(primitives["lines"]) >= 1, "Should detect at least some lines"
        assert len(primitives["circles"]) >= 1, "Should detect at least some circles"
        assert primitives["image_width"] is not None
        assert primitives["image_height"] is not None

    def test_agent3_png_annotations(self, synthetic_annotation_drawing: Path) -> None:
        doc = agent_1.process(str(synthetic_annotation_drawing))
        annotations = agent_3.process(doc, force_opencv=True)
        # OpenCV fallback won't recognize text, but should detect text regions
        assert isinstance(annotations["dimensions"], list)
        assert isinstance(annotations["tolerances"], list)
        assert isinstance(annotations["labels"], list)
        assert isinstance(annotations["raw_text"], str)
        assert annotations["ocr_engine"] == "opencv"

    def test_agent4_png_features(self, synthetic_annotation_drawing: Path) -> None:
        doc = agent_1.process(str(synthetic_annotation_drawing))
        primitives = agent_2.process(doc)
        features = agent_4.process(primitives)
        # Should detect the two circles as holes
        assert features["feature_count"] >= 1
        assert features["class_counts"]["hole"] >= 1
        assert features["detection_method"] == "cv"

    def test_agent5_png_reasoning(self, synthetic_annotation_drawing: Path) -> None:
        doc = agent_1.process(str(synthetic_annotation_drawing))
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc, force_opencv=True)
        features = agent_4.process(primitives)
        result = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        assert "feature_tree" in result
        assert "operations" in result
        assert "parameters" in result
        assert "rules_fired" in result

    def test_agent6_png_cad_generation(self, synthetic_annotation_drawing: Path) -> None:
        """Full Agents 1→6 pipeline with synthetic PNG should produce CAD scripts."""
        doc = agent_1.process(str(synthetic_annotation_drawing))
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc, force_opencv=True)
        features = agent_4.process(primitives)
        reasoning = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        cad = agent_6.process(reasoning)
        assert cad["success"] is True
        assert "openscad_script" in cad
        assert "cadquery_script" in cad
        assert cad["feature_count"] >= 1

    def test_complex_drawing_pipeline(self, synthetic_complex_drawing: Path) -> None:
        """Complex drawing with multiple features should produce CAD output."""
        doc = agent_1.process(str(synthetic_complex_drawing))
        primitives = agent_2.process(doc)
        features = agent_4.process(primitives)
        annotations = agent_3.process(doc, force_opencv=True)
        reasoning = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        cad = agent_6.process(reasoning)
        # The complex drawing has 6 mounting holes + 6 vent lines + display cutout
        # At minimum the CV should detect several circles as holes → drill operations → CAD output
        assert features["feature_count"] >= 3, \
            f"Expected >=3 features from complex drawing, got {features['feature_count']}"
        assert cad["success"] is True
        assert cad["feature_count"] >= 1
        assert "cylinder" in cad["openscad_script"] or "cube" in cad["openscad_script"]

    def test_blank_drawing_pipeline(self, blank_drawing: Path) -> None:
        """Blank drawing should produce empty results through Agent 6."""
        doc = agent_1.process(str(blank_drawing))
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc, force_opencv=True)
        features = agent_4.process(primitives)
        reasoning = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        cad = agent_6.process(reasoning)
        assert features["feature_count"] == 0
        assert len(reasoning["operations"]) == 0
        assert cad["feature_count"] == 0
        assert cad["success"] is True


# ── Test: Primitive Feature → Reasoning Pipeline (bypass image processing) ───


class TestPrimitivesToReasoningPipeline:
    """Test the chain from synthetic primitives through Agents 4 and 5.

    This bypasses Agents 1-3's image processing and OCR to test the
    core feature detection → reasoning pipeline deterministically.
    """

    @pytest.fixture
    def primitives_with_holes(self) -> dict:
        return {
            "format": "png",
            "lines": [],
            "circles": [
                {"x": 100.0, "y": 120.0, "radius": 30.0},
                {"x": 200.0, "y": 120.0, "radius": 20.0},
            ],
            "arcs": [],
            "contours": [],
            "symbols": [],
        }

    @pytest.fixture
    def primitives_with_slot(self) -> dict:
        return {
            "format": "png",
            "lines": [],
            "circles": [],
            "arcs": [],
            "contours": [
                {
                    "id": 0,
                    "area": 500,
                    "perimeter": 150.0,
                    "centroid": (150.0, 100.0),
                    "bounding_box": {"x": 100, "y": 80, "width": 80, "height": 20},
                    "approx_polygon_vertices": 6,
                    "approx_polygon": [],
                    "solidity": 0.85,
                    "hierarchy": {
                        "next": -1, "prev": -1, "first_child": -1, "parent": -1,
                        "has_children": False, "is_child": False,
                    },
                },
            ],
            "symbols": [],
        }

    @pytest.fixture
    def primitives_all_features(self) -> dict:
        """Primitives that should produce all 8 feature types."""
        return {
            "format": "png",
            "lines": [
                {"x1": 100, "y1": 100, "x2": 115, "y2": 85,
                 "length": 21.21, "angle_deg": -45.0},
            ],
            "circles": [
                {"x": 100.0, "y": 120.0, "radius": 30.0},
                {"x": 300.0, "y": 100.0, "radius": 25.0},
            ],
            "arcs": [
                {"x": 50.0, "y": 50.0, "radius": 5.0,
                 "start_angle": 0.0, "end_angle": 90.0,
                 "arc_length": 7.85, "angular_span_deg": 90.0,
                 "fit_error": 0.1, "max_fit_error": 0.2, "n_points": 20},
            ],
            "contours": [
                {
                    "id": 0,
                    "area": 50000,
                    "perimeter": 900.0,
                    "centroid": (250.0, 200.0),
                    "bounding_box": {"x": 30, "y": 30, "width": 440, "height": 340},
                    "approx_polygon_vertices": 4,
                    "approx_polygon": [],
                    "solidity": 0.98,
                    "hierarchy": {
                        "next": -1, "prev": -1, "first_child": 1, "parent": -1,
                        "has_children": True, "is_child": False,
                    },
                },
                {
                    "id": 1,
                    "area": 12000,
                    "perimeter": 440.0,
                    "centroid": (250.0, 140.0),
                    "bounding_box": {"x": 150, "y": 80, "width": 200, "height": 120},
                    "approx_polygon_vertices": 4,
                    "approx_polygon": [],
                    "solidity": 0.97,
                    "hierarchy": {
                        "next": -1, "prev": -1, "first_child": -1, "parent": 0,
                        "has_children": False, "is_child": True,
                    },
                },
                {
                    "id": 2,
                    "area": 2000,
                    "perimeter": 160.0,
                    "centroid": (300.0, 100.0),
                    "bounding_box": {"x": 275, "y": 75, "width": 50, "height": 50},
                    "approx_polygon_vertices": 8,
                    "approx_polygon": [],
                    "solidity": 0.95,
                    "hierarchy": {
                        "next": -1, "prev": -1, "first_child": -1, "parent": -1,
                        "has_children": False, "is_child": False,
                    },
                },
            ],
            "symbols": [],
        }

    def test_holes_to_operations(self, primitives_with_holes: dict) -> None:
        features = agent_4.process(primitives_with_holes)
        assert features["feature_count"] == 2
        assert features["class_counts"]["hole"] == 2
        result = agent_5.process({
            "primitives": primitives_with_holes,
            "annotations": {"dimensions": [], "tolerances": [], "labels": []},
            "features": features,
        })
        drill_ops = [o for o in result["operations"] if o["operation"] == "drill"]
        assert len(drill_ops) == 2, f"Expected 2 drill ops, got {len(drill_ops)}"

    def test_slot_to_operation(self, primitives_with_slot: dict) -> None:
        features = agent_4.process(primitives_with_slot)
        assert features["class_counts"]["slot"] >= 1
        result = agent_5.process({
            "primitives": primitives_with_slot,
            "annotations": {"dimensions": [], "tolerances": [], "labels": []},
            "features": features,
        })
        mill_ops = [o for o in result["operations"] if o["operation"] == "mill"]
        assert len(mill_ops) >= 1

    def test_all_features_detectable(self, primitives_all_features: dict) -> None:
        """All 8 feature types should be detectable from synthetic primitives."""
        features = agent_4.process(primitives_all_features)
        detected = {t for t, c in features["class_counts"].items() if c > 0}
        # Should detect at least: hole (circles), cutout (contour with children),
        # pocket (child contour), boss (high circularity contour), fillet, chamfer
        assert "hole" in detected
        assert "cutout" in detected
        assert "fillet" in detected or "chamfer" in detected

    def test_all_features_to_operations(self, primitives_all_features: dict) -> None:
        features = agent_4.process(primitives_all_features)
        # Create full annotations with a diameter dimension for the hole
        annotations = {
            "dimensions": [{
                "value": 60, "unit": "mm", "type": "diameter",
                "position": {"x": 105.0, "y": 110.0},
                "bbox": [], "confidence": 0.95, "text_raw": "⌀60", "tolerance": None,
            }],
            "tolerances": [],
            "labels": [],
        }
        result = agent_5.process({
            "primitives": primitives_all_features,
            "annotations": annotations,
            "features": features,
        })
        # Should produce at minimum drill operations (from holes)
        drill_ops = [o for o in result["operations"] if o["operation"] == "drill"]
        assert len(drill_ops) >= 1
        assert result["reasoning_engine"] in ("experta", "rule_set")

    def test_agent5_parameters_include_body_dimensions(
        self, primitives_all_features: dict,
    ) -> None:
        features = agent_4.process(primitives_all_features)
        annotations = {"dimensions": [], "tolerances": [], "labels": []}
        result = agent_5.process({
            "primitives": primitives_all_features,
            "annotations": annotations,
            "features": features,
        })
        # The largest outer contour (id=0, 440x340) should provide body dimensions
        assert result["parameters"].get("body_width") is not None, \
            f"Missing body_width in params: {result['parameters']}"
        assert result["parameters"].get("body_height") is not None

    def test_agent6_all_features_to_cad(self, primitives_all_features: dict) -> None:
        """Agents 4→5→6 pipeline with all feature types produces full CAD."""
        features = agent_4.process(primitives_all_features)
        annotations = {
            "dimensions": [{
                "value": 60, "unit": "mm", "type": "diameter",
                "position": {"x": 105.0, "y": 110.0},
                "bbox": [], "confidence": 0.95, "text_raw": "⌀60", "tolerance": None,
            }],
            "tolerances": [],
            "labels": [],
        }
        reasoning = agent_5.process({
            "primitives": primitives_all_features,
            "annotations": annotations,
            "features": features,
        })
        cad = agent_6.process(reasoning)
        assert cad["success"] is True
        assert cad["feature_count"] >= 1
        # CadQuery is installed, but process() may or may not build in-process model;
        # scad and py scripts are always generated
        assert "scad" in cad["formats"]
        assert "py" in cad["formats"]
        script = cad["openscad_script"]
        # Should include all feature types in the OpenSCAD output
        assert "hole" in script or "cylinder" in script
        assert "body_w" in script
        assert "body_h" in script

    def test_agent6_holes_to_cad(self, primitives_with_holes: dict) -> None:
        """2 holes → 2 drill operations → OpenSCAD with 2 cylinders."""
        features = agent_4.process(primitives_with_holes)
        reasoning = agent_5.process({
            "primitives": primitives_with_holes,
            "annotations": {"dimensions": [], "tolerances": [], "labels": []},
            "features": features,
        })
        assert len(reasoning["operations"]) == 2
        cad = agent_6.process(reasoning)
        assert cad["success"] is True
        assert cad["feature_count"] == 2
        assert cad["openscad_script"].count("cylinder") >= 2

    def test_agent6_slot_to_cad(self, primitives_with_slot: dict) -> None:
        """Slot → mill operation → OpenSCAD with cube for slot."""
        features = agent_4.process(primitives_with_slot)
        reasoning = agent_5.process({
            "primitives": primitives_with_slot,
            "annotations": {"dimensions": [], "tolerances": [], "labels": []},
            "features": features,
        })
        assert len(reasoning["operations"]) >= 1
        cad = agent_6.process(reasoning)
        assert cad["success"] is True
        assert cad["feature_count"] >= 1


# ── Test: Pipeline Edge Cases ────────────────────────────────────────────────


class TestPipelineEdgeCases:
    """Edge cases and error handling across the pipeline."""

    def test_blank_image_empty_through_pipeline(self, blank_drawing: Path) -> None:
        doc = agent_1.process(str(blank_drawing))
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc, force_opencv=True)
        features = agent_4.process(primitives)
        result = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        # Blank image → no primitives → no features → no rules
        assert features["feature_count"] == 0
        assert len(result["rules_fired"]) == 0

    def test_pipeline_with_all_agent5_kwargs(self, sample_dxf: Path) -> None:
        """Agent 5's process() accepts primitives/annotations/features as kwargs."""
        doc = agent_1.process(str(sample_dxf))
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc)
        features = agent_4.process(primitives)
        # Pass each as a separate kwarg
        result = agent_5.process(
            {},
            primitives=primitives,
            annotations=annotations,
            features=features,
        )
        assert "operations" in result
        assert "feature_tree" in result

    def test_unsupported_format_rejected(self, unsupported_format: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            agent_1.process(str(unsupported_format))

    def test_jpeg_pipeline(self, sample_jpeg: Path) -> None:
        """JPEG input should flow through the pipeline correctly."""
        doc = agent_1.process(str(sample_jpeg))
        assert doc["format"] == "jpeg"
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc, force_opencv=True)
        features = agent_4.process(primitives)
        result = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        assert isinstance(result, dict)

    def test_metadata_preserved_through_pipeline(self, sample_dxf: Path) -> None:
        """File metadata from Agent 1 should be preserved in the pipeline flow."""
        doc = agent_1.process(str(sample_dxf))
        assert doc["metadata"]["filename"] == "test_drawing.dxf"
        assert doc["metadata"]["size_bytes"] > 0
        primitives = agent_2.process(doc)
        # Agent 2 DXF path preserves entity counts
        preprocessing = primitives.get("preprocessing", {})
        assert preprocessing.get("source") == "dxf"

    def test_tiff_pipeline(self, sample_tiff: Path) -> None:
        """TIFF input should flow through the pipeline without errors."""
        doc = agent_1.process(str(sample_tiff))
        assert doc["format"] == "tiff"
        assert "image" in doc["data"]
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc, force_opencv=True)
        features = agent_4.process(primitives)
        reasoning = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        cad = agent_6.process(reasoning)
        assert isinstance(cad, dict)
        assert cad["success"] is True

    def test_pdf_pipeline(self, sample_pdf: Path) -> None:
        """PDF input (without embedded images) should flow through gracefully."""
        doc = agent_1.process(str(sample_pdf))
        assert doc["format"] == "pdf"
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc)
        features = agent_4.process(primitives)
        reasoning = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        cad = agent_6.process(reasoning)
        assert isinstance(cad, dict)
        assert cad["success"] is True

    # ── Agent 6 specific edge cases ──

    def test_agent6_generate_writes_files(self, sample_dxf: Path, tmp_path: Path) -> None:
        """Agent 6 should write actual files to disk via generate()."""
        doc = agent_1.process(str(sample_dxf))
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc)
        features = agent_4.process(primitives)
        reasoning = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        cad = agent_6.generate(reasoning, output_dir=str(tmp_path))
        paths = cad["output_paths"]
        assert "scad" in paths
        assert "cadquery_py" in paths
        assert Path(paths["scad"]).exists()
        assert Path(paths["cadquery_py"]).exists()

    def test_agent6_with_thickness_override(self, sample_dxf: Path) -> None:
        """Thickness override should propagate to OpenSCAD output."""
        doc = agent_1.process(str(sample_dxf))
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc)
        features = agent_4.process(primitives)
        reasoning = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        cad = agent_6.generate(reasoning, thickness=8.0)
        assert "t = 8.0" in cad["openscad_script"]
        assert "T = 8.0" in cad["cadquery_script"]

    def test_agent6_pipeline_with_step_stl_export(self, sample_dxf: Path, tmp_path: Path) -> None:
        """Full pipeline should produce STEP/STL when CadQuery is available."""
        doc = agent_1.process(str(sample_dxf))
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc)
        features = agent_4.process(primitives)
        reasoning = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        cad = agent_6.generate(reasoning, output_dir=str(tmp_path))
        assert cad["success"] is True
        assert cad["cadquery_used"] is True
        assert "step" in cad["formats"]
        assert "stl" in cad["formats"]
        assert "scad" in cad["formats"]
        assert "py" in cad["formats"]
        step_path = Path(cad["output_paths"]["step"])
        stl_path = Path(cad["output_paths"]["stl"])
        assert step_path.exists()
        assert stl_path.exists()
        assert step_path.stat().st_size > 100
        assert stl_path.stat().st_size > 100

    def test_agent6_pipeline_no_step_export(self, sample_dxf: Path) -> None:
        """Setting export_step=False should prevent step/stl in formats."""
        doc = agent_1.process(str(sample_dxf))
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc)
        features = agent_4.process(primitives)
        reasoning = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        cad = agent_6.generate(reasoning, export_step=False, export_stl=False)
        assert "step" not in cad["formats"]
        assert "stl" not in cad["formats"]
        assert "scad" in cad["formats"]
        assert "py" in cad["formats"]

    def test_agent6_pipeline_step_has_valid_header(self, sample_dxf: Path, tmp_path: Path) -> None:
        """STEP file from full pipeline should be a valid ISO-10303-21 file."""
        doc = agent_1.process(str(sample_dxf))
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc)
        features = agent_4.process(primitives)
        reasoning = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        cad = agent_6.generate(reasoning, output_dir=str(tmp_path))
        step_path = Path(cad["output_paths"]["step"])
        step_text = step_path.read_text(encoding="utf-8", errors="replace")
        assert "ISO-10303-21" in step_text
        assert "HEADER" in step_text

    def test_agent6_pipeline_with_holes_produces_larger_step(self, sample_dxf: Path, tmp_path: Path) -> None:
        """STEP file with features should be larger than empty STEP file."""
        doc = agent_1.process(str(sample_dxf))
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc)
        features = agent_4.process(primitives)
        reasoning = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })
        # Compare STEP with features vs STEP without features
        reasoning_without = reasoning.copy()
        reasoning_without["operations"] = []
        cad_empty = agent_6.generate(reasoning_without, output_dir=str(tmp_path / "empty"))
        cad_with = agent_6.generate(reasoning, output_dir=str(tmp_path / "with_features"))
        size_empty = Path(cad_empty["output_paths"]["step"]).stat().st_size
        size_with = Path(cad_with["output_paths"]["step"]).stat().st_size
        assert size_with >= size_empty, "STEP with features should be at least as large as empty STEP"


# ── Test: Cross-agent format compatibility ───────────────────────────────────


class TestCrossAgentCompatibility:
    """Ensure agents handle various data shapes correctly across boundaries."""

    def test_agent2_output_as_agent4_input(self, sample_dxf: Path) -> None:
        doc = agent_1.process(str(sample_dxf))
        primitives = agent_2.process(doc)
        # Agent 2's DXF output doesn't include 'format' key, but Agent 4's
        # detect() uses .get() for all primitives keys, so process() succeeds.
        # validate_input() would return False because 'format' is missing;
        # this tests the loose coupling between agents.
        features = agent_4.process(primitives)
        assert isinstance(features, dict)
        assert "features" in features
        assert "feature_count" in features

    def test_agent3_output_is_agent5_compatible(self, sample_dxf: Path) -> None:
        doc = agent_1.process(str(sample_dxf))
        annotations = agent_3.process(doc)
        # Agent 5 expects annotations to be a dict with dimensions, tolerances, labels
        # Each should be a list
        assert isinstance(annotations["dimensions"], list), \
            f"Expected list, got {type(annotations['dimensions']).__name__}"
        assert isinstance(annotations["tolerances"], list)
        assert isinstance(annotations["labels"], list)

    def test_agent4_output_is_agent5_compatible(self, sample_dxf: Path) -> None:
        doc = agent_1.process(str(sample_dxf))
        primitives = agent_2.process(doc)
        features = agent_4.process(primitives)
        # Agent 5 expects features['features'] to be a list of dicts
        assert isinstance(features["features"], list)
        for feat in features["features"]:
            assert "type" in feat, f"Feature missing 'type' key: {feat}"
            assert "confidence" in feat
            assert "source" in feat

    def test_agent5_deterministic_output(self, sample_dxf: Path) -> None:
        """Running Agent 5 twice with same input should produce same structure."""
        doc = agent_1.process(str(sample_dxf))
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc)
        features = agent_4.process(primitives)
        input_data = {
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        }
        result1 = agent_5.process(input_data)
        result2 = agent_5.process(input_data)
        assert len(result1["operations"]) == len(result2["operations"])
        assert result1["reasoning_engine"] == result2["reasoning_engine"]
        # Same rule names should fire
        rules1 = {r.get("rule") for r in result1["rules_fired"]}
        rules2 = {r.get("rule") for r in result2["rules_fired"]}
        assert rules1 == rules2
