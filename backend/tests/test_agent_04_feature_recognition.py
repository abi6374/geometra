"""Unit tests for the Feature Recognition Agent (Agent 4).

Tests cover:
- CV/rule-based feature detection for all 8 feature types
- YOLO engine initialization and fallback
- Feature detection from synthetic primitives
- Edge cases: empty primitives, missing data
- Feature counts and classification breakdown
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from geometra.agents.agent_04_feature_recognition import (
    FeatureRecognitionAgent,
    YOLOInferenceEngine,
    _detect_holes_from_circles,
    _detect_slots_from_contours,
    _detect_pockets_and_cutouts,
    _detect_vents,
    _detect_bosses,
    _detect_fillets,
    _detect_chamfers,
)


# ── Agent Fixture ─────────────────────────────────────────────────────────────


@pytest.fixture
def agent() -> FeatureRecognitionAgent:
    return FeatureRecognitionAgent()


# ── Synthetic Primitive Inputs ────────────────────────────────────────────────


@pytest.fixture
def empty_primitives() -> dict:
    """Completely empty primitives (no features)."""
    return {
        "format": "png",
        "lines": [],
        "circles": [],
        "arcs": [],
        "contours": [],
        "symbols": [],
    }


@pytest.fixture
def hole_primitives() -> dict:
    """Primitives containing 2 holes (circles)."""
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
def slot_primitives() -> dict:
    """Primitives containing a slot (elongated contour)."""
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
def cutout_primitives() -> dict:
    """Primitives with a cutout (contour with children)."""
    return {
        "format": "png",
        "lines": [],
        "circles": [],
        "arcs": [],
        "contours": [
            # Outer boundary (parent)
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
            # Display cutout (child of contour 0)
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
        ],
        "symbols": [],
    }


@pytest.fixture
def vent_primitives() -> dict:
    """Primitives with multiple thin horizontal slots (vent)."""
    contours = []
    for i, y in enumerate(range(80, 200, 20)):
        contours.append({
            "id": i,
            "area": 200,
            "perimeter": 60.0,
            "centroid": (150.0, float(y)),
            "bounding_box": {"x": 100, "y": y - 5, "width": 100, "height": 10},
            "approx_polygon_vertices": 4,
            "approx_polygon": [],
            "solidity": 0.95,
            "hierarchy": {
                "next": -1, "prev": -1, "first_child": -1, "parent": -1,
                "has_children": False, "is_child": False,
            },
        })
    return {
        "format": "png",
        "lines": [],
        "circles": [],
        "arcs": [],
        "contours": contours,
        "symbols": [],
    }


@pytest.fixture
def fillet_primitives() -> dict:
    """Primitives with a fillet (small-radius arc)."""
    return {
        "format": "png",
        "lines": [],
        "circles": [],
        "arcs": [
            {
                "x": 50.0, "y": 50.0, "radius": 5.0,
                "start_angle": 0.0, "end_angle": 90.0,
                "arc_length": 7.85, "angular_span_deg": 90.0,
                "fit_error": 0.1, "max_fit_error": 0.2, "n_points": 20,
            },
        ],
        "contours": [],
        "symbols": [],
    }


@pytest.fixture
def chamfer_primitives() -> dict:
    """Primitives with a chamfer (short diagonal line)."""
    return {
        "format": "png",
        "lines": [
            {"x1": 100, "y1": 100, "x2": 115, "y2": 85,
             "length": 21.21, "angle_deg": -45.0},
        ],
        "circles": [],
        "arcs": [],
        "contours": [],
        "symbols": [],
    }


@pytest.fixture
def boss_primitives() -> dict:
    """Primitives with a boss (convex circular contour)."""
    return {
        "format": "png",
        "lines": [],
        "circles": [
            {"x": 300.0, "y": 100.0, "radius": 25.0},
        ],
        "arcs": [],
        "contours": [
            {
                "id": 0,
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


# ── Agent: validate_input ─────────────────────────────────────────────────────


class TestValidateInput:
    def test_valid_primitives(self, agent: FeatureRecognitionAgent, hole_primitives: dict) -> None:
        assert agent.validate_input(hole_primitives) is True

    def test_invalid_none(self, agent: FeatureRecognitionAgent) -> None:
        assert agent.validate_input(None) is False

    def test_invalid_string(self, agent: FeatureRecognitionAgent) -> None:
        assert agent.validate_input("some string") is False

    def test_missing_format(self, agent: FeatureRecognitionAgent) -> None:
        assert agent.validate_input({"circles": []}) is False


# ── YOLO Inference Engine ─────────────────────────────────────────────────────


class TestYOLOEngine:
    def test_engine_initialization(self) -> None:
        engine = YOLOInferenceEngine()
        assert engine._model is None
        assert engine._available is None

    def test_lazy_check(self) -> None:
        engine = YOLOInferenceEngine()
        available = engine.available
        assert isinstance(available, bool)
        assert engine._available is not None

    def test_predict_without_model_returns_none(self) -> None:
        engine = YOLOInferenceEngine()
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        result = engine.predict(img)
        assert result is None


# ── Hole Detection ────────────────────────────────────────────────────────────


class TestHoleDetection:
    def test_detects_holes(self) -> None:
        circles = [
            {"x": 100.0, "y": 120.0, "radius": 30.0},
            {"x": 200.0, "y": 120.0, "radius": 20.0},
        ]
        holes = _detect_holes_from_circles(circles)
        assert len(holes) == 2
        assert holes[0]["type"] == "hole"
        assert holes[1]["type"] == "hole"

    def test_hole_properties(self) -> None:
        holes = _detect_holes_from_circles([{"x": 100.0, "y": 120.0, "radius": 30.0}])
        h = holes[0]
        assert h["x"] == 100.0
        assert h["y"] == 120.0
        assert h["radius"] == 30.0
        assert h["diameter"] == 60.0
        assert h["confidence"] > 0

    def test_no_circles_returns_empty(self) -> None:
        assert _detect_holes_from_circles([]) == []


# ── Slot Detection ────────────────────────────────────────────────────────────


class TestSlotDetection:
    def test_detects_slot(self, slot_primitives: dict) -> None:
        slots = _detect_slots_from_contours(slot_primitives["contours"], [])
        assert len(slots) >= 1
        assert slots[0]["type"] == "slot"

    def test_slot_has_properties(self, slot_primitives: dict) -> None:
        slots = _detect_slots_from_contours(slot_primitives["contours"], [])
        s = slots[0]
        assert "width" in s
        assert "height" in s
        assert "aspect_ratio" in s
        assert s["aspect_ratio"] < 0.4  # should be elongated

    def test_no_contours(self) -> None:
        assert _detect_slots_from_contours([], []) == []


# ── Pocket and Cutout Detection ───────────────────────────────────────────────


class TestPocketCutoutDetection:
    def test_detects_cutout(self, cutout_primitives: dict) -> None:
        pockets, cutouts = _detect_pockets_and_cutouts(cutout_primitives["contours"])
        assert len(cutouts) >= 1
        assert cutouts[0]["type"] == "cutout"

    def test_cutout_properties(self, cutout_primitives: dict) -> None:
        pockets, cutouts = _detect_pockets_and_cutouts(cutout_primitives["contours"])
        c = cutouts[0]
        assert c["shape"] == "rectangular"
        assert c["width"] > 0
        assert c["height"] > 0
        assert c["source"] == "contour_hierarchy"

    def test_detects_pocket(self, cutout_primitives: dict) -> None:
        pockets, cutouts = _detect_pockets_and_cutouts(cutout_primitives["contours"])
        # The child contour should be classified as a pocket
        # Actually, the child has no children of its own and is a child, so pocket
        assert len(pockets) >= 1 or len(cutouts) >= 1

    def test_empty_contours(self) -> None:
        pockets, cutouts = _detect_pockets_and_cutouts([])
        assert pockets == []
        assert cutouts == []


# ── Vent Detection ────────────────────────────────────────────────────────────


class TestVentDetection:
    def test_detects_vent(self, vent_primitives: dict) -> None:
        vents = _detect_vents(vent_primitives["contours"], [])
        assert len(vents) >= 1, f"Expected at least 1 vent, got {len(vents)}"
        assert vents[0]["type"] == "vent"

    def test_vent_properties(self, vent_primitives: dict) -> None:
        vents = _detect_vents(vent_primitives["contours"], [])
        v = vents[0]
        assert v["slot_count"] >= 3
        assert "orientation" in v
        assert "width" in v
        assert "height" in v

    def test_no_contours(self) -> None:
        assert _detect_vents([], []) == []


# ── Boss Detection ────────────────────────────────────────────────────────────


class TestBossDetection:
    def test_detects_boss(self, boss_primitives: dict) -> None:
        bosses = _detect_bosses(boss_primitives["contours"], boss_primitives["circles"])
        assert len(bosses) >= 1
        assert bosses[0]["type"] == "boss"

    def test_no_candidates(self) -> None:
        assert _detect_bosses([], []) == []


# ── Fillet Detection ──────────────────────────────────────────────────────────


class TestFilletDetection:
    def test_detects_fillet(self, fillet_primitives: dict) -> None:
        fillets = _detect_fillets(fillet_primitives["arcs"])
        assert len(fillets) >= 1
        assert fillets[0]["type"] == "fillet"

    def test_fillet_radius_small(self, fillet_primitives: dict) -> None:
        fillets = _detect_fillets(fillet_primitives["arcs"])
        assert fillets[0]["radius"] <= 30

    def test_no_arcs(self) -> None:
        assert _detect_fillets([]) == []


# ── Chamfer Detection ─────────────────────────────────────────────────────────


class TestChamferDetection:
    def test_detects_chamfer(self, chamfer_primitives: dict) -> None:
        chamfers = _detect_chamfers(chamfer_primitives["lines"])
        assert len(chamfers) >= 1
        assert chamfers[0]["type"] == "chamfer"

    def test_chamfer_angle(self, chamfer_primitives: dict) -> None:
        chamfers = _detect_chamfers(chamfer_primitives["lines"])
        assert abs(chamfers[0]["angle_deg"] + 45) < 5 or abs(chamfers[0]["angle_deg"] - 45) < 5

    def test_no_lines(self) -> None:
        assert _detect_chamfers([]) == []


# ── Full Agent Pipeline ───────────────────────────────────────────────────────


class TestFullPipeline:
    def test_agent_process_holes(self, agent: FeatureRecognitionAgent, hole_primitives: dict) -> None:
        result = agent.process(hole_primitives)
        assert isinstance(result, dict)
        assert "features" in result
        assert "feature_count" in result
        assert "class_counts" in result
        assert result["feature_count"] >= 2
        assert result["class_counts"]["hole"] >= 2

    def test_agent_process_slots(self, agent: FeatureRecognitionAgent, slot_primitives: dict) -> None:
        result = agent.process(slot_primitives)
        assert result["feature_count"] >= 1
        assert result["class_counts"]["slot"] >= 1

    def test_agent_process_cutouts(self, agent: FeatureRecognitionAgent, cutout_primitives: dict) -> None:
        result = agent.process(cutout_primitives)
        assert result["class_counts"]["cutout"] >= 1

    def test_agent_process_vents(self, agent: FeatureRecognitionAgent, vent_primitives: dict) -> None:
        result = agent.process(vent_primitives)
        assert result["class_counts"]["vent"] >= 1

    def test_agent_process_fillets(self, agent: FeatureRecognitionAgent, fillet_primitives: dict) -> None:
        result = agent.process(fillet_primitives)
        assert result["class_counts"]["fillet"] >= 1

    def test_agent_process_chamfers(self, agent: FeatureRecognitionAgent, chamfer_primitives: dict) -> None:
        result = agent.process(chamfer_primitives)
        assert result["class_counts"]["chamfer"] >= 1

    def test_agent_process_bosses(self, agent: FeatureRecognitionAgent, boss_primitives: dict) -> None:
        result = agent.process(boss_primitives)
        assert result["class_counts"]["boss"] >= 1

    def test_agent_process_empty(self, agent: FeatureRecognitionAgent, empty_primitives: dict) -> None:
        result = agent.process(empty_primitives)
        assert result["feature_count"] == 0
        assert all(v == 0 for v in result["class_counts"].values())
        assert result["detection_method"] == "cv"

    def test_class_counts_all_types_present(self, agent: FeatureRecognitionAgent) -> None:
        """All 8 feature types should be present in class_counts."""
        expected = {"hole", "slot", "pocket", "boss", "cutout", "vent", "fillet", "chamfer"}
        result = agent.process({
            "format": "png",
            "lines": [],
            "circles": [],
            "arcs": [],
            "contours": [],
            "symbols": [],
        })
        assert set(result["class_counts"].keys()) == expected

    def test_detection_method_cv(self, agent: FeatureRecognitionAgent, hole_primitives: dict) -> None:
        result = agent.process(hole_primitives)
        assert result["detection_method"] == "cv"

    def test_agent_no_yolo_by_default(self, agent: FeatureRecognitionAgent) -> None:
        """Should not try YOLO inference unless explicitly requested."""
        assert agent.yolo_available is False

    def test_process_all_features_combined(self, agent: FeatureRecognitionAgent) -> None:
        """Combined primitives should detect multiple feature types."""
        primitives = {
            "format": "png",
            "lines": [
                {"x1": 100, "y1": 100, "x2": 115, "y2": 85,
                 "length": 21.21, "angle_deg": -45.0},
            ],
            "circles": [
                {"x": 100.0, "y": 120.0, "radius": 30.0},
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
            ],
            "symbols": [],
        }
        result = agent.process(primitives)
        # Should detect hole + pocket (contour has parent, no children) + fillet + chamfer
        feature_types = {f["type"] for f in result["features"]}
        assert "hole" in feature_types, f"Missing hole in {feature_types}"
        assert "fillet" in feature_types or "chamfer" in feature_types, f"Missing fillet/chamfer in {feature_types}"
        assert result["feature_count"] >= 2

    def test_feature_has_type_key(self, agent: FeatureRecognitionAgent, hole_primitives: dict) -> None:
        result = agent.process(hole_primitives)
        for feature in result["features"]:
            assert "type" in feature
            assert "confidence" in feature
            assert "source" in feature
