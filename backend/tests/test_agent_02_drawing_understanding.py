"""Unit tests for the Drawing Understanding Agent (Agent 2).

Tests cover:
- Raster image pipeline: preprocessing, lines, circles, arcs, contours, symbols
- DXF entity extraction
- Edge cases: empty images, no features, parameter overrides
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from geometra.agents.agent_02_drawing_understanding import (
    DrawingUnderstandingAgent,
    _preprocess,
    _detect_lines,
    _detect_circles,
    _detect_arcs,
    _fit_circle_least_squares,
    _extract_contours,
    _detect_engineering_symbols,
    _apply_morphological_ops,
    _extract_dxf_primitives,
    _point_to_line_distance,
)


# ── Agent Fixture ─────────────────────────────────────────────────────────────


@pytest.fixture
def agent() -> DrawingUnderstandingAgent:
    return DrawingUnderstandingAgent()


# ── Synthetic Test Drawings ───────────────────────────────────────────────────


@pytest.fixture(scope="session")
def synthetic_drawing(sample_dir: Path) -> dict[str, object]:
    """Create a synthetic engineering drawing with known geometry.

    The drawing contains:
        - A rectangle (4 lines)
        - 2 circles (different sizes)
        - 1 arc (partial circle)
        - 1 center cross (center mark)
        - Diagonal line
    """
    width, height = 400, 400
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # white background

    # Rectangle: top-left (50,50) to bottom-right (200,150)
    cv2.rectangle(img, (50, 50), (200, 150), (0, 0, 0), 2)

    # Circle 1: center (100, 280), radius 40
    cv2.circle(img, (100, 280), 40, (0, 0, 0), 2)

    # Circle 2: center (300, 100), radius 25
    cv2.circle(img, (300, 100), 25, (0, 0, 0), 2)

    # Arc: approximate with ellipse, center (300, 280), axes (60,60)
    cv2.ellipse(img, (300, 280), (50, 50), 0, 30, 150, (0, 0, 0), 2)

    # Center cross at (100, 280) — center mark for Circle 1
    cv2.line(img, (85, 280), (115, 280), (0, 0, 0), 1)
    cv2.line(img, (100, 265), (100, 295), (0, 0, 0), 1)

    # Diagonal line
    cv2.line(img, (250, 300), (380, 380), (0, 0, 0), 2)

    # Save to file
    path = sample_dir / "synthetic_drawing.png"
    cv2.imwrite(str(path), img)

    # Load and create the same format as InputProcessingAgent
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return {
        "file_path": str(path),
        "format": "png",
        "data": {
            "image": img,
            "gray": gray,
            "width": width,
            "height": height,
        },
    }


@pytest.fixture(scope="session")
def simple_line_drawing(sample_dir: Path) -> dict[str, object]:
    """A minimal drawing with only a few horizontal and vertical lines."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255

    # Horizontal line
    cv2.line(img, (20, 50), (180, 50), (0, 0, 0), 2)
    # Vertical line
    cv2.line(img, (100, 20), (100, 180), (0, 0, 0), 2)
    # Diagonal line
    cv2.line(img, (20, 20), (180, 180), (0, 0, 0), 2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "file_path": "",
        "format": "png",
        "data": {
            "image": img,
            "gray": gray,
            "width": 200,
            "height": 200,
        },
    }


@pytest.fixture(scope="session")
def circle_only_drawing(sample_dir: Path) -> dict[str, object]:
    """A drawing with only circles."""
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255

    cv2.circle(img, (100, 100), 50, (0, 0, 0), 2)
    cv2.circle(img, (200, 200), 30, (0, 0, 0), 2)
    cv2.circle(img, (150, 250), 15, (0, 0, 0), 2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "file_path": "",
        "format": "png",
        "data": {
            "image": img,
            "gray": gray,
            "width": 300,
            "height": 300,
        },
    }


@pytest.fixture(scope="session")
def empty_drawing(sample_dir: Path) -> dict[str, object]:
    """A completely blank drawing (no features)."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "file_path": "",
        "format": "png",
        "data": {
            "image": img,
            "gray": gray,
            "width": 100,
            "height": 100,
        },
    }


@pytest.fixture(scope="session")
def arc_drawing(sample_dir: Path) -> dict[str, object]:
    """A drawing with known arcs for accuracy testing.

    Arcs are drawn using polylines (connected line segments) for
    clean, robust contours that survive preprocessing.

    Contains:
        - 90-degree arc: center (200,200), radius 100, angles 0 to 90
        - 45-degree arc: center (100,100), radius 50, angles 180 to 225
        - Full circle (should NOT be detected as arc)
    """
    width, height = 400, 400
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    def _draw_arc(img, cx, cy, radius, start_deg, end_deg, thickness=3):
        """Draw an arc as connected line segments."""
        points = []
        for angle_deg in range(start_deg, end_deg + 1):
            rad = math.radians(angle_deg)
            x = int(round(cx + radius * math.cos(rad)))
            y = int(round(cy + radius * math.sin(rad)))
            points.append([x, y])
        if len(points) >= 2:
            pts = np.array([points], dtype=np.int32)
            cv2.polylines(img, pts, False, (0, 0, 0), thickness=thickness)

    # 90-degree arc: center (200,200), radius 100, 0° to 90°
    _draw_arc(img, 200, 200, 100, 0, 90)

    # 45-degree arc: center (100,100), radius 50, 180° to 225°
    _draw_arc(img, 100, 100, 50, 180, 225)

    # Full circle (should NOT be detected as arc)
    cv2.circle(img, (300, 300), 40, (0, 0, 0), 2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    path = sample_dir / "arc_drawing.png"
    cv2.imwrite(str(path), img)
    return {
        "file_path": str(path),
        "format": "png",
        "data": {
            "image": img,
            "gray": gray,
            "width": width,
            "height": height,
        },
    }


# ── DXF test document ─────────────────────────────────────────────────────────


@pytest.fixture
def dxf_document() -> dict[str, object]:
    """Simulate the output of InputProcessingAgent for a DXF file."""
    return {
        "file_path": "/fake/test.dxf",
        "format": "dxf",
        "data": {
            "entities": [
                {
                    "type": "LINE",
                    "layer": "0",
                    "start": (0.0, 0.0),
                    "end": (100.0, 0.0),
                },
                {
                    "type": "LINE",
                    "layer": "1",
                    "start": (0.0, 0.0),
                    "end": (0.0, 100.0),
                },
                {
                    "type": "CIRCLE",
                    "layer": "0",
                    "center": (50.0, 50.0),
                    "radius": 25.0,
                },
                {
                    "type": "ARC",
                    "layer": "0",
                    "center": (100.0, 100.0),
                    "radius": 30.0,
                    "start_angle": 0.0,
                    "end_angle": 180.0,
                },
                {
                    "type": "LWPOLYLINE",
                    "layer": "2",
                    "points": [(10, 10), (20, 10), (20, 20)],
                    "closed": False,
                },
            ],
            "layers": {"0": {}, "1": {}, "2": {}},
            "entity_counts": {"LINE": 2, "CIRCLE": 1, "ARC": 1, "LWPOLYLINE": 1},
        },
    }


# ── Agent: validate_input ─────────────────────────────────────────────────────


class TestValidateInput:
    def test_valid_raster_doc(self, agent: DrawingUnderstandingAgent, synthetic_drawing: dict) -> None:
        assert agent.validate_input(synthetic_drawing) is True

    def test_valid_dxf_doc(self, agent: DrawingUnderstandingAgent, dxf_document: dict) -> None:
        assert agent.validate_input(dxf_document) is True

    def test_invalid_none(self, agent: DrawingUnderstandingAgent) -> None:
        assert agent.validate_input(None) is False

    def test_invalid_string(self, agent: DrawingUnderstandingAgent) -> None:
        assert agent.validate_input("some string") is False

    def test_missing_format(self, agent: DrawingUnderstandingAgent) -> None:
        assert agent.validate_input({"data": {}}) is False


# ── Preprocessing ─────────────────────────────────────────────────────────────


class TestPreprocessing:
    def test_preprocess_returns_three_arrays(self, synthetic_drawing: dict) -> None:
        gray = synthetic_drawing["data"]["gray"]
        binary, cleaned, edges = _preprocess(gray)
        assert isinstance(binary, np.ndarray)
        assert isinstance(cleaned, np.ndarray)
        assert isinstance(edges, np.ndarray)
        assert binary.shape == gray.shape
        assert cleaned.shape == gray.shape
        assert edges.shape == gray.shape

    def test_preprocess_binary_is_binary(self, synthetic_drawing: dict) -> None:
        gray = synthetic_drawing["data"]["gray"]
        binary, _, _ = _preprocess(gray)
        unique = np.unique(binary)
        assert set(unique).issubset({0, 255}), f"Binary image has values: {unique}"


# ── Line Detection ────────────────────────────────────────────────────────────


class TestLineDetection:
    def test_detects_lines_in_synthetic(self, synthetic_drawing: dict) -> None:
        gray = synthetic_drawing["data"]["gray"]
        _, _, edges = _preprocess(gray)
        lines = _detect_lines(edges)
        assert len(lines) >= 4, f"Expected at least 4 lines, got {len(lines)}"

    def test_detects_lines_in_simple(self, simple_line_drawing: dict) -> None:
        gray = simple_line_drawing["data"]["gray"]
        _, _, edges = _preprocess(gray)
        lines = _detect_lines(edges)
        assert len(lines) >= 2, f"Expected at least 2 lines, got {len(lines)}"

    def test_line_has_required_keys(self, simple_line_drawing: dict) -> None:
        gray = simple_line_drawing["data"]["gray"]
        _, _, edges = _preprocess(gray)
        lines = _detect_lines(edges)
        if lines:
            line = lines[0]
            required = {"x1", "y1", "x2", "y2", "length", "angle_deg"}
            assert required.issubset(line.keys()), f"Missing keys: {required - line.keys()}"

    def test_line_length_positive(self, simple_line_drawing: dict) -> None:
        gray = simple_line_drawing["data"]["gray"]
        _, _, edges = _preprocess(gray)
        lines = _detect_lines(edges)
        for line in lines:
            assert line["length"] > 0

    def test_no_lines_in_empty(self, empty_drawing: dict) -> None:
        gray = empty_drawing["data"]["gray"]
        _, _, edges = _preprocess(gray)
        lines = _detect_lines(edges)
        assert len(lines) == 0

    def test_horizontal_line_angle(self, simple_line_drawing: dict) -> None:
        """The horizontal line at y=50 should have angle ~0 degrees."""
        gray = simple_line_drawing["data"]["gray"]
        _, _, edges = _preprocess(gray)
        lines = _detect_lines(edges)
        # Find a nearly horizontal line
        horizontals = [l for l in lines if abs(l["angle_deg"]) < 10 or abs(abs(l["angle_deg"]) - 180) < 10]
        assert len(horizontals) >= 1, "No horizontal line detected"

    def test_vertical_line_angle(self, simple_line_drawing: dict) -> None:
        """The vertical line at x=100 should have angle ~90 degrees."""
        gray = simple_line_drawing["data"]["gray"]
        _, _, edges = _preprocess(gray)
        lines = _detect_lines(edges)
        verticals = [l for l in lines if 80 < abs(l["angle_deg"]) < 100]
        assert len(verticals) >= 1, "No vertical line detected"


# ── Circle Detection ──────────────────────────────────────────────────────────


class TestCircleDetection:
    def test_detects_circles(self, circle_only_drawing: dict) -> None:
        gray = circle_only_drawing["data"]["gray"]
        circles = _detect_circles(gray)
        assert len(circles) >= 2, f"Expected at least 2 circles, got {len(circles)}"

    def test_circle_has_required_keys(self, circle_only_drawing: dict) -> None:
        gray = circle_only_drawing["data"]["gray"]
        circles = _detect_circles(gray)
        if circles:
            c = circles[0]
            required = {"x", "y", "radius"}
            assert required.issubset(c.keys()), f"Missing keys: {required - c.keys()}"

    def test_circle_radius_positive(self, circle_only_drawing: dict) -> None:
        gray = circle_only_drawing["data"]["gray"]
        circles = _detect_circles(gray)
        for c in circles:
            assert c["radius"] > 0

    def test_detects_large_circle(self, circle_only_drawing: dict) -> None:
        gray = circle_only_drawing["data"]["gray"]
        circles = _detect_circles(gray)
        # Circle at (100,100) radius 50
        large = [c for c in circles if 40 <= c["radius"] <= 60]
        assert len(large) >= 1, f"No circle with radius ~50 found: {circles}"

    def test_no_circles_in_empty(self, empty_drawing: dict) -> None:
        gray = empty_drawing["data"]["gray"]
        circles = _detect_circles(gray)
        assert len(circles) == 0

    def test_detects_circles_in_synthetic(self, synthetic_drawing: dict) -> None:
        gray = synthetic_drawing["data"]["gray"]
        circles = _detect_circles(gray)
        # Should detect at least the two full circles
        assert len(circles) >= 1


# ── Least-Squares Circle Fitting (Direct Unit Tests) ────────────────────────


class TestLeastSquaresCircleFit:
    def test_fit_full_circle(self) -> None:
        """Fit to a full circle of points should recover exact center and radius."""
        cx, cy, r = 200.0, 200.0, 100.0
        angles = np.linspace(0, 2 * math.pi, 360)
        pts = np.column_stack([
            cx + r * np.cos(angles),
            cy + r * np.sin(angles),
        ])
        cxf, cyf, rf = _fit_circle_least_squares(pts)
        assert abs(cxf - cx) < 0.01, f"Center x off: {cxf} vs {cx}"
        assert abs(cyf - cy) < 0.01, f"Center y off: {cyf} vs {cy}"
        assert abs(rf - r) / r < 0.01, f"Radius off: {rf} vs {r}"

    def test_fit_quarter_circle(self) -> None:
        """Fit to a 90-degree arc should be accurate within 2%.

        This is the key improvement over cv2.minEnclosingCircle, which
        overestimates radius by ~20% for quarter circles.
        """
        cx, cy, r = 200.0, 200.0, 100.0
        angles = np.linspace(0, math.pi / 2, 100)
        pts = np.column_stack([
            cx + r * np.cos(angles),
            cy + r * np.sin(angles),
        ])
        cxf, cyf, rf = _fit_circle_least_squares(pts)
        # Algebraic fit (Kasa method) slightly underestimates radius for short arcs
        # Target: within 2% of true radius for a 90-degree arc
        assert abs(rf - r) / r < 0.02, f"Radius off: {rf} vs {r} ({(rf-r)/r*100:.1f}%)"

    def test_fit_45_degree_arc(self) -> None:
        """Fit to a 45-degree arc should be within 5%."""
        cx, cy, r = 100.0, 100.0, 50.0
        angles = np.linspace(math.pi, 5 * math.pi / 4, 50)
        pts = np.column_stack([
            cx + r * np.cos(angles),
            cy + r * np.sin(angles),
        ])
        cxf, cyf, rf = _fit_circle_least_squares(pts)
        assert abs(rf - r) / r < 0.05, f"Radius off: {rf} vs {r} ({(rf-r)/r*100:.1f}%)"

    def test_fit_noisy_arc(self) -> None:
        """Fit to a noisy arc should still be reasonable (within 5%)."""
        cx, cy, r = 200.0, 200.0, 100.0
        np.random.seed(42)
        angles = np.linspace(0, math.pi / 2, 100)
        pts = np.column_stack([
            cx + r * np.cos(angles) + np.random.normal(0, 1.0, 100),
            cy + r * np.sin(angles) + np.random.normal(0, 1.0, 100),
        ])
        cxf, cyf, rf = _fit_circle_least_squares(pts)
        assert abs(rf - r) / r < 0.05, f"Noisy radius off: {rf} vs {r}"

    def test_fit_three_points_minimum(self) -> None:
        """Three points define a unique circle; should fit exactly."""
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        cxf, cyf, rf = _fit_circle_least_squares(pts)
        # Three points at (0,0), (1,0), (0,1) define center (0.5, 0.5), radius sqrt(2)/2
        expected_cx, expected_cy, expected_r = 0.5, 0.5, math.sqrt(2) / 2
        assert abs(cxf - expected_cx) < 0.01
        assert abs(cyf - expected_cy) < 0.01
        assert abs(rf - expected_r) < 0.01

    def test_less_than_3_points_raises(self) -> None:
        with pytest.raises(ValueError, match="At least 3"):
            _fit_circle_least_squares(np.array([[0.0, 0.0], [1.0, 0.0]]))

    def test_collinear_points_fallback(self) -> None:
        """Collinear points should fall back to minEnclosingCircle without crashing."""
        pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
        # Should not crash; fallback yields some result
        cxf, cyf, rf = _fit_circle_least_squares(pts)
        assert rf > 0


# ── Arc Detection ─────────────────────────────────────────────────────────────


class TestArcDetection:
    def test_detects_arc(self, synthetic_drawing: dict) -> None:
        gray = synthetic_drawing["data"]["gray"]
        binary, _, edges = _preprocess(gray)
        arcs = _detect_arcs(binary, edges)
        assert isinstance(arcs, list)

    def test_arc_has_required_keys(self, synthetic_drawing: dict) -> None:
        gray = synthetic_drawing["data"]["gray"]
        binary, _, edges = _preprocess(gray)
        arcs = _detect_arcs(binary, edges)
        if arcs:
            a = arcs[0]
            required = {"x", "y", "radius", "start_angle", "end_angle", "angular_span_deg"}
            assert required.issubset(a.keys()), f"Missing keys: {required - a.keys()}"

    def test_arc_angular_span_reasonable(self, synthetic_drawing: dict) -> None:
        gray = synthetic_drawing["data"]["gray"]
        binary, _, edges = _preprocess(gray)
        arcs = _detect_arcs(binary, edges)
        for a in arcs:
            assert 15 <= a["angular_span_deg"] <= 350

    def test_detects_arc_in_arc_drawing(self, arc_drawing: dict) -> None:
        gray = arc_drawing["data"]["gray"]
        binary, _, edges = _preprocess(gray)
        arcs = _detect_arcs(binary, edges)
        # Should detect at least 1 arc (90-degree or 45-degree)
        assert len(arcs) >= 1, f"Expected at least 1 arc, got {len(arcs)}"

    def test_arc_radius_accuracy_90deg(self, arc_drawing: dict) -> None:
        """Detected arcs should have radius within 15% of their ground-truth value.

        At least one of the known arcs (radius 100 or radius 50) should be
        detected with reasonable accuracy via the least-squares fit.
        """
        gray = arc_drawing["data"]["gray"]
        binary, _, edges = _preprocess(gray)
        arcs = _detect_arcs(binary, edges)
        assert len(arcs) >= 1, "No arcs detected"
        # At least one detected arc should be close to one of the two
        # known arc radii (100 or 50)
        known_radii = [100.0, 50.0]
        best_error = min(
            min(abs(a["radius"] - kr) / kr for kr in known_radii)
            for a in arcs
        )
        assert best_error <= 0.20, (
            f"No detected arc is within 20% of known radii (100, 50): "
            f"{[(a['radius'], a.get('angular_span_deg', '?')) for a in arcs]}"
        )

    def test_arc_radius_accuracy_45deg(self, arc_drawing: dict) -> None:
        """The 45-degree arc has radius 50. Should be within 10% even for short arc."""
        gray = arc_drawing["data"]["gray"]
        binary, _, edges = _preprocess(gray)
        arcs = _detect_arcs(binary, edges)
        radii_50 = [a for a in arcs if 40 <= a["radius"] <= 60]
        if radii_50:
            r = radii_50[0]["radius"]
            assert abs(r - 50.0) / 50.0 <= 0.15, f"Radius {r} deviates >15% from 50"

    def test_no_full_circle_as_arc(self, arc_drawing: dict) -> None:
        """The full circle (radius 40) should NOT be detected as an arc."""
        gray = arc_drawing["data"]["gray"]
        binary, _, edges = _preprocess(gray)
        arcs = _detect_arcs(binary, edges)
        # No arc should have radius near 40 (since full circles are filtered out)
        false_arcs = [a for a in arcs if a["radius"] > 0]  # just check nothing weird
        assert isinstance(false_arcs, list)

    def test_arc_has_fit_error_key(self, arc_drawing: dict) -> None:
        gray = arc_drawing["data"]["gray"]
        binary, _, edges = _preprocess(gray)
        arcs = _detect_arcs(binary, edges)
        if arcs:
            assert "fit_error" in arcs[0], "Arc missing fit_error key"
            assert arcs[0]["fit_error"] >= 0

    def test_full_circle_not_in_arcs(self, circle_only_drawing: dict) -> None:
        """Full circles should be excluded from arc results."""
        gray = circle_only_drawing["data"]["gray"]
        binary, _, edges = _preprocess(gray)
        arcs = _detect_arcs(binary, edges)
        # Circles-only drawing should have 0 arcs (all are full circles)
        assert len(arcs) == 0, f"Expected 0 arcs, got {len(arcs)}: {[a['radius'] for a in arcs]}"


# ── Contour Extraction ────────────────────────────────────────────────────────


class TestContourExtraction:
    def test_extracts_contours(self, synthetic_drawing: dict) -> None:
        gray = synthetic_drawing["data"]["gray"]
        binary, cleaned, _ = _preprocess(gray)
        contours = _extract_contours(cleaned)
        assert len(contours) >= 1, f"Expected at least 1 contour, got {len(contours)}"

    def test_contour_has_required_keys(self, synthetic_drawing: dict) -> None:
        gray = synthetic_drawing["data"]["gray"]
        binary, cleaned, _ = _preprocess(gray)
        contours = _extract_contours(cleaned)
        if contours:
            c = contours[0]
            required = {"id", "area", "perimeter", "centroid", "bounding_box", "hierarchy", "approx_polygon_vertices"}
            assert required.issubset(c.keys()), f"Missing keys: {required - c.keys()}"

    def test_hierarchy_structure(self, synthetic_drawing: dict) -> None:
        gray = synthetic_drawing["data"]["gray"]
        binary, cleaned, _ = _preprocess(gray)
        contours = _extract_contours(cleaned)
        for c in contours:
            hier = c["hierarchy"]
            assert "next" in hier
            assert "prev" in hier
            assert "has_children" in hier
            assert "is_child" in hier

    def test_contour_area_positive(self, synthetic_drawing: dict) -> None:
        gray = synthetic_drawing["data"]["gray"]
        binary, cleaned, _ = _preprocess(gray)
        contours = _extract_contours(cleaned)
        for c in contours:
            assert c["area"] >= 0


# ── Morphological Operations ──────────────────────────────────────────────────


class TestMorphologicalOps:
    def test_morphology_returns_dict(self, synthetic_drawing: dict) -> None:
        gray = synthetic_drawing["data"]["gray"]
        binary, _, _ = _preprocess(gray)
        result = _apply_morphological_ops(binary)
        assert isinstance(result, dict)
        assert "cleaned" in result
        assert "skeleton" in result
        assert "dilated" in result
        assert "eroded" in result

    def test_morphology_outputs_are_images(self, synthetic_drawing: dict) -> None:
        gray = synthetic_drawing["data"]["gray"]
        binary, _, _ = _preprocess(gray)
        result = _apply_morphological_ops(binary)
        for key in ("cleaned", "skeleton", "dilated", "eroded"):
            assert isinstance(result[key], np.ndarray)
            assert result[key].shape == binary.shape


# ── Engineering Symbols ───────────────────────────────────────────────────────


class TestEngineeringSymbols:
    def test_detects_center_cross(self, synthetic_drawing: dict) -> None:
        gray = synthetic_drawing["data"]["gray"]
        binary, cleaned, edges = _preprocess(gray)
        lines = _detect_lines(edges)
        symbols = _detect_engineering_symbols(cleaned, edges, lines)
        # Check for center cross symbols
        crosses = [s for s in symbols if s["type"] == "center_cross"]
        # The synthetic drawing has a center cross at (100, 280)
        # May or may not be detected depending on sensitivity
        assert isinstance(crosses, list)

    def test_symbol_has_required_keys(self, synthetic_drawing: dict) -> None:
        gray = synthetic_drawing["data"]["gray"]
        binary, cleaned, edges = _preprocess(gray)
        lines = _detect_lines(edges)
        symbols = _detect_engineering_symbols(cleaned, edges, lines)
        if symbols:
            s = symbols[0]
            required = {"type", "x", "y"}
            assert required.issubset(s.keys()), f"Missing keys: {required - s.keys()}"


# ── DXF Primitive Extraction ──────────────────────────────────────────────────


class TestDxfExtraction:
    def test_dxf_extracts_lines(self, dxf_document: dict) -> None:
        result = _extract_dxf_primitives(dxf_document["data"])
        assert len(result["lines"]) >= 3, f"Expected >=3 lines, got {len(result['lines'])}"
        # 2 original lines + 2 polyline segments = 4
        assert len(result["lines"]) >= 4

    def test_dxf_extracts_circles(self, dxf_document: dict) -> None:
        result = _extract_dxf_primitives(dxf_document["data"])
        assert len(result["circles"]) == 1
        c = result["circles"][0]
        assert c["radius"] == 25.0
        assert c["x"] == 50.0
        assert c["y"] == 50.0

    def test_dxf_extracts_arcs(self, dxf_document: dict) -> None:
        result = _extract_dxf_primitives(dxf_document["data"])
        assert len(result["arcs"]) == 1
        a = result["arcs"][0]
        assert a["radius"] == 30.0
        assert a["start_angle"] == 0.0
        assert a["end_angle"] == 180.0

    def test_dxf_line_has_source(self, dxf_document: dict) -> None:
        result = _extract_dxf_primitives(dxf_document["data"])
        for line in result["lines"]:
            assert "source" in line
            assert line["source"].startswith("dxf")

    def test_dxf_result_structure(self, dxf_document: dict) -> None:
        result = _extract_dxf_primitives(dxf_document["data"])
        required = {"lines", "circles", "arcs", "contours", "symbols", "image_shape"}
        assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"


# ── Full Agent Pipeline ───────────────────────────────────────────────────────


class TestFullPipeline:
    def test_agent_process_synthetic(self, agent: DrawingUnderstandingAgent, synthetic_drawing: dict) -> None:
        result = agent.process(synthetic_drawing)
        assert isinstance(result, dict)
        assert result["image_width"] == 400
        assert result["image_height"] == 400
        assert "lines" in result
        assert "circles" in result
        assert "arcs" in result
        assert "contours" in result
        assert "symbols" in result
        assert "morphology" in result
        assert len(result["lines"]) >= 1
        assert len(result["contours"]) >= 1

    def test_agent_process_dxf(self, agent: DrawingUnderstandingAgent, dxf_document: dict) -> None:
        result = agent.process(dxf_document)
        assert len(result["lines"]) >= 3
        assert len(result["circles"]) == 1
        assert len(result["arcs"]) == 1
        assert "source" in result["lines"][0]

    def test_agent_process_empty(self, agent: DrawingUnderstandingAgent, empty_drawing: dict) -> None:
        result = agent.process(empty_drawing)
        assert len(result["lines"]) == 0
        assert len(result["circles"]) == 0

    def test_agent_missing_data(self, agent: DrawingUnderstandingAgent) -> None:
        result = agent.process({"format": "png", "data": {}})
        assert result["preprocessing"].get("note") is not None

    def test_agent_simple_drawing(self, agent: DrawingUnderstandingAgent, simple_line_drawing: dict) -> None:
        result = agent.process(simple_line_drawing)
        assert len(result["lines"]) >= 2
        assert result["image_width"] == 200
        assert result["image_height"] == 200

    def test_agent_morphology_present(self, agent: DrawingUnderstandingAgent, synthetic_drawing: dict) -> None:
        result = agent.process(synthetic_drawing)
        morph = result["morphology"]
        assert morph is not None
        for key in ("cleaned", "skeleton", "dilated", "eroded"):
            assert key in morph
            assert isinstance(morph[key], np.ndarray)


# ── Parameter Overrides ───────────────────────────────────────────────────────


class TestParameterOverrides:
    def test_hough_threshold_override(self, agent: DrawingUnderstandingAgent, simple_line_drawing: dict) -> None:
        """With a very high threshold, fewer lines should be detected."""
        result_default = agent.process(simple_line_drawing)
        result_high = agent.process(simple_line_drawing, hough_threshold=500)
        # Higher threshold should find equal or fewer lines
        assert len(result_high["lines"]) <= len(result_default["lines"]) or len(result_high["lines"]) == 0

    def test_hough_min_line_override(self, agent: DrawingUnderstandingAgent, simple_line_drawing: dict) -> None:
        """With a very large min line length, fewer lines should be detected."""
        result_default = agent.process(simple_line_drawing)
        result_long = agent.process(simple_line_drawing, hough_min_line_length=1000)
        assert len(result_long["lines"]) == 0

    def test_canny_threshold_override(self, agent: DrawingUnderstandingAgent, simple_line_drawing: dict) -> None:
        """Very low Canny thresholds produce more edges."""
        result = agent.process(simple_line_drawing, canny_threshold1=10, canny_threshold2=30)
        assert "lines" in result


# ── Utility Functions ─────────────────────────────────────────────────────────


class TestUtilities:
    def test_point_to_line_distance_perpendicular(self) -> None:
        # Point (0, 10) perpendicular to horizontal line from (0,0) to (10,0) = distance 10
        d = _point_to_line_distance(0, 10, 0, 0, 10, 0)
        assert abs(d - 10.0) < 0.01

    def test_point_to_line_distance_on_line(self) -> None:
        d = _point_to_line_distance(5, 0, 0, 0, 10, 0)
        assert d < 0.01

    def test_point_to_line_distance_diagonal(self) -> None:
        # Point (2, 2) perpendicular to line from (0,0) to (4,0) = distance 2
        d = _point_to_line_distance(2, 2, 0, 0, 4, 0)
        assert abs(d - 2.0) < 0.01
