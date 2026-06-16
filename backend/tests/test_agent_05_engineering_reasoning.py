"""Unit tests for the Engineering Reasoning Agent (Agent 5).

Tests cover:
- Experta rule engine initialization and fallback
- All 10 engineering rules (hole, slot, cutout, pocket, vent, boss, fillet, chamfer)
- Hole + dimension/tolerance/fit pairing via proximity
- NetworkX feature tree construction
- CAD operations list generation
- Body dimension extraction from contours and dimensions
- Edge cases: empty inputs, missing annotations, unpaired features
"""

from __future__ import annotations

import pytest

from geometra.agents.agent_05_engineering_reasoning import (
    EngineeringReasoningAgent,
    _create_feature_tree,
    _add_feature_node,
    _add_operation,
    _point_distance,
    _find_nearby_dimension,
    _find_nearby_tolerance,
    _extract_body_dimensions,
)


# ── Agent Fixture ─────────────────────────────────────────────────────────────


@pytest.fixture
def agent() -> EngineeringReasoningAgent:
    return EngineeringReasoningAgent()


# ── Synthetic Inputs ──────────────────────────────────────────────────────────


@pytest.fixture
def hole_annotation_input() -> dict:
    """Inputs with a hole and matching diameter annotation."""
    return {
        "primitives": {
            "circles": [{"x": 100.0, "y": 120.0, "radius": 30.0}],
            "contours": [],
            "lines": [],
            "arcs": [],
            "symbols": [],
        },
        "annotations": {
            "dimensions": [
                {"value": 60, "unit": "mm", "type": "diameter", "position": {"x": 105.0, "y": 110.0},
                 "bbox": [], "confidence": 0.95, "text_raw": "⌀60", "tolerance": None},
            ],
            "tolerances": [
                {"text_raw": "H7", "fit": "H7", "type": "fit", "nominal_size": 60,
                 "position": {"x": 100.0, "y": 155.0}},
            ],
            "labels": [],
        },
        "features": {
            "features": [
                {"type": "hole", "x": 100.0, "y": 120.0, "radius": 30.0, "diameter": 60.0,
                 "confidence": 0.85, "source": "circle_detection",
                 "properties": {"circular": True, "has_center_mark": True}},
            ],
            "feature_count": 1,
            "class_counts": {"hole": 1, "slot": 0, "pocket": 0, "boss": 0,
                             "cutout": 0, "vent": 0, "fillet": 0, "chamfer": 0},
            "detection_method": "cv",
        },
    }


@pytest.fixture
def multi_feature_input() -> dict:
    """Inputs with multiple feature types."""
    return {
        "primitives": {
            "circles": [{"x": 100.0, "y": 100.0, "radius": 20.0}],
            "contours": [
                {"id": 0, "area": 50000, "perimeter": 900.0, "centroid": (250.0, 200.0),
                 "bounding_box": {"x": 30, "y": 30, "width": 440, "height": 340},
                 "approx_polygon_vertices": 4, "approx_polygon": [], "solidity": 0.98,
                 "hierarchy": {"next": -1, "prev": -1, "first_child": 1, "parent": -1,
                               "has_children": True, "is_child": False}},
            ],
            "lines": [],
            "arcs": [],
            "symbols": [],
        },
        "annotations": {
            "dimensions": [],
            "tolerances": [],
            "labels": [],
        },
        "features": {
            "features": [
                {"type": "hole", "x": 100.0, "y": 100.0, "radius": 20.0, "diameter": 40.0,
                 "confidence": 0.85, "source": "circle_detection",
                 "properties": {"circular": True, "has_center_mark": True}},
                {"type": "slot", "x": 200.0, "y": 150.0, "width": 80, "height": 20,
                 "area": 1200, "aspect_ratio": 0.25, "circularity": 0.6,
                 "confidence": 0.7, "source": "contour_analysis",
                 "properties": {"closed": True, "vertex_count": 6}},
                {"type": "cutout", "x": 250.0, "y": 140.0, "width": 200, "height": 120,
                 "area": 12000, "shape": "rectangular",
                 "confidence": 0.75, "source": "contour_hierarchy",
                 "properties": {"vertex_count": 4, "solidity": 0.97, "has_children": True, "child_count": 0}},
                {"type": "fillet", "x": 50.0, "y": 50.0, "radius": 5.0,
                 "angular_span_deg": 90.0, "arc_length": 7.85,
                 "confidence": 0.7, "source": "arc_detection",
                 "properties": {"fit_error": 0.1, "n_points": 20}},
                {"type": "chamfer", "x1": 100, "y1": 100, "x2": 115, "y2": 85,
                 "length": 21.21, "angle_deg": -45.0,
                 "confidence": 0.55, "source": "line_analysis",
                 "properties": {"distance_from_45deg": 0.0}},
            ],
            "feature_count": 5,
            "class_counts": {"hole": 1, "slot": 1, "pocket": 0, "boss": 0,
                             "cutout": 1, "vent": 0, "fillet": 1, "chamfer": 1},
            "detection_method": "cv",
        },
    }


@pytest.fixture
def empty_input() -> dict:
    """Completely empty input."""
    return {
        "primitives": {"circles": [], "contours": [], "lines": [], "arcs": [], "symbols": []},
        "annotations": {"dimensions": [], "tolerances": [], "labels": []},
        "features": {"features": [], "feature_count": 0,
                     "class_counts": {c: 0 for c in
                         ["hole", "slot", "pocket", "boss", "cutout", "vent", "fillet", "chamfer"]},
                     "detection_method": "cv"},
    }


# ── Agent: validate_input ─────────────────────────────────────────────────────


class TestValidateInput:
    def test_valid_input(self, agent: EngineeringReasoningAgent, hole_annotation_input: dict) -> None:
        assert agent.validate_input(hole_annotation_input) is True

    def test_invalid_none(self, agent: EngineeringReasoningAgent) -> None:
        assert agent.validate_input(None) is False

    def test_missing_keys(self, agent: EngineeringReasoningAgent) -> None:
        assert agent.validate_input({"primitives": {}}) is False


# ── Feature Tree Construction ─────────────────────────────────────────────────


class TestFeatureTree:
    def test_create_tree_has_root(self) -> None:
        G = _create_feature_tree()
        assert "body" in G.nodes
        assert G.nodes["body"]["type"] == "body"

    def test_add_feature_node(self) -> None:
        G = _create_feature_tree()
        _add_feature_node(G, "hole_1", "hole", x=100, y=120, radius=30)
        assert "hole_1" in G.nodes
        assert G.nodes["hole_1"]["type"] == "hole"
        assert G.has_edge("body", "hole_1")

    def test_add_operation(self) -> None:
        G = _create_feature_tree()
        _add_feature_node(G, "hole_1", "hole", x=100, y=120)
        _add_operation(G, "drill", "hole_1", radius=30)
        op_nodes = [n for n in G.nodes if n.startswith("drill_")]
        assert len(op_nodes) == 1
        assert G.has_edge("hole_1", op_nodes[0])
        assert G.nodes[op_nodes[0]]["operation"] == "drill"


# ── Proximity Utilities ──────────────────────────────────────────────────────


class TestProximity:
    def test_point_distance(self) -> None:
        assert _point_distance(0, 0, 3, 4) == 5.0

    def test_find_nearby_dimension(self) -> None:
        dims = [{"value": 60, "type": "diameter", "position": {"x": 105, "y": 110}}]
        result = _find_nearby_dimension(100, 120, dims, threshold=50)
        assert result is not None
        assert result["value"] == 60

    def test_find_nearby_dimension_too_far(self) -> None:
        dims = [{"value": 60, "type": "diameter", "position": {"x": 500, "y": 500}}]
        result = _find_nearby_dimension(100, 120, dims, threshold=50)
        assert result is None

    def test_find_nearby_tolerance(self) -> None:
        tols = [{"type": "fit", "fit": "H7", "position": {"x": 100, "y": 155}}]
        result = _find_nearby_tolerance(100, 120, tols, threshold=50)
        assert result is not None
        assert result["fit"] == "H7"


# ── Body Dimension Extraction ─────────────────────────────────────────────────


class TestBodyDimensions:
    def test_from_contours(self) -> None:
        contours = [
            {"id": 0, "area": 50000, "bounding_box": {"x": 30, "y": 30, "width": 440, "height": 340},
             "hierarchy": {"is_child": False}},
            {"id": 1, "area": 12000, "bounding_box": {"x": 150, "y": 80, "width": 200, "height": 120},
             "hierarchy": {"is_child": True}},
        ]
        w, h = _extract_body_dimensions(contours, [], [])
        assert w == 440.0
        assert h == 340.0

    def test_from_dimensions(self) -> None:
        dims = [{"value": 440, "type": "linear"}, {"value": 340, "type": "linear"}]
        w, h = _extract_body_dimensions([], dims, [])
        assert w == 440.0
        assert h == 340.0

    def test_empty(self) -> None:
        w, h = _extract_body_dimensions([], [], [])
        assert w is None
        assert h is None


# ── Full Agent Pipeline ───────────────────────────────────────────────────────


class TestFullPipeline:
    def test_agent_process_hole_with_diameter(self, agent: EngineeringReasoningAgent,
                                                hole_annotation_input: dict) -> None:
        result = agent.process(hole_annotation_input)
        assert isinstance(result, dict)
        assert "feature_tree" in result
        assert "operations" in result
        assert "parameters" in result
        assert "rules_fired" in result
        assert "reasoning_engine" in result

    def test_hole_diameter_rule_fired(self, agent: EngineeringReasoningAgent,
                                       hole_annotation_input: dict) -> None:
        result = agent.process(hole_annotation_input)
        rules = result["rules_fired"]
        hole_rules = [r for r in rules if r.get("rule") == "hole_with_diameter"]
        assert len(hole_rules) >= 1, f"No hole_with_diameter rule fired: {rules}"

    def test_hole_fit_rule_fired(self, agent: EngineeringReasoningAgent,
                                  hole_annotation_input: dict) -> None:
        result = agent.process(hole_annotation_input)
        rules = result["rules_fired"]
        fit_rules = [r for r in rules if r.get("rule") == "hole_with_fit"]
        assert len(fit_rules) >= 1, f"No hole_with_fit rule fired: {rules}"

    def test_hole_has_operations(self, agent: EngineeringReasoningAgent,
                                  hole_annotation_input: dict) -> None:
        result = agent.process(hole_annotation_input)
        ops = result["operations"]
        drill_ops = [o for o in ops if o["operation"] == "drill"]
        assert len(drill_ops) >= 1

    def test_hole_has_fit_param(self, agent: EngineeringReasoningAgent,
                                 hole_annotation_input: dict) -> None:
        result = agent.process(hole_annotation_input)
        params = result["parameters"]
        fit_params = [k for k in params if "fit" in k]
        assert len(fit_params) >= 1

    def test_multi_feature_rules(self, agent: EngineeringReasoningAgent,
                                  multi_feature_input: dict) -> None:
        result = agent.process(multi_feature_input)
        rules = result["rules_fired"]
        rule_types = {r.get("rule") for r in rules}
        # Should have rules for hole, slot, cutout, fillet, chamfer
        assert "slot_feature" in rule_types
        assert "cutout_feature" in rule_types
        assert "fillet_feature" in rule_types
        assert "chamfer_feature" in rule_types

    def test_multi_feature_operations(self, agent: EngineeringReasoningAgent,
                                       multi_feature_input: dict) -> None:
        result = agent.process(multi_feature_input)
        ops = result["operations"]
        op_types = {o["operation"] for o in ops}
        assert "drill" in op_types
        assert "mill" in op_types
        assert "cut" in op_types
        assert "fillet" in op_types
        assert "chamfer" in op_types

    def test_empty_input(self, agent: EngineeringReasoningAgent, empty_input: dict) -> None:
        result = agent.process(empty_input)
        assert len(result["rules_fired"]) == 0
        assert len(result["operations"]) == 0
        assert len(result["feature_tree"]["nodes"]) > 0  # tree always has root node

    def test_feature_tree_structure(self, agent: EngineeringReasoningAgent,
                                     hole_annotation_input: dict) -> None:
        result = agent.process(hole_annotation_input)
        tree = result["feature_tree"]
        assert "nodes" in tree
        # NetworkX node_link_data uses 'edges' as the key (not 'links')
        assert "edges" in tree
        assert tree["directed"] is True

    def test_operations_ordered(self, agent: EngineeringReasoningAgent,
                                 multi_feature_input: dict) -> None:
        result = agent.process(multi_feature_input)
        ops = result["operations"]
        assert len(ops) >= 5  # drill, mill, cut, fillet, chamfer
        # Operations should maintain insertion order
        assert ops[0]["operation"] == "drill"  # holes first

    def test_reasoning_engine_string(self, agent: EngineeringReasoningAgent,
                                      hole_annotation_input: dict) -> None:
        result = agent.process(hole_annotation_input)
        engine = result["reasoning_engine"]
        assert engine in ("experta", "rule_set")

    def test_body_dimensions_in_params(self, agent: EngineeringReasoningAgent,
                                        multi_feature_input: dict) -> None:
        result = agent.process(multi_feature_input)
        params = result["parameters"]
        # The multi_feature_input has a large contour (50000 area, 440x340)
        assert params.get("body_width") is not None or params.get("body_height") is not None

    def test_process_no_rules_errors(self, agent: EngineeringReasoningAgent,
                                      empty_input: dict) -> None:
        """Processing empty input should not raise errors."""
        result = agent.process(empty_input)
        assert isinstance(result, dict)

    def test_feature_id_format(self, agent: EngineeringReasoningAgent,
                                hole_annotation_input: dict) -> None:
        result = agent.process(hole_annotation_input)
        for rule in result["rules_fired"]:
            fid = rule.get("feature_id", "")
            assert fid, f"Missing feature_id in rule: {rule}"

    def test_hole_diameter_value(self, agent: EngineeringReasoningAgent,
                                  hole_annotation_input: dict) -> None:
        result = agent.process(hole_annotation_input)
        rules = result["rules_fired"]
        for r in rules:
            if r.get("rule") == "hole_with_diameter":
                assert r.get("diameter") == 60
                break

    def test_agent_all_feature_types(self, agent: EngineeringReasoningAgent) -> None:
        """All 8 feature types should produce rules."""
        from geometra.agents.agent_04_feature_recognition import FEATURE_CLASSES

        feat_list = []
        for cls in FEATURE_CLASSES:
            feat: dict = {"type": cls}
            if cls == "hole":
                feat.update({"x": 100, "y": 100, "radius": 20, "diameter": 40})
            elif cls == "slot":
                feat.update({"x": 200, "y": 150, "width": 80, "height": 20, "area": 1200})
            elif cls == "cutout":
                feat.update({"x": 250, "y": 140, "width": 200, "height": 120, "shape": "rectangular", "area": 12000})
            elif cls == "pocket":
                feat.update({"x": 180, "y": 180, "width": 60, "height": 60, "shape": "rectangular"})
            elif cls == "vent":
                feat.update({"x": 150, "y": 250, "width": 100, "height": 80, "slot_count": 5,
                             "properties": {"orientation": "horizontal"}})
            elif cls == "boss":
                feat.update({"x": 300, "y": 80, "area": 2000,
                             "properties": {"approx_radius": 25}})
            elif cls == "fillet":
                feat.update({"x": 50, "y": 50, "radius": 5, "angular_span_deg": 90, "arc_length": 7.85})
            elif cls == "chamfer":
                feat.update({"x1": 100, "y1": 100, "x2": 115, "y2": 85, "length": 21.21, "angle_deg": -45.0})
            feat_list.append(feat)

        inp = {
            "primitives": {"circles": [], "contours": [], "lines": [], "arcs": [], "symbols": []},
            "annotations": {"dimensions": [], "tolerances": [], "labels": []},
            "features": {"features": feat_list, "feature_count": len(feat_list),
                         "class_counts": {c: 1 if c != "vent" else 1 for c in
                             ["hole", "slot", "pocket", "boss", "cutout", "vent", "fillet", "chamfer"]},
                         "detection_method": "cv"},
        }
        result = agent.process(inp)
        rule_types = {r.get("type") for r in result["rules_fired"]}
        for cls in FEATURE_CLASSES:
            assert cls in rule_types, f"Missing rule for feature type: {cls}"

    def test_operations_contain_feature_ids(self, agent: EngineeringReasoningAgent,
                                             multi_feature_input: dict) -> None:
        result = agent.process(multi_feature_input)
        for op in result["operations"]:
            assert "feature_id" in op
            assert op["feature_id"]
