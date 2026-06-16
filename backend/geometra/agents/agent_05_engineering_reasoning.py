"""Agent 5: Engineering Reasoning Agent.

Responsibilities:
Convert geometric primitives + annotations + detected features into a
parametric feature tree using a rule engine (Experta) and graph (NetworkX).

Rules implemented:
  1. Circle + Diameter annotation → Hole with size and location
  2. Hole + fit annotation (H7) → Hole with tolerance class
  3. Hole + tolerance (±0.1) → Hole with tolerance
  4. Slot contour + dimensions → Slot feature
  5. Cutout contour → Cutout / pocket with shape
  6. Vents grouped → Vent group feature
  7. Boss contour → Boss feature
  8. Fillet arc → Corner fillet treatment
  9. Chamfer line → Edge chamfer treatment
  10. Enclosure rectangle + dimensions → Base body

Libraries: Experta (rule engine), NetworkX (graph processing)

Pipeline:
  Agent 2 (primitives) ──┐
                          ├──→ Agent 5 → Parametric Feature Tree
  Agent 3 (annotations) ──┤              → CAD operations list
  Agent 4 (features)    ──┘              → Parameters dict
"""

from __future__ import annotations

import logging
import math
from typing import Any

import networkx as nx

from geometra.agents.base import BaseAgent

logger = logging.getLogger(__name__)

# ── Experta imports (lazy, with graceful fallback) ────────────────────────────

# Compatibility patch for frozendict (experta dependency) on Python 3.10+
# frozendict uses collections.Mapping which was removed in Python 3.10.
# This patch must run before frozendict is imported anywhere in the process.
import collections
import collections.abc
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping

# Additionally patch frozendict's internal import if already loaded
import sys
try:
    if "frozendict" in sys.modules:
        import frozendict
        if not hasattr(frozendict, "Mapping"):
            frozendict.Mapping = collections.abc.Mapping
except ImportError:
    pass
except AttributeError:
    pass

try:
    from experta import (
        AND,
        OR,
        P,
        KnowledgeEngine,
        Rule,
        Fact as ExpFact,
        MATCH,
    )

    EXPERTA_AVAILABLE = True
except ImportError:
    EXPERTA_AVAILABLE = False
    logger.warning("Experta not installed; rule engine disabled. Install with: pip install experta")


# ── Proximity matching constants ──────────────────────────────────────────────

# Maximum pixel distance for matching a dimension/annotation to a feature
PROXIMITY_THRESHOLD = 50

# Aspect ratio threshold for distinguishing slots vs rectangles
ASPECT_RATIO_THRESHOLD = 0.4

# Minimum contour area for a valid feature feature candidate
MIN_FEATURE_AREA = 100


# ── NetworkX Feature Tree Helpers ────────────────────────────────────────────


def _create_feature_tree() -> nx.DiGraph:
    """Create a new directed graph for the parametric feature tree.

    The tree structure:
        Root ("body") → Groups → Features → Parameters

    Returns:
        Empty directed graph with a root node.
    """
    G = nx.DiGraph()
    G.add_node("body", type="body", label="Main Body")
    return G


def _add_feature_node(
    G: nx.DiGraph,
    feature_id: str,
    feature_type: str,
    parent: str = "body",
    **params: Any,
) -> None:
    """Add a feature node to the tree with parameters."""
    G.add_node(
        feature_id,
        type=feature_type,
        label=feature_type.replace("_", " ").title(),
        **params,
    )
    G.add_edge(parent, feature_id)


def _add_operation(
    G: nx.DiGraph,
    operation: str,
    target_feature: str,
    **params: Any,
) -> None:
    """Add a CAD operation node attached to a feature."""
    op_id = f"{operation}_{target_feature}"
    G.add_node(
        op_id,
        type="operation",
        operation=operation,
        target=target_feature,
        **params,
    )
    G.add_edge(target_feature, op_id)


# ── Proximity utilities ──────────────────────────────────────────────────────


def _point_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _find_nearby_dimension(
    x: float,
    y: float,
    dimensions: list[dict[str, Any]],
    threshold: float = PROXIMITY_THRESHOLD,
) -> dict[str, Any] | None:
    """Find the nearest dimension annotation within threshold pixels."""
    best: dict[str, Any] | None = None
    best_dist = threshold
    for dim in dimensions:
        pos = dim.get("position", {})
        dx = pos.get("x", 0)
        dy = pos.get("y", 0)
        dist = _point_distance(x, y, dx, dy)
        if dist < best_dist:
            best_dist = dist
            best = dim
    return best


def _find_nearby_tolerance(
    x: float,
    y: float,
    tolerances: list[dict[str, Any]],
    threshold: float = PROXIMITY_THRESHOLD,
) -> dict[str, Any] | None:
    """Find the nearest tolerance annotation within threshold pixels."""
    best: dict[str, Any] | None = None
    best_dist = threshold
    for tol in tolerances:
        pos = tol.get("position", {})
        dx = pos.get("x", 0)
        dy = pos.get("y", 0)
        dist = _point_distance(x, y, dx, dy)
        if dist < best_dist:
            best_dist = dist
            best = tol
    return best


def _find_nearby_label(
    x: float,
    y: float,
    labels: list[dict[str, Any]],
    threshold: float = PROXIMITY_THRESHOLD,
) -> dict[str, Any] | None:
    """Find the nearest label within threshold pixels."""
    best: dict[str, Any] | None = None
    best_dist = threshold
    for label in labels:
        pos = label.get("position", {})
        dx = pos.get("x", 0)
        dy = pos.get("y", 0)
        dist = _point_distance(x, y, dx, dy)
        if dist < best_dist:
            best_dist = dist
            best = label
    return best


# ── Experta Knowledge Engine ────────────────────────────────────────────────


class EngineeringReasoningEngine(KnowledgeEngine):
    """Experta-based rule engine for engineering reasoning.

    Accepts facts representing:
      - Features (holes, slots, cutouts, vents, bosses, fillets, chamfers)
      - Dimensions (linear, diameter, radius)
      - Tolerances (fits, ± values)

    Each rule fires when the conditions are satisfied and produces
    a structured reasoning result in self.results.
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, Any]] = []

    # ── Rule 1: Hole + Diameter dimension → Sized hole ──
    @Rule(
        AND(
            ExpFact(type="hole"),
            ExpFact(type="diameter", value=P(lambda v: v is not None)),
        )
    )
    def hole_with_diameter(self) -> None:
        hole_facts = [f for f in self.facts.values() if isinstance(f, dict) and f.get("type") == "hole"]
        dim_facts = [f for f in self.facts.values() if isinstance(f, dict) and f.get("type") == "diameter"]
        for hf in hole_facts:
            hx = hf.get("x", 0)
            hy = hf.get("y", 0)
            for df in dim_facts:
                dx = df.get("position", {}).get("x", 0)
                dy = df.get("position", {}).get("y", 0)
                if _point_distance(hx, hy, dx, dy) < PROXIMITY_THRESHOLD:
                    self.results.append({
                        "rule": "hole_with_diameter",
                        "feature_id": f"hole_{hx:.0f}_{hy:.0f}",
                        "type": "hole",
                        "x": round(hx, 2),
                        "y": round(hy, 2),
                        "radius": hf.get("radius", 0),
                        "diameter": df.get("value", hf.get("diameter", 0)),
                        "source": "rule_hole_diameter",
                    })
                    break

    # ── Rule 2: Hole + fit tolerance → Hole with fit class ──
    @Rule(
        AND(
            ExpFact(type="hole"),
            ExpFact(type="fit", fit=P(lambda f: f is not None)),
        )
    )
    def hole_with_fit(self) -> None:
        hole_facts = [f for f in self.facts.values() if isinstance(f, dict) and f.get("type") == "hole"]
        fit_facts = [f for f in self.facts.values() if isinstance(f, dict) and f.get("type") == "fit"]
        for hf in hole_facts:
            hx = hf.get("x", 0)
            hy = hf.get("y", 0)
            for ff in fit_facts:
                fx = ff.get("position", {}).get("x", 0)
                fy = ff.get("position", {}).get("y", 0)
                if _point_distance(hx, hy, fx, fy) < PROXIMITY_THRESHOLD:
                    self.results.append({
                        "rule": "hole_with_fit",
                        "feature_id": f"hole_{hx:.0f}_{hy:.0f}",
                        "type": "hole",
                        "x": round(hx, 2),
                        "y": round(hy, 2),
                        "fit": ff.get("fit"),
                        "nominal_size": ff.get("nominal_size"),
                        "source": "rule_hole_fit",
                    })
                    break

    # ── Rule 3: Hole + tolerance (±) → Toleranced hole ──
    @Rule(
        AND(
            ExpFact(type="hole"),
            ExpFact(type="tolerance"),
        )
    )
    def hole_with_tolerance(self) -> None:
        hole_facts = [f for f in self.facts.values() if isinstance(f, dict) and f.get("type") == "hole"]
        tol_facts = [f for f in self.facts.values() if isinstance(f, dict) and f.get("type") == "tolerance"]
        for hf in hole_facts:
            hx = hf.get("x", 0)
            hy = hf.get("y", 0)
            for tf in tol_facts:
                tx = tf.get("position", {}).get("x", 0)
                ty = tf.get("position", {}).get("y", 0)
                if _point_distance(hx, hy, tx, ty) < PROXIMITY_THRESHOLD:
                    self.results.append({
                        "rule": "hole_with_tolerance",
                        "feature_id": f"hole_{hx:.0f}_{hy:.0f}",
                        "type": "hole",
                        "x": round(hx, 2),
                        "y": round(hy, 2),
                        "tolerance_value": tf.get("value"),
                        "tolerance_symbol": tf.get("symbol", "±"),
                        "source": "rule_hole_tolerance",
                    })
                    break

    # ── Rule 4: Slot → Slotted opening ──
    @Rule(ExpFact(type="slot"))
    def slot_feature(self) -> None:
        slot_facts = [f for f in self.facts.values() if isinstance(f, dict) and f.get("type") == "slot"]
        for sf in slot_facts:
            self.results.append({
                "rule": "slot_feature",
                "feature_id": f"slot_{sf.get('x', 0):.0f}_{sf.get('y', 0):.0f}",
                "type": "slot",
                "x": sf.get("x", 0),
                "y": sf.get("y", 0),
                "width": sf.get("width", 0),
                "height": sf.get("height", 0),
                "area": sf.get("area", 0),
                "source": "rule_slot",
            })

    # ── Rule 5: Cutout → Cutout opening ──
    @Rule(ExpFact(type="cutout"))
    def cutout_feature(self) -> None:
        cutout_facts = [f for f in self.facts.values() if isinstance(f, dict) and f.get("type") == "cutout"]
        for cf in cutout_facts:
            self.results.append({
                "rule": "cutout_feature",
                "feature_id": f"cutout_{cf.get('x', 0):.0f}_{cf.get('y', 0):.0f}",
                "type": "cutout",
                "x": cf.get("x", 0),
                "y": cf.get("y", 0),
                "width": cf.get("width", 0),
                "height": cf.get("height", 0),
                "shape": cf.get("shape", "rectangular"),
                "area": cf.get("area", 0),
                "source": "rule_cutout",
            })

    # ── Rule 6: Pocket → Pocket cavity ──
    @Rule(ExpFact(type="pocket"))
    def pocket_feature(self) -> None:
        pocket_facts = [f for f in self.facts.values() if isinstance(f, dict) and f.get("type") == "pocket"]
        for pf in pocket_facts:
            self.results.append({
                "rule": "pocket_feature",
                "feature_id": f"pocket_{pf.get('x', 0):.0f}_{pf.get('y', 0):.0f}",
                "type": "pocket",
                "x": pf.get("x", 0),
                "y": pf.get("y", 0),
                "width": pf.get("width", 0),
                "height": pf.get("height", 0),
                "shape": pf.get("shape", "rectangular"),
                "source": "rule_pocket",
            })

    # ── Rule 7: Vent → Ventilation group ──
    @Rule(ExpFact(type="vent"))
    def vent_feature(self) -> None:
        vent_facts = [f for f in self.facts.values() if isinstance(f, dict) and f.get("type") == "vent"]
        for vf in vent_facts:
            self.results.append({
                "rule": "vent_feature",
                "feature_id": f"vent_{vf.get('x', 0):.0f}_{vf.get('y', 0):.0f}",
                "type": "vent",
                "x": vf.get("x", 0),
                "y": vf.get("y", 0),
                "width": vf.get("width", 0),
                "height": vf.get("height", 0),
                "slot_count": vf.get("slot_count", 0),
                "orientation": vf.get("properties", {}).get("orientation", "horizontal"),
                "source": "rule_vent",
            })

    # ── Rule 8: Boss → Raised boss ──
    @Rule(ExpFact(type="boss"))
    def boss_feature(self) -> None:
        boss_facts = [f for f in self.facts.values() if isinstance(f, dict) and f.get("type") == "boss"]
        for bf in boss_facts:
            self.results.append({
                "rule": "boss_feature",
                "feature_id": f"boss_{bf.get('x', 0):.0f}_{bf.get('y', 0):.0f}",
                "type": "boss",
                "x": bf.get("x", 0),
                "y": bf.get("y", 0),
                "radius": bf.get("properties", {}).get("approx_radius", 0),
                "area": bf.get("area", 0),
                "source": "rule_boss",
            })

    # ── Rule 9: Fillet → Corner fillet ──
    @Rule(ExpFact(type="fillet"))
    def fillet_feature(self) -> None:
        fillet_facts = [f for f in self.facts.values() if isinstance(f, dict) and f.get("type") == "fillet"]
        for ff in fillet_facts:
            self.results.append({
                "rule": "fillet_feature",
                "feature_id": f"fillet_{ff.get('x', 0):.0f}_{ff.get('y', 0):.0f}",
                "type": "fillet",
                "x": ff.get("x", 0),
                "y": ff.get("y", 0),
                "radius": ff.get("radius", 0),
                "angular_span_deg": ff.get("angular_span_deg", 0),
                "source": "rule_fillet",
            })

    # ── Rule 10: Chamfer → Edge chamfer ──
    @Rule(ExpFact(type="chamfer"))
    def chamfer_feature(self) -> None:
        chamfer_facts = [f for f in self.facts.values() if isinstance(f, dict) and f.get("type") == "chamfer"]
        for cf in chamfer_facts:
            self.results.append({
                "rule": "chamfer_feature",
                "feature_id": f"chamfer_{cf.get('x1', 0):.0f}_{cf.get('y1', 0):.0f}",
                "type": "chamfer",
                "x1": cf.get("x1", 0),
                "y1": cf.get("y1", 0),
                "x2": cf.get("x2", 0),
                "y2": cf.get("y2", 0),
                "length": cf.get("length", 0),
                "angle_deg": cf.get("angle_deg", 0),
                "source": "rule_chamfer",
            })


# ── Main Agent ────────────────────────────────────────────────────────────────


class EngineeringReasoningAgent(BaseAgent):
    """Converts geometric primitives + annotations + detected features into a
    parametric feature tree using rule-based reasoning (Experta) and graph
    processing (NetworkX).

    The agent:
      1. Takes in primitives (Agent 2), annotations (Agent 3), and features (Agent 4)
      2. Feeds them into the Experta rule engine
      3. Rules combine geometry + dimensions + tolerances → engineering features
      4. Outputs a NetworkX feature tree + operations list + parameters dict
         for CAD generation (Agent 6)
    """

    def __init__(self) -> None:
        super().__init__()
        self._engine = EngineeringReasoningEngine() if EXPERTA_AVAILABLE else None

    @property
    def experta_available(self) -> bool:
        return EXPERTA_AVAILABLE and self._engine is not None

    def validate_input(self, input_data: Any) -> bool:
        """Validate that input has required keys."""
        if not isinstance(input_data, dict):
            return False
        required = {"primitives", "annotations", "features"}
        return all(k in input_data for k in required)

    def process(self, input_data: Any, **kwargs: Any) -> dict[str, Any]:
        """Execute reasoning. Accepts either a dict with 'primitives',
        'annotations', 'features' keys, or each as a separate kwarg.

        Args:
            input_data: Dict with primitives, annotations, features keys.
            **kwargs: Individual overrides for primitives, annotations, features.

        Returns:
            Parametric feature tree with:
                - feature_tree: NetworkX DiGraph serialized as adjacency dict
                - operations: ordered list of CAD build operations
                - parameters: global design parameters
                - rules_fired: list of reasoning results from the rule engine
                - reasoning_engine: which engine was used ('experta' | 'rule_set')
        """
        primitives = kwargs.get("primitives", input_data.get("primitives", {}))
        annotations = kwargs.get("annotations", input_data.get("annotations", {}))
        features = kwargs.get("features", input_data.get("features", {}))

        return self.reason(
            primitives=primitives,
            annotations=annotations,
            features=features,
        )

    def reason(
        self,
        primitives: dict[str, Any],
        annotations: dict[str, Any],
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply engineering rules to convert geometry + annotations + features
        into a parametric feature tree.

        Args:
            primitives: Output from DrawingUnderstandingAgent (Agent 2).
            annotations: Output from OCRAnnotationAgent (Agent 3).
            features: Output from FeatureRecognitionAgent (Agent 4).

        Returns:
            Parametric feature tree dict with:
                - feature_tree: NetworkX graph as adjacency list
                - operations: ordered list of CAD operations
                - parameters: global parameters dict
                - rules_fired: list of rule results
        """
        self.logger.info("Applying engineering reasoning")

        # Extract data from inputs
        dims = annotations.get("dimensions", [])
        tols = annotations.get("tolerances", [])
        labels = annotations.get("labels", [])
        feats = features.get("features", [])
        primitive_circles = primitives.get("circles", [])
        primitive_contours = primitives.get("contours", [])

        # ── Phase 1: Run the Expert System ──
        rules_fired: list[dict[str, Any]] = []

        if self.experta_available:
            rules_fired = self._run_experta_engine(
                features=feats,
                dimensions=dims,
                tolerances=tols,
            )
            engine_name = "experta"
        else:
            rules_fired = self._run_fallback_rules(
                features=feats,
                dimensions=dims,
                tolerances=tols,
                primitives=primitives,
            )
            engine_name = "rule_set"

        # ── Phase 2: Build the Parametric Feature Tree ──
        G = _create_feature_tree()
        ops: list[dict[str, Any]] = []
        params: dict[str, Any] = {}

        # Sort rules_fired by feature type priority for deterministic output
        # regardless of which rule engine was used
        feature_priority = {
            "hole": 0,
            "slot": 1,
            "cutout": 2,
            "pocket": 3,
            "vent": 4,
            "boss": 5,
            "fillet": 6,
            "chamfer": 7,
        }
        rules_fired.sort(key=lambda r: feature_priority.get(r.get("type", ""), 99))

        # Process each reasoning result into the feature tree
        for result in rules_fired:
            ftype = result.get("type", "")
            fid = result.get("feature_id", "")

            # Remove meta keys to avoid **kwargs conflicts with _add_feature_node
            result_params = {k: v for k, v in result.items() if k not in ("rule", "source", "feature_id", "type")}

            if ftype in ("hole", "slot", "cutout", "pocket", "vent", "boss", "fillet", "chamfer"):
                _add_feature_node(G, fid, ftype, **result_params)

                if ftype == "hole":
                    radius = result.get("radius", result.get("diameter", 0) / 2)
                    _add_operation(G, "drill", fid, depth="through", radius=radius)
                    ops.append({
                        "operation": "drill", "feature_id": fid,
                        "x": result.get("x", 0), "y": result.get("y", 0),
                        "radius": radius, "depth": "through",
                    })
                    if result.get("fit"):
                        params[f"{fid}_fit"] = result["fit"]
                    if result.get("tolerance_value"):
                        params[f"{fid}_tolerance"] = result["tolerance_value"]

                elif ftype == "cutout":
                    _add_operation(G, "cut", fid, shape=result.get("shape", "rectangular"))
                    ops.append({
                        "operation": "cut", "feature_id": fid,
                        "x": result.get("x", 0), "y": result.get("y", 0),
                        "width": result.get("width", 0), "height": result.get("height", 0),
                    })

                elif ftype == "slot":
                    _add_operation(G, "mill", fid)
                    ops.append({
                        "operation": "mill", "feature_id": fid,
                        "x": result.get("x", 0), "y": result.get("y", 0),
                        "width": result.get("width", 0), "height": result.get("height", 0),
                    })

                elif ftype == "pocket":
                    _add_operation(G, "pocket_mill", fid)
                    ops.append({
                        "operation": "pocket_mill", "feature_id": fid,
                        "x": result.get("x", 0), "y": result.get("y", 0),
                        "width": result.get("width", 0), "height": result.get("height", 0),
                    })

                elif ftype == "vent":
                    _add_operation(G, "vent_cut", fid, orientation=result.get("orientation", "horizontal"))
                    ops.append({
                        "operation": "vent_cut", "feature_id": fid,
                        "x": result.get("x", 0), "y": result.get("y", 0),
                        "width": result.get("width", 0), "height": result.get("height", 0),
                        "slot_count": result.get("slot_count", 0),
                    })

                elif ftype == "boss":
                    _add_operation(G, "boss_extrude", fid)
                    ops.append({
                        "operation": "boss_extrude", "feature_id": fid,
                        "x": result.get("x", 0), "y": result.get("y", 0),
                        "radius": result.get("radius", 0),
                    })

                elif ftype == "fillet":
                    _add_operation(G, "fillet", fid, radius=result.get("radius", 0))
                    ops.append({
                        "operation": "fillet", "feature_id": fid,
                        "x": result.get("x", 0), "y": result.get("y", 0),
                        "radius": result.get("radius", 0),
                    })

                elif ftype == "chamfer":
                    _add_operation(G, "chamfer", fid)
                    ops.append({
                        "operation": "chamfer", "feature_id": fid,
                        "x1": result.get("x1", 0), "y1": result.get("y1", 0),
                        "x2": result.get("x2", 0), "y2": result.get("y2", 0),
                    })

        # Extract body dimensions from primitives if available
        body_width, body_height = _extract_body_dimensions(primitive_contours, dims, primitive_circles)
        if body_width and body_height:
            params["body_width"] = body_width
            params["body_height"] = body_height
            G.nodes["body"]["width"] = body_width
            G.nodes["body"]["height"] = body_height

        # Serialize the graph for JSON output
        feature_tree_serialized = nx.node_link_data(G)

        self.logger.info(
            "Reasoning complete: %d rules fired, %d operations generated, %d parameters",
            len(rules_fired),
            len(ops),
            len(params),
        )

        return {
            "feature_tree": feature_tree_serialized,
            "operations": ops,
            "parameters": params,
            "rules_fired": rules_fired,
            "reasoning_engine": engine_name,
        }

    def _run_experta_engine(
        self,
        features: list[dict[str, Any]],
        dimensions: list[dict[str, Any]],
        tolerances: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Run the Experta rule engine with feature + dimension + tolerance facts."""
        if self._engine is None:
            return self._run_fallback_rules(
                features=features, dimensions=dimensions,
                tolerances=tolerances, primitives={},
            )

        self._engine.reset()
        self._engine.results = []

        # Declare feature facts
        for feat in features:
            ftype = feat.get("type", "")
            self._engine.declare(ExpFact(**feat))

        # Declare dimension facts
        for dim in dimensions:
            self._engine.declare(ExpFact(**dim))

        # Declare tolerance facts
        for tol in tolerances:
            self._engine.declare(ExpFact(**tol))

        # Run the engine
        self._engine.run()

        results = list(self._engine.results)

        # Post-process: add any unannotated holes that weren't matched by rules
        matched_hole_ids = set()
        for r in results:
            if r.get("type") == "hole":
                fid = r.get("feature_id", "")
                if fid:
                    matched_hole_ids.add(fid)

        for feat in features:
            if feat.get("type") != "hole":
                continue
            fx = feat.get("x", 0)
            fy = feat.get("y", 0)
            fid = f"hole_{fx:.0f}_{fy:.0f}"
            if fid not in matched_hole_ids:
                results.append({
                    "rule": "hole_with_diameter",
                    "feature_id": fid,
                    "type": "hole",
                    "x": round(fx, 2),
                    "y": round(fy, 2),
                    "radius": feat.get("radius", 0),
                    "diameter": feat.get("diameter", 0),
                    "source": "rule_hole_unannotated",
                })

        return results

    def _run_fallback_rules(
        self,
        features: list[dict[str, Any]],
        dimensions: list[dict[str, Any]],
        tolerances: list[dict[str, Any]],
        primitives: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Fallback: direct rule application without Experta.

        Apply the same rules imperatively when Experta is not available.
        """
        results: list[dict[str, Any]] = []
        used_features: set[int] = set()

        # Rule 1: Hole + nearby diameter dimension → Sized hole
        for i, feat in enumerate(features):
            if feat.get("type") != "hole":
                continue
            fx = feat.get("x", 0)
            fy = feat.get("y", 0)
            near_dim = _find_nearby_dimension(fx, fy, [d for d in dimensions if d.get("type") == "diameter"])
            if near_dim:
                results.append({
                    "rule": "hole_with_diameter",
                    "feature_id": f"hole_{fx:.0f}_{fy:.0f}",
                    "type": "hole",
                    "x": round(fx, 2),
                    "y": round(fy, 2),
                    "radius": feat.get("radius", feat.get("diameter", 0) / 2),
                    "diameter": near_dim.get("value", feat.get("diameter", 0)),
                    "source": "fallback_hole_diameter",
                })
                used_features.add(i)

        # Rule 2: Hole + fit tolerance
        for i, feat in enumerate(features):
            if feat.get("type") != "hole":
                continue
            if i in used_features:
                continue
            fx = feat.get("x", 0)
            fy = feat.get("y", 0)
            near_fit = _find_nearby_tolerance(fx, fy, [t for t in tolerances if t.get("type") == "fit"])
            if near_fit:
                results.append({
                    "rule": "hole_with_fit",
                    "feature_id": f"hole_{fx:.0f}_{fy:.0f}",
                    "type": "hole",
                    "x": round(fx, 2),
                    "y": round(fy, 2),
                    "fit": near_fit.get("fit"),
                    "nominal_size": near_fit.get("nominal_size"),
                    "source": "fallback_hole_fit",
                })
                used_features.add(i)

        # Rule 3: Hole + tolerance value
        for i, feat in enumerate(features):
            if feat.get("type") != "hole":
                continue
            if i in used_features:
                continue
            fx = feat.get("x", 0)
            fy = feat.get("y", 0)
            near_tol = _find_nearby_tolerance(fx, fy, [t for t in tolerances if t.get("type") == "tolerance"])
            if near_tol:
                results.append({
                    "rule": "hole_with_tolerance",
                    "feature_id": f"hole_{fx:.0f}_{fy:.0f}",
                    "type": "hole",
                    "x": round(fx, 2),
                    "y": round(fy, 2),
                    "tolerance_value": near_tol.get("value"),
                    "tolerance_symbol": near_tol.get("symbol", "±"),
                    "source": "fallback_hole_tolerance",
                })
                used_features.add(i)

        # Remaining holes without annotations
        for i, feat in enumerate(features):
            if feat.get("type") != "hole":
                continue
            if i in used_features:
                continue
            fx = feat.get("x", 0)
            fy = feat.get("y", 0)
            results.append({
                "rule": "hole_with_diameter",
                "feature_id": f"hole_{fx:.0f}_{fy:.0f}",
                "type": "hole",
                "x": round(fx, 2),
                "y": round(fy, 2),
                "radius": feat.get("radius", 0),
                "diameter": feat.get("diameter", 0),
                "source": "fallback_hole_unannotated",
            })

        # Rule 4-10: Direct feature passthrough
        for feat in features:
            ftype = feat.get("type", "")
            if ftype == "slot":
                results.append({
                    "rule": "slot_feature",
                    "feature_id": f"slot_{feat.get('x', 0):.0f}_{feat.get('y', 0):.0f}",
                    "type": "slot",
                    "x": feat.get("x", 0),
                    "y": feat.get("y", 0),
                    "width": feat.get("width", 0),
                    "height": feat.get("height", 0),
                    "area": feat.get("area", 0),
                    "source": "fallback_slot",
                })
            elif ftype == "cutout":
                results.append({
                    "rule": "cutout_feature",
                    "feature_id": f"cutout_{feat.get('x', 0):.0f}_{feat.get('y', 0):.0f}",
                    "type": "cutout",
                    "x": feat.get("x", 0),
                    "y": feat.get("y", 0),
                    "width": feat.get("width", 0),
                    "height": feat.get("height", 0),
                    "shape": feat.get("shape", "rectangular"),
                    "area": feat.get("area", 0),
                    "source": "fallback_cutout",
                })
            elif ftype == "pocket":
                results.append({
                    "rule": "pocket_feature",
                    "feature_id": f"pocket_{feat.get('x', 0):.0f}_{feat.get('y', 0):.0f}",
                    "type": "pocket",
                    "x": feat.get("x", 0),
                    "y": feat.get("y", 0),
                    "width": feat.get("width", 0),
                    "height": feat.get("height", 0),
                    "shape": feat.get("shape", "rectangular"),
                    "source": "fallback_pocket",
                })
            elif ftype == "vent":
                results.append({
                    "rule": "vent_feature",
                    "feature_id": f"vent_{feat.get('x', 0):.0f}_{feat.get('y', 0):.0f}",
                    "type": "vent",
                    "x": feat.get("x", 0),
                    "y": feat.get("y", 0),
                    "width": feat.get("width", 0),
                    "height": feat.get("height", 0),
                    "slot_count": feat.get("slot_count", 0),
                    "orientation": feat.get("properties", {}).get("orientation", "horizontal"),
                    "source": "fallback_vent",
                })
            elif ftype == "boss":
                results.append({
                    "rule": "boss_feature",
                    "feature_id": f"boss_{feat.get('x', 0):.0f}_{feat.get('y', 0):.0f}",
                    "type": "boss",
                    "x": feat.get("x", 0),
                    "y": feat.get("y", 0),
                    "radius": feat.get("properties", {}).get("approx_radius", 0),
                    "area": feat.get("area", 0),
                    "source": "fallback_boss",
                })
            elif ftype == "fillet":
                results.append({
                    "rule": "fillet_feature",
                    "feature_id": f"fillet_{feat.get('x', 0):.0f}_{feat.get('y', 0):.0f}",
                    "type": "fillet",
                    "x": feat.get("x", 0),
                    "y": feat.get("y", 0),
                    "radius": feat.get("radius", 0),
                    "angular_span_deg": feat.get("angular_span_deg", 0),
                    "source": "fallback_fillet",
                })
            elif ftype == "chamfer":
                results.append({
                    "rule": "chamfer_feature",
                    "feature_id": f"chamfer_{feat.get('x1', 0):.0f}_{feat.get('y1', 0):.0f}",
                    "type": "chamfer",
                    "x1": feat.get("x1", 0),
                    "y1": feat.get("y1", 0),
                    "x2": feat.get("x2", 0),
                    "y2": feat.get("y2", 0),
                    "length": feat.get("length", 0),
                    "angle_deg": feat.get("angle_deg", 0),
                    "source": "fallback_chamfer",
                })

        return results


# ── Body dimension extraction ─────────────────────────────────────────────────


def _extract_body_dimensions(
    contours: list[dict[str, Any]],
    dimensions: list[dict[str, Any]],
    circles: list[dict[str, Any]],
) -> tuple[float | None, float | None]:
    """Infer body dimensions from contours or dimension annotations.

    Looks for the largest outer contour as the body, or uses the
    first available linear dimensions as width/height.
    """
    # Try to get from the largest outer contour
    outer_contours = [
        c for c in contours
        if not c.get("hierarchy", {}).get("is_child", False)
    ]
    if outer_contours:
        largest = max(outer_contours, key=lambda c: c.get("area", 0))
        bbox = largest.get("bounding_box", {})
        w = bbox.get("width")
        h = bbox.get("height")
        if w and h:
            return float(w), float(h)

    # Fallback: use linear dimension values
    linear_dims = [d for d in dimensions if d.get("type") == "linear"]
    if len(linear_dims) >= 2:
        return float(linear_dims[0]["value"]), float(linear_dims[1]["value"])
    elif len(linear_dims) == 1:
        return float(linear_dims[0]["value"]), None

    return None, None
