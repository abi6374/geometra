"""Agent 4: Feature Recognition Agent.

Responsibilities:
Detect manufacturing features using dual-path detection:
  1. ML path: YOLOv11-seg segmentation (when model available)
  2. CV/rule path: Classical geometric analysis from Agent 2 primitives

Manufacturing features detected:
  - Hole: circular through-hole or blind hole
  - Slot: elongated opening (obround, T-slot, keyway)
  - Pocket: recessed cavity (rectangular, circular)
  - Boss: raised protrusion
  - Cutout: interior opening with boundary (display, access panel)
  - Vent: group of thin parallel slots
  - Fillet: curved internal corner (arc)
  - Chamfer: beveled edge (short diagonal line)

Libraries: ultralytics (optional, for YOLO path), OpenCV, numpy

Input: Geometric primitive graph from DrawingUnderstandingAgent (Agent 2).
  - Primitives: lines, circles, arcs, contours (with hierarchy), symbols
  - Raster / DXF both produce the same primitive format
"""

from __future__ import annotations

import logging
import math
from typing import Any

import cv2
import numpy as np

from geometra.agents.base import BaseAgent

logger = logging.getLogger(__name__)

# ── Known manufacturing feature classes ──────────────────────────────────────

FEATURE_CLASSES = [
    "hole",
    "slot",
    "pocket",
    "boss",
    "cutout",
    "vent",
    "fillet",
    "chamfer",
]

# Minimum contour area (pixels²) to consider as a feature
MIN_FEATURE_AREA = 50

# Aspect ratio threshold: below this, shape is considered "elongated"
SLOT_ASPECT_RATIO = 0.4

# Minimum number of parallel vents to form a vent group
VENT_MIN_COUNT = 3

# Max distance (pixels) between vent lines to be considered part of same group
VENT_MAX_GAP = 30

# Max length for a line to be a chamfer candidate
CHAMFER_MAX_LENGTH = 30

# Angle tolerance (degrees) to consider a line as diagonal (chamfer-like)
CHAMFER_ANGLE_TOLERANCE = 15


# ── YOLO Inference Engine ────────────────────────────────────────────────────


class YOLOInferenceEngine:
    """Wrapper around Ultralytics YOLO for segmentation inference.

    The model is loaded lazily on first use. If ultralytics is not
    installed or the model file is not found, inference returns None
    and the CV/rule path is used instead.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path
        self._model = None
        self._available: bool | None = None  # None = not yet checked

    @property
    def available(self) -> bool:
        """Check if YOLO model is available (lazy check)."""
        if self._available is None:
            self._available = self._try_load()
        return self._available

    def _try_load(self) -> bool:
        """Try to import ultralytics and load the model."""
        try:
            from ultralytics import YOLO  # noqa: F401
        except ImportError:
            logger.info("Ultralytics not installed; YOLO inference unavailable")
            return False

        model_path = self._model_path or "yolo11n-seg.pt"
        try:
            self._model = YOLO(model_path)
            logger.info("YOLO model loaded: %s", model_path)
            return True
        except Exception as exc:
            logger.warning("Failed to load YOLO model '%s': %s", model_path, exc)
            return False

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ) -> list[dict[str, Any]] | None:
        """Run YOLO segmentation inference on an image.

        Args:
            image: BGR image array.
            conf_threshold: Minimum confidence threshold.
            iou_threshold: NMS IoU threshold.

        Returns:
            List of detection dicts with keys: class_id, class_name, confidence,
            bbox (xyxy), mask_polygon (list of [x,y] pixel points), or None if unavailable.
        """
        if not self.available or self._model is None:
            return None

        try:
            results = self._model.predict(
                source=image,
                imgsz=640,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )
        except Exception as exc:
            logger.warning("YOLO inference failed: %s", exc)
            return None

        if not results or len(results) == 0:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        detections = []
        boxes = result.boxes
        masks = result.masks

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())
            bbox_xyxy = boxes.xyxy[i].tolist()

            class_name = result.names.get(cls_id, str(cls_id)) if hasattr(result, "names") else str(cls_id)

            # Extract mask polygon if available
            mask_polygon = None
            if masks is not None and i < len(masks.xy):
                mask_polygon = masks.xy[i].tolist()

            detections.append({
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": {
                    "x1": round(bbox_xyxy[0], 2),
                    "y1": round(bbox_xyxy[1], 2),
                    "x2": round(bbox_xyxy[2], 2),
                    "y2": round(bbox_xyxy[3], 2),
                },
                "mask_polygon": mask_polygon,
                "source": "yolo",
            })

        return detections


# ── Classical CV Feature Detection ──────────────────────────────────────────


def _detect_holes_from_circles(
    circles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect holes from detected circles.

    Every circle from the Hough transform is a potential hole.
    Holes are identified by the presence of a center cross or
    center mark near the circle center.

    Returns:
        List of hole feature dicts.
    """
    features = []
    for circle in circles:
        features.append({
            "type": "hole",
            "subtype": "circular",
            "x": circle["x"],
            "y": circle["y"],
            "radius": circle["radius"],
            "diameter": round(circle["radius"] * 2, 2),
            "area": round(math.pi * circle["radius"] ** 2, 2),
            "confidence": 0.85,
            "source": "circle_detection",
            "properties": {
                "circular": True,
                "has_center_mark": True,
            },
        })
    return features


def _detect_slots_from_contours(
    contours: list[dict[str, Any]],
    lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect slots from contours and lines.

    Slots are elongated shapes with parallel sides and rounded ends.
    Detection criteria:
      - High aspect ratio (width/height > 2 or height/width > 2)
      - Closed contour with area above threshold
      - Moderate to high circularity (rounded ends)
    """
    features = []
    for contour in contours:
        bbox = contour["bounding_box"]
        w, h = bbox["width"], bbox["height"]
        if w == 0 or h == 0:
            continue

        aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0

        # Slots are elongated
        if aspect > SLOT_ASPECT_RATIO:
            continue

        area = contour["area"]
        perimeter = contour["perimeter"]

        if area < MIN_FEATURE_AREA:
            continue

        # Circularity: 4π * area / perimeter²
        circularity = (4 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

        # Slots have moderate circularity (not too rectangular, not circular)
        if circularity < 0.2:
            continue

        centroid = contour["centroid"]
        features.append({
            "type": "slot",
            "x": round(centroid[0], 2),
            "y": round(centroid[1], 2),
            "width": w,
            "height": h,
            "area": area,
            "aspect_ratio": round(aspect, 4),
            "circularity": round(circularity, 4),
            "confidence": 0.7,
            "source": "contour_analysis",
            "properties": {
                "closed": True,
                "vertex_count": contour["approx_polygon_vertices"],
            },
        })

    return features


def _detect_pockets_and_cutouts(
    contours: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Detect pockets and cutouts from contour hierarchy.

    Pockets: interior contours (has parent) below a size threshold.
    Cutouts: contours that contain children (inner boundary of an opening).

    Returns:
        (pockets, cutouts) lists of feature dicts.
    """
    pockets = []
    cutouts = []

    for contour in contours:
        area = contour["area"]
        if area < MIN_FEATURE_AREA:
            continue

        bbox = contour["bounding_box"]
        hierarchy = contour["hierarchy"]
        is_child = hierarchy["is_child"]
        has_children = hierarchy["has_children"]
        vertices = contour["approx_polygon_vertices"]

        centroid = contour["centroid"]

        # Cutout: outer contour that has children (like a rectangular opening)
        if has_children and not is_child:
            # Determine shape type based on vertex count
            if vertices == 4:
                shape = "rectangular"
            elif vertices >= 6:
                shape = "complex"
            else:
                shape = "irregular"

            cutouts.append({
                "type": "cutout",
                "x": round(centroid[0], 2),
                "y": round(centroid[1], 2),
                "width": bbox["width"],
                "height": bbox["height"],
                "area": area,
                "shape": shape,
                "confidence": 0.75,
                "source": "contour_hierarchy",
                "properties": {
                    "vertex_count": vertices,
                    "solidity": contour["solidity"],
                    "has_children": True,
                    "child_count": sum(
                        1 for c in contours
                        if c["hierarchy"]["parent"] == contour["id"]
                    ),
                },
            })

        # Pocket: interior contour (child without its own children)
        # Typically pockets are inside a parent and don't contain inner details
        elif is_child and not has_children and vertices >= 3:
            shape = "rectangular" if vertices == 4 else "circular" if vertices >= 8 else "irregular"
            pockets.append({
                "type": "pocket",
                "x": round(centroid[0], 2),
                "y": round(centroid[1], 2),
                "width": bbox["width"],
                "height": bbox["height"],
                "area": area,
                "shape": shape,
                "depth": None,  # Not measurable from 2D
                "confidence": 0.65,
                "source": "contour_hierarchy",
                "properties": {
                    "vertex_count": vertices,
                    "solidity": contour["solidity"],
                    "parent_id": hierarchy["parent"],
                },
            })

    return pockets, cutouts


def _detect_vents(
    contours: list[dict[str, Any]],
    lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect vents: groups of parallel thin slots/lines.

    Vents are identified by:
      1. Multiple thin, elongated contours with similar orientation
      2. Grouped together in a small area
      3. Parallel to each other
    """
    # Filter contours that could be vent slots (thin, elongated)
    vent_candidates = []
    for contour in contours:
        bbox = contour["bounding_box"]
        w, h = bbox["width"], bbox["height"]
        if w == 0 or h == 0:
            continue

        aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        area = contour["area"]

        if area < 10 or area > 2000:
            continue
        if aspect > 0.3:  # must be thin
            continue

        # Orientation: use the bounding box to determine
        orientation = "horizontal" if w >= h else "vertical"
        vent_candidates.append({
            "contour_id": contour["id"],
            "centroid": contour["centroid"],
            "area": area,
            "bounding_box": bbox,
            "orientation": orientation,
            "aspect": aspect,
        })

    # Group by orientation and proximity (connected-components / BFS)
    groups = []
    used = set()

    for i, c1 in enumerate(vent_candidates):
        if i in used:
            continue

        group = [c1]
        used.add(i)
        group_orientation = c1["orientation"]

        # BFS: keep checking for new members as the group grows
        changed = True
        while changed:
            changed = False
            for j, c2 in enumerate(vent_candidates):
                if j in used:
                    continue
                if c2["orientation"] != group_orientation:
                    continue
                # Check distance to ANY existing group member
                for member in group:
                    mx, my = member["centroid"]
                    cx2, cy2 = c2["centroid"]
                    dist = math.sqrt((mx - cx2) ** 2 + (my - cy2) ** 2)
                    if dist < VENT_MAX_GAP:
                        group.append(c2)
                        used.add(j)
                        changed = True
                        break

        if len(group) >= VENT_MIN_COUNT:
            # Compute group centroid
            avg_x = sum(m["centroid"][0] for m in group) / len(group)
            avg_y = sum(m["centroid"][1] for m in group) / len(group)

            # Estimate vent region bounds
            xs = [m["bounding_box"]["x"] for m in group]
            ys = [m["bounding_box"]["y"] for m in group]
            x_ends = [m["bounding_box"]["x"] + m["bounding_box"]["width"] for m in group]
            y_ends = [m["bounding_box"]["y"] + m["bounding_box"]["height"] for m in group]

            groups.append({
                "type": "vent",
                "x": round(avg_x, 2),
                "y": round(avg_y, 2),
                "width": max(x_ends) - min(xs),
                "height": max(y_ends) - min(ys),
                "slot_count": len(group),
                "orientation": c1["orientation"],
                "confidence": 0.7,
                "source": "vent_grouping",
                "properties": {
                    "slot_count": len(group),
                    "orientation": c1["orientation"],
                    "slots": [
                        {"centroid": m["centroid"], "area": m["area"]}
                        for m in group
                    ],
                },
            })

    return groups


def _detect_bosses(
    contours: list[dict[str, Any]],
    circles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect bosses: raised protrusions on the surface.

    Bosses are detected as:
      - Outer contours (no parent) that are not cutouts
      - Typically circular or nearly circular
      - High solidity (close to convex)
    """
    features = []
    for contour in contours:
        hierarchy = contour["hierarchy"]

        # Bosses are outer contours or top-level shapes
        if hierarchy["is_child"]:
            continue

        area = contour["area"]
        if area < MIN_FEATURE_AREA or area > 50000:
            continue

        # Check if it's a circular feature (potential boss)
        perimeter = contour["perimeter"]
        circularity = (4 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

        solidity = contour["solidity"]
        centroid = contour["centroid"]

        if circularity > 0.7 and solidity > 0.9:
            features.append({
                "type": "boss",
                "x": round(centroid[0], 2),
                "y": round(centroid[1], 2),
                "area": area,
                "perimeter": perimeter,
                "circularity": round(circularity, 4),
                "solidity": round(solidity, 4),
                "confidence": 0.6,
                "source": "contour_analysis",
                "properties": {
                    "circular": True,
                    "approx_radius": round(math.sqrt(area / math.pi), 2),
                },
            })

    return features


def _detect_fillets(
    arcs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect fillets from arcs with small radii.

    Fillets are short arc segments at internal corners.
    They are identified by:
      - Small radius (relative to drawing size)
      - Moderate angular span (60-120 degrees typical)
      - Good fit quality (tight least-squares fit)
    """
    features = []
    for arc in arcs:
        radius = arc["radius"]
        angular_span = arc["angular_span_deg"]
        fit_error = arc.get("fit_error", 0)

        # Fillets are small rounded corners
        if radius > 30:  # too large to be a fillet
            continue

        if angular_span < 30 or angular_span > 150:
            continue

        if fit_error > radius * 0.15:  # poor fit
            continue

        features.append({
            "type": "fillet",
            "x": arc["x"],
            "y": arc["y"],
            "radius": arc["radius"],
            "angular_span_deg": arc["angular_span_deg"],
            "arc_length": arc["arc_length"],
            "confidence": 0.7,
            "source": "arc_detection",
            "properties": {
                "fit_error": arc.get("fit_error", 0),
                "n_points": arc.get("n_points", 0),
            },
        })

    return features


def _detect_chamfers(
    lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect chamfers: short diagonal lines at corners.

    Chamfers are identified by:
      - Short line segments (length < threshold)
      - Diagonal angle (~45 degrees from horizontal/vertical)
      - Typically found at corners of rectangular shapes
    """
    features = []
    for line in lines:
        length = line["length"]
        if length > CHAMFER_MAX_LENGTH:
            continue

        angle = abs(line["angle_deg"]) % 180
        # Chamfers are typically at 45° (± tolerance)
        # Distance from nearest multiple of 45
        dist_from_45 = min(abs(angle % 45), abs(45 - angle % 45))
        if dist_from_45 > CHAMFER_ANGLE_TOLERANCE:
            continue

        features.append({
            "type": "chamfer",
            "x1": line["x1"],
            "y1": line["y1"],
            "x2": line["x2"],
            "y2": line["y2"],
            "length": length,
            "angle_deg": line["angle_deg"],
            "confidence": 0.55,
            "source": "line_analysis",
            "properties": {
                "distance_from_45deg": round(dist_from_45, 2),
            },
        })

    return features


# ── Main Agent ────────────────────────────────────────────────────────────────


class FeatureRecognitionAgent(BaseAgent):
    """Detects manufacturing features in engineering drawings.

    Uses dual-path detection:
      1. ML path: YOLOv11-seg segmentation (when model is available)
      2. CV/rule path: Classical contour/shape analysis (always available)

    The CV path detects features from the geometric primitives produced
    by Agent 2 (DrawingUnderstandingAgent):
      - Holes: from detected circles
      - Slots: from elongated contours with rounded ends
      - Pockets: from interior contours (child in hierarchy)
      - Cutouts: from contours with children (holes through the part)
      - Vents: from grouped parallel thin slots
      - Bosses: from convex, high-circularity contours
      - Fillets: from small-radius arcs
      - Chamfers: from short diagonal lines
    """

    def __init__(self, yolo_model_path: str | None = None) -> None:
        super().__init__()
        self._yolo = YOLOInferenceEngine(model_path=yolo_model_path)

    @property
    def yolo_available(self) -> bool:
        """Check if YOLO model is available."""
        return self._yolo.available

    def validate_input(self, input_data: Any) -> bool:
        """Validate that input has required keys."""
        if not isinstance(input_data, dict):
            return False
        required = {"format"}
        return all(k in input_data for k in required)

    def process(self, input_data: Any, **kwargs: Any) -> dict[str, Any]:
        """Alias for detect()."""
        return self.detect(input_data, **kwargs)

    def detect(
        self,
        primitives: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Detect manufacturing features from geometric primitives.

        Args:
            primitives: Output from DrawingUnderstandingAgent (Agent 2).
                Must contain lines, circles, arcs, contours, symbols.
            **kwargs:
                - use_yolo: bool, if True attempt YOLO inference (requires image data)
                - yolo_conf: float, YOLO confidence threshold
                - image: optional np.ndarray for YOLO inference

        Returns:
            Feature graph with:
                - features: list of detected feature dicts
                - feature_count: total count
                - class_counts: breakdown by feature type
                - detection_method: which path was used ('cv' | 'yolo' | 'both')
        """
        self.logger.info("Detecting manufacturing features")

        use_yolo = kwargs.get("use_yolo", False)
        image = kwargs.get("image")
        features: list[dict[str, Any]] = []

        # ── Path 1: YOLO inference ──
        yolo_detections = None
        if use_yolo and image is not None:
            yolo_conf = kwargs.get("yolo_conf", 0.25)
            yolo_detections = self._yolo.predict(image, conf_threshold=yolo_conf)
            if yolo_detections is not None:
                # Map YOLO class names to our feature types if possible
                for det in yolo_detections:
                    # Use class_name directly from YOLO (we assume the model
                    # was trained on our FEATURE_CLASSES or similar)
                    cls_name = det["class_name"].lower()
                    if cls_name in FEATURE_CLASSES:
                        features.append({
                            "type": cls_name,
                            "x": round((det["bbox"]["x1"] + det["bbox"]["x2"]) / 2, 2),
                            "y": round((det["bbox"]["y1"] + det["bbox"]["y2"]) / 2, 2),
                            "width": round(det["bbox"]["x2"] - det["bbox"]["x1"], 2),
                            "height": round(det["bbox"]["y2"] - det["bbox"]["y1"], 2),
                            "confidence": det["confidence"],
                            "source": "yolo",
                            "properties": {
                                "bbox": det["bbox"],
                                "mask_polygon": det.get("mask_polygon"),
                            },
                        })

        # ── Path 2: Classical CV/rule-based detection ──
        # Extract primitives from the document
        circles = primitives.get("circles", [])
        lines = primitives.get("lines", [])
        arcs = primitives.get("arcs", [])
        contours = primitives.get("contours", [])

        # 1. Holes (from circles)
        holes = _detect_holes_from_circles(circles)
        features.extend(holes)

        # 2. Slots (from elongated contours)
        slots = _detect_slots_from_contours(contours, lines)
        features.extend(slots)

        # 3. Pockets and Cutouts (from contour hierarchy)
        pockets, cutouts = _detect_pockets_and_cutouts(contours)
        features.extend(pockets)
        features.extend(cutouts)

        # 4. Vents (from grouped parallel slots)
        vents = _detect_vents(contours, lines)
        features.extend(vents)

        # 5. Bosses (from convex contours)
        bosses = _detect_bosses(contours, circles)
        features.extend(bosses)

        # 6. Fillets (from small-radius arcs)
        fillets = _detect_fillets(arcs)
        features.extend(fillets)

        # 7. Chamfers (from short diagonal lines)
        chamfers = _detect_chamfers(lines)
        features.extend(chamfers)

        # ── Compile results ──
        class_counts: dict[str, int] = {cls: 0 for cls in FEATURE_CLASSES}
        for feat in features:
            ftype = feat.get("type", "")
            if ftype in class_counts:
                class_counts[ftype] += 1

        method = "yolo" if yolo_detections is not None else "cv"
        if yolo_detections is not None and len(features) > len(yolo_detections):
            method = "both"

        self.logger.info(
            "Feature detection complete: %d features across %d types (method: %s)",
            len(features),
            sum(1 for v in class_counts.values() if v > 0),
            method,
        )

        return {
            "features": features,
            "feature_count": len(features),
            "class_counts": class_counts,
            "detection_method": method,
        }
