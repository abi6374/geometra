"""Agent 2: Drawing Understanding Agent.

Responsibilities:
- Line detection (Probabilistic Hough Transform)
- Circle detection (Hough Circle Transform)
- Arc detection (contour-based angle sweep)
- Contour extraction (hierarchical)
- Morphological operations (denoise, close gaps, thin)
- Engineering symbol extraction (crosshairs, centerlines, arrows, center marks)

Libraries: OpenCV (cv2), scikit-image

Input: Standardized document from InputProcessingAgent.
  - Raster images (PNG, JPEG, TIFF): processed via OpenCV + skimage
  - DXF files: primitives extracted directly from entity data
"""
from __future__ import annotations

import logging
import math
from typing import Any

import cv2
import numpy as np
from geometra.agents.base import BaseAgent

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Hough Line Transform defaults
HOUGH_RHO = 1.0  # Distance resolution (pixels)
HOUGH_THETA = np.pi / 180  # Angle resolution (radians)
HOUGH_THRESHOLD = 80
HOUGH_MIN_LINE_LENGTH = 30
HOUGH_MAX_LINE_GAP = 10

# Hough Circle Transform defaults
CIRCLE_DP = 1.2
CIRCLE_MIN_DIST = 30
CIRCLE_PARAM1 = 80
CIRCLE_PARAM2 = 35
CIRCLE_MIN_RADIUS = 5
CIRCLE_MAX_RADIUS = 500

# Canny Edge Detection defaults
CANNY_THRESHOLD1 = 50
CANNY_THRESHOLD2 = 150
CANNY_APERTURE = 3

# Morphological kernel sizes
MORPH_KERNEL_SIZE = 3

# Arc detection: minimum circular arc length (as fraction of circumference)
MIN_ARC_FRACTION = 0.15
MAX_ARC_FRACTION = 0.85

# Symbol detection
CENTER_CROSS_SIZE = 15  # max half-size of a center cross
CENTER_CROSS_TOLERANCE = 0.3  # angle tolerance from 90 degrees

# Contour filtering
MIN_CONTOUR_AREA = 20  # minimum pixels to keep a contour
POLY_APPROX_EPSILON = 2.0  # Ramer-Douglas-Peucker epsilon


def _preprocess(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess a grayscale image for geometric primitive extraction.

    Steps:
        1. Gaussian blur for noise reduction
        2. Binary threshold (Otsu or adaptive)
        3. Morphological close to fill small gaps
        4. Canny edge detection

    Returns:
        (binary, cleaned, edges) tuple
    """
    # Denoise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

    # Binary threshold using Otsu
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological clean: remove small noise, close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # Canny edges
    edges = cv2.Canny(blurred, CANNY_THRESHOLD1, CANNY_THRESHOLD2, apertureSize=CANNY_APERTURE)

    return binary, cleaned, edges


def _detect_lines(
    edges: np.ndarray,
    threshold: int | None = None,
    min_line_length: int | None = None,
) -> list[dict[str, Any]]:
    """Detect line segments using Probabilistic Hough Line Transform.

    Args:
        edges: Canny edge image.
        threshold: Optional override for Hough accumulator threshold.
        min_line_length: Optional override for minimum line length.

    Returns:
        List of dicts with keys: x1, y1, x2, y2, length, angle_deg
    """
    t = threshold if threshold is not None else HOUGH_THRESHOLD
    mll = min_line_length if min_line_length is not None else HOUGH_MIN_LINE_LENGTH

    lines_p = cv2.HoughLinesP(
        edges,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=t,
        minLineLength=mll,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    if lines_p is None:
        return []

    result = []
    for line in lines_p:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        angle_deg = math.degrees(math.atan2(dy, dx))
        result.append({
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "length": round(length, 2),
            "angle_deg": round(angle_deg, 2),
        })

    return result


def _detect_circles(gray: np.ndarray) -> list[dict[str, Any]]:
    """Detect circles using Hough Circle Transform.

    Returns:
        List of dicts with keys: x, y, radius
    """
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=CIRCLE_DP,
        minDist=CIRCLE_MIN_DIST,
        param1=CIRCLE_PARAM1,
        param2=CIRCLE_PARAM2,
        minRadius=CIRCLE_MIN_RADIUS,
        maxRadius=CIRCLE_MAX_RADIUS,
    )

    if circles is None:
        return []

    result = []
    for c in circles[0]:
        result.append({
            "x": round(float(c[0]), 2),
            "y": round(float(c[1]), 2),
            "radius": round(float(c[2]), 2),
        })

    return result


def _fit_circle_least_squares(points: np.ndarray) -> tuple[float, float, float]:
    """Fit a circle to 2D points using algebraic least squares (Kasa method).

    Solves the linearized problem:
        minimize Σ( (x_i² + y_i²) + a*x_i + b*y_i + c )²

    where:
        a = -2*cx
        b = -2*cy
        c = cx² + cy² - r²

    This gives the exact algebraic fit for an overdetermined system.
    Unlike cv2.minEnclosingCircle, this is NOT biased by bounding-box
    extents, so it produces accurate radii for partial arcs.

    Args:
        points: (N, 2) numpy array of 2D points.

    Returns:
        (cx, cy, radius) of the best-fit circle.
    """
    if len(points) < 3:
        raise ValueError("At least 3 points are required for circle fitting")

    x = points[:, 0]
    y = points[:, 1]

    # Build the linear system: A @ [a, b, c] = B
    # where:  x_i*a + y_i*b + c = -(x_i² + y_i²)
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x * x + y * y)

    # Solve via least squares
    abc, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c = abc

    cx = -0.5 * a
    cy = -0.5 * b
    radius_sq = cx * cx + cy * cy - c

    if radius_sq <= 0:
        # Fallback: use bounding circle if algebraic fit degenerates
        center, r = cv2.minEnclosingCircle(points.astype(np.float32))
        return float(center[0]), float(center[1]), float(r)

    radius = math.sqrt(radius_sq)
    return float(cx), float(cy), float(radius)


def _detect_arcs(binary: np.ndarray, edges: np.ndarray) -> list[dict[str, Any]]:
    """Detect arcs (partial circles) via contour analysis and least-squares circle fitting.

    Strategy:
        1. Skeletonize the binary image to get single-pixel-wide centerlines
        2. Find contours of the skeleton (centerline arcs)
        3. Fit a circle to each contour using least-squares fitting (Kasa method)
        4. Classify as arc if the contour covers a fraction of the full circumference

    Skeletonization is critical: without it, binary contours wrap around the full
    line thickness (producing ribbon-like shapes), and edge contours are doubled
    (one on each side of the line). The skeleton gives a single clean centerline
    for accurate circle fitting.

    Returns:
        List of dicts with keys: x, y, radius, start_angle, end_angle, arc_length,
        fit_error (mean residual of the fit)
    """
    # Skeletonize the binary image to get single-pixel-wide centerlines
    # This produces contours that trace the actual arc curve, not the
    # outer boundary of the line thickness.
    try:
        from skimage.morphology import skeletonize as sk_skeletonize
        skeleton_bool = sk_skeletonize(binary > 0)
        skeleton = (skeleton_bool * 255).astype(np.uint8)
    except ImportError:
        skeleton = _opencv_thin(binary)

    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    arcs = []
    for contour in contours:
        # Get contour points as (N, 2) array
        points = contour[:, 0, :].astype(np.float64)  # shape (N, 2)
        if len(points) < 15:  # too few points for reliable fit
            continue

        # Fit circle using least squares
        try:
            cx, cy, radius = _fit_circle_least_squares(points)
        except (ValueError, np.linalg.LinAlgError):
            continue

        if radius < CIRCLE_MIN_RADIUS or radius > CIRCLE_MAX_RADIUS:
            continue

        # Compute fit quality: mean residual distance from circle
        dx = points[:, 0] - cx
        dy = points[:, 1] - cy
        distances = np.sqrt(dx * dx + dy * dy)
        residuals = np.abs(distances - radius)
        mean_residual = float(np.mean(residuals))
        max_residual = float(np.max(residuals))

        # Reject poor fits: mean residual > 10% of radius
        if radius > 0 and mean_residual > 0.1 * radius:
            continue

        # Determine if it's an arc (partial circle) by perimeter ratio
        perimeter = float(cv2.arcLength(contour, True))
        expected_perimeter = 2 * math.pi * radius
        if expected_perimeter <= 0:
            continue

        perimeter_ratio = perimeter / expected_perimeter

        # Compute angular span
        angles = np.degrees(np.arctan2(dy, dx))

        if len(angles) == 0:
            continue

        start_angle = float(np.min(angles))
        end_angle = float(np.max(angles))

        # Handle angle wrapping (arc crossing -180/180)
        if end_angle - start_angle > 180:
            wrapped = np.where(angles < 0, angles + 360, angles)
            start_angle = float(np.min(wrapped))
            end_angle = float(np.max(wrapped))

        angular_span = end_angle - start_angle

        # Classify: full circle (ratio > 0.85) vs arc (between thresholds)
        # Only emit arcs, not full circles
        if MIN_ARC_FRACTION <= perimeter_ratio < MAX_ARC_FRACTION:
            if angular_span < 15:  # too small to be meaningful
                continue

            arcs.append({
                "x": round(cx, 2),
                "y": round(cy, 2),
                "radius": round(radius, 2),
                "start_angle": round(start_angle, 2),
                "end_angle": round(end_angle, 2),
                "arc_length": round(perimeter, 2),
                "angular_span_deg": round(angular_span, 2),
                "fit_error": round(mean_residual, 4),
                "max_fit_error": round(max_residual, 4),
                "n_points": len(points),
            })

    return arcs


def _extract_contours(binary: np.ndarray) -> list[dict[str, Any]]:
    """Extract contours with hierarchy information.

    Returns:
        List of dicts with keys: area, perimeter, centroid, bounding_box,
        approx_polygon, convex, hierarchy_info
    """
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    result = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        M = cv2.moments(contour)
        cx = M["m10"] / M["m00"] if M["m00"] != 0 else 0
        cy = M["m01"] / M["m00"] if M["m00"] != 0 else 0

        x, y, w, h = cv2.boundingRect(contour)
        epsilon = POLY_APPROX_EPSILON
        approx = cv2.approxPolyDP(contour, epsilon, True)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # Hierarchy: [next, prev, first_child, parent]
        hier = hierarchy[0][i] if hierarchy is not None else [-1, -1, -1, -1]

        result.append({
            "id": i,
            "area": int(area),
            "perimeter": round(float(perimeter), 2),
            "centroid": (round(float(cx), 2), round(float(cy), 2)),
            "bounding_box": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
            },
            "approx_polygon_vertices": len(approx),
            "approx_polygon": approx.reshape(-1, 2).tolist(),
            "solidity": round(float(solidity), 4),
            "hierarchy": {
                "next": int(hier[0]),
                "prev": int(hier[1]),
                "first_child": int(hier[2]),
                "parent": int(hier[3]),
                "has_children": hier[2] != -1,
                "is_child": hier[3] != -1,
            },
        })

    return result


def _detect_engineering_symbols(binary: np.ndarray, edges: np.ndarray, lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect common engineering drawing symbols.

    Detects:
        - Center crosses / center marks: two perpendicular lines crossing
        - Dimension arrows: small triangular shapes near line endpoints
        - Centerlines: long thin dashed patterns
        - Crosshairs: small plus-sign shapes

    Returns:
        List of dicts with keys: type, location, properties
    """
    symbols = []

    # ── 1. Center crosses (center marks) ──
    # A center cross is two short perpendicular lines intersecting at their midpoints
    for i, line_a in enumerate(lines):
        mid_x_a = (line_a["x1"] + line_a["x2"]) / 2
        mid_y_a = (line_a["y1"] + line_a["y2"]) / 2
        len_a = line_a["length"]

        # Skip lines that are too long (likely drawing lines, not symbols)
        if len_a > CENTER_CROSS_SIZE * 2:
            continue

        for j, line_b in enumerate(lines):
            if j <= i:
                continue

            mid_x_b = (line_b["x1"] + line_b["x2"]) / 2
            mid_y_b = (line_b["y1"] + line_b["y2"]) / 2
            len_b = line_b["length"]

            if len_b > CENTER_CROSS_SIZE * 2:
                continue

            # Skip if lines have nearly the same angle (not perpendicular)
            raw_diff = abs(line_a["angle_deg"] - line_b["angle_deg"])
            if raw_diff < 30:
                continue

            # Check if midpoints are close (within a few pixels)
            mid_dist = math.sqrt((mid_x_a - mid_x_b) ** 2 + (mid_y_a - mid_y_b) ** 2)
            if mid_dist > 8:
                continue

            # Check if lines are nearly perpendicular
            angle_diff = abs(line_a["angle_deg"] - line_b["angle_deg"])
            angle_diff = min(angle_diff % 180, 180 - angle_diff % 180)

            if 85 <= angle_diff <= 95:
                symbols.append({
                    "type": "center_cross",
                    "x": round(float(mid_x_a), 2),
                    "y": round(float(mid_y_a), 2),
                    "properties": {
                        "line1_length": round(len_a, 2),
                        "line2_length": round(len_b, 2),
                        "line1_angle": line_a["angle_deg"],
                        "line2_angle": line_b["angle_deg"],
                    },
                })

    # ── 2. Centerlines (long thin lines passing through hole centers) ──
    # This is best done after circle detection, so we check if a line
    # passes near circle centers. We store this as a separate pass
    # (called from extract() after circles are detected).

    # ── 3. Arrowheads (small triangular contours) ──
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5 or area > 80:  # arrows are typically small
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        # Triangular shapes have approx 3 vertices
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 3:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                symbols.append({
                    "type": "arrowhead",
                    "x": round(float(cx), 2),
                    "y": round(float(cy), 2),
                    "properties": {
                        "area": int(area),
                        "orientation": _arrow_orientation(approx),
                    },
                })

    return symbols


def _arrow_orientation(triangle: np.ndarray) -> str:
    """Determine the orientation of a triangular arrowhead."""
    # Find the "tip" - the vertex farthest from the centroid
    M = cv2.moments(triangle)
    if M["m00"] == 0:
        return "unknown"
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    max_dist = 0
    tip = None
    for pt in triangle[:, 0, :]:
        d = math.sqrt((pt[0] - cx) ** 2 + (pt[1] - cy) ** 2)
        if d > max_dist:
            max_dist = d
            tip = pt

    if tip is None:
        return "unknown"

    dx = tip[0] - cx
    dy = tip[1] - cy
    angle = math.degrees(math.atan2(dy, dx))

    if -45 <= angle <= 45:
        return "right"
    elif 45 < angle <= 135:
        return "down"
    elif -135 <= angle < -45:
        return "up"
    else:
        return "left"


def _detect_centerlines(
    lines: list[dict[str, Any]],
    circles: list[dict[str, Any]],
    img_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    """Detect centerlines — long thin lines passing through circle/arc centers.

    Returns:
        List of dicts with type, start, end, passes_through_center
    """
    centerlines = []
    for line in lines:
        if line["length"] < 20:
            continue

        # Check if this line passes near any circle center
        for circle in circles:
            # Distance from line to circle center
            dist = _point_to_line_distance(
                circle["x"], circle["y"],
                line["x1"], line["y1"],
                line["x2"], line["y2"],
            )
            if dist <= circle["radius"] * 0.3:
                centerlines.append({
                    "type": "centerline",
                    "x1": line["x1"],
                    "y1": line["y1"],
                    "x2": line["x2"],
                    "y2": line["y2"],
                    "length": line["length"],
                    "angle_deg": line["angle_deg"],
                    "associated_circle_center": (circle["x"], circle["y"]),
                    "associated_circle_radius": circle["radius"],
                })
                break

    return centerlines


def _point_to_line_distance(px: float, py: float, x1: int, y1: int, x2: int, y2: int) -> float:
    """Compute the perpendicular distance from a point to a line segment."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))

    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


# ── Morphological helpers ─────────────────────────────────────────────────────


def _apply_morphological_ops(binary: np.ndarray) -> dict[str, Any]:
    """Apply morphological operations and return various views.

    Returns dict with:
        - cleaned: noise removed + gaps filled
        - skeleton: skeletonized (thinned) version
        - dilated: dilated version
        - eroded: eroded version
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))

    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    eroded = cv2.erode(closed, kernel, iterations=1)

    # Skeletonize using skimage if available, else use OpenCV thinning
    try:
        from skimage.morphology import skeletonize as sk_skeletonize

        skeleton_bool = sk_skeletonize(closed > 0)
        skeleton = (skeleton_bool * 255).astype(np.uint8)
    except ImportError:
        logger.warning("scikit-image not available; using fallback OpenCV thinning (may be less accurate)")
        # Fallback: OpenCV thinning via repeated erosion
        skeleton = _opencv_thin(closed)

    return {
        "cleaned": closed,
        "skeleton": skeleton,
        "dilated": dilated,
        "eroded": eroded,
    }


def _opencv_thin(binary: np.ndarray) -> np.ndarray:
    """Thin a binary image to a single-pixel-wide skeleton using morphological operations.

    Repeatedly applies an opening (erode + dilate) and subtracts the result
    from the current image to peel off removable border pixels.
    """
    thin = binary.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(thin) > 0:
        eroded = cv2.erode(thin, kernel)
        opened = cv2.dilate(eroded, kernel)
        # Pixels removed by the opening are the border pixels that can
        # be safely peeled off this iteration.
        removed = cv2.subtract(thin, opened)
        if cv2.countNonZero(removed) == 0:
            break
        thin = cv2.subtract(thin, removed)
    return thin


# ── DXF primitive extraction ──────────────────────────────────────────────────


def _extract_dxf_primitives(data: dict[str, Any]) -> dict[str, Any]:
    """Extract geometric primitives directly from DXF entity data
    (previously loaded by Agent 1: InputProcessingAgent).

    Returns the same structure as the raster extraction pipeline.
    """
    lines = []
    circles = []
    arcs = []
    contours = []
    symbols = []

    entities = data.get("entities", [])

    for entity in entities:
        etype = entity.get("type")

        if etype == "LINE":
            x1, y1 = entity.get("start", (0, 0))
            x2, y2 = entity.get("end", (0, 0))
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx * dx + dy * dy)
            angle_deg = math.degrees(math.atan2(dy, dx))
            lines.append({
                "x1": round(x1, 4),
                "y1": round(y1, 4),
                "x2": round(x2, 4),
                "y2": round(y2, 4),
                "length": round(length, 4),
                "angle_deg": round(angle_deg, 4),
                "layer": entity.get("layer"),
                "source": "dxf",
            })

        elif etype == "CIRCLE":
            cx, cy = entity.get("center", (0, 0))
            radius = entity.get("radius", 0)
            circles.append({
                "x": round(cx, 4),
                "y": round(cy, 4),
                "radius": round(radius, 4),
                "layer": entity.get("layer"),
                "source": "dxf",
            })

        elif etype == "ARC":
            cx, cy = entity.get("center", (0, 0))
            radius = entity.get("radius", 0)
            start_angle = entity.get("start_angle", 0)
            end_angle = entity.get("end_angle", 0)
            arc_length = 2 * math.pi * radius * (abs(end_angle - start_angle) / 360.0)
            arcs.append({
                "x": round(cx, 4),
                "y": round(cy, 4),
                "radius": round(radius, 4),
                "start_angle": round(start_angle, 4),
                "end_angle": round(end_angle, 4),
                "arc_length": round(arc_length, 4),
                "angular_span_deg": round(abs(end_angle - start_angle), 4),
                "layer": entity.get("layer"),
                "source": "dxf",
            })

        elif etype == "LWPOLYLINE":
            points = entity.get("points", [])
            closed = entity.get("closed", False)
            # Convert polyline to segments
            for k in range(len(points) - 1):
                x1, y1 = points[k]
                x2, y2 = points[k + 1]
                dx = x2 - x1
                dy = y2 - y1
                length = math.sqrt(dx * dx + dy * dy)
                lines.append({
                    "x1": round(x1, 4),
                    "y1": round(y1, 4),
                    "x2": round(x2, 4),
                    "y2": round(y2, 4),
                    "length": round(length, 4),
                    "angle_deg": round(math.degrees(math.atan2(dy, dx)), 4),
                    "layer": entity.get("layer"),
                    "source": "dxf_polyline",
                })
            if closed and len(points) >= 3:
                x1, y1 = points[-1]
                x2, y2 = points[0]
                dx = x2 - x1
                dy = y2 - y1
                length = math.sqrt(dx * dx + dy * dy)
                lines.append({
                    "x1": round(x1, 4),
                    "y1": round(y1, 4),
                    "x2": round(x2, 4),
                    "y2": round(y2, 4),
                    "length": round(length, 4),
                    "angle_deg": round(math.degrees(math.atan2(dy, dx)), 4),
                    "layer": entity.get("layer"),
                    "source": "dxf_polyline",
                })

    image_shape = None
    image_width = None
    image_height = None
    # Try to get image shape from header_vars if available
    header_vars = data.get("header_vars", {})
    extmin = header_vars.get("$EXTMIN")
    extmax = header_vars.get("$EXTMAX")
    if extmin and extmax:
        # Convert coordinate tuples to (height, width) format
        # matching the raster pipeline's (img_h, img_w) convention
        try:
            min_x, min_y = float(extmin.split(",")[0] if isinstance(extmin, str) else extmin[0] if hasattr(extmin, "__getitem__") else 0), float(extmin.split(",")[1] if isinstance(extmin, str) else extmin[1] if hasattr(extmin, "__getitem__") else 0)
            max_x, max_y = float(extmax.split(",")[0] if isinstance(extmax, str) else extmax[0] if hasattr(extmax, "__getitem__") else 0), float(extmax.split(",")[1] if isinstance(extmax, str) else extmax[1] if hasattr(extmax, "__getitem__") else 0)
            image_width = int(max_x - min_x)
            image_height = int(max_y - min_y)
            image_shape = (image_height, image_width)
        except (ValueError, TypeError, IndexError):
            pass

    return {
        "lines": lines,
        "circles": circles,
        "arcs": arcs,
        "contours": contours,
        "symbols": symbols,
        "image_shape": image_shape,
        "image_width": image_width,
        "image_height": image_height,
        "morphology": None,
        "preprocessing": {"source": "dxf", "entity_count": len(entities)},
    }


# ── Main Agent ────────────────────────────────────────────────────────────────


class DrawingUnderstandingAgent(BaseAgent):
    """Extracts geometric primitives from engineering drawings using classical CV.

    Supports:
        - Raster images (PNG, JPEG, TIFF): Full CV pipeline with Hough transforms,
          contour analysis, morphological operations, and symbol detection
        - DXF files: Direct primitive extraction from entity data
    """

    def validate_input(self, input_data: Any) -> bool:
        """Validate that input has the required keys."""
        if not isinstance(input_data, dict):
            return False
        required = {"format", "data"}
        return all(k in input_data for k in required)

    def process(self, input_data: Any, **kwargs: Any) -> dict[str, Any]:
        """Alias for extract()."""
        return self.extract(input_data, **kwargs)

    def extract(
        self,
        doc: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Extract geometric primitives from a normalized document.

        Args:
            doc: Standardized document from InputProcessingAgent.
                Must contain 'format' and 'data' keys.
            **kwargs: Optional overrides for detection parameters.

        Returns:
            Geometric primitive graph containing:
                - lines: detected line segments
                - circles: detected circles (full circles)
                - arcs: detected arcs (partial circles)
                - contours: extracted contours with hierarchy
                - symbols: detected engineering symbols
                - image_shape: dimensions of source image
                - image_width / image_height: pixel dimensions
                - morphology: results of morphological operations
                - preprocessing: preprocessing info and parameters used
        """
        self.logger.info("Extracting geometric primitives from %s", doc.get("format", "unknown"))

        doc_format = doc.get("format", "")
        doc_data = doc.get("data", {})

        # ── Handle DXF files ──
        if doc_format == "dxf":
            self.logger.info("DXF input: extracting primitives directly from entities")
            return _extract_dxf_primitives(doc_data)

        # ── Handle vector PDF (no raster image data) ──
        if doc_format == "pdf" and "image" not in doc_data:
            self.logger.warning("PDF without embedded image data; returning empty primitives")
            return {
                "lines": [],
                "circles": [],
                "arcs": [],
                "contours": [],
                "symbols": [],
                "image_shape": None,
                "image_width": None,
                "image_height": None,
                "morphology": None,
                "preprocessing": {"note": "PDF has no embedded raster data"},
            }

        # ── Handle raster images ──
        gray = doc_data.get("gray")
        if gray is None:
            # Try to load from image key
            img = doc_data.get("image")
            if img is not None:
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
            else:
                self.logger.warning("No image data available for primitive extraction")
                return {
                    "lines": [],
                    "circles": [],
                    "arcs": [],
                    "contours": [],
                    "symbols": [],
                    "image_shape": None,
                    "image_width": None,
                    "image_height": None,
                    "morphology": None,
                    "preprocessing": {"note": "No image data found in document"},
                }

        # Apply any user-specified parameter overrides
        hough_threshold = kwargs.get("hough_threshold", HOUGH_THRESHOLD)
        hough_min_line = kwargs.get("hough_min_line_length", HOUGH_MIN_LINE_LENGTH)
        canny_t1 = kwargs.get("canny_threshold1", CANNY_THRESHOLD1)
        canny_t2 = kwargs.get("canny_threshold2", CANNY_THRESHOLD2)
        circle_param2 = kwargs.get("circle_param2", CIRCLE_PARAM2)

        # ── Preprocessing ──
        binary, cleaned, edges = _preprocess(gray)

        # ── Morphology ──
        morph_results = _apply_morphological_ops(binary)

        # ── Line Detection (with optional parameter overrides) ──
        use_threshold = hough_threshold if hough_threshold != HOUGH_THRESHOLD else None
        use_min_line = hough_min_line if hough_min_line != HOUGH_MIN_LINE_LENGTH else None
        lines = _detect_lines(edges, threshold=use_threshold, min_line_length=use_min_line)

        # ── Circle Detection ──
        circles = _detect_circles(gray)

        # ── Arc Detection ──
        arcs = _detect_arcs(binary, edges)

        # ── Contour Extraction ──
        contours = _extract_contours(cleaned)

        # ── Engineering Symbols ──
        symbols = _detect_engineering_symbols(cleaned, edges, lines)
        centerlines = _detect_centerlines(lines, circles, gray.shape)
        symbols.extend(centerlines)

        img_h, img_w = gray.shape[:2]

        return {
            "lines": lines,
            "circles": circles,
            "arcs": arcs,
            "contours": contours,
            "symbols": symbols,
            "image_shape": (img_h, img_w),
            "image_width": img_w,
            "image_height": img_h,
            "morphology": {
                "cleaned": morph_results["cleaned"],
                "skeleton": morph_results["skeleton"],
                "dilated": morph_results["dilated"],
                "eroded": morph_results["eroded"],
            },
            "preprocessing": {
                "binary": binary,
                "edges": edges,
                "canny_threshold1": canny_t1,
                "canny_threshold2": canny_t2,
                "hough_threshold": hough_threshold,
                "hough_min_line_length": hough_min_line,
                "circle_param2": circle_param2,
            },
        }


