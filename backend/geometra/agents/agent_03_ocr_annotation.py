"""Agent 3: OCR and Annotation Agent.

Responsibilities:
- Extract dimensions (linear, diameter, radius) from engineering drawing images via OCR
- Extract tolerances and fits (e.g. H7, ±0.1)
- Extract feature labels (e.g. A-A, SECTION, notes)
- Parse text from DXF dimension entities
- Parse text from PDF text content
- Extract raw OCR text for downstream processing

Libraries: PaddleOCR (optional), OpenCV (fallback), regex

Input: Standardized document from InputProcessingAgent.
  - Raster images (PNG, JPEG, TIFF): OCR via PaddleOCR (with fallback)
  - DXF files: direct dimension entity extraction
  - PDF files: text content extraction
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

import cv2
import numpy as np

from geometra.agents.base import BaseAgent

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum confidence score for OCR results to be accepted
MIN_CONFIDENCE = 0.5

# Regex patterns for dimension parsing
RE_DIMENSION_NUMBER = re.compile(
    r"""
    [-+]?                    # optional sign
    (?:
        \d+(?:[.,]\d+)?      # integer or decimal (supports comma as decimal sep)
        |
        (?:[.,]\d+)          # decimal starting with dot/comma
    )
    """,
    re.VERBOSE,
)

RE_DIAMETER = re.compile(r"[⌀Ø]\s*(\d+(?:[.,]\d+)?)")
RE_RADIUS = re.compile(r"[Rr]\s*(\d+(?:[.,]\d+)?)")
RE_ANGULAR_DIM = re.compile(r"(\d+(?:[.,]\d+)?)\s*[°º]")
RE_TOLERANCE = re.compile(r"±\s*(\d+(?:[.,]\d+)?)")
RE_FIT = re.compile(r"(\d+(?:[.,]\d+)?)?\s*([HhPpKkNnJjFfGgDdEeCcBbAa][0-9]+)")
RE_SECTION_LABEL = re.compile(r"([A-Z])\s*[-–—]\s*([A-Z])")
RE_COORDINATE = re.compile(
    r"(?:X|x)\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*(?:Y|y)\s*[:=]?\s*(\d+(?:[.,]\d+)?)"
)



# Known dimension entity types in DXF
DXF_DIM_TYPES = {
    "DIMENSION",
    "DIMALIGNED",
    "DIMLINEAR",
    "DIMRADIUS",
    "DIMDIAMETER",
    "DIMANGULAR",
    "DIMORDINATE",
}


# ── Dimension Parsing ─────────────────────────────────────────────────────────


def parse_dimensions_from_text(
    text_blocks: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Parse OCR text blocks into structured dimensions, tolerances, and labels.

    Args:
        text_blocks: List of dicts with 'text', 'bbox', 'confidence'.

    Returns:
        Dict with keys: dimensions, tolerances, labels, raw_text
    """
    dimensions: list[dict[str, Any]] = []
    tolerances: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []

    for block in text_blocks:
        text = block.get("text", "").strip()
        bbox = block.get("bbox", [])
        confidence = block.get("confidence", 0.0)
        cx, cy = _bbox_center(bbox) if bbox else (0.0, 0.0)

        if not text or confidence < MIN_CONFIDENCE:
            continue

        # 1. Check for fit/tolerance annotations (e.g. "H7", "10 H7", "±0.1")
        fit_match = RE_FIT.search(text)
        tol_match = RE_TOLERANCE.search(text)

        if fit_match or tol_match:
            dim_value = _parse_number(fit_match.group(1)) if fit_match and fit_match.group(1) else None
            tolerance_entry: dict[str, Any] = {
                "text_raw": text,
                "position": {"x": round(cx, 2), "y": round(cy, 2)},
                "bbox": bbox,
                "confidence": round(confidence, 4),
            }

            if fit_match:
                tolerance_entry["type"] = "fit"
                tolerance_entry["fit"] = fit_match.group(2).strip()
                tolerance_entry["nominal_size"] = dim_value
                tolerances.append(tolerance_entry)

            if tol_match:
                tolerance_entry["type"] = "tolerance"
                tolerance_entry["value"] = _parse_number(tol_match.group(1))
                tolerance_entry["symbol"] = "±"
                tolerances.append(tolerance_entry)

            continue

        # 2. Check for diameter dimensions (e.g. "⌀12.5")
        diam_match = RE_DIAMETER.match(text)
        if diam_match:
            val = _parse_number(diam_match.group(1))
            dimensions.append({
                "value": val,
                "unit": "mm",
                "position": {"x": round(cx, 2), "y": round(cy, 2)},
                "bbox": bbox,
                "confidence": round(confidence, 4),
                "type": "diameter",
                "text_raw": text,
                "tolerance": None,
            })
            continue

        # 3. Check for radius dimensions (e.g. "R25")
        rad_match = RE_RADIUS.match(text)
        if rad_match:
            val = _parse_number(rad_match.group(1))
            dimensions.append({
                "value": val,
                "unit": "mm",
                "position": {"x": round(cx, 2), "y": round(cy, 2)},
                "bbox": bbox,
                "confidence": round(confidence, 4),
                "type": "radius",
                "text_raw": text,
                "tolerance": None,
            })
            continue

        # 4. Check for angular dimensions (e.g. "45°")
        ang_match = RE_ANGULAR_DIM.match(text)
        if ang_match:
            val = _parse_number(ang_match.group(1))
            dimensions.append({
                "value": val,
                "unit": "deg",
                "position": {"x": round(cx, 2), "y": round(cy, 2)},
                "bbox": bbox,
                "confidence": round(confidence, 4),
                "type": "angular",
                "text_raw": text,
                "tolerance": None,
            })
            continue

        # 5. Check for coordinate dimensions (e.g. "X:100 Y:50")
        coord_match = RE_COORDINATE.match(text)
        if coord_match:
            x_val = _parse_number(coord_match.group(1))
            y_val = _parse_number(coord_match.group(2))
            dimensions.append({
                "value": (x_val, y_val),
                "unit": "mm",
                "position": {"x": round(cx, 2), "y": round(cy, 2)},
                "bbox": bbox,
                "confidence": round(confidence, 4),
                "type": "coordinate",
                "text_raw": text,
                "tolerance": None,
            })
            continue

        # 6. Check for section labels (e.g. "A-A")
        section_match = RE_SECTION_LABEL.match(text)
        if section_match:
            labels.append({
                "text": text,
                "position": {"x": round(cx, 2), "y": round(cy, 2)},
                "bbox": bbox,
                "confidence": round(confidence, 4),
                "type": "section_label",
            })
            continue

        # 7. Check if it's a standalone number (potential linear dimension)
        num_match = RE_DIMENSION_NUMBER.match(text)
        if num_match and num_match.group() == text.strip():
            val = _parse_number(text)
            dimensions.append({
                "value": val,
                "unit": "mm",
                "position": {"x": round(cx, 2), "y": round(cy, 2)},
                "bbox": bbox,
                "confidence": round(confidence, 4),
                "type": "linear",
                "text_raw": text,
                "tolerance": None,
            })
            continue

        # 8. Remaining text → label
        if len(text) >= 2:
            labels.append({
                "text": text,
                "position": {"x": round(cx, 2), "y": round(cy, 2)},
                "bbox": bbox,
                "confidence": round(confidence, 4),
                "type": "annotation",
            })

    return {
        "dimensions": dimensions,
        "tolerances": tolerances,
        "labels": labels,
    }


def _parse_number(text: str) -> float | int:
    """Parse a number string, handling comma as decimal separator."""
    cleaned = text.strip().replace(",", ".")
    val = float(cleaned)
    return int(val) if val == int(val) else val


def _bbox_center(bbox: list[list[float]]) -> tuple[float, float]:
    """Compute the center point of a bounding box.

    bbox format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    """
    if not bbox:
        return (0.0, 0.0)
    xs = [pt[0] for pt in bbox]
    ys = [pt[1] for pt in bbox]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


# ── OCR Engine ────────────────────────────────────────────────────────────────


class OCREngine:
    """Wrapper around PaddleOCR with graceful fallback to OpenCV-based OCR.

    The engine attempts to load PaddleOCR on demand. If unavailable,
    it falls back to a simple OpenCV-based text detection (EAST or
    contour-based) with basic recognition via template matching/clustering.

    For testing, a mock OCR result can be injected.
    """

    def __init__(self) -> None:
        self._paddle_ocr = None
        self._paddle_available: bool | None = None  # None = not yet checked

    @property
    def paddle_available(self) -> bool:
        """Check if PaddleOCR is importable (lazy check)."""
        if self._paddle_available is None:
            try:
                from paddleocr import PaddleOCR  # noqa: F401

                self._paddle_available = True
            except ImportError:
                logger.warning(
                    "PaddleOCR is not installed. "
                    "Install with: pip install paddleocr paddlepaddle"
                )
                self._paddle_available = False
        return self._paddle_available

    def _get_paddle_ocr(self) -> Any:
        """Lazy-initialize the PaddleOCR instance."""
        if self._paddle_ocr is None and self.paddle_available:
            try:
                from paddleocr import PaddleOCR

                self._paddle_ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    show_log=False,
                )
            except Exception as exc:
                logger.error("Failed to initialize PaddleOCR: %s", exc)
                self._paddle_available = False
        return self._paddle_ocr

    def run_ocr(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Run OCR on an image and return structured text blocks.

        Args:
            image: BGR or grayscale numpy array.

        Returns:
            List of dicts with keys: text, bbox, confidence
        """
        if self.paddle_available:
            return self._run_paddle_ocr(image)
        else:
            return self._run_opencv_ocr(image)

    def _run_paddle_ocr(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Run PaddleOCR on the image."""
        ocr = self._get_paddle_ocr()
        if ocr is None:
            return self._run_opencv_ocr(image)

        try:
            result = ocr.ocr(image, cls=True)
        except Exception as exc:
            logger.warning("PaddleOCR inference failed: %s", exc)
            return self._run_opencv_ocr(image)

        if not result or not result[0]:
            return []

        text_blocks = []
        for line in result[0]:
            bbox = line[0]  # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            text, confidence = line[1]
            text_blocks.append({
                "text": str(text).strip(),
                "bbox": bbox,
                "confidence": float(confidence),
            })

        return text_blocks

    def _run_opencv_ocr(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Fallback OCR using OpenCV contour analysis + connected component text detection.

        This is a simplified text detection that uses:
        1. MSER (Maximally Stable Extremal Regions) for text region detection
        2. Contour filtering to find text-like regions
        3. Bounding box extraction

        Note: This does NOT perform actual text recognition - it only detects
        text regions. For actual recognition, PaddleOCR or an external engine is needed.

        Returns empty text blocks (caller should check if OCR is available).
        """
        logger.info("OpenCV fallback OCR: detecting text regions (no recognition)")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        # Use MSER for text region detection
        try:
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
        except Exception:
            # Fallback: simple threshold-based region detection
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            regions = [c for c in contours if 20 < cv2.contourArea(c) < 5000]

        # Filter and group text regions
        text_blocks = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            # Filter by aspect ratio and size (likely text)
            if w < 5 or h < 5 or w > 200 or h > 80:
                continue
            aspect_ratio = w / h
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                continue

            # Compute bbox in standard format
            bbox = [
                [float(x), float(y)],
                [float(x + w), float(y)],
                [float(x + w), float(y + h)],
                [float(x), float(y + h)],
            ]

            text_blocks.append({
                "text": "",  # No recognition available in fallback
                "bbox": bbox,
                "confidence": 0.0,
            })

        return text_blocks


# ── DXF dimension extraction ──────────────────────────────────────────────────


def _extract_dxf_dimensions(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract dimension information from DXF entity data.

    DXF dimension entities contain measurement values (actual dimension),
    text overrides, and geometry (definition points).

    Returns:
        List of dimension dicts.
    """
    dimensions = []
    entities = data.get("entities", [])

    for entity in entities:
        etype = entity.get("type", "")
        if etype not in DXF_DIM_TYPES and etype != "DIMENSION":
            continue

        measurement = entity.get("measurement", 0.0)
        text = entity.get("text", "")
        if not text:
            text = str(measurement)

        # Determine dimension type
        if etype == "DIMRADIUS" or etype == "DIMENSION":
            dim_type = "radius"
        elif etype == "DIMDIAMETER":
            dim_type = "diameter"
        elif etype in ("DIMALIGNED", "DIMLINEAR"):
            dim_type = "linear"
        elif etype == "DIMANGULAR":
            dim_type = "angular"
        elif etype == "DIMORDINATE":
            dim_type = "coordinate"
        else:
            dim_type = "linear"

        value = _parse_number(text) if RE_DIMENSION_NUMBER.match(text) else measurement
        position = entity.get("position")

        entry: dict[str, Any] = {
            "value": value,
            "unit": "deg" if dim_type == "angular" else "mm",
            "position": {
                "x": round(float(position[0]), 2) if position else 0.0,
                "y": round(float(position[1]), 2) if position else 0.0,
            },
            "bbox": [],
            "confidence": 0.99,
            "type": dim_type,
            "text_raw": text,
            "tolerance": None,
            "layer": entity.get("layer"),
            "source": "dxf",
        }
        dimensions.append(entry)

    return dimensions


def _extract_dxf_text_labels(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract text and label entities from DXF data.

    Returns:
        List of label dicts.
    """
    labels = []
    entities = data.get("entities", [])

    for entity in entities:
        etype = entity.get("type", "")
        if etype not in ("TEXT", "MTEXT"):
            continue

        text = entity.get("text", "")
        position = entity.get("position")

        if not text:
            continue

        label_type = "annotation"
        if RE_SECTION_LABEL.match(text):
            label_type = "section_label"
        elif RE_FIT.match(text):
            label_type = "fit"

        labels.append({
            "text": text,
            "position": {
                "x": round(float(position[0]), 2) if position else 0.0,
                "y": round(float(position[1]), 2) if position else 0.0,
            },
            "bbox": [],
            "confidence": 0.99,
            "type": label_type,
            "layer": entity.get("layer"),
            "source": "dxf",
        })

    return labels


# ── PDF text extraction ───────────────────────────────────────────────────────


def _extract_pdf_annotations(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract annotations from PDF text content.

    Parses raw text from PDF pages into structured annotations.

    Returns:
        List of annotation dicts.
    """
    text_content = data.get("text_content", "")
    pages = data.get("pages", [])

    if not text_content:
        return []

    blocks = []
    lines = [l.strip() for l in text_content.split("\n") if l.strip()]

    for line in lines:
        # Try to parse as a dimension
        num_match = RE_DIMENSION_NUMBER.match(line)
        fit_match = RE_FIT.search(line)
        tol_match = RE_TOLERANCE.search(line)

        entry_type = "annotation"
        if fit_match or tol_match:
            entry_type = "tolerance"
        elif num_match:
            entry_type = "dimension"

        blocks.append({
            "text": line,
            "position": {"x": 0.0, "y": 0.0},  # PDF position unavailable from text extract
            "bbox": [],
            "confidence": 0.9,
            "type": entry_type,
            "source": "pdf",
        })

    return blocks


# ── Main Agent ────────────────────────────────────────────────────────────────


class OCRAnnotationAgent(BaseAgent):
    """Extracts textual annotations, dimensions, and tolerances from engineering
    drawings using OCR (PaddleOCR) or direct entity extraction (DXF/PDF).

    Supports:
        - Raster images: OCR via PaddleOCR (with MSER-based fallback)
        - DXF files: direct dimension + text entity extraction
        - PDF files: text content extraction + parsing
    """

    def __init__(self) -> None:
        super().__init__()
        self._ocr_engine: OCREngine | None = None

    @property
    def ocr_engine(self) -> OCREngine:
        """Lazy-initialize OCR engine."""
        if self._ocr_engine is None:
            self._ocr_engine = OCREngine()
        return self._ocr_engine

    def validate_input(self, input_data: Any) -> bool:
        """Validate that input has required keys."""
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
        """Extract structured annotations from a normalized document.

        Args:
            doc: Standardized document from InputProcessingAgent.
                Must contain 'format' and 'data' keys.
            **kwargs:
                - force_opencv: bool, force OpenCV fallback even if PaddleOCR is available

        Returns:
            Structured annotation dataset with keys:
                - dimensions: list of extracted dimension annotations
                - tolerances: list of tolerance annotations
                - labels: list of feature labels
                - text_blocks: raw OCR text blocks
                - raw_text: concatenated raw OCR text
                - ocr_engine: which engine was used ('paddle', 'opencv', 'dxf', 'pdf')
        """
        self.logger.info("Extracting annotations from %s", doc.get("format", "unknown"))

        doc_format = doc.get("format", "")
        doc_data = doc.get("data", {})

        # ── DXF path ──
        if doc_format == "dxf":
            self.logger.info("DXF input: extracting dimensions and text from entities")
            dims = _extract_dxf_dimensions(doc_data)
            labels = _extract_dxf_text_labels(doc_data)

            # Parse tolerances from dimension text
            all_blocks = []
            for d in dims:
                all_blocks.append({
                    "text": d["text_raw"],
                    "bbox": d["bbox"],
                    "confidence": d.get("confidence", 0.99),
                })
            for lbl in labels:
                all_blocks.append({
                    "text": lbl["text"],
                    "bbox": lbl["bbox"],
                    "confidence": lbl.get("confidence", 0.99),
                })

            parsed = parse_dimensions_from_text(all_blocks)

            # Merge DXF dims with parsed (DXF dims are authoritative)
            merged_dims = dims + parsed.get("dimensions", [])
            merged_tols = parsed.get("tolerances", [])
            merged_labels = labels + parsed.get("labels", [])

            raw_text = " ".join(b["text"] for b in all_blocks)

            return {
                "dimensions": merged_dims,
                "tolerances": merged_tols,
                "labels": merged_labels,
                "text_blocks": all_blocks,
                "raw_text": raw_text,
                "ocr_engine": "dxf",
            }

        # ── PDF path ──
        if doc_format == "pdf" and "image" not in doc_data:
            self.logger.info("PDF input: extracting annotations from text content")
            pdf_anns = _extract_pdf_annotations(doc_data)

            # Parse PDF annotations through dimension parser
            pdf_blocks = [
                {"text": a["text"], "bbox": a["bbox"], "confidence": a.get("confidence", 0.9)}
                for a in pdf_anns
            ]
            parsed = parse_dimensions_from_text(pdf_blocks)

            raw_text = doc_data.get("text_content", "")

            return {
                "dimensions": parsed.get("dimensions", []),
                "tolerances": parsed.get("tolerances", []),
                "labels": parsed.get("labels", []),
                "text_blocks": pdf_blocks,
                "raw_text": raw_text,
                "ocr_engine": "pdf_text",
            }

        # ── Raster image path ──
        # Get the image to run OCR on
        image = doc_data.get("image")
        if image is None:
            self.logger.warning("No image data available for OCR")
            return {
                "dimensions": [],
                "tolerances": [],
                "labels": [],
                "text_blocks": [],
                "raw_text": "",
                "ocr_engine": "none",
            }

        # Ensure image is BGR (3-channel)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        force_opencv = kwargs.get("force_opencv", False)

        # Run OCR
        if force_opencv:
            text_blocks = self.ocr_engine._run_opencv_ocr(image)
            engine_name = "opencv"
        else:
            # Try PaddleOCR, fall back to OpenCV
            text_blocks = self.ocr_engine.run_ocr(image)
            engine_name = "paddle" if self.ocr_engine.paddle_available else "opencv"

        # Parse OCR results into structured annotations
        parsed = parse_dimensions_from_text(text_blocks)
        raw_text = " ".join(b["text"] for b in text_blocks)

        # Also extract text from DXF if present in data (PDF-with-image embedding case)
        if doc_format == "pdf" and "text_content" in doc_data:
            pdf_text = doc_data.get("text_content", "")
            if pdf_text:
                raw_text = pdf_text + " " + raw_text

        self.logger.info(
            "OCR complete: %d dimensions, %d tolerances, %d labels (engine: %s)",
            len(parsed["dimensions"]),
            len(parsed["tolerances"]),
            len(parsed["labels"]),
            engine_name,
        )

        return {
            "dimensions": parsed["dimensions"],
            "tolerances": parsed["tolerances"],
            "labels": parsed["labels"],
            "text_blocks": text_blocks,
            "raw_text": raw_text,
            "ocr_engine": engine_name,
        }


# ── Utility (exported for testing) ────────────────────────────────────────────


def _merge_dimension_groups(
    dimensions: list[dict[str, Any]],
    max_group_distance: float = 30.0,
) -> list[dict[str, Any]]:
    """Merge nearby dimension annotations that likely belong to the same
    dimension line.

    When OCR splits a dimension into multiple fragments (e.g. "10" and "H7"),
    this merges them if they are within the specified pixel distance.

    Args:
        dimensions: List of dimension dicts.
        max_group_distance: Maximum pixel distance to consider same group.

    Returns:
        Merged dimension list.
    """
    if not dimensions:
        return []

    merged = []
    used = set()

    for i, d1 in enumerate(dimensions):
        if i in used:
            continue

        group = [d1]
        used.add(i)

        for j, d2 in enumerate(dimensions):
            if j in used or i == j:
                continue

            dx = d1["position"]["x"] - d2["position"]["x"]
            dy = d1["position"]["y"] - d2["position"]["y"]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist <= max_group_distance:
                group.append(d2)
                used.add(j)

        if len(group) > 1:
            # Merge: concatenate text, average position
            merged_text = " ".join(d["text_raw"] for d in group)
            avg_x = sum(d["position"]["x"] for d in group) / len(group)
            avg_y = sum(d["position"]["y"] for d in group) / len(group)
            avg_conf = sum(d["confidence"] for d in group) / len(group)

            merged.append({
                "value": group[0]["value"],
                "unit": group[0]["unit"],
                "position": {"x": round(avg_x, 2), "y": round(avg_y, 2)},
                "bbox": group[0].get("bbox", []),
                "confidence": round(avg_conf, 4),
                "type": group[0]["type"],
                "text_raw": merged_text,
                "tolerance": None,
            })
        else:
            merged.append(d1)

    return merged
