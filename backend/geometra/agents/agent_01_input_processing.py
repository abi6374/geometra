"""Agent 1: Input Processing Agent.

Responsibilities:
- File validation (exists, size, supported format)
- Format identification (by extension + content sniffing)
- Metadata extraction (file system + format-specific)
- File normalization into a standardized document representation
- Load data for 2D formats (PDF, PNG, JPEG, TIFF, DXF) and 3D formats (STEP, STP, IGES, STL, OBJ)

Libraries: OpenCV (cv2), pdfplumber, ezdxf, Pillow, filetype
"""
from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Any

import cv2
from geometra.agents.base import BaseAgent
from geometra.config import settings

logger = logging.getLogger(__name__)

# ── Format constants ──────────────────────────────────────────────────────────

RASTER_2D_FORMATS = {"png", "jpeg", "jpg", "tiff", "tif"}
VECTOR_2D_FORMATS = {"pdf", "dxf", "dwg"}
_3D_FORMATS = {"step", "stp", "iges", "stl", "obj"}
ALL_2D_FORMATS = RASTER_2D_FORMATS | VECTOR_2D_FORMATS
ALL_SUPPORTED = ALL_2D_FORMATS | _3D_FORMATS


def _detect_format_by_content(path: Path) -> str | None:
    """Detect file format by reading magic bytes.

    Returns a format string (e.g. 'png', 'pdf', 'dxf') or None if unknown.
    """
    try:
        import filetype as ft

        kind = ft.guess(str(path))
        if kind is not None:
            mime = kind.mime.lower()
            ext = kind.extension.lower()
            # Map common MIME types back to our format names
            mime_to_format = {
                "application/pdf": "pdf",
                "image/png": "png",
                "image/jpeg": "jpeg",
                "image/tiff": "tiff",
                "image/vnd.dxf": "dxf",
                "application/dxf": "dxf",
                "model/stl": "stl",
                "application/vnd.ms-pki.stl": "stl",
            }
            if ext in ALL_SUPPORTED:
                return ext
            if mime in mime_to_format:
                return mime_to_format[mime]
    except ImportError:
        pass
    except Exception:
        logger.debug("filetype detection failed for %s", path, exc_info=True)

    # Fallback: manual magic byte checks
    return _manual_format_detect(path)


def _manual_format_detect(path: Path) -> str | None:
    """Manual magic-byte-based format detection fallback."""
    try:
        with open(path, "rb") as f:
            header = f.read(16)

        # PDF: starts with %PDF
        if header.startswith(b"%PDF"):
            return "pdf"

        # PNG: 8-byte signature
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            return "png"

        # JPEG: starts with FF D8 FF
        if header[:3] == b"\xff\xd8\xff":
            return "jpeg"

        # TIFF: starts with II (little-endian) or MM (big-endian)
        if header[:2] in (b"II", b"MM"):
            return "tiff"

        # DXF: starts with "  " or "0\nSECTION" or "0\nHEADER"
        if header.startswith(b"  ") or header.startswith(b"0\n"):
            text = header.decode("ascii", errors="ignore")
            if "SECTION" in text or "HEADER" in text or "ENTITIES" in text:
                return "dxf"

        # STEP/STP: starts with ISO-10303-21
        if header.startswith(b"ISO-10303-21"):
            return "step"

        # STL (ASCII): starts with "solid "
        if header.startswith(b"solid "):
            return "stl"

        # OBJ: starts with "# " or "v " or "vn " or "f "
        try:
            text = header.decode("ascii", errors="ignore")
            if any(text.startswith(prefix) for prefix in ("# ", "v ", "vn ", "vt ", "f ", "g ", "usemtl")):
                return "obj"
        except Exception:
            pass

    except OSError:
        pass

    # ── Checks requiring larger reads ──────────────────────────────────────────
    try:
        with open(path, "rb") as f:
            header_full = f.read(256)

        # IGES: first line ends with 'S' in column 73, or starts with 'S' in column 1
        # IGES uses 80-character fixed-width records. Column 73 is the section letter.
        try:
            first_line = header_full[:80].decode("ascii", errors="ignore")
            if len(first_line.rstrip()) >= 72 and first_line[72:73] in ("S", "G", "D", "P", "T"):
                return "iges"
        except Exception:
            pass

        # STL (binary): first 80 bytes are header, next 4 bytes = uint32 triangle count
        if len(header_full) >= 84:
            try:
                tri_count = struct.unpack_from("<I", header_full, 80)[0]
                # Reasonable triangle count for a real STL
                if 0 < tri_count < 10_000_000:
                    return "stl"
            except Exception:
                pass

    except OSError:
        pass

    return None


def _check_dwg(path: Path) -> bool:
    """Check if a file is a DWG by magic bytes."""
    try:
        with open(path, "rb") as f:
            header = f.read(12)
        # DWG files start with ACXXXX (e.g. AC1015, AC1027)
        return header.startswith(b"AC") and header[2:6].isdigit()
    except OSError:
        return False


# ── Image normalization helpers ───────────────────────────────────────────────


def _load_image(path: Path) -> dict[str, Any]:
    """Load a raster image and return normalized image data.

    Returns:
        dict with keys: image (np.ndarray BGR), gray (np.ndarray), dimensions (hwc)
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        # Try grayscale
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"OpenCV could not read image: {path}")
        gray = img
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    return {
        "image": img,
        "gray": gray,
        "width": w,
        "height": h,
        "channels": img.shape[2] if img.ndim == 3 else 1,
    }


def _load_pdf(path: Path) -> dict[str, Any]:
    """Load a PDF and extract pages, text, metadata, and embedded images.

    Returns:
        dict with keys: metadata, pages, text_content, embedded_images
    """
    import pdfplumber

    with pdfplumber.open(str(path)) as pdf:
        metadata = dict(pdf.metadata) if pdf.metadata else {}

        pages = []
        text_content = []
        embedded_images = []

        for page in pdf.pages:
            page_info = {
                "page_number": page.page_number,
                "width": float(page.width),
                "height": float(page.height),
            }
            pages.append(page_info)

            text = page.extract_text() or ""
            text_content.append(text)

            for img in page.images:
                stream = img.get("stream")
                if stream:
                    img_bytes = stream.get_data()
                    embedded_images.append({
                        "page": page.page_number,
                        "x0": float(img.get("x0", 0)),
                        "y0": float(img.get("y0", 0)),
                        "x1": float(img.get("x1", 0)),
                        "y1": float(img.get("y1", 0)),
                        "size_bytes": len(img_bytes),
                    })

        return {
            "metadata": metadata,
            "pages": pages,
            "text_content": "\n".join(text_content),
            "embedded_images": embedded_images,
            "page_count": len(pages),
        }


def _load_dxf(path: Path) -> dict[str, Any]:
    """Load a DXF file and extract entities, layers, and header info.

    Returns:
        dict with keys: dxf_version, entities, entity_counts, layers, header_vars
    """
    import ezdxf

    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()

    entities = []
    entity_counts: dict[str, int] = {}

    for entity in msp:
        entity_type = entity.dxftype()
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

        base = {
            "type": entity_type,
            "layer": entity.dxf.layer,
            "color": entity.dxf.color,
            "linetype": entity.dxf.linetype,
        }

        if entity_type == "LINE":
            entities.append({
                **base,
                "start": (float(entity.dxf.start.x), float(entity.dxf.start.y)),
                "end": (float(entity.dxf.end.x), float(entity.dxf.end.y)),
            })
        elif entity_type == "CIRCLE":
            entities.append({
                **base,
                "center": (float(entity.dxf.center.x), float(entity.dxf.center.y)),
                "radius": float(entity.dxf.radius),
            })
        elif entity_type == "ARC":
            entities.append({
                **base,
                "center": (float(entity.dxf.center.x), float(entity.dxf.center.y)),
                "radius": float(entity.dxf.radius),
                "start_angle": float(entity.dxf.start_angle),
                "end_angle": float(entity.dxf.end_angle),
            })
        elif entity_type == "LWPOLYLINE":
            pts = [(float(p[0]), float(p[1])) for p in entity.get_points()]
            entities.append({**base, "points": pts, "closed": entity.closed})
        elif entity_type == "TEXT":
            entities.append({
                **base,
                "text": entity.dxf.text,
                "position": (float(entity.dxf.insert.x), float(entity.dxf.insert.y)),
                "height": float(entity.dxf.height),
            })
        elif entity_type == "MTEXT":
            entities.append({
                **base,
                "text": entity.text,
                "position": (float(entity.dxf.insert.x), float(entity.dxf.insert.y)),
            })
        elif entity_type == "DIMENSION":
            entities.append({
                **base,
                "text": entity.dxf.text if hasattr(entity.dxf, "text") else "",
                "measurement": float(entity.get_measurement()) if hasattr(entity, "get_measurement") else 0.0,
            })
        elif entity_type == "INSERT":
            entities.append({
                **base,
                "block_name": entity.dxf.name if hasattr(entity.dxf, "name") else "",
                "position": (float(entity.dxf.insert.x), float(entity.dxf.insert.y)),
            })
        else:
            # Generic entity
            entities.append(base)

    # Layer info
    layers = {}
    for layer in doc.layers:
        layers[layer.dxf.name] = {
            "color": layer.color,
            "linetype": layer.dxf.linetype,
            "frozen": layer.is_frozen(),
            "on": layer.is_on(),
        }

    # Gather known header variables
    header_vars: dict[str, str] = {}
    known_vars = ["$ACADVER", "$INSBASE", "$EXTMIN", "$EXTMAX", "$LIMMIN", "$LIMMAX"]
    for var in known_vars:
        try:
            val = doc.header.get(var)
            if val is not None:
                header_vars[var] = str(val)
        except Exception:
            pass

    return {
        "dxf_version": doc.dxfversion,
        "entities": entities,
        "entity_counts": entity_counts,
        "entity_total": len(entities),
        "layers": layers,
        "header_vars": header_vars,
    }


def _extract_3d_metadata(path: Path, detected_format: str) -> dict[str, Any]:
    """Extract basic metadata from 3D format files without full parsing."""
    info: dict[str, Any] = {"format": detected_format}

    try:
        stat = path.stat()
        info["size_bytes"] = stat.st_size
    except OSError:
        pass

    # Read first few lines for text-based formats
    if detected_format in {"step", "stp", "iges", "obj"}:
        try:
            with open(path, "r", errors="ignore") as f:
                head = [f.readline().strip() for _ in range(10)]
            info["header_lines"] = [h for h in head if h]
        except Exception:
            pass

    if detected_format in {"stl"}:
        # Check if ASCII or binary STL
        try:
            with open(path, "rb") as f:
                h = f.read(6)
            info["stl_type"] = "ascii" if h.startswith(b"solid ") else "binary"
        except Exception:
            pass

    return info


# ── Main agent ────────────────────────────────────────────────────────────────


class InputProcessingAgent(BaseAgent):
    """Validates, identifies, and normalizes input files into a standardized
    document representation for downstream agents.

    Supports:
        - 2D raster: PNG, JPEG, TIFF (via OpenCV)
        - 2D vector: PDF (via pdfplumber), DXF (via ezdxf), DWG (detection only)
        - 3D models: STEP, STP, IGES, STL, OBJ (metadata + format detection)
    """

    def validate_input(self, input_data: Any) -> bool:
        if isinstance(input_data, (str, Path)):
            path = Path(str(input_data))
            return path.exists()
        if isinstance(input_data, bytes):
            return len(input_data) > 0
        return hasattr(input_data, "read")

    def process(self, input_data: Any, **kwargs: Any) -> dict[str, Any]:
        """Process an input file and return a standardized document.

        Args:
            input_data: Path to the input file (str or Path), bytes, or file-like object.
            **kwargs: Additional options:
                - max_size_mb: override max file size (default from settings)

        Returns:
            Standardized document representation with keys:
                - file_path: original path (or temp path for bytes/streams)
                - format: detected format (pdf, png, dxf, step, etc.)
                - detected_by: how format was determined ('extension' | 'content')
                - metadata: extracted metadata dict
                - is_3d: bool indicating if it's a 3D format
                - normalized_path: path to normalized copy
                - data: format-specific loaded data (images, pages, entities, etc.)
        """
        # Resolve input to a Path
        file_path, is_temp = self._resolve_input(input_data, **kwargs)
        self.logger.info("Processing input file: %s", file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        # Size validation
        max_bytes = (kwargs.get("max_size_mb") or settings.max_file_size_mb) * 1024 * 1024
        file_size = file_path.stat().st_size
        if file_size > max_bytes:
            raise ValueError(
                f"File exceeds size limit of {max_bytes // (1024*1024)} MB "
                f"(got {file_size / (1024*1024):.1f} MB)"
            )

        # Detect format: try extension first, then content sniffing
        ext = file_path.suffix.lower().lstrip(".")
        detected_format = ext if ext in ALL_SUPPORTED else None
        detected_by = "extension"

        if not detected_format:
            ext_by_content = _detect_format_by_content(file_path)
            if ext_by_content:
                detected_format = ext_by_content
                detected_by = "content"
            elif _check_dwg(file_path):
                detected_format = "dwg"
                detected_by = "content"

        if not detected_format:
            raise ValueError(
                f"Unsupported or unrecognized file format: {file_path.name}. "
                f"Supported formats: {', '.join(sorted(ALL_SUPPORTED))}"
            )

        # Normalize extension
        detected_format = detected_format.lower()
        if detected_format in {"jpg", "tif", "stp"}:
            detected_format = {"jpg": "jpeg", "tif": "tiff", "stp": "step"}.get(detected_format, detected_format)

        is_3d = detected_format in _3D_FORMATS

        # File-system metadata
        metadata = self._extract_metadata(file_path)
        metadata["detected_format"] = detected_format
        metadata["detected_by"] = detected_by

        # Format-specific loading
        data: dict[str, Any] = {}

        if detected_format in RASTER_2D_FORMATS:
            try:
                data = _load_image(file_path)
                metadata["width"] = data.get("width")
                metadata["height"] = data.get("height")
                metadata["channels"] = data.get("channels")
            except Exception as exc:
                self.logger.warning("Could not load image data: %s", exc)
                data = {"error": str(exc)}

        elif detected_format == "pdf":
            try:
                data = _load_pdf(file_path)
                metadata["page_count"] = data.get("page_count", 0)
            except Exception as exc:
                self.logger.warning("Could not load PDF data: %s", exc)
                data = {"error": str(exc)}

        elif detected_format == "dxf" or detected_format == "dwg":
            if detected_format == "dxf":
                try:
                    data = _load_dxf(file_path)
                    metadata["dxf_version"] = data.get("dxf_version")
                    metadata["entity_total"] = data.get("entity_total", 0)
                except Exception as exc:
                    self.logger.warning("Could not load DXF data: %s", exc)
                    data = {"error": str(exc)}
            else:
                # DWG: no full parser, just metadata
                data = {"note": "DWG format detected; full parsing not yet implemented"}

        elif detected_format in _3D_FORMATS:
            data = _extract_3d_metadata(file_path, detected_format)
            # For STL, also attempt trimesh loading if available
            if detected_format in {"stl", "obj"}:
                try:
                    import trimesh

                    mesh = trimesh.load(str(file_path))
                    data["mesh_info"] = {
                        "vertices": len(mesh.vertices) if hasattr(mesh, "vertices") else 0,
                        "faces": len(mesh.faces) if hasattr(mesh, "faces") else 0,
                        "bounds": list(mesh.bounds.flatten()) if hasattr(mesh, "bounds") and mesh.bounds is not None else [],
                    }
                except ImportError:
                    pass
                except Exception as exc:
                    self.logger.debug("trimesh could not load %s: %s", detected_format, exc)

        result = {
            "file_path": str(file_path),
            "format": detected_format,
            "detected_by": detected_by,
            "metadata": metadata,
            "is_3d": is_3d,
            "normalized_path": str(file_path),
            "data": data,
        }

        # Clean up temp file if we created one
        if is_temp:
            self.logger.debug("Temporary file: %s (will be cleaned up externally)", file_path)

        return result

    def _resolve_input(self, input_data: Any, **kwargs: Any) -> tuple[Path, bool]:
        """Convert input into a Path. Returns (path, is_temp_file).

        Accepts:
            - str or Path: used directly
            - bytes: written to a temp file
            - file-like object: read and written to a temp file
        """
        import tempfile

        if isinstance(input_data, (str, Path)):
            return Path(str(input_data)), False

        suffix = kwargs.get("suffix", "")

        if isinstance(input_data, bytes):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(input_data)
            tmp.close()
            return Path(tmp.name), True

        if hasattr(input_data, "read"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(input_data.read())
            tmp.close()
            return Path(tmp.name), True

        raise TypeError(f"Unsupported input type: {type(input_data).__name__}")

    def _extract_metadata(self, path: Path) -> dict[str, Any]:
        """Extract detailed file-system metadata."""
        stat = path.stat()
        return {
            "filename": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 3),
            "modified_time": stat.st_mtime,
        }
