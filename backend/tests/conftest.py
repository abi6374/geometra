"""Test fixtures: generate sample files of each supported format for testing."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="session")
def sample_dir() -> Generator[Path, None, None]:
    """Create a temporary directory to hold all sample files."""
    tmp = tempfile.mkdtemp(prefix="geometra_test_")
    yield Path(tmp)


@pytest.fixture(scope="session")
def sample_png(sample_dir: Path) -> Path:
    """Generate a simple 100x100 PNG image with some geometric shapes."""
    path = sample_dir / "test_drawing.png"
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255  # white

    # Draw a black rectangle
    cv2.rectangle(img, (20, 20), (180, 180), (0, 0, 0), 2)
    # Draw a circle
    cv2.circle(img, (100, 100), 50, (0, 0, 0), 2)
    # Draw a line
    cv2.line(img, (20, 100), (180, 100), (0, 0, 0), 2)

    cv2.imwrite(str(path), img)
    assert path.exists()
    return path


@pytest.fixture(scope="session")
def sample_jpeg(sample_dir: Path) -> Path:
    """Generate a simple 100x100 JPEG image."""
    path = sample_dir / "test_drawing.jpeg"
    img = np.ones((100, 100, 3), dtype=np.uint8) * 200
    cv2.rectangle(img, (10, 10), (90, 90), (50, 50, 50), 1)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert path.exists()
    return path


@pytest.fixture(scope="session")
def sample_tiff(sample_dir: Path) -> Path:
    """Generate a simple TIFF image."""
    path = sample_dir / "test_drawing.tiff"
    img = np.ones((100, 100), dtype=np.uint8) * 240
    cv2.rectangle(img, (10, 10), (90, 90), 30, 1)
    cv2.imwrite(str(path), img)
    assert path.exists()
    return path


@pytest.fixture(scope="session")
def sample_pdf(sample_dir: Path) -> Path:
    """Generate a minimal valid PDF with a single page."""
    path = sample_dir / "test_drawing.pdf"

    # Minimal valid PDF content
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n"
        b"<< /Type /Catalog /Pages 2 0 R >>\n"
        b"endobj\n"
        b"2 0 obj\n"
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n"
        b"endobj\n"
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
        b"   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n"
        b"endobj\n"
        b"4 0 obj\n"
        b"<< /Length 44 >>\n"
        b"stream\n"
        b"BT /F1 12 Tf 100 700 Td (Test) Tj ET\n"
        b"endstream\n"
        b"endobj\n"
        b"5 0 obj\n"
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n"
        b"endobj\n"
        b"xref\n"
        b"0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000363 00000 n \n"
        b"trailer\n"
        b"<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n"
        b"448\n"
        b"%%EOF\n"
    )

    path.write_bytes(pdf_content)
    assert path.exists()
    return path


@pytest.fixture(scope="session")
def sample_dxf(sample_dir: Path) -> Path:
    """Generate a minimal DXF file with a line and a circle."""
    path = sample_dir / "test_drawing.dxf"

    # Minimal DXF R12 content
    dxf_content = (
        "  0\nSECTION\n  2\nHEADER\n  9\n$ACADVER\n  1\nAC1009\n  0\nENDSEC\n"
        "  0\nSECTION\n  2\nENTITIES\n"
        "  0\nLINE\n  8\n0\n 10\n0.0\n 20\n0.0\n 11\n100.0\n 21\n100.0\n"
        "  0\nCIRCLE\n  8\n0\n 10\n50.0\n 20\n50.0\n 40\n25.0\n"
        "  0\nENDSEC\n  0\nEOF\n"
    )

    path.write_text(dxf_content)
    assert path.exists()
    return path


@pytest.fixture(scope="session")
def sample_step(sample_dir: Path) -> Path:
    """Generate a minimal STEP file (ISO 10303-21)."""
    path = sample_dir / "test_model.stp"

    step_content = (
        "ISO-10303-21;\n"
        "HEADER;\n"
        "FILE_DESCRIPTION(('Test Model'), '2;1');\n"
        "FILE_NAME('test_model.stp', '2024-01-01T00:00:00', (''), (''), '', '', '');\n"
        "FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));\n"
        "ENDSEC;\n"
        "DATA;\n"
        "#10 = CARTESIAN_POINT('Origin', (0., 0., 0.));\n"
        "#20 = DIRECTION('Axis', (1., 0., 0.));\n"
        "#30 = VECTOR('X-Axis', #20, 100.);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n"
    )

    path.write_text(step_content)
    assert path.exists()
    return path


@pytest.fixture(scope="session")
def sample_stl_ascii(sample_dir: Path) -> Path:
    """Generate a minimal ASCII STL file."""
    path = sample_dir / "test_model.stl"

    stl_content = (
        "solid test_model\n"
        "  facet normal 0 0 1\n"
        "    outer loop\n"
        "      vertex 0 0 0\n"
        "      vertex 1 0 0\n"
        "      vertex 0 1 0\n"
        "    endloop\n"
        "  endfacet\n"
        "  facet normal 0 0 1\n"
        "    outer loop\n"
        "      vertex 1 0 0\n"
        "      vertex 1 1 0\n"
        "      vertex 0 1 0\n"
        "    endloop\n"
        "  endfacet\n"
        "endsolid test_model\n"
    )

    path.write_text(stl_content)
    assert path.exists()
    return path


@pytest.fixture(scope="session")
def sample_stl_binary(sample_dir: Path) -> Path:
    """Generate a minimal binary STL file."""
    path = sample_dir / "test_binary.stl"

    # Binary STL: 80-byte header + 4-byte triangle count + triangles
    # Two triangles forming a square
    # 80-byte header: short ASCII label + padding
    header_label = b"Binary STL Test"  # 15 bytes
    header = header_label + b"\x00" * (80 - len(header_label))  # 80 bytes total
    tri_count = struct.pack("<I", 2)  # 2 triangles

    # Each triangle: 12 bytes normal + 12*3 bytes vertices + 2 bytes attribute
    # Triangle 1: normal (0,0,1), vertices (0,0,0), (1,0,0), (0,1,0)
    tri1 = struct.pack("<fff", 0, 0, 1)  # normal
    tri1 += struct.pack("<fff", 0, 0, 0)  # v1
    tri1 += struct.pack("<fff", 1, 0, 0)  # v2
    tri1 += struct.pack("<fff", 0, 1, 0)  # v3
    tri1 += struct.pack("<H", 0)  # attribute

    # Triangle 2: normal (0,0,1), vertices (1,0,0), (1,1,0), (0,1,0)
    tri2 = struct.pack("<fff", 0, 0, 1)  # normal
    tri2 += struct.pack("<fff", 1, 0, 0)  # v1
    tri2 += struct.pack("<fff", 1, 1, 0)  # v2
    tri2 += struct.pack("<fff", 0, 1, 0)  # v3
    tri2 += struct.pack("<H", 0)  # attribute

    data = header + tri_count + tri1 + tri2
    path.write_bytes(data)
    assert path.exists()
    return path


@pytest.fixture(scope="session")
def sample_obj(sample_dir: Path) -> Path:
    """Generate a minimal OBJ file."""
    path = sample_dir / "test_model.obj"

    obj_content = (
        "# Simple test model\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "v 1 1 0\n"
        "f 1 2 3\n"
        "f 2 4 3\n"
    )

    path.write_text(obj_content)
    assert path.exists()
    return path


@pytest.fixture(scope="session")
def sample_iges(sample_dir: Path) -> Path:
    """Generate a minimal IGES file."""
    path = sample_dir / "test_model.iges"

    # IGES format: 80-char fixed-width lines
    lines = [
        "                                                                        S      1",
        "1H,,1H;,4HTest,16HTest IGES Model,32HMinimal IGES for testing,G,1,1,5,0,0, G      2",
        "1,1.0,1,2,1,15,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, G      3",
        "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0;                                             G      4",
        "     110,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0, D      1",
        "     110,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0, D      2",
        "110,0.0,0.0,0.0,1.0,1.0,1.0;                                                  P      1",
        "S0000001G0000004D0000002P0000001                                              T      1",
    ]

    path.write_text("\n".join(lines))
    assert path.exists()
    return path


@pytest.fixture(scope="session")
def unsupported_format(sample_dir: Path) -> Path:
    """Generate a file with an unsupported format (.txt)."""
    path = sample_dir / "notes.txt"
    path.write_text("This is not a supported engineering drawing format.")
    return path


# ── Synthetic engineering drawings with dimension annotations ────────────────


@pytest.fixture(scope="session")
def synthetic_annotation_drawing(sample_dir: Path) -> Path:
    """Create a synthetic engineering drawing with dimension annotations rendered as text.

    Contains:
        - A rectangular enclosure (50,50) to (250,200)
        - A hole (circle) at (100, 120), radius 30
        - A hole (circle) at (200, 120), radius 20
        - Width dimension "200" above the rectangle
        - Height dimension "150" to the right of the rectangle
        - Diameter annotation "⌀60" near the first hole
        - Diameter annotation "⌀40" near the second hole
        - Tolerance annotation "±0.5" near dimensions
        - Fit annotation "H7" near the first hole
        - Section label "A-A" at the bottom
    """
    width, height = 400, 350
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Rectangle (enclosure)
    cv2.rectangle(img, (50, 50), (250, 200), (0, 0, 0), 2)

    # Hole 1 (circle)
    cv2.circle(img, (100, 120), 30, (0, 0, 0), 2)
    # Center mark for hole 1
    cv2.line(img, (90, 120), (110, 120), (0, 0, 0), 1)
    cv2.line(img, (100, 110), (100, 130), (0, 0, 0), 1)

    # Hole 2 (circle)
    cv2.circle(img, (200, 120), 20, (0, 0, 0), 2)
    # Center mark for hole 2
    cv2.line(img, (192, 120), (208, 120), (0, 0, 0), 1)
    cv2.line(img, (200, 112), (200, 128), (0, 0, 0), 1)

    # Width dimension "200" above rectangle
    cv2.line(img, (50, 35), (250, 35), (0, 0, 0), 1)  # dimension line
    cv2.line(img, (50, 30), (50, 50), (0, 0, 0), 1)  # extension line left
    cv2.line(img, (250, 30), (250, 50), (0, 0, 0), 1)  # extension line right
    cv2.putText(img, "200", (130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Height dimension "150" to the right
    cv2.line(img, (260, 50), (260, 200), (0, 0, 0), 1)  # dimension line
    cv2.line(img, (250, 50), (270, 50), (0, 0, 0), 1)  # extension line top
    cv2.line(img, (250, 200), (270, 200), (0, 0, 0), 1)  # extension line bottom
    cv2.putText(img, "150", (265, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Diameter annotation near hole 1
    cv2.putText(img, "⌀60", (60, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Diameter annotation near hole 2
    cv2.putText(img, "⌀40", (175, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Tolerance annotation
    cv2.putText(img, "±0.5", (290, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # Fit annotation near hole 1
    cv2.putText(img, "H7", (70, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # Section label
    cv2.putText(img, "A-A", (180, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    # Some additional annotations
    cv2.putText(img, "SECTION", (170, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    path = sample_dir / "synthetic_annotation.png"
    cv2.imwrite(str(path), img)
    assert path.exists()
    return path


@pytest.fixture(scope="session")
def synthetic_dimension_drawing(sample_dir: Path) -> Path:
    """A simpler synthetic drawing focused on dimensions only.

    Contains:
        - A simple rectangle (50,50) to (150, 100)
        - Width label "100"
        - Height label "50"
        - Radius annotation "R25" near a radius
        - Angular annotation "90°"
        - Coordinate label "X:100 Y:50"
    """
    width, height = 300, 200
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Rectangle
    cv2.rectangle(img, (50, 50), (150, 100), (0, 0, 0), 2)

    # Circle
    cv2.circle(img, (100, 150), 25, (0, 0, 0), 2)

    # Width dimension
    cv2.line(img, (50, 35), (150, 35), (0, 0, 0), 1)
    cv2.line(img, (50, 35), (50, 50), (0, 0, 0), 1)
    cv2.line(img, (150, 35), (150, 50), (0, 0, 0), 1)
    cv2.putText(img, "100", (85, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Height dimension
    cv2.line(img, (160, 50), (160, 100), (0, 0, 0), 1)
    cv2.line(img, (150, 50), (170, 50), (0, 0, 0), 1)
    cv2.line(img, (150, 100), (170, 100), (0, 0, 0), 1)
    cv2.putText(img, "50", (165, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Radius annotation
    cv2.putText(img, "R25", (75, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # Angular annotation
    cv2.putText(img, "90°", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # Coordinate label
    cv2.putText(img, "X:200 Y:150", (190, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

    path = sample_dir / "synthetic_dimensions.png"
    cv2.imwrite(str(path), img)
    assert path.exists()
    return path


@pytest.fixture(scope="session")
def synthetic_complex_drawing(sample_dir: Path) -> Path:
    """A more complex engineering drawing with multiple annotated features.

    Simulates an electrical enclosure front panel with:
        - Outer rectangle (enclosure boundary)
        - Cutout for display (inner rectangle)
        - Several mounting holes
        - Vent slots
        - Dimensions, tolerances, and labels
    """
    width, height = 500, 400
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Outer boundary
    cv2.rectangle(img, (30, 30), (470, 370), (0, 0, 0), 2)

    # Display cutout
    cv2.rectangle(img, (150, 80), (350, 200), (0, 0, 0), 1)

    # Mounting holes in corners
    hole_positions = [(60, 60), (440, 60), (60, 340), (440, 340)]
    for hx, hy in hole_positions:
        cv2.circle(img, (hx, hy), 8, (0, 0, 0), 1)
        cv2.circle(img, (hx, hy), 6, (0, 0, 0), -1)  # filled

    # Mounting holes around display
    hole_positions2 = [(150, 140), (350, 140)]
    for hx, hy in hole_positions2:
        cv2.circle(img, (hx, hy), 5, (0, 0, 0), 1)
        cv2.circle(img, (hx, hy), 3, (0, 0, 0), -1)

    # Vent slots (horizontal lines)
    for vy in range(250, 350, 20):
        cv2.line(img, (100, vy), (200, vy), (0, 0, 0), 1)

    # Overall width dimension "440"
    cv2.line(img, (30, 15), (470, 15), (0, 0, 0), 1)
    cv2.line(img, (30, 15), (30, 30), (0, 0, 0), 1)
    cv2.line(img, (470, 15), (470, 30), (0, 0, 0), 1)
    cv2.putText(img, "440", (230, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Overall height dimension "340"
    cv2.line(img, (480, 30), (480, 370), (0, 0, 0), 1)
    cv2.line(img, (470, 30), (490, 30), (0, 0, 0), 1)
    cv2.line(img, (470, 370), (490, 370), (0, 0, 0), 1)
    cv2.putText(img, "340", (485, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Hole diameter annotations
    cv2.putText(img, "⌀12", (45, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "⌀12", (425, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

    # Display cutout dimension
    cv2.putText(img, "200", (240, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

    # Tolerance on display cutout
    cv2.putText(img, "±0.2", (240, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

    # Section labels
    cv2.putText(img, "B-B", (400, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Material / feature labels
    cv2.putText(img, "VENT", (120, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "DISPLAY", (230, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

    path = sample_dir / "synthetic_complex.png"
    cv2.imwrite(str(path), img)
    assert path.exists()
    return path


@pytest.fixture(scope="session")
def blank_drawing(sample_dir: Path) -> Path:
    """A blank image with no annotations."""
    width, height = 200, 200
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    path = sample_dir / "blank.png"
    cv2.imwrite(str(path), img)
    assert path.exists()
    return path
