"""Unit tests for the Input Processing Agent (Agent 1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geometra.agents.agent_01_input_processing import (
    InputProcessingAgent,
    _detect_format_by_content,
    _manual_format_detect,
    _load_image,
    _load_pdf,
    _load_dxf,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def agent() -> InputProcessingAgent:
    return InputProcessingAgent()


# ── Basic validation ──────────────────────────────────────────────────────────


class TestInputValidation:
    def test_validate_input_with_path(self, agent: InputProcessingAgent, sample_png: Path) -> None:
        assert agent.validate_input(sample_png) is True

    def test_validate_input_with_string(self, agent: InputProcessingAgent, sample_png: Path) -> None:
        assert agent.validate_input(str(sample_png)) is True

    def test_validate_input_with_bytes(self, agent: InputProcessingAgent) -> None:
        assert agent.validate_input(b"test data") is True

    def test_validate_input_nonexistent_path(self, agent: InputProcessingAgent) -> None:
        assert agent.validate_input("/nonexistent/file.step") is False

    def test_validate_input_none(self, agent: InputProcessingAgent) -> None:
        assert agent.validate_input(None) is False

    def test_validate_input_empty_bytes(self, agent: InputProcessingAgent) -> None:
        assert agent.validate_input(b"") is False


# ── Format Detection ──────────────────────────────────────────────────────────


class TestFormatDetection:
    def test_detect_png_by_content(self, sample_png: Path) -> None:
        fmt = _detect_format_by_content(sample_png)
        assert fmt == "png", f"Expected 'png', got '{fmt}'"

    def test_detect_jpeg_by_content(self, sample_jpeg: Path) -> None:
        fmt = _detect_format_by_content(sample_jpeg)
        # filetype library returns 'jpg' (short form); process() normalizes it
        assert fmt in ("jpeg", "jpg"), f"Expected 'jpeg' or 'jpg', got '{fmt}'"

    def test_detect_tiff_by_content(self, sample_tiff: Path) -> None:
        fmt = _detect_format_by_content(sample_tiff)
        # filetype library returns 'tif' (short form); process() normalizes it
        assert fmt in ("tiff", "tif"), f"Expected 'tiff' or 'tif', got '{fmt}'"

    def test_detect_pdf_by_content(self, sample_pdf: Path) -> None:
        fmt = _detect_format_by_content(sample_pdf)
        assert fmt == "pdf", f"Expected 'pdf', got '{fmt}'"

    def test_detect_dxf_by_content(self, sample_dxf: Path) -> None:
        fmt = _detect_format_by_content(sample_dxf)
        assert fmt == "dxf", f"Expected 'dxf', got '{fmt}'"

    def test_detect_step_by_content(self, sample_step: Path) -> None:
        fmt = _detect_format_by_content(sample_step)
        assert fmt == "step", f"Expected 'step', got '{fmt}'"

    def test_detect_stl_ascii_by_content(self, sample_stl_ascii: Path) -> None:
        fmt = _detect_format_by_content(sample_stl_ascii)
        assert fmt == "stl", f"Expected 'stl', got '{fmt}'"

    def test_detect_stl_binary_by_content(self, sample_stl_binary: Path) -> None:
        fmt = _detect_format_by_content(sample_stl_binary)
        assert fmt == "stl", f"Expected 'stl', got '{fmt}'"

    def test_detect_obj_by_content(self, sample_obj: Path) -> None:
        fmt = _detect_format_by_content(sample_obj)
        assert fmt == "obj", f"Expected 'obj', got '{fmt}'"

    def test_detect_iges_by_content(self, sample_iges: Path) -> None:
        fmt = _detect_format_by_content(sample_iges)
        assert fmt == "iges", f"Expected 'iges', got '{fmt}'"

    def test_detect_unsupported_format(self, unsupported_format: Path) -> None:
        fmt = _detect_format_by_content(unsupported_format)
        assert fmt is None, f"Expected None, got '{fmt}'"

    def test_manual_detect_pdf(self, sample_pdf: Path) -> None:
        fmt = _manual_format_detect(sample_pdf)
        assert fmt == "pdf"

    def test_manual_detect_png(self, sample_png: Path) -> None:
        fmt = _manual_format_detect(sample_png)
        assert fmt == "png"

    def test_manual_detect_jpeg(self, sample_jpeg: Path) -> None:
        fmt = _manual_format_detect(sample_jpeg)
        assert fmt == "jpeg"

    def test_manual_detect_step(self, sample_step: Path) -> None:
        fmt = _manual_format_detect(sample_step)
        assert fmt == "step"


# ── Full Pipeline: process() ──────────────────────────────────────────────────


class TestProcess:
    def test_process_png(self, agent: InputProcessingAgent, sample_png: Path) -> None:
        result = agent.process(sample_png)
        assert result["format"] == "png"
        assert result["is_3d"] is False
        assert result["detected_by"] == "extension"
        assert result["file_path"] == str(sample_png)
        assert "data" in result
        assert "image" in result["data"]
        assert "gray" in result["data"]
        assert result["data"]["width"] == 200
        assert result["data"]["height"] == 200

    def test_process_jpeg(self, agent: InputProcessingAgent, sample_jpeg: Path) -> None:
        result = agent.process(sample_jpeg)
        assert result["format"] == "jpeg"
        assert result["is_3d"] is False
        assert "image" in result["data"]

    def test_process_tiff(self, agent: InputProcessingAgent, sample_tiff: Path) -> None:
        result = agent.process(sample_tiff)
        assert result["format"] == "tiff"
        assert result["is_3d"] is False
        assert "image" in result["data"]

    def test_process_pdf(self, agent: InputProcessingAgent, sample_pdf: Path) -> None:
        result = agent.process(sample_pdf)
        assert result["format"] == "pdf"
        assert result["is_3d"] is False
        data = result["data"]
        assert "pages" in data
        assert len(data["pages"]) >= 1
        assert data["page_count"] >= 1
        assert "metadata" in data
        assert "text_content" in data

    def test_process_dxf(self, agent: InputProcessingAgent, sample_dxf: Path) -> None:
        result = agent.process(sample_dxf)
        assert result["format"] == "dxf"
        assert result["is_3d"] is False
        data = result["data"]
        assert "entities" in data
        assert len(data["entities"]) >= 2  # line + circle
        assert data["entity_total"] >= 2
        assert "dxf_version" in data
        assert "layers" in data
        assert data["entity_counts"].get("LINE", 0) >= 1
        assert data["entity_counts"].get("CIRCLE", 0) >= 1

    def test_process_step(self, agent: InputProcessingAgent, sample_step: Path) -> None:
        result = agent.process(sample_step)
        assert result["format"] == "step"
        assert result["is_3d"] is True
        data = result["data"]
        assert data["format"] == "step"
        assert "header_lines" in data

    def test_process_stl_ascii(self, agent: InputProcessingAgent, sample_stl_ascii: Path) -> None:
        result = agent.process(sample_stl_ascii)
        assert result["format"] == "stl"
        assert result["is_3d"] is True

    def test_process_stl_binary(self, agent: InputProcessingAgent, sample_stl_binary: Path) -> None:
        result = agent.process(sample_stl_binary)
        assert result["format"] == "stl"
        assert result["is_3d"] is True

    def test_process_obj(self, agent: InputProcessingAgent, sample_obj: Path) -> None:
        result = agent.process(sample_obj)
        assert result["format"] == "obj"
        assert result["is_3d"] is True

    def test_process_iges(self, agent: InputProcessingAgent, sample_iges: Path) -> None:
        result = agent.process(sample_iges)
        assert result["format"] == "iges"
        assert result["is_3d"] is True

    def test_process_file_not_found(self, agent: InputProcessingAgent) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            agent.process("/nonexistent/foo.step")

    def test_process_unsupported_format(self, agent: InputProcessingAgent, unsupported_format: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported or unrecognized"):
            agent.process(unsupported_format)

    def test_process_bytes_input(self, agent: InputProcessingAgent) -> None:
        """Should accept bytes and detect format by content."""
        # Minimal PNG bytes
        png_bytes = (
            b"\x89PNG\r\n\x1a\n"  # signature
            + b"\x00\x00\x00\rIHDR"  # IHDR chunk
            + b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
            + b"\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\x0f\x00\x00\x00\x00\xff\xff\x03\x00"
            + b"\x00\x00\x04\x00\x01\x0c\x0c\x0c"
            + b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        result = agent.process(png_bytes)
        assert result["format"] == "png"

    def test_process_metadata_structure(self, agent: InputProcessingAgent, sample_png: Path) -> None:
        result = agent.process(sample_png)
        meta = result["metadata"]
        assert "filename" in meta
        assert "extension" in meta
        assert "size_bytes" in meta
        assert "size_mb" in meta
        assert meta["size_bytes"] > 0
        assert meta["size_mb"] > 0
        assert meta["width"] == 200
        assert meta["height"] == 200

    def test_process_png_detected_by_content(self, agent: InputProcessingAgent, sample_png: Path) -> None:
        """When extension is not recognized, fallback to content detection."""
        # Rename the PNG to have no extension
        path_no_ext = sample_png.with_name("test_no_ext")
        import shutil
        shutil.copy2(str(sample_png), str(path_no_ext))
        result = agent.process(path_no_ext)
        assert result["format"] == "png"
        assert result["detected_by"] == "content"


# ── Data Loading ──────────────────────────────────────────────────────────────


class TestDataLoading:
    def test_load_image_raster(self, sample_png: Path) -> None:
        data = _load_image(sample_png)
        assert "image" in data
        assert "gray" in data
        assert data["width"] == 200
        assert data["height"] == 200
        assert isinstance(data["image"], np.ndarray)
        assert isinstance(data["gray"], np.ndarray)
        assert data["gray"].ndim == 2  # grayscale is 2D

    def test_load_pdf_metadata(self, sample_pdf: Path) -> None:
        data = _load_pdf(sample_pdf)
        assert "pages" in data
        assert len(data["pages"]) >= 1
        page = data["pages"][0]
        assert "width" in page
        assert "height" in page
        assert page["width"] > 0
        assert page["height"] > 0

    def test_load_pdf_text(self, sample_pdf: Path) -> None:
        data = _load_pdf(sample_pdf)
        assert "text_content" in data
        # The test PDF contains "Test" on the page
        assert "Test" in data["text_content"]

    def test_load_dxf_entities(self, sample_dxf: Path) -> None:
        data = _load_dxf(sample_dxf)
        assert len(data["entities"]) >= 2
        assert data["entity_counts"].get("LINE", 0) == 1
        assert data["entity_counts"].get("CIRCLE", 0) == 1
        assert data["entity_total"] >= 2

    def test_load_dxf_line(self, sample_dxf: Path) -> None:
        data = _load_dxf(sample_dxf)
        lines = [e for e in data["entities"] if e["type"] == "LINE"]
        assert len(lines) == 1
        line = lines[0]
        assert line["start"] == (0.0, 0.0)
        assert line["end"] == (100.0, 100.0)

    def test_load_dxf_circle(self, sample_dxf: Path) -> None:
        data = _load_dxf(sample_dxf)
        circles = [e for e in data["entities"] if e["type"] == "CIRCLE"]
        assert len(circles) == 1
        circle = circles[0]
        assert circle["center"] == (50.0, 50.0)
        assert circle["radius"] == 25.0

    def test_load_dxf_version(self, sample_dxf: Path) -> None:
        data = _load_dxf(sample_dxf)
        assert "AC1009" in data["dxf_version"]  # R12


# ── Edge Cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_jpg_extension_normalized(self, agent: InputProcessingAgent, sample_dir: Path) -> None:
        """.jpg should be normalized to .jpeg"""
        import shutil
        # Copy the JPEG to a .jpg file
        src = sample_dir / "test_drawing.jpeg"
        dst = sample_dir / "test_drawing.jpg"
        shutil.copy2(str(src), str(dst))
        result = agent.process(dst)
        assert result["format"] == "jpeg"

    def test_tif_extension_normalized(self, agent: InputProcessingAgent, sample_dir: Path) -> None:
        """.tif should be normalized to .tiff"""
        import shutil
        src = sample_dir / "test_drawing.tiff"
        dst = sample_dir / "test_drawing.tif"
        shutil.copy2(str(src), str(dst))
        result = agent.process(dst)
        assert result["format"] == "tiff"

    def test_file_size_limit_exceeded(self, agent: InputProcessingAgent, sample_png: Path) -> None:
        with pytest.raises(ValueError, match="exceeds size limit"):
            agent.process(sample_png, max_size_mb=0.000001)

    def test_process_returns_all_required_keys(self, agent: InputProcessingAgent, sample_png: Path) -> None:
        result = agent.process(sample_png)
        required_keys = {"file_path", "format", "detected_by", "metadata", "is_3d", "normalized_path", "data"}
        assert required_keys.issubset(result.keys()), f"Missing keys: {required_keys - result.keys()}"

    def test_process_string_path(self, agent: InputProcessingAgent, sample_png: Path) -> None:
        """Should accept string paths, not just Path objects."""
        result = agent.process(str(sample_png))
        assert result["format"] == "png"

    def test_process_with_kwargs(self, agent: InputProcessingAgent, sample_png: Path) -> None:
        """Should accept extra kwargs without error."""
        result = agent.process(sample_png, extra_option=True, debug=True)
        assert result["format"] == "png"
