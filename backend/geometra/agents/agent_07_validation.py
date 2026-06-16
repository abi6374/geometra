"""Agent 7: Validation Agent.

Validates generated CAD output against the source document.

Validation checks:
1. CAD generation success
2. Feature count match between operation count and CAD output
3. File format completeness (required formats present)
4. File existence on disk
5. Basic structural validation of the CAD result

Note: Full geometry comparison (projection matching, dimension accuracy,
tolerance checking) requires OpenCascade integration and is a future milestone.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from geometra.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class ValidationAgent(BaseAgent):
    """Validates generated CAD models against the source engineering drawing."""

    def process(self, input_data: Any, **kwargs: Any) -> dict[str, Any]:
        """Alias for validate()."""
        if isinstance(input_data, (list, tuple)) and len(input_data) == 2:
            return self.validate(input_data[0], input_data[1])
        return self.validate(input_data, {})

    def validate(
        self,
        source_doc: dict[str, Any],
        cad_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate generated CAD against the source document.

        Args:
            source_doc: Original document from InputProcessingAgent.
            cad_result: Output from CADGenerationAgent or ThreeDToTwoDConversionAgent.

        Returns:
            Validation report with scores for each validation step.
        """
        self.logger.info("Validating generated CAD")

        details: dict[str, Any] = {}
        scores: list[float] = []

        # ── 1. CAD generation success ──
        success = cad_result.get("success", False)
        details["cad_generated"] = success
        scores.append(1.0 if success else 0.0)

        # ── 2. Feature count match ──
        feature_count = cad_result.get("feature_count", 0)
        if isinstance(source_doc, dict):
            # If source has an operations count, compare against it
            source_ops = len(source_doc.get("operations", []))
            if source_ops > 0:
                feature_match = min(1.0, feature_count / source_ops)
            else:
                feature_match = 1.0 if feature_count == 0 else 0.5
        else:
            feature_match = 0.5 if feature_count > 0 else 1.0
        details["feature_count_match"] = feature_match
        details["feature_count"] = feature_count
        scores.append(feature_match)

        # ── 3. File format completeness ──
        formats = cad_result.get("formats", [])
        required = {"scad", "py"}
        present = set(formats)
        format_score = len(present & required) / len(required) if required else 0.0
        details["formats_present"] = sorted(present)
        details["formats_missing"] = sorted(required - present)
        details["format_completeness"] = format_score
        scores.append(format_score)

        # ── 4. File existence on disk ──
        output_paths = cad_result.get("output_paths", {})
        files_exist = 0
        total_files = len(output_paths) or 1
        for fmt, path_str in output_paths.items():
            if Path(path_str).exists():
                files_exist += 1
                details.setdefault("files_found", []).append(fmt)
            else:
                details.setdefault("files_missing", []).append(fmt)
        file_score = files_exist / total_files
        details["file_existence"] = file_score
        scores.append(file_score)

        # ── 5. Script content validation (basic) ──
        openscad_script = cad_result.get("openscad_script", "")
        cadquery_script = cad_result.get("cadquery_script", "")
        has_scad_body = "body_w" in openscad_script and "body_h" in openscad_script
        has_cq_body = "Workplane" in cadquery_script
        script_score = 1.0 if has_scad_body and has_cq_body else 0.5 if has_scad_body or has_cq_body else 0.0
        details["script_validity"] = script_score
        details["has_scad_body"] = has_scad_body
        details["has_cq_body"] = has_cq_body
        scores.append(script_score)

        # ── Overall score ──
        overall_score = sum(scores) / len(scores) if scores else 0.0
        passed = overall_score >= 0.6 and success

        self.logger.info(
            "Validation complete: overall=%.2f, passed=%s (%d checks)",
            overall_score,
            passed,
            len(scores),
        )

        return {
            "overall_score": round(overall_score, 4),
            "feature_count_match": round(feature_match, 4),
            "format_completeness": round(format_score, 4),
            "file_existence": round(file_score, 4),
            "script_validity": round(script_score, 4),
            "dimension_accuracy": 0.0,  # Future: geometry comparison
            "hole_location_accuracy": 0.0,  # Future
            "tolerance_accuracy": 0.0,  # Future
            "projection_match": 0.0,  # Future
            "details": details,
            "passed": passed,
        }
