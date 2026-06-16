"""Agent 6: CAD Generation Agent.

Converts a parametric feature tree from Agent 5 into a 3D CAD model
via OpenSCAD scripts and CadQuery (for STEP/STL export).

Operation → CAD mapping:
  drill:       through-hole cylinder subtracted from body
  mill:        slot (rounded rectangular pocket)
  cut:         rectangular cutout through the body
  pocket_mill: rectangular pocket (blind, not through)
  vent_cut:    group of thin rectangular slots
  boss_extrude: raised circular boss added to body
  fillet:      edge rounding (annotated in OpenSCAD, applied in CadQuery)
  chamfer:     edge bevel (annotated in OpenSCAD, applied in CadQuery)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from geometra.agents.agent_06.cadquery_gen import (
    CADQUERY_AVAILABLE,
    build_cadquery_model,
    generate_cadquery_script,
    get_cadquery,
)
from geometra.agents.agent_06.openscad_gen import generate_openscad
from geometra.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class CADGenerationAgent(BaseAgent):
    """Generates parametric 3D CAD models from feature tree operations.

    Supports two output paths:
      1. OpenSCAD script (.scad) — always generated, no extra dependencies
      2. CadQuery in-process model + STEP/STL export — when CadQuery is installed

    Input: Agent 5 output dict with 'operations' and 'parameters' keys.
    """

    def __init__(self) -> None:
        super().__init__()
        self._cadquery_available = CADQUERY_AVAILABLE

    @property
    def cadquery_available(self) -> bool:
        return self._cadquery_available

    def validate_input(self, input_data: Any) -> bool:
        """Validate that input has required keys."""
        if not isinstance(input_data, dict):
            return False
        # Must have either operations (with optional parameters)
        # or a fully structured feature tree
        return "operations" in input_data

    def process(self, input_data: Any, **kwargs: Any) -> dict[str, Any]:
        """Alias for generate()."""
        return self.generate(input_data, **kwargs)

    def generate(
        self,
        param_tree: dict[str, Any],
        output_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate CAD models from the parametric feature tree.

        Args:
            param_tree: Output from EngineeringReasoningAgent with:
                - operations: list of CAD operation dicts
                - parameters: global design parameters dict
            output_dir: Directory for output files. Defaults to ./output/cad/
            **kwargs:
                - thickness: override body thickness (mm)
                - export_scad: bool, generate .scad file (default True)
                - export_step: bool, export STEP via CadQuery (default True)
                - export_stl: bool, export STL via CadQuery (default True)

        Returns:
            Dict with keys:
                - success: bool
                - openscad_script: the generated OpenSCAD script text
                - cadquery_script: the generated CadQuery script text
                - output_paths: dict of format → file path
                - formats: list of generated format strings
                - cadquery_used: whether CadQuery was available for export
                - feature_count: number of operations processed
        """
        self.logger.info("Generating CAD model from %d operations", len(param_tree.get("operations", [])))

        # Extract data
        operations = param_tree.get("operations", [])
        parameters = param_tree.get("parameters", {})

        # Apply overrides
        if "thickness" in kwargs:
            parameters["body_thickness"] = kwargs["thickness"]

        export_scad = kwargs.get("export_scad", True)
        export_step = kwargs.get("export_step", self._cadquery_available)
        export_stl = kwargs.get("export_stl", self._cadquery_available)

        # Resolve output directory
        out_dir = Path(output_dir) if output_dir else Path("output") / "cad"
        out_dir.mkdir(parents=True, exist_ok=True)

        output_paths: dict[str, str] = {}
        formats: list[str] = []

        # ── 1. Generate OpenSCAD script ──
        openscad_script = generate_openscad(operations, parameters)
        scad_path = out_dir / "model.scad"
        scad_path.write_text(openscad_script, encoding="utf-8")
        output_paths["scad"] = str(scad_path)
        formats.append("scad")
        self.logger.info("OpenSCAD script written to %s", scad_path)

        # ── 2. Generate CadQuery script ──
        cadquery_script = generate_cadquery_script(operations, parameters)
        cq_path = out_dir / "model_cadquery.py"
        cq_path.write_text(cadquery_script, encoding="utf-8")
        output_paths["cadquery_py"] = str(cq_path)
        formats.append("py")
        self.logger.info("CadQuery script written to %s", cq_path)

        # ── 3. Export STEP/STL via CadQuery (if available) ──
        cadquery_used = self._cadquery_available

        if self._cadquery_available:
            try:
                model = build_cadquery_model(operations, parameters)
                if model is not None:
                    _cq = get_cadquery()
                    if _cq is None:
                        raise RuntimeError("CadQuery module not accessible")
                    if export_step:
                        step_path = out_dir / "model.step"
                        _cq.exporters.export(model, str(step_path))
                        output_paths["step"] = str(step_path)
                        formats.append("step")
                        self.logger.info("STEP exported to %s", step_path)

                    if export_stl:
                        stl_path = out_dir / "model.stl"
                        _cq.exporters.export(model, str(stl_path), tolerance=0.1)
                        output_paths["stl"] = str(stl_path)
                        formats.append("stl")
                        self.logger.info("STL exported to %s", stl_path)
            except Exception as exc:
                self.logger.warning("CadQuery export failed: %s", exc)
                cadquery_used = False

        self.logger.info(
            "CAD generation complete: %d features, formats: %s",
            len(operations),
            ", ".join(formats),
        )

        return {
            "success": True,
            "openscad_script": openscad_script,
            "cadquery_script": cadquery_script,
            "output_paths": output_paths,
            "formats": formats,
            "cadquery_used": cadquery_used,
            "feature_count": len(operations),
        }
