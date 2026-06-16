"""CAD Generation Agent (Agent 6) subpackage.

Modules:
    openscad_gen  — OpenSCAD script generator
    cadquery_gen  — CadQuery script generator + in-process model builder
    agent         — CADGenerationAgent class

Usage:
    from geometra.agents.agent_06 import CADGenerationAgent
    from geometra.agents.agent_06.openscad_gen import generate_openscad
    from geometra.agents.agent_06.cadquery_gen import generate_cadquery_script, build_cadquery_model
"""

from geometra.agents.agent_06.agent import CADGenerationAgent
from geometra.agents.agent_06.cadquery_gen import (
    CADQUERY_AVAILABLE,
    build_cadquery_model,
    generate_cadquery_script,
)
from geometra.agents.agent_06.openscad_gen import DEFAULT_THICKNESS, generate_openscad

__all__ = [
    "CADGenerationAgent",
    "CADQUERY_AVAILABLE",
    "DEFAULT_THICKNESS",
    "build_cadquery_model",
    "generate_cadquery_script",
    "generate_openscad",
]
