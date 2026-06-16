"""Agent 6: CAD Generation Agent.

⚠️  This module is now a convenience re-export.
Please import from ``geometra.agents.agent_06`` instead:

    from geometra.agents.agent_06 import CADGenerationAgent
    from geometra.agents.agent_06.openscad_gen import generate_openscad
    from geometra.agents.agent_06.cadquery_gen import generate_cadquery_script, build_cadquery_model

All public symbols are re-exported below for backward compatibility.
"""

from geometra.agents.agent_06 import (  # noqa: F401
    CADGenerationAgent,
    CADQUERY_AVAILABLE,
    DEFAULT_THICKNESS,
    build_cadquery_model,
    generate_cadquery_script,
    generate_openscad,
)
