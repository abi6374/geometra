"""Base agent class that all Geometra agents inherit from."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all Geometra processing agents.

    Each agent implements a single stage in the conversion pipeline.
    Agents are designed to be independently runnable and testable.
    """

    def __init__(self) -> None:
        self.name: str = self.__class__.__name__
        self.logger = logging.getLogger(self.name)

    @abstractmethod
    def process(self, input_data: Any, **kwargs: Any) -> Any:
        """Execute the agent's core processing logic.

        Args:
            input_data: Input data for this agent (varies by agent).
            **kwargs: Additional options.

        Returns:
            Processing result (varies by agent).
        """
        ...

    def validate_input(self, input_data: Any) -> bool:
        """Validate that input data meets requirements.

        Override in subclasses to add specific validation.
        """
        return input_data is not None

    def validate_output(self, output: Any) -> bool:
        """Validate that output meets quality requirements.

        Override in subclasses to add specific validation.
        """
        return output is not None

    def __repr__(self) -> str:
        return f"<{self.name}>"
