"""The agent: observations in, decisions made, actions attempted, results recorded.

Nothing in this package may read privileged game state. The only inputs are
pixels and window bookkeeping; the only outputs are human-style input events.
"""

from __future__ import annotations

from .action import (
    Action,
    ActionExecutor,
    ActionKind,
    ActionResult,
    REQUIRED_PARAMETERS,
    validate_action,
)
from .decision import DecisionPolicy, NoOpDecisionPolicy
from .loop import AgentLoop, StepRecord
from .observation import Observation, Observer, WindowStatus

__all__ = [
    "Action",
    "ActionExecutor",
    "ActionKind",
    "ActionResult",
    "AgentLoop",
    "DecisionPolicy",
    "NoOpDecisionPolicy",
    "Observation",
    "Observer",
    "REQUIRED_PARAMETERS",
    "StepRecord",
    "WindowStatus",
    "validate_action",
]
