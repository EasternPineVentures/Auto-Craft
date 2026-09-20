"""Shared exception types for the control layer.

The agent loop needs to tell "the operator asked for something invalid" apart
from "safety refused to act", because only the second one should count toward the
consecutive-block budget that stops a runaway loop.
"""

from __future__ import annotations

__all__ = [
    "ControlError",
    "InputBlocked",
    "InvalidMouseDelta",
    "UnknownKeyError",
]


class ControlError(Exception):
    """Base class for control-layer failures."""


class InputBlocked(ControlError):
    """Raised when the safety guard refuses to inject input.

    Carries the human-readable reason so it can be recorded verbatim in
    telemetry rather than being re-derived.
    """

    def __init__(self, reason: str, *, action: str = "") -> None:
        self.reason = reason
        self.action = action
        super().__init__(reason if not action else f"{action}: {reason}")


class UnknownKeyError(ControlError, ValueError):
    """Raised when a key name is not in the supported key map."""


class InvalidMouseDelta(ControlError, ValueError):
    """Raised when a relative mouse movement exceeds the configured bound."""
