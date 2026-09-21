"""Structured events: the vocabulary a wake run narrates itself with.

WAKE-001 does not just move a camera and stop. It has to be possible to read back
*what it noticed, what it tried, and whether that helped*, in a form a human can
follow live and a later system can consume. That is what this module is for.

Two rules shape it:

* **Events are data, not prose.** Each :class:`WakeEvent` carries a machine kind,
  a small JSON payload of measured numbers, and a short plain-language line. The
  line is generated here from the kind and the payload so it can never drift out
  of sync with the numbers it describes.
* **Events describe behaviour, never semantics.** There is no ``TREE_FOUND`` and
  there never will be one in this milestone. A candidate is a "visually salient
  region", full stop. Nothing in this module knows what any of it *is*.

The event set is closed on purpose. The specification lists the events it wants,
and a later milestone (memory, personality, a story stream) is expected to read
these rather than invent new ones - which is also why nothing here reaches into
the control layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

__all__ = [
    "STREAM_SUMMARY",
    "WakeEvent",
    "WakeEventKind",
    "stream_summary",
]


class WakeEventKind(str, Enum):
    """Every event a WAKE-001 run can emit."""

    WAKE_STARTED = "WAKE_STARTED"
    VIEW_CAPTURED = "VIEW_CAPTURED"
    NEW_VIEW_OBSERVED = "NEW_VIEW_OBSERVED"
    VIEW_REVISITED = "VIEW_REVISITED"
    SCAN_MOVE = "SCAN_MOVE"
    CANDIDATE_FOUND = "CANDIDATE_FOUND"
    TARGET_SELECTED = "TARGET_SELECTED"
    CENTERING_PROGRESS = "CENTERING_PROGRESS"
    OVERSHOOT_DETECTED = "OVERSHOOT_DETECTED"
    TARGET_LOST = "TARGET_LOST"
    TARGET_REACQUIRED = "TARGET_REACQUIRED"
    STRATEGY_FAILED = "STRATEGY_FAILED"
    STUCK_PATTERN_DETECTED = "STUCK_PATTERN_DETECTED"
    STRATEGY_CHANGED = "STRATEGY_CHANGED"
    TARGET_CENTERED = "TARGET_CENTERED"
    WAKE_COMPLETE = "WAKE_COMPLETE"
    WAKE_ABORTED = "WAKE_ABORTED"
    SALIENCE_SCAN_SUMMARY = "SALIENCE_SCAN_SUMMARY"
    WINDOW_GEOMETRY_CHANGED = "WINDOW_GEOMETRY_CHANGED"


#: One plain-language line per event kind, for the streaming view of a run.
#:
#: These are deliberately flat statements of what the agent just did or saw. They
#: are not a running commentary and they are not chain-of-thought: the milestone
#: requires thoughts to stay display-only, and a summary line that only restates
#: an already-recorded event keeps that promise. Nothing here is ever fed back
#: into a decision.
STREAM_SUMMARY: Mapping[WakeEventKind, str] = {
    WakeEventKind.WAKE_STARTED: "Waking up and taking a first look.",
    WakeEventKind.VIEW_CAPTURED: "Looking around.",
    WakeEventKind.NEW_VIEW_OBSERVED: "New view found.",
    WakeEventKind.VIEW_REVISITED: "This looks like a view I already saw.",
    WakeEventKind.SCAN_MOVE: "Turning to inspect another direction.",
    WakeEventKind.CANDIDATE_FOUND: "Something over there stands out.",
    WakeEventKind.TARGET_SELECTED: "Candidate selected.",
    WakeEventKind.CENTERING_PROGRESS: "Target moved closer to centre.",
    WakeEventKind.OVERSHOOT_DETECTED: "That went past the centre.",
    WakeEventKind.TARGET_LOST: "Target lost.",
    WakeEventKind.TARGET_REACQUIRED: "Target found again.",
    WakeEventKind.STRATEGY_FAILED: "That did not work; trying something else.",
    WakeEventKind.STUCK_PATTERN_DETECTED: "I keep doing the same thing without progress.",
    WakeEventKind.STRATEGY_CHANGED: "Trying another view.",
    WakeEventKind.TARGET_CENTERED: "Target centred.",
    WakeEventKind.WAKE_COMPLETE: "Done - stopping here.",
    WakeEventKind.WAKE_ABORTED: "Stopping early.",
    WakeEventKind.SALIENCE_SCAN_SUMMARY: "Recording what this frame offered.",
    WakeEventKind.WINDOW_GEOMETRY_CHANGED: "The game window is not the size it was.",
}


def stream_summary(kind: WakeEventKind | str) -> str:
    """Return the plain-language line for ``kind``.

    Unknown kinds get a neutral line rather than raising: a display layer must
    never be the thing that breaks a run.
    """
    try:
        return STREAM_SUMMARY[WakeEventKind(kind)]
    except ValueError:
        return str(kind)


@dataclass(frozen=True)
class WakeEvent:
    """One thing that happened, with the numbers that justify it.

    ``detail`` holds only measured values - offsets, counts, similarities,
    distances. It never holds a frame, and it never holds an interpretation of
    what was seen.
    """

    kind: WakeEventKind
    message: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)
    index: int = 0
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", WakeEventKind(self.kind))
        object.__setattr__(self, "detail", dict(self.detail))
        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(self, "timestamp", float(self.timestamp))
        if not self.message:
            object.__setattr__(self, "message", stream_summary(self.kind))

    @classmethod
    def of(
        cls,
        kind: WakeEventKind,
        *,
        index: int = 0,
        timestamp: float = 0.0,
        message: str = "",
        **detail: Any,
    ) -> "WakeEvent":
        """Build an event, passing each keyword as a ``detail`` entry."""
        return cls(kind=kind, message=message, detail=detail, index=index, timestamp=timestamp)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view. Contains no pixel data."""
        return {
            "kind": self.kind.value,
            "message": self.message,
            "detail": {key: _plain(value) for key, value in self.detail.items()},
            "index": self.index,
            "timestamp": self.timestamp,
        }

    def describe(self) -> str:
        """Return a single printable line: the summary, then the numbers."""
        if not self.detail:
            return self.message
        numbers = ", ".join(f"{key}={_plain(value)}" for key, value in self.detail.items())
        return f"{self.message} ({numbers})"


def _plain(value: Any) -> Any:
    """Coerce a value to something JSON can hold and a human can read."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    return str(value)
