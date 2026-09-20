"""Display contracts for the AutoCraft observer.

The observer is an instrument, not a participant: the agent publishes state and
the observer shows it. Everything in this module is therefore a *display*
contract, which puts two requirements above convenience.

**It must be honestly serialisable.** :meth:`ObserverSnapshot.to_dict` produces
plain JSON types only - no numpy arrays, no ``Path`` objects, no pixel buffers -
so the page can never receive something that was not meant to be shown, and the
snapshot stays cheap to poll.

**It must never claim more than the agent knows.** Every field has a truthful
empty state. When there is no frame, the frame section says so. When nothing has
been interpreted, the belief list is empty and the page prints
"No interpreted objects yet." rather than inventing plausible-looking content.

Nothing here imports the control layer, and nothing here can send input. That
boundary is enforced by ``tests/test_observer.py``.

On affect: :class:`AffectState` is a *functional* model, not a claim about inner
experience. It exists so behaviour can carry continuity between steps, so
accumulated experience has somewhere to live, and so the state is legible to a
human watching. It is deliberately a plain, clamped, five-dimensional vector
with two arithmetic operations on it, and nothing in V0 drives those operations
from experience yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Sequence

from ..thoughts.model import ThoughtEvent

__all__ = [
    "AFFECT_BASELINE",
    "AFFECT_DIMENSIONS",
    "AffectState",
    "AgentEvent",
    "AgentMode",
    "Belief",
    "EmergencyStopState",
    "EventKind",
    "FrameFreshness",
    "FrameInfo",
    "ObserverError",
    "ObserverSnapshot",
    "RunMetrics",
    "SafetyStatus",
    "frame_freshness",
    "input_permitted",
    "with_events",
    "with_thoughts",
]


class ObserverError(ValueError):
    """Raised when a value cannot be represented as observer state."""


# ---------------------------------------------------------------------------
# affect
# ---------------------------------------------------------------------------

#: The five dimensions of the simulated affect model, in display order.
AFFECT_DIMENSIONS: tuple[str, ...] = (
    "curiosity",
    "confidence",
    "stress",
    "frustration",
    "energy",
)


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ObserverError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ObserverError(f"{name} must be a finite number, got {value!r}")
    return number


def _unit(value: Any, *, name: str) -> float:
    """Coerce to a normalised ``0.0``-``1.0`` value, clamping out-of-range input."""
    return min(1.0, max(0.0, _finite(value, name=name)))


@dataclass(frozen=True)
class AffectState:
    """A simulated affective state, every dimension normalised to ``0.0``-``1.0``.

    This is a functional model with a behavioural purpose: to give the agent
    continuity between steps, to give accumulated experience a place to live, and
    to make the agent's condition legible to a human. It is **not** a claim that
    the agent has feelings, and it is not a decorative animation - nothing in
    V0 randomises it, and the page shows exactly the numbers the agent reports.

    Numbers outside ``0.0``-``1.0`` are clamped rather than rejected, because
    clamping is what an accumulating model needs. Values that are not finite
    numbers at all are refused, because those indicate a bug rather than drift.

    The defaults are the documented resting state.
    """

    curiosity: float = 0.60
    confidence: float = 0.50
    stress: float = 0.10
    frustration: float = 0.00
    energy: float = 1.00

    def __post_init__(self) -> None:
        for name in AFFECT_DIMENSIONS:
            object.__setattr__(self, name, _unit(getattr(self, name), name=name))

    # -- inspection -------------------------------------------------------

    def as_dict(self) -> dict[str, float]:
        """Return the dimensions as a plain JSON-serialisable mapping."""
        return {name: round(getattr(self, name), 4) for name in AFFECT_DIMENSIONS}

    def as_tuple(self) -> tuple[float, ...]:
        """Return the dimensions as a tuple in display order."""
        return tuple(getattr(self, name) for name in AFFECT_DIMENSIONS)

    # -- the extension seam ----------------------------------------------

    def blend(self, other: "AffectState", weight: float) -> "AffectState":
        """Move ``weight`` (0..1) of the way toward ``other``.

        This is the single primitive the documented experience-driven rules need.
        Decay toward a baseline is ``state.blend(AFFECT_BASELINE, 1 - exp(-rate * dt))``;
        a partial recovery is the same call with a smaller weight. Nothing calls
        this from the agent yet - it exists so those rules can be added without
        reshaping the model.
        """
        if not isinstance(other, AffectState):
            raise ObserverError(f"blend() expects another AffectState, got {type(other).__name__}")
        fraction = _unit(weight, name="weight")
        return AffectState(
            **{
                name: getattr(self, name) + (getattr(other, name) - getattr(self, name)) * fraction
                for name in AFFECT_DIMENSIONS
            }
        )

    def adjust(self, **deltas: float) -> "AffectState":
        """Return a copy with clamped deltas applied to the named dimensions.

        The other primitive an experience hook needs: a success would be
        ``state.adjust(confidence=+0.05, frustration=-0.10)``. Unknown dimension
        names are refused so a typo cannot silently do nothing.
        """
        unknown = sorted(set(deltas) - set(AFFECT_DIMENSIONS))
        if unknown:
            known = ", ".join(AFFECT_DIMENSIONS)
            raise ObserverError(f"unknown affect dimension(s) {unknown}; known dimensions: {known}")
        return AffectState(
            **{
                name: getattr(self, name) + _finite(deltas.get(name, 0.0), name=name)
                for name in AFFECT_DIMENSIONS
            }
        )


#: The documented resting state: mildly curious, neutral confidence, low stress,
#: no frustration, fully rested.
AFFECT_BASELINE = AffectState()


# ---------------------------------------------------------------------------
# enums
# ---------------------------------------------------------------------------


class AgentMode(str, Enum):
    """What the agent is doing right now, as one of a fixed set of words."""

    IDLE = "IDLE"
    OBSERVING = "OBSERVING"
    ACTING = "ACTING"
    PAUSED = "PAUSED"
    SAFE_STOP = "SAFE_STOP"
    ERROR = "ERROR"


class EventKind(str, Enum):
    """Category of a timeline entry, used only for display grouping."""

    OBSERVE = "observe"
    ACTION = "action"
    SAFETY = "safety"
    ERROR = "error"
    MEMORY = "memory"
    INFO = "info"


class FrameFreshness(str, Enum):
    """How current the displayed game frame is."""

    NONE = "none"
    LIVE = "live"
    RECENT = "recent"
    STALE = "stale"


class EmergencyStopState(str, Enum):
    """Whether the emergency stop has fired."""

    READY = "ready"
    TRIGGERED = "triggered"


def frame_freshness(
    age_seconds: float | None,
    *,
    live_seconds: float,
    stale_seconds: float,
) -> FrameFreshness:
    """Classify a frame's age.

    ``None`` means no frame has been published at all, which is distinct from a
    frame that is merely old - the page says "no frame yet" rather than "stale",
    because those are different problems for whoever is watching.

    A negative age (the frame timestamp is ahead of the reading clock) is treated
    as live rather than as an error: it means the two clocks disagree by
    microseconds, not that anything is wrong.
    """
    if age_seconds is None:
        return FrameFreshness.NONE
    if not math.isfinite(age_seconds):
        return FrameFreshness.STALE
    if age_seconds <= live_seconds:
        return FrameFreshness.LIVE
    if age_seconds <= stale_seconds:
        return FrameFreshness.RECENT
    return FrameFreshness.STALE


def input_permitted(
    *,
    window_found: bool,
    window_foreground: bool,
    emergency_stop: EmergencyStopState,
    require_foreground: bool,
    control_available: bool = True,
) -> bool:
    """Whether the safety layer would allow input, given these published facts.

    This mirrors the refusal conditions in
    :meth:`~autocraft.control.safety.SafetyGuard.authorize` and nothing else: an
    active emergency stop refuses, and losing the foreground lock refuses when
    the lock is enabled. It is a pure function of facts the agent publishes, so
    the observer never has to probe the live guard - probing would call
    ``GetForegroundWindow`` on every poll and, worse, ``authorize`` records a
    safety event each time it refuses, so a dashboard that asked it directly
    could fill the safety log with refusals that never happened.

    ``control_available`` is false for a run that has no safety guard at all,
    such as a read-only ``observe`` session. Without it the page would report
    "input enabled" for a process that cannot inject anything, which is exactly
    the kind of flattering claim this layer is meant to avoid.

    ``tests/test_observer.py`` cross-checks this against ``authorize`` so the two
    cannot drift apart.
    """
    if not control_available:
        return False
    if emergency_stop is EmergencyStopState.TRIGGERED:
        return False
    if require_foreground and not (window_found and window_foreground):
        return False
    return True


# ---------------------------------------------------------------------------
# structured interpretation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Belief:
    """One structured interpretation the agent is willing to state.

    Used for both the belief list and the detected-entity list. V0 has no
    perception, so both are empty and the page shows the empty state. The shape
    exists now so a later perception stage has a contract to fill in rather than
    a dashboard to rewrite.
    """

    label: str
    confidence: float = 0.0
    source: str = "unknown"
    timestamp: float | None = None

    def __post_init__(self) -> None:
        label = str(self.label).strip()
        if not label:
            raise ObserverError("belief label must not be empty")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "confidence", _unit(self.confidence, name="confidence"))
        object.__setattr__(self, "source", str(self.source))
        if self.timestamp is not None:
            object.__setattr__(self, "timestamp", float(self.timestamp))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class AgentEvent:
    """One entry in the recent-event timeline."""

    timestamp: float
    message: str
    kind: EventKind = EventKind.INFO

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "message", str(self.message))
        if not isinstance(self.kind, EventKind):
            object.__setattr__(self, "kind", EventKind(str(self.kind)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {"timestamp": self.timestamp, "kind": self.kind.value, "message": self.message}


# ---------------------------------------------------------------------------
# operational read-outs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunMetrics:
    """Counters for the current run. Every value is measured, never estimated."""

    run_id: str = "-"
    started_at: float | None = None
    runtime_seconds: float = 0.0
    frames_observed: int = 0
    capture_fps: float = 0.0
    loop_rate: float = 0.0
    actions_attempted: int = 0
    actions_executed: int = 0
    actions_blocked: int = 0
    safety_events: int = 0
    errors: int = 0
    #: How many spontaneous thoughts were expressed. Counted, not estimated - and
    #: never used to decide anything, because a thought is not an instruction.
    thoughts_expressed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "runtime_seconds": round(self.runtime_seconds, 3),
            "frames_observed": self.frames_observed,
            "capture_fps": round(self.capture_fps, 2),
            "loop_rate": round(self.loop_rate, 2),
            "actions_attempted": self.actions_attempted,
            "actions_executed": self.actions_executed,
            "actions_blocked": self.actions_blocked,
            "safety_events": self.safety_events,
            "errors": self.errors,
            "thoughts_expressed": self.thoughts_expressed,
        }


@dataclass(frozen=True)
class SafetyStatus:
    """The safety layer's current verdict, as reported by the safety layer.

    The observer copies these values out of :class:`~autocraft.control.safety.SafetyGuard`
    and never recomputes or overrides them. ``input_enabled`` is the guard's own
    answer to "would you permit input right now", not a guess made by the page.
    """

    window_found: bool = False
    window_foreground: bool = False
    input_enabled: bool = False
    emergency_stop: EmergencyStopState = EmergencyStopState.READY
    held_inputs: tuple[str, ...] = ()
    blocked_streak: int = 0
    last_block_reason: str | None = None
    control_available: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "window_found", bool(self.window_found))
        object.__setattr__(self, "window_foreground", bool(self.window_foreground))
        object.__setattr__(self, "input_enabled", bool(self.input_enabled))
        object.__setattr__(self, "control_available", bool(self.control_available))
        object.__setattr__(self, "held_inputs", tuple(str(item) for item in self.held_inputs))
        object.__setattr__(self, "blocked_streak", int(self.blocked_streak))
        if not isinstance(self.emergency_stop, EmergencyStopState):
            object.__setattr__(self, "emergency_stop", EmergencyStopState(str(self.emergency_stop)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "window_found": self.window_found,
            "window_foreground": self.window_foreground,
            "input_enabled": self.input_enabled,
            "emergency_stop": self.emergency_stop.value,
            "held_inputs": list(self.held_inputs),
            "blocked_streak": self.blocked_streak,
            "last_block_reason": self.last_block_reason,
            "control_available": self.control_available,
        }


@dataclass(frozen=True)
class LookReport:
    """The most recent LOOK-001 measurement, exactly as it was measured.

    LOOK-001 injects a known mouse delta and measures the picture's response. The
    page shows those measurements and no verdict about them, because the
    milestone defines no pass mark: ``reversibility_ratio`` is reported as a
    number, not as a grade.

    Two honesty rules are structural rather than stylistic. First, every measured
    field defaults to ``None`` and ``available`` defaults to ``False``, so the
    empty state is "no LOOK measurement has been taken" and never a row of
    zeroes that reads like a real result. Second, ``pixels_per_delta_x`` and
    ``pixels_per_delta_y`` are image pixels per unit of injected mouse delta as
    *measured*; they are not a conversion AutoCraft is entitled to assume, and
    the panel says so.
    """

    available: bool = False
    status: str = "not-run"
    experiment: str = "LOOK-001"
    trial_index: int | None = None
    trial_count: int = 0
    dx: int | None = None
    dy: int | None = None
    settle_seconds: float | None = None
    window_width: int = 0
    window_height: int = 0
    movements_sent: int = 0
    mean_absolute_difference: float | None = None
    rmse: float | None = None
    changed_fraction: float | None = None
    block_grid: int = 0
    block_map: tuple[tuple[float, ...], ...] = ()
    shift_x: float | None = None
    shift_y: float | None = None
    shift_quality: float | None = None
    shift_available: bool = False
    pixels_per_delta_x: float | None = None
    pixels_per_delta_y: float | None = None
    reversibility_ratio: float | None = None
    reversibility_note: str = ""
    capture_seconds: float | None = None
    stop_reason: str = ""
    measured_at: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "available", bool(self.available))
        object.__setattr__(self, "shift_available", bool(self.shift_available))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "experiment", str(self.experiment))
        object.__setattr__(self, "trial_count", int(self.trial_count))
        object.__setattr__(self, "window_width", int(self.window_width))
        object.__setattr__(self, "window_height", int(self.window_height))
        object.__setattr__(self, "movements_sent", int(self.movements_sent))
        object.__setattr__(self, "block_grid", int(self.block_grid))
        object.__setattr__(
            self,
            "block_map",
            tuple(tuple(float(value) for value in row) for row in self.block_map),
        )
        if self.trial_index is not None:
            object.__setattr__(self, "trial_index", int(self.trial_index))
        if self.dx is not None:
            object.__setattr__(self, "dx", int(self.dx))
        if self.dy is not None:
            object.__setattr__(self, "dy", int(self.dy))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view. Contains no pixel data."""

        def number(value: float | None, digits: int) -> float | None:
            return None if value is None else round(float(value), digits)

        return {
            "available": self.available,
            "status": self.status,
            "experiment": self.experiment,
            "trial_index": self.trial_index,
            "trial_count": self.trial_count,
            "dx": self.dx,
            "dy": self.dy,
            "settle_seconds": number(self.settle_seconds, 4),
            "window": {"width": self.window_width, "height": self.window_height},
            "movements_sent": self.movements_sent,
            "mean_absolute_difference": number(self.mean_absolute_difference, 4),
            "rmse": number(self.rmse, 4),
            "changed_fraction": number(self.changed_fraction, 6),
            "block_grid": self.block_grid,
            "block_map": [[round(value, 2) for value in row] for row in self.block_map],
            "shift": {
                "available": self.shift_available,
                "x": number(self.shift_x, 3),
                "y": number(self.shift_y, 3),
                "quality": number(self.shift_quality, 4),
                "method": "phase-correlation",
            },
            "pixels_per_delta": {
                "x": number(self.pixels_per_delta_x, 5),
                "y": number(self.pixels_per_delta_y, 5),
            },
            "reversibility_ratio": number(self.reversibility_ratio, 5),
            "reversibility_note": self.reversibility_note,
            "capture_seconds": number(self.capture_seconds, 4),
            "stop_reason": self.stop_reason,
            "measured_at": self.measured_at,
        }


@dataclass(frozen=True)
class FrameInfo:
    """Metadata about the frame the page is currently showing.

    Deliberately metadata only. The pixels travel separately over ``/frame`` so
    that polling the state stays cheap and a multi-megabyte image never lands in
    a JSON payload.
    """

    available: bool = False
    captured_at: float | None = None
    age_seconds: float | None = None
    freshness: FrameFreshness = FrameFreshness.NONE
    width: int = 0
    height: int = 0
    source_width: int = 0
    source_height: int = 0
    signature: str | None = None
    mean_luma: float | None = None
    reference: str | None = None
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "available": self.available,
            "captured_at": self.captured_at,
            "age_seconds": None if self.age_seconds is None else round(self.age_seconds, 3),
            "freshness": self.freshness.value,
            "width": self.width,
            "height": self.height,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "signature": self.signature,
            "mean_luma": None if self.mean_luma is None else round(self.mean_luma, 2),
            "reference": self.reference,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# the snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObserverSnapshot:
    """Everything the page needs to answer "what is the agent doing right now".

    The field list is the observer specification's contract. Two of its entries
    are intentionally honest about V0's limits: ``beliefs`` and
    ``detected_entities`` exist and are typed, but nothing fills them, because
    AutoCraft has no perception yet and a dashboard that shows invented
    perception would be worse than one that shows none.

    ``short_term_memory`` is the newest few timeline entries, newest first, for
    the compact memory panel. ``recent_events`` is the full bounded timeline in
    chronological order for the event log. They come from the same source and are
    two views of it, not two systems.

    ``latest_thought`` and ``thought_history`` carry the expression layer. They
    are display-only: the page renders them, and nothing reads them back to
    decide anything.

    ``look`` carries the most recent LOOK-001 measurement. Like the thought
    fields it is display-only, and like ``beliefs`` it starts out honestly empty:
    a run that never measured anything shows nothing rather than zeroes.
    """

    run_id: str = "-"
    timestamp: float = 0.0
    demo: bool = False
    mode: AgentMode = AgentMode.IDLE
    mode_since: float | None = None
    mode_note: str = ""
    current_goal: str | None = None
    current_intention: str | None = None
    confidence: float | None = None
    last_action: str | None = None
    action_result: str | None = None
    observation_summary: str | None = None
    beliefs: tuple[Belief, ...] = ()
    detected_entities: tuple[Belief, ...] = ()
    affect: AffectState = field(default_factory=AffectState)
    latest_thought: ThoughtEvent | None = None
    thought_history: tuple[ThoughtEvent, ...] = ()
    recent_events: tuple[AgentEvent, ...] = ()
    short_term_memory: tuple[AgentEvent, ...] = ()
    metrics: RunMetrics = field(default_factory=RunMetrics)
    safety: SafetyStatus = field(default_factory=SafetyStatus)
    frame: FrameInfo = field(default_factory=FrameInfo)
    look: LookReport = field(default_factory=LookReport)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", str(self.run_id))
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "demo", bool(self.demo))
        object.__setattr__(self, "beliefs", tuple(self.beliefs))
        object.__setattr__(self, "detected_entities", tuple(self.detected_entities))
        object.__setattr__(self, "thought_history", tuple(self.thought_history))
        object.__setattr__(self, "recent_events", tuple(self.recent_events))
        object.__setattr__(self, "short_term_memory", tuple(self.short_term_memory))
        if not isinstance(self.mode, AgentMode):
            object.__setattr__(self, "mode", AgentMode(str(self.mode)))
        if self.confidence is not None:
            object.__setattr__(self, "confidence", _unit(self.confidence, name="confidence"))

    def to_dict(self) -> dict[str, Any]:
        """Return the snapshot as plain JSON types.

        This is the only shape the HTTP layer ever emits for ``/api/snapshot``,
        which is what makes the read-only guarantee checkable: the page receives
        a description of the agent and nothing else.
        """
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "demo": self.demo,
            "mode": self.mode.value,
            "mode_since": self.mode_since,
            "mode_note": self.mode_note,
            "current_goal": self.current_goal,
            "current_intention": self.current_intention,
            "confidence": None if self.confidence is None else round(self.confidence, 4),
            "last_action": self.last_action,
            "action_result": self.action_result,
            "observation_summary": self.observation_summary,
            "beliefs": [belief.to_dict() for belief in self.beliefs],
            "detected_entities": [entity.to_dict() for entity in self.detected_entities],
            "affect_state": self.affect.as_dict(),
            "latest_thought": None if self.latest_thought is None else self.latest_thought.to_dict(),
            "thought_history": [thought.to_dict() for thought in self.thought_history],
            "recent_events": [event.to_dict() for event in self.recent_events],
            "short_term_memory": [event.to_dict() for event in self.short_term_memory],
            "metrics": self.metrics.to_dict(),
            "safety": self.safety.to_dict(),
            "frame": self.frame.to_dict(),
            "look": self.look.to_dict(),
        }


def with_thoughts(
    snapshot: ObserverSnapshot,
    thoughts: Sequence[ThoughtEvent],
) -> ObserverSnapshot:
    """Return ``snapshot`` with its thought views derived from ``thoughts``.

    ``thoughts`` is chronological. The latest is surfaced on its own because the
    page gives it a large panel, and the rest become the compact history, newest
    first, matching the specification's example stream view.
    """
    ordered = tuple(thoughts)
    return replace(
        snapshot,
        latest_thought=ordered[-1] if ordered else None,
        thought_history=tuple(reversed(ordered)),
    )


def with_events(
    snapshot: ObserverSnapshot,
    events: Sequence[AgentEvent],
    *,
    short_term_limit: int,
) -> ObserverSnapshot:
    """Return ``snapshot`` with its two event views derived from ``events``.

    Kept as a free function so the derivation rule - chronological for the log,
    newest-first for the memory panel - is stated once and can be tested on its
    own.
    """
    return replace(
        snapshot,
        recent_events=tuple(events),
        short_term_memory=tuple(reversed(list(events)[-short_term_limit:])),
    )

