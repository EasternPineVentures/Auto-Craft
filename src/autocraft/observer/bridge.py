"""The adapter that turns a running agent loop into observer publications.

This is the one place where the two sides meet, and the meeting is deliberately
one-directional: the bridge is handed observations and step records and calls
``publish_*``. Nothing here returns a value the loop acts on, and nothing here
can reach an executor, a keyboard or a safety guard. The safety numbers the page
shows are supplied by the caller as a plain mapping, so the display layer never
holds a reference to the authority that could stop a run.

Keeping the adapter here rather than in the loop means the agent layer stays
unaware that a page exists at all: ``AgentLoop`` gained two optional callbacks
and no knowledge of HTTP, JSON or JavaScript.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .snapshot import EventKind, AgentMode

__all__ = ["LoopPublisher"]


class LoopPublisher:
    """Feeds one :class:`~autocraft.observer.state.ObserverState` from a live run.

    Wire it up as the loop's two callbacks::

        publisher = LoopPublisher(state, goal="TREE-001")
        loop = AgentLoop(..., on_observation=publisher.on_observation,
                         progress=publisher.on_step)

    Both callbacks are total: a publication failure is recorded as an observer
    error rather than raised, because a broken dashboard must never be able to
    end a run.
    """

    def __init__(
        self,
        state: Any,
        *,
        goal: str | None = None,
        intention: str | None = None,
        safety: Callable[[], Mapping[str, Any]] | None = None,
        confidence: Callable[[Any], float | None] | None = None,
        express_thoughts: bool = True,
    ) -> None:
        """
        Args:
            state: The :class:`ObserverState` to publish into.
            goal: What the run is trying to accomplish, if anything. V0 has no
                goal, and passing ``None`` leaves the field honestly empty.
            intention: How the agent intends to pursue the goal.
            safety: Zero-argument provider returning the safety layer's own
                verdict as a mapping of :meth:`ObserverState.publish_safety`
                keywords. Supplied by the caller so this module never holds a
                reference to the guard.
            confidence: Optional ``observation -> float | None`` provider. V0 has
                no perception, so the default leaves confidence unset rather than
                inventing a number.
            express_thoughts: Whether to offer each step to the thought engine.
        """
        self._state = state
        self._goal = goal
        self._intention = intention
        self._safety = safety
        self._confidence = confidence
        self._express_thoughts = bool(express_thoughts)

    # -- lifecycle --------------------------------------------------------

    def begin(self, run_id: str, *, now: float | None = None) -> None:
        """Reset the observer for a fresh run and state the goal, if there is one."""
        self._guard(lambda: self._state.begin_run(run_id, now=now, goal=self._goal or ""))
        if self._intention:
            self._guard(lambda: self._state.publish_goal(self._goal, intention=self._intention))
        self._publish_safety()
        self._guard(lambda: self._state.publish_mode(AgentMode.OBSERVING, note="run started", now=now))

    def finish(self, record: Any, *, now: float | None = None) -> None:
        """Record the end of the run, including the recorder's own status."""
        status = str(getattr(record, "status", "") or "")
        reason = str(getattr(record, "stop_reason", "") or "")
        if status in {"error", "failed"}:
            mode, kind = AgentMode.ERROR, EventKind.ERROR
        elif status == "stopped":
            mode, kind = AgentMode.SAFE_STOP, EventKind.SAFETY
        else:
            mode, kind = AgentMode.IDLE, EventKind.INFO
        message = f"Run finished: {status or 'unknown'} ({reason or 'no reason recorded'})."
        self._guard(lambda: self._state.publish_mode(mode, note=reason or status, now=now))
        self._guard(lambda: self._state.publish_event(message, kind=kind, now=now))
        self._publish_safety()

    # -- loop callbacks ---------------------------------------------------

    def on_observation(self, observation: Any) -> None:
        """Publish the frame and the window status for one step.

        Called before the policy runs, so the page shows what the agent is about
        to decide *from* rather than what it decided. The frame is handed to the
        state object, which downscales and encodes it once; this method never
        copies pixels itself.
        """
        frame = getattr(observation, "frame", None)
        window = getattr(observation, "window", None)
        summary = self._summarise(observation)
        found = bool(getattr(window, "found", False))
        foreground = bool(getattr(window, "is_foreground", False))
        self._guard(
            lambda: self._state.publish_observation(
                summary=summary,
                window_found=found,
                window_foreground=foreground,
                now=float(getattr(observation, "timestamp", 0.0)) or None,
            )
        )
        if frame is not None:
            self._guard(lambda: self._state.publish_frame(frame, source=getattr(frame, "source", None)))
        self._publish_safety()

    def on_step(self, record: Any) -> None:
        """Publish one step's action, result, counters and (optionally) a thought."""
        action = _as_mapping(getattr(record, "action", None))
        result = _as_mapping(getattr(record, "result", None))
        description = str(action.get("description") or action.get("kind") or "unknown action")
        outcome = _describe_outcome(result, getattr(record, "notes", ""))
        self._guard(lambda: self._state.publish_action(description, result=outcome))
        self._guard(
            lambda: self._state.publish_action_outcome(
                attempted=bool(result.get("attempted", False)),
                executed=bool(result.get("executed", False)),
                blocked_reason=_optional_str(result.get("blocked_reason")),
                error=_optional_str(result.get("error")),
            )
        )
        if self._confidence is not None:
            try:
                value = self._confidence(record)
            except Exception:  # a confidence provider must not break the run
                value = None
            self._guard(lambda: self._state.publish_confidence(value))
        self._publish_safety()
        if self._express_thoughts:
            self._guard(lambda: self._state.maybe_express_thought())

    # -- internals --------------------------------------------------------

    def _publish_safety(self) -> None:
        if self._safety is None:
            return
        try:
            payload = dict(self._safety())
        except Exception:  # the safety provider is best effort too
            return
        self._guard(lambda: self._state.publish_safety(**payload))

    def _guard(self, call: Callable[[], None]) -> None:
        try:
            call()
        except Exception as exc:
            self._state.publish_error(
                f"Observer publication failed ({type(exc).__name__}: {exc})."
            )

    @staticmethod
    def _summarise(observation: Any) -> str:
        """Describe one observation in the operator's terms, without inventing."""
        window = getattr(observation, "window", None)
        if window is None:
            return "No window information was returned."
        if not getattr(window, "found", False):
            reason = str(getattr(window, "reason", "") or "no matching window")
            return f"Target window not found: {reason}."
        if getattr(window, "minimized", False):
            return "Target window is minimised, so its client area cannot be captured."
        region = getattr(window, "region", None)
        size = ""
        if region is not None:
            try:
                size = f" at {region.width}x{region.height}"
            except Exception:  # pragma: no cover - defensive against odd regions
                size = ""
        if not getattr(window, "is_foreground", False):
            return f"Target window is visible{size} but not focused, so input is refused."
        frame = getattr(observation, "frame", None)
        if frame is None:
            error = str(getattr(observation, "capture_error", "") or "")
            return f"Target window is focused{size}; the capture returned no frame. {error}".strip()
        return f"Captured the focused target window's client area{size}."


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_str(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def _describe_outcome(result: Mapping[str, Any], notes: Any) -> str:
    """Render a result the way the executor reported it, not more flatteringly."""
    blocked = _optional_str(result.get("blocked_reason"))
    if blocked is not None:
        return f"Refused by the safety layer: {blocked}"
    error = _optional_str(result.get("error"))
    if error is not None:
        return f"Failed: {error}"
    if not result.get("attempted", False):
        note = _optional_str(notes)
        return f"No input was sent. {note}".strip() if note else "No input was sent."
    if result.get("executed", False):
        return "Input reached the game."
    return "Input was attempted but did not reach the game."


def publish_safety_from(guard: Any) -> dict[str, Any]:
    """Read the safety guard's own verdict as ``publish_safety`` keywords.

    This lives beside the bridge so the CLI has one place to call, and it only
    ever *reads*. The returned mapping is plain data: once it is out of here the
    page cannot reach the guard again.
    """
    keys: Sequence[str] = tuple(getattr(guard, "held_keys", ()) or ())
    buttons: Sequence[str] = tuple(getattr(guard, "held_buttons", ()) or ())
    events = tuple(getattr(guard, "events", ()) or ())
    stop_requested = bool(getattr(guard, "stop_requested", False))
    return {
        "emergency_stop": "triggered" if stop_requested else "ready",
        "held_inputs": tuple([*keys, *buttons]),
        "blocked_streak": int(getattr(guard, "blocked_streak", 0) or 0),
        "last_block_reason": _optional_str(getattr(guard, "last_block_reason", None)),
        "safety_event_total": len(events),
    }
