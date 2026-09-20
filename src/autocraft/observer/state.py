"""The observer's read side, its write side, and the demo source.

The page is a *display*. It never asks the agent for anything and it never tells
the agent anything. What makes that true is the direction of the arrow in this
module: the agent's loop calls :meth:`ObserverState.publish_*`, the HTTP layer
calls :meth:`ObserverState.snapshot`, and nothing in the observer package ever
calls into the agent, the safety guard or the input backend.

Three consequences, all deliberate:

* **Publication is best-effort.** Every ``publish_*`` method swallows its own
  failures and records an event instead of raising. A diagnostic page must never
  be able to break a run, and it must certainly never be able to stop a stop.
* **Nothing here performs input.** There is no import of
  :mod:`autocraft.control` anywhere in this package, which is what makes the
  read-only guarantee a structural fact rather than a promise.
* **The demo source is a separate constructor.** :func:`demo_state` builds an
  :class:`ObserverState` whose every value comes from a script, whose snapshot is
  flagged ``demo=True``, and whose thoughts are stamped with the demo trigger. A
  demo cannot be confused with a live run because it is built by a different
  function and says so in its own payload.

Display frames are downscaled and JPEG-encoded once per published frame, on the
publishing thread, so the HTTP handler only ever copies bytes. Downscaling uses
block averaging rather than stride sampling: a rendered 3D scene sampled by
stride shimmers badly, and the whole point of the game view is that the operator
can tell what the agent is looking at.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..vision.frame import Frame, ScreenRegion
from ..thoughts.engine import ThoughtEngine, scripted_engine
from ..thoughts.model import ThoughtContext, ThoughtEvent
from .snapshot import (
    AFFECT_BASELINE,
    AffectState,
    AgentEvent,
    AgentMode,
    Belief,
    EmergencyStopState,
    EventKind,
    FrameFreshness,
    FrameInfo,
    LookReport,
    WakeReport,
    ObserverError,
    ObserverSnapshot,
    RunMetrics,
    SafetyStatus,
    frame_freshness,
    input_permitted,
    with_events,
    with_thoughts,
)

__all__ = [
    "DEMO_SCRIPT",
    "DEMO_FRAME_LIVE_SECONDS",
    "DEMO_FRAME_STALE_SECONDS",
    "EncodedFrame",
    "FRAME_CONTENT_TYPE",
    "JPEG_QUALITY",
    "ObserverState",
    "RATE_WINDOW_SECONDS",
    "SHORT_TERM_LIMIT",
    "demo_frame",
    "demo_state",
    "downscale_image",
    "encode_jpeg",
]

#: Trailing window over which frame and step rates are measured.
RATE_WINDOW_SECONDS = 5.0

#: Upper bound on the timestamp samples kept for rate measurement. A 5 s window
#: at 100 fps needs 500; 1024 leaves headroom without letting a long run grow.
_RATE_SAMPLES = 1024

#: How many timeline entries the compact memory panel shows.
SHORT_TERM_LIMIT = 8

#: Seconds between demo script steps.
DEMO_TICK_SECONDS = 2.0

#: Freshness thresholds used by the demo, whose scripted cadence is
#: ``DEMO_TICK_SECONDS``. Reusing the production thresholds would label a
#: two-second-old scripted frame "recent" purely because the demo runs slower
#: than a real loop, which would misdescribe the demo's own cadence.
DEMO_FRAME_LIVE_SECONDS = DEMO_TICK_SECONDS * 1.5
DEMO_FRAME_STALE_SECONDS = DEMO_TICK_SECONDS * 4.0

#: Quality of the display tile. This is a dashboard image, not a record, and the
#: saved frames in ``data/captures`` remain full quality.
JPEG_QUALITY = 85

#: Content type served from ``/frame``.
FRAME_CONTENT_TYPE = "image/jpeg"


# ---------------------------------------------------------------------------
# display frames
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncodedFrame:
    """An encoded display frame ready to be written to a socket."""

    data: bytes
    content_type: str = FRAME_CONTENT_TYPE
    captured_at: float = 0.0
    width: int = 0
    height: int = 0
    source_width: int = 0
    source_height: int = 0
    signature: str | None = None
    mean_luma: float | None = None
    source: str = "unknown"

    @property
    def size(self) -> int:
        """Encoded size in bytes."""
        return len(self.data)


def downscale_image(image: np.ndarray, max_width: int) -> np.ndarray:
    """Return ``image`` reduced to at most ``max_width`` pixels wide.

    Uses area averaging - every source pixel contributes - implemented as a
    reshape-and-mean, so no interpolation dependency is needed and the result is
    exact. A few pixels may be trimmed from the right and bottom edges when the
    dimensions are not divisible by the reduction factor, which is invisible at
    dashboard scale and much cheaper than padding.

    Returns ``image`` unchanged when it is already narrow enough or when the
    reduction cannot be expressed.
    """
    if max_width <= 0:
        return image
    if image.ndim != 3:
        raise ObserverError(f"expected an H x W x C image, got shape {image.shape!r}")
    height, width = image.shape[0], image.shape[1]
    if width <= max_width:
        return image
    factor = int(np.ceil(width / max_width))
    usable_w = width - (width % factor)
    usable_h = height - (height % factor)
    if factor < 2 or usable_w < factor or usable_h < factor:
        return image
    cropped = image[:usable_h, :usable_w]
    channels = image.shape[2]
    reduced = cropped.reshape(
        usable_h // factor, factor, usable_w // factor, factor, channels
    ).mean(axis=(1, 3), dtype=np.float64)
    return np.rint(reduced).astype(np.uint8)


def encode_jpeg(image: np.ndarray, *, quality: int = JPEG_QUALITY) -> bytes:
    """Encode an image as JPEG bytes.

    OpenCV is imported lazily, matching the rest of the project: it is a declared
    dependency but nothing pays for it until an image actually has to be encoded.
    The array must already be in the channel order OpenCV expects.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - opencv-python is a core dependency
        raise ObserverError(
            "encoding the display frame requires opencv-python; install it with "
            "'pip install opencv-python'"
        ) from exc
    ok, buffer = cv2.imencode(".jpg", np.ascontiguousarray(image), [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ObserverError("could not encode the display frame as JPEG")
    return bytes(buffer.tobytes())


def _rate(times: Sequence[float], now: float, *, window: float = RATE_WINDOW_SECONDS) -> float:
    """Events per second over the trailing ``window``, from event timestamps.

    Rates are computed from when things happened rather than when they were
    polled, so the read-out stays correct whatever rhythm the page happens to
    request snapshots at. Returns ``0.0`` until there are two samples, because a
    single event is not a rate.
    """
    recent = [stamp for stamp in times if now - stamp <= window]
    if len(recent) < 2:
        return 0.0
    span = recent[-1] - recent[0]
    if span <= 0:
        return 0.0
    return (len(recent) - 1) / span


# ---------------------------------------------------------------------------
# observer state
# ---------------------------------------------------------------------------


class ObserverState:
    """The published state of one agent run, safe to read from another thread.

    The agent writes through the ``publish_*`` methods; the HTTP layer reads
    through :meth:`snapshot`. All access is behind a lock, and reads never mutate
    anything, so a page can poll as often as it likes without perturbing the run.
    """

    def __init__(
        self,
        config: Any,
        *,
        clock: Callable[[], float] = time.time,
        demo: bool = False,
        affect: AffectState | None = None,
        thought_engine: ThoughtEngine | None = None,
        demo_script: Sequence[Any] = (),
    ) -> None:
        self._config = config
        self._clock = clock
        self._lock = threading.Lock()
        self._demo = bool(demo)

        self._run_id = "-"
        self._started_at = float(clock())
        self._mode = AgentMode.IDLE
        self._mode_since = self._started_at
        self._mode_note = ""

        self._goal: str = ""
        self._intention: str = ""
        self._observation_summary = ""
        self._last_action: str = ""
        self._action_result: str = ""
        self._confidence: float | None = None
        self._beliefs: tuple[Belief, ...] = ()
        self._entities: tuple[Belief, ...] = ()
        self._affect = affect if affect is not None else AFFECT_BASELINE

        self._events: deque[AgentEvent] = deque(maxlen=max(10, int(config.observer_max_events)))

        self._frames_observed = 0
        self._actions_attempted = 0
        self._actions_executed = 0
        self._actions_blocked = 0
        self._safety_events = 0
        self._errors = 0

        self._frame: EncodedFrame | None = None
        self._frame_at: float | None = None
        # Both timestamp deques are bounded: the rate helpers only ever look at a
        # fixed window, so an unbounded deque would grow for the length of a run
        # to answer a question about the last few seconds.
        self._frame_times: deque[float] = deque(maxlen=_RATE_SAMPLES)
        self._step_times: deque[float] = deque(maxlen=_RATE_SAMPLES)

        self._window_found = False
        self._window_foreground = False
        self._held_inputs: tuple[str, ...] = ()
        self._emergency_stop = EmergencyStopState.READY
        self._blocked_streak = 0
        self._last_block_reason: str | None = None
        self._control_available = True

        self._thought_engine = thought_engine
        self._thoughts: deque[ThoughtEvent] = deque(maxlen=max(1, int(config.thought_history_max)))

        self._look = LookReport()
        self._wake = WakeReport()

        self._demo_records: tuple[_DemoRecord, ...] = tuple(demo_script)
        self._demo_index = 0
        self._demo_ticks = 0
        self._last_demo_tick: float | None = None

    # -- read side --------------------------------------------------------

    @property
    def demo(self) -> bool:
        """True when every value here came from a script."""
        return self._demo

    @property
    def config(self) -> Any:
        """The configuration this state was built with."""
        return self._config

    def snapshot(self, *, now: float | None = None) -> ObserverSnapshot:
        """Return an immutable view of everything the page displays.

        Reads under the lock and computes derived values - frame age, freshness,
        rates - from stored timestamps. It does not mutate anything, so polling
        cannot affect the run.
        """
        moment = float(now) if now is not None else float(self._clock())
        with self._lock:
            events = tuple(self._events)
            snapshot = ObserverSnapshot(
                run_id=self._run_id,
                timestamp=moment,
                demo=self._demo,
                mode=self._mode,
                mode_since=self._mode_since,
                mode_note=self._mode_note,
                current_goal=self._goal or None,
                current_intention=self._intention or None,
                confidence=self._confidence,
                last_action=self._last_action or None,
                action_result=self._action_result or None,
                observation_summary=self._observation_summary or None,
                beliefs=self._beliefs,
                detected_entities=self._entities,
                affect=self._affect,
                metrics=self._metrics_locked(moment),
                safety=self._safety_locked(),
                frame=self._frame_info_locked(moment),
                look=self._look,
                wake=self._wake,
            )
            snapshot = with_events(snapshot, events, short_term_limit=SHORT_TERM_LIMIT)
            return with_thoughts(snapshot, tuple(self._thoughts))

    def snapshot_dict(self, *, now: float | None = None) -> dict[str, Any]:
        """Return :meth:`snapshot` as plain JSON types."""
        return self.snapshot(now=now).to_dict()

    def encoded_frame(self) -> EncodedFrame | None:
        """Return the current display frame, or ``None`` when there is none."""
        with self._lock:
            return self._frame

    @property
    def events(self) -> tuple[AgentEvent, ...]:
        """The timeline, chronological, as published so far."""
        with self._lock:
            return tuple(self._events)

    @property
    def thoughts(self) -> tuple[ThoughtEvent, ...]:
        """Thoughts expressed so far, oldest first."""
        with self._lock:
            return tuple(self._thoughts)

    @property
    def look(self) -> LookReport:
        """The most recent LOOK-001 measurement, or an empty report."""
        with self._lock:
            return self._look

    @property
    def wake(self) -> WakeReport:
        """The most recent WAKE-001 attempt, or an empty report."""
        with self._lock:
            return self._wake

    # -- write side -------------------------------------------------------

    def begin_run(self, run_id: str, *, now: float | None = None, goal: str = "") -> None:
        """Reset per-run counters and record the run identifier."""
        moment = float(now) if now is not None else float(self._clock())
        with self._lock:
            self._run_id = str(run_id)
            self._started_at = moment
            self._mode = AgentMode.IDLE
            self._mode_since = moment
            self._mode_note = "run started"
            self._goal = str(goal)
            self._intention = ""
            self._last_action = ""
            self._action_result = ""
            self._observation_summary = ""
            self._confidence = None
            self._frames_observed = 0
            self._actions_attempted = 0
            self._actions_executed = 0
            self._actions_blocked = 0
            self._safety_events = 0
            self._errors = 0
            self._frame_times.clear()
            self._step_times.clear()
            self._look = LookReport()
            self._wake = WakeReport()
        self.publish_event("Run started.", kind=EventKind.INFO, now=moment)

    def publish_mode(self, mode: AgentMode | str, *, note: str = "", now: float | None = None) -> None:
        """Record the agent's current mode, noting when it last changed."""
        moment = float(now) if now is not None else float(self._clock())
        resolved = mode if isinstance(mode, AgentMode) else AgentMode(str(mode))
        with self._lock:
            if resolved is not self._mode:
                self._mode = resolved
                self._mode_since = moment
            if note:
                self._mode_note = str(note)

    def publish_goal(self, goal: str | None, *, intention: str | None = None) -> None:
        """Record what the agent is trying to accomplish and how, this step."""
        with self._lock:
            self._goal = "" if goal is None else str(goal)
            if intention is not None:
                self._intention = str(intention)

    def publish_affect(self, affect: AffectState) -> None:
        """Record the current simulated affect state."""
        if not isinstance(affect, AffectState):
            raise ObserverError("publish_affect expects an AffectState")
        with self._lock:
            self._affect = affect

    def publish_beliefs(self, beliefs: Sequence[Belief], *, entities: Sequence[Belief] = ()) -> None:
        """Record structured interpretations. V0 has none, and says so."""
        with self._lock:
            self._beliefs = tuple(beliefs)
            self._entities = tuple(entities)

    def publish_event(
        self,
        message: str,
        *,
        kind: EventKind | str = EventKind.INFO,
        now: float | None = None,
    ) -> None:
        """Append one entry to the timeline."""
        moment = float(now) if now is not None else float(self._clock())
        resolved = kind if isinstance(kind, EventKind) else EventKind(str(kind))
        with self._lock:
            self._events.append(AgentEvent(timestamp=moment, message=str(message), kind=resolved))

    def publish_thought(self, thought: ThoughtEvent) -> None:
        """Record an expressed thought. Display-only: nothing reads it back."""
        with self._lock:
            self._thoughts.append(thought)

    def publish_look(
        self,
        report: LookReport | None = None,
        *,
        now: float | None = None,
        **fields: Any,
    ) -> LookReport:
        """Record a LOOK-001 measurement for the visual-motion panel.

        Accepts either a finished :class:`LookReport` or keyword overrides merged
        onto the current one, so the CLI can update the panel per trial without
        rebuilding the whole report. Display-only: nothing reads this back, and
        publishing it cannot inject input or change a run.
        """
        moment = float(now) if now is not None else float(self._clock())
        with self._lock:
            if report is not None:
                merged = report
            else:
                merged = replace(self._look, **fields)
            if merged.measured_at is None:
                merged = replace(merged, measured_at=moment)
            self._look = merged
            return merged

    def publish_wake(
        self,
        report: WakeReport | None = None,
        *,
        now: float | None = None,
        **fields: Any,
    ) -> WakeReport:
        """Record a WAKE-001 attempt for the wake panel.

        Accepts either a finished :class:`WakeReport` or keyword overrides merged
        onto the current one, so the CLI can refresh the panel after every step
        without rebuilding the whole report. Display-only: nothing reads this
        back, and publishing it cannot inject input or change a run. In
        particular it cannot influence which region the agent picks or how far it
        turns, which is what keeps the page a window rather than a control.
        """
        moment = float(now) if now is not None else float(self._clock())
        with self._lock:
            if report is not None:
                merged = report
            else:
                merged = replace(self._wake, **fields)
            if merged.measured_at is None:
                merged = replace(merged, measured_at=moment)
            self._wake = merged
            return merged

    def publish_frame(
        self,
        frame: Frame,
        *,
        now: float | None = None,
        source: str | None = None,
    ) -> None:
        """Publish the latest captured frame as a downscaled JPEG display tile.

        Downscaling and encoding happen here, once per frame, so the HTTP handler
        only ever copies bytes it already has. Signature and mean luminance are
        measured on the tile rather than on the source frame: hashing a
        multi-megapixel frame on every step would cost more than the capture it
        describes, and the tile is what the operator is actually looking at.
        """
        moment = float(now) if now is not None else float(self._clock())
        try:
            rgb = np.asarray(frame.image)
            tile = downscale_image(rgb, int(self._config.observer_frame_max_width))
            if tile is not rgb:
                tile = np.ascontiguousarray(tile)
            tile_frame = Frame(
                image=tile,
                timestamp=float(getattr(frame, "timestamp", moment)),
                region=frame.region if isinstance(frame.region, ScreenRegion) else None,
                source=str(source or getattr(frame, "source", "unknown")),
            )
            data = encode_jpeg(tile[:, :, ::-1])
        except Exception as exc:  # best effort: a bad frame must not break a run
            self._record_failure("Could not publish the display frame.", exc)
            return
        with self._lock:
            self._frames_observed += 1
            self._frame = EncodedFrame(
                data=data,
                captured_at=moment,
                width=int(tile.shape[1]),
                height=int(tile.shape[0]),
                source_width=int(rgb.shape[1]),
                source_height=int(rgb.shape[0]),
                signature=tile_frame.signature(),
                mean_luma=tile_frame.mean_luma(),
                source=tile_frame.source,
            )
            self._frame_at = moment
            self._frame_times.append(moment)

    def publish_observation(
        self,
        *,
        summary: str | None = None,
        window_found: bool | None = None,
        window_foreground: bool | None = None,
        now: float | None = None,
    ) -> None:
        """Record what the agent currently sees, and whether the target is usable.

        The timeline gains an event only when the description actually changes, so
        a long observation session shows the moment the target window appeared or
        went away rather than one line per captured frame.
        """
        moment = float(now) if now is not None else float(self._clock())
        changed: str | None = None
        with self._lock:
            if summary is not None and str(summary) != self._observation_summary:
                self._observation_summary = str(summary)
                changed = self._observation_summary
            if window_found is not None:
                self._window_found = bool(window_found)
            if window_foreground is not None:
                self._window_foreground = bool(window_foreground)
            self._step_times.append(moment)
        if changed is not None:
            self.publish_event(changed, kind=EventKind.INFO, now=moment)

    def publish_confidence(self, confidence: float | None) -> None:
        """Record how confident the agent is in its current interpretation."""
        with self._lock:
            self._confidence = None if confidence is None else float(confidence)

    def publish_action(self, description: str, *, result: str = "", now: float | None = None) -> None:
        """Record the last action and its result, for the state panel."""
        moment = float(now) if now is not None else float(self._clock())
        with self._lock:
            self._last_action = str(description)
            self._action_result = str(result)
            if self._mode is AgentMode.IDLE:
                self._mode = AgentMode.ACTING
                self._mode_since = moment

    def publish_action_outcome(
        self,
        *,
        attempted: bool,
        executed: bool,
        blocked_reason: str | None = None,
        error: str | None = None,
    ) -> None:
        """Count one action outcome exactly as the executor reported it.

        The three counters mirror :class:`~autocraft.agent.action.ActionResult`'s
        orthogonal flags - attempted, executed, refused - so the page cannot
        claim an action reached the game when it did not.
        """
        with self._lock:
            if attempted:
                self._actions_attempted += 1
            if executed:
                self._actions_executed += 1
            if blocked_reason is not None:
                self._actions_blocked += 1
            if error is not None:
                self._errors += 1

    def publish_safety(
        self,
        *,
        window_found: bool | None = None,
        window_foreground: bool | None = None,
        emergency_stop: EmergencyStopState | str | None = None,
        held_inputs: Sequence[str] | None = None,
        blocked_streak: int | None = None,
        last_block_reason: str | None = None,
        safety_event_total: int | None = None,
        control_available: bool | None = None,
        now: float | None = None,
    ) -> None:
        """Copy the safety layer's own verdict across. Never recompute it here.

        ``safety_event_total`` is the guard's *cumulative* event count rather
        than a flag, so publishing on every poll is idempotent and the page cannot
        inflate the number by refreshing. It is applied with ``max`` because a
        counter only ever grows within a run.

        ``control_available`` says whether a safety guard exists at all. A
        read-only ``observe`` session has none, and the page must say so rather
        than imply that input is one keystroke away.
        """
        moment = float(now) if now is not None else float(self._clock())
        resolved_stop: EmergencyStopState | None = None
        if emergency_stop is not None:
            resolved_stop = (
                emergency_stop
                if isinstance(emergency_stop, EmergencyStopState)
                else EmergencyStopState(str(emergency_stop))
            )
        with self._lock:
            if window_found is not None:
                self._window_found = bool(window_found)
            if window_foreground is not None:
                self._window_foreground = bool(window_foreground)
            if resolved_stop is not None:
                self._emergency_stop = resolved_stop
            if held_inputs is not None:
                self._held_inputs = tuple(str(item) for item in held_inputs)
            if blocked_streak is not None:
                self._blocked_streak = int(blocked_streak)
            if last_block_reason is not None:
                self._last_block_reason = str(last_block_reason)
            if safety_event_total is not None:
                self._safety_events = max(self._safety_events, int(safety_event_total))
            if control_available is not None:
                self._control_available = bool(control_available)
        if resolved_stop is EmergencyStopState.TRIGGERED:
            self.publish_event("Emergency stop triggered.", kind=EventKind.SAFETY, now=moment)

    def publish_error(self, message: str, *, now: float | None = None) -> None:
        """Record an error against the run and on the timeline."""
        with self._lock:
            self._errors += 1
        self.publish_event(message, kind=EventKind.ERROR, now=now)

    # -- thoughts ---------------------------------------------------------

    def attach_thought_engine(self, engine: ThoughtEngine | None) -> None:
        """Attach the expression layer. ``None`` disables thoughts entirely."""
        self._thought_engine = engine

    def maybe_express_thought(
        self,
        *,
        now: float | None = None,
        memories: Sequence[str] | None = None,
    ) -> ThoughtEvent | None:
        """Offer the current state to the thought engine and publish any result.

        The context is assembled here, from what the agent has already published,
        so a generator is never handed more than the specification allows. When no
        memories are supplied the run's own timeline is used, because that is
        AutoCraft's memory system in V0 - which means a thought can only refer to
        something that was genuinely recorded.
        """
        engine = self._thought_engine
        if engine is None:
            return None
        moment = float(now) if now is not None else float(self._clock())
        with self._lock:
            context = ThoughtContext(
                goal=self._goal,
                intention=self._intention,
                observation_summary=self._observation_summary,
                recent_events=tuple(event.message for event in self._events),
                memories=tuple(memories)
                if memories is not None
                else tuple(event.message for event in self._events),
                affect=self._affect.as_dict(),
                mode=self._mode.value,
                recent_action_rate=_rate(self._step_times, moment),
                now=moment,
            )
        thought = engine.maybe_think(context)
        if thought is not None:
            self.publish_thought(thought)
        return thought

    # -- demo -------------------------------------------------------------

    def tick(self, *, now: float | None = None) -> ObserverSnapshot:
        """Advance a demo script one step. A no-op for a live run."""
        moment = float(now) if now is not None else float(self._clock())
        if not self._demo or not self._demo_records:
            return self.snapshot(now=moment)
        record = self._demo_records[self._demo_index % len(self._demo_records)]
        self._demo_index += 1
        self._demo_ticks += 1
        self._last_demo_tick = moment

        with self._lock:
            self._mode = record.mode
            self._mode_since = moment
            self._mode_note = record.note
            self._goal = record.goal
            self._intention = record.intention
            self._observation_summary = record.observation
            self._last_action = record.action
            self._action_result = record.result
            self._window_found = record.window_found
            self._window_foreground = record.window_foreground
            self._affect = self._affect.adjust(**dict(record.affect_delta))
            if record.counts_action:
                self._actions_attempted += 1
                if record.result.startswith("Completed"):
                    self._actions_executed += 1
                else:
                    self._actions_blocked += 1
            # The scripted step *is* this run's loop, so its rate is measurable.
            self._step_times.append(moment)
        self.publish_event(record.note, kind=record.kind, now=moment)
        self.publish_frame(demo_frame(now=moment), now=moment, source="demo")
        self.maybe_express_thought(now=moment)
        return self.snapshot(now=moment)

    @property
    def demo_ticks(self) -> int:
        """How many script steps the demo has taken."""
        return self._demo_ticks

    @property
    def seconds_until_demo_tick(self) -> float:
        """Seconds until the next script step is due, for the server's ticker."""
        if not self._demo:
            return float("inf")
        if self._last_demo_tick is None:
            return 0.0
        return max(0.0, DEMO_TICK_SECONDS - (float(self._clock()) - self._last_demo_tick))

    # -- internals --------------------------------------------------------

    def _record_failure(self, message: str, exc: BaseException) -> None:
        """Note a publication failure without propagating it."""
        try:
            with self._lock:
                self._errors += 1
                self._events.append(
                    AgentEvent(
                        timestamp=float(self._clock()),
                        message=f"{message} ({type(exc).__name__}: {exc})",
                        kind=EventKind.ERROR,
                    )
                )
        except Exception:  # pragma: no cover - the fallback must not raise either
            pass

    def _metrics_locked(self, moment: float) -> RunMetrics:
        return RunMetrics(
            run_id=self._run_id,
            started_at=self._started_at,
            runtime_seconds=max(0.0, moment - self._started_at),
            frames_observed=self._frames_observed,
            capture_fps=_rate(self._frame_times, moment),
            loop_rate=_rate(self._step_times, moment),
            actions_attempted=self._actions_attempted,
            actions_executed=self._actions_executed,
            actions_blocked=self._actions_blocked,
            safety_events=self._safety_events,
            errors=self._errors,
            thoughts_expressed=len(self._thoughts),
        )

    def _safety_locked(self) -> SafetyStatus:
        return SafetyStatus(
            window_found=self._window_found,
            window_foreground=self._window_foreground,
            input_enabled=input_permitted(
                window_found=self._window_found,
                window_foreground=self._window_foreground,
                emergency_stop=self._emergency_stop,
                require_foreground=bool(getattr(self._config, "require_foreground", True)),
                control_available=self._control_available,
            ),
            emergency_stop=self._emergency_stop,
            held_inputs=self._held_inputs,
            blocked_streak=self._blocked_streak,
            last_block_reason=self._last_block_reason,
            control_available=self._control_available,
        )

    def _frame_info_locked(self, moment: float) -> FrameInfo:
        if self._frame is None or self._frame_at is None:
            return FrameInfo(available=False, freshness=FrameFreshness.NONE)
        age = max(0.0, moment - self._frame_at)
        return FrameInfo(
            available=True,
            captured_at=self._frame_at,
            age_seconds=age,
            freshness=frame_freshness(
                age,
                live_seconds=float(self._config.observer_frame_live_seconds),
                stale_seconds=float(self._config.observer_frame_stale_seconds),
            ),
            width=self._frame.width,
            height=self._frame.height,
            source_width=self._frame.source_width,
            source_height=self._frame.source_height,
            signature=self._frame.signature,
            mean_luma=self._frame.mean_luma,
            reference="/frame",
            source=self._frame.source,
        )


# ---------------------------------------------------------------------------
# demo source
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DemoRecord:
    """One step of the demo script. Not a simulation of the agent's logic."""

    mode: AgentMode
    goal: str
    intention: str
    observation: str
    action: str
    result: str
    note: str
    kind: EventKind = EventKind.INFO
    affect_delta: Mapping[str, float] = field(default_factory=dict)
    window_found: bool = True
    window_foreground: bool = True
    counts_action: bool = True


#: A short, plainly fictional run used to populate the page when there is no
#: agent. Every value is authored here, none of it is measured, and the snapshot
#: it produces is flagged ``demo`` so the page can say so.
DEMO_SCRIPT: tuple[_DemoRecord, ...] = (
    _DemoRecord(
        mode=AgentMode.OBSERVING,
        goal="Find wood",
        intention="Look around before committing to a direction",
        observation="Demo frame: a horizon, some trees, nothing close.",
        action="No action",
        result="Nothing attempted",
        note="DEMO: frame captured.",
        kind=EventKind.OBSERVE,
        affect_delta={"curiosity": 0.05},
        counts_action=False,
    ),
    _DemoRecord(
        mode=AgentMode.ACTING,
        goal="Find wood",
        intention="Rotate toward the nearest visible tree",
        observation="Demo frame: a tree is left of centre.",
        action="Rotate camera +8 degrees",
        result="Completed",
        note="DEMO: camera movement requested.",
        affect_delta={"curiosity": 0.03, "confidence": 0.04},
    ),
    _DemoRecord(
        mode=AgentMode.ACTING,
        goal="Find wood",
        intention="Approach the tree until it fills the crosshair",
        observation="Demo frame: trunk near the centre, ground close.",
        action="Move forward for 0.4 s",
        result="Completed",
        note="DEMO: action accepted by the safety layer.",
        affect_delta={"confidence": 0.05, "stress": -0.02},
    ),
    _DemoRecord(
        mode=AgentMode.ACTING,
        goal="Find wood",
        intention="Hold attack on the trunk",
        observation="Demo frame: crosshair on bark.",
        action="Hold attack for 1.2 s",
        result="Completed",
        note="DEMO: visual change detected.",
        affect_delta={"confidence": 0.06, "frustration": -0.04},
    ),
    _DemoRecord(
        mode=AgentMode.OBSERVING,
        goal="Collect a second log",
        intention="Re-acquire the tree after the first break",
        observation="Demo frame: the trunk is gone, leaves remain.",
        action="No action",
        result="Nothing attempted",
        note="DEMO: verifying the change.",
        kind=EventKind.OBSERVE,
        affect_delta={"curiosity": 0.04},
        counts_action=False,
    ),
    _DemoRecord(
        mode=AgentMode.ACTING,
        goal="Collect a second log",
        intention="Step toward the next trunk",
        observation="Demo frame: two trunks ahead.",
        action="Move forward for 0.3 s",
        result="Refused: target window is not foreground",
        note="DEMO: action refused by the safety layer.",
        kind=EventKind.SAFETY,
        affect_delta={"frustration": 0.08, "stress": 0.05, "confidence": -0.05},
        window_foreground=False,
    ),
    _DemoRecord(
        mode=AgentMode.PAUSED,
        goal="Collect a second log",
        intention="Wait for the window to come back to the foreground",
        observation="Demo frame: unchanged, focus lost.",
        action="No action",
        result="Nothing attempted",
        note="DEMO: paused, waiting for focus.",
        kind=EventKind.SAFETY,
        affect_delta={"stress": -0.03, "frustration": -0.02},
        counts_action=False,
    ),
    _DemoRecord(
        mode=AgentMode.OBSERVING,
        goal="Collect a second log",
        intention="Resume once the target is focused again",
        observation="Demo frame: target focused again.",
        action="No action",
        result="Nothing attempted",
        note="DEMO: focus restored.",
        kind=EventKind.SAFETY,
        affect_delta={"stress": -0.04, "confidence": 0.03},
        counts_action=False,
    ),
)


def demo_frame(*, now: float = 0.0, width: int = 640, height: int = 360) -> Frame:
    """Build a synthetic frame for demo mode.

    A gradient with a slow-moving block, drawn from ``now`` so successive ticks
    differ visibly and the frame age shown on the page means something. It is
    obviously not Luanti: the page labels it DEMO and the frame source says so
    too, because a synthetic picture presented as a real capture would be a lie
    about what the agent can see.
    """
    rows = np.linspace(24.0, 96.0, height, dtype=np.float64)[:, None]
    cols = np.linspace(0.0, 18.0, width, dtype=np.float64)[None, :]
    base = np.repeat(rows + cols, 3, axis=0).reshape(height, width, 3)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = np.clip(base[:, :, 0] * 0.5, 0, 255)
    image[:, :, 1] = np.clip(base[:, :, 1] * 0.9, 0, 255)
    image[:, :, 2] = np.clip(base[:, :, 2] * 1.2, 0, 255)

    block_w, block_h = max(8, width // 8), max(8, height // 5)
    span = max(1, width - block_w)
    left = int((now * 37.0) % span)
    top = height // 2 - block_h // 2
    image[top : top + block_h, left : left + block_w] = (235, 210, 90)

    horizon = height // 2
    image[horizon : horizon + 2, :] = (140, 130, 120)
    return Frame(
        image=image,
        timestamp=float(now),
        region=ScreenRegion(0, 0, width, height),
        source="demo",
    )


def demo_state(
    config: Any,
    *,
    clock: Callable[[], float] = time.time,
    thoughts: tuple[str, ...] | None = None,
    advance: bool = True,
) -> ObserverState:
    """Build an :class:`ObserverState` driven entirely by :data:`DEMO_SCRIPT`.

    The affect model is not randomised to look busy. The script applies the
    changes a real experience would produce - success raises confidence and
    lowers frustration, a refusal raises frustration and lowers confidence - so
    the demo shows what the affect contract is for, rather than pretending the
    agent is feeling something.
    """
    state = ObserverState(
        replace(
            config,
            observer_frame_live_seconds=DEMO_FRAME_LIVE_SECONDS,
            observer_frame_stale_seconds=DEMO_FRAME_STALE_SECONDS,
        ),
        clock=clock,
        demo=True,
        affect=AFFECT_BASELINE,
        thought_engine=scripted_engine(config, clock=clock, thoughts=thoughts),
        demo_script=DEMO_SCRIPT,
    )
    state.begin_run("demo-0000", now=float(clock()), goal="Find wood")
    state.publish_event("DEMO MODE: all values on this page are scripted.", kind=EventKind.INFO)
    if advance:
        state.tick()
    return state
