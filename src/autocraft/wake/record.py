"""The WAKE-001 result record.

The milestone requires that every meaningful action and observation is recorded
truthfully, so this module's whole job is to make an honest record cheap to write
and hard to overstate.

Two rules are structural rather than stylistic, and they are the same two the
LOOK panel follows:

* Every measured field defaults to ``None`` (or ``False``), so a run that never
  measured something cannot render as a row of zeroes that reads like a result.
  "No candidate was found" and "the run never got that far" are different facts
  and they are stored differently.
* ``successful_completion`` is ``None`` until the run says otherwise. It is not a
  default-true flag: a WAKE run that ends by exhausting its budget is a truthful
  outcome, not a failure to be smoothed over, and the record says which one
  happened in words as well as in the flag.

The result is written to ``wake_result.json`` in the run's directory and is
deliberately small and greppable. No pixel data is serialised here; frames belong
to the telemetry layer, which references them by path.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .events import WakeEvent, stream_summary

__all__ = [
    "EXPERIMENT_NAME",
    "RESULT_FILENAME",
    "WakeRecorder",
    "WakeResult",
]

#: The milestone this record belongs to.
EXPERIMENT_NAME = "WAKE-001"

#: Name of the result file inside a run directory.
RESULT_FILENAME = "wake_result.json"

#: Outcome labels. ``COMPLETED`` means the agent stopped because it was finished,
#: ``FAILED`` means it stopped because it ran out of something, and ``ABORTED``
#: means the world stopped being safe to act on.
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_ABORTED = "aborted"


@dataclass(frozen=True)
class WakeResult:
    """One WAKE-001 run, as measured.

    Attributes:
        experiment: The milestone name.
        run_id: The telemetry run this belongs to.
        started_at: Wall clock at the start.
        finished_at: Wall clock at the end, or ``None`` while running.
        status: ``running``, ``completed``, ``failed`` or ``aborted``.
        state: The behaviour layer's terminal state, e.g. ``COMPLETE``.
        stop_reason: Plain words for why the run stopped. Always set on a
            finished run, including a successful one.
        directory: Where the record lives.
        window_width: Width of the window the run acted on.
        window_height: Height of the window the run acted on.
        max_moves: The movement budget the run was given.
        moves_sent: Movements actually handed to the actuator.
        scan_moves: Movements spent looking around.
        unique_views: Distinct views the run recognised.
        revisited_views: Looks that matched a view it had already seen.
        candidate_count: Salient regions the run considered.
        target_changes: How many times the selected region changed.
        centering_moves: Movements spent bringing the target to the centre.
        overshoots: Times a correction crossed the centre and had to come back.
        failed_strategies: Approaches given up on.
        stuck_patterns_detected: Repetition loops the guard noticed.
        stuck_patterns_broken: Repetition loops that were escaped.
        final_target_offset: The last measured offset from the centre, in pixels.
        final_target_distance: The last measured distance from the centre, in pixels.
        target_centre: Where the selected region was last measured, in pixels.
        target_bbox: The selected region's last bounding box, in pixels.
        target_salience: How much the selected region stood out from its
            neighbours when it was chosen, ``0..1``.
        confidence: How sure the run was of the target's location, ``0..1``.
        progress: The measured distance after each centring move, in pixels.
        strategy: The strategy in force when the run stopped.
        dead_repetition_ratio: Share of measured outcomes that made no progress.
        productive_repetition_ratio: Share that measurably reduced the distance.
        mapping_source: Where the mouse-to-pixel mapping came from, e.g.
            ``unmeasured``, ``look-001`` or ``self-measured``.
        mapping_quality: How good that mapping is, ``0..1``.
        pixels_per_delta_x: Measured horizontal pixels per mouse count.
        pixels_per_delta_y: Measured vertical pixels per mouse count.
        events: The structured event stream, in order.
        steps: How many loop steps the run took.
        notes: Free-text notes worth keeping with the record.
        measured_at: Wall clock when the record was last written.
    """

    experiment: str = EXPERIMENT_NAME
    run_id: str = ""
    started_at: float = 0.0
    finished_at: float | None = None
    status: str = STATUS_RUNNING
    state: str = ""
    stop_reason: str = ""
    directory: str | None = None
    window_width: int = 0
    window_height: int = 0
    max_moves: int = 0
    moves_sent: int = 0
    scan_moves: int = 0
    unique_views: int = 0
    revisited_views: int = 0
    candidate_count: int = 0
    target_changes: int = 0
    centering_moves: int = 0
    overshoots: int = 0
    failed_strategies: int = 0
    stuck_patterns_detected: int = 0
    stuck_patterns_broken: int = 0
    final_target_offset: tuple[float, float] | None = None
    final_target_distance: float | None = None
    target_centre: tuple[float, float] | None = None
    target_bbox: tuple[int, int, int, int] | None = None
    target_salience: float | None = None
    confidence: float | None = None
    progress: tuple[float, ...] = ()
    strategy: str = ""
    dead_repetition_ratio: float | None = None
    productive_repetition_ratio: float | None = None
    mapping_source: str = ""
    mapping_quality: float | None = None
    pixels_per_delta_x: float | None = None
    pixels_per_delta_y: float | None = None
    events: tuple[WakeEvent, ...] = ()
    steps: int = 0
    notes: tuple[str, ...] = ()
    measured_at: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment", str(self.experiment))
        object.__setattr__(self, "run_id", str(self.run_id))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "state", str(self.state))
        object.__setattr__(self, "stop_reason", str(self.stop_reason))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "progress", tuple(float(value) for value in self.progress))
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))
        for name in (
            "window_width",
            "window_height",
            "max_moves",
            "moves_sent",
            "scan_moves",
            "unique_views",
            "revisited_views",
            "candidate_count",
            "target_changes",
            "centering_moves",
            "overshoots",
            "failed_strategies",
            "stuck_patterns_detected",
            "stuck_patterns_broken",
            "steps",
        ):
            object.__setattr__(self, name, int(getattr(self, name)))
        if self.target_bbox is not None:
            object.__setattr__(self, "target_bbox", tuple(int(v) for v in self.target_bbox))
        if self.final_target_offset is not None:
            object.__setattr__(
                self, "final_target_offset", tuple(float(v) for v in self.final_target_offset)
            )
        if self.target_centre is not None:
            object.__setattr__(self, "target_centre", tuple(float(v) for v in self.target_centre))
        for name in ("target_salience", "dead_repetition_ratio", "productive_repetition_ratio"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))

    @property
    def duration(self) -> float:
        """Wall-clock seconds the run covered."""
        if self.finished_at is None:
            return 0.0
        return max(0.0, float(self.finished_at) - float(self.started_at))

    @property
    def successful_completion(self) -> bool | None:
        """Whether the run ended because the target was centred.

        ``None`` while the run is still going, and on a run that never got far
        enough to be judged either way. It is deliberately not a bool: a run that
        stopped early has no verdict, and defaulting it to ``False`` would read as
        "the agent tried and failed" when the truth may be "the operator stopped
        it".
        """
        if self.status == STATUS_RUNNING:
            return None
        return self.state == "COMPLETE"

    @property
    def summary_line(self) -> str:
        """One plain sentence describing how the run ended."""
        if self.status == STATUS_RUNNING:
            return "Still running."
        return self.stop_reason or f"Stopped with status {self.status}."

    def stream_lines(self, *, limit: int | None = None) -> list[str]:
        """The plain-language event lines, oldest first.

        A stream shows sentences, not identifiers, so each line comes from the
        event's own kind rather than from a second hand-written list that could
        drift away from the events actually emitted.
        """
        lines = [event.message or stream_summary(event.kind) for event in self.events]
        if limit is not None and limit >= 0:
            return lines[-int(limit) :]
        return lines

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view. Contains no pixel data."""

        def number(value: float | None, digits: int) -> float | None:
            return None if value is None else round(float(value), digits)

        return {
            "experiment": self.experiment,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration": round(self.duration, 4),
            "status": self.status,
            "state": self.state,
            "stop_reason": self.stop_reason,
            "directory": self.directory,
            "window": {"width": self.window_width, "height": self.window_height},
            "budget": {"max_moves": self.max_moves},
            "moves_sent": self.moves_sent,
            "scan_moves": self.scan_moves,
            "unique_views": self.unique_views,
            "revisited_views": self.revisited_views,
            "candidate_count": self.candidate_count,
            "target_changes": self.target_changes,
            "centering_moves": self.centering_moves,
            "overshoots": self.overshoots,
            "failed_strategies": self.failed_strategies,
            "stuck_patterns_detected": self.stuck_patterns_detected,
            "stuck_patterns_broken": self.stuck_patterns_broken,
            "final_target_offset": (
                None
                if self.final_target_offset is None
                else [round(float(v), 2) for v in self.final_target_offset]
            ),
            "final_target_distance": number(self.final_target_distance, 2),
            "target": {
                "centre": (
                    None
                    if self.target_centre is None
                    else [round(float(v), 2) for v in self.target_centre]
                ),
                "bbox": None if self.target_bbox is None else list(self.target_bbox),
                "salience": number(self.target_salience, 4),
                "confidence": number(self.confidence, 4),
            },
            "progress": [round(float(value), 2) for value in self.progress],
            "strategy": self.strategy,
            "dead_repetition_ratio": number(self.dead_repetition_ratio, 4),
            "productive_repetition_ratio": number(self.productive_repetition_ratio, 4),
            "mapping": {
                "source": self.mapping_source,
                "quality": number(self.mapping_quality, 4),
                "pixels_per_delta_x": number(self.pixels_per_delta_x, 5),
                "pixels_per_delta_y": number(self.pixels_per_delta_y, 5),
            },
            "successful_completion": self.successful_completion,
            "steps": self.steps,
            "events": [event.to_dict() for event in self.events],
            "stream": self.stream_lines(),
            "notes": list(self.notes),
            "measured_at": self.measured_at,
        }


#: Kept with every record. These are the things a reader of a WAKE result most
#: needs to know before drawing any conclusion from it, and they are the same
#: caveats the milestone itself is careful about.
LIMITATION_NOTES: tuple[str, ...] = (
    "A 'target' here means a visually salient region and nothing more. WAKE-001 "
    "trains no object detector and recognises no object; the milestone forbids it.",
    "The mouse-to-pixel mapping is measured, not assumed. When mapping.source is "
    "'unmeasured' the corrections were sized by the band's fixed count and the run "
    "had no idea how far a count would move the view.",
    "A run that ends with status 'failed' has not necessarily malfunctioned. "
    "Exhausting a movement budget without centring the target is a truthful "
    "outcome and is reported as one.",
    "View similarity is visual only. VIEW_REVISITED means 'this looks very similar "
    "to a view I recently saw' and never 'same place in the world'.",
)


class WakeRecorder:
    """Writes a WAKE-001 result, rewriting it as the run progresses.

    The file is written once at construction and again at every update, so a run
    that is killed mid-way still leaves a readable record of what it had measured
    rather than nothing at all.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        run_id: str,
        plan: Mapping[str, Any] | None = None,
        settings: Mapping[str, Any] | None = None,
        notes: Sequence[str] = LIMITATION_NOTES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._run_id = str(run_id)
        self._plan = dict(plan or {})
        self._settings = dict(settings or {})
        self._notes = tuple(str(note) for note in notes)
        self._started_at = float(clock())
        self._finished_at: float | None = None
        self._status = STATUS_RUNNING
        self._state = ""
        self._stop_reason = ""
        self._metrics: dict[str, Any] = {}
        self._events: tuple[WakeEvent, ...] = ()
        self._steps = 0
        self._window: tuple[int, int] = (0, 0)
        self._closed = False
        self._write()

    # -- read side --------------------------------------------------------

    @property
    def directory(self) -> Path:
        """The run directory this record lives in."""
        return self._directory

    @property
    def result_path(self) -> Path:
        """Path of ``wake_result.json``."""
        return self._directory / RESULT_FILENAME

    @property
    def run_id(self) -> str:
        """The run this record belongs to."""
        return self._run_id

    @property
    def closed(self) -> bool:
        """True once :meth:`finish` has run."""
        return self._closed

    @property
    def window(self) -> tuple[int, int]:
        """The window size recorded so far, ``(0, 0)`` until a frame is seen."""
        return self._window

    def result(self) -> WakeResult:
        """The record as it currently stands."""
        metrics = self._metrics
        offset = metrics.get("final_target_offset")
        return WakeResult(
            experiment=EXPERIMENT_NAME,
            run_id=self._run_id,
            started_at=self._started_at,
            finished_at=self._finished_at,
            status=self._status,
            state=self._state,
            stop_reason=self._stop_reason,
            directory=str(self._directory),
            window_width=self._window[0],
            window_height=self._window[1],
            max_moves=int(metrics.get("max_moves", 0) or 0),
            moves_sent=int(metrics.get("moves_sent", 0) or 0),
            scan_moves=int(metrics.get("scan_moves", 0) or 0),
            unique_views=int(metrics.get("unique_views", 0) or 0),
            revisited_views=int(metrics.get("revisited_views", 0) or 0),
            candidate_count=int(metrics.get("candidate_count", 0) or 0),
            target_changes=int(metrics.get("target_changes", 0) or 0),
            centering_moves=int(metrics.get("centering_moves", 0) or 0),
            overshoots=int(metrics.get("overshoots", 0) or 0),
            failed_strategies=int(metrics.get("failed_strategies", 0) or 0),
            stuck_patterns_detected=int(metrics.get("stuck_patterns_detected", 0) or 0),
            stuck_patterns_broken=int(metrics.get("stuck_patterns_broken", 0) or 0),
            final_target_offset=None if offset is None else (float(offset[0]), float(offset[1])),
            final_target_distance=_optional_float(metrics.get("final_target_distance")),
            target_centre=_optional_pair(metrics.get("target_centre")),
            target_bbox=_optional_bbox(metrics.get("target_bbox")),
            target_salience=_optional_float(metrics.get("target_salience")),
            confidence=_optional_float(metrics.get("confidence")),
            progress=tuple(metrics.get("progress") or ()),
            strategy=str(metrics.get("strategy", "")),
            dead_repetition_ratio=_optional_float(metrics.get("dead_repetition_ratio")),
            productive_repetition_ratio=_optional_float(
                metrics.get("productive_repetition_ratio")
            ),
            mapping_source=str(metrics.get("mapping_source", "")),
            mapping_quality=_optional_float(metrics.get("mapping_quality")),
            pixels_per_delta_x=_optional_float(metrics.get("pixels_per_delta_x")),
            pixels_per_delta_y=_optional_float(metrics.get("pixels_per_delta_y")),
            events=self._events,
            steps=self._steps,
            notes=self._notes,
            measured_at=float(self._clock()),
        )

    # -- write side -------------------------------------------------------

    def note_window(self, width: int, height: int) -> None:
        """Remember the size of the window the run is acting on."""
        self._window = (int(width), int(height))
        self._write()

    def add_note(self, text: str) -> None:
        """Append a note explaining something about this particular run.

        Separate from the standing :data:`LIMITATION_NOTES`, which describe the
        milestone. This is for what happened here: a run the loop had to stop, a
        window that was a different size than expected, and so on.
        """
        if self._closed:
            raise RuntimeError("this WAKE record is already closed")
        text = str(text)
        if text and text not in self._notes:
            self._notes = self._notes + (text,)
            self._write()

    def update(
        self,
        *,
        metrics: Mapping[str, Any] | None = None,
        events: Sequence[WakeEvent] | None = None,
        steps: int | None = None,
        state: str | None = None,
    ) -> None:
        """Replace the measured part of the record and rewrite the file."""
        if self._closed:
            raise RuntimeError("this WAKE record is already closed")
        if metrics is not None:
            self._metrics = dict(metrics)
        if events is not None:
            self._events = tuple(events)
        if steps is not None:
            self._steps = int(steps)
        if state is not None:
            self._state = str(state)
        self._write()

    def finish(self, *, status: str, stop_reason: str, state: str | None = None) -> WakeResult:
        """Close the record with its outcome and write it for the last time."""
        if self._closed:
            raise RuntimeError("this WAKE record is already closed")
        self._status = str(status)
        self._stop_reason = str(stop_reason)
        if state is not None:
            self._state = str(state)
        self._finished_at = float(self._clock())
        self._closed = True
        self._write()
        return self.result()

    def _write(self) -> None:
        """Write the record, never letting a write failure break the run.

        A telemetry failure must not be the thing that stops an agent mid-action;
        the console summary still reports everything either way.
        """
        try:
            payload = self.result().to_dict()
            # ``target`` belongs to the measurement: it is the region the agent
            # actually chose. The window and handle the run was pointed at are the
            # plan, and keeping them under a different key is what stops the plan
            # from overwriting what was observed.
            payload["plan"] = self._plan
            payload["settings"] = self._settings
            self.result_path.write_text(
                json.dumps(payload, indent=2, sort_keys=False) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass


def _optional_float(value: Any) -> float | None:
    """Coerce a metric to a float, keeping ``None`` as "not measured"."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_pair(value: Any) -> tuple[float, float] | None:
    """Coerce a two-element metric to a float pair, or ``None``."""
    if value is None:
        return None
    try:
        pair = tuple(value)
    except TypeError:
        return None
    if len(pair) != 2:
        return None
    left, right = _optional_float(pair[0]), _optional_float(pair[1])
    if left is None or right is None:
        return None
    return (left, right)


def _optional_bbox(value: Any) -> tuple[int, int, int, int] | None:
    """Coerce a four-element metric to an integer bounding box, or ``None``."""
    if value is None:
        return None
    try:
        box = tuple(value)
    except TypeError:
        return None
    if len(box) != 4:
        return None
    try:
        return (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    except (TypeError, ValueError):
        return None
