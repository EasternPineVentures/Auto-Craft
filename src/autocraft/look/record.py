"""The LOOK-001 experiment record: what was asked, what was measured, what was saved.

Two shapes matter here:

* :class:`LookTrialResult` is the in-memory outcome of one trial. It holds the
  three frames, because the operator asked for them on disk.
* :class:`LookResult` is the serialised run. It holds metadata, per-frame
  summaries and measurements - and no pixels, ever. ``look_result.json`` is
  written from this type, and :meth:`LookTrialResult.to_dict` reaches pixels only
  through :meth:`~autocraft.vision.frame.Frame.to_dict`, which is documented
  never to include them.

The write path is streaming on purpose: ``look_result.json`` is rewritten after
every trial rather than once at the end, so a run that is interrupted - by the
operator, by focus loss, or by the emergency stop - still leaves a readable
record of the trials that did complete.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..vision.frame import Frame
from .metrics import (
    MAPPING_NOTE,
    FrameDifference,
    ShiftEstimate,
    difference_image,
    save_grayscale,
)

__all__ = [
    "ARTIFACT_DIFFERENCE_AB",
    "ARTIFACT_DIFFERENCE_AC",
    "ARTIFACT_FRAME_A",
    "ARTIFACT_FRAME_B",
    "ARTIFACT_FRAME_C",
    "EXPERIMENT_NAME",
    "LIMITATION_NOTES",
    "LookRecorder",
    "LookResult",
    "LookTrialResult",
    "TRIAL_COMPLETED",
    "TRIAL_FAILED",
    "TRIAL_INTERRUPTED",
    "TrialSpec",
]

#: The specification's name for this milestone. It appears in the record so a
#: file found months later says which experiment produced it.
EXPERIMENT_NAME = "LOOK-001"

#: The record's own filename, inside the look directory.
RESULT_FILENAME = "look_result.json"

TRIAL_COMPLETED = "completed"
TRIAL_INTERRUPTED = "interrupted"
TRIAL_FAILED = "failed"

ARTIFACT_FRAME_A = "frame_a.png"
ARTIFACT_FRAME_B = "frame_b.png"
ARTIFACT_FRAME_C = "frame_c.png"
ARTIFACT_DIFFERENCE_AB = "difference_ab.png"
ARTIFACT_DIFFERENCE_AC = "difference_ac.png"

#: What this experiment is and is not, written into every record.
#:
#: These live beside the record rather than at the call site because they are
#: true of the experiment itself: a record found on disk months later must
#: carry its own limits, whoever ran it and whatever they were hoping to prove.
LIMITATION_NOTES = (
    "Measured sensorimotor mapping for one bounded movement. "
    "Not evidence that anything understands the camera.",
    "No pass/fail threshold is applied. The operator reads the numbers.",
    "The pixel displacement is measured by phase correlation, not assumed "
    "from the mouse delta; the ratio between them is reported, never required "
    "to be a particular value.",
)


@dataclass(frozen=True)
class TrialSpec:
    """One bounded movement-and-reverse trial, as planned before it ran.

    Recorded separately from the outcome so the record can distinguish "we meant
    to send this" from "we sent this".
    """

    index: int
    dx: int
    dy: int
    settle_seconds: float

    @property
    def reverse_dx(self) -> int:
        """The outbound delta negated on x."""
        return -self.dx

    @property
    def reverse_dy(self) -> int:
        """The outbound delta negated on y."""
        return -self.dy

    def describe(self) -> str:
        """One-line description of the movement pair."""
        return f"({self.dx:+d}, {self.dy:+d}) then ({self.reverse_dx:+d}, {self.reverse_dy:+d})"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "index": self.index,
            "dx": self.dx,
            "dy": self.dy,
            "settle_seconds": round(self.settle_seconds, 4),
        }


@dataclass(frozen=True)
class LookTrialResult:
    """The measured outcome of one trial.

    ``frame_b`` is ``None`` when the trial stopped before the second capture, and
    ``frame_c`` is ``None`` when it stopped before the third. A missing frame is
    the honest representation of "we never got there" and is preferred over a
    placeholder that would read as a measurement.
    """

    spec: TrialSpec
    status: str
    stop_reason: str
    window_width: int
    window_height: int
    capture_seconds: float
    movements_sent: int
    measured_at: float
    frame_a: Frame | None = None
    frame_b: Frame | None = None
    frame_c: Frame | None = None
    a_to_b: FrameDifference | None = None
    a_to_c: FrameDifference | None = None
    shift: ShiftEstimate | None = None
    pixels_per_delta_x: float | None = None
    pixels_per_delta_y: float | None = None
    reversibility_ratio: float | None = None
    reversibility_note: str = ""
    artifacts: Mapping[str, str] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """True when every capture and measurement this trial needed happened."""
        return self.status == TRIAL_COMPLETED

    def frames(self) -> tuple[tuple[str, Frame], ...]:
        """The frames that were captured, in capture order, with their roles."""
        pairs: list[tuple[str, Frame]] = []
        if self.frame_a is not None:
            pairs.append(("a", self.frame_a))
        if self.frame_b is not None:
            pairs.append(("b", self.frame_b))
        if self.frame_c is not None:
            pairs.append(("c", self.frame_c))
        return tuple(pairs)

    def report_fields(self) -> dict[str, Any]:
        """The measured primitives the observer's LOOK panel displays.

        Returns plain values only, and only ones that were actually measured. The
        observer adds the run-level fields (status, trial count, timestamp) so
        this module never has to know how the page is shaped.
        """
        return {
            "trial_index": self.spec.index,
            "dx": self.spec.dx,
            "dy": self.spec.dy,
            "settle_seconds": self.spec.settle_seconds,
            "window_width": int(self.window_width),
            "window_height": int(self.window_height),
            "movements_sent": int(self.movements_sent),
            "mean_absolute_difference": (
                None if self.a_to_b is None else self.a_to_b.mean_absolute_difference
            ),
            "rmse": None if self.a_to_b is None else self.a_to_b.rmse,
            "changed_fraction": None if self.a_to_b is None else self.a_to_b.changed_fraction,
            "block_grid": 0 if self.a_to_b is None else self.a_to_b.block_grid,
            "block_map": () if self.a_to_b is None else self.a_to_b.block_map,
            "shift_x": None if self.shift is None or not self.shift.available else self.shift.x,
            "shift_y": None if self.shift is None or not self.shift.available else self.shift.y,
            "shift_quality": (
                None if self.shift is None or not self.shift.available else self.shift.quality
            ),
            "shift_available": bool(self.shift is not None and self.shift.available),
            "pixels_per_delta_x": self.pixels_per_delta_x,
            "pixels_per_delta_y": self.pixels_per_delta_y,
            "reversibility_ratio": self.reversibility_ratio,
            "reversibility_note": self.reversibility_note,
            "capture_seconds": float(self.capture_seconds),
            "stop_reason": self.stop_reason,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view. Contains no pixel data."""
        return {
            "index": self.spec.index,
            "dx": self.spec.dx,
            "dy": self.spec.dy,
            "settle_seconds": round(self.spec.settle_seconds, 4),
            "status": self.status,
            "stop_reason": self.stop_reason,
            "measured_at": round(self.measured_at, 3),
            "movements_sent": int(self.movements_sent),
            "capture_seconds": round(float(self.capture_seconds), 4),
            "window": {"width": int(self.window_width), "height": int(self.window_height)},
            "frames": dict(self.artifacts),
            "frame_a": None if self.frame_a is None else self.frame_a.to_dict(),
            "frame_b": None if self.frame_b is None else self.frame_b.to_dict(),
            "frame_c": None if self.frame_c is None else self.frame_c.to_dict(),
            "a_to_b": None if self.a_to_b is None else self.a_to_b.to_dict(),
            "a_to_c": None if self.a_to_c is None else self.a_to_c.to_dict(),
            "shift": None if self.shift is None else self.shift.to_dict(),
            "pixels_per_delta": {
                "x": None if self.pixels_per_delta_x is None else round(self.pixels_per_delta_x, 5),
                "y": None if self.pixels_per_delta_y is None else round(self.pixels_per_delta_y, 5),
                "note": MAPPING_NOTE,
            },
            "reversibility_ratio": (
                None if self.reversibility_ratio is None else round(self.reversibility_ratio, 5)
            ),
            "reversibility_note": self.reversibility_note,
        }


@dataclass(frozen=True)
class LookResult:
    """The whole experiment, as written to ``look_result.json``."""

    experiment: str
    run_id: str
    started_at: float
    finished_at: float | None
    status: str
    stop_reason: str
    directory: str
    target: Mapping[str, Any]
    settings: Mapping[str, Any]
    plan: tuple[TrialSpec, ...]
    trials: tuple[LookTrialResult, ...]
    movements_sent: int
    notes: tuple[str, ...] = ()

    @property
    def duration(self) -> float | None:
        """Wall-clock length of the experiment, or ``None`` while it is open."""
        if self.finished_at is None:
            return None
        return max(0.0, self.finished_at - self.started_at)

    @property
    def completed_trials(self) -> int:
        """How many trials ran to completion."""
        return sum(1 for trial in self.trials if trial.complete)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view. Contains no pixel data."""
        return {
            "experiment": self.experiment,
            "run_id": self.run_id,
            "started_at": round(self.started_at, 3),
            "finished_at": None if self.finished_at is None else round(self.finished_at, 3),
            "duration_seconds": None if self.duration is None else round(self.duration, 3),
            "status": self.status,
            "stop_reason": self.stop_reason,
            "directory": self.directory,
            "target": dict(self.target),
            "settings": dict(self.settings),
            "plan": [spec.to_dict() for spec in self.plan],
            "movements_sent": int(self.movements_sent),
            "trials_completed": self.completed_trials,
            "trials": [trial.to_dict() for trial in self.trials],
            "notes": list(self.notes),
        }


class LookRecorder:
    """Writes one LOOK-001 experiment into a ``look`` directory.

    Layout, with the specification's filenames:

    * a single-trial run writes ``frame_a.png``, ``frame_b.png``, ``frame_c.png``,
      ``difference_ab.png``, ``difference_ac.png`` and ``look_result.json``
      directly into the look directory;
    * a multi-trial run - a calibration series, or ``--steps`` above one - writes
      the same five images per trial inside ``trial-NN/`` so no trial overwrites
      another.

    Both cases keep ``look_result.json`` at the top, because that is the file an
    operator or a later milestone reads.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        run_id: str,
        plan: Sequence[TrialSpec],
        target: Mapping[str, Any] | None = None,
        settings: Mapping[str, Any] | None = None,
        notes: Sequence[str] = LIMITATION_NOTES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._run_id = str(run_id)
        self._plan = tuple(plan)
        self._target = dict(target or {})
        self._settings = dict(settings or {})
        self._notes = tuple(str(note) for note in notes)
        self._trials: list[LookTrialResult] = []
        self._started_at = float(clock())
        self._finished_at: float | None = None
        self._status = "running"
        self._stop_reason = ""
        self._closed = False
        self._write_result()

    # -- read side --------------------------------------------------------

    @property
    def directory(self) -> Path:
        """The look directory."""
        return self._directory

    @property
    def result_path(self) -> Path:
        """Path of ``look_result.json``."""
        return self._directory / RESULT_FILENAME

    @property
    def run_id(self) -> str:
        """The run this experiment belongs to."""
        return self._run_id

    @property
    def plan(self) -> tuple[TrialSpec, ...]:
        """The planned trials, in order."""
        return self._plan

    @property
    def trials(self) -> tuple[LookTrialResult, ...]:
        """Trials recorded so far."""
        return tuple(self._trials)

    @property
    def closed(self) -> bool:
        """True once :meth:`finish` has run."""
        return self._closed

    def trial_directory(self, index: int) -> Path:
        """Where one trial's images belong.

        Single-trial runs write into the look directory itself, which is what
        makes the documented manual trial produce exactly the paths the
        specification lists.
        """
        if len(self._plan) <= 1:
            return self._directory
        return self._directory / f"trial-{int(index):02d}"

    def result(self) -> LookResult:
        """The record as it currently stands."""
        return LookResult(
            experiment=EXPERIMENT_NAME,
            run_id=self._run_id,
            started_at=self._started_at,
            finished_at=self._finished_at,
            status=self._status,
            stop_reason=self._stop_reason,
            directory=str(self._directory),
            target=self._target,
            settings=self._settings,
            plan=self._plan,
            trials=tuple(self._trials),
            movements_sent=sum(trial.movements_sent for trial in self._trials),
            notes=self._notes,
        )

    # -- write side -------------------------------------------------------

    def record(self, result: LookTrialResult) -> LookTrialResult:
        """Save one trial's images, add it to the record and rewrite the JSON.

        Returns the trial with its :attr:`LookTrialResult.artifacts` filled in, so
        the caller can report the exact paths that were written. Only frames that
        exist are written: an interrupted trial leaves a shorter file list rather
        than a misleading placeholder.

        The difference images are written only for the pairs whose difference was
        actually *measured*. A trial whose measurement failed keeps its raw frames
        - those are exactly what the operator needs to see what happened - but no
        difference image is invented for a pair the runner could not compare, and
        a failed measurement cannot make the record itself fail.
        """
        if self._closed:
            raise RuntimeError("this LOOK record is already closed")
        directory = self.trial_directory(result.spec.index)
        directory.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, str] = {}

        def save(name: str, path: Path) -> None:
            artifacts[name] = _relative(path, self._directory)

        if result.frame_a is not None:
            save(ARTIFACT_FRAME_A, result.frame_a.save(directory / ARTIFACT_FRAME_A))
        if result.frame_b is not None:
            save(ARTIFACT_FRAME_B, result.frame_b.save(directory / ARTIFACT_FRAME_B))
        if result.frame_c is not None:
            save(ARTIFACT_FRAME_C, result.frame_c.save(directory / ARTIFACT_FRAME_C))
        if result.a_to_b is not None and result.frame_a is not None and result.frame_b is not None:
            image = difference_image(result.frame_a, result.frame_b)
            save(
                ARTIFACT_DIFFERENCE_AB,
                save_grayscale(image, directory / ARTIFACT_DIFFERENCE_AB),
            )
        if result.a_to_c is not None and result.frame_a is not None and result.frame_c is not None:
            image = difference_image(result.frame_a, result.frame_c)
            save(
                ARTIFACT_DIFFERENCE_AC,
                save_grayscale(image, directory / ARTIFACT_DIFFERENCE_AC),
            )

        stored = replace(result, artifacts=artifacts)
        self._trials.append(stored)
        self._write_result()
        return stored

    def finish(
        self,
        *,
        status: str,
        stop_reason: str,
        finished_at: float | None = None,
    ) -> LookResult:
        """Close the record. Idempotent, so a ``finally`` block is safe."""
        if self._closed:
            return self.result()
        self._status = str(status)
        self._stop_reason = str(stop_reason)
        self._finished_at = float(self._clock() if finished_at is None else finished_at)
        self._closed = True
        self._write_result()
        return self.result()

    def _write_result(self) -> Path:
        payload = self.result().to_dict()
        text = json.dumps(payload, indent=2, sort_keys=False)
        self.result_path.write_text(text + "\n", encoding="utf-8")
        return self.result_path


def _relative(path: Path, root: Path) -> str:
    """Express ``path`` relative to ``root``, with forward slashes.

    Forward slashes keep the record readable on the platform that did not write
    it. When the path somehow lies outside the root the absolute path is used
    rather than a wrong relative one.
    """
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:  # pragma: no cover - only if a caller writes outside
        return path.as_posix()
