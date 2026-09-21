"""Driving the scene model against a live or recorded frame source.

:class:`StabilityModel` knows how to learn from and score frames. It knows nothing
about *where* frames come from, which is deliberate: the model is pure arithmetic
over arrays and can be tested with synthetic ones. This module is the thin layer
that supplies real frames and accumulates what came back.

The session takes a plain callable that returns a frame, so the same code drives a
live screen, a saved PNG directory, or a test fixture without any branching. It
never sends input of any kind. There is no actuator here to send it with.

Two phases are reported separately and never mixed:

* **warm-up** - frames the model is learning from. No scores exist yet, and
  reporting any would be inventing numbers.
* **steady** - frames scored against the fitted model.

If the run ends during warm-up the report says so and carries no scores, rather
than presenting a half-fitted model's output as if it meant something.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from ..vision.frame import Frame
from .features import cell_features, describe
from .stability import SceneScore, StabilityModel

__all__ = ["FrameSource", "PerceptionReport", "PerceptionSession", "SessionRow"]

#: A source of frames. Returns ``None`` to signal that this frame failed.
FrameSource = Callable[[], "Frame | None"]


@dataclass
class SessionRow:
    """One frame's entry in the run timeline."""

    index: int
    phase: str
    """``"warmup"`` or ``"steady"``."""

    seconds: float
    """Seconds since the session started."""

    score: SceneScore | None
    """``None`` during warm-up, where no score can honestly exist."""

    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
            "phase": self.phase,
            "seconds": round(self.seconds, 4),
            "features": self.features,
        }
        if self.score is not None:
            payload["score"] = self.score.to_dict()
        return payload


@dataclass
class PerceptionReport:
    """Everything a ``perceive-test`` run measured, in one serialisable object."""

    grid: int
    fit_frames: int
    rows: list[SessionRow] = field(default_factory=list)
    capture_failures: int = 0
    duration_seconds: float = 0.0
    stop_reason: str = ""
    model: dict[str, Any] = field(default_factory=dict)

    @property
    def warmup_frames(self) -> int:
        return sum(1 for row in self.rows if row.phase == "warmup")

    @property
    def steady_frames(self) -> int:
        return sum(1 for row in self.rows if row.phase == "steady")

    @property
    def scores(self) -> list[SceneScore]:
        return [row.score for row in self.rows if row.score is not None]

    @property
    def is_complete(self) -> bool:
        """Whether the run got past warm-up and has anything to report."""
        return self.steady_frames > 0

    def changed_counts(self) -> list[int]:
        return [score.changed_cells for score in self.scores]

    def total_excesses(self) -> list[float]:
        return [score.total_excess for score in self.scores]

    def accumulated_excess(self) -> list[list[float]]:
        """Sum every steady frame's overshoot map into one grid.

        This is the map that answers "where in the picture did anything happen
        during this run", which is usually the single most useful output.
        """
        if not self.scores:
            return [[0.0] * self.grid for _ in range(self.grid)]
        total = self.scores[0].excess.copy()
        for score in self.scores[1:]:
            total = total + score.excess
        return [[float(value) for value in row] for row in total]

    def summary(self) -> dict[str, Any]:
        """Aggregate statistics over the steady frames."""
        counts = self.changed_counts()
        excesses = self.total_excesses()
        cell_count = self.grid * self.grid
        payload: dict[str, Any] = {
            "grid": self.grid,
            "cell_count": cell_count,
            "fit_frames": self.fit_frames,
            "warmup_frames": self.warmup_frames,
            "steady_frames": self.steady_frames,
            "capture_failures": self.capture_failures,
            "duration_seconds": round(self.duration_seconds, 4),
            "stop_reason": self.stop_reason,
            "complete": self.is_complete,
        }
        if counts:
            payload["changed_cells_mean"] = round(sum(counts) / len(counts), 4)
            payload["changed_cells_min"] = min(counts)
            payload["changed_cells_max"] = max(counts)
            payload["changed_fraction_mean"] = round(
                sum(counts) / len(counts) / cell_count, 6
            )
            payload["total_excess_mean"] = round(sum(excesses) / len(excesses), 6)
            payload["total_excess_max"] = round(max(excesses), 6)
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "model": self.model,
            "timeline": [row.to_dict() for row in self.rows],
            "accumulated_excess": self.accumulated_excess(),
        }


class PerceptionSession:
    """Runs a frame source through a scene model and records what happened.

    Args:
        model: The scene model to fit and score with. A fresh one is expected; a
            model that is already fitted will go straight to steady state.
        capture: Callable returning a :class:`~autocraft.vision.frame.Frame`, or
            ``None`` if that frame could not be captured.
        clock: Monotonic clock, injectable so tests need not sleep.
        adapt: Whether to call :meth:`StabilityModel.update` on steady frames. Off
            leaves the model frozen at the end of warm-up, which is what a caller
            wants when the question is "is this frame different from the scene I
            learned", not "has the scene moved on".
    """

    def __init__(
        self,
        model: StabilityModel,
        capture: FrameSource,
        *,
        clock: Callable[[], float] = time.monotonic,
        adapt: bool = True,
    ) -> None:
        self.model = model
        self.capture = capture
        self.clock = clock
        self.adapt = bool(adapt)

    def run(
        self,
        *,
        seconds: float | None = None,
        steps: int | None = None,
        max_seconds: float | None = None,
    ) -> PerceptionReport:
        """Observe until a bound is reached, then return the report.

        At least one of ``seconds`` or ``steps`` is required. There is no
        unbounded mode: an experiment that cannot say when it stops cannot be
        reviewed afterwards, and every other bounded loop in the project follows
        the same rule.

        ``max_seconds`` is a separate, absolute ceiling that also covers time spent
        capturing. ``seconds`` measures the same clock but exists as the
        user-facing duration, so a run can be given both without them being
        redundant: the tighter one wins.
        """
        if seconds is None and steps is None:
            raise ValueError("at least one of seconds or steps is required")
        if seconds is not None and seconds <= 0:
            raise ValueError(f"seconds must be positive, got {seconds}")
        if steps is not None and steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")

        started = self.clock()
        report = PerceptionReport(
            grid=self.model.grid,
            fit_frames=self.model.fit_frames,
            model=self.model.summary(),
        )
        stop_reason = "step limit reached" if steps is not None else "time limit reached"
        index = 0

        while True:
            if steps is not None and index >= steps:
                stop_reason = f"reached the {steps}-step limit"
                break

            elapsed = self.clock() - started
            if seconds is not None and elapsed >= seconds:
                stop_reason = f"reached the {seconds:g}s limit"
                break
            if max_seconds is not None and elapsed >= max_seconds:
                stop_reason = f"reached the {max_seconds:g}s ceiling"
                break

            frame = self.capture()
            if frame is None:
                report.capture_failures += 1
                index += 1
                continue

            self._record(report, index, frame, self.clock() - started)
            index += 1

        report.duration_seconds = self.clock() - started
        report.stop_reason = stop_reason
        report.model = self.model.summary()
        return report

    def frames(self, frames: Iterator[Frame]) -> PerceptionReport:
        """Fit and score a fixed sequence of frames, ignoring the clock.

        The recorded-data path: replay an existing capture instead of watching the
        screen. Used by tests and by anyone re-analysing a saved run.
        """
        started = self.clock()
        report = PerceptionReport(
            grid=self.model.grid,
            fit_frames=self.model.fit_frames,
            model=self.model.summary(),
        )
        index = 0
        for frame in frames:
            self._record(report, index, frame, self.clock() - started)
            index += 1

        report.duration_seconds = self.clock() - started
        report.stop_reason = f"replayed {index} recorded frame(s)"
        report.model = self.model.summary()
        return report

    def _record(
        self, report: PerceptionReport, index: int, frame: Frame, elapsed: float
    ) -> None:
        """Add one frame to the report, fitting or scoring as appropriate."""
        features = cell_features(frame.image, self.model.grid)

        if not self.model.is_fitted:
            self.model.observe(features)
            report.rows.append(
                SessionRow(
                    index=index,
                    phase="warmup",
                    seconds=elapsed,
                    score=None,
                    features=describe(features),
                )
            )
            if self.model.is_fitted:
                # Warm-up completed on this frame, so the summary captured when the
                # report was created is now stale.
                report.model = self.model.summary()
            return

        score = self.model.score(features, index=index)
        if self.adapt:
            self.model.update(features, score=score)
        report.rows.append(
            SessionRow(
                index=index,
                phase="steady",
                seconds=elapsed,
                score=score,
                features=describe(features),
            )
        )
