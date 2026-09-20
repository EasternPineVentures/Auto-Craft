"""Experiment telemetry.

Every run gets its own directory under ``data/runs/<run-id>/``:

* ``steps.ndjson`` - one JSON object per loop step, appended as the run goes, so
  a crashed or Ctrl-C'd run still leaves a readable trace.
* ``run.json`` - the finished run summary, written once at the end.
* ``frames/`` - only created when the caller asked the loop to persist frames.

Raw pixels are never serialised into JSON. Frames are referenced by path, and a
coarse signature and diff scalar stand in for their content. A run that captures
thousands of frames should produce a small, greppable JSON file, not a gigabyte
of base64.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4

__all__ = ["RunRecorder", "RunRecord", "new_run_id"]


def new_run_id(now: float | None = None) -> str:
    """Return a sortable, collision-resistant run identifier.

    Format: ``20260214T101530Z-9f3a1c2d`` - UTC timestamp plus a short random
    suffix, so runs sort chronologically and two runs in the same second do not
    collide.
    """
    stamp = datetime.fromtimestamp(now if now is not None else time.time(), tz=timezone.utc)
    return f"{stamp.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"


@dataclass(frozen=True)
class RunRecord:
    """The complete, serialisable record of one agent run."""

    run_id: str
    started_at: float
    finished_at: float | None = None
    status: str = "running"
    stop_reason: str = ""
    config: Mapping[str, Any] = field(default_factory=dict)
    step_count: int = 0
    steps: tuple[Mapping[str, Any], ...] = ()
    safety_events: tuple[Mapping[str, Any], ...] = ()
    errors: tuple[Mapping[str, Any], ...] = ()
    directory: str | None = None

    def __post_init__(self) -> None:
        # A record built by hand (or restored from JSON without a step_count) still
        # has to report how many steps it holds.
        if not self.step_count and self.steps:
            object.__setattr__(self, "step_count", len(self.steps))

    @property
    def duration(self) -> float:
        """Wall-clock seconds the run covered."""
        if self.finished_at is None:
            return 0.0
        return max(0.0, self.finished_at - self.started_at)

    @property
    def executed_steps(self) -> int:
        """Number of steps whose action actually ran."""
        return sum(1 for step in self.steps if (step.get("result") or {}).get("executed"))

    @property
    def blocked_steps(self) -> int:
        """Number of steps the safety guard refused."""
        return sum(1 for step in self.steps if (step.get("result") or {}).get("blocked_reason"))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration": self.duration,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "config": dict(self.config),
            "step_count": self.step_count,
            "executed_steps": self.executed_steps,
            "blocked_steps": self.blocked_steps,
            "steps": [dict(step) for step in self.steps],
            "safety_events": [dict(event) for event in self.safety_events],
            "errors": [dict(error) for error in self.errors],
            "directory": self.directory,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunRecord":
        """Rebuild a record from its serialised form."""
        return cls(
            run_id=str(payload["run_id"]),
            started_at=float(payload["started_at"]),
            finished_at=payload.get("finished_at"),
            status=str(payload.get("status", "unknown")),
            stop_reason=str(payload.get("stop_reason", "")),
            config=dict(payload.get("config", {})),
            step_count=int(payload.get("step_count", 0)),
            steps=tuple(payload.get("steps", ())),
            safety_events=tuple(payload.get("safety_events", ())),
            errors=tuple(payload.get("errors", ())),
            directory=payload.get("directory"),
        )


class RunRecorder:
    """Writes a run's telemetry to disk, streaming steps as they happen."""

    def __init__(
        self,
        directory: str | Path,
        *,
        run_id: str | None = None,
        config: Mapping[str, Any] | None = None,
        clock: Callable[[], float] = time.time,
        frames_dir: str | Path | None = None,
    ) -> None:
        self._clock = clock
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id or new_run_id(self._clock())
        self._config = dict(config or {})
        self._frames_dir = Path(frames_dir) if frames_dir is not None else self._directory / "frames"
        self._steps: list[dict[str, Any]] = []
        self._safety_events: list[dict[str, Any]] = []
        self._errors: list[dict[str, Any]] = []
        self._started_at = self._clock()
        self._closed = False
        self._record: RunRecord | None = None
        self._steps_path = self._directory / "steps.ndjson"
        self._record_path = self._directory / "run.json"
        self._steps_path.write_text("", encoding="utf-8")

    @classmethod
    def new_run(
        cls,
        base_dir: str | Path,
        *,
        config: Mapping[str, Any] | None = None,
        clock: Callable[[], float] = time.time,
        run_id: str | None = None,
    ) -> "RunRecorder":
        """Create a recorder writing into its own ``base_dir/<run-id>/`` folder.

        This is the constructor callers should normally use: giving every run its
        own directory is what keeps a previous run's telemetry from being
        overwritten by the next one.
        """
        identifier = run_id or new_run_id(clock())
        return cls(Path(base_dir) / identifier, run_id=identifier, config=config, clock=clock)

    # -- properties -------------------------------------------------------

    @property
    def run_id(self) -> str:
        """Identifier for this run."""
        return self._run_id

    @property
    def directory(self) -> Path:
        """Directory holding this run's artifacts."""
        return self._directory

    @property
    def frames_dir(self) -> Path:
        """Directory that will hold saved frames, created on first use."""
        return self._frames_dir

    @property
    def steps_path(self) -> Path:
        """Path of the streaming step log."""
        return self._steps_path

    @property
    def record_path(self) -> Path:
        """Path of the final run summary."""
        return self._record_path

    @property
    def steps(self) -> tuple[Mapping[str, Any], ...]:
        """Steps recorded so far."""
        return tuple(self._steps)

    # -- recording --------------------------------------------------------

    def frame_path(self, name: str) -> Path:
        """Return a path inside this run's frames directory, creating it."""
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        return self._frames_dir / name

    def _ensure_open(self) -> None:
        """Refuse writes that would land after ``run.json`` was already written.

        Steps appended after the summary would sit in ``steps.ndjson`` but not in
        ``run.json``, so the two artifacts would disagree about what happened.
        Failing loudly keeps a finished run immutable and honest.
        """
        if self._closed:
            raise RuntimeError(
                f"run {self._run_id} is already closed; telemetry can no longer be recorded"
            )

    def record_step(self, step: Mapping[str, Any]) -> Mapping[str, Any]:
        """Append one step to memory and to ``steps.ndjson``."""
        self._ensure_open()
        payload = dict(step)
        self._steps.append(payload)
        with self._steps_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        return payload

    def record_safety_event(self, event: Any) -> Mapping[str, Any]:
        """Record a safety event, accepting a ``SafetyEvent`` or a plain mapping."""
        self._ensure_open()
        payload = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        self._safety_events.append(payload)
        return payload

    def record_safety_events(self, events: Iterable[Any]) -> None:
        """Record several safety events in order."""
        for event in events:
            self.record_safety_event(event)

    def record_error(self, message: str, *, context: str = "") -> Mapping[str, Any]:
        """Record an error without aborting the run."""
        self._ensure_open()
        payload = {"at": self._clock(), "message": message, "context": context}
        self._errors.append(payload)
        return payload

    def finish(
        self,
        *,
        status: str,
        stop_reason: str,
        finished_at: float | None = None,
    ) -> RunRecord:
        """Write ``run.json`` and return the completed record.

        Idempotent: once the summary has been written, later calls return the
        existing record rather than overwriting it with a different status. The
        loop calls this from a ``finally`` block, so a second call must never
        rewrite the real outcome.
        """
        if self._closed and self._record is not None:
            return self._record
        record = RunRecord(
            run_id=self._run_id,
            started_at=self._started_at,
            finished_at=finished_at if finished_at is not None else self._clock(),
            status=status,
            stop_reason=stop_reason,
            config=self._config,
            step_count=len(self._steps),
            steps=tuple(self._steps),
            safety_events=tuple(self._safety_events),
            errors=tuple(self._errors),
            directory=str(self._directory),
        )
        self._record_path.write_text(
            json.dumps(record.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._closed = True
        self._record = record
        return record

    def close(self) -> None:
        """Mark the recorder closed without writing a summary."""
        self._closed = True

    @property
    def closed(self) -> bool:
        """True once the run summary has been written."""
        return self._closed

    def __enter__(self) -> "RunRecorder":
        return self

    def __exit__(self, exc_type: object, exc: object, _tb: object) -> None:
        # Leaving the block must always leave a usable run.json behind, and must
        # say so honestly when the block ended by raising.
        if self._record is None:
            if exc is None:
                self.finish(status="finished", stop_reason="context manager exit")
            else:
                self.finish(
                    status="error",
                    stop_reason=f"{type(exc).__name__}: {exc}",
                )
        else:
            self.close()
