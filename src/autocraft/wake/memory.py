"""Short-term experience: what the agent has just looked at and just tried.

The milestone asks for bounded working memory and warns, in the same breath, not
to build a vector database and not to build long-term memory. So this module is
deliberately small and deliberately forgetful. Everything here is a fixed-size
deque or a capped list; when it is full, the oldest thing falls out and the agent
genuinely no longer remembers it.

Three kinds of memory live here, and they are separate because they answer
separate questions.

**View memory** answers *"have I looked this way already?"* It works on a coarse
luminance grid, correlated against the grids already stored. A match produces the
event ``VIEW_REVISITED``, and that event means one thing only: *this looks very
similar to a view I recently saw*. It does **not** mean "I am in the same place
in the world", and nothing downstream is allowed to read it that way. Two
different spots under a featureless sky can correlate; the agent has no way to
tell and does not pretend to.

**Candidate memory** answers *"have I already considered this thing?"* It matches
candidates by their appearance descriptor rather than by position, because
position is exactly what changes when the camera moves. A region that has already
been selected keeps a ``seen`` count, which is what the selection score uses to
avoid picking the same target over and over.

**Action memory** answers *"what did I try, and did it help?"* Each entry records
the strategy, the movement, the view before and after, and - where the behaviour
layer had a measurable quantity - how that quantity changed. It is the raw
material for the anti-repetition guard in :mod:`autocraft.wake.progress`.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

import numpy as np

from ..vision.frame import Frame
from .salience import CandidateTarget, descriptor_similarity

__all__ = [
    "DEFAULT_SIMILAR_THRESHOLD",
    "DEFAULT_VIEW_GRID",
    "ActionRecord",
    "FailedAttempt",
    "ProgressSample",
    "ShortTermExperience",
    "ViewFingerprint",
    "ViewMemory",
    "ViewObservation",
    "ViewRecord",
]

#: Cells per axis of a view fingerprint. Eight is coarse on purpose: a fine grid
#: would make two looks at the same scene score as different purely because the
#: camera settled a few pixels away.
DEFAULT_VIEW_GRID = 8

#: Similarity at or above which two views count as "the same view". On the
#: correlation scale used here, 0.92 means a Pearson correlation of about 0.84
#: between the two coarse grids - plainly the same structure, not a coincidence.
DEFAULT_SIMILAR_THRESHOLD = 0.92

#: Similarity at or above which two candidates count as the same region. Uses the
#: descriptor scale, where 0.7 corresponds to a descriptor distance below 0.84
#: standard deviations across seven appearance features.
DEFAULT_MERGE_THRESHOLD = 0.7


@dataclass(frozen=True)
class ViewFingerprint:
    """A coarse, comparable summary of one frame.

    The values are raw block luminances in 0..255, kept rather than normalised so
    that the fingerprint can also report the frame's overall brightness. Structure
    is compared by correlation, which is what makes the comparison robust to the
    global brightness shift a change of exposure or sky produces.
    """

    grid: int
    values: tuple[float, ...]
    mean_luma: float
    key: str

    @classmethod
    def of(cls, frame: Frame | np.ndarray, *, grid: int = DEFAULT_VIEW_GRID) -> "ViewFingerprint":
        """Build a fingerprint from a frame or an ``H x W x 3`` uint8 array."""
        image = frame.image if isinstance(frame, Frame) else np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected an H x W x 3 image, got shape {getattr(image, 'shape', None)}")
        grid = max(1, int(grid))
        height, width = int(image.shape[0]), int(image.shape[1])
        # Integer band bounds, so the whole frame is covered and no remainder is
        # silently dropped the way a plain reshape would drop it.
        rows = _bounds(height, grid)
        columns = _bounds(width, grid)
        values: list[float] = []
        for row in rows:
            for column in columns:
                block = image[row[0] : row[1], column[0] : column[1]]
                values.append(float(block.mean()))
        return cls(
            grid=grid,
            values=tuple(values),
            mean_luma=float(np.mean(values)) if values else 0.0,
            key=_digest(values),
        )

    @property
    def size(self) -> int:
        """Number of cells in the fingerprint."""
        return len(self.values)

    def similarity(self, other: "ViewFingerprint") -> float:
        """How alike two views are, in 0..1, where 1.0 is the same structure.

        The measure is the Pearson correlation of the two coarse grids, remapped
        from -1..1 into 0..1. Correlation ignores a global brightness shift, which
        is what makes it the right measure here: turning the camera changes the
        sky gradient and the exposure, and those are not what "a different view"
        means.

        When either view is essentially featureless - a frame of one flat colour -
        correlation is undefined, and this falls back to comparing overall
        brightness. That is an honest degradation rather than a fabricated
        structural claim, and it is why a uniformly dark view does not match
        every other uniformly dark view with a perfect score by accident: two
        featureless views at different brightness still score below 1.0.
        """
        if self.size != other.size or self.size == 0:
            return 0.0
        left = np.asarray(self.values, dtype=np.float64)
        right = np.asarray(other.values, dtype=np.float64)
        left = left - left.mean()
        right = right - right.mean()
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm < 1e-9 or right_norm < 1e-9:
            return max(0.0, 1.0 - abs(self.mean_luma - other.mean_luma) / 255.0)
        correlation = float(np.dot(left, right) / (left_norm * right_norm))
        return min(1.0, max(0.0, (correlation + 1.0) / 2.0))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view. Contains the grid, not the frame."""
        return {
            "grid": self.grid,
            "key": self.key,
            "mean_luma": round(self.mean_luma, 3),
            "values": [round(value, 3) for value in self.values],
        }


def _bounds(size: int, parts: int) -> list[tuple[int, int]]:
    """Split ``size`` pixels into ``parts`` contiguous bands covering all of it.

    The remainder is spread over the leading bands, so every pixel belongs to
    exactly one band. This is the same partition
    :func:`autocraft.perception.features._band_bounds` uses; it is duplicated
    rather than imported because it is four lines and the import would drag the
    whole feature module in for one helper.
    """
    parts = max(1, min(int(parts), max(1, int(size))))
    base, extra = divmod(int(size), parts)
    bands: list[tuple[int, int]] = []
    start = 0
    for index in range(parts):
        width = base + (1 if index < extra else 0)
        bands.append((start, start + width))
        start += width
    return bands


def _digest(values: Sequence[float]) -> str:
    """A short, stable key for a fingerprint's grid values."""
    payload = ",".join(f"{value:.2f}" for value in values)
    return hashlib.blake2b(payload.encode("ascii"), digest_size=6).hexdigest()


@dataclass(frozen=True)
class ViewRecord:
    """One stored view: its fingerprint, when it was taken, and its label."""

    fingerprint: ViewFingerprint
    index: int
    timestamp: float
    seen: int = 1

    @property
    def key(self) -> str:
        """The fingerprint's short key."""
        return self.fingerprint.key

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "key": self.key,
            "index": self.index,
            "timestamp": round(self.timestamp, 4),
            "seen": self.seen,
            "mean_luma": round(self.fingerprint.mean_luma, 3),
        }


@dataclass(frozen=True)
class ViewObservation:
    """The result of showing one view to :class:`ViewMemory`."""

    record: ViewRecord
    similarity: float
    matched: ViewRecord | None

    @property
    def revisited(self) -> bool:
        """True when this view matched a stored one closely enough."""
        return self.matched is not None

    @property
    def novelty(self) -> float:
        """How unlike anything stored this view is, in 0..1."""
        return max(0.0, 1.0 - self.similarity)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "key": self.record.key,
            "similarity": round(self.similarity, 4),
            "novelty": round(self.novelty, 4),
            "revisited": self.revisited,
            "matched_key": None if self.matched is None else self.matched.key,
            "matched_index": None if self.matched is None else self.matched.index,
            "times_seen": self.record.seen,
        }


class ViewMemory:
    """A bounded set of views the agent has recently seen.

    Bounded by construction. Once the deque is full the oldest view is gone, and
    a later frame that would have matched it is correctly reported as new - the
    agent has forgotten it, and saying otherwise would be a lie about its own
    memory.
    """

    def __init__(
        self,
        *,
        grid: int = DEFAULT_VIEW_GRID,
        limit: int = 24,
        similar_threshold: float = DEFAULT_SIMILAR_THRESHOLD,
    ) -> None:
        self.grid = max(1, int(grid))
        self.limit = max(1, int(limit))
        self.similar_threshold = float(similar_threshold)
        self._records: deque[ViewRecord] = deque(maxlen=self.limit)
        self._unique = 0
        self._revisits = 0

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> tuple[ViewRecord, ...]:
        """Stored views, oldest first."""
        return tuple(self._records)

    @property
    def unique_count(self) -> int:
        """How many views were judged new when first seen."""
        return self._unique

    @property
    def revisit_count(self) -> int:
        """How many views matched something already stored."""
        return self._revisits

    def reset(self) -> None:
        """Forget everything."""
        self._records.clear()
        self._unique = 0
        self._revisits = 0

    def best_match(self, fingerprint: ViewFingerprint) -> tuple[ViewRecord | None, float]:
        """Return the most similar stored view and its similarity, if any."""
        best: ViewRecord | None = None
        best_similarity = 0.0
        for record in self._records:
            similarity = record.fingerprint.similarity(fingerprint)
            if similarity > best_similarity:
                best = record
                best_similarity = similarity
        return best, best_similarity

    def observe(
        self,
        fingerprint: ViewFingerprint,
        *,
        index: int = 0,
        timestamp: float = 0.0,
    ) -> ViewObservation:
        """Store a view, reporting whether it was one already known."""
        matched, similarity = self.best_match(fingerprint)
        if matched is not None and similarity >= self.similar_threshold:
            self._revisits += 1
            updated = replace(matched, seen=matched.seen + 1)
            # Move the refreshed record to the back so "recently seen" means
            # recently seen, not "seen once a long time ago".
            self._records.remove(matched)
            self._records.append(updated)
            return ViewObservation(record=updated, similarity=similarity, matched=updated)
        self._unique += 1
        record = ViewRecord(fingerprint=fingerprint, index=int(index), timestamp=float(timestamp), seen=1)
        self._records.append(record)
        return ViewObservation(record=record, similarity=similarity, matched=None)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "grid": self.grid,
            "limit": self.limit,
            "similar_threshold": self.similar_threshold,
            "stored": len(self._records),
            "unique_views": self._unique,
            "revisited_views": self._revisits,
            "records": [record.to_dict() for record in self._records],
        }


@dataclass(frozen=True)
class ActionRecord:
    """One movement the behaviour layer asked for, and what came of it.

    ``progress`` is signed and in the units of whatever was being measured, so a
    positive value always means "closer to what I wanted" and a negative value
    always means "further away". ``None`` means the layer had nothing measurable
    to say about this action - which is a different claim from zero.
    """

    index: int
    strategy: str
    kind: str
    dx: int = 0
    dy: int = 0
    view_key_before: str = ""
    view_key_after: str = ""
    view_similarity: float | None = None
    outcome: str = "unknown"
    progress: float | None = None
    distance_before: float | None = None
    distance_after: float | None = None

    @property
    def movement(self) -> tuple[int, int]:
        """The movement as ``(dx, dy)``."""
        return (self.dx, self.dy)

    @property
    def improved(self) -> bool | None:
        """True when progress was measured and positive, None when unmeasured."""
        return None if self.progress is None else self.progress > 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "index": self.index,
            "strategy": self.strategy,
            "kind": self.kind,
            "dx": self.dx,
            "dy": self.dy,
            "view_key_before": self.view_key_before,
            "view_key_after": self.view_key_after,
            "view_similarity": None if self.view_similarity is None else round(self.view_similarity, 4),
            "outcome": self.outcome,
            "progress": None if self.progress is None else round(self.progress, 4),
            "distance_before": None if self.distance_before is None else round(self.distance_before, 3),
            "distance_after": None if self.distance_after is None else round(self.distance_after, 3),
        }


@dataclass(frozen=True)
class FailedAttempt:
    """A strategy that was tried on a target and did not help."""

    strategy: str
    reason: str
    index: int = 0
    view_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "strategy": self.strategy,
            "reason": self.reason,
            "index": self.index,
            "view_key": self.view_key,
        }


@dataclass(frozen=True)
class ProgressSample:
    """One measurement of how far the current target is from where it should be."""

    index: int
    strategy: str
    distance: float
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "index": self.index,
            "strategy": self.strategy,
            "distance": round(self.distance, 3),
            "confidence": round(self.confidence, 4),
        }


class ShortTermExperience:
    """Everything the agent remembers about the last few seconds of its run.

    All of it is bounded. The constructor's limits are the only thing standing
    between "working memory" and "a leak that grows for the length of a run", so
    they are required, not optional.
    """

    def __init__(
        self,
        *,
        view_grid: int = DEFAULT_VIEW_GRID,
        view_limit: int = 24,
        similar_threshold: float = DEFAULT_SIMILAR_THRESHOLD,
        action_limit: int = 64,
        candidate_limit: int = 32,
        failure_limit: int = 32,
        progress_limit: int = 64,
        merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
    ) -> None:
        self.views = ViewMemory(grid=view_grid, limit=view_limit, similar_threshold=similar_threshold)
        self.merge_threshold = float(merge_threshold)
        self.actions: deque[ActionRecord] = deque(maxlen=max(1, int(action_limit)))
        self.candidates: deque[CandidateTarget] = deque(maxlen=max(1, int(candidate_limit)))
        self.failed_attempts: deque[FailedAttempt] = deque(maxlen=max(1, int(failure_limit)))
        self.progress_history: deque[ProgressSample] = deque(maxlen=max(1, int(progress_limit)))
        self.current_strategy: str = ""
        self.strategy_attempt_count: int = 0
        self.selected_target: CandidateTarget | None = None

    def reset(self) -> None:
        """Forget everything. Called at the start of every run."""
        self.views.reset()
        self.actions.clear()
        self.candidates.clear()
        self.failed_attempts.clear()
        self.progress_history.clear()
        self.current_strategy = ""
        self.strategy_attempt_count = 0
        self.selected_target = None

    # -- views ------------------------------------------------------------

    def note_view(self, fingerprint: ViewFingerprint, *, index: int, timestamp: float) -> ViewObservation:
        """Record a view and report whether it was one already known."""
        return self.views.observe(fingerprint, index=index, timestamp=timestamp)

    @property
    def unique_views(self) -> int:
        """Distinct views seen this run."""
        return self.views.unique_count

    @property
    def revisited_views(self) -> int:
        """Times a view matched one already stored."""
        return self.views.revisit_count

    # -- candidates -------------------------------------------------------

    def note_candidate(self, candidate: CandidateTarget) -> CandidateTarget:
        """Record a candidate, merging it with an earlier sighting if it is one.

        Matching is by appearance descriptor, not by position, because position
        is precisely what a camera movement changes. A merged candidate keeps its
        original identity and its ``seen`` count grows, which is what stops the
        selection score from nominating the same region forever.

        Returns:
            The stored candidate, which is the merged one when a match was found.
        """
        best: CandidateTarget | None = None
        best_similarity = 0.0
        for stored in self.candidates:
            similarity = descriptor_similarity(stored.descriptor, candidate.descriptor)
            if similarity > best_similarity:
                best = stored
                best_similarity = similarity
        if best is not None and best_similarity >= self.merge_threshold:
            merged = replace(
                candidate,
                seen=best.seen + 1,
                novelty=min(best.novelty, 1.0 - best_similarity),
                persistence=min(1.0, best.persistence + 0.5),
            )
            self.candidates.remove(best)
            self.candidates.append(merged)
            return merged
        fresh = replace(candidate, seen=max(1, candidate.seen), novelty=1.0, persistence=0.5)
        self.candidates.append(fresh)
        return fresh

    def mark_candidate_failed(self, candidate: CandidateTarget, reason: str, *, index: int, view_key: str = "") -> None:
        """Record that a candidate could not be centred, and why."""
        self.failed_attempts.append(FailedAttempt(strategy=self.current_strategy, reason=reason, index=index, view_key=view_key))
        self.candidates.append(replace(candidate, seen=candidate.seen + 1))

    def strongest_candidates(self, limit: int = 8) -> list[CandidateTarget]:
        """Stored candidates, most persistent and least examined first."""
        ordered = sorted(self.candidates, key=lambda item: (-item.persistence, item.seen, item.bbox))
        return ordered[: max(1, int(limit))]

    # -- actions and progress ---------------------------------------------

    def note_action(self, record: ActionRecord) -> ActionRecord:
        """Record a movement and its outcome."""
        self.actions.append(record)
        return record

    def note_progress(self, sample: ProgressSample) -> ProgressSample:
        """Record a measurement of how far the target is from where it should be."""
        self.progress_history.append(sample)
        return sample

    def set_strategy(self, strategy: str) -> None:
        """Switch strategy, resetting the per-strategy attempt counter."""
        if strategy != self.current_strategy:
            self.current_strategy = str(strategy)
            self.strategy_attempt_count = 0

    def note_strategy_attempt(self) -> int:
        """Count one attempt at the current strategy and return the new count."""
        self.strategy_attempt_count += 1
        return self.strategy_attempt_count

    @property
    def last_action(self) -> ActionRecord | None:
        """The most recent action, or ``None``."""
        return self.actions[-1] if self.actions else None

    @property
    def last_progress(self) -> ProgressSample | None:
        """The most recent progress measurement, or ``None``."""
        return self.progress_history[-1] if self.progress_history else None

    def distance_series(self, limit: int = 8) -> tuple[float, ...]:
        """The last few measured distances, oldest first."""
        values = [sample.distance for sample in self.progress_history]
        return tuple(values[-max(1, int(limit)) :])

    def recent_failure_count(self, strategy: str, limit: int = 8) -> int:
        """How many of the last ``limit`` failures were this strategy's."""
        recent = list(self.failed_attempts)[-max(1, int(limit)) :]
        return sum(1 for failure in recent if failure.strategy == strategy)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view. Contains no pixel data."""
        return {
            "current_strategy": self.current_strategy,
            "strategy_attempt_count": self.strategy_attempt_count,
            "selected_target": None if self.selected_target is None else self.selected_target.to_dict(),
            "views": self.views.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "actions": [record.to_dict() for record in self.actions],
            "failed_attempts": [failure.to_dict() for failure in self.failed_attempts],
            "progress_history": [sample.to_dict() for sample in self.progress_history],
        }


def describe_actions(records: Iterable[ActionRecord]) -> list[str]:
    """Render a sequence of actions as short readable lines, for the terminal."""
    lines: list[str] = []
    for record in records:
        movement = f"({record.dx:+d}, {record.dy:+d})"
        progress = "n/a" if record.progress is None else f"{record.progress:+.1f}"
        lines.append(f"{record.index:>3} {record.strategy:<18} {movement:>12}  {record.outcome:<10} progress {progress}")
    return lines


def summarise_memory(memory: ShortTermExperience) -> dict[str, Any]:
    """A compact, human-facing summary of the short-term experience."""
    distances = memory.distance_series()
    return {
        "unique_views": memory.unique_views,
        "revisited_views": memory.revisited_views,
        "candidates": len(memory.candidates),
        "actions": len(memory.actions),
        "failed_attempts": len(memory.failed_attempts),
        "current_strategy": memory.current_strategy,
        "distance_series": [round(value, 1) for value in distances],
        "distance_improved": _improved(distances),
    }


def _improved(distances: Sequence[float]) -> bool | None:
    """True when the last distance is below the first, None when too few."""
    if len(distances) < 2:
        return None
    return bool(distances[-1] < distances[0] - 1e-9)
