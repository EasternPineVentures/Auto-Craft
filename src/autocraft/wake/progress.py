"""Did that help? And am I doing the same useless thing again?

This module holds the two questions WAKE-001 is required to be able to answer
about its own behaviour, and it holds them together because they are two halves
of one idea.

The first half is **progress**. The milestone is explicit that the agent may not
decide whether to continue on a bare success/failure flag: a run that only records
"did it work: no" cannot tell the difference between a strategy that is slowly
converging and one that is thrashing. So every verdict here carries a signed
magnitude - a distance that got smaller by 46 pixels, a view that gained 0.7 of
novelty - and ``None`` where nothing was measurable. ``None`` is not zero. A
strategy whose effect could not be measured has not been shown to fail.

The second half is **repetition**, and it draws a line the milestone insists on:

* **Productive repetition is fine and must not be penalised.** Closing on a
  target at 164 -> 97 -> 43 -> 11 pixels is four looks in the same direction and
  it is exactly what the agent is supposed to do. Anything that flagged this as
  a loop would break the milestone's own success criterion.
* **Dead repetition is not.** The same action, from a view that has not
  meaningfully changed, with no measured improvement, is a loop - and the third
  one should not be attempted blindly.

The distinction is drawn from measurements only: a signature of (view, action),
the measured progress series, and how many times the signature has repeated
consecutively. When a loop is detected the response is to *change strategy*. It
is never to add random movement, which is why nothing in this module can produce
a movement at all - it only ever says "this is a loop" and "that strategy is
temporarily unavailable".
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .memory import ViewFingerprint

__all__ = [
    "DEFAULT_COOLDOWN_FAILURES",
    "DEFAULT_PROGRESS_EPSILON",
    "DEFAULT_STUCK_REPEATS",
    "Cooldown",
    "ProgressModel",
    "ProgressVerdict",
    "RepetitionGuard",
    "RepetitionVerdict",
    "StrategyCooldown",
]

#: Smallest measured improvement that counts as progress, in the units being
#: measured - pixels of distance for centring, and a fraction of novelty for
#: scanning. One pixel is chosen because below that the measurement is noise and
#: calling it progress would let the agent congratulate itself on nothing.
DEFAULT_PROGRESS_EPSILON = 1.0

#: How many times the same (view, action) may repeat without progress before the
#: pattern is called stuck. Three, because the milestone says so: "the third
#: identical failed strategy should not be attempted blindly".
DEFAULT_STUCK_REPEATS = 3

#: How many failures a strategy may accumulate before it is taken out of
#: circulation. The milestone's own worked example uses three.
DEFAULT_COOLDOWN_FAILURES = 3

#: How many recent signatures the reported cycle shows. Long enough to see the
#: shape of a loop, short enough to read.
_CYCLE_LENGTH = 4


@dataclass(frozen=True)
class ProgressVerdict:
    """Whether an action improved the situation, and by how much.

    ``delta`` is signed: positive means the situation improved. ``measured``
    being False means there was nothing to compare, which is a different claim
    from "nothing improved".
    """

    measured: bool
    improved: bool | None
    delta: float | None
    reason: str = ""

    @property
    def productive(self) -> bool:
        """True when this action measurably helped."""
        return bool(self.measured and self.improved)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "measured": self.measured,
            "improved": self.improved,
            "delta": None if self.delta is None else round(self.delta, 4),
            "reason": self.reason,
        }


class ProgressModel:
    """Tracks whether the run is getting anywhere, with magnitudes not flags.

    It carries two separate progress notions because the two phases of the run
    are measured in different units and it would be dishonest to add them:

    * scanning is measured by *new information* - how many of the looks taken so
      far produced a view that was genuinely new;
    * centring is measured by *distance* - how far the selected target sits from
      the centre of the screen.

    The ratios at the bottom are the ones the milestone names explicitly. They are
    computed over attempts, not over actions, so an action that could not be
    measured does not silently count as a failure.
    """

    def __init__(self, *, epsilon: float = DEFAULT_PROGRESS_EPSILON) -> None:
        self.epsilon = max(0.0, float(epsilon))
        self.scans = 0
        self.new_views = 0
        self.attempts = 0
        self.productive = 0
        self.dead = 0
        self._progress_values: deque[float] = deque(maxlen=64)

    def reset(self) -> None:
        """Forget every measurement."""
        self.scans = 0
        self.new_views = 0
        self.attempts = 0
        self.productive = 0
        self.dead = 0
        self._progress_values.clear()

    def note_scan(self, *, new_view: bool) -> None:
        """Count one look, and whether it produced a view that was new."""
        self.scans += 1
        if new_view:
            self.new_views += 1

    def assess(self, *, before: float | None, after: float | None, unit: str = "px") -> ProgressVerdict:
        """Compare two measurements of the same quantity.

        Args:
            before: The value before the action, or ``None`` when unmeasured.
            after: The value after the action, or ``None`` when unmeasured.
            unit: Only used in the human-readable reason.

        Returns:
            A :class:`ProgressVerdict`. Unmeasurable comparisons report
            ``measured=False`` rather than pretending nothing improved.
        """
        self.attempts += 1
        if before is None or after is None:
            return ProgressVerdict(measured=False, improved=None, delta=None, reason="not measurable")
        if not (math.isfinite(float(before)) and math.isfinite(float(after))):
            return ProgressVerdict(measured=False, improved=None, delta=None, reason="not finite")
        delta = float(before) - float(after)
        self._progress_values.append(delta)
        if delta > self.epsilon:
            self.productive += 1
            return ProgressVerdict(
                measured=True,
                improved=True,
                delta=delta,
                reason=f"{delta:.1f} {unit} closer",
            )
        # Anything that did not measurably help counts as dead repetition, including
        # a change inside the noise band. "Nothing moved" is not progress, and a
        # ratio that reported zero dead repetition through a run where nothing ever
        # moved would be worse than useless - it would look like success.
        self.dead += 1
        if delta < -self.epsilon:
            return ProgressVerdict(
                measured=True,
                improved=False,
                delta=delta,
                reason=f"{-delta:.1f} {unit} further away",
            )
        return ProgressVerdict(
            measured=True,
            improved=False,
            delta=delta,
            reason=f"changed by {delta:.1f} {unit}, inside the {self.epsilon:g} {unit} noise band",
        )

    @property
    def new_view_fraction(self) -> float:
        """Fraction of looks so far that produced a genuinely new view."""
        if self.scans <= 0:
            return 0.0
        return self.new_views / self.scans

    @property
    def novelty_gain(self) -> float:
        """Mean measured novelty gain per look. Zero when nothing was measured."""
        return self.new_view_fraction

    @property
    def dead_repetition_ratio(self) -> float:
        """Fraction of measurable attempts that did not measurably help."""
        if self.attempts <= 0:
            return 0.0
        return self.dead / self.attempts

    @property
    def productive_repetition_ratio(self) -> float:
        """Fraction of measurable attempts that measurably helped.

        The two ratios are complements over the attempts that could be measured, so
        a run where nothing ever moved reads as all dead repetition rather than as
        no repetition at all.
        """
        if self.attempts <= 0:
            return 0.0
        return self.productive / self.attempts

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "scans": self.scans,
            "new_views": self.new_views,
            "new_view_fraction": round(self.new_view_fraction, 4),
            "novelty_gain": round(self.novelty_gain, 4),
            "attempts": self.attempts,
            "productive": self.productive,
            "dead": self.dead,
            "dead_repetition_ratio": round(self.dead_repetition_ratio, 4),
            "productive_repetition_ratio": round(self.productive_repetition_ratio, 4),
            "epsilon": self.epsilon,
        }


@dataclass(frozen=True)
class RepetitionVerdict:
    """The repetition guard's ruling on one attempted action."""

    stuck: bool
    signature: str
    repeats: int
    cycle: tuple[str, ...] = ()
    reason: str = ""
    recent_progress: tuple[float | None, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "stuck": self.stuck,
            "signature": self.signature,
            "repeats": self.repeats,
            "cycle": list(self.cycle),
            "reason": self.reason,
            "recent_progress": [None if value is None else round(value, 4) for value in self.recent_progress],
        }


class RepetitionGuard:
    """Detects dead repetition and stays quiet about productive repetition.

    A pattern is a (view, action) signature. It is *dead* when the same signature
    is attempted ``repeats`` times in a row and none of those attempts produced
    measured progress. Anything less than that is not called a loop:

    * a single repeat is not enough, because correcting an overshoot legitimately
      means moving back the way you came;
    * measured progress resets the streak, which is what makes 164 -> 97 -> 43 ->
      11 pixels repeatable in the same direction forever without complaint.

    The guard reports; it never moves anything. Its only output is a verdict and
    the advice to change strategy.
    """

    def __init__(self, *, repeats: int = DEFAULT_STUCK_REPEATS, epsilon: float = DEFAULT_PROGRESS_EPSILON) -> None:
        self.repeats = max(2, int(repeats))
        self.epsilon = max(0.0, float(epsilon))
        self.stuck_patterns_detected = 0
        self.stuck_patterns_broken = 0
        self._streak: dict[str, int] = {}
        self._recent: deque[str] = deque(maxlen=_CYCLE_LENGTH)
        self._recent_progress: deque[float | None] = deque(maxlen=_CYCLE_LENGTH)
        self._last_signature = ""
        self._in_pattern = False

    def reset(self) -> None:
        """Forget every pattern."""
        self.stuck_patterns_detected = 0
        self.stuck_patterns_broken = 0
        self._streak.clear()
        self._recent.clear()
        self._recent_progress.clear()
        self._last_signature = ""
        self._in_pattern = False

    @staticmethod
    def signature(*, view_key: str, kind: str, dx: int, dy: int, strategy: str = "") -> str:
        """Build the compact fingerprint the guard matches on.

        It deliberately includes the view: the same movement from a different
        view is not the same attempt, and treating it as one would flag ordinary
        scanning as a loop.
        """
        return f"{view_key}:{kind}:{int(dx):+d},{int(dy):+d}:{strategy}"

    def observe(
        self,
        signature: str,
        *,
        progress: float | None = None,
    ) -> RepetitionVerdict:
        """Record one attempt and rule on whether it is part of a dead loop.

        Args:
            signature: From :meth:`signature`.
            progress: Signed measured progress, or ``None`` when unmeasured.

        Returns:
            A :class:`RepetitionVerdict`. ``stuck`` becomes True only on the
            ``repeats``-th consecutive non-improving attempt of one signature.
        """
        improved = progress is not None and progress > self.epsilon
        if signature == self._last_signature and not improved:
            self._streak[signature] = self._streak.get(signature, 1) + 1
        else:
            if self._in_pattern:
                self.stuck_patterns_broken += 1
            self._streak = {signature: 1}
            self._in_pattern = False
        self._last_signature = signature
        self._recent.append(signature)
        self._recent_progress.append(progress)

        count = self._streak.get(signature, 1)
        stuck = count >= self.repeats
        if stuck:
            if not self._in_pattern:
                self.stuck_patterns_detected += 1
            self._in_pattern = True
        reason = ""
        if stuck:
            reason = (
                f"the same action was attempted {count} times from an unchanged view "
                "with no measured improvement"
            )
        return RepetitionVerdict(
            stuck=stuck,
            signature=signature,
            repeats=count,
            cycle=tuple(self._recent),
            reason=reason,
            recent_progress=tuple(self._recent_progress),
        )

    def note_strategy_change(self) -> None:
        """Record that the behaviour layer actually changed strategy.

        This is the only thing that clears a detected pattern's streak other than
        measured progress, and it is what makes ``stuck_patterns_broken`` an
        honest count of loops that were genuinely escaped rather than merely
        re-labelled.
        """
        if self._in_pattern:
            self.stuck_patterns_broken += 1
        self._in_pattern = False
        self._streak.clear()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "repeats": self.repeats,
            "epsilon": self.epsilon,
            "stuck_patterns_detected": self.stuck_patterns_detected,
            "stuck_patterns_broken": self.stuck_patterns_broken,
            "last_signature": self._last_signature,
            "streak": self._streak.get(self._last_signature, 0),
            "in_pattern": self._in_pattern,
        }


@dataclass(frozen=True)
class Cooldown:
    """A strategy that has failed enough times to be set aside."""

    strategy: str
    failures: int
    reason: str
    released: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "strategy": self.strategy,
            "failures": self.failures,
            "reason": self.reason,
            "active": not self.released,
        }


class StrategyCooldown:
    """Makes a repeatedly failing strategy temporarily unavailable.

    "Temporarily" is a claim about the *visual situation*, not about the clock.
    The milestone allows a cooled strategy back as soon as the situation
    materially changes, and material change is measured the same way view memory
    measures it: the current view's coarse fingerprint is compared with the one
    that was on screen when the strategy was set aside. If they are no longer
    similar, the situation has changed and the strategy is released.

    That makes the cooldown exactly as long as it needs to be and no longer - and
    it means a cooled strategy can never become permanently unavailable, which
    would leave the agent with nothing left to try.
    """

    def __init__(
        self,
        *,
        failures: int = DEFAULT_COOLDOWN_FAILURES,
        similar_threshold: float = 0.92,
    ) -> None:
        self.failures = max(1, int(failures))
        self.similar_threshold = float(similar_threshold)
        self._counts: dict[str, int] = {}
        self._entries: dict[str, tuple[Cooldown, ViewFingerprint | None]] = {}

    def reset(self) -> None:
        """Release everything and forget every failure."""
        self._counts.clear()
        self._entries.clear()

    def record_failure(
        self,
        strategy: str,
        *,
        reason: str,
        fingerprint: ViewFingerprint | None = None,
    ) -> Cooldown:
        """Count one failure, cooling the strategy down once it has had enough."""
        count = self._counts.get(strategy, 0) + 1
        self._counts[strategy] = count
        if count < self.failures:
            return Cooldown(strategy=strategy, failures=count, reason=reason, released=True)
        entry = Cooldown(strategy=strategy, failures=count, reason=reason, released=False)
        self._entries[strategy] = (entry, fingerprint)
        return entry

    def is_available(self, strategy: str, *, fingerprint: ViewFingerprint | None = None) -> bool:
        """True when ``strategy`` may be attempted in the current situation.

        A released strategy is always available. A cooled one becomes available
        again - and is released - the moment the current view is no longer similar
        to the view it failed in.
        """
        pair = self._entries.get(strategy)
        if pair is None:
            return True
        entry, cooled_fingerprint = pair
        if entry.released:
            return True
        if fingerprint is None or cooled_fingerprint is None:
            return False
        if cooled_fingerprint.similarity(fingerprint) < self.similar_threshold:
            self._entries[strategy] = (replace(entry, released=True), cooled_fingerprint)
            return True
        return False

    def release(self, strategy: str) -> None:
        """Release a strategy explicitly."""
        pair = self._entries.get(strategy)
        if pair is None:
            return
        entry, fingerprint = pair
        self._entries[strategy] = (replace(entry, released=True), fingerprint)

    def active(self) -> tuple[Cooldown, ...]:
        """Cooled strategies that are still unavailable."""
        return tuple(entry for entry, _ in self._entries.values() if not entry.released)

    def available(self, strategies: Sequence[str], *, fingerprint: ViewFingerprint | None = None) -> tuple[str, ...]:
        """The subset of ``strategies`` currently available, order preserved."""
        return tuple(strategy for strategy in strategies if self.is_available(strategy, fingerprint=fingerprint))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "failures_before_cooldown": self.failures,
            "counts": dict(self._counts),
            "entries": [entry.to_dict() for entry, _ in self._entries.values()],
            "active": [entry.strategy for entry in self.active()],
        }


def summarise_repetition(
    progress: ProgressModel,
    guard: RepetitionGuard,
    cooldown: StrategyCooldown,
) -> Mapping[str, Any]:
    """A compact, human-facing summary of the progress and repetition state."""
    return {
        "progress": progress.to_dict(),
        "repetition": guard.to_dict(),
        "cooldown": cooldown.to_dict(),
    }
