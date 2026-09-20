"""The thought engine: whether to speak, and how often.

Two jobs, kept separate because they are testable separately.

:class:`ThoughtPolicy` holds the *deterministic* gates - is the feature on, has
the cooldown elapsed, is the per-minute ceiling reached, is the agent in one of
its occasional long silences - plus the probability a thought is attempted. The
gates are pure functions of timestamps, so they can be tested without touching a
random source; the probability is a function of the affect state and of whether
anything meaningful recently happened, so "affect influences generation" is a
property of a pure function rather than of a code path.

:class:`ThoughtEngine` owns the stateful part: the history, the last-thought
timestamp, and the scheduled silence.

**This module cannot act.** It holds no reference to an executor, a keyboard, a
mouse or a safety guard, and :meth:`ThoughtEngine.maybe_think` returns a
:class:`~autocraft.thoughts.model.ThoughtEvent` to its caller instead of doing
anything with it. The specification's rule is that a thought must never cause
keyboard or mouse input, and the way that is guaranteed here is that there is no
object in this file capable of producing input.

The randomness is bounded in the sense the specification asks for. There is a
minimum interval, a per-minute ceiling, a lower probability while the agent is
busy, and a scheduled silence after speaking so the rhythm is irregular rather
than a metronome.
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Callable

from .generate import ScriptedThoughtGenerator, TemplateThoughtGenerator, ThoughtGenerator
from .model import ThoughtContext, ThoughtEvent

__all__ = [
    "DEMO_POLICY",
    "ThoughtEngine",
    "ThoughtPolicy",
]

#: Events that make a thought more likely, because something worth reacting to
#: just happened.
_MEANINGFUL_MARKERS = (
    "refused",
    "failed",
    "blocked",
    "emergency",
    "accepted",
    "visual change detected",
    "first",
    "run complete",
)


@dataclass(frozen=True)
class ThoughtPolicy:
    """Rate limits and probabilities governing when AutoCraft speaks.

    Attributes:
        enabled: Master switch. Off means the agent never expresses a thought.
        min_interval_seconds: Hard cooldown between thoughts.
        max_per_minute: Ceiling on thoughts in any trailing 60 seconds.
        base_probability: Chance of speaking at an opportunity that passes the
            gates, before affect and event adjustments.
        busy_action_rate: Loop iterations per second above which the agent counts
            as busy and gets quieter.
        busy_damping: Multiplier applied to the probability while busy.
        quiet_chance: Chance, after speaking, of scheduling an extended silence.
        quiet_min_seconds: Shortest scheduled silence.
        quiet_max_seconds: Longest scheduled silence.
    """

    enabled: bool = True
    min_interval_seconds: float = 25.0
    max_per_minute: float = 3.0
    base_probability: float = 0.06
    busy_action_rate: float = 4.0
    busy_damping: float = 0.15
    quiet_chance: float = 0.35
    quiet_min_seconds: float = 30.0
    quiet_max_seconds: float = 150.0

    def silence_reason(
        self,
        *,
        now: float,
        last_at: float | None,
        timestamps: deque[float],
        quiet_until: float | None,
        context: ThoughtContext,
    ) -> str | None:
        """Return why the agent should stay quiet, or ``None`` to proceed.

        Deterministic: no random source is touched, so a test can assert each
        gate on its own.
        """
        if not self.enabled:
            return "disabled"
        if quiet_until is not None and now < quiet_until:
            return "scheduled silence"
        if last_at is not None and now - last_at < self.min_interval_seconds:
            return "cooldown"
        if self.max_per_minute > 0:
            recent = [stamp for stamp in timestamps if now - stamp <= 60.0]
            if len(recent) >= self.max_per_minute:
                return "per-minute ceiling"
        return None

    def probability(self, context: ThoughtContext) -> float:
        """Chance of speaking now, adjusted by affect and recent events.

        Every adjustment is multiplicative and bounded, so affect *influences* the
        outcome without ever forcing it - which is what the specification asks
        for. High curiosity raises the chance, low energy lowers it, and a recent
        meaningful event raises it further.
        """
        if not self.enabled:
            return 0.0
        chance = self.base_probability
        curiosity = context.affect_of("curiosity", 0.5)
        energy = context.affect_of("energy", 1.0)
        stress = context.affect_of("stress", 0.1)
        chance *= 0.4 + 1.6 * curiosity
        chance *= 0.5 + 0.9 * energy
        chance *= 1.0 + 0.5 * stress
        if context.recent_action_rate > self.busy_action_rate:
            chance *= self.busy_damping
        recent = " ".join(context.recent_events[-6:]).lower()
        if any(marker in recent for marker in _MEANINGFUL_MARKERS):
            chance *= 1.6
        if context.goal:
            chance *= 1.1
        return min(1.0, max(0.0, chance))


class ThoughtEngine:
    """Decides whether to express a thought, and keeps the recent history.

    The caller supplies a :class:`~autocraft.thoughts.model.ThoughtContext` and
    receives either a thought or ``None``. Nothing else happens: the engine has no
    way to act on what it produces, so the expression layer cannot become a
    control path.
    """

    def __init__(
        self,
        config: Any,
        *,
        generator: ThoughtGenerator | None = None,
        policy: ThoughtPolicy | None = None,
        clock: Callable[[], float] = time.time,
        random_source: random.Random | None = None,
    ) -> None:
        self._clock = clock
        self._rng = random_source if random_source is not None else random.Random()
        self._generator = generator if generator is not None else TemplateThoughtGenerator(
            random_source=self._rng
        )
        self._policy = policy if policy is not None else _policy_from_config(config)
        self._history: deque[ThoughtEvent] = deque(maxlen=max(1, int(config.thought_history_max)))
        self._timestamps: deque[float] = deque()
        self._last_at: float | None = None
        self._quiet_until: float | None = None
        self._last_silence_reason: str | None = None

    @property
    def policy(self) -> ThoughtPolicy:
        """The policy in force."""
        return self._policy

    @property
    def generator_name(self) -> str:
        """Provenance string of the active generator."""
        return str(getattr(self._generator, "name", type(self._generator).__name__))

    @property
    def history(self) -> tuple[ThoughtEvent, ...]:
        """Thoughts expressed so far, oldest first, bounded by config."""
        return tuple(self._history)

    @property
    def last_thought_at(self) -> float | None:
        """When the most recent thought was expressed, or ``None``."""
        return self._last_at

    @property
    def last_silence_reason(self) -> str | None:
        """Why the last opportunity passed without a thought, for diagnostics."""
        return self._last_silence_reason

    def maybe_think(self, context: ThoughtContext) -> ThoughtEvent | None:
        """Express a thought if the policy allows, otherwise return ``None``."""
        now = float(context.now) if context.now else float(self._clock())
        self._prune(now)
        reason = self._policy.silence_reason(
            now=now,
            last_at=self._last_at,
            timestamps=self._timestamps,
            quiet_until=self._quiet_until,
            context=context,
        )
        if reason is not None:
            self._last_silence_reason = reason
            return None
        if self._rng.random() >= self._policy.probability(context):
            self._last_silence_reason = "probability"
            return None

        prepared = self._prepare(context, now)
        thought = self._generator.generate(prepared)
        if thought is None:
            # The generator had nothing to say. This is a quiet tick, not a
            # failure, and it must not consume the cooldown.
            self._last_silence_reason = "generator quiet"
            return None

        self._history.append(thought)
        self._timestamps.append(now)
        self._last_at = now
        self._last_silence_reason = None
        self._schedule_quiet(now)
        return thought

    def reset(self) -> None:
        """Clear the history and the cooldown, keeping the generator's position."""
        self._history.clear()
        self._timestamps.clear()
        self._last_at = None
        self._quiet_until = None
        self._last_silence_reason = None

    def _prepare(self, context: ThoughtContext, now: float) -> ThoughtContext:
        """Fill in the two fields the engine owns rather than the caller."""
        return replace(
            context,
            now=now,
            seconds_since_last_thought=None if self._last_at is None else max(0.0, now - self._last_at),
            recent_thoughts=tuple(
                thought.text for thought in reversed(self._history) if thought.text
            ),
        )

    def _schedule_quiet(self, now: float) -> None:
        """Sometimes go quiet for a while, so the rhythm is not a metronome."""
        if self._rng.random() < self._policy.quiet_chance:
            self._quiet_until = now + self._rng.uniform(
                self._policy.quiet_min_seconds, self._policy.quiet_max_seconds
            )
        else:
            self._quiet_until = None

    def _prune(self, now: float) -> None:
        """Drop timestamps older than the rate window."""
        while self._timestamps and now - self._timestamps[0] > 60.0:
            self._timestamps.popleft()


def _policy_from_config(config: Any) -> ThoughtPolicy:
    """Build a policy from the configuration, falling back to defaults."""
    return ThoughtPolicy(
        enabled=bool(getattr(config, "thoughts_enabled", True)),
        min_interval_seconds=float(getattr(config, "thought_min_interval_seconds", 25.0)),
        max_per_minute=float(getattr(config, "thought_max_per_minute", 3.0)),
    )


#: The policy scripted demo output runs under.
#:
#: A demo ticker advances roughly thirty times faster than a real run, so the
#: production cooldown would silence the entire demo. This keeps the *shape* of
#: the cadence - a bounded probability, plus an occasional multi-tick silence - so
#: the demo shows irregularity rather than one thought per tick, without
#: pretending that 0.5 s is a realistic interval between thoughts.
DEMO_POLICY = ThoughtPolicy(
    enabled=True,
    min_interval_seconds=0.0,
    max_per_minute=0.0,
    base_probability=0.22,
    busy_action_rate=1e9,
    busy_damping=1.0,
    quiet_chance=0.3,
    quiet_min_seconds=6.0,
    quiet_max_seconds=20.0,
)


def scripted_engine(
    config: Any,
    *,
    clock: Callable[[], float] = time.time,
    thoughts: tuple[str, ...] | None = None,
    policy: ThoughtPolicy | None = None,
) -> ThoughtEngine:
    """Build an engine that replays labelled demo thoughts.

    Used by demo mode and by the tests that need a thought source which is
    deterministic without being a stub. The rate limits still apply, so the demo
    shows the real cadence rather than one thought per frame - but with
    :data:`DEMO_POLICY`, because a demo ticker advances far faster than a real run
    and the production cooldown would otherwise silence the whole demo.
    """
    return ThoughtEngine(
        config,
        generator=ScriptedThoughtGenerator(thoughts),
        policy=policy if policy is not None else DEMO_POLICY,
        clock=clock,
        random_source=random.Random(0),
    )
