"""The thought model: what AutoCraft says out loud about itself.

A :class:`ThoughtEvent` is an **expression**, not an instruction. It is generated
from state that already exists, it is displayed, and it stops there. Nothing in
this module can reach an input backend, and
:class:`~autocraft.thoughts.engine.ThoughtEngine` returns thoughts to its caller
rather than acting on them. The flow is one way:

    experience -> affect -> thought -> display

and never

    thought -> game action

Actual game actions keep going through planning and the safety layer, unchanged.

The spec's framing is worth restating because it governs every judgement in this
module: these are *deliberately generated, user-facing character expressions*.
They are not hidden chain-of-thought, they are not a transcript of model
reasoning, and nothing here reads or reveals private reasoning.
"""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

__all__ = [
    "THOUGHT_TONES",
    "THOUGHT_TRIGGERS",
    "ThoughtContext",
    "ThoughtError",
    "ThoughtEvent",
    "ThoughtTone",
    "ThoughtTrigger",
]


class ThoughtError(ValueError):
    """Raised when a thought cannot be constructed honestly."""


class ThoughtTone(str, Enum):
    """How a thought reads.

    Deliberately short. The specification says the taxonomy may grow and that V0
    should not over-engineer it, so this is the working set the page can style,
    not an attempt at completeness.
    """

    NEUTRAL = "neutral"
    HUMOROUS = "humorous"
    CURIOUS = "curious"
    HOPEFUL = "hopeful"
    EXCITED = "excited"
    SAD = "sad"
    FRUSTRATED = "frustrated"
    ANXIOUS = "anxious"
    DRAMATIC = "dramatic"
    ABSURD = "absurd"
    REFLECTIVE = "reflective"


class ThoughtTrigger(str, Enum):
    """What prompted a thought, drawn from the specification's trigger list."""

    DISCOVERY = "discovery"
    SUCCESS = "success"
    FAILURE = "failure"
    DANGER = "danger"
    MEMORY = "memory"
    IDLE = "idle"
    MILESTONE = "milestone"
    RANDOM_REFLECTION = "random_reflection"
    #: Synthetic. Only ever produced by demo and test sources, and the page shows
    #: the DEMO badge alongside it, so it cannot be confused with a real thought.
    DEMO = "demo"


THOUGHT_TONES: tuple[str, ...] = tuple(tone.value for tone in ThoughtTone)
THOUGHT_TRIGGERS: tuple[str, ...] = tuple(trigger.value for trigger in ThoughtTrigger)


def _finite(value: Any, *, name: str) -> float:
    """Coerce ``value`` to a finite float, rejecting bools and non-numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ThoughtError(f"{name} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ThoughtError(f"{name} must be finite, got {value!r}")
    return number


def _unit(value: Any, *, name: str) -> float:
    """Coerce ``value`` into the inclusive 0..1 range.

    Clamped rather than rejected: intensity and importance come out of a
    stochastic generator, and the boundary is the right place to absorb an
    overshoot. Fields a human types by hand - tone, trigger, text - are rejected
    instead, because those are mistakes rather than rounding.
    """
    return min(1.0, max(0.0, _finite(value, name=name)))


@dataclass(frozen=True)
class ThoughtEvent:
    """One user-facing character expression.

    Attributes:
        text: The sentence shown on stream. Must not be empty.
        tone: A :class:`ThoughtTone` value, or its string form.
        intensity: How strongly it reads, 0.0 - 1.0.
        trigger_type: A :class:`ThoughtTrigger` value, or its string form.
        trigger_reference: Short free-form pointer at what set it off, such as a
            step index or an event label. Never a payload.
        related_goal: The goal in force when it was generated, if any.
        related_memory: The stored event it refers to, if any. Only ever set from
            real memory - see
            :class:`~autocraft.thoughts.generate.TemplateThoughtGenerator`.
        affect_snapshot: The affect dimensions at generation time, as plain
            floats. A mapping rather than an ``AffectState`` so the thought model
            stays independent of the observer package.
        importance: How much this is worth keeping, 0.0 - 1.0. The seam a later
            stage uses to promote a thought into long-term memory.
        generated_by: Which generator produced it, for honesty about provenance.
        id: Short identifier, generated when omitted.
        timestamp: Unix seconds when it was created.
    """

    text: str
    tone: ThoughtTone | str = ThoughtTone.NEUTRAL
    intensity: float = 0.5
    trigger_type: ThoughtTrigger | str = ThoughtTrigger.RANDOM_REFLECTION
    trigger_reference: str = ""
    related_goal: str | None = None
    related_memory: str | None = None
    affect_snapshot: Mapping[str, float] = field(default_factory=dict)
    importance: float = 0.3
    generated_by: str = "unknown"
    id: str = ""
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        text = str(self.text).strip()
        if not text:
            raise ThoughtError("a thought must have text")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "tone", _coerce(ThoughtTone, self.tone, name="tone"))
        object.__setattr__(
            self, "trigger_type", _coerce(ThoughtTrigger, self.trigger_type, name="trigger_type")
        )
        object.__setattr__(self, "intensity", _unit(self.intensity, name="intensity"))
        object.__setattr__(self, "importance", _unit(self.importance, name="importance"))
        object.__setattr__(self, "timestamp", _finite(self.timestamp, name="timestamp"))
        object.__setattr__(self, "id", str(self.id) or secrets.token_hex(6))
        object.__setattr__(self, "generated_by", str(self.generated_by))
        object.__setattr__(self, "trigger_reference", str(self.trigger_reference))
        object.__setattr__(self, "related_goal", _optional_text(self.related_goal))
        object.__setattr__(self, "related_memory", _optional_text(self.related_memory))
        object.__setattr__(self, "affect_snapshot", _affect_snapshot(self.affect_snapshot))

    @property
    def tone_value(self) -> str:
        """The tone as a plain string, for rendering and serialisation."""
        return self.tone.value

    @property
    def trigger_value(self) -> str:
        """The trigger as a plain string, for rendering and serialisation."""
        return self.trigger_type.value

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-JSON mapping. Every value is a str, float, bool or None."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "text": self.text,
            "tone": self.tone.value,
            "intensity": round(self.intensity, 3),
            "trigger_type": self.trigger_type.value,
            "trigger_reference": self.trigger_reference,
            "related_goal": self.related_goal,
            "related_memory": self.related_memory,
            "affect_snapshot": dict(self.affect_snapshot),
            "importance": round(self.importance, 3),
            "generated_by": self.generated_by,
        }


@dataclass(frozen=True)
class ThoughtContext:
    """The only information a thought generator is given.

    The specification is explicit that a generator must not receive the whole
    application state, so this is a closed, explicit list: what the agent is
    trying to do, what it just saw, what happened lately, what it actually
    remembers, and how it currently feels.

    ``memories`` is the honesty seam. It holds strings taken from the agent's own
    recorded events, and a generator that wants to say "this looks like the place
    I fell" must find that fall in here. If it is empty there is nothing to
    remember, and a generator is required to say nothing about memory rather than
    invent one.
    """

    goal: str | None = None
    intention: str | None = None
    observation_summary: str | None = None
    recent_events: tuple[str, ...] = ()
    memories: tuple[str, ...] = ()
    #: Texts of the thoughts just expressed, newest first, so a generator can
    #: avoid repeating itself without being handed the whole history object.
    recent_thoughts: tuple[str, ...] = ()
    affect: Mapping[str, float] = field(default_factory=dict)
    mode: str = "idle"
    #: Seconds since the previous thought, or ``None`` when there has not been one.
    seconds_since_last_thought: float | None = None
    #: Loop iterations per second, used to keep the agent quiet while it is busy.
    recent_action_rate: float = 0.0
    now: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal", _optional_text(self.goal))
        object.__setattr__(self, "intention", _optional_text(self.intention))
        object.__setattr__(self, "observation_summary", _optional_text(self.observation_summary))
        object.__setattr__(self, "recent_events", _text_tuple(self.recent_events, name="recent_events"))
        object.__setattr__(self, "memories", _text_tuple(self.memories, name="memories"))
        object.__setattr__(
            self, "recent_thoughts", _text_tuple(self.recent_thoughts, name="recent_thoughts")
        )
        object.__setattr__(self, "affect", _affect_snapshot(self.affect))
        object.__setattr__(self, "mode", str(self.mode))
        object.__setattr__(self, "now", _finite(self.now, name="now"))
        object.__setattr__(
            self,
            "recent_action_rate",
            max(0.0, _finite(self.recent_action_rate, name="recent_action_rate")),
        )
        if self.seconds_since_last_thought is not None:
            object.__setattr__(
                self,
                "seconds_since_last_thought",
                max(
                    0.0,
                    _finite(self.seconds_since_last_thought, name="seconds_since_last_thought"),
                ),
            )

    def affect_of(self, dimension: str, default: float = 0.0) -> float:
        """Read one affect dimension, falling back to ``default`` when absent."""
        return _unit(self.affect.get(dimension, default), name=f"affect.{dimension}")

    def remembers(self, needle: str) -> str | None:
        """Return the first stored memory containing ``needle``, case-insensitively.

        This is how a generator references the past without inventing it: it can
        only name something the agent actually recorded. A generator that bypasses
        this and writes a memory from thin air is a bug, and
        ``tests/test_thoughts.py`` checks the shipped generators against an empty
        memory set to prove they do not.
        """
        lowered = needle.lower()
        for memory in self.memories:
            if lowered in memory.lower():
                return memory
        return None


def _coerce(enum_type: type[Enum], value: Any, *, name: str) -> Any:
    """Coerce ``value`` to a member of ``enum_type``, rejecting unknown names."""
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError:
            pass
    allowed = ", ".join(member.value for member in enum_type)
    raise ThoughtError(f"{name} must be one of: {allowed} (got {value!r})")


def _optional_text(value: Any) -> str | None:
    """Normalise an optional string, treating blanks as absent."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _text_tuple(values: Any, *, name: str) -> tuple[str, ...]:
    """Normalise an iterable of strings, dropping blanks."""
    if isinstance(values, str):
        raise ThoughtError(f"{name} must be an iterable of strings, not a single string")
    cleaned: list[str] = []
    for item in values:
        text = _optional_text(item)
        if text is not None:
            cleaned.append(text)
    return tuple(cleaned)


def _affect_snapshot(values: Any) -> dict[str, float]:
    """Normalise an affect mapping into plain finite floats in 0..1."""
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ThoughtError(f"affect_snapshot must be a mapping, got {type(values).__name__}")
    snapshot: dict[str, float] = {}
    for key, value in values.items():
        snapshot[str(key)] = _unit(value, name=f"affect_snapshot[{key}]")
    return snapshot
