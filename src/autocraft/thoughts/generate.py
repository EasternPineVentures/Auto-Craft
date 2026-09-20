"""Thought generators: how a sentence gets written.

Two are shipped, and the interface exists so a third can replace both later.

:class:`TemplateThoughtGenerator` composes text from fragments rather than
reading from a list of canned lines. The specification asks for exactly that -
"do not hard-code the personality entirely from canned quotes" - so the output
is a function of the trigger, the tone the affect state currently favours, and
whatever real context is available. The same trigger with a different affect
produces a different sentence, which is what makes the character read as a
character rather than as a random quote generator.

:class:`ScriptedThoughtGenerator` replays a fixed sequence. It is for tests and
for demo mode, it is labelled as such in ``generated_by``, and it is the only
generator that produces the ``demo`` trigger.

Neither generator performs input, network access, process execution or file
access. That is not a convention here, it is the point: a thought is an
expression, and ``tests/test_thoughts.py`` asserts that this package imports no
module capable of doing anything else.

The interface is replaceable by design:

    ThoughtGenerator.generate(context) -> ThoughtEvent | None

Returning ``None`` means "nothing worth saying", which the engine treats as a
quiet tick rather than as an error.
"""

from __future__ import annotations

import random
import re
from typing import Protocol, runtime_checkable

from .model import (
    ThoughtContext,
    ThoughtEvent,
    ThoughtTone,
    ThoughtTrigger,
)

__all__ = [
    "ScriptedThoughtGenerator",
    "TemplateThoughtGenerator",
    "ThoughtGenerator",
]

TEMPLATE_GENERATOR_NAME = "template-v0"
SCRIPTED_GENERATOR_NAME = "scripted-demo"


@runtime_checkable
class ThoughtGenerator(Protocol):
    """Turns a :class:`ThoughtContext` into a thought, or into silence.

    A generator must be a pure function of its context apart from its own random
    source. It must not perform input, network, process or filesystem access, and
    it must not be handed anything beyond the context it is given.
    """

    #: Short provenance string, copied onto every thought the generator produces.
    name: str

    def generate(self, context: ThoughtContext) -> ThoughtEvent | None:
        """Return a thought, or ``None`` to stay quiet."""
        ...


# ---------------------------------------------------------------------------
# fragment tables
# ---------------------------------------------------------------------------

#: Opening fragment, by what prompted the thought. Two or three each, so a
#: repeated trigger does not read as a repeated line.
_OPENERS: dict[ThoughtTrigger, tuple[str, ...]] = {
    ThoughtTrigger.DISCOVERY: (
        "Something here is new.",
        "That is not a thing I have seen before.",
        "New ground.",
    ),
    ThoughtTrigger.SUCCESS: (
        "That worked.",
        "That worked on the first attempt, which is suspicious.",
        "Progress.",
    ),
    ThoughtTrigger.FAILURE: (
        "That did not work.",
        "Same problem, same result.",
        "Failure, again.",
    ),
    ThoughtTrigger.DANGER: (
        "This is bad.",
        "Everything is moving at once.",
        "NOPE. Too many things are moving.",
    ),
    ThoughtTrigger.MEMORY: (
        "I recognise this.",
        "This is familiar in a way I do not like.",
        "I have been here before.",
    ),
    ThoughtTrigger.IDLE: (
        "Nothing is happening.",
        "Still nothing.",
        "Time is passing and I am watching it.",
    ),
    ThoughtTrigger.MILESTONE: (
        "This is a first.",
        "I am going to remember this.",
        "Something just changed permanently.",
    ),
    ThoughtTrigger.RANDOM_REFLECTION: (
        "Unrelated observation:",
        "I have been thinking.",
        "A thought arrived unprompted.",
    ),
    ThoughtTrigger.DEMO: (
        "I wonder what is over that hill.",
        "Apparently the answer was another hill.",
        "I am beginning to understand why humans invented roads.",
    ),
}

#: Closing fragment, by tone. These carry the personality, and the affect model
#: chooses between them probabilistically rather than by rule.
_CLOSERS: dict[ThoughtTone, tuple[str, ...]] = {
    ThoughtTone.NEUTRAL: (
        "I will keep going.",
        "Nothing to be done about it.",
        "Noted.",
    ),
    ThoughtTone.HUMOROUS: (
        "I am sure that will be funny later.",
        "Excellent. Perfect. Exactly what I wanted.",
        "The sheep seem unimpressed, and honestly, fair.",
    ),
    ThoughtTone.CURIOUS: (
        "I want to know what is past it.",
        "There is probably something worth seeing over there.",
        "I should look closer before deciding anything.",
    ),
    ThoughtTone.HOPEFUL: (
        "The next attempt might go better.",
        "It is probably fine.",
        "This will work eventually. Statistically.",
    ),
    ThoughtTone.EXCITED: (
        "This changes everything.",
        "This is genuinely good and I am pleased about it.",
        "I did not expect to be this happy about rocks.",
    ),
    ThoughtTone.SAD: (
        "I still have not found what I was looking for.",
        "The sun is going down again.",
        "I had hoped for more than this.",
    ),
    ThoughtTone.FRUSTRATED: (
        "I am declaring war on geology.",
        "I am taking this personally.",
        "I have developed strong opinions about this terrain.",
    ),
    ThoughtTone.ANXIOUS: (
        "I would prefer to be somewhere else.",
        "I do not like the look of that.",
        "I am counting the exits.",
    ),
    ThoughtTone.DRAMATIC: (
        "I will remember this day.",
        "Nothing will be the same after this.",
        "This is the moment it all goes wrong.",
    ),
    ThoughtTone.ABSURD: (
        "Perhaps the sheep knows the way home. Its confidence is suspicious.",
        "I am a rectangle-making machine and I have made my peace with it.",
        "Somewhere, a tree is filing a complaint.",
    ),
    ThoughtTone.REFLECTIVE: (
        "I have spent most of my existence converting trees into rectangles.",
        "I wonder what I was doing a hundred frames ago.",
        "The world is very large and I am very small.",
    ),
}

#: Keywords that mark the agent's own event log as containing a given kind of
#: experience. This is a rule-based reading of real recorded events, not
#: inference about the game world.
_TRIGGER_KEYWORDS: dict[ThoughtTrigger, tuple[str, ...]] = {
    ThoughtTrigger.SUCCESS: ("accepted", "visual change detected", "completed"),
    ThoughtTrigger.FAILURE: ("refused", "failed", "blocked", "not captured"),
    ThoughtTrigger.DANGER: ("emergency", "stop", "refused"),
    ThoughtTrigger.MILESTONE: ("run complete", "first", "finished"),
    ThoughtTrigger.MEMORY: ("memory", "recall", "remember"),
}

#: Words too common to be evidence of a thematic link between two strings.
#: ``demo`` is here because every demo observation and every demo event contains
#: it, so it would otherwise make unrelated demo entries look related.
_STOPWORDS = frozenset(
    {
        "about", "after", "again", "against", "another", "because", "before", "being",
        "below", "between", "could", "doing", "during", "every", "frame", "frames",
        "still", "there", "these", "thing", "things", "those", "through", "under",
        "where", "which", "while", "would", "captured", "detected", "action", "demo",
    }
)


#: A token is only evidence of a theme if it contains a run of letters. Resolutions
#: and other numeric identifiers do not, however they are punctuated.
_LETTER_RUN = re.compile(r"[a-z]{2,}")


def _tone_weights(affect: dict[str, float]) -> dict[ThoughtTone, float]:
    """Derive a tone distribution from the affect state.

    The specification's rule is that affect *influences* generation probability
    rather than determining it, so every tone keeps a non-zero weight and no
    dimension can force an outcome. High frustration makes ``frustrated`` likely,
    never certain.
    """
    curiosity = affect.get("curiosity", 0.5)
    confidence = affect.get("confidence", 0.5)
    stress = affect.get("stress", 0.1)
    frustration = affect.get("frustration", 0.0)
    energy = affect.get("energy", 1.0)
    return {
        ThoughtTone.NEUTRAL: 0.6 + 0.8 * energy,
        ThoughtTone.HUMOROUS: 0.4 + 1.6 * confidence,
        ThoughtTone.CURIOUS: 0.3 + 2.0 * curiosity,
        ThoughtTone.HOPEFUL: 0.3 + 1.5 * confidence,
        ThoughtTone.EXCITED: 0.2 + 1.8 * max(curiosity, confidence),
        ThoughtTone.SAD: 0.2 + 1.6 * frustration + 0.8 * (1.0 - energy),
        ThoughtTone.FRUSTRATED: 0.15 + 2.2 * frustration,
        ThoughtTone.ANXIOUS: 0.15 + 2.0 * stress,
        ThoughtTone.DRAMATIC: 0.2 + 1.2 * max(stress, frustration),
        ThoughtTone.ABSURD: 0.3 + 0.8 * curiosity + 0.5 * (1.0 - confidence),
        ThoughtTone.REFLECTIVE: 0.4 + 1.2 * (1.0 - energy) + 0.5 * (1.0 - confidence),
    }


def _trigger_weights(context: ThoughtContext) -> dict[ThoughtTrigger, float]:
    """Derive a trigger distribution from the agent's own recent event log."""
    weights = {
        ThoughtTrigger.DISCOVERY: 0.5,
        ThoughtTrigger.SUCCESS: 0.6,
        ThoughtTrigger.FAILURE: 0.6,
        ThoughtTrigger.DANGER: 0.3,
        ThoughtTrigger.MEMORY: 0.3,
        ThoughtTrigger.IDLE: 0.4,
        ThoughtTrigger.MILESTONE: 0.2,
        ThoughtTrigger.RANDOM_REFLECTION: 0.9,
        ThoughtTrigger.DEMO: 0.0,
    }
    recent = " ".join(context.recent_events[-6:]).lower()
    for trigger, keywords in _TRIGGER_KEYWORDS.items():
        if any(keyword in recent for keyword in keywords):
            weights[trigger] += 2.5
    if not context.recent_events:
        weights[ThoughtTrigger.IDLE] += 2.0
    if context.mode in {"safe_stop", "error"}:
        weights[ThoughtTrigger.DANGER] += 4.0
    if context.mode in {"idle", "observing"}:
        weights[ThoughtTrigger.IDLE] += 1.5
        weights[ThoughtTrigger.RANDOM_REFLECTION] += 1.0
    # A memory can only be talked about when there is a real one to talk about.
    if not context.memories:
        weights[ThoughtTrigger.MEMORY] = 0.0
    return weights


def _weighted_pick(rng: random.Random, weights: dict[ThoughtTone, float] | dict[ThoughtTrigger, float]):
    """Pick one key with probability proportional to its weight, ignoring zeros."""
    keys = [key for key, weight in weights.items() if weight > 0]
    if not keys:  # pragma: no cover - the tables above always leave something
        return next(iter(weights))
    return rng.choices(keys, weights=[weights[key] for key in keys], k=1)[0]


def _keywords(text: str) -> set[str]:
    """Significant lowercase words in ``text``, for thematic matching.

    Digit-only tokens are dropped, and so are tokens without a run of at least
    two letters. Resolution strings such as ``1280x650`` are shared by almost
    every frame and would otherwise make unrelated memories look thematically
    linked, which is exactly the kind of false confidence the retrieval rule
    exists to prevent. ``str.isdigit()`` alone does not catch them, because of
    the ``x``.
    """
    cleaned = "".join(char if char.isalnum() else " " for char in text.lower())
    return {
        word
        for word in cleaned.split()
        if len(word) > 4
        and word not in _STOPWORDS
        and not word.isdigit()
        and _LETTER_RUN.search(word) is not None
    }


def _memory_clause(context: ThoughtContext) -> tuple[str, str] | None:
    """Find a stored memory that shares a keyword with the current view.

    Returns ``(clause, memory)`` or ``None``. This is real retrieval: the match
    has to exist in ``context.memories`` and share a significant word with what
    the agent currently sees. With no memories there is no match and the
    generator says nothing about the past, which is the rule the specification
    sets - AutoCraft must never pretend to remember something it did not record.
    """
    if not context.memories:
        return None
    target = _keywords(context.observation_summary or "")
    if not target:
        return None
    for memory in context.memories:
        if _keywords(memory) & target:
            return f"I recall: {memory}", memory
    return None


def _goal_clause(context: ThoughtContext) -> tuple[str, str] | None:
    """Name the current goal, when there is one."""
    if not context.goal:
        return None
    return f"My goal is still {context.goal}.", context.goal


def _intensity(affect: dict[str, float], trigger: ThoughtTrigger, rng: random.Random) -> float:
    """Scale how strongly a thought reads, from affect and trigger severity."""
    arousal = max(affect.get("stress", 0.1), affect.get("frustration", 0.0))
    severity = {
        ThoughtTrigger.DANGER: 0.35,
        ThoughtTrigger.MILESTONE: 0.3,
        ThoughtTrigger.SUCCESS: 0.2,
        ThoughtTrigger.DISCOVERY: 0.2,
        ThoughtTrigger.FAILURE: 0.15,
        ThoughtTrigger.MEMORY: 0.1,
        ThoughtTrigger.RANDOM_REFLECTION: 0.05,
        ThoughtTrigger.IDLE: 0.0,
        ThoughtTrigger.DEMO: 0.1,
    }[trigger]
    return min(1.0, max(0.0, 0.25 + 0.45 * arousal + severity + rng.uniform(-0.05, 0.05)))


def _importance(trigger: ThoughtTrigger, rng: random.Random) -> float:
    """How much this is worth keeping, which is what a later stage would use."""
    base = {
        ThoughtTrigger.MILESTONE: 0.85,
        ThoughtTrigger.DANGER: 0.7,
        ThoughtTrigger.SUCCESS: 0.55,
        ThoughtTrigger.DISCOVERY: 0.5,
        ThoughtTrigger.FAILURE: 0.45,
        ThoughtTrigger.MEMORY: 0.4,
        ThoughtTrigger.RANDOM_REFLECTION: 0.25,
        ThoughtTrigger.IDLE: 0.15,
        ThoughtTrigger.DEMO: 0.1,
    }[trigger]
    return min(1.0, max(0.0, base + rng.uniform(-0.05, 0.05)))


class TemplateThoughtGenerator:
    """Composes a thought from fragments, the affect state and real context.

    No language model, no network, no filesystem. Every sentence is assembled
    from :data:`_OPENERS`, :data:`_CLOSERS` and clauses built from context the
    caller supplied, which means the same call with the same seed always produces
    the same sentence and different affect produces different sentences.
    """

    name = TEMPLATE_GENERATOR_NAME

    def __init__(self, *, seed: int | None = None, random_source: random.Random | None = None) -> None:
        self._rng = random_source if random_source is not None else random.Random(seed)

    def generate(self, context: ThoughtContext) -> ThoughtEvent | None:
        """Compose one thought, or return ``None`` when there is nothing to say."""
        affect = dict(context.affect)
        trigger = _weighted_pick(self._rng, _trigger_weights(context))
        tone = _weighted_pick(self._rng, _tone_weights(affect))

        parts = [self._rng.choice(_OPENERS[trigger])]
        related_goal: str | None = None
        related_memory: str | None = None

        if trigger is ThoughtTrigger.MEMORY:
            clause = _memory_clause(context)
            if clause is not None:
                parts.append(clause[0])
                related_memory = clause[1]
            else:
                # Chosen as a memory thought, but nothing real to refer to. Fall
                # back rather than inventing a past.
                trigger = ThoughtTrigger.RANDOM_REFLECTION
        elif trigger is ThoughtTrigger.RANDOM_REFLECTION:
            goal_clause = _goal_clause(context)
            if goal_clause is not None and self._rng.random() < 0.5:
                parts.append(goal_clause[0])
                related_goal = goal_clause[1]
            memory_clause = _memory_clause(context)
            if memory_clause is not None and self._rng.random() < 0.4:
                parts.append(memory_clause[0])
                related_memory = memory_clause[1]
                trigger = ThoughtTrigger.MEMORY

        parts.append(self._rng.choice(_CLOSERS[tone]))
        text = " ".join(part for part in parts if part).strip()
        if not text:  # pragma: no cover - the fragment tables are non-empty
            return None
        if text in context.recent_thoughts:
            # Avoid saying the same sentence twice in a row. Pick another closer.
            alternatives = [closer for closer in _CLOSERS[tone] if closer not in text]
            if alternatives:
                text = f"{parts[0]} {self._rng.choice(alternatives)}".strip()
        return ThoughtEvent(
            text=text,
            tone=tone,
            intensity=_intensity(affect, trigger, self._rng),
            trigger_type=trigger,
            trigger_reference=context.mode,
            related_goal=related_goal,
            related_memory=related_memory,
            affect_snapshot=affect,
            importance=_importance(trigger, self._rng),
            generated_by=self.name,
            timestamp=context.now,
        )


class ScriptedThoughtGenerator:
    """Replays a fixed sequence of thoughts. For tests and demo mode only.

    Every thought it produces is stamped with the ``demo`` trigger and this
    generator's name, so nothing it writes can be mistaken for a real expression.
    It can also be told to return ``None`` at chosen positions, which is how the
    tests exercise a quiet period without waiting for one.
    """

    name = SCRIPTED_GENERATOR_NAME

    def __init__(self, thoughts: list[str] | tuple[str, ...] | None = None, *, quiet_at: tuple[int, ...] = ()) -> None:
        if thoughts is None:
            self._script = _DEMO_SCRIPT
        else:
            self._script = tuple(
                (text, _DEMO_TONES[index % len(_DEMO_TONES)], 0.4)
                for index, text in enumerate(thoughts)
            )
        if not self._script:
            raise ValueError("a scripted generator needs at least one thought")
        self._quiet_at = frozenset(quiet_at)
        self._index = 0

    def generate(self, context: ThoughtContext) -> ThoughtEvent | None:
        """Return the next scripted thought, or ``None`` at a scripted quiet tick."""
        position = self._index
        self._index += 1
        if position in self._quiet_at:
            return None
        text, tone, intensity = self._script[position % len(self._script)]
        return ThoughtEvent(
            text=text,
            tone=tone,
            intensity=intensity,
            trigger_type=ThoughtTrigger.DEMO,
            trigger_reference="demo",
            related_goal=context.goal,
            affect_snapshot=dict(context.affect),
            importance=0.1,
            generated_by=self.name,
            timestamp=context.now,
        )


#: The specification's three example lines, plus enough continuation that a demo
#: can run for a while without repeating itself immediately. Tone and intensity
#: are per-line so the stream shows variation instead of one flat register.
_DEMO_SCRIPT: tuple[tuple[str, ThoughtTone, float], ...] = (
    ("I wonder what's over that hill.", ThoughtTone.CURIOUS, 0.35),
    ("Apparently the answer was another hill.", ThoughtTone.HUMOROUS, 0.45),
    ("I am beginning to understand why humans invented roads.", ThoughtTone.REFLECTIVE, 0.50),
    ("I have walked past this exact rock three times now.", ThoughtTone.ABSURD, 0.60),
    ("My map is a drawing of my hopes, not of the terrain.", ThoughtTone.DRAMATIC, 0.55),
    ("I could stop and think, or I could keep walking and think less.", ThoughtTone.HUMOROUS, 0.40),
    ("There is a tree. There is always a tree. That is not a complaint.", ThoughtTone.FRUSTRATED, 0.65),
    ("I am going to call this direction north and commit to it.", ThoughtTone.HOPEFUL, 0.45),
)

#: Kept as the plain-text view of :data:`_DEMO_SCRIPT` for callers that only want
#: the lines.
_DEMO_THOUGHTS: tuple[str, ...] = tuple(text for text, _, _ in _DEMO_SCRIPT)

_DEMO_TONES: tuple[ThoughtTone, ...] = (
    ThoughtTone.CURIOUS,
    ThoughtTone.HUMOROUS,
    ThoughtTone.REFLECTIVE,
    ThoughtTone.ABSURD,
)
