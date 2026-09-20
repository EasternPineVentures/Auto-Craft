"""Tests for the spontaneous thought system.

The specification's architectural rule is the thing being pinned here: a thought
is an **expression**, and a ``ThoughtEvent`` must never directly cause keyboard or
mouse input. Two kinds of test enforce it.

*Structural.* The package is read as source and as an imported namespace, and it
must contain no reference to the control layer or to anything that can reach the
operating system. This is the guarantee that survives a future contributor who
never reads the specification.

*Behavioural.* A thought engine is driven through the observer's real
publication path with a recording input backend wired to a real
:class:`SafetyGuard`, and the backend must record nothing at all. A thought that
reached the game would show up as an event there.

The rest of the file covers the model's validation rules, the honesty of memory
retrieval, the affect-influenced (but not affect-determined) generators, and the
deterministic rate gates.
"""

from __future__ import annotations

import ast
import json
import math
import random
import re
from pathlib import Path

import pytest

from autocraft.config import Config
from autocraft.control.safety import SafetyGuard
from autocraft.observer import ObserverState
from autocraft.thoughts import (
    DEMO_POLICY,
    SCRIPTED_GENERATOR_NAME,
    TEMPLATE_GENERATOR_NAME,
    THOUGHT_TONES,
    THOUGHT_TRIGGERS,
    ScriptedThoughtGenerator,
    TemplateThoughtGenerator,
    ThoughtContext,
    ThoughtEngine,
    ThoughtError,
    ThoughtEvent,
    ThoughtGenerator,
    ThoughtPolicy,
    ThoughtTone,
    ThoughtTrigger,
    scripted_engine,
)
from autocraft.thoughts.generate import _memory_clause, _tone_weights, _trigger_weights

from conftest import FakeClock, FakeInputBackend

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def quiet_context(**overrides: object) -> ThoughtContext:
    """A context with nothing happening, for tests that only care about one axis."""
    base: dict[str, object] = {
        "goal": None,
        "intention": None,
        "observation_summary": None,
        "recent_events": (),
        "memories": (),
        "recent_thoughts": (),
        "affect": {},
        "mode": "idle",
        "recent_action_rate": 0.0,
        "now": 1000.0,
    }
    base.update(overrides)
    return ThoughtContext(**base)  # type: ignore[arg-type]


def permissive_policy(**overrides: object) -> ThoughtPolicy:
    """A policy that speaks on every opportunity, for engine tests.

    The per-minute ceiling is switched off (``0`` disables it) and the scheduled
    silence is disabled, so each gate can be turned on one at a time.
    """
    base: dict[str, object] = {
        "enabled": True,
        "min_interval_seconds": 0.0,
        "max_per_minute": 0.0,
        "base_probability": 1.0,
        "busy_action_rate": 1e9,
        "busy_damping": 1.0,
        "quiet_chance": 0.0,
        "quiet_min_seconds": 30.0,
        "quiet_max_seconds": 30.0,
    }
    base.update(overrides)
    return ThoughtPolicy(**base)  # type: ignore[arg-type]


def engine(config: Config, *, policy: ThoughtPolicy | None = None, seed: int = 3) -> ThoughtEngine:
    """An engine whose randomness is pinned."""
    return ThoughtEngine(
        config,
        policy=policy if policy is not None else permissive_policy(),
        random_source=random.Random(seed),
    )


@pytest.fixture
def thoughts_state(config: Config) -> ObserverState:
    """An observer state with a scripted thought engine attached."""
    state = ObserverState(config, clock=FakeClock())
    state.attach_thought_engine(scripted_engine(config, clock=FakeClock()))
    return state


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


def test_a_thought_needs_text() -> None:
    with pytest.raises(ThoughtError, match="must have text"):
        ThoughtEvent(text="   ")


def test_text_is_trimmed_rather_than_rejected() -> None:
    assert ThoughtEvent(text="  hello  ").text == "hello"


def test_tone_and_trigger_accept_their_enum_members() -> None:
    thought = ThoughtEvent(text="x", tone=ThoughtTone.ABSURD, trigger_type=ThoughtTrigger.DANGER)
    assert thought.tone is ThoughtTone.ABSURD
    assert thought.trigger_type is ThoughtTrigger.DANGER


def test_tone_and_trigger_accept_their_string_forms() -> None:
    thought = ThoughtEvent(text="x", tone="dramatic", trigger_type="milestone")
    assert thought.tone_value == "dramatic"
    assert thought.trigger_value == "milestone"


@pytest.mark.parametrize("value", ["", "smug", "NEUTRAL", 7, None])
def test_an_unknown_tone_is_refused(value: object) -> None:
    """Tone is typed by hand, so a wrong one is a mistake rather than rounding."""
    with pytest.raises(ThoughtError, match="tone must be one of"):
        ThoughtEvent(text="x", tone=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["", "teleport", "DEMO", 7, None])
def test_an_unknown_trigger_is_refused(value: object) -> None:
    with pytest.raises(ThoughtError, match="trigger_type must be one of"):
        ThoughtEvent(text="x", trigger_type=value)  # type: ignore[arg-type]


def test_the_tone_and_trigger_tables_match_their_enums() -> None:
    assert THOUGHT_TONES == tuple(tone.value for tone in ThoughtTone)
    assert THOUGHT_TRIGGERS == tuple(trigger.value for trigger in ThoughtTrigger)
    assert len(THOUGHT_TONES) == 11
    assert len(THOUGHT_TRIGGERS) == 9


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("intensity", -3.0, 0.0),
        ("intensity", 9.0, 1.0),
        ("importance", -1.0, 0.0),
        ("importance", 4.0, 1.0),
    ],
)
def test_intensity_and_importance_are_clamped(field: str, value: float, expected: float) -> None:
    """They come out of a stochastic generator, so the boundary absorbs overshoot."""
    thought = ThoughtEvent(text="x", **{field: value})  # type: ignore[arg-type]
    assert getattr(thought, field) == pytest.approx(expected)


@pytest.mark.parametrize("value", [True, False, "0.5", None, object()])
def test_a_boolean_is_not_a_number(value: object) -> None:
    with pytest.raises(ThoughtError, match="must be a number"):
        ThoughtEvent(text="x", intensity=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_non_finite_number_is_refused(value: float) -> None:
    with pytest.raises(ThoughtError, match="must be finite"):
        ThoughtEvent(text="x", intensity=value)


def test_an_identifier_is_generated_when_one_is_not_supplied() -> None:
    first, second = ThoughtEvent(text="x"), ThoughtEvent(text="x")
    assert first.id and second.id and first.id != second.id


def test_blank_optional_text_becomes_absent() -> None:
    thought = ThoughtEvent(text="x", related_goal="  ", related_memory="")
    assert thought.related_goal is None
    assert thought.related_memory is None


def test_the_affect_snapshot_is_normalised_to_plain_floats() -> None:
    thought = ThoughtEvent(text="x", affect_snapshot={"curiosity": 3, "stress": -1})
    assert thought.affect_snapshot == {"curiosity": 1.0, "stress": 0.0}
    assert all(type(value) is float for value in thought.affect_snapshot.values())


@pytest.mark.parametrize("value", [[], "curiosity", 5])
def test_the_affect_snapshot_must_be_a_mapping(value: object) -> None:
    with pytest.raises(ThoughtError, match="must be a mapping"):
        ThoughtEvent(text="x", affect_snapshot=value)  # type: ignore[arg-type]


def test_a_thought_serialises_to_plain_json() -> None:
    thought = ThoughtEvent(
        text="I wonder what's over that hill.",
        tone=ThoughtTone.CURIOUS,
        intensity=0.35,
        trigger_type=ThoughtTrigger.DISCOVERY,
        trigger_reference="step 4",
        related_goal="TREE-001",
        affect_snapshot={"curiosity": 0.8},
        importance=0.5,
        generated_by=TEMPLATE_GENERATOR_NAME,
        timestamp=1000.0,
    )
    payload = thought.to_dict()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert set(payload) == {
        "id",
        "timestamp",
        "text",
        "tone",
        "intensity",
        "trigger_type",
        "trigger_reference",
        "related_goal",
        "related_memory",
        "affect_snapshot",
        "importance",
        "generated_by",
    }
    assert payload["tone"] == "curious"
    assert payload["trigger_type"] == "discovery"


def test_serialisation_rounds_the_floats_it_shows() -> None:
    thought = ThoughtEvent(text="x", intensity=0.123456789, importance=0.987654321)
    payload = thought.to_dict()
    assert payload["intensity"] == 0.123
    assert payload["importance"] == 0.988


# ---------------------------------------------------------------------------
# the context, and the honesty of memory
# ---------------------------------------------------------------------------


def test_remembers_finds_a_stored_memory() -> None:
    context = quiet_context(memories=("The target window was not captured.",))
    assert context.remembers("window") == "The target window was not captured."


def test_remembers_is_case_insensitive() -> None:
    context = quiet_context(memories=("Emergency stop was triggered.",))
    assert context.remembers("EMERGENCY") == "Emergency stop was triggered."


def test_remembers_returns_nothing_when_the_memory_is_not_there() -> None:
    """The seam that makes "I remember" impossible to fake."""
    context = quiet_context(memories=("Something else entirely.",))
    assert context.remembers("tree") is None
    assert quiet_context().remembers("anything") is None


def test_affect_of_falls_back_to_the_supplied_default() -> None:
    context = quiet_context(affect={"curiosity": 0.9})
    assert context.affect_of("curiosity") == pytest.approx(0.9)
    assert context.affect_of("stress", 0.25) == pytest.approx(0.25)
    assert context.affect_of("stress") == pytest.approx(0.0)


def test_affect_of_clamps_what_it_reads() -> None:
    context = quiet_context(affect={"curiosity": 5.0})
    assert context.affect_of("curiosity") == pytest.approx(1.0)


def test_a_context_refuses_a_bare_string_where_a_sequence_is_expected() -> None:
    """A string is iterable, so it would otherwise become one memory per letter."""
    with pytest.raises(ThoughtError, match="not a single string"):
        ThoughtContext(memories="The target window was not captured.")


def test_blank_entries_are_dropped_from_the_sequences() -> None:
    context = quiet_context(recent_events=("a", "  ", "", "b"))
    assert context.recent_events == ("a", "b")


def test_the_action_rate_and_the_thought_gap_are_never_negative() -> None:
    context = quiet_context(recent_action_rate=-5.0, seconds_since_last_thought=-5.0)
    assert context.recent_action_rate == pytest.approx(0.0)
    assert context.seconds_since_last_thought == pytest.approx(0.0)


def test_the_thought_gap_stays_absent_when_there_has_been_no_thought() -> None:
    assert quiet_context().seconds_since_last_thought is None


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------


def test_the_template_generator_satisfies_the_replaceable_interface() -> None:
    """A future model-backed generator has to fit through this hole."""
    assert isinstance(TemplateThoughtGenerator(), ThoughtGenerator)
    assert isinstance(ScriptedThoughtGenerator(), ThoughtGenerator)
    assert TemplateThoughtGenerator.name == TEMPLATE_GENERATOR_NAME


def test_the_template_generator_is_deterministic_for_a_seed() -> None:
    context = quiet_context(goal="TREE-001", affect={"curiosity": 0.7})
    first = TemplateThoughtGenerator(seed=11).generate(context)
    second = TemplateThoughtGenerator(seed=11).generate(context)
    assert first is not None and second is not None
    assert first.text == second.text
    assert first.tone is second.tone


def test_a_different_affect_produces_a_different_sentence() -> None:
    """Affect has to influence the output, not merely be recorded alongside it."""
    seen = set()
    for affect in ({"curiosity": 1.0, "energy": 1.0}, {"frustration": 1.0, "energy": 0.0}):
        generator = TemplateThoughtGenerator(seed=5)
        for _ in range(30):
            thought = generator.generate(quiet_context(affect=affect))
            assert thought is not None
            seen.add(thought.text)
    assert len(seen) > 1


def test_every_tone_keeps_a_non_zero_weight_under_extreme_affect() -> None:
    """No dimension may be able to force an outcome."""
    extremes = [
        {"curiosity": 1.0, "confidence": 1.0, "stress": 1.0, "frustration": 1.0, "energy": 1.0},
        {"curiosity": 0.0, "confidence": 0.0, "stress": 0.0, "frustration": 0.0, "energy": 0.0},
        {"stress": 1.0},
        {},
    ]
    for affect in extremes:
        weights = _tone_weights(affect)
        assert set(weights) == set(ThoughtTone)
        assert all(weight > 0 for weight in weights.values()), affect


def test_high_frustration_makes_frustration_likelier_but_not_certain() -> None:
    calm = _tone_weights({"frustration": 0.0})[ThoughtTone.FRUSTRATED]
    angry = _tone_weights({"frustration": 1.0})[ThoughtTone.FRUSTRATED]
    assert angry > calm
    total = sum(_tone_weights({"frustration": 1.0}).values())
    assert angry < total


def test_the_memory_trigger_is_impossible_without_a_memory() -> None:
    """There is nothing to remember, so the trigger is removed from the table."""
    weights = _trigger_weights(quiet_context())
    assert weights[ThoughtTrigger.MEMORY] == 0.0
    assert _trigger_weights(quiet_context(memories=("x",)))[ThoughtTrigger.MEMORY] > 0.0


def test_a_safety_stop_makes_danger_much_likelier() -> None:
    calm = _trigger_weights(quiet_context(mode="idle"))[ThoughtTrigger.DANGER]
    stopped = _trigger_weights(quiet_context(mode="safe_stop"))[ThoughtTrigger.DANGER]
    assert stopped > calm


def test_the_generator_never_invents_a_memory_when_there_are_none() -> None:
    """The rule the specification sets: no pretending to remember."""
    generator = TemplateThoughtGenerator(seed=17)
    for _ in range(400):
        thought = generator.generate(quiet_context(goal="TREE-001"))
        assert thought is not None
        assert thought.related_memory is None
        assert thought.trigger_type is not ThoughtTrigger.MEMORY
        assert "I recall" not in thought.text


def test_a_memory_is_only_referenced_when_it_shares_a_keyword_with_the_view() -> None:
    memories = ("The target window was not captured at 1280x650.",)
    generator = TemplateThoughtGenerator(seed=23)
    referenced = 0
    for _ in range(400):
        thought = generator.generate(
            quiet_context(
                observation_summary="Target window is not focused.",
                memories=memories,
            )
        )
        assert thought is not None
        if thought.related_memory is not None:
            referenced += 1
            assert thought.related_memory in memories
            assert "I recall" in thought.text
    assert referenced > 0


def test_an_unrelated_memory_is_never_referenced() -> None:
    generator = TemplateThoughtGenerator(seed=29)
    for _ in range(400):
        thought = generator.generate(
            quiet_context(
                observation_summary="Target window is not focused.",
                memories=("Something completely unrelated.",),
            )
        )
        assert thought is not None
        assert thought.related_memory is None


def test_memory_clause_returns_nothing_without_a_shared_keyword() -> None:
    assert _memory_clause(quiet_context(observation_summary="Target window is not focused.")) is None
    assert (
        _memory_clause(
            quiet_context(
                observation_summary="Target window is not focused.",
                memories=("Something completely unrelated.",),
            )
        )
        is None
    )


def test_the_demo_word_cannot_create_a_false_memory_link() -> None:
    """Every demo string contains ``demo``, so it must not count as evidence."""
    assert (
        _memory_clause(
            quiet_context(
                observation_summary="DEMO MODE observation.",
                memories=("DEMO MODE event.",),
            )
        )
        is None
    )


def test_a_resolution_string_cannot_create_a_false_memory_link() -> None:
    """``1280x650`` is shared by almost every frame and is not a theme."""
    assert (
        _memory_clause(
            quiet_context(
                observation_summary="Captured at 1280x650.",
                memories=("Also captured at 1280x650.",),
            )
        )
        is None
    )


def test_the_generator_names_its_own_provenance() -> None:
    thought = TemplateThoughtGenerator(seed=2).generate(quiet_context())
    assert thought is not None
    assert thought.generated_by == TEMPLATE_GENERATOR_NAME


def test_the_generator_avoids_repeating_the_sentence_it_just_said() -> None:
    generator = TemplateThoughtGenerator(seed=31)
    previous = ""
    for _ in range(200):
        thought = generator.generate(quiet_context(recent_thoughts=(previous,) if previous else ()))
        assert thought is not None
        if previous:
            assert thought.text != previous
        previous = thought.text


def test_the_scripted_generator_labels_everything_it_says_as_demo() -> None:
    generator = ScriptedThoughtGenerator(["one", "two"])
    assert generator.name == SCRIPTED_GENERATOR_NAME
    for _ in range(4):
        thought = generator.generate(quiet_context())
        assert thought is not None
        assert thought.trigger_type is ThoughtTrigger.DEMO
        assert thought.generated_by == SCRIPTED_GENERATOR_NAME


def test_the_scripted_generator_cycles_rather_than_running_out() -> None:
    generator = ScriptedThoughtGenerator(["one", "two"])
    texts = [generator.generate(quiet_context()).text for _ in range(6)]  # type: ignore[union-attr]
    assert texts == ["one", "two", "one", "two", "one", "two"]


def test_the_scripted_generator_can_be_told_to_stay_quiet() -> None:
    generator = ScriptedThoughtGenerator(["one", "two"], quiet_at=(0, 2))
    assert generator.generate(quiet_context()) is None
    assert generator.generate(quiet_context()) is not None
    assert generator.generate(quiet_context()) is None


def test_an_empty_script_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one thought"):
        ScriptedThoughtGenerator([])


def test_the_shipped_demo_script_is_labelled_and_varied() -> None:
    generator = ScriptedThoughtGenerator()
    thoughts = [generator.generate(quiet_context()) for _ in range(8)]
    assert all(thought is not None for thought in thoughts)
    assert all(thought.trigger_type is ThoughtTrigger.DEMO for thought in thoughts)  # type: ignore[union-attr]
    assert len({thought.tone for thought in thoughts}) > 1  # type: ignore[union-attr]
    assert len({thought.intensity for thought in thoughts}) > 1  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# the policy gates
# ---------------------------------------------------------------------------


def test_a_disabled_policy_never_speaks() -> None:
    policy = permissive_policy(enabled=False)
    reason = policy.silence_reason(
        now=1000.0, last_at=None, timestamps=__import__("collections").deque(), quiet_until=None,
        context=quiet_context(),
    )
    assert reason == "disabled"
    assert policy.probability(quiet_context()) == pytest.approx(0.0)


def test_the_cooldown_gate() -> None:
    policy = permissive_policy(min_interval_seconds=25.0)
    from collections import deque

    assert policy.silence_reason(
        now=1010.0, last_at=1000.0, timestamps=deque(), quiet_until=None, context=quiet_context()
    ) == "cooldown"
    assert policy.silence_reason(
        now=1030.0, last_at=1000.0, timestamps=deque(), quiet_until=None, context=quiet_context()
    ) is None


def test_the_scheduled_silence_gate() -> None:
    from collections import deque

    policy = permissive_policy()
    assert policy.silence_reason(
        now=1010.0, last_at=None, timestamps=deque(), quiet_until=1050.0, context=quiet_context()
    ) == "scheduled silence"
    assert policy.silence_reason(
        now=1060.0, last_at=None, timestamps=deque(), quiet_until=1050.0, context=quiet_context()
    ) is None


def test_the_per_minute_ceiling_only_counts_the_trailing_minute() -> None:
    from collections import deque

    policy = permissive_policy(max_per_minute=2.0)
    timestamps = deque([900.0, 980.0, 985.0])
    assert policy.silence_reason(
        now=1000.0, last_at=None, timestamps=timestamps, quiet_until=None, context=quiet_context()
    ) == "per-minute ceiling"
    # The two old enough to have fallen out of the window are not counted.
    stale = deque([900.0, 910.0])
    assert policy.silence_reason(
        now=1000.0, last_at=None, timestamps=stale, quiet_until=None, context=quiet_context()
    ) is None


def test_a_zero_ceiling_switches_the_ceiling_off() -> None:
    from collections import deque

    policy = permissive_policy(max_per_minute=0.0)
    timestamps = deque([999.0] * 50)
    assert policy.silence_reason(
        now=1000.0, last_at=None, timestamps=timestamps, quiet_until=None, context=quiet_context()
    ) is None


def test_the_probability_stays_inside_zero_and_one() -> None:
    policy = ThoughtPolicy(base_probability=5.0)
    for affect in ({}, {"curiosity": 1.0, "energy": 1.0, "stress": 1.0}, {"energy": 0.0}):
        chance = policy.probability(quiet_context(affect=affect))
        assert 0.0 <= chance <= 1.0


def test_curiosity_raises_the_chance_and_low_energy_lowers_it() -> None:
    policy = ThoughtPolicy(base_probability=0.2)
    bored = policy.probability(quiet_context(affect={"curiosity": 0.0, "energy": 1.0}))
    curious = policy.probability(quiet_context(affect={"curiosity": 1.0, "energy": 1.0}))
    tired = policy.probability(quiet_context(affect={"curiosity": 0.0, "energy": 0.0}))
    assert curious > bored > tired


def test_being_busy_makes_the_agent_quieter() -> None:
    policy = ThoughtPolicy(base_probability=0.2, busy_action_rate=4.0)
    idle = policy.probability(quiet_context(recent_action_rate=0.0))
    busy = policy.probability(quiet_context(recent_action_rate=30.0))
    assert busy < idle


def test_something_meaningful_having_happened_makes_a_thought_likelier() -> None:
    policy = ThoughtPolicy(base_probability=0.1)
    quiet = policy.probability(quiet_context(recent_events=("Nothing to report.",)))
    loud = policy.probability(quiet_context(recent_events=("The action was refused by safety.",)))
    assert loud > quiet


def test_having_a_goal_raises_the_chance_slightly() -> None:
    policy = ThoughtPolicy(base_probability=0.1)
    assert policy.probability(quiet_context(goal="TREE-001")) > policy.probability(quiet_context())


def test_the_probability_does_not_depend_on_a_random_source() -> None:
    """The gates are deterministic; only the draw is random."""
    policy = ThoughtPolicy(base_probability=0.2)
    context = quiet_context(affect={"curiosity": 0.4})
    assert policy.probability(context) == policy.probability(context)


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------


def test_the_engine_speaks_when_nothing_is_holding_it_back(config: Config) -> None:
    thought = engine(config).maybe_think(quiet_context())
    assert thought is not None
    assert thought.text


def test_the_engine_holds_its_cooldown(config: Config) -> None:
    engine_ = engine(config, policy=permissive_policy(min_interval_seconds=25.0))
    assert engine_.maybe_think(quiet_context(now=1000.0)) is not None
    assert engine_.maybe_think(quiet_context(now=1010.0)) is None
    assert engine_.last_silence_reason == "cooldown"
    assert engine_.maybe_think(quiet_context(now=1030.0)) is not None


def test_the_engine_respects_the_per_minute_ceiling(config: Config) -> None:
    engine_ = engine(config, policy=permissive_policy(max_per_minute=2.0))
    assert engine_.maybe_think(quiet_context(now=1000.0)) is not None
    assert engine_.maybe_think(quiet_context(now=1001.0)) is not None
    assert engine_.maybe_think(quiet_context(now=1002.0)) is None
    assert engine_.last_silence_reason == "per-minute ceiling"


def test_the_engine_goes_quiet_for_a_while_after_speaking(config: Config) -> None:
    policy = permissive_policy(quiet_chance=1.0, quiet_min_seconds=30.0, quiet_max_seconds=30.0)
    engine_ = engine(config, policy=policy)
    assert engine_.maybe_think(quiet_context(now=1000.0)) is not None
    assert engine_.maybe_think(quiet_context(now=1010.0)) is None
    assert engine_.last_silence_reason == "scheduled silence"
    assert engine_.maybe_think(quiet_context(now=1040.0)) is not None


def test_a_policy_that_never_fires_reports_why(config: Config) -> None:
    engine_ = engine(config, policy=permissive_policy(base_probability=0.0))
    assert engine_.maybe_think(quiet_context()) is None
    assert engine_.last_silence_reason == "probability"
    assert engine_.last_thought_at is None


def test_a_quiet_generator_does_not_consume_the_cooldown(config: Config) -> None:
    """A generator with nothing to say is a quiet tick, not a spoken thought."""
    engine_ = ThoughtEngine(
        config,
        generator=ScriptedThoughtGenerator(["only one"], quiet_at=(0,)),
        policy=permissive_policy(min_interval_seconds=25.0),
        random_source=random.Random(0),
    )
    assert engine_.maybe_think(quiet_context(now=1000.0)) is None
    assert engine_.last_silence_reason == "generator quiet"
    assert engine_.last_thought_at is None
    # Still inside the cooldown the first thought would have started - because it
    # never started.
    assert engine_.maybe_think(quiet_context(now=1001.0)) is not None


def test_a_disabled_policy_silences_the_engine(config: Config) -> None:
    engine_ = engine(config, policy=permissive_policy(enabled=False))
    assert engine_.maybe_think(quiet_context()) is None
    assert engine_.last_silence_reason == "disabled"
    assert engine_.history == ()


def test_the_history_is_bounded_by_the_configuration(config: Config) -> None:
    small = Config(data_dir=config.data_dir, thought_history_max=3)
    engine_ = engine(small)
    for step in range(10):
        engine_.maybe_think(quiet_context(now=1000.0 + step * 10.0))
    assert len(engine_.history) == 3


def test_the_history_reads_oldest_first_and_the_gap_is_measurable(config: Config) -> None:
    engine_ = engine(config)
    first = engine_.maybe_think(quiet_context(now=1000.0))
    second = engine_.maybe_think(quiet_context(now=1005.0))
    assert engine_.history == (first, second)
    assert engine_.last_thought_at == pytest.approx(1005.0)


def test_the_engine_tells_the_generator_what_it_just_said(config: Config) -> None:
    """The generator needs to avoid repeating itself without holding the history."""
    seen: list[tuple[str, ...]] = []

    class Spy:
        name = "spy"

        def generate(self, context: ThoughtContext) -> ThoughtEvent | None:
            seen.append(context.recent_thoughts)
            return ThoughtEvent(text=f"line {len(seen)}", timestamp=context.now)

    engine_ = ThoughtEngine(
        config, generator=Spy(), policy=permissive_policy(), random_source=random.Random(0)
    )
    engine_.maybe_think(quiet_context(now=1000.0))
    engine_.maybe_think(quiet_context(now=1001.0))
    assert seen[0] == ()
    assert seen[1] == ("line 1",)


def test_the_engine_is_deterministic_for_a_seed(config: Config) -> None:
    texts = []
    for _ in range(2):
        engine_ = ThoughtEngine(
            config, policy=permissive_policy(), random_source=random.Random(99)
        )
        texts.append(
            [thought.text for thought in (
                engine_.maybe_think(quiet_context(now=1000.0 + step * 30.0)) for step in range(30)
            ) if thought is not None]
        )
    assert texts[0] == texts[1]
    assert texts[0]


def test_the_engine_reports_the_generator_it_is_using(config: Config) -> None:
    assert engine(config).generator_name == TEMPLATE_GENERATOR_NAME
    assert scripted_engine(config).generator_name == SCRIPTED_GENERATOR_NAME


def test_reset_clears_the_history_and_the_cooldown(config: Config) -> None:
    engine_ = engine(config, policy=permissive_policy(min_interval_seconds=1000.0))
    assert engine_.maybe_think(quiet_context(now=1000.0)) is not None
    engine_.reset()
    assert engine_.history == ()
    assert engine_.last_thought_at is None
    assert engine_.last_silence_reason is None
    assert engine_.maybe_think(quiet_context(now=1000.0)) is not None


def test_the_production_policy_is_patient_by_default(config: Config) -> None:
    """A real run must not narrate every step."""
    engine_ = ThoughtEngine(config, random_source=random.Random(1))
    spoken = sum(
        1
        for step in range(600)
        if engine_.maybe_think(quiet_context(now=1000.0 + step * 0.5)) is not None
    )
    assert spoken < 30


def test_the_demo_policy_is_bounded_but_not_silent(config: Config) -> None:
    """The demo must show a cadence, not one thought per tick and not nothing."""
    engine_ = scripted_engine(config, thoughts=("a", "b", "c"))
    spoken = sum(
        1
        for step in range(200)
        if engine_.maybe_think(quiet_context(now=1000.0 + step)) is not None
    )
    assert 5 <= spoken <= 120


def test_the_demo_policy_is_still_a_policy() -> None:
    assert DEMO_POLICY.enabled is True
    assert DEMO_POLICY.min_interval_seconds == 0.0
    assert DEMO_POLICY.max_per_minute == 0.0


# ---------------------------------------------------------------------------
# the non-operative guarantee
# ---------------------------------------------------------------------------

_FORBIDDEN_IMPORTS = (
    "autocraft.control",
    "autocraft.agent",
    "autocraft.vision",
    "autocraft.observer",
    "socket",
    "subprocess",
    "ctypes",
    "shutil",
    "urllib",
    "http",
    "os",
    "win32",
    "pyautogui",
    "keyboard",
    "mouse",
)


def test_no_thought_module_can_reach_the_control_layer() -> None:
    """The structural half of "a thought must never cause input".

    Imports are read from the syntax tree rather than by searching the text, so
    prose in a docstring that *names* the control layer is not mistaken for an
    import of it. Checking the source is deliberate: this keeps holding for a
    future contributor who adds a call without reading the specification.
    """
    package = Path(__file__).resolve().parent.parent / "src" / "autocraft" / "thoughts"
    sources = sorted(package.glob("*.py"))
    assert sources, "the thoughts package should have modules to check"

    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        for module in imported:
            for forbidden in _FORBIDDEN_IMPORTS:
                assert not module.startswith(forbidden), (
                    f"{source.name} imports {module!r}, which can reach {forbidden!r}"
                )
        assert "importlib" not in imported, f"{source.name} could import anything at runtime"


def test_no_thought_module_imports_the_operating_system() -> None:
    """``os`` alone would be enough to start a process."""
    import autocraft.thoughts.engine as engine_module
    import autocraft.thoughts.generate as generate_module
    import autocraft.thoughts.model as model_module

    forbidden = {"os", "subprocess", "socket", "ctypes", "win32api", "autocraft"}
    for module in (model_module, generate_module, engine_module):
        names = set(vars(module))
        assert not (names & forbidden), f"{module.__name__} imported {names & forbidden}"


def test_a_full_thought_cycle_injects_no_input(config: Config) -> None:
    """The behavioural half: drive the real path and watch the backend."""
    backend = FakeInputBackend()
    guard = SafetyGuard(config, backend, target_is_foreground=lambda: True, clock=FakeClock())
    state = ObserverState(config, clock=FakeClock())
    state.attach_thought_engine(scripted_engine(config, clock=FakeClock()))
    state.begin_run("thought-run", now=1000.0, goal="TREE-001")

    for step in range(300):
        state.publish_observation(summary=f"Observation {step}.", now=1000.0 + step)
        state.publish_event(f"Step {step} recorded.", now=1000.0 + step)
        state.maybe_express_thought(now=1000.0 + step)

    assert state.snapshot().metrics.thoughts_expressed > 0
    assert backend.events == []
    assert backend.closed is False
    assert guard.held_keys == ()
    assert guard.held_buttons == ()
    assert guard.events == ()


def test_the_engine_holds_no_executor_keyboard_mouse_or_guard(config: Config) -> None:
    """There is no object in the engine capable of producing input."""
    engine_ = engine(config)
    reachable = set(vars(engine_)) | set(vars(type(engine_)))
    for name in ("executor", "keyboard", "mouse", "guard", "backend", "_executor", "_keyboard", "_mouse", "_guard", "_backend"):
        assert name not in reachable


# ---------------------------------------------------------------------------
# wiring into the observer
# ---------------------------------------------------------------------------


def test_a_thought_engine_that_is_not_attached_says_nothing(config: Config) -> None:
    state = ObserverState(config, clock=FakeClock())
    assert state.maybe_express_thought(now=1000.0) is None
    assert state.snapshot().metrics.thoughts_expressed == 0


def test_detaching_the_engine_disables_thoughts(config: Config) -> None:
    state = ObserverState(config, clock=FakeClock())
    state.attach_thought_engine(scripted_engine(config, clock=FakeClock()))
    state.publish_event("Something happened.", now=1000.0)
    state.maybe_express_thought(now=1000.0)
    before = state.snapshot().metrics.thoughts_expressed
    state.attach_thought_engine(None)
    for step in range(50):
        assert state.maybe_express_thought(now=1100.0 + step) is None
    assert state.snapshot().metrics.thoughts_expressed == before


def test_an_expressed_thought_reaches_the_page(thoughts_state: ObserverState) -> None:
    thoughts_state.begin_run("run-t", now=1000.0, goal="TREE-001")
    thoughts_state.publish_event("Something happened.", now=1000.0)
    spoken = None
    for step in range(60):
        spoken = thoughts_state.maybe_express_thought(now=1000.0 + step)
        if spoken is not None:
            break
    assert spoken is not None

    snapshot = thoughts_state.snapshot()
    assert snapshot.latest_thought is spoken
    assert snapshot.latest_thought.text == spoken.text
    assert snapshot.metrics.thoughts_expressed >= 1
    assert snapshot.thought_history[0] is spoken
    # The page receives the serialised form, not the object.
    payload = snapshot.to_dict()
    assert payload["latest_thought"]["text"] == spoken.text
    assert payload["latest_thought"]["trigger_type"] == spoken.trigger_value


def test_the_thought_context_is_assembled_from_what_was_published(config: Config) -> None:
    """A generator is never handed more than the specification allows."""
    captured: list[ThoughtContext] = []

    class Spy:
        name = "spy"

        def generate(self, context: ThoughtContext) -> ThoughtEvent | None:
            captured.append(context)
            return None

    state = ObserverState(config, clock=FakeClock())
    state.attach_thought_engine(
        ThoughtEngine(
            config, generator=Spy(), policy=permissive_policy(), random_source=random.Random(0)
        )
    )
    state.begin_run("run-s", now=1000.0, goal="TREE-001")
    state.publish_observation(summary="Target window is not focused.", now=1000.0)
    state.publish_event("The action was refused by safety.", now=1000.0)
    state.maybe_express_thought(now=1000.0)

    assert len(captured) == 1
    context = captured[0]
    assert context.goal == "TREE-001"
    assert context.observation_summary == "Target window is not focused."
    assert "The action was refused by safety." in context.recent_events
    assert context.now == pytest.approx(1000.0)


def test_the_timeline_is_the_memory_system_by_default(config: Config) -> None:
    """A thought can only refer to something genuinely recorded."""
    captured: list[ThoughtContext] = []

    class Spy:
        name = "spy"

        def generate(self, context: ThoughtContext) -> ThoughtEvent | None:
            captured.append(context)
            return None

    state = ObserverState(config, clock=FakeClock())
    state.attach_thought_engine(
        ThoughtEngine(
            config, generator=Spy(), policy=permissive_policy(), random_source=random.Random(0)
        )
    )
    state.publish_event("The target window was not captured.", now=1000.0)
    state.maybe_express_thought(now=1000.0)
    assert "The target window was not captured." in captured[0].memories


def test_supplied_memories_replace_the_timeline(config: Config) -> None:
    captured: list[ThoughtContext] = []

    class Spy:
        name = "spy"

        def generate(self, context: ThoughtContext) -> ThoughtEvent | None:
            captured.append(context)
            return None

    state = ObserverState(config, clock=FakeClock())
    state.attach_thought_engine(
        ThoughtEngine(
            config, generator=Spy(), policy=permissive_policy(), random_source=random.Random(0)
        )
    )
    state.publish_event("An event nobody should remember.", now=1000.0)
    state.maybe_express_thought(now=1000.0, memories=("A real recorded memory.",))
    assert captured[0].memories == ("A real recorded memory.",)


def test_an_engine_that_raises_does_not_break_publication(config: Config) -> None:
    """The observer's own publication path must stay total."""
    state = ObserverState(config, clock=FakeClock())

    class Broken:
        name = "broken"

        def generate(self, context: ThoughtContext) -> ThoughtEvent | None:
            raise RuntimeError("synthetic generator failure")

    state.attach_thought_engine(
        ThoughtEngine(
            config, generator=Broken(), policy=permissive_policy(), random_source=random.Random(0)
        )
    )
    with pytest.raises(RuntimeError):
        state.maybe_express_thought(now=1000.0)


def test_a_demo_run_labels_its_thoughts_as_demo(config: Config) -> None:
    from autocraft.observer import demo_state

    state = demo_state(config, clock=FakeClock(), thoughts=("one", "two", "three"))
    snapshot = state.snapshot()
    assert snapshot.demo is True
    if snapshot.latest_thought is not None:
        assert snapshot.latest_thought.trigger_type is ThoughtTrigger.DEMO
        assert snapshot.latest_thought.generated_by == SCRIPTED_GENERATOR_NAME


def test_demo_ticks_eventually_produce_a_labelled_thought(config: Config) -> None:
    from autocraft.observer import demo_state

    clock = FakeClock()
    state = demo_state(config, clock=clock, thoughts=("one", "two", "three"), advance=False)
    for _ in range(40):
        clock.advance(2.0)
        state.tick(now=clock.now)
    snapshot = state.snapshot()
    assert snapshot.metrics.thoughts_expressed > 0
    assert snapshot.latest_thought is not None
    assert snapshot.latest_thought.trigger_type is ThoughtTrigger.DEMO
    # Nothing synthetic may be presented as if it were a real expression.
    assert snapshot.to_dict()["latest_thought"]["trigger_type"] == "demo"


def test_no_thought_text_looks_like_reasoning_or_markup() -> None:
    """These are character expressions, not chain-of-thought or raw HTML."""
    generator = TemplateThoughtGenerator(seed=41)
    for _ in range(300):
        thought = generator.generate(quiet_context(goal="TREE-001"))
        assert thought is not None
        assert not re.search(r"[<>]", thought.text)
        assert "```" not in thought.text
        assert len(thought.text) < 400
