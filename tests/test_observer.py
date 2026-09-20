"""Tests for the local observer page.

Three things are being pinned here, in order of importance.

1. **The observer cannot control the agent.** The HTTP surface answers ``GET``
   and refuses every other verb, it binds loopback by default, it rejects a
   request whose ``Host`` header names somewhere else, and polling it leaves the
   published state byte-identical. There is no route that writes anything.
2. **The display contract is honest.** A frame that has never been published is
   reported as absent rather than stale, a run with no safety guard reports that
   control is unavailable rather than "input enabled", and the safety verdict
   shown on the page is the one ``SafetyGuard.authorize`` would actually give.
3. **The published shape is plain JSON.** ``/api/snapshot`` is serialised with
   ``allow_nan=False``, so a value that is not representable would be an error
   rather than a silently malformed page.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from autocraft.config import Config
from autocraft.control.safety import SafetyGuard
from autocraft.observer import (
    AFFECT_BASELINE,
    AFFECT_DIMENSIONS,
    AgentEvent,
    AgentMode,
    AffectState,
    Belief,
    EmergencyStopState,
    EventKind,
    FrameFreshness,
    FrameInfo,
    LoopPublisher,
    ObserverError,
    ObserverServer,
    ObserverSnapshot,
    ObserverState,
    RunMetrics,
    SafetyStatus,
    demo_frame,
    demo_state,
    downscale_image,
    encode_jpeg,
    frame_freshness,
    input_permitted,
    publish_safety_from,
    resolve_bind_host,
    with_events,
    with_thoughts,
)
from autocraft.observer.state import _rate
from autocraft.thoughts import ThoughtEvent, ThoughtTone, ThoughtTrigger
from autocraft.vision.frame import Frame, ScreenRegion

from conftest import FakeClock, FakeInputBackend

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def solid_frame(width: int = 40, height: int = 20, *, value: int = 60) -> Frame:
    """A flat RGB frame of a known size."""
    image = np.full((height, width, 3), value, dtype=np.uint8)
    return Frame(
        image=image,
        timestamp=1000.0,
        region=ScreenRegion(0, 0, width, height),
        source="test",
    )


def striped_frame(width: int = 40, height: int = 20) -> Frame:
    """A frame with a hard left/right contrast, so averaging is observable."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, width // 2 :] = 200
    return Frame(image=image, timestamp=1000.0, region=ScreenRegion(0, 0, width, height))


def http_get(port: int, path: str, *, host_header: str | None = None) -> tuple[int, dict[str, str], bytes]:
    """Issue a raw GET so the ``Host`` header can be chosen."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        headers = {} if host_header is None else {"Host": host_header}
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def http_method(port: int, method: str, path: str = "/api/snapshot") -> tuple[int, str | None]:
    """Issue a non-GET request and return the status and the ``Allow`` header."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        response.read()
        return response.status, response.getheader("Allow")
    finally:
        connection.close()


@dataclass(frozen=True)
class _Record:
    """The subset of a step record the bridge reads."""

    action: Mapping[str, Any]
    result: Mapping[str, Any]
    notes: str = ""


@pytest.fixture
def state(config: Config) -> ObserverState:
    """A bare, non-demo observer state."""
    return ObserverState(config, clock=FakeClock())


@pytest.fixture
def server(state: ObserverState):
    """A running observer server on an ephemeral loopback port."""
    running = ObserverServer(state, host="127.0.0.1", port=0)
    running.start()
    try:
        yield running
    finally:
        running.stop()


# ---------------------------------------------------------------------------
# affect
# ---------------------------------------------------------------------------


def test_affect_defaults_are_the_documented_resting_state() -> None:
    affect = AffectState()
    assert affect.as_dict() == {
        "curiosity": 0.60,
        "confidence": 0.50,
        "stress": 0.10,
        "frustration": 0.00,
        "energy": 1.00,
    }
    assert affect.as_tuple() == tuple(affect.as_dict()[name] for name in AFFECT_DIMENSIONS)
    assert AFFECT_BASELINE == affect


def test_affect_clamps_out_of_range_values() -> None:
    affect = AffectState(curiosity=4.0, stress=-3.0, energy=0.5)
    assert affect.curiosity == 1.0
    assert affect.stress == 0.0
    assert affect.energy == 0.5


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "0.5", None])
def test_affect_refuses_values_that_are_not_finite_numbers(bad: object) -> None:
    with pytest.raises(ObserverError):
        AffectState(curiosity=bad)  # type: ignore[arg-type]


def test_affect_refuses_booleans_because_true_is_not_one() -> None:
    with pytest.raises(ObserverError):
        AffectState(stress=True)


def test_affect_blend_moves_toward_the_target_and_clamps_the_weight() -> None:
    start = AffectState(curiosity=0.0, confidence=1.0)
    target = AffectState(curiosity=1.0, confidence=0.0)

    halfway = start.blend(target, 0.5)
    assert halfway.curiosity == pytest.approx(0.5)
    assert halfway.confidence == pytest.approx(0.5)

    assert start.blend(target, 2.0) == target
    assert start.blend(target, -1.0) == start


def test_affect_blend_refuses_anything_that_is_not_an_affect_state() -> None:
    with pytest.raises(ObserverError):
        AffectState().blend({"curiosity": 1.0}, 0.5)  # type: ignore[arg-type]


def test_affect_adjust_applies_deltas_and_rejects_unknown_dimensions() -> None:
    adjusted = AffectState(curiosity=0.5).adjust(curiosity=0.2, frustration=0.1)
    assert adjusted.curiosity == pytest.approx(0.7)
    assert adjusted.frustration == pytest.approx(0.1)

    with pytest.raises(ObserverError, match="unknown affect dimension"):
        AffectState().adjust(hunger=0.1)


# ---------------------------------------------------------------------------
# freshness
# ---------------------------------------------------------------------------


def test_freshness_distinguishes_absent_from_old() -> None:
    assert frame_freshness(None, live_seconds=1.0, stale_seconds=5.0) is FrameFreshness.NONE


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (0.0, FrameFreshness.LIVE),
        (1.0, FrameFreshness.LIVE),
        (-0.5, FrameFreshness.LIVE),
        (1.001, FrameFreshness.RECENT),
        (5.0, FrameFreshness.RECENT),
        (5.001, FrameFreshness.STALE),
        (3600.0, FrameFreshness.STALE),
        (float("inf"), FrameFreshness.STALE),
    ],
)
def test_freshness_boundaries(age: float, expected: FrameFreshness) -> None:
    assert frame_freshness(age, live_seconds=1.0, stale_seconds=5.0) is expected


# ---------------------------------------------------------------------------
# the safety verdict must agree with the guard
# ---------------------------------------------------------------------------


def _guard(config: Config, *, foreground: bool, stopped: bool = False) -> SafetyGuard:
    backend = FakeInputBackend()
    guard = SafetyGuard(
        config,
        backend,
        target_is_foreground=lambda: foreground,
        clock=FakeClock(),
        wall_clock=FakeClock(),
    )
    if stopped:
        guard.trigger_emergency_stop("test")
    return guard


@pytest.mark.parametrize("foreground", [True, False])
@pytest.mark.parametrize("stopped", [True, False])
def test_input_permitted_agrees_with_the_guard(config: Config, *, foreground: bool, stopped: bool) -> None:
    """The page's verdict and the guard's must not drift apart."""
    guard = _guard(config, foreground=foreground, stopped=stopped)
    allowed = guard.authorize("test action").allowed

    published = input_permitted(
        window_found=True,
        window_foreground=foreground,
        emergency_stop=(
            EmergencyStopState.TRIGGERED if stopped else EmergencyStopState.READY
        ),
        require_foreground=config.require_foreground,
    )
    assert published == allowed


def test_input_permitted_is_stricter_than_the_guard_when_no_window_is_found(config: Config) -> None:
    """A missing target window can only ever refuse, never permit."""
    guard = _guard(config, foreground=True)
    assert guard.authorize("test action").allowed is True
    assert (
        input_permitted(
            window_found=False,
            window_foreground=True,
            emergency_stop=EmergencyStopState.READY,
            require_foreground=True,
        )
        is False
    )


def test_input_permitted_is_false_when_there_is_no_safety_layer_at_all() -> None:
    """A read-only session must not advertise input it cannot perform."""
    assert (
        input_permitted(
            window_found=True,
            window_foreground=True,
            emergency_stop=EmergencyStopState.READY,
            require_foreground=True,
            control_available=False,
        )
        is False
    )


# ---------------------------------------------------------------------------
# the display contract
# ---------------------------------------------------------------------------


def test_snapshot_carries_every_field_the_specification_asks_for() -> None:
    """The published shape is the specification's contract, spelled out."""
    payload = ObserverSnapshot().to_dict()
    assert set(payload) == {
        "run_id",
        "timestamp",
        "demo",
        "mode",
        "mode_since",
        "mode_note",
        "current_goal",
        "current_intention",
        "confidence",
        "last_action",
        "action_result",
        "observation_summary",
        "beliefs",
        "detected_entities",
        "affect_state",
        "latest_thought",
        "thought_history",
        "recent_events",
        "short_term_memory",
        "metrics",
        "safety",
        "frame",
        "look",
    }
    # The four flat read-outs the page shows at the top level, mapped onto the
    # nested groups that actually own them.
    assert payload["mode"] == "IDLE"  # the agent's state
    assert payload["metrics"]["capture_fps"] == 0.0
    assert payload["metrics"]["loop_rate"] == 0.0
    assert payload["safety"]["window_found"] is False
    assert payload["safety"]["window_foreground"] is False
    assert payload["safety"]["input_enabled"] is False
    assert payload["safety"]["emergency_stop"] == "ready"
    assert payload["frame"]["reference"] is None
    assert payload["affect_state"] == AFFECT_BASELINE.as_dict()


def test_snapshot_serialises_to_plain_json() -> None:
    """Strict JSON is the real assertion: nothing is a custom type or NaN."""
    snapshot = ObserverSnapshot(
        run_id="run-1",
        timestamp=1000.0,
        mode=AgentMode.ACTING,
        beliefs=(Belief(label="a tree", confidence=0.4, source="vision"),),
        detected_entities=(Belief(label="trunk", confidence=0.2),),
        recent_events=(AgentEvent(timestamp=1.0, message="hello", kind=EventKind.INFO),),
        metrics=RunMetrics(run_id="run-1", frames_observed=3),
    )
    payload = snapshot.to_dict()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_snapshot_coerces_strings_and_clamps_confidence() -> None:
    snapshot = ObserverSnapshot(mode="OBSERVING", confidence=3.0, beliefs=[Belief(label="x")])
    assert snapshot.mode is AgentMode.OBSERVING
    assert snapshot.confidence == 1.0
    assert isinstance(snapshot.beliefs, tuple)


def test_snapshot_refuses_a_mode_it_does_not_have() -> None:
    with pytest.raises(ValueError):
        ObserverSnapshot(mode="TELEPORTING")


def test_belief_refuses_an_empty_label() -> None:
    with pytest.raises(ObserverError):
        Belief(label="   ")


def test_with_events_orders_the_log_forward_and_the_memory_panel_newest_first() -> None:
    events = tuple(AgentEvent(timestamp=float(index), message=str(index)) for index in range(6))
    snapshot = with_events(ObserverSnapshot(), events, short_term_limit=3)

    assert [event.message for event in snapshot.recent_events] == ["0", "1", "2", "3", "4", "5"]
    assert [event.message for event in snapshot.short_term_memory] == ["5", "4", "3"]


def test_with_events_handles_an_empty_timeline() -> None:
    snapshot = with_events(ObserverSnapshot(), (), short_term_limit=3)
    assert snapshot.recent_events == ()
    assert snapshot.short_term_memory == ()


def test_with_thoughts_puts_the_newest_thought_on_top_and_in_the_history() -> None:
    thoughts = tuple(
        ThoughtEvent(text=f"thought {index}", tone=ThoughtTone.NEUTRAL, timestamp=float(index))
        for index in range(3)
    )
    snapshot = with_thoughts(ObserverSnapshot(), thoughts)

    assert snapshot.latest_thought is not None
    assert snapshot.latest_thought.text == "thought 2"
    assert [thought.text for thought in snapshot.thought_history] == [
        "thought 2",
        "thought 1",
        "thought 0",
    ]


def test_with_thoughts_leaves_the_panel_empty_rather_than_inventing_one() -> None:
    snapshot = with_thoughts(ObserverSnapshot(), ())
    assert snapshot.latest_thought is None
    assert snapshot.thought_history == ()


# ---------------------------------------------------------------------------
# frame tiles
# ---------------------------------------------------------------------------


def test_downscale_returns_the_same_array_when_already_narrow_enough() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    assert downscale_image(image, 20) is image
    assert downscale_image(image, 0) is image


def test_downscale_averages_rather_than_sampling() -> None:
    """A hard edge must survive as a hard edge, not as aliasing."""
    image = np.zeros((8, 16, 3), dtype=np.uint8)
    image[:, 8:] = 200
    reduced = downscale_image(image, 8)

    assert reduced.shape == (4, 8, 3)  # both axes are halved by the factor
    assert set(np.unique(reduced[:, :, 0])) <= {0, 200}
    # Both halves are still represented, which stride sampling could have lost.
    assert reduced[0, 0, 0] == 0
    assert reduced[0, -1, 0] == 200


def test_downscale_trims_to_a_divisible_size_without_padding() -> None:
    """A few edge pixels are dropped rather than padded; the means stay exact."""
    image = np.full((10, 10, 3), 128, dtype=np.uint8)
    reduced = downscale_image(image, 4)

    assert reduced.shape == (3, 3, 3)  # factor 3, so 9x9 of the 10x10 survives
    assert np.all(reduced == 128)


def test_downscale_refuses_an_image_that_is_not_three_dimensional() -> None:
    with pytest.raises(ObserverError):
        downscale_image(np.zeros((10, 10), dtype=np.uint8), 5)


def test_encode_jpeg_produces_a_jpeg() -> None:
    data = encode_jpeg(np.zeros((8, 8, 3), dtype=np.uint8))
    assert data.startswith(b"\xff\xd8\xff")  # JPEG SOI marker
    assert data.endswith(b"\xff\xd9")  # JPEG EOI marker


def test_rate_is_zero_until_there_are_two_samples() -> None:
    assert _rate([], 100.0) == 0.0
    assert _rate([99.0], 100.0) == 0.0
    assert _rate([99.0, 99.0], 100.0) == 0.0  # no span, so no rate


def test_rate_is_measured_from_event_timestamps() -> None:
    assert _rate([96.0, 97.0, 98.0, 99.0], 100.0) == pytest.approx(1.0)
    # Samples older than the window are ignored.
    assert _rate([0.0, 98.0, 99.0], 100.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# observer state
# ---------------------------------------------------------------------------


def test_published_frame_is_downscaled_to_the_configured_width(state: ObserverState, config: Config) -> None:
    frame = striped_frame(width=config.observer_frame_max_width * 3)
    state.publish_frame(frame, now=1000.0)

    encoded = state.encoded_frame()
    assert encoded is not None
    assert encoded.content_type == "image/jpeg"
    assert encoded.width <= config.observer_frame_max_width
    assert encoded.source_width == frame.width
    assert encoded.source_height == frame.height
    assert encoded.size > 0

    info = state.snapshot(now=1000.0).frame
    assert info.available is True
    assert info.source_width == frame.width
    assert info.reference == "/frame"
    assert info.signature


def test_frame_absent_is_reported_as_absent_not_stale(state: ObserverState) -> None:
    info = state.snapshot(now=1000.0).frame
    assert info.available is False
    assert info.freshness is FrameFreshness.NONE
    assert info.age_seconds is None
    assert state.encoded_frame() is None


def test_frame_age_and_freshness_follow_the_configured_thresholds(
    state: ObserverState, config: Config
) -> None:
    state.publish_frame(solid_frame(), now=1000.0)

    live = state.snapshot(now=1000.0 + config.observer_frame_live_seconds).frame
    assert live.freshness is FrameFreshness.LIVE

    recent = state.snapshot(now=1000.0 + config.observer_frame_stale_seconds).frame
    assert recent.freshness is FrameFreshness.RECENT
    assert recent.age_seconds == pytest.approx(config.observer_frame_stale_seconds)

    stale = state.snapshot(now=1000.0 + config.observer_frame_stale_seconds + 1.0).frame
    assert stale.freshness is FrameFreshness.STALE


def test_publishing_a_broken_frame_records_an_error_instead_of_raising(state: ObserverState) -> None:
    class NotAnImage:
        image = None
        timestamp = 0.0
        region = None
        source = "broken"

    state.publish_frame(NotAnImage())  # type: ignore[arg-type]

    snapshot = state.snapshot(now=1000.0)
    assert snapshot.frame.available is False
    assert snapshot.metrics.errors == 1
    assert any(event.kind is EventKind.ERROR for event in snapshot.recent_events)


def test_safety_publication_is_idempotent(state: ObserverState) -> None:
    """Publishing the same total repeatedly must not inflate the counter."""
    for _ in range(5):
        state.publish_safety(safety_event_total=3, window_found=True, window_foreground=True)

    assert state.snapshot().metrics.safety_events == 3


def test_safety_event_total_only_grows(state: ObserverState) -> None:
    state.publish_safety(safety_event_total=4)
    state.publish_safety(safety_event_total=2)
    assert state.snapshot().metrics.safety_events == 4


def test_emergency_stop_makes_the_page_report_input_disabled(state: ObserverState) -> None:
    state.publish_safety(
        window_found=True,
        window_foreground=True,
        emergency_stop=EmergencyStopState.TRIGGERED,
    )
    snapshot = state.snapshot()
    assert snapshot.safety.input_enabled is False
    assert snapshot.safety.emergency_stop is EmergencyStopState.TRIGGERED
    assert any(event.kind is EventKind.SAFETY for event in snapshot.recent_events)


def test_a_run_without_a_safety_layer_reports_control_unavailable(state: ObserverState) -> None:
    state.publish_safety(
        window_found=True, window_foreground=True, control_available=False
    )
    safety = state.snapshot().safety
    assert safety.control_available is False
    assert safety.input_enabled is False
    assert safety.to_dict()["control_available"] is False


def test_observation_events_appear_only_when_the_description_changes(state: ObserverState) -> None:
    for _ in range(20):
        state.publish_observation(summary="Target window not found.", now=1000.0)

    assert len(state.events) == 1

    state.publish_observation(summary="Captured a frame.", now=1001.0)
    assert [event.message for event in state.events] == [
        "Target window not found.",
        "Captured a frame.",
    ]


def test_polling_does_not_change_the_published_state(state: ObserverState) -> None:
    """Reads are pure, which is what makes "read-only" checkable."""
    state.publish_frame(solid_frame(), now=1000.0)
    state.publish_safety(window_found=True, window_foreground=True)
    state.publish_event("something happened", now=1000.0)

    first = state.snapshot(now=1000.0).to_dict()
    for _ in range(10):
        state.snapshot(now=1000.0)
    second = state.snapshot(now=1000.0).to_dict()

    assert first == second


def test_action_outcomes_count_exactly_what_the_executor_reported(state: ObserverState) -> None:
    state.publish_action_outcome(attempted=True, executed=True)
    state.publish_action_outcome(attempted=True, executed=False, blocked_reason="not foreground")
    state.publish_action_outcome(attempted=False, executed=False)

    metrics = state.snapshot().metrics
    assert metrics.actions_attempted == 2
    assert metrics.actions_executed == 1
    assert metrics.actions_blocked == 1


def test_the_timeline_is_bounded_by_the_configured_limit(config: Config) -> None:
    state = ObserverState(config, clock=FakeClock())
    for index in range(config.observer_max_events + 50):
        state.publish_event(f"event {index}", now=float(index))

    events = state.events
    assert len(events) == config.observer_max_events
    assert events[-1].message == f"event {config.observer_max_events + 49}"


def test_begin_run_sets_the_identity_and_the_goal(state: ObserverState) -> None:
    state.begin_run("run-42", now=1000.0, goal="Find wood")
    state.publish_goal("Find wood", intention="Look around first")

    snapshot = state.snapshot(now=1002.0)
    assert snapshot.run_id == "run-42"
    assert snapshot.current_goal == "Find wood"
    assert snapshot.current_intention == "Look around first"
    assert snapshot.metrics.runtime_seconds == pytest.approx(2.0)


def test_mode_changes_are_timestamped_and_explained(state: ObserverState) -> None:
    state.publish_mode(AgentMode.OBSERVING, note="looking", now=1000.0)
    state.publish_mode(AgentMode.SAFE_STOP, note="emergency stop", now=1005.0)

    snapshot = state.snapshot(now=1006.0)
    assert snapshot.mode is AgentMode.SAFE_STOP
    assert snapshot.mode_note == "emergency stop"
    assert snapshot.mode_since == pytest.approx(1005.0)


def test_beliefs_and_entities_stay_empty_because_there_is_no_perception(state: ObserverState) -> None:
    snapshot = state.snapshot()
    assert snapshot.beliefs == ()
    assert snapshot.detected_entities == ()
    assert snapshot.to_dict()["beliefs"] == []
    assert snapshot.to_dict()["detected_entities"] == []


def test_thoughts_are_counted_and_surfaced_but_never_acted_on(state: ObserverState) -> None:
    thought = ThoughtEvent(
        text="I wonder what is over that hill.",
        tone=ThoughtTone.CURIOUS,
        trigger_type=ThoughtTrigger.DISCOVERY,
    )
    state.publish_thought(thought)

    snapshot = state.snapshot()
    assert snapshot.metrics.thoughts_expressed == 1
    assert snapshot.latest_thought == thought
    assert snapshot.thought_history == (thought,)


# ---------------------------------------------------------------------------
# demo mode
# ---------------------------------------------------------------------------


def test_demo_state_is_labelled_and_produces_a_frame(config: Config) -> None:
    state = demo_state(config, clock=FakeClock())
    assert state.demo is True

    snapshot = state.snapshot()
    assert snapshot.demo is True
    assert snapshot.frame.available is True
    assert snapshot.frame.source == "demo"
    assert snapshot.frame.freshness is FrameFreshness.LIVE
    assert any("DEMO" in event.message for event in snapshot.recent_events)


def test_demo_ticks_advance_and_are_repeatable(config: Config) -> None:
    clock = FakeClock()
    state = demo_state(config, clock=clock)

    first = state.tick()
    clock.advance(2.0)
    second = state.tick()

    assert state.demo_ticks == 3  # demo_state ticks once when it is built
    assert first.observation_summary != second.observation_summary
    assert second.metrics.frames_observed == 3


def test_demo_frames_differ_over_time_so_the_age_means_something() -> None:
    early = demo_frame(now=0.0)
    later = demo_frame(now=5.0)
    assert early.signature() != later.signature()
    assert early.source == "demo"


def test_demo_state_reports_a_loop_rate(config: Config) -> None:
    """The demo's scripted step is its loop, so a rate is measurable."""
    clock = FakeClock()
    state = demo_state(config, clock=clock)
    for _ in range(4):
        clock.advance(2.0)
        state.tick()

    metrics = state.snapshot().metrics
    assert metrics.loop_rate == pytest.approx(0.5)
    assert metrics.capture_fps == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# the bridge
# ---------------------------------------------------------------------------


def test_publisher_makes_a_run_visible(state: ObserverState) -> None:
    publisher = LoopPublisher(
        state,
        goal="Find wood",
        intention="Look around",
        safety=lambda: {"window_found": True, "window_foreground": True},
        confidence=lambda _record: 0.7,
    )
    publisher.begin("run-7")
    publisher.on_observation(object())
    publisher.on_step(
        _Record(
            action={"kind": "key_press", "description": "press w"},
            result={"attempted": True, "executed": True},
            notes="",
        )
    )
    publisher.finish(None)

    snapshot = state.snapshot()
    assert snapshot.run_id == "run-7"
    assert snapshot.current_goal == "Find wood"
    assert snapshot.current_intention == "Look around"
    assert snapshot.confidence == pytest.approx(0.7)
    assert snapshot.safety.window_found is True
    assert snapshot.safety.input_enabled is True
    assert snapshot.metrics.actions_attempted == 1
    assert snapshot.metrics.actions_executed == 1


def test_publisher_records_a_broken_publication_instead_of_raising() -> None:
    """A dashboard that cannot be updated must never be able to end a run."""

    class BrokenState:
        demo = False

        def __init__(self) -> None:
            self.errors: list[str] = []

        def publish_error(self, message: str) -> None:
            self.errors.append(message)

        def begin_run(self, *_args, **_kwargs):
            raise RuntimeError("synthetic failure")

        def publish_goal(self, *_args, **_kwargs):
            raise RuntimeError("synthetic failure")

        def publish_mode(self, *_args, **_kwargs):
            raise RuntimeError("synthetic failure")

        def publish_event(self, *_args, **_kwargs):
            raise RuntimeError("synthetic failure")

        def publish_observation(self, *_args, **_kwargs):
            raise RuntimeError("synthetic failure")

        def publish_safety(self, *_args, **_kwargs):
            raise RuntimeError("synthetic failure")

    broken = BrokenState()
    publisher = LoopPublisher(broken, goal=None, safety=lambda: {"window_found": True})  # type: ignore[arg-type]
    publisher.begin("run-8")
    publisher.on_observation(object())
    publisher.finish(None)

    assert broken.errors
    assert all("synthetic failure" in message for message in broken.errors)


def test_publisher_survives_a_safety_provider_that_fails(state: ObserverState) -> None:
    def explode() -> dict[str, object]:
        raise RuntimeError("no safety layer here")

    publisher = LoopPublisher(state, goal="Find wood", safety=explode)
    publisher.begin("run-8")

    # The provider failing is not an error: there is simply nothing to publish.
    assert state.snapshot().safety.window_found is False
    assert state.snapshot().metrics.errors == 0


def test_publish_safety_from_reads_the_guard_without_holding_it(config: Config) -> None:
    guard = _guard(config, foreground=True)
    guard.register_key_down("w")
    guard.register_button_down("left")
    guard.record_block("not foreground")

    facts = publish_safety_from(guard)
    assert facts["emergency_stop"] == "ready"
    assert facts["held_inputs"] == ("w", "left")
    assert facts["blocked_streak"] == 1
    assert facts["last_block_reason"] == "not foreground"
    assert facts["safety_event_total"] == len(guard.events)

    guard.trigger_emergency_stop("test")
    assert publish_safety_from(guard)["emergency_stop"] == "triggered"


def test_publish_safety_from_returns_plain_data_not_the_guard() -> None:
    """Once the facts are out, the display layer cannot reach the authority."""

    class Bare:
        held_keys = ("w",)
        held_buttons = ()
        events = ()
        stop_requested = False
        blocked_streak = 0
        last_block_reason = ""

    facts = publish_safety_from(Bare())
    assert facts["held_inputs"] == ("w",)
    assert facts["last_block_reason"] is None
    assert facts["safety_event_total"] == 0
    assert set(facts) == {
        "emergency_stop",
        "held_inputs",
        "blocked_streak",
        "last_block_reason",
        "safety_event_total",
    }


# ---------------------------------------------------------------------------
# the HTTP surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/", "text/html"),
        ("/index.html", "text/html"),
        ("/app.js", "application/javascript"),
        ("/styles.css", "text/css"),
        ("/api/snapshot", "application/json"),
        ("/api/health", "application/json"),
    ],
)
def test_get_routes_serve_the_page(server: ObserverServer, path: str, content_type: str) -> None:
    status, headers, body = http_get(server.port, path)
    assert status == 200
    assert headers["Content-Type"].startswith(content_type)
    assert body


def test_health_reports_the_thresholds_the_page_must_use(server: ObserverServer, config: Config) -> None:
    _status, _headers, body = http_get(server.port, "/api/health")
    payload = json.loads(body)

    assert payload["status"] == "ok"
    assert payload["demo"] is False
    assert payload["poll_ms"] == config.observer_poll_ms
    assert payload["frame_live_seconds"] == config.observer_frame_live_seconds
    assert payload["frame_stale_seconds"] == config.observer_frame_stale_seconds


def test_snapshot_route_serves_the_published_state(server: ObserverServer, state: ObserverState) -> None:
    state.begin_run("run-9", now=1000.0, goal="Find wood")
    state.publish_frame(solid_frame(), now=1000.0)

    _status, _headers, body = http_get(server.port, "/api/snapshot")
    payload = json.loads(body)

    assert payload["run_id"] == "run-9"
    assert payload["current_goal"] == "Find wood"
    assert payload["frame"]["available"] is True
    assert payload["frame"]["reference"] == "/frame"


def test_frame_route_is_404_until_a_frame_exists(server: ObserverServer, state: ObserverState) -> None:
    status, _headers, _body = http_get(server.port, "/frame")
    assert status == 404

    state.publish_frame(solid_frame(), now=1000.0)
    status, headers, body = http_get(server.port, "/frame")
    assert status == 200
    assert headers["Content-Type"] == "image/jpeg"
    assert body.startswith(b"\xff\xd8\xff")


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
def test_every_method_other_than_get_is_refused(server: ObserverServer, method: str) -> None:
    status, allow = http_method(server.port, method)
    assert status == 405
    assert allow == "GET"


def test_a_request_from_another_host_is_refused(server: ObserverServer) -> None:
    status, _headers, _body = http_get(server.port, "/api/snapshot", host_header="evil.example.com")
    assert status == 403

    status, _headers, _body = http_get(server.port, "/api/snapshot", host_header="127.0.0.1")
    assert status == 200


def test_unknown_paths_are_404_and_the_favicon_is_empty(server: ObserverServer) -> None:
    assert http_get(server.port, "/nope")[0] == 404
    assert http_get(server.port, "/../autocraft/config.py")[0] == 404
    assert http_get(server.port, "/favicon.ico")[0] == 204


def test_serving_the_page_cannot_change_the_run(server: ObserverServer, state: ObserverState) -> None:
    """The whole point of the layer: the page is a window, not a control."""
    state.begin_run("run-10", now=1000.0, goal="Find wood")
    state.publish_frame(solid_frame(), now=1000.0)
    before = state.snapshot(now=1000.0).to_dict()

    for path in ("/", "/app.js", "/styles.css", "/api/snapshot", "/api/health", "/frame"):
        for _ in range(3):
            urllib.request.urlopen(f"http://127.0.0.1:{server.port}{path}", timeout=5).read()
    for method in ("POST", "PUT", "PATCH", "DELETE", "OPTIONS"):
        http_method(server.port, method)

    assert state.snapshot(now=1000.0).to_dict() == before


def test_the_page_never_embeds_pixels_in_its_state(server: ObserverServer, state: ObserverState) -> None:
    """Frames travel over /frame, so a poll stays small and cannot leak a frame."""
    state.publish_frame(solid_frame(width=200, height=100), now=1000.0)
    _status, _headers, body = http_get(server.port, "/api/snapshot")

    assert len(body) < 8192
    assert "image" not in json.loads(body)["frame"]


def test_a_demo_server_reports_that_it_is_demo(config: Config) -> None:
    demo = demo_state(config, clock=FakeClock())
    running = ObserverServer(demo, host="127.0.0.1", port=0)
    running.start()
    try:
        _status, _headers, body = http_get(running.port, "/api/health")
        assert json.loads(body)["demo"] is True
        _status, _headers, body = http_get(running.port, "/api/snapshot")
        assert json.loads(body)["demo"] is True
    finally:
        running.stop()


def test_the_web_directory_holds_the_three_assets() -> None:
    from autocraft.observer import default_web_dir

    directory = default_web_dir()
    assert directory.is_dir()
    for name in ("index.html", "app.js", "styles.css"):
        assert (directory / name).is_file()


def test_index_html_does_not_request_an_image_it_does_not_have() -> None:
    """A src-less <img> is deliberate: the browser must not fetch a broken URL."""
    from autocraft.observer import default_web_dir

    markup = (default_web_dir() / "index.html").read_text(encoding="utf-8")
    assert 'id="frame-image"' in markup

    start = markup.index('id="frame-image"')
    tag = markup[markup.rindex("<", 0, start) : markup.index(">", start)]
    assert "src=" not in tag


# ---------------------------------------------------------------------------
# binding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_hosts_are_accepted_by_default(host: str) -> None:
    assert resolve_bind_host(host) == host


def test_none_means_the_documented_default() -> None:
    assert resolve_bind_host(None) == "127.0.0.1"


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "example.com", "", "   "])
def test_a_non_loopback_bind_is_refused_unless_asked_for(host: str) -> None:
    with pytest.raises(ObserverError):
        resolve_bind_host(host)


def test_a_non_loopback_bind_is_allowed_when_it_is_explicit() -> None:
    assert resolve_bind_host("0.0.0.0", allow_remote=True) == "0.0.0.0"


def test_a_remote_bind_skips_the_host_header_check(config: Config) -> None:
    """An explicit remote bind means the caller's own network is the boundary."""
    state = ObserverState(config, clock=FakeClock())
    running = ObserverServer(state, host="0.0.0.0", port=0, allow_remote=True)
    running.start()
    try:
        status, _headers, _body = http_get(running.port, "/api/snapshot", host_header="evil.example.com")
        assert status == 200
    finally:
        running.stop()


def test_a_second_server_cannot_quietly_share_the_port(config: Config) -> None:
    """Two observers on one port would be a page nobody could trust."""
    state = ObserverState(config, clock=FakeClock())
    first = ObserverServer(state, host="127.0.0.1", port=0)
    first.start()
    try:
        with pytest.raises(OSError):
            ObserverServer(state, host="127.0.0.1", port=first.port)
    finally:
        first.stop()


def test_missing_web_assets_are_reported_not_ignored(config: Config, tmp_path: Path) -> None:
    state = ObserverState(config, clock=FakeClock())
    with pytest.raises(ObserverError):
        ObserverServer(state, host="127.0.0.1", port=0, web_dir=tmp_path / "absent")


def test_a_missing_asset_file_answers_500_rather_than_a_silent_blank(config: Config, tmp_path: Path) -> None:
    empty = tmp_path / "web"
    empty.mkdir()
    state = ObserverState(config, clock=FakeClock())
    running = ObserverServer(state, host="127.0.0.1", port=0, web_dir=empty)
    running.start()
    try:
        status, _headers, body = http_get(running.port, "/")
        assert status == 500
        assert b"missing" in body
    finally:
        running.stop()


def test_urllib_cannot_reach_a_route_that_does_not_exist(server: ObserverServer) -> None:
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(f"http://127.0.0.1:{server.port}/api/control", timeout=5)
    assert caught.value.code == 404


# ---------------------------------------------------------------------------
# the server object
# ---------------------------------------------------------------------------


def test_stopping_the_server_is_idempotent(config: Config) -> None:
    state = ObserverState(config, clock=FakeClock())
    running = ObserverServer(state, host="127.0.0.1", port=0)
    running.start()
    running.stop()
    running.stop()


def test_the_context_manager_stops_the_socket(config: Config) -> None:
    state = ObserverState(config, clock=FakeClock())
    with ObserverServer(state, host="127.0.0.1", port=0) as running:
        assert http_get(running.port, "/api/health")[0] == 200
        port = running.port
    with pytest.raises(OSError):
        http_get(port, "/api/health")


def test_a_demo_server_ticks_its_script(config: Config) -> None:
    demo = demo_state(config, clock=FakeClock())
    running = ObserverServer(demo, host="127.0.0.1", port=0, tick_seconds=0.05)
    running.start()
    try:
        before = demo.demo_ticks
        for _ in range(60):
            if demo.demo_ticks > before:
                break
            import time as _time

            _time.sleep(0.05)
        assert demo.demo_ticks > before
    finally:
        running.stop()
