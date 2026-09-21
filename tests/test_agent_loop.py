"""The bounded agent loop, the NoOp policy, and observation plumbing.

The loop is tested end to end with fake backends, which means these tests
exercise the real OBSERVE -> DECIDE -> SAFETY -> ACT -> VERIFY -> RECORD path
without a game and without injecting anything.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from autocraft.agent.action import Action, ActionExecutor, ActionKind
from autocraft.agent.decision import DecisionPolicy, NoOpDecisionPolicy
from autocraft.agent.loop import AgentLoop
from autocraft.agent.observation import Observation, Observer, WindowStatus
from autocraft.config import DEFAULT_WAKE_VERIFY_SETTLE_SECONDS, Config, load_config
from autocraft.control.keyboard import Keyboard
from autocraft.control.mouse import Mouse
from autocraft.control.safety import SafetyGuard
from autocraft.telemetry.recorder import RunRecorder
from autocraft.vision.capture import ScreenCapturer
from autocraft.vision.frame import Frame, ScreenRegion
from autocraft.vision.window import TargetStatus, WindowLocator

from conftest import FakeCaptureBackend, FakeInputBackend


class ScriptedPolicy:
    """A policy that plays back a fixed list of actions."""

    name = "scripted"

    def __init__(self, actions: list[Action]) -> None:
        self._actions = list(actions)
        self.calls = 0
        self.resets = 0

    def decide(self, observation: Observation) -> Action:
        self.calls += 1
        if self._actions:
            return self._actions.pop(0)
        return Action.noop()

    def reset(self) -> None:
        self.resets += 1


def build_loop(
    config: Config,
    fake_windows,
    fake_capture,
    fake_input,
    clock,
    *,
    policy: DecisionPolicy | None = None,
    recorder: RunRecorder | None = None,
    **kwargs,
) -> tuple[AgentLoop, SafetyGuard]:
    """Assemble a loop wired entirely to fakes."""
    locator = WindowLocator(fake_windows, config.target_title_patterns, clock=clock, rediscover_after=0.0)
    observer = Observer(locator, ScreenCapturer(fake_capture, clock=clock), clock=clock)
    guard = SafetyGuard(config, fake_input, target_is_foreground=locator.is_target_foreground, clock=clock)
    executor = ActionExecutor(
        Keyboard(guard, fake_input, config, sleeper=clock.sleep),
        Mouse(guard, fake_input, config, sleeper=clock.sleep),
        clock=clock,
    )
    loop = AgentLoop(
        config=config,
        observer=observer,
        guard=guard,
        policy=policy,
        executor=executor,
        recorder=recorder,
        clock=clock,
        sleeper=clock.sleep,
        **kwargs,
    )
    return loop, guard


class TestNoOpPolicy:
    """V0 ships exactly one policy and it does nothing."""

    def test_always_returns_noop(self) -> None:
        policy = NoOpDecisionPolicy()
        observation = Observation(index=0, timestamp=0.0, window=WindowStatus(found=True))
        for _ in range(5):
            assert policy.decide(observation).kind is ActionKind.NOOP

    def test_noop_is_not_input(self) -> None:
        assert Action.noop().is_input is False

    def test_reset_is_harmless(self) -> None:
        NoOpDecisionPolicy().reset()

    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(NoOpDecisionPolicy(), DecisionPolicy)

    def test_policy_does_not_use_the_frame(self) -> None:
        """A NoOp policy must not pretend to look at the pixels."""
        frame = Frame(image=np.zeros((8, 8, 3), dtype=np.uint8), timestamp=0.0)
        observation = Observation(index=0, timestamp=0.0, window=WindowStatus(found=True), frame=frame)
        assert NoOpDecisionPolicy().decide(observation).kind is ActionKind.NOOP


class TestBoundedRun:
    """There must be no way to start an unbounded run."""

    def test_requires_a_bound(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        loop, _ = build_loop(config, fake_windows, fake_capture, fake_input, clock)
        with pytest.raises(ValueError):
            loop.run()

    @pytest.mark.parametrize("kwargs", [{"max_steps": 0}, {"max_steps": -1}, {"max_seconds": 0}])
    def test_rejects_non_positive_bounds(self, config, fake_windows, fake_capture, fake_input, clock, kwargs) -> None:
        loop, _ = build_loop(config, fake_windows, fake_capture, fake_input, clock)
        with pytest.raises(ValueError):
            loop.run(**kwargs)

    def test_runs_exactly_max_steps(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        recorder = RunRecorder(tmp_path, clock=clock)
        loop, _ = build_loop(config, fake_windows, fake_capture, fake_input, clock, recorder=recorder)
        record = loop.run(max_steps=4)
        assert record.step_count == 4
        assert len(record.steps) == 4

    def test_stops_when_max_seconds_is_exceeded(self, config, fake_windows, fake_capture, fake_input, clock) -> None:
        loop, _ = build_loop(config, fake_windows, fake_capture, fake_input, clock)
        record = loop.run(max_seconds=0.001)
        assert record.step_count >= 1
        assert "time" in record.stop_reason.lower() or "max_seconds" in record.stop_reason.lower()

    def test_finishes_the_record(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        recorder = RunRecorder(tmp_path, clock=clock)
        loop, _ = build_loop(config, fake_windows, fake_capture, fake_input, clock, recorder=recorder)
        loop.run(max_steps=2)
        assert recorder.closed is True
        payload = json.loads(recorder.record_path.read_text(encoding="utf-8"))
        assert payload["step_count"] == 2
        # "completed" means the loop ran to its own bound; "finished" is reserved
        # for a caller closing the recorder by hand.
        assert payload["status"] == "completed"
        assert payload["stop_reason"]


class TestObserveStage:
    """Observation must never inject input, and capture failure must not crash."""

    def test_noop_run_injects_nothing(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        recorder = RunRecorder(tmp_path, clock=clock)
        loop, _ = build_loop(config, fake_windows, fake_capture, fake_input, clock, recorder=recorder)
        loop.run(max_steps=3)
        assert fake_input.key_downs == []
        assert fake_input.key_ups == []
        assert fake_input.button_downs == []
        assert fake_input.button_ups == []
        assert fake_input.moves == []

    def test_frames_are_captured_from_the_client_area(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        recorder = RunRecorder(tmp_path, clock=clock)
        loop, _ = build_loop(config, fake_windows, fake_capture, fake_input, clock, recorder=recorder)
        loop.run(max_steps=2)
        assert fake_capture.calls
        assert all(region == ScreenRegion(100, 50, 320, 240) for region in fake_capture.calls)

    def test_capture_failure_is_recorded_not_raised(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        """A dead capture device must not end the run; it must be reported.

        The loop keeps stepping with frameless observations so the failure is
        visible in telemetry rather than being a crash with no trace.
        """
        fake_capture.fail = True
        recorder = RunRecorder(tmp_path, clock=clock)
        loop, _ = build_loop(config, fake_windows, fake_capture, fake_input, clock, recorder=recorder)
        record = loop.run(max_steps=2)
        assert record.step_count == 2
        assert record.status == "completed"
        assert all(step["observation"]["has_frame"] is False for step in record.steps)
        assert fake_input.key_downs == []

    def test_missing_target_still_produces_observations(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        fake_windows.remove(0x100)
        fake_windows.remove(0x200)
        recorder = RunRecorder(tmp_path, clock=clock)
        loop, _ = build_loop(config, fake_windows, fake_capture, fake_input, clock, recorder=recorder)
        record = loop.run(max_steps=2)
        assert record.step_count == 2


class TestSafetyStage:
    """Safety decisions must be visible in the record, not silently swallowed."""

    def test_noop_is_not_gated_by_the_safety_guard(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        """Doing nothing is not an injection, so it can never be "blocked".

        Gating a no-op would report a run that never touched the game as blocked,
        and would spend the consecutive-block budget while doing it.
        """
        recorder = RunRecorder(tmp_path, clock=clock)
        loop, guard = build_loop(config, fake_windows, fake_capture, fake_input, clock, recorder=recorder)
        fake_windows.focus(0x200)
        record = loop.run(max_steps=config.max_consecutive_blocks + 4)
        assert record.status == "completed"
        assert record.blocked_steps == 0
        assert guard.blocked_streak == 0
        assert all(step["result"]["blocked_reason"] is None for step in record.steps)

    def test_focus_loss_blocks_the_scripted_action(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        policy = ScriptedPolicy([Action.key_tap("w")] * 4)
        recorder = RunRecorder(tmp_path, clock=clock)
        loop, _ = build_loop(config, fake_windows, fake_capture, fake_input, clock, policy=policy, recorder=recorder)
        # Discovery happens on the first step; steal focus before it.
        fake_windows.focus(0x200)
        record = loop.run(max_steps=2)
        assert fake_input.key_downs == []
        assert record.blocked_steps == 2

    def test_repeated_blocks_stop_the_run(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        policy = ScriptedPolicy([Action.key_tap("w")] * 20)
        recorder = RunRecorder(tmp_path, clock=clock)
        loop, _ = build_loop(config, fake_windows, fake_capture, fake_input, clock, policy=policy, recorder=recorder)
        fake_windows.focus(0x200)
        record = loop.run(max_steps=20)
        assert record.step_count <= config.max_consecutive_blocks + 1
        assert "block" in record.stop_reason.lower()
        assert fake_input.key_downs == []

    def test_policy_stop_action_ends_the_run(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        policy = ScriptedPolicy([Action.key_tap("w"), Action.stop("script says stop")])
        recorder = RunRecorder(tmp_path, clock=clock)
        loop, _ = build_loop(config, fake_windows, fake_capture, fake_input, clock, policy=policy, recorder=recorder)
        record = loop.run(max_steps=10)
        assert record.step_count == 2
        assert "stop" in record.stop_reason.lower()
        # The W tap happened while focused, then the loop stopped.
        assert fake_input.key_downs == [ord("W")]

    def test_emergency_stop_halts_the_run(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        from autocraft.control.keymap import virtual_key_for

        stop_vk = virtual_key_for(config.emergency_stop_key)
        pressed = {"value": False}
        locator = WindowLocator(fake_windows, config.target_title_patterns, clock=clock, rediscover_after=0.0)
        observer = Observer(locator, ScreenCapturer(fake_capture, clock=clock), clock=clock)
        guard = SafetyGuard(
            config,
            fake_input,
            target_is_foreground=locator.is_target_foreground,
            clock=clock,
            key_probe=lambda vk: vk == stop_vk and pressed["value"],
        )
        recorder = RunRecorder(tmp_path, clock=clock)

        class StopOnSecondStep(ScriptedPolicy):
            def decide(self, observation: Observation) -> Action:
                pressed["value"] = True
                return super().decide(observation)

        loop = AgentLoop(
            config=config,
            observer=observer,
            guard=guard,
            policy=StopOnSecondStep([]),
            recorder=recorder,
            clock=clock,
            sleeper=clock.sleep,
        )
        record = loop.run(max_steps=10)
        assert record.step_count <= 2
        assert "emergency" in record.stop_reason.lower() or "stop" in record.stop_reason.lower()
        assert fake_input.key_downs == []

    def test_hold_limit_backstop_runs_each_step(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        """A key that somehow stays held is released by the loop, not the caller."""
        recorder = RunRecorder(tmp_path, clock=clock)
        loop, guard = build_loop(config, fake_windows, fake_capture, fake_input, clock, recorder=recorder)
        guard.register_key_down("w")
        clock.advance(config.max_key_hold_seconds + 1.0)
        loop.run(max_steps=1)
        assert ord("W") in fake_input.key_ups

    def test_everything_is_released_when_the_run_ends(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        recorder = RunRecorder(tmp_path, clock=clock)
        loop, guard = build_loop(config, fake_windows, fake_capture, fake_input, clock, recorder=recorder)
        guard.register_key_down("w")
        guard.register_button_down("left")
        loop.run(max_steps=2)
        assert ord("W") in fake_input.key_ups
        assert "left" in fake_input.button_ups
        assert guard.held_keys == ()
        assert guard.held_buttons == ()


class TestActStage:
    """Actions that pass safety must actually reach the backend."""

    def test_focused_tap_reaches_the_backend(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        policy = ScriptedPolicy([Action.key_tap("w")])
        recorder = RunRecorder(tmp_path, clock=clock)
        loop, _ = build_loop(config, fake_windows, fake_capture, fake_input, clock, policy=policy, recorder=recorder)
        record = loop.run(max_steps=1)
        assert fake_input.key_downs == [ord("W")]
        assert fake_input.key_ups == [ord("W")]
        assert record.executed_steps == 1

    def test_mouse_move_respects_the_delta_limit(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        policy = ScriptedPolicy([Action.mouse_move(config.max_mouse_delta * 10, 0)])
        recorder = RunRecorder(tmp_path, clock=clock)
        loop, _ = build_loop(config, fake_windows, fake_capture, fake_input, clock, policy=policy, recorder=recorder)
        loop.run(max_steps=1)
        assert fake_input.moves == []

    def test_step_records_carry_action_and_result(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        policy = ScriptedPolicy([Action.key_tap("w")])
        recorder = RunRecorder(tmp_path, clock=clock)
        loop, _ = build_loop(config, fake_windows, fake_capture, fake_input, clock, policy=policy, recorder=recorder)
        record = loop.run(max_steps=1)
        step = record.steps[0]
        assert step["action"]["kind"] == "key_tap"
        assert step["result"]["executed"] is True
        assert step["observation"]["has_frame"] is True


class TestVerifyStage:
    """VERIFY must report a measurement, never a claim of success."""

    def test_identical_frames_report_no_change(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        policy = ScriptedPolicy([Action.key_tap("w")])
        recorder = RunRecorder(tmp_path, clock=clock)
        loop, _ = build_loop(config, fake_windows, fake_capture, fake_input, clock, policy=policy, recorder=recorder)
        record = loop.run(max_steps=1)
        step = record.steps[0]
        assert step["verification_measured"] is True
        assert step["frame_difference"] == 0.0
        assert step["frame_changed"] is False

    def test_changed_frames_are_detected(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        flat = np.zeros((240, 320, 4), dtype=np.uint8)
        flat[..., 3] = 255
        bright = np.full((240, 320, 4), 200, dtype=np.uint8)
        bright[..., 3] = 255
        fake_capture.frames = [flat.copy(), bright.copy()]
        policy = ScriptedPolicy([Action.key_tap("w")])
        recorder = RunRecorder(tmp_path, clock=clock)
        loop, _ = build_loop(config, fake_windows, fake_capture, fake_input, clock, policy=policy, recorder=recorder)
        record = loop.run(max_steps=1)
        step = record.steps[0]
        assert step["frame_difference"] > 0.5
        assert step["frame_changed"] is True

    def test_threshold_decides_the_verdict(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        """The same pixel change is "changed" or "not changed" purely by threshold.

        A one-level shift across the whole frame is a real 0.0039 mean difference.
        Setting the threshold below it must flip the verdict, which proves the
        comparison uses the measured difference rather than any hidden state.
        """
        near = np.zeros((240, 320, 4), dtype=np.uint8)
        near[..., 3] = 255
        shifted = near.copy()
        shifted[..., 0] = 1

        def run(threshold: float) -> dict:
            fake_capture.frames = [near.copy(), shifted.copy()]
            recorder = RunRecorder(tmp_path / f"t{threshold}", clock=clock)
            loop, _ = build_loop(
                config,
                fake_windows,
                fake_capture,
                fake_input,
                clock,
                policy=ScriptedPolicy([Action.key_tap("w")]),
                recorder=recorder,
                change_threshold=threshold,
            )
            return loop.run(max_steps=1).steps[0]

        measured = run(0.01)
        assert 0.0 < measured["frame_difference"] < 0.01
        assert measured["frame_changed"] is False

        sensitive = run(0.001)
        assert sensitive["frame_difference"] == measured["frame_difference"]
        assert sensitive["frame_changed"] is True


class RedrawAwareInputBackend(FakeInputBackend):
    """An input backend that also says when the game will have redrawn.

    The game draws an injected movement asynchronously. The moment the move is
    sent is therefore the moment the next frame becomes *due*, not the moment it
    becomes *available* - which is the whole reason the loop has to wait before
    it captures the frame it judges the move by.
    """

    def __init__(self, clock, redraw_after: float) -> None:
        super().__init__()
        self._clock = clock
        self._redraw_after = redraw_after
        self.redraw_at: float | None = None

    def move_relative(self, dx: int, dy: int) -> None:
        super().move_relative(dx, dy)
        self.redraw_at = self._clock() + self._redraw_after


class RedrawAwareCaptureBackend(FakeCaptureBackend):
    """A capture backend whose picture only shows a movement once it is drawn.

    Before the first movement, and until the redraw deadline passes, it returns
    a flat dark frame. After the deadline it returns a flat bright one. That is
    enough for :meth:`Frame.difference` to have something real to measure.
    """

    def __init__(self, clock, inputs: RedrawAwareInputBackend, *, drawn_value: int = 200) -> None:
        super().__init__(value=0)
        self._clock = clock
        self._inputs = inputs
        self._drawn_value = drawn_value

    def grab(self, region: ScreenRegion) -> np.ndarray:
        drawn = self._inputs.redraw_at is not None and self._clock() >= self._inputs.redraw_at
        self.value = self._drawn_value if drawn else 0
        return super().grab(region)


class TestVerifySettle:
    """VERIFY must never judge a movement by the picture from before it.

    The live defect this pins: a capture issued immediately after injecting a
    movement samples the desktop before the game has drawn that movement, so it
    returns the pre-movement picture, the difference measures as zero, and every
    movement looks like it did nothing. Measured on the reference machine, the
    first capture after a movement reads 0.00000 and the second, ~115 ms later,
    reads about 0.104.
    """

    def _step(self, config, fake_windows, clock, tmp_path, *, settle: float) -> dict:
        """Run one mouse-move step against a game that redraws after 50 ms."""
        inputs = RedrawAwareInputBackend(clock, redraw_after=0.05)
        capture = RedrawAwareCaptureBackend(clock, inputs)
        recorder = RunRecorder(tmp_path, clock=clock)
        loop, _ = build_loop(
            config,
            fake_windows,
            capture,
            inputs,
            clock,
            policy=ScriptedPolicy([Action.mouse_move(20, 0)]),
            recorder=recorder,
            verify_settle_seconds=settle,
        )
        return loop.run(max_steps=1).steps[0]

    def test_a_movement_is_judged_after_the_game_redraws(self, config, fake_windows, clock, tmp_path) -> None:
        step = self._step(config, fake_windows, clock, tmp_path, settle=0.15)
        assert step["verification_measured"] is True
        assert step["frame_difference"] > 0.5
        assert step["frame_changed"] is True

    def test_without_the_wait_the_movement_looks_like_it_did_nothing(self, config, fake_windows, clock, tmp_path) -> None:
        step = self._step(config, fake_windows, clock, tmp_path, settle=0.0)
        assert step["verification_measured"] is True
        assert step["frame_difference"] == 0.0
        assert step["frame_changed"] is False

    def test_the_wait_is_reported_in_the_step_timings(self, config, fake_windows, clock, tmp_path) -> None:
        step = self._step(config, fake_windows, clock, tmp_path, settle=0.15)
        assert step["timings"]["verify_settle"] == pytest.approx(150.0, abs=1.0)
        assert step["timings"]["verify"] >= step["timings"]["verify_settle"]
        assert step["timings"]["total"] >= step["timings"]["verify"]

    def test_no_wait_is_reported_when_none_is_configured(self, config, fake_windows, clock, tmp_path) -> None:
        """A skipped phase is omitted, not reported as a misleading zero."""
        step = self._step(config, fake_windows, clock, tmp_path, settle=0.0)
        assert "verify_settle" not in step["timings"]

    def test_the_wait_only_precedes_the_verification_capture(self, config, fake_windows, clock, tmp_path) -> None:
        """One wait per movement, not one per capture."""
        self._step(config, fake_windows, clock, tmp_path, settle=0.15)
        assert clock.sleeps == [0.15]

    def test_the_shipped_configuration_waits(self, config) -> None:
        assert DEFAULT_WAKE_VERIFY_SETTLE_SECONDS > 0
        assert config.wake_verify_settle_seconds == DEFAULT_WAKE_VERIFY_SETTLE_SECONDS


class TestFramePersistence:
    """Frame saving is opt-in so a default run stays cheap."""

    def test_no_frames_saved_by_default(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        recorder = RunRecorder(tmp_path, clock=clock)
        loop, _ = build_loop(config, fake_windows, fake_capture, fake_input, clock, recorder=recorder)
        loop.run(max_steps=3)
        assert list(recorder.frames_dir.glob("*.png")) == []

    def test_frames_saved_when_requested(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        pytest.importorskip("cv2")
        recorder = RunRecorder(tmp_path, clock=clock)
        loop, _ = build_loop(config, fake_windows, fake_capture, fake_input, clock, recorder=recorder, save_frames_every=1)
        loop.run(max_steps=2)
        assert len(list(recorder.frames_dir.glob("*.png"))) == 2


class TestProgressCallback:
    """The CLI prints progress through this hook."""

    def test_callback_runs_once_per_step(self, config, fake_windows, fake_capture, fake_input, clock, tmp_path) -> None:
        seen = []
        recorder = RunRecorder(tmp_path, clock=clock)
        loop, _ = build_loop(config, fake_windows, fake_capture, fake_input, clock, recorder=recorder, progress=seen.append)
        loop.run(max_steps=3)
        assert len(seen) == 3
        assert [record.index for record in seen] == [0, 1, 2]


class TestObserverUnit:
    """Observer behaviour in isolation."""

    def test_observation_reports_window_status(self, config, fake_windows, fake_capture, clock) -> None:
        locator = WindowLocator(fake_windows, config.target_title_patterns, clock=clock)
        observer = Observer(locator, ScreenCapturer(fake_capture, clock=clock), clock=clock)
        observation = observer.observe(3)
        assert observation.index == 3
        assert observation.window.found is True
        assert observation.window.is_foreground is True
        assert observation.has_frame is True
        assert observation.can_act is True

    def test_observation_without_capture(self, config, fake_windows, fake_capture, clock) -> None:
        locator = WindowLocator(fake_windows, config.target_title_patterns, clock=clock)
        observer = Observer(locator, ScreenCapturer(fake_capture, clock=clock), clock=clock)
        observation = observer.observe(0, capture=False)
        assert observation.has_frame is False
        assert fake_capture.calls == []

    def test_capture_failure_is_attached_to_the_observation(self, config, fake_windows, fake_capture, clock) -> None:
        fake_capture.fail = True
        locator = WindowLocator(fake_windows, config.target_title_patterns, clock=clock)
        observer = Observer(locator, ScreenCapturer(fake_capture, clock=clock), clock=clock)
        observation = observer.observe(0)
        assert observation.frame is None
        assert observation.capture_error is not None

    def test_save_frame_writes_a_png(self, config, fake_windows, fake_capture, clock, tmp_path) -> None:
        pytest.importorskip("cv2")
        locator = WindowLocator(fake_windows, config.target_title_patterns, clock=clock)
        observer = Observer(locator, ScreenCapturer(fake_capture, clock=clock), clock=clock)
        observation = observer.observe(0)
        path = observer.save_frame(observation, tmp_path, stem="shot")
        assert path.is_file()
        assert path.name == "shot.png"

    def test_save_frame_without_a_frame_is_a_no_op(self, config, fake_windows, fake_capture, clock, tmp_path) -> None:
        """Nothing to save is not an error; the caller already has the failure."""
        locator = WindowLocator(fake_windows, config.target_title_patterns, clock=clock)
        observer = Observer(locator, ScreenCapturer(fake_capture, clock=clock), clock=clock)
        observation = observer.observe(0, capture=False)
        assert observer.save_frame(observation, tmp_path) is None

    def test_observation_serialisation_omits_pixels(self, config, fake_windows, fake_capture, clock) -> None:
        locator = WindowLocator(fake_windows, config.target_title_patterns, clock=clock)
        observer = Observer(locator, ScreenCapturer(fake_capture, clock=clock), clock=clock)
        payload = json.loads(json.dumps(observer.observe(0).to_dict()))
        assert payload["window"]["found"] is True
        assert payload["frame"]["width"] == 320
        assert "image" not in payload["frame"]


class TestWindowStatus:
    """The serialisable window snapshot."""

    def test_from_target_status_when_found(self) -> None:
        status = TargetStatus(
            found=True,
            window=__import__("autocraft.vision.window", fromlist=["WindowInfo"]).WindowInfo(
                handle=1, title="Luanti", region=ScreenRegion(1, 2, 3, 4), visible=True, minimized=False
            ),
            is_foreground=True,
        )
        snapshot = WindowStatus.from_target_status(status)
        assert snapshot.found is True
        assert snapshot.title == "Luanti"
        assert snapshot.region == ScreenRegion(1, 2, 3, 4)
        assert snapshot.is_foreground is True

    def test_from_target_status_when_missing(self) -> None:
        snapshot = WindowStatus.from_target_status(TargetStatus(found=False, reason="nothing matched"))
        assert snapshot.found is False
        assert snapshot.reason == "nothing matched"
        assert snapshot.region is None

    def test_to_dict_serialises_region(self) -> None:
        snapshot = WindowStatus(found=True, region=ScreenRegion(1, 2, 3, 4))
        assert snapshot.to_dict()["region"]["width"] == 3
