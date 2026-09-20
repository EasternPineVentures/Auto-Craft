"""Safety: foreground lock, hold limits, release-all, rate limiting, e-stop.

Every test here uses :class:`FakeInputBackend`, so a passing suite is proof that
the safety machinery works without a single real key event.
"""

from __future__ import annotations

import pytest

from autocraft.control.errors import ControlError, InputBlocked, InvalidMouseDelta
from autocraft.control.keyboard import Keyboard
from autocraft.control.mouse import MOUSE_BUTTONS, Mouse
from autocraft.control.safety import SafetyGuard


@pytest.fixture
def guard(config, fake_input, clock):
    """A guard whose target window is foreground, with injected time."""
    return SafetyGuard(config, fake_input, target_is_foreground=lambda: True, clock=clock)


@pytest.fixture
def keyboard(guard, fake_input, config, clock):
    """Keyboard primitives bound to the test guard and the fake clock.

    The fake clock is the sleeper too, so a tap advances simulated time instead of
    really sleeping and the hold it actually used is observable.
    """
    return Keyboard(guard, fake_input, config, sleeper=clock.sleep)


@pytest.fixture
def mouse(guard, fake_input, config, clock):
    """Mouse primitives bound to the test guard and the fake clock."""
    return Mouse(guard, fake_input, config, sleeper=clock.sleep)


class TestForegroundLock:
    """Input must be refused whenever the game is not the foreground window."""

    def test_refuses_when_target_not_foreground(self, config, fake_input, clock) -> None:
        guard = SafetyGuard(config, fake_input, target_is_foreground=lambda: False, clock=clock)
        decision = guard.authorize("key_tap w")
        assert decision.allowed is False
        assert "foreground" in decision.reason.lower()

    def test_allows_when_target_is_foreground(self, guard) -> None:
        assert guard.authorize("key_tap w").allowed is True

    def test_keyboard_refuses_and_injects_nothing(self, config, fake_input, clock) -> None:
        guard = SafetyGuard(config, fake_input, target_is_foreground=lambda: False, clock=clock)
        keyboard = Keyboard(guard, fake_input, config)
        with pytest.raises(InputBlocked):
            keyboard.tap("w")
        assert fake_input.key_downs == []
        assert fake_input.key_ups == []

    def test_mouse_refuses_and_injects_nothing(self, config, fake_input, clock) -> None:
        guard = SafetyGuard(config, fake_input, target_is_foreground=lambda: False, clock=clock)
        mouse = Mouse(guard, fake_input, config)
        with pytest.raises(InputBlocked):
            mouse.move_relative(5, 5)
        assert fake_input.moves == []

    def test_focus_loss_between_press_and_release_still_releases(self, config, fake_input, clock) -> None:
        """A release must never be refused, even after focus is lost."""
        focused = {"value": True}
        guard = SafetyGuard(config, fake_input, target_is_foreground=lambda: focused["value"], clock=clock)
        keyboard = Keyboard(guard, fake_input, config)
        keyboard.press("w")
        focused["value"] = False
        keyboard.release("w")
        assert fake_input.key_ups == [ord("W")]
        assert guard.held_keys == ()

    def test_press_is_refused_after_focus_is_lost(self, config, fake_input, clock) -> None:
        focused = {"value": True}
        guard = SafetyGuard(config, fake_input, target_is_foreground=lambda: focused["value"], clock=clock)
        keyboard = Keyboard(guard, fake_input, config)
        focused["value"] = False
        with pytest.raises(InputBlocked):
            keyboard.press("w")
        assert fake_input.key_downs == []

    def test_probe_error_fails_closed(self, config, fake_input, clock) -> None:
        def explode() -> bool:
            raise RuntimeError("probe exploded")

        guard = SafetyGuard(config, fake_input, target_is_foreground=explode, clock=clock)
        decision = guard.authorize("key_tap w")
        assert decision.allowed is False
        assert "probe failed" in decision.reason

    def test_foreground_lock_can_be_disabled_explicitly(self, config, fake_input, clock) -> None:
        from dataclasses import replace

        relaxed = replace(config, require_foreground=False)
        guard = SafetyGuard(relaxed, fake_input, target_is_foreground=lambda: False, clock=clock)
        assert guard.authorize("key_tap w").allowed is True


class TestKeyStateTracking:
    """The guard must know exactly what is held so release-all is complete."""

    def test_press_registers_a_held_key(self, keyboard, guard) -> None:
        keyboard.press("w")
        assert "w" in guard.held_keys

    def test_release_clears_it(self, keyboard, guard) -> None:
        keyboard.press("w")
        keyboard.release("w")
        assert "w" not in guard.held_keys

    def test_duplicate_press_does_not_reset_the_hold_timer(self, guard, fake_input, config, clock) -> None:
        """The hold clock starts at the first press, not the most recent one.

        A policy that re-presses the same key every step must not be able to hold
        it forever by accident. Two presses are made, the second after most of the
        limit has already elapsed, and the release must fire on total elapsed time.
        """
        keyboard = Keyboard(guard, fake_input, config, sleeper=clock.sleep)
        limit = config.max_key_hold_seconds
        keyboard.press("w")
        clock.advance(limit * 0.75)
        keyboard.press("w")
        clock.advance(limit * 0.5)  # 1.25 * limit in total, 0.5 * limit since the re-press
        released = guard.enforce_hold_limits()
        assert "w" in released

    def test_releasing_a_key_that_is_not_held_still_sends_the_key_up(self, keyboard, fake_input, guard) -> None:
        """A stray key-up is harmless; a missing one is not.

        Release paths deliberately skip the "is it held?" bookkeeping question and
        just send the event, so they can never leave a key physically stuck.
        """
        keyboard.release("q")
        assert fake_input.key_ups == [ord("Q")]
        assert guard.held_keys == ()

    def test_held_keys_reports_names(self, keyboard, guard) -> None:
        keyboard.press("w")
        keyboard.press("a")
        assert set(guard.held_keys) == {"w", "a"}


class TestHoldLimits:
    """A key-up that never arrives must not leave the game stuck."""

    def test_enforce_releases_a_key_held_too_long(self, guard, keyboard, fake_input, config, clock) -> None:
        keyboard.press("w")
        clock.advance(config.max_key_hold_seconds + 0.01)
        released = guard.enforce_hold_limits()
        assert "w" in released
        assert ord("W") in fake_input.key_ups
        assert "w" not in guard.held_keys

    def test_enforce_leaves_short_holds_alone(self, guard, keyboard, fake_input, config, clock) -> None:
        keyboard.press("w")
        clock.advance(config.max_key_hold_seconds / 2)
        assert guard.enforce_hold_limits() == ()
        assert "w" in guard.held_keys

    def test_enforce_releases_a_button_held_too_long(self, guard, mouse, fake_input, config, clock) -> None:
        mouse.press("left")
        clock.advance(config.max_key_hold_seconds + 0.01)
        released = guard.enforce_hold_limits()
        assert "mouse:left" in released
        assert fake_input.button_ups == ["left"]

    def test_tap_is_clamped_to_the_hold_limit(self, guard, keyboard, fake_input, config, clock) -> None:
        """A caller cannot request an arbitrarily long hold."""
        keyboard.tap("w", hold_seconds=config.max_key_hold_seconds * 100)
        assert "w" not in guard.held_keys
        assert clock.sleeps == [config.max_key_hold_seconds]

    def test_tap_returns_the_hold_it_actually_used(self, keyboard, config) -> None:
        assert keyboard.tap("w", hold_seconds=0.01) == pytest.approx(0.01)

    def test_negative_hold_is_rejected(self, keyboard) -> None:
        with pytest.raises(ValueError):
            keyboard.tap("w", hold_seconds=-0.5)

    def test_release_failure_keeps_the_key_tracked(self, guard, fake_input, config, clock) -> None:
        """Failing to release must not also forget the key.

        ``release_all`` retries whatever the guard still believes is held, so
        dropping the bookkeeping on a backend error would silently strand a key.
        """
        keyboard = Keyboard(guard, fake_input, config, sleeper=clock.sleep)
        keyboard.press("w")
        fake_input.fail_on.add("key_up")
        with pytest.raises(ControlError):
            keyboard.release("w")
        assert "w" in guard.held_keys
        fake_input.fail_on.clear()
        guard.release_all("retry")
        assert "w" not in guard.held_keys
        assert fake_input.key_ups == [ord("W")]


class TestReleaseAll:
    """Release-all is the single guarantee the whole safety story rests on."""

    def test_releases_every_held_key_and_button(self, guard, keyboard, mouse, fake_input) -> None:
        keyboard.press("w")
        keyboard.press("space")
        mouse.press("left")
        mouse.press("right")
        released = guard.release_all("test")
        assert set(released) == {"w", "space", "mouse:left", "mouse:right"}
        assert set(fake_input.key_ups) == {ord("W"), ord(" ")}
        assert set(fake_input.button_ups) == {"left", "right"}
        assert guard.held_keys == ()
        assert guard.held_buttons == ()

    def test_release_all_is_idempotent(self, guard, keyboard, fake_input) -> None:
        keyboard.press("w")
        guard.release_all()
        count = len(fake_input.key_ups)
        guard.release_all()
        assert len(fake_input.key_ups) == count

    def test_release_all_works_while_stopped(self, guard, keyboard, fake_input) -> None:
        """The emergency stop must not prevent the release it triggers."""
        keyboard.press("w")
        guard.trigger_emergency_stop("test")
        assert fake_input.key_ups == [ord("W")]
        assert guard.held_keys == ()

    def test_release_all_works_while_not_foreground(self, config, fake_input, clock) -> None:
        guard = SafetyGuard(config, fake_input, target_is_foreground=lambda: False, clock=clock)
        keyboard = Keyboard(guard, fake_input, config)
        # Press is refused, so simulate a key that is somehow already held.
        guard.register_key_down("w")
        released = guard.release_all("focus lost")
        assert released == ("w",)
        assert fake_input.key_ups == [ord("W")]

    def test_release_survives_a_failing_backend(self, guard, fake_input) -> None:
        """A failed release must be retried later, not forgotten.

        The held record is the guard's only memory of what may still be
        physically down. Dropping it because the backend errored would strand
        the input in the game forever, with nothing left to retry.
        """
        guard.register_key_down("w")
        fake_input.fail_on.add("key_up")
        # Must not raise: refusing to release is the unsafe behaviour.
        assert guard.release_all("backend is broken") == ()
        assert guard.held_keys == ("w",)
        assert any(event.kind == "release_failed" for event in guard.events)

        # Once the backend recovers, the same release-all finishes the job.
        fake_input.fail_on.clear()
        assert guard.release_all("backend recovered") == ("w",)
        assert guard.held_keys == ()
        assert fake_input.key_ups == [ord("W")]

    def test_context_manager_releases_on_exit(self, config, fake_input, clock) -> None:
        guard = SafetyGuard(config, fake_input, target_is_foreground=lambda: True, clock=clock)
        keyboard = Keyboard(guard, fake_input, config)
        with guard:
            keyboard.press("w")
        assert ord("W") in fake_input.key_ups


class TestForcedReleaseFailureTracking:
    """Held state may only be dropped once the backend release actually worked.

    The invariant is one-directional: a release that did *not* happen must not
    be recorded as one that did. Every path below therefore checks both that the
    input stays tracked and that the failure is visible in the event log.
    """

    def test_a_failed_key_release_keeps_the_key_retryable(self, guard, fake_input) -> None:
        # 1. input tracked held
        guard.register_key_down("w")
        assert guard.held_keys == ("w",)

        # 2. backend release fails
        fake_input.fail_on.add("key_up")

        # 3. release_all returns safely
        first = guard.release_all("forced")

        # 4. input remains tracked
        assert first == ()
        assert guard.held_keys == ("w",)

        # 5. release_failed recorded
        failed = [event for event in guard.events if event.kind == "release_failed"]
        assert len(failed) == 1
        assert "w" in failed[0].detail

        # 6. backend recovers
        fake_input.fail_on.clear()

        # 7. second release_all retries
        second = guard.release_all("retry")

        # 8. release succeeds
        assert second == ("w",)
        assert fake_input.key_ups == [ord("W")]

        # 9. input leaves held tracking
        assert guard.held_keys == ()

    def test_a_failed_button_release_keeps_the_button_retryable(self, guard, fake_input) -> None:
        # 1. input tracked held
        guard.register_button_down("left")
        assert guard.held_buttons == ("left",)

        # 2. backend release fails
        fake_input.fail_on.add("mouse_button_up")

        # 3. release_all returns safely
        first = guard.release_all("forced")

        # 4. input remains tracked
        assert first == ()
        assert guard.held_buttons == ("left",)

        # 5. release_failed recorded
        failed = [event for event in guard.events if event.kind == "release_failed"]
        assert len(failed) == 1
        assert "left" in failed[0].detail

        # 6. backend recovers
        fake_input.fail_on.clear()

        # 7. second release_all retries
        second = guard.release_all("retry")

        # 8. release succeeds
        assert second == ("mouse:left",)
        assert fake_input.button_ups == ["left"]

        # 9. input leaves held tracking
        assert guard.held_buttons == ()

    def test_a_key_and_a_button_can_fail_independently(self, guard, fake_input) -> None:
        guard.register_key_down("w")
        guard.register_button_down("right")
        fake_input.fail_on.add("key_up")

        released = guard.release_all("partial failure")

        # The button came up, so it is reported and forgotten; the key did not,
        # so it is neither reported nor forgotten.
        assert released == ("mouse:right",)
        assert guard.held_keys == ("w",)
        assert guard.held_buttons == ()

    def test_release_keys_does_not_report_a_release_that_did_not_happen(self, guard, fake_input) -> None:
        guard.register_key_down("w")
        fake_input.fail_on.add("key_up")
        assert guard.release_keys("forced") == ()

    def test_release_buttons_does_not_report_a_release_that_did_not_happen(
        self, guard, fake_input
    ) -> None:
        guard.register_button_down("left")
        fake_input.fail_on.add("mouse_button_up")
        assert guard.release_buttons("forced") == ()

    def test_enforce_hold_limits_does_not_claim_a_failed_release(
        self, guard, fake_input, config, clock
    ) -> None:
        guard.register_key_down("w")
        clock.advance(config.max_key_hold_seconds + 0.01)
        fake_input.fail_on.add("key_up")

        assert guard.enforce_hold_limits() == ()
        # Still tracked, so the next release-all can try again.
        assert guard.held_keys == ("w",)

    def test_a_failing_release_never_raises_from_emergency_stop(self, guard, fake_input) -> None:
        guard.register_key_down("w")
        fake_input.fail_on.add("key_up")
        # The emergency stop must complete even when the backend is broken; an
        # exception here would abort the cleanup that everything else depends on.
        guard.trigger_emergency_stop("test")
        assert guard.stop_requested is True
        assert guard.held_keys == ("w",)

    def test_a_failing_release_never_raises_from_shutdown(self, guard, fake_input) -> None:
        guard.register_key_down("w")
        fake_input.fail_on.add("key_up")
        guard.shutdown("test")
        assert guard.held_keys == ("w",)

    def test_an_unknown_held_key_name_is_dropped_not_retried_forever(
        self, guard, fake_input
    ) -> None:
        """A name with no virtual key can never be released, so it cannot be kept.

        Keeping it would be a phantom hold that every future release-all would
        fail on. It also used to raise ``KeyError`` from inside the release path.
        """
        guard.register_key_down("definitely-not-a-key")

        assert guard.release_all("forced") == ()
        assert guard.held_keys == ()
        assert fake_input.key_ups == []
        assert any(event.kind == "error" for event in guard.events)


class TestEmergencyStop:
    """The e-stop is checked by polling, so it is testable without a global hook."""

    def test_pressing_the_stop_key_requests_a_stop(self, config, fake_input, clock) -> None:
        from autocraft.control.keymap import virtual_key_for

        stop_vk = virtual_key_for(config.emergency_stop_key)
        guard = SafetyGuard(
            config,
            fake_input,
            target_is_foreground=lambda: True,
            clock=clock,
            key_probe=lambda vk: vk == stop_vk,
        )
        assert guard.check_emergency_stop() is True
        assert guard.stop_requested is True

    def test_stop_key_is_checked_by_polling_not_by_hooking(self, config, fake_input, clock) -> None:
        """The stop key is read, never consumed, so the game still receives it.

        A global hotkey registration would swallow the key; polling
        ``GetAsyncKeyState`` leaves the game's own binding intact.
        """
        from autocraft.control.keymap import virtual_key_for

        stop_vk = virtual_key_for(config.emergency_stop_key)
        seen: list[int] = []

        def probe(vk: int) -> bool:
            seen.append(vk)
            return False

        guard = SafetyGuard(
            config, fake_input, target_is_foreground=lambda: True, clock=clock, key_probe=probe
        )
        assert guard.check_emergency_stop() is False
        assert seen == [stop_vk]
        assert guard.stop_requested is False

    def test_unpressed_stop_key_does_not_stop(self, guard) -> None:
        assert guard.check_emergency_stop() is False
        assert guard.stop_requested is False

    def test_stop_blocks_all_further_input(self, guard) -> None:
        guard.trigger_emergency_stop("test")
        decision = guard.authorize("key_tap w")
        assert decision.allowed is False
        assert "emergency stop" in decision.reason

    def test_stop_releases_held_state(self, guard, keyboard, mouse, fake_input) -> None:
        keyboard.press("w")
        mouse.press("left")
        guard.trigger_emergency_stop("test")
        assert set(fake_input.key_ups) == {ord("W")}
        assert fake_input.button_ups == ["left"]

    def test_reset_clears_the_stop(self, guard) -> None:
        guard.trigger_emergency_stop("test")
        guard.reset()
        assert guard.stop_requested is False
        assert guard.authorize("key_tap w").allowed is True


class TestRateLimiting:
    """Pacing slows input down; it must never consume the block budget."""

    def test_waits_when_actions_come_too_fast(self, fake_input, clock) -> None:
        from dataclasses import replace

        from autocraft.config import load_config

        config = replace(load_config(env={}), min_action_interval=0.5)
        guard = SafetyGuard(config, fake_input, target_is_foreground=lambda: True, clock=clock, sleeper=clock.sleep)
        guard.note_action()
        slept = guard.wait_for_rate_limit()
        assert slept == pytest.approx(0.5)

    def test_does_not_wait_when_the_interval_has_passed(self, fake_input, clock) -> None:
        from dataclasses import replace

        from autocraft.config import load_config

        config = replace(load_config(env={}), min_action_interval=0.5)
        guard = SafetyGuard(config, fake_input, target_is_foreground=lambda: True, clock=clock, sleeper=clock.sleep)
        guard.note_action()
        clock.advance(1.0)
        assert guard.wait_for_rate_limit() == 0.0

    def test_rate_limiting_does_not_block_authorisation(self, fake_input, clock) -> None:
        from dataclasses import replace

        from autocraft.config import load_config

        config = replace(load_config(env={}), min_action_interval=10.0)
        guard = SafetyGuard(config, fake_input, target_is_foreground=lambda: True, clock=clock)
        guard.note_action()
        assert guard.authorize("key_tap w").allowed is True
        assert guard.blocked_streak == 0


class TestBlockBudget:
    """Repeated refusals must eventually stop the loop rather than spin."""

    def test_streak_grows_and_triggers_a_stop(self, guard, config) -> None:
        for _ in range(config.max_consecutive_blocks - 1):
            guard.record_block("not foreground")
        assert guard.should_stop_for_blocks is False
        guard.record_block("not foreground")
        assert guard.should_stop_for_blocks is True

    def test_success_resets_the_streak(self, guard) -> None:
        guard.record_block("nope")
        guard.record_block("nope")
        guard.record_success()
        assert guard.blocked_streak == 0

    def test_last_reason_is_retained(self, guard) -> None:
        guard.record_block("target window is not the foreground window")
        assert "foreground" in guard.last_block_reason


class TestSafetyEvents:
    """Safety events are the audit trail; they must be serialisable."""

    def test_focus_loss_is_recorded(self, config, fake_input, clock) -> None:
        guard = SafetyGuard(config, fake_input, target_is_foreground=lambda: False, clock=clock)
        guard.authorize("key_tap w")
        assert any("focus" in event.kind or "focus" in event.detail for event in guard.events)

    def test_summary_is_json_serialisable(self, guard, keyboard) -> None:
        import json

        keyboard.press("w")
        guard.release_all("test")
        payload = json.loads(json.dumps(guard.summary()))
        assert "held_keys" in payload
        assert payload["held_keys"] == []

    def test_event_dicts_are_json_serialisable(self, guard, keyboard) -> None:
        import json

        keyboard.press("w")
        guard.release_all("test")
        events = guard.event_dicts()
        assert events
        json.loads(json.dumps(events))


class TestKeyboardPrimitives:
    """Keyboard behaviour that is not about the safety guard."""

    def test_press_uses_the_scan_code_backend(self, keyboard, fake_input) -> None:
        keyboard.press("w")
        assert fake_input.key_downs == [ord("W")]

    def test_unknown_key_raises(self, keyboard) -> None:
        with pytest.raises(ControlError):
            keyboard.press("definitely-not-a-key")

    def test_tap_presses_then_releases(self, keyboard, fake_input) -> None:
        keyboard.tap("space")
        assert fake_input.key_downs == [ord(" ")]
        assert fake_input.key_ups == [ord(" ")]

    def test_tap_releases_even_if_the_hold_sleep_is_interrupted(self, keyboard, fake_input, guard) -> None:
        """Ctrl+C during a hold must not leave the key down."""

        class Boom(Exception):
            pass

        original = guard.config

        def explode(_seconds: float) -> None:
            raise Boom

        keyboard._sleep = explode  # type: ignore[attr-defined]
        with pytest.raises(Boom):
            keyboard.tap("w")
        assert fake_input.key_ups == [ord("W")]
        assert original is guard.config

    def test_held_keys_property(self, keyboard) -> None:
        keyboard.press("w")
        assert "w" in keyboard.held_keys


class TestMousePrimitives:
    """Mouse behaviour: relative-only motion and a hard delta ceiling."""

    def test_relative_move_is_forwarded(self, mouse, fake_input) -> None:
        mouse.move_relative(7, -3)
        assert fake_input.moves == [(7, -3)]

    def test_delta_over_the_limit_is_rejected_not_clamped(self, mouse, fake_input, config) -> None:
        """Rejection protects the agent's intent from being silently altered."""
        with pytest.raises(InvalidMouseDelta):
            mouse.move_relative(config.max_mouse_delta + 1, 0)
        assert fake_input.moves == []

    def test_delta_at_the_limit_is_allowed(self, mouse, fake_input, config) -> None:
        mouse.move_relative(config.max_mouse_delta, 0)
        assert fake_input.moves == [(config.max_mouse_delta, 0)]

    def test_negative_delta_over_the_limit_is_rejected(self, mouse, config) -> None:
        with pytest.raises(InvalidMouseDelta):
            mouse.move_relative(0, -(config.max_mouse_delta + 1))

    def test_zero_move_is_allowed(self, mouse, fake_input) -> None:
        mouse.move_relative(0, 0)
        assert fake_input.moves == [(0, 0)]

    def test_click_presses_and_releases(self, mouse, fake_input) -> None:
        mouse.click("left")
        assert fake_input.button_downs == ["left"]
        assert fake_input.button_ups == ["left"]

    @pytest.mark.parametrize("button", MOUSE_BUTTONS)
    def test_every_documented_button_works(self, mouse, fake_input, button: str) -> None:
        mouse.click(button)
        assert fake_input.button_downs == [button]

    def test_unknown_button_raises(self, mouse) -> None:
        with pytest.raises(ControlError):
            mouse.press("trigger")

    def test_button_state_is_tracked(self, mouse, guard) -> None:
        mouse.press("left")
        assert "left" in guard.held_buttons
        mouse.release("left")
        assert guard.held_buttons == ()

    def test_mouse_release_all(self, mouse, fake_input) -> None:
        mouse.press("left")
        mouse.press("right")
        released = mouse.release_all()
        assert set(released) == {"left", "right"}
        assert fake_input.button_ups == ["left", "right"]
