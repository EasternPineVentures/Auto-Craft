"""The action vocabulary and its validation.

This file defines every way AutoCraft is allowed to influence the game, so it is
the most safety-critical interface in the project. All of it is pure data, which
means it can be tested exhaustively with no hardware present.
"""

from __future__ import annotations

import pytest

from autocraft.agent.action import (
    REQUIRED_PARAMETERS,
    Action,
    ActionKind,
    validate_action,
)
from autocraft.control.errors import InputBlocked
from autocraft.control.keymap import known_key_names


class TestFactoryConstructors:
    """Each factory must produce a valid action for the matching kind."""

    @pytest.mark.parametrize(
        "action,kind",
        [
            (Action.noop(), ActionKind.NOOP),
            (Action.key_press("w"), ActionKind.KEY_PRESS),
            (Action.key_release("w"), ActionKind.KEY_RELEASE),
            (Action.key_tap("w"), ActionKind.KEY_TAP),
            (Action.mouse_move(10, -5), ActionKind.MOUSE_MOVE),
            (Action.mouse_click(), ActionKind.MOUSE_CLICK),
            (Action.mouse_down(), ActionKind.MOUSE_DOWN),
            (Action.mouse_up(), ActionKind.MOUSE_UP),
            (Action.stop(), ActionKind.STOP),
        ],
    )
    def test_factories_build_valid_actions(self, action: Action, kind: ActionKind) -> None:
        assert action.kind is kind
        assert validate_action(action) is None

    def test_key_names_are_normalised(self) -> None:
        assert Action.key_press("W").parameters["key"] == "w"
        assert Action.key_press("  SPACE ").parameters["key"] == "space"

    def test_synonym_keys_resolve_to_one_canonical_name(self) -> None:
        """Aliases are folded to the canonical name at the factory boundary.

        ``esc`` and ``escape`` are the same physical key, so an agent that uses
        either spelling must produce an identical action - otherwise telemetry
        would record two different actions for one behaviour.
        """
        assert Action.key_press("esc").parameters == Action.key_press("escape").parameters
        assert Action.key_press("esc").parameters["key"] == "escape"
        assert Action.key_press("ESCAPE").parameters["key"] == "escape"
        assert Action.key_press("spacebar").parameters["key"] == "space"

    def test_unknown_key_is_rejected_at_construction(self) -> None:
        """Failing at construction beats failing halfway through an action."""
        from autocraft.control.errors import UnknownKeyError

        with pytest.raises(UnknownKeyError):
            Action.key_press("teleport-key")

    def test_actions_get_unique_ids(self) -> None:
        ids = {Action.noop().action_id for _ in range(50)}
        assert len(ids) == 50

    def test_actions_are_immutable(self) -> None:
        action = Action.key_press("w")
        with pytest.raises(Exception):
            action.kind = ActionKind.NOOP  # type: ignore[misc]


class TestInputClassification:
    """NOOP and STOP must be classified as non-input, everything else as input."""

    @pytest.mark.parametrize("kind", [ActionKind.NOOP, ActionKind.STOP])
    def test_non_input_kinds(self, kind: ActionKind) -> None:
        assert Action(kind).is_input is False

    @pytest.mark.parametrize(
        "kind",
        [
            ActionKind.KEY_PRESS,
            ActionKind.KEY_RELEASE,
            ActionKind.KEY_TAP,
            ActionKind.MOUSE_MOVE,
            ActionKind.MOUSE_CLICK,
            ActionKind.MOUSE_DOWN,
            ActionKind.MOUSE_UP,
        ],
    )
    def test_input_kinds(self, kind: ActionKind) -> None:
        assert Action(kind).is_input is True

    def test_every_kind_has_a_required_parameter_entry(self) -> None:
        for kind in ActionKind:
            assert kind in REQUIRED_PARAMETERS


class TestValidation:
    """Validation must reject anything the primitives could not execute safely."""

    def test_rejects_non_action(self) -> None:
        assert validate_action("w") is not None  # type: ignore[arg-type]
        assert validate_action(None) is not None  # type: ignore[arg-type]

    @pytest.mark.parametrize("key", ["", "not-a-key", "w2", "superkey", " ", "space bar"])
    def test_rejects_unknown_keys(self, key: str) -> None:
        from autocraft.control.errors import UnknownKeyError

        with pytest.raises(UnknownKeyError):
            Action.key_press(key)

    @pytest.mark.parametrize("key", [None, 42, b"w"])
    def test_rejects_non_string_keys(self, key: object) -> None:
        from autocraft.control.errors import UnknownKeyError

        with pytest.raises(UnknownKeyError):
            Action.key_press(key)  # type: ignore[arg-type]

    def test_accepts_every_documented_key(self) -> None:
        """Every name the keymap advertises must be usable as an action."""
        for name in known_key_names():
            assert validate_action(Action.key_press(name)) is None, name

    @pytest.mark.parametrize("dx,dy", [(0.5, 0), ("10", 0), (None, 0), (True, 0)])
    def test_rejects_non_integer_mouse_deltas(self, dx: object, dy: object) -> None:
        action = Action(ActionKind.MOUSE_MOVE, {"dx": dx, "dy": dy})
        assert validate_action(action) is not None

    def test_accepts_zero_and_negative_mouse_deltas(self) -> None:
        assert validate_action(Action.mouse_move(0, 0)) is None
        assert validate_action(Action.mouse_move(-500, -500)) is None

    @pytest.mark.parametrize("button", ["", "primary", "wheel", "side", "LEFT"])
    def test_rejects_unknown_mouse_buttons(self, button: str) -> None:
        assert validate_action(Action.mouse_click(button)) is not None

    @pytest.mark.parametrize("button", ["left", "right", "middle", "x1", "x2"])
    def test_accepts_documented_mouse_buttons(self, button: str) -> None:
        assert validate_action(Action.mouse_click(button)) is None

    @pytest.mark.parametrize("hold", [-1, "soon", None])
    def test_rejects_bad_hold_durations(self, hold: object) -> None:
        assert validate_action(Action(ActionKind.KEY_TAP, {"key": "w", "hold_seconds": hold})) is not None

    def test_rejects_negative_intended_duration(self) -> None:
        action = Action(ActionKind.NOOP, {}, intended_duration=-0.1)
        assert validate_action(action) is not None

    def test_rejects_missing_parameters(self) -> None:
        assert validate_action(Action(ActionKind.KEY_PRESS, {})) is not None
        assert validate_action(Action(ActionKind.MOUSE_MOVE, {"dx": 1})) is not None

    def test_rejects_unknown_kind(self) -> None:
        """An action can only ever be built with a kind the vocabulary defines."""
        with pytest.raises(ValueError):
            Action("teleport", {"x": 1})  # type: ignore[arg-type]

    def test_validation_refuses_an_unknown_kind(self) -> None:
        """Defence in depth: validation does not trust the object it is handed.

        ``__post_init__`` already refuses unknown kinds, so this reaches past it
        with ``object.__setattr__`` to prove the validator itself would also
        refuse one.
        """
        action = Action.noop()
        object.__setattr__(action, "kind", "teleport")
        assert validate_action(action) is not None


class TestSerialisation:
    """Actions must round-trip through JSON for telemetry."""

    @pytest.mark.parametrize(
        "action",
        [
            Action.noop(),
            Action.key_tap("space", 0.1),
            Action.mouse_move(-12, 34),
            Action.mouse_click("right"),
            Action.stop("done"),
        ],
    )
    def test_round_trip(self, action: Action) -> None:
        restored = Action.from_dict(action.to_dict())
        assert restored.kind is action.kind
        assert dict(restored.parameters) == dict(action.parameters)
        assert restored.action_id == action.action_id
        assert restored.intended_duration == action.intended_duration

    def test_to_dict_is_json_serialisable(self) -> None:
        import json

        payload = json.loads(json.dumps(Action.key_tap("w", 0.05).to_dict()))
        assert payload["kind"] == "key_tap"
        assert payload["parameters"]["key"] == "w"

    def test_describe_is_human_readable(self) -> None:
        assert "noop" in Action.noop().describe().lower()
        assert "w" in Action.key_tap("w").describe()
        assert "12" in Action.mouse_move(12, 0).describe()


class TestActionExecutor:
    """The executor must translate refusals into results, never exceptions."""

    def test_blocked_action_is_reported_not_raised(self, config, fake_input, clock) -> None:
        from autocraft.agent.action import ActionExecutor
        from autocraft.control.keyboard import Keyboard
        from autocraft.control.mouse import Mouse
        from autocraft.control.safety import SafetyGuard

        guard = SafetyGuard(config, fake_input, target_is_foreground=lambda: False, clock=clock)
        executor = ActionExecutor(Keyboard(guard, fake_input, config), Mouse(guard, fake_input, config), clock=clock)
        result = executor.execute(Action.key_tap("w"))
        assert result.attempted is True
        assert result.executed is False
        assert result.was_blocked is True
        assert result.ok is False
        assert fake_input.key_downs == []
        assert fake_input.key_ups == []

    def test_noop_is_neither_attempted_nor_executed(self, config, fake_input, clock) -> None:
        """A no-op must not be able to claim it performed an action.

        ``executed`` records that input reached the game. A no-op injects
        nothing, so a run made only of no-ops must report zero executed steps -
        otherwise the telemetry overstates what the agent did.
        """
        from autocraft.agent.action import ActionExecutor
        from autocraft.control.keyboard import Keyboard
        from autocraft.control.mouse import Mouse
        from autocraft.control.safety import SafetyGuard

        guard = SafetyGuard(config, fake_input, target_is_foreground=lambda: True, clock=clock)
        executor = ActionExecutor(Keyboard(guard, fake_input, config), Mouse(guard, fake_input, config), clock=clock)
        for action in (Action.noop(), Action.stop("nothing to do")):
            result = executor.execute(action)
            assert result.attempted is False
            assert result.executed is False
            assert result.was_blocked is False
            assert result.error is None
            assert result.ok is True
        assert fake_input.key_downs == []
        assert fake_input.key_ups == []

    def test_noop_is_ok_even_when_the_target_is_not_foreground(self, config, fake_input, clock) -> None:
        """There is nothing to gate, so an unfocused target cannot block a no-op."""
        from autocraft.agent.action import ActionExecutor
        from autocraft.control.keyboard import Keyboard
        from autocraft.control.mouse import Mouse
        from autocraft.control.safety import SafetyGuard

        guard = SafetyGuard(config, fake_input, target_is_foreground=lambda: False, clock=clock)
        executor = ActionExecutor(Keyboard(guard, fake_input, config), Mouse(guard, fake_input, config), clock=clock)
        result = executor.execute(Action.noop())
        assert result.ok is True
        assert result.was_blocked is False

    def test_allowed_action_reaches_the_backend(self, config, fake_input, clock) -> None:
        from autocraft.agent.action import ActionExecutor
        from autocraft.control.keyboard import Keyboard
        from autocraft.control.mouse import Mouse
        from autocraft.control.safety import SafetyGuard

        guard = SafetyGuard(config, fake_input, target_is_foreground=lambda: True, clock=clock)
        executor = ActionExecutor(Keyboard(guard, fake_input, config), Mouse(guard, fake_input, config), clock=clock)
        result = executor.execute(Action.key_tap("w"))
        assert result.attempted is True
        assert result.executed is True
        assert result.ok is True
        assert fake_input.key_downs == [ord("W")]
        assert fake_input.key_ups == [ord("W")]

    def test_invalid_action_never_reaches_the_backend(self, config, fake_input, clock) -> None:
        from autocraft.agent.action import ActionExecutor
        from autocraft.control.keyboard import Keyboard
        from autocraft.control.mouse import Mouse
        from autocraft.control.safety import SafetyGuard

        guard = SafetyGuard(config, fake_input, target_is_foreground=lambda: True, clock=clock)
        executor = ActionExecutor(Keyboard(guard, fake_input, config), Mouse(guard, fake_input, config), clock=clock)
        result = executor.execute(Action(ActionKind.KEY_PRESS, {"key": "nonsense"}))
        assert result.executed is False
        assert result.was_blocked is False
        assert result.error is not None
        assert fake_input.key_downs == []

    def test_backend_failure_is_captured_as_an_error(self, config, fake_input, clock) -> None:
        from autocraft.agent.action import ActionExecutor
        from autocraft.control.keyboard import Keyboard
        from autocraft.control.mouse import Mouse
        from autocraft.control.safety import SafetyGuard

        fake_input.fail_on.add("key_down")
        guard = SafetyGuard(config, fake_input, target_is_foreground=lambda: True, clock=clock)
        executor = ActionExecutor(Keyboard(guard, fake_input, config), Mouse(guard, fake_input, config), clock=clock)
        result = executor.execute(Action.key_press("w"))
        assert result.executed is False
        assert result.error is not None
        assert "synthetic failure" in result.error

    def test_result_serialisation(self, config, fake_input, clock) -> None:
        import json

        from autocraft.agent.action import ActionExecutor
        from autocraft.control.keyboard import Keyboard
        from autocraft.control.mouse import Mouse
        from autocraft.control.safety import SafetyGuard

        guard = SafetyGuard(config, fake_input, target_is_foreground=lambda: True, clock=clock)
        executor = ActionExecutor(Keyboard(guard, fake_input, config), Mouse(guard, fake_input, config), clock=clock)
        payload = json.loads(json.dumps(executor.execute(Action.noop()).to_dict()))
        assert payload["kind"] == "noop"

    def test_input_blocked_carries_the_reason(self, config, fake_input, clock) -> None:
        from autocraft.control.safety import SafetyGuard

        guard = SafetyGuard(config, fake_input, target_is_foreground=lambda: False, clock=clock)
        with pytest.raises(InputBlocked) as excinfo:
            guard.authorize_or_raise("key_tap w")
        assert "foreground" in str(excinfo.value).lower()
