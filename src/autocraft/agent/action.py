"""Actions: the complete vocabulary of things AutoCraft may do.

The action set is intentionally tiny and entirely human: hold a key, release a
key, tap a key, move the mouse relatively, press or release a mouse button. There
is no "mine this block" or "walk to these coordinates", because those would
require privileged game state and would break the pixels-in / controls-out
contract.

Validation is a pure function so it can be tested without any hardware, and the
executor turns a validated action into a primitive call and an
:class:`ActionResult`. Refusals by the safety guard are reported as
``blocked_reason``; malformed actions are reported as ``error``. The distinction
matters: only the first counts toward the loop's consecutive-block budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping
from uuid import uuid4

from ..control.errors import ControlError, InputBlocked
from ..control.keyboard import Keyboard
from ..control.keymap import is_known_key, normalise_key
from ..control.mouse import MOUSE_BUTTONS, Mouse

__all__ = [
    "Action",
    "ActionExecutor",
    "ActionKind",
    "ActionResult",
    "validate_action",
]


class ActionKind(str, Enum):
    """Every action AutoCraft can express."""

    NOOP = "noop"
    KEY_PRESS = "key_press"
    KEY_RELEASE = "key_release"
    KEY_TAP = "key_tap"
    MOUSE_MOVE = "mouse_move"
    MOUSE_CLICK = "mouse_click"
    MOUSE_DOWN = "mouse_down"
    MOUSE_UP = "mouse_up"
    STOP = "stop"


#: Parameters each kind requires, used for validation and documentation.
REQUIRED_PARAMETERS: Mapping[ActionKind, tuple[str, ...]] = {
    ActionKind.NOOP: (),
    ActionKind.KEY_PRESS: ("key",),
    ActionKind.KEY_RELEASE: ("key",),
    ActionKind.KEY_TAP: ("key",),
    ActionKind.MOUSE_MOVE: ("dx", "dy"),
    ActionKind.MOUSE_CLICK: ("button",),
    ActionKind.MOUSE_DOWN: ("button",),
    ActionKind.MOUSE_UP: ("button",),
    ActionKind.STOP: (),
}


@dataclass(frozen=True)
class Action:
    """A requested action, before any safety decision has been made.

    ``intended_duration`` is what the caller *wants*; the primitives clamp it to
    ``config.max_key_hold_seconds``, and telemetry records both.
    """

    kind: ActionKind
    parameters: Mapping[str, Any] = field(default_factory=dict)
    intended_duration: float = 0.0
    created_at: float = field(default_factory=time.time)
    action_id: str = field(default_factory=lambda: uuid4().hex[:12])

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ActionKind(self.kind))
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "intended_duration", float(self.intended_duration))
        object.__setattr__(self, "created_at", float(self.created_at))

    # -- constructors -----------------------------------------------------

    @classmethod
    def noop(cls) -> "Action":
        """Do nothing this step. The default for the V0 decision policy."""
        return cls(ActionKind.NOOP)

    @classmethod
    def key_press(cls, key: str) -> "Action":
        """Hold a key down until released."""
        return cls(ActionKind.KEY_PRESS, {"key": normalise_key(key)})

    @classmethod
    def key_release(cls, key: str) -> "Action":
        """Release a held key."""
        return cls(ActionKind.KEY_RELEASE, {"key": normalise_key(key)})

    @classmethod
    def key_tap(cls, key: str, hold_seconds: float = 0.06) -> "Action":
        """Press and release a key after a short hold."""
        return cls(ActionKind.KEY_TAP, {"key": normalise_key(key), "hold_seconds": hold_seconds}, hold_seconds)

    @classmethod
    def mouse_move(cls, dx: int, dy: int) -> "Action":
        """Move the mouse by a relative delta."""
        return cls(ActionKind.MOUSE_MOVE, {"dx": dx, "dy": dy})

    @classmethod
    def mouse_click(cls, button: str = "left", hold_seconds: float = 0.05) -> "Action":
        """Press and release a mouse button."""
        return cls(ActionKind.MOUSE_CLICK, {"button": button, "hold_seconds": hold_seconds}, hold_seconds)

    @classmethod
    def mouse_down(cls, button: str = "left") -> "Action":
        """Hold a mouse button down."""
        return cls(ActionKind.MOUSE_DOWN, {"button": button})

    @classmethod
    def mouse_up(cls, button: str = "left") -> "Action":
        """Release a mouse button."""
        return cls(ActionKind.MOUSE_UP, {"button": button})

    @classmethod
    def stop(cls, reason: str = "policy requested stop") -> "Action":
        """Ask the loop to stop. Not an input event."""
        return cls(ActionKind.STOP, {"reason": reason})

    # -- reporting --------------------------------------------------------

    @property
    def is_input(self) -> bool:
        """True when this action would inject keyboard or mouse input."""
        return self.kind not in (ActionKind.NOOP, ActionKind.STOP)

    def describe(self) -> str:
        """Return a short human-readable description for logs and telemetry."""
        if self.kind is ActionKind.NOOP:
            return "noop"
        if self.kind is ActionKind.STOP:
            return f"stop ({self.parameters.get('reason', '')})"
        if self.kind is ActionKind.KEY_TAP:
            return f"tap {self.parameters.get('key')} for {self.parameters.get('hold_seconds', 0):g}s"
        if self.kind is ActionKind.MOUSE_MOVE:
            return f"move mouse by ({self.parameters.get('dx')}, {self.parameters.get('dy')})"
        if self.kind is ActionKind.MOUSE_CLICK:
            return f"click {self.parameters.get('button', 'left')}"
        if self.kind in (ActionKind.KEY_PRESS, ActionKind.KEY_RELEASE):
            verb = "press" if self.kind is ActionKind.KEY_PRESS else "release"
            return f"{verb} {self.parameters.get('key')}"
        verb = "hold" if self.kind is ActionKind.MOUSE_DOWN else "release"
        return f"{verb} mouse {self.parameters.get('button', 'left')}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "action_id": self.action_id,
            "kind": self.kind.value,
            "parameters": dict(self.parameters),
            "intended_duration": self.intended_duration,
            "created_at": self.created_at,
            "description": self.describe(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Action":
        """Rebuild an action from its serialised form."""
        return cls(
            kind=ActionKind(payload["kind"]),
            parameters=dict(payload.get("parameters", {})),
            intended_duration=float(payload.get("intended_duration", 0.0)),
            created_at=float(payload.get("created_at", 0.0)),
            action_id=str(payload.get("action_id", uuid4().hex[:12])),
        )


@dataclass(frozen=True)
class ActionResult:
    """What actually happened when an action was attempted."""

    action_id: str
    kind: ActionKind
    attempted: bool
    executed: bool
    blocked_reason: str | None = None
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    description: str = ""

    @property
    def duration(self) -> float:
        """Wall-clock seconds the attempt took."""
        return max(0.0, self.finished_at - self.started_at)

    @property
    def ok(self) -> bool:
        """True when the action was not blocked and did not fail.

        A no-op counts as ok: there was nothing to inject, and nothing went
        wrong. ``ok`` is about the absence of a refusal or an error, not about
        whether input reached the game - that is what ``executed`` records.
        """
        return self.blocked_reason is None and self.error is None

    @property
    def was_blocked(self) -> bool:
        """True when the safety guard refused the action."""
        return self.blocked_reason is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "action_id": self.action_id,
            "kind": self.kind.value,
            "description": self.description,
            "attempted": self.attempted,
            "executed": self.executed,
            "blocked_reason": self.blocked_reason,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration": self.duration,
        }


def validate_action(action: Action) -> str | None:
    """Return a human-readable problem with ``action``, or ``None`` if it is fine.

    Pure and hardware-free, so the whole action vocabulary can be tested without
    touching a keyboard or a game.
    """
    if not isinstance(action, Action):
        return f"expected an Action, got {type(action).__name__}"
    try:
        kind = ActionKind(action.kind)
    except ValueError:
        return f"unknown action kind {action.kind!r}"

    parameters = action.parameters
    missing = [name for name in REQUIRED_PARAMETERS[kind] if name not in parameters]
    if missing:
        return f"{kind.value} is missing required parameter(s): {', '.join(missing)}"

    if action.intended_duration < 0:
        return f"intended_duration must not be negative, got {action.intended_duration}"

    if kind in (ActionKind.KEY_PRESS, ActionKind.KEY_RELEASE, ActionKind.KEY_TAP):
        key = parameters["key"]
        if not is_known_key(key):
            return f"unknown key {key!r}"

    if kind is ActionKind.KEY_TAP:
        hold = parameters.get("hold_seconds", 0.06)
        if not isinstance(hold, (int, float)) or isinstance(hold, bool) or hold < 0:
            return f"hold_seconds must be a non-negative number, got {hold!r}"

    if kind is ActionKind.MOUSE_MOVE:
        for axis in ("dx", "dy"):
            value = parameters[axis]
            if isinstance(value, bool) or not isinstance(value, int):
                return f"{axis} must be an integer, got {value!r}"

    if kind in (ActionKind.MOUSE_CLICK, ActionKind.MOUSE_DOWN, ActionKind.MOUSE_UP):
        button = parameters["button"]
        if button not in MOUSE_BUTTONS:
            return f"unknown mouse button {button!r}"

    if kind is ActionKind.MOUSE_CLICK:
        hold = parameters.get("hold_seconds", 0.05)
        if not isinstance(hold, (int, float)) or isinstance(hold, bool) or hold < 0:
            return f"hold_seconds must be a non-negative number, got {hold!r}"

    return None


class ActionExecutor:
    """Runs actions through the keyboard and mouse primitives.

    Every input call passes through the safety guard inside those primitives, so
    a refusal surfaces here as ``blocked_reason`` rather than an exception.
    """

    def __init__(
        self,
        keyboard: Keyboard,
        mouse: Mouse,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._keyboard = keyboard
        self._mouse = mouse
        self._clock = clock

    def execute(self, action: Action) -> ActionResult:
        """Attempt ``action`` and report exactly what happened."""
        started = self._clock()
        problem = validate_action(action)
        if problem is not None:
            return ActionResult(
                action_id=action.action_id,
                kind=action.kind,
                attempted=False,
                executed=False,
                error=problem,
                started_at=started,
                finished_at=self._clock(),
                description=action.describe(),
            )

        if action.kind in (ActionKind.NOOP, ActionKind.STOP):
            # Nothing was injected and nothing failed: neither attempted nor
            # executed. Reporting executed=True here would let a run that only
            # no-op'd claim it performed actions.
            return ActionResult(
                action_id=action.action_id,
                kind=action.kind,
                attempted=False,
                executed=False,
                started_at=started,
                finished_at=self._clock(),
                description=action.describe(),
            )

        try:
            self._dispatch(action)
        except InputBlocked as exc:
            return ActionResult(
                action_id=action.action_id,
                kind=action.kind,
                attempted=True,
                executed=False,
                blocked_reason=exc.reason,
                started_at=started,
                finished_at=self._clock(),
                description=action.describe(),
            )
        except (ControlError, ValueError, TypeError) as exc:
            return ActionResult(
                action_id=action.action_id,
                kind=action.kind,
                attempted=True,
                executed=False,
                error=f"{type(exc).__name__}: {exc}",
                started_at=started,
                finished_at=self._clock(),
                description=action.describe(),
            )
        return ActionResult(
            action_id=action.action_id,
            kind=action.kind,
            attempted=True,
            executed=True,
            started_at=started,
            finished_at=self._clock(),
            description=action.describe(),
        )

    def _dispatch(self, action: Action) -> None:
        parameters = action.parameters
        kind = action.kind
        if kind is ActionKind.KEY_PRESS:
            self._keyboard.press(parameters["key"])
        elif kind is ActionKind.KEY_RELEASE:
            self._keyboard.release(parameters["key"])
        elif kind is ActionKind.KEY_TAP:
            self._keyboard.tap(parameters["key"], parameters.get("hold_seconds", 0.06))
        elif kind is ActionKind.MOUSE_MOVE:
            self._mouse.move_relative(parameters["dx"], parameters["dy"])
        elif kind is ActionKind.MOUSE_CLICK:
            self._mouse.click(parameters.get("button", "left"), parameters.get("hold_seconds", 0.05))
        elif kind is ActionKind.MOUSE_DOWN:
            self._mouse.press(parameters.get("button", "left"))
        elif kind is ActionKind.MOUSE_UP:
            self._mouse.release(parameters.get("button", "left"))
        else:  # pragma: no cover - NOOP/STOP are handled before dispatch
            raise ControlError(f"action kind {kind.value!r} is not dispatchable")
