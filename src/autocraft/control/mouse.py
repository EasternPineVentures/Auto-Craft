"""Mouse primitives: relative movement and button events, safety-guarded.

First-person camera control needs *relative* motion, not absolute positioning:
the game consumes deltas and applies its own sensitivity. Absolute cursor
placement would also fight with the game's cursor capture. So the only movement
primitive here is :meth:`Mouse.move_relative`.

One honest limitation is documented rather than hidden: Windows pointer
acceleration still applies to the injected stream, so a given delta does not
produce a fixed in-game rotation. Measuring that mapping is the job of the next
milestone, not this one.
"""

from __future__ import annotations

import time
from typing import Callable

from ..config import Config
from .errors import ControlError, InvalidMouseDelta
from .safety import SafetyGuard
from .win32_input import BUTTON_FLAGS, InputBackend

__all__ = ["DEFAULT_CLICK_SECONDS", "MOUSE_BUTTONS", "Mouse"]

#: Mouse button names accepted by the primitives.
MOUSE_BUTTONS: tuple[str, ...] = ("left", "right", "middle", "x1", "x2")

#: Default press duration for a click.
DEFAULT_CLICK_SECONDS = 0.05


class Mouse:
    """Validated, safety-guarded mouse events."""

    def __init__(
        self,
        guard: SafetyGuard,
        backend: InputBackend,
        config: Config,
        *,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._guard = guard
        self._backend = backend
        self._config = config
        self._sleep = sleeper

    @property
    def held_buttons(self) -> tuple[str, ...]:
        """Names of mouse buttons currently believed to be held."""
        return self._guard.held_buttons

    @property
    def max_delta(self) -> int:
        """Largest absolute per-axis delta accepted in one command."""
        return self._config.max_mouse_delta

    def move_relative(self, dx: int, dy: int) -> tuple[int, int]:
        """Move the pointer by a relative delta.

        Raises:
            InvalidMouseDelta: If either axis exceeds ``max_mouse_delta``. The
                delta is rejected rather than clamped: silently changing how far
                the agent asked to look would corrupt any later learning signal.
            InputBlocked: If the safety guard refuses the action.
        """
        dx, dy = _as_int(dx, "dx"), _as_int(dy, "dy")
        limit = self._config.max_mouse_delta
        if abs(dx) > limit or abs(dy) > limit:
            raise InvalidMouseDelta(
                f"relative move ({dx}, {dy}) exceeds max_mouse_delta={limit} per axis"
            )
        self._guard.authorize_or_raise(f"move mouse by ({dx}, {dy})")
        self._guard.wait_for_rate_limit()
        try:
            self._backend.move_relative(dx, dy)
        except ControlError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise backend failures
            raise ControlError(f"failed to move mouse by ({dx}, {dy}): {exc}") from exc
        self._guard.note_action()
        return dx, dy

    def press(self, button: str = "left") -> None:
        """Press and hold a mouse button.

        Raises:
            ControlError: If the button name is unknown or the backend fails.
            InputBlocked: If the safety guard refuses the action.
        """
        name = _normalise_button(button)
        self._guard.authorize_or_raise(f"press mouse {name}")
        self._guard.wait_for_rate_limit()
        try:
            self._backend.mouse_button_down(name)
        except ControlError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise backend failures
            raise ControlError(f"failed to press mouse button {name!r}: {exc}") from exc
        self._guard.note_action()
        self._guard.register_button_down(name)

    def release(self, button: str = "left") -> None:
        """Release a mouse button. Never blocked, safe when not held.

        Raises:
            ControlError: If the button name is unknown or the backend fails. The
                button stays tracked as held so ``release_all`` retries it.
        """
        name = _normalise_button(button)
        try:
            self._backend.mouse_button_up(name)
        except ControlError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise backend failures
            raise ControlError(f"failed to release mouse button {name!r}: {exc}") from exc
        self._guard.register_button_up(name)

    def click(self, button: str = "left", hold_seconds: float = DEFAULT_CLICK_SECONDS) -> float:
        """Press and release a button. Returns the actual hold time."""
        if hold_seconds < 0:
            raise ValueError(f"hold_seconds must not be negative, got {hold_seconds}")
        hold = min(float(hold_seconds), self._config.max_key_hold_seconds)
        self.press(button)
        try:
            if hold > 0:
                self._sleep(hold)
        except BaseException:
            # Never let a failed release replace the real reason we are unwinding.
            self._release_quietly(button)
            raise
        self.release(button)
        return hold

    def _release_quietly(self, button: str) -> None:
        """Best-effort release used while an exception is already propagating."""
        try:
            self.release(button)
        except ControlError:
            # The guard still tracks the button, so release_all will retry it.
            pass

    def release_all(self) -> tuple[str, ...]:
        """Release every mouse button the guard believes is held."""
        return self._guard.release_buttons()


def _normalise_button(button: str) -> str:
    if not isinstance(button, str):
        raise ControlError(f"mouse button must be a string, got {type(button).__name__}")
    name = button.strip().lower()
    if name not in MOUSE_BUTTONS:
        supported = ", ".join(MOUSE_BUTTONS)
        raise ControlError(f"unknown mouse button {button!r}; supported: {supported}")
    if name not in BUTTON_FLAGS:  # pragma: no cover - defensive consistency check
        raise ControlError(f"mouse button {name!r} has no Win32 flag mapping")
    return name


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise InvalidMouseDelta(f"{label} must be an integer, got {value!r}") from exc
    return value
