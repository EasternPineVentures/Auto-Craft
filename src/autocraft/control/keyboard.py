"""Keyboard primitives: named keys in, safe press/release out.

This module owns two things and nothing else:

* the mapping from readable key names (``"w"``, ``"space"``, ``"f8"``) to Windows
  virtual-key codes, so no magic numbers leak into the agent, and
* a thin facade that routes every key event through the safety guard.

It contains no policy. Whether a key *should* be pressed is the decision layer's
job; whether it *may* be pressed is the guard's.
"""

from __future__ import annotations

import time
from typing import Callable

from ..config import Config
from .errors import ControlError, InputBlocked, UnknownKeyError
from .keymap import (
    KEY_NAME_TO_VK,
    VK_TO_KEY_NAME,
    is_known_key,
    known_key_names,
    normalise_key,
    virtual_key_for,
)
from .safety import SafetyGuard
from .win32_input import InputBackend

__all__ = [
    "DEFAULT_TAP_SECONDS",
    "KEY_NAME_TO_VK",
    "Keyboard",
    "VK_TO_KEY_NAME",
    "ControlError",
    "InputBlocked",
    "UnknownKeyError",
    "is_known_key",
    "known_key_names",
    "normalise_key",
    "virtual_key_for",
]

#: Default press duration for a tap. Long enough for a game's input poll to see
#: it, short enough that a stuck tap cannot do much.
DEFAULT_TAP_SECONDS = 0.06


def _clamp_hold(hold_seconds: float, limit: float) -> float:
    """Clamp a requested hold to the configured cap.

    Raises:
        ValueError: If ``hold_seconds`` is negative.
    """
    if hold_seconds < 0:
        raise ValueError(f"hold_seconds must not be negative, got {hold_seconds}")
    return min(float(hold_seconds), limit)


class Keyboard:
    """Validated, safety-guarded keyboard events.

    Release paths are deliberately *not* subject to the safety guard: refusing to
    release a held key would be the unsafe behaviour, so ``release`` always goes
    straight to the backend.
    """

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
    def held_keys(self) -> tuple[str, ...]:
        """Names of keys currently believed to be held down."""
        return self._guard.held_keys

    def press(self, key: str) -> None:
        """Press and hold a key.

        Raises:
            UnknownKeyError: If the key name is not recognised.
            InputBlocked: If the safety guard refuses the action.
            ControlError: If the backend fails to inject the key. The key is not
                registered as held in that case, so it is never "released" later.
        """
        name = normalise_key(key)
        vk = virtual_key_for(name)
        self._guard.authorize_or_raise(f"press {name}")
        self._guard.wait_for_rate_limit()
        try:
            self._backend.key_down(vk)
        except ControlError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise backend failures
            raise ControlError(f"failed to press key {name!r}: {exc}") from exc
        self._guard.note_action()
        self._guard.register_key_down(name)

    def release(self, key: str) -> None:
        """Release a key. Never blocked by safety, and safe when not held.

        Raises:
            UnknownKeyError: If the key name is not recognised.
            ControlError: If the backend fails. The key stays tracked as held so
                ``release_all`` retries it, which is the safe direction to fail.
        """
        name = normalise_key(key)
        vk = virtual_key_for(name)
        try:
            self._backend.key_up(vk)
        except ControlError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise backend failures
            raise ControlError(f"failed to release key {name!r}: {exc}") from exc
        self._guard.register_key_up(name)

    def tap(self, key: str, hold_seconds: float = DEFAULT_TAP_SECONDS) -> float:
        """Press a key, hold briefly, release. Returns the actual hold time.

        The hold is clamped to ``config.max_key_hold_seconds`` so a caller cannot
        accidentally exceed the configured bound even before the guard's periodic
        hold-limit sweep runs.
        """
        hold = _clamp_hold(hold_seconds, self._config.max_key_hold_seconds)
        self.press(key)
        try:
            if hold > 0:
                self._sleep(hold)
        except BaseException:
            # Release on the way out, but never let a failed release replace the
            # real reason we are unwinding (a KeyboardInterrupt must stay one).
            self._release_quietly(key)
            raise
        self.release(key)
        return hold

    def _release_quietly(self, key: str) -> None:
        """Best-effort release used while an exception is already propagating."""
        try:
            self.release(key)
        except ControlError:
            # The guard still tracks the key, so release_all will retry it.
            pass

    def release_all(self) -> tuple[str, ...]:
        """Release every key the guard believes is held."""
        return self._guard.release_keys()
