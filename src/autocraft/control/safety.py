"""The safety guard: the single authority on whether input may be injected.

Every requirement in the V0 spec that is about *not* doing damage lives here:

* **Target-window lock.** Input is refused unless the intended game window is the
  foreground window. This is the check that stops AutoCraft from typing into a
  chat window that happened to steal focus.
* **Emergency stop.** A configurable key is polled with ``GetAsyncKeyState``. It
  is deliberately *not* registered as a global hotkey: polling does not consume
  the key, so the game still receives it, and AutoCraft never steals input from
  anything else. Pressing it releases every held key and button and stops the
  loop.
* **Release guarantee.** Every key and button is tracked by name. On a hold-limit
  breach, an emergency stop, an exception, Ctrl+C, or process exit, the guard
  attempts to release everything it believes is held.
* **Rate limiting.** Injected events are spaced by ``min_action_interval``.
* **Bounded failure.** Consecutive refusals are counted, and the loop stops once
  the configured budget is exhausted instead of retrying forever.

The guard does not know about windows, frames or the game. It is given one
callable that answers "is the target foreground?" and one backend to release
through, which keeps it testable with fakes and honest about its single job.
"""

from __future__ import annotations

import atexit
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..config import Config
from .errors import InputBlocked
from .keymap import KEY_NAME_TO_VK, normalise_key, virtual_key_for
from .win32_input import InputBackend

__all__ = [
    "SafetyDecision",
    "SafetyEvent",
    "SafetyGuard",
]

#: Safety event kinds, kept as plain strings so telemetry stays readable.
KIND_EMERGENCY_STOP = "emergency_stop"
KIND_FOCUS_LOST = "focus_lost"
KIND_HOLD_LIMIT = "hold_limit"
KIND_RELEASE = "release"
KIND_RELEASE_FAILED = "release_failed"
KIND_BLOCKED = "blocked"
KIND_ERROR = "error"
KIND_LIFECYCLE = "lifecycle"


@dataclass(frozen=True)
class SafetyDecision:
    """The guard's answer to "may this action be injected?"."""

    allowed: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {"allowed": self.allowed, "reason": self.reason}


@dataclass(frozen=True)
class SafetyEvent:
    """One notable safety-relevant moment, recorded for telemetry."""

    timestamp: float
    kind: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {"timestamp": self.timestamp, "kind": self.kind, "detail": self.detail}


class SafetyGuard:
    """Tracks held input, enforces the safety policy and releases on demand."""

    def __init__(
        self,
        config: Config,
        backend: InputBackend,
        *,
        target_is_foreground: Callable[[], bool],
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
        key_probe: Callable[[int], bool] | None = None,
    ) -> None:
        """
        Args:
            config: Effective configuration, including every safety limit.
            backend: The input backend used to release held keys and buttons.
            target_is_foreground: Callable returning True when the intended game
                window currently has focus. Called on every authorisation.
            clock: Monotonic clock, injectable so tests need no real time.
            wall_clock: Wall clock used for event timestamps.
            sleeper: Sleep function, injectable so rate limiting is testable.
            key_probe: Overrides the backend's key-state probe. Used for the
                emergency-stop poll and for tests.
        """
        self._config = config
        self._backend = backend
        self._target_is_foreground = target_is_foreground
        self._clock = clock
        self._wall_clock = wall_clock
        self._sleep = sleeper
        self._key_probe = key_probe if key_probe is not None else backend.is_key_pressed

        self._emergency_stop_name = normalise_key(config.emergency_stop_key)
        self._emergency_stop_vk = virtual_key_for(config.emergency_stop_key)

        self._held_keys: dict[str, float] = {}
        self._held_buttons: dict[str, float] = {}
        self._events: list[SafetyEvent] = []

        self._stopped = False
        self._stop_reason = ""
        self._last_action_at = float("-inf")
        self._blocked_streak = 0
        self._last_block_reason = ""
        self._atexit_registered = False

    # -- introspection ----------------------------------------------------

    @property
    def config(self) -> Config:
        """The configuration this guard enforces."""
        return self._config

    @property
    def backend(self) -> InputBackend:
        """The input backend used for releases and key-state probes."""
        return self._backend

    @property
    def emergency_stop_key(self) -> str:
        """Canonical name of the emergency-stop key."""
        return self._emergency_stop_name

    @property
    def emergency_stop_vk(self) -> int:
        """Virtual-key code of the emergency-stop key."""
        return self._emergency_stop_vk

    @property
    def held_keys(self) -> tuple[str, ...]:
        """Names of keys currently believed to be held, in press order."""
        return tuple(self._held_keys)

    @property
    def held_buttons(self) -> tuple[str, ...]:
        """Names of mouse buttons currently believed to be held."""
        return tuple(self._held_buttons)

    @property
    def events(self) -> tuple[SafetyEvent, ...]:
        """Every safety event recorded so far."""
        return tuple(self._events)

    @property
    def stop_requested(self) -> bool:
        """True once an emergency stop has been triggered."""
        return self._stopped

    @property
    def stop_reason(self) -> str:
        """Why the emergency stop fired, if it did."""
        return self._stop_reason

    @property
    def blocked_streak(self) -> int:
        """Number of consecutive refusals or failures."""
        return self._blocked_streak

    @property
    def last_block_reason(self) -> str:
        """Reason for the most recent refusal or failure."""
        return self._last_block_reason

    @property
    def should_stop_for_blocks(self) -> bool:
        """True when the consecutive-block budget is exhausted."""
        return self._blocked_streak >= self._config.max_consecutive_blocks

    # -- authorisation ----------------------------------------------------

    def authorize(self, action: str) -> SafetyDecision:
        """Decide whether ``action`` may be injected right now.

        Only two things can refuse input: an active emergency stop, and loss of
        the foreground lock. Rate limiting deliberately does not refuse - it
        waits - because a refusal would otherwise consume the block budget for a
        purely timing-related reason.
        """
        if self._stopped:
            return SafetyDecision(False, f"emergency stop active ({self._stop_reason})")
        if self._config.require_foreground:
            try:
                focused = bool(self._target_is_foreground())
            except Exception as exc:  # noqa: BLE001 - fail closed on any probe error
                self._record(KIND_ERROR, f"foreground probe failed: {exc}")
                return SafetyDecision(False, f"foreground probe failed: {exc}")
            if not focused:
                self._record(KIND_FOCUS_LOST, f"refused to {action}: target window is not foreground")
                return SafetyDecision(
                    False,
                    "target window is not the foreground window (foreground lock active)",
                )
        return SafetyDecision(True)

    def authorize_or_raise(self, action: str) -> None:
        """Raise :class:`InputBlocked` unless ``action`` is allowed."""
        decision = self.authorize(action)
        if not decision.allowed:
            raise InputBlocked(decision.reason, action=action)

    # -- pacing -----------------------------------------------------------

    def wait_for_rate_limit(self) -> float:
        """Sleep until at least ``min_action_interval`` has passed. Returns slept time."""
        interval = self._config.min_action_interval
        if interval <= 0:
            return 0.0
        remaining = interval - (self._clock() - self._last_action_at)
        if remaining <= 0:
            return 0.0
        self._sleep(remaining)
        return remaining

    def note_action(self) -> None:
        """Record that an event was just injected, for rate limiting."""
        self._last_action_at = self._clock()

    # -- held-state bookkeeping -------------------------------------------

    def register_key_down(self, name: str) -> None:
        """Note that a key is now held."""
        self._held_keys.setdefault(name, self._clock())

    def register_key_up(self, name: str) -> None:
        """Note that a key is no longer held."""
        self._held_keys.pop(name, None)

    def register_button_down(self, button: str) -> None:
        """Note that a mouse button is now held."""
        self._held_buttons.setdefault(button, self._clock())

    def register_button_up(self, button: str) -> None:
        """Note that a mouse button is no longer held."""
        self._held_buttons.pop(button, None)

    # -- enforcement ------------------------------------------------------

    def enforce_hold_limits(self) -> tuple[str, ...]:
        """Release anything held longer than ``max_key_hold_seconds``.

        This is the backstop for a key-up event that never arrived, for example
        because the game crashed mid-hold.
        """
        now = self._clock()
        limit = self._config.max_key_hold_seconds
        released: list[str] = []

        for name, pressed_at in list(self._held_keys.items()):
            if now - pressed_at >= limit:
                if self._force_release_key(name, f"held longer than {limit:g}s"):
                    released.append(name)
        for button, pressed_at in list(self._held_buttons.items()):
            if now - pressed_at >= limit:
                if self._force_release_button(button, f"held longer than {limit:g}s"):
                    released.append(f"mouse:{button}")
        return tuple(released)

    def release_keys(self, reason: str = "release requested") -> tuple[str, ...]:
        """Release every held key, returning only the names that came up.

        A key whose backend release fails is *not* reported here: the caller
        must not be told a key was released when it may still be physically
        down. It stays tracked, and the next ``release_all`` retries it.
        """
        released: list[str] = []
        for name in list(self._held_keys):
            if self._force_release_key(name, reason):
                released.append(name)
        return tuple(released)

    def release_buttons(self, reason: str = "release requested") -> tuple[str, ...]:
        """Release every held mouse button, returning only the ones that came up.

        Failing releases are excluded for the same reason as ``release_keys``:
        a report of "released" that did not happen is worse than a report of
        nothing, because it hides the input that is still down.
        """
        released: list[str] = []
        for button in list(self._held_buttons):
            if self._force_release_button(button, reason):
                released.append(button)
        return tuple(released)

    def release_all(self, reason: str = "release_all") -> tuple[str, ...]:
        """Release every held key and mouse button. Safe to call at any time."""
        released = list(self.release_keys(reason)) + [
            f"mouse:{button}" for button in self.release_buttons(reason)
        ]
        if released:
            self._record(KIND_RELEASE, f"{reason}: released {', '.join(released)}")
        return tuple(released)

    def _force_release_key(self, name: str, reason: str) -> bool:
        """Release one key, forgetting it only once the backend accepts it.

        The held record is the guard's only memory of what may still be
        physically down. Removing it *before* the backend call would mean a
        failed release silently strands the key: ``release_all`` retries
        whatever is still tracked, so forgetting a key the backend refused to
        release makes the retry impossible and the key stays down forever.

        Returns True when the key was released, False when it was not.
        """
        try:
            virtual_key = KEY_NAME_TO_VK[name]
        except KeyError:
            # Not a name the backend can act on, so no retry could ever
            # succeed and no such key can be physically held. Drop it rather
            # than tracking an entry that can never be released.
            self._held_keys.pop(name, None)
            self._record(
                KIND_ERROR, f"cannot release unknown key {name!r}; dropped from held state"
            )
            return False
        try:
            self._backend.key_up(virtual_key)
        except Exception as exc:  # noqa: BLE001 - release attempts must not raise
            self._record(KIND_RELEASE_FAILED, f"key_up {name} failed: {exc}")
            return False
        self._held_keys.pop(name, None)
        self._record(
            KIND_HOLD_LIMIT if "held longer" in reason else KIND_RELEASE,
            f"released key {name} ({reason})",
        )
        return True

    def _force_release_button(self, button: str, reason: str) -> bool:
        """Release one mouse button, forgetting it only on success.

        Same invariant as ``_force_release_key``: a button the backend could
        not release stays tracked so a later ``release_all`` retries it.

        Returns True when the button was released, False when it was not.
        """
        try:
            self._backend.mouse_button_up(button)
        except Exception as exc:  # noqa: BLE001 - release attempts must not raise
            self._record(KIND_RELEASE_FAILED, f"mouse_button_up {button} failed: {exc}")
            return False
        self._held_buttons.pop(button, None)
        self._record(
            KIND_HOLD_LIMIT if "held longer" in reason else KIND_RELEASE,
            f"released mouse button {button} ({reason})",
        )
        return True

    # -- emergency stop ---------------------------------------------------

    def check_emergency_stop(self) -> bool:
        """Poll the emergency-stop key and act on it.

        Returns True once an emergency stop has fired, including on later calls,
        so callers can simply test the result each iteration.
        """
        if self._stopped:
            return True
        try:
            pressed = bool(self._key_probe(self._emergency_stop_vk))
        except Exception as exc:  # noqa: BLE001 - never crash the safety path
            self._record(KIND_ERROR, f"emergency-stop probe failed: {exc}")
            return False
        if pressed:
            self.trigger_emergency_stop(f"{self._emergency_stop_name} pressed")
            return True
        return False

    def trigger_emergency_stop(self, reason: str) -> tuple[str, ...]:
        """Stop everything: latch the stop flag and release all held input."""
        if not self._stopped:
            self._stopped = True
            self._stop_reason = reason
            self._record(KIND_EMERGENCY_STOP, reason)
        return self.release_all("emergency stop")

    # -- failure budget ---------------------------------------------------

    def record_block(self, reason: str) -> int:
        """Note a refused or failed action and return the new streak length."""
        self._blocked_streak += 1
        self._last_block_reason = reason
        self._record(KIND_BLOCKED, f"streak {self._blocked_streak}: {reason}")
        return self._blocked_streak

    def record_success(self) -> None:
        """Reset the consecutive-block counter after a successful action."""
        self._blocked_streak = 0
        self._last_block_reason = ""

    # -- lifecycle --------------------------------------------------------

    def install_atexit(self) -> None:
        """Release held input if the process exits without unwinding."""
        if not self._atexit_registered:
            atexit.register(self._atexit_release)
            self._atexit_registered = True

    def _atexit_release(self) -> None:  # pragma: no cover - exercised via atexit
        self.release_all("process exit")

    def shutdown(self, reason: str = "shutdown") -> tuple[str, ...]:
        """Release held input and record that the guard is done."""
        released = self.release_all(reason)
        self._record(KIND_LIFECYCLE, reason)
        return released

    def reset(self) -> None:
        """Clear all state, including a latched emergency stop.

        Intended for tests and for starting a fresh run in the same process.
        """
        self._held_keys.clear()
        self._held_buttons.clear()
        self._events.clear()
        self._stopped = False
        self._stop_reason = ""
        self._last_action_at = float("-inf")
        self._blocked_streak = 0
        self._last_block_reason = ""

    def _record(self, kind: str, detail: str) -> SafetyEvent:
        event = SafetyEvent(timestamp=self._wall_clock(), kind=kind, detail=detail)
        self._events.append(event)
        return event

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the guard's state."""
        return {
            "emergency_stop_key": self._emergency_stop_name,
            "require_foreground": self._config.require_foreground,
            "max_key_hold_seconds": self._config.max_key_hold_seconds,
            "max_mouse_delta": self._config.max_mouse_delta,
            "min_action_interval": self._config.min_action_interval,
            "max_consecutive_blocks": self._config.max_consecutive_blocks,
            "stop_requested": self._stopped,
            "stop_reason": self._stop_reason,
            "held_keys": list(self._held_keys),
            "held_buttons": list(self._held_buttons),
            "blocked_streak": self._blocked_streak,
            "last_block_reason": self._last_block_reason,
            "event_count": len(self._events),
        }

    def event_dicts(self) -> list[dict[str, Any]]:
        """Return every recorded safety event as JSON-serialisable dicts."""
        return [event.to_dict() for event in self._events]

    def __enter__(self) -> "SafetyGuard":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.shutdown("context exit")


def event_kinds() -> Mapping[str, str]:
    """Return the safety event kinds, for documentation and tests."""
    return {
        "emergency_stop": KIND_EMERGENCY_STOP,
        "focus_lost": KIND_FOCUS_LOST,
        "hold_limit": KIND_HOLD_LIMIT,
        "release": KIND_RELEASE,
        "release_failed": KIND_RELEASE_FAILED,
        "blocked": KIND_BLOCKED,
        "error": KIND_ERROR,
        "lifecycle": KIND_LIFECYCLE,
    }
