"""Control layer: the only way AutoCraft can affect the world.

Actuators are keyboard and mouse events, nothing else. No teleportation, no
server commands, no file writes into the game. Every event passes through
:class:`~autocraft.control.safety.SafetyGuard`, which owns the foreground lock,
the emergency stop and the release guarantee.
"""

from __future__ import annotations

from .errors import ControlError, InputBlocked, InvalidMouseDelta, UnknownKeyError
from .keyboard import DEFAULT_TAP_SECONDS, Keyboard
from .keymap import (
    KEY_NAME_TO_VK,
    VK_TO_KEY_NAME,
    is_known_key,
    known_key_names,
    normalise_key,
    virtual_key_for,
)
from .mouse import DEFAULT_CLICK_SECONDS, MOUSE_BUTTONS, Mouse
from .safety import SafetyDecision, SafetyEvent, SafetyGuard
from .win32_input import (
    InputBackend,
    InputBackendError,
    UnsupportedPlatformError,
    Win32InputBackend,
)

__all__ = [
    "DEFAULT_CLICK_SECONDS",
    "DEFAULT_TAP_SECONDS",
    "KEY_NAME_TO_VK",
    "MOUSE_BUTTONS",
    "VK_TO_KEY_NAME",
    "ControlError",
    "InputBackend",
    "InputBackendError",
    "InputBlocked",
    "InvalidMouseDelta",
    "Keyboard",
    "Mouse",
    "SafetyDecision",
    "SafetyEvent",
    "SafetyGuard",
    "UnknownKeyError",
    "UnsupportedPlatformError",
    "Win32InputBackend",
    "is_known_key",
    "known_key_names",
    "normalise_key",
    "virtual_key_for",
]
