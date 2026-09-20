"""Readable key names to Windows virtual-key codes.

Kept separate from :mod:`autocraft.control.keyboard` so that configuration
validation and the safety guard can resolve key names without importing the
keyboard facade (which would create an import cycle).
"""

from __future__ import annotations

from typing import Mapping

from .errors import UnknownKeyError

__all__ = [
    "KEY_NAME_TO_VK",
    "VK_TO_KEY_NAME",
    "is_known_key",
    "known_key_names",
    "normalise_key",
    "virtual_key_for",
]

_ALIASES: dict[str, str] = {
    "esc": "escape",
    "return": "enter",
    "spacebar": "space",
    "ctrl": "control",
    "lctrl": "lcontrol",
    "rctrl": "rcontrol",
    "menu": "alt",
    "altgr": "ralt",
    "win": "lwin",
    "super": "lwin",
    "meta": "lwin",
    "del": "delete",
    "ins": "insert",
    "pgup": "pageup",
    "pgdn": "pagedown",
    "prtsc": "printscreen",
    "caps": "capslock",
    "num": "numlock",
    "scroll": "scrolllock",
    "apostrophe": "quote",
    "grave": "backtick",
    "tilde": "backtick",
}


def _build_key_map() -> dict[str, int]:
    keys: dict[str, int] = {}

    for index, letter in enumerate("abcdefghijklmnopqrstuvwxyz"):
        keys[letter] = 0x41 + index
    for digit in range(10):
        keys[str(digit)] = 0x30 + digit
    for number in range(1, 25):
        keys[f"f{number}"] = 0x6F + number

    keys.update(
        {
            "backspace": 0x08,
            "tab": 0x09,
            "enter": 0x0D,
            "shift": 0x10,
            "lshift": 0xA0,
            "rshift": 0xA1,
            "control": 0x11,
            "lcontrol": 0xA2,
            "rcontrol": 0xA3,
            "alt": 0x12,
            "lalt": 0xA4,
            "ralt": 0xA5,
            "pause": 0x13,
            "capslock": 0x14,
            "escape": 0x1B,
            "space": 0x20,
            "pageup": 0x21,
            "pagedown": 0x22,
            "end": 0x23,
            "home": 0x24,
            "left": 0x25,
            "up": 0x26,
            "right": 0x27,
            "down": 0x28,
            "printscreen": 0x2C,
            "insert": 0x2D,
            "delete": 0x2E,
            "lwin": 0x5B,
            "rwin": 0x5C,
            "apps": 0x5D,
            "numpad0": 0x60,
            "numpad1": 0x61,
            "numpad2": 0x62,
            "numpad3": 0x63,
            "numpad4": 0x64,
            "numpad5": 0x65,
            "numpad6": 0x66,
            "numpad7": 0x67,
            "numpad8": 0x68,
            "numpad9": 0x69,
            "multiply": 0x6A,
            "add": 0x6B,
            "subtract": 0x6D,
            "decimal": 0x6E,
            "divide": 0x6F,
            "numlock": 0x90,
            "scrolllock": 0x91,
            "semicolon": 0xBA,
            "equals": 0xBB,
            "comma": 0xBC,
            "minus": 0xBD,
            "period": 0xBE,
            "slash": 0xBF,
            "backtick": 0xC0,
            "bracketleft": 0xDB,
            "backslash": 0xDC,
            "bracketright": 0xDD,
            "quote": 0xDE,
        }
    )
    for alias, canonical in _ALIASES.items():
        if canonical in keys:
            keys.setdefault(alias, keys[canonical])
    return keys


#: Readable key name -> Windows virtual-key code.
KEY_NAME_TO_VK: Mapping[str, int] = _build_key_map()

#: Reverse map used when reporting which keys are held.
VK_TO_KEY_NAME: Mapping[int, str] = {vk: name for name, vk in KEY_NAME_TO_VK.items()}


def normalise_key(name: str) -> str:
    """Return the canonical lowercase name for a key, applying aliases.

    Raises:
        UnknownKeyError: If the name is not recognised.
    """
    if not isinstance(name, str):
        raise UnknownKeyError(f"key name must be a string, got {type(name).__name__}")
    candidate = _ALIASES.get(name.strip().lower(), name.strip().lower())
    if candidate not in KEY_NAME_TO_VK:
        raise UnknownKeyError(
            f"unknown key {name!r}; run 'python -m autocraft keys' to list supported names"
        )
    return candidate


def virtual_key_for(name: str) -> int:
    """Return the virtual-key code for a key name."""
    return KEY_NAME_TO_VK[normalise_key(name)]


def is_known_key(name: object) -> bool:
    """True when ``name`` resolves to a supported key."""
    if not isinstance(name, str):
        return False
    try:
        normalise_key(name)
    except UnknownKeyError:
        return False
    return True


def known_key_names() -> tuple[str, ...]:
    """Return every supported key name, sorted."""
    return tuple(sorted(KEY_NAME_TO_VK))
