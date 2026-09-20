"""Raw Windows input injection via ``SendInput``.

Why not a GUI automation library
--------------------------------
First-person games do not consume input through the ordinary window message
queue; they read the keyboard and mouse through Raw Input / DirectInput, and
most of them ignore synthetic ``WM_KEYDOWN`` messages entirely. ``SendInput`` is
the documented way to place events into the same system input stream a real
device writes to, which is what makes it work here. AutoCraft therefore talks to
``SendInput`` directly through ``ctypes`` instead of pulling in a heavier
automation dependency.

Two details that matter for games:

* **Scan codes.** SDL2 and friends often key off the hardware scan code, not the
  virtual-key code. Events are sent with ``KEYEVENTF_SCANCODE`` and translated
  through ``MapVirtualKeyW`` so the game sees what a physical key press looks
  like.
* **Relative mouse motion.** ``MOUSEEVENTF_MOVE`` without ``MOUSEEVENTF_ABSOLUTE``
  produces the relative deltas a mouse-look control needs. Note that Windows
  pointer acceleration still applies to this stream; a future calibration step
  (VISION-001 / LOOK-001) will need to measure that mapping.

Everything here is deliberately dumb: no state, no policy, no safety. The
:class:`~autocraft.control.safety.SafetyGuard` owns those decisions.
"""

from __future__ import annotations

import ctypes
import sys
from typing import Protocol, runtime_checkable

from .errors import ControlError

__all__ = [
    "ButtonCode",
    "InputBackend",
    "InputBackendError",
    "UnsupportedPlatformError",
    "Win32InputBackend",
    "extended_key_vks",
]


class InputBackendError(ControlError):
    """Raised when the operating system rejects an injected event."""


class UnsupportedPlatformError(RuntimeError):
    """Raised when a Win32-only feature is used on another platform."""


# ---------------------------------------------------------------------------
# Win32 constants
# ---------------------------------------------------------------------------

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_XDOWN = 0x0080
MOUSEEVENTF_XUP = 0x0100

XBUTTON1 = 0x0001
XBUTTON2 = 0x0002

MAPVK_VK_TO_VSC = 0

#: Keys whose scan code is shared with another key and that are only
#: distinguishable by the extended-key flag.
_EXTENDED_KEY_VKS: frozenset[int] = frozenset(
    {
        0x21,  # VK_PRIOR    (Page Up)
        0x22,  # VK_NEXT     (Page Down)
        0x23,  # VK_END
        0x24,  # VK_HOME
        0x25,  # VK_LEFT
        0x26,  # VK_UP
        0x27,  # VK_RIGHT
        0x28,  # VK_DOWN
        0x2C,  # VK_SNAPSHOT (Print Screen)
        0x2D,  # VK_INSERT
        0x2E,  # VK_DELETE
        0x6F,  # VK_DIVIDE   (numpad /)
        0x90,  # VK_NUMLOCK
        0xA3,  # VK_RCONTROL
        0xA5,  # VK_RMENU    (right Alt)
    }
)

#: Mouse button name -> ``(down flag, up flag, mouseData)``.
BUTTON_FLAGS: dict[str, tuple[int, int, int]] = {
    "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, 0),
    "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, 0),
    "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP, 0),
    "x1": (MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP, XBUTTON1),
    "x2": (MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP, XBUTTON2),
}


def extended_key_vks() -> frozenset[int]:
    """Virtual keys that require ``KEYEVENTF_EXTENDEDKEY``."""
    return _EXTENDED_KEY_VKS


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class InputBackend(Protocol):
    """The only channel through which AutoCraft can affect the world.

    Implementations must be *dumb*: validate nothing, decide nothing, just place
    the event. Tests supply a recording fake so no real input is ever injected
    during automated runs.
    """

    def key_down(self, virtual_key: int) -> None:
        """Press and hold one key."""
        ...

    def key_up(self, virtual_key: int) -> None:
        """Release one key."""
        ...

    def mouse_button_down(self, button: str) -> None:
        """Press and hold one mouse button."""
        ...

    def mouse_button_up(self, button: str) -> None:
        """Release one mouse button."""
        ...

    def move_relative(self, dx: int, dy: int) -> None:
        """Move the pointer by a relative delta."""
        ...

    def is_key_pressed(self, virtual_key: int) -> bool:
        """Report whether a key is currently physically down."""
        ...

    def close(self) -> None:
        """Release any backend resources."""
        ...


# ---------------------------------------------------------------------------
# Win32 implementation
# ---------------------------------------------------------------------------

_INPUT_SIZE = 0


def _build_structures() -> dict[str, object]:
    """Declare the ``INPUT`` union exactly as ``winuser.h`` does."""
    from ctypes import wintypes

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_void_p),
        ]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_void_p),
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        ]

    class _INPUTUNION(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]

    return {
        "wintypes": wintypes,
        "KEYBDINPUT": KEYBDINPUT,
        "MOUSEINPUT": MOUSEINPUT,
        "INPUT": INPUT,
    }


class Win32InputBackend:
    """``SendInput``-based input backend.

    The ``user32`` handle is resolved on first use, so importing this module does
    not require Windows.
    """

    def __init__(self) -> None:
        self._api: dict[str, object] | None = None

    def _ensure_api(self) -> dict[str, object]:
        if self._api is not None:
            return self._api
        if sys.platform != "win32":
            raise UnsupportedPlatformError("input injection requires Windows")
        from ctypes import wintypes

        structures = _build_structures()
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        input_type = structures["INPUT"]
        user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(input_type), ctypes.c_int]
        user32.SendInput.restype = wintypes.UINT
        user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
        user32.MapVirtualKeyW.restype = wintypes.UINT
        user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        user32.GetAsyncKeyState.restype = wintypes.SHORT
        structures["user32"] = user32
        self._api = structures
        return self._api

    # -- event plumbing ---------------------------------------------------

    def _send(self, event: object) -> None:
        api = self._ensure_api()
        user32 = api["user32"]
        input_type = api["INPUT"]
        size = ctypes.sizeof(input_type)
        sent = user32.SendInput(1, ctypes.byref(event), size)
        if sent != 1:
            error = ctypes.get_last_error()
            raise InputBackendError(
                f"SendInput injected {sent} of 1 events (Win32 error {error}); "
                "the desktop may be locked or input is blocked"
            )

    def _keyboard_event(self, virtual_key: int, *, up: bool) -> object:
        api = self._ensure_api()
        user32 = api["user32"]
        scan = int(user32.MapVirtualKeyW(int(virtual_key) & 0xFF, MAPVK_VK_TO_VSC))
        flags = 0
        if scan:
            # Prefer scan-code events: games that read Raw Input / DirectInput
            # generally ignore virtual-key-only events.
            flags |= KEYEVENTF_SCANCODE
        if int(virtual_key) in _EXTENDED_KEY_VKS:
            flags |= KEYEVENTF_EXTENDEDKEY
        if up:
            flags |= KEYEVENTF_KEYUP
        event = api["INPUT"]()
        event.type = INPUT_KEYBOARD
        event.ki = api["KEYBDINPUT"](
            wVk=0 if scan else int(virtual_key),
            wScan=scan,
            dwFlags=flags,
            time=0,
            dwExtraInfo=None,
        )
        return event

    def _mouse_event(self, flags: int, *, dx: int = 0, dy: int = 0, data: int = 0) -> object:
        api = self._ensure_api()
        event = api["INPUT"]()
        event.type = INPUT_MOUSE
        event.mi = api["MOUSEINPUT"](dx=int(dx), dy=int(dy), mouseData=int(data), dwFlags=flags, time=0, dwExtraInfo=None)
        return event

    # -- InputBackend -----------------------------------------------------

    def key_down(self, virtual_key: int) -> None:
        """Press and hold a key identified by virtual-key code."""
        self._send(self._keyboard_event(virtual_key, up=False))

    def key_up(self, virtual_key: int) -> None:
        """Release a key identified by virtual-key code."""
        self._send(self._keyboard_event(virtual_key, up=True))

    def mouse_button_down(self, button: str) -> None:
        """Press and hold a mouse button."""
        try:
            down, _up, data = BUTTON_FLAGS[button]
        except KeyError as exc:
            raise InputBackendError(f"unknown mouse button {button!r}") from exc
        self._send(self._mouse_event(down, data=data))

    def mouse_button_up(self, button: str) -> None:
        """Release a mouse button."""
        try:
            _down, up, data = BUTTON_FLAGS[button]
        except KeyError as exc:
            raise InputBackendError(f"unknown mouse button {button!r}") from exc
        self._send(self._mouse_event(up, data=data))

    def move_relative(self, dx: int, dy: int) -> None:
        """Move the pointer by a relative delta (no absolute flag)."""
        self._send(self._mouse_event(MOUSEEVENTF_MOVE, dx=dx, dy=dy))

    def is_key_pressed(self, virtual_key: int) -> bool:
        """Report whether a key is physically down anywhere on the desktop.

        Used only for the emergency-stop poll. This reads global key state and
        does not consume the key, so the game still receives it normally.
        """
        api = self._ensure_api()
        return bool(api["user32"].GetAsyncKeyState(int(virtual_key) & 0xFF) & 0x8000)

    def close(self) -> None:
        """No-op: ``SendInput`` holds no session state."""
        return None


#: Alias kept for callers that prefer the descriptive name.
ButtonCode = str
