"""Win32 window discovery and client-area geometry.

The agent only ever needs two things from the operating system: which window is
the game, and where that window's *client area* sits on the desktop. Everything
else about the game must come from pixels.

Win32 bindings are created lazily so this module imports cleanly on any platform
and so the pure helpers (title matching, geometry) stay testable without a
window server.
"""

from __future__ import annotations

import ctypes
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

from .frame import ScreenRegion

__all__ = [
    "TargetStatus",
    "UnsupportedPlatformError",
    "WindowBackend",
    "WindowError",
    "WindowInfo",
    "WindowLocator",
    "Win32WindowBackend",
    "ensure_dpi_awareness",
    "title_matches",
]


class UnsupportedPlatformError(RuntimeError):
    """Raised when a Win32-only feature is used on another platform."""


class WindowError(RuntimeError):
    """Raised when a Win32 window query fails unexpectedly."""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def title_matches(title: str, patterns: Sequence[str]) -> bool:
    """True when ``title`` contains any pattern, compared case-insensitively.

    Empty patterns never match, so a blank title is never mistaken for the game.
    """
    if not title:
        return False
    haystack = title.casefold()
    return any(pattern.strip().casefold() in haystack for pattern in patterns if pattern.strip())


@dataclass(frozen=True)
class WindowInfo:
    """A top-level window and the screen rectangle of its client area."""

    handle: int
    title: str
    region: ScreenRegion
    visible: bool
    minimized: bool
    process_id: int = 0

    def is_capturable(self) -> bool:
        """True when the client area can actually be captured right now."""
        return self.visible and not self.minimized and not self.region.is_empty()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "handle": self.handle,
            "title": self.title,
            "region": self.region.to_dict(),
            "visible": self.visible,
            "minimized": self.minimized,
            "process_id": self.process_id,
            "capturable": self.is_capturable(),
        }


@dataclass(frozen=True)
class TargetStatus:
    """The current answer to "where is the game, and may we act on it?"."""

    found: bool
    window: WindowInfo | None = None
    is_foreground: bool = False
    foreground_handle: int = 0
    foreground_title: str = ""
    reason: str = ""

    @property
    def can_capture(self) -> bool:
        """True when a frame can be grabbed from the target right now."""
        return bool(self.found and self.window is not None and self.window.is_capturable())

    @property
    def can_inject_input(self) -> bool:
        """True when input may be sent under the foreground-window lock.

        Deliberately says nothing about whether the caller *should* act; the
        safety guard owns that decision.
        """
        return bool(self.found and self.is_foreground and self.window is not None and not self.window.minimized)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "found": self.found,
            "window": self.window.to_dict() if self.window is not None else None,
            "is_foreground": self.is_foreground,
            "foreground_handle": self.foreground_handle,
            "foreground_title": self.foreground_title,
            "reason": self.reason,
            "can_capture": self.can_capture,
            "can_inject_input": self.can_inject_input,
        }


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class WindowBackend:
    """Interface the locator needs from the operating system.

    Keeping this behind a protocol means window logic is testable with a fake
    backend, and a future non-Windows backend would only have to implement these
    three methods.
    """

    def iter_windows(self) -> Iterator[WindowInfo]:
        """Yield every visible top-level window."""
        raise NotImplementedError

    def foreground_handle(self) -> int:
        """Return the handle of the current foreground window, or 0."""
        raise NotImplementedError

    def describe(self, handle: int) -> WindowInfo | None:
        """Return details for one handle, or ``None`` if it is gone."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Win32 implementation
# ---------------------------------------------------------------------------

_WIN32: dict[str, Any] | None = None

#: Virtual key for a window handle comparison is unnecessary; these constants
#: only drive DPI awareness so capture coordinates match physical pixels.
_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4


def _win32() -> dict[str, Any]:
    """Bind and cache the Win32 entry points AutoCraft uses."""
    global _WIN32
    if _WIN32 is not None:
        return _WIN32
    if sys.platform != "win32":
        raise UnsupportedPlatformError("AutoCraft V0 window discovery requires Windows")

    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    user32.EnumWindows.argtypes = [ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM), wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetClientRect.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.ClientToScreen.restype = wintypes.BOOL
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD

    _WIN32 = {"user32": user32, "kernel32": kernel32, "wintypes": wintypes}
    return _WIN32


def ensure_dpi_awareness() -> str:
    """Opt the process into physical-pixel coordinates and report what happened.

    Without this, Windows silently virtualises coordinates on scaled displays
    and captured regions land in the wrong place. Best effort: the return value
    records which API succeeded so the CLI can report it honestly.
    """
    if sys.platform != "win32":
        return "not-applicable"
    try:
        user32 = _win32()["user32"]
    except UnsupportedPlatformError:  # pragma: no cover - platform guard
        return "not-applicable"

    for name, call in (
        ("per-monitor-v2", lambda: user32.SetProcessDpiAwarenessContext(_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)),
        ("system", lambda: user32.SetProcessDPIAware()),
    ):
        try:
            if call():
                return name
        except (AttributeError, OSError):
            continue
    try:
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        if shcore.SetProcessDpiAwareness(2) == 0:
            return "per-monitor"
    except (AttributeError, OSError):
        pass
    return "unchanged"


class Win32WindowBackend(WindowBackend):
    """Window queries backed by ``user32``."""

    def __init__(self, *, skip_empty_titles: bool = True) -> None:
        self._api = _win32()
        self._skip_empty_titles = skip_empty_titles

    # -- helpers ----------------------------------------------------------

    def _title(self, handle: int) -> str:
        wintypes = self._api["wintypes"]
        length = self._api["user32"].GetWindowTextLengthW(wintypes.HWND(handle))
        if length <= 0:
            return ""
        buffer = ctypes.create_unicode_buffer(length + 1)
        self._api["user32"].GetWindowTextW(wintypes.HWND(handle), buffer, length + 1)
        return buffer.value

    def _client_region(self, handle: int) -> ScreenRegion:
        wintypes = self._api["wintypes"]
        user32 = self._api["user32"]
        rect = wintypes.RECT()
        if not user32.GetClientRect(wintypes.HWND(handle), ctypes.byref(rect)):
            raise WindowError(f"GetClientRect failed for window {handle:#x}")
        origin = wintypes.POINT(0, 0)
        if not user32.ClientToScreen(wintypes.HWND(handle), ctypes.byref(origin)):
            raise WindowError(f"ClientToScreen failed for window {handle:#x}")
        return ScreenRegion.from_client_rect(
            (rect.left, rect.top, rect.right, rect.bottom),
            (origin.x, origin.y),
        )

    def _process_id(self, handle: int) -> int:
        wintypes = self._api["wintypes"]
        pid = wintypes.DWORD(0)
        self._api["user32"].GetWindowThreadProcessId(wintypes.HWND(handle), ctypes.byref(pid))
        return int(pid.value)

    # -- WindowBackend ----------------------------------------------------

    def iter_windows(self) -> Iterator[WindowInfo]:
        wintypes = self._api["wintypes"]
        user32 = self._api["user32"]
        collected: list[WindowInfo] = []

        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def _collect(hwnd: int, _lparam: int) -> bool:
            info = self._describe_handle(int(hwnd))
            if info is not None:
                collected.append(info)
            return True

        callback = callback_type(_collect)  # keep a reference alive for the call
        if not user32.EnumWindows(callback, 0):
            error = ctypes.get_last_error()
            # ERROR_SUCCESS just means enumeration finished; only raise otherwise.
            if error not in (0, 18):
                raise WindowError(f"EnumWindows failed with error {error}")
        return iter(collected)

    def foreground_handle(self) -> int:
        wintypes = self._api["wintypes"]
        return int(self._api["user32"].GetForegroundWindow() or 0)

    def describe(self, handle: int) -> WindowInfo | None:
        return self._describe_handle(int(handle))

    # -- internals --------------------------------------------------------

    def _describe_handle(self, handle: int) -> WindowInfo | None:
        if not handle:
            return None
        wintypes = self._api["wintypes"]
        user32 = self._api["user32"]
        if not user32.IsWindow(wintypes.HWND(handle)):
            return None
        title = self._title(handle)
        if self._skip_empty_titles and not title:
            return None
        visible = bool(user32.IsWindowVisible(wintypes.HWND(handle)))
        minimized = bool(user32.IsIconic(wintypes.HWND(handle)))
        try:
            region = self._client_region(handle)
        except WindowError:
            region = ScreenRegion(0, 0, 0, 0)
        return WindowInfo(
            handle=handle,
            title=title,
            region=region,
            visible=visible,
            minimized=minimized,
            process_id=self._process_id(handle),
        )


# ---------------------------------------------------------------------------
# Locator
# ---------------------------------------------------------------------------


class WindowLocator:
    """Resolves and caches the game window for a set of title patterns.

    The cached handle keeps the per-action foreground check cheap: comparing a
    cached handle against ``GetForegroundWindow`` avoids re-enumerating every
    top-level window several times a second.
    """

    def __init__(
        self,
        backend: WindowBackend,
        patterns: Sequence[str],
        *,
        clock: Callable[[], float] = time.monotonic,
        rediscover_after: float = 1.0,
    ) -> None:
        self._backend = backend
        self._patterns = tuple(patterns)
        self._clock = clock
        self._rediscover_after = float(rediscover_after)
        self._handle: int | None = None
        self._cached: WindowInfo | None = None
        self._last_refresh = float("-inf")

    @property
    def patterns(self) -> tuple[str, ...]:
        """Configured title patterns."""
        return self._patterns

    @property
    def handle(self) -> int | None:
        """Cached target window handle, or ``None`` before discovery."""
        return self._handle

    def forget(self) -> None:
        """Drop the cached handle, forcing rediscovery on the next call."""
        self._handle = None
        self._cached = None
        self._last_refresh = float("-inf")

    def find_all(self) -> list[WindowInfo]:
        """Return every visible top-level window whose title matches."""
        matches: list[WindowInfo] = []
        for window in self._backend.iter_windows():
            if window.visible and title_matches(window.title, self._patterns):
                matches.append(window)
        return matches

    def refresh(self) -> TargetStatus:
        """Re-enumerate windows, choose a target and cache it."""
        matches = self.find_all()
        foreground = self._backend.foreground_handle()
        foreground_title = ""
        if foreground:
            described = self._backend.describe(foreground)
            foreground_title = described.title if described is not None else ""

        self._last_refresh = self._clock()
        if not matches:
            self._handle = None
            self._cached = None
            return TargetStatus(
                found=False,
                is_foreground=False,
                foreground_handle=foreground,
                foreground_title=foreground_title,
                reason=f"no visible window matched {list(self._patterns)}",
            )

        # Prefer the window that is already foreground; otherwise the largest
        # capturable one, which is the game rather than a small launcher or a
        # minimised window left over from a previous session.
        target = next((w for w in matches if w.handle == foreground), None)
        if target is None:
            pool = [w for w in matches if w.is_capturable()] or matches
            target = max(pool, key=lambda w: (w.region.area, w.handle))

        self._handle = target.handle
        self._cached = target
        reason = "" if target.is_capturable() else "target window is minimised or has an empty client area"
        return TargetStatus(
            found=True,
            window=target,
            is_foreground=target.handle == foreground,
            foreground_handle=foreground,
            foreground_title=foreground_title,
            reason=reason,
        )

    def status(self, *, force: bool = False) -> TargetStatus:
        """Return the target status, reusing the cache when it is still fresh."""
        if force or self._handle is None:
            return self.refresh()
        if self._clock() - self._last_refresh >= self._rediscover_after:
            return self.refresh()

        window = self._cached
        if window is None:
            return self.refresh()
        described = self._backend.describe(window.handle)
        if described is None:
            return self.refresh()

        foreground = self._backend.foreground_handle()
        foreground_title = described.title if described.handle == foreground else ""
        if not foreground_title and foreground:
            other = self._backend.describe(foreground)
            foreground_title = other.title if other is not None else ""
        self._cached = described
        reason = "" if described.is_capturable() else "target window is minimised or has an empty client area"
        return TargetStatus(
            found=True,
            window=described,
            is_foreground=described.handle == foreground,
            foreground_handle=foreground,
            foreground_title=foreground_title,
            reason=reason,
        )

    def is_target_foreground(self) -> bool:
        """Cheap foreground-lock predicate used by the safety guard."""
        handle = self._handle
        if handle is None:
            status = self.refresh()
            handle = status.window.handle if status.window is not None else None
            if handle is None:
                return False
        return self._backend.foreground_handle() == handle

    def target_region(self) -> ScreenRegion:
        """Return the cached client region, refreshing once if needed.

        Raises:
            WindowError: If no target window is available.
        """
        window = self._cached
        if window is None or self._handle is None:
            status = self.refresh()
            window = status.window
        if window is None:
            raise WindowError("no target window found")
        return window.region
