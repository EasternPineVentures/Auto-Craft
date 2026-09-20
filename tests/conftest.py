"""Shared test doubles.

Every fake here replaces a hardware or OS boundary, which is the whole point:
the safety logic, the agent loop, action validation and telemetry can all be
tested on a machine with no game running and, more importantly, with **no input
ever injected**. Nothing in this file calls ``SendInput``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autocraft.config import Config
from autocraft.vision.frame import ScreenRegion
from autocraft.vision.window import WindowInfo

# ---------------------------------------------------------------------------
# window fakes
# ---------------------------------------------------------------------------


class FakeWindowBackend:
    """A ``WindowBackend`` backed by a mutable list of fake windows."""

    def __init__(self, windows: list[WindowInfo] | None = None, foreground: int = 0) -> None:
        self.windows = list(windows or [])
        self.foreground = foreground
        self.enumeration_count = 0

    def add(self, handle: int, title: str, *, region: ScreenRegion | None = None, minimized: bool = False, visible: bool = True) -> WindowInfo:
        window = WindowInfo(
            handle=handle,
            title=title,
            region=region if region is not None else ScreenRegion(0, 0, 800, 600),
            visible=visible,
            minimized=minimized,
            process_id=handle,
        )
        self.windows.append(window)
        return window

    def iter_windows(self):
        self.enumeration_count += 1
        return iter(list(self.windows))

    def foreground_handle(self) -> int:
        return self.foreground

    def describe(self, handle: int) -> WindowInfo | None:
        for window in self.windows:
            if window.handle == handle:
                return window
        return None

    def focus(self, handle: int) -> None:
        """Make ``handle`` the foreground window."""
        self.foreground = handle

    def remove(self, handle: int) -> None:
        """Drop a window, simulating the game closing."""
        self.windows = [window for window in self.windows if window.handle != handle]
        if self.foreground == handle:
            self.foreground = 0


# ---------------------------------------------------------------------------
# capture fakes
# ---------------------------------------------------------------------------


class FakeCaptureBackend:
    """A ``CaptureBackend`` returning synthetic BGRA buffers."""

    def __init__(self, *, value: int = 32, fail: bool = False) -> None:
        self.value = value
        self.fail = fail
        self.calls: list[ScreenRegion] = []
        self.closed = False
        self.frames: list[np.ndarray] | None = None

    def grab(self, region: ScreenRegion) -> np.ndarray:
        self.calls.append(region)
        if self.fail:
            raise RuntimeError("synthetic grab failure")
        if self.frames is not None:
            if not self.frames:
                raise RuntimeError("synthetic frame supply exhausted")
            return self.frames.pop(0)
        buffer = np.zeros((region.height, region.width, 4), dtype=np.uint8)
        buffer[..., 0] = self.value  # B
        buffer[..., 1] = self.value  # G
        buffer[..., 2] = self.value  # R
        buffer[..., 3] = 255  # A
        return buffer

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# input fakes
# ---------------------------------------------------------------------------


class FakeInputBackend:
    """An ``InputBackend`` that records events instead of injecting them."""

    def __init__(self) -> None:
        self.key_downs: list[int] = []
        self.key_ups: list[int] = []
        self.button_downs: list[str] = []
        self.button_ups: list[str] = []
        self.moves: list[tuple[int, int]] = []
        self.pressed_keys: set[int] = set()
        self.closed = False
        self.fail_on: set[str] = set()

    def _maybe_fail(self, name: str) -> None:
        if name in self.fail_on:
            raise RuntimeError(f"synthetic failure in {name}")

    def key_down(self, virtual_key: int) -> None:
        self._maybe_fail("key_down")
        self.key_downs.append(virtual_key)
        self.pressed_keys.add(virtual_key)

    def key_up(self, virtual_key: int) -> None:
        self._maybe_fail("key_up")
        self.key_ups.append(virtual_key)
        self.pressed_keys.discard(virtual_key)

    def mouse_button_down(self, button: str) -> None:
        self._maybe_fail("mouse_button_down")
        self.button_downs.append(button)

    def mouse_button_up(self, button: str) -> None:
        self._maybe_fail("mouse_button_up")
        self.button_ups.append(button)

    def move_relative(self, dx: int, dy: int) -> None:
        self._maybe_fail("move_relative")
        self.moves.append((dx, dy))

    def is_key_pressed(self, virtual_key: int) -> bool:
        return virtual_key in self.pressed_keys

    def close(self) -> None:
        self.closed = True

    @property
    def events(self) -> list[tuple[str, object]]:
        """All recorded events in the order they happened."""
        merged: list[tuple[float, str, object]] = []
        for index, key in enumerate(self.key_downs):
            merged.append((index, "key_down", key))
        for index, key in enumerate(self.key_ups):
            merged.append((index, "key_up", key))
        for index, button in enumerate(self.button_downs):
            merged.append((index, "button_down", button))
        for index, button in enumerate(self.button_ups):
            merged.append((index, "button_up", button))
        for index, move in enumerate(self.moves):
            merged.append((index, "move", move))
        merged.sort(key=lambda item: item[0])
        return [(name, value) for _, name, value in merged]


class FakeClock:
    """A monotonic clock the test drives by hand."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """A configuration with small, fast limits for tests.

    ``data_dir`` is redirected into pytest's temporary directory so running the
    suite never leaves run directories behind in the working tree.
    """
    return Config(
        data_dir=tmp_path / "data",
        target_title_patterns=("Luanti", "VoxelLibre"),
        capture_fps=100.0,
        min_action_interval=0.0,
        max_consecutive_blocks=3,
    )


@pytest.fixture
def fake_windows() -> FakeWindowBackend:
    """A window backend with one focused Luanti window and one unrelated window."""
    backend = FakeWindowBackend(foreground=0x100)
    backend.add(0x100, "Luanti 5.17.0 [Singleplayer]", region=ScreenRegion(100, 50, 320, 240))
    backend.add(0x200, "Notepad", region=ScreenRegion(0, 0, 640, 480))
    return backend


@pytest.fixture
def fake_capture() -> FakeCaptureBackend:
    """A capture backend returning a flat synthetic frame."""
    return FakeCaptureBackend()


@pytest.fixture
def fake_input() -> FakeInputBackend:
    """An input backend that records rather than injects."""
    return FakeInputBackend()


@pytest.fixture
def clock() -> FakeClock:
    """A hand-driven clock."""
    return FakeClock()
