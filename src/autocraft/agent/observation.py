"""Observation: everything the agent is permitted to know at one instant.

An observation is a frame plus the window bookkeeping needed to say *where* that
frame came from and whether acting on it is currently legal. It contains no game
state: no coordinates, no entities, no inventory. If a future version of
AutoCraft needs ground truth, that belongs to a separate evaluator, never here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..vision.capture import CaptureError, ScreenCapturer
from ..vision.frame import Frame, ScreenRegion
from ..vision.window import TargetStatus, WindowLocator

__all__ = ["Observation", "Observer", "WindowGeometry", "WindowStatus"]


@dataclass(frozen=True)
class WindowGeometry:
    """Where a frame came from, and how big it was.

    Two sizes are recorded rather than one, because they are not guaranteed to
    agree and neither is guaranteed to hold still. ``client_*`` is the target
    window's client area as the locator reported it; ``frame_*`` is the size of
    the image actually captured. A live run once reported a 3222-pixel-wide client
    area at startup and a 3591-pixel-wide frame on every step afterwards, and the
    record had nowhere to say so - so the two are written down separately, at the
    moment they were taken, for every observation.
    """

    handle: int | None = None
    # Zero means "not measured", not "measured as zero". The seed is taken
    # before the first capture, so it has a client area and no frame yet.
    client_width: int = 0
    client_height: int = 0
    frame_width: int = 0
    frame_height: int = 0

    @property
    def empty(self) -> bool:
        """True when this records no window and no frame at all.

        All-or-nothing, so it cannot speak for a geometry that measured some of
        its fields. ``changes_from`` compares field by field for that reason.
        """
        return (
            self.handle is None
            and self.client_width == 0
            and self.client_height == 0
            and self.frame_width == 0
            and self.frame_height == 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "handle": self.handle,
            "client_width": self.client_width,
            "client_height": self.client_height,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
        }

    def changes_from(self, other: "WindowGeometry") -> tuple[str, ...]:
        """Field names that differ from ``other``, in a fixed order.

        An empty geometry on either side is not a change. Losing sight of the
        window for a step and getting it back has not resized anything, and
        reporting it as a geometry change would bury the real one in noise.

        Neither is a field only one side ever measured. Zero is how this type
        spells "not known", and the seed is taken before the first capture, so it
        carries a 0x0 frame. Comparing that 0x0 against the first real frame
        reports a change that did not happen, and it did so on every run: the
        first event of the third live WAKE-001 run was ``WINDOW_GEOMETRY_CHANGED``
        with ``before.frame_width 0`` and ``after.frame_width 2102``, while the
        client area was 2102x1061 on both sides and had not moved at all.

        That is not only noise. The event calls ``_rebaseline_after_resize``,
        which discards the view memory, the progress model, the repetition guard,
        the strategy cooldowns and the measured calibration, so a step that
        happened to capture no frame would silently wipe the run's state.

        The client area is still compared whenever both sides measured it, which
        is what catches the disagreement this instrumentation exists for: the
        second live run reported a 3222-pixel client area and a 3591-pixel frame,
        and that is a ``client_width`` change with a real number on both sides.
        """
        if self.empty or other.empty:
            return ()
        changed = []
        if self.handle is not None and other.handle is not None and self.handle != other.handle:
            changed.append("handle")
        for name in ("client_width", "client_height", "frame_width", "frame_height"):
            mine = getattr(self, name)
            theirs = getattr(other, name)
            if mine and theirs and mine != theirs:
                changed.append(name)
        return tuple(changed)

    @classmethod
    def of(
        cls,
        *,
        handle: int | None,
        region: ScreenRegion | None,
        frame: Frame | None = None,
    ) -> "WindowGeometry":
        """Build a geometry from a locator window, a region and an optional frame."""
        return cls(
            handle=None if handle is None else int(handle),
            client_width=int(region.width) if region is not None else 0,
            client_height=int(region.height) if region is not None else 0,
            frame_width=int(frame.width) if frame is not None else 0,
            frame_height=int(frame.height) if frame is not None else 0,
        )


@dataclass(frozen=True)
class WindowStatus:
    """A serialisable snapshot of the target window's state."""

    found: bool
    title: str = ""
    is_foreground: bool = False
    minimized: bool = False
    region: ScreenRegion | None = None
    reason: str = ""
    handle: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "found": self.found,
            "title": self.title,
            "is_foreground": self.is_foreground,
            "minimized": self.minimized,
            "region": self.region.to_dict() if self.region is not None else None,
            "reason": self.reason,
            "handle": self.handle,
        }

    @classmethod
    def from_target_status(cls, status: TargetStatus) -> "WindowStatus":
        """Build a status from a window-locator result."""
        window = status.window
        return cls(
            found=status.found,
            title=window.title if window is not None else "",
            is_foreground=status.is_foreground,
            minimized=bool(window.minimized) if window is not None else False,
            region=window.region if window is not None else None,
            reason=status.reason,
            handle=int(window.handle) if window is not None else None,
        )


@dataclass(frozen=True)
class Observation:
    """One step's view of the world.

    ``frame`` is the rendered client area; ``capture_path`` points at the saved
    copy when the caller asked for one. Raw pixels are never written into
    telemetry - only the path and a coarse signature.
    """

    index: int
    timestamp: float
    window: WindowStatus
    frame: Frame | None = None
    capture_path: Path | None = None
    capture_error: str | None = None
    #: The geometry this observation replaced, when it differs from the previous
    #: one. ``None`` on the first observation and whenever nothing moved.
    geometry_changed_from: WindowGeometry | None = None

    @property
    def geometry(self) -> WindowGeometry:
        """The window handle, client area and captured frame size, all in one place."""
        return WindowGeometry.of(handle=self.window.handle, region=self.window.region, frame=self.frame)

    @property
    def geometry_change(self) -> tuple[str, ...]:
        """Names of the geometry fields that changed since the previous observation."""
        if self.geometry_changed_from is None:
            return ()
        return self.geometry.changes_from(self.geometry_changed_from)

    @property
    def has_frame(self) -> bool:
        """True when this observation carries a captured frame."""
        return self.frame is not None

    @property
    def can_act(self) -> bool:
        """True when the window is present, unminimised and focused."""
        return self.window.found and self.window.is_foreground and not self.window.minimized

    def to_dict(self, *, include_frame: bool = True) -> dict[str, Any]:
        """Return a JSON-serialisable view that never embeds pixel data."""
        geometry = self.geometry
        payload: dict[str, Any] = {
            "index": self.index,
            "timestamp": self.timestamp,
            "window": self.window.to_dict(),
            "has_frame": self.has_frame,
            "capture_path": str(self.capture_path) if self.capture_path is not None else None,
            "capture_error": self.capture_error,
            # Spelled out at the top level as well as inside ``window``: these are
            # the numbers a geometry disagreement shows up in, and a reader should
            # not have to know that ``window.region`` is the client area.
            "window_handle": geometry.handle,
            "client_width": geometry.client_width,
            "client_height": geometry.client_height,
            "frame_width": geometry.frame_width,
            "frame_height": geometry.frame_height,
        }
        if self.geometry_changed_from is not None:
            payload["geometry_changed_from"] = self.geometry_changed_from.to_dict()
            payload["geometry_changed_fields"] = list(self.geometry_change)
        if include_frame and self.frame is not None:
            payload["frame"] = self.frame.to_dict()
        return payload


class Observer:
    """Produces :class:`Observation` objects from the vision layer.

    Observing is always safe: it never injects input and never refuses to look.
    It reports focus state so the safety layer can decide about acting.
    """

    def __init__(
        self,
        locator: WindowLocator,
        capturer: ScreenCapturer,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._locator = locator
        self._capturer = capturer
        self._clock = clock
        self._last_geometry: WindowGeometry | None = None

    @property
    def locator(self) -> WindowLocator:
        """The window locator in use."""
        return self._locator

    @property
    def capturer(self) -> ScreenCapturer:
        """The screen capturer in use."""
        return self._capturer

    @property
    def last_geometry(self) -> WindowGeometry | None:
        """The most recent non-empty geometry observed, if any."""
        return self._last_geometry

    def seed_geometry(self, geometry: WindowGeometry) -> None:
        """Prime the geometry baseline without taking an observation.

        A caller that has already asked the locator where the window is - the
        wake-test command does, before the focus countdown - can hand that answer
        in here, so the very first observation is compared against it. That is the
        comparison that matters: the plan's geometry is sampled before the countdown
        and the run's is sampled after it, which is exactly where a disagreement
        would first appear.
        """
        if not geometry.empty:
            self._last_geometry = geometry

    def observe(self, index: int = 0, *, capture: bool = True, force_discovery: bool = False) -> Observation:
        """Grab one observation.

        A capture failure is recorded on the observation rather than raised: a
        transient grab failure should not tear down a run, and the loop needs to
        see it in telemetry.

        The observation also reports whether the window's geometry has moved since
        the last one. AutoCraft never resizes the game window; it only writes down
        when someone else did.
        """
        status = self._locator.status(force=force_discovery)
        window_status = WindowStatus.from_target_status(status)
        timestamp = self._clock()

        frame: Frame | None = None
        error: str | None = None
        if capture and status.can_capture and status.window is not None:
            try:
                frame = self._capturer.capture_window(status.window)
            except CaptureError as exc:
                error = str(exc)
        elif capture and not status.can_capture:
            error = status.reason or "target window is not capturable"

        geometry = WindowGeometry.of(handle=window_status.handle, region=window_status.region, frame=frame)
        previous = self._last_geometry
        changed_from: WindowGeometry | None = None
        if previous is not None and geometry.changes_from(previous):
            changed_from = previous
        if not geometry.empty:
            self._last_geometry = geometry

        return Observation(
            index=index,
            timestamp=timestamp,
            window=window_status,
            frame=frame,
            capture_error=error,
            geometry_changed_from=changed_from,
        )

    def save_frame(
        self,
        observation: Observation,
        directory: str | Path,
        *,
        stem: str | None = None,
    ) -> Path | None:
        """Write the observation's frame to disk and return the path.

        Returns ``None`` when the observation has no frame.
        """
        if observation.frame is None:
            return None
        base = Path(directory)
        name = stem or f"frame-{observation.index:06d}-{int(observation.timestamp * 1000)}"
        suffix = observation.frame.metadata.get("extension", "png")
        return observation.frame.save(base / f"{name}.{suffix}")
