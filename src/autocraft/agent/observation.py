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

__all__ = ["Observation", "Observer", "WindowStatus"]


@dataclass(frozen=True)
class WindowStatus:
    """A serialisable snapshot of the target window's state."""

    found: bool
    title: str = ""
    is_foreground: bool = False
    minimized: bool = False
    region: ScreenRegion | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "found": self.found,
            "title": self.title,
            "is_foreground": self.is_foreground,
            "minimized": self.minimized,
            "region": self.region.to_dict() if self.region is not None else None,
            "reason": self.reason,
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
        payload: dict[str, Any] = {
            "index": self.index,
            "timestamp": self.timestamp,
            "window": self.window.to_dict(),
            "has_frame": self.has_frame,
            "capture_path": str(self.capture_path) if self.capture_path is not None else None,
            "capture_error": self.capture_error,
        }
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

    @property
    def locator(self) -> WindowLocator:
        """The window locator in use."""
        return self._locator

    @property
    def capturer(self) -> ScreenCapturer:
        """The screen capturer in use."""
        return self._capturer

    def observe(self, index: int = 0, *, capture: bool = True, force_discovery: bool = False) -> Observation:
        """Grab one observation.

        A capture failure is recorded on the observation rather than raised: a
        transient grab failure should not tear down a run, and the loop needs to
        see it in telemetry.
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

        return Observation(
            index=index,
            timestamp=timestamp,
            window=window_status,
            frame=frame,
            capture_error=error,
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
