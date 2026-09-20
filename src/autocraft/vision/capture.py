"""Client-area screen capture.

Only the game window's client area is ever captured, so the agent's observation
is the rendered game surface and nothing else. The backend sits behind a small
protocol so tests and dry runs can substitute a synthetic frame source.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from .frame import Frame, FrameError, ScreenRegion
from .window import WindowInfo

__all__ = ["CaptureBackend", "CaptureError", "MssCaptureBackend", "ScreenCapturer"]


class CaptureError(RuntimeError):
    """Raised when a frame could not be grabbed."""


@runtime_checkable
class CaptureBackend(Protocol):
    """Minimal frame source.

    Implementations return a BGRA ``H x W x 4`` ``uint8`` array for the
    requested region. ``mss`` already works in that format, so the protocol
    matches it rather than forcing an extra copy.
    """

    def grab(self, region: ScreenRegion) -> np.ndarray:
        """Return a BGRA buffer covering ``region``."""
        ...

    def close(self) -> None:
        """Release any underlying resources."""
        ...


class MssCaptureBackend:
    """``mss``-based backend: fast, dependency-light desktop capture."""

    def __init__(self, *, mss_factory: Callable[[], Any] | None = None) -> None:
        self._mss_factory = mss_factory
        self._session: Any = None

    def _ensure_session(self) -> Any:
        if self._session is None:
            if self._mss_factory is not None:
                self._session = self._mss_factory()
            else:
                try:
                    import mss  # noqa: PLC0415 - lazy so import cost is paid on first use
                except ImportError as exc:  # pragma: no cover - dependency is declared
                    raise CaptureError(
                        "screen capture requires mss; install with 'pip install mss'"
                    ) from exc
                # mss renamed the class from ``mss.mss`` to ``mss.MSS``; the old
                # name still works but is deprecated, so prefer the new one.
                factory = getattr(mss, "MSS", None) or mss.mss
                self._session = factory()
        return self._session

    def grab(self, region: ScreenRegion) -> np.ndarray:
        """Grab one frame. The returned buffer is a copy, safe to keep."""
        if region.is_empty():
            raise CaptureError(f"refusing to capture an empty region: {region.to_dict()}")
        session = self._ensure_session()
        try:
            shot = session.grab(region.to_mss_monitor())
        except Exception as exc:  # noqa: BLE001 - mss raises platform-specific types
            raise CaptureError(f"capture failed for region {region.to_dict()}: {exc}") from exc
        return np.asarray(shot)[:, :, :4].copy()

    def close(self) -> None:
        """Close the underlying ``mss`` session if one was opened."""
        session, self._session = self._session, None
        if session is not None:
            try:
                session.close()
            except Exception:  # noqa: BLE001 - closing must never mask the real error
                pass


class ScreenCapturer:
    """Turns a screen region into a :class:`Frame`.

    Adds no cleverness: it validates the region, delegates to the backend and
    converts BGRA to RGB. Keeping it this thin is what makes capture easy to
    fake in tests.
    """

    def __init__(
        self,
        backend: CaptureBackend,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._backend = backend
        self._clock = clock

    @property
    def backend(self) -> CaptureBackend:
        """The wrapped capture backend."""
        return self._backend

    def capture_region(self, region: ScreenRegion, *, source: str = "mss") -> Frame:
        """Capture ``region`` and return a frame.

        Raises:
            CaptureError: If the region is empty or the grab fails.
            FrameError: If the backend returned a buffer of the wrong shape.
        """
        if region.is_empty():
            raise CaptureError(f"refusing to capture an empty region: {region.to_dict()}")
        timestamp = self._clock()
        try:
            buffer = self._backend.grab(region)
        except CaptureError:
            raise
        except Exception as exc:  # noqa: BLE001 - a backend failure is a capture failure
            # Wrapping here is what makes the Observer's transient-failure handling
            # cover every backend, not just the mss one that already wraps errors.
            raise CaptureError(f"capture backend failed for region {region.to_dict()}: {exc}") from exc
        frame = Frame.from_bgra(
            buffer,
            timestamp=timestamp,
            region=region,
            source=source,
            metadata={"backend": type(self._backend).__name__},
        )
        if not frame.matches_region():
            raise FrameError(
                f"backend returned {frame.width}x{frame.height} for a "
                f"{region.width}x{region.height} region"
            )
        return frame

    def capture_window(self, window: WindowInfo, *, source: str = "mss") -> Frame:
        """Capture the client area of one window.

        Raises:
            CaptureError: If the window is minimised or has no client area.
        """
        if window.minimized:
            raise CaptureError(f"window {window.handle:#x} is minimised; restore it before capturing")
        if window.region.is_empty():
            raise CaptureError(f"window {window.handle:#x} has an empty client area")
        return self.capture_region(window.region, source=source)

    def close(self) -> None:
        """Close the capture backend."""
        self._backend.close()

    def __enter__(self) -> "ScreenCapturer":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
