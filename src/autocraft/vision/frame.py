"""Frame and screen-geometry primitives.

This module is deliberately free of Win32 and capture-backend details so the
geometry and frame maths can be unit tested anywhere.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = ["Frame", "FrameError", "ScreenRegion"]


class FrameError(RuntimeError):
    """Raised when a frame is malformed or cannot be written."""


@dataclass(frozen=True)
class ScreenRegion:
    """A rectangle in *screen* (virtual desktop) pixel coordinates.

    ``width`` and ``height`` are always positive for a usable region; use
    :meth:`is_empty` to detect degenerate rectangles produced by a minimised or
    hidden window.
    """

    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        for name in ("left", "top", "width", "height"):
            value = getattr(self, name)
            if not isinstance(value, int):
                object.__setattr__(self, name, int(value))

    @property
    def right(self) -> int:
        """Exclusive right edge in screen coordinates."""
        return self.left + self.width

    @property
    def bottom(self) -> int:
        """Exclusive bottom edge in screen coordinates."""
        return self.top + self.height

    @property
    def area(self) -> int:
        """Pixel area of the region."""
        return self.width * self.height

    def is_empty(self) -> bool:
        """True when the region has no pixels to capture."""
        return self.width <= 0 or self.height <= 0

    def contains(self, x: int, y: int) -> bool:
        """True when the screen point lies inside the region."""
        return self.left <= x < self.right and self.top <= y < self.bottom

    def centre(self) -> tuple[int, int]:
        """Return the screen point at the middle of the region."""
        return self.left + self.width // 2, self.top + self.height // 2

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serialisable view."""
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
            "right": self.right,
            "bottom": self.bottom,
        }

    def to_mss_monitor(self) -> dict[str, int]:
        """Return this region in the dict form ``mss`` expects."""
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    @classmethod
    def from_mss_monitor(cls, monitor: dict[str, int]) -> "ScreenRegion":
        """Build a region from an ``mss`` monitor dictionary."""
        return cls(
            left=int(monitor["left"]),
            top=int(monitor["top"]),
            width=int(monitor["width"]),
            height=int(monitor["height"]),
        )

    @classmethod
    def from_client_rect(
        cls,
        client_rect: Sequence[int],
        client_origin: tuple[int, int],
    ) -> "ScreenRegion":
        """Convert a Win32 client rectangle into a screen-space region.

        ``GetClientRect`` reports the client area relative to the window's own
        client origin, so it always starts at ``(0, 0)``; ``ClientToScreen``
        supplies where that origin sits on the desktop. Combining them is the
        only reliable way to capture just the render surface, excluding title
        bar, borders and menu bar.

        Args:
            client_rect: ``(left, top, right, bottom)`` from ``GetClientRect``.
            client_origin: ``(x, y)`` from ``ClientToScreen``.

        Returns:
            The equivalent rectangle in screen coordinates.
        """
        left, top, right, bottom = (int(value) for value in client_rect)
        origin_x, origin_y = (int(value) for value in client_origin)
        return cls(
            left=origin_x + left,
            top=origin_y + top,
            width=max(0, right - left),
            height=max(0, bottom - top),
        )


@dataclass(frozen=True)
class Frame:
    """One captured image of the game's client area.

    ``image`` is an ``H x W x 3`` ``uint8`` array in **RGB** order. Backends that
    hand back BGRA (``mss`` does) must convert before constructing a frame, so
    nothing downstream has to think about channel order.

    Frames are intentionally never serialised into telemetry; the recorder keeps
    a file path instead. See :meth:`signature` and :meth:`difference` for the
    cheap comparisons the agent loop uses.
    """

    image: np.ndarray
    timestamp: float
    region: ScreenRegion | None = None
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        image = self.image
        if not isinstance(image, np.ndarray):
            raise FrameError("frame image must be a numpy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise FrameError(f"frame image must be H x W x 3, got shape {image.shape!r}")
        if image.dtype != np.uint8:
            raise FrameError(f"frame image must be uint8, got {image.dtype!r}")
        if image.size == 0:
            raise FrameError("frame image is empty")
        object.__setattr__(self, "timestamp", float(self.timestamp))

    # -- geometry ---------------------------------------------------------

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return int(self.image.shape[1])

    @property
    def shape(self) -> tuple[int, int, int]:
        """Frame shape as ``(height, width, channels)``."""
        return (self.height, self.width, 3)

    @property
    def nbytes(self) -> int:
        """Raw pixel byte count."""
        return int(self.image.nbytes)

    def matches_region(self) -> bool:
        """True when the image dimensions agree with the recorded region."""
        if self.region is None:
            return True
        return self.height == self.region.height and self.width == self.region.width

    # -- comparison -------------------------------------------------------

    def block_means(self, grid: int = 16) -> np.ndarray:
        """Average luminance over a ``grid x grid`` partition of the frame.

        This is the cheap perceptual summary used for change detection: it
        survives tiny rendering noise and antialiasing while still responding to
        camera rotation and large scene changes. Implemented with pure numpy so
        it costs no extra dependency at import time.

        The average is taken directly over the colour channels rather than
        through a separate greyscale pass. The two are mathematically identical
        - the mean of the channel means is the mean of the block - but the direct
        form avoids allocating a full-frame float image, which is the dominant
        cost on a multi-megapixel frame. The result is exact, not sampled: every
        pixel inside the grid contributes.
        """
        if grid < 1:
            raise FrameError(f"grid must be at least 1, got {grid}")
        image = self.image
        rows, cols = image.shape[0], image.shape[1]
        band_h, band_w = rows // grid, cols // grid
        if band_h < 1 or band_w < 1:
            return np.full((grid, grid), float(image.mean()), dtype=np.float64)
        cropped = image[: band_h * grid, : band_w * grid]
        return cropped.reshape(grid, band_h, grid, band_w, 3).mean(axis=(1, 3, 4), dtype=np.float64)

    def signature(self, grid: int = 16) -> str:
        """Short stable hash of the frame's coarse appearance."""
        quantised = np.clip(self.block_means(grid), 0.0, 255.0).astype(np.uint8)
        return hashlib.sha1(quantised.tobytes()).hexdigest()[:16]

    def difference(self, other: "Frame", grid: int = 16) -> float:
        """Mean absolute luminance difference against another frame, scaled 0..1.

        ``0.0`` means the coarse appearance is identical. Values around ``0.01``
        are usually rendering noise; larger values mean the picture really
        changed.
        """
        if not isinstance(other, Frame):
            raise FrameError("difference() expects another Frame")
        mine = self.block_means(grid)
        theirs = other.block_means(grid)
        return float(np.abs(mine - theirs).mean() / 255.0)

    def mean_luma(self) -> float:
        """Average luminance of the frame, 0..255."""
        return float(self.image.mean())

    # -- output -----------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the frame to disk and return the resolved path.

        Uses OpenCV rather than Pillow so AutoCraft keeps a single image
        dependency. The array is converted RGB -> BGR because that is the order
        OpenCV's encoders expect.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2 = _opencv()
        bgr = self.image[:, :, ::-1]
        if not cv2.imwrite(str(target), bgr):
            raise FrameError(f"could not write frame to {target}")
        return target

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary. Never includes pixel data."""
        payload: dict[str, Any] = {
            "timestamp": self.timestamp,
            "width": self.width,
            "height": self.height,
            "channels": 3,
            "source": self.source,
            "signature": self.signature(),
            "mean_luma": round(self.mean_luma(), 3),
            "region": self.region.to_dict() if self.region is not None else None,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_bgra(
        cls,
        buffer: np.ndarray,
        *,
        timestamp: float,
        region: ScreenRegion | None = None,
        source: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> "Frame":
        """Build a frame from a BGRA buffer, as produced by ``mss``."""
        array = np.asarray(buffer)
        if array.ndim != 3 or array.shape[2] < 3:
            raise FrameError(f"BGRA buffer must be H x W x 4, got shape {array.shape!r}")
        rgb = np.ascontiguousarray(array[:, :, 2::-1])
        return cls(
            image=rgb,
            timestamp=timestamp,
            region=region,
            source=source,
            metadata=dict(metadata or {}),
        )


def _opencv() -> Any:
    try:
        import cv2  # noqa: PLC0415 - intentionally lazy: only saving needs it
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise FrameError(
            "saving frames requires opencv-python; install with 'pip install opencv-python'"
        ) from exc
    return cv2
