"""Vision layer: what the agent is allowed to see.

Sensors convert the operating system's rendering surface into frames. Nothing in
this package may read game state; a frame is the entire observation channel.
"""

from __future__ import annotations

from .capture import CaptureBackend, CaptureError, MssCaptureBackend, ScreenCapturer
from .frame import Frame, FrameError, ScreenRegion
from .window import (
    TargetStatus,
    UnsupportedPlatformError,
    WindowBackend,
    WindowError,
    WindowInfo,
    WindowLocator,
    Win32WindowBackend,
    coordinate_scaling_note,
    ensure_dpi_awareness,
    title_matches,
)

__all__ = [
    "CaptureBackend",
    "CaptureError",
    "Frame",
    "FrameError",
    "MssCaptureBackend",
    "ScreenCapturer",
    "ScreenRegion",
    "TargetStatus",
    "UnsupportedPlatformError",
    "Win32WindowBackend",
    "coordinate_scaling_note",
    "WindowBackend",
    "WindowError",
    "WindowInfo",
    "WindowLocator",
    "ensure_dpi_awareness",
    "title_matches",
]
