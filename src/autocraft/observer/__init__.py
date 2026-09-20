"""The observer: a local, read-only window onto the agent's own state.

The specification's rule for this layer is that the agent publishes state and the
observer displays state, and that the observer must never be able to control the
agent. That rule is enforced structurally rather than promised:

* nothing in this package imports :mod:`autocraft.control`, so there is no input
  backend, safety guard or executor reachable from here;
* the HTTP surface implements ``GET`` only and no route mutates anything;
* the bind address is loopback unless a caller explicitly opts out.

Three parts:

* :mod:`~autocraft.observer.snapshot` - the display contract. Frozen dataclasses
  that serialise to plain JSON, plus the pure helpers :func:`frame_freshness` and
  :func:`input_permitted`.
* :mod:`~autocraft.observer.state` - :class:`ObserverState`, the agent's write
  side and the page's read side, plus the demo source.
* :mod:`~autocraft.observer.bridge` - :class:`LoopPublisher`, the adapter that
  feeds an :class:`ObserverState` from a running loop's two callbacks.
* :mod:`~autocraft.observer.server` - :class:`ObserverServer`, a GET-only
  stdlib HTTP server bound to loopback.

Typical use::

    state = ObserverState(config)
    with ObserverServer(state, port=config.observer_port) as server:
        print(server.base_url)
        # the agent calls state.publish_* as it runs
"""

from __future__ import annotations

from .bridge import LoopPublisher, publish_safety_from
from .server import ObserverServer, build_server, default_web_dir, resolve_bind_host
from .snapshot import (
    AFFECT_BASELINE,
    AFFECT_DIMENSIONS,
    AffectState,
    AgentEvent,
    AgentMode,
    Belief,
    EmergencyStopState,
    EventKind,
    FrameFreshness,
    FrameInfo,
    ObserverError,
    ObserverSnapshot,
    RunMetrics,
    SafetyStatus,
    frame_freshness,
    input_permitted,
    with_events,
    with_thoughts,
)
from .state import (
    DEMO_SCRIPT,
    FRAME_CONTENT_TYPE,
    JPEG_QUALITY,
    RATE_WINDOW_SECONDS,
    SHORT_TERM_LIMIT,
    EncodedFrame,
    ObserverState,
    demo_frame,
    demo_state,
    downscale_image,
    encode_jpeg,
)

__all__ = [
    "AFFECT_BASELINE",
    "AFFECT_DIMENSIONS",
    "AffectState",
    "AgentEvent",
    "AgentMode",
    "Belief",
    "DEMO_SCRIPT",
    "EncodedFrame",
    "EmergencyStopState",
    "EventKind",
    "FRAME_CONTENT_TYPE",
    "FrameFreshness",
    "FrameInfo",
    "JPEG_QUALITY",
    "LoopPublisher",
    "ObserverError",
    "ObserverServer",
    "ObserverSnapshot",
    "ObserverState",
    "RATE_WINDOW_SECONDS",
    "RunMetrics",
    "SHORT_TERM_LIMIT",
    "SafetyStatus",
    "build_server",
    "default_web_dir",
    "demo_frame",
    "demo_state",
    "downscale_image",
    "encode_jpeg",
    "frame_freshness",
    "input_permitted",
    "publish_safety_from",
    "resolve_bind_host",
    "with_events",
    "with_thoughts",
]
