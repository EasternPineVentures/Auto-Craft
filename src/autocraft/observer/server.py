"""The observer's HTTP surface: GET-only, loopback-only, read-only.

The specification asks for a local observer page and warns against a large web
application, so this is the standard library's HTTP server and nothing else.
There is no framework, no templating, no build step and no websocket: the page
polls two endpoints, which is the whole protocol.

Three properties are enforced here rather than documented as intentions.

**Read-only.** Only ``GET`` is implemented. Every other method answers ``405``
with ``Allow: GET``. No route mutates anything: ``/api/snapshot`` serialises the
published state, ``/frame`` copies bytes that were encoded when the frame was
published. The observer therefore cannot control the agent, because there is no
request it can make that would.

**Local.** The bind address is checked against
:data:`~autocraft.config.LOOPBACK_HOSTS` and a non-loopback host is refused
unless the caller explicitly passes ``allow_remote=True``, which only the CLI's
``--allow-remote`` flag does. When bound locally, a ``Host`` header naming
something other than a loopback name is rejected as well, so a page on another
site cannot reach the state through a rebinding trick.

**Safe with nothing running.** Starting the server without an agent produces a
page that says so. It does not discover a window, capture anything, or touch the
input layer - it has no way to, since nothing in this package imports
:mod:`autocraft.control`.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..config import DEFAULT_OBSERVER_HOST, LOOPBACK_HOSTS
from .snapshot import ObserverError
from .state import DEMO_TICK_SECONDS, ObserverState

__all__ = [
    "ObserverServer",
    "build_server",
    "resolve_bind_host",
]

_LOGGER = logging.getLogger("autocraft.observer.server")

#: Static files the server will serve, by URL path. A whitelist rather than a
#: directory walk, so a crafted path cannot reach outside the web directory.
_STATIC_FILES: dict[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}

_NO_STORE = "no-store, no-cache, must-revalidate"


def resolve_bind_host(host: str | None, *, allow_remote: bool = False) -> str:
    """Validate the bind address and return it.

    ``None`` means "use the default", which is loopback.

    Raises:
        ObserverError: if ``host`` is not a loopback name and ``allow_remote`` is
            false. The specification says not to expose a network listener
            publicly by default, and a default that can only bind loopback is the
            only version of that promise a reader can verify.
    """
    candidate = DEFAULT_OBSERVER_HOST if host is None else str(host).strip()
    if not candidate:
        raise ObserverError("observer host must not be empty")
    if candidate in LOOPBACK_HOSTS:
        return candidate
    if not allow_remote:
        raise ObserverError(
            f"refusing to bind the observer to {candidate!r}: it is not a loopback address. "
            "Pass allow_remote=True (CLI: --allow-remote) to expose it deliberately."
        )
    return candidate


class _ObserverHTTPServer(ThreadingHTTPServer):
    """Threading HTTP server carrying the state and the web directory."""

    daemon_threads = True
    # ``SO_REUSEADDR`` means the opposite of what its name suggests on Windows:
    # it lets a second socket bind a port that is already being served, so two
    # observers would silently share one URL and the operator could not tell
    # which process answered. Off there; on elsewhere so a restart is not
    # blocked by sockets still in TIME_WAIT.
    allow_reuse_address = os.name != "nt"

    def __init__(
        self,
        address: tuple[str, int],
        state: ObserverState,
        web_dir: Path,
        *,
        allow_remote: bool,
    ) -> None:
        self.state = state
        self.web_dir = Path(web_dir)
        self.allow_remote = bool(allow_remote)
        super().__init__(address, _ObserverRequestHandler)


class _ObserverRequestHandler(BaseHTTPRequestHandler):
    """Serves the page, the snapshot and the current frame. GET only."""

    server_version = "AutoCraftObserver/0.1"
    sys_version = ""

    # -- helpers ----------------------------------------------------------

    @property
    def _state(self) -> ObserverState:
        return self.server.state  # type: ignore[attr-defined]

    def _send(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", _NO_STORE)
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, payload: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _send_text(self, status: HTTPStatus, message: str, *, allow: str | None = None) -> None:
        headers = {"Allow": allow} if allow else None
        self._send(status, message.encode("utf-8"), "text/plain; charset=utf-8", headers=headers)

    def _host_is_local(self) -> bool:
        """True when the request's Host header names a loopback address.

        Only enforced when the server is bound locally: a remote bind was asked
        for explicitly, and there the caller's own network is the boundary. A
        missing Host header is accepted, because a local client that omits it is
        still local.
        """
        if getattr(self.server, "allow_remote", False):
            return True
        header = self.headers.get("Host")
        if not header:
            return True
        name = header.rsplit(":", 1)[0].strip("[]")
        return name in LOOPBACK_HOSTS

    # -- methods ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        """Serve the page, the snapshot or the frame."""
        if not self._host_is_local():
            self._send_text(HTTPStatus.FORBIDDEN, "the observer only answers loopback requests\n")
            return
        path = urlparse(self.path).path
        if path in _STATIC_FILES:
            self._serve_static(path)
            return
        if path == "/api/snapshot":
            try:
                payload = self._state.snapshot_dict()
            except Exception as exc:  # a display failure must not become a 500 storm
                _LOGGER.warning("could not build a snapshot: %s", exc)
                self._send_json(
                    {"error": "snapshot unavailable", "detail": str(exc)},
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            self._send_json(payload)
            return
        if path == "/api/health":
            self._send_json(self._health_payload())
            return
        if path == "/frame":
            self._serve_frame()
            return
        if path == "/favicon.ico":
            self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
            return
        self._send_text(HTTPStatus.NOT_FOUND, f"no such resource: {path}\n")

    def do_HEAD(self) -> None:  # noqa: N802
        """Answer ``405``: the observer serves GET only, and says so plainly."""
        self._reject_method()

    def do_POST(self) -> None:  # noqa: N802
        """Answer ``405``: the observer cannot be asked to change anything."""
        self._reject_method()

    def do_PUT(self) -> None:  # noqa: N802
        """Answer ``405``."""
        self._reject_method()

    def do_PATCH(self) -> None:  # noqa: N802
        """Answer ``405``."""
        self._reject_method()

    def do_DELETE(self) -> None:  # noqa: N802
        """Answer ``405``."""
        self._reject_method()

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Answer ``405`` rather than advertising CORS: this is not an API."""
        self._reject_method()

    def _reject_method(self) -> None:
        self._send_text(
            HTTPStatus.METHOD_NOT_ALLOWED,
            "the observer is read-only; only GET is supported\n",
            allow="GET",
        )

    # -- payloads ---------------------------------------------------------

    def _health_payload(self) -> dict[str, Any]:
        """Report liveness plus the few numbers the page would otherwise guess.

        The freshness thresholds live in the configuration, so sending them means
        the page's LIVE/RECENT/STALE badge cannot drift away from what the server
        calls fresh.
        """
        config = self._state.config
        return {
            "status": "ok",
            "demo": self._state.demo,
            "version": self.server_version,
            "poll_ms": int(config.observer_poll_ms),
            "frame_live_seconds": float(config.observer_frame_live_seconds),
            "frame_stale_seconds": float(config.observer_frame_stale_seconds),
        }

    def _serve_static(self, path: str) -> None:
        filename, content_type = _STATIC_FILES[path]
        target = (self.server.web_dir / filename).resolve()  # type: ignore[attr-defined]
        try:
            body = target.read_bytes()
        except OSError:
            self._send_text(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"the observer's web assets are missing: {target}\n",
            )
            return
        self._send(HTTPStatus.OK, body, content_type)

    def _serve_frame(self) -> None:
        frame = self._state.encoded_frame()
        if frame is None:
            self._send_text(HTTPStatus.NOT_FOUND, "no frame has been published yet\n")
            return
        self._send(HTTPStatus.OK, frame.data, frame.content_type)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Route request logging to the module logger instead of stderr."""
        _LOGGER.debug("%s - %s", self.address_string(), format % args)


def build_server(
    state: ObserverState,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    web_dir: Path | None = None,
    allow_remote: bool = False,
) -> _ObserverHTTPServer:
    """Create the observer's HTTP server without starting it."""
    bind_host = resolve_bind_host(host, allow_remote=allow_remote)
    directory = Path(web_dir) if web_dir is not None else default_web_dir()
    if not directory.is_dir():
        raise ObserverError(f"observer web directory not found: {directory}")
    return _ObserverHTTPServer((bind_host, int(port)), state, directory, allow_remote=allow_remote)


def default_web_dir() -> Path:
    """Return the packaged web asset directory."""
    return Path(__file__).resolve().parent.parent / "web"


class ObserverServer:
    """A running observer server, optionally with its demo ticker.

    Used as a context manager by the CLI so the socket is closed even if the run
    raises. Starting it does not enable game control: it opens a listening socket
    on loopback and reads state that somebody else publishes.
    """

    def __init__(
        self,
        state: ObserverState,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        web_dir: Path | None = None,
        allow_remote: bool = False,
        tick_seconds: float = DEMO_TICK_SECONDS,
    ) -> None:
        self.state = state
        self.httpd = build_server(
            state, host=host, port=port, web_dir=web_dir, allow_remote=allow_remote
        )
        self._allow_remote = bool(allow_remote)
        self._tick_seconds = max(0.05, float(tick_seconds))
        self._thread: threading.Thread | None = None
        self._ticker: threading.Thread | None = None
        self._stop = threading.Event()

    # -- lifecycle --------------------------------------------------------

    @property
    def host(self) -> str:
        """The address actually bound, which resolves ``port=0``."""
        return str(self.httpd.server_address[0])

    @property
    def port(self) -> int:
        """The port actually bound."""
        return int(self.httpd.server_address[1])

    @property
    def base_url(self) -> str:
        """The page's URL, with brackets for an IPv6 literal."""
        host = self.host
        if ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self.port}"

    @property
    def allow_remote(self) -> bool:
        """Whether a non-loopback bind was explicitly permitted."""
        return self._allow_remote

    def start(self) -> str:
        """Start serving in a background thread and return the page URL."""
        if self._thread is not None:
            return self.base_url
        self._thread = threading.Thread(
            target=self.httpd.serve_forever, name="autocraft-observer", daemon=True
        )
        self._thread.start()
        if self.state.demo:
            self._ticker = threading.Thread(
                target=self._demo_loop, name="autocraft-observer-demo", daemon=True
            )
            self._ticker.start()
        return self.base_url

    def serve_forever(self) -> None:
        """Serve in the calling thread until interrupted."""
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:  # pragma: no cover - interactive convenience
            pass

    def stop(self) -> None:
        """Stop the ticker and close the listening socket. Idempotent."""
        self._stop.set()
        self.httpd.shutdown()
        self.httpd.server_close()
        for thread in (self._ticker, self._thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)
        self._ticker = None
        self._thread = None

    def __enter__(self) -> "ObserverServer":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    # -- demo -------------------------------------------------------------

    def _demo_loop(self) -> None:
        """Advance the demo script on a wall-clock cadence, never faster."""
        while not self._stop.is_set():
            if self._stop.wait(self._tick_seconds):
                return
            try:
                self.state.tick()
            except Exception as exc:  # pragma: no cover - a demo must not kill the server
                _LOGGER.warning("demo tick failed: %s", exc)
