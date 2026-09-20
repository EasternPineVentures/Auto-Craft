"""AutoCraft's command line interface.

Design rule for every command here: **looking is always safe, touching is always
explicit.** ``status``, ``capture`` and ``observe`` read pixels and print facts -
they never inject input, so they can be run freely. ``input-test`` and
``look-test`` are the only commands that can move the mouse or press a key, they
say so before they do anything, and they refuse to run without ``--yes``.

Nothing here starts autonomous play. The only shipped decision policy is the
no-op policy, and the loop refuses to run without an explicit bound.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .agent.action import Action, ActionExecutor
from .agent.decision import NoOpDecisionPolicy
from .agent.loop import AgentLoop
from .agent.observation import Observer
from .config import Config, ConfigError, DEFAULT_CONFIG_FILENAME, load_config
from .control.errors import ControlError, InputBlocked
from .control.keyboard import Keyboard
from .control.keymap import known_key_names
from .control.mouse import Mouse
from .control.safety import SafetyGuard
from .look import (
    EVENT_ERROR,
    EVENT_INFO,
    EVENT_OBSERVE,
    EVENT_SAFETY,
    EXPERIMENT_NAME,
    TRIAL_COMPLETED,
    FocusCheck,
    LookRecorder,
    LookRunner,
    LookTrialResult,
    MoveOutcome,
    TrialSpec,
)
from .observer import (
    LoopPublisher,
    ObserverError,
    ObserverServer,
    ObserverState,
    demo_state,
    publish_safety_from,
    resolve_bind_host,
)
from .observer.snapshot import AgentMode
from .telemetry.recorder import RunRecorder
from .vision.capture import CaptureError, MssCaptureBackend, ScreenCapturer
from .vision.window import (
    TargetStatus,
    WindowBackend,
    WindowError,
    WindowLocator,
    Win32WindowBackend,
    ensure_dpi_awareness,
)

__all__ = ["build_parser", "main"]


# ---------------------------------------------------------------------------
# runtime assembly
# ---------------------------------------------------------------------------


@dataclass
class Runtime:
    """The assembled pieces of one AutoCraft session."""

    config: Config
    window_backend: WindowBackend
    locator: WindowLocator
    capturer: ScreenCapturer
    observer: Observer
    dpi_mode: str
    config_source: str

    def close(self) -> None:
        """Release capture resources."""
        try:
            self.capturer.close()
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass


def _build_runtime(config: Config, *, config_source: str) -> Runtime:
    dpi_mode = ensure_dpi_awareness()
    window_backend = Win32WindowBackend()
    locator = WindowLocator(window_backend, config.target_title_patterns)
    capturer = ScreenCapturer(MssCaptureBackend())
    return Runtime(
        config=config,
        window_backend=window_backend,
        locator=locator,
        capturer=capturer,
        observer=Observer(locator, capturer),
        dpi_mode=dpi_mode,
        config_source=config_source,
    )


def _build_guard(config: Config, locator: WindowLocator) -> SafetyGuard:
    """Create the safety guard, binding the foreground lock to the locator."""
    from .control.win32_input import UnsupportedPlatformError, Win32InputBackend

    try:
        backend = Win32InputBackend()
    except UnsupportedPlatformError as exc:
        raise ConfigError(str(exc)) from exc
    return SafetyGuard(config, backend, target_is_foreground=locator.is_target_foreground)


def _input_backend_report() -> tuple[bool, str]:
    """Report whether input injection is available, without injecting anything."""
    try:
        from .control.win32_input import Win32InputBackend

        backend = Win32InputBackend()
    except Exception as exc:  # noqa: BLE001 - reported to the user, never raised
        return False, f"unavailable ({type(exc).__name__}: {exc})"
    try:
        backend.close()
    except Exception:  # noqa: BLE001
        pass
    return True, "ready (Windows SendInput)"


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------


def _print_table(rows: Sequence[tuple[str, Any]]) -> None:
    width = max((len(label) for label, _ in rows), default=0)
    for label, value in rows:
        print(f"  {label.ljust(width)} : {value}")


#: Remaining seconds below which the countdown stops ticking. Guards against a
#: final epsilon tick from float subtraction.
_COUNTDOWN_EPSILON = 1e-6


def _is_interactive(stream: Any) -> bool:
    """Whether ``stream`` is a terminal, tolerating streams that cannot say."""
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def _focus_countdown(seconds: float, *, stream: Any) -> None:
    """Wait out the focus handoff, counting down when attached to a terminal.

    A silent wait is indistinguishable from a hang, and the operator is looking
    at the game window rather than the shell, so they need to see that the
    command is still alive and how long is left. Off a terminal (pipes,
    redirected logs, the test suite) this degrades to one plain sleep so no
    carriage-return noise ends up in captured output.
    """
    if seconds <= 0:
        return
    if not _is_interactive(stream):
        stream.write(f"  sending in {seconds:g}s...\n")
        stream.flush()
        time.sleep(seconds)
        return
    remaining = seconds
    while remaining > _COUNTDOWN_EPSILON:
        stream.write(f"\r  sending in {remaining:>4.1f}s...   ")
        stream.flush()
        step = min(1.0, remaining)
        time.sleep(step)
        remaining -= step
    stream.write("\r" + " " * 32 + "\r")
    stream.flush()


def _rerun_hint(args: argparse.Namespace) -> str:
    """The exact ``input-test`` invocation that repeats this run for real.

    Echoing the operator's own flags back means the suggested command is
    copy-pasteable and cannot drift from what they actually asked for.
    """
    parts = ["autocraft", "input-test", "--action", args.action]
    if args.action == "key-tap":
        parts += ["--key", args.key, "--hold", f"{args.hold:g}"]
    else:
        parts += ["--dx", str(args.dx), "--dy", str(args.dy)]
    if args.focus_delay is not None:
        parts += ["--focus-delay", f"{args.focus_delay:g}"]
    parts.append("--yes")
    return " ".join(parts)


def _status_lines(status: TargetStatus) -> list[tuple[str, Any]]:
    window = status.window
    region = window.region if window is not None else None
    return [
        ("target found", "yes" if status.found else "no"),
        ("window title", window.title if window is not None else "-"),
        ("window handle", f"0x{window.handle:X}" if window is not None else "-"),
        ("process id", window.process_id if window is not None else "-"),
        ("foreground", "yes" if status.is_foreground else "no"),
        ("minimized", "yes" if (window.minimized if window is not None else False) else "no"),
        (
            "client area",
            f"{region.width}x{region.height} at ({region.left},{region.top})" if region is not None else "-",
        ),
        ("capturable", "yes" if status.can_capture else "no"),
        ("input allowed", "yes" if status.can_inject_input else "no"),
        ("reason", status.reason or "-"),
    ]


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    """Report whether AutoCraft can see and reach the game right now."""
    config, source = _load(args)
    runtime = _build_runtime(config, config_source=source)
    try:
        status = runtime.locator.status(force=True)
        input_ok, input_note = _input_backend_report()
        print("AutoCraft status")
        print(f"  version          : {__version__}")
        print(f"  config source    : {source}")
        print(f"  title patterns   : {', '.join(config.target_title_patterns)}")
        print(f"  DPI awareness    : {runtime.dpi_mode}")
        print(f"  emergency stop   : {config.emergency_stop_key}")
        print()
        print("Target window")
        _print_table(_status_lines(status))
        print()
        print("Subsystems")
        _print_table(
            [
                ("window discovery", "ready (Win32)"),
                ("screen capture", "ready (mss)"),
                ("input injection", input_note if not input_ok else "ready (Win32 SendInput)"),
                ("decision policy", "noop (no autonomous play in V0)"),
            ]
        )
        print()
        matches = runtime.locator.find_all()
        if matches:
            print("All matching windows")
            for window in matches:
                region = window.region
                print(
                    f"  - 0x{window.handle:X} {window.title!r} "
                    f"{region.width}x{region.height} at ({region.left},{region.top})"
                )
        else:
            print("All matching windows: none")
        print()
        print("Verdict")
        if not status.found:
            print("  not ready - no window matched the title patterns above.")
            print("  start the game, or adjust target_title_patterns in the config.")
        elif status.window is not None and status.window.minimized:
            print("  not ready - the target window is minimized; restore it first.")
        else:
            print("  ready - the game window was found and can be captured.")
            if not status.is_foreground:
                print("  note - 'input allowed: no' is expected here: this terminal has focus,")
                print("         and the foreground lock only permits input while the game is")
                print("         focused. input-test gives you a countdown to switch to it.")
            print("  next - autocraft input-test --action key-tap --key w --hold 0.05 --yes")
        return 0 if status.found else 1
    finally:
        runtime.close()


def cmd_capture(args: argparse.Namespace) -> int:
    """Save exactly one client-area frame. Never injects input."""
    config, source = _load(args)
    config.ensure_directories()
    runtime = _build_runtime(config, config_source=source)
    try:
        status = runtime.locator.status(force=True)
        if not status.found or status.window is None:
            print(f"no target window matched {config.target_title_patterns!r}", file=sys.stderr)
            return 1
        if status.window.minimized:
            print("target window is minimized; restore it before capturing", file=sys.stderr)
            return 1

        if args.output:
            destination = Path(args.output)
        else:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            destination = config.captures_dir / f"capture-{stamp}.{config.image_format}"

        frame = runtime.capturer.capture_window(status.window)
        written = frame.save(destination)
        print("captured one client-area frame")
        _print_table(
            [
                ("window", f"{status.window.title!r} (0x{status.window.handle:X})"),
                ("region", f"{frame.width}x{frame.height} at ({frame.region.left},{frame.region.top})" if frame.region else "-"),
                ("frame", f"{frame.width}x{frame.height} RGB uint8"),
                ("path", written),
                ("bytes", frame.nbytes),
                ("mean luma", f"{frame.mean_luma():.2f}"),
                ("signature", frame.signature()),
            ]
        )
        print()
        print("No input was sent. This command only observes.")
        return 0
    except CaptureError as exc:
        print(f"capture failed: {exc}", file=sys.stderr)
        return 2
    finally:
        runtime.close()


def _capture_rate_warning(effective_fps: float, target_fps: float) -> str | None:
    """Return an explanation when observation missed its target rate.

    A grab costs time proportional to the client area, and that cost competes
    with the game's own event loop, so falling short of the target is worth
    explaining rather than just reporting. Returns ``None`` when the target was
    met closely enough to be uninteresting.
    """
    if target_fps <= 0 or effective_fps >= target_fps * 0.75:
        return None
    return (
        f"WARNING: {effective_fps:.2f} fps is well below the {target_fps:g} fps target.\n"
        "  A full-screen grab costs time proportional to the client area, and that cost\n"
        "  competes with the game's own event loop (Luanti logs 'SDL_PollEvent took too\n"
        "  long' when it is starved, which shows up as stutter).\n"
        "  Remedies, cheapest first:\n"
        "    - run the game in a smaller window rather than full-screen\n"
        f"    - lower capture_fps in autocraft.toml (currently {target_fps:g})"
    )


def _fan_out(callbacks: Sequence[Any]) -> Any:
    """Combine step callbacks into one, preserving order and dropping ``None``.

    No exception handling here on purpose: the observer publisher is already
    total, so anything that does raise is a real bug and should not be hidden
    behind a silent dashboard.
    """
    live = [callback for callback in callbacks if callback is not None]
    if not live:
        return None
    if len(live) == 1:
        return live[0]

    def _call(payload: Any) -> None:
        for callback in live:
            callback(payload)

    return _call


def _start_observer(
    config: Config,
    *,
    run_id: str,
    goal: str | None = None,
    intention: str | None = None,
    safety: Any = None,
    express_thoughts: bool = True,
    state: ObserverState | None = None,
) -> tuple[ObserverServer | None, LoopPublisher | None]:
    """Start the read-only observer page for a run.

    Deliberately constructs nothing from :mod:`autocraft.control`: starting the
    page must never make game control possible, so no input backend and no safety
    guard are created here. The caller supplies whatever safety facts it already
    has, or none.

    A page that cannot bind its port is reported and skipped, and the pair
    ``(None, None)`` is returned. The run is the important thing and the page is
    an accessory, so a port already in use must not end an observation.

    ``state`` lets a caller that needs to publish beyond the two loop callbacks -
    ``look-test`` publishing a measurement, for instance - keep the
    :class:`ObserverState` it will write through. The page is still built and
    owned entirely here.
    """
    if state is None:
        state = ObserverState(config)
    try:
        server = ObserverServer(
            state, host=config.observer_host, port=config.observer_port
        )
    except (OSError, ObserverError) as exc:
        print(
            f"observer page unavailable: could not bind "
            f"{config.observer_host}:{config.observer_port} ({exc})",
            file=sys.stderr,
        )
        return None, None
    publisher = LoopPublisher(
        state,
        goal=goal,
        intention=intention,
        safety=safety,
        express_thoughts=express_thoughts,
    )
    publisher.begin(run_id)
    url = server.start()
    print(f"observer page: {url}  (read-only, loopback, Ctrl+C to stop)")
    if state.demo:
        print("observer page is showing DEMO data")
    return server, publisher


def cmd_observer(args: argparse.Namespace) -> int:
    """Serve the local observer page. Read-only: it cannot touch the game."""
    config, _source = _load(args)
    host = args.host if args.host is not None else config.observer_host
    try:
        bind_host = resolve_bind_host(host, allow_remote=bool(args.allow_remote))
    except ValueError as exc:
        print(f"observer: {exc}", file=sys.stderr)
        return 2
    port = int(args.port if args.port is not None else config.observer_port)
    if port <= 0 or port > 65535:
        print(f"observer: port {port} is outside 1-65535", file=sys.stderr)
        return 2

    if args.demo:
        state = demo_state(config)
    else:
        state = ObserverState(config)
        state.begin_run("-", goal="")
        state.publish_safety(control_available=False)
        state.publish_mode(
            AgentMode.IDLE,
            note="observer only: no agent loop is running and no input is possible",
        )
        state.publish_event(
            "Observer started with no agent attached. Start 'loop --observer' or "
            "'observe --observer' to publish a live run.",
            now=time.time(),
        )

    try:
        server = ObserverServer(
            state, host=bind_host, port=port, allow_remote=bool(args.allow_remote)
        )
    except (OSError, ObserverError) as exc:
        print(f"observer: could not bind {bind_host}:{port} ({exc})", file=sys.stderr)
        return 1
    try:
        url = server.start()
    except OSError as exc:
        server.stop()
        print(f"observer: could not bind {bind_host}:{port} ({exc})", file=sys.stderr)
        return 1
    print(f"observer page: {url}")
    if state.demo:
        print("DEMO data: every value on this page is scripted, not measured.")
    if bind_host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            f"WARNING: bound to {bind_host}, which is not loopback. "
            "Anyone who can reach this port can read the run state."
        )
    print("read-only: the page cannot inject input, and this command never starts the agent")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping observer")
    finally:
        server.stop()
    return 0


def cmd_observe(args: argparse.Namespace) -> int:
    """Observe for a bounded time. Reports rates. Never injects input."""
    config, source = _load(args)
    config.ensure_directories()
    runtime = _build_runtime(config, config_source=source)
    recorder = RunRecorder.new_run(
        config.runs_dir,
        config=config.to_dict(),
    )
    frames_dir = recorder.frames_dir
    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    captured = 0
    failed = 0
    focus_lost = 0
    saved = 0
    differences: list[float] = []
    previous = None
    started = time.time()
    deadline = started + args.seconds

    print(f"observing for {args.seconds:g}s at up to {config.capture_fps:g} fps (observation only, no input)")
    print(f"run id: {recorder.run_id}")
    observer_server = None
    publisher = None
    if args.observer:
        observer_server, publisher = _start_observer(
            config,
            run_id=recorder.run_id,
            goal="Observe only: V0 sets no goal.",
            intention="Watch the target window and report what changes.",
            safety=lambda: {"control_available": False},
        )
    try:
        while time.time() < deadline:
            loop_started = time.monotonic()
            observation = runtime.observer.observe(captured + failed)
            if publisher is not None:
                publisher.on_observation(observation)
            if observation.frame is None:
                failed += 1
            else:
                captured += 1
                if previous is not None:
                    differences.append(previous.difference(observation.frame))
                previous = observation.frame
                if args.save_frames and saved < args.save_frames:
                    path = runtime.observer.save_frame(
                        observation,
                        frames_dir,
                        stem=f"observe-{captured:06d}",
                    )
                    if path is not None:
                        saved += 1
            if not observation.window.is_foreground:
                focus_lost += 1
            recorder.record_step(
                {
                    "index": observation.index,
                    "observation": observation.to_dict(),
                    "action": None,
                    "result": None,
                }
            )
            elapsed = time.time() - started
            if args.duration_report and captured % args.duration_report == 0:
                print(f"  t={elapsed:6.2f}s frames={captured} failed={failed}")
            if args.steps is not None and captured + failed >= args.steps:
                break
            interval = config.step_interval
            if interval > 0:
                remaining = interval - (time.monotonic() - loop_started)
                if remaining > 0:
                    time.sleep(remaining)

        elapsed = max(1e-6, time.time() - started)
        status = runtime.locator.status()
        record = recorder.finish(
            status="observed",
            stop_reason=f"observation window of {args.seconds:g}s ended",
        )
        if publisher is not None:
            publisher.finish(record)
        mean_diff = sum(differences) / len(differences) if differences else 0.0
        effective_fps = captured / elapsed
        print()
        print("Observation summary")
        _print_table(
            [
                ("duration", f"{elapsed:.2f}s"),
                ("frames captured", captured),
                ("frames failed", failed),
                ("effective fps", f"{effective_fps:.2f}"),
                ("target fps", f"{config.capture_fps:g}"),
                ("frames saved", saved),
                ("mean frame difference", f"{mean_diff:.3f}"),
                ("steps where target was not foreground", focus_lost),
                ("final focus state", "foreground" if status.is_foreground else "not foreground"),
                ("telemetry", recorder.record_path),
            ]
        )
        if captured:
            warning = _capture_rate_warning(effective_fps, config.capture_fps)
            if warning is not None:
                print()
                print(warning)
        print()
        print(f"No input was sent. Step log: {recorder.steps_path}")
        print(f"Run status: {record.status}")
        return 0
    except KeyboardInterrupt:
        recorder.record_error("KeyboardInterrupt", context="observe")
        recorder.finish(status="interrupted", stop_reason="Ctrl+C")
        print("\ninterrupted by Ctrl+C; no input was sent")
        return 130
    finally:
        if observer_server is not None:
            observer_server.stop()
        runtime.close()


#: Seconds the operator gets to bring the game window forward in ``input-test``.
#: A first run is awkward: the operator has to read the message, find the game
#: window and click into it, all while the shell that launched the command is
#: covering it. Ten seconds is comfortable for that without leaving the operator
#: wondering whether the command has hung. Overridable with ``--focus-delay``,
#: and patched to zero by the tests so no real delay is needed.
FOCUS_HANDOFF_SECONDS = 10.0


def _confirm_target_focus(locator: WindowLocator, *, expected_handle: int) -> tuple[bool, str]:
    """Re-check the target window immediately before injection.

    Returns ``(ready, reason)``. The check is repeated *after* the focus
    handoff on purpose: the window matched before the countdown may since have
    been closed, minimised, or replaced by a different window, and the
    foreground lock is a promise about the *exact* window the operator was told
    to focus - not merely about "some matching window".
    """
    status = locator.status(force=True)
    if not status.found or status.window is None:
        return False, "the target window disappeared during the focus countdown"
    if status.window.minimized:
        return False, "the target window is minimized"
    if status.window.handle != expected_handle:
        return False, (
            f"the target window changed during the focus countdown "
            f"(expected 0x{expected_handle:X}, found 0x{status.window.handle:X})"
        )
    if not status.is_foreground:
        return False, "the target window is not the foreground window (foreground lock active)"
    return True, ""


def cmd_input_test(args: argparse.Namespace) -> int:
    """The one command that can inject input. Explicit, bounded, single action.

    The ordering here is load-bearing. ``input-test`` is normally launched from
    a shell, so at the first check the *shell* is the foreground window -
    refusing on that would make the documented smoke test impossible to run.
    So: locate and vet the target, print the bounded action, demand ``--yes``
    before any injection machinery exists, and only then give the operator a
    window to focus the game. Focus is re-checked after that handoff, and the
    guard independently re-checks it again inside the actuator call.
    """
    # Argument validation comes first so a malformed handoff window is reported
    # as a usage error regardless of what the window layer currently reports.
    handoff = FOCUS_HANDOFF_SECONDS if args.focus_delay is None else args.focus_delay
    if not math.isfinite(handoff) or handoff < 0:
        print(
            f"invalid --focus-delay {args.focus_delay!r}: expected a non-negative number of seconds",
            file=sys.stderr,
        )
        return 2

    config, source = _load(args)
    config.ensure_directories()
    runtime = _build_runtime(config, config_source=source)
    try:
        status = runtime.locator.status(force=True)
        if not status.found or status.window is None:
            print(f"no target window matched {config.target_title_patterns!r}; nothing sent", file=sys.stderr)
            return 1
        if status.window.minimized:
            print("the target window is minimized; nothing sent", file=sys.stderr)
            return 1

        action = _smoke_action(args)
        if action is None:
            print(f"unknown smoke action {args.action!r}", file=sys.stderr)
            return 2

        target_title = status.window.title
        target_handle = status.window.handle

        print("INPUT SMOKE TEST")
        print()
        print("  this is the only AutoCraft command that sends input to the game")
        print(f"  press {config.emergency_stop_key.upper()} at any time to abort and release everything")
        print()
        _print_table(
            [
                (
                    "mode",
                    "LIVE - one input action will be sent"
                    if args.yes
                    else "DRY RUN - nothing will be sent",
                ),
                ("target", repr(target_title)),
                ("handle", f"0x{target_handle:X}"),
                ("action", action.describe()),
            ]
        )

        if not args.yes:
            print()
            print("refusing to send input without --yes; nothing was sent")
            print("to send it for real, re-run with --yes:")
            print()
            print(f"    {_rerun_hint(args)}")
            return 3

        guard = _build_guard(config, runtime.locator)
        guard.install_atexit()
        keyboard = Keyboard(guard, guard.backend, config)
        mouse = Mouse(guard, guard.backend, config)
        executor = ActionExecutor(keyboard, mouse)
        recorder = RunRecorder.new_run(config.runs_dir, config=config.to_dict())

        exit_code = 1
        result_dict: dict[str, Any] = {
            "attempted": False,
            "executed": False,
            "description": action.describe(),
        }
        try:
            guard.release_all("input-test start")

            print()
            _print_table([("run id", recorder.run_id)])
            print()
            print("  Focus the game window now.")
            print("  It is checked again immediately before anything is sent.")
            _focus_countdown(handoff, stream=sys.stdout)

            ready, refusal = _confirm_target_focus(runtime.locator, expected_handle=target_handle)
            if not ready:
                result_dict["blocked_reason"] = refusal
                print()
                sys.stdout.flush()
                print(f"  refused: {refusal}", file=sys.stderr)
                print("  nothing was sent", file=sys.stderr)
                print(f"  hint: click into {target_title!r} during the countdown,", file=sys.stderr)
                print("        or allow more time with --focus-delay 20", file=sys.stderr)
                exit_code = 1
            else:
                # Deliberate defence in depth, not duplicated bookkeeping: the
                # check above reads the window, this one goes through the guard
                # that will actually gate the actuator, and it is what catches
                # focus changing in the moment between the two.
                decision = guard.authorize(action.describe())
                if not decision.allowed:
                    result_dict["blocked_reason"] = decision.reason
                    print()
                    sys.stdout.flush()
                    print(f"  blocked: {decision.reason}", file=sys.stderr)
                    print("  nothing was sent", file=sys.stderr)
                    exit_code = 1
                else:
                    guard.wait_for_rate_limit()
                    result = executor.execute(action)
                    guard.note_action()
                    result_dict = result.to_dict()
                    print()
                    rows: list[tuple[str, Any]] = [
                        ("attempted", result.attempted),
                        ("executed", result.executed),
                        ("duration", f"{result.duration:.3f}s"),
                    ]
                    if result.blocked_reason:
                        rows.append(("blocked", result.blocked_reason))
                    if result.error:
                        rows.append(("error", result.error))
                    _print_table(rows)
                    print()
                    print(
                        "  sent - the game has been given this input"
                        if result.ok
                        else "  not sent - the action did not complete"
                    )
                    exit_code = 0 if result.ok else 1
        finally:
            # Every exit path releases, including Ctrl+C during the countdown and
            # every refusal above. ``release_all`` already covers mouse buttons,
            # so there is no separate button release to report.
            released = guard.release_all("input-test end")
            guard.shutdown("input-test finished")
            recorder.record_step(
                {
                    "index": 0,
                    "observation": {"window": status.to_dict()},
                    "action": action.to_dict(),
                    "result": result_dict,
                }
            )
            recorder.record_safety_events(guard.events)
            record = recorder.finish(
                status="smoke-test", stop_reason="single bounded action completed"
            )
            print()
            _print_table(
                [
                    ("released inputs", ", ".join(released) or "none"),
                    ("telemetry", record.directory),
                ]
            )
            print()
            print("Nothing else will be sent. AutoCraft is idle.")
        return exit_code
    finally:
        runtime.close()


def _smoke_action(args: argparse.Namespace) -> Action | None:
    """Build the single bounded action used by the smoke test."""
    if args.action == "key-tap":
        return Action.key_tap(args.key, args.hold)
    if args.action == "mouse-move":
        return Action.mouse_move(args.dx, args.dy)
    return None


# ---------------------------------------------------------------------------
# LOOK-001
# ---------------------------------------------------------------------------

#: Client-area size above which LOOK-001 warns that the window is large.
#: The specification's recommendation is roughly 1280x650 to 1280x720. This is a
#: warning only: nothing here resizes, moves or reconfigures the game window,
#: because a command that reconfigures the thing it is measuring would change the
#: measurement.
LOOK_RECOMMENDED_MAX_WIDTH = 1280
LOOK_RECOMMENDED_MAX_HEIGHT = 720


def _look_plan(args: argparse.Namespace, config: Config) -> tuple[tuple[TrialSpec, ...], str]:
    """Build the bounded trial plan, or raise ``ValueError`` with the reason.

    Every plan is finite. There is no unbounded mode, no "keep going until it
    works", and no calibration that runs as long as it likes: the series length
    comes from ``look_calibration_deltas`` and is capped by ``look_max_steps``.
    """
    settle = config.look_settle_seconds if args.settle is None else args.settle
    if not math.isfinite(settle) or settle < 0:
        raise ValueError(f"invalid --settle {args.settle!r}: expected a non-negative number of seconds")
    if settle > 10:
        raise ValueError(f"invalid --settle {settle:g}: a settle above 10s is not a bounded wait")

    steps = int(args.steps)
    if steps < 1:
        raise ValueError(f"invalid --steps {args.steps}: at least one trial is required")
    if steps > config.look_max_steps:
        raise ValueError(
            f"invalid --steps {args.steps}: the configured bound is "
            f"look_max_steps={config.look_max_steps}"
        )

    if args.calibrate_horizontal or args.calibrate_vertical:
        deltas = tuple(int(value) for value in config.look_calibration_deltas)
        if len(deltas) > config.look_max_steps:
            raise ValueError(
                f"look_calibration_deltas has {len(deltas)} entries, above the "
                f"look_max_steps bound of {config.look_max_steps}"
            )
        if args.calibrate_horizontal:
            specs = tuple(
                TrialSpec(index=index, dx=delta, dy=0, settle_seconds=settle)
                for index, delta in enumerate(deltas)
            )
            return specs, f"horizontal calibration series at x deltas {list(deltas)}"
        specs = tuple(
            TrialSpec(index=index, dx=0, dy=delta, settle_seconds=settle)
            for index, delta in enumerate(deltas)
        )
        return specs, f"vertical calibration series at y deltas {list(deltas)}"

    limit = int(config.max_mouse_delta)
    if abs(int(args.dx)) > limit or abs(int(args.dy)) > limit:
        raise ValueError(
            f"delta ({args.dx}, {args.dy}) exceeds max_mouse_delta={limit} per axis; "
            "AutoCraft refuses an oversized movement rather than clamping it"
        )
    if args.dx == 0 and args.dy == 0:
        raise ValueError("delta (0, 0) would not move anything, so nothing could be measured")
    specs = tuple(
        TrialSpec(index=index, dx=int(args.dx), dy=int(args.dy), settle_seconds=settle)
        for index in range(steps)
    )
    return specs, f"{steps} x ({args.dx:+d}, {args.dy:+d}) then the exact reverse"


def _look_rerun_hint(args: argparse.Namespace) -> str:
    """The exact ``look-test`` invocation that repeats this run for real.

    Echoing the operator's own flags back means the suggested command is
    copy-pasteable and cannot drift from what they actually asked for.
    """
    parts = ["autocraft", "look-test"]
    if args.calibrate_horizontal:
        parts.append("--calibrate-horizontal")
    elif args.calibrate_vertical:
        parts.append("--calibrate-vertical")
    else:
        parts += ["--dx", str(args.dx), "--dy", str(args.dy), "--steps", str(args.steps)]
    if args.settle is not None:
        parts += ["--settle", f"{args.settle:g}"]
    if args.focus_delay is not None:
        parts += ["--focus-delay", f"{args.focus_delay:g}"]
    if args.output:
        parts += ["--output", args.output]
    parts.append("--yes")
    return " ".join(parts)


def _look_window_warning(width: int, height: int, config: Config) -> str | None:
    """Explain an oversized client area, without doing anything about it.

    A big window is not an error - the operator may want one - but the
    specification recommends roughly 1280x650 to 1280x720, because a phase
    correlation over a very large frame is slower and its estimate is no more
    meaningful for the extra pixels. This reports that and stops.
    """
    pixels = int(width) * int(height)
    if pixels <= config.look_large_window_pixels:
        return None
    return (
        f"WARNING: the client area is {width}x{height} ({pixels} pixels), above the\n"
        f"  configured look_large_window_pixels={config.look_large_window_pixels}.\n"
        f"  LOOK-001 is recommended at about "
        f"{LOOK_RECOMMENDED_MAX_WIDTH}x{LOOK_RECOMMENDED_MAX_HEIGHT} or smaller.\n"
        "  AutoCraft will not resize your window - that would change the thing being\n"
        "  measured. Resize it yourself, or raise look_large_window_pixels if you mean it."
    )


def _look_move(mouse: Mouse, direction: str) -> Any:
    """Wrap :meth:`Mouse.move_relative` as a :class:`MoveOutcome` factory.

    This is the only place LOOK-001 reaches the actuator, and it reaches it
    through the existing V0 :class:`~autocraft.control.mouse.Mouse`, so the guard,
    the rate limit, the per-axis bound and the release bookkeeping are all exactly
    the ones the rest of AutoCraft uses. There is no second input path.
    """

    def move(dx: int, dy: int) -> MoveOutcome:
        try:
            mouse.move_relative(dx, dy)
        except InputBlocked as exc:
            return MoveOutcome(sent=False, detail=f"{direction} ({dx}, {dy}): {exc}", refused=True)
        except (ControlError, ValueError) as exc:
            return MoveOutcome(sent=False, detail=f"{direction} ({dx}, {dy}): {exc}")
        except Exception as exc:  # noqa: BLE001 - a broken backend must not escape
            return MoveOutcome(sent=False, detail=f"{direction} ({dx}, {dy}): {type(exc).__name__}: {exc}")
        return MoveOutcome(sent=True, detail=f"{direction} ({dx}, {dy})")

    return move


def _look_event_printer(state: Any, quiet: bool):
    """Timeline printer for the LOOK sequence, and optional observer mirror."""

    def emit(message: str, kind: str) -> None:
        if state is not None:
            try:
                state.publish_event(message, kind=kind)
            except Exception:  # noqa: BLE001 - display must never break the experiment
                pass
        if quiet:
            return
        marker = {EVENT_ERROR: "!", EVENT_SAFETY: "~", EVENT_OBSERVE: ".", EVENT_INFO: "-"}.get(kind, "-")
        print(f"  {marker} {message}", flush=True)

    return emit


def cmd_look_test(args: argparse.Namespace) -> int:
    """LOOK-001: measure the picture's response to one known mouse movement.

    The same ordering rule as ``input-test`` applies. ``look-test`` is normally
    launched from a shell, so at the first check the *shell* is the foreground
    window, and refusing on that would make the documented trial impossible to
    run. So: locate and vet the target, validate the whole bounded plan, print
    exactly what will be sent, demand ``--yes`` before any injection machinery
    exists, and only then give the operator a window to focus the game. Focus is
    re-checked before every single movement, and the guard re-checks it again
    inside the actuator call.

    What this command does not do: it does not resize the window, it does not
    decide whether the result is good, and it does not claim to have learned
    anything. It injects a known delta, measures the frames, and writes the
    numbers down.
    """
    handoff = FOCUS_HANDOFF_SECONDS if args.focus_delay is None else args.focus_delay
    if not math.isfinite(handoff) or handoff < 0:
        print(
            f"invalid --focus-delay {args.focus_delay!r}: expected a non-negative number of seconds",
            file=sys.stderr,
        )
        return 2

    config, source = _load(args)
    config.ensure_directories()

    try:
        plan, plan_note = _look_plan(args, config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    movements_planned = 2 * len(plan)

    runtime = _build_runtime(config, config_source=source)
    try:
        status = runtime.locator.status(force=True)
        if not status.found or status.window is None:
            print(
                f"no target window matched {config.target_title_patterns!r}; nothing sent",
                file=sys.stderr,
            )
            return 1
        if status.window.minimized:
            print("the target window is minimized; nothing sent", file=sys.stderr)
            return 1

        target_title = status.window.title
        target_handle = status.window.handle
        region = status.window.region
        window_width, window_height = region.width, region.height
        look_dir = Path(args.output) if args.output else Path(runtime.config.runs_dir) / "look"

        print("LOOK-001: MEASURED SENSORIMOTOR MAPPING")
        print()
        print("  LIVE INPUT WILL OCCUR: this command moves the mouse pointer")
        print(f"  press {config.emergency_stop_key.upper()} at any time to abort and release everything")
        print()
        _print_table(
            [
                (
                    "mode",
                    f"LIVE - {movements_planned} mouse movement(s) will be sent"
                    if args.yes
                    else f"DRY RUN - {movements_planned} movement(s) planned, nothing will be sent",
                ),
                ("target", repr(target_title)),
                ("handle", f"0x{target_handle:X}"),
                ("client area", f"{window_width}x{window_height}"),
                ("sequence", plan_note),
                ("per trial", plan[0].describe()),
                ("trials", f"{len(plan)} (bounded; there is no unbounded mode)"),
                ("settle", f"{plan[0].settle_seconds:g}s between movement and capture"),
                ("calibration", "no" if not (args.calibrate_horizontal or args.calibrate_vertical) else "yes"),
                ("output", str(look_dir)),
                ("observer", "shown at the end" if args.observer else "off (pass --observer to watch live)"),
                ("config", source),
            ]
        )

        warning = _look_window_warning(window_width, window_height, config)
        if warning is not None:
            print()
            print(warning)

        if not args.yes:
            print()
            print("refusing to send input without --yes; nothing was sent")
            print("to run the measurement for real, re-run with --yes:")
            print()
            print(f"    {_look_rerun_hint(args)}")
            return 3

        guard = _build_guard(config, runtime.locator)
        guard.install_atexit()
        mouse = Mouse(guard, guard.backend, config)
        recorder = RunRecorder.new_run(config.runs_dir, config=config.to_dict())

        state = ObserverState(config)
        observer_server = None
        publisher = None
        if args.observer:
            observer_server, publisher = _start_observer(
                config,
                run_id=recorder.run_id,
                goal="Measure how the picture responds to one known mouse movement.",
                intention="Inject a known delta, capture A/B/C, and record what changed.",
                safety=lambda: publish_safety_from(guard),
                express_thoughts=False,
                state=state,
            )
        look_state = state if publisher is not None else None

        emit = _look_event_printer(look_state, args.quiet)
        look_recorder = LookRecorder(
            look_dir,
            run_id=recorder.run_id,
            plan=plan,
            target={
                "title": target_title,
                "handle": target_handle,
                "width": window_width,
                "height": window_height,
            },
            settings={
                "block_grid": config.look_block_grid,
                "max_steps": config.look_max_steps,
                "large_window_pixels": config.look_large_window_pixels,
                "calibration_deltas": list(config.look_calibration_deltas),
                "max_mouse_delta": config.max_mouse_delta,
            },
            clock=time.time,
        )

        def verify() -> FocusCheck:
            ready, reason = _confirm_target_focus(runtime.locator, expected_handle=target_handle)
            return FocusCheck(ready=ready, reason=reason)

        def on_frame(frame: Any) -> None:
            if look_state is None:
                return
            try:
                look_state.publish_frame(frame, source="look-test")
            except Exception:  # noqa: BLE001 - display must never break the experiment
                pass

        def on_trial(trial: LookTrialResult) -> None:
            if look_state is None:
                return
            try:
                look_state.publish_look(
                    available=True,
                    status=trial.status,
                    experiment=EXPERIMENT_NAME,
                    trial_count=len(plan),
                    **trial.report_fields(),
                )
            except Exception:  # noqa: BLE001 - display must never break the experiment
                pass

        runner = LookRunner(
            recorder=look_recorder,
            capture=lambda: runtime.capturer.capture_window(status.window),
            move=_look_move(mouse, "look-test"),
            verify=verify,
            stop_requested=lambda: bool(guard.stop_requested),
            release=guard.release_all,
            window_size=(window_width, window_height),
            block_grid=config.look_block_grid,
            clock=time.time,
            sleeper=time.sleep,
            on_event=emit,
            on_frame=on_frame,
            on_trial=on_trial,
        )

        result = None
        interrupted = False
        try:
            guard.release_all("look-test start")

            print()
            _print_table([("run id", recorder.run_id), ("experiment", EXPERIMENT_NAME)])
            print()
            print("  Focus the game window now.")
            print("  It is checked again before every movement, not just once.")
            _focus_countdown(handoff, stream=sys.stdout)

            ready, refusal = _confirm_target_focus(runtime.locator, expected_handle=target_handle)
            if not ready:
                print()
                sys.stdout.flush()
                print(f"  refused: {refusal}", file=sys.stderr)
                print("  nothing was sent", file=sys.stderr)
                print(f"  hint: click into {target_title!r} during the countdown,", file=sys.stderr)
                print("        or allow more time with --focus-delay 20", file=sys.stderr)
                result = look_recorder.finish(status="interrupted", stop_reason=refusal)
                exit_code = 1
            else:
                result = runner.run(plan)
                exit_code = 0 if result.status == TRIAL_COMPLETED else 1
        except KeyboardInterrupt:
            interrupted = True
            result = look_recorder.finish(status="interrupted", stop_reason="Ctrl+C")
            print("\ninterrupted by Ctrl+C; releasing everything")
            exit_code = 130
        finally:
            released = guard.release_all("look-test end")
            guard.shutdown("look-test finished")
            if result is None:
                result = look_recorder.finish(status="interrupted", stop_reason="stopped before the sequence began")
            recorder.record_step(
                {
                    "index": 0,
                    "observation": {"window": status.to_dict()},
                    "action": {"kind": "look-test", "plan": [spec.to_dict() for spec in plan]},
                    "result": result.to_dict(),
                }
            )
            recorder.record_safety_events(guard.events)
            run_record = recorder.finish(
                status="look-test",
                stop_reason=result.stop_reason or "look-test finished",
            )
            if publisher is not None:
                publisher.finish(run_record)
            if observer_server is not None:
                observer_server.stop()

            _print_look_summary(result, released=released, telemetry=run_record.directory)
            if interrupted:
                print()
                print("Nothing else will be sent. AutoCraft is idle.")
        return exit_code
    finally:
        runtime.close()


def _print_look_summary(result: Any, *, released: Sequence[str], telemetry: str) -> None:
    """Print the measured numbers, and only the measured numbers."""
    print()
    print("LOOK-001 summary")
    trials = tuple(result.trials)
    rows: list[tuple[str, Any]] = [
        ("experiment", result.experiment),
        ("status", result.status),
        ("stop reason", result.stop_reason),
        ("trials recorded", f"{len(trials)} of {len(result.plan)}"),
        ("movements sent", result.movements_sent),
    ]
    if result.duration is not None:
        rows.append(("duration", f"{result.duration:.2f}s"))
    rows.append(("result file", str(Path(result.directory) / "look_result.json")))
    rows.append(("released inputs", ", ".join(released) or "none"))
    rows.append(("telemetry", telemetry))
    _print_table(rows)

    for trial in trials:
        print()
        print(f"  trial {trial.spec.index + 1}: {trial.spec.describe()}")
        _print_table(_look_trial_rows(trial))
    print()
    print("No pass/fail threshold was applied. Read the numbers above, and see")
    print("look_result.json for the per-block difference map.")
    print("Nothing else will be sent. AutoCraft is idle.")


def _look_trial_rows(trial: Any) -> list[tuple[str, Any]]:
    """One trial's measured primitives, formatted for the terminal."""
    rows: list[tuple[str, Any]] = [("status", trial.status)]
    if trial.stop_reason:
        rows.append(("stopped because", trial.stop_reason))
    if trial.frame_a is not None:
        rows.append(("frame size", f"{trial.window_width}x{trial.window_height}"))
    rows.append(("movements sent", trial.movements_sent))
    rows.append(("capture time", f"{trial.capture_seconds * 1000:.0f} ms"))
    if trial.a_to_b is not None:
        rows += [
            ("A to B mean abs diff", f"{trial.a_to_b.mean_absolute_difference:.3f} luma levels"),
            ("A to B rmse", f"{trial.a_to_b.rmse:.3f}"),
            ("A to B changed pixels", f"{trial.a_to_b.changed_fraction * 100:.2f}%"),
            ("difference grid", f"{trial.a_to_b.block_grid} x {trial.a_to_b.block_grid} blocks"),
        ]
    if trial.a_to_c is not None:
        rows.append(("A to C mean abs diff", f"{trial.a_to_c.mean_absolute_difference:.3f} luma levels"))
    if trial.shift is not None:
        if trial.shift.available:
            rows.append(
                ("estimated shift", f"({trial.shift.x:+.2f}, {trial.shift.y:+.2f}) px, quality {trial.shift.quality:.3f}")
            )
        else:
            rows.append(("estimated shift", f"not available ({trial.shift.reason})"))
    if trial.pixels_per_delta_x is not None:
        rows.append(("pixels per delta x", f"{trial.pixels_per_delta_x:.4f}"))
    if trial.pixels_per_delta_y is not None:
        rows.append(("pixels per delta y", f"{trial.pixels_per_delta_y:.4f}"))
    rows.append(
        (
            "reversibility",
            "not defined (A and B were indistinguishable)"
            if trial.reversibility_ratio is None
            else f"{trial.reversibility_ratio:.3f}",
        )
    )
    if trial.reversibility_note:
        rows.append(("note", trial.reversibility_note))
    for name, path in trial.artifacts.items():
        rows.append((f"file {name}", path))
    return rows


def cmd_loop(args: argparse.Namespace) -> int:
    """Run the bounded agent loop. V0's only policy is no-op."""
    config, source = _load(args)
    config.ensure_directories()
    runtime = _build_runtime(config, config_source=source)
    observer_server = None
    publisher = None
    try:
        guard = _build_guard(config, runtime.locator)
        recorder = RunRecorder.new_run(config.runs_dir, config=config.to_dict())
        if args.observer:
            observer_server, publisher = _start_observer(
                config,
                run_id=recorder.run_id,
                goal="Run the bounded V0 loop without touching the game.",
                intention="Observe, decide with the no-op policy, and record every step.",
                safety=lambda: publish_safety_from(guard),
            )
        progress = _fan_out(
            [_progress_printer(args.quiet), None if publisher is None else publisher.on_step]
        )
        loop = AgentLoop(
            config=config,
            observer=runtime.observer,
            guard=guard,
            policy=NoOpDecisionPolicy(),
            recorder=recorder,
            save_frames_every=args.save_frames_every,
            progress=progress,
            on_observation=None if publisher is None else publisher.on_observation,
        )
        print(f"policy: {loop.policy_name} (V0 cannot play the game by design)")
        print(f"bounds: steps={args.steps} seconds={args.seconds}")
        print(f"run id: {recorder.run_id}")
        record = loop.run(max_steps=args.steps, max_seconds=args.seconds)
        if publisher is not None:
            publisher.finish(record)
        print()
        print("Run summary")
        _print_table(
            [
                ("run id", record.run_id),
                ("status", record.status),
                ("stop reason", record.stop_reason),
                ("duration", f"{record.duration:.2f}s"),
                ("steps", record.step_count),
                ("executed", record.executed_steps),
                ("blocked", record.blocked_steps),
                ("safety events", len(record.safety_events)),
                ("telemetry", record.directory),
            ]
        )
        return 0
    finally:
        if observer_server is not None:
            observer_server.stop()
        runtime.close()


def _progress_printer(quiet: bool):
    if quiet:
        return None

    def _print(step) -> None:
        result = step.result or {}
        if result.get("blocked_reason"):
            flag = "blocked"
        elif not result.get("attempted"):
            flag = "no action"
        elif result.get("executed"):
            flag = "ok"
        else:
            flag = "-"
        print(
            f"  step {step.index:5d}  {step.action.get('description', '?'):<32} "
            f"{flag:<10} diff={step.frame_difference if step.frame_difference is not None else '-'}",
            flush=True,
        )

    return _print


def cmd_keys(args: argparse.Namespace) -> int:
    """List every key name the control layer accepts."""
    names = known_key_names()
    if args.json:
        print(json.dumps(names, indent=2))
        return 0
    per_line = 8
    print(f"{len(names)} supported key names:")
    for index in range(0, len(names), per_line):
        print("  " + "  ".join(name.ljust(9) for name in names[index : index + per_line]))
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Show the effective configuration and where it came from."""
    config, source = _load(args)
    if args.json:
        print(json.dumps({"source": source, "config": config.to_dict()}, indent=2, sort_keys=True))
        return 0
    print("Effective AutoCraft configuration")
    print(f"  source : {source}")
    print()
    _print_table(sorted(config.to_dict().items()))
    print()
    print("Every limit above is a safety bound. See README.md for what each one means.")
    return 0


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def _load(args: argparse.Namespace) -> tuple[Config, str]:
    """Load configuration, honouring ``--config``."""
    path = Path(args.config) if args.config else None
    try:
        config = load_config(path)
    except ConfigError as exc:
        raise SystemExit(f"configuration error: {exc}") from exc
    if path is not None:
        source = str(path)
    else:
        default = Path(DEFAULT_CONFIG_FILENAME)
        source = str(default) if default.is_file() else "built-in defaults (no autocraft.toml found)"
    return config, source


def build_parser() -> argparse.ArgumentParser:
    """Construct the AutoCraft argument parser."""
    parser = argparse.ArgumentParser(
        prog="autocraft",
        description=(
            "AutoCraft V0: a pixel-in, human-controls-out foundation for an embodied agent. "
            "Observation commands never send input; only 'input-test' and 'look-test' can, "
            "and only with --yes."
        ),
    )
    parser.add_argument("--version", action="version", version=f"autocraft {__version__}")
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="path to a TOML config file (default: ./autocraft.toml, else built-in defaults)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="report whether the game window can be found and captured")
    p_status.set_defaults(func=cmd_status)

    p_capture = sub.add_parser("capture", help="save one client-area frame (never sends input)")
    p_capture.add_argument("--output", metavar="PATH", help="destination image path")
    p_capture.set_defaults(func=cmd_capture)

    p_observe = sub.add_parser("observe", help="observe for a bounded time (never sends input)")
    p_observe.add_argument("--seconds", type=float, default=10.0, help="how long to observe (default: 10)")
    p_observe.add_argument("--steps", type=int, default=None, help="stop after this many frames")
    p_observe.add_argument("--save-frames", type=int, default=0, help="save at most N frames into the run directory")
    p_observe.add_argument("--duration-report", type=int, default=0, help="print progress every N frames")
    p_observe.add_argument(
        "--observer",
        action="store_true",
        help="also serve the read-only observer page for this run (never enables input)",
    )
    p_observe.set_defaults(func=cmd_observe)

    p_loop = sub.add_parser("loop", help="run the bounded agent loop (V0 policy is no-op)")
    p_loop.add_argument("--steps", type=int, default=5, help="maximum loop iterations (default: 5)")
    p_loop.add_argument("--seconds", type=float, default=10.0, help="maximum wall-clock seconds (default: 10)")
    p_loop.add_argument("--save-frames-every", type=int, default=0, help="persist a frame every N steps (default: 0)")
    p_loop.add_argument("--quiet", action="store_true", help="do not print per-step progress")
    p_loop.add_argument(
        "--observer",
        action="store_true",
        help="also serve the read-only observer page for this run (does not add any control path)",
    )
    p_loop.set_defaults(func=cmd_loop)

    p_observer = sub.add_parser(
        "observer",
        help="serve the local read-only observer page (never starts the agent, never sends input)",
    )
    p_observer.add_argument("--demo", action="store_true", help="show clearly-labelled scripted demo data")
    p_observer.add_argument("--host", default=None, help="bind address (default: config observer_host, loopback)")
    p_observer.add_argument("--port", type=int, default=None, help="bind port (default: config observer_port)")
    p_observer.add_argument(
        "--allow-remote",
        action="store_true",
        help="permit a non-loopback bind; off by default because the page is unauthenticated",
    )
    p_observer.set_defaults(func=cmd_observer)

    p_input = sub.add_parser(
        "input-test",
        help="EXPLICIT smoke test: send exactly one tiny input action, then release everything",
    )
    p_input.add_argument(
        "--action",
        choices=("key-tap", "mouse-move"),
        default="key-tap",
        help="which single action to send (default: key-tap)",
    )
    p_input.add_argument("--key", default="w", help="key for --action key-tap (default: w)")
    p_input.add_argument("--hold", type=float, default=0.05, help="hold seconds for --action key-tap (default: 0.05)")
    p_input.add_argument("--dx", type=int, default=10, help="x delta for --action mouse-move (default: 10)")
    p_input.add_argument("--dy", type=int, default=0, help="y delta for --action mouse-move (default: 0)")
    p_input.add_argument(
        "--focus-delay",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "seconds to wait for you to focus the game before the final foreground "
            f"check (default: {FOCUS_HANDOFF_SECONDS:g})"
        ),
    )
    p_input.add_argument("--yes", action="store_true", help="required confirmation; without it nothing is sent")
    p_input.set_defaults(func=cmd_input_test)

    p_look = sub.add_parser(
        "look-test",
        help="LOOK-001: inject one known mouse movement and measure how far the picture moved",
    )
    p_look.add_argument("--dx", type=int, default=10, help="x delta to inject per trial (default: 10)")
    p_look.add_argument("--dy", type=int, default=0, help="y delta to inject per trial (default: 0)")
    p_look.add_argument(
        "--steps",
        type=int,
        default=1,
        help="how many identical trials to run (default: 1; capped by config look_max_steps)",
    )
    p_look.add_argument(
        "--settle",
        type=float,
        default=None,
        metavar="SECONDS",
        help="seconds to let the picture settle between a movement and its capture (default: config look_settle_seconds)",
    )
    p_look.add_argument(
        "--focus-delay",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "seconds to wait for you to focus the game before the final foreground "
            f"check (default: {FOCUS_HANDOFF_SECONDS:g})"
        ),
    )
    p_look.add_argument(
        "--output",
        metavar="PATH",
        help="directory for the frames and look_result.json (default: <runs dir>/look)",
    )
    look_calibration = p_look.add_mutually_exclusive_group()
    look_calibration.add_argument(
        "--calibrate-horizontal",
        action="store_true",
        help="run the bounded x calibration series instead of a single delta",
    )
    look_calibration.add_argument(
        "--calibrate-vertical",
        action="store_true",
        help="run the bounded y calibration series instead of a single delta",
    )
    p_look.add_argument("--observer", action="store_true", help="also serve the read-only observer page for this run")
    p_look.add_argument("--quiet", action="store_true", help="do not print per-step progress")
    p_look.add_argument("--yes", action="store_true", help="required confirmation; without it nothing is sent")
    p_look.set_defaults(func=cmd_look_test)

    p_keys = sub.add_parser("keys", help="list supported key names")
    p_keys.add_argument("--json", action="store_true", help="emit JSON")
    p_keys.set_defaults(func=cmd_keys)

    p_config = sub.add_parser("config", help="show the effective configuration")
    p_config.add_argument("--json", action="store_true", help="emit JSON")
    p_config.set_defaults(func=cmd_config)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except WindowError as exc:
        print(f"window error: {exc}", file=sys.stderr)
        return 2
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover - exercised through __main__.py
    sys.exit(main())
