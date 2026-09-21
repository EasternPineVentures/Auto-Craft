"""AutoCraft's command line interface.

Design rule for every command here: **looking is always safe, touching is always
explicit.** ``status``, ``capture`` and ``observe`` read pixels and print facts -
they never inject input, so they can be run freely. ``input-test``, ``look-test``
and ``wake-test`` are the only commands that can move the mouse or press a key,
they say so before they do anything, and they refuse to run without ``--yes``.

``wake-test`` is the first command that decides its own movements rather than
executing a plan handed to it, so it also prints its whole movement budget before
asking for confirmation. Every one of those movements is bounded by the config and
every one of them passes through the same safety guard the simpler commands use.

Nothing here starts autonomous play. ``wake-test`` is one bounded behaviour that
ends by itself, and the loop refuses to run without an explicit bound.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import __version__
from .agent.action import Action, ActionExecutor
from .agent.decision import NoOpDecisionPolicy
from .agent.loop import AgentLoop
from .agent.observation import Observer, WindowGeometry
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
from .observer.snapshot import AgentMode, EventKind, WakeReport
from .perception import PerceptionReport, PerceptionSession, StabilityModel
from .telemetry.recorder import RunRecord, RunRecorder
from .thoughts.model import ThoughtEvent, ThoughtTone, ThoughtTrigger
from .vision.capture import CaptureError, MssCaptureBackend, ScreenCapturer
from .vision.window import (
    TargetStatus,
    WindowBackend,
    WindowError,
    WindowLocator,
    Win32WindowBackend,
    coordinate_scaling_note,
    ensure_dpi_awareness,
)
from .wake import (
    EXPERIMENT_NAME as WAKE_EXPERIMENT_NAME,
    STATUS_ABORTED as WAKE_STATUS_ABORTED,
    STATUS_COMPLETED as WAKE_STATUS_COMPLETED,
    WakeDecisionPolicy,
    WakeEvent,
    WakeRecorder,
    WakeResult,
    WakeRunner,
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
        caution = coordinate_scaling_note(None if status.window is None else status.window.handle)
        if caution:
            print()
            print(f"  caution - {caution}.")
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


def cmd_perceive_test(args: argparse.Namespace) -> int:
    """VISION-001: learn the scene, then report every frame that surprises it.

    Observation only. This command has no mouse, no keyboard, and no executor: it
    cannot send input even if it wanted to, because nothing that injects is
    constructed here.

    The model it uses is fitted online from the run's own warm-up frames. Nothing
    is downloaded, nothing is pretrained, and no model API is called - the whole
    layer is a per-cell statistic computed with numpy. That is a deliberate
    reading of the project's rules, and it is stated here so the owner can
    disagree with it.
    """
    config, source = _load(args)
    config.ensure_directories()
    runtime = _build_runtime(config, config_source=source)
    recorder = RunRecorder.new_run(config.runs_dir, config=config.to_dict())
    try:
        status = runtime.locator.status(force=True)
        if not status.found or status.window is None:
            print(
                f"no target window matched {config.target_title_patterns!r}; nothing observed",
                file=sys.stderr,
            )
            return 1
        if status.window.minimized:
            print("the target window is minimized; nothing observed", file=sys.stderr)
            return 1

        region = status.window.region
        warning = _large_window_warning(region.width, region.height, config)
        if warning is not None:
            print(warning)
            print()

        model = StabilityModel(
            grid=config.perception_grid,
            fit_frames=config.perception_fit_frames,
            sigma=config.perception_sigma,
            floor=config.perception_floor,
            adapt_rate=config.perception_adapt_rate,
        )
        counter = itertools.count()

        def next_frame():
            observation = runtime.observer.observe(next(counter))
            return observation.frame

        session = PerceptionSession(model, next_frame)

        print("VISION-001: LEARNED SCENE MODEL")
        print()
        print("  OBSERVATION ONLY: no mouse movement, no key press, no click")
        print("  the model is fitted from this run's own frames; nothing is pretrained")
        print()
        bound = f"{args.seconds:g}s" if args.seconds is not None else ""
        if args.steps is not None:
            bound = f"{bound} or {args.steps} frame(s)" if bound else f"{args.steps} frame(s)"
        _print_table(
            [
                ("target", repr(status.window.title)),
                ("handle", f"0x{status.window.handle:X}"),
                ("client area", f"{region.width}x{region.height}"),
                ("grid", f"{config.perception_grid} x {config.perception_grid} cells"),
                ("warm-up", f"{config.perception_fit_frames} frames"),
                ("sigma", f"{config.perception_sigma:g}"),
                ("floor", f"{config.perception_floor:g} luma levels"),
                ("adapt rate", f"{config.perception_adapt_rate:g}"),
                ("bound", bound),
            ]
        )
        print()
        print("Watching. Nothing will be sent at any point.")
        print()

        report = session.run(seconds=args.seconds, steps=args.steps)
        _report_perceive(
            report,
            model,
            recorder,
            config,
            target_title=status.window.title,
            width=region.width,
            height=region.height,
        )
        return 0
    except KeyboardInterrupt:
        recorder.record_error("KeyboardInterrupt", context="perceive-test")
        recorder.finish(status="interrupted", stop_reason="Ctrl+C")
        print("\ninterrupted by Ctrl+C; no input was sent")
        return 130
    finally:
        runtime.close()


def _report_perceive(
    report: PerceptionReport,
    model: StabilityModel,
    recorder: RunRecorder,
    config: Config,
    *,
    target_title: str,
    width: int,
    height: int,
) -> None:
    """Print and persist what the scene model measured."""
    summary = report.summary()
    print("Perception summary")
    rows: list[tuple[str, Any]] = [
        ("run id", recorder.run_id),
        ("stop reason", report.stop_reason),
        ("duration", f"{report.duration_seconds:.2f}s"),
        ("frames", f"{len(report.rows)} ({report.warmup_frames} warm-up, "
                   f"{report.steady_frames} scored)"),
        ("capture failures", report.capture_failures),
        ("grid", f"{report.grid} x {report.grid} = {report.grid * report.grid} cells"),
    ]
    if not report.is_complete:
        rows.append(("result", "warm-up never completed, so no frame was scored"))
    else:
        rows.extend(
            [
                ("changed cells, mean", f"{summary['changed_cells_mean']:.2f}"),
                ("changed cells, range", f"{summary['changed_cells_min']}"
                                          f"..{summary['changed_cells_max']}"),
                ("changed fraction, mean", f"{summary['changed_fraction_mean']:.4%}"),
                ("total excess, mean", f"{summary['total_excess_mean']:.4f}"),
                ("total excess, max", f"{summary['total_excess_max']:.4f}"),
            ]
        )
    rows.append(("telemetry", recorder.directory))
    _print_table(rows)

    # Persist the measurement before printing anything else. The capture is the
    # expensive, unrepeatable part of this command, and a formatting error in the
    # reporting path below has already cost one live run its result file: the
    # numbers must reach disk even if a later print statement raises. An
    # incomplete run is persisted too - "the warm-up never finished" is itself a
    # measurement, and `summary.complete` records which case this was.
    payload = report.to_dict()
    payload["run_id"] = recorder.run_id
    payload["target"] = {"title": target_title, "client_area": [width, height]}
    payload["config"] = {
        "grid": config.perception_grid,
        "sigma": config.perception_sigma,
        "floor": config.perception_floor,
        "fit_frames": config.perception_fit_frames,
        "adapt_rate": config.perception_adapt_rate,
    }
    result_path = Path(recorder.directory) / "perceive_result.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for row in report.rows:
        recorder.record_step(
            {
                "index": row.index,
                "phase": row.phase,
                "seconds": round(row.seconds, 4),
                "score": None if row.score is None else row.score.to_dict(),
            }
        )

    if not report.is_complete:
        print()
        print("The run ended before the model had a warm-up window, so nothing was")
        print("scored. Raise --seconds or lower perception_fit_frames and try again.")
        print("No input was sent.")
        recorder.finish(status="observed", stop_reason=report.stop_reason)
        return

    print()
    print("Where the scene moved (accumulated overshoot, one character per cell)")
    accumulated = report.accumulated_excess()
    for line in _render_map(accumulated):
        print(f"  {line}")
    print("  '.' is nothing unexpected; heavier characters are more overshoot.")

    print()
    print("Last scored frame")
    last = report.scores[-1]
    _print_table(
        [
            ("changed cells", f"{last.changed_cells} of {last.cell_count}"),
            ("total excess", f"{last.total_excess:.4f}"),
            ("largest overshoot", f"{last.max_excess:.4f} luma levels"),
            ("mean deviation", f"{last.mean_deviation:.4f} luma levels"),
            ("mean allowance", f"{last.mean_allowance:.4f} luma levels"),
        ]
    )
    for line in last.excess_ascii():
        print(f"  {line}")

    print()
    print("What the model learned")
    _print_table(_perceive_model_rows(model))

    share = model.floor_share()
    print()
    print(f"  {share:.1%} of the learned allowance came from the floor rather than")
    print("  from the measured wobble of the scene. On a scene that is genuinely")
    print("  still this is expected and correct: the floor is what stops a perfectly")
    print("  static picture from being reported as noisy. It is printed because it")
    print("  is also the number that would reveal a model that had learned nothing.")

    record = recorder.finish(status="observed", stop_reason=report.stop_reason)

    print()
    print("No threshold was applied and no verdict was reached: read the numbers")
    print("above and decide what they mean.")
    print(f"Per-frame detail: {result_path}")
    print(f"Run status: {record.status}")
    print("No input was sent. AutoCraft is idle.")


def _perceive_model_rows(model: StabilityModel) -> list[tuple[str, Any]]:
    """The learned parameters, as printable rows."""
    summary = model.summary()
    rows: list[tuple[str, Any]] = [
        ("grid", summary["grid"]),
        ("fit frames", summary["fit_frames"]),
        ("frames seen", summary["frames_seen"]),
        ("sigma", f"{summary['sigma']:g}"),
        ("floor", f"{summary['floor']:g} luma levels"),
        ("adapt rate", f"{summary['adapt_rate']:g}"),
    ]
    fit = summary.get("fit")
    if isinstance(fit, dict):
        for key, value in fit.items():
            label = key.replace("_", " ")
            if isinstance(value, bool):
                rows.append((label, value))
            elif isinstance(value, int):
                rows.append((label, value))
            elif isinstance(value, float):
                rows.append((label, f"{value:.6f}"))
            elif isinstance(value, list) and all(
                isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
            ):
                rows.append((label, "[" + ", ".join(f"{item:.2e}" for item in value) + "]"))
            elif isinstance(value, list):
                rows.append((label, ", ".join(str(item) for item in value)))
            else:
                rows.append((label, value))
    if "floor_share" in summary:
        rows.append(("floor share", f"{summary['floor_share']:.6f}"))
    return rows


def _render_map(block: list[list[float]], levels: str = " .:-=+*#%@") -> list[str]:
    """Render a grid of numbers as ASCII, scaled to its own peak."""
    if not block or not block[0]:
        return []
    peak = max((abs(value) for row in block for value in row), default=0.0)
    if peak <= 0.0:
        return ["." * len(block[0]) for _ in block]
    lines = []
    for row in block:
        lines.append(
            "".join(levels[int(min(abs(value) / peak, 1.0) * (len(levels) - 1))] for value in row)
        )
    return lines


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

    limit = int(config.max_mouse_delta)

    if args.calibrate_horizontal or args.calibrate_vertical:
        deltas = tuple(int(value) for value in config.look_calibration_deltas)
        if len(deltas) > config.look_max_steps:
            raise ValueError(
                f"look_calibration_deltas has {len(deltas)} entries, above the "
                f"look_max_steps bound of {config.look_max_steps}"
            )
        # The series goes through the same per-axis bound as a single delta. The
        # actuator refuses an oversized one anyway, but that refusal lands
        # mid-trial, after a frame has already been captured, and it would turn
        # the series into a row of failures with nothing to say why.
        oversized = [delta for delta in deltas if delta > limit]
        if oversized:
            raise ValueError(
                f"look_calibration_deltas contains {max(oversized)}, above "
                f"max_mouse_delta={limit} per axis; AutoCraft refuses an "
                "oversized movement rather than clamping it"
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


def _large_window_warning(width: int, height: int, config: Config) -> str | None:
    """Explain an oversized client area, without doing anything about it.

    A big window is not an error - the operator may want one - but a large frame
    costs proportionally more to capture and analyse, and the extra pixels add no
    information that a smaller window would not have carried. This reports that
    and stops.
    """
    pixels = int(width) * int(height)
    if pixels <= config.look_large_window_pixels:
        return None
    return (
        f"WARNING: the client area is {width}x{height} ({pixels} pixels), above the\n"
        f"  configured look_large_window_pixels={config.look_large_window_pixels}.\n"
        f"  AutoCraft is recommended at about "
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

        warning = _large_window_warning(window_width, window_height, config)
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


# ---------------------------------------------------------------------------
# WAKE-001: wake up, look around, notice something, turn toward it, stop
# ---------------------------------------------------------------------------

#: The panel fields a :class:`WakeReport` will accept, read off the dataclass
#: itself. The policy's report carries more than the panel needs (run timing,
#: the verdict) and less than it needs (the window size, the run id), so the
#: mapping is done by name and filtered here rather than by hand-written list
#: that could drift the moment either side gains a field.
_WAKE_PANEL_FIELDS: frozenset[str] = frozenset(
    field.name for field in dataclasses.fields(WakeReport)
) - {
    # Passed explicitly by the publisher, so they must not also arrive through
    # the filtered report: a duplicate keyword would raise inside the display
    # path, where the failure is deliberately swallowed, and the panel would
    # then silently never update.
    "available",
    "status",
    "experiment",
    "run_id",
    "window_width",
    "window_height",
    "state",
    "stop_reason",
}


def _wake_panel_fields(report: Mapping[str, Any]) -> dict[str, Any]:
    """Project the policy's flat report onto the fields the panel understands."""
    return {key: value for key, value in report.items() if key in _WAKE_PANEL_FIELDS}


def _wake_rerun_hint(args: argparse.Namespace) -> str:
    """The exact ``wake-test`` invocation that repeats this run for real."""
    parts = ["autocraft", "wake-test"]
    if args.observer:
        parts.append("--observer")
    if args.quiet:
        parts.append("--quiet")
    if args.focus_delay is not None:
        parts += ["--focus-delay", f"{args.focus_delay:g}"]
    parts.append("--yes")
    return " ".join(parts)


def _wake_event_printer(state: ObserverState | None, quiet: bool) -> Any:
    """Print WAKE events as they happen, and mirror them to the page when live.

    The panel write is wrapped because the page is an accessory: a broken display
    must not be able to stop a run that is otherwise doing what it was asked.
    """

    def emit(event: WakeEvent) -> None:
        if not quiet:
            print(f"  . {event.message or event.describe()}")
            sys.stdout.flush()
        if state is None:
            return
        try:
            state.publish_event(event.message or event.describe(), kind=EventKind.ACTION)
        except Exception:  # noqa: BLE001 - display must never break the run
            pass

    return emit


def _wake_publisher(state: ObserverState | None, *, run_id: str, window: tuple[int, int]) -> Any:
    """Build the ``on_status`` callback that refreshes the WAKE panel per step."""
    if state is None:
        return None

    def publish(report: Mapping[str, Any]) -> None:
        try:
            state.publish_wake(
                available=True,
                status="running",
                state=str(report.get("state", "")),
                run_id=run_id,
                window_width=window[0],
                window_height=window[1],
                stop_reason=str(report.get("stop_reason", "")),
                **_wake_panel_fields(report),
            )
        except Exception:  # noqa: BLE001 - display must never break the run
            pass

    return publish


def _wake_plan(config: Config) -> list[tuple[str, Any]]:
    """What ``wake-test`` will and will not do, in the order it will do it."""
    return [
        (
            "behaviour",
            f"look around (max {config.wake_max_scan_moves} scans of "
            f"{config.wake_scan_counts} counts), pick 1 of up to "
            f"{config.wake_max_target_candidates} salient regions, centre it",
        ),
        (
            "movement budget",
            f"{config.wake_max_moves} mouse movement(s) maximum - there is no unbounded mode",
        ),
        (
            "centring budget",
            f"{config.wake_max_center_moves} correction(s) per candidate, "
            f"dead zone {config.wake_dead_zone_px:g} px",
        ),
        ("time limit", f"{config.wake_max_seconds:g}s"),
        ("confidence floor", f"{config.wake_min_target_confidence:g} (below this the target is dropped, not guessed)"),
        ("keyboard", "none - this command never presses a key"),
        ("game state read", "none - pixels in, mouse movements out"),
    ]


def _wake_plan_payload(config: Config) -> list[dict[str, str]]:
    """The plan in the shape the run telemetry stores it.

    ``_wake_plan`` yields ``(label, text)`` pairs, which is what the printed
    table wants. The record wants one object per step: a list of single-key
    objects is unreadable, and a list of bare pairs loses the labels.

    This is a separate function rather than a comprehension at the call site
    because the two shapes are easy to confuse, and confusing them fails in a
    way that names neither. ``dict(row)`` on a ``(label, text)`` pair does not
    build ``{label: text}`` - it treats the pair as a sequence of key/value
    pairs and tries to unpack the label itself. That is how a nine-character
    label produced ``dictionary update sequence element #0 has length 9; 2 is
    required`` during the first live wake-test run, after the behaviour had
    already finished and stopped.
    """
    return [{"step": str(label), "detail": str(text)} for label, text in _wake_plan(config)]


def _note_reporting_failure(what: str, exc: BaseException) -> None:
    """Say that reporting failed without letting it stop anything else.

    Reporting is not the run. A finished measurement must still be handed to
    the operator, the input must still be released, and the observer must still
    be stopped - so a failure here is printed and stepped over.
    """
    print(f"warning: {what} failed: {type(exc).__name__}: {exc}", file=sys.stderr)


def _finish_wake_run_recorder(
    recorder: RunRecorder, result: WakeResult, guard: Any, *, owned: bool
) -> RunRecord | None:
    """Sweep any unrecorded safety events, then close out the run telemetry.

    The order is release, sweep, finalize, close - and nothing may append after
    the close. The agent loop performs all of it from its own ``finally`` block
    when it runs, so on the normal path there is nothing left to sweep and the
    ``finish`` below is the idempotent second call that hands back the same
    record.

    ``owned`` says whether this command owns the shutdown. It does when the
    behaviour never ran - a refusal, or an error on the way in - and then the
    safety events in ``guard`` have never been written and must be. When the loop
    did run, it already recorded the tail it produced, and appending here would
    both duplicate what it wrote and, since it also closed the recorder, raise
    ``run ... is already closed``. That warning was the whole of the second live
    run's cleanup defect: the sweep was unconditional, so the successful path
    always tried to write to a closed recorder.

    The sweep is also passed the whole event list rather than a slice, because it
    only runs on the path where the loop never got to record anything.
    """
    run_record: RunRecord | None = None
    if owned and not recorder.closed:
        try:
            recorder.record_safety_events(guard.events)
        except Exception as exc:  # noqa: BLE001 - telemetry must not break the run
            _note_reporting_failure("the final safety sweep", exc)
    try:
        run_record = recorder.finish(
            status="wake-test",
            stop_reason=result.stop_reason or "wake-test finished",
        )
    except Exception as exc:  # noqa: BLE001 - telemetry must not break the run
        _note_reporting_failure("the run telemetry summary", exc)
    return run_record


def _present_wake_outcome(
    result: WakeResult,
    *,
    run_record: RunRecord | None,
    wake_state: Any,
    publisher: Any,
    observer_server: Any,
    policy: WakeDecisionPolicy,
    window: tuple[int, int],
    released: bool,
    interrupted: bool,
) -> None:
    """Publish, stop, and print - each attempted on its own.

    Every piece here is presentation or cleanup, and none of them is allowed to
    skip the others. On the first live wake-test run a single presentational
    exception inside the finalisation block skipped the observer shutdown and
    the whole printed summary, so the operator saw a traceback instead of the
    numbers the run had already measured and saved.
    """
    if wake_state is not None:
        try:
            wake_state.publish_wake(
                available=True,
                status=result.status,
                state=result.state,
                run_id=run_record.run_id if run_record is not None else "",
                window_width=window[0],
                window_height=window[1],
                stop_reason=result.stop_reason,
                **_wake_panel_fields(policy.report()),
            )
        except Exception as exc:  # noqa: BLE001 - display must never break the run
            _note_reporting_failure("the live panel update", exc)
    if publisher is not None:
        try:
            if run_record is not None:
                publisher.finish(run_record)
        except Exception as exc:  # noqa: BLE001 - display must never break the run
            _note_reporting_failure("the observer publisher", exc)
    if observer_server is not None:
        try:
            observer_server.stop()
        except Exception as exc:  # noqa: BLE001 - cleanup must never break the run
            _note_reporting_failure("stopping the observer server", exc)
    try:
        _print_wake_summary(
            result,
            released=released,
            telemetry=run_record.directory if run_record is not None else "",
        )
    except Exception as exc:  # noqa: BLE001 - the summary is reporting, not the run
        _note_reporting_failure("the summary", exc)
    if interrupted:
        print()
        print("Nothing else will be sent. AutoCraft is idle.")


def cmd_wake_test(args: argparse.Namespace) -> int:
    """WAKE-001: wake up, look around, turn toward something that stands out.

    The same ordering rule as ``look-test`` applies, and it matters more here
    because this is the first command that decides its own movements. ``wake-test``
    is launched from a shell, so at the first check the *shell* is the foreground
    window and refusing on that would make the demo impossible to run. So: locate
    and vet the target, print the entire bounded plan, demand ``--yes`` before any
    injection machinery exists, and only then give the operator a window to focus
    the game. Focus is re-checked before every movement.

    What this command does not do: it does not resize the window, it does not know
    what anything on screen *is*, and it does not claim the region it picks is
    meaningful. It picks a region that differs from its neighbours, turns toward
    it, and writes down what happened.
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
        wake_dir = Path(runtime.config.runs_dir) / "wake"

        # The geometry printed above and written into the plan was sampled now,
        # before the focus countdown. The run's own observations are sampled after
        # it, by a different component. Priming the observer with this answer is
        # what makes the first observation able to say whether the two agree, and
        # a live run has already been recorded where they did not.
        runtime.observer.seed_geometry(
            WindowGeometry.of(handle=target_handle, region=region)
        )

        print("WAKE-001: WAKE UP, LOOK AROUND, TURN TOWARD SOMETHING")
        print()
        print("  LIVE INPUT WILL OCCUR: this command moves the mouse pointer on its own")
        print(f"  press {config.emergency_stop_key.upper()} at any time to abort and release everything")
        print()
        _print_table(
            [
                (
                    "mode",
                    f"LIVE - up to {config.wake_max_moves} mouse movement(s) will be sent"
                    if args.yes
                    else f"DRY RUN - up to {config.wake_max_moves} movement(s) planned, nothing will be sent",
                ),
                ("target", repr(target_title)),
                ("handle", f"0x{target_handle:X}"),
                ("client area", f"{window_width}x{window_height}"),
            ]
            + _wake_plan(config)
            + [
                ("output", str(wake_dir)),
                ("observer", "live" if args.observer else "off (pass --observer to watch live)"),
                ("config", source),
            ]
        )

        warning = _large_window_warning(window_width, window_height, config)
        if warning is not None:
            print()
            print(warning)

        if not args.yes:
            print()
            print("refusing to send input without --yes; nothing was sent")
            print("to run the behaviour for real, re-run with --yes:")
            print()
            print(f"    {_wake_rerun_hint(args)}")
            return 3

        guard = _build_guard(config, runtime.locator)
        guard.install_atexit()
        mouse = Mouse(guard, guard.backend, config)
        keyboard = Keyboard(guard, guard.backend, config)
        executor = ActionExecutor(keyboard, mouse)
        recorder = RunRecorder.new_run(config.runs_dir, config=config.to_dict())
        # Record what this command intends *before* anything moves. The agent
        # loop writes run.json and closes the recorder when the behaviour ends,
        # so a step appended afterwards is rejected - and a rejected append must
        # never be what stops the run from being finalized. Writing it here is
        # also the honest order: this row describes the plan, not the outcome.
        # The outcome is the steps the loop appends below it, plus the
        # measurement in wake_result.json. Index -1 keeps this row ahead of the
        # loop's step 0 instead of colliding with it.
        recorder.record_step(
            {
                "index": -1,
                "observation": {"window": status.to_dict()},
                "action": {"kind": "wake-test", "plan": _wake_plan_payload(config)},
            }
        )

        state = ObserverState(config)
        observer_server = None
        publisher = None
        if args.observer:
            observer_server, publisher = _start_observer(
                config,
                run_id=recorder.run_id,
                goal="Wake up, look around, and turn toward something that stands out.",
                intention="Scan a few views, pick a salient region, centre it, and stop.",
                safety=lambda: publish_safety_from(guard),
                express_thoughts=False,
                state=state,
            )
        wake_state = state if publisher is not None else None

        policy = WakeDecisionPolicy(
            salience_grid=config.wake_salience_grid,
            view_grid=config.wake_view_grid,
            scan_counts=config.wake_scan_counts,
            max_scan_moves=config.wake_max_scan_moves,
            max_target_candidates=config.wake_max_target_candidates,
            max_center_moves=config.wake_max_center_moves,
            max_moves=config.wake_max_moves,
            max_mouse_delta=config.max_mouse_delta,
            dead_zone_px=config.wake_dead_zone_px,
            min_target_confidence=config.wake_min_target_confidence,
            thought_hook=None if wake_state is None else _wake_thought_hook(wake_state),
        )

        wake_recorder = WakeRecorder(
            wake_dir,
            run_id=recorder.run_id,
            plan={
                "title": target_title,
                "handle": target_handle,
                "width": window_width,
                "height": window_height,
            },
            settings={
                "salience_grid": config.wake_salience_grid,
                "view_grid": config.wake_view_grid,
                "scan_counts": config.wake_scan_counts,
                "max_scan_moves": config.wake_max_scan_moves,
                "max_target_candidates": config.wake_max_target_candidates,
                "max_center_moves": config.wake_max_center_moves,
                "max_moves": config.wake_max_moves,
                "max_seconds": config.wake_max_seconds,
                "dead_zone_px": config.wake_dead_zone_px,
                "min_target_confidence": config.wake_min_target_confidence,
            },
            clock=time.time,
        )

        def on_observation(observation: Any) -> None:
            if wake_state is None:
                return
            try:
                if observation.has_frame:
                    wake_state.publish_frame(observation.frame, source="wake-test")
            except Exception:  # noqa: BLE001 - display must never break the run
                pass

        runner = WakeRunner(
            config=config,
            observer=runtime.observer,
            guard=guard,
            policy=policy,
            recorder=wake_recorder,
            executor=executor,
            run_recorder=recorder,
            max_seconds=config.wake_max_seconds,
            clock=time.time,
            sleeper=time.sleep,
            on_event=_wake_event_printer(wake_state, args.quiet),
            on_status=_wake_publisher(
                wake_state, run_id=recorder.run_id, window=(window_width, window_height)
            ),
        )

        result: WakeResult | None = None
        interrupted = False
        # Whether the agent loop ran to completion. The loop performs the whole
        # release-sweep-finalize-close sequence itself, so when it did, this
        # command must not try to sweep again into a recorder that has closed.
        behaviour_ran = False
        try:
            guard.release_all("wake-test start")

            print()
            _print_table([("run id", recorder.run_id), ("experiment", WAKE_EXPERIMENT_NAME)])
            print()
            caution = coordinate_scaling_note(target_handle)
            if caution:
                print(f"  caution - {caution}.")
                print("  the run is still recorded, but what it looks at may not be the game.")
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
                result = wake_recorder.finish(
                    status=WAKE_STATUS_ABORTED, stop_reason=refusal, state=policy.state.value
                )
                exit_code = 1
            else:
                result = runner.run()
                behaviour_ran = True
                exit_code = 0 if result.status == WAKE_STATUS_COMPLETED else 1
        except KeyboardInterrupt:
            interrupted = True
            result = wake_recorder.finish(
                status=WAKE_STATUS_ABORTED, stop_reason="Ctrl+C", state=policy.state.value
            )
            print("\ninterrupted by Ctrl+C; releasing everything")
            exit_code = 130
        finally:
            # Order matters, and it is the whole point of this block, because it
            # is the order the telemetry contract requires:
            #
            #   1. the behaviour has ended
            #   2. release held input
            #   3. record the final safety events
            #   4. finalize the required run telemetry
            #   5. close the recorder
            #   6. and only then, nonessential presentation
            #
            # Releasing input is safety, not reporting, so it goes first and
            # unguarded. ``shutdown`` records a lifecycle event, so it has to
            # precede the sweep that is supposed to capture it. The measurement
            # (wake_result.json) is finalized before any reporting is attempted.
            # Reporting then happens through helpers that attempt each piece
            # separately, so one presentational failure can no longer skip the
            # observer shutdown or the printed summary - which is exactly what
            # happened on the first live run.
            released = guard.release_all("wake-test end")
            guard.shutdown("wake-test finished")
            if result is None:
                result = wake_recorder.finish(
                    status=WAKE_STATUS_ABORTED,
                    stop_reason="stopped before the behaviour began",
                    state=policy.state.value,
                )
            run_record = _finish_wake_run_recorder(recorder, result, guard, owned=not behaviour_ran)
            _present_wake_outcome(
                result,
                run_record=run_record,
                wake_state=wake_state,
                publisher=publisher,
                observer_server=observer_server,
                policy=policy,
                window=(window_width, window_height),
                released=released,
                interrupted=interrupted,
            )
        return exit_code
    finally:
        runtime.close()


def _wake_thought_hook(state: ObserverState) -> Any:
    """A display-only thought sink. Nothing reads these back.

    The behaviour layer hands over a one-word mood and a ready-made sentence. The
    sentence is passed through unchanged and the mood only picks the tone, so the
    thought model is not asked to invent anything about what the agent is seeing.
    """
    tones = {member.value: member for member in ThoughtTone}

    def hook(mood: str, text: str) -> None:
        try:
            state.publish_thought(
                ThoughtEvent(
                    text=str(text),
                    tone=tones.get(str(mood), ThoughtTone.NEUTRAL),
                    trigger_type=ThoughtTrigger.DISCOVERY,
                    trigger_reference="wake-001",
                    generated_by="wake-template",
                )
            )
        except Exception:  # noqa: BLE001 - display must never break the run
            pass

    return hook


def _wake_progress_series(progress: Sequence[float]) -> str:
    """Render the distance series as a single arrow chain, oldest first."""
    if not progress:
        return "not measured"
    return " -> ".join(f"{float(value):g}" for value in progress)


def _wake_cadence_rows(cadence: Mapping[str, float]) -> list[tuple[str, str]]:
    """Turn the measured cadence into printable rows, or nothing at all.

    Nothing is printed for a run that took no timed steps, because a table of
    dashes reads like a measurement that came back zero.
    """
    if not cadence:
        return []
    labels = (
        ("capture_mean", "mean capture"),
        ("decide_mean", "mean decision"),
        ("act_mean", "mean movement"),
        ("verify_mean", "mean verify"),
        ("since_previous_move_mean", "mean gap between movements"),
        ("total_mean", "mean step time"),
    )
    rows = [(label, f"{cadence[key]:.1f} ms") for key, label in labels if key in cadence]
    if not rows:
        return []
    steps = int(cadence.get("steps_timed", 0))
    moves = int(cadence.get("moves_timed", 0))
    rows.append(("measured over", f"{steps} step(s), {moves} movement(s)"))
    return rows


def _print_wake_summary(result: WakeResult, *, released: bool, telemetry: Any) -> None:
    """Print what the run measured, and no verdict about it."""
    print()
    print("WAKE-001 summary")
    _print_table(
        [
            ("experiment", result.experiment),
            ("status", result.status),
            ("state", result.state),
            ("stop reason", result.summary_line),
            ("run id", result.run_id),
            ("duration", f"{result.duration:.2f}s"),
            ("result file", str(Path(result.directory) / "wake_result.json")),
            (
                "successful completion",
                "not judged (the run never reached a terminal state)"
                if result.successful_completion is None
                else ("yes" if result.successful_completion else "no"),
            ),
            ("movements sent", f"{result.moves_sent} of {result.max_moves} allowed"),
            ("scan moves", result.scan_moves),
            ("centring moves", result.centering_moves),
            ("unique views", f"{result.unique_views} ({result.revisited_views} revisits)"),
            ("candidate regions", result.candidate_count),
            ("target changes", result.target_changes),
            ("overshoots", result.overshoots),
            ("failed strategies", result.failed_strategies),
            (
                "stuck patterns",
                f"{result.stuck_patterns_detected} detected, {result.stuck_patterns_broken} broken",
            ),
        ]
    )

    rows: list[tuple[str, Any]] = []
    if result.window_width and result.window_height:
        rows.append(("frame size", f"{result.window_width}x{result.window_height}"))
    if result.target_centre is not None:
        rows.append(("target centre", f"({result.target_centre[0]:.1f}, {result.target_centre[1]:.1f}) px"))
    if result.target_bbox is not None:
        rows.append(("target bbox", "({}, {}, {}, {})".format(*result.target_bbox)))
    if result.target_salience is not None:
        rows.append(("target salience", f"{result.target_salience:.3f}"))
    if result.final_target_offset is not None:
        rows.append(
            (
                "final target offset",
                f"({result.final_target_offset[0]:+.1f}, {result.final_target_offset[1]:+.1f}) px",
            )
        )
    if result.final_target_distance is not None:
        rows.append(("final distance", f"{result.final_target_distance:.1f} px from centre"))
    if result.confidence is not None:
        rows.append(("match confidence", f"{result.confidence:.3f}"))
    rows.append(("progress", _wake_progress_series(result.progress)))
    if result.strategy:
        rows.append(("last strategy", result.strategy))
    if result.mapping_source == "unmeasured":
        rows.append(
            (
                "mouse mapping",
                "unmeasured - corrections were sized by a fixed band count, "
                "so the agent did not know how far a count would move the view",
            )
        )
    else:
        rows.append(
            (
                "mouse mapping",
                f"{result.mapping_source} "
                f"{result.pixels_per_delta_x if result.pixels_per_delta_x is not None else '-'} px/count x, "
                f"{result.pixels_per_delta_y if result.pixels_per_delta_y is not None else '-'} px/count y",
            )
        )
    if result.dead_repetition_ratio is not None:
        rows.append(("dead repetition", f"{result.dead_repetition_ratio:.3f}"))
    if result.productive_repetition_ratio is not None:
        rows.append(("productive repetition", f"{result.productive_repetition_ratio:.3f}"))
    rows.append(("released inputs", "none" if not released else ", ".join(released)))
    rows.append(("telemetry", telemetry))
    cadence = _wake_cadence_rows(result.cadence)
    if rows:
        print()
        _print_table(rows)
    if cadence:
        print()
        _print_table(cadence)

    lines = result.stream_lines()
    if lines:
        print()
        print("  what it did, in its own words:")
        for line in lines:
            print(f"  - {line}")

    if result.notes:
        print()
        print("  notes recorded with the result:")
        for note in result.notes:
            print(f"  - {note}")

    print()
    print("No pass/fail threshold was applied. Read the numbers above, and see")
    print("wake_result.json for the per-event detail.")
    print("Nothing else will be sent. AutoCraft is idle.")


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
            "Observation commands never send input; only 'input-test', 'look-test' and "
            "'wake-test' can, and only with --yes."
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

    p_perceive = sub.add_parser(
        "perceive-test",
        help="learn the scene from live frames and report what changed (never sends input)",
    )
    p_perceive.add_argument("--seconds", type=float, default=30.0, help="how long to watch (default: 30)")
    p_perceive.add_argument("--steps", type=int, default=None, help="stop after this many frames")
    p_perceive.set_defaults(func=cmd_perceive_test)

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

    p_wake = sub.add_parser(
        "wake-test",
        help="WAKE-001: look around, pick a region that stands out, turn toward it, and stop",
        description=(
            "Wake up, look around, notice a visually salient region, turn toward it, "
            "centre it, and stop. This command chooses its own mouse movements, so it "
            "says what it will do, prints the whole movement budget, and refuses to run "
            "without --yes. It never presses a key and never reads game state: pixels "
            "in, mouse movements out."
        ),
    )
    p_wake.add_argument(
        "--focus-delay",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "seconds to wait for you to focus the game before the final foreground "
            f"check (default: {FOCUS_HANDOFF_SECONDS:g})"
        ),
    )
    p_wake.add_argument("--observer", action="store_true", help="also serve the read-only observer page for this run")
    p_wake.add_argument("--quiet", action="store_true", help="do not print per-step progress")
    p_wake.add_argument("--yes", action="store_true", help="required confirmation; without it nothing is sent")
    p_wake.set_defaults(func=cmd_wake_test)

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
