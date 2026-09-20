"""AutoCraft's command line interface.

Design rule for every command here: **looking is always safe, touching is always
explicit.** ``status``, ``capture`` and ``observe`` read pixels and print facts -
they never inject input, so they can be run freely. ``input-test`` is the only
command that can move the mouse or press a key, it says so before it does
anything, and it refuses to run without ``--yes``.

Nothing here starts autonomous play. The only shipped decision policy is the
no-op policy, and the loop refuses to run without an explicit bound.
"""

from __future__ import annotations

import argparse
import json
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
from .control.keyboard import Keyboard
from .control.keymap import known_key_names
from .control.mouse import Mouse
from .control.safety import SafetyGuard
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
    try:
        while time.time() < deadline:
            loop_started = time.monotonic()
            observation = runtime.observer.observe(captured + failed)
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
        runtime.close()


def cmd_input_test(args: argparse.Namespace) -> int:
    """The one command that can inject input. Explicit, bounded, single action."""
    config, source = _load(args)
    config.ensure_directories()
    runtime = _build_runtime(config, config_source=source)
    try:
        status = runtime.locator.status(force=True)
        if not status.found or status.window is None:
            print(f"no target window matched {config.target_title_patterns!r}; nothing sent", file=sys.stderr)
            return 1
        if not status.is_foreground:
            print(
                "the target window is not the foreground window; refusing to inject input",
                file=sys.stderr,
            )
            return 1
        if status.window.minimized:
            print("the target window is minimized; nothing sent", file=sys.stderr)
            return 1

        action = _smoke_action(args)
        if action is None:
            print(f"unknown smoke action {args.action!r}", file=sys.stderr)
            return 2

        print("INPUT SMOKE TEST")
        print(f"  target    : {status.window.title!r} (0x{status.window.handle:X})")
        print(f"  action    : {action.describe()}")
        print(f"  press {config.emergency_stop_key.upper()} at any time to abort and release everything")
        print("  this is the only AutoCraft command that sends input to the game")

        if not args.yes:
            print()
            print("refusing to send input without --yes; nothing was sent")
            print(f"re-run as: autocraft input-test --action {args.action} --yes")
            return 3

        guard = _build_guard(config, runtime.locator)
        keyboard = Keyboard(guard, guard.backend, config)
        mouse = Mouse(guard, guard.backend, config)
        executor = ActionExecutor(keyboard, mouse)
        recorder = RunRecorder.new_run(config.runs_dir, config=config.to_dict())
        guard.install_atexit()
        guard.release_all("input-test start")

        print()
        print(f"  sending in 1s - switch to the game window now (run id {recorder.run_id})")
        for remaining in (1,):
            time.sleep(remaining)

        decision = guard.authorize(action.describe())
        if not decision.allowed:
            result_dict = {
                "attempted": False,
                "executed": False,
                "blocked_reason": decision.reason,
                "description": action.describe(),
            }
            print(f"  blocked: {decision.reason}")
            exit_code = 1
        else:
            guard.wait_for_rate_limit()
            result = executor.execute(action)
            guard.note_action()
            result_dict = result.to_dict()
            print(f"  attempted : {result.attempted}")
            print(f"  executed  : {result.executed}")
            print(f"  duration  : {result.duration:.3f}s")
            if result.blocked_reason:
                print(f"  blocked   : {result.blocked_reason}")
            if result.error:
                print(f"  error     : {result.error}")
            exit_code = 0 if result.ok else 1

        released_keys = guard.release_all("input-test end")
        released_buttons = guard.release_buttons("input-test end")
        recorder.record_step(
            {
                "index": 0,
                "observation": {"window": status.to_dict()},
                "action": action.to_dict(),
                "result": result_dict,
            }
        )
        recorder.record_safety_events(guard.events)
        record = recorder.finish(status="smoke-test", stop_reason="single bounded action completed")
        guard.shutdown("input-test finished")
        print()
        print("  released keys   :", ", ".join(released_keys) or "none")
        print("  released buttons:", ", ".join(released_buttons) or "none")
        print("  telemetry       :", record.directory)
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


def cmd_loop(args: argparse.Namespace) -> int:
    """Run the bounded agent loop. V0's only policy is no-op."""
    config, source = _load(args)
    config.ensure_directories()
    runtime = _build_runtime(config, config_source=source)
    try:
        guard = _build_guard(config, runtime.locator)
        recorder = RunRecorder.new_run(config.runs_dir, config=config.to_dict())
        loop = AgentLoop(
            config=config,
            observer=runtime.observer,
            guard=guard,
            policy=NoOpDecisionPolicy(),
            recorder=recorder,
            save_frames_every=args.save_frames_every,
            progress=_progress_printer(args.quiet),
        )
        print(f"policy: {loop.policy_name} (V0 cannot play the game by design)")
        print(f"bounds: steps={args.steps} seconds={args.seconds}")
        print(f"run id: {recorder.run_id}")
        record = loop.run(max_steps=args.steps, max_seconds=args.seconds)
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
            "Observation commands never send input; only 'input-test' can, and only with --yes."
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
    p_observe.set_defaults(func=cmd_observe)

    p_loop = sub.add_parser("loop", help="run the bounded agent loop (V0 policy is no-op)")
    p_loop.add_argument("--steps", type=int, default=5, help="maximum loop iterations (default: 5)")
    p_loop.add_argument("--seconds", type=float, default=10.0, help="maximum wall-clock seconds (default: 10)")
    p_loop.add_argument("--save-frames-every", type=int, default=0, help="persist a frame every N steps (default: 0)")
    p_loop.add_argument("--quiet", action="store_true", help="do not print per-step progress")
    p_loop.set_defaults(func=cmd_loop)

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
    p_input.add_argument("--yes", action="store_true", help="required confirmation; without it nothing is sent")
    p_input.set_defaults(func=cmd_input_test)

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
