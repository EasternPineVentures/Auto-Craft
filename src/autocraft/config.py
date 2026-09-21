"""Configuration for AutoCraft.

Every limit that bounds game input lives here as an explicit, documented field.
Nothing that keeps the experiment safe is hidden as a magic number deeper in
the code, and ``python -m autocraft config`` prints the effective values.

Values are resolved in increasing priority:

1. the dataclass defaults in this module,
2. an ``autocraft.toml`` file (see ``autocraft.example.toml``),
3. ``AUTOCRAFT_*`` environment variables.

Only Windows is supported in V0 because window discovery and input injection use
Win32 directly.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "Config",
    "ConfigError",
    "DEFAULT_CAPTURE_FPS",
    "DEFAULT_CONFIG_FILENAME",
    "DEFAULT_EMERGENCY_STOP_KEY",
    "DEFAULT_LOOK_BLOCK_GRID",
    "DEFAULT_LOOK_CALIBRATION_DELTAS",
    "DEFAULT_LOOK_LARGE_WINDOW_PIXELS",
    "DEFAULT_LOOK_MAX_STEPS",
    "DEFAULT_LOOK_SETTLE_SECONDS",
    "DEFAULT_VERIFY_SETTLE_SECONDS",
    "DEFAULT_MAX_CONSECUTIVE_BLOCKS",
    "DEFAULT_MAX_KEY_HOLD_SECONDS",
    "DEFAULT_MAX_MOUSE_DELTA",
    "DEFAULT_MIN_ACTION_INTERVAL",
    "DEFAULT_OBSERVER_HOST",
    "DEFAULT_OBSERVER_PORT",
    "DEFAULT_PERCEPTION_ADAPT_RATE",
    "DEFAULT_PERCEPTION_FIT_FRAMES",
    "DEFAULT_PERCEPTION_FLOOR",
    "DEFAULT_PERCEPTION_GRID",
    "DEFAULT_PERCEPTION_SIGMA",
    "DEFAULT_TARGET_TITLE_PATTERNS",
    "DEFAULT_WAKE_DEAD_ZONE_PX",
    "DEFAULT_WAKE_MAX_CENTER_MOVES",
    "DEFAULT_WAKE_MAX_MOVES",
    "DEFAULT_WAKE_MAX_SCAN_MOVES",
    "DEFAULT_WAKE_MAX_SECONDS",
    "DEFAULT_WAKE_MAX_TARGET_CANDIDATES",
    "DEFAULT_WAKE_MIN_TARGET_CONFIDENCE",
    "DEFAULT_WAKE_VERIFY_SETTLE_SECONDS",
    "DEFAULT_WAKE_SALIENCE_GRID",
    "DEFAULT_WAKE_SCAN_COUNTS",
    "DEFAULT_WAKE_VIEW_GRID",
    "ENV_PREFIX",
    "SUPPORTED_IMAGE_FORMATS",
    "load_config",
]

ENV_PREFIX = "AUTOCRAFT_"
DEFAULT_CONFIG_FILENAME = "autocraft.toml"

#: A window is a candidate target when its title contains any of these,
#: compared case-insensitively. Kept deliberately loose so both the game name
#: and the distribution name are matched.
DEFAULT_TARGET_TITLE_PATTERNS: tuple[str, ...] = ("Luanti", "VoxelLibre")

#: Observation rate, and therefore the upper bound on how often the loop acts.
DEFAULT_CAPTURE_FPS: float = 10.0

#: Non-intercepted kill switch. Polled with ``GetAsyncKeyState`` rather than
#: registered as a global hotkey, so AutoCraft never steals the key from the
#: game or from any other application.
DEFAULT_EMERGENCY_STOP_KEY: str = "f8"

#: Longest a single key or mouse button may stay logically held before the loop
#: force-releases it. Bounds the damage from a dropped key-up event.
DEFAULT_MAX_KEY_HOLD_SECONDS: float = 2.0

#: Largest absolute per-axis relative mouse delta accepted in one command.
DEFAULT_MAX_MOUSE_DELTA: int = 200

#: Minimum spacing between injected input events, in seconds.
DEFAULT_MIN_ACTION_INTERVAL: float = 0.02

#: Consecutive safety-blocked actions before the loop gives up and stops. This
#: is what prevents a "window lost focus" condition from becoming a spin loop
#: that keeps hammering the safety guard.
DEFAULT_MAX_CONSECUTIVE_BLOCKS: int = 5

DEFAULT_IMAGE_FORMAT = "png"
SUPPORTED_IMAGE_FORMATS: tuple[str, ...] = ("png", "jpg", "jpeg", "bmp")

#: The observer page is a local instrument, not a service. It binds the loopback
#: interface unless the operator explicitly opts into something wider, and it is
#: read-only in every configuration: it can display agent state but has no code
#: path that reaches the control layer.
DEFAULT_OBSERVER_HOST: str = "127.0.0.1"
DEFAULT_OBSERVER_PORT: int = 8765

#: Hosts the observer will bind without an explicit ``--allow-remote``. Anything
#: else is refused so a diagnostic page never becomes an open network listener
#: by accident.
LOOPBACK_HOSTS: tuple[str, ...] = ("127.0.0.1", "localhost", "::1")

#: LOOK-001 settings. The milestone injects a known mouse movement and measures
#: the picture's response, so every bound on that experiment lives here too.

#: Seconds to wait after a movement before capturing the next frame, giving the
#: game time to finish its own camera interpolation. Too short and the capture
#: lands mid-motion; too long and unrelated animation has time to change the
#: scene instead.
DEFAULT_LOOK_SETTLE_SECONDS: float = 0.15

#: Seconds to wait after an injected movement before capturing the frame the
#: agent will compare with the pre-movement frame. This is the same physical
#: quantity LOOK-001 calls its settle and it defaults to the same value, because
#: the reason is the same in both experiments: a capture issued immediately
#: after a movement samples the desktop before the game has drawn the movement,
#: so it returns the *pre*-movement picture and the movement measures as having
#: changed nothing. Measured live on the reference machine: the first capture
#: after a movement reads 0.00000 up to about 50 ms, the second reads about
#: 0.104, and the change first crosses the loop's 0.01 threshold at 112-128 ms.
#: 150 ms is a three-times margin over that. Zero disables the wait and
#: reintroduces the stale-capture defect, so it is only for tests.
DEFAULT_VERIFY_SETTLE_SECONDS: float = 0.15

#: Coarse partition for the difference map. 8x8 keeps the whole map to 64
#: numbers, small enough to ride along in the observer's JSON payload.
DEFAULT_LOOK_BLOCK_GRID: int = 8

#: Hard ceiling on planned trials in one ``look-test`` invocation. LOOK-001 has
#: no unbounded mode, and this is the number that enforces that.
DEFAULT_LOOK_MAX_STEPS: int = 10

#: Window area above which ``look-test`` warns. 1280x720 is the size the
#: milestone recommends; a larger window is not refused, only flagged, because
#: AutoCraft must never resize the game window to suit itself.
DEFAULT_LOOK_LARGE_WINDOW_PIXELS: int = 1280 * 720

#: The calibration series: injected deltas, in mouse counts, each tried in both
#: directions. Bounded on purpose - calibration is a probe, not a sweep.
#:
#: A series has to bracket the answer, so this one ascends to
#: :data:`DEFAULT_MAX_MOUSE_DELTA`, the largest delta the actuator will accept in
#: one command. It used to stop at 20. The first live trial then showed that a
#: delta of 10 in a 3222x1928 window moved the picture by less than the method
#: can resolve, so a series topping out at 20 would spend every trial below the
#: resolution of the instrument and report near-zero estimates for all of them.
DEFAULT_LOOK_CALIBRATION_DELTAS: tuple[int, ...] = (5, 10, 25, 50, 100, 200)

#: VISION-001 settings. The perception layer learns what each part of the scene
#: normally does, so these bound that learning rather than any game input. The
#: whole ``autocraft.perception`` package is read-only: it can look at the game
#: and has no code path that reaches :mod:`autocraft.control`.

#: Partition the frame is read through. 16x16 gives 256 cells, coarse enough that
#: a cell holds real texture rather than a single noisy pixel, and fine enough
#: that a small on-screen change still lands in its own cell.
DEFAULT_PERCEPTION_GRID: int = 16

#: How many robust standard deviations above a cell's typical frame-to-frame
#: movement count as "this cell changed". The bound itself is learned per cell
#: from real frames; this is only the multiplier applied to the learned spread.
DEFAULT_PERCEPTION_SIGMA: float = 4.0

#: Luma levels. No learned bound is allowed below this. Without a floor, a cell
#: that happened to sit perfectly still while the model was fitted would get a
#: bound of zero and then flag every subsequent frame as changed.
DEFAULT_PERCEPTION_FLOOR: float = 2.0

#: Frames the perception layer learns "normal" from before it starts reporting.
#: A run shorter than this never gets past the learning phase, and says so rather
#: than reporting numbers it cannot support.
DEFAULT_PERCEPTION_FIT_FRAMES: int = 20

#: Per frame, how far an *unchanged* cell's learned baseline drifts toward the
#: current frame, in ``0.0 .. 1.0``. Zero disables adaptation entirely.
#:
#: Adaptation exists because of a measured failure. In the first real run, the
#: scene's ambient lighting shifted at about frame 28 and stayed shifted. A model
#: fitted on frames 0-19 and then frozen reported that shift as a change on every
#: remaining frame forever - precisely the failure this layer exists to prevent.
#: Letting quiet cells drift absorbs a persistent scene change; cells currently
#: flagged as changed are held back, so a transient change still stands out until
#: it either goes away or proves it is the new normal.
DEFAULT_PERCEPTION_ADAPT_RATE: float = 0.05

#: WAKE-001 settings. The behaviour layer decides where to look and how far to
#: turn; these are its budgets and its thresholds. They are the *only* things
#: that bound the run: every one of them is a ceiling, and the milestone requires
#: that a run always terminates, so none of them may be disabled.

#: Cells per axis for the salience map. Eight is deliberately coarse. This is
#: attention, not measurement: it has to say *roughly where* something stands out,
#: and the pixel-accurate part is done afterwards by template matching.
DEFAULT_WAKE_SALIENCE_GRID: int = 8

#: Cells per axis for the view fingerprint, which decides whether two looks are
#: the same view. Same grid as the salience map for the same reason.
DEFAULT_WAKE_VIEW_GRID: int = 8

#: One scan step, in mouse counts. This is how far the camera is turned when
#: deliberately looking around. It is a *count*, not a distance, because LOOK-001
#: never measured how far a count moves the view; the policy re-measures after
#: every scan step rather than assuming this reached anywhere in particular.
DEFAULT_WAKE_SCAN_COUNTS: int = 60

#: SCANNING's movement budget. Twelve deliberate looks is enough to visit the
#: eight compass directions plus the starting view twice over.
DEFAULT_WAKE_MAX_SCAN_MOVES: int = 12

#: How many candidate regions may be attempted before the run gives up and says
#: so. Three, because the point is to stop rather than to keep trying: one failed
#: target must not be allowed to consume the whole run.
DEFAULT_WAKE_MAX_TARGET_CANDIDATES: int = 3

#: CENTERING's movement budget *per candidate*. Eight closed-loop corrections is
#: enough to converge from anywhere on screen at any mapping the LOOK-001 floor
#: admits, with room to spare for an overshoot and its correction.
DEFAULT_WAKE_MAX_CENTER_MOVES: int = 8

#: Total movement budget for the run, across scanning and every candidate.
#: This is the outermost bound and the one the safety story rests on.
DEFAULT_WAKE_MAX_MOVES: int = 45

#: Wall-clock ceiling, in seconds, for a whole wake run. A movement budget alone
#: does not bound time: a window that takes a second to capture would let 45
#: movements take three quarters of a minute. Two minutes is comfortably more
#: than a healthy run needs and short enough that a wedged one gives the operator
#: their screen back.
DEFAULT_WAKE_MAX_SECONDS: float = 120.0

#: How close to the frame centre, in pixels, counts as centred. Twelve pixels is
#: about one mouse count at the finest mapping LOOK-001 could not rule out, so it
#: is a tolerance the motor layer can actually hit rather than a target it would
#: dither around forever.
DEFAULT_WAKE_DEAD_ZONE_PX: float = 12.0

#: Below this match confidence the selected region is treated as lost and the
#: run reacquires or abandons it. The value is a floor against inventing a target
#: location: a matcher that always answers finds something even when the region is
#: gone, and this is what stops that answer being believed.
DEFAULT_WAKE_MIN_TARGET_CONFIDENCE: float = 0.35

#: Seconds to wait after an injected movement before capturing the frame the
#: policy is allowed to judge the movement by. WAKE-001 moves the view far more
#: often than LOOK-001 does - up to 45 movements in a run - and until this
#: existed the loop captured the verification frame with no wait at all, so
#: every movement was judged by the picture from *before* it happened. See
#: ``DEFAULT_VERIFY_SETTLE_SECONDS`` for the measurement behind the number.
DEFAULT_WAKE_VERIFY_SETTLE_SECONDS: float = DEFAULT_VERIFY_SETTLE_SECONDS


class ConfigError(ValueError):
    """Raised when configuration values are missing, malformed or unsafe."""


@dataclass(frozen=True)
class Config:
    """Effective AutoCraft configuration.

    Attributes:
        target_title_patterns: Case-insensitive title substrings that identify
            the game window.
        require_foreground: When true (the default and the safe setting) input
            is refused unless the target window is the foreground window.
        capture_fps: Observation rate for the agent loop.
        image_format: File extension used when saving frames.
        emergency_stop_key: Key name polled for the kill switch.
        max_key_hold_seconds: Upper bound on a single held key/button.
        max_mouse_delta: Upper bound on |dx| and |dy| per mouse command.
        min_action_interval: Minimum spacing between injected input events.
        max_consecutive_blocks: Blocked actions tolerated before the loop stops.
        data_dir: Root directory for generated captures and run records.
        observer_host: Interface the observer page binds.
        observer_port: TCP port the observer page binds.
        observer_frame_max_width: Display frames are downscaled to at most this
            many pixels wide before being handed to the browser. This bounds the
            cost of the dashboard and keeps it off the capture-critical path.
        observer_frame_live_seconds: Frame age at or below which the game view is
            reported as ``live``.
        observer_frame_stale_seconds: Frame age above which the game view is
            reported as ``stale``. Must exceed ``observer_frame_live_seconds``.
        observer_max_events: Upper bound on retained timeline entries. The
            observer must not grow without limit during a long run.
        observer_poll_ms: How often the page re-reads the snapshot. Sent to the
            browser so the page's cadence and the server's expectations cannot
            drift apart.
        thoughts_enabled: Whether AutoCraft expresses thoughts at all.
        thought_min_interval_seconds: Hard cooldown between expressed thoughts.
        thought_max_per_minute: Ceiling on thoughts in any trailing minute.
        thought_history_max: How many recent thoughts the observer retains.
        look_settle_seconds: Seconds to wait after a LOOK-001 movement before
            capturing the next frame.
        look_block_grid: Coarse partition size for the LOOK-001 difference map.
        look_max_steps: Ceiling on planned trials in one ``look-test`` run. There
            is no unbounded mode.
        look_large_window_pixels: Window area above which ``look-test`` warns that
            the target is larger than the recommended 1280x720.
        look_calibration_deltas: Injected mouse deltas, in counts, tried in both
            directions by ``--calibrate-horizontal`` and
            ``--calibrate-vertical``.
        perception_grid: Coarse partition the perception layer reads a frame
            through.
        perception_sigma: Robust standard deviations above a cell's learned
            typical frame-to-frame movement that count as a change.
        perception_floor: Lower bound, in luma levels, on any learned change
            bound.
        perception_fit_frames: Frames the perception layer learns "normal" from
            before it reports anything.
        perception_adapt_rate: Per frame, how far an unchanged cell's learned
            baseline drifts toward the current frame. Zero freezes the model at
            the end of its learning phase.
        wake_salience_grid: Cells per axis for the salience map the attention
            layer reads a frame through.
        wake_view_grid: Cells per axis for the view fingerprint that decides
            whether two looks are the same view.
        wake_scan_counts: One deliberate look-around step, in mouse counts.
        wake_max_scan_moves: SCANNING's movement budget.
        wake_max_target_candidates: How many candidate regions may be attempted
            before the run reports failure. One failed target may not consume the
            whole run.
        wake_max_center_moves: CENTERING's movement budget per candidate.
        wake_max_moves: Total movement budget for the run.
        wake_max_seconds: Wall-clock ceiling for the run. A movement budget alone
            does not bound time.
        wake_dead_zone_px: How close to the frame centre, in pixels, counts as
            centred.
        wake_min_target_confidence: Below this match confidence the selected
            region is treated as lost rather than believed.
        wake_verify_settle_seconds: Seconds to wait after an injected movement
            before capturing the frame the policy judges that movement by. Zero
            disables the wait, which makes every movement look like it changed
            nothing because the capture returns the pre-movement picture.
    """

    target_title_patterns: tuple[str, ...] = DEFAULT_TARGET_TITLE_PATTERNS
    require_foreground: bool = True
    capture_fps: float = DEFAULT_CAPTURE_FPS
    image_format: str = DEFAULT_IMAGE_FORMAT
    emergency_stop_key: str = DEFAULT_EMERGENCY_STOP_KEY
    max_key_hold_seconds: float = DEFAULT_MAX_KEY_HOLD_SECONDS
    max_mouse_delta: int = DEFAULT_MAX_MOUSE_DELTA
    min_action_interval: float = DEFAULT_MIN_ACTION_INTERVAL
    max_consecutive_blocks: int = DEFAULT_MAX_CONSECUTIVE_BLOCKS
    data_dir: Path = Path("data")
    observer_host: str = DEFAULT_OBSERVER_HOST
    observer_port: int = DEFAULT_OBSERVER_PORT
    observer_frame_max_width: int = 960
    observer_frame_live_seconds: float = 1.0
    observer_frame_stale_seconds: float = 5.0
    observer_max_events: int = 200
    observer_poll_ms: int = 1000
    thoughts_enabled: bool = True
    thought_min_interval_seconds: float = 25.0
    thought_max_per_minute: float = 3.0
    thought_history_max: int = 20
    look_settle_seconds: float = DEFAULT_LOOK_SETTLE_SECONDS
    look_block_grid: int = DEFAULT_LOOK_BLOCK_GRID
    look_max_steps: int = DEFAULT_LOOK_MAX_STEPS
    look_large_window_pixels: int = DEFAULT_LOOK_LARGE_WINDOW_PIXELS
    look_calibration_deltas: tuple[int, ...] = DEFAULT_LOOK_CALIBRATION_DELTAS
    perception_grid: int = DEFAULT_PERCEPTION_GRID
    perception_sigma: float = DEFAULT_PERCEPTION_SIGMA
    perception_floor: float = DEFAULT_PERCEPTION_FLOOR
    perception_fit_frames: int = DEFAULT_PERCEPTION_FIT_FRAMES
    perception_adapt_rate: float = DEFAULT_PERCEPTION_ADAPT_RATE
    wake_salience_grid: int = DEFAULT_WAKE_SALIENCE_GRID
    wake_view_grid: int = DEFAULT_WAKE_VIEW_GRID
    wake_scan_counts: int = DEFAULT_WAKE_SCAN_COUNTS
    wake_max_scan_moves: int = DEFAULT_WAKE_MAX_SCAN_MOVES
    wake_max_target_candidates: int = DEFAULT_WAKE_MAX_TARGET_CANDIDATES
    wake_max_center_moves: int = DEFAULT_WAKE_MAX_CENTER_MOVES
    wake_max_moves: int = DEFAULT_WAKE_MAX_MOVES
    wake_max_seconds: float = DEFAULT_WAKE_MAX_SECONDS
    wake_dead_zone_px: float = DEFAULT_WAKE_DEAD_ZONE_PX
    wake_min_target_confidence: float = DEFAULT_WAKE_MIN_TARGET_CONFIDENCE
    wake_verify_settle_seconds: float = DEFAULT_WAKE_VERIFY_SETTLE_SECONDS

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_title_patterns", tuple(str(p) for p in self.target_title_patterns))
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "image_format", str(self.image_format).lower().lstrip("."))
        object.__setattr__(self, "emergency_stop_key", str(self.emergency_stop_key).strip().lower())
        object.__setattr__(self, "capture_fps", float(self.capture_fps))
        object.__setattr__(self, "max_key_hold_seconds", float(self.max_key_hold_seconds))
        object.__setattr__(self, "max_mouse_delta", int(self.max_mouse_delta))
        object.__setattr__(self, "min_action_interval", float(self.min_action_interval))
        object.__setattr__(self, "max_consecutive_blocks", int(self.max_consecutive_blocks))
        object.__setattr__(self, "require_foreground", bool(self.require_foreground))
        object.__setattr__(self, "observer_host", str(self.observer_host).strip())
        object.__setattr__(self, "observer_port", int(self.observer_port))
        object.__setattr__(self, "observer_frame_max_width", int(self.observer_frame_max_width))
        object.__setattr__(self, "observer_frame_live_seconds", float(self.observer_frame_live_seconds))
        object.__setattr__(self, "observer_frame_stale_seconds", float(self.observer_frame_stale_seconds))
        object.__setattr__(self, "observer_max_events", int(self.observer_max_events))
        object.__setattr__(self, "observer_poll_ms", int(self.observer_poll_ms))
        object.__setattr__(self, "thoughts_enabled", bool(self.thoughts_enabled))
        object.__setattr__(self, "thought_min_interval_seconds", float(self.thought_min_interval_seconds))
        object.__setattr__(self, "thought_max_per_minute", float(self.thought_max_per_minute))
        object.__setattr__(self, "thought_history_max", int(self.thought_history_max))
        object.__setattr__(self, "look_settle_seconds", float(self.look_settle_seconds))
        object.__setattr__(self, "look_block_grid", int(self.look_block_grid))
        object.__setattr__(self, "look_max_steps", int(self.look_max_steps))
        object.__setattr__(self, "look_large_window_pixels", int(self.look_large_window_pixels))
        object.__setattr__(
            self,
            "look_calibration_deltas",
            tuple(int(delta) for delta in self.look_calibration_deltas),
        )
        object.__setattr__(self, "perception_grid", int(self.perception_grid))
        object.__setattr__(self, "perception_sigma", float(self.perception_sigma))
        object.__setattr__(self, "perception_floor", float(self.perception_floor))
        object.__setattr__(self, "perception_fit_frames", int(self.perception_fit_frames))
        object.__setattr__(self, "perception_adapt_rate", float(self.perception_adapt_rate))
        object.__setattr__(self, "wake_salience_grid", int(self.wake_salience_grid))
        object.__setattr__(self, "wake_view_grid", int(self.wake_view_grid))
        object.__setattr__(self, "wake_scan_counts", int(self.wake_scan_counts))
        object.__setattr__(self, "wake_max_scan_moves", int(self.wake_max_scan_moves))
        object.__setattr__(self, "wake_max_target_candidates", int(self.wake_max_target_candidates))
        object.__setattr__(self, "wake_max_center_moves", int(self.wake_max_center_moves))
        object.__setattr__(self, "wake_max_moves", int(self.wake_max_moves))
        object.__setattr__(self, "wake_max_seconds", float(self.wake_max_seconds))
        object.__setattr__(self, "wake_dead_zone_px", float(self.wake_dead_zone_px))
        object.__setattr__(self, "wake_min_target_confidence", float(self.wake_min_target_confidence))
        object.__setattr__(self, "wake_verify_settle_seconds", float(self.wake_verify_settle_seconds))
        self._validate()

    # -- derived paths ----------------------------------------------------

    @property
    def captures_dir(self) -> Path:
        """Where one-off ``autocraft capture`` frames are written."""
        return self.data_dir / "captures"

    @property
    def runs_dir(self) -> Path:
        """Where per-run telemetry directories are written."""
        return self.data_dir / "runs"

    @property
    def models_dir(self) -> Path:
        """Where fitted perception models are written.

        Models are artifacts of a run, not of a checkout, so they live under
        ``data/`` alongside captures and runs and are never committed.
        """
        return self.data_dir / "models"

    @property
    def step_interval(self) -> float:
        """Target seconds between agent-loop iterations."""
        return 1.0 / self.capture_fps

    def ensure_directories(self) -> tuple[Path, Path]:
        """Create the data directories if needed and return ``(captures, runs)``."""
        for directory in (self.captures_dir, self.runs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self.captures_dir, self.runs_dir

    # -- reporting --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view, used by the CLI and telemetry."""
        return {
            "target_title_patterns": list(self.target_title_patterns),
            "require_foreground": self.require_foreground,
            "capture_fps": self.capture_fps,
            "image_format": self.image_format,
            "emergency_stop_key": self.emergency_stop_key,
            "max_key_hold_seconds": self.max_key_hold_seconds,
            "max_mouse_delta": self.max_mouse_delta,
            "min_action_interval": self.min_action_interval,
            "max_consecutive_blocks": self.max_consecutive_blocks,
            "data_dir": str(self.data_dir),
            "captures_dir": str(self.captures_dir),
            "runs_dir": str(self.runs_dir),
            "observer_host": self.observer_host,
            "observer_port": self.observer_port,
            "observer_frame_max_width": self.observer_frame_max_width,
            "observer_frame_live_seconds": self.observer_frame_live_seconds,
            "observer_frame_stale_seconds": self.observer_frame_stale_seconds,
            "observer_max_events": self.observer_max_events,
            "observer_poll_ms": self.observer_poll_ms,
            "thoughts_enabled": self.thoughts_enabled,
            "thought_min_interval_seconds": self.thought_min_interval_seconds,
            "thought_max_per_minute": self.thought_max_per_minute,
            "thought_history_max": self.thought_history_max,
            "look_settle_seconds": self.look_settle_seconds,
            "look_block_grid": self.look_block_grid,
            "look_max_steps": self.look_max_steps,
            "look_large_window_pixels": self.look_large_window_pixels,
            "look_calibration_deltas": list(self.look_calibration_deltas),
            "perception_grid": self.perception_grid,
            "perception_sigma": self.perception_sigma,
            "perception_floor": self.perception_floor,
            "perception_fit_frames": self.perception_fit_frames,
            "perception_adapt_rate": self.perception_adapt_rate,
            "wake_salience_grid": self.wake_salience_grid,
            "wake_view_grid": self.wake_view_grid,
            "wake_scan_counts": self.wake_scan_counts,
            "wake_max_scan_moves": self.wake_max_scan_moves,
            "wake_max_target_candidates": self.wake_max_target_candidates,
            "wake_max_center_moves": self.wake_max_center_moves,
            "wake_max_moves": self.wake_max_moves,
            "wake_max_seconds": self.wake_max_seconds,
            "wake_dead_zone_px": self.wake_dead_zone_px,
            "wake_min_target_confidence": self.wake_min_target_confidence,
            "wake_verify_settle_seconds": self.wake_verify_settle_seconds,
        }

    # -- validation -------------------------------------------------------

    def _validate(self) -> None:
        if not self.target_title_patterns:
            raise ConfigError("target_title_patterns must contain at least one pattern")
        if any(not pattern.strip() for pattern in self.target_title_patterns):
            raise ConfigError("target_title_patterns must not contain blank patterns")
        if not self.emergency_stop_key:
            raise ConfigError("emergency_stop_key must not be empty")
        if any(ch.isspace() for ch in self.emergency_stop_key):
            raise ConfigError(f"emergency_stop_key must be a single key name, got {self.emergency_stop_key!r}")
        if self.capture_fps <= 0:
            raise ConfigError(f"capture_fps must be greater than 0, got {self.capture_fps}")
        if self.max_key_hold_seconds <= 0:
            raise ConfigError(
                f"max_key_hold_seconds must be greater than 0, got {self.max_key_hold_seconds}"
            )
        if self.max_mouse_delta < 1:
            raise ConfigError(f"max_mouse_delta must be at least 1, got {self.max_mouse_delta}")
        if self.min_action_interval < 0:
            raise ConfigError(
                f"min_action_interval must not be negative, got {self.min_action_interval}"
            )
        if self.max_consecutive_blocks < 1:
            raise ConfigError(
                f"max_consecutive_blocks must be at least 1, got {self.max_consecutive_blocks}"
            )
        if self.image_format not in SUPPORTED_IMAGE_FORMATS:
            supported = ", ".join(SUPPORTED_IMAGE_FORMATS)
            raise ConfigError(f"image_format must be one of {supported}, got {self.image_format!r}")
        if not self.observer_host:
            raise ConfigError("observer_host must not be empty")
        if any(ch.isspace() for ch in self.observer_host):
            raise ConfigError(f"observer_host must not contain whitespace, got {self.observer_host!r}")
        if not 1 <= self.observer_port <= 65535:
            raise ConfigError(f"observer_port must be between 1 and 65535, got {self.observer_port}")
        if self.observer_frame_max_width < 64:
            raise ConfigError(
                f"observer_frame_max_width must be at least 64, got {self.observer_frame_max_width}"
            )
        if self.observer_frame_live_seconds <= 0:
            raise ConfigError(
                "observer_frame_live_seconds must be greater than 0, "
                f"got {self.observer_frame_live_seconds}"
            )
        if self.observer_frame_stale_seconds <= self.observer_frame_live_seconds:
            raise ConfigError(
                "observer_frame_stale_seconds must be greater than observer_frame_live_seconds, "
                f"got {self.observer_frame_stale_seconds} <= {self.observer_frame_live_seconds}"
            )
        if self.observer_max_events < 10:
            raise ConfigError(
                f"observer_max_events must be at least 10, got {self.observer_max_events}"
            )
        if self.observer_poll_ms < 200:
            raise ConfigError(
                f"observer_poll_ms must be at least 200, got {self.observer_poll_ms}"
            )
        if self.thought_min_interval_seconds < 0:
            raise ConfigError(
                "thought_min_interval_seconds must not be negative, "
                f"got {self.thought_min_interval_seconds}"
            )
        if self.thought_max_per_minute < 0:
            raise ConfigError(
                "thought_max_per_minute must not be negative, "
                f"got {self.thought_max_per_minute}"
            )
        if self.thought_history_max < 1:
            raise ConfigError(
                f"thought_history_max must be at least 1, got {self.thought_history_max}"
            )
        if self.look_settle_seconds <= 0:
            raise ConfigError(
                f"look_settle_seconds must be greater than 0, got {self.look_settle_seconds}"
            )
        if self.look_settle_seconds > 10:
            raise ConfigError(
                "look_settle_seconds must be at most 10; a longer settle lets the scene "
                f"change for reasons other than the injected movement, got {self.look_settle_seconds}"
            )
        if self.look_block_grid < 1:
            raise ConfigError(f"look_block_grid must be at least 1, got {self.look_block_grid}")
        if self.look_max_steps < 1:
            raise ConfigError(f"look_max_steps must be at least 1, got {self.look_max_steps}")
        if self.look_large_window_pixels < 1:
            raise ConfigError(
                f"look_large_window_pixels must be at least 1, got {self.look_large_window_pixels}"
            )
        if not self.look_calibration_deltas:
            raise ConfigError("look_calibration_deltas must contain at least one delta")
        if any(delta < 1 for delta in self.look_calibration_deltas):
            raise ConfigError(
                "look_calibration_deltas must all be at least 1, "
                f"got {list(self.look_calibration_deltas)}"
            )
        if self.perception_grid < 1:
            raise ConfigError(f"perception_grid must be at least 1, got {self.perception_grid}")
        if self.perception_grid > 128:
            raise ConfigError(
                "perception_grid must be at most 128; a finer grid gives cells too small "
                f"to hold any texture, got {self.perception_grid}"
            )
        if self.perception_sigma <= 0:
            raise ConfigError(
                f"perception_sigma must be greater than 0, got {self.perception_sigma}"
            )
        if self.perception_floor < 0:
            raise ConfigError(
                f"perception_floor must not be negative, got {self.perception_floor}"
            )
        if self.perception_fit_frames < 2:
            raise ConfigError(
                "perception_fit_frames must be at least 2; a spread needs at least one "
                f"frame-to-frame difference, got {self.perception_fit_frames}"
            )
        if not 0.0 <= self.perception_adapt_rate <= 1.0:
            raise ConfigError(
                "perception_adapt_rate must be between 0 and 1; it is the fraction "
                "of the gap a quiet cell closes each frame, so a value above 1 would "
                f"overshoot the current frame, got {self.perception_adapt_rate}"
            )
        if self.wake_salience_grid < 1:
            raise ConfigError(
                f"wake_salience_grid must be at least 1, got {self.wake_salience_grid}"
            )
        if self.wake_salience_grid > 64:
            raise ConfigError(
                "wake_salience_grid must be at most 64; the salience map is attention, "
                "not measurement, and a grid finer than the frame's own texture turns "
                f"noise into candidates, got {self.wake_salience_grid}"
            )
        if self.wake_view_grid < 1:
            raise ConfigError(f"wake_view_grid must be at least 1, got {self.wake_view_grid}")
        if self.wake_view_grid > 64:
            raise ConfigError(
                "wake_view_grid must be at most 64; the view fingerprint compares how a "
                "scene looks in coarse blocks, and a finer grid makes two looks of the "
                f"same place look different, got {self.wake_view_grid}"
            )
        if self.wake_scan_counts < 1:
            raise ConfigError(
                f"wake_scan_counts must be at least 1, got {self.wake_scan_counts}"
            )
        if self.wake_scan_counts > self.max_mouse_delta:
            raise ConfigError(
                "wake_scan_counts must not exceed max_mouse_delta; the safety guard "
                "refuses a larger movement, so a larger scan step would be a request "
                f"that is always denied, got {self.wake_scan_counts} against "
                f"{self.max_mouse_delta}"
            )
        if self.wake_max_scan_moves < 1:
            raise ConfigError(
                f"wake_max_scan_moves must be at least 1, got {self.wake_max_scan_moves}"
            )
        if self.wake_max_target_candidates < 1:
            raise ConfigError(
                "wake_max_target_candidates must be at least 1; a run that may not "
                f"attempt a single target cannot succeed, got {self.wake_max_target_candidates}"
            )
        if self.wake_max_center_moves < 1:
            raise ConfigError(
                f"wake_max_center_moves must be at least 1, got {self.wake_max_center_moves}"
            )
        if self.wake_max_moves < self.wake_max_scan_moves + self.wake_max_center_moves:
            raise ConfigError(
                "wake_max_moves must leave room for at least one candidate: it must be "
                f"at least wake_max_scan_moves + wake_max_center_moves "
                f"({self.wake_max_scan_moves} + {self.wake_max_center_moves}), "
                f"got {self.wake_max_moves}"
            )
        if self.wake_max_seconds <= 0:
            raise ConfigError(
                "wake_max_seconds must be greater than 0; a wake run has to be bounded "
                f"in time as well as in movements, got {self.wake_max_seconds}"
            )
        if self.wake_dead_zone_px < 0:
            raise ConfigError(
                f"wake_dead_zone_px must not be negative, got {self.wake_dead_zone_px}"
            )
        if not 0.0 <= self.wake_min_target_confidence <= 1.0:
            raise ConfigError(
                "wake_min_target_confidence must be between 0 and 1; it is compared "
                "against a matcher's own score, so a value outside that range would "
                f"either lose every target or believe every match, got "
                f"{self.wake_min_target_confidence}"
            )
        if self.wake_verify_settle_seconds < 0:
            raise ConfigError(
                "wake_verify_settle_seconds must not be negative, got "
                f"{self.wake_verify_settle_seconds}"
            )
        if self.wake_verify_settle_seconds > 10:
            raise ConfigError(
                "wake_verify_settle_seconds must be at most 10; a longer settle lets "
                "the scene change for reasons other than the injected movement, got "
                f"{self.wake_verify_settle_seconds}"
            )


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _as_bool(value: Any, *, source: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
    raise ConfigError(f"{source}: expected a boolean, got {value!r}")


def _as_float(value: Any, *, source: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{source}: expected a number, got {value!r}") from exc


def _as_int(value: Any, *, source: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{source}: expected an integer, got {value!r}") from exc


def _as_str_list(value: Any, *, source: str) -> tuple[str, ...]:
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = [str(part).strip() for part in value]
    else:
        raise ConfigError(f"{source}: expected a string or list of strings, got {value!r}")
    items = [item for item in items if item]
    if not items:
        raise ConfigError(f"{source}: at least one pattern is required")
    return tuple(items)


def _as_int_list(value: Any, *, source: str) -> tuple[int, ...]:
    """Parse a list of positive integers from a comma-separated string or list."""
    if isinstance(value, str):
        items: list[Any] = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise ConfigError(f"{source}: expected an integer or list of integers, got {value!r}")
    if not items:
        raise ConfigError(f"{source}: at least one value is required")
    parsed: list[int] = []
    for item in items:
        try:
            parsed.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{source}: expected an integer, got {item!r}") from exc
    return tuple(parsed)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_TOML_SECTIONS: dict[str, dict[str, tuple[str, Callable[..., Any]]]] = {
    "target": {
        "title_patterns": ("target_title_patterns", _as_str_list),
        "require_foreground": ("require_foreground", _as_bool),
    },
    "capture": {
        "fps": ("capture_fps", _as_float),
        "format": ("image_format", lambda v, source: str(v)),
    },
    "safety": {
        "emergency_stop_key": ("emergency_stop_key", lambda v, source: str(v)),
        "max_key_hold_seconds": ("max_key_hold_seconds", _as_float),
        "max_mouse_delta": ("max_mouse_delta", _as_int),
        "min_action_interval": ("min_action_interval", _as_float),
        "max_consecutive_blocks": ("max_consecutive_blocks", _as_int),
    },
    "paths": {
        "data_dir": ("data_dir", lambda v, source: Path(str(v))),
    },
    "observer": {
        "host": ("observer_host", lambda v, source: str(v)),
        "port": ("observer_port", _as_int),
        "frame_max_width": ("observer_frame_max_width", _as_int),
        "frame_live_seconds": ("observer_frame_live_seconds", _as_float),
        "frame_stale_seconds": ("observer_frame_stale_seconds", _as_float),
        "max_events": ("observer_max_events", _as_int),
        "poll_ms": ("observer_poll_ms", _as_int),
    },
    "thoughts": {
        "enabled": ("thoughts_enabled", _as_bool),
        "min_interval_seconds": ("thought_min_interval_seconds", _as_float),
        "max_per_minute": ("thought_max_per_minute", _as_float),
        "history_max": ("thought_history_max", _as_int),
    },
    "look": {
        "settle_seconds": ("look_settle_seconds", _as_float),
        "block_grid": ("look_block_grid", _as_int),
        "max_steps": ("look_max_steps", _as_int),
        "large_window_pixels": ("look_large_window_pixels", _as_int),
        "calibration_deltas": ("look_calibration_deltas", _as_int_list),
    },
    "perception": {
        "grid": ("perception_grid", _as_int),
        "sigma": ("perception_sigma", _as_float),
        "floor": ("perception_floor", _as_float),
        "fit_frames": ("perception_fit_frames", _as_int),
        "adapt_rate": ("perception_adapt_rate", _as_float),
    },
    "wake": {
        "salience_grid": ("wake_salience_grid", _as_int),
        "view_grid": ("wake_view_grid", _as_int),
        "scan_counts": ("wake_scan_counts", _as_int),
        "max_scan_moves": ("wake_max_scan_moves", _as_int),
        "max_target_candidates": ("wake_max_target_candidates", _as_int),
        "max_center_moves": ("wake_max_center_moves", _as_int),
        "max_moves": ("wake_max_moves", _as_int),
        "max_seconds": ("wake_max_seconds", _as_float),
        "dead_zone_px": ("wake_dead_zone_px", _as_float),
        "min_target_confidence": ("wake_min_target_confidence", _as_float),
        "verify_settle_seconds": ("wake_verify_settle_seconds", _as_float),
    },
}

_ENV_KEYS: dict[str, tuple[str, Callable[..., Any]]] = {
    "TARGET_TITLE_PATTERNS": ("target_title_patterns", _as_str_list),
    "REQUIRE_FOREGROUND": ("require_foreground", _as_bool),
    "CAPTURE_FPS": ("capture_fps", _as_float),
    "IMAGE_FORMAT": ("image_format", lambda v, source: str(v)),
    "EMERGENCY_STOP_KEY": ("emergency_stop_key", lambda v, source: str(v)),
    "MAX_KEY_HOLD_SECONDS": ("max_key_hold_seconds", _as_float),
    "MAX_MOUSE_DELTA": ("max_mouse_delta", _as_int),
    "MIN_ACTION_INTERVAL": ("min_action_interval", _as_float),
    "MAX_CONSECUTIVE_BLOCKS": ("max_consecutive_blocks", _as_int),
    "DATA_DIR": ("data_dir", lambda v, source: Path(str(v))),
    "OBSERVER_HOST": ("observer_host", lambda v, source: str(v)),
    "OBSERVER_PORT": ("observer_port", _as_int),
    "OBSERVER_FRAME_MAX_WIDTH": ("observer_frame_max_width", _as_int),
    "OBSERVER_FRAME_LIVE_SECONDS": ("observer_frame_live_seconds", _as_float),
    "OBSERVER_FRAME_STALE_SECONDS": ("observer_frame_stale_seconds", _as_float),
    "OBSERVER_MAX_EVENTS": ("observer_max_events", _as_int),
    "OBSERVER_POLL_MS": ("observer_poll_ms", _as_int),
    "THOUGHTS_ENABLED": ("thoughts_enabled", _as_bool),
    "THOUGHT_MIN_INTERVAL_SECONDS": ("thought_min_interval_seconds", _as_float),
    "THOUGHT_MAX_PER_MINUTE": ("thought_max_per_minute", _as_float),
    "THOUGHT_HISTORY_MAX": ("thought_history_max", _as_int),
    "LOOK_SETTLE_SECONDS": ("look_settle_seconds", _as_float),
    "LOOK_BLOCK_GRID": ("look_block_grid", _as_int),
    "LOOK_MAX_STEPS": ("look_max_steps", _as_int),
    "LOOK_LARGE_WINDOW_PIXELS": ("look_large_window_pixels", _as_int),
    "LOOK_CALIBRATION_DELTAS": ("look_calibration_deltas", _as_int_list),
    "PERCEPTION_GRID": ("perception_grid", _as_int),
    "PERCEPTION_SIGMA": ("perception_sigma", _as_float),
    "PERCEPTION_FLOOR": ("perception_floor", _as_float),
    "PERCEPTION_FIT_FRAMES": ("perception_fit_frames", _as_int),
    "PERCEPTION_ADAPT_RATE": ("perception_adapt_rate", _as_float),
    "WAKE_SALIENCE_GRID": ("wake_salience_grid", _as_int),
    "WAKE_VIEW_GRID": ("wake_view_grid", _as_int),
    "WAKE_SCAN_COUNTS": ("wake_scan_counts", _as_int),
    "WAKE_MAX_SCAN_MOVES": ("wake_max_scan_moves", _as_int),
    "WAKE_MAX_TARGET_CANDIDATES": ("wake_max_target_candidates", _as_int),
    "WAKE_MAX_CENTER_MOVES": ("wake_max_center_moves", _as_int),
    "WAKE_MAX_MOVES": ("wake_max_moves", _as_int),
    "WAKE_MAX_SECONDS": ("wake_max_seconds", _as_float),
    "WAKE_DEAD_ZONE_PX": ("wake_dead_zone_px", _as_float),
    "WAKE_MIN_TARGET_CONFIDENCE": ("wake_min_target_confidence", _as_float),
    "WAKE_VERIFY_SETTLE_SECONDS": ("wake_verify_settle_seconds", _as_float),
}


def _read_toml(path: str | Path | None, *, search_dir: str | Path | None) -> dict[str, Any]:
    if path is not None:
        candidate = Path(path)
        if not candidate.is_file():
            raise ConfigError(f"config file not found: {candidate}")
    else:
        candidate = Path(search_dir) if search_dir is not None else Path.cwd()
        candidate = candidate / DEFAULT_CONFIG_FILENAME
        if not candidate.is_file():
            return {}
    try:
        with candidate.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{candidate}: invalid TOML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"{candidate}: could not be read: {exc}") from exc


def _values_from_toml(data: Mapping[str, Any]) -> dict[str, Any]:
    unknown_sections = sorted(set(data) - set(_TOML_SECTIONS))
    if unknown_sections:
        known = ", ".join(sorted(_TOML_SECTIONS))
        raise ConfigError(f"unknown config section(s) {unknown_sections}; known sections: {known}")

    values: dict[str, Any] = {}
    for section, fields in _TOML_SECTIONS.items():
        table = data.get(section, {})
        if not isinstance(table, Mapping):
            raise ConfigError(f"[{section}] must be a table")
        unknown_fields = sorted(set(table) - set(fields))
        if unknown_fields:
            known = ", ".join(sorted(fields))
            raise ConfigError(f"unknown key(s) in [{section}]: {unknown_fields}; known keys: {known}")
        for key, raw in table.items():
            field_name, coerce = fields[key]
            values[field_name] = coerce(raw, source=f"[{section}].{key}")
    return values


def _values_from_env(env: Mapping[str, str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for suffix, (field_name, coerce) in _ENV_KEYS.items():
        raw = env.get(ENV_PREFIX + suffix)
        if raw is None or raw == "":
            continue
        values[field_name] = coerce(raw, source=ENV_PREFIX + suffix)
    return values


def load_config(
    path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    search_dir: str | Path | None = None,
) -> Config:
    """Build the effective :class:`Config`.

    Args:
        path: Explicit TOML file to read. When omitted, ``autocraft.toml`` is
            used if it exists in ``search_dir`` (default: the working directory).
        env: Environment mapping to read overrides from. Defaults to
            ``os.environ``; injectable for tests.
        search_dir: Directory searched for the default config file.

    Raises:
        ConfigError: If a file is unreadable, a value is malformed, or the
            resulting configuration would be unsafe.
    """
    environment = os.environ if env is None else env
    values = _values_from_toml(_read_toml(path, search_dir=search_dir))
    values.update(_values_from_env(environment))
    config = Config(**values)

    # Imported here rather than at module scope: keeping the key table out of
    # the config module means neither can create an import cycle with the other.
    from .control.keymap import is_known_key

    if not is_known_key(config.emergency_stop_key):
        raise ConfigError(
            f"emergency_stop_key {config.emergency_stop_key!r} is not a recognised key name; "
            "run 'python -m autocraft keys' to list supported names"
        )
    return config
