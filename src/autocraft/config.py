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
    "DEFAULT_MAX_CONSECUTIVE_BLOCKS",
    "DEFAULT_MAX_KEY_HOLD_SECONDS",
    "DEFAULT_MAX_MOUSE_DELTA",
    "DEFAULT_MIN_ACTION_INTERVAL",
    "DEFAULT_TARGET_TITLE_PATTERNS",
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
