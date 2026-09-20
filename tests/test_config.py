"""Configuration loading, defaults and validation.

The safety limits in :class:`Config` are the documented contract for how hard
AutoCraft is allowed to push the game, so they get tested like any other
interface.
"""

from __future__ import annotations

import pytest

from autocraft.config import Config, ConfigError, load_config


class TestDefaults:
    """The documented defaults must survive a load with no file and no env."""

    def test_loads_with_no_file_and_no_env(self) -> None:
        config = load_config(env={})
        assert config.target_title_patterns == ("Luanti", "VoxelLibre")

    def test_default_capture_rate_is_positive(self) -> None:
        config = load_config(env={})
        assert config.capture_fps > 0
        assert config.step_interval > 0

    def test_default_emergency_stop_is_f8(self) -> None:
        assert load_config(env={}).emergency_stop_key == "f8"

    def test_safety_limits_are_finite_and_small(self) -> None:
        """No safety limit may be zero, negative or effectively unbounded."""
        config = load_config(env={})
        assert 0 < config.max_key_hold_seconds <= 10.0
        assert 0 < config.max_mouse_delta <= 2000
        assert 0 < config.max_consecutive_blocks <= 100
        assert config.min_action_interval >= 0

    def test_paths_are_derived_from_the_data_dir(self) -> None:
        config = load_config(env={})
        assert config.captures_dir.name == "captures"
        assert config.runs_dir.name == "runs"
        assert config.captures_dir.parent == config.data_dir


class TestEnvironmentOverrides:
    """Environment variables override the file, which overrides defaults."""

    def test_override_capture_fps(self) -> None:
        config = load_config(env={"AUTOCRAFT_CAPTURE_FPS": "25"})
        assert config.capture_fps == 25.0

    def test_override_emergency_stop_key(self) -> None:
        config = load_config(env={"AUTOCRAFT_EMERGENCY_STOP_KEY": "f12"})
        assert config.emergency_stop_key == "f12"

    def test_override_title_patterns_as_comma_list(self) -> None:
        config = load_config(env={"AUTOCRAFT_TARGET_TITLE_PATTERNS": "Luanti, VoxelLibre , Custom"})
        assert config.target_title_patterns == ("Luanti", "VoxelLibre", "Custom")

    def test_override_hold_limit(self) -> None:
        config = load_config(env={"AUTOCRAFT_MAX_KEY_HOLD_SECONDS": "1.5"})
        assert config.max_key_hold_seconds == 1.5

    def test_boolean_override(self) -> None:
        assert load_config(env={"AUTOCRAFT_REQUIRE_FOREGROUND": "false"}).require_foreground is False
        assert load_config(env={"AUTOCRAFT_REQUIRE_FOREGROUND": "1"}).require_foreground is True


class TestValidation:
    """Bad configuration must fail loudly at load time, not mid-run."""

    @pytest.mark.parametrize("value", ["0", "-5"])
    def test_rejects_non_positive_capture_fps(self, value: str) -> None:
        with pytest.raises(ConfigError):
            load_config(env={"AUTOCRAFT_CAPTURE_FPS": value})

    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_rejects_non_positive_hold_limit(self, value: str) -> None:
        with pytest.raises(ConfigError):
            load_config(env={"AUTOCRAFT_MAX_KEY_HOLD_SECONDS": value})

    def test_rejects_unknown_key_name(self) -> None:
        with pytest.raises(ConfigError):
            load_config(env={"AUTOCRAFT_EMERGENCY_STOP_KEY": "not-a-key"})

    def test_rejects_non_numeric_number(self) -> None:
        with pytest.raises(ConfigError):
            load_config(env={"AUTOCRAFT_CAPTURE_FPS": "fast"})

    def test_rejects_empty_title_patterns(self) -> None:
        with pytest.raises(ConfigError):
            load_config(env={"AUTOCRAFT_TARGET_TITLE_PATTERNS": "  ,  "})

    def test_rejects_unknown_config_key(self, tmp_path) -> None:
        path = tmp_path / "autocraft.toml"
        path.write_text("[capture]\nfps = 10\nnonsense = true\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(path, env={})

    def test_rejects_missing_config_file(self, tmp_path) -> None:
        with pytest.raises(ConfigError):
            load_config(tmp_path / "absent.toml", env={})


class TestTomlLoading:
    """A TOML file is the supported way to persist non-default settings."""

    def test_reads_values_from_toml(self, tmp_path) -> None:
        path = tmp_path / "autocraft.toml"
        path.write_text(
            "[target]\ntitle_patterns = [\"VoxelLibre\"]\n\n"
            "[capture]\nfps = 7\n\n"
            "[safety]\nemergency_stop_key = \"f9\"\nmax_mouse_delta = 120\n",
            encoding="utf-8",
        )
        config = load_config(path, env={})
        assert config.target_title_patterns == ("VoxelLibre",)
        assert config.capture_fps == 7.0
        assert config.emergency_stop_key == "f9"
        assert config.max_mouse_delta == 120

    def test_env_beats_toml(self, tmp_path) -> None:
        path = tmp_path / "autocraft.toml"
        path.write_text("[capture]\nfps = 7\n", encoding="utf-8")
        assert load_config(path, env={"AUTOCRAFT_CAPTURE_FPS": "30"}).capture_fps == 30.0

    def test_example_config_is_valid(self) -> None:
        """The shipped example template must actually load."""
        from pathlib import Path

        example = Path(__file__).resolve().parents[1] / "autocraft.example.toml"
        assert example.is_file(), "autocraft.example.toml is missing"
        config = load_config(example, env={})
        assert isinstance(config, Config)
        assert config.target_title_patterns


class TestDerivedProperties:
    """Derived values used by the loop and CLI."""

    def test_step_interval_is_the_inverse_of_fps(self) -> None:
        config = load_config(env={"AUTOCRAFT_CAPTURE_FPS": "20"})
        assert config.step_interval == pytest.approx(0.05)

    def test_ensure_directories_creates_both(self, tmp_path) -> None:
        config = load_config(env={"AUTOCRAFT_DATA_DIR": str(tmp_path / "data")})
        config.ensure_directories()
        assert config.captures_dir.is_dir()
        assert config.runs_dir.is_dir()

    def test_to_dict_is_json_serialisable(self) -> None:
        import json

        payload = json.loads(json.dumps(load_config(env={}).to_dict()))
        assert payload["target_title_patterns"]
        assert "max_mouse_delta" in payload

    def test_config_is_frozen(self) -> None:
        config = load_config(env={})
        with pytest.raises(Exception):
            config.capture_fps = 999.0  # type: ignore[misc]
