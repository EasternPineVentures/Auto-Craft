"""CLI-level guarantees: the safety gate and the capture-rate warning.

The most important promise AutoCraft makes is that nothing autonomously drives
the game. That promise lives in the CLI, so it is tested here rather than left
to inspection: ``input-test`` must refuse before it constructs any object
capable of injecting input.

These tests never touch real windows or real input. ``_load`` and
``_build_runtime`` are replaced with fakes, and the input classes are replaced
with sentinels that fail loudly if anything reaches for them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from autocraft import cli
from autocraft.config import Config
from autocraft.vision.frame import ScreenRegion
from autocraft.vision.window import TargetStatus, WindowInfo


class ForbiddenInputBackend:
    """A backend that fails the test if anything tries to inject input."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("input was constructed during a command that must never inject")


@dataclass
class FakeLocator:
    """A locator that always reports a matching, focused window."""

    status_calls: list[bool]

    def status(self, *, force: bool = False) -> TargetStatus:
        self.status_calls.append(force)
        window = WindowInfo(
            handle=0x40724,
            title="Luanti 5.17.0",
            region=ScreenRegion(0, 105, 3840, 1950),
            visible=True,
            minimized=False,
            process_id=1234,
        )
        return TargetStatus(
            found=True,
            window=window,
            is_foreground=True,
            foreground_handle=window.handle,
            foreground_title=window.title,
            reason="",
        )

    def is_target_foreground(self) -> bool:
        return True


@dataclass
class FakeRuntime:
    """The subset of ``Runtime`` that the CLI commands actually use."""

    locator: FakeLocator

    def close(self) -> None:
        """``cmd_input_test`` closes the runtime on every exit path."""


def _fake_runtime_factory(locator: FakeLocator):
    def factory(config: Config, *, config_source: str) -> FakeRuntime:
        return FakeRuntime(locator=locator)

    return factory


@pytest.fixture
def no_real_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any attempt to build input machinery an immediate failure."""
    for name in ("Keyboard", "Mouse", "ActionExecutor", "SafetyGuard"):
        monkeypatch.setattr(cli, name, ForbiddenInputBackend)


@pytest.fixture
def fake_cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> FakeLocator:
    """Point the CLI at a fake window and a throwaway data directory."""
    config = Config(data_dir=tmp_path / "data")
    locator = FakeLocator(status_calls=[])
    monkeypatch.setattr(cli, "_load", lambda args: (config, "test"))
    monkeypatch.setattr(cli, "_build_runtime", _fake_runtime_factory(locator))
    return locator


class TestCaptureRateWarning:
    """The warning that explains a shortfall must not fire when there is none."""

    def test_silent_when_the_target_is_met(self) -> None:
        assert cli._capture_rate_warning(10.0, 10.0) is None

    def test_silent_at_the_tolerance_boundary(self) -> None:
        # 75% of target is the documented edge of "close enough".
        assert cli._capture_rate_warning(7.5, 10.0) is None

    def test_fires_just_below_the_boundary(self) -> None:
        assert cli._capture_rate_warning(7.4, 10.0) is not None

    def test_silent_when_there_is_no_target(self) -> None:
        assert cli._capture_rate_warning(0.0, 0.0) is None
        assert cli._capture_rate_warning(5.0, -1.0) is None

    def test_explains_a_measured_shortfall(self) -> None:
        warning = cli._capture_rate_warning(2.38, 10.0)
        assert warning is not None
        assert "2.38" in warning
        assert "10" in warning
        # The remedy is the actionable part; a bare number would not help.
        assert "smaller window" in warning
        assert "capture_fps" in warning


class TestInputTestSafetyGate:
    """``input-test`` is the only command that may inject, and only with --yes."""

    def test_refuses_without_yes(self, fake_cli_env, no_real_input, capsys) -> None:
        args = cli.build_parser().parse_args(["input-test"])
        assert args.yes is False

        code = cli.cmd_input_test(args)

        assert code == 3
        output = capsys.readouterr().out
        assert "refusing to send input without --yes" in output
        assert "nothing was sent" in output

    def test_refusal_happens_before_any_input_object_exists(
        self, fake_cli_env, no_real_input
    ) -> None:
        # ``no_real_input`` turns every input class into a hard failure, so
        # reaching this assertion proves the gate ran first.
        args = cli.build_parser().parse_args(["input-test"])
        assert cli.cmd_input_test(args) == 3

    def test_refuses_when_the_target_is_not_foreground(
        self, monkeypatch: pytest.MonkeyPatch, fake_cli_env, no_real_input, capsys
    ) -> None:
        monkeypatch.setattr(fake_cli_env, "status", _unfocused_status)
        args = cli.build_parser().parse_args(["input-test", "--yes"])

        assert cli.cmd_input_test(args) == 1
        assert "not the foreground window" in capsys.readouterr().err

    def test_refuses_when_no_window_matches(
        self, monkeypatch: pytest.MonkeyPatch, fake_cli_env, no_real_input, capsys
    ) -> None:
        monkeypatch.setattr(fake_cli_env, "status", _missing_status)
        args = cli.build_parser().parse_args(["input-test", "--yes"])

        assert cli.cmd_input_test(args) == 1
        assert "no target window matched" in capsys.readouterr().err

    def test_rejects_an_unknown_smoke_action(self, fake_cli_env, no_real_input) -> None:
        # argparse enforces the vocabulary before the command body runs at all.
        with pytest.raises(SystemExit) as excinfo:
            cli.build_parser().parse_args(["input-test", "--yes", "--action", "nope"])
        assert excinfo.value.code == 2


def _focused_status(*, force: bool = False) -> TargetStatus:
    window = WindowInfo(
        handle=0x40724,
        title="Luanti 5.17.0",
        region=ScreenRegion(0, 105, 3840, 1950),
        visible=True,
        minimized=False,
        process_id=1234,
    )
    return TargetStatus(
        found=True,
        window=window,
        is_foreground=True,
        foreground_handle=window.handle,
        foreground_title=window.title,
        reason="",
    )


def _unfocused_status(*, force: bool = False) -> TargetStatus:
    status = _focused_status()
    return TargetStatus(
        found=status.found,
        window=status.window,
        is_foreground=False,
        foreground_handle=0x999,
        foreground_title="Some Other App",
        reason="another window has focus",
    )


def _missing_status(*, force: bool = False) -> TargetStatus:
    return TargetStatus(
        found=False,
        window=None,
        is_foreground=False,
        foreground_handle=0x999,
        foreground_title="Some Other App",
        reason="no window title matched",
    )


class TestParserSurface:
    """The command surface is part of the contract; pin it down."""

    def test_all_documented_commands_exist(self) -> None:
        parser = cli.build_parser()
        for command in ("status", "capture", "observe", "input-test", "keys", "config"):
            args = parser.parse_args([command])
            assert callable(args.func)

    def test_input_test_defaults_to_not_sending(self) -> None:
        parser = cli.build_parser()
        assert parser.parse_args(["input-test"]).yes is False

    def test_loop_always_gets_a_bound(self) -> None:
        # ``AgentLoop.run`` refuses an unbounded run; the CLI must therefore
        # always hand it one, even when the user passes no flags.
        parser = cli.build_parser()
        args = parser.parse_args(["loop"])
        assert args.steps is not None or args.seconds is not None
        assert args.steps == 5
        assert args.seconds == 10.0
