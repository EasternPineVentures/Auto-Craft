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

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from autocraft import cli
from autocraft.agent.action import Action, ActionResult
from autocraft.agent.observation import Observation
from autocraft.config import Config
from autocraft.control.safety import SafetyDecision
from autocraft.vision.frame import Frame, ScreenRegion
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


class ForbiddenCapturer:
    """A capturer that fails the test if anything tries to read the screen."""

    def capture_window(self, window: object) -> object:
        raise AssertionError("the screen was captured during a command that must not capture")


@dataclass
class FakeRuntime:
    """The subset of ``Runtime`` that the CLI commands actually use."""

    locator: FakeLocator
    config: Config
    #: A hard failure rather than ``None``: a command that reaches for the
    #: screen when it should not gets a loud error instead of an
    #: ``AttributeError`` that a later refactor might accidentally swallow.
    capturer: object = field(default_factory=ForbiddenCapturer)

    def close(self) -> None:
        """``cmd_input_test`` closes the runtime on every exit path."""


def _fake_runtime_factory(locator: FakeLocator):
    def factory(config: Config, *, config_source: str) -> FakeRuntime:
        return FakeRuntime(locator=locator, config=config)

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

    def test_an_unfocused_window_is_not_refused_before_the_handoff(
        self, monkeypatch: pytest.MonkeyPatch, fake_cli_env, capsys
    ) -> None:
        # Regression: the foreground check used to run *before* the operator was
        # given any chance to focus the game. Launched from a shell the shell is
        # foreground, so that check refused immediately and made the documented
        # smoke test impossible to run. With no --yes the command must still
        # stop at the --yes gate rather than at a focus refusal.
        monkeypatch.setattr(fake_cli_env, "status", _unfocused_status)

        assert _run_input_test() == 3
        assert "not the foreground window" not in capsys.readouterr().err

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


def _minimized_status(*, force: bool = False) -> TargetStatus:
    status = _focused_status()
    assert status.window is not None
    window = WindowInfo(
        handle=status.window.handle,
        title=status.window.title,
        region=status.window.region,
        visible=True,
        minimized=True,
        process_id=status.window.process_id,
    )
    return TargetStatus(
        found=True,
        window=window,
        is_foreground=False,
        foreground_handle=0x999,
        foreground_title="Some Other App",
        reason="target window is minimized",
    )


def _replaced_window_status(*, force: bool = False) -> TargetStatus:
    """A different matching window, focused — e.g. a second game instance."""
    window = WindowInfo(
        handle=0x51A00,
        title="Luanti 5.17.0",
        region=ScreenRegion(0, 105, 1920, 1080),
        visible=True,
        minimized=False,
        process_id=4321,
    )
    return TargetStatus(
        found=True,
        window=window,
        is_foreground=True,
        foreground_handle=window.handle,
        foreground_title=window.title,
        reason="",
    )


@dataclass
class SequenceLocator:
    """A locator that answers each ``status()`` call from a script.

    ``input-test`` queries the window twice on purpose: once to discover and vet
    the target, and once after the focus countdown. One canned answer cannot
    tell those two queries apart, so the script - plus the recorded ``force``
    flags - is what actually pins the ordering. The last entry repeats once the
    script runs out.
    """

    statuses: list[TargetStatus]
    status_calls: list[bool] = field(default_factory=list)

    def status(self, *, force: bool = False) -> TargetStatus:
        self.status_calls.append(force)
        index = min(len(self.status_calls) - 1, len(self.statuses) - 1)
        return self.statuses[index]

    def is_target_foreground(self) -> bool:
        return bool(self.statuses) and self.statuses[-1].is_foreground


class FakeGuard:
    """A stand-in guard that records every decision the CLI asked it for."""

    def __init__(self, *, allow: bool = True) -> None:
        self.allow = allow
        # ``Keyboard``/``Mouse`` are faked here, so nothing ever calls into
        # this; the CLI only passes it along.
        self.backend = object()
        self.authorize_calls: list[str] = []
        self.release_all_calls: list[str] = []
        self.release_buttons_calls: list[str] = []
        self.shutdown_calls: list[str] = []
        self.atexit_installed = False
        self.events: list[object] = []

    def install_atexit(self) -> None:
        self.atexit_installed = True

    def release_all(self, reason: str = "release_all") -> list[str]:
        self.release_all_calls.append(reason)
        return []

    def release_buttons(self, reason: str = "release_all") -> list[str]:
        self.release_buttons_calls.append(reason)
        return []

    def authorize(self, description: str) -> SafetyDecision:
        self.authorize_calls.append(description)
        if self.allow:
            return SafetyDecision(True)
        return SafetyDecision(
            False, "target window is not the foreground window (foreground lock active)"
        )

    def wait_for_rate_limit(self) -> float:
        return 0.0

    def note_action(self) -> None:
        pass

    def shutdown(self, reason: str = "shutdown") -> list[str]:
        self.shutdown_calls.append(reason)
        return []


class FakeExecutor:
    """Records whether the actuator layer was ever reached."""

    def __init__(self) -> None:
        self.actions: list[Action] = []

    def execute(self, action: Action) -> ActionResult:
        self.actions.append(action)
        return ActionResult(
            action_id=action.action_id,
            kind=action.kind,
            attempted=True,
            executed=True,
            description=action.describe(),
        )


@dataclass
class SmokeHarness:
    """Everything a test needs to observe one ``input-test`` run."""

    locator: SequenceLocator
    guard: FakeGuard
    executor: FakeExecutor


def _install_smoke_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    statuses: list[TargetStatus],
    *,
    guard_allows: bool = True,
) -> SmokeHarness:
    """Wire ``input-test`` to fakes all the way down to the actuator boundary.

    ``FOCUS_HANDOFF_SECONDS`` is zeroed so the ordering is exercised without a
    real sleep - the delay is a courtesy to the operator, not part of the
    safety property.
    """
    config = Config(data_dir=tmp_path / "data")
    locator = SequenceLocator(statuses=list(statuses))
    guard = FakeGuard(allow=guard_allows)
    executor = FakeExecutor()
    monkeypatch.setattr(cli, "_load", lambda args: (config, "test"))
    monkeypatch.setattr(cli, "_build_runtime", _fake_runtime_factory(locator))
    monkeypatch.setattr(cli, "_build_guard", lambda config, locator: guard)
    monkeypatch.setattr(cli, "Keyboard", lambda *a, **k: object())
    monkeypatch.setattr(cli, "Mouse", lambda *a, **k: object())
    monkeypatch.setattr(cli, "ActionExecutor", lambda *a, **k: executor)
    monkeypatch.setattr(cli, "FOCUS_HANDOFF_SECONDS", 0.0)
    return SmokeHarness(locator=locator, guard=guard, executor=executor)


def _run_input_test(*argv: str) -> int:
    args = cli.build_parser().parse_args(["input-test", *argv])
    return cli.cmd_input_test(args)


class TestInputTestFocusOrdering:
    """The foreground handoff must be usable when launched from a shell.

    ``input-test`` is normally typed into PowerShell, so at the first window
    query the *shell* is foreground. Checking foreground there would make the
    documented smoke test impossible to run; the check has to come after the
    operator has been told to focus the game.
    """

    def test_a_shell_being_foreground_at_launch_is_not_a_refusal(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        harness = _install_smoke_harness(
            monkeypatch, tmp_path, [_unfocused_status(), _focused_status()]
        )

        code = _run_input_test("--yes")

        # The whole point of the fix: an unfocused window at the first query
        # must not make the command unusable before the focus handoff.
        assert code == 0
        assert len(harness.executor.actions) == 1
        assert harness.guard.authorize_calls, "the guard must gate the injection"

    def test_the_target_is_queried_twice_and_both_are_forced(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        harness = _install_smoke_harness(
            monkeypatch, tmp_path, [_unfocused_status(), _focused_status()]
        )

        _run_input_test("--yes")

        # Both queries must force re-discovery: the second one is the whole
        # point, and a cached answer would not prove focus actually moved.
        assert harness.locator.status_calls == [True, True]

    def test_refuses_when_focus_never_arrives(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        harness = _install_smoke_harness(
            monkeypatch, tmp_path, [_unfocused_status(), _unfocused_status()]
        )

        assert _run_input_test("--yes") == 1
        assert "not the foreground window" in capsys.readouterr().err
        assert harness.executor.actions == []

    def test_refuses_when_the_target_vanishes_during_the_countdown(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        harness = _install_smoke_harness(
            monkeypatch, tmp_path, [_focused_status(), _missing_status()]
        )

        assert _run_input_test("--yes") == 1
        assert "disappeared during the focus countdown" in capsys.readouterr().err
        assert harness.executor.actions == []

    def test_refuses_when_the_target_becomes_minimized(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        harness = _install_smoke_harness(
            monkeypatch, tmp_path, [_focused_status(), _minimized_status()]
        )

        assert _run_input_test("--yes") == 1
        assert "minimized" in capsys.readouterr().err
        assert harness.executor.actions == []

    def test_refuses_when_the_target_window_is_replaced(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        # A different matching window is focused. "Some Luanti window has
        # focus" is not the promise; the exact vetted window is.
        harness = _install_smoke_harness(
            monkeypatch, tmp_path, [_focused_status(), _replaced_window_status()]
        )

        assert _run_input_test("--yes") == 1
        assert "changed during the focus countdown" in capsys.readouterr().err
        assert harness.executor.actions == []

    def test_the_guard_still_vetoes_after_the_cli_check_passes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        # Focus moves in the instant between the CLI's re-query and the
        # actuator call. The CLI cannot see that; the guard must, because the
        # foreground lock is enforced inside the actuator, not by the CLI.
        harness = _install_smoke_harness(
            monkeypatch, tmp_path, [_focused_status(), _focused_status()], guard_allows=False
        )

        assert _run_input_test("--yes") == 1
        assert "not the foreground window" in capsys.readouterr().err
        assert harness.guard.authorize_calls, "the guard must be consulted before injecting"
        assert harness.executor.actions == []

    @pytest.mark.parametrize(
        "statuses",
        [
            pytest.param([_unfocused_status(), _unfocused_status()], id="never-focused"),
            pytest.param([_focused_status(), _missing_status()], id="vanished"),
            pytest.param([_focused_status(), _minimized_status()], id="minimized"),
            pytest.param([_focused_status(), _replaced_window_status()], id="replaced"),
            pytest.param([_missing_status()], id="missing-from-the-start"),
            pytest.param([_minimized_status()], id="minimized-from-the-start"),
        ],
    )
    def test_no_input_on_any_refusal_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, statuses
    ) -> None:
        harness = _install_smoke_harness(monkeypatch, tmp_path, statuses)

        assert _run_input_test("--yes") == 1
        assert harness.executor.actions == []
        assert harness.guard.authorize_calls == []

    @pytest.mark.parametrize(
        "statuses",
        [
            pytest.param([_unfocused_status(), _unfocused_status()], id="never-focused"),
            pytest.param([_focused_status(), _missing_status()], id="vanished"),
            pytest.param([_focused_status(), _minimized_status()], id="minimized"),
            pytest.param([_focused_status(), _replaced_window_status()], id="replaced"),
        ],
    )
    def test_every_exit_path_releases(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, statuses
    ) -> None:
        harness = _install_smoke_harness(monkeypatch, tmp_path, statuses)

        _run_input_test("--yes")

        assert "input-test start" in harness.guard.release_all_calls
        assert "input-test end" in harness.guard.release_all_calls
        assert harness.guard.shutdown_calls == ["input-test finished"]

    def test_no_separate_button_release_after_release_all(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        # ``release_all`` already covers mouse buttons, so a following
        # ``release_buttons`` could only ever return nothing - and reporting
        # "released buttons: none" when a button might still be down would be a
        # lie, not a cosmetic detail.
        harness = _install_smoke_harness(
            monkeypatch, tmp_path, [_focused_status(), _focused_status()]
        )

        assert _run_input_test("--yes") == 0
        assert harness.guard.release_buttons_calls == []

    def test_the_operator_is_given_a_focus_window_before_the_final_check(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        # The delay is not cosmetic: it is the handoff that makes the documented
        # workflow possible. It must happen before the final query, and the
        # operator must be told about it.
        sleeps: list[float] = []
        harness = _install_smoke_harness(
            monkeypatch, tmp_path, [_unfocused_status(), _focused_status()]
        )
        monkeypatch.setattr(cli, "FOCUS_HANDOFF_SECONDS", 2.5)
        monkeypatch.setattr(cli.time, "sleep", sleeps.append)

        assert _run_input_test("--yes") == 0

        assert sleeps == [2.5]
        assert "sending in 2.5s" in capsys.readouterr().out
        assert harness.locator.status_calls == [True, True]

    def test_ctrl_c_during_the_countdown_still_releases(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        harness = _install_smoke_harness(
            monkeypatch, tmp_path, [_focused_status(), _focused_status()]
        )
        # A real countdown must exist for the interrupt to land inside it; the
        # harness otherwise zeroes the handoff, which skips the wait entirely.
        monkeypatch.setattr(cli, "FOCUS_HANDOFF_SECONDS", 5.0)

        def interrupted(seconds: float) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(cli.time, "sleep", interrupted)

        with pytest.raises(KeyboardInterrupt):
            _run_input_test("--yes")

        assert "input-test end" in harness.guard.release_all_calls
        assert harness.guard.shutdown_calls == ["input-test finished"]
        assert harness.executor.actions == []

    def test_the_gate_runs_before_any_injection_machinery(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        # Even with a perfectly good target and no --yes, nothing that could
        # inject may be constructed.
        _install_smoke_harness(monkeypatch, tmp_path, [_focused_status(), _focused_status()])

        def forbidden(*args: object, **kwargs: object) -> object:
            raise AssertionError("injection machinery was built before the --yes gate")

        for name in ("Keyboard", "Mouse", "ActionExecutor", "_build_guard"):
            monkeypatch.setattr(cli, name, forbidden)

        assert _run_input_test() == 3
        assert "refusing to send input without --yes" in capsys.readouterr().out


class TestFocusDelay:
    """The handoff window is operator-tunable, but never unbounded or malformed."""

    def test_focus_delay_overrides_the_configured_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        sleeps: list[float] = []
        _install_smoke_harness(
            monkeypatch, tmp_path, [_focused_status(), _focused_status()]
        )
        monkeypatch.setattr(cli.time, "sleep", sleeps.append)

        assert _run_input_test("--yes", "--focus-delay", "2.5") == 0

        assert sleeps == [2.5]
        assert "sending in 2.5s" in capsys.readouterr().out

    def test_the_configured_default_is_used_without_the_flag(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        sleeps: list[float] = []
        _install_smoke_harness(
            monkeypatch, tmp_path, [_focused_status(), _focused_status()]
        )
        monkeypatch.setattr(cli, "FOCUS_HANDOFF_SECONDS", 3.0)
        monkeypatch.setattr(cli.time, "sleep", sleeps.append)

        assert _run_input_test("--yes") == 0

        assert sleeps == [3.0]
        assert "sending in 3s" in capsys.readouterr().out

    def test_a_zero_focus_delay_skips_the_wait_entirely(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        sleeps: list[float] = []
        _install_smoke_harness(
            monkeypatch, tmp_path, [_focused_status(), _focused_status()]
        )
        monkeypatch.setattr(cli.time, "sleep", sleeps.append)

        assert _run_input_test("--yes", "--focus-delay", "0") == 0

        assert sleeps == []

    @pytest.mark.parametrize("value", ["-1", "nan", "inf", "-0.5"])
    def test_a_malformed_focus_delay_is_a_usage_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys, value: str
    ) -> None:
        # Rejected before the window layer is consulted at all: the operator
        # asked for something impossible, so the message must not depend on
        # whatever the desktop happens to look like right now.
        harness = _install_smoke_harness(
            monkeypatch, tmp_path, [_focused_status(), _focused_status()]
        )

        assert _run_input_test("--yes", "--focus-delay", value) == 2

        captured = capsys.readouterr()
        assert "invalid --focus-delay" in captured.err
        assert "nothing was sent" not in captured.out
        assert harness.locator.status_calls == []
        assert harness.executor.actions == []
        assert harness.guard.authorize_calls == []

    def test_the_default_is_used_when_the_flag_is_absent(self) -> None:
        parser = cli.build_parser()
        assert parser.parse_args(["input-test"]).focus_delay is None
        assert parser.parse_args(["input-test", "--focus-delay", "20"]).focus_delay == 20.0


class TestRerunHint:
    """The suggested re-run must be copy-pasteable and faithful to the flags."""

    def _hint(self, *argv: str) -> str:
        parser = cli.build_parser()
        return cli._rerun_hint(parser.parse_args(["input-test", *argv]))

    def test_key_hint_echoes_the_key_and_hold(self) -> None:
        hint = self._hint("--action", "key-tap", "--key", "w", "--hold", "0.05")
        assert hint == "autocraft input-test --action key-tap --key w --hold 0.05 --yes"

    def test_mouse_hint_echoes_the_delta(self) -> None:
        hint = self._hint("--action", "mouse-move", "--dx", "10", "--dy", "-5")
        assert hint == "autocraft input-test --action mouse-move --dx 10 --dy -5 --yes"

    def test_the_hint_includes_an_explicit_focus_delay(self) -> None:
        hint = self._hint("--focus-delay", "20")
        assert "--focus-delay 20" in hint
        assert hint.endswith("--yes")

    def test_the_hint_omits_the_focus_delay_when_it_was_not_set(self) -> None:
        assert "--focus-delay" not in self._hint()

    def test_the_dry_run_prints_the_exact_hint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        _install_smoke_harness(
            monkeypatch, tmp_path, [_focused_status(), _focused_status()]
        )

        assert _run_input_test("--action", "key-tap", "--key", "w", "--hold", "0.05") == 3

        captured = capsys.readouterr()
        assert "refusing to send input without --yes" in captured.out
        assert (
            "autocraft input-test --action key-tap --key w --hold 0.05 --yes" in captured.out
        )


@dataclass
class LookHarness:
    """Everything a test needs to observe one ``look-test`` run."""

    config: Config
    locator: SequenceLocator
    guard: FakeGuard

    @property
    def look_dir(self) -> Path:
        """Where ``look-test`` writes its frames and its result file."""
        return Path(self.config.runs_dir) / "look"

    def result_payload(self) -> dict:
        """The parsed ``look_result.json`` from this run."""
        import json

        path = self.look_dir / "look_result.json"
        return json.loads(path.read_text(encoding="utf-8"))


def _install_look_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    statuses: list[TargetStatus],
    *,
    guard_allows: bool = True,
) -> LookHarness:
    """Wire ``look-test`` to fakes all the way down to the actuator boundary.

    No real window, no real sleep, and - the point of the exercise - no real
    mouse: ``Keyboard`` and ``Mouse`` become plain objects, so a test that
    reaches the actuator reaches something with no ``move_relative`` at all
    rather than moving the operator's pointer. The runtime's capturer is a hard
    failure for the same reason. ``FOCUS_HANDOFF_SECONDS`` is zeroed so the
    ordering is exercised without waiting.
    """
    config = Config(data_dir=tmp_path / "data")
    locator = SequenceLocator(statuses=list(statuses))
    guard = FakeGuard(allow=guard_allows)
    monkeypatch.setattr(cli, "_load", lambda args: (config, "test"))
    monkeypatch.setattr(cli, "_build_runtime", _fake_runtime_factory(locator))
    monkeypatch.setattr(cli, "_build_guard", lambda config, locator: guard)
    monkeypatch.setattr(cli, "Keyboard", lambda *a, **k: object())
    monkeypatch.setattr(cli, "Mouse", lambda *a, **k: object())
    monkeypatch.setattr(cli, "FOCUS_HANDOFF_SECONDS", 0.0)
    return LookHarness(config=config, locator=locator, guard=guard)


def _run_look_test(*argv: str) -> int:
    args = cli.build_parser().parse_args(["look-test", *argv])
    return cli.cmd_look_test(args)


class TestLookTestSafetyGate:
    """``look-test`` may only inject with ``--yes``, and never before vetting."""

    def test_defaults_to_not_sending(self) -> None:
        assert cli.build_parser().parse_args(["look-test"]).yes is False

    def test_refuses_without_yes(self, fake_cli_env, no_real_input, capsys) -> None:
        assert _run_look_test() == 3

        output = capsys.readouterr().out
        assert "refusing to send input without --yes" in output
        assert "nothing was sent" in output

    def test_refusal_happens_before_any_input_object_exists(
        self, fake_cli_env, no_real_input
    ) -> None:
        # ``no_real_input`` turns every input class into a hard failure, so
        # reaching this assertion proves the gate ran before the guard, the
        # mouse and the actuator existed.
        assert _run_look_test() == 3

    def test_the_refusal_does_not_create_the_result_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        # A refused run must leave nothing behind that could later be mistaken
        # for the output of a measurement.
        harness = _install_look_harness(monkeypatch, tmp_path, [_focused_status()])

        assert _run_look_test() == 3
        assert not harness.look_dir.exists()

    def test_the_refusal_prints_the_exact_rerun_command(
        self, fake_cli_env, no_real_input, capsys
    ) -> None:
        assert _run_look_test("--dx", "12", "--dy", "-4", "--steps", "3") == 3

        assert "\n    autocraft look-test --dx 12 --dy -4 --steps 3 --yes\n" in capsys.readouterr().out

    def test_the_refusal_warns_about_an_oversized_window(
        self, fake_cli_env, no_real_input, capsys
    ) -> None:
        # The fake target is 3840x1950, far above the recommended ~1280x720.
        assert _run_look_test() == 3

        output = capsys.readouterr().out
        assert "WARNING: the client area is 3840x1950" in output
        # The warning explains itself and stops; it does not act on the window.
        assert "will not resize your window" in output
        assert "Resize it yourself" in output

    def test_the_dry_run_reports_the_plan_without_claiming_a_result(
        self, fake_cli_env, no_real_input, capsys
    ) -> None:
        assert _run_look_test("--dx", "10", "--dy", "0", "--steps", "1") == 3

        output = capsys.readouterr().out
        assert "LOOK-001: MEASURED SENSORIMOTOR MAPPING" in output
        assert "DRY RUN - 2 movement(s) planned, nothing will be sent" in output
        # The standing warning that this command can move the pointer stays,
        # even on a dry run: the operator has to know what --yes would do.
        assert "LIVE INPUT WILL OCCUR" in output


class TestLookTestTargetVetting:
    """A missing or minimized target is refused before the ``--yes`` gate."""

    def test_refuses_when_no_window_matches(
        self, monkeypatch: pytest.MonkeyPatch, fake_cli_env, no_real_input, capsys
    ) -> None:
        monkeypatch.setattr(fake_cli_env, "status", _missing_status)

        assert _run_look_test("--yes") == 1
        assert "no target window matched" in capsys.readouterr().err

    def test_refuses_a_minimized_window(
        self, monkeypatch: pytest.MonkeyPatch, fake_cli_env, no_real_input, capsys
    ) -> None:
        monkeypatch.setattr(fake_cli_env, "status", _minimized_status)

        assert _run_look_test("--yes") == 1
        assert "is minimized" in capsys.readouterr().err


class TestLookTestBoundedPlan:
    """Every plan is finite, and an impossible one is refused with a reason."""

    @pytest.mark.parametrize(
        "argv",
        [
            ("--steps", "0"),
            ("--steps", "11"),
            ("--dx", "0", "--dy", "0"),
            ("--dx", "99999"),
            ("--dy", "-99999"),
            ("--settle", "-1"),
            ("--settle", "60"),
            ("--focus-delay", "-1"),
        ],
    )
    def test_an_impossible_plan_is_refused(
        self, fake_cli_env, no_real_input, argv: tuple[str, ...]
    ) -> None:
        assert _run_look_test(*argv) == 2

    def test_the_refusal_names_the_configured_bound(
        self, fake_cli_env, no_real_input, capsys
    ) -> None:
        assert _run_look_test("--steps", "11") == 2
        assert "look_max_steps=10" in capsys.readouterr().err

    def test_an_oversized_delta_is_refused_rather_than_clamped(
        self, fake_cli_env, no_real_input, capsys
    ) -> None:
        assert _run_look_test("--dx", "99999") == 2
        assert "rather than clamping it" in capsys.readouterr().err

    def test_a_zero_delta_is_refused(self, fake_cli_env, no_real_input, capsys) -> None:
        # Nothing moved means nothing could be measured, so the run is not a
        # measurement of zero - it is not a measurement at all.
        assert _run_look_test("--dx", "0", "--dy", "0") == 2
        assert "nothing could be measured" in capsys.readouterr().err

    def test_a_negative_settle_is_refused(self, fake_cli_env, no_real_input, capsys) -> None:
        assert _run_look_test("--settle", "-1") == 2
        assert "non-negative number of seconds" in capsys.readouterr().err

    def test_an_unbounded_settle_is_refused(self, fake_cli_env, no_real_input, capsys) -> None:
        assert _run_look_test("--settle", "60") == 2
        assert "not a bounded wait" in capsys.readouterr().err


class TestLookTestCalibration:
    """The calibration series comes from config and is bounded by look_max_steps."""

    def test_the_horizontal_series_is_the_configured_one(
        self, fake_cli_env, no_real_input, capsys
    ) -> None:
        assert _run_look_test("--calibrate-horizontal") == 3

        output = capsys.readouterr().out
        assert "horizontal calibration series at x deltas [5, 10, 25, 50, 100, 200]" in output
        assert "6 (bounded; there is no unbounded mode)" in output
        # Six trials, each moving out and back.
        assert "12 movement(s) planned" in output

    def test_the_vertical_series_runs_on_the_other_axis(
        self, fake_cli_env, no_real_input, capsys
    ) -> None:
        assert _run_look_test("--calibrate-vertical") == 3

        output = capsys.readouterr().out
        assert "vertical calibration series at y deltas [5, 10, 25, 50, 100, 200]" in output
        assert "  calibration : yes" in output

    def test_the_rerun_hint_keeps_the_calibration_flag(
        self, fake_cli_env, no_real_input, capsys
    ) -> None:
        assert _run_look_test("--calibrate-vertical") == 3

        assert "\n    autocraft look-test --calibrate-vertical --yes\n" in capsys.readouterr().out

    def test_the_calibration_flags_are_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli.build_parser().parse_args(
                ["look-test", "--calibrate-horizontal", "--calibrate-vertical"]
            )
        assert excinfo.value.code == 2

    def test_a_series_longer_than_the_bound_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        # A config edit must not be able to produce an unbounded series: the
        # length comes from look_calibration_deltas, so that list is what has to
        # be capped by look_max_steps.
        config = Config(
            data_dir=tmp_path / "data",
            look_calibration_deltas=tuple(range(1, 12)),
        )
        monkeypatch.setattr(cli, "_load", lambda args: (config, "test"))

        assert _run_look_test("--calibrate-horizontal") == 2

        stderr = capsys.readouterr().err
        assert "11 entries, above the look_max_steps bound of 10" in stderr

    def test_a_series_above_the_per_axis_bound_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        # The series goes through the same per-axis bound as a single --dx. The
        # actuator refuses an oversized delta anyway, but only mid-trial, after a
        # frame has been captured - so the series is refused here instead, while
        # nothing has been sent and no injection machinery exists.
        config = Config(
            data_dir=tmp_path / "data",
            look_calibration_deltas=(5, 10, 500),
        )
        monkeypatch.setattr(cli, "_load", lambda args: (config, "test"))

        assert _run_look_test("--calibrate-horizontal") == 2

        stderr = capsys.readouterr().err
        assert "look_calibration_deltas contains 500, above max_mouse_delta=200" in stderr
        assert "rather than clamping it" in stderr


class TestLookTestFocusOrdering:
    """The foreground handoff must be usable when launched from a shell.

    ``look-test`` is normally typed into PowerShell, so at the discovery query
    the *shell* is foreground. Refusing there would make the documented trial
    impossible to run, so the foreground check belongs after the countdown -
    and it must still stop the run dead when it fails.
    """

    def test_a_shell_being_foreground_at_launch_is_not_a_refusal(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        _install_look_harness(monkeypatch, tmp_path, [_unfocused_status()])

        assert _run_look_test("--yes") == 1

        captured = capsys.readouterr()
        assert "not the foreground window" in captured.err
        assert "nothing was sent" in captured.err

    def test_losing_focus_during_the_countdown_stops_the_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        # Focused at discovery, unfocused by the time the countdown ends.
        harness = _install_look_harness(
            monkeypatch, tmp_path, [_focused_status(), _unfocused_status()]
        )

        assert _run_look_test("--yes") == 1

        captured = capsys.readouterr()
        assert "refused: the target window is not the foreground window" in captured.err
        # No measurement was taken, and the record says so rather than
        # presenting an empty run as a completed one.
        payload = harness.result_payload()
        assert payload["status"] == "interrupted"
        assert payload["trials"] == []
        assert payload["movements_sent"] == 0
        assert "foreground" in payload["stop_reason"]

    def test_the_run_never_reaches_the_actuator_without_focus(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        harness = _install_look_harness(
            monkeypatch, tmp_path, [_focused_status(), _unfocused_status()]
        )

        assert _run_look_test("--yes") == 1

        # ``Mouse`` is a bare ``object()`` here, so an injected movement would
        # raise rather than move the pointer; the guard records the decisions it
        # was asked for, and there must be none.
        assert harness.guard.authorize_calls == []


# --------------------------------------------------------------------------
# VISION-001: perceive-test
# --------------------------------------------------------------------------


def _perceive_status(*, force: bool = False) -> TargetStatus:
    """A modest, focused target: small enough that no size warning fires."""
    window = WindowInfo(
        handle=0x40724,
        title="Luanti 5.17.0",
        region=ScreenRegion(0, 0, 1280, 720),
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


def _perceive_minimized_status(*, force: bool = False) -> TargetStatus:
    status = _perceive_status()
    assert status.window is not None
    window = WindowInfo(
        handle=status.window.handle,
        title=status.window.title,
        region=status.window.region,
        visible=True,
        minimized=True,
        process_id=status.window.process_id,
    )
    return TargetStatus(
        found=True,
        window=window,
        is_foreground=False,
        foreground_handle=0x999,
        foreground_title="Some Other App",
        reason="target window is minimized",
    )


def _plain_frame(size: int = 64, value: int = 40) -> Frame:
    """A frame with no structure at all: every cell is the same."""
    return Frame(
        image=np.full((size, size, 3), value, dtype=np.uint8),
        timestamp=time.time(),
    )


def _bright_quadrant_frame(size: int = 64, *, quadrant: int = 0, value: int = 220) -> Frame:
    """A frame whose top-left quarter is far brighter than the rest."""
    image = np.full((size, size, 3), 40, dtype=np.uint8)
    half = size // 2
    row = 0 if quadrant < 2 else half
    col = 0 if quadrant % 2 == 0 else half
    image[row : row + half, col : col + half] = value
    return Frame(image=image, timestamp=time.time())


class FakeObserver:
    """An observer that hands out canned frames and counts what was asked of it.

    The real :class:`~autocraft.agent.observation.Observer` captures the screen.
    This one cannot, which is the point: the tests below prove that
    ``perceive-test`` reads the world only through the observer seam, and that
    its model sees exactly the frames it was handed and no others.
    """

    def __init__(self, frames: list[Frame], window: WindowInfo) -> None:
        self.frames = list(frames)
        self.window = window
        self.indices: list[int] = []

    def observe(
        self,
        index: int = 0,
        *,
        capture: bool = True,
        force_discovery: bool = False,
    ) -> Observation:
        self.indices.append(index)
        if not self.frames:
            raise AssertionError("perceive-test asked for more frames than the test supplied")
        # The last frame repeats, so a test that under-supplies frames fails on
        # the assertion above rather than on a run that mysteriously never ends.
        frame = self.frames[0] if len(self.frames) == 1 else self.frames.pop(0)
        return Observation(
            index=index,
            timestamp=time.time(),
            window=self.window,
            frame=frame,
        )


@dataclass
class PerceiveRuntime:
    """The subset of ``Runtime`` that ``cmd_perceive_test`` actually uses."""

    locator: SequenceLocator
    config: Config
    observer: FakeObserver
    #: A hard failure: the command must read frames through the observer, never
    #: by reaching for the capturer itself.
    capturer: object = field(default_factory=ForbiddenCapturer)

    def close(self) -> None:
        """``cmd_perceive_test`` closes the runtime on every exit path."""


@dataclass
class PerceiveHarness:
    """Everything a test needs to observe one ``perceive-test`` run."""

    config: Config
    locator: SequenceLocator
    observer: FakeObserver

    @property
    def result_path(self) -> Path:
        """The one ``perceive_result.json`` this run wrote."""
        found = sorted(Path(self.config.runs_dir).glob("*/perceive_result.json"))
        assert len(found) == 1, f"expected exactly one result file, found {found}"
        return found[0]

    def result_payload(self) -> dict:
        """The parsed ``perceive_result.json`` from this run."""
        import json

        return json.loads(self.result_path.read_text(encoding="utf-8"))


def _install_perceive_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    statuses: list[TargetStatus],
    *,
    frames: list[Frame] | None = None,
    **config_kwargs: object,
) -> PerceiveHarness:
    """Wire ``perceive-test`` to fakes with no input machinery reachable.

    ``_build_guard`` is replaced with a hard failure as well as the input
    classes, so a run that reaches this harness cannot authorise a movement, let
    alone make one. Any of those being constructed fails the test loudly.
    """
    config = Config(data_dir=tmp_path / "data", **config_kwargs)
    locator = SequenceLocator(statuses=list(statuses))
    # The observer is handed a window even when the script's first status has
    # none, so a "no window matched" test fails on the command's refusal rather
    # than on the harness.
    window = statuses[0].window or _perceive_status().window
    assert window is not None
    observer = FakeObserver(frames if frames is not None else [], window)

    def factory(config: Config, *, config_source: str) -> PerceiveRuntime:
        return PerceiveRuntime(locator=locator, config=config, observer=observer)

    monkeypatch.setattr(cli, "_load", lambda args: (config, "test"))
    monkeypatch.setattr(cli, "_build_runtime", factory)
    monkeypatch.setattr(cli, "_build_guard", ForbiddenInputBackend)
    for name in ("Keyboard", "Mouse", "ActionExecutor", "SafetyGuard"):
        monkeypatch.setattr(cli, name, ForbiddenInputBackend)
    return PerceiveHarness(config=config, locator=locator, observer=observer)


def _run_perceive_test(*argv: str) -> int:
    args = cli.build_parser().parse_args(["perceive-test", *argv])
    return cli.cmd_perceive_test(args)


class TestPerceiveTestObservationOnly:
    """``perceive-test`` measures the scene. It must never be able to act on it."""

    def test_it_prints_an_observation_only_banner(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        _install_perceive_harness(
            monkeypatch,
            tmp_path,
            [_perceive_status()],
            frames=[_plain_frame()],
            perception_fit_frames=2,
        )

        assert _run_perceive_test("--steps", "4") == 0

        output = capsys.readouterr().out
        assert "OBSERVATION ONLY" in output
        assert "nothing is pretrained" in output
        assert "No input was sent" in output

    def test_no_input_machinery_is_ever_constructed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        # The harness makes every input class - and the safety guard that would
        # authorise a movement - raise on construction. Reaching exit 0 proves
        # none of them existed.
        _install_perceive_harness(
            monkeypatch,
            tmp_path,
            [_perceive_status()],
            frames=[_plain_frame()],
            perception_fit_frames=2,
        )

        assert _run_perceive_test("--steps", "4") == 0

    def test_it_reads_frames_only_through_the_observer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        # The runtime's capturer raises if anything calls it. The command still
        # has to produce a fitted model, so the frames can only have arrived via
        # ``observer.observe``.
        harness = _install_perceive_harness(
            monkeypatch,
            tmp_path,
            [_perceive_status()],
            frames=[_plain_frame()],
            perception_fit_frames=2,
        )

        assert _run_perceive_test("--steps", "4") == 0
        assert harness.result_payload()["model"]["fitted"] is True
        assert harness.observer.indices == [0, 1, 2, 3]

    def test_the_result_records_no_verdict(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        # The layer measures; the operator decides. A machine-readable "pass"
        # would quietly turn this into an evaluator.
        harness = _install_perceive_harness(
            monkeypatch,
            tmp_path,
            [_perceive_status()],
            frames=[_plain_frame()],
            perception_fit_frames=2,
        )

        assert _run_perceive_test("--steps", "4") == 0

        payload = harness.result_payload()
        for forbidden in ("pass", "passed", "fail", "failed", "verdict", "ok"):
            assert forbidden not in payload
            assert forbidden not in payload["summary"]

    def test_it_says_no_verdict_was_reached(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        _install_perceive_harness(
            monkeypatch,
            tmp_path,
            [_perceive_status()],
            frames=[_plain_frame()],
            perception_fit_frames=2,
        )

        assert _run_perceive_test("--steps", "4") == 0

        output = capsys.readouterr().out
        assert "no verdict was reached" in output


class TestPerceiveTestTargetVetting:
    """Nothing is observed when there is nothing valid to observe."""

    def test_a_missing_window_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        harness = _install_perceive_harness(monkeypatch, tmp_path, [_missing_status()])

        assert _run_perceive_test("--steps", "4") == 1

        captured = capsys.readouterr()
        assert "nothing observed" in captured.err
        assert harness.observer.indices == []
        assert not list(Path(harness.config.runs_dir).glob("*/perceive_result.json"))

    def test_a_minimized_window_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        harness = _install_perceive_harness(
            monkeypatch, tmp_path, [_perceive_minimized_status()]
        )

        assert _run_perceive_test("--steps", "4") == 1

        captured = capsys.readouterr()
        assert "minimized" in captured.err
        assert harness.observer.indices == []

    def test_a_window_bigger_than_recommended_only_warns(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        # ``_focused_status`` is 3840x1950, well over the recommended size. The
        # command must warn and then measure anyway - it never resizes the window.
        _install_perceive_harness(
            monkeypatch,
            tmp_path,
            [_focused_status()],
            frames=[_plain_frame()],
            perception_fit_frames=2,
        )

        assert _run_perceive_test("--steps", "4") == 0

        output = capsys.readouterr().out
        assert "will not resize" in output
        assert "3840x1950" in output


class TestPerceiveTestBounds:
    """A run always stops, and says which bound stopped it."""

    def test_steps_bound_the_run(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        harness = _install_perceive_harness(
            monkeypatch,
            tmp_path,
            [_perceive_status()],
            frames=[_plain_frame()],
            perception_fit_frames=2,
        )

        assert _run_perceive_test("--steps", "7", "--seconds", "600") == 0

        payload = harness.result_payload()
        assert payload["summary"]["stop_reason"] == "reached the 7-step limit"
        assert len(harness.observer.indices) == 7

    def test_the_seconds_default_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        args = cli.build_parser().parse_args(["perceive-test"])
        assert args.seconds == 30.0
        assert args.steps is None

    def test_a_warm_up_that_never_finishes_is_reported_as_such(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
    ) -> None:
        harness = _install_perceive_harness(
            monkeypatch,
            tmp_path,
            [_perceive_status()],
            frames=[_plain_frame()],
            perception_fit_frames=20,
        )

        assert _run_perceive_test("--steps", "5") == 0

        output = capsys.readouterr().out
        assert "warm-up never completed" in output
        assert "The run ended before the model had a warm-up window" in output

        # The measurement is persisted even when it is incomplete: "the warm-up
        # never finished" is a result, and the record says which case it was.
        payload = harness.result_payload()
        assert payload["summary"]["complete"] is False
        assert payload["summary"]["steady_frames"] == 0
        # An unscored row carries no ``score`` key at all rather than a null one.
        assert "score" not in payload["timeline"][-1]


class TestPerceiveTestLearnsTheScene:
    """The command has to actually learn, or it is just a screenshot loop."""

    def test_the_model_is_fitted_from_this_runs_own_frames(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        harness = _install_perceive_harness(
            monkeypatch,
            tmp_path,
            [_perceive_status()],
            frames=[_plain_frame()],
            perception_fit_frames=3,
            perception_grid=4,
        )

        assert _run_perceive_test("--steps", "6") == 0

        model = harness.result_payload()["model"]
        assert model["fitted"] is True
        assert model["frames_seen"] == 3
        assert model["fit"]["frames"] == 2  # the first frame is only a baseline
        assert model["fit"]["cells"] == 16
        assert model["grid"] == 4

    def test_a_still_scene_is_reported_as_still(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        harness = _install_perceive_harness(
            monkeypatch,
            tmp_path,
            [_perceive_status()],
            frames=[_plain_frame()],
            perception_fit_frames=3,
            perception_grid=4,
        )

        assert _run_perceive_test("--steps", "6") == 0

        summary = harness.result_payload()["summary"]
        assert summary["steady_frames"] == 3
        assert summary["changed_cells_mean"] == 0.0
        assert summary["changed_cells_max"] == 0

    def test_a_change_after_the_warm_up_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        # Three identical frames to learn "nothing moves here", then a quarter of
        # the picture jumps from 40 to 220 luma. The learned allowance is the
        # 2.0 floor, so all four cells in that quarter must be flagged.
        harness = _install_perceive_harness(
            monkeypatch,
            tmp_path,
            [_perceive_status()],
            frames=[
                _plain_frame(),
                _plain_frame(),
                _plain_frame(),
                _bright_quadrant_frame(),
            ],
            perception_fit_frames=3,
            perception_grid=4,
        )

        assert _run_perceive_test("--steps", "6") == 0

        summary = harness.result_payload()["summary"]
        assert summary["steady_frames"] == 3
        assert summary["changed_cells_mean"] == 4.0
        assert summary["changed_cells_max"] == 4
        assert summary["total_excess_mean"] > 0.0

        # The change is localised, not smeared over the whole grid.
        assert summary["changed_fraction_mean"] == 0.25

    def test_the_accumulated_map_points_at_where_it_moved(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        harness = _install_perceive_harness(
            monkeypatch,
            tmp_path,
            [_perceive_status()],
            frames=[
                _plain_frame(),
                _plain_frame(),
                _plain_frame(),
                _bright_quadrant_frame(),
            ],
            perception_fit_frames=3,
            perception_grid=4,
        )

        assert _run_perceive_test("--steps", "6") == 0

        accumulated = harness.result_payload()["accumulated_excess"]
        moved = [
            (row, col)
            for row, values in enumerate(accumulated)
            for col, value in enumerate(values)
            if value > 0.0
        ]
        assert moved == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_the_measurement_is_persisted_before_it_is_printed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        # A live run once lost 45 seconds of capture to a formatting error in the
        # reporting path. The numbers now reach disk first, so a crash while
        # printing cannot destroy the measurement.
        harness = _install_perceive_harness(
            monkeypatch,
            tmp_path,
            [_perceive_status()],
            frames=[_plain_frame()],
            perception_fit_frames=3,
            perception_grid=4,
        )

        def explode(model: object) -> list[tuple[str, object]]:
            raise RuntimeError("a formatting error in the reporting path")

        monkeypatch.setattr(cli, "_perceive_model_rows", explode)

        with pytest.raises(RuntimeError, match="formatting error"):
            _run_perceive_test("--steps", "6")

        payload = harness.result_payload()
        assert payload["summary"]["complete"] is True
        assert payload["summary"]["steady_frames"] == 3
        assert len(payload["timeline"]) == 6


class TestParserSurface:
    """The command surface is part of the contract; pin it down."""

    def test_all_documented_commands_exist(self) -> None:
        parser = cli.build_parser()
        for command in (
            "status",
            "capture",
            "observe",
            "input-test",
            "look-test",
            "perceive-test",
            "keys",
            "config",
        ):
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
