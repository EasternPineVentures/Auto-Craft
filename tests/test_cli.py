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

from dataclasses import dataclass, field

import pytest

from autocraft import cli
from autocraft.agent.action import Action, ActionResult
from autocraft.config import Config
from autocraft.control.safety import SafetyDecision
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
        assert "not the foreground window" in capsys.readouterr().out
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
