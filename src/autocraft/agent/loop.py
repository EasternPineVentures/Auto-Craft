"""The agent loop: OBSERVE -> DECIDE -> SAFETY -> ACT -> VERIFY -> RECORD.

This is the skeleton the specification asks for, and it is deliberately boring.
It cannot run unbounded: :meth:`AgentLoop.run` refuses to start unless the caller
supplies a step limit, a time limit, or both. Every input event goes through the
safety guard, every step is recorded, and the guard's release-all runs on the way
out no matter how the loop ends.

VERIFY here is honest about what it is: it measures whether the pixels changed
after an action. It does not claim the action achieved anything. Task-level
scoring is a future evaluator's job, not the agent's.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..config import Config
from ..control.safety import SafetyGuard
from ..telemetry.recorder import RunRecord, RunRecorder
from .action import Action, ActionExecutor, ActionKind, ActionResult
from .decision import DecisionPolicy, NoOpDecisionPolicy
from .observation import Observation, Observer

__all__ = ["AgentLoop", "StepRecord"]


@dataclass(frozen=True)
class StepRecord:
    """One iteration of the loop, in serialisable form."""

    index: int
    observation: Mapping[str, Any]
    action: Mapping[str, Any]
    result: Mapping[str, Any]
    verification_measured: bool = False
    frame_changed: bool | None = None
    frame_difference: float | None = None
    safety_events: tuple[Mapping[str, Any], ...] = ()
    notes: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def duration(self) -> float:
        """Wall-clock seconds this step took."""
        return max(0.0, self.finished_at - self.started_at)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "index": self.index,
            "observation": dict(self.observation),
            "action": dict(self.action),
            "result": dict(self.result),
            "verification_measured": self.verification_measured,
            "frame_changed": self.frame_changed,
            "frame_difference": self.frame_difference,
            "safety_events": [dict(event) for event in self.safety_events],
            "notes": self.notes,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration": self.duration,
        }


class AgentLoop:
    """Runs bounded OBSERVE -> DECIDE -> SAFETY -> ACT -> VERIFY -> RECORD cycles."""

    def __init__(
        self,
        *,
        config: Config,
        observer: Observer,
        guard: SafetyGuard,
        policy: DecisionPolicy | None = None,
        executor: ActionExecutor | None = None,
        recorder: RunRecorder | None = None,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
        progress: Callable[[StepRecord], None] | None = None,
        change_threshold: float = 0.01,
        save_frames_every: int = 0,
    ) -> None:
        """
        Args:
            config: Effective configuration.
            observer: Produces observations. Never injects input.
            guard: The single safety authority.
            policy: Decision policy. Defaults to :class:`NoOpDecisionPolicy`.
            executor: Action executor. Built from ``guard`` when omitted.
            recorder: Telemetry sink. One is created under ``config.runs_dir``
                when omitted, so a run is always recorded.
            clock: Wall clock, injectable for tests.
            sleeper: Sleep function, injectable for tests.
            progress: Optional callback invoked after each step.
            change_threshold: Mean absolute luminance difference (0..1 scale)
                counted as "the image changed" during VERIFY. The default of
                ``0.01`` corresponds to about 2.5 levels out of 255, which is
                above typical rendering noise but far below a camera turn.
            save_frames_every: Persist a frame every N steps; 0 disables frame
                persistence entirely, which is the default.
        """
        self._config = config
        self._observer = observer
        self._guard = guard
        self._policy = policy if policy is not None else NoOpDecisionPolicy()
        self._executor = executor
        self._recorder = recorder
        self._clock = clock
        self._sleep = sleeper
        self._progress = progress
        self._change_threshold = float(change_threshold)
        self._save_frames_every = int(save_frames_every)

    @property
    def policy_name(self) -> str:
        """Name of the active decision policy."""
        return getattr(self._policy, "name", type(self._policy).__name__)

    @property
    def recorder(self) -> RunRecorder | None:
        """The telemetry recorder, once a run has started."""
        return self._recorder

    # -- run --------------------------------------------------------------

    def run(
        self,
        *,
        max_steps: int | None = None,
        max_seconds: float | None = None,
    ) -> RunRecord:
        """Run the loop until a bound, a safety stop, or a policy stop.

        At least one bound is required. There is no "run forever" option: an
        unbounded loop driving a real game is exactly the failure mode the safety
        requirements exist to prevent.

        Returns:
            The completed :class:`RunRecord`.
        """
        if max_steps is None and max_seconds is None:
            raise ValueError("AgentLoop.run requires max_steps, max_seconds, or both")
        if max_steps is not None and max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        if max_seconds is not None and max_seconds <= 0:
            raise ValueError(f"max_seconds must be positive, got {max_seconds}")

        config = self._config
        config.ensure_directories()
        if self._recorder is None:
            self._recorder = RunRecorder.new_run(
                config.runs_dir,
                config=config.to_dict(),
                clock=self._clock,
            )
        recorder = self._recorder
        if self._executor is None:
            from ..control.keyboard import Keyboard
            from ..control.mouse import Mouse

            self._executor = ActionExecutor(
                Keyboard(self._guard, self._guard.backend, config, sleeper=self._sleep),
                Mouse(self._guard, self._guard.backend, config, sleeper=self._sleep),
                clock=self._clock,
            )
        executor = self._executor

        self._guard.install_atexit()
        self._guard.release_all("run start")
        self._policy.reset()

        started_at = self._clock()
        deadline = started_at + max_seconds if max_seconds is not None else None
        status = "completed"
        stop_reason = self._bound_reason(max_steps, max_seconds)
        step_index = 0
        event_cursor = len(self._guard.events)

        try:
            while True:
                if max_steps is not None and step_index >= max_steps:
                    stop_reason = f"step limit reached ({max_steps})"
                    break
                if deadline is not None and self._clock() >= deadline:
                    stop_reason = f"time limit reached ({max_seconds:g}s)"
                    break
                if self._guard.check_emergency_stop():
                    status = "stopped"
                    stop_reason = f"emergency stop ({self._guard.stop_reason})"
                    break

                released = self._guard.enforce_hold_limits()
                if released:
                    recorder.record_safety_events(self._guard.events[event_cursor:])
                    event_cursor = len(self._guard.events)

                record, step_index, stop_note = self._step(
                    step_index,
                    recorder,
                    event_cursor,
                )
                event_cursor = len(self._guard.events)
                if record is not None:
                    recorder.record_step(record.to_dict())
                    if self._progress is not None:
                        self._progress(record)
                if stop_note is not None:
                    stop_reason = stop_note
                    status = "stopped"
                    break

                if self._guard.should_stop_for_blocks:
                    status = "blocked"
                    stop_reason = (
                        f"safety stop: {self._guard.blocked_streak} consecutive blocked actions "
                        f"(last: {self._guard.last_block_reason})"
                    )
                    break

                self._pace(started_at, step_index)
        except KeyboardInterrupt:
            status = "interrupted"
            stop_reason = "interrupted by Ctrl+C"
            recorder.record_error("KeyboardInterrupt", context="agent loop")
        except Exception as exc:  # noqa: BLE001 - a crashed loop must still release input
            status = "error"
            stop_reason = f"{type(exc).__name__}: {exc}"
            recorder.record_error(f"{type(exc).__name__}: {exc}", context="agent loop")
        finally:
            self._guard.release_all("run end")
            recorder.record_safety_events(self._guard.events[event_cursor:])
            record = recorder.finish(
                status=status,
                stop_reason=stop_reason,
                finished_at=self._clock(),
            )
            self._guard.shutdown("run finished")
        return record

    # -- internals --------------------------------------------------------

    def _step(
        self,
        index: int,
        recorder: RunRecorder,
        event_cursor: int,
    ) -> tuple[StepRecord | None, int, str | None]:
        """Run one cycle. Returns ``(record, next_index, stop_note)``."""
        started_at = self._clock()
        observation = self._observer.observe(index)
        observation_payload = observation.to_dict()

        if observation.frame is not None and self._save_frames_every > 0:
            if index % self._save_frames_every == 0:
                path = recorder.frame_path(f"frame-{index:06d}.png")
                observation.frame.save(path)
                observation_payload["capture_path"] = str(path)

        action = self._policy.decide(observation)
        if action.kind is ActionKind.STOP:
            record = StepRecord(
                index=index,
                observation=observation_payload,
                action=action.to_dict(),
                result=ActionResult(
                    action_id=action.action_id,
                    kind=action.kind,
                    attempted=False,
                    executed=True,
                    started_at=started_at,
                    finished_at=self._clock(),
                    description=action.describe(),
                ).to_dict(),
                safety_events=tuple(event.to_dict() for event in self._guard.events[event_cursor:]),
                notes=str(action.parameters.get("reason", "policy requested stop")),
                started_at=started_at,
                finished_at=self._clock(),
            )
            return record, index + 1, f"policy stop ({action.parameters.get('reason', '')})"

        # The guard gates *injection*, so it is only consulted for actions that
        # would inject something. Asking permission to do nothing would report a
        # no-op as "blocked" and spend the consecutive-block budget, which can
        # stop a run that never touched the game.
        if action.is_input:
            decision = self._guard.authorize(action.describe())
            if not decision.allowed:
                result = ActionResult(
                    action_id=action.action_id,
                    kind=action.kind,
                    attempted=False,
                    executed=False,
                    blocked_reason=decision.reason,
                    started_at=started_at,
                    finished_at=self._clock(),
                    description=action.describe(),
                )
                self._guard.record_block(decision.reason)
            else:
                self._guard.wait_for_rate_limit()
                result = self._executor.execute(action)
                self._guard.note_action()
                if result.was_blocked:
                    self._guard.record_block(result.blocked_reason or "blocked")
                elif result.error:
                    self._guard.record_block(result.error)
                    recorder.record_error(result.error, context=f"step {index} {action.describe()}")
                else:
                    self._guard.record_success()
        else:
            result = self._executor.execute(action)

        measured = False
        frame_changed: bool | None = None
        difference: float | None = None
        if result.ok and action.is_input and observation.frame is not None:
            follow_up = self._observer.observe(index, capture=True)
            if follow_up.frame is not None:
                difference = observation.frame.difference(follow_up.frame)
                frame_changed = difference >= self._change_threshold
                measured = True

        record = StepRecord(
            index=index,
            observation=observation_payload,
            action=action.to_dict(),
            result=result.to_dict(),
            verification_measured=measured,
            frame_changed=frame_changed,
            frame_difference=difference,
            safety_events=tuple(event.to_dict() for event in self._guard.events[event_cursor:]),
            started_at=started_at,
            finished_at=self._clock(),
        )
        return record, index + 1, None

    def _pace(self, started_at: float, steps_done: int) -> None:
        """Sleep so the loop does not spin faster than the configured step rate."""
        interval = self._config.step_interval
        if interval <= 0:
            return
        target = started_at + steps_done * interval
        remaining = target - self._clock()
        if remaining > 0:
            self._sleep(remaining)

    @staticmethod
    def _bound_reason(max_steps: int | None, max_seconds: float | None) -> str:
        parts = []
        if max_steps is not None:
            parts.append(f"max_steps={max_steps}")
        if max_seconds is not None:
            parts.append(f"max_seconds={max_seconds:g}")
        return "loop finished (" + ", ".join(parts) + ")"
