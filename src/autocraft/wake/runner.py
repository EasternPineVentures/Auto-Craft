"""The bounded WAKE-001 run.

This module is the thin shell around the behaviour layer. It owns exactly three
things the policy cannot own for itself:

1. **The step bound.** :class:`~autocraft.agent.loop.AgentLoop` needs a step or
   time limit because there is deliberately no "run forever" option. The policy
   bounds its own movements, but a movement and a step are not the same thing, so
   the runner translates one bound into the other.
2. **The record.** The milestone requires that every meaningful action and
   observation is recorded. The runner collects the policy's event stream and its
   metrics and writes them, including when the run is cut short by something other
   than the policy.
3. **The wiring.** The policy is plugged into the existing
   :class:`~autocraft.agent.loop.AgentLoop` rather than into a second loop written
   for this milestone. The safety guard, the executor, the telemetry recorder and
   the observation callback all keep the meaning they already had.

It deliberately does not decide anything. There is no branch here that chooses a
movement, and there is no path by which this module can inject input: the
executor inside :class:`AgentLoop` is the only thing that can, and it is reached
only through the guard.

The status mapping is worth stating because it is the one place the runner
interprets anything. The policy's terminal state is the truth when it has one:

* ``COMPLETE`` -> ``completed`` - the agent stopped because it was finished;
* ``FAILED`` -> ``failed`` - the agent stopped because it ran out of something;
* ``SAFE_STOP`` -> ``aborted`` - the world stopped being safe to act on.

If the policy has **no** terminal state, the loop stopped it instead - a step
limit, an emergency stop, a crash, a Ctrl+C. That is reported as ``aborted`` with
the loop's own reason, because an agent that was interrupted did not decide
anything and the record must not imply that it did.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from ..agent.loop import AgentLoop
from ..agent.observation import Observation, Observer
from ..config import Config
from ..telemetry.recorder import RunRecorder
from .events import WakeEvent
from .policy import TERMINAL_STATES, WakeDecisionPolicy, WakeState
from .record import (
    STATUS_ABORTED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    WakeRecorder,
    WakeResult,
)

__all__ = [
    "DEFAULT_MAX_SECONDS",
    "WakeRunner",
    "status_for_state",
]

#: The default wall-clock ceiling. The milestone requires a bounded run, and a
#: movement budget alone does not bound *time*: a window that takes a second to
#: capture would let 45 movements take a minute. Two minutes is comfortably more
#: than a healthy run needs and short enough that a wedged one gives the operator
#: their screen back.
DEFAULT_MAX_SECONDS = 120.0


def status_for_state(state: WakeState) -> str:
    """Map the policy's final state onto the record's status vocabulary.

    The rule is state-driven on purpose, and the fall-through is the important
    part: a policy that never reached a terminal state did not decide to stop, so
    whatever ended the run, the agent did not. Reporting that as ``failed`` would
    blame the behaviour for a step limit, a wall-clock limit or a wedged window,
    none of which are things it chose.
    """
    if state is WakeState.COMPLETE:
        return STATUS_COMPLETED
    if state is WakeState.FAILED:
        return STATUS_FAILED
    return STATUS_ABORTED


class WakeRunner:
    """Runs one bounded WAKE-001 attempt and records it.

    Args:
        config: Effective configuration.
        observer: Produces observations. Never injects input.
        guard: The single safety authority. Typed loosely on purpose: the guard is
            handed straight to :class:`AgentLoop` and this module never calls a
            method on it, so importing its class from the control package would buy
            nothing and would give the wake package a route to the input layer it
            has no business having.
        policy: The behaviour layer. Its budgets are the run's budgets.
        recorder: Where the milestone result is written.
        executor: Action executor. Built by the loop from ``guard`` when omitted.
        run_recorder: Telemetry sink. The loop creates one when omitted.
        max_seconds: Wall-clock ceiling. ``None`` uses :data:`DEFAULT_MAX_SECONDS`;
            a value is never allowed to be absent, because an unbounded run is
            exactly what the safety requirements exist to prevent.
        clock: Wall clock, injectable for tests.
        sleeper: Sleep function, injectable for tests.
        on_event: Optional callback for each :class:`WakeEvent`, called as soon as
            the policy emits it. Display only; the return value is ignored.
        on_status: Optional callback with the policy's flat report after every
            observation. Display only, and read-only by construction.
    """

    def __init__(
        self,
        *,
        config: Config,
        observer: Observer,
        guard: Any,
        policy: WakeDecisionPolicy,
        recorder: WakeRecorder,
        executor: Any | None = None,
        run_recorder: RunRecorder | None = None,
        max_seconds: float | None = None,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
        on_event: Callable[[WakeEvent], None] | None = None,
        on_status: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.observer = observer
        self.guard = guard
        self.policy = policy
        self.recorder = recorder
        self.executor = executor
        self.run_recorder = run_recorder
        self.max_seconds = DEFAULT_MAX_SECONDS if max_seconds is None else float(max_seconds)
        if self.max_seconds <= 0:
            raise ValueError(f"max_seconds must be positive, got {max_seconds}")
        self.clock = clock
        self.sleeper = sleeper
        self._on_event = on_event
        self._on_status = on_status
        self._events: list[WakeEvent] = []
        self._steps = 0
        self._loop: AgentLoop | None = None

    # -- read side --------------------------------------------------------

    @property
    def events(self) -> tuple[WakeEvent, ...]:
        """Every event the policy has emitted, oldest first."""
        return tuple(self._events)

    @property
    def steps(self) -> int:
        """Loop steps taken so far."""
        return self._steps

    @property
    def max_steps(self) -> int:
        """The step bound handed to the loop."""
        return self.policy.step_budget

    # -- run --------------------------------------------------------------

    def run(self) -> WakeResult:
        """Run the policy inside the existing agent loop, then record the outcome.

        Returns the finished :class:`WakeResult`. The loop always terminates: it
        is bounded by steps *and* seconds, and the policy additionally bounds its
        own movements.
        """
        loop = AgentLoop(
            config=self.config,
            observer=self.observer,
            guard=self.guard,
            policy=self.policy,
            executor=self.executor,
            recorder=self.run_recorder,
            clock=self.clock,
            sleeper=self.sleeper,
            progress=self._after_step,
            on_observation=self._after_observation,
        )
        self._loop = loop
        record = loop.run(max_steps=self.max_steps, max_seconds=self.max_seconds)
        self._drain()
        self.recorder.update(
            metrics=self.policy.report(),
            events=self._events,
            steps=self._steps,
            state=self.policy.state.value,
        )

        state = self.policy.state
        status = status_for_state(state)
        stop_reason = self.policy.stop_reason or record.stop_reason
        for note in self._notes(state, record.stop_reason):
            # A run the loop had to stop needs saying so in the record itself.
            # Someone reading wake_result.json months later will not have the
            # telemetry next to it, and "aborted" on its own would not say which
            # bound ran out.
            self.recorder.add_note(note)
        return self.recorder.finish(
            status=status,
            stop_reason=stop_reason,
            state=state.value,
        )

    # -- callbacks --------------------------------------------------------

    def _after_observation(self, observation: Observation) -> None:
        """Drain the previous step's events and refresh the display.

        Called as soon as an observation exists and *before* the policy decides,
        so what is published is the state the previous decision produced. That is
        the honest thing to show: the panel must not be able to display an outcome
        for a decision that has not happened yet.
        """
        frame = observation.frame
        if frame is not None and self.recorder.window == (0, 0):
            # The window the run actually looked at is a measurement, not a
            # setting: it is whatever the first frame turned out to be.
            self.recorder.note_window(int(frame.width), int(frame.height))
        self._drain()
        self._publish_status()

    def _after_step(self, record: Any) -> None:
        """Count a completed step and refresh the record on disk.

        The record is rewritten per step rather than only at the end, so a run
        that is killed mid-way still leaves what it had measured. A partial record
        is worth more than an empty one, and it cannot be mistaken for a finished
        run because its status is still ``running``.
        """
        self._steps += 1
        self.recorder.update(
            metrics=self.policy.report(),
            events=self._events,
            steps=self._steps,
            state=self.policy.state.value,
        )

    def _drain(self) -> None:
        """Forward every pending event, in order, to the callback and the record."""
        pending = self.policy.drain_events()
        if not pending:
            return
        self._events.extend(pending)
        if self._on_event is not None:
            for event in pending:
                self._on_event(event)

    def _publish_status(self) -> None:
        """Hand the display layer a flat snapshot of the policy's state."""
        if self._on_status is None:
            return
        report = dict(self.policy.report())
        report["steps"] = self._steps
        report["max_steps"] = self.max_steps
        self._on_status(report)

    # -- recording helpers ------------------------------------------------

    def _notes(self, state: WakeState, stop_reason: str) -> tuple[str, ...]:
        """Notes explaining a run the policy did not end by itself."""
        if state in TERMINAL_STATES:
            return ()
        detail = f" The loop gave as its reason: {stop_reason}" if stop_reason else ""
        return (
            "This run ended without the behaviour layer reaching a terminal state, "
            "so the agent never decided to stop; the run was stopped from outside "
            f"it.{detail}",
        )
