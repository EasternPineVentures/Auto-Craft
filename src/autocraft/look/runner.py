"""The bounded LOOK-001 trial runner.

This module executes the experiment's fixed sequence and nothing else. It has no
input backend, no guard and no Win32 import: everything that touches the world
arrives as an injected callable. That is not decoration - it is what lets the
runner be tested exhaustively with fake input, which is the only way a test suite
may exercise it (no test may move the real mouse), and it keeps the sequence
itself auditable in one screen.

The sequence, per trial:

1. verify the target window is still the exact window we vetted;
2. capture frame A;
3. verify again, then inject the outbound delta ``(+dx, +dy)``;
4. settle;
5. capture frame B;
6. verify again, then inject the exact reverse ``(-dx, -dy)``;
7. settle;
8. capture frame C;
9. measure A-vs-B, A-vs-C, the displacement estimate and the reversibility
   ratio, and hand the result to the recorder.

Abort rules, all of which stop the experiment rather than continuing to the next
trial, because the specification requires that nothing further be injected once
the world stops being what we vetted:

* the target is not the foreground window, has gone away, or is a *different*
  window with a matching title -> interrupted, everything released, frames so far
  kept;
* the emergency stop is latched, or was requested between steps -> interrupted;
* a capture fails -> failed;
* an actuator refuses or errors -> failed, with the guard's own reason;
* a measurement cannot be taken, for example because the window was resized
  mid-trial -> failed, with the reason.

In every abort path the runner releases held input before returning, and it
never injects the reverse movement unless the outbound movement was actually
sent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

from ..vision.capture import CaptureError
from ..vision.frame import Frame, FrameError
from .metrics import (
    DEFAULT_BLOCK_GRID,
    LookError,
    ShiftEstimate,
    difference_metrics,
    estimate_shift,
    pixels_per_delta,
    reversibility_note,
    reversibility_ratio,
)
from .record import (
    TRIAL_COMPLETED,
    TRIAL_FAILED,
    TRIAL_INTERRUPTED,
    LookRecorder,
    LookResult,
    LookTrialResult,
    TrialSpec,
)

__all__ = [
    "FocusCheck",
    "LookRunner",
    "MoveOutcome",
    "capture_failure_reason",
]


@dataclass(frozen=True)
class MoveOutcome:
    """What the actuator layer reported for one requested movement.

    ``sent`` is the only field the sequence branches on. ``refused`` separates
    "the safety layer would not allow it" from "the backend failed", because
    those two are worth reporting differently even though both stop the trial.
    """

    sent: bool
    detail: str = ""
    refused: bool = False


@dataclass(frozen=True)
class FocusCheck:
    """Whether the vetted target window is still the window in front."""

    ready: bool
    reason: str = ""


CaptureFn = Callable[[], Frame]
MoveFn = Callable[[int, int], MoveOutcome]
VerifyFn = Callable[[], FocusCheck]
StopRequestedFn = Callable[[], bool]
ReleaseFn = Callable[[str], None]
EventFn = Callable[[str, str], None]
FrameFn = Callable[[Frame], None]
TrialFn = Callable[[LookTrialResult], None]
StartFn = Callable[[TrialSpec, int], None]

#: Event kinds the runner emits. Plain strings, because the runner does not know
#: the observer exists; the caller maps them onto its own event vocabulary.
EVENT_INFO = "info"
EVENT_OBSERVE = "observe"
EVENT_SAFETY = "safety"
EVENT_ERROR = "error"


def capture_failure_reason(role: str, exc: BaseException) -> str:
    """Uniform wording for a failed capture, used by every call site."""
    return f"frame {role} could not be captured ({type(exc).__name__}: {exc})"


class LookRunner:
    """Runs the LOOK-001 sequence against injected callables.

    Args:
        recorder: Where results are written.
        capture: Returns the current frame of the vetted target window. May raise
            :class:`~autocraft.vision.capture.CaptureError` or
            :class:`~autocraft.vision.frame.FrameError`.
        move: Injects a relative mouse delta. Must not raise; it reports refusal
            or failure through :class:`MoveOutcome`.
        verify: Re-checks that the target is still the exact vetted window and is
            in front. Must not raise.
        stop_requested: True once the emergency stop has been latched.
        release: Releases all held input. Must not raise, and is called on every
            abort path before the runner returns.
        window_size: The vetted window's size, used for the record when a trial
            fails before any frame exists.
        block_grid: Coarse partition size for the difference map.
        clock: Wall clock, for timestamps and capture timing.
        sleeper: Sleep function, injected so tests need no real delay.
        on_event: Optional ``(message, kind)`` callback for the timeline.
        on_frame: Optional callback for each captured frame, for the display.
        on_trial: Optional callback for each recorded trial.
        on_trial_start: Optional ``(spec, total)`` callback before each trial.
    """

    def __init__(
        self,
        *,
        recorder: LookRecorder,
        capture: CaptureFn,
        move: MoveFn,
        verify: VerifyFn,
        stop_requested: StopRequestedFn,
        release: ReleaseFn,
        window_size: tuple[int, int] = (0, 0),
        block_grid: int = DEFAULT_BLOCK_GRID,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
        on_event: EventFn | None = None,
        on_frame: FrameFn | None = None,
        on_trial: TrialFn | None = None,
        on_trial_start: StartFn | None = None,
    ) -> None:
        self._recorder = recorder
        self._capture = capture
        self._move = move
        self._verify = verify
        self._stop_requested = stop_requested
        self._release = release
        self._window_width, self._window_height = (int(window_size[0]), int(window_size[1]))
        self._block_grid = int(block_grid)
        self._clock = clock
        self._sleeper = sleeper
        self._on_event = on_event
        self._on_frame = on_frame
        self._on_trial = on_trial
        self._on_trial_start = on_trial_start
        self._last_capture_reason = ""

    # -- public entry point ----------------------------------------------

    def run(self, plan: Sequence[TrialSpec]) -> LookResult:
        """Run every planned trial, or stop at the first abort.

        Returns the finished record. The experiment stops at the first trial that
        does not complete; it never continues to a later trial after an abort.
        """
        specs = tuple(plan)
        for spec in specs:
            if self._on_trial_start is not None:
                self._on_trial_start(spec, len(specs))
            result = self._recorder.record(self._run_trial(spec))
            if self._on_trial is not None:
                self._on_trial(result)
            if not result.complete:
                return self._recorder.finish(
                    status=result.status,
                    stop_reason=result.stop_reason,
                )
        return self._recorder.finish(
            status=TRIAL_COMPLETED,
            stop_reason=f"all {len(specs)} planned trial(s) completed",
        )

    # -- one trial -------------------------------------------------------

    def _run_trial(self, spec: TrialSpec) -> LookTrialResult:
        frames: dict[str, Frame] = {}
        capture_seconds = 0.0
        movements = 0
        window_width, window_height = self._window_width, self._window_height

        def abort(status: str, reason: str) -> LookTrialResult:
            self._release_on_abort(reason)
            return self._build(
                spec,
                status=status,
                stop_reason=reason,
                frames=frames,
                movements=movements,
                capture_seconds=capture_seconds,
                window_width=window_width,
                window_height=window_height,
            )

        # 1. The window must still be the one we vetted, before anything at all.
        check = self._verify()
        if not check.ready:
            return abort(TRIAL_INTERRUPTED, check.reason)
        if self._stop_requested():
            return abort(TRIAL_INTERRUPTED, "emergency stop was requested before any movement")

        # 2. Frame A: the baseline the other two are measured against.
        captured = self._capture_into(frames, "a", capture_seconds)
        if captured is None:
            return abort(TRIAL_FAILED, self._last_capture_reason)
        frames["a"], capture_seconds = captured
        window_width, window_height = frames["a"].width, frames["a"].height
        self._emit_frame(frames["a"])

        # 3. Re-verify immediately before injecting, then inject the outbound delta.
        check = self._verify()
        if not check.ready:
            return abort(TRIAL_INTERRUPTED, check.reason)
        if self._stop_requested():
            return abort(TRIAL_INTERRUPTED, "emergency stop was requested before the outbound movement")
        outcome = self._move(spec.dx, spec.dy)
        if not outcome.sent:
            return abort(TRIAL_FAILED, _move_failure("outbound", outcome))
        movements += 1

        # 4-5. Let the picture come to rest, then capture frame B.
        self._settle(spec.settle_seconds)
        if self._stop_requested():
            return abort(TRIAL_INTERRUPTED, "emergency stop was requested after the outbound movement")
        captured = self._capture_into(frames, "b", capture_seconds)
        if captured is None:
            return abort(TRIAL_FAILED, self._last_capture_reason)
        frames["b"], capture_seconds = captured
        self._emit_frame(frames["b"])

        # 6. Re-verify, then inject the exact reverse.
        check = self._verify()
        if not check.ready:
            return abort(TRIAL_INTERRUPTED, check.reason)
        if self._stop_requested():
            return abort(TRIAL_INTERRUPTED, "emergency stop was requested before the reverse movement")
        outcome = self._move(spec.reverse_dx, spec.reverse_dy)
        if not outcome.sent:
            return abort(TRIAL_FAILED, _move_failure("reverse", outcome))
        movements += 1

        # 7-8. Settle again, then capture frame C.
        self._settle(spec.settle_seconds)
        if self._stop_requested():
            return abort(TRIAL_INTERRUPTED, "emergency stop was requested after the reverse movement")
        captured = self._capture_into(frames, "c", capture_seconds)
        if captured is None:
            return abort(TRIAL_FAILED, self._last_capture_reason)
        frames["c"], capture_seconds = captured
        self._emit_frame(frames["c"])

        # 9. Measure. A failure here is a failed trial, not a silent zero.
        try:
            result = self._measure(
                spec,
                frames,
                movements=movements,
                capture_seconds=capture_seconds,
                window_width=window_width,
                window_height=window_height,
            )
        except LookError as exc:
            return abort(TRIAL_FAILED, f"the trial could not be measured ({exc})")
        self._emit(_measured_summary(spec, result), EVENT_OBSERVE)
        return result

    # -- measurement -----------------------------------------------------

    def _measure(
        self,
        spec: TrialSpec,
        frames: dict[str, Frame],
        *,
        movements: int,
        capture_seconds: float,
        window_width: int,
        window_height: int,
    ) -> LookTrialResult:
        frame_a, frame_b, frame_c = frames["a"], frames["b"], frames["c"]
        a_to_b = difference_metrics(frame_a, frame_b, block_grid=self._block_grid)
        a_to_c = difference_metrics(frame_a, frame_c, block_grid=self._block_grid)
        shift = self._estimate_shift(frame_a, frame_b)
        per_x: float | None = None
        per_y: float | None = None
        if shift.available:
            per_x, per_y = pixels_per_delta(shift, dx=spec.dx, dy=spec.dy)
        ratio = reversibility_ratio(a_to_b.mean_absolute_difference, a_to_c.mean_absolute_difference)
        return LookTrialResult(
            spec=spec,
            status=TRIAL_COMPLETED,
            stop_reason="",
            window_width=window_width,
            window_height=window_height,
            capture_seconds=capture_seconds,
            movements_sent=movements,
            measured_at=float(self._clock()),
            frame_a=frame_a,
            frame_b=frame_b,
            frame_c=frame_c,
            a_to_b=a_to_b,
            a_to_c=a_to_c,
            shift=shift,
            pixels_per_delta_x=per_x,
            pixels_per_delta_y=per_y,
            reversibility_ratio=ratio,
            reversibility_note=reversibility_note(
                a_to_b.mean_absolute_difference, a_to_c.mean_absolute_difference
            ),
        )

    def _estimate_shift(self, mine: Frame, theirs: Frame) -> ShiftEstimate:
        """Estimate displacement, degrading to an explained absence.

        A missing displacement estimate does not invalidate the difference
        measurements, so this never fails the trial: it reports that the estimate
        was not available and why.
        """
        try:
            return estimate_shift(mine, theirs)
        except LookError as exc:
            self._emit(f"Displacement estimate unavailable: {exc}", EVENT_INFO)
            return ShiftEstimate(available=False, reason=str(exc))

    # -- helpers ---------------------------------------------------------

    def _capture_into(
        self,
        frames: dict[str, Frame],
        role: str,
        elapsed: float,
    ) -> tuple[Frame, float] | None:
        """Capture one frame, timing it, or record why it failed."""
        started = float(self._clock())
        try:
            frame = self._capture()
        except (CaptureError, FrameError, LookError) as exc:
            self._last_capture_reason = capture_failure_reason(role.upper(), exc)
            self._emit(self._last_capture_reason, EVENT_ERROR)
            return None
        except Exception as exc:  # noqa: BLE001 - a broken backend must not escape
            self._last_capture_reason = capture_failure_reason(role.upper(), exc)
            self._emit(self._last_capture_reason, EVENT_ERROR)
            return None
        seconds = max(0.0, float(self._clock()) - started)
        self._emit(f"Captured frame {role.upper()} in {seconds * 1000:.0f} ms.", EVENT_OBSERVE)
        return frame, elapsed + seconds

    def _settle(self, seconds: float) -> None:
        if seconds > 0:
            self._sleeper(float(seconds))

    def _release_on_abort(self, reason: str) -> None:
        """Release held input, and never let a release failure hide the abort."""
        try:
            self._release(f"look-test aborted: {reason}")
        except Exception as exc:  # noqa: BLE001 - the abort itself matters more
            self._emit(f"Releasing held input reported: {exc}", EVENT_SAFETY)

    def _emit(self, message: str, kind: str = EVENT_INFO) -> None:
        if self._on_event is not None:
            self._on_event(message, kind)

    def _emit_frame(self, frame: Frame) -> None:
        if self._on_frame is not None:
            self._on_frame(frame)

    def _build(
        self,
        spec: TrialSpec,
        *,
        status: str,
        stop_reason: str,
        frames: dict[str, Frame],
        movements: int,
        capture_seconds: float,
        window_width: int,
        window_height: int,
    ) -> LookTrialResult:
        return LookTrialResult(
            spec=spec,
            status=status,
            stop_reason=stop_reason,
            window_width=int(window_width),
            window_height=int(window_height),
            capture_seconds=float(capture_seconds),
            movements_sent=int(movements),
            measured_at=float(self._clock()),
            frame_a=frames.get("a"),
            frame_b=frames.get("b"),
            frame_c=frames.get("c"),
            reversibility_note=_abort_note(status, stop_reason),
        )

    #: Set by :meth:`_capture_into` so :meth:`_run_trial` can report the reason
    #: without threading an exception object through the sequence.
    _last_capture_reason: str


def _move_failure(direction: str, outcome: MoveOutcome) -> str:
    """Uniform wording for a movement that did not reach the game."""
    what = "the safety layer refused it" if outcome.refused else "the backend did not send it"
    detail = f": {outcome.detail}" if outcome.detail else ""
    return f"the {direction} movement was not injected ({what}){detail}"


def _abort_note(status: str, stop_reason: str) -> str:
    """Note left in place of a reversibility reading on a trial that stopped early."""
    if status == TRIAL_COMPLETED:
        return ""
    return f"not measured: {stop_reason}"


def _measured_summary(spec: TrialSpec, result: LookTrialResult) -> str:
    """One timeline line describing a completed trial's two headline numbers."""
    difference = None if result.a_to_b is None else result.a_to_b.mean_absolute_difference
    ratio = result.reversibility_ratio
    parts = [f"LOOK trial {spec.index + 1} measured"]
    if difference is not None:
        parts.append(f"A-to-B {difference:.2f} luma levels")
    parts.append(
        "reversibility not defined" if ratio is None else f"reversibility {ratio:.3f}"
    )
    return ", ".join(parts) + "."
