"""WAKE-001's behaviour layer: wake up, look around, pick something, centre it.

This is the middle layer of the three the milestone insists stay separate.
Perception (:mod:`autocraft.wake.salience`, :mod:`autocraft.wake.memory`) answers
*what changed and what looks interesting*. Motor
(:mod:`autocraft.wake.centering`, and the existing guard and executor) answers
*how do I move the mouse safely*. This module answers *what small thing should I
attempt next*, and it is the only one of the three that is allowed to decide.

It is a :class:`~autocraft.agent.decision.DecisionPolicy` - the seam that already
exists - so WAKE-001 is a new policy plugged into the existing
:class:`~autocraft.agent.loop.AgentLoop`, not a second agent architecture. The
loop calls :meth:`WakeDecisionPolicy.decide` with a fresh
:class:`~autocraft.agent.observation.Observation` each step and executes whatever
:class:`~autocraft.agent.action.Action` comes back, which is what makes the
milestone's "every correction must be based on the latest frame" rule structural
rather than a promise: there is no queue here and no way to emit a precomputed
series, because ``decide`` can only ever return one action and only ever sees the
frame it was just handed.

The states are inspectable and the transitions are explicit, because the
milestone asks for both::

    STARTING ──▶ SCANNING ──▶ SELECTING ──▶ CENTERING ──▶ COMPLETE
                    ▲             ▲             │
                    │             │             ▼
                    └─────────────┴──────── REACQUIRING
                                                 │
                                                 ▼
                                    FAILED / SAFE_STOP

``SCANNING`` inspects a fixed, deterministic sequence of directions and returns to
where it started. ``SELECTING`` scores whatever candidates accumulated and picks
one. ``CENTERING`` closes the loop on that one, one correction at a time.
``REACQUIRING`` is what happens when a correction loses it. Every state has a
budget and every budget ends in a clean terminal state, so WAKE-001 always
terminates.

Two things this policy deliberately does not do. It never says what anything *is*:
a target is a "selected visual region", and there is no code path here that could
name one. And it never moves without having just looked - the only input it emits
is derived from the frame in the observation it was handed, so a stale belief
cannot turn into a movement.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable

import numpy as np

from ..agent.action import Action
from ..agent.observation import Observation
from ..vision.frame import Frame
from .centering import (
    BAND_COUNTS,
    BAND_ORDER,
    CenteringController,
    CenteringMove,
    MotionCalibration,
    band_for_distance,
    direction_of,
    strategy_name,
)
from .events import WakeEvent, WakeEventKind
from .memory import (
    DEFAULT_SIMILAR_THRESHOLD,
    DEFAULT_VIEW_GRID,
    ActionRecord,
    ProgressSample,
    ShortTermExperience,
    ViewFingerprint,
)
from .progress import (
    DEFAULT_COOLDOWN_FAILURES,
    DEFAULT_PROGRESS_EPSILON,
    DEFAULT_STUCK_REPEATS,
    ProgressModel,
    RepetitionGuard,
    RepetitionVerdict,
    StrategyCooldown,
)
from .salience import (
    CANDIDATE_LIMIT,
    DEFAULT_MIN_SALIENCE,
    DEFAULT_PEAK_FRACTION,
    CandidateTarget,
    Relocation,
    SalienceDiagnostics,
    SalienceMap,
    SelectionWeights,
    TargetPatch,
    descriptor_similarity,
    find_candidates,
    locate,
    relocate,
    salience_map,
    score_candidates,
    select_candidate,
)

__all__ = [
    "DEFAULT_MAX_CAPTURE_FAILURES",
    "DEFAULT_MAX_CENTER_MOVES",
    "DEFAULT_MAX_MOVES",
    "DEFAULT_MAX_REACQUIRE_MOVES",
    "DEFAULT_MAX_SCAN_MOVES",
    "DEFAULT_MAX_TARGET_CANDIDATES",
    "DEFAULT_MIN_TARGET_CONFIDENCE",
    "DEFAULT_MIN_VIEWS_BEFORE_SELECTING",
    "DEFAULT_SCAN_COUNTS",
    "SCAN_OFFSETS",
    "WakeDecisionPolicy",
    "WakeState",
]


class WakeState(str, Enum):
    """What the behaviour layer is doing right now.

    A ``str`` enum so it serialises into the observer snapshot and the run record
    as the word itself rather than an integer, which matters because these names
    are what the owner reads while watching.
    """

    STARTING = "STARTING"
    SCANNING = "SCANNING"
    SELECTING = "SELECTING"
    CENTERING = "CENTERING"
    REACQUIRING = "REACQUIRING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    SAFE_STOP = "SAFE_STOP"


#: States from which no further input will ever be sent.
TERMINAL_STATES = frozenset({WakeState.COMPLETE, WakeState.FAILED, WakeState.SAFE_STOP})

#: The scan plan, as offsets *from the view the run started in*, in units of
#: :data:`DEFAULT_SCAN_COUNTS` mouse counts.
#:
#: This is the milestone's "inspect left, inspect right, inspect further left and
#: right, return toward the remembered view", written out rather than generated.
#: A fixed plan is used instead of a random walk for three reasons. It is
#: reproducible, so a run can be explained afterwards. It covers both axes rather
#: than only the one the first movement happened to favour. And the last entry
#: returns to the origin, which is the starting view - so the run deliberately
#: looks at the view it began with, which is what gives view memory something real
#: to notice and what the ``VIEW_REVISITED`` event is for.
SCAN_OFFSETS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (-1, 0),
    (2, 0),
    (-2, 0),
    (0, 1),
    (0, -1),
    (0, 2),
    (0, -2),
    (0, 0),
)

#: One scan step, in mouse counts. Below ``max_mouse_delta`` on purpose: a scan is
#: meant to show a *new* view, not to fling the camera so far that nothing from the
#: previous view survives. The ``--dx 200`` probe is the evidence for that - two
#: hundred counts moved the view far enough that no shared feature remained.
DEFAULT_SCAN_COUNTS = 60

#: How many scan movements SCANNING may spend before it has to select from whatever
#: it has seen. The milestone's suggested value.
DEFAULT_MAX_SCAN_MOVES = 12

#: How many different candidates may be attempted in one run. "Do not allow one
#: failed target to consume the whole run."
DEFAULT_MAX_TARGET_CANDIDATES = 3

#: How many centring movements one candidate gets before it is abandoned.
DEFAULT_MAX_CENTER_MOVES = 8

#: How many recovery movements may be spent looking for a lost target before its
#: candidate is marked failed and another is chosen.
DEFAULT_MAX_REACQUIRE_MOVES = 3

#: How many consecutive captures may fail before the run aborts. A capture failure
#: is not something the policy can fix by moving.
DEFAULT_MAX_CAPTURE_FAILURES = 3

#: How many distinct views SCANNING must have seen before SELECTING is allowed.
#: The milestone lists "explores more than one view" as a success criterion, so
#: picking a target from the very first view would fail it.
DEFAULT_MIN_VIEWS_BEFORE_SELECTING = 2

#: Below this the relocation is not believed and the target counts as lost.
#:
#: Set at the descriptor matcher's own floor rather than above it, on purpose. The
#: reported confidence is the lower of the patch matcher's and the descriptor
#: matcher's, so raising this above that floor would make the *patch* matcher's
#: opinion irrelevant and quietly turn this into a descriptor-only system. Equal
#: thresholds mean the two have to agree that the target is there.
DEFAULT_MIN_TARGET_CONFIDENCE = 0.35

#: Total input movements a run may ever send, whatever the state. The per-state
#: budgets already bound this; the total exists so that a bug in one budget cannot
#: produce an unbounded run, and so the CLI has one number to print.
DEFAULT_MAX_MOVES = (
    DEFAULT_MAX_SCAN_MOVES
    + DEFAULT_MAX_TARGET_CANDIDATES * (DEFAULT_MAX_CENTER_MOVES + DEFAULT_MAX_REACQUIRE_MOVES)
)

#: Display-only thought templates. The milestone allows these in demo and observer
#: mode and is explicit that they must not influence movement, so they are emitted
#: through a callback that returns nothing and is called *after* the action for the
#: step has already been chosen. Nothing in this module reads anything back from it.
_THOUGHTS = {
    "curious": "Something over there looks different.",
    "confused": "I think I've already looked this way.",
    "frustrated": "That did not move where I expected.",
    "reflective": "I am starting to understand how looking works.",
    "satisfied": "That is centred now.",
}

#: A thought may only be emitted every this many steps, so the observer feed reads
#: as occasional remarks rather than a running commentary.
_THOUGHT_INTERVAL = 3


@dataclass
class _Target:
    """The candidate currently being centred, with everything needed to chase it."""

    candidate: CandidateTarget
    patch: TargetPatch
    offset: tuple[float, float]
    attempts: int = 0
    losses: int = 0
    best_distance: float | None = None


class WakeDecisionPolicy:
    """Look around, choose a visually salient region, centre it, stop.

    Satisfies :class:`~autocraft.agent.decision.DecisionPolicy`: it has ``name``,
    ``decide`` and ``reset``, and nothing else is required of it.

    Args:
        salience_grid: Cells per axis for the salience map. Eight is the milestone's
            default and is deliberately coarse: this is attention, not measurement.
        view_grid: Cells per axis for the view fingerprint.
        max_mouse_delta: Hard ceiling on any single movement, in counts. The guard
            enforces this too; the policy respects it so that it never even asks
            for something that would be refused.
        scan_counts: One scan step, in counts.
        max_scan_moves: SCANNING's movement budget.
        max_target_candidates: How many candidates may be attempted.
        max_center_moves: CENTERING's movement budget per candidate.
        max_reacquire_moves: REACQUIRING's movement budget per candidate.
        max_capture_failures: Consecutive capture failures tolerated.
        max_moves: Total movement budget for the run.
        dead_zone_px: How close to the frame centre counts as centred.
        min_target_confidence: Below this the target is treated as lost.
        min_views_before_selecting: Distinct views required before selecting.
        min_salience: Salience floor for a cell to be part of a candidate.
        peak_fraction: Fraction of the peak cell salience a cell must reach.
        weights: The candidate scoring weights, passed through unchanged.
        tie_epsilon: Scores within this of the best are treated as equal, and the
            choice among them may vary. Zero means always take the best.
        rng: Random source used *only* for that tie-break. Seeded in tests.
        coarse_scale: Block-average factor for the coarse locate pass.
        refine_window_px: Fine locate window. ``None`` lets :func:`~autocraft.wake.salience.locate`
            size it from ``coarse_scale``.
        calibration: A :class:`~autocraft.wake.centering.MotionCalibration` to adopt.
            ``None`` starts an unmeasured one, which is the honest default: LOOK-001
            never measured the mapping.
        progress_epsilon: Distance improvement, in pixels, that counts as progress.
        stuck_repeats: Identical unproductive repetitions before a loop is declared.
        cooldown_failures: Failures before a strategy is cooled down.
        clock: Monotonic clock, injectable for tests.
        thought_hook: Display-only ``callable(mood, text)``. Its return value is
            ignored and it is never consulted when choosing an action.
    """

    name = "wake-001"

    def __init__(
        self,
        *,
        salience_grid: int = 8,
        view_grid: int = DEFAULT_VIEW_GRID,
        max_mouse_delta: int = 200,
        scan_counts: int = DEFAULT_SCAN_COUNTS,
        max_scan_moves: int = DEFAULT_MAX_SCAN_MOVES,
        max_target_candidates: int = DEFAULT_MAX_TARGET_CANDIDATES,
        max_center_moves: int = DEFAULT_MAX_CENTER_MOVES,
        max_reacquire_moves: int = DEFAULT_MAX_REACQUIRE_MOVES,
        max_capture_failures: int = DEFAULT_MAX_CAPTURE_FAILURES,
        max_moves: int | None = None,
        dead_zone_px: float = 12.0,
        min_target_confidence: float = DEFAULT_MIN_TARGET_CONFIDENCE,
        min_views_before_selecting: int = DEFAULT_MIN_VIEWS_BEFORE_SELECTING,
        min_salience: float = DEFAULT_MIN_SALIENCE,
        peak_fraction: float = DEFAULT_PEAK_FRACTION,
        weights: SelectionWeights | None = None,
        tie_epsilon: float = 0.05,
        rng: random.Random | None = None,
        coarse_scale: int = 4,
        refine_window_px: int | None = None,
        calibration: MotionCalibration | None = None,
        progress_epsilon: float = DEFAULT_PROGRESS_EPSILON,
        stuck_repeats: int = DEFAULT_STUCK_REPEATS,
        cooldown_failures: int = DEFAULT_COOLDOWN_FAILURES,
        clock: Callable[[], float] = time.monotonic,
        thought_hook: Callable[[str, str], None] | None = None,
    ) -> None:
        if max_mouse_delta < 1:
            raise ValueError("max_mouse_delta must be at least 1")
        if scan_counts < 1:
            raise ValueError("scan_counts must be at least 1")
        if max_scan_moves < 0 or max_center_moves < 0 or max_reacquire_moves < 0:
            raise ValueError("movement budgets cannot be negative")
        if max_target_candidates < 1:
            raise ValueError("max_target_candidates must be at least 1")
        if min_views_before_selecting < 1:
            raise ValueError("min_views_before_selecting must be at least 1")
        if not 0.0 <= tie_epsilon:
            raise ValueError("tie_epsilon cannot be negative")

        self.salience_grid = int(salience_grid)
        self.view_grid = int(view_grid)
        self.max_mouse_delta = int(max_mouse_delta)
        self.scan_counts = int(scan_counts)
        self.max_scan_moves = int(max_scan_moves)
        self.max_target_candidates = int(max_target_candidates)
        self.max_center_moves = int(max_center_moves)
        self.max_reacquire_moves = int(max_reacquire_moves)
        self.max_capture_failures = int(max_capture_failures)
        self.max_moves = int(max_moves) if max_moves is not None else DEFAULT_MAX_MOVES
        self.dead_zone_px = float(dead_zone_px)
        self.min_target_confidence = float(min_target_confidence)
        self.min_views_before_selecting = int(min_views_before_selecting)
        self.min_salience = float(min_salience)
        self.peak_fraction = float(peak_fraction)
        self.weights = weights
        self.tie_epsilon = float(tie_epsilon)
        self.coarse_scale = int(coarse_scale)
        self.refine_window_px = refine_window_px
        self.thought_hook = thought_hook
        self.clock = clock
        self._rng = rng if rng is not None else random.Random()

        self.memory = ShortTermExperience(view_grid=self.view_grid, similar_threshold=DEFAULT_SIMILAR_THRESHOLD)
        self.progress = ProgressModel(epsilon=progress_epsilon)
        self.guard = RepetitionGuard(repeats=stuck_repeats, epsilon=progress_epsilon)
        self.cooldown = StrategyCooldown(failures=cooldown_failures)
        self.calibration = calibration if calibration is not None else MotionCalibration()
        self.controller = CenteringController(
            max_mouse_delta=self.max_mouse_delta,
            dead_zone_px=self.dead_zone_px,
            calibration=self.calibration,
        )
        self.reset()

    # ------------------------------------------------------------------ lifecycle

    def reset(self) -> None:
        """Return to the starting state, keeping the accumulated calibration.

        The calibration survives on purpose: it is a measurement of how the game
        responds to this mouse, not a belief about this run, and throwing it away
        between runs would make the agent relearn the same thing every time. It
        is still bounded - :class:`~autocraft.wake.centering.MotionCalibration`
        keeps only a fixed number of samples and re-medians them.
        """
        self.state = WakeState.STARTING
        self.step_index = -1
        self.started_at = self.clock()
        self.finished_at: float | None = None
        self.events: list[WakeEvent] = []
        self.memory.reset()
        self.progress.reset()
        self.guard.reset()
        self.cooldown.reset()
        self.controller.reset()

        self.frame_size: tuple[int, int] | None = None
        self.start_fingerprint: ViewFingerprint | None = None
        self.scan_index = 0
        self.scan_offset = (0, 0)
        self.scan_moves = 0
        self.candidate_count = 0
        self.target_changes = 0
        self.last_selected_descriptor: tuple[float, ...] | None = None
        self.centering_moves = 0
        self.reacquire_moves = 0
        self.capture_failures = 0
        self.moves = 0
        # Run-level tallies. The per-target counters above are reset whenever a
        # target is abandoned, which is right for a budget but wrong for a report:
        # a run that spent twenty corrections across three targets must not record
        # zero centring moves merely because the last target was given up on.
        self.total_centering_moves = 0
        self.run_progress: list[float] = []
        self.last_seen_offset: tuple[float, float] | None = None
        self.last_seen_distance: float | None = None
        self.target: _Target | None = None
        self.last_offset: tuple[float, float] | None = None
        self.last_relocation_confidence: float | None = None
        self.last_move: CenteringMove | None = None
        self.offset_history: list[float] = []
        self.recent_event: WakeEvent | None = None
        self.last_ruling: RepetitionVerdict | None = None
        self.stop_reason = ""
        self.thought_tick = 0
        self._last_salience: SalienceMap | None = None
        # Diagnostics for the salience pipeline. The scan accumulator describes
        # one frame and is replaced at the start of every scan; the relocation one
        # accumulates across the run, because the refusals it counts can only
        # happen once a target has been chosen.
        self._scan_diagnostics = SalienceDiagnostics()
        self._relocation_diagnostics = SalienceDiagnostics()
        self._last_view_key = ""

    # ------------------------------------------------------------------- protocol

    def decide(self, observation: Observation) -> Action:
        """Choose the next action from the observation just taken.

        Args:
            observation: The loop's latest observation, with its frame if capture
                succeeded.

        Returns:
            One :class:`~autocraft.agent.action.Action`. Terminal states return a
            stop action, which the executor treats as a no-op and the loop's guard
            never has to authorise.
        """
        self.step_index = int(observation.index)

        if self.state in TERMINAL_STATES:
            return Action.stop(self.stop_reason or self.state.value)

        self._note_geometry(observation)

        if observation.frame is None:
            self.capture_failures += 1
            if self.capture_failures >= self.max_capture_failures:
                return self._abort(
                    f"capture failed {self.capture_failures} times in a row",
                    state=WakeState.SAFE_STOP,
                )
            # Nothing to look at, so nothing to decide. Retrying costs no input.
            return Action.noop()

        self.capture_failures = 0

        if not observation.can_act:
            return self._abort(
                f"the game window is not available to act on ({observation.window.reason or 'not focused'})",
                state=WakeState.SAFE_STOP,
            )

        if self.moves >= self.max_moves:
            return self._finish(
                WakeState.FAILED,
                f"the movement budget of {self.max_moves} was exhausted",
            )

        frame = observation.frame
        self.frame_size = (int(frame.width), int(frame.height))

        if self.state is WakeState.STARTING:
            return self._start(frame)
        if self.state is WakeState.SCANNING:
            return self._scan(frame)
        if self.state is WakeState.SELECTING:
            return self._select(frame)
        if self.state is WakeState.CENTERING:
            return self._centre(frame)
        if self.state is WakeState.REACQUIRING:
            return self._reacquire(frame)
        # Unreachable while the terminal check above holds, but a state machine that
        # can silently fall through is a state machine that hangs.
        return self._abort(f"unexpected state {self.state.value}", state=WakeState.SAFE_STOP)

    # --------------------------------------------------------------------- states

    def _note_geometry(self, observation: Observation) -> None:
        """Record a change in the window's geometry, and rebaseline for it.

        The geometry recorded on each observation already carries the before and
        after numbers, so this turns them into an event and then acts on them.
        AutoCraft still does not resize the game window and does not restart the
        run: sizing the window is the operator's business, and the run's job is to
        say plainly what it saw. What it must not do is go on measuring in a scale
        that no longer exists.
        """
        previous = observation.geometry_changed_from
        if previous is None:
            return
        current = observation.geometry
        dropped = self._rebaseline_after_resize()
        self._emit(
            WakeEventKind.WINDOW_GEOMETRY_CHANGED,
            changed=list(observation.geometry_change),
            before=previous.to_dict(),
            after=current.to_dict(),
            rebaselined=bool(dropped),
            dropped=dropped,
        )

    def _rebaseline_after_resize(self) -> list[str]:
        """Discard everything that was measured in the old pixel scale.

        A resize does not change the game, but it changes what a pixel means.
        Every quantity cleared here is either a fingerprint taken over a grid that
        no longer exists, or a distance in pixels that no longer covers the same
        part of the scene. Carrying any of them across the change would leave the
        agent acting on a stale scale - and acting confidently, because a stale
        number does not look stale.

        The scan position is deliberately kept. The camera is pointing where it
        was pointing; only the units changed. Restarting the scan would re-send
        movements that have already been made, which is the dead repetition this
        milestone exists to avoid.

        The run's tallies are deliberately kept too. They are a record of what
        happened, and what happened is that the window changed size mid-run.

        Returns:
            The names of what was dropped, for the event that reports the change.
        """
        dropped: list[str] = []
        if self.memory.unique_views:
            dropped.append("view memory")
        self.memory.reset()
        self.progress.reset()
        self.guard.reset()
        self.cooldown.reset()
        self.controller.reset()
        if self.start_fingerprint is not None:
            dropped.append("start view")
        self.start_fingerprint = None
        if self.target is not None:
            dropped.append("chosen target")
        self.target = None
        if self.offset_history:
            dropped.append("offset history")
        self.offset_history = []
        for name in (
            "last_offset",
            "last_seen_offset",
            "last_seen_distance",
            "last_relocation_confidence",
            "last_move",
        ):
            if getattr(self, name) is not None:
                dropped.append(name.replace("_", " "))
            setattr(self, name, None)
        # The calibration is the one piece of state `reset()` keeps on purpose, and
        # the one place where a resize genuinely invalidates a measurement: it is
        # pixels per mouse count, and the pixels have changed size. Keeping it
        # would make every predicted correction wrong by the resize factor. An
        # adopted LOOK-001 ratio is cleared too - it was measured at the old size.
        if (
            self.calibration.samples
            or self.calibration.pixels_per_count_x is not None
            or self.calibration.pixels_per_count_y is not None
        ):
            dropped.append("motion calibration")
        self.calibration.reset()
        self._last_salience = None
        self.centering_moves = 0
        self.reacquire_moves = 0
        if self.state in (WakeState.SELECTING, WakeState.CENTERING, WakeState.REACQUIRING):
            # A target chosen under the old geometry is not a target any more, so
            # there is nothing left to centre on. Scanning again is the honest
            # response; it costs a step and invents nothing.
            self.state = WakeState.SCANNING
            dropped.append("target in progress")
        return dropped

    def _start(self, frame: Frame) -> Action:
        """Take in the starting view, then begin scanning in the same step."""
        self.start_fingerprint = ViewFingerprint.of(frame, grid=self.view_grid)
        self.memory.note_view(self.start_fingerprint, index=self.step_index, timestamp=self.clock())
        self._emit(
            WakeEventKind.WAKE_STARTED,
            index=self.step_index,
            window_width=int(frame.width),
            window_height=int(frame.height),
            grid=self.salience_grid,
            max_scan_moves=self.max_scan_moves,
            max_center_moves=self.max_center_moves,
            max_target_candidates=self.max_target_candidates,
        )
        self.state = WakeState.SCANNING
        # Falling straight through is deliberate: the frame in hand is a perfectly
        # good thing to look at, and spending a step to say hello would be theatre.
        return self._scan(frame)

    def _scan(self, frame: Frame) -> Action:
        """Look at the current view, then either select a target or look elsewhere."""
        self._observe_view(frame)

        candidates = self._find(frame)
        self._emit_scan_summary(frame, candidates)
        if candidates:
            for candidate in candidates:
                self.memory.note_candidate(candidate)
            self.candidate_count = max(self.candidate_count, len(candidates))
            self._emit(
                WakeEventKind.CANDIDATE_FOUND,
                index=self.step_index,
                count=len(candidates),
                best_salience=round(max(c.salience for c in candidates), 4),
                best_centre=[round(value, 1) for value in candidates[0].centre],
            )

        enough_views = self.memory.unique_views >= self.min_views_before_selecting
        has_candidates = bool(self.memory.strongest_candidates(CANDIDATE_LIMIT))
        out_of_scan_budget = self.scan_moves >= self.max_scan_moves

        if (enough_views and has_candidates) or out_of_scan_budget:
            if not has_candidates:
                return self._finish(
                    WakeState.FAILED,
                    f"nothing visually salient was found in {self.scan_moves} scan movement(s)",
                )
            self.state = WakeState.SELECTING
            return self._select(frame)

        if self.scan_moves >= self.max_scan_moves:
            return self._finish(
                WakeState.FAILED,
                f"the scan budget of {self.max_scan_moves} was exhausted before a target was chosen",
            )

        return self._scan_move()

    def _scan_move(self) -> Action:
        """Emit one movement of the fixed scan plan, and count it."""
        target_x, target_y = SCAN_OFFSETS[self.scan_index % len(SCAN_OFFSETS)]
        self.scan_index += 1
        wanted = (
            (target_x - self.scan_offset[0]) * self.scan_counts,
            (target_y - self.scan_offset[1]) * self.scan_counts,
        )
        dx = _clamp_int(wanted[0], self.max_mouse_delta)
        dy = _clamp_int(wanted[1], self.max_mouse_delta)

        if dx == 0 and dy == 0:
            # The plan asked for the view we are already in. Skipping ahead keeps
            # the run from spending a movement, and therefore a capture, on going
            # nowhere.
            return self._scan_move()

        # Record where the movement actually takes us, not where the plan hoped:
        # a clamped step is a real step, and pretending otherwise would put the
        # "return to the starting view" entry somewhere other than the start.
        self.scan_offset = (self.scan_offset[0] + dx, self.scan_offset[1] + dy)
        self.scan_moves += 1
        self.moves += 1
        self._emit(
            WakeEventKind.SCAN_MOVE,
            index=self.step_index,
            dx=dx,
            dy=dy,
            scan_move=self.scan_moves,
            offset=list(self.scan_offset),
            unique_views=self.memory.unique_views,
        )
        return Action.mouse_move(dx, dy)

    def _select(self, frame: Frame) -> Action:
        """Choose one candidate from everything the scan accumulated."""
        candidates = self.memory.strongest_candidates(CANDIDATE_LIMIT)
        if not candidates:
            return self._finish(WakeState.FAILED, "no candidates were available to select")

        fingerprint = ViewFingerprint.of(frame, grid=self.view_grid)
        # A cooled strategy is one that failed repeatedly from a view very much like
        # this one. Filtering here is what makes the cooldown do anything at all:
        # without it, the policy would simply pick the same candidate again.
        available = [
            candidate
            for candidate in candidates
            if self.cooldown.is_available(_approach_strategy(candidate, self.frame_size), fingerprint=fingerprint)
        ]
        # A cooldown may legitimately leave nothing: every strategy tried so far may
        # be resting. Retrying a strategy that has already failed three times from a
        # view that still looks the same is exactly the dead repetition this
        # milestone forbids, so stop and say so rather than quietly picking the
        # least-bad option.
        if not available:
            resting = ", ".join(sorted({_approach_strategy(c, self.frame_size) for c in candidates}))
            return self._finish(
                WakeState.FAILED,
                f"every approach strategy is resting after repeated failure ({resting})",
            )
        pool = available

        if self.last_selected_descriptor is not None:
            # Prefer a region genuinely different from the one just abandoned. The
            # test is appearance, not position: the memory merges sightings by
            # descriptor, so a box-based comparison would call the same region new
            # every time the camera moved.
            fresh = [
                candidate
                for candidate in pool
                if descriptor_similarity(self.last_selected_descriptor, candidate.descriptor)
                < self.memory.merge_threshold
            ]
            if fresh and len(fresh) != len(pool):
                self._emit(
                    WakeEventKind.STRATEGY_CHANGED,
                    reason="the previous target was abandoned; choosing another candidate",
                    candidates_left=len(fresh),
                )
            pool = fresh or pool

        chosen = select_candidate(pool, tie_epsilon=self.tie_epsilon, rng=self._rng)
        if chosen is None:
            return self._finish(WakeState.FAILED, "candidate selection returned nothing")

        if self.last_selected_descriptor is not None and (
            descriptor_similarity(self.last_selected_descriptor, chosen.descriptor) < self.memory.merge_threshold
        ):
            self.target_changes += 1
        self.last_selected_descriptor = chosen.descriptor

        patch = TargetPatch.of(frame, chosen.bbox, centre=chosen.centre, pad=_patch_pad(chosen))
        self.target = _Target(candidate=chosen, patch=patch, offset=(0.0, 0.0))
        self.last_offset = None
        self.last_move = None
        self.last_relocation_confidence = None
        self.controller.reset()
        self.centering_moves = 0
        self.reacquire_moves = 0
        self.offset_history = []

        self._emit(
            WakeEventKind.TARGET_SELECTED,
            index=self.step_index,
            bbox=list(chosen.bbox),
            centre=[round(value, 1) for value in chosen.centre],
            salience=round(chosen.salience, 4),
            novelty=round(chosen.novelty, 4),
            persistence=round(chosen.persistence, 4),
            confidence=round(chosen.confidence, 4),
            score=round(chosen.score, 4),
            seen=chosen.seen,
        )
        self.state = WakeState.CENTERING
        return self._centre(frame)

    def _centre(self, frame: Frame) -> Action:
        """Close the loop on the selected target, one correction at a time."""
        target = self.target
        if target is None:
            self.state = WakeState.SELECTING
            return self._select(frame)

        found = self._locate(frame, target)
        if found is None:
            return self._lose_target("the selected region could not be found in the new view", frame)

        relocation, confidence = found
        self.last_relocation_confidence = confidence
        if confidence < self.min_target_confidence:
            return self._lose_target(
                f"the selected region was only matched with confidence {confidence:.2f}, "
                f"below the {self.min_target_confidence:.2f} needed to aim at it",
                frame,
            )

        offset = _offset_from_centre(relocation.centre, self.frame_size)
        distance = math.hypot(offset[0], offset[1])

        previous_offset = self.last_offset
        previous_distance = None if previous_offset is None else math.hypot(*previous_offset)

        # Feed the observed displacement of the target into the calibration, using
        # the counts actually sent last step. This is how the mapping gets measured:
        # not from a probe that moves too far to see anything, but from the small
        # corrections this loop makes and then looks at.
        if self.last_move is not None and previous_offset is not None:
            self.controller.note_observation(
                shift_x=offset[0] - previous_offset[0],
                shift_y=offset[1] - previous_offset[1],
                confidence=confidence,
            )

        verdict = None
        if previous_distance is not None:
            verdict = self.progress.assess(before=previous_distance, after=distance)
            self.memory.note_progress(
                ProgressSample(
                    index=self.step_index,
                    strategy=self.controller.band,
                    distance=distance,
                    confidence=confidence,
                )
            )

        self.last_offset = offset
        self.offset_history.append(distance)
        self.run_progress.append(distance)
        self.last_seen_offset = offset
        self.last_seen_distance = distance
        if target.best_distance is None or distance < target.best_distance:
            target.best_distance = distance

        if distance <= self.dead_zone_px:
            return self._centred(distance, confidence)

        if self.centering_moves >= self.max_center_moves:
            return self._abandon_target(
                f"the centring budget of {self.max_center_moves} was spent with the target still "
                f"{distance:.1f} px from centre",
                frame,
            )

        move = self.controller.decide(
            offset_x=offset[0],
            offset_y=offset[1],
            frame_width=int(frame.width),
            frame_height=int(frame.height),
            confidence=confidence,
        )
        if move.is_noop:
            # Outside the dead zone but no correction wanted. The controller only
            # does this when its own band logic and this dead zone disagree, which
            # means the numbers are not trustworthy enough to keep acting on.
            return self._abandon_target(f"the centring controller declined to move: {move.reason}", frame)

        if move.overshoot:
            self._emit(
                WakeEventKind.OVERSHOOT_DETECTED,
                index=self.step_index,
                axis=list(move.reversed_axes),
                previous_offset=[round(value, 1) for value in (previous_offset or offset)],
                band=move.band,
                dx=move.dx,
                dy=move.dy,
                overshoots=self.controller.overshoots,
            )

        stuck = self._check_repetition(move, distance, verdict, frame)
        if stuck is not None:
            return stuck

        return self._emit_move(move, distance, confidence)

    def _centred(self, distance: float, confidence: float) -> Action:
        """The success path: the selected region is inside the dead zone."""
        self._emit(
            WakeEventKind.TARGET_CENTERED,
            index=self.step_index,
            distance=round(distance, 1),
            confidence=round(confidence, 4),
            moves=self.centering_moves,
            history=[round(value, 1) for value in self.offset_history],
        )
        return self._finish(
            WakeState.COMPLETE,
            f"the selected region is {distance:.1f} px from the centre of the view, "
            f"inside the {self.dead_zone_px:.1f} px dead zone",
            event=WakeEventKind.WAKE_COMPLETE,
        )

    def _lose_target(self, reason: str, frame: Frame) -> Action:
        """The target vanished from the view. Start recovering, bounded."""
        target = self.target
        if target is None:
            self.state = WakeState.SELECTING
            return self._select(frame)
        target.losses += 1
        self._emit(
            WakeEventKind.TARGET_LOST,
            index=self.step_index,
            reason=reason,
            losses=target.losses,
            confidence=None if self.last_relocation_confidence is None else round(self.last_relocation_confidence, 4),
            centre=[round(value, 1) for value in target.candidate.centre],
        )
        # The offset is no longer known, and a stale one would be fed to the
        # calibration on the next look as though the target had teleported.
        self.last_offset = None
        self.last_move = None
        if target.losses > self.max_reacquire_moves:
            return self._abandon_target(f"the selected region was lost {target.losses} times", frame)
        self.state = WakeState.REACQUIRING
        return self._reacquire(frame)

    def _reacquire(self, frame: Frame) -> Action:
        """Look for the lost target, turning back if it is still not there."""
        target = self.target
        if target is None:
            self.state = WakeState.SELECTING
            return self._select(frame)

        found = self._locate(frame, target, wide=True)
        if found is not None and found[1] >= self.min_target_confidence:
            relocation, confidence = found
            self.last_relocation_confidence = confidence
            self.last_offset = None
            self.last_move = None
            self.controller.reset()
            self._emit(
                WakeEventKind.TARGET_REACQUIRED,
                index=self.step_index,
                centre=[round(value, 1) for value in relocation.centre],
                confidence=round(confidence, 4),
                attempts=target.losses,
            )
            self.state = WakeState.CENTERING
            return self._centre(frame)

        if self.reacquire_moves >= self.max_reacquire_moves:
            return self._abandon_target(
                f"the selected region was not found again after {self.reacquire_moves} recovery movement(s)",
                frame,
            )

        return self._recovery_move(frame)

    def _recovery_move(self, frame: Frame) -> Action:
        """Turn part of the way back toward where the target was last seen.

        Deliberately the reverse of the last centring movement, at half size. If a
        correction overshot and pushed the target out of the view, the fix is to
        undo part of that correction, not to try something new - and half rather
        than all of it, because the target is not necessarily exactly where the last
        offset said. This is the milestone's RECOVERY strategy: reacquire, then
        abandon. It is not random and it is not a search pattern.
        """
        if self.last_move is not None and (self.last_move.dx or self.last_move.dy):
            dx = _clamp_int(-self.last_move.dx // 2, self.max_mouse_delta)
            dy = _clamp_int(-self.last_move.dy // 2, self.max_mouse_delta)
            reason = "turning part of the way back, the last correction lost the target"
        else:
            return self._abandon_target(
                "the selected region was lost and there was no recent correction to undo",
                frame,
            )

        self.reacquire_moves += 1
        self.moves += 1
        self.scan_offset = (self.scan_offset[0] + dx, self.scan_offset[1] + dy)
        self._emit(
            WakeEventKind.SCAN_MOVE,
            index=self.step_index,
            dx=dx,
            dy=dy,
            scan_move=self.scan_moves,
            offset=list(self.scan_offset),
            unique_views=self.memory.unique_views,
            recovery=True,
            reason=reason,
        )
        # The movement that lost the target is no longer the latest one, so it must
        # not be undone twice.
        self.last_move = None
        return Action.mouse_move(dx, dy)

    def _abandon_target(self, reason: str, frame: Frame | None = None) -> Action:
        """Give up on this candidate and let another one have the budget."""
        target = self.target
        if target is not None:
            view_key = self.memory.views.records[-1].key if len(self.memory.views) else ""
            fingerprint = self.memory.views.records[-1].fingerprint if len(self.memory.views) else None
            self.memory.mark_candidate_failed(target.candidate, reason, index=self.step_index, view_key=view_key)
            strategy = _approach_strategy(target.candidate, self.frame_size)
            cooldown = self.cooldown.record_failure(strategy, reason=reason, fingerprint=fingerprint)
            self._emit(
                WakeEventKind.STRATEGY_FAILED,
                index=self.step_index,
                strategy=strategy,
                reason=reason,
                failures=cooldown.failures,
                cooldown_active=not cooldown.released,
            )
            self._emit(
                WakeEventKind.STRATEGY_CHANGED,
                index=self.step_index,
                reason="changing target after a failed strategy",
                from_strategy=strategy,
            )
            # A different target is a different strategy, so the loop that was
            # detected - if any - has genuinely been escaped.
            self.guard.note_strategy_change()
        self.target = None
        self.last_offset = None
        self.last_move = None
        self.last_relocation_confidence = None
        self.controller.reset()
        self.centering_moves = 0
        self.reacquire_moves = 0
        self.offset_history = []
        attempted = len(self.memory.failed_attempts)
        if attempted >= self.max_target_candidates:
            return self._finish(
                WakeState.FAILED,
                f"every one of the {attempted} candidate(s) attempted was abandoned; last reason: {reason}",
            )
        self.state = WakeState.SELECTING
        if frame is None:
            return Action.noop()
        return self._select(frame)

    # ------------------------------------------------------------------ repetition

    def _check_repetition(
        self,
        move: CenteringMove,
        distance: float,
        verdict: Any,
        frame: Frame,
    ) -> Action | None:
        """Notice a loop and change strategy rather than repeating it again.

        Args:
            move: The correction about to be sent.
            distance: The target's current distance from the centre.
            verdict: The :class:`~autocraft.wake.progress.ProgressVerdict` from
                comparing this look with the previous one, if there was one.
            frame: The frame in hand, so abandoning can select immediately.

        Returns:
            An action when a loop was detected and something else must be tried, or
            ``None`` when the movement is fine to send.
        """
        fingerprint = self.memory.views.records[-1].fingerprint if len(self.memory.views) else None
        signature = RepetitionGuard.signature(
            view_key=fingerprint.key if fingerprint is not None else "",
            kind="mouse_move",
            dx=move.dx,
            dy=move.dy,
            strategy=move.strategy,
        )
        # The guard wants *signed progress*, not a distance: positive means the
        # correction helped. Passing the distance instead would make every step that
        # reduced it look like improvement and every step that increased it look like
        # a loop, which is backwards.
        progress = None if verdict is None or verdict.delta is None else float(verdict.delta)
        ruling = self.guard.observe(signature, progress=progress)
        self.last_ruling = ruling
        if not ruling.stuck:
            return None

        self._emit(
            WakeEventKind.STUCK_PATTERN_DETECTED,
            index=self.step_index,
            signature=ruling.signature,
            repeats=ruling.repeats,
            cycle=list(ruling.cycle),
            strategy=move.strategy,
            reason=ruling.reason,
            recent_progress=[None if value is None else round(value, 1) for value in ruling.recent_progress],
            distance=round(distance, 1),
        )

        # The response is to change strategy, never to add random movement. There are
        # exactly two strategies available here and both are deterministic: make a
        # smaller correction, or chase something else.
        position = BAND_ORDER.index(move.band) if move.band in BAND_ORDER else len(BAND_ORDER) - 1
        if position + 1 < len(BAND_ORDER):
            smaller = BAND_ORDER[position + 1]
            # Forcing the band downward is a real strategy change and is recorded as
            # one; the controller still refuses to enlarge a step afterwards, so this
            # cannot be undone on the next look.
            self.controller.band = smaller
            self.memory.set_strategy(f"centre_{move.direction}_{smaller}")
            self._emit(
                WakeEventKind.STRATEGY_CHANGED,
                index=self.step_index,
                reason=f"{ruling.repeats} identical unproductive corrections in a row",
                from_strategy=move.strategy,
                to_strategy=f"centre_{move.direction}_{smaller}",
                new_band=smaller,
                band_cap=BAND_COUNTS[smaller],
            )
            forced = self.controller.decide(
                offset_x=self.last_offset[0] if self.last_offset else 0.0,
                offset_y=self.last_offset[1] if self.last_offset else 0.0,
                frame_width=int(frame.width),
                frame_height=int(frame.height),
                confidence=self.last_relocation_confidence or 1.0,
            )
            if not forced.is_noop:
                return self._emit_move(forced, distance, self.last_relocation_confidence or 1.0)
            return self._abandon_target(
                f"the same correction was attempted {ruling.repeats} times and no smaller step was available",
                frame,
            )

        return self._abandon_target(
            f"the same correction was attempted {ruling.repeats} times with no measurable progress",
            frame,
        )

    # -------------------------------------------------------------------- helpers

    def _observe_view(self, frame: Frame) -> None:
        """Fingerprint the view and record whether it has been seen before."""
        fingerprint = ViewFingerprint.of(frame, grid=self.view_grid)
        observation = self.memory.note_view(fingerprint, index=self.step_index, timestamp=self.clock())
        # Kept so the salience summary can name the view it is describing. A
        # summary of a frame is only useful if you can tell which view it was.
        self._last_view_key = fingerprint.key
        self.progress.note_scan(new_view=not observation.revisited)
        if observation.revisited:
            self._emit(
                WakeEventKind.VIEW_REVISITED,
                index=self.step_index,
                similarity=round(observation.similarity, 4),
                times_seen=observation.record.seen,
                unique_views=self.memory.unique_views,
                revisited_views=self.memory.revisited_views,
                # The honest reading, spelled out where it is produced, because this
                # is the single easiest place in the whole milestone to overclaim.
                meaning="this looks very similar to a view seen recently, not 'this is the same place'",
            )
        else:
            self._emit(
                WakeEventKind.NEW_VIEW_OBSERVED,
                index=self.step_index,
                similarity=round(observation.similarity, 4),
                unique_views=self.memory.unique_views,
                revisited_views=self.memory.revisited_views,
            )
        self._emit(WakeEventKind.VIEW_CAPTURED, index=self.step_index, size=list(self.frame_size or ()))

    def _find(self, frame: Frame) -> list[CandidateTarget]:
        """Find and score the candidates in this frame.

        The diagnostics accumulator is replaced here, so it always describes the
        frame this call was given rather than a mixture of frames.
        """
        self._scan_diagnostics = SalienceDiagnostics()
        self._last_salience = salience_map(frame, grid=self.salience_grid)
        candidates = find_candidates(
            frame,
            grid=self.salience_grid,
            precomputed=self._last_salience,
            peak_fraction=self.peak_fraction,
            min_salience=self.min_salience,
            limit=CANDIDATE_LIMIT,
            diagnostics=self._scan_diagnostics,
        )
        if not candidates:
            return []
        return score_candidates(candidates, frame_size=self.frame_size or (frame.width, frame.height), weights=self.weights)

    def _emit_scan_summary(self, frame: Frame, candidates: list[CandidateTarget]) -> None:
        """Say why this frame yielded the candidates it did.

        A run that ends with "nothing visually salient was found" is telling the
        truth and explaining nothing: several different rules can empty the
        candidate list, and until this event existed they were indistinguishable
        in the record. Every number here is read back out of the pipeline that
        actually ran - the same thresholds, the same grouping, the same refusals -
        so it is evidence about this frame rather than a second opinion on it.
        Nothing here changes a decision; the event is written after the decision
        has already been made.
        """
        detail = self._scan_diagnostics.to_dict()
        # The four relocation counters come from the run-level accumulator, not
        # from this frame's: the rules they count live in the patch matcher, which
        # only runs once a target has been chosen. Reporting them here keeps every
        # refusal reason in one place, and `relocation_attempts` is what tells a
        # reader whether a zero means "nothing was refused" or "nothing was ever
        # attempted" - which, for a run that found no candidates at all, is the
        # difference between a mystery and an answer.
        relocation = self._relocation_diagnostics
        detail.update(
            view_id=self._last_view_key,
            frame_width=int(frame.width),
            frame_height=int(frame.height),
            surviving_candidates=len(candidates),
            candidates_in_memory=len(self.memory.strongest_candidates(CANDIDATE_LIMIT)),
            relocation_attempts=relocation.relocation_attempts,
            rejected_search_room=relocation.rejected_search_room,
            rejected_score=relocation.rejected_score,
            rejected_flat_patch=relocation.rejected_flat_patch,
        )
        self._emit(WakeEventKind.SALIENCE_SCAN_SUMMARY, **detail)

    def _locate(self, frame: Frame, target: _Target, *, wide: bool = False) -> tuple[Any, float] | None:
        """Find the selected region in a new frame, and say how sure we are.

        Two passes, because they fail differently and a target is only lost when
        both agree it is gone. The patch matcher is the precise one and gives a
        pixel-accurate centre, but it can lock onto the wrong thing when the scene
        has changed. The descriptor matcher is the robust one - it compares how a
        cell *looks relative to the rest of the frame*, so it survives the exposure
        shift that turning the camera produces - but it only resolves to a cell.
        The confidence reported is the lower of the two, so neither can talk the
        other into a match that is not there.
        """
        predicted = target.candidate.centre
        window = self.refine_window_px
        if wide:
            window = None

        self._relocation_diagnostics.relocation_attempts += 1
        found = locate(
            frame,
            target.patch,
            predicted=predicted,
            window=window,
            coarse_scale=self.coarse_scale,
            diagnostics=self._relocation_diagnostics,
        )
        if found is None:
            return None

        confidence = found.confidence
        cell = self._cell_at(found.centre, frame)
        if cell is not None:
            descriptor_confidence = self._descriptor_confidence(frame, target, cell)
            if descriptor_confidence is not None:
                confidence = min(confidence, descriptor_confidence)

        # Keep the patch current. A stale patch is how a target gets "lost" while
        # it is still plainly on screen: turn the camera and the appearance changes
        # enough that yesterday's crop no longer matches.
        target.patch = TargetPatch.of(frame, target.candidate.bbox, centre=found.centre, pad=_patch_pad(target.candidate))
        target.candidate = _recentre(target.candidate, found.centre)
        return found, float(confidence)

    def _cell_at(self, centre: tuple[float, float], frame: Frame) -> int | None:
        """Which salience-grid cell contains a pixel position."""
        if not self.frame_size:
            return None
        width, height = self.frame_size
        grid = self.salience_grid
        column = int(centre[0] * grid / max(1, width))
        row = int(centre[1] * grid / max(1, height))
        if not (0 <= row < grid and 0 <= column < grid):
            return None
        return row * grid + column

    def _descriptor_confidence(self, frame: Frame, target: _Target, cell: int) -> float | None:
        """Cross-check the patch matcher against the descriptor matcher.

        Deliberately one-sided: it can lower the confidence, never raise it. When
        the two matchers agree on which cell the target is in, the patch matcher's
        own score stands and this returns ``None``. When they disagree, the two are
        telling different stories about where the target is, and the honest thing to
        report is how well the descriptor actually matched where it went - which
        caps the confidence rather than being averaged with it.
        """
        found = relocate(
            target.candidate,
            frame,
            predicted=target.candidate.centre,
            search_radius=self._search_radius(),
            grid=self.salience_grid,
            min_confidence=0.0,
        )
        if found is None:
            return None
        if found.cell == cell:
            return None
        return float(found.confidence)

    def _search_radius(self) -> float:
        """A descriptor search radius comparable to a cell, not to a pixel.

        The descriptor matcher works on the salience grid, so a radius smaller than
        a cell would exclude the cell the target is actually in - the target would
        be reported lost by a radius that was meant to be helpful. Two cells is
        enough to cover a one-cell error and small enough that the search stays a
        local one.
        """
        if not self.frame_size:
            return 1.0
        width, height = self.frame_size
        grid = max(1, self.salience_grid)
        return 2.0 * max(width / grid, height / grid)

    def _emit_move(self, move: CenteringMove, distance: float, confidence: float) -> Action:
        """Send one centring correction and record what it was meant to achieve."""
        dx = _clamp_int(move.dx, self.max_mouse_delta)
        dy = _clamp_int(move.dy, self.max_mouse_delta)
        if dx == 0 and dy == 0:
            return self._abandon_target("the centring controller produced a movement of zero")

        self.centering_moves += 1
        self.total_centering_moves += 1
        self.moves += 1
        self.last_move = move
        self.memory.note_action(
            ActionRecord(
                index=self.step_index,
                strategy=move.strategy,
                kind="mouse_move",
                dx=dx,
                dy=dy,
                outcome="attempted",
                distance_before=distance,
            )
        )
        self.memory.set_strategy(move.strategy)
        self.memory.note_strategy_attempt()
        self._emit(
            WakeEventKind.CENTERING_PROGRESS,
            index=self.step_index,
            dx=dx,
            dy=dy,
            band=move.band,
            direction=move.direction,
            distance=round(distance, 1),
            expected_pixels=None if move.expected_pixels is None else round(move.expected_pixels, 1),
            confidence=round(confidence, 4),
            move=self.centering_moves,
            moves_left=max(0, self.max_center_moves - self.centering_moves),
            mapping_source=self.calibration.source,
            reason=move.reason,
        )
        return Action.mouse_move(dx, dy)

    def _abort(self, reason: str, *, state: WakeState = WakeState.SAFE_STOP) -> Action:
        """Stop for a reason outside the agent's control."""
        return self._finish(state, reason, event=WakeEventKind.WAKE_ABORTED)

    def _finish(
        self,
        state: WakeState,
        reason: str,
        *,
        event: WakeEventKind = WakeEventKind.WAKE_ABORTED,
    ) -> Action:
        """Enter a terminal state, record why, and return the stop action."""
        self.state = state
        self.stop_reason = reason
        if self.finished_at is None:
            self.finished_at = self.clock()
        self._emit(
            event,
            index=self.step_index,
            state=state.value,
            reason=reason,
            moves=self.moves,
            runtime_seconds=round(self.runtime, 3),
            final_offset=None if self.last_offset is None else [round(v, 1) for v in self.last_offset],
        )
        return Action.stop(reason)

    def _emit(self, kind: WakeEventKind, **detail: Any) -> None:
        """Record a structured event, and offer a display-only thought about it.

        The step index and timestamp always come from the run itself, so a call
        site that passes either in the detail has it dropped rather than being
        allowed to overwrite what the event actually happened at.
        """
        detail.pop("index", None)
        detail.pop("timestamp", None)
        event = WakeEvent.of(kind, index=self.step_index, timestamp=self.clock(), **detail)
        self.events.append(event)
        self.recent_event = event
        self._maybe_think(kind)

    def _maybe_think(self, kind: WakeEventKind) -> None:
        """Hand a template thought to the display hook, if there is one.

        Display only, in the strongest sense available: the hook is called after
        the action for this step has already been chosen, its return value is
        discarded, and nothing here reads it back. A thought cannot move the mouse
        because there is no path from this function to an
        :class:`~autocraft.agent.action.Action`.
        """
        if self.thought_hook is None:
            return
        self.thought_tick += 1
        if self.thought_tick % _THOUGHT_INTERVAL != 0:
            return
        mood = {
            WakeEventKind.NEW_VIEW_OBSERVED: "curious",
            WakeEventKind.CANDIDATE_FOUND: "curious",
            WakeEventKind.VIEW_REVISITED: "confused",
            WakeEventKind.OVERSHOOT_DETECTED: "frustrated",
            WakeEventKind.STUCK_PATTERN_DETECTED: "frustrated",
            WakeEventKind.TARGET_CENTERED: "satisfied",
            WakeEventKind.WAKE_COMPLETE: "reflective",
        }.get(kind)
        if mood is None:
            return
        try:
            self.thought_hook(mood, _THOUGHTS[mood])
        except Exception:  # noqa: BLE001 - a display hook must never break a run
            return

    # --------------------------------------------------------------------- output

    @property
    def runtime(self) -> float:
        """Seconds since the run started, frozen once it has finished."""
        end = self.finished_at if self.finished_at is not None else self.clock()
        return max(0.0, end - self.started_at)

    @property
    def finished(self) -> bool:
        """True once no further input will be sent."""
        return self.state in TERMINAL_STATES

    @property
    def successful(self) -> bool:
        """True only when the run ended because a target was centred."""
        return self.state is WakeState.COMPLETE

    @property
    def step_budget(self) -> int:
        """How many loop steps this policy can legitimately need.

        The policy bounds *movements*; the loop bounds *steps*, and they are not
        the same number. A step that cannot move - a failed capture, or the final
        step that returns the stop action - still consumes a step. The runner
        needs a step bound that cannot truncate a well-behaved run, so it is the
        movement budget plus the allowance the policy itself makes for a failed
        capture, plus one for the stop.

        This is a ceiling for the loop, not a decision. The policy still stops
        itself on its own terms; the loop bound exists so that a policy bug cannot
        leave the agent running.
        """
        return self.max_moves + self.max_capture_failures + 1

    def drain_events(self) -> list[WakeEvent]:
        """Return the events recorded since the last call, and forget them."""
        events, self.events = self.events, []
        return events

    def metrics(self) -> dict[str, Any]:
        """The milestone's metric set, with ``None`` where nothing was measurable."""
        guard = self.guard.to_dict()
        progress = self.progress.to_dict()
        return {
            "state": self.state.value,
            "scan_moves": self.scan_moves,
            "unique_views": self.memory.unique_views,
            "revisited_views": self.memory.revisited_views,
            "candidate_count": self.candidate_count,
            "target_changes": self.target_changes,
            "centering_moves": self.total_centering_moves,
            "overshoots": self.controller.overshoots,
            "failed_strategies": len(self.memory.failed_attempts),
            "stuck_patterns_detected": guard.get("stuck_patterns_detected", 0),
            "stuck_patterns_broken": guard.get("stuck_patterns_broken", 0),
            "final_target_offset": None
            if self.last_seen_offset is None
            else [round(v, 1) for v in self.last_seen_offset],
            "final_target_distance": self.last_seen_distance
            if self.last_seen_distance is None
            else round(self.last_seen_distance, 1),
            "runtime": round(self.runtime, 3),
            "successful_completion": self.successful,
            "dead_repetition_ratio": progress.get("dead_repetition_ratio"),
            "productive_repetition_ratio": progress.get("productive_repetition_ratio"),
            "moves_sent": self.moves,
            "stop_reason": self.stop_reason,
        }

    def status(self) -> dict[str, Any]:
        """The compact read-out the observer panel and the stream overlay show.

        Deliberately short. The milestone asks for something readable while
        streaming, not a debug wall, so this carries the nine things a viewer needs
        and nothing else.
        """
        target = self.target
        offset = self.last_offset
        return {
            "state": self.state.value,
            "target": None
            if target is None
            else {
                "centre": [round(v, 1) for v in target.candidate.centre],
                "bbox": list(target.candidate.bbox),
                "salience": round(target.candidate.salience, 3),
                "seen": target.candidate.seen,
            },
            "target_offset": None if offset is None else [round(v, 1) for v in offset],
            "confidence": None if self.last_relocation_confidence is None else round(self.last_relocation_confidence, 3),
            "progress": [round(v, 1) for v in self.offset_history[-6:]],
            "strategy": None if self.last_move is None else self.last_move.strategy,
            "repeat_guard": {
                "stuck": bool(self.last_ruling is not None and self.last_ruling.stuck),
                "repeats": 0 if self.last_ruling is None else self.last_ruling.repeats,
                "patterns_detected": self.guard.stuck_patterns_detected,
                "patterns_broken": self.guard.stuck_patterns_broken,
                "cooldowns": len(self.cooldown.active()),
            },
            "unique_views": self.memory.unique_views,
            "recent_event": None if self.recent_event is None else self.recent_event.message,
            "moves": self.moves,
            "max_moves": self.max_moves,
        }

    def summary(self) -> dict[str, Any]:
        """Everything a run record or a terminal summary needs."""
        return {
            "policy": self.name,
            "state": self.state.value,
            "finished": self.finished,
            "successful": self.successful,
            "stop_reason": self.stop_reason,
            "metrics": self.metrics(),
            "calibration": self.calibration.to_dict(),
            "memory": self.memory.to_dict(),
            "repetition": {
                "progress": self.progress.to_dict(),
                "guard": self.guard.to_dict(),
                "cooldown": self.cooldown.to_dict(),
            },
            "centering": self.controller.to_dict(),
        }

    def report(self) -> dict[str, Any]:
        """A flat snapshot for the run record and the observer panel.

        :meth:`metrics` is the milestone's own metric list and stays exactly that
        list; this is the wider set a record and a live panel need, gathered in
        one place so neither of them has to reach into the policy's attributes and
        guess at the shape.

        Every field that has not been measured yet is ``None``. That is the whole
        point: a panel showing a real zero and a panel showing "not measured" must
        not be able to look the same.
        """
        metrics = self.metrics()
        status = self.status()
        target = status.get("target") or {}
        guard = status.get("repeat_guard") or {}
        mapping = self.calibration.to_dict()
        metrics.update(
            {
                "max_moves": self.max_moves,
                # The run's own distance series, not the current target's. The
                # panel shows the current target (status()["progress"]); the
                # record wants the shape of the whole run.
                "progress": [round(v, 1) for v in self.run_progress[-24:]],
                "strategy": status.get("strategy") or "",
                "target_centre": target.get("centre"),
                "target_bbox": target.get("bbox"),
                "target_salience": target.get("salience"),
                "target_seen": target.get("seen", 0),
                "target_offset": status.get("target_offset"),
                "target_distance": metrics.get("final_target_distance"),
                "confidence": status.get("confidence"),
                "recent_event": status.get("recent_event") or "",
                "repeat_guard_stuck": bool(guard.get("stuck", False)),
                "repeat_guard_repeats": int(guard.get("repeats", 0)),
                "cooldowns_active": int(guard.get("cooldowns", 0)),
                "mapping_source": mapping.get("source", "unmeasured"),
                # A quality of 0.0 next to a real one reads as a measured
                # judgement. When there is no mapping there is nothing to judge,
                # so the record says "not measured" instead.
                "mapping_quality": mapping.get("quality") if mapping.get("measured") else None,
                "pixels_per_delta_x": mapping.get("pixels_per_count_x"),
                "pixels_per_delta_y": mapping.get("pixels_per_count_y"),
            }
        )
        return metrics


# ------------------------------------------------------------------- pure helpers


def _clamp_int(value: int | float, limit: int) -> int:
    """Round to an int and clamp to ``[-limit, +limit]``.

    A real ``int`` comes out, never a ``numpy`` scalar or a ``bool``, because
    :func:`~autocraft.agent.action.validate_action` insists on one and rejects the
    others. ``bool`` is rejected on purpose upstream - ``True`` is ``1`` and that
    kind of accident is exactly what a mouse movement should not be made of.
    """
    limit = int(limit)
    number = int(round(float(value)))
    if number > limit:
        return limit
    if number < -limit:
        return -limit
    return number


def _offset_from_centre(centre: tuple[float, float], frame_size: tuple[int, int] | None) -> tuple[float, float]:
    """Where a point sits relative to the frame's centre, in pixels."""
    if frame_size is None:
        return (0.0, 0.0)
    width, height = frame_size
    return (float(centre[0]) - width / 2.0, float(centre[1]) - height / 2.0)


def _patch_pad(candidate: CandidateTarget) -> int:
    """How much surround to include when cropping a candidate as a template.

    A little, proportional to the region, so the matcher has some context to lock
    onto rather than only the region's interior - which for a uniform patch scores
    as nothing at all.
    """
    return max(2, int(round(0.15 * max(1, min(candidate.width, candidate.height)))))


def _recentre(candidate: CandidateTarget, centre: tuple[float, float]) -> CandidateTarget:
    """A copy of a candidate whose centre has been measured more precisely.

    The salience layer can only put a candidate's centre at a cell centroid, and
    the grid on a 3222-pixel-wide window is coarse enough that this is a long way
    off. Once the patch matcher has found the region exactly, the better number
    should be what the next search predicts from - otherwise every step re-searches
    from a position that is systematically a fraction of a cell away.

    The bounding box moves with the centre, keeping its size. Leaving it behind
    would matter: the next patch is cropped from the box, and a box that no longer
    contains the region crops the wrong thing.
    """
    x, y = float(centre[0]), float(centre[1])
    half_width = candidate.width / 2.0
    half_height = candidate.height / 2.0
    bbox = (
        int(round(x - half_width)),
        int(round(y - half_height)),
        int(round(x + half_width)),
        int(round(y + half_height)),
    )
    return replace(candidate, centre=(x, y), bbox=bbox)


def _approach_strategy(candidate: CandidateTarget, frame_size: tuple[int, int] | None) -> str:
    """The name of the strategy that would be used to approach a candidate.

    Used as the key for the strategy cooldown, so a candidate on the left and one
    on the right are different strategies and failing at one does not disable the
    other. Named from the direction of the first correction the candidate would
    need, which is what the cooldown is really about.
    """
    if frame_size is None:
        return "centre_still_coarse"
    width, height = frame_size
    offset_x = candidate.centre[0] - width / 2.0
    offset_y = candidate.centre[1] - height / 2.0
    distance = math.hypot(offset_x, offset_y)
    band = band_for_distance(distance, frame_width=width, frame_height=height)
    return strategy_name(direction_of(int(round(offset_x)), int(round(offset_y))), band)
