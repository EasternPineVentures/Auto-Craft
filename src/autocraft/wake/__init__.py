"""WAKE-001: look around, notice something, turn toward it, centre it, stop.

The milestone asks for one bounded piece of behaviour and refuses every wider one.
On wake, the agent should capture the world, deliberately inspect more than one
direction, notice a visually interesting region *without knowing what it is*, turn
toward it, and bring it to the centre of the view - then stop and say what
happened.

What this package is:

* :mod:`autocraft.wake.events` - the structured event vocabulary. Events are data,
  and the plain-language line a stream shows is generated from the event kind so
  the prose cannot drift away from the numbers.
* :mod:`autocraft.wake.salience` - PERCEPTION. Where is there something worth
  looking at? Five deterministic cues over the existing VISION-001 cell features,
  a candidate per connected region, and a pixel-accurate matcher for finding a
  region again in a later frame.
* :mod:`autocraft.wake.memory` - bounded working memory: which views have I looked
  at, which candidates have I already considered, what did I try and did it help.
* :mod:`autocraft.wake.progress` - BEHAVIOUR. Did that action improve things, and
  am I going round in circles? Verdicts carry a signed magnitude, never a bare
  success/failure boolean.
* :mod:`autocraft.wake.centering` - MOTOR. Closed-loop correction: one movement at
  a time, each based on the latest frame, with overshoot treated as normal.
* :mod:`autocraft.wake.policy` - the state machine that ties the three layers
  together behind the existing :class:`~autocraft.agent.decision.DecisionPolicy`
  seam. It is not a second agent architecture.

What this package is deliberately not:

* It does not recognise anything. There is no object detector, no classifier, no
  trained vision model and no model API. A "target" here means a *visually salient
  region* and nothing more; the type is named
  :class:`~autocraft.wake.salience.CandidateTarget` so that no reader mistakes it
  for a recognised object.
* It does not know the game. Nothing here reads privileged state, and nothing
  names a tree, a block or a mob. A test asserts that this package never imports
  the control or evaluator layers.
* It does not plan a route or a camera path. Every correction is decided from the
  frame in hand and at most one movement is returned per decision, which is what
  makes "re-observe after every movement" a structural fact rather than a promise.
* It does not use randomness to look alive. The one place randomness is permitted
  is a tie between two candidates whose scores are within an explicit epsilon, and
  it is off by default.
* It does not write long-term memory. The working memory is a fixed number of
  deques and is deliberately forgetful.

The honest name for what this does is *bounded visual curiosity*: it can find a
region that stands out from its surroundings and bring it to the middle of the
screen. It is not evidence that anything understands what it is looking at.
"""

from __future__ import annotations

from .centering import (
    BAND_COUNTS,
    BAND_ORDER,
    DEFAULT_DEAD_ZONE_PX,
    DEFAULT_GAIN,
    CenteringController,
    CenteringMove,
    MotionCalibration,
    band_for_distance,
    direction_of,
    strategy_name,
)
from .events import (
    STREAM_SUMMARY,
    WakeEvent,
    WakeEventKind,
    stream_summary,
)
from .memory import (
    DEFAULT_MERGE_THRESHOLD,
    DEFAULT_SIMILAR_THRESHOLD,
    DEFAULT_VIEW_GRID,
    ActionRecord,
    FailedAttempt,
    ProgressSample,
    ShortTermExperience,
    ViewFingerprint,
    ViewMemory,
    ViewObservation,
    ViewRecord,
    describe_actions,
    summarise_memory,
)
from .policy import (
    DEFAULT_MAX_CAPTURE_FAILURES,
    DEFAULT_MAX_CENTER_MOVES,
    DEFAULT_MAX_MOVES,
    DEFAULT_MAX_REACQUIRE_MOVES,
    DEFAULT_MAX_SCAN_MOVES,
    DEFAULT_MAX_TARGET_CANDIDATES,
    DEFAULT_MIN_TARGET_CONFIDENCE,
    DEFAULT_MIN_VIEWS_BEFORE_SELECTING,
    DEFAULT_SCAN_COUNTS,
    SCAN_OFFSETS,
    TERMINAL_STATES,
    WakeDecisionPolicy,
    WakeState,
)
from .progress import (
    DEFAULT_COOLDOWN_FAILURES,
    DEFAULT_PROGRESS_EPSILON,
    DEFAULT_STUCK_REPEATS,
    Cooldown,
    ProgressModel,
    ProgressVerdict,
    RepetitionGuard,
    RepetitionVerdict,
    StrategyCooldown,
    summarise_repetition,
)
from .record import (
    EXPERIMENT_NAME,
    LIMITATION_NOTES,
    RESULT_FILENAME,
    STATUS_ABORTED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    WakeRecorder,
    WakeResult,
)
from .runner import (
    DEFAULT_MAX_SECONDS,
    WakeRunner,
    status_for_state,
)
from .salience import (
    CANDIDATE_LIMIT,
    DEFAULT_MATCH_CONFIDENCE,
    DEFAULT_MIN_SALIENCE,
    DEFAULT_PEAK_FRACTION,
    DEFAULT_REFINE_SCORE,
    MATCH_SIGMA,
    SALIENCE_WEIGHTS,
    CandidateTarget,
    Relocation,
    SalienceError,
    SalienceMap,
    SelectionWeights,
    TargetPatch,
    candidate_from_cells,
    descriptor_similarity,
    find_candidates,
    locate,
    refine,
    relocate,
    salience_map,
    score_candidates,
    select_candidate,
)

__all__ = [
    "BAND_COUNTS",
    "BAND_ORDER",
    "CANDIDATE_LIMIT",
    "DEFAULT_COOLDOWN_FAILURES",
    "DEFAULT_DEAD_ZONE_PX",
    "DEFAULT_GAIN",
    "DEFAULT_MATCH_CONFIDENCE",
    "DEFAULT_MAX_CAPTURE_FAILURES",
    "DEFAULT_MAX_CENTER_MOVES",
    "DEFAULT_MAX_MOVES",
    "DEFAULT_MAX_REACQUIRE_MOVES",
    "DEFAULT_MAX_SCAN_MOVES",
    "DEFAULT_MAX_SECONDS",
    "DEFAULT_MAX_TARGET_CANDIDATES",
    "DEFAULT_MERGE_THRESHOLD",
    "DEFAULT_MIN_SALIENCE",
    "DEFAULT_MIN_TARGET_CONFIDENCE",
    "DEFAULT_MIN_VIEWS_BEFORE_SELECTING",
    "DEFAULT_PEAK_FRACTION",
    "DEFAULT_PROGRESS_EPSILON",
    "DEFAULT_REFINE_SCORE",
    "DEFAULT_SCAN_COUNTS",
    "DEFAULT_SIMILAR_THRESHOLD",
    "DEFAULT_STUCK_REPEATS",
    "DEFAULT_VIEW_GRID",
    "EXPERIMENT_NAME",
    "LIMITATION_NOTES",
    "MATCH_SIGMA",
    "RESULT_FILENAME",
    "SALIENCE_WEIGHTS",
    "SCAN_OFFSETS",
    "STATUS_ABORTED",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_RUNNING",
    "STREAM_SUMMARY",
    "TERMINAL_STATES",
    "ActionRecord",
    "CandidateTarget",
    "CenteringController",
    "CenteringMove",
    "Cooldown",
    "FailedAttempt",
    "MotionCalibration",
    "ProgressModel",
    "ProgressSample",
    "ProgressVerdict",
    "Relocation",
    "RepetitionGuard",
    "RepetitionVerdict",
    "SalienceError",
    "SalienceMap",
    "SelectionWeights",
    "ShortTermExperience",
    "StrategyCooldown",
    "TargetPatch",
    "ViewFingerprint",
    "ViewMemory",
    "ViewObservation",
    "ViewRecord",
    "WakeDecisionPolicy",
    "WakeEvent",
    "WakeEventKind",
    "WakeRecorder",
    "WakeResult",
    "WakeRunner",
    "WakeState",
    "band_for_distance",
    "candidate_from_cells",
    "describe_actions",
    "descriptor_similarity",
    "direction_of",
    "find_candidates",
    "locate",
    "refine",
    "relocate",
    "salience_map",
    "score_candidates",
    "select_candidate",
    "status_for_state",
    "strategy_name",
    "stream_summary",
    "summarise_memory",
    "summarise_repetition",
]
