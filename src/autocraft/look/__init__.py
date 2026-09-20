"""LOOK-001: a bounded measurement of how the picture responds to a known input.

The milestone asks one narrow question and refuses to answer any wider one: if a
known relative mouse movement is injected, how much does the captured picture
change, where did it appear to move, and does the change come back when the
movement is reversed?

What this package is:

* :mod:`autocraft.look.metrics` - frame-difference scalars, a coarse block map, a
  phase-correlation displacement estimate, and the reversibility ratio. Pure
  arithmetic over two frames, with no access to input.
* :mod:`autocraft.look.record` - the experiment record: what was planned, what was
  measured, and which files were written. Never serialises pixels.
* :mod:`autocraft.look.runner` - the fixed sequence, executed against injected
  callables so it can be tested without touching real input.

What this package is deliberately not:

* It does not classify anything. There is no object detection, no segmentation,
  no model of any kind, trained or otherwise. The only estimate here is a
  closed-form phase correlation.
* It does not assume a conversion between mouse counts and pixels.
  :func:`~autocraft.look.metrics.pixels_per_delta` computes the ratio that was
  *observed*, and :data:`~autocraft.look.metrics.MAPPING_NOTE` says what that
  number is and is not.
* It does not decide whether a result is good. There is no threshold and no
  verdict; the operator reads the numbers.
* It does not inject input itself. The sequence calls the existing V0 actuator
  path through injected callables; there is no second input implementation, and a
  test asserts that this package never imports the control layer.

The honest name for what this measures is a *measured sensorimotor mapping* for
one bounded movement. It is not evidence that anything understands a camera.
"""

from __future__ import annotations

from .metrics import (
    DEFAULT_BLOCK_GRID,
    DEFAULT_CHANGED_THRESHOLD,
    MAPPING_NOTE,
    FrameDifference,
    LookError,
    ShiftEstimate,
    difference_image,
    difference_metrics,
    estimate_shift,
    luminance,
    pixels_per_delta,
    reversibility_note,
    reversibility_ratio,
    save_grayscale,
)
from .record import (
    ARTIFACT_DIFFERENCE_AB,
    ARTIFACT_DIFFERENCE_AC,
    ARTIFACT_FRAME_A,
    ARTIFACT_FRAME_B,
    ARTIFACT_FRAME_C,
    EXPERIMENT_NAME,
    LIMITATION_NOTES,
    TRIAL_COMPLETED,
    TRIAL_FAILED,
    TRIAL_INTERRUPTED,
    LookRecorder,
    LookResult,
    LookTrialResult,
    TrialSpec,
)
from .runner import (
    EVENT_ERROR,
    EVENT_INFO,
    EVENT_OBSERVE,
    EVENT_SAFETY,
    FocusCheck,
    LookRunner,
    MoveOutcome,
    capture_failure_reason,
)

__all__ = [
    "ARTIFACT_DIFFERENCE_AB",
    "ARTIFACT_DIFFERENCE_AC",
    "ARTIFACT_FRAME_A",
    "ARTIFACT_FRAME_B",
    "ARTIFACT_FRAME_C",
    "DEFAULT_BLOCK_GRID",
    "DEFAULT_CHANGED_THRESHOLD",
    "EVENT_ERROR",
    "EVENT_INFO",
    "EVENT_OBSERVE",
    "EVENT_SAFETY",
    "EXPERIMENT_NAME",
    "LIMITATION_NOTES",
    "MAPPING_NOTE",
    "TRIAL_COMPLETED",
    "TRIAL_FAILED",
    "TRIAL_INTERRUPTED",
    "FocusCheck",
    "FrameDifference",
    "LookError",
    "LookRecorder",
    "LookResult",
    "LookRunner",
    "LookTrialResult",
    "MoveOutcome",
    "ShiftEstimate",
    "TrialSpec",
    "capture_failure_reason",
    "difference_image",
    "difference_metrics",
    "estimate_shift",
    "luminance",
    "pixels_per_delta",
    "reversibility_note",
    "reversibility_ratio",
    "save_grayscale",
]
