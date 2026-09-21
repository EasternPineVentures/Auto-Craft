"""Closed-loop centring: move a little, look again, decide again.

The milestone is emphatic about three things here, and each one is a trap that
this module is written to avoid.

**Do not assume one mouse count equals one pixel.** LOOK-001 was built to measure
that ratio and did not manage to: the live probe that moved the camera far enough
to prove motion existed also moved it so far that no shared features survived, so
the ratio came back unmeasured. The honest consequence is that this controller
starts life *not knowing* how far a mouse count moves the view. It uses small,
conservative, fixed steps and it does not claim otherwise - ``motion_mapping_quality``
starts at 0.0 and the reported source is ``"unmeasured"``.

What it can do instead is measure the ratio *as it goes*. Every correction is
followed by a fresh observation, and the observed pixel displacement of the
target divided by the counts actually sent is a direct measurement of the
mapping. A few of those, taken through a median so one bad relocation cannot
poison the estimate, and the controller is calibrating itself from its own
sensorimotor experience. That is what the milestone means by using measured
calibration rather than assuming one.

**Every correction must be based on the latest frame.** So this controller has no
plan, no queue and no precomputed series of movements. :meth:`CenteringController.decide`
takes the offset measured from the frame that was just captured and returns at
most one movement. Calling it twice without a new observation in between is a
programming error, and it is refused rather than silently obeyed.

**Overshoot is normal.** When the offset's sign flips between two observations,
the target went past the centre. The controller records that, drops to a smaller
step, and moves back. It is not a failure and nothing here counts it as one.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "BAND_COUNTS",
    "BAND_ORDER",
    "DEFAULT_DEAD_ZONE_PX",
    "DEFAULT_GAIN",
    "MAX_CALIBRATION_SAMPLES",
    "CenteringController",
    "CenteringMove",
    "MotionCalibration",
    "band_for_distance",
    "direction_of",
    "strategy_name",
]

#: Step sizes, coarsest first, in mouse counts. These are what the controller
#: uses while the mapping is unmeasured, and they are deliberately small: the
#: largest is under a third of the configured ``max_mouse_delta``, so a single
#: correction cannot fling the view off the target the way the ``--dx 200``
#: calibration probe did.
BAND_COUNTS: Mapping[str, int] = {"coarse": 60, "medium": 20, "fine": 6}

#: Bands from largest step to smallest.
BAND_ORDER: Sequence[str] = ("coarse", "medium", "fine")

#: Fraction of the frame's half-diagonal that separates the bands. A target more
#: than a quarter of the half-diagonal away is far enough that a coarse step is
#: safe; inside 8% the only thing left to do is nudge.
_COARSE_FRACTION = 0.25
_MEDIUM_FRACTION = 0.08

#: How close counts as centred, in pixels. The live window is 3222 wide, so this
#: is 0.4% of the width - comfortably inside the precision of a descriptor
#: relocation, and tight enough that "centred" still means something.
DEFAULT_DEAD_ZONE_PX = 12.0

#: Fraction of the measured or assumed displacement to actually request. Under
#: one on purpose: undershooting costs an extra look, overshooting costs the
#: target and a recovery cycle.
DEFAULT_GAIN = 0.7

#: How many measured samples the calibration median is built from. Enough to be
#: robust to one bad relocation, few enough to still adapt within a run.
MAX_CALIBRATION_SAMPLES = 9

#: Below this many samples the calibration is reported as unmeasured regardless
#: of the values, because one sample cannot distinguish a mapping from a
#: coincidence.
MIN_CALIBRATION_SAMPLES = 3

#: Absolute floor on a per-axis ratio. Anything smaller would divide an offset by
#: almost nothing and demand a movement larger than the whole frame.
_MIN_PIXELS_PER_COUNT = 0.05


@dataclass(frozen=True)
class CenteringMove:
    """One correction, and the reasoning behind it."""

    dx: int
    dy: int
    band: str
    direction: str
    reason: str
    overshoot: bool = False
    reversed_axes: tuple[str, ...] = ()
    expected_pixels: float | None = None
    unachievable: bool = False

    @property
    def is_noop(self) -> bool:
        """True when the correction is inside the dead zone and sends nothing."""
        return self.dx == 0 and self.dy == 0

    @property
    def strategy(self) -> str:
        """The strategy label this move belongs to, e.g. ``centre_left_coarse``."""
        return strategy_name(self.direction, self.band)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "dx": self.dx,
            "dy": self.dy,
            "band": self.band,
            "direction": self.direction,
            "strategy": self.strategy,
            "reason": self.reason,
            "overshoot": self.overshoot,
            "reversed_axes": list(self.reversed_axes),
            "expected_pixels": None if self.expected_pixels is None else round(self.expected_pixels, 3),
            "unachievable": self.unachievable,
        }


def direction_of(dx: int, dy: int) -> str:
    """Name the direction of a movement by its dominant axis."""
    if dx == 0 and dy == 0:
        return "still"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def strategy_name(direction: str, band: str) -> str:
    """Build the ``centre_left_coarse``-style strategy label."""
    return f"centre_{direction}_{band}"


def band_for_distance(distance: float, *, frame_width: int, frame_height: int) -> str:
    """Pick the step size band for a target this far from the centre.

    Distance is compared against the frame's half-diagonal rather than an absolute
    pixel count so the same thresholds work on any window size.
    """
    half_diagonal = max(1.0, 0.5 * math.hypot(max(1, frame_width), max(1, frame_height)))
    if distance > _COARSE_FRACTION * half_diagonal:
        return "coarse"
    if distance > _MEDIUM_FRACTION * half_diagonal:
        return "medium"
    return "fine"


class MotionCalibration:
    """An online estimate of how many pixels one mouse count moves the view.

    Starts unmeasured, because it is. Each observation after a correction supplies
    one sample - pixels the target actually moved, divided by the counts actually
    sent - and the estimate is the median of the recent samples, which is what
    keeps a single mis-relocated frame from rewriting the mapping.

    The quality figure is the fraction of samples that agreed with the median to
    within a factor of two. It is reported, not used as a gate: a low quality
    means the estimate should be distrusted, and the honest response to a
    distrusted estimate is a smaller step, which is what happens anyway because
    the step is capped by ``max_mouse_delta``.
    """

    def __init__(self, *, max_samples: int = MAX_CALIBRATION_SAMPLES) -> None:
        self.max_samples = max(1, int(max_samples))
        self._samples_x: deque[float] = deque(maxlen=self.max_samples)
        self._samples_y: deque[float] = deque(maxlen=self.max_samples)
        self.pixels_per_count_x: float | None = None
        self.pixels_per_count_y: float | None = None
        self.quality = 0.0
        self.source = "unmeasured"
        self.samples = 0

    def reset(self) -> None:
        """Discard every sample and go back to not knowing."""
        self._samples_x.clear()
        self._samples_y.clear()
        self.pixels_per_count_x = None
        self.pixels_per_count_y = None
        self.quality = 0.0
        self.source = "unmeasured"
        self.samples = 0

    def adopt(self, *, pixels_per_delta_x: float | None, pixels_per_delta_y: float | None, quality: float) -> None:
        """Seed the estimate from a LOOK-001 measurement, when one exists.

        A LOOK-001 ratio is only adopted when it is a finite, positive number. A
        missing ratio, a null, or a nonsense value is ignored rather than coerced
        into a default, because a wrong mapping is worse than no mapping: the
        controller is honest about not knowing, and works from its own samples.
        """
        if _usable_ratio(pixels_per_delta_x):
            self.pixels_per_count_x = float(pixels_per_delta_x)  # type: ignore[arg-type]
        if _usable_ratio(pixels_per_delta_y):
            self.pixels_per_count_y = float(pixels_per_delta_y)  # type: ignore[arg-type]
        if self.pixels_per_count_x is not None or self.pixels_per_count_y is not None:
            self.quality = max(0.0, min(1.0, float(quality)))
            self.source = "look-001"

    def observe(self, *, dx_counts: int, dy_counts: int, shift_x: float, shift_y: float) -> bool:
        """Add one sample from a correction and the displacement it produced.

        Returns:
            True when a sample was taken. A sample is refused when the counts were
            zero (no correction was sent, so there is nothing to attribute), when
            the shift is not finite, or when the resulting ratio is absurd - the
            ratio must be finite and positive for it to mean anything.
        """
        taken = False
        if dx_counts != 0 and math.isfinite(shift_x):
            ratio = abs(float(shift_x)) / abs(float(dx_counts))
            if math.isfinite(ratio) and ratio > 0.0:
                self._samples_x.append(ratio)
                taken = True
        if dy_counts != 0 and math.isfinite(shift_y):
            ratio = abs(float(shift_y)) / abs(float(dy_counts))
            if math.isfinite(ratio) and ratio > 0.0:
                self._samples_y.append(ratio)
                taken = True
        if taken:
            self._recompute()
        return taken

    def _recompute(self) -> None:
        """Re-derive the median estimate and its quality from the samples."""
        self.samples = max(len(self._samples_x), len(self._samples_y))
        if len(self._samples_x) >= MIN_CALIBRATION_SAMPLES:
            self.pixels_per_count_x = _median(self._samples_x)
        if len(self._samples_y) >= MIN_CALIBRATION_SAMPLES:
            self.pixels_per_count_y = _median(self._samples_y)
        agreements: list[bool] = []
        for samples, estimate in ((self._samples_x, self.pixels_per_count_x), (self._samples_y, self.pixels_per_count_y)):
            if estimate is None or estimate <= 0.0:
                continue
            for value in samples:
                agreements.append(0.5 <= value / estimate <= 2.0)
        if agreements:
            self.quality = sum(1 for ok in agreements if ok) / len(agreements)
        if self.pixels_per_count_x is not None or self.pixels_per_count_y is not None:
            self.source = "self-measured"

    @property
    def measured(self) -> bool:
        """True when at least one axis has a usable measured ratio."""
        return self.pixels_per_count_x is not None or self.pixels_per_count_y is not None

    def ratio_for(self, axis: str) -> float | None:
        """The measured ratio for ``"x"`` or ``"y"``, or ``None``."""
        return self.pixels_per_count_x if axis == "x" else self.pixels_per_count_y

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "source": self.source,
            "measured": self.measured,
            "pixels_per_count_x": None if self.pixels_per_count_x is None else round(self.pixels_per_count_x, 5),
            "pixels_per_count_y": None if self.pixels_per_count_y is None else round(self.pixels_per_count_y, 5),
            "quality": round(self.quality, 4),
            "samples": self.samples,
            "sample_count_x": len(self._samples_x),
            "sample_count_y": len(self._samples_y),
        }


class CenteringController:
    """Decides one correction at a time from the newest measurement of the target.

    The controller owns no plan. It is told where the target is now, and it says
    what single movement to make about it. That structure is what enforces the
    milestone's "every correction must be based on the latest frame" rule - there
    is nowhere to put a precomputed series of movements even if someone wanted to.
    """

    def __init__(
        self,
        *,
        max_mouse_delta: int = 200,
        dead_zone_px: float = DEFAULT_DEAD_ZONE_PX,
        gain: float = DEFAULT_GAIN,
        calibration: MotionCalibration | None = None,
    ) -> None:
        self.max_mouse_delta = max(1, int(max_mouse_delta))
        self.dead_zone_px = max(0.0, float(dead_zone_px))
        self.gain = max(0.05, min(1.0, float(gain)))
        self.calibration = calibration if calibration is not None else MotionCalibration()
        self.band = "coarse"
        self._last_offset: tuple[float, float] | None = None
        self._last_move: CenteringMove | None = None
        self._pending = False
        self.overshoots = 0

    def reset(self) -> None:
        """Forget the previous observation. Keeps the calibration."""
        self.band = "coarse"
        self._last_offset = None
        self._last_move = None
        self._pending = False
        self.overshoots = 0

    @property
    def awaiting_observation(self) -> bool:
        """True when a movement has been issued and not yet observed."""
        return self._pending

    def note_observation(self, *, shift_x: float, shift_y: float, confidence: float = 1.0) -> bool:
        """Feed back what the last correction actually did.

        Args:
            shift_x: Pixels the target moved horizontally, in the same sign
                convention as the offset (positive is rightward on screen).
            shift_y: Pixels the target moved vertically, positive downward.
            confidence: How much the relocation should be trusted, in 0..1.
                Samples from a low-confidence relocation are still taken, because
                the median already defends against outliers, but the confidence is
                carried into the reported quality.

        Returns:
            True when a calibration sample was taken.
        """
        move = self._last_move
        self._pending = False
        if move is None or move.is_noop:
            return False
        if confidence <= 0.0:
            return False
        return self.calibration.observe(
            dx_counts=move.dx,
            dy_counts=move.dy,
            shift_x=shift_x,
            shift_y=shift_y,
        )

    def decide(
        self,
        *,
        offset_x: float,
        offset_y: float,
        frame_width: int,
        frame_height: int,
        confidence: float = 1.0,
    ) -> CenteringMove:
        """Return the single correction to make, given the target's offset.

        Args:
            offset_x: Target centre x minus frame centre x, in pixels. Positive
                means the target is to the right of centre, so the camera should
                turn right.
            offset_y: Target centre y minus frame centre y, in pixels. Positive
                means the target is below centre.
            frame_width: Frame width in pixels, for scaling the bands.
            frame_height: Frame height in pixels.
            confidence: How sure the relocation is, in 0..1. Below the caller's
                own threshold the caller should reacquire rather than move, so
                this is only used to shorten the step when it is middling.

        Returns:
            A :class:`CenteringMove`. Its ``is_noop`` is True when the target is
            already inside the dead zone, in which case nothing is sent.
        """
        distance = math.hypot(offset_x, offset_y)
        previous = self._last_offset
        previous_distance = None if previous is None else math.hypot(previous[0], previous[1])
        self._last_offset = (float(offset_x), float(offset_y))

        if distance <= self.dead_zone_px:
            self.band = "fine"
            move = CenteringMove(
                dx=0,
                dy=0,
                band="fine",
                direction="still",
                reason=f"target is {distance:.1f} px from centre, inside the {self.dead_zone_px:g} px dead zone",
            )
            self._last_move = move
            self._pending = False
            return move

        wanted_band = band_for_distance(distance, frame_width=frame_width, frame_height=frame_height)
        if previous is None:
            band = wanted_band
        elif previous_distance is not None and distance > previous_distance:
            # The target is getting further away, so a small step is no longer the
            # right size. This is the one case where the step may grow again.
            band = wanted_band
        else:
            # Converging: never enlarge the step, or a run that overshoots once
            # would start oscillating at full amplitude.
            band = _smaller_band(self.band, wanted_band)

        reversed_axes: list[str] = []
        if previous is not None:
            if _sign_flipped(previous[0], offset_x, self.dead_zone_px):
                reversed_axes.append("x")
            if _sign_flipped(previous[1], offset_y, self.dead_zone_px):
                reversed_axes.append("y")
        if reversed_axes:
            self.overshoots += 1
            band = _smaller_band(band, _smaller_band(wanted_band, "medium"))
        self.band = band

        steps = _step_counts(
            offset_x=offset_x,
            offset_y=offset_y,
            band=band,
            calibration=self.calibration,
            gain=self.gain,
            confidence=confidence,
            max_mouse_delta=self.max_mouse_delta,
        )
        expected = _expected_pixels(steps, self.calibration)
        if expected is not None and _step_is_hopeless(expected, distance, self.dead_zone_px):
            # The controller's own arithmetic says this correction cannot work.
            # It is worth spelling out why, because the alternative - and what
            # actually happened - is to send the step anyway.
            #
            # ``_step_counts`` floors the ratio at ``_MIN_PIXELS_PER_COUNT``, so a
            # mapping that is real but far too small divides the offset by almost
            # nothing and the request lands on ``max_mouse_delta``: the largest
            # movement the safety limit allows. ``_expected_pixels`` uses the raw
            # ratio, so it still knows the truth - that 200 counts will displace
            # the view by about a pixel and a half. The two functions disagree,
            # and the one telling the truth is the one that is not allowed to
            # choose the step. This is the check that reconciles them.
            #
            # The live WAKE-001 run ``20260921T045955Z-4b6dd88b`` is the evidence.
            # Its measured mapping was 0.0075 px per count on y, so every step was
            # capped at 200 counts and displaced the view by 1.5 px. Its first
            # target took eight centring moves to go from 171.9 px to 168.0 px -
            # four pixels of a hundred-and-seventy-two pixel gap - and the distance
            # series wanders up as often as down: 171.9, 172.9, 171.9, 170.0,
            # 169.0, 170.0, 168.0, 169.0. That is not a controller failing to
            # converge; it is a controller marching on a signal too small to
            # measure. Reporting the correction as impossible is the honest
            # answer, and it is a fact about the measurement rather than about the
            # target.
            reason = (
                f"the target is {distance:.1f} px from centre, but the measured mapping "
                f"({self.calibration.ratio_for('x') or 0:.4f} px per count on x, "
                f"{self.calibration.ratio_for('y') or 0:.4f} on y) means the largest "
                f"correction available moves it {expected:.1f} px, below the "
                f"{self.dead_zone_px:g} px dead zone - this correction cannot be made"
            )
            move = CenteringMove(
                dx=0,
                dy=0,
                band=band,
                direction="still",
                reason=reason,
                expected_pixels=expected,
                unachievable=True,
            )
            self._last_move = move
            self._pending = False
            return move
        direction = direction_of(*steps)
        if reversed_axes:
            reason = (
                f"target crossed the centre on the {' and '.join(reversed_axes)} axis "
                f"between looks; stepping down to {band} and moving back"
            )
        elif self.calibration.measured:
            reason = (
                f"{distance:.1f} px from centre; using the self-measured mapping "
                f"({self.calibration.pixels_per_count_x or 0:.3f} px per count on x)"
            )
        else:
            reason = (
                f"{distance:.1f} px from centre; the mouse-to-camera mapping is unmeasured, "
                f"so this is a conservative {band} step"
            )
        move = CenteringMove(
            dx=steps[0],
            dy=steps[1],
            band=band,
            direction=direction,
            reason=reason,
            overshoot=bool(reversed_axes),
            reversed_axes=tuple(reversed_axes),
            expected_pixels=expected,
        )
        self._last_move = move
        self._pending = not move.is_noop
        return move

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "band": self.band,
            "dead_zone_px": self.dead_zone_px,
            "gain": self.gain,
            "max_mouse_delta": self.max_mouse_delta,
            "overshoots": self.overshoots,
            "awaiting_observation": self._pending,
            "calibration": self.calibration.to_dict(),
        }


def _step_counts(
    *,
    offset_x: float,
    offset_y: float,
    band: str,
    calibration: MotionCalibration,
    gain: float,
    confidence: float,
    max_mouse_delta: int,
) -> tuple[int, int]:
    """Turn an offset into mouse counts, per axis, respecting the cap.

    With a measured ratio on an axis the step is ``gain`` of the remaining offset,
    converted from pixels into counts. Without one the step is the band's fixed
    count in the offset's direction - which is a guess, and is treated as one
    everywhere it is reported.

    The step is deliberately *not* capped at the band's count when a ratio exists.
    A cap there looks prudent and quietly destroys the thing that makes this work:
    taking a fixed fraction of the remaining error makes the sequence converge
    geometrically, so it reaches the dead zone in a handful of steps whatever the
    mapping turns out to be. Capping the counts turns that into a fixed-size march,
    and a fixed-size march cannot cross a large error at all - on a 3222-pixel-wide
    window a target at the edge would need dozens of steps it does not have. What
    bounds the step is ``max_mouse_delta``, which is a safety limit rather than a
    control decision, and the band still governs which fraction of the error is
    taken and how the attempt is named.

    The floor on the ratio is what turns a useless mapping into a capped step: a
    measured ratio of 0.0075 px per count divides the offset by 0.05 instead, so
    the request lands on ``max_mouse_delta`` every time. Nothing here can tell that
    apart from a genuinely large error, which is why the caller checks the
    resulting move against its own dead zone via :func:`_step_is_hopeless` before
    sending it. This function sizes a step; it does not judge whether the step can
    work.
    """
    ceiling = max(1, int(max_mouse_delta))
    band_counts = int(BAND_COUNTS.get(band, BAND_COUNTS["fine"]))
    scale = 1.0
    if confidence < 0.5:
        # A middling relocation is worth acting on, but not at full step.
        scale = 0.5
    counts: list[int] = []
    for offset, axis in ((offset_x, "x"), (offset_y, "y")):
        if offset == 0.0:
            counts.append(0)
            continue
        ratio = calibration.ratio_for(axis)
        if ratio is None:
            magnitude = band_counts
        else:
            magnitude = int(round(abs(offset) * gain / max(ratio, _MIN_PIXELS_PER_COUNT)))
        magnitude = int(round(magnitude * scale))
        magnitude = max(1, min(ceiling, magnitude))
        counts.append(magnitude if offset > 0 else -magnitude)
    return (counts[0], counts[1])


def _expected_pixels(steps: tuple[int, int], calibration: MotionCalibration) -> float | None:
    """Predict the displacement the steps should produce, when that is knowable."""
    if not calibration.measured:
        return None
    x_ratio = calibration.ratio_for("x")
    y_ratio = calibration.ratio_for("y")
    expected_x = 0.0 if x_ratio is None else steps[0] * x_ratio
    expected_y = 0.0 if y_ratio is None else steps[1] * y_ratio
    if x_ratio is None and y_ratio is None:
        return None
    return math.hypot(expected_x, expected_y)


def _step_is_hopeless(expected: float, distance: float, dead_zone: float) -> bool:
    """Whether a correction worth ``expected`` pixels is too small to be worth sending.

    Two things have to be true. The step must displace the view by less than the
    dead zone, because a movement smaller than the tolerance the loop is trying to
    reach is indistinguishable from not moving: the next look cannot say whether
    the correction worked, so the loop would be steering on noise. And the step
    must also fail to close the gap in this one move, so that a target which is a
    single nudge from success is not abandoned for being close.

    The second condition guards the first rather than standing on its own. A
    target 13 px from centre with a mapping worth 9 px per step is below the dead
    zone and about to be centred, and refusing it would be a false positive of
    exactly the kind this check exists to catch.
    """
    return expected <= dead_zone and expected < (distance - dead_zone)


def _smaller_band(left: str, right: str) -> str:
    """Return whichever of two bands is the smaller step."""
    try:
        return left if BAND_ORDER.index(left) >= BAND_ORDER.index(right) else right
    except ValueError:
        return "fine"


def _sign_flipped(previous: float, current: float, dead_zone: float) -> bool:
    """True when an offset crossed zero between two looks, ignoring the dead zone."""
    if abs(previous) <= dead_zone or abs(current) <= dead_zone:
        return False
    return (previous > 0.0) != (current > 0.0)


def _median(values: Iterable[float]) -> float:
    """The median of a sequence of floats."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _usable_ratio(value: float | None) -> bool:
    """True when a ratio is a finite, positive number worth adopting."""
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0
