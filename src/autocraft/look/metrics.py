"""Frame-difference metrics and a coarse displacement estimate.

LOOK-001 asks one narrow question: if a known mouse delta is injected, how much
does the picture change, and does the change come back when the delta is
reversed? Answering it needs three measurements, and all three live here:

* :func:`difference_metrics` - how far apart two frames are, per pixel, as three
  scalars plus a coarse block map for the debug overlay.
* :func:`estimate_shift` - how far the picture appears to have moved, by phase
  correlation on luminance, with a confidence response.
* :func:`reversibility_ratio` - the ratio that says whether the picture returned.

Three deliberate omissions:

* Nothing here treats a mouse delta and a pixel shift as the same quantity.
  :func:`pixels_per_delta` computes the ratio explicitly, and
  :data:`MAPPING_NOTE` is the wording that must travel with it: the ratio is a
  *measurement of the game's own camera convention*, not a conversion factor
  AutoCraft is entitled to assume.
* Nothing here returns a pass/fail verdict. LOOK-001 has no threshold on
  purpose. It reports numbers and lets the operator judge them.
* Nothing here imports the control layer. The measurement code cannot inject
  input, and a test asserts it.

There is no trained model anywhere in this module, and there will not be one:
the estimate is a closed-form phase correlation over two grayscale arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_BLOCK_GRID",
    "DEFAULT_CHANGED_THRESHOLD",
    "MAPPING_NOTE",
    "FrameDifference",
    "LookError",
    "ShiftEstimate",
    "difference_image",
    "difference_metrics",
    "estimate_shift",
    "luminance",
    "pixels_per_delta",
    "reversibility_note",
    "reversibility_ratio",
    "save_grayscale",
]


class LookError(RuntimeError):
    """Raised when a frame pair cannot be measured."""


#: The coarse partition used for the difference map. 8x8 is the specification's
#: suggestion: fine enough to show *where* the picture moved, coarse enough that
#: the whole map is 64 numbers and can ride along in the observer's JSON payload.
DEFAULT_BLOCK_GRID = 8

#: Luminance levels a pixel must move by before it counts as "changed". Rendering
#: noise and antialiasing routinely move a pixel by a couple of levels; 8 is
#: comfortably above that and still far below a real camera rotation.
DEFAULT_CHANGED_THRESHOLD = 8.0

#: Rec.601 luma weights, in the RGB order :class:`~autocraft.vision.frame.Frame`
#: stores. Using one fixed weighting keeps two runs comparable.
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float64)

#: Wording that must travel with any mouse-delta-to-pixel ratio. The ratio is a
#: measurement of the game's camera convention; AutoCraft does not get to assume
#: it, and the sign is as likely to be negative as positive.
MAPPING_NOTE = (
    "image pixels of displacement per unit of injected mouse delta, as measured. "
    "The sign follows the game's own camera convention and is not assumed."
)

#: Below this many pixels on either axis the phase-correlation estimate is
#: numerically meaningless, so the module says so instead of returning noise.
MIN_ESTIMATE_EDGE = 8


def _image_of(frame: Any) -> np.ndarray:
    """Accept either a :class:`~autocraft.vision.frame.Frame` or an array."""
    image = getattr(frame, "image", frame)
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise LookError(f"expected an H x W x 3 image, got shape {array.shape!r}")
    return array


def luminance(frame: Any) -> np.ndarray:
    """Return the frame's luminance as an ``H x W`` float64 array.

    ``0.0`` is black, ``255.0`` is white. Colour is discarded here and only here,
    because every metric in this module is about where brightness moved, not
    which way the colour went.
    """
    array = _image_of(frame)
    return array[:, :, :3].astype(np.float64) @ LUMA_WEIGHTS


def _block_reduce(delta: np.ndarray, grid: int) -> tuple[tuple[float, ...], ...]:
    """Average an ``H x W`` array over a ``grid x grid`` partition.

    Exact, not sampled: every pixel contributes to exactly one block, matching
    the convention :meth:`~autocraft.vision.frame.Frame.block_means` already uses
    so the two summaries agree about what a "block" is. When the frame is too
    small to give every block a pixel, every block gets the frame-wide mean
    rather than an invented value.
    """
    if grid < 1:
        raise LookError(f"block grid must be at least 1, got {grid}")
    rows, cols = delta.shape
    band_h, band_w = rows // grid, cols // grid
    if band_h < 1 or band_w < 1:
        flat = float(delta.mean())
        return tuple(tuple(flat for _ in range(grid)) for _ in range(grid))
    cropped = delta[: band_h * grid, : band_w * grid]
    reduced = cropped.reshape(grid, band_h, grid, band_w).mean(axis=(1, 3))
    return tuple(tuple(float(value) for value in row) for row in reduced)


@dataclass(frozen=True)
class FrameDifference:
    """How far apart two frames are, measured per pixel.

    All scalars are in luminance levels (``0..255``), except
    ``changed_fraction`` which is a fraction of the frame.

    ``mean_absolute_difference`` is the number the reversibility ratio uses: it
    is the least sensitive of the three to a single hot pixel and the most stable
    between runs.
    """

    mean_absolute_difference: float
    rmse: float
    changed_fraction: float
    max_absolute_difference: float
    block_grid: int
    block_map: tuple[tuple[float, ...], ...]
    changed_threshold: float

    @property
    def identical(self) -> bool:
        """True when no pixel moved at all."""
        return self.max_absolute_difference <= 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view. Contains no pixel data."""
        return {
            "mean_absolute_difference": round(self.mean_absolute_difference, 4),
            "rmse": round(self.rmse, 4),
            "changed_fraction": round(self.changed_fraction, 6),
            "max_absolute_difference": round(self.max_absolute_difference, 2),
            "changed_threshold": self.changed_threshold,
            "block_grid": self.block_grid,
            "block_map": [[round(value, 2) for value in row] for row in self.block_map],
        }


def difference_metrics(
    mine: Any,
    theirs: Any,
    *,
    block_grid: int = DEFAULT_BLOCK_GRID,
    changed_threshold: float = DEFAULT_CHANGED_THRESHOLD,
) -> FrameDifference:
    """Measure the per-pixel difference between two frames.

    Args:
        mine: The reference frame, as a ``Frame`` or an ``H x W x 3`` array.
        theirs: The frame to compare against it. Must have the same shape.
        block_grid: Coarse partition size for :attr:`FrameDifference.block_map`.
        changed_threshold: Luminance levels above which a pixel counts as
            changed, for :attr:`FrameDifference.changed_fraction`.

    Raises:
        LookError: If the two frames do not have the same shape.
    """
    mine_luma = luminance(mine)
    theirs_luma = luminance(theirs)
    if mine_luma.shape != theirs_luma.shape:
        raise LookError(
            f"frames must have the same shape, got {mine_luma.shape!r} and "
            f"{theirs_luma.shape!r}; a resized window cannot be measured"
        )
    signed = mine_luma - theirs_luma
    delta = np.abs(signed)
    return FrameDifference(
        mean_absolute_difference=float(delta.mean()),
        rmse=float(np.sqrt(np.mean(np.square(signed)))),
        changed_fraction=float((delta > float(changed_threshold)).mean()),
        max_absolute_difference=float(delta.max()) if delta.size else 0.0,
        block_grid=int(block_grid),
        block_map=_block_reduce(delta, int(block_grid)),
        changed_threshold=float(changed_threshold),
    )


def difference_image(mine: Any, theirs: Any) -> np.ndarray:
    """Return the absolute luminance difference as a ``uint8`` grayscale image.

    This is what gets written to ``difference_ab.png`` so the operator can see
    *where* the change happened instead of only how much. Values are the raw
    luminance delta, so a black pixel means "unchanged" and white means "the
    brightness moved by a full 255 levels".
    """
    mine_luma = luminance(mine)
    theirs_luma = luminance(theirs)
    if mine_luma.shape != theirs_luma.shape:
        raise LookError(
            f"frames must have the same shape, got {mine_luma.shape!r} and {theirs_luma.shape!r}"
        )
    delta = np.abs(mine_luma - theirs_luma)
    return np.clip(delta, 0.0, 255.0).astype(np.uint8)


@dataclass(frozen=True)
class ShiftEstimate:
    """The picture's apparent displacement between two frames, in pixels.

    Sign convention: positive ``x`` means scene content moved *right* between the
    first frame and the second, positive ``y`` means it moved *down*. The
    estimate is a property of the image pair, never of the mouse delta that
    produced it - see :func:`pixels_per_delta` for the ratio between the two.

    ``quality`` is the phase-correlation response: it says how strongly the two
    frames agree on one translation, not whether that translation is correct. It
    is reported rather than acted upon, and it is deliberately not normalised -
    OpenCV does not promise a ``0..1`` range, and a strong, well-textured match
    can come back slightly above ``1.0``. Read it as an ordering, not a score.
    A camera with no flat texture ahead produces a low response and a
    meaningless vector.
    """

    x: float = 0.0
    y: float = 0.0
    quality: float = 0.0
    method: str = "phase-correlation"
    available: bool = False
    reason: str = ""

    @property
    def magnitude(self) -> float:
        """Length of the estimated displacement vector, in pixels."""
        return float(np.hypot(self.x, self.y))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "available": self.available,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "magnitude": round(self.magnitude, 3),
            "quality": round(self.quality, 4),
            "method": self.method,
            "reason": self.reason,
        }


def estimate_shift(mine: Any, theirs: Any, *, window: bool = True) -> ShiftEstimate:
    """Estimate how far the picture moved between two frames.

    Uses OpenCV's phase correlation on luminance - a closed-form frequency-domain
    alignment, no model and no training. A Hann window is applied first because
    phase correlation assumes the image wraps around at its edges and a captured
    game view does not; without the window the discontinuity at the border
    dominates the result.

    Args:
        mine: The reference frame.
        theirs: The frame to compare against it. Must have the same shape.
        window: Apply the Hann window. Only turn this off for synthetic input
            that genuinely wraps.

    Returns:
        A :class:`ShiftEstimate`. ``available`` is False when the frames are too
        small to align, in which case ``reason`` explains why.

    Raises:
        LookError: If the shapes differ or OpenCV is unavailable.
    """
    mine_luma = luminance(mine)
    theirs_luma = luminance(theirs)
    if mine_luma.shape != theirs_luma.shape:
        raise LookError(
            f"frames must have the same shape, got {mine_luma.shape!r} and {theirs_luma.shape!r}"
        )
    rows, cols = mine_luma.shape
    if rows < MIN_ESTIMATE_EDGE or cols < MIN_ESTIMATE_EDGE:
        return ShiftEstimate(
            available=False,
            reason=f"a {cols}x{rows} frame is too small to align",
        )
    cv2 = _opencv()
    first = np.ascontiguousarray(mine_luma, dtype=np.float32)
    second = np.ascontiguousarray(theirs_luma, dtype=np.float32)
    try:
        if window:
            hann = cv2.createHanningWindow((cols, rows), cv2.CV_32F)
            (dx, dy), quality = cv2.phaseCorrelate(first, second, hann)
        else:
            (dx, dy), quality = cv2.phaseCorrelate(first, second)
    except cv2.error as exc:  # pragma: no cover - depends on the OpenCV build
        raise LookError(f"phase correlation failed: {exc}") from exc
    return ShiftEstimate(x=float(dx), y=float(dy), quality=float(quality), available=True)


def pixels_per_delta(
    shift: ShiftEstimate,
    *,
    dx: int,
    dy: int,
) -> tuple[float | None, float | None]:
    """Return image pixels per unit of injected mouse delta, per axis.

    This is the closest thing LOOK-001 produces to a sensorimotor mapping, and it
    is deliberately the *last* number computed rather than the first: the mouse
    delta and the pixel shift are measured independently and only divided here.

    A component is ``None`` when the corresponding delta was zero, because a
    ratio with no movement in the denominator is undefined rather than infinite.
    The sign is reported exactly as measured. See :data:`MAPPING_NOTE`.
    """
    per_x = None if int(dx) == 0 else float(shift.x) / float(dx)
    per_y = None if int(dy) == 0 else float(shift.y) / float(dy)
    return per_x, per_y


def reversibility_ratio(
    outbound: float,
    inbound: float,
    *,
    identical_epsilon: float = 0.0,
) -> float | None:
    """``difference(A, C) / difference(A, B)``. Lower means the picture returned.

    ``outbound`` is the A-to-B difference produced by the movement; ``inbound``
    is the A-to-C difference remaining after the movement was reversed. ``0.0``
    means the scene came back exactly, ``1.0`` means the reversal undid nothing,
    and values above ``1.0`` mean the scene ended up further away than it started.

    Returns ``None`` when ``outbound`` is not greater than ``identical_epsilon``,
    because the division has no meaning: if the movement changed nothing, there
    is no change for the reversal to undo. There is no threshold here and no
    verdict - the specification forbids inventing one.
    """
    if not (float(outbound) > float(identical_epsilon)):
        return None
    return float(inbound) / float(outbound)


def reversibility_note(outbound: float, inbound: float) -> str:
    """Plain-language reading of a :func:`reversibility_ratio` result."""
    ratio = reversibility_ratio(outbound, inbound)
    if ratio is None:
        return (
            "frames A and B were identical, so the movement produced no measurable "
            "change and a reversibility ratio does not exist"
        )
    if float(inbound) <= 0.0:
        return "frames A and C were identical: the picture returned exactly"
    return (
        f"frames A and C differ by {ratio:.3f} of the A-to-B difference; "
        "lower means the picture came back"
    )


def save_grayscale(image: np.ndarray, path: str | Path) -> Path:
    """Write a single-channel image to disk and return the path.

    Kept next to :func:`difference_image` because that is the only producer. Uses
    OpenCV for the same reason :meth:`~autocraft.vision.frame.Frame.save` does:
    one image dependency, not two.
    """
    cv2 = _opencv()
    array = np.asarray(image)
    if array.ndim != 2:
        raise LookError(f"expected an H x W grayscale image, got shape {array.shape!r}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), np.ascontiguousarray(array)):
        raise LookError(f"could not write {target}")
    return target


def _opencv() -> Any:
    try:
        import cv2  # noqa: PLC0415 - intentionally lazy: only alignment and saving need it
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise LookError(
            "measuring displacement requires opencv-python; "
            "install with 'pip install opencv-python'"
        ) from exc
    return cv2
