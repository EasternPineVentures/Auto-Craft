"""Attention without understanding: which regions of a frame stand out, and why.

This module answers exactly one question - *where in this picture is there
something worth looking at?* - and it answers it with arithmetic on the pixels.
It does not know what anything is. It has never heard of a tree, a wall, a mob or
a doorway, and the milestone forbids it from learning: no detector is trained
here, no semantic class is ever produced, and the output type is called
``CandidateTarget`` precisely so that no later reader can mistake a bright
textured blob for a recognised object.

The cues are the classic bottom-up ones, all of them cheap, all of them
deterministic:

``texture``    local luminance variation inside a cell
``edge``       local edge energy
``colour``     local colour spread
``contrast``   how far a cell's brightness sits from the frame's typical cell
``neighbour``  how far a cell's brightness sits from the cells around it

Each cue is normalised *within the frame* by dividing by that cue's own maximum
over the grid. That is deliberate and it is the whole trick: "salient" is a
relative claim, and a cue that is uniformly present everywhere carries no
information about where to look. A consequence worth stating plainly, because a
test depends on it: on a **uniform** frame every cue's maximum is zero, every cue
is therefore zero, and the salience map is empty. A featureless view produces no
candidates rather than an arbitrary one.

Normalising by the maximum is also the design's main weakness. A single blown-out
pixel can compress every other cue towards zero, so salience is only meaningful
within the frame it was computed from and must never be compared across frames.
Nothing here does compare it across frames: persistence and novelty, the two
cross-frame quantities, are computed in :mod:`autocraft.wake.memory` from
candidate *positions and descriptors*, not from raw salience.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

import numpy as np

from ..perception.features import CELL_FEATURE_NAMES, cell_boxes, cell_features, cell_luma
from ..vision.frame import Frame

__all__ = [
    "CANDIDATE_LIMIT",
    "DEFAULT_MIN_SALIENCE",
    "DEFAULT_MATCH_CONFIDENCE",
    "DEFAULT_PEAK_FRACTION",
    "DEFAULT_REFINE_SCORE",
    "MATCH_SIGMA",
    "SALIENCE_WEIGHTS",
    "CandidateTarget",
    "Relocation",
    "SalienceError",
    "SalienceMap",
    "TargetPatch",
    "candidate_from_cells",
    "descriptor_similarity",
    "find_candidates",
    "locate",
    "refine",
    "relocate",
    "score_candidates",
    "select_candidate",
    "SelectionWeights",
]


class SalienceError(RuntimeError):
    """Raised when refinement cannot run because OpenCV is missing."""

#: Feature columns used as salience cues, in the order the weights below expect.
_CUE_FEATURES = {
    "texture": "std_luma",
    "edge": "edge_energy",
    "colour": "colour_spread",
}

#: How much each cue contributes, before normalisation. These are the documented
#: magic numbers the milestone asks not to hide.
#:
#: ``texture`` and ``edge`` lead because a rendered 3D scene concentrates its
#: structure in exactly those - a face, a trunk, a fence, a block boundary all
#: read as local variation. ``colour`` is lower because Luanti's daytime palette
#: is broadly uniform and colour spread alone lights up large flat regions.
#: ``contrast`` and ``neighbour`` are the "different from what is around it"
#: terms; together they are weighted equal to texture so that a bright uniform
#: patch surrounded by darkness can compete with a busy patch in a busy scene.
SALIENCE_WEIGHTS: dict[str, float] = {
    "texture": 0.30,
    "edge": 0.25,
    "colour": 0.15,
    "contrast": 0.15,
    "neighbour": 0.15,
}

#: Fraction of the peak cell salience a cell must reach to be part of a
#: candidate. Below 1.0 so that a candidate is a *region* and not a single cell.
DEFAULT_PEAK_FRACTION = 0.45

#: Absolute floor on cell salience. Guards the degenerate case where the whole
#: frame is nearly flat: without it, the single least-flat cell of a featureless
#: view would be normalised to 1.0 and nominated as a candidate.
DEFAULT_MIN_SALIENCE = 0.15

#: Hard cap on how many candidates one frame may yield.
CANDIDATE_LIMIT = 8

#: Width of the Gaussian used to turn a descriptor distance into a match
#: confidence: ``confidence = exp(-d^2 / 2)`` on z-scored features. With seven
#: feature dimensions, two unrelated cells sit about ``sqrt(7)`` apart, which
#: scores about 0.03, while a cell that has genuinely stayed the same scores
#: near 1.0. So the useful range is wide and the threshold sits in an empty part
#: of it rather than on a cliff edge.
MATCH_SIGMA = 1.0

#: Minimum normalised cross-correlation for a refined patch match to be believed.
#: Deliberately low: the patch is cropped from an earlier frame of a moving
#: camera, so it is never expected to match perfectly, and a strict threshold
#: would report "target lost" for a target that is plainly still there. The
#: coarse descriptor match and the caller's own confidence gate are what reject
#: genuinely absent targets.
DEFAULT_REFINE_SCORE = 0.4

#: Minimum descriptor-match confidence for a coarse relocation to be believed.
#: A cell of a completely different scene still scores a little above zero -
#: z-scored features of seven dimensions happen to align by chance - so a floor of
#: zero would let a target that has left the frame be "found" somewhere arbitrary.
#: Two unrelated cells sit around ``sqrt(7)`` apart, which scores about 0.03, so
#: this floor sits in an empty part of the range rather than on a cliff edge.
DEFAULT_MATCH_CONFIDENCE = 0.35

#: Below this standard deviation a patch is featureless, and a featureless
#: template matches every location equally well. OpenCV returns meaningless
#: numbers for such a template, so it is refused rather than trusted.
_MIN_PATCH_CONTRAST = 1.0


def _has_search_room(
    region_width: int,
    region_height: int,
    patch_width: int,
    patch_height: int,
) -> bool:
    """Whether a template has somewhere to move inside its search region.

    A matcher that is offered exactly one legal position has not located
    anything: it has echoed back the position it was handed, with a perfect
    score, because a template always matches itself perfectly. That is not a
    weak measurement, it is not a measurement at all, and it is the one failure
    mode that cannot be detected by looking at the score. So it is refused by
    geometry instead. The template needs room along *either* axis - a region
    exactly as wide as the patch can still say whether the target moved up or
    down.
    """
    return int(region_width) > int(patch_width) or int(region_height) > int(patch_height)


def _spans_frame(bbox: Sequence[int], width: int, height: int) -> bool:
    """Whether a box covers the entire frame.

    A region that covers everything has no surroundings, so it is not a local
    feature of the scene at all - its "salience" is the frame's global contrast
    wearing a region's clothing. Such a box also cannot be tracked, because a
    frame-sized patch leaves the matcher no position to search. Both problems
    are the same problem, and the honest answer to it is to decline the region
    rather than to name the whole screen as a target.
    """
    return (
        int(bbox[0]) <= 0
        and int(bbox[1]) <= 0
        and int(bbox[2]) >= int(width)
        and int(bbox[3]) >= int(height)
    )

#: How much of the relocation score is given up for being far from where the
#: target was predicted to be. Small, because the prediction is only a hint - the
#: point of relocating is to follow the target, not to stay put.
_PREDICTION_PENALTY = 0.15


@dataclass(frozen=True)
class SelectionWeights:
    """The structured target-selection score, with its weights made explicit.

    ``score = salience*w_s + novelty*w_n + persistence*w_p
              - recently_seen*w_r - distance*w_d``

    where ``distance`` is the candidate's distance from the frame centre divided
    by the frame's half-diagonal, so it is in 0..1 regardless of window size.

    The values encode one judgement each:

    * ``salience`` leads at 1.0 because it is the only term measured directly
      from the current frame; everything else is history or geometry.
    * ``novelty`` at 0.6 is a strong second. The milestone's target is something
      *interesting*, and a thing already examined closely is less interesting
      than an equally salient thing that has not been.
    * ``persistence`` at 0.4 breaks ties in favour of what has stayed put. A
      region that is salient in one frame and gone in the next is usually
      animation, and chasing it wastes the centring budget.
    * ``recently_seen`` at 0.9 is the largest penalty and it is meant to be. It
      must be able to overcome salience, or the agent would re-pick the same
      region after failing to centre it, forever.
    * ``distance`` at 0.3 is a *tie-break*, not a filter. It expresses a mild
      preference for a target that is already close to the crosshair, because
      centring it costs fewer movements. It is far too small to make the agent
      ignore a much better candidate on the far side of the screen.
    """

    salience: float = 1.0
    novelty: float = 0.6
    persistence: float = 0.4
    recently_seen: float = 0.9
    distance: float = 0.3

    def __post_init__(self) -> None:
        for name in ("salience", "novelty", "persistence", "recently_seen", "distance"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} weight must be finite and non-negative, got {value!r}")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serialisable view."""
        return {
            "salience": self.salience,
            "novelty": self.novelty,
            "persistence": self.persistence,
            "recently_seen": self.recently_seen,
            "distance": self.distance,
        }


@dataclass(frozen=True)
class CandidateTarget:
    """A visually salient region. Not an object, and not a guess about one.

    ``bbox`` and ``centre`` are in pixels of the frame the candidate was found
    in, because that is the only frame it is known to be in. ``descriptor`` is
    the mean standardised appearance of its cells, which is what makes it
    findable again after the camera moves.

    ``confidence`` is evidence strength - how much of the frame's salience this
    region actually holds - not a probability that anything is there. An empty
    frame cannot produce a confident candidate because it cannot produce a
    candidate at all.
    """

    bbox: tuple[int, int, int, int]
    centre: tuple[float, float]
    cells: tuple[int, ...] = ()
    salience: float = 0.0
    novelty: float = 0.0
    persistence: float = 0.0
    confidence: float = 0.0
    score: float = 0.0
    descriptor: tuple[float, ...] = ()
    seen: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", tuple(int(value) for value in self.bbox))
        object.__setattr__(self, "centre", (float(self.centre[0]), float(self.centre[1])))
        object.__setattr__(self, "cells", tuple(int(value) for value in self.cells))
        object.__setattr__(self, "descriptor", tuple(float(value) for value in self.descriptor))
        for name in ("salience", "novelty", "persistence", "confidence", "score"):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "seen", int(self.seen))

    @property
    def width(self) -> int:
        """Width of the bounding box in pixels."""
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        """Height of the bounding box in pixels."""
        return max(0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> int:
        """Area of the bounding box in pixels."""
        return self.width * self.height

    def with_score(self, score: float) -> "CandidateTarget":
        """Return a copy carrying ``score``."""
        return replace(self, score=float(score))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "bbox": list(self.bbox),
            "centre": [round(self.centre[0], 2), round(self.centre[1], 2)],
            "cells": list(self.cells),
            "cell_count": len(self.cells),
            "width": self.width,
            "height": self.height,
            "salience": round(self.salience, 4),
            "novelty": round(self.novelty, 4),
            "persistence": round(self.persistence, 4),
            "confidence": round(self.confidence, 4),
            "score": round(self.score, 4),
            "seen": self.seen,
        }


@dataclass(frozen=True)
class Relocation:
    """Where a previously selected target appears to be now.

    ``centre`` is the matched cell's centre in the new frame's pixels;
    ``confidence`` is the descriptor match strength; ``distance`` is how far the
    match is from where the target was predicted to be.
    """

    centre: tuple[float, float]
    confidence: float
    distance: float
    cell: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "centre": [round(self.centre[0], 2), round(self.centre[1], 2)],
            "confidence": round(self.confidence, 4),
            "distance": round(self.distance, 2),
            "cell": self.cell,
        }


@dataclass(frozen=True)
class SalienceMap:
    """The per-cell salience of one frame, before any grouping."""

    grid: int
    values: tuple[tuple[float, ...], ...]
    cues: dict[str, tuple[tuple[float, ...], ...]] = field(default_factory=dict)

    @property
    def peak(self) -> float:
        """The highest cell salience, or 0.0 for an empty map."""
        return max((max(row) for row in self.values), default=0.0)

    def flat(self) -> tuple[float, ...]:
        """The map as a row-major flat tuple."""
        return tuple(value for row in self.values for value in row)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "grid": self.grid,
            "peak": round(self.peak, 4),
            "values": [[round(value, 4) for value in row] for row in self.values],
        }

    def ascii(self, levels: str = " .:-=+*#%@") -> list[str]:
        """Render the map as text, one string per row. For terminal output."""
        peak = self.peak
        out: list[str] = []
        for row in self.values:
            line = []
            for value in row:
                fraction = 0.0 if peak <= 0.0 else min(1.0, max(0.0, value / peak))
                index = int(round(fraction * (len(levels) - 1)))
                line.append(levels[index])
            out.append("".join(line))
        return out


# -- cue extraction ---------------------------------------------------------


def _normalise(values: np.ndarray) -> np.ndarray:
    """Scale a non-negative cue to 0..1 by its own maximum over the frame.

    A cue whose maximum is zero is zero everywhere: it carries no information
    about *where* to look, which is the only question being asked.
    """
    peak = float(np.max(values)) if values.size else 0.0
    if not math.isfinite(peak) or peak <= 1e-9:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip(values.astype(np.float64) / peak, 0.0, 1.0)


def _neighbour_difference(luma: np.ndarray) -> np.ndarray:
    """How far each cell's brightness sits from the mean of its neighbours.

    Neighbours are the up-to-four orthogonal cells. Slicing rather than rolling
    means the edge of the grid simply has fewer neighbours instead of a wrapped
    partner on the opposite side, which would make opposite edges of the frame
    look alike.
    """
    total = np.zeros_like(luma, dtype=np.float64)
    count = np.zeros_like(luma, dtype=np.float64)
    for source, destination in (
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
    ):
        total[destination] += luma[source]
        count[destination] += 1.0
    count = np.where(count <= 0.0, 1.0, count)
    return np.abs(luma - total / count)


def salience_map(frame: Frame | np.ndarray, *, grid: int = 8) -> SalienceMap:
    """Compute the per-cell salience of one frame.

    Args:
        frame: A :class:`~autocraft.vision.frame.Frame` or an ``H x W x 3``
            uint8 array.
        grid: Cells per axis.

    Returns:
        A :class:`SalienceMap` whose values are in 0..1 and are comparable only
        with other cells of the *same* frame.
    """
    image = frame.image if isinstance(frame, Frame) else np.asarray(frame)
    features = cell_features(image, grid)
    luma = cell_luma(features)

    cues: dict[str, np.ndarray] = {}
    for name, column in _CUE_FEATURES.items():
        cues[name] = _normalise(features[:, :, CELL_FEATURE_NAMES.index(column)])
    typical = float(np.median(luma))
    cues["contrast"] = _normalise(np.abs(luma - typical))
    cues["neighbour"] = _normalise(_neighbour_difference(luma))

    total = np.zeros_like(luma, dtype=np.float64)
    for name, weight in SALIENCE_WEIGHTS.items():
        total += weight * cues[name]

    return SalienceMap(
        grid=int(grid),
        values=tuple(tuple(float(value) for value in row) for row in total),
        cues={name: tuple(tuple(float(value) for value in row) for row in values) for name, values in cues.items()},
    )


# -- candidate grouping -----------------------------------------------------


def _components(selected: set[int], grid: int) -> list[list[int]]:
    """Group cell indices into 8-connected regions, in a deterministic order.

    Eight-connectivity rather than four on purpose. A salient blob usually shows
    up as a ring of cells around a hollow centre - the interior of a uniform patch
    has no internal texture or edges to speak of, so only its boundary cells score
    - and under four-connectivity that ring's corners are not joined to anything,
    which splits one visible region into several arbitrary pieces. Diagonal
    neighbours are therefore treated as connected.

    The order is deterministic: groups always start from the lowest remaining
    index and are returned sorted, so the same frame always yields the same
    candidates in the same order.
    """
    remaining = set(selected)
    groups: list[list[int]] = []
    while remaining:
        start = min(remaining)
        remaining.discard(start)
        group = [start]
        queue: deque[int] = deque([start])
        while queue:
            cell = queue.popleft()
            row, column = divmod(cell, grid)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, column + dc
                    if not (0 <= nr < grid and 0 <= nc < grid):
                        continue
                    other = nr * grid + nc
                    if other in remaining:
                        remaining.discard(other)
                        group.append(other)
                        queue.append(other)
        groups.append(sorted(group))
    return groups


def _merge_overlapping(groups: list[list[int]], grid: int) -> list[list[int]]:
    """Join components whose cell footprints touch, directly or through another.

    The salience threshold is a single number, so one visual region can cross it
    in a few places and dip below it in between - the interior of a uniform patch
    scores nothing while its boundary scores highly, and a corner can fall just
    under the line. Left alone, that turns one region into several candidates and
    the agent ends up choosing between fragments of the same thing.

    Two groups are joined when their bounding boxes intersect. Because a group's
    box is the union of whole cells, touching boxes do intersect, so this is a
    closing operation expressed on boxes. It is applied repeatedly until nothing
    more joins, so chains of fragments collapse together, and the result is
    deterministic: groups are always visited lowest-index-first.
    """
    current = [list(group) for group in groups]
    changed = True
    while changed and len(current) > 1:
        changed = False
        boxes = [_group_box(group, grid) for group in current]
        merged: list[list[int]] = []
        consumed = [False] * len(current)
        for index in range(len(current)):
            if consumed[index]:
                continue
            group = list(current[index])
            box = boxes[index]
            for other in range(index + 1, len(current)):
                if consumed[other] or not _boxes_intersect(box, boxes[other]):
                    continue
                group.extend(current[other])
                box = _union_box(box, boxes[other])
                consumed[other] = True
                changed = True
            merged.append(sorted(set(group)))
        current = merged
    return sorted(current, key=lambda group: group[0])


def _group_box(group: Sequence[int], grid: int) -> tuple[int, int, int, int]:
    """The ``(row0, col0, row1, col1)`` cell extent of a group."""
    rows = [cell // grid for cell in group]
    columns = [cell % grid for cell in group]
    return (min(rows), min(columns), max(rows), max(columns))


def _boxes_intersect(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    """True when two inclusive cell boxes share at least one cell."""
    return not (left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1])


def _union_box(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """The smallest inclusive cell box containing both."""
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


def _descriptor_of(features: np.ndarray, cells: Sequence[int]) -> tuple[float, ...]:
    """Mean appearance of a set of cells, standardised across the whole grid.

    Standardising per feature dimension across the frame is what makes the
    descriptor comparable between two frames: it removes any global change in
    exposure or brightness and keeps only the *relative* pattern of appearance.

    Each cell's contribution is its own 3x3 neighbourhood mean rather than its
    raw value. That matters because a salient region is usually several cells
    wide, and a single cell's raw appearance is noisy enough that the *interior*
    of a uniform patch - which has no texture at all - would look like the best
    match for a textured region. Averaging over the neighbourhood gives every
    cell a little context, so a region descriptor describes a region.
    """
    grid = int(features.shape[0])
    smoothed = _smoothed_standard(features, grid)
    block = smoothed[list(cells)]
    return tuple(float(value) for value in block.mean(axis=0))


def _smoothed_standard(features: np.ndarray, grid: int) -> np.ndarray:
    """Standardise every cell's features, then average each over its 3x3 block.

    Returns:
        A ``(grid * grid, features)`` array in row-major cell order, ready to be
        matched against a descriptor. Edge cells are padded by replication rather
        than wrapped, so a cell on the frame's edge is never described in terms of
        the opposite edge's appearance.
    """
    standardised = _standardise(features)
    cube = standardised.reshape(grid, grid, -1)
    padded = np.pad(cube, ((1, 1), (1, 1), (0, 0)), mode="edge")
    total = np.zeros_like(cube)
    for dr in range(3):
        for dc in range(3):
            total += padded[dr : dr + grid, dc : dc + grid]
    return (total / 9.0).reshape(-1, cube.shape[-1])


def find_candidates(
    frame: Frame | np.ndarray,
    *,
    grid: int = 8,
    precomputed: SalienceMap | None = None,
    peak_fraction: float = DEFAULT_PEAK_FRACTION,
    min_salience: float = DEFAULT_MIN_SALIENCE,
    limit: int = CANDIDATE_LIMIT,
) -> list[CandidateTarget]:
    """Find the salient regions of one frame, strongest first.

    Cells at or above ``peak_fraction`` of the frame's peak salience are grouped
    into connected regions; each region becomes one candidate. Regions are ranked
    by their mean cell salience, which is a *size-blind* ranking on purpose - a
    large dim patch should not out-rank a small bright one, and merging the two
    would leave the agent unable to choose between them.

    Args:
        frame: A :class:`~autocraft.vision.frame.Frame` or an ``H x W x 3``
            uint8 array.
        grid: Cells per axis.
        precomputed: A salience map already computed for this frame, to avoid
            recomputing it.
        peak_fraction: Threshold as a fraction of the frame's peak salience.
        min_salience: Absolute floor; below it the frame is treated as flat.
        limit: Maximum number of candidates to return.

    Returns:
        Candidates sorted by salience, strongest first. Empty for a flat frame,
        and empty for a frame whose only candidate would be the whole frame -
        see :func:`_spans_frame`.
    """
    image = frame.image if isinstance(frame, Frame) else np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an H x W x 3 image, got shape {getattr(image, 'shape', None)}")
    height, width = int(image.shape[0]), int(image.shape[1])

    salience = salience_map(image, grid=grid) if precomputed is None else precomputed
    values = np.asarray(salience.values, dtype=np.float64)
    peak = float(values.max()) if values.size else 0.0
    if peak < min_salience:
        return []

    threshold = max(min_salience, peak_fraction * peak)
    chosen = {int(index) for index in np.flatnonzero(values.reshape(-1) >= threshold)}
    if not chosen:
        return []

    features = cell_features(image, grid)
    boxes = cell_boxes(width, height, grid)
    flat = values.reshape(-1)

    found: list[CandidateTarget] = []
    for group in _merge_overlapping(_components(chosen, grid), grid):
        weights = flat[group]
        cell_boxes_group = [boxes[cell] for cell in group]
        x0 = min(box[0] for box in cell_boxes_group)
        y0 = min(box[1] for box in cell_boxes_group)
        x1 = max(box[2] for box in cell_boxes_group)
        y1 = max(box[3] for box in cell_boxes_group)
        if _spans_frame((x0, y0, x1, y1), width, height):
            # Every cell cleared the threshold, which on a low-contrast frame
            # means the threshold said nothing about the scene. The whole frame
            # is not a region, and it cannot be tracked, so it is not offered.
            continue
        total = float(weights.sum())
        if total <= 0.0:
            continue
        centres = np.array(
            [((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0) for box in cell_boxes_group],
            dtype=np.float64,
        )
        centre = (centres * weights[:, None]).sum(axis=0) / total
        salience_value = float(weights.mean())
        # Confidence is how much of the frame's total salience this region holds,
        # scaled so that a region holding half the frame's salience reads 1.0.
        share = total / float(flat.sum()) if float(flat.sum()) > 0.0 else 0.0
        confidence = min(1.0, share / 0.5)
        found.append(
            CandidateTarget(
                bbox=(x0, y0, x1, y1),
                centre=(float(centre[0]), float(centre[1])),
                cells=tuple(group),
                salience=salience_value,
                confidence=confidence,
                descriptor=_descriptor_of(features, group),
            )
        )

    found.sort(key=lambda candidate: (-candidate.salience, candidate.bbox))
    return found[: max(1, int(limit))]


def candidate_from_cells(
    frame: Frame | np.ndarray,
    cells: Iterable[int],
    *,
    grid: int = 8,
    precomputed: SalienceMap | None = None,
) -> CandidateTarget | None:
    """Build one candidate from an explicit set of cell indices.

    Used by tests and by the policy when a target has been followed to a new set
    of cells and needs its bounding box and descriptor rebuilt.
    """
    image = frame.image if isinstance(frame, Frame) else np.asarray(frame)
    height, width = int(image.shape[0]), int(image.shape[1])
    chosen = sorted(int(cell) for cell in cells)
    if not chosen:
        return None
    salience = salience_map(image, grid=grid) if precomputed is None else precomputed
    flat = np.asarray(salience.values, dtype=np.float64).reshape(-1)
    boxes = cell_boxes(width, height, grid)
    features = cell_features(image, grid)
    group = [boxes[cell] for cell in chosen]
    x0 = min(box[0] for box in group)
    y0 = min(box[1] for box in group)
    x1 = max(box[2] for box in group)
    y1 = max(box[3] for box in group)
    if _spans_frame((x0, y0, x1, y1), width, height):
        # Same refusal as find_candidates: a box covering the whole frame is not
        # a region, and a frame-sized patch cannot be located.
        return None
    weights = flat[chosen]
    total = float(weights.sum())
    centres = np.array([((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0) for box in group], dtype=np.float64)
    if total > 0.0:
        centre = (centres * weights[:, None]).sum(axis=0) / total
    else:
        centre = centres.mean(axis=0)
    share = total / float(flat.sum()) if float(flat.sum()) > 0.0 else 0.0
    return CandidateTarget(
        bbox=(x0, y0, x1, y1),
        centre=(float(centre[0]), float(centre[1])),
        cells=tuple(chosen),
        salience=float(weights.mean()) if weights.size else 0.0,
        confidence=min(1.0, share / 0.5),
        descriptor=_descriptor_of(features, chosen),
    )


# -- scoring and selection --------------------------------------------------


@dataclass(frozen=True)
class TargetPatch:
    """A target's appearance, cropped from a frame, plus where it came from.

    The geometry is carried with the pixels because a cropped patch on its own is
    ambiguous: matching it later finds where the *patch* went, and the patch's
    centre is not the target's centre whenever padding was added or the bounding
    box was not symmetric about the target. Keeping the origin and the target
    centre together makes that correction impossible to forget.
    """

    image: np.ndarray
    origin: tuple[int, int]
    centre: tuple[float, float]

    @classmethod
    def of(
        cls,
        frame: Frame | np.ndarray,
        bbox: Sequence[int],
        *,
        centre: tuple[float, float] | None = None,
        pad: int = 0,
    ) -> "TargetPatch":
        """Crop a target out of a frame.

        Args:
            frame: A :class:`~autocraft.vision.frame.Frame` or ``H x W x 3`` array.
            bbox: ``(x0, y0, x1, y1)`` in pixels, half-open.
            centre: The target's own centre, which is what the caller will want to
                track. Defaults to the bounding box's centre.
            pad: Extra pixels on every side. A little padding gives the matcher
                some surround to lock onto instead of only the target's interior.
        """
        source = frame.image if isinstance(frame, Frame) else np.asarray(frame)
        height, width = int(source.shape[0]), int(source.shape[1])
        x0 = max(0, min(width - 1, int(bbox[0]) - int(pad)))
        y0 = max(0, min(height - 1, int(bbox[1]) - int(pad)))
        x1 = max(x0 + 1, min(width, int(bbox[2]) + int(pad)))
        y1 = max(y0 + 1, min(height, int(bbox[3]) + int(pad)))
        if centre is None:
            centre = ((int(bbox[0]) + int(bbox[2])) / 2.0, (int(bbox[1]) + int(bbox[3])) / 2.0)
        return cls(
            image=np.ascontiguousarray(source[y0:y1, x0:x1]),
            origin=(x0, y0),
            centre=(float(centre[0]), float(centre[1])),
        )

    @property
    def size(self) -> tuple[int, int]:
        """The patch's ``(width, height)`` in pixels."""
        return (int(self.image.shape[1]), int(self.image.shape[0]))

    @property
    def contrast(self) -> float:
        """Standard deviation of the patch's pixels, across all channels."""
        if self.image.size == 0:
            return 0.0
        return float(self.image.reshape(-1, self.image.shape[-1]).astype(np.float64).std())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view. The pixels are not included."""
        return {
            "origin": list(self.origin),
            "centre": [round(value, 2) for value in self.centre],
            "width": self.size[0],
            "height": self.size[1],
            "contrast": round(self.contrast, 3),
        }


def refine(
    frame: Frame | np.ndarray,
    patch: TargetPatch,
    *,
    predicted: tuple[float, float] | None = None,
    window: int | None = None,
    min_score: float = DEFAULT_REFINE_SCORE,
    scale: int = 1,
) -> Relocation | None:
    """Locate a patch by normalised cross-correlation.

    ``TM_CCOEFF_NORMED`` is used rather than a plain difference because it is
    invariant to a change in brightness and contrast, which is the same reason the
    view fingerprint correlates rather than subtracts: turning the camera changes
    the sky gradient and the exposure, and that is not what "the target moved"
    means.

    Args:
        frame: The frame to search.
        patch: The target's appearance and geometry, from an earlier frame.
        predicted: Where the target's *centre* is expected to be, in pixels. When
            omitted the whole frame is searched.
        window: Half-width in pixels of the region to search around ``predicted``.
            When omitted, or when ``predicted`` is omitted, the whole frame is used.
        min_score: Reject a match scoring below this.
        scale: Integer block-average factor to search at. A result found at scale
            ``s`` is accurate to about ``s`` original pixels, which is why
            :func:`locate` follows a coarse pass with a full-resolution one.

    Returns:
        A :class:`Relocation` whose ``centre`` is the target's centre, or ``None``
        when the patch is too flat to match, the search region is smaller than the
        patch or exactly the patch, or the best score is below ``min_score``.
        Returning ``None`` for a flat patch is the only honest answer: a
        featureless template matches every location equally well, and reporting the
        best of those would be inventing a position. Returning ``None`` when the
        region is exactly the patch is the same honesty applied to geometry: a
        template always matches itself perfectly, so the "match" would be the
        patch's own assumed position, not a measurement of where it went.
    """
    image = frame.image if isinstance(frame, Frame) else np.asarray(frame)
    height, width = int(image.shape[0]), int(image.shape[1])
    template = np.asarray(patch.image)
    if template.ndim != 3 or template.shape[2] != 3 or template.size == 0:
        return None
    if patch.contrast < _MIN_PATCH_CONTRAST:
        return None

    factor = max(1, int(scale))
    if factor > 1:
        return _refine_scaled(
            image,
            patch,
            predicted=predicted,
            window=window,
            min_score=min_score,
            factor=factor,
        )

    patch_height, patch_width = int(template.shape[0]), int(template.shape[1])
    if patch_height > height or patch_width > width:
        return None

    if predicted is None or window is None:
        left, top, right, bottom = 0, 0, width, height
    else:
        # ``predicted`` is the target's centre, so the search region is the
        # prediction expanded by half the patch plus the radius. The
        # centre-to-corner offset is applied here, once, rather than left to the
        # caller to remember.
        radius = max(1, int(window))
        half_width = patch_width / 2.0
        half_height = patch_height / 2.0
        left = max(0, int(math.floor(predicted[0] - half_width - radius)))
        top = max(0, int(math.floor(predicted[1] - half_height - radius)))
        right = min(width, int(math.ceil(predicted[0] + half_width + radius)))
        bottom = min(height, int(math.ceil(predicted[1] + half_height + radius)))
    if right - left < patch_width or bottom - top < patch_height:
        return None
    if not _has_search_room(right - left, bottom - top, patch_width, patch_height):
        # The region is exactly the patch, so there is one legal position and the
        # match would be the patch's own origin echoed back at a perfect score.
        # That is the identity, not a location.
        return None
    region = np.ascontiguousarray(image[top:bottom, left:right])

    cv2 = _opencv()
    result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    if not math.isfinite(float(score)) or float(score) < float(min_score):
        return None
    matched_origin = (left + int(location[0]), top + int(location[1]))
    return _relocation_of(patch, matched_origin, float(score), width)


def _refine_scaled(
    image: np.ndarray,
    patch: TargetPatch,
    *,
    predicted: tuple[float, float] | None,
    window: int | None,
    min_score: float,
    factor: int,
) -> Relocation | None:
    """The ``scale > 1`` branch of :func:`refine`, kept separate for readability."""
    small_frame = _downscale(image, factor)
    small_patch = _downscale(patch.image, factor)
    if small_frame is None or small_patch is None:
        return None
    small_height, small_width = int(small_frame.shape[0]), int(small_frame.shape[1])
    patch_height, patch_width = int(small_patch.shape[0]), int(small_patch.shape[1])
    if patch_height > small_height or patch_width > small_width:
        return None

    if predicted is None or window is None:
        left, top, right, bottom = 0, 0, small_width, small_height
    else:
        radius = max(1, int(round(window / factor)))
        centre_x = predicted[0] / factor
        centre_y = predicted[1] / factor
        left = max(0, int(math.floor(centre_x - patch_width / 2.0 - radius)))
        top = max(0, int(math.floor(centre_y - patch_height / 2.0 - radius)))
        right = min(small_width, int(math.ceil(centre_x + patch_width / 2.0 + radius)))
        bottom = min(small_height, int(math.ceil(centre_y + patch_height / 2.0 + radius)))
    if right - left < patch_width or bottom - top < patch_height:
        return None
    if not _has_search_room(right - left, bottom - top, patch_width, patch_height):
        # One legal position at this scale too, so the coarse pass would return
        # the patch's own origin. Refusing lets locate fall through to the
        # full-resolution pass, which applies the same rule.
        return None
    region = np.ascontiguousarray(small_frame[top:bottom, left:right])

    cv2 = _opencv()
    result = cv2.matchTemplate(region, small_patch, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    if not math.isfinite(float(score)) or float(score) < float(min_score):
        return None
    # The match is accurate to about one downscaled pixel, so the origin is scaled
    # back up and the residual error is left for the full-resolution pass.
    matched_origin = ((left + int(location[0])) * factor, (top + int(location[1])) * factor)
    return _relocation_of(patch, matched_origin, float(score), small_width)


def locate(
    frame: Frame | np.ndarray,
    patch: TargetPatch,
    *,
    predicted: tuple[float, float] | None = None,
    window: int | None = None,
    coarse_scale: int = 4,
    min_score: float = DEFAULT_REFINE_SCORE,
    fine_margin: int = 12,
) -> Relocation | None:
    """Find a patch anywhere in a frame, to pixel precision.

    A single full-resolution correlation over a 3222-pixel-wide frame with a large
    patch costs most of a second, and the salience grid - eight cells across that
    frame - resolves position only to about four hundred pixels, nowhere near
    enough to centre something to within twelve. So this runs two passes: a cheap
    block-averaged pass over the whole frame, then a full-resolution pass in a
    small window around the coarse answer. The coarse pass is accurate to about
    ``coarse_scale`` pixels, so the fine window is that plus a margin, which
    bounds the expensive second pass regardless of frame size.

    Args:
        frame: The frame to search.
        patch: The target's appearance and geometry.
        predicted: Where the target's centre is expected to be. Used only to bias
            the coarse pass; a target that has moved further is still found.
        window: Half-width in pixels of the fine search window. Defaults to four
            times ``coarse_scale``, which comfortably covers the coarse pass's own
            error.
        coarse_scale: Block-average factor for the first pass.
        min_score: Reject a match scoring below this.
        fine_margin: Extra pixels added to the fine window beyond the coarse pass's
            error bound.

    Returns:
        The best :class:`Relocation`, or ``None`` if neither pass found a match.
        The returned ``confidence`` is whichever pass won, so a coarse pass that
        happened to be optimistic cannot inflate a full-resolution score.
    """
    factor = max(1, int(coarse_scale))
    coarse: Relocation | None = None
    if factor > 1:
        coarse = refine(frame, patch, predicted=predicted, window=None, min_score=min_score, scale=factor)
    if coarse is None:
        # Nothing from the coarse pass, or no coarse pass was asked for: fall back
        # to a full-resolution search, which is slower but never wrong.
        return refine(frame, patch, predicted=predicted, window=window, min_score=min_score)

    fine_window = max(int(fine_margin), 4 * factor) if window is None else int(window)
    fine = refine(frame, patch, predicted=coarse.centre, window=fine_window, min_score=min_score)
    if fine is None:
        return coarse
    # Prefer the full-resolution answer, but never let it be worse than the coarse
    # one: a bigger score is a better match, and the two can disagree when the
    # target is partly occluded.
    return fine if fine.confidence >= coarse.confidence else coarse


def _relocation_of(
    patch: TargetPatch,
    matched_origin: tuple[int, int],
    score: float,
    width: int,
) -> Relocation:
    """Turn a matched patch origin into a relocation of the target's centre."""
    centre = (
        patch.centre[0] + (matched_origin[0] - patch.origin[0]),
        patch.centre[1] + (matched_origin[1] - patch.origin[1]),
    )
    cell = int(round(centre[1])) * int(width) + int(round(centre[0]))
    return Relocation(centre=centre, confidence=float(score), distance=0.0, cell=cell)


def _downscale(image: np.ndarray, factor: int) -> np.ndarray | None:
    """Block-average an image by an integer factor, discarding the remainder.

    Block averaging rather than stride sampling, for the same reason the observer's
    display frames do it: a stride sample of a rendered 3D scene shimmers, and the
    template matcher would then be matching aliasing rather than content. The
    remainder rows and columns are dropped, which is harmless because the dropped
    strip is narrower than one output pixel.

    Returns:
        The reduced image, or ``None`` when the input is too small to reduce.
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        return None
    height, width = int(array.shape[0]), int(array.shape[1])
    rows = height // factor
    columns = width // factor
    if rows < 1 or columns < 1:
        return None
    trimmed = array[: rows * factor, : columns * factor].astype(np.float32)
    return trimmed.reshape(rows, factor, columns, factor, 3).mean(axis=(1, 3)).astype(np.uint8)


def _opencv() -> Any:
    """Import OpenCV lazily, so the pure-numpy path needs no compiled dependency."""
    try:
        import cv2  # noqa: PLC0415 - intentionally lazy: only refinement needs it
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise SalienceError(
            "refining a target to pixel precision requires opencv-python; "
            "install with 'pip install opencv-python'"
        ) from exc
    return cv2


def _standardise(features: np.ndarray) -> np.ndarray:
    """Z-score every feature dimension across the grid."""
    flat = features.reshape(-1, features.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    spread = flat.std(axis=0)
    spread = np.where(spread < 1e-6, 1.0, spread)
    return (flat - mean) / spread


def descriptor_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """How alike two candidate descriptors are, in 0..1.

    Uses the same ``exp(-d^2 / 2)`` form as relocation, on the same z-scored
    scale, so a threshold means the same thing in both places: about 0.7 is "the
    same region seen again", and unrelated regions sit near 0.03.

    Returns 0.0 when either descriptor is missing or the lengths differ, because
    "cannot compare" must never read as "identical".
    """
    if not left or not right or len(left) != len(right):
        return 0.0
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    distance = float(np.linalg.norm(difference))
    return float(math.exp(-(distance**2) / (2.0 * MATCH_SIGMA**2)))


def relocate(
    target: CandidateTarget,
    frame: Frame | np.ndarray,
    *,
    predicted: tuple[float, float] | None = None,
    search_radius: float,
    grid: int = 8,
    min_confidence: float = DEFAULT_MATCH_CONFIDENCE,
) -> Relocation | None:
    """Find ``target`` again in a new frame by its appearance.

    Matching is by the standardised descriptor over every cell of the frame, then
    a small penalty for cells far from where the target was predicted to be. The
    prediction is a hint, not a constraint: a target that moved further than
    expected is still found, just at a slightly lower score.

    Args:
        target: The candidate to look for.
        frame: The frame to look in.
        predicted: Where the target was expected to be, in pixels. Defaults to
            the target's own last known centre.
        search_radius: Radius in pixels inside which a match is considered at all.
        grid: Cells per axis.
        min_confidence: Reject matches weaker than this.

    Returns:
        The best :class:`Relocation`, or ``None`` when nothing cleared
        ``min_confidence`` inside the radius.
    """
    image = frame.image if isinstance(frame, Frame) else np.asarray(frame)
    height, width = int(image.shape[0]), int(image.shape[1])
    if not target.descriptor:
        return None
    guess = predicted if predicted is not None else target.centre
    boxes = cell_boxes(width, height, grid)
    features = cell_features(image, grid)
    standardised = _smoothed_standard(features, int(grid))
    reference = np.asarray(target.descriptor, dtype=np.float64)
    if reference.size != standardised.shape[1]:
        return None
    distances = np.linalg.norm(standardised - reference, axis=1)
    confidence = np.exp(-(distances**2) / (2.0 * MATCH_SIGMA**2))

    radius = max(1.0, float(search_radius))
    best: Relocation | None = None
    best_score = -math.inf
    for cell, box in enumerate(boxes):
        centre = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        offset = math.hypot(centre[0] - guess[0], centre[1] - guess[1])
        if offset > radius:
            continue
        score = float(confidence[cell]) - _PREDICTION_PENALTY * (offset / radius)
        if score > best_score:
            best_score = score
            best = Relocation(centre=centre, confidence=float(confidence[cell]), distance=offset, cell=cell)
    if best is None or best.confidence < float(min_confidence):
        return None
    return best


def score_candidates(
    candidates: Sequence[CandidateTarget],
    *,
    frame_size: tuple[int, int],
    weights: SelectionWeights | None = None,
) -> list[CandidateTarget]:
    """Score candidates and return them best-first.

    The distance term is the candidate centre's distance from the frame centre
    divided by the frame's half-diagonal, so it is in 0..1 for any window size.
    """
    active = weights if weights is not None else SelectionWeights()
    width, height = int(frame_size[0]), int(frame_size[1])
    half_diagonal = max(1.0, math.hypot(width / 2.0, height / 2.0))
    centre_x, centre_y = width / 2.0, height / 2.0

    scored: list[CandidateTarget] = []
    for candidate in candidates:
        distance = math.hypot(candidate.centre[0] - centre_x, candidate.centre[1] - centre_y) / half_diagonal
        distance = min(1.0, max(0.0, distance))
        value = (
            active.salience * candidate.salience
            + active.novelty * candidate.novelty
            + active.persistence * candidate.persistence
            - active.recently_seen * _seen_penalty(candidate)
            - active.distance * distance
        )
        scored.append(candidate.with_score(value))
    scored.sort(key=lambda candidate: (-candidate.score, candidate.bbox))
    return scored


def _seen_penalty(candidate: CandidateTarget) -> float:
    """How strongly a candidate should be penalised for having been examined.

    ``seen`` is the number of times this region has already been selected or
    found, so the penalty saturates at 1.0 after the second sighting. It
    saturates rather than growing because an unbounded penalty would let a
    never-seen region beat an overwhelmingly better one purely on freshness.
    """
    return min(1.0, max(0.0, (candidate.seen - 1) / 1.0))


def select_candidate(
    candidates: Sequence[CandidateTarget],
    *,
    tie_epsilon: float = 0.0,
    rng: Any | None = None,
) -> CandidateTarget | None:
    """Pick one candidate, best-first, with bounded variation on near-ties.

    ``tie_epsilon`` is the only place randomness is permitted in the behaviour
    layer, and it is bounded by construction: it can choose among candidates
    whose scores are within ``tie_epsilon`` of the best, and it can never choose
    a candidate outside that band. When one candidate is clearly superior the
    choice is deterministic whatever ``rng`` does.

    Args:
        candidates: Already scored, best-first.
        tie_epsilon: Absolute score band treated as a tie. Zero disables it.
        rng: A ``random.Random``-like object. Omitted means deterministic.

    Returns:
        The chosen candidate, or ``None`` for an empty sequence.
    """
    if not candidates:
        return None
    best = candidates[0]
    if tie_epsilon <= 0.0 or rng is None or len(candidates) == 1:
        return best
    tied = [candidate for candidate in candidates if best.score - candidate.score <= tie_epsilon]
    if len(tied) == 1:
        return best
    return tied[int(rng.randrange(len(tied)))]
