"""Per-cell appearance features: pixels in, numbers out.

VISION-001 rests on one testable idea: a cell's *appearance* carries information
about its *temporal behaviour*. A patch of animated liquid looks different from a
patch of static stone, so how much a cell will move from frame to frame can be
predicted from a single frame, before any second frame exists. That prediction is
what lets the change detector stop calling an animated texture a change.

This module produces the appearance half of that prediction. It reads a frame
through a coarse grid and describes each cell with a handful of cheap numbers.

The luma used here is the plain mean across the three channels, not a Rec.601
weighting, because that is exactly what
:meth:`autocraft.vision.frame.Frame.block_means` already uses. Matching it means
the perception layer and the rest of the project cannot disagree about what "a
cell's brightness" means.

Nothing here imports :mod:`autocraft.control`. The whole package is read-only by
construction: it can look at the game and has no way to touch it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from ..vision.frame import Frame

__all__ = [
    "CELL_FEATURE_NAMES",
    "cell_boxes",
    "cell_features",
    "cell_luma",
    "describe",
    "feature_matrix",
    "features_of",
    "iter_cells",
]


#: The appearance numbers produced for each cell, in order.
#:
#: The first four are brightness and colour, which describe *what* is there. The
#: last three are texture, which describes *how much is going on* there. Texture
#: is what actually predicts motion, but colour is what separates materials, and
#: keeping both lets the learner discover which one matters instead of being told.
CELL_FEATURE_NAMES: tuple[str, ...] = (
    "mean_r",
    "mean_g",
    "mean_b",
    "mean_luma",
    "std_luma",
    "edge_energy",
    "colour_spread",
)


def _band_bounds(size: int, grid: int) -> list[tuple[int, int]]:
    """Split ``size`` pixels into ``grid`` near-equal ``(start, stop)`` bands.

    Uses even division with the remainder spread over the first bands, so every
    pixel belongs to exactly one band and the partition always covers the whole
    frame. This deliberately differs from :meth:`Frame.block_means`, which crops
    the frame to a whole number of bands and throws the remainder away: an
    animated strip at the very edge of the screen must land inside a cell, not be
    cropped out of the measurement. The two agree exactly whenever the frame size
    divides evenly by the grid.
    """
    base = size // grid
    extra = size - base * grid
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(grid):
        width = base + (1 if index < extra else 0)
        bounds.append((start, start + width))
        start += width
    return bounds


def cell_boxes(width: int, height: int, grid: int) -> list[tuple[int, int, int, int]]:
    """Return the ``(x0, y0, x1, y1)`` pixel box of every cell, row-major.

    Exposed so a caller can point at a cell on screen without re-deriving the
    partition, and so the partition itself can be tested directly.
    """
    if grid < 1:
        raise ValueError(f"grid must be at least 1, got {grid}")
    if width < 1 or height < 1:
        raise ValueError(f"frame must have positive size, got {width}x{height}")
    columns = _band_bounds(width, grid)
    rows = _band_bounds(height, grid)
    return [(x0, y0, x1, y1) for y0, y1 in rows for x0, x1 in columns]


def iter_cells(image: np.ndarray, grid: int) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield ``(row, column, patch)`` for every cell of ``image``.

    Each patch is a view, not a copy, so walking the grid costs no extra memory
    on a multi-megapixel frame.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be H x W x 3, got shape {image.shape!r}")
    rows = _band_bounds(int(image.shape[0]), grid)
    columns = _band_bounds(int(image.shape[1]), grid)
    for row, (y0, y1) in enumerate(rows):
        for column, (x0, x1) in enumerate(columns):
            yield row, column, image[y0:y1, x0:x1]


def cell_features(image: np.ndarray, grid: int) -> np.ndarray:
    """Describe every cell of ``image`` using :data:`CELL_FEATURE_NAMES`.

    Args:
        image: An ``H x W x 3`` array. ``uint8`` is the expected input, but any
            numeric dtype works because everything is promoted to ``float32``.
        grid: Partition size, so ``grid x grid`` cells come back.

    Returns:
        A ``(grid, grid, len(CELL_FEATURE_NAMES))`` ``float32`` array. Cell
        ``[r, c]`` covers pixel box ``cell_boxes(width, height, grid)[r * grid + c]``.
    """
    if grid < 1:
        raise ValueError(f"grid must be at least 1, got {grid}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be H x W x 3, got shape {image.shape!r}")

    height, width = int(image.shape[0]), int(image.shape[1])
    if height < grid or width < grid:
        raise ValueError(
            f"a {width}x{height} frame cannot be split into {grid}x{grid} cells; "
            "every cell would be empty"
        )

    row_bounds = _band_bounds(height, grid)
    column_bounds = _band_bounds(width, grid)

    pixels = np.asarray(image, dtype=np.float32)
    luma = pixels.mean(axis=2)

    # Shape-preserving gradient, so the gradient map lines up with the luma map
    # and both reduce through the same cell boundaries in one pass. np.diff would
    # shrink the array and force a second, differently aligned partition.
    gradient = np.zeros_like(luma)
    gradient[:-1, :] += np.abs(np.diff(luma, axis=0))
    gradient[:, :-1] += np.abs(np.diff(luma, axis=1))

    features = np.zeros((grid, grid, len(CELL_FEATURE_NAMES)), dtype=np.float32)
    for row, (y0, y1) in enumerate(row_bounds):
        for column, (x0, x1) in enumerate(column_bounds):
            patch = pixels[y0:y1, x0:x1]
            patch_luma = luma[y0:y1, x0:x1]
            channel_means = patch.mean(axis=(0, 1))
            features[row, column] = (
                channel_means[0],
                channel_means[1],
                channel_means[2],
                float(patch_luma.mean()),
                float(patch_luma.std()),
                float(gradient[y0:y1, x0:x1].mean()),
                float(channel_means.max() - channel_means.min()),
            )
    return features


def cell_luma(features: np.ndarray) -> np.ndarray:
    """Pull the per-cell mean luma back out of a feature cube.

    The stability model only needs brightness, and this is the value
    :func:`cell_features` already computed, so recovering it here avoids walking
    the frame a second time.
    """
    if features.ndim != 3:
        raise ValueError(f"features must be grid x grid x features, got shape {features.shape!r}")
    return np.ascontiguousarray(features[:, :, CELL_FEATURE_NAMES.index("mean_luma")])


def feature_matrix(features: np.ndarray) -> np.ndarray:
    """Flatten a ``(grid, grid, F)`` cube into ``(grid * grid, F)`` rows.

    Row order is row-major, matching :func:`cell_boxes`, so a row index turns back
    into a cell with ``divmod(index, grid)``.
    """
    if features.ndim != 3:
        raise ValueError(f"features must be grid x grid x features, got shape {features.shape!r}")
    return features.reshape(-1, features.shape[2])


def features_of(frame: "Frame", grid: int) -> np.ndarray:
    """Extract features straight from a captured frame."""
    return cell_features(frame.image, grid)


def describe(features: np.ndarray) -> dict[str, Any]:
    """Return a JSON-serialisable mean of each feature, for telemetry."""
    matrix = feature_matrix(features)
    return {
        name: round(float(matrix[:, index].mean()), 4)
        for index, name in enumerate(CELL_FEATURE_NAMES)
    }
