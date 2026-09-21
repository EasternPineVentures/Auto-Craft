"""A scene model that learns where change is expected, from the run's own frames.

The problem this solves is concrete and was measured, not imagined. A 60-frame
capture of a real Luanti scene (run ``20260920T041914Z-65077437``) showed a
picture that is almost entirely frozen, with one wide animated band of liquid
spanning most of the width. That band changes on *every* frame. Everything else
changes on almost none of them.

Any single fixed threshold is wrong for that scene. Set the cutoff above the band
and genuine changes elsewhere are missed; set it below and the band is reported as
a change forever, drowning the signal. The scene, not the threshold, is the thing
that has to be described.

So this module fits a description of the scene instead. It watches a warm-up
window of frames and learns two things:

1. **How bright each cell usually is, and how much it wobbles** — so a cell that
   flickers constantly is not mistaken for a cell that just changed.
2. **What a cell's motion looks like from its appearance** — a ridge regression
   from the seven cheap per-cell features in :mod:`autocraft.perception.features`
   to that cell's observed frame-to-frame change. This is the part that makes it
   a model rather than a filter: it predicts, before a second frame exists, how
   much movement a *given patch of texture* is entitled to.

A cell is then called changed only when its brightness moves further than both
its own learned wobble *and* its own appearance-predicted allowance. A quiet,
smooth, empty cell gets a tight allowance and is sensitive; a rippling water cell
gets a wide allowance and is ignored. The bound is per-cell, so both can be true
in the same frame.

Why a linear model rather than a neural one
-------------------------------------------
The dataset that motivated this is a single run of 60 frames. A linear fit on
seven features has eight coefficients; a network has more parameters than that
run has samples, and would be fitting noise. The features were chosen to be the
things that plausibly matter, so the honest model here is the small one. If the
data later justifies something bigger, this class is the seam to replace, and
:mod:`autocraft.perception.policy` is what would consume the replacement.

The fit is deliberately kept dependency-free: numpy only, no scikit-learn, no
torch, no network access. Nothing is downloaded and nothing is pretrained — the
model is a summary of the frames this process just watched, which is measurement,
not a trained vision model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .features import CELL_FEATURE_NAMES, cell_luma

__all__ = [
    "DEFAULT_RIDGE",
    "SceneScore",
    "StabilityModel",
    "StabilitySample",
]


#: Ridge penalty applied to the appearance fit.
#:
#: Small enough not to distort a well-conditioned fit, large enough that a
#: feature which never varies in the warm-up window cannot blow the coefficients
#: up. The intercept is left unpenalised, since shrinking it would bias every
#: prediction toward zero.
DEFAULT_RIDGE: float = 1.0e-3


@dataclass
class StabilitySample:
    """One frame's contribution to the fit: appearance in, movement out."""

    features: np.ndarray
    """``(grid, grid, F)`` appearance cube."""

    movement: np.ndarray
    """``(grid, grid)`` absolute luma change from the previous frame."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "movement_mean": round(float(self.movement.mean()), 6),
            "movement_max": round(float(self.movement.max()), 6),
        }


@dataclass
class SceneScore:
    """The verdict of one frame against the fitted scene model.

    Every field is a measurement. Nothing here is a pass/fail judgement: the
    project's rule is that AutoCraft reports what it saw and the owner decides
    what it means.
    """

    index: int
    """Frame index this score belongs to."""

    changed_cells: int
    """Cells whose brightness moved past their learned allowance."""

    cell_count: int
    """Total cells in the grid, so ``changed_cells`` has a denominator."""

    total_excess: float
    """Sum over cells of how far past the allowance the brightness went.

    This is the graded quantity: a frame that barely trips a hundred cells scores
    lower than a frame that blows one cell wide open, which a bare count cannot
    express.
    """

    max_excess: float
    """The single largest overshoot, in luma levels."""

    mean_deviation: float
    """Mean absolute brightness deviation across all cells, in luma levels."""

    mean_allowance: float
    """Mean learned allowance across all cells, in luma levels.

    Worth watching on its own: if this collapses, the model has stopped expecting
    anything and will report noise as change.
    """

    excess: np.ndarray = field(repr=False)
    """``(grid, grid)`` per-cell overshoot, for a difference map."""

    deviation: np.ndarray = field(repr=False)
    """``(grid, grid)`` per-cell absolute brightness deviation."""

    allowance: np.ndarray = field(repr=False)
    """``(grid, grid)`` per-cell learned allowance."""

    grid: int = 16

    @property
    def changed_fraction(self) -> float:
        """Fraction of cells called changed, in ``0.0 .. 1.0``."""
        if self.cell_count <= 0:
            return 0.0
        return self.changed_cells / self.cell_count

    @property
    def excess_map(self) -> np.ndarray:
        """The per-cell overshoot map, as a float array."""
        return self.excess

    def peak_cell(self) -> tuple[int, int] | None:
        """``(row, column)`` of the cell that overshot most, or ``None``.

        Returns ``None`` when nothing overshot at all, which is the common case on
        a quiet frame. Callers that steer by this must treat ``None`` as "no
        reason to move" rather than as a cell at the origin.
        """
        if self.excess.size == 0 or self.max_excess <= 0.0:
            return None
        row, column = np.unravel_index(int(np.argmax(self.excess)), self.excess.shape)
        return int(row), int(column)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable summary, with no pixel data."""
        return {
            "index": int(self.index),
            "grid": int(self.grid),
            "changed_cells": int(self.changed_cells),
            "cell_count": int(self.cell_count),
            "changed_fraction": round(self.changed_fraction, 6),
            "total_excess": round(float(self.total_excess), 6),
            "max_excess": round(float(self.max_excess), 6),
            "mean_deviation": round(float(self.mean_deviation), 6),
            "mean_allowance": round(float(self.mean_allowance), 6),
        }

    def excess_ascii(self, levels: str = " .:-=+*#%@") -> list[str]:
        """Render the overshoot map as text, one string per grid row.

        Uses only ASCII so it survives a cp1252 console, and is scaled to each
        frame's own maximum so a quiet frame is still legible.
        """
        return _render(self.excess, levels)

    def deviation_ascii(self, levels: str = " .:-=+*#%@") -> list[str]:
        """Render the raw deviation map as text, for comparison."""
        return _render(self.deviation, levels)


def _render(block: np.ndarray, levels: str) -> list[str]:
    if block.size == 0:
        return []
    peak = float(np.abs(block).max())
    if peak <= 0.0:
        return [" " * block.shape[1] for _ in range(block.shape[0])]
    scaled = np.clip(np.abs(block) / peak, 0.0, 1.0) * (len(levels) - 1)
    return ["".join(levels[int(value)] for value in row) for row in scaled]


class StabilityModel:
    """Learns which parts of a scene are allowed to move, and by how much.

    Lifecycle::

        model = StabilityModel(grid=16, fit_frames=20)
        while not model.is_fitted:
            model.observe(features_of(frame))   # warm-up, no scoring yet
        score = model.score(features_of(frame)) # steady state

    Calling :meth:`score` before the fit completes raises :class:`RuntimeError`
    rather than guessing, because a model that has seen no frames has no basis for
    any allowance and would silently report everything as normal.
    """

    def __init__(
        self,
        *,
        grid: int = 16,
        fit_frames: int = 20,
        sigma: float = 4.0,
        floor: float = 2.0,
        changed_threshold: float = 8.0,
        adapt_rate: float = 0.0,
        ridge: float = DEFAULT_RIDGE,
        feature_names: Sequence[str] = CELL_FEATURE_NAMES,
    ) -> None:
        if grid < 1:
            raise ValueError(f"grid must be at least 1, got {grid}")
        if fit_frames < 2:
            raise ValueError(f"fit_frames must be at least 2, got {fit_frames}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if floor < 0:
            raise ValueError(f"floor must be non-negative, got {floor}")
        if ridge < 0:
            raise ValueError(f"ridge must be non-negative, got {ridge}")
        if not 0.0 <= adapt_rate <= 1.0:
            raise ValueError(f"adapt_rate must be between 0 and 1, got {adapt_rate}")

        self.grid = int(grid)
        self.fit_frames = int(fit_frames)
        self.sigma = float(sigma)
        self.floor = float(floor)
        self.changed_threshold = float(changed_threshold)
        self.adapt_rate = float(adapt_rate)
        self.ridge = float(ridge)
        self.feature_names = tuple(feature_names)

        self._centre: np.ndarray | None = None
        self._spread: np.ndarray | None = None
        self._coefficients: np.ndarray | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_scale: np.ndarray | None = None
        self._previous: np.ndarray | None = None
        self._samples: list[StabilitySample] = []
        self._seen = 0
        self._fit: dict[str, Any] | None = None

    # ------------------------------------------------------------------ state

    @property
    def frames_seen(self) -> int:
        """How many frames have been observed, including the warm-up."""
        return self._seen

    @property
    def is_fitted(self) -> bool:
        """Whether a full warm-up window has been observed and fitted."""
        return self._coefficients is not None

    @property
    def warmup_remaining(self) -> int:
        """Frames still needed before scoring becomes possible."""
        return max(0, self.fit_frames - self._seen)

    def reset(self) -> None:
        """Forget every frame seen and return to an unfitted state.

        Configuration is untouched. Use this when the thing being looked at has
        changed enough that the old scene is no longer the baseline - a new
        window, a different level - rather than merely drifting.
        """
        self._centre = None
        self._spread = None
        self._coefficients = None
        self._feature_mean = None
        self._feature_scale = None
        self._previous = None
        self._samples = []
        self._seen = 0
        self._fit = None

    @property
    def centre(self) -> np.ndarray:
        """The learned per-cell mean luma, or raise if not fitted."""
        self._require_fitted()
        assert self._centre is not None
        return self._centre

    @property
    def spread(self) -> np.ndarray:
        """The learned per-cell luma wobble."""
        self._require_fitted()
        assert self._spread is not None
        return self._spread

    def allowance(self, features: np.ndarray) -> np.ndarray:
        """Predict the per-cell change allowance for a frame's appearance.

        Combines two independent reasons a cell is entitled to move: it wobbles a
        lot historically (``sigma * spread``), and its texture is the kind that
        moves (the fitted regression). The floor keeps a perfectly still cell from
        producing a bound of zero, which would make it maximally sensitive to a
        single quantisation step.

        The result is clipped to be non-negative, because a regression can
        predict a negative movement for an appearance it never saw in warm-up.
        """
        self._require_fitted()
        assert self._coefficients is not None
        assert self._spread is not None
        assert self._feature_mean is not None
        assert self._feature_scale is not None

        if features.shape[0] != self.grid or features.shape[1] != self.grid:
            raise ValueError(
                f"expected a {self.grid}x{self.grid} feature grid, got "
                f"{features.shape[0]}x{features.shape[1]}"
            )

        rows = features.reshape(-1, features.shape[2]).astype(np.float64)
        normalised = (rows - self._feature_mean) / self._feature_scale
        # Coefficients are stored as (features + 1, cells): one column per cell, so
        # the shared design matrix multiplies each column independently. The dot
        # product is per-cell, hence the explicit einsum rather than `@`, which
        # would broadcast the (features, cells) operand into a cells x cells block.
        predicted = (
            np.einsum("cp,pc->c", normalised, self._coefficients[:-1])
            + self._coefficients[-1]
        )

        # Centre the allowance on the fit's own residual scale so an unusually
        # active appearance is not treated as certainly-still.
        statistical = self.sigma * self._spread.reshape(-1)
        learned = np.maximum(predicted, 0.0)
        bound = np.maximum(learned, statistical)
        bound = np.maximum(bound, self.floor)
        return np.ascontiguousarray(bound.reshape(self.grid, self.grid))

    # ------------------------------------------------------------------- fit

    def observe(self, features: np.ndarray) -> StabilitySample | None:
        """Add a frame to the warm-up window.

        The first frame only establishes a baseline, so it contributes no sample.
        Once ``fit_frames`` frames have arrived the model fits itself and becomes
        scoreable; further calls to :meth:`observe` are ignored so a long run
        cannot drift away from the scene it was fitted against.

        Returns:
            The sample contributed, or ``None`` for the first frame or after the
            fit has completed.
        """
        self._check_shape(features)
        if self.is_fitted:
            return None

        luma = cell_luma(features)
        self._seen += 1

        if self._previous is None:
            self._previous = luma
            return None

        sample = StabilitySample(
            features=np.asarray(features, dtype=np.float64),
            movement=np.abs(luma - self._previous),
        )
        self._previous = luma
        self._samples.append(sample)

        if self._seen >= self.fit_frames:
            self._fit_now()
        return sample

    def fit(self, frames: Iterable[np.ndarray]) -> "StabilityModel":
        """Convenience: observe a whole sequence of feature cubes at once."""
        for features in frames:
            self.observe(features)
        return self

    def _fit_now(self) -> None:
        if not self._samples:
            raise RuntimeError(
                "cannot fit a scene model from a single frame; at least two are needed "
                "to observe any movement at all"
            )

        movements = np.stack([sample.movement for sample in self._samples])
        appearances = np.stack([sample.features for sample in self._samples])

        self._centre = np.stack(
            [cell_luma(sample.features) for sample in self._samples]
        ).mean(axis=0)
        # A robust spread: the 90th percentile of observed movement, not the max,
        # so one stray frame cannot widen a cell's allowance permanently. Scaled by
        # 1/1.645 to approximate a standard deviation for a normal distribution,
        # which is what `sigma` is multiplied against.
        percentile = np.percentile(movements, 90.0, axis=0)
        self._spread = np.ascontiguousarray(percentile / 1.6448536269514722)

        targets = movements.reshape(len(self._samples), -1)
        rows = appearances.reshape(len(self._samples), -1, appearances.shape[-1])
        design = np.concatenate([rows, np.ones((len(self._samples), rows.shape[1], 1))], axis=2)

        self._feature_mean = rows.reshape(-1, rows.shape[-1]).mean(axis=0)
        scale = rows.reshape(-1, rows.shape[-1]).std(axis=0)
        # A feature that never varies during warm-up carries no information; give
        # it unit scale so its coefficient is shrunk to ~0 by the ridge rather
        # than producing a division by zero.
        self._feature_scale = np.where(scale > 1e-9, scale, 1.0)
        normalised = (design[:, :, :-1] - self._feature_mean) / self._feature_scale
        design = np.concatenate([normalised, np.ones((len(self._samples), rows.shape[1], 1))], axis=2)

        self._coefficients = self._solve(design, targets)
        predictions = (
            np.einsum("fcp,pc->fc", design[:, :, :-1], self._coefficients[:-1])
            + self._coefficients[-1]
        )
        residual = targets - predictions

        self._fit = {
            "frames": len(self._samples),
            "cells": int(targets.shape[1]),
            "features": list(self.feature_names),
            "movement_mean": round(float(movements.mean()), 6),
            "movement_p90": round(float(percentile.mean()), 6),
            "movement_max": round(float(movements.max()), 6),
            "spread_mean": round(float(self._spread.mean()), 6),
            "residual_rmse": round(float(math.sqrt(float((residual ** 2).mean()))), 6),
            "residual_mean_abs": round(float(np.abs(residual).mean()), 6),
            "intercept_mean": round(float(self._coefficients[-1].mean()), 6),
            "coefficient_mean_abs": [
                round(float(np.abs(self._coefficients[index]).mean()), 6)
                for index in range(len(self.feature_names))
            ],
        }

    def _solve(self, design: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Solve one independent ridge regression per cell.

        ``design`` is ``(frames, cells, features + 1)`` and ``targets`` is
        ``(frames, cells)``. Every cell shares the same design matrix, so the
        normal equations are built once and reused for all cells, with only the
        right-hand side differing. That turns ``cells`` separate least-squares
        problems into one matrix solve.

        Returns:
            A ``(features + 1, cells)`` array — one column of coefficients per
            cell, with the intercept last. Callers must respect that orientation;
            transposing it silently produces a shape error at best.
        """
        frames, cells, width = design.shape
        normal = np.einsum("fcp,fcq->pq", design, design)

        penalty = np.eye(width) * self.ridge
        penalty[-1, -1] = 0.0  # never shrink the intercept
        normal = normal + penalty

        rhs = np.einsum("fcp,fc->cp", design, targets)
        # normal @ W = rhs.T gives W with shape (features + 1, cells), which is the
        # orientation the rest of the class expects.
        try:
            return np.linalg.solve(normal, rhs.T)
        except np.linalg.LinAlgError:
            # A degenerate warm-up window (for example every frame identical)
            # leaves the system singular. Fall back to a pseudo-inverse, which
            # yields the minimum-norm answer instead of failing the run.
            return np.linalg.pinv(normal) @ rhs.T

    # ----------------------------------------------------------------- score

    def score(self, features: np.ndarray, *, index: int = -1) -> SceneScore:
        """Measure one frame against the fitted scene.

        A cell counts as changed when its brightness deviation exceeds the
        allowance predicted for its appearance. ``total_excess`` accumulates the
        size of the overshoot, so the result is graded rather than binary.
        """
        self._require_fitted()
        self._check_shape(features)

        assert self._centre is not None
        luma = cell_luma(features)
        deviation = np.abs(luma - self._centre)
        allowance = self.allowance(features)
        excess = np.maximum(deviation - allowance, 0.0)

        return SceneScore(
            index=int(index),
            changed_cells=int(np.count_nonzero(excess > 0.0)),
            cell_count=int(self.grid * self.grid),
            total_excess=float(excess.sum()),
            max_excess=float(excess.max()) if excess.size else 0.0,
            mean_deviation=float(deviation.mean()) if deviation.size else 0.0,
            mean_allowance=float(allowance.mean()) if allowance.size else 0.0,
            excess=excess,
            deviation=deviation,
            allowance=allowance,
            grid=self.grid,
        )

    # -------------------------------------------------------------- adaptation

    def update(self, features: np.ndarray, *, score: SceneScore | None = None) -> bool:
        """Fold one frame into the model, so a persistent change becomes normal.

        This is deliberately separate from :meth:`score`. ``score`` is a pure
        measurement of a frame against the model as it stands; ``update`` is the
        act of learning, and a caller that wants a fixed reference frame can
        simply never call it.

        Only the *centre* is held back for flagged cells. A cell currently sitting
        outside its allowance keeps its old baseline, so a transient change stays
        visible until it either goes away or the quiet cells around it drift to
        meet it. That asymmetry is the whole point: the model follows the scene,
        but it follows it reluctantly, and never at the moment something happens.

        The *spread*, by contrast, learns from every cell including flagged ones.
        That distinction was forced by measurement rather than chosen. In the
        first real run the animated band was still during warm-up and active
        afterwards, so a frozen spread left the model reporting the band's normal
        rippling as a change on every frame forever - the exact failure this layer
        exists to prevent. Holding the centre and learning the spread fixes both
        halves: a persistent change still stands out as a deviation from the
        anchored centre, while a recurring movement widens the allowance until it
        is recognised as ordinary. A one-off event inflates the spread only by
        ``adapt_rate`` of its own size, so a single surprise cannot desensitise a
        cell.

        Args:
            features: The frame to fold in, as a ``grid x grid x F`` cube.
            score: The result of scoring this same frame, if already computed.
                Passing it avoids a second allowance calculation; omitting it
                recomputes the score internally.

        Returns:
            ``True`` if the frame was folded in, ``False`` if adaptation is
            disabled or the model is not yet fitted.
        """
        self._require_fitted()
        if self.adapt_rate <= 0.0:
            return False

        self._check_shape(features)
        assert self._centre is not None
        assert self._spread is not None

        if score is None:
            score = self.score(features)

        luma = cell_luma(features)
        quiet = score.excess <= 0.0

        self._centre = np.where(
            quiet, self._centre + self.adapt_rate * (luma - self._centre), self._centre
        )

        if self._previous is not None:
            movement = np.abs(luma - self._previous)
            self._spread = self._spread + self.adapt_rate * (movement - self._spread)
        self._previous = luma
        return True

    def floor_share(self) -> float:
        """Fraction of cells whose allowance is currently set by the floor alone.

        A diagnostic worth reading before trusting the model. A share near 1.0
        means the learned term is contributing almost nothing and the floor is
        doing all the work, so the model is closer to a fixed threshold than its
        name suggests. That is not necessarily wrong - a genuinely static scene
        *should* sit on the floor - but it is the honest thing to report.
        """
        self._require_fitted()
        assert self._spread is not None
        statistical = self.sigma * self._spread
        return float(np.mean(statistical <= self.floor))

    # ----------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        """Serialise the fitted model, for writing to ``data/models/``.

        Only learned parameters are stored — never pixels. The file is a summary
        of a run, which is why it belongs under the data directory and not in the
        checkout.
        """
        if not self.is_fitted:
            raise RuntimeError("cannot serialise a model that has not been fitted")
        assert self._centre is not None
        assert self._spread is not None
        assert self._coefficients is not None
        assert self._feature_mean is not None
        assert self._feature_scale is not None
        return {
            "version": 1,
            "kind": "autocraft.perception.StabilityModel",
            "grid": self.grid,
            "fit_frames": self.fit_frames,
            "sigma": self.sigma,
            "floor": self.floor,
            "ridge": self.ridge,
            "adapt_rate": self.adapt_rate,
            "changed_threshold": self.changed_threshold,
            "feature_names": list(self.feature_names),
            "centre": self._centre.tolist(),
            "spread": self._spread.tolist(),
            "coefficients": self._coefficients.tolist(),
            "feature_mean": self._feature_mean.tolist(),
            "feature_scale": self._feature_scale.tolist(),
            "fit": self._fit,
        }

    def save(self, path: Path) -> Path:
        """Write the fitted model to ``path`` as JSON, creating parents."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "StabilityModel":
        """Read back a model written by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            grid=int(payload["grid"]),
            fit_frames=int(payload["fit_frames"]),
            sigma=float(payload["sigma"]),
            floor=float(payload["floor"]),
            ridge=float(payload.get("ridge", DEFAULT_RIDGE)),
            adapt_rate=float(payload.get("adapt_rate", 0.0)),
            changed_threshold=float(payload.get("changed_threshold", 8.0)),
            feature_names=payload.get("feature_names", CELL_FEATURE_NAMES),
        )
        model._centre = np.asarray(payload["centre"], dtype=np.float64)
        model._spread = np.asarray(payload["spread"], dtype=np.float64)
        model._coefficients = np.asarray(payload["coefficients"], dtype=np.float64)
        model._feature_mean = np.asarray(payload["feature_mean"], dtype=np.float64)
        model._feature_scale = np.asarray(payload["feature_scale"], dtype=np.float64)
        model._fit = payload.get("fit")
        model._seen = model.fit_frames
        return model

    def summary(self) -> dict[str, Any]:
        """A JSON-serialisable description of the model's state."""
        payload: dict[str, Any] = {
            "grid": self.grid,
            "fit_frames": self.fit_frames,
            "frames_seen": self.frames_seen,
            "fitted": self.is_fitted,
            "sigma": self.sigma,
            "floor": self.floor,
            "adapt_rate": self.adapt_rate,
            "changed_threshold": self.changed_threshold,
        }
        if self._fit is not None:
            payload["fit"] = dict(self._fit)
            payload["floor_share"] = round(self.floor_share(), 6)
        return payload

    # ----------------------------------------------------------------- guards

    def _check_shape(self, features: np.ndarray) -> None:
        if features.ndim != 3:
            raise ValueError(
                f"features must be grid x grid x features, got shape {features.shape!r}"
            )
        if features.shape[0] != self.grid or features.shape[1] != self.grid:
            raise ValueError(
                f"expected a {self.grid}x{self.grid} feature grid, got "
                f"{features.shape[0]}x{features.shape[1]}"
            )

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"the scene model is not fitted yet: {self.warmup_remaining} more frame(s) "
                "of warm-up are needed before it can score anything"
            )
