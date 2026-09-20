"""Learned, self-fitted perception: the agent working out what it is looking at.

This package exists because a measurement said a fixed threshold cannot work.
A 60-frame capture of a real Luanti scene contained one wide animated band of
liquid that changes on every single frame, and a mostly frozen world everywhere
else. No single cutoff separates those two things. A per-cell model of the scene
can, because it learns that the band is *expected* to move.

What lives here:

* :mod:`autocraft.perception.features` turns a frame into cheap per-cell
  appearance numbers.
* :mod:`autocraft.perception.stability` fits a model of the scene and scores new
  frames against it, with a per-cell allowance derived from appearance.

Three rules hold the package together, and all three are enforced by tests rather
than by good intentions:

1. **Pixels in, measurements out.** Nothing here reads game state, memory, or
   files inside the game. The only input is a captured frame.
2. **No actuation.** The package never imports :mod:`autocraft.control`, so it
   cannot move the mouse or press a key even by accident. It observes.
3. **No verdicts.** Every number produced is a measurement. Whether a number means
   "interesting" is the owner's call, not this package's.

The models here are fitted online from frames this process just watched. Nothing
is downloaded, nothing is pretrained, and the only numerical dependency is numpy.
That matters for the project's contract: this is a summary of a run, not a
trained vision model imported from elsewhere.
"""

from __future__ import annotations

from .features import CELL_FEATURE_NAMES, cell_boxes, cell_features, cell_luma
from .features import describe, feature_matrix, features_of, iter_cells
from .policy import PerceptionDecisionPolicy
from .session import FrameSource, PerceptionReport, PerceptionSession, SessionRow
from .stability import DEFAULT_RIDGE, SceneScore, StabilityModel, StabilitySample

__all__ = [
    "CELL_FEATURE_NAMES",
    "DEFAULT_RIDGE",
    "FrameSource",
    "PerceptionDecisionPolicy",
    "PerceptionReport",
    "PerceptionSession",
    "SceneScore",
    "SessionRow",
    "StabilityModel",
    "StabilitySample",
    "cell_boxes",
    "cell_features",
    "cell_luma",
    "describe",
    "feature_matrix",
    "features_of",
    "iter_cells",
]
