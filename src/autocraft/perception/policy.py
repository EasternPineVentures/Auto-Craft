"""A decision policy that steers by what changed in the picture.

This is the layer that makes the scene model *do* something. It plugs into
:class:`~autocraft.agent.decision.DecisionPolicy`, the project's existing seam
between seeing and acting, so it inherits the loop's bounds and the safety
guard's vetoes rather than bypassing them.

What it does is deliberately narrow. Each step it looks at the fitted scene model's
overshoot map and, if some cell is doing something the model did not expect, it
aims the camera a little way toward that cell. Nothing else. It has no notion of
goals, objects, or success; it is a reflex, not a plan.

Two rules it will not break:

* **It never acts before the model is fitted.** During warm-up it returns
  ``noop``, because a policy that moves before it can see is just noise with a
  mouse attached.
* **It never proposes more than ``max_mouse_delta``.** The safety guard would
  clamp it anyway; proposing a value the guard must reject would make the
  telemetry lie about what the policy asked for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..agent.action import Action
from ..agent.observation import Observation
from ..vision.frame import Frame
from .features import cell_features
from .stability import SceneScore, StabilityModel

__all__ = ["PerceptionDecisionPolicy"]


@dataclass
class PerceptionDecisionPolicy:
    """Move toward whatever the scene model found surprising.

    Args:
        model: The scene model to fit and score with.
        max_mouse_delta: Hard ceiling on the proposed delta, in pixels. Should be
            the same value the safety guard enforces.
        gain: Fraction of the way to the target cell to travel in one step. The
            default is deliberately timid: the mouse-to-camera mapping has not
            been measured yet (that is what LOOK-001 exists to do), so the policy
            must not assume a large delta produces a large camera movement.
        adapt: Whether to keep adapting the model while acting. On by default;
            without it the model accumulates the drift caused by its own motion.
        name: Policy name, reported by the loop.
    """

    model: StabilityModel
    max_mouse_delta: int = 20
    gain: float = 0.25
    adapt: bool = True
    name: str = "perception-glimpse"

    #: Cells the last :meth:`decide` call flagged, for reporting.
    last_score: SceneScore | None = field(default=None, init=False)
    #: ``(row, column)`` of the last cell the policy aimed at, if any.
    last_target: tuple[int, int] | None = field(default=None, init=False)
    #: How many decisions so far produced a movement rather than a no-op.
    moves: int = field(default=0, init=False)
    #: How many decisions were made at all.
    decisions: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.max_mouse_delta = max(0, int(self.max_mouse_delta))
        self.gain = max(0.0, min(1.0, float(self.gain)))

    # -- DecisionPolicy ---------------------------------------------------

    def decide(self, observation: Observation) -> Action:
        """Return the next action for ``observation``.

        Always an ``Action``; never ``None`` and never a non-integer delta.
        """
        self.decisions += 1

        frame = observation.frame
        if frame is None:
            self.last_score = None
            self.last_target = None
            return Action.noop()

        features = cell_features(frame.image, self.model.grid)

        if not self.model.is_fitted:
            self.model.observe(features)
            self.last_score = None
            self.last_target = None
            return Action.noop()

        score = self.model.score(features, index=observation.index)
        if self.adapt:
            self.model.update(features, score=score)
        self.last_score = score

        target = score.peak_cell()
        if target is None or self.max_mouse_delta <= 0:
            self.last_target = None
            return Action.noop()

        self.last_target = target
        dx, dy = self._delta_to(target, frame)
        if dx == 0 and dy == 0:
            return Action.noop()
        self.moves += 1
        return Action.mouse_move(dx, dy)

    def reset(self) -> None:
        """Forget the scene and start over.

        Resetting the model is the honest meaning of resetting this policy: a
        policy whose knowledge is the model has not reset if the model persists.
        """
        self.model.reset()
        self.last_score = None
        self.last_target = None
        self.moves = 0
        self.decisions = 0

    # -- geometry ---------------------------------------------------------

    def _delta_to(self, cell: tuple[int, int], frame: Frame) -> tuple[int, int]:
        """Aim the camera from the picture's centre toward ``cell``.

        The offset is measured in cell units and scaled by the grid, so the result
        is independent of the capture resolution.
        """
        row, column = cell
        grid = self.model.grid
        centre = (grid - 1) / 2.0
        # A positive dx should move the view right, so the target's own offset is
        # the direction to travel; the picture follows the camera.
        offset_x = column - centre
        offset_y = row - centre

        reach = self.max_mouse_delta * self.gain
        dx = int(round(offset_x / centre * reach)) if centre > 0 else 0
        dy = int(round(offset_y / centre * reach)) if centre > 0 else 0

        limit = self.max_mouse_delta
        dx = max(-limit, min(limit, dx))
        dy = max(-limit, min(limit, dy))

        if dx == 0 and dy == 0:
            # The target is off-centre but rounds to nothing. Nudge along whichever
            # axis is furthest off so the policy makes progress instead of
            # stalling one step short forever.
            if abs(offset_x) >= abs(offset_y) and offset_x != 0:
                dx = 1 if offset_x > 0 else -1
            elif offset_y != 0:
                dy = 1 if offset_y > 0 else -1
        return dx, dy

    # -- reporting --------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "decisions": self.decisions,
            "moves": self.moves,
            "max_mouse_delta": self.max_mouse_delta,
            "gain": self.gain,
            "fitted": self.model.is_fitted,
        }
        if self.last_score is not None:
            payload["last_changed_cells"] = self.last_score.changed_cells
            payload["last_total_excess"] = round(self.last_score.total_excess, 6)
        if self.last_target is not None:
            payload["last_target"] = list(self.last_target)
        return payload
