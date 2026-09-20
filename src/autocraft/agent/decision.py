"""Decision layer: observation in, action out.

V0 ships exactly one policy, :class:`NoOpDecisionPolicy`, which always chooses to
do nothing. That is not a placeholder to be embarrassed about - it is what makes
the whole pipeline demonstrably safe to run: the loop, telemetry, safety and
capture paths can all be exercised end to end without AutoCraft ever touching the
game.

The :class:`DecisionPolicy` protocol is the seam where real behaviour will
eventually arrive (visual servoing, then something learned). Keeping it a
one-method protocol means a policy can be swapped without the loop, the safety
guard or the actuators knowing anything changed.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .action import Action
from .observation import Observation

__all__ = ["DecisionPolicy", "NoOpDecisionPolicy"]


@runtime_checkable
class DecisionPolicy(Protocol):
    """A replaceable mapping from observation to intended action."""

    name: str

    def decide(self, observation: Observation) -> Action:
        """Choose an action for this observation.

        Implementations must not inject input, read game state, or block for long
        periods; they only return an intent that the executor and safety guard
        then vet.
        """
        ...

    def reset(self) -> None:
        """Drop any per-run state so the policy can be reused."""
        ...


class NoOpDecisionPolicy:
    """Always chooses ``noop``.

    This is the default in every code path in V0. AutoCraft cannot start playing
    the game by accident, because the only shipped policy never asks it to.
    """

    name = "noop"

    def decide(self, observation: Observation) -> Action:
        """Return a no-op action, ignoring the observation."""
        return Action.noop()

    def reset(self) -> None:
        """No state to clear."""
        return None
