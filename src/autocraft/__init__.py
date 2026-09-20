"""AutoCraft -- an embodied game-playing agent experiment.

Long-term contract: **pixels in, human-style controls out.**

The player agent sees only rendered frames and acts only through keyboard and
mouse primitives, exactly like a human sitting at the machine. Privileged
game-state access is deliberately out of bounds for the agent; anything that
needs ground truth belongs to a future, strictly separate evaluator.

V0 provides the nervous system, not the brain:

* target-window discovery and client-area capture
* safe keyboard / mouse primitives
* focus-lock, rate limiting, emergency stop and guaranteed release
* a bounded OBSERVE -> DECIDE -> SAFETY -> ACT -> VERIFY -> RECORD skeleton
* run telemetry

Nothing in this package starts controlling the game on its own. Input injection
requires an explicit, foreground-checked command.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
