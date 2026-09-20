"""AutoCraft's expression layer.

Thoughts are *expressions*, not commands. A :class:`~autocraft.thoughts.model.ThoughtEvent`
is text the agent chose to say about what is happening to it, and there is no
path from this package to keyboard or mouse input. The specification's rule is
that a thought must never directly cause game input, and the structural reason
that holds is that nothing here can produce input: no module in this package
imports the control layer, opens a socket, starts a process or touches the
filesystem.

The flow is one-directional::

    experience -> affect changes -> thought generated -> displayed to viewer

and never::

    thought generated -> execute game action

Three pieces, deliberately separable:

* :mod:`~autocraft.thoughts.model` - the :class:`ThoughtEvent` contract and the
  :class:`ThoughtContext` a generator is allowed to see.
* :mod:`~autocraft.thoughts.generate` - :class:`ThoughtGenerator`, plus a
  fragment-composing template generator and a scripted one for demo and tests.
* :mod:`~autocraft.thoughts.engine` - :class:`ThoughtEngine`, which decides
  *whether* to speak: cooldown, per-minute ceiling, scheduled silences, and a
  probability the affect state influences but does not dictate.

A language model can replace the generator later without touching anything else,
because the only thing the engine needs from it is
``generate(context) -> ThoughtEvent | None``.
"""

from __future__ import annotations

from .engine import DEMO_POLICY, ThoughtEngine, ThoughtPolicy, scripted_engine
from .generate import (
    SCRIPTED_GENERATOR_NAME,
    TEMPLATE_GENERATOR_NAME,
    ScriptedThoughtGenerator,
    TemplateThoughtGenerator,
    ThoughtGenerator,
)
from .model import (
    THOUGHT_TONES,
    THOUGHT_TRIGGERS,
    ThoughtContext,
    ThoughtError,
    ThoughtEvent,
    ThoughtTone,
    ThoughtTrigger,
)

__all__ = [
    "DEMO_POLICY",
    "SCRIPTED_GENERATOR_NAME",
    "TEMPLATE_GENERATOR_NAME",
    "THOUGHT_TONES",
    "THOUGHT_TRIGGERS",
    "ScriptedThoughtGenerator",
    "TemplateThoughtGenerator",
    "ThoughtContext",
    "ThoughtEngine",
    "ThoughtError",
    "ThoughtEvent",
    "ThoughtGenerator",
    "ThoughtPolicy",
    "ThoughtTone",
    "ThoughtTrigger",
    "scripted_engine",
]
