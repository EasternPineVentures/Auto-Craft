# AutoCraft

An experimental project to build an agent that learns to play a Minecraft-like
voxel survival game by **looking at the screen** and **using the keyboard and
mouse** — the same way a person does.

AutoCraft V0 is deliberately small. It contains no learning, no perception
model, and no game knowledge. It is the nervous system: find the game window,
see the pixels, and move the hands — safely.

---

## The long-term contract

> **Pixels in, human-style controls out.**

The agent may only ever use:

| Direction | Allowed |
|---|---|
| **Input to the agent** | Frames rendered by the game window |
| **Output from the agent** | Keyboard events, mouse movement, mouse buttons |

The agent must **never** use privileged game state. Specifically, it may not
read or use:

- player coordinates or orientation
- inventory, hotbar, health, or hunger values
- "nearest entity / nearest tree / nearest block" APIs
- the game's internal object model
- the world database or map files
- server commands, chat commands, or teleportation
- helper mods written to make the agent's job easier

`AGENT != EVALUATOR`. A future scoring system is allowed to know the truth
about the world in order to grade a run. The agent itself is not. If the agent
cannot perceive it through pixels, it does not know it.

That constraint is the whole point of the project. An agent with coordinates
and a `find_nearest_tree()` function has not learned to play a game; it has
learned to call a function. AutoCraft exists to find out how far the harder
version gets.

---

## What V0 actually does

V0 is a foundation, not a player. It implements stages A–J of the initial
specification:

| Stage | Capability |
|---|---|
| A | Locate the Luanti / VoxelLibre window on Windows |
| B | Determine the **client area** bounds (excluding title bar and borders) |
| C | Capture frames from the client area only |
| D | Save a frame to disk for human inspection |
| E | Safe keyboard primitives (press, release, tap, release-all) |
| F | Safe relative mouse-movement primitives |
| G | Mouse-button primitives |
| H | `OBSERVE → DECIDE → SAFETY → ACT → VERIFY → RECORD` loop skeleton |
| I | Basic experiment telemetry (one directory per run) |
| J | Safety controls: focus lock, emergency stop, guaranteed key release |

The decision layer is a **`NoOpDecisionPolicy`**. It always chooses to do
nothing. There is no AI in V0, on purpose: the loop skeleton had to be
trustworthy before anything is allowed to plug into it.

---

## The first real benchmark: TREE-001

Not attempted in V0. It is recorded here so the foundation can be judged
against the thing it has to support.

Starting from an unknown position in a normal world:

1. Observe the scene.
2. Look around to search for a tree.
3. Visually identify a tree from pixels alone.
4. Rotate toward it.
5. Approach it.
6. Stop at a sensible distance.
7. Centre the crosshair on the trunk.
8. Hold the attack/dig input.
9. Break exactly one log.
10. Visually verify that the world changed.

Every one of those steps must be done from pixels and human controls. No
coordinates, no object detection hints, no privileged state.

---

## What AutoCraft is NOT doing yet

Explicitly out of scope for V0, and mostly out of scope for the project's
near future:

- no LLM or any model API integration
- no reinforcement learning
- no object detector, no YOLO, no trained vision model
- no crafting logic
- no navigation or pathfinding
- no survival logic (no hunger, health, or combat behaviour)
- no tree recognition
- no privileged game-state access of any kind
- no autonomous survival play
- no agent frameworks (LangChain or similar)
- no vector databases, Docker, web dashboard, Electron, React, or databases
- no cloud services, multiplayer, or chat integrations
- no game mods, and no modification of the game installation

---

## Architecture

The boundaries below are the reason the code is split the way it is. Each one
can be replaced without touching the others.

```
GAME          Luanti / VoxelLibre  (never modified, never inspected internally)
  |
SENSORS       vision/     screen capture -> Frame (a numpy BGRA array)
  |
AGENT         agent/      Observation -> Decision -> Action
  |
ACTUATORS     control/    Action -> real keyboard / mouse events
  |
SAFETY        control/safety.py   gates every actuator call
  |
EVALUATOR     (future)    scores runs using privileged truth the agent never sees
```

```
src/autocraft/
  config.py              frozen Config: window patterns, limits, directories
  cli.py                 the seven commands
  vision/
    window.py            Win32 window discovery + client-area geometry
    capture.py           mss-backed client-area capture
    frame.py             ScreenRegion + Frame (block means, difference, save)
  control/
    keymap.py            key names <-> virtual key codes
    win32_input.py       SendInput via ctypes (scan codes)
    safety.py            SafetyGuard: focus lock, e-stop, release-all, limits
    keyboard.py          keyboard primitives
    mouse.py             mouse primitives
  agent/
    observation.py       Observer: the only path from pixels into the agent
    action.py            the human-input vocabulary + ActionExecutor
    decision.py          DecisionPolicy protocol + NoOpDecisionPolicy
    loop.py              the bounded OBSERVE..RECORD loop
  telemetry/
    recorder.py          RunRecorder: one directory per run
```

---

## Setup

**Requirements**

- Windows 10 or 11 (window discovery and input injection are Win32-specific)
- Python 3.11 or newer
- Luanti 5.17+ or VoxelLibre, installed and running

**Install**

```powershell
git clone https://github.com/EasternPineVentures/Auto-Craft.git
cd Auto-Craft
python -m pip install -e ".[dev]"
```

**Dependencies, and why each one is here**

| Package | Why |
|---|---|
| `mss` | Fast, dependency-light screen capture. No GUI toolkit dragged in. |
| `opencv-python` | Only used to write PNGs and resize. Kept to a single image library. |
| `numpy` | Frames are arrays. All the frame maths is vectorised. |
| `pytest` (dev) | The test suite. |

Window discovery and input injection use `ctypes` against Win32 directly —
there is no PyAutoGUI, no pywin32, and no GUI-automation framework. Those
libraries are convenient, but for a first-person game they add latency,
swallow input, or capture the wrong surface. Doing it directly keeps the
behaviour inspectable and the failure modes honest.

**Configuration**

Copy `autocraft.example.toml` to `autocraft.toml` and edit. Every safety
limit has a documented default and a comment explaining it; none of them are
hidden magic numbers.

---

## Commands

Run everything with `python -m autocraft <command>`. Looking is always safe.
Touching the game is always explicit.

| Command | Sends input? | What it does |
|---|---|---|
| `status` | no | Reports whether the game window was found, whether it is foreground, and its client-area geometry. |
| `capture` | no | Saves exactly one client-area frame. Prints the path and dimensions. |
| `observe` | no | Observes for a bounded time. Reports frames captured, achieved FPS, dropped/failed frames, and focus state. |
| `loop` | no | Runs the bounded agent loop with the no-op policy. |
| `input-test` | **yes** | Explicit smoke test. Requires `--yes`. Sends one tiny bounded action and then releases everything. |
| `keys` | no | Lists every supported key name. |
| `config` | no | Prints the effective configuration and where it came from. |

```powershell
python -m autocraft status
python -m autocraft capture
python -m autocraft observe --seconds 5 --steps 60
python -m autocraft loop --steps 5 --seconds 10
python -m autocraft keys
```

The loop refuses to start unless you give it `--steps` or `--seconds`. There
is no "run forever" option, by design.

---

## Safety

The rules below are enforced in code, not by convention.

**Foreground-window lock.** Input is injected only while the intended game
window is the foreground window. If focus moves to anything else — you alt-tab
away, a dialog appears — every held key and button is released and the loop
stops. AutoCraft will not type into your editor because a window appeared.

**Emergency stop.** Press **F8** (configurable) to stop immediately. Held
inputs are released and the run ends. AutoCraft polls the key with
`GetAsyncKeyState` rather than registering a global hotkey, so the key is not
swallowed and the game still sees it.

Honest limitation: this is a polled check, not an OS-level interrupt. It is
evaluated between actions, so it can be delayed by at most one action. A truly
immediate kill switch would need a separate low-level keyboard hook or a
helper process; that is not implemented in V0.

**Release guarantee.** Every key and button is tracked while held. Everything
is released on a normal exit, on an exception, on `Ctrl+C`, on focus loss, and
on emergency stop. A failed release leaves the input tracked as still held so
the next release-all retries it, and `atexit` runs a final release.

**Bounded actions.** Key holds are clamped to a maximum duration. Mouse
movement is clamped to a maximum delta per command. Deltas above the limit are
**rejected, not silently clamped** — the agent's intent is never quietly
rewritten into a different action.

**Rate limiting.** Actions are paced by a minimum interval. Rate limiting
never consumes the "too many consecutive failures" budget; it waits instead of
refusing, so pacing cannot masquerade as a safety stop.

**No autonomy by default.** `status`, `capture`, `observe`, `loop`, `keys`,
and `config` never inject input. Only `input-test` can, and it requires an
explicit `--yes`, states clearly what it is about to do, verifies the game
window is foreground, sends exactly one small action, releases it, and exits.

**AutoCraft never controls the game process.** It reads pixels and window
geometry, and it can emit key presses and mouse events. It does not close,
minimise, focus, move, resize, or terminate any window, and it sends no window
messages. The only events it can emit without being asked are key-*up* and
button-*up* releases, which exist so nothing is left held. If the game closes
while AutoCraft is running, AutoCraft did not close it.

---

## Tests

```powershell
python -m pytest -q
```

The suite runs without Luanti and **never injects real input**. Hardware and
game dependencies sit behind small backend protocols (`WindowBackend`,
`CaptureBackend`, `InputBackend`), which the tests replace with fakes. What is
covered:

- window geometry and client-area transformations
- target-window matching and foreground detection
- action validation and the input vocabulary
- mouse-delta limits
- key state tracking and release-all behaviour
- foreground-window rejection
- the no-op decision policy
- bounded agent loop behaviour, including capture failure
- telemetry serialisation
- the CLI safety gate: `input-test` refuses before any input object exists
- the capture-rate warning

The suite is 273 tests and runs in about 1.5 seconds. Everything that talks to
the real OS is exercised manually, through the commands above.

---

## Telemetry

Each run gets its own directory:

```
data/runs/<run-id>/
  steps.ndjson     one JSON object per step, streamed as the run progresses
  run.json         the final summary, written when the run finishes
  frames/          saved frames, only if frame saving was requested
```

`<run-id>` is a UTC timestamp plus a short random suffix, so runs sort
chronologically and two runs in the same second cannot collide.

Raw pixels are never serialised into the JSON. A step references a saved
frame by path instead, which keeps the records small enough to read.

---

## Known limitations

These are measured on the development machine, not estimates.

**Capture cost scales with the client area, and it competes with the game.**
A full-screen grab at 3840x1950 costs roughly 400 ms, which caps observation at
about 2.4 fps against a 10 fps target. The same grab on a 1280x650 window costs
about 18 ms, which clears 10 fps comfortably. This is a property of desktop
capture, not of AutoCraft's loop. Because the grab occupies the desktop, it also
starves the game's own event loop: Luanti logs `Irrlicht: SDL_PollEvent took too
long` when this happens, and the visible symptom is stutter. `observe` now warns
when it misses its target by more than 25% and prints the remedies. Run the game
in a smaller window than full-screen for anything interactive.

**The emergency stop is a polled check, not an OS-level interrupt.** AutoCraft
reads `GetAsyncKeyState` for the configured key. It is deliberately not a
registered global hotkey, because a registered hotkey would swallow the key and
the game would stop receiving it. The consequence is honest but real: the key is
noticed between steps, so a step that blocks inside the capture or input backend
delays the response. A guaranteed asynchronous kill switch needs extra OS work
(a low-level keyboard hook, or a watchdog process) and is not implemented.

**Input injection has not been exercised against the game.** See the Tests
section: nothing in the test suite sends real input, and `input-test` refuses to
run without an explicit `--yes`. The SendInput path is therefore verified by
construction and by unit tests against a fake backend, not by having actually
moved a character.

---

## Where this goes next

**VISION-001 / LOOK-001** is the recommended next milestone: capture a
continuous stream of gameplay frames while deliberately rotating the camera,
and measure what actually changes on screen. Before any of TREE-001 can be
attempted, the project needs to know what "looking around" does to pixels, and
how much of a frame is stable while the camera moves.

That work is not started. V0 stops at a foundation that is boring, inspectable,
and trustworthy.
