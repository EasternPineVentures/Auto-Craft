# AutoCraft

An experimental project to build an agent that learns to play a Minecraft-like
voxel survival game by **looking at the screen** and **using the keyboard and
mouse** — the same way a person does.

AutoCraft V0 is deliberately small. It contains no game knowledge and no
learning of its own: the `loop` command's decision policy always chooses to do
nothing. It is the nervous system: find the game window, see the pixels, and
move the hands — safely. The one thing added on top of that foundation is a
scene model that learns what each part of the picture normally does, so that
"the picture did not move" can be told apart from "the picture moved by less
than I can see". It is read-only, it is not wired into the loop, and it is
documented in full under
[The one exception: the scene model in `perception/`](#the-one-exception-the-scene-model-in-perception).

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

The decision layer the `loop` command uses is a **`NoOpDecisionPolicy`**. It
always chooses to do nothing. There is no AI in the loop, on purpose: the loop
skeleton had to be trustworthy before anything is allowed to plug into it.

Since then the owner lifted the LLM/model-API prohibition for one narrow purpose,
and `perception/` now exists: a scene model fitted online from the run's own
frames, plus a `PerceptionDecisionPolicy` that implements the same
`DecisionPolicy` protocol. It is read-only, it is not wired into `loop`, and the
distinction is set out in full below — see
[The one exception: the scene model in `perception/`](#the-one-exception-the-scene-model-in-perception).

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
- no vector databases, Docker, Electron, React, or databases
- no cloud services, multiplayer, or chat integrations
- no game mods, and no modification of the game installation
- no world map, no long-term memory expansion, no story engine
- no multi-agent system, no simulated society, no economy
- no pass/fail verdict on a measurement, and no "the camera moved" claim

The local observer page is a deliberate, narrow exception to "no web
dashboard", and it is drawn tightly: it is read-only, it is loopback-only, it
has no build step, no framework, and no dependency the agent does not already
need. See [The observer page](#the-observer-page).

### The one exception: the scene model in `perception/`

The project owner lifted the **LLM / model-API** prohibition so that the
perception layer could be built. It is worth being precise about what was
lifted, because the list above is otherwise still in force.

**What `perception/` is.** It fits a small statistical model of the scene from
the run's *own* frames. For each cell of a 16x16 partition it learns a robust
centre, a robust spread, and a ridge regression from seven cheap appearance
features to the cell's observed frame-to-frame movement. All of it is
computed at run time by the process itself, in numpy, on frames it just
captured. Nothing is downloaded, nothing is pre-trained, and no network call is
made.

**What it is not.** It is not a trained vision model: there are no weights
shipped, nothing was trained offline on a corpus, and the "model" is a handful
of per-cell means and a 8x256 coefficient matrix that is thrown away at the end
of a run unless you ask for it to be saved. It contains no object detector, no
classifier, and no notion of what anything on screen *is*. It answers one
question — "does this cell look like the cell I learned?" — and nothing else.

**Why it is measurement rather than capability.** The distinction the list is
protecting is between *measuring the instrument* and *giving the agent new
powers*. This layer is the former: it is the thing that makes the question "did
the camera actually move?" answerable at all, which is exactly what the
LOOK-001 trial could not settle. It is read-only, it is in the same
`AGENT != EVALUATOR` position as `vision/`, and it has no code path that
reaches `control/` — pinned by an import-structure test, not by convention.

Everything else on the list, including "no reinforcement learning" and "no
object detector, no trained vision model", is unchanged and unplanned.

---

## Architecture

The boundaries below are the reason the code is split the way it is. Each one
can be replaced without touching the others.

```
GAME          Luanti / VoxelLibre  (never modified, never inspected internally)
  |
SENSORS       vision/     screen capture -> Frame (a numpy BGRA array)
  |
PERCEPTION    perception/ Frame -> per-cell scene model -> SceneScore
  |
AGENT         agent/      Observation -> Decision -> Action
  |
ACTUATORS     control/    Action -> real keyboard / mouse events
  |
SAFETY        control/safety.py   gates every actuator call
  |
EVALUATOR     (future)    scores runs using privileged truth the agent never sees

                    ---- published state, one direction only ----

DISPLAY       observer/   ObserverSnapshot -> HTTP -> browser
  ^
EXPRESSION    thoughts/   ThoughtEvent (generated, never acted on)
```

The last two boxes are deliberately **below** the arrow, not beside it.
`observer/` and `thoughts/` read published facts; nothing in them can reach
`control/`. That is enforced by import structure and pinned by tests.
`perception/` sits in the same position: it is a sensor, not an actuator, and
the same test forbids it from importing the control layer.

```
src/autocraft/
  config.py              frozen Config: window patterns, limits, directories
  cli.py                 the nine commands
  vision/
    window.py            Win32 window discovery + client-area geometry
    capture.py           mss-backed client-area capture
    frame.py             ScreenRegion + Frame (block means, difference, save)
  perception/
    features.py          per-cell appearance: seven features + the cell partition
    stability.py         StabilityModel: the learned scene model + SceneScore
    session.py           PerceptionSession: a frame source through the model
    policy.py            PerceptionDecisionPolicy: a DecisionPolicy over the model
  look/
    metrics.py           frame maths: difference, shift, reversibility
    record.py            the LOOK-001 result schema
    runner.py            the fixed nine-step trial sequence
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
  thoughts/
    model.py             ThoughtEvent, ThoughtContext, the tone vocabulary
    generate.py          ThoughtGenerator protocol + template / scripted sources
    engine.py            ThoughtPolicy: cooldown, rate ceiling, quiet periods
  observer/
    snapshot.py          the display contract (pure data, no I/O)
    state.py             ObserverState: the agent's write side, the page's read side
    server.py            GET-only HTTP surface on loopback
    bridge.py            LoopPublisher: the one place the agent meets the page
  telemetry/
    recorder.py          RunRecorder: one directory per run
  web/
    index.html           the instrumentation page
    styles.css           the dark theme
    app.js               vanilla-JS polling client, no build step
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
| `opencv-python` | Writes and resizes PNGs, and provides the phase correlation LOOK-001 measures with. Kept to a single image library. |
| `numpy` | Frames are arrays. All the frame maths is vectorised. |
| `pytest` (dev) | The test suite. |

Window discovery and input injection use `ctypes` against Win32 directly —
there is no PyAutoGUI, no pywin32, and no GUI-automation framework. Those
libraries are convenient, but for a first-person game they add latency,
swallow input, or capture the wrong surface. Doing it directly keeps the
behaviour inspectable and the failure modes honest.

### Working from a git worktree

`pip install -e .` writes one machine-wide `.pth` file naming the `src/`
directory it was run in. If you keep several worktrees of this repository, that
file points at whichever one you installed from last, and `python -m autocraft`
resolves there from *any* directory — including a different worktree whose
branch may not have the command you are typing. The symptom is a confusing
`invalid choice: 'look-test' (choose from status, capture, ...)` for a command
that plainly exists on your branch.

The quickest way to see which checkout you are actually running:

```powershell
python -c "import autocraft; print(autocraft.__file__)"
```

If that names a different worktree than the one you are standing in, that is
your answer. Two ways to keep it straight:

```powershell
# Point this shell at the checkout you actually mean, for this session:
$env:PYTHONPATH = "C:\path\to\this\worktree\src"

# Or make one worktree the canonical install, and run from there:
python -m pip install -e ".[dev]"
```

Set `PYTHONPATH` as its own command, on its own line. Pasting it together with
the `python -m autocraft` line makes PowerShell treat it as a *continuation* of
that command — you will see a `>>` prompt, the assignment is swallowed as extra
arguments, and the variable is never set.

The test suite is unaffected either way: `pyproject.toml` sets
`pythonpath = ["src"]`, so pytest always imports the worktree it is running in.
Only `python -m autocraft` is affected.

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
| `input-test` | **yes** | Explicit smoke test. Prints the bounded action, requires `--yes`, gives you 10 seconds (`--focus-delay`) to focus the game, re-verifies that exact window is foreground, sends one tiny bounded action and then releases everything. |
| `look-test` | **yes** | LOOK-001. Injects one known relative mouse movement, captures the picture before it, after it, and after the exact reverse, and reports the measured pixel displacement and its reversibility. Requires `--yes`; every plan is bounded. |
| `perceive-test` | no | VISION-001. Observes for a bounded time, learns what each cell of the scene normally does from the run's own frames, then reports per-cell change as it goes. Never sends input and has no `--yes` flag. |
| `wake-test` | **yes** | WAKE-001. Looks around with bounded mouse movements, picks the most visually salient region it can find, centres it in the view, and stops. Requires `--yes`; every plan is bounded and the run always terminates. |
| `keys` | no | Lists every supported key name. |
| `config` | no | Prints the effective configuration and where it came from. |
| `observer` | no | Serves the local read-only observer page. Never enables control. |

```powershell
python -m autocraft status
python -m autocraft capture
python -m autocraft observe --seconds 5 --steps 60
python -m autocraft observe --seconds 30 --steps 300 --observer
python -m autocraft loop --steps 5 --seconds 10
python -m autocraft perceive-test --seconds 30
python -m autocraft wake-test --yes
python -m autocraft observer --demo
python -m autocraft keys
```

The loop refuses to start unless you give it `--steps` or `--seconds`. There
is no "run forever" option, by design.

`--observer` is available on `observe`, `loop`, `look-test` and `wake-test` and
does exactly one thing: it starts the page and publishes that run's state to it.
It does not change what the command does, and it never enables input. If the
port is already busy, the run continues and the page is simply unavailable.

### The first input smoke test

This is the only command that touches the game, and the only one where the
order of operations matters:

```powershell
python -m autocraft input-test --action key-tap --key w --hold 0.05 --yes
```

1. Run it from a shell in this repository, with the game already running.
2. It prints exactly what it intends to send, and refuses outright if the
   target window is missing or minimised.
3. It then starts a countdown. **Click into the game during the countdown.**
   The shell is the foreground window when you launch the command — that is
   expected, and it is exactly why the foreground check happens *after* the
   handoff rather than before it.
4. Immediately before sending, it re-queries the window and requires that
   exact window to be foreground. If focus is wrong, nothing is sent and the
   command exits non-zero. Re-running is harmless.
5. Press `F8` at any time to abort and release everything.

**Expect exactly one brief input and then nothing.** `input-test` sends a single
bounded action and exits; it is a smoke test that proves the actuator reaches
the game, not a controller. `--hold 0.05` is a 50 ms tap, so the character takes
one short step. To see something more legible, lengthen the hold — `--hold 1.0`
walks for a full second (holds are clamped to `max_key_hold_seconds`, 2.0 by
default). Re-run the command as many times as you like; each run is one action.

The countdown is 10 seconds by default. If that is tight on a first run, allow
more time with `--focus-delay 20`, or pass `--focus-delay 0` to skip the wait
entirely once you know the game already has focus.

Leaving off `--yes` is a safe dry run: it prints the action, prints the exact
command to repeat it for real, and exits with code `3` without ever
constructing any injection machinery.

### The mouse smoke test

The other actuator class has its own variant, which turns the camera rather than
moving the character:

```powershell
python -m autocraft input-test --action mouse-move --dx 10 --dy 0 --yes
```

Same ordering and the same rules. `dx`/`dy` are **relative** counts, not screen
coordinates, so display scaling and DPI awareness cannot distort them. Deltas
above `max_mouse_delta` (200 by default) are refused rather than clamped, so a
typo fails loudly instead of yanking the view around.

### The LOOK-001 measurement

`input-test` proves the actuator reaches the game. `look-test` asks the next
question: **when the pointer moves by a known number of counts, how far does the
picture move, and does moving back put it back?** That is a measurement, not a
capability — it is the number TREE-001 would need before anything can be aimed.

```powershell
python -m autocraft look-test --dx 200 --dy 0 --steps 1 --yes
```

One trial is a fixed nine-step sequence, and nothing about it is adaptive:

1. Verify the target window is foreground.
2. Capture frame **A**.
3. Send the outbound movement (`--dx`, `--dy`).
4. Wait `--settle` seconds (config `look_settle_seconds`, 0.15 by default).
5. Capture frame **B**.
6. Send the exact reverse movement (`-dx`, `-dy`).
7. Wait `--settle` seconds again.
8. Capture frame **C**.
9. Measure and record. Nothing is decided.

`--steps N` repeats that sequence N times in N separate directories, which is
how you see whether the measurement is repeatable. Focus is re-verified before
*every* movement, and the guard re-checks it again inside the actuator call.

#### What it reports

- **A→B and A→C mean absolute luma difference, RMSE, and changed-pixel
  fraction**, plus a `block_grid` × `block_grid` map (config `look_block_grid`,
  8 by default) so you can see *where* the frame changed rather than only that
  it did.
- **Estimated pixel shift**, from OpenCV phase correlation, in pixels, with the
  response value that came with it.
- **Pixels per delta**, the mapping ratio on each axis. `null` on an axis you
  did not move, rather than `0` or `inf`.
- **Reversibility ratio**: the outbound A→B displacement divided by the A→C
  displacement that should undo it. `1.0` means the reverse returned the picture
  to where it started. It is `null` when A and B were indistinguishable, because
  the ratio would be a division by nothing.

The quality number that comes back from phase correlation is deliberately **not**
normalised to `0..1`. It is a relative response: read it as an ordering (this
match is stronger than that one), never as a score with a meaning of its own.

It is, however, used for one decision. Phase correlation fits a translation and
only a translation, so a camera that has **turned** leaves it without a peak — and
it still returns a near-zero vector, which reads as "the camera did not move".
Below a response of `0.5` the shift is therefore reported as *not available* with
the reason, and `pixels per delta` is `null` on both axes rather than a ratio
divided by a number nobody measured. The floor is measured, not chosen: on frames
this project captures, a real translation of a textured scene responds
`0.93–1.00`, two unrelated frames respond `0.02`, and a 5° turn responds `0.17`.

That floor answers "does one translation explain this pair". It does **not**
answer "is there anything worth looking at" — two featureless frames agree
perfectly on the translation `(0, 0)` and pass it.

#### The first live trial

`look-test` has now been run once against Luanti 5.17.0, recorded as run
`20260920T034854Z-e339cbaa`: one trial at `(+10, +0)` in a 3222x1928 client
area. That is the `--dx 10` example this section used to document, and the run is
the reason the example is now `--dx 200`.

The mechanism worked end to end. Two movements were sent, three frames were
captured, the record was written, and `released inputs: none`. Reversibility was
excellent: A→C differs by 0.033 luma levels against 4.655 for A→B, and
`difference_ac.png` is 15 KB against 1.4 MB for `difference_ab.png`.

**The mapping was not measured.** The 4.655-luma A→B difference is not a camera
pan. It is 0.910% of pixels, in two compact clusters at the top-right corner and
the left edge, whose 1D profiles disagree about the shift and whose brightness
gain is about 1.0 — an overlay flickering, not the scene moving. Measuring the
textured content directly is decisive: the bottom band (rows 1440–1928, patch
standard deviation 19–53) spans the frame, and phase correlation across it
returns `dx 0.00, dy 0.00` at quality 0.93–0.998 for every patch — left, centre
and right alike, and 0.60 at the extreme right edge still gave `dx 0.04`. A
translation would show a constant non-zero shift; a camera yaw would show a shift
that varies with `x`. Neither is present. Relative to each other, the frames did
not move.

So 10 counts moved the picture by less than this method can resolve — under about
0.02 px on this content. That is a statement about the size of the injected
delta, not about the instrument: the same run measured its own control, A against
C, at zero shift with quality up to 1.000.

The consequence is the opposite of "the mapping is tiny". A 10-count delta moving
the picture by less than 0.02 px implies a sensitivity well below anything a game
ships, which makes it more likely that the camera look was never engaged — Luanti
applies mouse motion to the camera only while the pointer is captured. A large
delta settles which it is, which is why the calibration series ascends to
`max_mouse_delta`.

#### The second live trial

The first trial left one question open: would a 200-count delta also move the
picture by nothing, which would mean the camera look was never engaged? It would
not.

`input-test --action mouse-move --dx 200 --dy 0` sent the movement and reported
`attempted: True`, `executed: True` in 0.001 s, and the camera visibly swung. That
is the direct delivery confirmation the first trial could not supply: a difference
that is small says nothing about *why* it is small.

`look-test --dx 200 --dy 0 --steps 1` was then run as `20260921T044151Z-9c47b3d5`
against a 2102x1061 client area. It changed the picture enormously:

| measurement | trial 1 (`--dx 10`) | trial 2 (`--dx 200`) |
|---|---|---|
| A→B mean absolute difference | 4.655 luma levels | **31.335** |
| A→B changed pixels | 12.98% | **57.12%** |
| A→C mean absolute difference | 0.033 | **0.222** |
| reversibility | 0.007 | 0.007 |

Two conclusions follow, and neither depends on the shift estimate. **The actuator
works**: 200 counts moved the picture far more than 10 counts did. And **the change
was the mouse's**: A→C returned to within 0.222 luma levels of A, and the clouds
drifting across that sky do not reverse.

**The mapping still was not measured, and now we know why.** The same trial
reported `estimated shift (-0.10, +0.05) px` at quality `0.196`, and a
`pixels per delta x` of `-0.0005`. A 57% pixel change cannot be a
`-0.0005 px per count` camera. Phase correlation fits a translation and only a
translation; the response of `0.196` is the estimator saying it found no such
translation, and the vector printed next to that response meant nothing. So the
trial's own numbers contradicted each other — a large change, a fully reversible
one, and a displacement of approximately zero — and the displacement was printed
anyway.

It is not printed any more. A response below `0.5` is reported as an unavailable
shift with the reason, and `pixels per delta` is `null` rather than a ratio
computed from a number the estimator had already declined to stand behind.

The best-supported reading of what the camera actually did is that it **turned**.
The picture changed by a large amount, came back exactly, and no single
translation explains the pair — which is what a camera turning looks like, and is
precisely the motion a translation estimator cannot fit. How many degrees a count
turns it is not yet measured, because nothing here estimates rotation yet.

Two limits on these numbers. `--dx 200` is far above LOOK-001's design point: it
demonstrates that input reaches the camera, not what the mapping is. And the run
was at 2102x1061 — about 2.2 million pixels, well above the ~1 million the
experiment is designed around — so the frame is larger than it should be and the
estimate is correspondingly less precise.

#### Calibration

The mapping is not assumed to be linear, so `look-test` will run a bounded
series instead of a single delta:

```powershell
python -m autocraft look-test --calibrate-horizontal --yes
python -m autocraft look-test --calibrate-vertical --yes
```

The series is config `look_calibration_deltas` (`[5, 10, 25, 50, 100, 200]` by
default), one trial per delta, and it is capped by config `look_max_steps` (10 by
default). A series longer than that bound is refused outright, so editing the
config cannot produce an unbounded run.

A series has to bracket the answer, so the default one ascends to
`max_mouse_delta` — the largest delta the actuator will accept in a single
command. Every entry also goes through that same per-axis bound, and the whole
series is refused if any entry exceeds it. The actuator would reject an oversized
delta anyway, but only after a frame had already been captured, which would leave
a row of failed trials and nothing to say why.

#### Bounds and refusals

Every plan is finite; there is no unbounded mode. These are all usage errors
(exit `2`), refused before the window layer is consulted:

| Input | Why it is refused |
|---|---|
| `--dx 0 --dy 0` | Nothing moved means nothing could be measured. That is not a measurement of zero. |
| `--dx`/`--dy` above `max_mouse_delta` | Refused, not clamped — the same rule as `input-test`. |
| A calibration series with an entry above `max_mouse_delta` | Refused up front, rather than failing one trial at a time mid-run. |
| `--steps 0` or above `look_max_steps` | The trial count is capped by config. |
| `--settle` negative, `nan`, `inf`, or above 10s | A settle is a bounded wait, not a delay you can make arbitrary. |
| `--focus-delay` negative, `nan` or `inf` | Rejected before anything else happens. |

Leaving off `--yes` is a dry run: it prints the whole plan, the window geometry,
the exact command to repeat it for real, and exits `3` without constructing any
injection machinery or creating the output directory.

#### What it does not do

- It does **not** resize, move, or reconfigure the game window. A command that
  reconfigures the thing it is measuring would change the measurement. If the
  client area is large it says so and stops; resize it yourself.
- It does **not** apply a pass/fail threshold. There is no `is_good()`, no
  score, no verdict. It prints numbers and writes them down.
- It does **not** claim anything understands the camera. The mapping is
  measured from the mouse delta, never assumed from it.

#### Files

Everything lands in `data/runs/look/` (or `--output PATH`):

```
look_result.json    the numbers, the plan, the settings, and the limitation notes
frame_a.png         before the movement
frame_b.png         after the outbound movement
frame_c.png         after the exact reverse
difference_ab.png   |A - B|, the same map the block grid is computed from
difference_ac.png   |A - C|
trial-00/ …         one directory per trial when --steps is above 1
```

Raw pixels are never serialised into the JSON; the record references frames by
filename. A trial whose frames cannot be compared — the window was resized
mid-trial, so A and B have different shapes — is recorded as `failed` with the
reason, and it keeps its raw frames rather than inventing a difference image for
a comparison that could not be made.

---

### The VISION-001 scene model

`look-test` measures the actuator against the picture. `perceive-test` asks the
question underneath it: **what does this scene normally do, and did anything
actually change?** Without an answer, "the picture did not move" and "the picture
moved by less than I can see" are the same observation — which is exactly the
hole the first LOOK-001 trial fell into.

```powershell
python -m autocraft perceive-test --seconds 30
```

It never sends input, it has no `--yes` flag, and it has no dry-run mode because
there is nothing to authorise. The test suite goes further than the flag: it
replaces `Keyboard`, `Mouse`, `ActionExecutor`, `SafetyGuard` **and** the guard
factory with hard failures, so a `perceive-test` run that reaches exit `0` has
proved that no input machinery was even constructed.

#### Why a fixed threshold cannot do this

The obvious implementation is "a pixel changed if it differs by more than *N*".
That was measured against 60 real frames of the live game first, and it does not
work:

- The scene is **almost entirely static**. Only 3.12% of pixels ever differ by
  more than 8 luma levels, and they sit in one wide horizontal band
  (`y 1521..1693`, `x 619..2692`).
- That band is **animated texture cycling**, not noise. Some consecutive frames
  are *pixel-identical*; others differ by a maximum of *exactly* 77.0 in 0.0029%
  of pixels. A fixed cutoff either ignores the animation or calls it a change,
  depending on which frames it happens to compare.
- The scene's overall brightness **drifts** — mean luma went 15.927 → 16.290 over
  50 frames in the quiet recorded set, and 41.05 → 73.44 over 40 seconds in one
  live run. Any global threshold is chasing that drift.

On the recorded set, a flat `8.0` cutoff measured against the fitted centre
reported a mean of **0.70** changed cells (range 0–1) — it **misses the animated
band entirely**. Measured frame-to-frame instead it sees *nothing at all*: across
all 59 consecutive pairs, **zero** cells ever differ by more than 8.0, because the
animation cycles past in steps smaller than that. The learned model, on the same
frames, reported a mean of **16.50** (range 1–21) and localized the band to rows
12–13, columns 3–13 of a 16x16 grid, with **zero** false positives across the
other 87% of the grid.

It is worth being blunt about what that does and does not prove. With
`floor_share` at 0.996, the model's answer *on this recording* is a flat `2.0`
cutoff with extra bookkeeping — the learned spread and the regression term
contribute almost nothing, so the same 16.50 comes out of the floor alone. What
the recorded set proves is that a fixed cutoff chosen by hand fails: `8.0` misses
the band, and `2.0` works, and nothing in the data says which of those two you
are holding. What sets the bound is what the model is for; whether it sets it
*well* is what the two live runs below are evidence about, and the answer there is
not flattering.

#### What it learns

For every cell of the 16x16 partition (config `perception_grid`), from a warm-up
window of frames (config `perception_fit_frames`, 20 by default):

- a robust **centre** — the mean luma the cell normally shows;
- a robust **spread** — the 90th percentile of the cell's observed frame-to-frame
  movement, divided by 1.6449 so it reads as a standard deviation;
- a per-cell **ridge regression** from seven cheap appearance features
  (`mean_r`, `mean_g`, `mean_b`, `mean_luma`, `std_luma`, `edge_energy`,
  `colour_spread`) to the movement that cell is observed to make.

A cell counts as changed when it exceeds
`max(floor, sigma * spread, predicted_movement)` — config `perception_floor`
(2.0 luma levels) stops a cell that happened to be perfectly still during
learning from getting a bound of zero, and `perception_sigma` (4.0) is the
multiplier on the learned spread.

Once fitted, the model **adapts**: on each frame an *unchanged* cell's centre
drifts toward what it currently sees, at config `perception_adapt_rate` (0.05 per
frame). That is what lets it follow slow lighting change instead of calling it a
change forever. A cell that is currently **flagged** keeps its centre — so real
motion cannot desensitise the model — but its *spread* still learns from every
frame, because a cell that is repeatedly surprising should get a wider idea of
normal, while one surprise should not.

#### What it reports

- **Warm-up versus steady.** A run shorter than `perception_fit_frames` never
  leaves the learning phase. It says so, reports `complete: false`, and still
  writes its measurement — "the warm-up never finished" is a result.
- **Per-frame change**: how many of the 256 cells changed, what fraction of the
  grid that is, and the total and maximum excess above each cell's own allowance.
- **An accumulated excess map** — where change kept happening across the whole
  run, rendered as ASCII on the terminal and as a 16x16 grid in the JSON. This is
  the part that answers "did anything move", and it is why the animated band is
  visible as a band rather than as a count.
- **The model's own fit numbers**, including `floor_share`: the fraction of cells
  whose bound came from the floor rather than from anything learned. A high
  `floor_share` means the model learned almost nothing and is running on the
  floor.
- **No verdict.** There is no pass/fail threshold, no `is_good()`, and no claim
  that anything was recognized. It prints numbers and writes them down.

#### The two live runs, and why they disagree

`perceive-test` has been run twice against Luanti 5.17.0, both with the game
focused and neither sending input. **The two runs disagree sharply, and that
disagreement is the finding.** Quoting them both is more useful than quoting
either.

| | Recorded set (unfocused, 60 frames) | Live run 2 (`…90360bf0`, 40 s) | Live run 1 (`…34371370`, 45 s) |
|---|---|---|---|
| warm-up | 20 frames | 20 frames | 20 frames |
| scored frames | 40 | 36 | 45 |
| `floor_share` | 0.996 | 0.000 | — |
| fitted `movement_mean` | 0.0023 | 21.30 | — |
| fitted `spread_mean` | 0.0035 | 33.33 | — |
| changed cells per frame (mean) | 16.50 | 0.22 | 5.29 |
| changed cells (range) | 1–21 | 0–4 | 0–55 |
| total excess (mean / max) | — | 1.78 / 43.12 | 70.77 / 833.02 |

Three things are visible in that table and none of them is a bug to hide:

**The model learns whatever the warm-up window contains.** In the quiet recorded
set the warm-up saw almost no movement, so `floor_share` was 0.996 — 99.6% of
cells were running on the floor bound rather than on anything learned, and the
model then reported 16.5 changed cells per frame against a scene that was
essentially static. That is an honest over-report: with nothing learned, the
floor is doing all the work.

**The live run's warm-up captured a transient.** Live run 2 fitted
`movement_mean` 21.30 and `spread_mean` 33.33, which produced a mean allowance of
**70.0** luma levels — far above the 13.2 mean deviation it was actually seeing.
With an allowance that wide, the same scene scores 0.22 changed cells per frame.
The model is not wrong; it was calibrated against 20 frames that happened to
contain a lot of motion.

**And the scene itself genuinely changed mid-run.** This is the more important
half. Over live run 2, mean luma rose **41.05 → 73.44**, `std_luma` 13.29 →
19.74, and `colour_spread` 15.58 → 33.97. That is a large, systematic change in
what the camera was looking at, *after* the model had already been fitted. A
longer warm-up would not fix this: the model's answer to it is the `adapt_rate`
centre drift, and 0.05 per frame at the capture rate this machine achieves
(about 2.4 fps at 3222x1928) is far too slow to track a change that size.

So the model's weakest points are named and measured: **the warm-up window can
catch a transient, and the adaptation rate cannot follow a scene that changes
this much.** Both are configuration, both are recorded in every run's JSON, and
the accumulated excess map is what makes either visible. The next design step is
to fit against a median-of-medians warm-up and to warn when the fitted
`movement_mean` looks like a transient; neither is implemented yet.

#### Files

Everything lands in `data/runs/<run-id>/`:

```
perceive_result.json   summary, timeline, model fit, and the accumulated map
steps.ndjson           one JSON object per frame, streamed as the run goes
```

The result file is written **before** any of the pretty-printing, and it is
written for incomplete runs too. A live run once lost 45 seconds of measurement
to a formatting error in the reporting path; that ordering is now pinned by a
test that makes the printing fail on purpose and asserts the numbers still
reached disk.

A fitted model can also be saved and reloaded (`StabilityModel.save` /
`load`), which is what `data/models/` is for. Nothing writes there
automatically, the directory is gitignored, and a model that has not been fitted
refuses to serialise rather than writing an empty file.

---

### The WAKE-001 first behaviour

```powershell
python -m autocraft wake-test --yes
```

VISION-001 can tell *that* the scene changed. WAKE-001 is the first behaviour
built on top of that: **wake up, look around, notice a visually interesting
region, turn toward it, centre it in the view, and stop.**

It is deliberately not TREE-001. There is no object detector, no semantic
class, and no privileged game state anywhere in this path. The thing it finds is
a *visually salient region*, and that is the only thing the code or the output
will ever call it. Nothing here knows what a tree is.

#### What it actually does

1. **Looks.** A fixed ring of bounded relative mouse movements, each followed by
   a fresh capture. Every view is fingerprinted and stored in a small bounded
   memory, so the agent can notice it is looking at somewhere it has already
   been.
2. **Scores.** Each frame is divided into a grid of cells and each cell is scored
   with deterministic, local evidence: texture, edge density, colour spread,
   local contrast, and how much it differs from its neighbours. Cells are merged
   into a small candidate set, at most `wake_max_target_candidates` of them. A
   merged region that covers the whole frame is refused rather than offered:
   a region the size of the frame has no surroundings to be salient *against*,
   and no room to be located *in*.
3. **Chooses.** Candidates are ranked by
   `salience + novelty + persistence - recently_seen_penalty - excessive_distance_penalty`.
   The weights are a named, documented constant
   (`SelectionWeights`), not numbers buried in the ranking code.
4. **Centres.** A closed loop: measure the offset from the crosshair, move a
   bounded number of counts, re-capture, re-locate, adjust. Coarse band first,
   then medium, then fine. Every correction is sized from the *latest* frame.
5. **Stops.** On being centred inside the dead zone, or on confidence falling
   below the floor, or on a budget running out, or on a safety abort. All four
   paths terminate the run and are recorded with the reason that ended it.

Every verification capture — the one after a scan movement and the one after a
centring movement alike — waits `wake_verify_settle_seconds` before it looks,
because the game draws a movement asynchronously and a capture taken immediately
samples the frame from *before* it. See
[the third round](#the-third-round-correct-geometry-and-a-sensor-that-looked-too-early).

#### The mapping is measured, not assumed

The centring controller never assumes `1 mouse count = 1 pixel`. It starts with
a fixed band count and **self-measures**: every movement is followed by a fresh
capture, and the observed shift divided by the counts sent becomes a sample in a
`MotionCalibration`. Once an axis has three samples the controller switches to
`source: self-measured` and sizes its corrections from the ratio it actually
observed. Until then it says `source: unmeasured` and every record, every
console summary and every panel says so too.

The LOOK-001 calibration can be adopted as a starting point, but it is not
required and it is never invented: `mapping_quality` is reported as `null` when
there was no measurement, not as a confident number.

#### Anti-repetition, because repetition is the failure mode

The spec is explicit that the third identical failed strategy must not be tried
blindly. Two mechanisms enforce that:

- **A progress model.** Distance from the crosshair is assessed on every
  centring step and classified as improved or not against an epsilon. It tracks
  `dead_repetition_ratio` and `productive_repetition_ratio` separately, so
  164 → 97 → 43 → 11 px reads as *productive* while 114 → 114 → 114 reads as
  *dead*.
- **A repetition guard.** A signature is built from the view, the strategy and
  the movement, and the guard counts how many times in a row it has been seen
  *without progress*. On the third, it raises `STUCK_PATTERN_DETECTED` and the
  policy **changes strategy** — it forces a smaller movement band — rather than
  adding random jitter. A strategy that keeps failing goes on a cooldown and
  becomes unavailable; it can come back if the visual situation changes.

Randomness exists in exactly one place, the candidate tie-break in
`select_candidate`, it is seeded in tests, and it can never override a safety
rule or a progress decision.

#### Bounded, always

The behaviour can never run away. Each of these is a configured ceiling, and
hitting one ends the run rather than extending it:

| Bound | Default | What it stops |
|---|---|---|
| `wake_max_scan_moves` | 12 | looking around forever |
| `wake_max_target_candidates` | 3 | one run chasing every region in the frame |
| `wake_max_center_moves` | 8 | grinding on a target that will not centre |
| `wake_max_moves` | 45 | the whole run's movement budget |
| `wake_max_seconds` | 120 | wall clock |

`max_moves` must be at least `max_scan_moves + max_center_moves`; a configuration
that would let one phase starve the other is rejected at load time.

#### Configuration

```toml
[wake]
salience_grid = 8            # cells per side for the salience map
view_grid = 8                # cells per side for the view fingerprint
scan_counts = 60             # mouse counts per scan step
max_scan_moves = 12
max_target_candidates = 3
max_center_moves = 8
max_moves = 45
max_seconds = 120.0
dead_zone_px = 12.0          # "centred enough" radius
min_target_confidence = 0.35 # below this, do not pretend to know where it is
verify_settle_seconds = 0.15 # wait for the game to redraw before VERIFY looks
```

Every field has a matching `AUTOCRAFT_WAKE_*` environment override, and all of
them appear in every run's `wake_result.json`.

#### Files

```
data/runs/<run-id>/wake_result.json   what was measured, and what it is not
```

`wake_result.json` carries the metrics, the event stream in order, the
per-target detail, and the mapping provenance. It never contains raw pixels, and
it never contains a pass/fail verdict — the numbers are there to be read, not
scored. Four standing limitation notes travel with every record, so the file
cannot be read out of context.

The record is rewritten after **every** step rather than only at the end. A run
that is killed mid-way leaves behind what it had measured, and it cannot be
mistaken for a finished run because its status is still `running`.

The window size in the record is taken from the first frame that actually
arrived, not from the plan: what the agent looked at is a measurement. Every
observation additionally records the window handle, the client-area size and the
captured-frame size, so a disagreement between the plan and the frame is visible
in the record instead of being averaged away — see
[the second live run](#the-window-was-two-different-sizes-in-one-run).

#### Known weaknesses

- **The mouse mapping is unmeasured until a run measures it.** LOOK-001
  established that relative mouse input moves the camera, but the pixels-per-count
  ratio is only weakly constrained: a 200-count probe moved the view so far that
  no shared features survived to fit a transform to. A run therefore begins by
  not knowing how far a count will move the view, and the first centring
  corrections are sized by a fixed band count. Self-measurement fixes this within
  a few movements — but only if the target stays findable.
- **The floor on the measured ratio is a real limit.** `_MIN_PIXELS_PER_COUNT`
  is 0.05. If the true ratio were smaller than that, the 200-count hardware
  ceiling would mean no bounded number of corrections could ever close a
  large offset. That is deliberate — it fails honestly instead of sending
  thousands of counts — but it is a genuine inability, and it is recorded rather
  than hidden.
- **Salience is not semantics.** The chosen region is whatever has the most
  local structure, novelty and persistence. On a busy scene it will sometimes
  pick something a human would not.
- **A low-contrast frame used to yield the whole frame as its one region.** The
  first live run found this the hard way (see the section below). Each salience
  cue is normalised by its own frame maximum, so on a frame with almost no
  variation the little there is gets inflated to fill the range, every cell
  clears the threshold, and the map's single group spans the entire frame. A
  region the size of the frame is not a region and cannot be tracked — it leaves
  the matcher exactly one legal position, so it "matches" itself wherever it is
  assumed to be, at a confidence of 1.0, and the distance to it can never change.
  This is now refused in three places rather than tuned around: `find_candidates`
  and `candidate_from_cells` decline a box that covers the frame, and `refine`
  and `_refine_scaled` decline a search region that offers the template only one
  position. The refusal is geometric, not a size cap, so a large-but-bounded
  region is still located normally. The honest consequence is that a sky-only
  scene now scans, finds nothing, and says so, instead of "successfully" centring
  the sky.
- **Re-acquisition is shallow.** If a target is lost mid-centring, the agent
  gets `wake_max_center_moves` attempts to find it again by searching outward.
  If it stays lost, the target is abandoned and the run ends rather than
  wandering.

#### The first live run, and what it exposed

The first live WAKE-001 run happened on the operator's machine against Luanti
5.17.0, in a 3222x1928 client area:

```powershell
python -m autocraft wake-test --focus-delay 20 --yes
```

Run `20260920T063805Z-7b7a82a2`; the measurement is
`data/runs/wake/wake_result.json`. The run failed, and the operator's own
description of it is the most valuable artefact this milestone produced:

> Looked around, picked a candidate, then it seemed like it started going down
> and looking at the ground. Not centring anything. It stopped early after
> repeated attempts. Then the terminal threw a crash after it stopped.

Both halves of that report are reproducible from the record, and neither is a
tuning problem. This section is the forensic pass: what was recorded, what was
**not** recorded, and the two defects the run proved.

The run is preserved exactly as it happened. Nothing here was rewritten to make
the run look better, and the fix described below is a general defect fix, not a
change made until this one run looked right.

##### What the record says, in the run's own words

`run.json` for this run is complete: `status: "stopped"`, `step_count: 16`,
`executed_steps: 16`, `blocked_steps: 0`, **`safety_events: []`**, `errors: []`,
`duration: 40.27 s`, `stop_reason: "policy stop (every one of the 3 candidate(s)
attempted was abandoned; last reason: the same correction was attempted 3 times
with no measurable progress)"`. So the run was recorded cleanly by the loop
before anything else went wrong. **There were no safety events at all** in this
run — nothing was ever blocked, rate-limited, or emergency-stopped.

The console printed 41 lines. Counted by kind, they are:

| kind | count |
|---|---|
| `WAKE_STARTED` | 1 |
| `VIEW_REVISITED` | 1 |
| `VIEW_CAPTURED` | 2 |
| `CANDIDATE_FOUND` | 2 |
| `SCAN_MOVE` | 1 |
| `NEW_VIEW_OBSERVED` | 1 |
| `TARGET_SELECTED` | 3 |
| `CENTERING_PROGRESS` | 14 |
| `STUCK_PATTERN_DETECTED` | 5 |
| `STRATEGY_CHANGED` | 7 |
| `STRATEGY_FAILED` | 3 |
| `WAKE_ABORTED` | 1 |

The 14 `CENTERING_PROGRESS` events correspond to the 14 centring moves; the 15th
mouse move is the single scan move.

##### Every movement the run made

Reconstructed from `steps.ndjson` (16 rows) and the event stream. `t` is seconds
since step 0's `started_at`. `distance` is the target's distance from the frame
centre as reported by the policy. `move` is the policy's own centring-move
counter, which resets whenever a new target is selected.

| # | t (s) | dx | dy | band | distance | conf | move | what else happened at this step |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | +60 | 0 | — | — | — | scan 1 | the scan move; no target yet |
| 1 | 1.651 | +20 | +20 | medium | 216.4 | 1.000 | 1 | target A selected: centre (1788.0, 1088.6) |
| 2 | 2.789 | +20 | +20 | medium | 216.4 | 0.640 | 2 | |
| 3 | 3.937 | +6 | +6 | fine | 216.4 | 0.550 | 3 | stuck: 3x (+20,+20) medium; band forced to fine |
| 4 | 5.086 | +6 | +6 | fine | 216.4 | 0.646 | 4 | |
| 5 | 6.259 | +6 | +6 | fine | 216.4 | 0.664 | 5 | |
| 6 | 8.364 | −6 | −6 | fine | 130.4 | 0.902 | 1 | stuck; target A failed; target B selected: centre (1533.5, 859.1) |
| 7 | 9.476 | −6 | −6 | fine | 130.4 | 0.686 | 2 | |
| 8 | 11.526 | +20 | +20 | medium | 216.4 | 0.846 | 1 | stuck; target B failed; target A selected again |
| 9 | 12.647 | +10 | +10 | medium | 216.4 | 0.479 | 2 | |
| 10 | 13.818 | +20 | +20 | medium | 216.4 | 0.710 | 3 | the only step whose VERIFY frame registered as changed |
| 11 | 14.976 | +20 | +20 | medium | 216.4 | 0.635 | 4 | |
| 12 | 16.124 | +6 | +6 | fine | 216.4 | 0.655 | 5 | stuck; band forced to fine |
| 13 | 17.294 | +6 | +6 | fine | 216.4 | 0.815 | 6 | |
| 14 | 18.451 | +6 | +6 | fine | 216.4 | 0.692 | 7 | |
| 15 | 19.593 | — | — | — | — | — | — | stuck; target A failed again; run aborted, no move sent |

Two things are visible in that table before any interpretation. **Every centring
move has `dx == dy`** — not one is purely horizontal or vertical. And **the
reported `distance` never changes within a target**: 216.4 for target A across
eleven separate moves, 130.4 for target B across three. The policy believed it
was making progress fourteen times and made none.

##### What the record does *not* contain

The record cannot answer several of the questions this table invites, and it is
better to say so than to infer:

- **per-step target centre, before and after** — missing. Only the run-level
  `final_target_offset` and the centres inside the three `TARGET_SELECTED`
  events exist.
- **per-step target distance before *and* after** — only the single `distance`
  the policy used to size the move is recorded, not the measured distance on the
  far side of it. That is why "no progress" has to be inferred from the
  *constancy* of the number rather than read off two numbers.
- **per-step state, strategy and stuck status** — missing from `steps.ndjson`.
  They appear only inside the corresponding events, which is a different file.
- **per-step progress classification** — missing. Only the run-level
  `dead_repetition_ratio: 1.0` and `productive_repetition_ratio: 0.0` exist.
- **per-step candidate identity** — missing. The three `TARGET_SELECTED` events
  carry the centres, and the run-level `target_changes: 2` counts the switches,
  but a step cannot be joined to the candidate it was acting on.
- **per-step mapping source** — missing. It is run-level and constant:
  `mapping.source: "unmeasured"`.
- **frame images** — none survived. `save_frames_every` defaults to `0` and
  `wake-test` exposes no flag for it, so every step's `capture_path` is `None`.
  The only frame that left a trace is step 0's `observation.frame` payload:
  3222x1928, `mean_luma` 137.132, screen rect (615, 85)-(3837, 2013).

The record also cannot say *how far* a mouse count moves the view, and does not
pretend to: `mapping` is `{"pixels_per_delta_x": null, "pixels_per_delta_y":
null, "quality": null, "source": "unmeasured"}`.

##### Why the camera drifted downward

The drift is not a vertical bias, and it is not the scene. It is a single
defect with a chain that the record and an offline reproduction both confirm.

**1. The salience map is normalised per frame, so a low-contrast frame is
inflated.** `salience_map` divides each cue by its own frame maximum
(`_normalise`). On the sky-and-ground view the run was looking at, the raw
spread is small, so the normalisation stretches it.

**2. The threshold is relative to that peak, so on a low-contrast frame every
cell passes it.** `find_candidates` thresholds at
`max(min_salience, peak_fraction * peak)` = `max(0.15, 0.45 * peak)`. On a
synthetic low-contrast sky the peak comes out at 0.9657, so the threshold is
0.4346 and **all 64 of 64 cells clear it**. They merge into one group whose
bounding box is the entire frame. All three `TARGET_SELECTED` events in the live
run carry exactly that:

```json
{"bbox": [0, 0, 3222, 1928], "centre": [1788.0, 1088.6], "confidence": 1.0, "salience": 0.3399}
```

**3. A frame-sized candidate makes relocation a fixed point.** `_locate` builds
`TargetPatch.of(frame, candidate.bbox)`. With a full-frame bbox **the patch is
the whole frame**. Correlating a whole-frame template against a whole-frame
region leaves `matchTemplate` exactly one legal position, so the match is always
at the patch's own origin and `_relocation_of` returns `patch.centre` unchanged
— with `confidence 1.0`, because a perfect self-match is a perfect match. The
"found" location is not a measurement of the target; it is the target's own
assumed position echoed back. Reproduced from scratch offline: `locate` on a
full-frame patch returns exactly the frame centre `(1611.0, 964.0)` at
`confidence 1.0`, while the same call on a properly sized 320x240 patch returns
the correct `(1560.0, 820.0)`.

**4. So the offset never changes, and the loop cannot converge.** The target
centres are `(1788.0, 1088.6)` and `(1533.5, 859.1)`; the frame centre is
`(1611.0, 964.0)`. The offsets are therefore `(+177.0, +124.6)` (distance
216.4) and `(−77.5, −104.9)` (distance 130.4) — which are exactly the run-level
`final_target_offset: [177.0, 124.6]` and exactly the distances in the movement
table. **The centre never moved once.** Distance stayed at 216.4, the progress
model scored every attempt `0.0`, the repetition guard fired every third
attempt, and the run walked through three candidates and 15 moves to a truthful
`failed`.

**5. And the diagonal is a consequence, not a cause.** With no measured
pixels-per-count ratio, `_step_counts` sizes each axis **independently** by the
band's fixed count, so any offset with two non-zero components yields
`|dx| == |dy|`. That is why every correction in the table is a perfect
diagonal, and why the reported `direction` is always `"right"` or `"left"` —
`direction_of` names an axis from the dominant component, and the vertical half
of every one of those moves is invisible in the report. `(+20, +20)` is not a
"right" correction. It is a right-and-down correction, fourteen times in a row,
which is the downward drift the operator saw.

**What this rules out.** The vertical-calibration hypothesis is not merely
unsupported, it is impossible here: `_centre` feeds
`note_observation(shift_x=offset_x − previous_offset_x, ...)`, and because the
relocation is a fixed point that shift is exactly `0` every time.
`MotionCalibration.observe` rejects any sample whose ratio is not `> 0`, so no
sample is ever taken and the calibration stays `unmeasured` for the whole run.
There was never a bias to have. A HUD or hotbar region is a real secondary
attraction on a busy frame, but it is not what happened here — the candidate was
the whole frame, HUD included.

##### The fix: a region that covers the frame is not a region

The defect is fixed at the source, in `wake/salience.py`, by refusing the two
degenerate shapes rather than by tuning any threshold. Two small predicates do
it:

- `_spans_frame(bbox, width)` — true when a box spans the frame's width.
  `find_candidates` skips such a group, and
  `candidate_from_cells` returns `None` for it. This is the upstream fix and the
  clean statement of the defect: a region that wide has no surroundings, so it is
  not a local feature of the scene at all, and its "salience" is the frame's
  global contrast wearing a region's clothing. It is untrackable for the same
  reason — it occupies every column the matcher can search, so the only position
  it can be found at is the one it was told to look at.
- `_has_search_room(region_width, region_height, patch_width, patch_height)` —
  false when a search region offers the template only one legal position.
  `refine` and `_refine_scaled` return `None` in that case. This is the general
  rule the first one is a special case of, and it protects every caller of the
  matcher, not just the policy: **a template always matches itself perfectly, so
  a lone position is not a location.** A score cannot detect this — the
  self-match scores 1.0 — so it is refused by geometry instead. The rule allows
  room along *either* axis, because a region exactly as wide as the patch can
  still say whether the target moved up or down.

Both predicates are deliberately narrow, and the tests pin that narrowness. A
region larger than the frame is refused too, and a large-but-bounded 120x90 region
in a 160x120 frame is still found and still relocated. The rule is about spanning
the frame, not about size.

The width test is a later extension of the original rule, which required all four
edges and so let a full-width band through on the reasoning that it "is a real
feature". The live WAKE-001 run `20260921T045955Z-4b6dd88b` retired that reasoning.
It selected `[0, 266, 2102, 1061]`, `[0, 0, 2102, 665]` and `[16, 300, 2118, 1095]`
out of a 2102x1061 frame — every one of them the full frame width — and measured
the same unusable mapping from each: 0.0075 px per mouse count on y, so the
largest movement the safety cap allows displaced the view by 1.5 px against a
12 px dead zone. A target that wide has no horizontal position left to correct
with, which is why the mapping it yields is not a mapping at all. Height is
deliberately *not* tested. A tall narrow region still carries a usable horizontal
position, and no live run has offered one.

There is no scene-specific threshold anywhere in the fix, no "never look down",
no hardcoded direction, and no change to `BAND_COUNTS`, the salience weights, the
progress epsilon or the repetition guard. The three candidates, the 15 moves, the
constant 216.4 px distance and the diagonal are all consequences of the target
never moving; fixing the target fixes them.

The honest consequence is a different and better failure. On a synthetic
sky-only scene the old code selected the full frame, ran 14 centring moves at a
distance that never changed, and reached `WakeState.COMPLETE` — it believed it had
centred the sky. The same scene now scans, finds nothing, and ends `FAILED` with
`"nothing visually salient was found in 12 scan movement(s)"`: zero
`TARGET_SELECTED` events, zero `CENTERING_PROGRESS` events, `candidate_count 0`.
A truthful "there was nothing to look at" is the outcome this milestone is
supposed to produce.

Six tests cover it, using the panning `FakeWorld` scene and a synthetic gradient
sky. Five of them fail against the pre-fix code — the frame-sized candidate, the
direct `candidate_from_cells` call, the over-broad-region boundary, the
frame-sized patch through both matcher passes, and the sky-only run. The sixth,
`test_a_bounded_region_is_still_located`, passes both before and after, and
exists precisely so that over-refusal cannot be mistaken for a fix.

##### Why the motion was choppy

Measured over the 15 mouse moves, decomposing each step at
`action.created_at`, `result.started_at`/`result.finished_at` and
`finished_at`:

| phase | min | median | mean | max | total | share |
|---|---|---|---|---|---|---|
| pre-move: capture, salience, match, decide, focus check | 0.656 | 1.027 | **1.160** | 1.992 | 17.394 | **90.609%** |
| mouse execution (`SendInput` returns) | 0.0002 | 0.0002 | **0.0003** | 0.0011 | 0.005 | **0.026%** |
| post-move: VERIFY capture + `frame.difference` | 0.107 | 0.117 | **0.120** | 0.139 | 1.797 | **9.361%** |
| whole step | 0.774 | 1.156 | 1.280 | 2.103 | 19.196 | |
| inter-step gap | 0.0017 | 0.0021 | 0.0021 | 0.0034 | 0.032 | 0.16% |

The run took 20.249 s of behaviour (the record's 40.27 s `duration` includes the
operator's own `--focus-delay 20` countdown). The step rate was therefore about
**0.78 moves per second**, and the mouse itself accounts for 0.026% of the time
between moves. **There was no deliberate settle delay in that run**: `_pace` is
a no-op unless a step interval is configured, the inter-step gap is 2 ms, and the
only gap between injecting a move and looking at the result was the VERIFY
capture itself (~0.09 s of grab). So the choppiness is not a pacing choice and
not a slow input path — it is capture and decision latency on a 3222x1928 frame,
and the frame size is the reason.

The absence of a settle is no longer true, and the paragraph below is why. WAKE
now waits `wake_verify_settle_seconds` (0.15 s) before the VERIFY capture; see
[the third round](#the-third-round-correct-geometry-and-a-sensor-that-looked-too-early).

An offline reproduction of the same degenerate case on this machine attributes
the cost directly:

| operation at 3222x1928 | 3222x1928 | 1280x720 |
|---|---|---|
| `salience_map` (grid 8) | 0.221 s | 0.038 s |
| `find_candidates` (grid 8) | 0.454 s | 0.077 s |
| `locate`, coarse pass (`refine` scale 4) | 0.637 s | 0.091 s |
| `locate`, fine pass (`refine` window 16) | 0.567 s | 0.071 s |
| `locate`, both passes | 1.228 s | 0.164 s |

The degenerate full-frame template match is the single largest term, and it is a
direct consequence of the defect above: the search window is bounded by design,
but a frame-sized *template* is not, so both passes end up correlating the whole
frame against itself. The offline figures were taken at a different time from
the live run and are the same order of magnitude as, not a decomposition of, the
recorded 1.16 s pre-move mean; the recorded numbers are the authoritative ones.

Two secondary readings fall out of the same data. `frame_changed` is `False` for
14 of the 15 moves even though the view demonstrably moved (mean luma falls
137.1 to 82.0 across the run). `Frame.difference` compares 16x16 block means and
divides by 255, and the loop's threshold is `0.01` — documented as "about 2.5
levels out of 255" — while the recorded differences are 0.000128 to 0.001579,
about 0.03 to 0.4 levels. Only step 10, at 0.0424, tripped it. Given that VERIFY
runs about 0.09 s after the injection and every later step sees a large change,
the honest reading is that **the game has not redrawn by the time VERIFY looks**:
the latency between a relative mouse move and a visible response is bounded
below by ~0.1 s and above by ~0.65 s by this record, and is not pinned more
precisely. VERIFY is currently paying ~0.12 s per step for information that
arrives too early to be true. That is a measurement, not a defect in the
sensor — and it is worth fixing, but not in this pass.

It was pinned in a later pass and fixed: the window is 112-128 ms, and VERIFY now
waits 0.15 s before looking. See
[the third round](#the-third-round-correct-geometry-and-a-sensor-that-looked-too-early).

##### The crash, and the rule it produced

After the loop had already stopped cleanly, the CLI raised:

```
ValueError: dictionary update sequence element #0 has length 9; 2 is required
```

Two defects, one visible and one stacked behind it.

The visible one: the plan was serialised with `[dict(row) for row in
_wake_plan(config)]`. `_wake_plan` returns seven `(label, text)` pairs, and
`dict()` on a 2-tuple treats it as a key/value pair — so it tried to unpack the
nine-character string `"behaviour"` into a key and a value. The plan is now
serialised by an explicit `_wake_plan_payload` that names its two fields, and
`_wake_plan` itself is untouched because the table printer consumes those
tuples.

The stacked one: `AgentLoop.run()`'s `finally` closes the run recorder, so the
CLI's later attempt to write the plan step into the same recorder raised
`RuntimeError: run ... is already closed`. That is why the run's `run.json` was
complete while the command still died.

The rule this produced is a durability invariant, and it is now enforced
structurally rather than by care: **the measurement is written and the run is
closed before anything non-essential is allowed to fail.** The plan step is
recorded *before* the loop runs, while the recorder is open. Closing out is
guarded, and a failure to close is reported rather than raised. And the four
presentation steps — publishing to the observer page, finishing the publisher,
stopping the observer server, printing the summary — are each attempted
separately, each one's failure reported as a `warning:` line on stderr, and none
of them able to skip the cleanup that follows it or lose the result file.

The regression coverage drives the whole `wake-test` command through the real
finalisation path, including terminal-completion and early-stop, and the tests
fail against the old code with the byte-for-byte error above.

##### Smooth mouse motion: designed, deferred

The measured choppiness has a second cause that this pass did **not** fix: a
relative move is one instantaneous `SendInput` jump. The design for
`move_relative_smooth(total_dx, total_dy, duration, microsteps)` is recorded here
so it can be implemented deliberately rather than improvised:

- The total is divided into `microsteps` integer deltas that **sum exactly** to
  the requested total on each axis, with the rounding remainder distributed
  rather than dropped, so smoothing can never change where the view ends up.
- The total is clamped by `max_mouse_delta` **before** division, not per
  microstep, so the existing hardware ceiling keeps its meaning.
- The sleep between microsteps lives in the input layer's injected sleeper, not
  in the policy, and the policy makes no decision inside the interpolation. A
  smooth move is one action with one `ActionRecord`, one duration and one
  verification — interpolation is not a place for autonomy.
- Focus re-check, rate limiting, hold limits and the emergency stop are enforced
  on **every** microstep, not only at the start, and an abort mid-move leaves
  the already-sent deltas sent and says so in the record. A partially applied
  move is reported as partial, never as complete.
- Held-input semantics are unchanged: smoothing applies to relative movement
  only, and no key state is interpolated.
- Tests drive it with fake input only, and assert the sum invariant, the clamp
  order, abort mid-move, and that no real input is ever constructed.

This is deliberately larger than the correction pass and is deferred. It is a
motion-quality change, and it would be dishonest to ship it in the same commit
as a defect fix, because then the next live run could not attribute what changed.

##### The next live run

The operator resizes Luanti by hand to about **1280x720** before the run.
AutoCraft never resizes the window: the window is the thing being measured, and
changing it to make a measurement easier would make the measurement worthless.
The 3222x1928 size is what produced both the ~1.3 s step time and the
low-contrast normalisation blow-up above, and `look_large_window_pixels` (921600)
warns rather than acts for exactly this reason.

The comparison axes for run 2 against run 1, in the order they should be read:

1. **Cadence** — step time and the pre-move share. Run 1: 1.28 s per step, 90.6%
   pre-move. At 1280x720 the offline figures above put the decision cost at
   about 0.22 s instead of 1.46 s, so a step time in the 0.3-0.4 s range is the
   prediction, not a result.
2. **Candidate choice** — does `CANDIDATE_FOUND` report more than one candidate,
   and is any `TARGET_SELECTED` bbox still the whole frame? Run 1: one
   candidate, frame-sized, three times.
3. **Target path** — does the reported `distance` change between consecutive
   moves, or is it constant again? Run 1: constant, which is the signature of
   the fixed point.
4. **Dead repetition** — `dead_repetition_ratio` and `productive_repetition_ratio`.
   Run 1: 1.0 and 0.0.
5. **Downward drift** — are the corrections still `dx == dy` on every move, and
   does the view's mean luma fall by 55 levels again? Run 1: yes and yes.
6. **Truthful emptiness** — if a view has nothing in it, does the run say so, or
   does it name the whole frame a target? Run 1: the latter, three times. The
   fix should make this the "nothing visually salient was found" outcome, and a
   run that reports that has succeeded at being honest, not failed at centring.

Until run 2 exists, no claim is made that the defect is gone. The fix is
implemented and offline-verified — the frame-sized candidate is refused, the
frame-sized patch cannot be located, and a sky-only scene ends with a truthful
"nothing there" — but an offline reproduction is not the operator's screen. Only
a live run can say the view stops walking into the ground.

That run then happened. It answered the question, and it surfaced two defects
nobody had been looking for.

#### The second live run, and what it exposed

Run 2 happened on the same machine, at the same 3222x1928 client area, with the
same command:

```powershell
python -m autocraft wake-test --focus-delay 20 --yes
```

Run `20260921T004907Z-b066177e`. It failed, and **this time the failure is the
fix working**:

```
status            : FAILED
stop reason       : nothing visually salient was found in 12 scan movement(s)
scan moves        : 12
centring moves    : 0
unique views      : 5 (9 revisits)
candidate regions : 0
target changes    : 0
stuck patterns    : 0
duration          : 29.77s
```

(`FAILED` is how the WAKE report renders it; `run.json` records the same run as
`status: "stopped"` with that stop reason, which is the loop's own vocabulary for
"the policy decided there was nothing to do".)

The frame-sized false target is gone. Run 1 named the whole frame a target threetimes and then walked into the ground chasing it; run 2 scanned its bounded set
of views, found nothing it could call a region, and **said so**. A run that
reports "nothing there" has succeeded at being honest, not failed at centring,
and that outcome is preserved exactly as it happened — it was not tuned until it
looked better.

What run 2 did expose was two defects in the machinery around the behaviour, and
one thing nobody had measured.

##### The window was two different sizes in one run

The startup banner reported a client area of **3222x1928**. The finished report
said the frame was **3591x1928**:

| | left | top | width | height |
|---|---|---|---|---|
| startup (`status`) | 615 | 85 | **3222** | 1928 |
| every loop step | 246 | 85 | **3591** | 1928 |

Same right edge (3837), same top and bottom, same height — the capture region
was 369 px wider on the left, and every one of the 13 loop steps agreed with the
wider figure. The two numbers were not competing measurements of one window;
they came from **two different code paths**, which is why the discrepancy
survived a run that otherwise worked:

- The startup banner, the printed table and the plan step all come from a single
  `WindowLocator.status(force=True)` taken *before* the focus countdown.
- The `window` field in the finished report comes from the **first captured
  frame** (`WakeRunner._after_observation`), not from the locator at all.

`ScreenCapturer.capture_region` refuses to return a frame whose size does not
match the region it was asked for, so the wider region was a real, separately
obtained `ScreenRegion` — the locator's own answer had changed between the
pre-countdown reading and the first capture. The handle check before injection
passed, so this is not "it measured the wrong window", and the two readings share
a right edge, a top and a bottom, differing only on the left: a genuine geometry
change on the same handle is the leading explanation, and the 20-second focus
handoff is exactly the window in which the operator is being asked to click into
the game.

The useful negative evidence came from the operator's next run, at a resized
**2246x1159**: there, the startup region and *every* loop step were byte-identical.
So this is not a systematic DPI-scaling bug that would fire on every run — it is a
timing-and-selection effect, which is exactly the kind of thing that only shows up
live. The preserved record cannot settle which of those it is; the next run's
instrumentation can.

AutoCraft does not resize the window to make this go away, and it does not paper
over the disagreement. Instead **every observation now carries its own geometry**:

```
window_handle, client_width, client_height, frame_width, frame_height
```

and the policy emits `WINDOW_GEOMETRY_CHANGED` with `changed`, `before` and
`after` when the client area and the captured frame stop agreeing. The locator's
pre-countdown answer is seeded as the baseline, so the comparison is against the
reading the operator was actually shown.

##### That question led somewhere much bigger: the process was not DPI aware

Following the discrepancy meant comparing AutoCraft's capture against the whole
virtual desktop, and the capture turned out to be pointed at the wrong place. On
a 125%-scaled secondary monitor, `GetClientRect` and `ClientToScreen` were
returning coordinates in the system-DPI virtual space instead of physical pixels,
so the region handed to `mss` was about 2.4x too wide linearly — 5.8x by area —
and landed mostly outside the game window on a dark desktop.

That is the real explanation of the 65%-black capture that had been blamed on
Luanti throttling while unfocused. It is not throttling: a correctly-addressed
capture of the same *unfocused* window is 0.00% black.

The cause was one missing ctypes declaration. `SetProcessDpiAwarenessContext`
takes a `DPI_AWARENESS_CONTEXT`, which is a **handle, not an `int`** — and it was
the one user32 call in `_win32()` with no `argtypes`, so the literal `-4` was
marshalled as a 32-bit int. The call failed with `ERROR_INVALID_PARAMETER` (87)
and the process silently stayed DPI-unaware. Every other call was declared, which
is exactly why this survived review: the code read as though awareness had been
requested, and the return value was never checked.

`ensure_dpi_awareness()` now binds the argument as a pointer, falls back to
`shcore.SetProcessDpiAwareness(2)` if the context call is unavailable, and
**returns the mode it actually measured** rather than the mode it attempted.
That measured mode is what `status` and `wake-test` print, and both print a
caution naming the consequence when it is not per-monitor: relative mouse counts
are unaffected, but the capture region may not be the game.

Same live window, same handle, `GetDpiForWindow` 120 under both:

| | unaware (before) | per-monitor (after) |
|---|---|---|
| `GetClientRect` | 1960x1041 | **817x434** |
| capture mean luma | 13.08 | 41.52 |
| capture black | 65.2% | **0.00%** |

**This invalidates the absolute resolutions quoted throughout this section.**
Every run so far was executed by a DPI-unaware process, so the client areas it
recorded — 3222x1928, 2246x1159, 1960x1041 — were inflated by about 2.4x
linearly and 5.8x by area. The *proportional* findings survive, because both
runs were inflated by the same factor on the same monitor: behaviour time still
tracks pixel area almost exactly, and the focus countdown is still a fixed ~20 s
constant. But the real client area was always about **817x434 (354,578 pixels)**
— comfortably under the ~1,000,000 pixel figure this milestone treats as the
design point. The size gate was never genuinely failing. The operator was
resizing a window that had already been the right size; the reported size was
the defect.

##### A resize is now acted on, not merely reported

The run-2 instrumentation recorded geometry changes, but the policy only emitted
an event and then carried on — a resize was detected and ignored. That is worse
than not detecting it, because every quantity the agent holds is measured in the
pixel scale that just changed: view fingerprints over a grid that no longer
exists, offsets in pixels that no longer cover the same part of the scene, and a
pixels-per-count calibration taken at the old size.

`_note_geometry` now calls a rebaseline on a change. It discards everything
measured in the old scale and **names what it dropped** in the event (`dropped`,
alongside `rebaselined`):

| dropped | why it cannot survive the change |
|---|---|
| view memory, start view | fingerprints taken over a grid that no longer exists |
| chosen target, offset history, last offset / distance / confidence / move | pixel distances that no longer cover the same part of the scene |
| motion calibration | pixels per mouse count, measured at the old size — including an adopted LOOK-001 ratio |
| progress, guards, cooldowns, controller | counters and deadlines scoped to the old scale |

Two things are deliberately **kept**:

- **The scan position.** The camera is still pointing where it was pointing; only
  the units changed. Restarting the scan would re-send movements that have
  already been made, which is the dead repetition this milestone exists to avoid.
- **The run's tallies** (`moves`, `scan_moves`, `candidate_count`,
  `target_changes`). They are a record of what happened, and what happened is
  that the window changed size mid-run.

A run that was centring on a target when the resize arrived returns to
`SCANNING`: a target chosen under the old geometry is not a target any more, so
there is nothing left to centre on. It costs a step and invents nothing.
AutoCraft still never resizes the window itself, and a resize is still not a
reason to restart the run.

##### The final safety sweep ran after the recorder had closed

Run 2 completed its primary result and then printed:

```
warning: the final safety sweep failed:
RuntimeError: run <id> is already closed; telemetry can no longer be recorded
```

The cause was an ordering bug, not a reporting bug. `AgentLoop.run()`'s `finally`
block released inputs, then recorded safety events, then finalised the run, then
shut the guard down — but `SafetyGuard.shutdown()` **is itself a safety event**
(it records a `KIND_LIFECYCLE` entry). So the shutdown event was produced *after*
the last sweep had already run, and was silently lost on every wake run. The CLI
then swept `guard.events` a second time, after `AgentLoop` had already finalised
the recorder, which is what raised.

Two things were wrong and both are fixed:

1. The order is now **release → shutdown → record the final safety events →
   finalise the run**. The lifecycle event exists before the sweep that carries
   it, so it is recorded rather than dropped.
2. The CLI's sweep now runs **only when the loop never ran**. On the path where
   the behaviour ran, the loop already swept its own events. As it stood, the
   CLI passed the *entire* event list rather than a cursor, so had that call ever
   succeeded it would have duplicated every safety event in the record.

Nothing suppresses the warning; the ordering that produced it is gone, and a
regression test asserts a successful cleanup prints no warning at all.

##### Nobody had measured where the time went

`steps.ndjson` recorded one number per step: `duration`. It did not record how
much of that was capture, how much was deciding, or how long the game sat still
between two inputs — so "the run took 29.77 s" was the only cadence fact
available, and it turned out to be misleading. Every step now records a phase
breakdown:

```json
"timings": {
  "capture": 141.2, "decide": 3.4, "act": 0.9, "verify": 120.0,
  "since_previous_move": 289.7, "total": 300.1
}
```

milliseconds, per step, and `wake_result.json` carries the means plus the sample
counts (`steps_timed`, `moves_timed`) so a mean over three steps can never be
read as a mean over thirty. Nothing smooths them and nothing reads them back.

Reading the two preserved runs with that decomposition finally explains the
headline durations:

| | run 2 (3222x1928) | re-run (2246x1159) |
|---|---|---|
| recorded `duration` | 29.77 s | 24.11 s |
| **focus countdown** | **20.02 s** | **20.02 s** |
| behaviour (sum of steps) | 9.71 s | 4.07 s |
| mean step | 747.2 ms | 313.1 ms |
| slowest / fastest step | 870.8 / 617.4 ms | 440.0 / 246.0 ms |
| gap between steps | 2-4 ms | 2-3 ms |

Two things fall out of this, and both are more useful than the headline number:

- **67-83% of the run's recorded duration is the focus countdown**, not
  behaviour. The `--focus-delay 20` handoff is inside the run, so `duration`
  mixes a fixed human-paced wait into a measurement of the agent. Comparing two
  runs by `duration` is comparing their countdowns.
- **Behaviour time scales with pixels, and nothing else does.** Halving the
  resolution by area (6.21 → 2.60 Mpx, a factor of 2.39) cut the mean step from
  747 ms to 313 ms — a factor of 2.39. The gap between steps stayed at 2-4 ms, so
  the loop is never idle and `step_interval` is doing nothing here. Capture and
  analysis *are* the step time.

The 2246x1159 re-run is still above the 1 Mpx the design assumes, and it produced
the same outcome as run 2 — 5 unique views, 9 revisits, 0 candidates, the same
stop reason. A smaller window made it faster; it did not make it see anything.
(As the DPI finding above shows, that 2246x1159 was itself an inflated reading;
the real client area was under 1 Mpx the whole time.)

##### The third round: correct geometry, and a sensor that looked too early

Three live runs followed the DPI fix — `20260921T031753Z-7ccf8a25`,
`20260921T032850Z-e77e7581` and `20260921T033013Z-2ec9ec4b`. They are the first
runs whose geometry is real, and the things the earlier passes fixed stayed
fixed: `safety_events` is recorded (Finding 2), the startup
`WINDOW_GEOMETRY_CHANGED` is the benign first-capture `0x0 -> real` population,
salience reaches `TARGET_SELECTED`, and no run invented a frame-sized target.

They also all failed, and in a way that pointed at the sensor rather than the
policy. `frame_changed` was `True` on **0 of 29** moves in the first,
**2 of 30** in the second and **0 of 6** in the third, which drove
`dead_repetition 1.000` and `productive_repetition 0.000`: the agent believed
every movement it made had changed nothing, and abandoned every candidate it had
found. Two explanations were tested and both were wrong.

- **"The mouse input never reaches the game."** Falsified. `move_relative` is a
  correct `SendInput` relative move, and on a verified-static scene the same move
  produced a `0.104` frame difference.
- **"The window lost focus mid-run."** Falsified. `is_foreground` is `True` on
  every real step of every run; only the synthetic header row is `False`.
- **"The pointer needs recentring."** Not applicable. The cursor is captured by
  the game (an SDL pointer grab) and pinned to the window centre — `GetCursorPos`
  returns exactly `(2880, 1661)`, and `SetCursorPos` returns `1` with a
  `last_error` of `0` while changing nothing. There is no drifting pointer to
  recentre.

What the runs were actually seeing is a **capture race**, and it is deterministic.
With a verified-static baseline — no hands on the keyboard or mouse, which is a
prerequisite for any probe here, because a probe run while the operator is playing
measures the operator — the same movement was captured twice, immediately and
then ~115 ms later, four times over:

| round | capture #1, ~50 ms after the move | capture #2, ~115 ms after the move |
|---|---|---|
| 1 | `0.00000` | `0.10356` |
| 2 | `0.00000` | `0.10359` |
| 3 | `0.00000` | `0.10471` |
| 4 | `0.00000` | `0.10475` |

The earliest capture that crossed the loop's `0.01` threshold was 112, 128, 112
and 121 ms. The immediate capture is not merely early — it is deterministically
**one redraw behind**: a `(+200, 0)` move read `0.00000` while the immediately
following `(-200, 0)` move read exactly `0.08997`, byte-identical to the settled
value from the move before it. So `VERIFY` was comparing the frame before the
move with the frame before the move, and correctly reporting that nothing had
changed.

**LOOK-001 already had this solved and WAKE had not inherited it.**
`look/runner.py` settles before both of its captures, driven by
`look_settle_seconds` (0.15 s), because the LOOK-001 measurement cannot work
without it. `AgentLoop._step` — the path WAKE-001 uses — had no settle at all.

WAKE now waits before the verification capture:

| setting | default | what it is |
|---|---|---|
| `wake_verify_settle_seconds` | `0.15` | wait after injecting a move, before the VERIFY capture |

It applies **only** to the post-move verification capture, not to the pre-move
observation, and it reuses the loop's injectable sleeper, so tests never really
sleep. 0.15 s is a three-times margin over the measured 112-128 ms crossing, and
it is the same value and the same physics as LOOK-001's established settle, so
there is one precedent rather than two. The loop records the wait as a
`verify_settle` timing, and `wake_result.json` carries a `verify_settle_mean`
row beside the other cadence means.

Two consequences worth stating plainly. The cadence will be about 150 ms slower
per movement step, and that cost is deliberate — it buys a measurement that is
true rather than one that is merely prompt. And the earlier runs' `frame_changed`
and repetition figures are now known to be **unusable as evidence about the
policy**: they were measuring the capture race. Comparing run to run across this
change means comparing `verify_settle_mean` alongside the other cadence means
rather than against them.

The third run's records also disagree with the first two on repetition
(`dead_repetition 0.0`, `productive 1.0` where the others have `1.0` and `0.0`).
That difference is not explained yet, and with a settle in place it is finally
worth asking, because the flag now measures the movement instead of the race.

#### Before the next live run

Both of the runs above were recorded by a DPI-unaware process, and step time
tracks frame area almost exactly, so the numbers they produced describe the
inflation as much as they describe the agent. The runs after the DPI fix are the
first whose resolution, cadence and geometry are all real, and they are the ones
described in
[the third round](#the-third-round-correct-geometry-and-a-sensor-that-looked-too-early)
above.

**AutoCraft will not resize the window for you** — the window is the thing being
measured — so the size is a manual step, but the first thing to check is not the
size:

1. Confirm AutoCraft can see the game, and that it is DPI aware:

   ```powershell
   python -m autocraft status
   ```

   `DPI awareness` must read **`per-monitor`**. If it does not, the capture
   region may not be the game, and `status` prints a caution saying so — fix that
   before reading anything else. On this machine the real client area is about
   **817x434**, well under the ~1,000,000 pixels the design assumes. Roughly
   **1280x720** or anything else below that figure is also fine. There is no
   longer a size gate to chase.

2. Then run the behaviour:

   ```powershell
   python -m autocraft wake-test --focus-delay 20 --yes
   ```

Read the result against the two runs above on: mean step time, capture duration,
unique views, revisits, candidate count, **candidate rejection reasons** (the
`SALIENCE_SCAN_SUMMARY` events, which say *why* nothing was found rather than only
that nothing was), centring moves, target path, dead repetition, and the stop
reason. There is no pass/fail threshold: a run that scans a bounded set of views
and reports honestly that it found nothing is a valid result, and lowering the
salience threshold until a candidate appears would destroy the measurement the
whole milestone rests on. Note also that the absolute resolutions and cadence
figures from the earlier runs are inflated, so compare *shapes* — does time track
area, do views repeat, does anything survive selection — not the raw
milliseconds. Read `verify_settle_mean` alongside the other cadence means rather
than against them: it is a deliberate 0.15 s per movement step, and without
subtracting it the new run will look slower than the old ones for a reason that is
not the agent's.

The runs after the DPI fix are the first that record a per-phase `timings`
breakdown and a per-observation `WINDOW_GEOMETRY_CHANGED` comparison, so their
numbers can say where the time went and whether the geometry moved.

#### What it does not do

No semantic object recognition, no tree recognition, no LLM in the motor path,
no vision-language API, no reinforcement learning, no pathfinding, no walking,
jumping, mining, crafting, or survival, no long-term memory, no multi-agent
logic. `ThoughtEvent` is display-only: the thought hook is called *after* a
decision is made, its return value is discarded, and its failure is caught. A
thought can never move the mouse.

---

## The observer page

```powershell
python -m autocraft observer            # http://127.0.0.1:8765
python -m autocraft observer --demo     # same page, synthetic data
```

A local, read-only page that shows what AutoCraft is doing internally: the
latest captured frame, the current mode, goal and intention, the last action
and its result, confidence, recent events, the current simulated affect, the
safety state, and — during `look-test` — the **Visual motion** panel, and —
during `wake-test` — the **Wake behaviour** panel.

The Wake behaviour panel shows what the behaviour layer is doing right now: the
state, the current strategy, the target's centre, offset and distance from the
crosshair, its salience, the match confidence, the progress series, the repeat
guard, the unique and revisited view counts, the active cooldowns, and the most
recent event. Every number on it comes from the same flat report the policy
publishes, so the panel cannot disagree with the record on disk; the one field
the page adds is `measured_at`, which is a fact about the display rather than
about the run.

The Visual motion panel shows the measured primitives of the LOOK-001 run: the
difference statistics, the per-block difference map, the estimated shift, the
pixels-per-delta ratio, and the reversibility figure.

Both panels read `status: not-run` until something has actually been measured,
and neither fills itself in from the plan.

It exists so a run can be watched and streamed without reading a log.

It is **loopback-only by default**. Binding to a non-loopback address requires
an explicit `--allow-remote`, and requests carrying a non-loopback `Host`
header are refused with `403` even then.

**Read-only, structurally.** Three properties hold together, and each is
pinned by a test:

1. `observer/` does not import `control/`. The display layer has no reachable
   path to a keyboard or mouse.
2. Every route is `GET`. Any other verb gets `405` with `Allow: GET`. There is
   no endpoint that accepts a command, a mode change, or an action.
3. The safety panel is computed from *published facts* by a pure function
   (`input_permitted`), not by querying the live guard. Asking the real guard
   would append a refusal to the safety log every time the page refreshed,
   which would make the log a record of the dashboard's polling rather than of
   the agent's behaviour.

**Truthful empty states.** V0 has no object detector, no tree recognition, and
no privileged state. Beliefs and detected entities are therefore displayed as
empty, and say so. The page never invents perception to look busier than the
agent is. The VISION-001 scene model is not published to the page either:
`perceive-test` writes its result to `data/runs/<run-id>/perceive_result.json`,
and nothing on the page pretends otherwise.

**One-directional by construction.** `LoopPublisher` receives safety as a
provider function and confidence as a callback, so the bridge never holds a
reference to the guard and the page never holds a reference to the agent.
`AgentLoop` gained a single `on_observation` hook; it still knows nothing about
HTTP, JSON, or JavaScript.

**Demo mode is labelled.** `--demo` fills the page with synthetic goals,
events, affect and metrics, and every surface of the page says `DEMO`. The
demo ticker is deterministic — it does not randomise the mood to look alive.

---

## Thoughts

`ThoughtEvent` is an **expression layer**, not a decision layer. The flow is
one-directional:

```
experience -> affect -> thought generated -> displayed to a viewer
```

A `ThoughtEvent` can never cause keyboard or mouse input. `thoughts/` holds no
reference to an executor, a keyboard, a mouse, or the safety guard, and does not
import `control/`. The flow that is *wrong* — `thought -> execute game action` —
is not reachable from the code.

Thoughts are generated, user-facing character expressions. They are **not**
hidden chain-of-thought: the model's reasoning is never surfaced, and the
generator is a template composer rather than a model. The `ThoughtGenerator`
protocol (`generate(context) -> ThoughtEvent | None`) is the seam a future
model-backed generator would plug into, and nothing else would change.

Each thought carries a tone (`neutral`, `humorous`, `curious`, `hopeful`,
`excited`, `sad`, `frustrated`, `anxious`, `dramatic`, `absurd`, `reflective`),
an intensity, a trigger (`discovery`, `success`, `failure`, `danger`, `memory`,
`idle`, `milestone`, `random_reflection`, `demo`), and the affect snapshot it
was generated under.

**Bounded, not chatty.** `ThoughtPolicy` enforces a minimum interval, a
per-minute ceiling, and occasional long quiet periods. Probability rises after
meaningful events and is damped during rapid action, so thoughts do not appear
on a metronome or once per action. Nothing is random merely to look alive.

**Affect influences, but does not dictate.** High curiosity biases toward
speculative tones, high frustration toward irritation, low energy toward
shorter and rarer thoughts. The mapping is a weighting, not a lookup table.

**Memory is retrieved, never invented.** `ThoughtContext.remembers(needle)`
returns a stored memory containing that needle, or `None`. A thought that says
"earlier I…" is composed only from a memory that actually exists in the
context. If there is nothing to recall, the generator has no clause to reach
for. This is the honesty seam of the whole feature, and it is pinned by
property-style tests that assert no memory text is ever fabricated.

**Extreme thoughts stay non-operative.** Intensity and language can be
dramatic; that changes nothing structural. Thoughts cannot bypass the safety
layer, modify permissions, damage the host, initiate real-world actions, or
reach external information.

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

**Bounded actions.** Key hold duration is clamped to the configured maximum
(`max_key_hold_seconds`) — holding longer than asked for is always safe, so a
long hold is shortened rather than refused. Mouse deltas are different: a
relative move whose delta exceeds `max_mouse_delta` on either axis is
**rejected, not clamped** — the agent's intent is never quietly rewritten into
a different action. So: holds may be clamped, mouse deltas are refused.

**Rate limiting.** Actions are paced by a minimum interval. Rate limiting
never consumes the "too many consecutive failures" budget; it waits instead of
refusing, so pacing cannot masquerade as a safety stop.

**No autonomy by default.** `status`, `capture`, `observe`, `loop`, `keys`,
`config`, and `observer` never inject input. Only `input-test` and `look-test`
can, and each requires an explicit `--yes`. `input-test` states clearly what it
is about to do, sends exactly one small action, releases it, and exits. Both
commands check foreground in the only order that works from a shell: locate and
vet the target, print the bounded plan, demand `--yes` *before* any injection
machinery is constructed, and only then start a countdown for you to focus the
game. After the countdown they re-query the window and refuse unless that exact
window is foreground; the guard then re-checks foreground again inside the
actuator call itself, so focus moving in between is still caught. `look-test`
does that check before *every* movement in the sequence, not once. Every
refusal path injects nothing and every exit path releases everything.

**LOOK-001 injects through the same actuator as everything else.** `look-test`
has no input path of its own: it calls `Mouse.move_relative` through the
existing V0 guard, so the foreground lock, the rate limit, the per-axis bound
and the release bookkeeping are exactly the ones described above. Its movements
are bounded twice over — a plan of at most `look_max_steps` trials, and a
per-axis delta capped by `max_mouse_delta` — and the calibration series is
capped by the same `look_max_steps` bound, so editing the config cannot produce
an unbounded run.

**The observer cannot reach the actuators.** The observer page and the thought
system are downstream of the agent, not upstream of it. Starting the page
constructs nothing from `control/`, so it cannot make game control possible,
and no amount of traffic to it can produce an input event. `--observer` on a
run only publishes state to the page; it does not enable anything.

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
- mouse-delta limits: deltas above the maximum are rejected, not clamped
- key state tracking and release-all behaviour, including that a failed
  backend release keeps the input tracked so a later release-all retries it
- foreground-window rejection
- the no-op decision policy
- bounded agent loop behaviour, including capture failure
- telemetry serialisation
- the CLI safety gate: `input-test` refuses before any input object exists
- the `input-test` ordering: a shell being foreground at launch is fine, and
  the post-countdown re-query refuses on lost focus, a vanished target, a
  newly-minimised target, or a changed window handle — with zero input sent
  on every refusal path
- the focus handoff itself: the configured default applies when no flag is
  given, `--focus-delay` overrides it, `0` skips the wait, and a negative,
  `nan` or `inf` delay is a usage error rejected before the window layer is
  consulted
- the re-run hint: the command `input-test` prints back is built from the
  operator's own flags, so it cannot drift from what they asked for
- the capture-rate warning
- the display contract: exact snapshot keys, strict-JSON round-trip, frame
  downscaling and JPEG encoding
- the HTTP surface: every `GET` route, `405` + `Allow: GET` for every other
  verb, `403` on a foreign `Host` header, read-only invariance under traffic
- `input_permitted` cross-checked against `SafetyGuard.authorize`, so the
  dashboard's safety panel cannot drift from the real refusal rules
- the thought model, the policy gates, and the engine's cooldown and ceilings
- the non-operative guarantee: an import-structure check, an attribute scan,
  and a real 300-step run asserting zero backend events and zero held inputs
- the memory-honesty property: no thought may reference a memory that is not
  in its context
- the LOOK-001 frame maths: identical and different frames, the block map's
  shape, synthetic shifts in both directions on both axes, the divide-by-zero
  cases returning `null` rather than a number, and reversibility
- the LOOK-001 runner sequence: a complete trial writes all five artifacts, and
  focus loss before the first movement, focus loss between `+dx` and `-dx`, a
  capture failure, a vanished target, and an emergency stop all abort with a
  reason and no further movement
- the LOOK-001 record: strict-JSON round-trip, no raw pixels in the JSON, the
  difference images present only for comparisons that were actually measured,
  and the limitation notes travelling with every record
- the LOOK-001 structural prohibitions, checked as **imports** and as the
  package's declared public surface: no `look` module can reach the control
  layer, none imports a trained model, and none exposes a pass/fail verdict
- the `look-test` safety gate: refusal before any input object exists, no output
  directory created on a refused run, and a dry run that reports the plan
  without claiming a result
- the `look-test` ordering: a shell being foreground at launch is fine, and
  losing focus during the countdown stops the run with the record marked
  `interrupted` and zero movements sent
- the `look-test` bounds: a zero delta, an oversized delta, an out-of-range step
  count, a non-positive or unbounded settle, and a calibration series longer
  than `look_max_steps` are each refused with the configured bound named
- the VISION-001 cell partition: `cell_luma` against `Frame.block_means` on an
  even split, the deliberate divergence between the two layers on an uneven
  frame, and that the perception partition covers every pixel exactly once
- the VISION-001 feature vector: every cell feature, the ragged-frame refusal,
  and the declared feature-name order matching what the extractor returns
- the VISION-001 model: fitting, that a still scene is reported as still, that a
  post-warm-up change is localized to the cells it happened in, that adaptation
  drifts a quiet cell's centre and leaves a flagged one's alone, that one large
  surprise cannot desensitise a cell, that the spread still learns from a
  flagged cell, strict-JSON round-trip through `save`/`load`, and that an
  unfitted model refuses to serialise
- the VISION-001 session: the stop reasons for a step limit, a time limit and a
  replayed frame set, capture failures counted rather than raised, and an
  incomplete warm-up reported as incomplete
- the VISION-001 policy: that it is a real `DecisionPolicy`, that it never
  exceeds `max_mouse_delta`, and that it nudges one axis toward the worst cell
  rather than stalling
- the VISION-001 structural prohibitions, checked as **imports** and as the
  package's declared public surface — including a test that pins the
  relative-import resolver itself, because a resolver that silently returns
  nothing would make every other structural check pass forever without checking
  anything
- the `perceive-test` safety gate: no input machinery is constructed, frames
  arrive only through the observer, the record contains no verdict, and the
  command says so
- the `perceive-test` target vetting: a missing window and a minimised window are
  each refused with no result file written, and an oversized window warns but
  still measures
- the `perceive-test` bounds: `--steps` bounds the run and the observer is called
  exactly that many times, and an incomplete warm-up is both reported and
  persisted with `complete: false`
- the `perceive-test` persist-before-print ordering: the printing is made to fail
  on purpose, and the result file still holds the full measurement
- the WAKE-001 view memory: that an identical view is recognised as a revisit, a
  novel view is not, and the memory stays inside its limit
- the WAKE-001 salience: that a region of structure is found and a uniform scene
  offers nothing, that the returned candidates are real `CandidateTarget`s, that
  a candidate marked failed is not offered again, and that the scoring weights
  are explicit and documented
- the WAKE-001 frame-covering refusal, the fix for the first live run's defect: a
  synthetic low-contrast sky offers no candidate at all, `candidate_from_cells`
  refuses a box that covers the frame, a box larger than the frame is refused
  while a full-width band is not, a frame-sized patch is refused by both matcher
  passes, a large-but-bounded region is still located — so that over-refusal
  cannot be mistaken for a fix — and a sky-only scene runs to `FAILED` with a
  truthful "nothing visually salient" reason, zero `TARGET_SELECTED` events and
  zero `CENTERING_PROGRESS` events instead of centring the sky
- the WAKE-001 centring: that the movement band follows the distance, that a
  sign flip is recorded as an overshoot with the reversed axis named, that a
  panning scene converges inside the dead zone, that the controller self-measures
  its mapping and switches source once it has samples, and that a mapping limited
  by the hardware ceiling does not converge — recorded as the honest failure it
  is rather than asserted away
- the WAKE-001 anti-repetition: that a productive streak is not stuck, that a
  move which stops helping becomes stuck, that an unmeasured attempt counts as
  no improvement, and that a repeatedly failing strategy goes on cooldown
- the WAKE-001 bounds: that every ceiling is enforced and that a configuration
  letting one phase starve another is rejected
- the WAKE-001 record: strict-JSON round-trip, no raw pixels, the standing
  limitation notes, and that the measured target is not overwritten by the run
  plan
- the WAKE-001 runner, end to end on a synthetic panning scene: a complete run
  that centres its target and reports `completed`, and a run the loop has to cut
  short that reports `aborted` with the reason that ended it
- the WAKE-001 live publisher: that the CLI's status callback actually reaches
  the observer panel, driven with the real report shape, because the display path
  swallows its own failures and a bad keyword there would leave the panel empty
  for a whole run without raising anywhere
- the WAKE-001 structural prohibitions, checked as **imports** and as the
  package's declared public surface: no `wake` module can reach the control
  layer, none imports a trained model or a model API, none opens a network
  connection or a subprocess, and none exposes a pass/fail verdict
- the `wake-test` safety gate: refusal before any input object exists, the plan
  printed without claiming a result, and a dry run that sends nothing
- the `wake-test` finalisation path, driven through the whole command: that the
  plan is serialised as named fields and never as `(label, text)` pairs, that the
  plan step is recorded while the recorder is still open, that the run is
  finished exactly once, and that an early stop and a terminal completion both
  reach the same close-out
- the `wake-test` durability rule: the summary, the observer page publish, the
  publisher and the observer server are each made to fail on purpose, one at a
  time, and the result file is still written and the cleanup still runs — with
  the failure reported as a warning rather than swallowed
- the `wake-test` crash regression: the exact `dict()`-on-a-2-tuple defect that
  killed the first live run is pinned as still raising, so the fix cannot be
  undone by a plausible-looking tidy-up
- the WAKE-001 salience funnel, keyed on the case run 2 actually hit: a
  sky-only frame reports `every_group_refused` with `rejected_frame_span: 1`
  beside `surviving_candidates: 0`, so a diagnostic can never read as
  "candidates survived" when none did. The relocation refusals
  (`rejected_search_room`, `rejected_score`, `rejected_flat_patch`) and the
  top-cell-score list are covered too, and the summary is asserted to carry no
  pixel data
- the WAKE-001 geometry instrumentation: `WINDOW_GEOMETRY_CHANGED` fires on a
  real client-area/frame-size disagreement and *not* on an unchanged run, every
  observation records its handle and both sizes, and the real `Observer` — not a
  fabricated change — is driven through the disagreement so the detection seam
  itself is covered
- the WAKE-001 resize response: a resize discards the view memory, the chosen
  target, the offset history and the motion calibration, and says so by name in
  the event; an ordinary step and an ordinary `reset()` keep the calibration; the
  scan position and the run's tallies survive; a centring run returns to
  scanning; and a rebaseline reports only what it actually dropped
- the WAKE-001 DPI awareness: a fresh process really does end up per-monitor
  aware, the mode reported is the mode measured rather than the mode attempted,
  and the awareness context is bound as a handle rather than an int — the exact
  defect that made the capture point at the wrong place
- the WAKE-001 cadence: the per-phase breakdown reaches the step record, the
  means and their sample counts reach `wake_result.json`, an unreadable timing
  is dropped rather than failing the record, a run that timed nothing reports an
  empty cadence rather than zeros, and a successful cleanup prints no warning
- the WAKE-001 verify settle: a movement is judged only after the game has
  redrawn, so a redraw-aware fake reports the change and the same fake with the
  wait disabled reproduces the live defect exactly (difference `0.0`,
  `frame_changed` false); the wait appears in the step timings and is absent when
  it is disabled; it precedes the verification capture once per movement rather
  than once per capture; the shipped default is non-zero; and the runner threads
  the configured value from `Config` into the loop

The suite is 898 tests and runs in about 40 seconds. Everything that talks to
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

`look-test` writes its own experiment record alongside the standard run
directory, in `data/runs/look/` by default. See
[the LOOK-001 measurement](#the-look-001-measurement) for the layout.

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

**Both actuator classes are verified against the live game.** The test suite
still never sends real input — the `SendInput` path is covered by unit tests
against a fake backend — so the live evidence is a manual smoke test, not an
automated one. Two such smoke tests have passed against Luanti 5.17.0:

| Command | Reported | In-game effect |
|---|---|---|
| `input-test --action key-tap --key w --hold 0.05 --yes` | `executed: true`, `duration: 0.051s`, `released inputs: none`, exit `0` | character moved |
| `input-test --action mouse-move --dx 10 --dy 0 --yes` | `executed: true`, `duration: 0.001s`, `released inputs: none`, exit `0` | camera turned |

So window discovery, the foreground lock, the focus handoff, and real
`SendInput` for *both* keyboard and relative mouse reach the actual game.
Cleanup left no tracked held input in either run.

The read-only half of the boundary was exercised first, against the same live
window: window discovery, client-area geometry, `capture`, and the foreground
and minimised checks all behave as documented. An earlier `input-test` attempt
refused at the post-countdown focus check because the launching shell still held
focus when the handoff elapsed. The refusal worked exactly as intended, which is
why the default countdown is now 10 seconds and `--focus-delay` exists.

**What two bounded smoke tests do not establish.** They prove the actuator
wiring, nothing more. Still unverified: sustained autonomous movement, repeated
closed-loop control, camera calibration, the relationship between `dx`/`dy` and
how far the view actually rotates, navigation, model-backed decisions, and
long-running autonomous play.

**The `--dx 200` probe has now been run, and it moved the camera — but it did not
produce a mapping.** This is the correction to an earlier reading, so it is worth
stating precisely. The completed run
(`data/runs/20260920T051311Z-1145fdca`) reports a shift of `0.006 px` and a
`pixels_per_delta.x` of `3e-05`, which reads as "nothing moved". **That reading is
wrong.** On the raw frames, no rigid transform reproduces frame B from frame A:
translation is best at exactly `(0, 0)` with a 0.00% gain, a large-shift search
reaches only 3.7–12.96% at mutually inconsistent offsets, vertical shift is
0.00%, the best rotation is `0°` with a monotone decline either way, the best
zoom is scale exactly `1.00`, ECC reaches `cc 0.594–0.628` against `0.9998` for
the control, ORB finds only 8/31 inliers, and the best achievable correlation of B
against *any* shifted A is 0.6249 versus 0.5364 for identity. Meanwhile the
reverse `-200` delta restored the view essentially exactly: `cc 0.9998`,
1824/1825 ORB inliers, an identity homography to 0.02 px, and every matched patch
at `NCC 1.0000`. The `0.006 px` figure is a phase-correlation artifact — the two
views share so little that the estimator has nothing to lock onto.

So: **relative mouse input demonstrably causes large in-game camera motion, and it
is exactly reversible.** That settles the engagement question in the affirmative.
It does **not** give a pixels-per-count ratio, because a 200-count delta moves the
view far enough that no shared features survive to fit a transform to. The ratio
has to come from a middle value (roughly 25–50 counts), which is what the
calibration series is for. An earlier analysis of these same frames concluded the
opposite; that analysis applied `cv2.equalizeHist` to a near-black scene, which
amplifies quantization noise into apparent structure. The re-derivation on raw
luminance is what is reported here.

**The mapping is therefore still unmeasured, and WAKE-001 runs without it.**
`wake-test` begins by not knowing how far a count moves the view and sizes its
first corrections from a fixed band count; it self-measures as it goes and reports
`source: unmeasured` until it has three samples on an axis. This is honest, but it
means the first behaviour runs with a weaker prior than the design intends.

**The LOOK-001 trial has been run twice, and neither measured the mapping.** The
tool exists — `look-test`, documented above — and one live trial at `(+10, +0)` is
on record. It established that the mechanism works (two movements sent, three
frames captured, nothing left held) and that reversibility is excellent (0.007).
It did **not** establish a pixels-per-delta ratio: the picture did not move by a
measurable amount at 10 counts, so the ratio that run reported came from an
estimate indistinguishable from zero.

The follow-up probe at `(+200, +0)` did move the camera, decisively, and it too
reported no usable ratio — see
[the correction above](#known-limitations) for the evidence and for why a
200-count delta is too large to fit a transform to. There is still no live mapping
number and still no evidence about repeatability.

**A measured ratio is a ratio for one configuration.** Even once the trial has
run, a pixels-per-delta figure is specific to that window size, that field of
view, that in-game sensitivity, and that mouse setting. It is not a property of
AutoCraft, and it does not transfer to a different setup without being measured
again.

**The VISION-001 scene model has two measured weak points.** Both are visible in
the two live `perceive-test` runs and neither is fixed:

- **The warm-up window can catch a transient.** Live run 2 fitted a mean
  frame-to-frame movement of 21.30 and a mean spread of 33.33, giving a mean
  allowance of 70.0 luma levels against a mean observed deviation of 13.2. The
  model then reported 0.22 changed cells per frame — an honest reading of a model
  calibrated against 20 unusually busy frames. On the quiet recorded set the same
  code reported `floor_share` 0.996, meaning 99.6% of cells were running on the
  configured floor rather than on anything learned. A median-of-medians fit and a
  warning when the fitted movement looks like a transient are the obvious next
  steps; neither is implemented.
- **The adaptation rate cannot follow a scene that changes a lot.** In live run 2
  the mean luma rose 41.05 → 73.44 and `colour_spread` 15.58 → 33.97 over 40
  seconds, *after* the model was fitted. The only mechanism for that is the
  `adapt_rate` centre drift, and 0.05 per frame at this machine's ~2.4 fps
  capture rate is far too slow. A longer warm-up does not help this case.

There is also **no verdict anywhere** in the layer, deliberately. It reports
changed cells, excess, and where the change accumulated; deciding whether any of
that means "the camera moved" is a separate, unbuilt thing.

**The camera-engagement question is now answered, and the mapping question is
not.** The first LOOK-001 trial at `(+10, +0)` measured no picture movement. The
`(+200, +0)` probe answered why the question was hard to read and settled the
engagement half of it: on the raw frames no rigid transform reproduces B from A,
while the reverse delta restores the view essentially exactly. Relative mouse
input does turn the camera. What is still missing is the *scale* of that turn,
because 200 counts moves the view past the point where any shared features
survive. That is a middle-value calibration problem, not an engagement problem.

The follow-up probe to run is therefore a smaller one:

```powershell
python -m autocraft look-test --dx 40 --dy 0 --steps 1 --yes
```

and then the full series:

```powershell
python -m autocraft look-test --calibrate-horizontal --yes
```

---

## Where this goes next

**The first behaviour is now built, has now run live three times, and the
mapping is still the open question.** `wake-test` (WAKE-001) looks around, picks
the most visually salient region it can find, centres it, and stops — bounded at
every phase, with the mouse mapping self-measured as it goes. It has now been run
against a real Luanti scene three times, and none of those runs has yet produced
a target (see
[the second live run, and what it exposed](#the-second-live-run-and-what-it-exposed)).

**The mapping is what it is still missing.** LOOK-001's `(+200, +0)` probe
settled that relative mouse input turns the camera; it did not produce a
pixels-per-count ratio, because 200 counts moves the view far enough that no
shared features survive to fit a transform to. The next step is a smaller delta:

```powershell
python -m autocraft look-test --dx 40 --dy 0 --steps 1 --yes
```

and then the series:

```powershell
python -m autocraft look-test --calibrate-horizontal --yes
```

Click inside the game window during the countdown before running either.

(If either reports `invalid choice: 'look-test'`, you are running from a different
worktree than the branch — see
[Working from a git worktree](#working-from-a-git-worktree).)

**After that**, the ordering that follows from what has actually been measured:

1. **Run WAKE-001 live again with the DPI defect fixed, and read what it
   records.** The behaviour has now faced a real scene three times without ever
   finding a target, and every one of those runs was made by a DPI-unaware
   process, so the 6.2 and 2.6 million pixel client areas it recorded were
   inflated readings of a window that was under a million pixels all along. The
   next live run is the first one that is both instrumented and measured in real
   pixels, so its numbers can mean what they say.
2. **Feed the measured mapping back in.** Once `look-test` produces a ratio,
   `wake-test` can adopt it as a starting calibration instead of beginning from a
   fixed band count, which is the difference between a first correction that is
   roughly right and one that is sized by a guess.
3. **Settle the warm-up and adaptation weaknesses.** A median-of-medians fit and
   a transient warning, both driven by the two live runs above.
4. **Capture a focused series with deliberate movement.** Every capture so far
   was either unfocused or had no camera movement, so there is still no data
   about what a moving camera does to these per-cell statistics. That is the
   first thing TREE-001 would need.
5. **Only then** a movement-sensitive decision policy. `PerceptionDecisionPolicy`
   exists and is tested, but it is unvalidated against real motion, and
   validating it needs step 4.

A perception layer that cannot yet tell a camera pan from an animated river is
not a foundation for tree recognition, and pretending otherwise would undo the
point of measuring. V0 still stops at a foundation that is boring, inspectable,
and trustworthy.
