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

                    ---- published state, one direction only ----

DISPLAY       observer/   ObserverSnapshot -> HTTP -> browser
  ^
EXPRESSION    thoughts/   ThoughtEvent (generated, never acted on)
```

The last two boxes are deliberately **below** the arrow, not beside it.
`observer/` and `thoughts/` read published facts; nothing in them can reach
`control/`. That is enforced by import structure and pinned by tests.

```
src/autocraft/
  config.py              frozen Config: window patterns, limits, directories
  cli.py                 the eight commands
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
| `keys` | no | Lists every supported key name. |
| `config` | no | Prints the effective configuration and where it came from. |
| `observer` | no | Serves the local read-only observer page. Never enables control. |

```powershell
python -m autocraft status
python -m autocraft capture
python -m autocraft observe --seconds 5 --steps 60
python -m autocraft observe --seconds 30 --steps 300 --observer
python -m autocraft loop --steps 5 --seconds 10
python -m autocraft observer --demo
python -m autocraft keys
```

The loop refuses to start unless you give it `--steps` or `--seconds`. There
is no "run forever" option, by design.

`--observer` is available on `observe`, `loop` and `look-test` and does exactly
one thing: it starts the page and publishes that run's state to it. It does not
change what the command does, and it never enables input. If the port is
already busy, the run continues and the page is simply unavailable.

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
python -m autocraft look-test --dx 10 --dy 0 --steps 1 --yes
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

#### Calibration

The mapping is not assumed to be linear, so `look-test` will run a bounded
series instead of a single delta:

```powershell
python -m autocraft look-test --calibrate-horizontal --yes
python -m autocraft look-test --calibrate-vertical --yes
```

The series is config `look_calibration_deltas` (`[2, 5, 10, 20]` by default),
one trial per delta, and it is capped by config `look_max_steps` (10 by
default). A series longer than that bound is refused outright, so editing the
config cannot produce an unbounded run.

#### Bounds and refusals

Every plan is finite; there is no unbounded mode. These are all usage errors
(exit `2`), refused before the window layer is consulted:

| Input | Why it is refused |
|---|---|
| `--dx 0 --dy 0` | Nothing moved means nothing could be measured. That is not a measurement of zero. |
| `--dx`/`--dy` above `max_mouse_delta` | Refused, not clamped — the same rule as `input-test`. |
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

## The observer page

```powershell
python -m autocraft observer            # http://127.0.0.1:8765
python -m autocraft observer --demo     # same page, synthetic data
```

A local, read-only page that shows what AutoCraft is doing internally: the
latest captured frame, the current mode, goal and intention, the last action
and its result, confidence, recent events, the current simulated affect, the
safety state, and — during `look-test` — the **Visual motion** panel.

The Visual motion panel shows the measured primitives of the LOOK-001 run: the
difference statistics, the per-block difference map, the estimated shift, the
pixels-per-delta ratio, and the reversibility figure. It reads
`status: not-run` until a trial has actually been measured, and it never fills
itself in from the plan.

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

**Truthful empty states.** V0 has no perception: no object detector, no tree
recognition, no privileged state. Beliefs and detected entities are therefore
displayed as empty, and say so. The page never invents perception to look
busier than the agent is.

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

The suite is 643 tests and runs in about 21 seconds. Everything that talks to
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
how far the view actually rotates, perception, navigation, model-backed
decisions, and long-running autonomous play. A `mouse-move` of `(10, 0)` turned
the camera *some* amount; it says nothing about the pixels-per-count ratio.

**The LOOK-001 trial has not been run against the live game.** The tool for it
exists — `look-test`, documented above — and the machinery behind it is covered
by the suite, but every one of those tests uses synthetic frames and a fake
mouse. So at this commit there is **no live measurement**: no pixels-per-delta
ratio, no reversibility figure, and no evidence about how repeatable either is.
Those numbers are what the next manual run produces, and until it happens they
should be treated as unknown rather than as approximately known.

**A measured ratio is a ratio for one configuration.** Even once the trial has
run, a pixels-per-delta figure is specific to that window size, that field of
view, that in-game sensitivity, and that mouse setting. It is not a property of
AutoCraft, and it does not transfer to a different setup without being measured
again.

---

## Where this goes next

**VISION-001 / LOOK-001** is the recommended next milestone: capture a
continuous stream of gameplay frames while deliberately rotating the camera,
and measure what actually changes on screen. Before any of TREE-001 can be
attempted, the project needs to know what "looking around" does to pixels, and
how much of a frame is stable while the camera moves.

The bounded version of that is implemented: `look-test` injects a known relative
mouse movement, captures A/B/C, and reports the measured displacement and its
reversibility, with a calibration series for the mapping ratio. What it has not
done is run against the live game. That is the next step, and it is a manual
one:

```powershell
python -m autocraft look-test --dx 10 --dy 0 --steps 1
```

Everything beyond that — a continuous look stream, stability maps, a
perception layer — is still not started. V0 stops at a foundation that is
boring, inspectable, and trustworthy.
