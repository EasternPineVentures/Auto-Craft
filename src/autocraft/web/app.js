/* AutoCraft Observer - polling client.
 *
 * No framework, no build step, no websocket: two GET requests and a local
 * clock. The page only ever reads. There is no fetch() that is not a GET, and
 * no code path here that could ask the agent to do anything.
 *
 * Two details worth knowing:
 *
 *  - The frame is fetched only when its identity changes. The snapshot carries
 *    `frame.captured_at`, which is a new value every time the agent publishes a
 *    frame, so the <img> src is rebuilt from it. Polling the state therefore
 *    stays cheap and the JPEG is not re-downloaded 4 times a second.
 *
 *  - The frame age is computed locally between polls, from the server/local clock
 *    offset measured at each poll. Without that, a "0.4 s old" reading would sit
 *    frozen for a whole poll interval and the LIVE badge would lag reality.
 */

"use strict";

(function () {
  const FALLBACK_POLL_MS = 1000;
  const AGE_TICK_MS = 250;

  const AFFECT_ORDER = ["curiosity", "confidence", "stress", "frustration", "energy"];

  const nodes = {
    demoBadge: document.getElementById("demo-badge"),
    demoNotice: document.getElementById("demo-notice"),
    connection: document.getElementById("connection"),
    runId: document.getElementById("run-id"),

    frameFreshness: document.getElementById("frame-freshness"),
    frameAge: document.getElementById("frame-age"),
    frameImage: document.getElementById("frame-image"),
    frameEmpty: document.getElementById("frame-empty"),
    frameSize: document.getElementById("frame-size"),
    frameSource: document.getElementById("frame-source"),
    frameLuma: document.getElementById("frame-luma"),
    frameSignature: document.getElementById("frame-signature"),

    lookStatus: document.getElementById("look-status"),
    lookTrial: document.getElementById("look-trial"),
    lookEmpty: document.getElementById("look-empty"),
    lookBody: document.getElementById("look-body"),
    lookDelta: document.getElementById("look-delta"),
    lookMoves: document.getElementById("look-moves"),
    lookWindow: document.getElementById("look-window"),
    lookSettle: document.getElementById("look-settle"),
    lookCapture: document.getElementById("look-capture"),
    lookMad: document.getElementById("look-mad"),
    lookRmse: document.getElementById("look-rmse"),
    lookChanged: document.getElementById("look-changed"),
    lookShiftX: document.getElementById("look-shift-x"),
    lookShiftY: document.getElementById("look-shift-y"),
    lookQuality: document.getElementById("look-quality"),
    lookReversibility: document.getElementById("look-reversibility"),
    lookGridSize: document.getElementById("look-grid-size"),
    lookGrid: document.getElementById("look-grid"),
    lookNote: document.getElementById("look-note"),

    wakeStatus: document.getElementById("wake-status"),
    wakeMoves: document.getElementById("wake-moves"),
    wakeEmpty: document.getElementById("wake-empty"),
    wakeBody: document.getElementById("wake-body"),
    wakeState: document.getElementById("wake-state"),
    wakeStrategy: document.getElementById("wake-strategy"),
    wakeTarget: document.getElementById("wake-target"),
    wakeOffset: document.getElementById("wake-offset"),
    wakeConfidence: document.getElementById("wake-confidence"),
    wakeViews: document.getElementById("wake-views"),
    wakeGuard: document.getElementById("wake-guard"),
    wakeEvent: document.getElementById("wake-event"),
    wakeProgress: document.getElementById("wake-progress"),
    wakeProgressScale: document.getElementById("wake-progress-scale"),
    wakeProgressNote: document.getElementById("wake-progress-note"),
    wakeNote: document.getElementById("wake-note"),

    safetyPanel: document.querySelector(".panel-safety"),
    safetyVerdict: document.getElementById("safety-verdict"),
    sWindow: document.getElementById("s-window"),
    sFocus: document.getElementById("s-focus"),
    sInput: document.getElementById("s-input"),
    sEstop: document.getElementById("s-estop"),
    sHeld: document.getElementById("s-held"),

    mode: document.getElementById("mode"),
    modeNote: document.getElementById("mode-note"),
    goal: document.getElementById("goal"),
    intention: document.getElementById("intention"),
    lastAction: document.getElementById("last-action"),
    actionResult: document.getElementById("action-result"),
    observation: document.getElementById("observation"),
    confidenceValue: document.getElementById("confidence-value"),
    confidenceBar: document.getElementById("confidence-bar"),
    beliefsEmpty: document.getElementById("beliefs-empty"),
    beliefList: document.getElementById("belief-list"),

    affectList: document.getElementById("affect-list"),

    thoughtTone: document.getElementById("thought-tone"),
    thoughtIntensity: document.getElementById("thought-intensity"),
    thoughtText: document.getElementById("thought-text"),
    thoughtMeta: document.getElementById("thought-meta"),
    thoughtHistory: document.getElementById("thought-history"),

    mRun: document.getElementById("m-run"),
    mRuntime: document.getElementById("m-runtime"),
    mFrames: document.getElementById("m-frames"),
    mCaptureFps: document.getElementById("m-capture-fps"),
    mLoopRate: document.getElementById("m-loop-rate"),
    mAttempted: document.getElementById("m-attempted"),
    mExecuted: document.getElementById("m-executed"),
    mBlocked: document.getElementById("m-blocked"),
    mSafety: document.getElementById("m-safety"),
    mErrors: document.getElementById("m-errors"),
    mThoughts: document.getElementById("m-thoughts"),
    mStreak: document.getElementById("m-streak"),
    blockReason: document.getElementById("block-reason"),

    eventList: document.getElementById("event-list"),
    eventsEmpty: document.getElementById("events-empty"),
    eventCount: document.getElementById("event-count"),
    memoryList: document.getElementById("memory-list"),
    memoryEmpty: document.getElementById("memory-empty"),

    footerStatus: document.getElementById("footer-status"),
    footerPoll: document.getElementById("footer-poll"),
  };

  const view = {
    liveSeconds: 1.0,
    staleSeconds: 5.0,
    pollMs: FALLBACK_POLL_MS,
    offset: 0,
    haveOffset: false,
    frameKey: null,
    connected: false,
    lastEventsKey: "",
    lastMemoryKey: "",
    affectRows: new Map(),
    beliefKey: "",
    historyKey: "",
  };

  // -- helpers ------------------------------------------------------------

  const localNow = () => Date.now() / 1000;
  const serverNow = () => localNow() + view.offset;

  function setText(node, value) {
    if (!node) return;
    const text = value === null || value === undefined || value === "" ? "-" : String(value);
    if (node.textContent !== text) node.textContent = text;
  }

  function setBar(node, value) {
    if (!node) return;
    const clamped = Math.max(0, Math.min(1, Number(value) || 0));
    node.style.setProperty("--v", String(clamped));
  }

  function setClass(node, className) {
    if (node && node.className !== className) node.className = className;
  }

  function clock(seconds, withSeconds) {
    if (seconds === null || seconds === undefined) return "-";
    const date = new Date(seconds * 1000);
    return date.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: withSeconds ? "2-digit" : undefined,
      hour12: false,
    });
  }

  function duration(seconds) {
    if (seconds === null || seconds === undefined || !isFinite(seconds)) return "-";
    if (seconds < 60) return seconds.toFixed(1) + " s";
    const minutes = Math.floor(seconds / 60);
    const rest = Math.floor(seconds % 60);
    if (minutes < 60) return minutes + " m " + String(rest).padStart(2, "0") + " s";
    const hours = Math.floor(minutes / 60);
    return hours + " h " + String(minutes % 60).padStart(2, "0") + " m";
  }

  function freshness(age) {
    if (age === null || age === undefined || !isFinite(age)) return "none";
    if (age <= view.liveSeconds) return "live";
    if (age <= view.staleSeconds) return "recent";
    return "stale";
  }

  function freshnessLabel(level) {
    if (level === "live") return "LIVE";
    if (level === "recent") return "RECENT";
    if (level === "stale") return "STALE";
    return "NO FRAME";
  }

  function freshnessClass(level) {
    if (level === "live") return "badge badge-ok";
    if (level === "recent") return "badge badge-warn";
    if (level === "stale") return "badge badge-danger";
    return "badge badge-muted";
  }

  function renderFrameAge() {
    const frame = view.snapshot && view.snapshot.frame;
    if (!frame || !frame.available || !frame.captured_at) {
      setText(nodes.frameAge, "-");
      setClass(nodes.frameFreshness, freshnessClass("none"));
      setText(nodes.frameFreshness, freshnessLabel("none"));
      return;
    }
    const age = Math.max(0, (view.haveOffset ? serverNow() : localNow()) - frame.captured_at);
    const level = freshness(age);
    setText(nodes.frameAge, level === "live" ? "age " + age.toFixed(1) + " s" : "age " + duration(age));
    setClass(nodes.frameFreshness, freshnessClass(level));
    setText(nodes.frameFreshness, freshnessLabel(level));
  }

  function renderFrame(snapshot) {
    const frame = snapshot.frame || {};
    if (!frame.available) {
      nodes.frameImage.hidden = true;
      nodes.frameEmpty.hidden = false;
      view.frameKey = null;
    } else {
      nodes.frameImage.hidden = false;
      nodes.frameEmpty.hidden = true;
      if (view.frameKey !== frame.captured_at) {
        view.frameKey = frame.captured_at;
        nodes.frameImage.src = "/frame?t=" + encodeURIComponent(String(frame.captured_at));
      }
    }
    setText(nodes.frameSize, frame.available ? frame.width + " x " + frame.height : "-");
    setText(
      nodes.frameSource,
      frame.available && frame.source_width
        ? frame.source + " (" + frame.source_width + " x " + frame.source_height + ")"
        : frame.source || "-"
    );
    setText(nodes.frameLuma, frame.mean_luma === null || frame.mean_luma === undefined ? "-" : frame.mean_luma);
    setText(nodes.frameSignature, frame.signature);
    renderFrameAge();
  }

  // -- visual motion (LOOK-001) -------------------------------------------

  function renderLookGrid(report) {
    const grid = report.block_map || [];
    const size = report.block_grid || 0;
    nodes.lookGrid.replaceChildren();
    if (!grid.length) {
      setText(nodes.lookGridSize, "-");
      nodes.lookGrid.style.setProperty("--cols", "1");
      return;
    }
    setText(nodes.lookGridSize, size + " x " + grid.length);
    nodes.lookGrid.style.setProperty("--cols", String(grid[0].length));
    // One cell per block, filled by that block's mean absolute difference on a
    // fixed 0-255 scale. This is a readout of the stored measurement, not a
    // recomputation, so the grid can never disagree with the numbers beside it.
    grid.forEach(function (row) {
      row.forEach(function (value) {
        const cell = document.createElement("span");
        const level = Math.max(0, Math.min(1, Number(value) / 255));
        cell.className = "look-cell";
        cell.style.setProperty("--v", level.toFixed(4));
        cell.title = Number(value).toFixed(1);
        nodes.lookGrid.append(cell);
      });
    });
  }

  function renderLook(snapshot) {
    const look = snapshot.look || {};
    const available = Boolean(look.available);
    nodes.lookEmpty.hidden = available;
    nodes.lookBody.hidden = !available;

    setText(nodes.lookStatus, look.status || "not run");
    setClass(
      nodes.lookStatus,
      available
        ? look.status === "completed"
          ? "badge badge-ok"
          : "badge badge-warn"
        : "badge badge-muted"
    );

    if (!available) {
      setText(nodes.lookTrial, "-");
      nodes.lookGrid.replaceChildren();
      return;
    }

    setText(
      nodes.lookTrial,
      "trial " + (look.trial_index === null || look.trial_index === undefined ? "?" : look.trial_index + 1) +
        " of " + (look.trial_count || 0)
    );
    const dx = look.dx === null || look.dx === undefined ? "-" : look.dx;
    const dy = look.dy === null || look.dy === undefined ? "-" : look.dy;
    setText(nodes.lookDelta, "(" + dx + ", " + dy + ")");
    setText(nodes.lookMoves, look.movements_sent);
    const window = look.window || {};
    setText(nodes.lookWindow, window.width && window.height ? window.width + " x " + window.height : "-");
    setText(nodes.lookSettle, look.settle_seconds === null || look.settle_seconds === undefined ? "-" : look.settle_seconds + " s");
    setText(nodes.lookCapture, look.capture_seconds === null || look.capture_seconds === undefined ? "-" : look.capture_seconds + " s");
    setText(nodes.lookMad, look.mean_absolute_difference);
    setText(nodes.lookRmse, look.rmse);
    setText(
      nodes.lookChanged,
      look.changed_fraction === null || look.changed_fraction === undefined
        ? "-"
        : (Number(look.changed_fraction) * 100).toFixed(2) + "%"
    );

    const shift = look.shift || {};
    setText(nodes.lookShiftX, shift.available ? shift.x + " px" : "not estimated");
    setText(nodes.lookShiftY, shift.available ? shift.y + " px" : "not estimated");
    setText(nodes.lookQuality, shift.available ? shift.quality : "-");

    const ratio = look.reversibility_ratio;
    setText(
      nodes.lookReversibility,
      ratio === null || ratio === undefined
        ? "not measurable"
        : Number(ratio).toFixed(3)
    );

    const perDelta = look.pixels_per_delta || {};
    if (perDelta.x === null || perDelta.x === undefined) {
      setText(nodes.lookNote, look.reversibility_note || "-");
    } else {
      setText(
        nodes.lookNote,
        Number(perDelta.x).toFixed(4) + " px per unit of injected delta. " +
          (look.reversibility_note || "")
      );
    }

    renderLookGrid(look);
  }

  // -- waking behaviour (WAKE-001) ----------------------------------------

  // The centring history as a small column chart: one bar per measured
  // distance, oldest on the left, scaled against the first and largest sample.
  // A falling staircase is the whole point of the panel, so the bar heights are
  // the raw measurement and nothing is smoothed or rescaled to look better.
  function renderWakeProgress(wake) {
    const progress = wake.progress || [];
    nodes.wakeProgress.replaceChildren();
    if (!progress.length) {
      setText(nodes.wakeProgressScale, "-");
      setText(nodes.wakeProgressNote, "no centring move has been measured yet");
      nodes.wakeProgress.style.setProperty("--cols", "1");
      return;
    }
    const peak = Math.max.apply(null, progress.map(Number));
    nodes.wakeProgress.style.setProperty("--cols", String(progress.length));
    setText(nodes.wakeProgressScale, peak.toFixed(1) + " px -> 0");
    progress.forEach(function (value) {
      const bar = document.createElement("span");
      const level = peak > 0 ? Math.max(0, Math.min(1, Number(value) / peak)) : 0;
      bar.className = "wake-bar";
      bar.style.setProperty("--v", level.toFixed(4));
      bar.title = Number(value).toFixed(1) + " px";
      nodes.wakeProgress.append(bar);
    });
    const first = Number(progress[0]);
    const last = Number(progress[progress.length - 1]);
    setText(
      nodes.wakeProgressNote,
      progress.length +
        " move(s): " +
        first.toFixed(1) +
        " px -> " +
        last.toFixed(1) +
        " px" +
        (last < first ? " (closer)" : " (not closer)")
    );
  }

  function renderWake(snapshot) {
    const wake = snapshot.wake || {};
    const available = Boolean(wake.available);
    nodes.wakeEmpty.hidden = available;
    nodes.wakeBody.hidden = !available;

    setText(nodes.wakeStatus, wake.status || "not run");
    setClass(
      nodes.wakeStatus,
      available
        ? wake.status === "completed"
          ? "badge badge-ok"
          : wake.status === "failed"
            ? "badge badge-warn"
            : "badge badge-muted"
        : "badge badge-muted"
    );

    if (!available) {
      setText(nodes.wakeMoves, "-");
      setText(nodes.wakeState, "STARTING");
      setText(nodes.wakeStrategy, "-");
      setText(nodes.wakeTarget, "none");
      setText(nodes.wakeOffset, "-");
      setText(nodes.wakeConfidence, "-");
      setText(nodes.wakeViews, "-");
      setText(nodes.wakeGuard, "clear");
      setText(nodes.wakeEvent, "-");
      nodes.wakeProgress.replaceChildren();
      return;
    }

    setText(
      nodes.wakeMoves,
      (wake.moves_sent || 0) + " / " + (wake.max_moves || 0) + " moves"
    );
    setText(nodes.wakeState, wake.state || "-");
    setText(nodes.wakeStrategy, wake.strategy || "no strategy chosen yet");

    const target = wake.target || {};
    const centre = target.centre;
    if (centre && centre.length === 2) {
      setText(nodes.wakeTarget, "(" + centre[0].toFixed(0) + ", " + centre[1].toFixed(0) + ") px");
    } else {
      setText(nodes.wakeTarget, "none selected");
    }

    const offset = wake.target_offset;
    if (offset && offset.length === 2) {
      const distance = wake.target_distance;
      setText(
        nodes.wakeOffset,
        "(" + Number(offset[0]).toFixed(1) + ", " + Number(offset[1]).toFixed(1) + ") px" +
          (distance === null || distance === undefined ? "" : "  |  " + Number(distance).toFixed(1) + " px out")
      );
    } else {
      setText(nodes.wakeOffset, "-");
    }

    const confidence = wake.confidence;
    setText(
      nodes.wakeConfidence,
      confidence === null || confidence === undefined ? "not measured" : Number(confidence).toFixed(3)
    );
    setText(
      nodes.wakeViews,
      (wake.unique_views || 0) + " (" + (wake.revisited_views || 0) + " revisits)"
    );

    const guard = wake.repeat_guard || {};
    if (guard.stuck) {
      setText(nodes.wakeGuard, "STUCK x" + (guard.repeats || 0) + " - changing strategy");
      setClass(nodes.wakeGuard, "mono wake-guard-stuck");
    } else {
      setText(
        nodes.wakeGuard,
        guard.cooldowns_active
          ? "clear (" + guard.cooldowns_active + " strategy on cooldown)"
          : "clear"
      );
      setClass(nodes.wakeGuard, "mono");
    }

    setText(nodes.wakeEvent, wake.recent_event || "-");

    // The mapping line is the one place the panel states a limitation rather
    // than a number, because "unmeasured" is the honest and interesting case.
    const mapping = wake.mapping || {};
    if (mapping.source === "unmeasured") {
      setText(
        nodes.wakeNote,
        "Mouse mapping unmeasured: corrections were sized by a fixed band count, " +
          "so the agent did not know how far a count would move the view."
      );
    } else {
      setText(
        nodes.wakeNote,
        "Mouse mapping " + mapping.source + ": " +
          Number(mapping.pixels_per_delta_x).toFixed(4) + " px/count x, " +
          Number(mapping.pixels_per_delta_y).toFixed(4) + " px/count y."
      );
    }

    renderWakeProgress(wake);
  }

  // -- safety -------------------------------------------------------------

  function safetyVerdict(safety) {
    if (safety.emergency_stop === "triggered") {
      return { label: "EMERGENCY STOP", badge: "badge badge-danger", panel: "panel panel-safety is-stop" };
    }
    if (!safety.window_found) {
      return { label: "target missing", badge: "badge badge-warn", panel: "panel panel-safety is-warn" };
    }
    if (!safety.window_foreground) {
      return { label: "not focused", badge: "badge badge-warn", panel: "panel panel-safety is-warn" };
    }
    if (!safety.input_enabled) {
      return { label: "input blocked", badge: "badge badge-warn", panel: "panel panel-safety is-warn" };
    }
    return { label: "nominal", badge: "badge badge-ok", panel: "panel panel-safety" };
  }

  function renderSafety(snapshot) {
    const safety = snapshot.safety || {};
    const verdict = safetyVerdict(safety);
    setText(nodes.safetyVerdict, verdict.label);
    setClass(nodes.safetyVerdict, verdict.badge);
    setClass(nodes.safetyPanel, verdict.panel);

    setText(nodes.sWindow, safety.window_found ? "found" : "missing");
    setClass(nodes.sWindow, "status-value " + (safety.window_found ? "ok" : "bad"));

    setText(nodes.sFocus, safety.window_foreground ? "foreground" : "background");
    setClass(nodes.sFocus, "status-value " + (safety.window_foreground ? "ok" : "warn"));

    setText(nodes.sInput, safety.input_enabled ? "enabled" : "blocked");
    setClass(nodes.sInput, "status-value " + (safety.input_enabled ? "ok" : "warn"));

    setText(nodes.sEstop, safety.emergency_stop === "triggered" ? "TRIGGERED" : "ready");
    setClass(nodes.sEstop, "status-value " + (safety.emergency_stop === "triggered" ? "bad" : "ok"));

    const held = safety.held_inputs || [];
    setText(nodes.sHeld, held.length ? held.join(", ") : "none");
    setClass(nodes.sHeld, "status-value " + (held.length ? "warn" : ""));
  }

  // -- agent state --------------------------------------------------------

  function modeClass(mode) {
    if (mode === "SAFE_STOP" || mode === "ERROR") return "badge badge-danger";
    if (mode === "PAUSED") return "badge badge-warn";
    if (mode === "ACTING") return "badge badge-tone";
    return "badge badge-mode";
  }

  function renderState(snapshot) {
    setText(nodes.mode, snapshot.mode);
    setClass(nodes.mode, modeClass(snapshot.mode));
    setText(nodes.modeNote, snapshot.mode_note);
    setText(nodes.goal, snapshot.current_goal);
    setText(nodes.intention, snapshot.current_intention);
    setText(nodes.lastAction, snapshot.last_action);
    setText(nodes.actionResult, snapshot.action_result);
    setText(nodes.observation, snapshot.observation_summary);
    setText(
      nodes.confidenceValue,
      snapshot.confidence === null || snapshot.confidence === undefined
        ? "-"
        : snapshot.confidence.toFixed(2)
    );
    setBar(nodes.confidenceBar, snapshot.confidence === null ? 0 : snapshot.confidence);
    renderBeliefs(snapshot);
  }

  function renderBeliefs(snapshot) {
    const beliefs = (snapshot.beliefs || []).concat(snapshot.detected_entities || []);
    const key = JSON.stringify(beliefs);
    if (key === view.beliefKey) return;
    view.beliefKey = key;
    nodes.beliefList.replaceChildren();
    nodes.beliefsEmpty.hidden = beliefs.length > 0;
    beliefs.forEach(function (belief) {
      const item = document.createElement("li");
      const label = document.createElement("span");
      label.textContent = belief.label;
      const confidence = document.createElement("span");
      confidence.className = "mono muted";
      confidence.textContent = Number(belief.confidence).toFixed(2);
      item.append(label, confidence);
      nodes.beliefList.append(item);
    });
  }

  // -- affect -------------------------------------------------------------

  function ensureAffectRows() {
    if (view.affectRows.size === AFFECT_ORDER.length) return;
    nodes.affectList.replaceChildren();
    view.affectRows.clear();
    AFFECT_ORDER.forEach(function (name) {
      const item = document.createElement("li");
      const label = document.createElement("span");
      label.className = "affect-name";
      label.textContent = name;
      const bar = document.createElement("div");
      bar.className = "bar bar-" + name;
      const fill = document.createElement("span");
      bar.append(fill);
      const value = document.createElement("span");
      value.className = "affect-value";
      item.append(label, bar, value);
      nodes.affectList.append(item);
      view.affectRows.set(name, { fill: fill, value: value });
    });
  }

  function renderAffect(snapshot) {
    ensureAffectRows();
    const affect = snapshot.affect_state || {};
    AFFECT_ORDER.forEach(function (name) {
      const row = view.affectRows.get(name);
      if (!row) return;
      const value = Number(affect[name]);
      const safe = isFinite(value) ? value : 0;
      setBar(row.fill, safe);
      if (row.value.textContent !== safe.toFixed(2)) row.value.textContent = safe.toFixed(2);
    });
  }

  // -- thought ------------------------------------------------------------

  function renderThought(snapshot) {
    const latest = snapshot.latest_thought;
    if (!latest) {
      setText(nodes.thoughtTone, "-");
      setClass(nodes.thoughtTone, "badge badge-tone");
      setText(nodes.thoughtIntensity, "-");
      setText(
        nodes.thoughtText,
        "Nothing expressed yet. Thoughts are paced by a cooldown, so silence is normal."
      );
      setText(nodes.thoughtMeta, "");
    } else {
      setText(nodes.thoughtTone, latest.tone);
      setClass(nodes.thoughtTone, "badge badge-tone");
      setText(nodes.thoughtIntensity, "intensity " + Number(latest.intensity).toFixed(2));
      setText(nodes.thoughtText, latest.text);
      const parts = [
        clock(latest.timestamp, true),
        latest.trigger_type,
        latest.generated_by,
      ];
      if (latest.related_goal) parts.push("goal: " + latest.related_goal);
      if (latest.related_memory) parts.push("memory: " + latest.related_memory);
      setText(nodes.thoughtMeta, parts.join("  ·  "));
    }
    renderThoughtHistory(snapshot);
  }

  function renderThoughtHistory(snapshot) {
    const history = snapshot.thought_history || [];
    const key = history.length + "|" + (history.length ? history[0].id : "");
    if (key === view.historyKey) return;
    view.historyKey = key;
    nodes.thoughtHistory.replaceChildren();
    // The first entry is the one already shown in the large panel.
    history.slice(1).forEach(function (thought) {
      const item = document.createElement("li");
      const time = document.createElement("span");
      time.className = "thought-time";
      time.textContent = clock(thought.timestamp, false);
      const text = document.createElement("span");
      text.textContent = thought.text;
      item.append(time, text);
      nodes.thoughtHistory.append(item);
    });
  }

  // -- metrics ------------------------------------------------------------

  function renderMetrics(snapshot) {
    const metrics = snapshot.metrics || {};
    setText(nodes.mRun, metrics.run_id);
    setText(nodes.mRuntime, duration(metrics.runtime_seconds));
    setText(nodes.mFrames, metrics.frames_observed);
    setText(nodes.mCaptureFps, Number(metrics.capture_fps || 0).toFixed(2));
    setText(nodes.mLoopRate, Number(metrics.loop_rate || 0).toFixed(2));
    setText(nodes.mAttempted, metrics.actions_attempted);
    setText(nodes.mExecuted, metrics.actions_executed);
    setText(nodes.mBlocked, metrics.actions_blocked);
    setText(nodes.mSafety, metrics.safety_events);
    setText(nodes.mErrors, metrics.errors);
    setText(nodes.mThoughts, metrics.thoughts_expressed);
    setText(nodes.mStreak, (snapshot.safety || {}).blocked_streak);
    const reason = (snapshot.safety || {}).last_block_reason;
    setText(nodes.blockReason, reason ? "last refusal: " + reason : "");
  }

  // -- events -------------------------------------------------------------

  function fillEventList(list, events, newestFirst) {
    list.replaceChildren();
    const ordered = newestFirst ? events.slice().reverse() : events;
    ordered.forEach(function (event) {
      const item = document.createElement("li");
      const time = document.createElement("span");
      time.className = "event-time";
      time.textContent = clock(event.timestamp, true);
      const kind = document.createElement("span");
      kind.className = "event-kind k-" + event.kind;
      kind.textContent = event.kind;
      const message = document.createElement("span");
      message.className = "event-message";
      message.textContent = event.message;
      item.append(time, kind, message);
      list.append(item);
    });
  }

  function keyOf(events) {
    if (!events.length) return "0";
    const first = events[0];
    const last = events[events.length - 1];
    return events.length + "|" + first.timestamp + "|" + last.timestamp + "|" + last.message;
  }

  function renderEvents(snapshot) {
    const events = snapshot.recent_events || [];
    setText(nodes.eventCount, events.length + " retained");
    nodes.eventsEmpty.hidden = events.length > 0;
    const key = keyOf(events);
    if (key !== view.lastEventsKey) {
      view.lastEventsKey = key;
      fillEventList(nodes.eventList, events, true);
    }
  }

  function renderMemory(snapshot) {
    const memory = snapshot.short_term_memory || [];
    nodes.memoryEmpty.hidden = memory.length > 0;
    const key = keyOf(memory);
    if (key !== view.lastMemoryKey) {
      view.lastMemoryKey = key;
      // Already newest-first, so the most recent entry stays at the top.
      fillEventList(nodes.memoryList, memory, false);
    }
  }

  // -- connection ---------------------------------------------------------

  function setConnected(connected) {
    if (view.connected === connected) return;
    view.connected = connected;
    setClass(nodes.connection, connected ? "badge badge-ok" : "badge badge-danger");
    setText(nodes.connection, connected ? "connected" : "disconnected");
    setText(nodes.footerStatus, connected ? "receiving state" : "no contact with the observer server");
  }

  // -- polling ------------------------------------------------------------

  function render(snapshot) {
    view.snapshot = snapshot;
    setText(nodes.runId, snapshot.run_id);
    nodes.demoBadge.hidden = !snapshot.demo;
    nodes.demoNotice.hidden = !snapshot.demo;
    renderFrame(snapshot);
    renderLook(snapshot);
  renderWake(snapshot);
    renderSafety(snapshot);
    renderState(snapshot);
    renderAffect(snapshot);
    renderThought(snapshot);
    renderMetrics(snapshot);
    renderEvents(snapshot);
    renderMemory(snapshot);
  }

  async function pollSnapshot() {
    try {
      const response = await fetch("/api/snapshot", { cache: "no-store" });
      if (!response.ok) throw new Error("HTTP " + response.status);
      const snapshot = await response.json();
      view.offset = Number(snapshot.timestamp) - localNow();
      view.haveOffset = isFinite(view.offset);
      setConnected(true);
      render(snapshot);
    } catch (error) {
      setConnected(false);
    }
  }

  async function readHealth() {
    try {
      const response = await fetch("/api/health", { cache: "no-store" });
      if (!response.ok) throw new Error("HTTP " + response.status);
      const health = await response.json();
      if (isFinite(health.frame_live_seconds)) view.liveSeconds = health.frame_live_seconds;
      if (isFinite(health.frame_stale_seconds)) view.staleSeconds = health.frame_stale_seconds;
      if (isFinite(health.poll_ms) && health.poll_ms >= 200) view.pollMs = health.poll_ms;
    } catch (error) {
      // The snapshot poll reports connectivity; the health probe is optional.
    }
    setText(nodes.footerPoll, view.pollMs + " ms");
  }

  async function loop() {
    await pollSnapshot();
    window.setTimeout(loop, view.pollMs);
  }

  function start() {
    window.setInterval(renderFrameAge, AGE_TICK_MS);
    readHealth();
    window.setInterval(readHealth, 30000);
    loop();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
