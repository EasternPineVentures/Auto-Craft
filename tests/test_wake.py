"""Tests for WAKE-001: look around, notice something, turn toward it, stop.

What is pinned here, and why:

* **The behaviour always terminates.** WAKE-001 is the first thing in AutoCraft
  that decides its own movements, so the bound is the safety property that
  matters most. ``test_a_blank_scene_gives_up_within_the_budget`` and the BOUNDS
  group below run the whole policy against scenes it cannot succeed on and assert
  it stops anyway.
* **The centring step is proportional to the measured mapping, not to the band.**
  This was a real bug. Capping the step at the band's fixed count meant a
  measured ``0.5`` px/count mapping needed 29 moves to close 1611 px, against a
  budget of 8. ``test_a_measured_mapping_sizes_the_step_not_the_band`` is what
  keeps the cap from coming back.
* **A uniform scene yields no target.** The salience stage must be able to say
  "nothing here stands out" rather than picking the least flat cell.
* **Repetition is only dead when it stops making progress.** A descending
  distance series is allowed to repeat the same movement; a flat one is not.
  ``test_a_productive_streak_is_not_stuck`` pins the distinction.
* **The package cannot inject input and cannot recognise anything.** Pinned
  structurally, by reading the syntax tree, so it keeps holding for a contributor
  who never reads this file.
* **The observer stays read-only.** The WAKE panel is a display, and the import
  ban is what makes that a structural fact rather than a promise.

No test injects real input. Every frame is synthetic and every actuator is a fake,
which is what the milestone requires.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autocraft.agent.action import Action, ActionKind
from autocraft.agent.decision import DecisionPolicy
from autocraft.agent.observation import Observation, WindowStatus
from autocraft.config import Config
from autocraft.observer.snapshot import WakeReport
from autocraft.observer.state import ObserverState
from autocraft.telemetry.recorder import RunRecorder
from autocraft.vision.frame import Frame
from autocraft.wake import (
    EXPERIMENT_NAME,
    LIMITATION_NOTES,
    STATUS_ABORTED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STREAM_SUMMARY,
    CenteringController,
    MotionCalibration,
    ProgressModel,
    RepetitionGuard,
    ShortTermExperience,
    StrategyCooldown,
    TargetPatch,
    ViewFingerprint,
    ViewMemory,
    WakeDecisionPolicy,
    WakeEvent,
    WakeEventKind,
    WakeRecorder,
    WakeResult,
    WakeRunner,
    WakeState,
    band_for_distance,
    find_candidates,
    locate,
    relocate,
    select_candidate,
    status_for_state,
    stream_summary,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def scene(seed: int = 7, height: int = 120, width: int = 160) -> np.ndarray:
    """A low-contrast noisy scene, the way the real world region looks."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 14, size=(height, width, 3), dtype=np.uint8)


def with_blob(
    image: np.ndarray, *, y0: int, y1: int, x0: int, x1: int, level: int = 230
) -> np.ndarray:
    """A copy of ``image`` with one bright striped rectangle painted on it."""
    out = image.copy()
    for y in range(y0, y1):
        for x in range(x0, x1):
            band = level if ((x + y) // 5) % 2 == 0 else level // 3
            out[y, x] = (band, band, band)
    return out


def frame_of(image: np.ndarray, index: int = 0) -> Frame:
    return Frame(image, float(index))


def panned(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """``image`` shifted by whole pixels, filling the gap with the background."""
    out = np.full_like(image, 6)
    height, width = image.shape[:2]
    sy0, sy1 = max(0, dy), min(height, height + dy)
    sx0, sx1 = max(0, dx), min(width, width + dx)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0:sy1, sx0:sx1] = image[sy0 - dy : sy1 - dy, sx0 - dx : sx1 - dx]
    return out


def gradient_sky(height: int = 120, width: int = 160, seed: int = 7) -> np.ndarray:
    """A smooth vertical gradient with faint noise: the shape of an empty sky.

    This is the frame that broke the first live run. Every salience cue is
    normalised by its own frame maximum, so a frame with almost no variation has
    its small variation inflated to fill the range, every cell clears the
    threshold, and the "region" the map reports is the entire frame.
    """
    rng = np.random.default_rng(seed)
    ramp = np.linspace(150.0, 168.0, height, dtype=np.float32)[:, None, None]
    image = np.repeat(ramp, width, axis=1).repeat(3, axis=2)
    image += rng.normal(0.0, 1.2, size=image.shape).astype(np.float32)
    return np.clip(image, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# VIEW MEMORY
# ---------------------------------------------------------------------------


def test_an_identical_view_is_recognised_as_a_revisit() -> None:
    memory = ViewMemory()
    image = with_blob(scene(), y0=30, y1=70, x0=40, x1=90)

    first = memory.observe(ViewFingerprint.of(frame_of(image)), index=0, timestamp=0.0)
    assert first.revisited is False
    assert memory.unique_count == 1

    again = memory.observe(ViewFingerprint.of(frame_of(image)), index=1, timestamp=1.0)
    assert again.revisited is True
    assert again.similarity == pytest.approx(1.0)
    assert memory.unique_count == 1
    assert memory.revisit_count == 1


def test_a_different_view_is_not_a_revisit() -> None:
    """VIEW_REVISITED must not fire on a view the agent has never seen."""
    memory = ViewMemory()
    left = with_blob(scene(), y0=30, y1=70, x0=20, x1=60)
    right = with_blob(scene(), y0=30, y1=70, x0=100, x1=140)

    memory.observe(ViewFingerprint.of(frame_of(left)), index=0, timestamp=0.0)
    record, similarity = memory.best_match(ViewFingerprint.of(frame_of(right)))
    assert record is not None
    assert similarity < 0.92, "an unrelated view should not look like a revisit"


def test_a_scene_shifted_along_the_ground_still_matches_itself() -> None:
    """The fingerprint has to survive a camera pan, or memory is useless here."""
    image = with_blob(scene(), y0=30, y1=70, x0=60, x1=110)
    memory = ViewMemory()
    memory.observe(ViewFingerprint.of(frame_of(image)), index=0, timestamp=0.0)

    _, same = memory.best_match(ViewFingerprint.of(frame_of(image)))
    _, moved = memory.best_match(ViewFingerprint.of(frame_of(panned(image, 1, 0))))
    assert same == pytest.approx(1.0)
    assert moved > same - 0.2, "a one-pixel pan must not read as a different place"


def test_view_memory_is_bounded() -> None:
    memory = ViewMemory(limit=3)
    for index in range(6):
        image = with_blob(scene(seed=index), y0=20, y1=60, x0=10 + index, x1=40 + index)
        memory.observe(ViewFingerprint.of(frame_of(image)), index=index, timestamp=float(index))
    assert len(memory.records) <= 3
    assert len(memory) <= 3


def test_short_term_experience_counts_unique_and_revisited_views() -> None:
    experience = ShortTermExperience()
    image = with_blob(scene(), y0=30, y1=70, x0=40, x1=90)
    fingerprint = ViewFingerprint.of(frame_of(image))

    experience.note_view(fingerprint, index=0, timestamp=0.0)
    experience.note_view(fingerprint, index=1, timestamp=1.0)
    assert experience.unique_views == 1
    assert experience.revisited_views == 1


# ---------------------------------------------------------------------------
# SALIENCE
# ---------------------------------------------------------------------------


def test_a_uniform_scene_offers_no_candidate() -> None:
    """The honest answer to a blank view is "nothing stands out".

    A *constant* frame is the case that matters. A noisy low-contrast frame does
    contain local structure, so salience legitimately finds something in it; what
    must never happen is a featureless frame producing a candidate out of
    quantization noise.
    """
    assert find_candidates(np.zeros((120, 160, 3), dtype=np.uint8), grid=8) == []
    assert find_candidates(np.full((120, 160, 3), 20, dtype=np.uint8), grid=8) == []


def test_one_bright_region_becomes_one_candidate_around_it() -> None:
    image = with_blob(scene(), y0=30, y1=70, x0=40, x1=90)
    candidates = find_candidates(image, grid=8)
    assert len(candidates) == 1, f"expected a single blob, got {len(candidates)}"
    candidate = candidates[0]
    x0, y0, x1, y1 = candidate.bbox
    assert x0 <= 40 and x1 >= 90 and y0 <= 30 and y1 >= 70
    assert 40 <= candidate.centre[0] <= 90
    assert 30 <= candidate.centre[1] <= 70
    assert candidate.salience > 0.15


def test_two_separated_regions_become_two_candidates() -> None:
    image = scene()
    image = with_blob(image, y0=10, y1=40, x0=10, x1=40)
    image = with_blob(image, y0=80, y1=110, x0=110, x1=150)
    candidates = find_candidates(image, grid=8)
    assert len(candidates) == 2
    centres = sorted(candidate.centre[0] for candidate in candidates)
    assert centres[0] < 60 < centres[1]


def test_touching_regions_are_merged_rather_than_split() -> None:
    """A single visual object must not be reported as several candidates.

    Fragmentation is what makes a candidate set meaningless: three candidates
    that are all one thing turn "pick a target" into "pick a fragment".
    """
    image = scene()
    image = with_blob(image, y0=40, y1=60, x0=20, x1=60)
    image = with_blob(image, y0=60, y1=80, x0=60, x1=100)
    assert len(find_candidates(image, grid=8)) == 1


def test_a_frame_covering_region_is_not_a_candidate() -> None:
    """The whole frame is not a region, and cannot be treated as one.

    On the first live run every cell of an empty sky cleared the salience
    threshold, so the map's single group spanned the entire frame. That group
    was accepted as a target, and a target the size of the frame cannot be
    tracked: it leaves the matcher no position to search, so it "matches" itself
    perfectly wherever it is assumed to be, and the distance to it can never
    change. Declining the region is the only honest answer.
    """
    sky = gradient_sky()
    assert find_candidates(sky, grid=8) == [], "a sky-only frame has no region in it"


def test_a_frame_covering_region_is_refused_when_named_directly() -> None:
    """The same refusal must hold for the explicit "use these cells" entry point."""
    from autocraft.wake.salience import candidate_from_cells

    sky = gradient_sky()
    every_cell = tuple(range(64))
    assert candidate_from_cells(sky, every_cell, grid=8) is None


def test_a_region_spanning_the_frame_width_is_refused() -> None:
    """The rule is about spanning the frame, not about an exact match on its size.

    The full-width band used to be allowed through here, on the reasoning that a
    band spanning the full width "is a real feature". The live WAKE-001 run
    ``20260921T045955Z-4b6dd88b`` is what retired that reasoning: it selected
    three boxes - ``[0, 266, 2102, 1061]``, ``[0, 0, 2102, 665]`` and
    ``[16, 300, 2118, 1095]`` - out of a 2102x1061 frame, all three the full frame
    width, and measured the same unusable mapping from each. A region that wide
    occupies every column the matcher can search, so the only place it can be
    found is where it was told to look. See :func:`_spans_frame` for the full
    account.
    """
    from autocraft.wake.salience import _spans_frame

    assert _spans_frame((0, 0, 160, 120), 160) is True
    assert _spans_frame((-4, -4, 200, 200), 160) is True
    # Wider than the frame, and reaching past it on both sides.
    assert _spans_frame((-40, 40, 200, 80), 160) is True
    # The band that used to survive. It is the ``20260921T045955Z-4b6dd88b`` case
    # in miniature: full width, part height, so it has no horizontal position to
    # be tracked by.
    assert _spans_frame((0, 40, 160, 80), 160) is True
    # A box that leaves a column to spare is still a region, and still survives.
    assert _spans_frame((10, 10, 150, 110), 160) is False
    assert _spans_frame((0, 0, 159, 120), 160) is False
    # Height is deliberately not tested - see the docstring.
    assert _spans_frame((10, 0, 150, 120), 160) is False


def test_a_full_width_band_is_not_offered_as_a_candidate() -> None:
    """The live WAKE-001 run ``20260921T045955Z-4b6dd88b``, in one assertion.

    The run selected three boxes out of a 2102x1061 frame and every one of them
    was the full frame width: ``[0, 266, 2102, 1061]``, ``[0, 0, 2102, 665]`` and
    ``[16, 300, 2118, 1095]``. A bright horizontal band is the same shape in
    miniature. It is a real feature of the picture and it is still not a region,
    because a box that wide occupies every column the matcher can search.
    """
    from autocraft.wake.salience import SalienceDiagnostics

    image = with_blob(scene(), y0=20, y1=70, x0=0, x1=160)
    diagnostics = SalienceDiagnostics()
    found = find_candidates(image, grid=8, diagnostics=diagnostics)

    assert found == [], "a full-width band is not a target"
    assert diagnostics.groups == 1, "the band does form a group; that is the problem"
    assert diagnostics.rejected_frame_span == 1
    assert diagnostics.surviving_candidates == 0


def test_a_frame_sized_patch_cannot_be_located() -> None:
    """A template always matches itself perfectly, so a lone position is no location.

    With a frame-sized patch, every pass - the block-averaged one and the
    full-resolution one - is offered exactly one legal position, and reports it
    back at a confidence of 1.0. That is the identity, not a measurement of where
    the target went. ``locate`` used to return the patch's own centre here, which
    is what made the centring loop a fixed point: the target never moved, the
    distance never changed, and the same correction was re-issued until the
    budget ran out.
    """
    from autocraft.wake.salience import refine

    sky = gradient_sky()
    frame = frame_of(sky)
    patch = TargetPatch(image=sky, origin=(0, 0), centre=(80.0, 60.0))

    assert refine(frame, patch) is None
    assert refine(frame, patch, scale=4) is None
    assert locate(frame, patch) is None
    assert locate(frame, patch, window=16) is None


def test_a_bounded_region_is_still_located() -> None:
    """Refusing the frame-sized region must not refuse real regions.

    The rule is geometry, not a size cap: a region that covers the frame has one
    legal position, and a region that does not has many.
    """
    image = with_blob(scene(), y0=25, y1=95, x0=25, x1=135)
    frame = frame_of(image)
    candidate = find_candidates(frame, grid=8)[0]
    assert 0 < candidate.bbox[0] and candidate.bbox[2] < image.shape[1], "the region must be bounded"
    patch = TargetPatch.of(frame, candidate.bbox, centre=candidate.centre)

    shifted = frame_of(panned(image, 6, 4))
    found = locate(shifted, patch, predicted=candidate.centre)
    assert found is not None
    assert found.centre[0] == pytest.approx(candidate.centre[0] + 6, abs=2.0)
    assert found.centre[1] == pytest.approx(candidate.centre[1] + 4, abs=2.0)


def test_a_sky_only_scene_selects_no_target_at_all() -> None:
    """The operator-visible face of the defect: it must not chase the sky.

    The first live run printed "Target moved closer to centre." fourteen times
    while the measured distance stayed at 216.4 px, and drifted the camera down
    because every "right" correction was equally downward. None of that should
    happen on a scene with nothing in it: the run should scan, find nothing, and
    say so. That line has since been corrected - see the stream-narration tests -
    but the defect it described is what this test still guards.
    """
    policy = WakeDecisionPolicy(max_moves=45)
    sky = gradient_sky(height=240, width=320)
    _run_policy(policy, [sky])

    assert policy.state is WakeState.FAILED
    kinds = [event.kind for event in policy.drain_events()]
    assert kinds.count(WakeEventKind.TARGET_SELECTED) == 0
    assert kinds.count(WakeEventKind.CENTERING_PROGRESS) == 0
    assert policy.scan_moves <= policy.max_scan_moves
    assert "nothing visually salient" in (policy.stop_reason or "")


# ---------------------------------------------------------------------------
# SALIENCE DIAGNOSTICS
#
# Three different rules can empty the candidate list, and before this they all
# returned the same bare ``[]``. The second live run ended with "nothing
# visually salient was found in 12 scan movement(s)" - a true statement that
# explained nothing. These tests pin the evidence that was added to explain it.
# ---------------------------------------------------------------------------


def _diagnostics_for(image: np.ndarray, **kwargs: object) -> dict:
    from autocraft.wake.salience import SalienceDiagnostics

    accumulator = SalienceDiagnostics()
    find_candidates(image, grid=8, diagnostics=accumulator, **kwargs)  # type: ignore[arg-type]
    return accumulator.to_dict()


def test_a_flat_frame_says_the_frame_itself_was_flat() -> None:
    """A featureless frame is not "the threshold found nothing", it is "no contrast"."""
    report = _diagnostics_for(np.zeros((120, 160, 3), dtype=np.uint8))

    assert report["refusal"] == "flat_frame"
    assert report["max_cell_score"] == 0.0
    assert report["groups"] == 0, "no group should have formed"
    assert report["cells_above_threshold"] == 0


def test_a_sky_frame_says_the_group_was_refused_for_spanning_the_frame() -> None:
    """The second live run, in one assertion.

    Every cell of the sky cleared the threshold, so the map formed exactly one
    group - and that group was the whole frame, which is refused. The record has
    to say that, because it is the difference between "this frame offered nothing"
    and "this frame offered one thing and it was not a region".
    """
    report = _diagnostics_for(gradient_sky())

    assert report["refusal"] == "every_group_refused"
    assert report["groups"] == 1, "the sky does form a group; that is the problem"
    assert report["cells_above_threshold"] == 64, "every cell cleared the threshold"
    assert report["rejected_frame_span"] == 1
    assert report["surviving_candidates"] == 0
    assert report["max_cell_score"] > report["min_salience"], "the frame was not flat"


def test_a_frame_that_yields_a_candidate_reports_no_refusal() -> None:
    """A refusal is only recorded when something was refused."""
    image = with_blob(scene(), y0=30, y1=70, x0=40, x1=90)
    report = _diagnostics_for(image)

    assert report["refusal"] == ""
    assert report["surviving_candidates"] == 1
    assert report["groups"] == 1
    assert report["rejected_frame_span"] == 0


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"min_salience": 99.0}, "flat_frame"),
        ({"peak_fraction": 2.0}, "below_threshold"),
    ],
)
def test_the_diagnostics_name_which_rule_emptied_the_list(
    kwargs: dict, expected: str
) -> None:
    """Each early exit has its own name, so a zero is not a shrug.

    ``min_salience`` above the frame's peak means the frame was judged flat;
    ``peak_fraction`` above 1.0 leaves the threshold unreachable, which is a
    threshold decision rather than a flatness one. Those are different facts.
    """
    image = with_blob(scene(), y0=30, y1=70, x0=40, x1=90)
    report = _diagnostics_for(image, **kwargs)

    assert report["refusal"] == expected
    assert report["surviving_candidates"] == 0


def test_the_top_cell_scores_are_bounded_and_ordered() -> None:
    """The strongest cells are kept as evidence, and only a handful of them."""
    image = with_blob(scene(), y0=30, y1=70, x0=40, x1=90)
    report = _diagnostics_for(image)
    scores = report["top_cell_scores"]

    assert len(scores) == 8, "eight is the cap, and a full grid has more cells than that"
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(report["max_cell_score"], abs=1e-4)


def test_the_top_cell_scores_never_exceed_the_cap() -> None:
    from autocraft.wake.salience import _TOP_CELL_SCORES

    report = _diagnostics_for(gradient_sky(height=240, width=320))
    assert len(report["top_cell_scores"]) == _TOP_CELL_SCORES
    assert _TOP_CELL_SCORES < 8 * 8, "the cap has to be smaller than the grid to mean anything"


def test_the_diagnostics_carry_no_pixel_data() -> None:
    """This is telemetry, not a frame dump: every value is a number or a label."""
    report = _diagnostics_for(gradient_sky())

    assert json.loads(json.dumps(report)) == report, "it must survive a round trip"
    for key, value in report.items():
        assert isinstance(value, (int, float, str, list)), f"{key} is {type(value).__name__}"
        if isinstance(value, list):
            assert all(isinstance(item, float) for item in value), key
    assert len(report) < 24, "a diagnostic that grows without bound is a second problem"


def test_the_diagnostics_are_per_call_and_not_shared() -> None:
    """Two accumulators, two frames, no cross-talk.

    Module-level counters would have been the easy way to do this and would have
    made the numbers depend on whatever else happened to run first.
    """
    from autocraft.wake.salience import SalienceDiagnostics

    sky = SalienceDiagnostics()
    blob = SalienceDiagnostics()
    find_candidates(gradient_sky(), grid=8, diagnostics=sky)
    find_candidates(with_blob(scene(), y0=30, y1=70, x0=40, x1=90), grid=8, diagnostics=blob)

    assert sky.rejected_frame_span == 1 and blob.rejected_frame_span == 0
    assert blob.surviving_candidates == 1 and sky.surviving_candidates == 0


def test_a_relocation_refused_for_search_room_says_so() -> None:
    """The patch matcher's own refusals are counted too, in their own slots."""
    from autocraft.wake.salience import SalienceDiagnostics, refine

    sky = gradient_sky()
    patch = TargetPatch(image=sky, origin=(0, 0), centre=(80.0, 60.0))
    accumulator = SalienceDiagnostics()

    assert refine(frame_of(sky), patch, diagnostics=accumulator) is None
    assert accumulator.rejected_search_room == 1
    assert accumulator.rejected_score == 0
    assert accumulator.rejected_flat_patch == 0


def test_a_relocation_refused_for_its_score_says_so() -> None:
    from autocraft.wake.salience import SalienceDiagnostics, refine

    image = with_blob(scene(), y0=25, y1=95, x0=25, x1=135)
    frame = frame_of(image)
    candidate = find_candidates(frame, grid=8)[0]
    patch = TargetPatch.of(frame, candidate.bbox, centre=candidate.centre)
    accumulator = SalienceDiagnostics()

    assert refine(frame, patch, min_score=1.5, diagnostics=accumulator) is None
    assert accumulator.rejected_score == 1
    assert accumulator.rejected_search_room == 0


def test_a_relocation_refused_for_a_flat_patch_says_so() -> None:
    from autocraft.wake.salience import SalienceDiagnostics, refine

    flat = np.zeros((60, 60, 3), dtype=np.uint8)
    patch = TargetPatch.of(flat, (10, 10, 50, 50))
    accumulator = SalienceDiagnostics()

    assert refine(flat, patch, window=None, diagnostics=accumulator) is None
    assert accumulator.rejected_flat_patch == 1


def test_the_scan_summary_is_emitted_once_per_scanned_frame() -> None:
    """One event per frame the scan looked at, and no more."""
    policy = WakeDecisionPolicy(max_moves=45)
    _run_policy(policy, [gradient_sky(height=240, width=320)])

    summaries = [e for e in policy.drain_events() if e.kind is WakeEventKind.SALIENCE_SCAN_SUMMARY]
    assert len(summaries) == policy.scan_moves + 1, (
        "the starting view is scanned too, so there is one summary per frame in hand"
    )


def test_the_scan_summary_explains_the_second_live_run() -> None:
    """The whole point: a zero-candidate run has to say why it found nothing.

    The second live run scanned twelve movements, found nothing, and said so.
    These are the numbers that turn that sentence into an explanation.
    """
    policy = WakeDecisionPolicy(max_moves=45)
    _run_policy(policy, [gradient_sky(height=240, width=320)])

    summaries = [e for e in policy.drain_events() if e.kind is WakeEventKind.SALIENCE_SCAN_SUMMARY]
    assert summaries, "a scan that found nothing must still report"
    detail = summaries[0].detail

    assert set(detail) >= {
        "view_id",
        "max_cell_score",
        "groups",
        "rejected_frame_span",
        "rejected_search_room",
        "rejected_score",
        "surviving_candidates",
        "relocation_attempts",
        "frame_width",
        "frame_height",
    }
    assert detail["surviving_candidates"] == 0
    assert detail["groups"] == 1
    assert detail["rejected_frame_span"] == 1
    assert detail["refusal"] == "every_group_refused"
    assert detail["frame_width"] == 320 and detail["frame_height"] == 240
    assert detail["view_id"], "the frame has to be identifiable in the record"


def test_the_scan_summary_says_when_nothing_was_ever_relocated() -> None:
    """``relocation_attempts == 0`` is what makes the three zeros mean something.

    A run that never chose a target reports zero rejections for search room, for
    score and for a flat patch - not because nothing was refused, but because
    nothing was ever attempted. Without the attempt count the three zeros read as
    three findings.
    """
    policy = WakeDecisionPolicy(max_moves=45)
    _run_policy(policy, [gradient_sky(height=240, width=320)])

    detail = [
        e.detail for e in policy.drain_events() if e.kind is WakeEventKind.SALIENCE_SCAN_SUMMARY
    ][-1]
    assert detail["relocation_attempts"] == 0
    assert detail["rejected_search_room"] == 0
    assert detail["rejected_score"] == 0
    assert detail["rejected_flat_patch"] == 0


def test_the_scan_summary_holds_no_pixels() -> None:
    policy = WakeDecisionPolicy(max_moves=45)
    _run_policy(policy, [gradient_sky(height=240, width=320)])

    for event in policy.drain_events():
        if event.kind is WakeEventKind.SALIENCE_SCAN_SUMMARY:
            payload = event.to_dict()
            assert json.loads(json.dumps(payload)) == payload
            assert "image" not in json.dumps(payload)


def test_a_geometry_change_is_recorded_as_an_event() -> None:
    """Finding 1: the run has to say when the window is not the size it was.

    The second live run printed a 3222x1928 client area at startup and reported
    3591x1928 in its final summary, and the record had nowhere to put either
    number. Now every observation carries both sizes and the change is an event.

    This builds the change from a real observation's own geometry so that only
    the client width differs - which is the point: the event has to name the
    client area and the captured frame separately, not blur them into one
    "the window changed" flag.
    """
    from dataclasses import replace

    policy = WakeDecisionPolicy(max_moves=45)
    policy.reset()
    image = with_blob(scene(), y0=30, y1=70, x0=40, x1=90)
    status = _status_with_region(width=3591, height=1928, handle=11)
    first = Observation(index=0, timestamp=0.0, window=status, frame=frame_of(image))
    before = replace(first.geometry, client_width=3222)

    policy.decide(first)
    policy.decide(
        Observation(
            index=1,
            timestamp=1.0,
            window=status,
            frame=frame_of(image, 1),
            geometry_changed_from=before,
        )
    )

    events = [e for e in policy.drain_events() if e.kind is WakeEventKind.WINDOW_GEOMETRY_CHANGED]
    assert len(events) == 1, "only the observation that carries a change may emit one"
    detail = events[0].detail
    assert detail["before"] == before.to_dict()
    assert detail["after"]["client_width"] == 3591
    assert detail["after"]["frame_width"] == 160, "the captured frame is the fake 160-wide one"
    assert detail["changed"] == ["client_width"], (
        "the height did not move, and the event must not claim it did"
    )


def _status_with_region(*, width: int, height: int, handle: int) -> WindowStatus:
    from autocraft.vision.frame import ScreenRegion

    return WindowStatus(
        found=True,
        title="Fake",
        is_foreground=True,
        handle=handle,
        region=ScreenRegion(left=0, top=0, width=width, height=height),
    )


# ---------------------------------------------------------------------------
# THE RESIZE RESPONSE
# ---------------------------------------------------------------------------
#
# A window that changes size mid-run changes what a pixel means, and every
# quantity the policy carries is denominated in pixels: view fingerprints over a
# grid that no longer exists, a target chosen at the old size, offset history and
# a pixels-per-mouse-count calibration measured at the old size. Before this the
# run logged the change and carried all of it forward, which is worse than having
# none of it, because a stale number does not look stale.

# A bounded blob kept well away from the frame edges, so the candidate group
# never touches a border and is refused for spanning the frame.
_BLOB_KWARGS = {"y0": 20, "y1": 55, "x0": 25, "x1": 65}


def _drive_to_a_live_target(
    policy: WakeDecisionPolicy,
    *,
    width: int = 1280,
    height: int = 720,
    limit: int = 6,
) -> Observation:
    """Drive the policy until it has picked the blob out of the frame.

    The world pans a few pixels each step on purpose. The scan refuses to choose
    anything until it has seen more than one distinct view, which is what keeps it
    from locking onto the first frame it is handed - so a frozen frame never
    reaches a target at all.

    The blob stays well inside the frame, so the target is bounded and the
    selection succeeds. By then there is a chosen target, a start view, offset
    history and view memory, all denominated in the old pixel scale.
    """
    base = with_blob(scene(), **_BLOB_KWARGS)
    status = _status_with_region(width=width, height=height, handle=11)
    last: Observation | None = None
    for step in range(limit):
        last = Observation(
            index=step,
            timestamp=float(step),
            window=status,
            frame=frame_of(panned(base, 8 * step, 0), step),
        )
        policy.decide(last)
        assert policy.state not in {
            WakeState.COMPLETE,
            WakeState.FAILED,
            WakeState.SAFE_STOP,
        }, "the run ended before the resize could be tested"
        if policy.state in {WakeState.SELECTING, WakeState.CENTERING, WakeState.REACQUIRING}:
            return last
    raise AssertionError("the policy never picked the blob out of the frame")


def _resize(
    policy: WakeDecisionPolicy,
    last: Observation,
    *,
    width: int = 640,
    height: int = 360,
) -> dict:
    """Hand the policy a look captured after the window changed size.

    Returns:
        The detail of the single WINDOW_GEOMETRY_CHANGED event it produced.
    """
    index = last.index + 1
    image = with_blob(scene(), **_BLOB_KWARGS)
    policy.decide(
        Observation(
            index=index,
            timestamp=float(index),
            window=_status_with_region(width=width, height=height, handle=11),
            frame=frame_of(image, index),
            geometry_changed_from=last.geometry,
        )
    )
    events = [e for e in policy.drain_events() if e.kind is WakeEventKind.WINDOW_GEOMETRY_CHANGED]
    assert len(events) == 1, "a resize must be reported exactly once"
    return events[0].detail


def test_a_resize_drops_everything_measured_in_the_old_pixel_scale() -> None:
    policy = WakeDecisionPolicy(max_moves=45)
    policy.reset()
    last = _drive_to_a_live_target(policy)

    before_views = policy.memory.unique_views
    assert before_views >= 1, "the test is vacuous unless a view was stored"
    assert policy.start_fingerprint is not None
    assert policy.target is not None, "the bounded blob should have been selected"

    detail = _resize(policy, last)

    # The resize step carries on into the scan, which records the single view it
    # has just taken. Every view from before the change is gone, so only that one
    # survives - and the view it recorded is in the new pixel scale, not the old.
    assert policy.memory.unique_views == 1
    assert before_views > policy.memory.unique_views
    assert policy.start_fingerprint is None
    assert policy.target is None
    assert policy.offset_history == []
    assert policy.last_offset is None
    assert policy.last_seen_offset is None
    assert policy.last_seen_distance is None
    assert policy.last_move is None
    assert policy.last_relocation_confidence is None
    assert detail["rebaselined"] is True


def test_the_resize_event_names_what_it_dropped() -> None:
    """The event has to say what was invalidated, not merely that something was.

    An operator reading the record needs to know whether the run quietly kept
    acting on a target it chose at a size that no longer exists.
    """
    policy = WakeDecisionPolicy(max_moves=45)
    policy.reset()
    last = _drive_to_a_live_target(policy)

    dropped = _resize(policy, last)["dropped"]

    assert isinstance(dropped, list)
    assert all(isinstance(name, str) for name in dropped)
    assert "view memory" in dropped
    assert "start view" in dropped
    assert "chosen target" in dropped


def test_a_rebaseline_reports_only_what_it_actually_dropped() -> None:
    """A fixed list would be a lie. An empty run drops nothing and must say so."""
    policy = WakeDecisionPolicy(max_moves=45)
    policy.reset()

    assert policy._rebaseline_after_resize() == []

    last = _drive_to_a_live_target(policy)
    assert policy._rebaseline_after_resize(), "a run in progress has something to drop"


def test_a_resize_does_not_rewind_the_scan() -> None:
    """Replaying movements already made is the dead repetition WAKE-001 exists to avoid.

    The camera is still pointing where it was pointing; only the units changed, so
    the scan position and the run's tallies carry over untouched.
    """
    policy = WakeDecisionPolicy(max_moves=45)
    policy.reset()
    last = _drive_to_a_live_target(policy)
    before = (
        policy.scan_index,
        policy.scan_offset,
        policy.moves,
        policy.scan_moves,
        policy.total_centering_moves,
    )

    _resize(policy, last)

    after = (
        policy.scan_index,
        policy.scan_offset,
        policy.moves,
        policy.scan_moves,
        policy.total_centering_moves,
    )
    assert after[0] >= before[0], "the scan must never rewind to a view it has already paid for"
    assert after[2] >= before[2], "the movement tally is a record of what happened"
    assert after[3] >= before[3]
    assert after[4] >= before[4]


def test_a_resize_returns_a_centring_run_to_scanning() -> None:
    """A target chosen at the old size is not a target any more.

    Scanning again costs one step and invents nothing, which is the honest
    response. The alternative - centring on a stale rectangle - would send real
    input toward a place the target is no longer known to be.
    """
    policy = WakeDecisionPolicy(max_moves=45)
    policy.reset()
    last = _drive_to_a_live_target(policy)
    assert policy.state in {WakeState.SELECTING, WakeState.CENTERING, WakeState.REACQUIRING}

    _resize(policy, last)

    assert policy.state is not WakeState.CENTERING
    assert policy.target is None


def test_an_ordinary_reset_keeps_the_calibration_but_a_resize_does_not() -> None:
    """This is the one piece of state ``reset()`` keeps on purpose, and a resize kills it.

    The calibration is pixels per mouse count, so a resize scales every predicted
    correction by the resize factor. ``reset()`` keeps it because it is a
    measurement of how the game responds to this mouse rather than a belief about
    this run - but a measurement taken at the old size is exactly what a resize
    invalidates.
    """
    policy = WakeDecisionPolicy(max_moves=45)
    policy.calibration.adopt(pixels_per_delta_x=2.5, pixels_per_delta_y=2.5, quality=0.9)
    policy.reset()
    assert policy.calibration.pixels_per_count_x == 2.5, (
        "reset() keeps the calibration on purpose"
    )

    last = _drive_to_a_live_target(policy)
    detail = _resize(policy, last)

    assert policy.calibration.pixels_per_count_x is None
    assert policy.calibration.pixels_per_count_y is None
    assert policy.calibration.samples == 0
    assert policy.calibration.source == "unmeasured"
    assert "motion calibration" in detail["dropped"]


def test_an_unchanged_window_drops_nothing() -> None:
    """The rebaseline must be driven by a real change, not by every step."""
    policy = WakeDecisionPolicy(max_moves=45)
    policy.reset()
    last = _drive_to_a_live_target(policy)
    index = last.index + 1
    image = with_blob(scene(), **_BLOB_KWARGS)

    policy.decide(
        Observation(
            index=index,
            timestamp=float(index),
            window=_status_with_region(width=1280, height=720, handle=11),
            frame=frame_of(image, index),
        )
    )

    assert policy.memory.unique_views >= 1, "an ordinary step must not clear the view memory"
    assert policy.start_fingerprint is not None
    kinds = [e.kind for e in policy.drain_events()]
    assert kinds.count(WakeEventKind.WINDOW_GEOMETRY_CHANGED) == 0


def test_a_real_observer_hands_the_policy_a_geometry_change(
    fake_windows, fake_capture, clock
) -> None:
    """The detection seam and the response seam, joined.

    Every other test in this group hands the policy a geometry change it built
    itself. This one lets the real Observer produce it, because the rebaseline
    only ever runs if observe() actually sets geometry_changed_from - and until
    this test existed, the real Observer had never been driven through a change
    at all.
    """
    from autocraft.agent.observation import Observer
    from autocraft.vision.capture import ScreenCapturer
    from autocraft.vision.frame import ScreenRegion
    from autocraft.vision.window import WindowLocator

    fake_windows.windows.clear()
    fake_windows.add(0x100, "Luanti 5.17.0", region=ScreenRegion(100, 50, 1280, 720))
    observer = Observer(
        WindowLocator(fake_windows, ["Luanti"], clock=clock, rediscover_after=0.0),
        ScreenCapturer(fake_capture, clock=clock),
        clock=clock,
    )
    policy = WakeDecisionPolicy(max_moves=45)
    policy.reset()

    first = observer.observe(index=0)
    policy.decide(first)
    assert first.geometry_changed_from is None, "the first look has nothing to compare against"
    assert policy.start_fingerprint is not None

    # The window changes size under the run, and the Observer is what notices.
    fake_windows.windows.clear()
    fake_windows.add(0x100, "Luanti 5.17.0", region=ScreenRegion(100, 50, 640, 360))
    resized = observer.observe(index=1)

    assert resized.geometry_changed_from is not None, "the Observer must notice the change"
    assert "client_width" in resized.geometry_change

    policy.decide(resized)

    events = [e for e in policy.drain_events() if e.kind is WakeEventKind.WINDOW_GEOMETRY_CHANGED]
    assert len(events) == 1, "the Observer's change must reach the policy exactly once"
    detail = events[0].detail
    assert detail["rebaselined"] is True
    assert "view memory" in detail["dropped"]
    assert "start view" in detail["dropped"]
    assert detail["after"]["client_width"] == 640
    assert policy.start_fingerprint is None


def test_no_geometry_event_without_a_change() -> None:
    policy = WakeDecisionPolicy(max_moves=45)
    _run_policy(policy, [with_blob(scene(), y0=30, y1=70, x0=40, x1=90)])

    kinds = [e.kind for e in policy.drain_events()]
    assert kinds.count(WakeEventKind.WINDOW_GEOMETRY_CHANGED) == 0


def test_an_observation_reports_the_geometry_it_was_captured_with() -> None:
    """Every observation carries handle, client area and captured frame size.

    The client area and the captured frame are recorded separately on purpose:
    a disagreement between them is exactly what the second live run hid.
    """
    status = _status_with_region(width=1280, height=720, handle=99)
    observation = Observation(
        index=0, timestamp=0.0, window=status, frame=frame_of(scene())
    )

    payload = observation.to_dict()
    assert payload["window_handle"] == 99
    assert payload["client_width"] == 1280
    assert payload["client_height"] == 720
    assert payload["frame_width"] == 160, "the frame is the fake 160x120 one, not the client area"
    assert payload["frame_height"] == 120
    assert observation.geometry.empty is False
    assert observation.geometry_change == (), "nothing to compare against on the first look"


def test_the_geometry_names_the_fields_that_changed() -> None:
    from autocraft.agent.observation import WindowGeometry

    before = WindowGeometry(
        handle=11, client_width=3222, client_height=1928, frame_width=3222, frame_height=1928
    )
    after = WindowGeometry(
        handle=11, client_width=3591, client_height=1928, frame_width=3591, frame_height=1928
    )

    assert after.changes_from(before) == ("client_width", "frame_width")
    assert after.changes_from(after) == ()
    assert after.changes_from(WindowGeometry()) == (), (
        "an empty geometry is a missing reading, not a change"
    )
    assert WindowGeometry().empty is True
    assert after.empty is False


def test_salience_map_renders_as_rows_of_text() -> None:
    from autocraft.wake import SalienceMap

    grid = SalienceMap(4, tuple(tuple(0.0 for _ in range(4)) for _ in range(4)))
    lines = grid.ascii()
    assert isinstance(lines, list)
    assert len(lines) == 4
    assert all(isinstance(line, str) for line in lines)


def test_selecting_from_no_candidates_is_not_an_error() -> None:
    assert select_candidate([]) is None


def test_selection_prefers_the_higher_score() -> None:
    image = scene()
    image = with_blob(image, y0=10, y1=40, x0=10, x1=40, level=120)
    image = with_blob(image, y0=80, y1=115, x0=110, x1=155, level=245)
    candidates = find_candidates(image, grid=8)
    chosen = select_candidate(candidates, tie_epsilon=0.0)
    assert chosen is not None
    assert chosen.score == max(candidate.score for candidate in candidates)


def test_a_flat_patch_is_refused() -> None:
    """Matching a featureless template would report a location out of noise."""
    flat = np.zeros((60, 60, 3), dtype=np.uint8)
    patch = TargetPatch.of(flat, (10, 10, 50, 50))
    assert locate(flat, patch, window=None) is None


def test_a_moved_target_is_found_again() -> None:
    image = with_blob(scene(), y0=30, y1=70, x0=60, x1=110)
    candidate = find_candidates(image, grid=8)[0]
    patch = TargetPatch.of(image, candidate.bbox, centre=candidate.centre)

    shifted = panned(image, 20, 8)
    found = locate(shifted, patch, predicted=candidate.centre, window=None)
    assert found is not None
    assert found.centre[0] == pytest.approx(candidate.centre[0] + 20, abs=2.0)
    assert found.centre[1] == pytest.approx(candidate.centre[1] + 8, abs=2.0)


def test_relocating_an_absent_target_is_refused_by_default() -> None:
    """The default confidence floor must not be 0.0.

    With a zero floor, a target that has left the frame is still "found" at a
    confidence of 0.001 and the centring loop chases noise forever.
    """
    image = with_blob(scene(), y0=30, y1=70, x0=60, x1=110)
    candidate = find_candidates(image, grid=8)[0]
    elsewhere = with_blob(scene(seed=99), y0=90, y1=115, x0=10, x1=40)
    assert relocate(candidate, elsewhere, search_radius=24, grid=8) is None


# ---------------------------------------------------------------------------
# STREAM NARRATION
#
# The one-line summaries are display-only, and the rule that matters is that a
# line may not claim an outcome the event does not carry. ``CENTERING_PROGRESS``
# is emitted before the movement is sent, so it carries an attempt - and the line
# used to assert the result anyway.
# ---------------------------------------------------------------------------


def test_every_event_kind_has_a_line() -> None:
    missing = [kind for kind in WakeEventKind if kind not in STREAM_SUMMARY]
    assert missing == [], f"no summary line for {missing}"


def test_a_centring_line_does_not_claim_the_correction_worked() -> None:
    """The line was wrong on 9 of the first run's 20 corrections.

    ``_emit_move`` emits ``CENTERING_PROGRESS`` immediately before the movement
    goes out, so the distance it moved the target is not known when the line is
    written - and in the first live run the distance grew rather than shrank on 9
    of 20 attempts, while the line read "Target moved closer to centre." every
    time. A summary may describe the attempt; it may not describe the result.
    """
    line = stream_summary(WakeEventKind.CENTERING_PROGRESS)
    assert "closer" not in line
    assert "moved" not in line
    assert "attempt" in line.lower()


def test_a_summary_line_is_returned_for_a_kind_named_as_a_string() -> None:
    assert stream_summary("WAKE_STARTED") == stream_summary(WakeEventKind.WAKE_STARTED)


# ---------------------------------------------------------------------------
# CENTERING
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("distance", "expected"),
    [(1611.0, "coarse"), (400.0, "medium"), (200.0, "medium"), (100.0, "fine"), (5.0, "fine")],
)
def test_the_band_follows_the_distance(distance: float, expected: str) -> None:
    assert band_for_distance(distance, frame_width=3222, frame_height=1928) == expected


def test_a_measured_mapping_sizes_the_step_not_the_band() -> None:
    """The regression that motivated removing the band cap.

    With a real 0.5 px/count mapping and a 1611 px offset, the proportional step
    is far larger than the coarse band's 60 counts. Capping it at 60 would need
    29 moves to converge against a budget of 8; the step here is at the safety
    ceiling instead, which is a bound on the *hardware*, not on the estimate.
    """
    calibration = MotionCalibration()
    for _ in range(4):
        calibration.observe(dx_counts=100, dy_counts=0, shift_x=50.0, shift_y=0.0)
    assert calibration.measured is True

    controller = CenteringController(max_mouse_delta=200, calibration=calibration)
    move = controller.decide(offset_x=1611.0, offset_y=0.0, frame_width=3222, frame_height=1928)
    assert move.band == "coarse"
    assert move.dx == 200, "a measured mapping must not be capped at the band's count"
    assert move.dx <= 200, "and it must still respect the safety ceiling"


def test_an_unmeasured_mapping_falls_back_to_the_band_count() -> None:
    controller = CenteringController(max_mouse_delta=200)
    move = controller.decide(offset_x=1611.0, offset_y=0.0, frame_width=3222, frame_height=1928)
    assert move.band == "coarse"
    assert move.dx == 60
    assert move.expected_pixels is None


def test_a_centred_target_produces_no_movement() -> None:
    controller = CenteringController(dead_zone_px=12.0)
    move = controller.decide(offset_x=4.0, offset_y=-3.0, frame_width=3222, frame_height=1928)
    assert move.is_noop is True
    assert move.band == "fine"


def test_the_step_never_exceeds_the_configured_ceiling() -> None:
    calibration = MotionCalibration()
    for _ in range(4):
        calibration.observe(dx_counts=10, dy_counts=10, shift_x=0.5, shift_y=0.5)
    controller = CenteringController(max_mouse_delta=25, calibration=calibration)
    for offset in (5000.0, 800.0, 90.0):
        move = controller.decide(
            offset_x=offset, offset_y=offset, frame_width=3222, frame_height=1928
        )
        assert abs(move.dx) <= 25
        assert abs(move.dy) <= 25


def test_a_sign_flip_is_recorded_as_an_overshoot() -> None:
    """Crossing the centre is normal, and has to be visible rather than silent."""
    controller = CenteringController(dead_zone_px=2.0)
    first = controller.decide(offset_x=40.0, offset_y=0.0, frame_width=3222, frame_height=1928)
    assert first.overshoot is False
    assert first.dx > 0, "the target is right of centre, so the view turns right"

    controller.note_observation(shift_x=-80.0, shift_y=0.0)
    second = controller.decide(offset_x=-40.0, offset_y=0.0, frame_width=3222, frame_height=1928)
    assert second.overshoot is True
    assert second.reversed_axes == ("x",)
    assert second.dx < 0, "an overshoot has to come back the other way"
    assert controller.overshoots >= 1


def test_calibration_needs_several_samples_before_it_is_believed() -> None:
    """One measurement is not a mapping."""
    calibration = MotionCalibration()
    calibration.observe(dx_counts=100, dy_counts=0, shift_x=50.0, shift_y=0.0)
    assert calibration.measured is False
    assert calibration.pixels_per_count_x is None

    for _ in range(2):
        calibration.observe(dx_counts=100, dy_counts=0, shift_x=50.0, shift_y=0.0)
    assert calibration.measured is True
    assert calibration.pixels_per_count_x == pytest.approx(0.5)


def test_calibration_refuses_a_measurement_it_cannot_use() -> None:
    calibration = MotionCalibration()
    for _ in range(5):
        calibration.observe(dx_counts=0, dy_counts=0, shift_x=99.0, shift_y=99.0)
    assert calibration.measured is False, "a zero mouse count says nothing about the mapping"

    for _ in range(5):
        calibration.observe(dx_counts=10, dy_counts=0, shift_x=float("nan"), shift_y=0.0)
    assert calibration.measured is False


def test_calibration_can_adopt_a_previous_runs_mapping() -> None:
    calibration = MotionCalibration()
    calibration.adopt(pixels_per_delta_x=2.5, pixels_per_delta_y=2.5, quality=0.8)
    assert calibration.measured is True
    assert calibration.pixels_per_count_x == pytest.approx(2.5)
    assert calibration.source == "look-001"


def test_centring_converges_on_a_panning_scene() -> None:
    """Closed-loop: every correction is sized from the frame it just measured."""
    calibration = MotionCalibration()
    for _ in range(4):
        calibration.observe(dx_counts=100, dy_counts=0, shift_x=800.0, shift_y=0.0)
    assert calibration.pixels_per_count_x == pytest.approx(8.0)

    controller = CenteringController(dead_zone_px=12.0, calibration=calibration)
    offset = 1611.0
    distances = [offset]
    for _ in range(8):
        move = controller.decide(
            offset_x=offset, offset_y=0.0, frame_width=3222, frame_height=1928
        )
        if move.is_noop:
            break
        moved = move.dx * 8.0  # the ground truth the controller is discovering
        offset -= moved
        controller.note_observation(shift_x=-moved, shift_y=0.0)
        distances.append(abs(offset))

    assert distances[-1] < distances[0], f"did not converge: {distances}"
    assert distances[-1] < 30.0, f"converged poorly: {distances}"
    assert len(distances) <= 6, f"took too many moves: {distances}"


def test_a_ceiling_limited_mapping_is_bounded_by_the_hardware_not_the_estimate() -> None:
    """The honest failure mode left in the centring loop.

    At 0.5 px/count the 200-count safety ceiling only moves the view 100 px, so
    closing 1611 px needs about 17 corrections against a budget of 8. The loop
    stops short and says so rather than being "fixed" by pretending a count moves
    further than it does.
    """
    calibration = MotionCalibration()
    for _ in range(4):
        calibration.observe(dx_counts=100, dy_counts=0, shift_x=50.0, shift_y=0.0)

    controller = CenteringController(dead_zone_px=12.0, calibration=calibration)
    offset = 1611.0
    for _ in range(8):
        move = controller.decide(
            offset_x=offset, offset_y=0.0, frame_width=3222, frame_height=1928
        )
        assert move.dx == 200, "the ceiling is the bound here, and it is honest"
        offset -= move.dx * 0.5
    assert offset > 800.0, "eight moves genuinely cannot close this distance"


def test_a_mapping_too_small_to_move_the_target_is_refused_rather_than_capped() -> None:
    """The live WAKE-001 run ``20260921T045955Z-4b6dd88b``, in one assertion.

    That run measured 0.0075 px per mouse count on y - a real measurement, taken
    from consistent observations, of a target that could not be relocated.
    ``_step_counts`` floors the ratio at 0.05, so every request landed on the
    200-count safety ceiling and displaced the view by 1.5 px. The controller then
    spent eight centring moves on its first target taking the distance from
    171.9 px to 168.0 px - four pixels of a hundred-and-seventy-two pixel gap,
    against a 12 px dead zone - and the distance series reads 171.9, 172.9, 171.9,
    170.0, 169.0, 170.0, 168.0, 169.0: eight moves, no convergence, and nothing in
    the record saying why.

    The controller's own ``expected_pixels`` knew. It said 1.5 px while the step
    was the largest the hardware would take, and that number was reported, never
    consulted. This is the assertion that it is consulted.
    """
    calibration = MotionCalibration()
    for _ in range(4):
        calibration.observe(dx_counts=0, dy_counts=200, shift_x=0.0, shift_y=1.5)
    assert calibration.measured is True
    assert calibration.pixels_per_count_y == pytest.approx(0.0075)

    controller = CenteringController(
        dead_zone_px=12.0, max_mouse_delta=200, calibration=calibration
    )
    move = controller.decide(offset_x=0.0, offset_y=169.0, frame_width=2102, frame_height=1061)

    assert move.expected_pixels == pytest.approx(1.5, abs=0.01)
    assert move.unachievable is True
    assert move.dx == 0 and move.dy == 0, "the 200-count cap must not be sent"
    assert move.is_noop is True, "the caller treats this as a declined correction"
    assert "cannot be made" in move.reason
    assert controller.to_dict()["awaiting_observation"] is False


def test_a_target_one_nudge_from_success_is_not_called_unachievable() -> None:
    """The guard on that refusal, so it cannot abandon a target it can still fix.

    A step smaller than the dead zone is normally a step the loop cannot verify,
    because the next look cannot tell it from no movement. It is not that when the
    single step is enough to land inside the dead zone: a target 13 px out with a
    mapping worth 9 px per count is about to be centred, and calling it
    unachievable would be the false positive this check exists to avoid.
    """
    calibration = MotionCalibration()
    for _ in range(4):
        calibration.observe(dx_counts=100, dy_counts=0, shift_x=900.0, shift_y=0.0)
    assert calibration.pixels_per_count_x == pytest.approx(9.0)

    controller = CenteringController(
        dead_zone_px=12.0, max_mouse_delta=200, calibration=calibration
    )
    move = controller.decide(offset_x=13.0, offset_y=0.0, frame_width=3222, frame_height=1928)

    assert move.unachievable is False
    assert move.expected_pixels == pytest.approx(9.0)
    assert move.dx > 0, "one step from the dead zone is still a step worth taking"


# ---------------------------------------------------------------------------
# ANTI-REPETITION
# ---------------------------------------------------------------------------


def _signature(**overrides: object) -> str:
    fields = {"view_key": "view-a", "kind": "move", "dx": 10, "dy": 0, "strategy": "centre_left"}
    fields.update(overrides)
    return RepetitionGuard.signature(**fields)  # type: ignore[arg-type]


def test_the_same_failing_move_is_flagged_after_the_configured_repeats() -> None:
    guard = RepetitionGuard(repeats=3)
    verdicts = [guard.observe(_signature(), progress=0.0) for _ in range(4)]
    assert [verdict.stuck for verdict in verdicts] == [False, False, True, True]
    assert verdicts[-1].repeats >= 3
    assert guard.to_dict()["stuck_patterns_detected"] >= 1


def test_a_productive_streak_is_not_stuck() -> None:
    """The same move four times is not a loop while it is measurably working.

    ``progress`` is *signed improvement*, not a distance: positive means the
    correction helped. 164 -> 97 -> 43 -> 11 px is four positive deltas, so the
    guard must keep its hands off it.
    """
    guard = RepetitionGuard(repeats=3)
    distances = (164.0, 97.0, 43.0, 11.0)
    for before, after in zip(distances, distances[1:]):
        verdict = guard.observe(_signature(), progress=before - after)
        assert verdict.stuck is False, "progress must excuse a repeated movement"
    assert guard.to_dict()["stuck_patterns_detected"] == 0


def test_a_move_that_stops_helping_becomes_stuck() -> None:
    guard = RepetitionGuard(repeats=3)
    for _ in range(3):
        verdict = guard.observe(_signature(), progress=0.0)
    assert verdict.stuck is True
    assert verdict.repeats == 3
    assert guard.to_dict()["stuck_patterns_detected"] == 1


def test_an_unmeasured_attempt_counts_as_no_improvement() -> None:
    """Silence is not evidence of progress."""
    guard = RepetitionGuard(repeats=3)
    for _ in range(3):
        verdict = guard.observe(_signature())
    assert verdict.stuck is True


def test_changing_strategy_breaks_the_streak() -> None:
    guard = RepetitionGuard(repeats=3)
    for _ in range(3):
        guard.observe(_signature(), progress=0.0)
    guard.note_strategy_change()
    verdict = guard.observe(_signature(strategy="centre_up"), progress=0.0)
    assert verdict.stuck is False
    assert guard.to_dict()["stuck_patterns_broken"] >= 1


def test_a_failing_strategy_goes_on_cooldown_and_can_come_back() -> None:
    cooldown = StrategyCooldown(failures=2)
    assert cooldown.is_available("centre_left") is True

    cooldown.record_failure("centre_left", reason="no closer")
    assert cooldown.is_available("centre_left") is True
    cooldown.record_failure("centre_left", reason="no closer")
    assert cooldown.is_available("centre_left") is False
    assert cooldown.active()

    cooldown.release("centre_left")
    assert cooldown.is_available("centre_left") is True


def test_progress_is_measured_rather_than_guessed() -> None:
    model = ProgressModel(epsilon=1.0)
    closer = model.assess(before=164.0, after=97.0)
    assert closer.measured is True and closer.improved is True and closer.productive is True

    stuck = model.assess(before=97.0, after=97.4)
    assert stuck.measured is True and stuck.improved is False and stuck.productive is False

    blind = model.assess(before=None, after=None)
    assert blind.measured is False, "no measurement is not the same as no progress"


def test_dead_and_productive_repetition_ratios_are_reported() -> None:
    model = ProgressModel(epsilon=1.0)
    model.assess(before=200.0, after=100.0)
    model.assess(before=100.0, after=100.0)
    model.assess(before=100.0, after=100.0)
    summary = model.to_dict()
    assert 0.0 < summary["dead_repetition_ratio"] < 1.0
    assert 0.0 < summary["productive_repetition_ratio"] < 1.0


def test_a_failed_candidate_is_not_offered_again() -> None:
    image = with_blob(scene(), y0=30, y1=70, x0=40, x1=90)
    candidate = find_candidates(image, grid=8)[0]
    experience = ShortTermExperience()
    experience.note_candidate(candidate)
    assert experience.strongest_candidates()

    experience.mark_candidate_failed(candidate, "lost it", index=1)
    remaining = experience.strongest_candidates()
    assert candidate not in remaining


# ---------------------------------------------------------------------------
# BOUNDS
# ---------------------------------------------------------------------------


def _observation(image: np.ndarray | None, index: int = 0, *, foreground: bool = True) -> Observation:
    status = WindowStatus(found=True, title="Fake", is_foreground=foreground)
    frame = None if image is None else frame_of(image, index)
    return Observation(index=index, timestamp=float(index), window=status, frame=frame)


def _run_policy(
    policy: WakeDecisionPolicy,
    frames: list[np.ndarray | None],
    *,
    limit: int = 400,
) -> list[str]:
    """Drive the policy directly, recording the state after each decision."""
    states: list[str] = []
    policy.reset()
    for step in range(limit):
        image = frames[min(step, len(frames) - 1)]
        action = policy.decide(_observation(image, step))
        states.append(policy.state.value)
        if policy.state in {WakeState.COMPLETE, WakeState.FAILED, WakeState.SAFE_STOP}:
            break
        assert action.kind in {ActionKind.NOOP, ActionKind.MOUSE_MOVE, ActionKind.STOP}, (
            "WAKE-001 may only look and move the mouse"
        )
    return states


def test_a_blank_scene_gives_up_within_the_budget() -> None:
    """The most important bound: a scene with nothing in it still terminates."""
    policy = WakeDecisionPolicy(max_moves=45)
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    states = _run_policy(policy, [blank])
    assert states[-1] in {WakeState.FAILED.value, WakeState.COMPLETE.value}
    assert policy.moves <= 45
    assert len(states) <= policy.step_budget


def test_the_scan_budget_is_respected() -> None:
    policy = WakeDecisionPolicy(max_scan_moves=4, scan_counts=60)
    _run_policy(policy, [scene(seed=2)])
    assert policy.scan_moves <= 4


def test_a_run_with_no_frames_does_not_hang() -> None:
    policy = WakeDecisionPolicy()
    states = _run_policy(policy, [None])
    assert states[-1] in {
        WakeState.FAILED.value,
        WakeState.SAFE_STOP.value,
        WakeState.COMPLETE.value,
    }


def test_losing_the_window_stops_safely() -> None:
    policy = WakeDecisionPolicy()
    policy.reset()
    image = with_blob(scene(), y0=30, y1=70, x0=40, x1=90)
    policy.decide(_observation(image, 0))
    policy.decide(_observation(image, 1))
    for step in range(8):
        policy.decide(_observation(image, step + 2, foreground=False))
        if policy.state is WakeState.SAFE_STOP:
            break
    assert policy.state is WakeState.SAFE_STOP


def test_the_step_budget_leaves_room_for_the_capture_failures() -> None:
    policy = WakeDecisionPolicy(max_moves=10, max_capture_failures=3)
    assert policy.step_budget == 10 + 3 + 1


def test_the_policy_satisfies_the_decision_policy_protocol() -> None:
    """The behaviour layer must plug into the existing seam, not a new one."""
    policy = WakeDecisionPolicy()
    assert isinstance(policy, DecisionPolicy)
    assert policy.name == "wake-001"
    assert callable(policy.reset)


def test_the_policy_report_is_json_serialisable() -> None:
    policy = WakeDecisionPolicy()
    policy.reset()
    payload = policy.report()
    json.dumps(payload)
    assert payload["state"] == WakeState.STARTING.value
    assert "mapping_source" in payload
    assert "dead_repetition_ratio" in payload
    assert "productive_repetition_ratio" in payload


def test_the_policy_report_agrees_with_the_panel() -> None:
    """The record, the report and the panel must not drift apart.

    ``measured_at`` is the one field the policy cannot supply: the observer stamps
    it when the panel is published, because it is the time the page was told, not
    the time the behaviour decided.
    """
    import dataclasses

    report = WakeDecisionPolicy().report()
    panel = {field.name for field in dataclasses.fields(WakeReport)}
    shared = panel - {
        "available",
        "status",
        "experiment",
        "run_id",
        "window_width",
        "window_height",
        "measured_at",
    }
    assert shared <= set(report), f"the panel wants {sorted(shared - set(report))}"


# ---------------------------------------------------------------------------
# RECORD
# ---------------------------------------------------------------------------


def test_a_running_record_is_not_a_verdict(tmp_path: Path) -> None:
    recorder = WakeRecorder(tmp_path / "run", run_id="r1", clock=lambda: 1.0)
    result = recorder.result()
    assert result.status == STATUS_RUNNING
    assert result.successful_completion is None, "an unfinished run has not succeeded or failed"
    assert result.experiment == EXPERIMENT_NAME


def test_the_record_is_rewritten_as_the_run_progresses(tmp_path: Path) -> None:
    """A run that is killed still leaves behind what it measured."""
    recorder = WakeRecorder(tmp_path / "run", run_id="r1", clock=lambda: 1.0)
    recorder.update(metrics={"moves_sent": 4, "scan_moves": 3}, steps=4)
    payload = json.loads(recorder.result_path.read_text(encoding="utf-8"))
    assert payload["moves_sent"] == 4
    assert payload["status"] == STATUS_RUNNING


def test_finishing_records_a_truthful_verdict(tmp_path: Path) -> None:
    recorder = WakeRecorder(tmp_path / "run", run_id="r1", clock=lambda: 1.0)
    centred = recorder.finish(status=STATUS_COMPLETED, stop_reason="centred", state="COMPLETE")
    assert centred.successful_completion is True

    recorder = WakeRecorder(tmp_path / "run2", run_id="r2", clock=lambda: 1.0)
    gave_up = recorder.finish(status=STATUS_FAILED, stop_reason="out of moves", state="FAILED")
    assert gave_up.successful_completion is False


def test_the_limitation_notes_travel_with_every_record(tmp_path: Path) -> None:
    recorder = WakeRecorder(tmp_path / "run", run_id="r1", clock=lambda: 1.0)
    result = recorder.finish(status=STATUS_COMPLETED, stop_reason="centred", state="COMPLETE")
    for note in LIMITATION_NOTES:
        assert note in result.notes
    assert any("salient region and nothing more" in note for note in result.notes)
    assert any("measured, not assumed" in note for note in result.notes)


def test_the_result_stream_is_readable() -> None:
    result = WakeResult(
        status=STATUS_COMPLETED,
        state="COMPLETE",
        events=(
            WakeEvent.of(WakeEventKind.WAKE_STARTED, index=0),
            WakeEvent.of(WakeEventKind.TARGET_CENTERED, index=9),
            WakeEvent.of(WakeEventKind.WAKE_COMPLETE, index=9),
        ),
    )
    lines = result.stream_lines()
    assert len(lines) == 3
    assert lines[0]
    assert len(result.stream_lines(limit=2)) == 2


# ---------------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------------


def test_the_status_mapping_never_blames_the_agent_for_a_bound() -> None:
    """A step limit is not the behaviour failing, and must not be reported as one."""
    assert status_for_state(WakeState.COMPLETE) == STATUS_COMPLETED
    assert status_for_state(WakeState.FAILED) == STATUS_FAILED
    for state in (
        WakeState.STARTING,
        WakeState.SCANNING,
        WakeState.SELECTING,
        WakeState.CENTERING,
        WakeState.REACQUIRING,
        WakeState.SAFE_STOP,
    ):
        assert status_for_state(state) == STATUS_ABORTED, state


# -- a fake world, so the runner can be exercised without a screen ----------


class FakeWorld:
    """One fixed scene seen through a camera the mouse pans.

    The whole scene moves with the camera. A fake that slid the interesting region
    over a fixed background would let a matcher lock onto the background and
    produce a constant, unchanging offset that looks exactly like a centring bug.
    """

    def __init__(self, width: int = 320, height: int = 240, ratio: float = 0.5) -> None:
        self.width, self.height = width, height
        self.ratio = ratio
        self.scene_width, self.scene_height = 1600, 1200
        self.blob_x, self.blob_y = 400, 600
        rng = np.random.default_rng(7)
        self.scene = rng.integers(0, 14, size=(self.scene_height, self.scene_width, 3), dtype=np.uint8)
        for y in range(self.blob_y - 34, self.blob_y + 35):
            for x in range(self.blob_x - 34, self.blob_x + 35):
                band = 210 if ((x + y) // 6) % 2 == 0 else 80
                self.scene[y, x] = (band, band, band)
        self.camera_x = float(self.blob_x + 110)
        self.camera_y = float(self.blob_y)

    def frame(self, index: int) -> Frame:
        x0 = int(round(self.camera_x - self.width / 2))
        y0 = int(round(self.camera_y - self.height / 2))
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(self.scene_width, x0 + self.width), min(self.scene_height, y0 + self.height)
        if sx1 > sx0 and sy1 > sy0:
            canvas[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = self.scene[sy0:sy1, sx0:sx1]
        return Frame(canvas, float(index))

    def move(self, dx: int, dy: int) -> None:
        self.camera_x += dx * self.ratio
        self.camera_y += dy * self.ratio


class FakeObserver:
    def __init__(self, world: FakeWorld, *, capture: bool = True) -> None:
        self.world = world
        self.capture = capture
        self.calls = 0

    def observe(self, index: int = 0, *, capture: bool = True, force_discovery: bool = False):
        self.calls += 1
        status = WindowStatus(found=True, title="Fake", is_foreground=True)
        frame = self.world.frame(index) if (capture and self.capture) else None
        return Observation(index=index, timestamp=0.0, window=status, frame=frame)


class FakeGuard:
    """The guard surface the loop uses. It never refuses anything here."""

    def __init__(self) -> None:
        self.events: list = []
        self.backend = None
        self.blocked_streak = 0
        self.last_block_reason = ""
        self.stop_reason = ""
        self.should_stop_for_blocks = False

    def install_atexit(self) -> None: ...
    def shutdown(self, reason: str) -> None: ...
    def release_all(self, reason: str) -> None: ...
    def check_emergency_stop(self) -> bool:
        return False

    def enforce_hold_limits(self) -> list:
        return []

    def authorize(self, description: str):
        return SimpleNamespace(allowed=True, reason="")

    def wait_for_rate_limit(self) -> None: ...
    def note_action(self) -> None: ...
    def record_block(self, reason: str) -> None: ...
    def record_success(self) -> None: ...


class FakeExecutor:
    def __init__(self, world: FakeWorld) -> None:
        self.world = world
        self.moves: list[tuple[int, int]] = []

    def execute(self, action: Action):
        executed = False
        if action.kind is ActionKind.MOUSE_MOVE:
            dx = action.parameters["dx"]
            dy = action.parameters["dy"]
            self.moves.append((dx, dy))
            self.world.move(dx, dy)
            executed = True
        payload = {
            "action_id": action.action_id,
            "kind": action.kind.value,
            "attempted": action.kind is not ActionKind.NOOP,
            "executed": executed,
            "blocked_reason": None,
            "error": None,
            "ok": True,
            "description": action.describe(),
        }
        return SimpleNamespace(
            action_id=action.action_id,
            kind=action.kind,
            attempted=payload["attempted"],
            executed=executed,
            blocked_reason=None,
            error=None,
            was_blocked=False,
            started_at=0.0,
            finished_at=0.0,
            description=action.describe(),
            ok=True,
            to_dict=lambda: dict(payload),
        )


def _build_runner(
    tmp_path: Path,
    *,
    world: FakeWorld | None = None,
    config: Config | None = None,
    policy: WakeDecisionPolicy | None = None,
    max_seconds: float | None = None,
    clock=None,
    sleeper=None,
    name: str = "run",
):
    world = world if world is not None else FakeWorld()
    config = config if config is not None else Config(data_dir=tmp_path / "data")
    policy = policy if policy is not None else WakeDecisionPolicy()
    clock = clock if clock is not None else (lambda: 0.0)
    recorder = WakeRecorder(
        tmp_path / name,
        run_id=name,
        plan={"title": "Fake", "handle": 0x1234},
        settings={"dead_zone_px": 12.0},
        clock=clock,
    )
    events: list = []
    statuses: list = []
    runner = WakeRunner(
        config=config,
        observer=FakeObserver(world),
        guard=FakeGuard(),
        policy=policy,
        recorder=recorder,
        executor=FakeExecutor(world),
        max_seconds=max_seconds,
        clock=clock,
        sleeper=sleeper if sleeper is not None else (lambda _: None),
        on_event=events.append,
        on_status=statuses.append,
        run_recorder=RunRecorder.new_run(config.runs_dir, clock=clock),
    )
    return runner, recorder, policy, events, statuses


def test_the_runner_turns_a_panning_scene_into_a_centred_target(tmp_path: Path) -> None:
    runner, recorder, policy, events, statuses = _build_runner(tmp_path)
    result = runner.run()

    assert result.status == STATUS_COMPLETED
    assert result.state == WakeState.COMPLETE.value
    assert result.successful_completion is True
    assert result.moves_sent <= result.max_moves
    assert result.centering_moves >= 1
    assert result.final_target_distance is not None
    assert result.final_target_distance < 60.0, result.final_target_distance
    assert result.progress and result.progress[-1] < result.progress[0]
    assert result.mapping_source == "self-measured"
    assert result.pixels_per_delta_x == pytest.approx(0.5, abs=0.05)

    # the record on disk agrees with the returned result
    payload = json.loads(recorder.result_path.read_text(encoding="utf-8"))
    assert payload["status"] == result.status
    assert payload["mapping"]["source"] == result.mapping_source
    assert payload["target"]["salience"] is not None
    assert payload["target"]["centre"] is not None
    assert payload["plan"]["title"] == "Fake"
    assert payload["window"] == {"width": 320, "height": 240}
    assert payload["dead_repetition_ratio"] is not None

    # the event stream and the status callback both saw the run
    assert events and statuses
    kinds = {event.kind for event in events}
    assert WakeEventKind.WAKE_STARTED in kinds
    assert WakeEventKind.TARGET_CENTERED in kinds
    assert WakeEventKind.WAKE_COMPLETE in kinds
    assert all(status["state"] for status in statuses)


def test_the_loop_stopping_a_run_is_reported_as_aborted_with_a_reason(tmp_path: Path) -> None:
    """A run cut short from outside was not the behaviour's own failure."""
    ticks = {"n": 0}

    def fast_clock() -> float:
        ticks["n"] += 1
        return float(ticks["n"])

    runner, recorder, policy, _, _ = _build_runner(
        tmp_path, max_seconds=3.0, clock=fast_clock, name="cut-short"
    )
    result = runner.run()

    assert result.status == STATUS_ABORTED, result.status
    assert result.state not in {WakeState.COMPLETE.value, WakeState.FAILED.value}
    assert any("never decided to stop" in note for note in result.notes), result.notes
    assert "time limit" in result.stop_reason

    payload = json.loads(recorder.result_path.read_text(encoding="utf-8"))
    assert payload["status"] == STATUS_ABORTED
    assert any("never decided to stop" in note for note in payload["notes"])


def test_the_runner_always_terminates_on_a_scene_with_nothing_in_it(tmp_path: Path) -> None:
    blank = np.zeros((240, 320, 3), dtype=np.uint8)

    class BlankObserver(FakeObserver):
        def observe(self, index: int = 0, *, capture: bool = True, force_discovery: bool = False):
            self.calls += 1
            return Observation(
                index=index,
                timestamp=0.0,
                window=WindowStatus(found=True, title="Fake", is_foreground=True),
                frame=Frame(blank.copy(), float(index)),
            )

    world = FakeWorld()
    config = Config(data_dir=tmp_path / "data")
    policy = WakeDecisionPolicy()
    recorder = WakeRecorder(tmp_path / "blank", run_id="blank", clock=lambda: 0.0)
    runner = WakeRunner(
        config=config,
        observer=BlankObserver(world),
        guard=FakeGuard(),
        policy=policy,
        recorder=recorder,
        executor=FakeExecutor(world),
        clock=lambda: 0.0,
        sleeper=lambda _: None,
    )
    result = runner.run()

    assert result.status in {STATUS_FAILED, STATUS_ABORTED}
    assert result.steps <= runner.max_steps
    assert result.moves_sent <= result.max_moves


# ---------------------------------------------------------------------------
# cadence: where the time in a step actually goes
# ---------------------------------------------------------------------------


def _ticking_clock(step_seconds: float = 0.005):
    """A clock that advances a fixed amount per reading.

    Cadence is the only thing in AutoCraft that is measured rather than decided,
    so it is the one place a test needs a clock it controls. The absolute values
    below are artefacts of how many times the loop reads the clock; what the
    tests assert is the *shape* - which phases are recorded, that they add up
    sensibly, and that nothing is smoothed or invented.
    """
    ticks = {"n": 0}

    def clock() -> float:
        ticks["n"] += 1
        return ticks["n"] * step_seconds

    return clock


def test_the_runner_reports_how_long_each_phase_of_a_step_took(tmp_path: Path) -> None:
    runner, recorder, _, _, _ = _build_runner(tmp_path, clock=_ticking_clock(), name="cadence")
    result = runner.run()

    cadence = result.cadence
    # Every phase the loop can observe, named for what it is.
    assert set(cadence) >= {
        "capture_mean",
        "decide_mean",
        "act_mean",
        "total_mean",
        "steps_timed",
        "moves_timed",
    }, sorted(cadence)
    assert "verify_mean" in cadence, "a movement that changed the frame was not timed"
    assert "since_previous_move_mean" in cadence

    # The counts say how much evidence each mean rests on, so a mean over three
    # steps can never be read as a mean over thirty.
    assert cadence["steps_timed"] == float(result.steps)
    assert cadence["moves_timed"] == float(result.moves_sent)

    # Means, not totals: a per-step figure cannot exceed the whole run.
    for key, value in cadence.items():
        assert value >= 0.0, (key, value)
    assert cadence["total_mean"] < result.duration * 1000.0
    assert cadence["capture_mean"] <= cadence["total_mean"]
    assert cadence["act_mean"] <= cadence["total_mean"]


def test_the_cadence_is_written_to_the_result_file(tmp_path: Path) -> None:
    runner, recorder, _, _, _ = _build_runner(tmp_path, clock=_ticking_clock(), name="cadence-file")
    result = runner.run()

    payload = json.loads(recorder.result_path.read_text(encoding="utf-8"))
    assert payload["cadence"] == result.cadence
    assert payload["cadence"]["total_mean"] > 0.0


def test_a_run_that_timed_nothing_reports_no_cadence_rather_than_zeros() -> None:
    """An empty cadence says "not measured"; zeros would say "measured as zero"."""
    result = WakeResult(run_id="untimed", experiment="WAKE-001")

    assert result.cadence == {}
    assert result.to_dict()["cadence"] == {}


def test_unreadable_cadence_entries_are_dropped_rather_than_failing_the_record() -> None:
    result = WakeResult(
        run_id="partial",
        experiment="WAKE-001",
        cadence={"capture_mean": 12.5, "decide_mean": None, "act_mean": "nonsense"},
    )

    assert result.cadence == {"capture_mean": 12.5}


def test_the_gap_between_movements_is_measured_and_not_smoothed(tmp_path: Path) -> None:
    """The felt cadence is the pause between inputs, so it is recorded as given."""
    runner, recorder, _, _, _ = _build_runner(tmp_path, clock=_ticking_clock(), name="gap")
    result = runner.run()

    payload = json.loads(recorder.result_path.read_text(encoding="utf-8"))
    gap = payload["cadence"]["since_previous_move_mean"]
    total = payload["cadence"]["total_mean"]

    # A mean of raw intervals, not of an averaged or decayed series: it sits
    # between the shortest and longest possible step and cannot be smaller than
    # the first interval's floor.
    assert 0.0 < gap
    assert gap <= result.duration * 1000.0
    assert total > 0.0


def test_the_first_movement_of_a_run_is_timed_but_has_no_predecessor(tmp_path: Path) -> None:
    """The first input has no previous movement, so its interval is zero.

    Recording it as zero keeps ``moves_timed`` equal to the number of movements
    actually sent, which is what makes the mean trustworthy.
    """
    runner, recorder, _, _, _ = _build_runner(tmp_path, clock=_ticking_clock(), name="first")
    result = runner.run()

    payload = json.loads(recorder.result_path.read_text(encoding="utf-8"))
    assert payload["cadence"]["moves_timed"] == float(result.moves_sent)
    assert result.moves_sent >= 1

    intervals = [
        row["timings"]["since_previous_move"]
        for row in runner.run_recorder.steps
        if "since_previous_move" in row.get("timings", {})
    ]
    assert intervals, "no step recorded an inter-movement interval"
    assert intervals[0] == 0.0, intervals


def test_every_step_records_a_phase_breakdown_without_pixels(tmp_path: Path) -> None:
    runner, _, _, _, _ = _build_runner(tmp_path, clock=_ticking_clock(), name="phases")
    runner.run()

    rows = [row for row in runner.run_recorder.steps if "timings" in row]
    assert rows
    for row in rows:
        timings = row["timings"]
        # Every step captures a frame and asks the policy what to do, so those
        # two phases are always present. ``act`` is only present when something
        # was actually sent, which is why the run's final STOP step has no
        # movement phase - reporting one would invent a movement that never
        # happened.
        assert set(timings) >= {"capture", "decide", "total"}
        assert all(isinstance(value, float) for value in timings.values())
        assert all(value >= 0.0 for value in timings.values())
        assert timings["total"] >= timings["decide"]
    assert any("act" in row["timings"] for row in rows), "no step timed a movement"


def test_the_runner_threads_the_configured_verify_settle_into_the_loop(tmp_path: Path) -> None:
    """The settle only fixes anything if the run actually configures it.

    The loop has its own default, but WAKE-001 is the experiment that needs the
    wait, so a runner that failed to pass its configured value through would
    silently go back to judging every movement by the picture from before it.
    """
    slept: list[float] = []
    # Deliberately not a multiple of the step interval: the loop paces itself to
    # the configured step rate through the same sleeper, so the settle has to be
    # a value that pacing cannot produce for the count below to mean anything.
    config = Config(data_dir=tmp_path / "data", wake_verify_settle_seconds=0.35)
    runner, _, _, _, _ = _build_runner(
        tmp_path, config=config, sleeper=slept.append, name="settle"
    )
    result = runner.run()

    waited = [
        row for row in runner.run_recorder.steps if "verify_settle" in row.get("timings", {})
    ]
    assert result.moves_sent > 0
    assert len(waited) == result.moves_sent
    assert slept.count(0.35) == result.moves_sent


def test_no_verify_settle_is_recorded_when_the_wait_is_disabled(tmp_path: Path) -> None:
    slept: list[float] = []
    config = Config(data_dir=tmp_path / "data", wake_verify_settle_seconds=0.0)
    runner, _, _, _, _ = _build_runner(
        tmp_path, config=config, sleeper=slept.append, name="no-settle"
    )
    result = runner.run()

    assert result.moves_sent > 0
    assert 0.35 not in slept
    assert not any("verify_settle" in row.get("timings", {}) for row in runner.run_recorder.steps)


# ---------------------------------------------------------------------------
# the observer panel
# ---------------------------------------------------------------------------


def test_the_wake_report_starts_out_honestly_empty() -> None:
    payload = WakeReport().to_dict()
    assert payload["available"] is False
    assert payload["status"] == "not-run"
    assert payload["progress"] == []
    assert payload["target"]["centre"] is None
    assert payload["mapping"]["source"] == "unmeasured"
    assert payload["mapping"]["pixels_per_delta_x"] is None


def test_a_wake_report_round_trips_through_the_observer(tmp_path: Path) -> None:
    state = ObserverState(Config(data_dir=tmp_path / "data"))
    assert state.wake.available is False
    assert state.snapshot_dict()["wake"]["available"] is False

    state.begin_run("r1")
    state.publish_wake(
        available=True,
        status="running",
        state="CENTERING",
        max_moves=45,
        moves_sent=6,
        target_centre=(10.0, 20.0),
        target_bbox=(1, 2, 3, 4),
        target_salience=0.4,
        target_seen=2,
        target_offset=(100.0, 40.0),
        target_distance=107.7,
        progress=(400.0, 180.0, 40.0),
        confidence=0.8,
        strategy="centre_left_medium",
        recent_event="Attempting a centring correction.",
        mapping_source="self-measured",
        pixels_per_delta_x=0.5,
    )

    published = state.snapshot_dict()["wake"]
    assert published["available"] is True
    assert published["state"] == "CENTERING"
    assert published["progress"] == [400.0, 180.0, 40.0]
    assert published["target"]["centre"] == [10.0, 20.0]
    assert published["target"]["bbox"] == [1, 2, 3, 4]
    assert published["target"]["salience"] == pytest.approx(0.4)
    assert published["target"]["seen"] == 2
    assert published["target_offset"] == [100.0, 40.0]
    assert published["target_distance"] == pytest.approx(107.7)
    assert published["mapping"]["source"] == "self-measured"
    assert published["mapping"]["pixels_per_delta_x"] == pytest.approx(0.5)
    assert published["measured_at"] is not None

    # a partial update merges rather than replacing the whole panel
    state.publish_wake(moves_sent=7)
    assert state.snapshot_dict()["wake"]["moves_sent"] == 7
    assert state.snapshot_dict()["wake"]["state"] == "CENTERING"
    assert state.snapshot_dict()["wake"]["target"]["centre"] == [10.0, 20.0]


def test_a_new_run_resets_the_wake_panel(tmp_path: Path) -> None:
    state = ObserverState(Config(data_dir=tmp_path / "data"))
    state.publish_wake(available=True, status="completed", state="COMPLETE")
    assert state.snapshot_dict()["wake"]["available"] is True

    state.begin_run("r2")
    assert state.snapshot_dict()["wake"]["available"] is False
    assert state.snapshot_dict()["wake"]["state"] == "STARTING"


def test_a_malformed_wake_field_is_refused_with_a_clear_message() -> None:
    from autocraft.observer.snapshot import ObserverError

    with pytest.raises(ObserverError, match="2 numbers"):
        WakeReport(target_centre=(1.0, 2.0, 3.0))


def test_the_live_publisher_reaches_the_panel(tmp_path: Path) -> None:
    """The CLI publisher must actually land on the panel, not fail silently.

    ``_wake_publisher`` swallows display failures on purpose, so a bad keyword
    would not raise anywhere visible - the panel would simply stay empty for the
    whole run. This drives the real publisher with the real report shape and
    insists the panel comes alive.
    """
    from autocraft.cli import _wake_panel_fields, _wake_publisher

    runner, _recorder, policy, _events, _statuses = _build_runner(tmp_path)
    state = ObserverState(Config(data_dir=tmp_path / "data"))
    state.begin_run("r1")
    publish = _wake_publisher(state, run_id="r1", window=(320, 240))
    assert publish is not None

    # the same keys the runner hands the display layer
    report = dict(policy.report())
    report["steps"] = 0
    report["max_steps"] = runner.max_steps
    publish(report)

    panel = state.snapshot_dict()["wake"]
    assert panel["available"] is True
    assert panel["state"] == WakeState.STARTING.value
    assert panel["window"]["width"] == 320
    assert panel["window"]["height"] == 240
    assert panel["run_id"] == "r1"

    # the explicitly passed keys are not also sent through the filtered report
    fields = _wake_panel_fields(report)
    assert "state" not in fields
    assert "stop_reason" not in fields
    assert "run_id" not in fields
    assert "available" not in fields
    assert "target_centre" in fields


# ---------------------------------------------------------------------------
# structural: what the package is forbidden from becoming
# ---------------------------------------------------------------------------

_WAKE_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "autocraft" / "wake"
_PACKAGE_DOTTED = "autocraft.wake"

#: The wake layer may move the mouse. That is the whole point of it. What it may
#: not do is reach past the action abstraction into the input layer itself, and it
#: may not open a socket or a subprocess to get around the ban.
_FORBIDDEN_IMPORTS = (
    "autocraft.control",
    "socket",
    "subprocess",
    "ctypes",
    "shutil",
    "urllib",
    "http",
    "win32",
    "pyautogui",
    "keyboard",
    "mouse",
    "pynput",
)

#: The DO-NOT-BUILD list, checked as imports so a docstring naming the ban is not
#: mistaken for a violation. The owner's green light lifted the ban on model
#: *APIs*; it did not authorise a learned model inside the behaviour layer.
_FORBIDDEN_MODEL_IMPORTS = (
    "torch",
    "tensorflow",
    "keras",
    "jax",
    "sklearn",
    "onnx",
    "onnxruntime",
    "ultralytics",
    "cv2.dnn",
    "clip",
    "open_clip",
    "transformers",
    "stable_baselines3",
    "gymnasium",
    "gym",
)

_VERDICT_TOKENS = ("pass", "fail", "verdict", "good", "bad", "acceptable", "grade")

#: Names that carry that vocabulary without being a verdict about the run.
#: ``ProgressVerdict`` and ``RepetitionVerdict`` are assessments of a *single*
#: measurement - did the distance fall, did this movement repeat - and
#: ``FailedAttempt`` records one approach given up on. None of them says the run
#: was good or bad; the policy reports numbers and the owner decides what they
#: mean. The test below is what stops an ``is_acceptable()`` appearing later.
_NOT_A_VERDICT = frozenset(
    {"FAILED", "failed_strategies", "STRATEGY_FAILED", "FailedAttempt", "ProgressVerdict", "RepetitionVerdict"}
)

_ALLOWED_ACTION_IMPORTS = frozenset({"Action", "ActionKind"})


def _wake_sources() -> list[Path]:
    sources = sorted(_WAKE_PACKAGE.glob("*.py"))
    assert sources, "the wake package should have modules to check"
    return sources


def _absolute_module(node: ast.ImportFrom) -> str:
    """Resolve an ``ImportFrom`` to a fully dotted path.

    The wake modules import relatively (``from ..agent.action import Action``), so
    ``node.module`` alone is ``"agent.action"`` and comparing it against
    ``"autocraft.agent.action"`` would never match - a structural test that
    silently checks nothing. ``node.level`` is how many packages to walk up.
    """
    if not node.level:
        return node.module or ""
    parts = _PACKAGE_DOTTED.split(".")
    prefix = ".".join(parts[: len(parts) - (node.level - 1)])
    return f"{prefix}.{node.module}" if node.module else prefix


def _imports(source: Path) -> list[tuple[str, list[str]]]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    found: list[tuple[str, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, []) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((_absolute_module(node), [alias.name for alias in node.names]))
    return found


def test_relative_imports_are_resolved_to_absolute_paths() -> None:
    """The checks below are only meaningful if they can see the imports."""
    policy = _imports(_WAKE_PACKAGE / "policy.py")
    assert ("autocraft.agent.action", ["Action"]) in policy

    record = _imports(_WAKE_PACKAGE / "record.py")
    assert any(
        module == "autocraft.wake.events" and "WakeEvent" in names for module, names in record
    ), record


def test_no_wake_module_reaches_past_the_action_seam() -> None:
    """The behaviour layer decides; it does not inject.

    Movement goes out as an :class:`Action` and is executed by whatever executor
    the loop was given. Importing the control layer would let the policy press
    keys directly, which is exactly the layer boundary the milestone draws.
    """
    for source in _wake_sources():
        for module, names in _imports(source):
            for forbidden in _FORBIDDEN_IMPORTS:
                assert not module.startswith(forbidden), (
                    f"{source.name} imports {module!r}, which can reach {forbidden!r}"
                )
            assert module != "importlib", f"{source.name} could import anything at runtime"
            if module.startswith("autocraft.agent.action"):
                unexpected = sorted(set(names) - _ALLOWED_ACTION_IMPORTS)
                assert not unexpected, (
                    f"{source.name} takes {unexpected} from the action module; only "
                    f"{sorted(_ALLOWED_ACTION_IMPORTS)} may be used, because the "
                    "executor is what injects input"
                )


def test_the_wake_package_imports_no_trained_model() -> None:
    for source in _wake_sources():
        for module, _ in _imports(source):
            for forbidden in _FORBIDDEN_MODEL_IMPORTS:
                assert module != forbidden and not module.startswith(f"{forbidden}."), (
                    f"{source.name} imports the model framework {module!r}"
                )


def test_the_wake_package_reads_no_privileged_game_state() -> None:
    """Pixels in, mouse movements out. There is no third channel."""
    forbidden_modules = ("luanti", "minetest", "mcpi", "autocraft.game", "autocraft.world")
    for source in _wake_sources():
        for module, _ in _imports(source):
            for forbidden in forbidden_modules:
                assert not module.startswith(forbidden), f"{source.name} imports {module!r}"


def test_the_wake_package_opens_no_network_or_subprocess() -> None:
    """Nothing in the behaviour layer may reach out of the process.

    ``run`` is checked only as a *bare* call. ``loop.run(...)`` and
    ``runner.run()`` are this package driving its own loop, and flagging them
    would be a false positive; ``subprocess`` and ``os`` are already banned from
    being imported, so an attribute call cannot be reaching a shell.
    """
    for source in _wake_sources():
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            attribute = getattr(node.func, "attr", None)
            bare = getattr(node.func, "id", None)
            assert attribute not in {"urlopen", "Popen", "system", "connect"}, (
                f"{source.name} calls .{attribute}()"
            )
            assert bare not in {"urlopen", "Popen", "system", "connect", "run"}, (
                f"{source.name} calls {bare}()"
            )


def test_the_wake_package_exposes_no_pass_fail_verdict() -> None:
    """WAKE-001 reports numbers. It does not decide whether the run was good."""
    import autocraft.wake as wake

    judged = [
        name
        for name in wake.__all__
        if callable(getattr(wake, name))
        and name not in _NOT_A_VERDICT
        and any(token in name.lower() for token in _VERDICT_TOKENS)
    ]
    assert judged == [], f"the wake package exports verdict-shaped callables: {judged}"


def test_the_thought_system_still_cannot_reach_the_control_layer() -> None:
    """A thought is a caption. It must not be able to move anything."""
    thoughts = Path(__file__).resolve().parent.parent / "src" / "autocraft" / "thoughts"
    for source in sorted(thoughts.glob("*.py")):
        for module, _ in _imports(source):
            assert not module.startswith("autocraft.control"), (
                f"thoughts/{source.name} imports {module!r}"
            )
            assert not module.startswith("autocraft.agent.action"), (
                f"thoughts/{source.name} imports {module!r}"
            )


def test_the_observer_package_is_still_read_only() -> None:
    """The WAKE panel is a display, so the package that serves it must not touch."""
    observer = Path(__file__).resolve().parent.parent / "src" / "autocraft" / "observer"
    for source in sorted(observer.glob("*.py")):
        for module, _ in _imports(source):
            assert not module.startswith("autocraft.control"), (
                f"observer/{source.name} imports {module!r}"
            )


def test_the_wake_package_never_presses_a_key() -> None:
    """Looking and moving are allowed. Typing is not part of this milestone."""
    for source in _wake_sources():
        text = source.read_text(encoding="utf-8")
        assert "key_tap" not in text, f"{source.name} mentions a key press"
        assert "ActionKind.KEY" not in text, f"{source.name} constructs a key action"


def test_the_public_surface_is_declared_explicitly() -> None:
    import autocraft.wake as wake

    for name in wake.__all__:
        assert hasattr(wake, name), f"{name} is in __all__ but not importable"
    for expected in ("WakeDecisionPolicy", "WakeState", "WakeRunner", "WakeRecorder", "WakeResult"):
        assert expected in wake.__all__
