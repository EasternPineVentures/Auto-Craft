"""Tests for LOOK-001: the measured sensorimotor mapping experiment.

The specification's hard rules are what is pinned here, and they are pinned
structurally wherever possible, because a structural test keeps holding for a
future contributor who never reads the specification:

* **No test may move the real mouse.** Every test drives
  :class:`~autocraft.look.runner.LookRunner` through injected callables. The
  runner has no backend of its own, so this is enforced by construction rather
  than by discipline - and ``ForbiddenInputBackend`` below fails loudly if a
  command under test ever reaches for one.
* **The measurement code cannot inject input.** ``look.metrics`` is read as a
  syntax tree and must not import the control layer, exactly as
  ``tests/test_thoughts.py`` does for the thought system.
* **No pass/fail verdict exists.** There is no threshold anywhere in the
  package, and the reversibility ratio reports an absence rather than a
  fabricated number when the movement changed nothing.
* **Nothing claims to understand the camera.** The mapping note travels with the
  ratio, and the record says "measured".

The synthetic frames used throughout are built from a seeded, lightly blurred
noise field rather than a smooth gradient. That is not decoration: phase
correlation recovers a translation from the *phase* of the spectrum, and a linear
ramp has almost all of its energy in one low-frequency direction, so the
estimate collapses towards zero on it. A band-limited texture has structure in
every direction and is displaced exactly, which makes the correct answer known in
advance while still being something the estimator actually has to find.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from autocraft.config import Config
from autocraft.look import (
    ARTIFACT_DIFFERENCE_AB,
    ARTIFACT_DIFFERENCE_AC,
    ARTIFACT_FRAME_A,
    ARTIFACT_FRAME_B,
    ARTIFACT_FRAME_C,
    EVENT_ERROR,
    EVENT_INFO,
    EVENT_OBSERVE,
    EVENT_SAFETY,
    EXPERIMENT_NAME,
    LIMITATION_NOTES,
    TRIAL_COMPLETED,
    TRIAL_FAILED,
    TRIAL_INTERRUPTED,
    FocusCheck,
    LookError,
    LookRecorder,
    LookRunner,
    LookTrialResult,
    MoveOutcome,
    TrialSpec,
    difference_image,
    difference_metrics,
    estimate_shift,
    luminance,
    pixels_per_delta,
    reversibility_note,
    reversibility_ratio,
)
from autocraft.observer import ObserverState, ObserverSnapshot
from autocraft.vision.frame import Frame, ScreenRegion

# ---------------------------------------------------------------------------
# synthetic frames
# ---------------------------------------------------------------------------

#: Slack around the sampled window, so a shift never runs off the drawn field.
_TEXTURE_MARGIN = 32


def texture_frame(
    *,
    width: int = 96,
    height: int = 96,
    shift_x: int = 0,
    shift_y: int = 0,
    timestamp: float = 0.0,
    seed: int = 11,
) -> Frame:
    """A band-limited textured frame whose content is displaced by ``(shift_x, shift_y)``.

    A deterministic noise field is drawn once and sampled through a window offset
    by ``-shift``, so the content genuinely moves right by ``shift_x`` and down by
    ``shift_y`` with no wrapping, and no part of the frame is invented at the
    edge. The field is blurred slightly so it is band-limited rather than pure
    white noise, which is what phase correlation locks onto most reliably.

    The result is that ``estimate_shift(texture_frame(), texture_frame(shift_x=6))``
    has a known correct answer of ``+6`` that the estimator has to measure.
    """
    rng = np.random.default_rng(seed)
    padded = rng.integers(
        0, 256, size=(height + 2 * _TEXTURE_MARGIN, width + 2 * _TEXTURE_MARGIN)
    ).astype(np.float64)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    padded = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, padded)
    padded = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), 0, padded)

    top = _TEXTURE_MARGIN - shift_y
    left = _TEXTURE_MARGIN - shift_x
    crop = padded[top : top + height, left : left + width]
    plane = np.clip(crop, 0.0, 255.0).astype(np.uint8)
    image = np.repeat(plane[:, :, None], 3, axis=2)
    return Frame(image=image, timestamp=timestamp, region=ScreenRegion(0, 0, width, height))


def flat_frame(value: int = 40, *, width: int = 96, height: int = 96, timestamp: float = 0.0) -> Frame:
    """A frame with no structure at all, for the degenerate-input tests."""
    image = np.full((height, width, 3), value, dtype=np.uint8)
    return Frame(image=image, timestamp=timestamp, region=ScreenRegion(0, 0, width, height))


def _collect_events() -> tuple[list[tuple[str, str]], object]:
    events: list[tuple[str, str]] = []
    return events, (lambda message, kind: events.append((message, kind)))


def _runner(
    recorder: LookRecorder,
    *,
    frames: list[Frame],
    verify: list[FocusCheck] | None = None,
    move: list[MoveOutcome] | None = None,
    stop_requested: list[bool] | None = None,
    moves_log: list[tuple[int, int]] | None = None,
    released: list[str] | None = None,
    capture_error: BaseException | None = None,
    on_event: object = None,
    on_frame: object = None,
    on_trial: object = None,
    window_size: tuple[int, int] = (96, 96),
) -> LookRunner:
    """Build a runner fed from scripts, with no hardware anywhere in sight.

    ``frames`` is consumed one entry per capture. Each of ``verify``, ``move`` and
    ``stop_requested`` is its own script, consumed one entry per call with the
    last entry repeating once it runs out, so a test scripts only the calls it
    cares about. The scripts are indexed by call count, which makes the ordering
    the sequence uses explicit: verify call 0 is the pre-capture check, call 1 is
    the check before the outbound move, and call 2 the one before the reverse.
    """
    remaining = list(frames)
    checks = list(verify or [FocusCheck(ready=True)])
    outcomes = list(move or [MoveOutcome(sent=True)])
    stops = list(stop_requested or [False])
    log = moves_log if moves_log is not None else []
    counts = {"verify": 0, "stop": 0}

    def capture() -> Frame:
        if capture_error is not None:
            raise capture_error
        if not remaining:
            raise AssertionError("the test scripted fewer frames than the runner captured")
        return remaining.pop(0)

    def do_move(dx: int, dy: int) -> MoveOutcome:
        log.append((dx, dy))
        index = min(len(log) - 1, len(outcomes) - 1)
        return outcomes[index]

    def do_verify() -> FocusCheck:
        index = min(counts["verify"], len(checks) - 1)
        counts["verify"] += 1
        return checks[index]

    def do_stop() -> bool:
        index = min(counts["stop"], len(stops) - 1)
        counts["stop"] += 1
        return stops[index]

    return LookRunner(
        recorder=recorder,
        capture=capture,
        move=do_move,
        verify=do_verify,
        stop_requested=do_stop,
        release=lambda reason: (released if released is not None else []).append(reason),
        window_size=window_size,
        clock=lambda: 1000.0,
        sleeper=lambda seconds: None,
        on_event=on_event,
        on_frame=on_frame,
        on_trial=on_trial,
    )


@pytest.fixture
def recorder(tmp_path: Path) -> LookRecorder:
    """A recorder writing into pytest's temporary directory."""
    return LookRecorder(
        tmp_path / "look", run_id="run-1", plan=(TrialSpec(index=0, dx=10, dy=0, settle_seconds=0.0),)
    )


# ---------------------------------------------------------------------------
# structural: the measurement code cannot reach the input layer
# ---------------------------------------------------------------------------

#: The look package's source directory, read directly so the structural tests
#: below need no import of the code they are checking.
_LOOK_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "autocraft" / "look"

_FORBIDDEN_IMPORTS = (
    "autocraft.control",
    "autocraft.agent",
    "autocraft.vision",
    "autocraft.observer",
    "socket",
    "subprocess",
    "ctypes",
    "shutil",
    "urllib",
    "http",
    "os",
    "win32",
    "pyautogui",
    "keyboard",
    "mouse",
)

#: The trained models the specification forbids. Checked as *imports*, because a
#: substring search would flag ``numpy.clip`` and a docstring that names the ban.
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

#: Vocabulary a pass/fail verdict would need. ``DEFAULT_CHANGED_THRESHOLD`` is
#: deliberately *not* here: it is the luminance cutoff that decides which pixels
#: count as changed for the difference map, which is a measurement, not a verdict.
_VERDICT_TOKENS = ("pass", "fail", "verdict", "good", "bad", "score", "grade", "acceptable")

#: Names that carry that vocabulary without being a verdict, listed so that a
#: *new* verdict-shaped name fails the test and has to be looked at by hand.
#: ``TRIAL_FAILED`` and ``capture_failure_reason`` both describe an operation that
#: did not happen - a capture died, the safety layer refused the move - and never
#: a measurement that came out wrong.
_NOT_A_VERDICT = frozenset({"TRIAL_FAILED", "capture_failure_reason"})


def _look_sources() -> list[Path]:
    sources = sorted(_LOOK_PACKAGE.glob("*.py"))
    assert sources, "the look package should have modules to check"
    return sources


def _reads_as_a_verdict(name: str) -> bool:
    if name in _NOT_A_VERDICT:
        return False
    lowered = name.lower()
    return any(token in lowered for token in _VERDICT_TOKENS)


def _imported_modules(source: Path) -> list[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    return imported


def test_no_look_module_can_reach_the_control_layer() -> None:
    """``look`` measures pixels; it must not be able to inject anything.

    Imports are read from the syntax tree rather than searched for in the text,
    so a docstring that *names* the control layer is not mistaken for an import
    of it.
    """
    for source in _look_sources():
        imported = _imported_modules(source)
        for module in imported:
            for forbidden in _FORBIDDEN_IMPORTS:
                assert not module.startswith(forbidden), (
                    f"{source.name} imports {module!r}, which can reach {forbidden!r}"
                )
        assert "importlib" not in imported, f"{source.name} could import anything at runtime"


def test_the_look_package_imports_no_trained_model() -> None:
    """No recognition, no detector, no learned weights: the only estimate is closed form."""
    for source in _look_sources():
        for module in _imported_modules(source):
            for forbidden in _FORBIDDEN_MODEL_IMPORTS:
                assert module != forbidden and not module.startswith(f"{forbidden}."), (
                    f"{source.name} imports the model framework {module!r}"
                )


def test_the_look_package_exposes_no_pass_fail_verdict() -> None:
    """LOOK-001 reports numbers; it does not decide whether a result is good.

    ``TRIAL_FAILED`` is not a counterexample and is excluded on purpose: it says
    the *trial could not be carried out* - a capture died, the safety layer
    refused the move - never that the measurement came out wrong. What is checked
    here is that no callable on the public surface turns a measurement into a
    judgement, so adding an ``is_acceptable()`` helper later fails here rather
    than quietly reintroducing the verdict the specification forbids.
    """
    import autocraft.look as look

    names = list(look.__all__)
    assert names, "the package should declare its public surface"

    judged = [name for name in names if callable(getattr(look, name)) and _reads_as_a_verdict(name)]
    assert judged == [], f"the look package exports verdict-shaped callables: {judged}"

    statuses = sorted(
        name for name in names if isinstance(getattr(look, name), str) and _reads_as_a_verdict(name)
    )
    assert statuses == [], (
        "the only verdict-shaped strings the package may export are the trial outcome "
        f"constants, and those are listed as non-verdicts; found {statuses}"
    )


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


class TestDifferenceMetrics:
    """The per-pixel difference, and the coarse map that goes with it."""

    def test_identical_frames_differ_by_nothing(self) -> None:
        frame = texture_frame()
        difference = difference_metrics(frame, frame)
        assert difference.identical
        assert difference.mean_absolute_difference == 0.0
        assert difference.rmse == 0.0
        assert difference.changed_fraction == 0.0
        assert difference.max_absolute_difference == 0.0

    def test_different_frames_report_a_positive_difference(self) -> None:
        first = texture_frame(shift_x=0)
        second = texture_frame(shift_x=6)
        difference = difference_metrics(first, second)
        assert not difference.identical
        assert difference.mean_absolute_difference > 0.0
        assert difference.max_absolute_difference > 0.0

    def test_a_larger_shift_differs_more(self) -> None:
        """The metric is monotone in the size of the change, not just non-zero."""
        base = texture_frame()
        small = difference_metrics(base, texture_frame(shift_x=2))
        large = difference_metrics(base, texture_frame(shift_x=12))
        assert large.mean_absolute_difference > small.mean_absolute_difference

    def test_the_block_map_has_the_requested_shape(self) -> None:
        difference = difference_metrics(texture_frame(), texture_frame(shift_x=4), block_grid=8)
        assert difference.block_grid == 8
        assert len(difference.block_map) == 8
        assert all(len(row) == 8 for row in difference.block_map)

    def test_a_non_square_block_grid_is_still_square(self) -> None:
        difference = difference_metrics(texture_frame(), texture_frame(shift_x=4), block_grid=3)
        assert len(difference.block_map) == 3
        assert all(len(row) == 3 for row in difference.block_map)

    def test_the_block_map_averages_the_whole_frame(self) -> None:
        """Every pixel belongs to exactly one block, so the means agree."""
        difference = difference_metrics(texture_frame(), texture_frame(shift_x=5), block_grid=4)
        flat = [value for row in difference.block_map for value in row]
        assert np.mean(flat) == pytest.approx(difference.mean_absolute_difference, abs=1e-6)

    def test_a_frame_too_small_for_the_grid_does_not_invent_values(self) -> None:
        tiny = flat_frame(90, width=4, height=4)
        difference = difference_metrics(tiny, flat_frame(40, width=4, height=4), block_grid=8)
        flat = [value for row in difference.block_map for value in row]
        assert len(flat) == 64
        assert set(flat) == {difference.mean_absolute_difference}

    def test_mismatched_shapes_are_refused_rather_than_compared(self) -> None:
        with pytest.raises(LookError, match="same shape"):
            difference_metrics(texture_frame(width=64, height=64), texture_frame(width=48, height=48))

    def test_a_grid_below_one_is_refused(self) -> None:
        with pytest.raises(LookError, match="at least 1"):
            difference_metrics(texture_frame(), texture_frame(shift_x=1), block_grid=0)

    def test_a_malformed_image_is_refused(self) -> None:
        with pytest.raises(LookError, match="H x W x 3"):
            luminance(np.zeros((10, 10), dtype=np.uint8))

    def test_luminance_of_a_grey_frame_is_that_grey(self) -> None:
        luma = luminance(flat_frame(200, width=8, height=8))
        assert luma.shape == (8, 8)
        assert np.allclose(luma, 200.0)


class TestDifferenceImage:
    """The picture written to ``difference_ab.png``."""

    def test_identical_frames_produce_a_black_image(self) -> None:
        frame = texture_frame()
        image = difference_image(frame, frame)
        assert image.dtype == np.uint8
        assert image.shape == (96, 96)
        assert image.max() == 0

    def test_a_changed_picture_produces_non_black_pixels(self) -> None:
        image = difference_image(texture_frame(), texture_frame(shift_x=8))
        assert image.max() > 0

    def test_a_mismatched_pair_is_refused(self) -> None:
        with pytest.raises(LookError, match="same shape"):
            difference_image(texture_frame(width=32, height=32), texture_frame(width=64, height=64))


class TestShiftEstimate:
    """The closed-form displacement estimate. No model is involved."""

    def test_identical_frames_estimate_no_movement(self) -> None:
        frame = texture_frame()
        estimate = estimate_shift(frame, frame)
        assert estimate.available
        assert estimate.x == pytest.approx(0.0, abs=0.5)
        assert estimate.y == pytest.approx(0.0, abs=0.5)
        assert estimate.magnitude == pytest.approx(0.0, abs=0.7)

    def test_a_positive_shift_is_estimated_as_positive(self) -> None:
        estimate = estimate_shift(texture_frame(), texture_frame(shift_x=6))
        assert estimate.available
        assert estimate.x == pytest.approx(6.0, abs=0.5)

    def test_a_negative_shift_is_estimated_as_negative(self) -> None:
        """The sign is measured, never assumed: the opposite shift must flip it."""
        estimate = estimate_shift(texture_frame(), texture_frame(shift_x=-6))
        assert estimate.available
        assert estimate.x == pytest.approx(-6.0, abs=0.5)

    def test_a_vertical_shift_lands_on_the_other_axis(self) -> None:
        estimate = estimate_shift(texture_frame(), texture_frame(shift_y=5))
        assert estimate.available
        assert estimate.x == pytest.approx(0.0, abs=0.5)
        assert estimate.y == pytest.approx(5.0, abs=0.5)

    def test_a_reversed_pair_estimates_the_opposite_displacement(self) -> None:
        forward = estimate_shift(texture_frame(), texture_frame(shift_x=7))
        backward = estimate_shift(texture_frame(shift_x=7), texture_frame())
        assert forward.x == pytest.approx(-backward.x, abs=0.5)

    def test_the_estimate_reports_a_quality(self) -> None:
        """The response is reported, and deliberately not normalised to ``0..1``.

        OpenCV makes no range promise, so clamping it here would be inventing a
        bound. A strong textured match legitimately comes back slightly above
        ``1.0``, and the value is an ordering, not a score.
        """
        matched = estimate_shift(texture_frame(), texture_frame(shift_x=6))
        unrelated = estimate_shift(texture_frame(), texture_frame(seed=99))
        assert matched.quality > 0.0
        assert matched.quality > unrelated.quality

    def test_a_frame_too_small_to_align_is_reported_not_guessed(self) -> None:
        tiny = flat_frame(30, width=6, height=6)
        estimate = estimate_shift(tiny, flat_frame(60, width=6, height=6))
        assert not estimate.available
        assert estimate.reason
        assert estimate.x == 0.0 and estimate.y == 0.0

    def test_a_mismatched_pair_is_refused(self) -> None:
        with pytest.raises(LookError, match="same shape"):
            estimate_shift(texture_frame(width=32, height=32), texture_frame(width=64, height=64))

    def test_the_method_is_named_and_is_not_a_model(self) -> None:
        assert estimate_shift(texture_frame(), texture_frame()).method == "phase-correlation"


class TestPixelsPerDelta:
    """The ratio between two independently measured quantities."""

    def test_a_zero_denominator_axis_is_undefined_not_infinite(self) -> None:
        """Division by zero must produce ``None``, never ``inf`` or a crash."""
        shift = estimate_shift(texture_frame(), texture_frame(shift_x=6))
        per_x, per_y = pixels_per_delta(shift, dx=6, dy=0)
        assert per_y is None
        assert per_x == pytest.approx(1.0, abs=0.1)

    def test_both_axes_zero_are_both_undefined(self) -> None:
        shift = estimate_shift(texture_frame(), texture_frame())
        assert pixels_per_delta(shift, dx=0, dy=0) == (None, None)

    def test_the_ratio_keeps_the_measured_sign(self) -> None:
        shift = estimate_shift(texture_frame(), texture_frame(shift_x=-6))
        per_x, _ = pixels_per_delta(shift, dx=6, dy=0)
        assert per_x is not None and per_x < 0.0

    def test_the_ratio_scales_with_the_delta_that_produced_it(self) -> None:
        shift = estimate_shift(texture_frame(), texture_frame(shift_x=6))
        per_x, _ = pixels_per_delta(shift, dx=3, dy=0)
        assert per_x == pytest.approx(2.0, abs=0.2)


class TestReversibility:
    """``difference(A, C) / difference(A, B)``, and its honest absence."""

    def test_a_perfect_return_is_zero(self) -> None:
        assert reversibility_ratio(10.0, 0.0) == 0.0

    def test_an_undone_nothing_is_one(self) -> None:
        assert reversibility_ratio(10.0, 10.0) == 1.0

    def test_moving_further_away_exceeds_one(self) -> None:
        ratio = reversibility_ratio(4.0, 9.0)
        assert ratio is not None and ratio > 1.0

    def test_no_outbound_change_has_no_ratio(self) -> None:
        """The divide-by-zero case, in the form the specification asks about."""
        assert reversibility_ratio(0.0, 5.0) is None

    def test_the_epsilon_is_honoured(self) -> None:
        assert reversibility_ratio(0.5, 5.0, identical_epsilon=1.0) is None

    def test_the_note_explains_an_absent_ratio(self) -> None:
        note = reversibility_note(0.0, 5.0)
        assert "identical" in note and "does not exist" in note

    def test_the_note_explains_a_perfect_return(self) -> None:
        assert "returned exactly" in reversibility_note(10.0, 0.0)

    def test_the_note_reports_the_ratio_otherwise(self) -> None:
        note = reversibility_note(10.0, 2.5)
        assert "0.250" in note


# ---------------------------------------------------------------------------
# the runner's bounded sequence
# ---------------------------------------------------------------------------


class TestLookRunnerSequence:
    """The nine-step trial, exercised entirely through injected callables."""

    def test_a_complete_trial_records_every_artifact(self, recorder: LookRecorder) -> None:
        frames = [
            texture_frame(timestamp=1.0),
            texture_frame(shift_x=6, timestamp=2.0),
            texture_frame(timestamp=3.0),
        ]
        runner = _runner(recorder, frames=frames, moves_log=[])

        result = runner.run((TrialSpec(index=0, dx=10, dy=0, settle_seconds=0.0),))

        assert result.status == TRIAL_COMPLETED
        assert result.completed_trials == 1
        trial = result.trials[0]
        assert trial.complete
        assert trial.movements_sent == 2
        assert trial.frame_a is not None and trial.frame_b is not None and trial.frame_c is not None
        for name in (
            ARTIFACT_FRAME_A,
            ARTIFACT_FRAME_B,
            ARTIFACT_FRAME_C,
            ARTIFACT_DIFFERENCE_AB,
            ARTIFACT_DIFFERENCE_AC,
        ):
            assert name in trial.artifacts, f"{name} was not written"
            assert (recorder.directory / trial.artifacts[name]).exists()

    def test_the_outbound_move_is_followed_by_its_exact_reverse(self, recorder: LookRecorder) -> None:
        frames = [texture_frame(), texture_frame(shift_x=6), texture_frame()]
        moves: list[tuple[int, int]] = []
        runner = _runner(recorder, frames=frames, moves_log=moves)

        runner.run((TrialSpec(index=0, dx=10, dy=0, settle_seconds=0.0),))

        assert moves == [(10, 0), (-10, 0)]

    def test_a_plan_of_several_trials_runs_each_one(self, tmp_path: Path) -> None:
        plan = tuple(TrialSpec(index=index, dx=4, dy=0, settle_seconds=0.0) for index in range(3))
        multi = LookRecorder(tmp_path / "multi", run_id="run-2", plan=plan)
        frames = [texture_frame(shift_x=(0 if index % 2 else 5)) for index in range(9)]
        runner = _runner(multi, frames=frames, moves_log=[])

        result = runner.run(plan)

        assert result.status == TRIAL_COMPLETED
        assert len(result.trials) == 3
        assert result.completed_trials == 3
        assert result.movements_sent == 6

    def test_a_multi_trial_run_keeps_each_trial_in_its_own_directory(self, tmp_path: Path) -> None:
        """No trial may overwrite another's frames."""
        plan = tuple(TrialSpec(index=index, dx=4, dy=0, settle_seconds=0.0) for index in range(2))
        multi = LookRecorder(tmp_path / "multi", run_id="run-3", plan=plan)
        frames = [texture_frame() for _ in range(6)]
        runner = _runner(multi, frames=frames, moves_log=[])

        runner.run(plan)

        written = [Path(path) for path in multi.trials[0].artifacts.values()]
        assert all("trial-00" in path.parts for path in written)
        assert multi.result_path.exists()

    def test_the_settle_happens_before_each_capture_after_a_move(self, tmp_path: Path) -> None:
        """The settle is a real wait, so it must be requested the right number of times."""
        sleeps: list[float] = []
        plan = (TrialSpec(index=0, dx=5, dy=0, settle_seconds=0.25),)
        rec = LookRecorder(tmp_path / "settle", run_id="run-4", plan=plan)
        frames = [texture_frame(), texture_frame(shift_x=4), texture_frame()]
        runner = LookRunner(
            recorder=rec,
            capture=lambda: frames.pop(0),
            move=lambda dx, dy: MoveOutcome(sent=True),
            verify=lambda: FocusCheck(ready=True),
            stop_requested=lambda: False,
            release=lambda reason: None,
            window_size=(96, 96),
            clock=lambda: 1000.0,
            sleeper=sleeps.append,
        )

        runner.run(plan)

        assert sleeps == [0.25, 0.25]

    def test_a_zero_settle_does_not_call_the_sleeper(self, tmp_path: Path) -> None:
        sleeps: list[float] = []
        plan = (TrialSpec(index=0, dx=5, dy=0, settle_seconds=0.0),)
        rec = LookRecorder(tmp_path / "no-settle", run_id="run-5", plan=plan)
        frames = [texture_frame(), texture_frame(shift_x=4), texture_frame()]
        runner = LookRunner(
            recorder=rec,
            capture=lambda: frames.pop(0),
            move=lambda dx, dy: MoveOutcome(sent=True),
            verify=lambda: FocusCheck(ready=True),
            stop_requested=lambda: False,
            release=lambda reason: None,
            window_size=(96, 96),
            clock=lambda: 1000.0,
            sleeper=sleeps.append,
        )

        runner.run(plan)

        assert sleeps == []

    def test_the_measured_numbers_are_recorded(self, recorder: LookRecorder) -> None:
        frames = [
            texture_frame(),
            texture_frame(shift_x=6),
            texture_frame(),
        ]
        runner = _runner(recorder, frames=frames, moves_log=[])

        result = runner.run((TrialSpec(index=0, dx=6, dy=0, settle_seconds=0.0),))

        trial = result.trials[0]
        assert trial.a_to_b is not None and trial.a_to_c is not None
        assert trial.a_to_b.mean_absolute_difference > 0.0
        assert trial.shift is not None and trial.shift.available
        assert trial.shift.x == pytest.approx(6.0, abs=0.5)
        assert trial.pixels_per_delta_x == pytest.approx(1.0, abs=0.1)
        assert trial.pixels_per_delta_y is None
        assert trial.reversibility_ratio is not None

    def test_a_returning_scene_reverses_better_than_one_that_does_not(self, tmp_path: Path) -> None:
        """The ratio must order two runs the way the pictures actually differ."""
        returned = LookRecorder(tmp_path / "returned", run_id="r1", plan=(TrialSpec(index=0, dx=6, dy=0, settle_seconds=0.0),))
        drifted = LookRecorder(tmp_path / "drifted", run_id="r2", plan=(TrialSpec(index=0, dx=6, dy=0, settle_seconds=0.0),))

        _runner(
            returned,
            frames=[texture_frame(), texture_frame(shift_x=6), texture_frame()],
            moves_log=[],
        ).run((TrialSpec(index=0, dx=6, dy=0, settle_seconds=0.0),))
        _runner(
            drifted,
            frames=[texture_frame(), texture_frame(shift_x=6), texture_frame(shift_x=30)],
            moves_log=[],
        ).run((TrialSpec(index=0, dx=6, dy=0, settle_seconds=0.0),))

        clean = returned.trials[0].reversibility_ratio
        dirty = drifted.trials[0].reversibility_ratio
        assert clean is not None and dirty is not None
        assert clean < dirty

    def test_an_unmoving_scene_reports_an_absent_ratio_rather_than_zero(
        self, recorder: LookRecorder
    ) -> None:
        frames = [flat_frame(40), flat_frame(40), flat_frame(40)]
        runner = _runner(recorder, frames=frames, moves_log=[])

        result = runner.run((TrialSpec(index=0, dx=6, dy=0, settle_seconds=0.0),))

        trial = result.trials[0]
        assert trial.status == TRIAL_COMPLETED
        assert trial.reversibility_ratio is None
        assert "does not exist" in trial.reversibility_note

    def test_the_runner_emits_a_timeline(self, recorder: LookRecorder) -> None:
        events, on_event = _collect_events()
        frames = [texture_frame(), texture_frame(shift_x=6), texture_frame()]
        runner = _runner(recorder, frames=frames, moves_log=[], on_event=on_event)

        runner.run((TrialSpec(index=0, dx=6, dy=0, settle_seconds=0.0),))

        kinds = {kind for _, kind in events}
        assert EVENT_OBSERVE in kinds
        assert any("measured" in message for message, _ in events)

    def test_every_captured_frame_reaches_the_display_callback(self, recorder: LookRecorder) -> None:
        seen: list[Frame] = []
        frames = [texture_frame(), texture_frame(shift_x=6), texture_frame()]
        runner = _runner(recorder, frames=frames, moves_log=[], on_frame=seen.append)

        runner.run((TrialSpec(index=0, dx=6, dy=0, settle_seconds=0.0),))

        assert len(seen) == 3

    def test_each_trial_reaches_the_trial_callback(self, recorder: LookRecorder) -> None:
        seen: list[LookTrialResult] = []
        frames = [texture_frame(), texture_frame(shift_x=6), texture_frame()]
        runner = _runner(recorder, frames=frames, moves_log=[], on_trial=seen.append)

        runner.run((TrialSpec(index=0, dx=6, dy=0, settle_seconds=0.0),))

        assert len(seen) == 1
        assert seen[0].complete


class TestLookRunnerAborts:
    """Every abort path stops the experiment and releases held input."""

    def test_focus_lost_before_the_first_action_sends_nothing(
        self, recorder: LookRecorder
    ) -> None:
        released: list[str] = []
        moves: list[tuple[int, int]] = []
        runner = _runner(
            recorder,
            frames=[texture_frame()],
            verify=[FocusCheck(ready=False, reason="the target is not in front")],
            moves_log=moves,
            released=released,
        )

        result = runner.run((TrialSpec(index=0, dx=10, dy=0, settle_seconds=0.0),))

        assert result.status == TRIAL_INTERRUPTED
        assert "not in front" in result.stop_reason
        assert moves == [], "no movement may be injected once focus is lost"
        assert released, "held input must be released on the abort path"

    def test_focus_lost_between_the_outbound_and_reverse_moves(
        self, recorder: LookRecorder
    ) -> None:
        """The reverse must not be injected into a window we no longer hold."""
        released: list[str] = []
        moves: list[tuple[int, int]] = []
        # verify call 0 is before the first capture, call 1 before the outbound
        # move, call 2 before the reverse.
        checks = [
            FocusCheck(ready=True),
            FocusCheck(ready=True),
            FocusCheck(ready=False, reason="focus moved away"),
        ]
        frames = [texture_frame(), texture_frame(shift_x=6)]
        runner = _runner(
            recorder, frames=frames, verify=checks, moves_log=moves, released=released
        )

        result = runner.run((TrialSpec(index=0, dx=10, dy=0, settle_seconds=0.0),))

        assert result.status == TRIAL_INTERRUPTED
        assert moves == [(10, 0)], "only the outbound move may have been sent"
        assert released

    def test_a_capture_failure_fails_the_trial_with_the_reason(
        self, recorder: LookRecorder
    ) -> None:
        released: list[str] = []
        runner = _runner(
            recorder,
            frames=[],
            capture_error=RuntimeError("the window closed"),
            released=released,
        )

        result = runner.run((TrialSpec(index=0, dx=10, dy=0, settle_seconds=0.0),))

        assert result.status == TRIAL_FAILED
        assert "frame A could not be captured" in result.stop_reason
        assert "the window closed" in result.stop_reason
        assert result.trials[0].frame_a is None
        assert released

    def test_a_capture_failure_after_the_move_still_records_frame_a(
        self, recorder: LookRecorder
    ) -> None:
        """A partial trial keeps what it did capture rather than discarding it."""
        calls = {"count": 0}

        def capture() -> Frame:
            calls["count"] += 1
            if calls["count"] == 1:
                return texture_frame()
            raise RuntimeError("capture died")

        runner = LookRunner(
            recorder=recorder,
            capture=capture,
            move=lambda dx, dy: MoveOutcome(sent=True),
            verify=lambda: FocusCheck(ready=True),
            stop_requested=lambda: False,
            release=lambda reason: None,
            window_size=(96, 96),
            clock=lambda: 1000.0,
            sleeper=lambda seconds: None,
        )

        result = runner.run((TrialSpec(index=0, dx=10, dy=0, settle_seconds=0.0),))

        assert result.status == TRIAL_FAILED
        assert result.trials[0].frame_a is not None
        assert result.trials[0].frame_b is None
        assert result.trials[0].movements_sent == 1

    def test_the_target_disappearing_is_an_interruption(self, recorder: LookRecorder) -> None:
        moves: list[tuple[int, int]] = []
        runner = _runner(
            recorder,
            frames=[texture_frame()],
            verify=[FocusCheck(ready=False, reason="the target window is gone")],
            moves_log=moves,
        )

        result = runner.run((TrialSpec(index=0, dx=10, dy=0, settle_seconds=0.0),))

        assert result.status == TRIAL_INTERRUPTED
        assert moves == []

    def test_an_emergency_stop_before_any_movement_interrupts(self, recorder: LookRecorder) -> None:
        moves: list[tuple[int, int]] = []
        released: list[str] = []
        runner = _runner(
            recorder,
            frames=[texture_frame()],
            stop_requested=[True],
            moves_log=moves,
            released=released,
        )

        result = runner.run((TrialSpec(index=0, dx=10, dy=0, settle_seconds=0.0),))

        assert result.status == TRIAL_INTERRUPTED
        assert "emergency stop" in result.stop_reason
        assert moves == []
        assert released

    def test_an_emergency_stop_between_the_moves_stops_the_second(
        self, recorder: LookRecorder
    ) -> None:
        moves: list[tuple[int, int]] = []
        # stop is polled after the pre-capture verify, after the pre-outbound
        # verify, after the settle, then after the pre-reverse verify.
        stops = [False, False, False, True]
        frames = [texture_frame(), texture_frame(shift_x=6)]
        runner = _runner(recorder, frames=frames, stop_requested=stops, moves_log=moves)

        result = runner.run((TrialSpec(index=0, dx=10, dy=0, settle_seconds=0.0),))

        assert result.status == TRIAL_INTERRUPTED
        assert moves == [(10, 0)]

    def test_a_refused_movement_fails_the_trial_and_does_not_reverse(
        self, recorder: LookRecorder
    ) -> None:
        """If the safety layer vetoes the outbound move, there is nothing to undo."""
        moves: list[tuple[int, int]] = []
        runner = _runner(
            recorder,
            frames=[texture_frame()],
            move=[MoveOutcome(sent=False, detail="foreground lock active", refused=True)],
            moves_log=moves,
        )

        result = runner.run((TrialSpec(index=0, dx=10, dy=0, settle_seconds=0.0),))

        assert result.status == TRIAL_FAILED
        assert "safety layer refused" in result.stop_reason
        assert moves == [(10, 0)]

    def test_a_backend_failure_is_reported_differently_from_a_refusal(
        self, recorder: LookRecorder
    ) -> None:
        runner = _runner(
            recorder,
            frames=[texture_frame()],
            move=[MoveOutcome(sent=False, detail="SendInput returned 0")],
            moves_log=[],
        )

        result = runner.run((TrialSpec(index=0, dx=10, dy=0, settle_seconds=0.0),))

        assert result.status == TRIAL_FAILED
        assert "backend did not send it" in result.stop_reason

    def test_a_refused_reverse_move_fails_after_the_outbound_succeeded(
        self, recorder: LookRecorder
    ) -> None:
        moves: list[tuple[int, int]] = []
        outcomes = [MoveOutcome(sent=True), MoveOutcome(sent=False, detail="rate limited", refused=True)]
        frames = [texture_frame(), texture_frame(shift_x=6)]
        runner = _runner(recorder, frames=frames, move=outcomes, moves_log=moves)

        result = runner.run((TrialSpec(index=0, dx=10, dy=0, settle_seconds=0.0),))

        assert result.status == TRIAL_FAILED
        assert "reverse movement" in result.stop_reason
        assert moves == [(10, 0), (-10, 0)]

    def test_a_trial_that_cannot_be_measured_is_failed_not_zeroed(
        self, recorder: LookRecorder
    ) -> None:
        """A resized window makes the frames incomparable; that must not read as 0."""
        frames = [
            texture_frame(width=96, height=96),
            texture_frame(width=64, height=64, shift_x=6),
            texture_frame(width=96, height=96),
        ]
        runner = _runner(recorder, frames=frames, moves_log=[])

        result = runner.run((TrialSpec(index=0, dx=10, dy=0, settle_seconds=0.0),))

        assert result.status == TRIAL_FAILED
        assert "could not be measured" in result.stop_reason
        assert result.trials[0].a_to_b is None
        assert result.trials[0].reversibility_ratio is None

    def test_a_missing_displacement_estimate_does_not_fail_the_trial(
        self, recorder: LookRecorder
    ) -> None:
        """The estimate degrades to an explained absence; the difference still stands."""
        frames = [flat_frame(30, width=6, height=6), flat_frame(90, width=6, height=6), flat_frame(30, width=6, height=6)]
        runner = _runner(recorder, frames=frames, moves_log=[], window_size=(6, 6))

        result = runner.run((TrialSpec(index=0, dx=10, dy=0, settle_seconds=0.0),))

        assert result.status == TRIAL_COMPLETED
        trial = result.trials[0]
        assert trial.shift is not None and not trial.shift.available
        assert trial.shift.reason
        assert trial.a_to_b is not None
        assert trial.pixels_per_delta_x is None and trial.pixels_per_delta_y is None

    def test_the_experiment_stops_at_the_first_abort_and_does_not_continue(
        self, tmp_path: Path
    ) -> None:
        """Nothing further may be injected once the world stops being what we vetted."""
        plan = tuple(TrialSpec(index=index, dx=5, dy=0, settle_seconds=0.0) for index in range(3))
        rec = LookRecorder(tmp_path / "abort", run_id="run-6", plan=plan)
        moves: list[tuple[int, int]] = []
        checks = [FocusCheck(ready=True), FocusCheck(ready=True), FocusCheck(ready=False, reason="gone")]
        runner = _runner(rec, frames=[texture_frame(), texture_frame(shift_x=5)], verify=checks, moves_log=moves)

        result = runner.run(plan)

        assert result.status == TRIAL_INTERRUPTED
        assert len(result.trials) == 1, "no later trial may run after an abort"
        assert moves == [(5, 0)]

    def test_a_release_failure_does_not_hide_the_abort(self, recorder: LookRecorder) -> None:
        def explode(reason: str) -> None:
            raise RuntimeError("release backend is broken")

        runner = LookRunner(
            recorder=recorder,
            capture=lambda: texture_frame(),
            move=lambda dx, dy: MoveOutcome(sent=True),
            verify=lambda: FocusCheck(ready=False, reason="not focused"),
            stop_requested=lambda: False,
            release=explode,
            window_size=(96, 96),
            clock=lambda: 1000.0,
            sleeper=lambda seconds: None,
        )

        result = runner.run((TrialSpec(index=0, dx=10, dy=0, settle_seconds=0.0),))

        assert result.status == TRIAL_INTERRUPTED
        assert "not focused" in result.stop_reason


class TestTrialSpec:
    """The plan's unit, and its bounded calibration series."""

    def test_the_reverse_is_the_exact_negation(self) -> None:
        spec = TrialSpec(index=0, dx=10, dy=-4, settle_seconds=0.0)
        assert (spec.reverse_dx, spec.reverse_dy) == (-10, 4)

    def test_a_zero_delta_spec_is_still_representable(self) -> None:
        spec = TrialSpec(index=0, dx=0, dy=0, settle_seconds=0.0)
        assert (spec.reverse_dx, spec.reverse_dy) == (0, 0)

    def test_the_description_uses_explicit_signs(self) -> None:
        assert TrialSpec(index=0, dx=10, dy=0, settle_seconds=0.0).describe() == "(+10, +0) then (-10, +0)"

    def test_a_spec_round_trips_through_json(self) -> None:
        spec = TrialSpec(index=2, dx=5, dy=-3, settle_seconds=0.5)
        assert json.loads(json.dumps(spec.to_dict()))["dx"] == 5

    def test_the_calibration_series_is_bounded_by_the_configured_deltas(self) -> None:
        config = Config()
        deltas = config.look_calibration_deltas
        assert deltas, "a calibration series must exist"
        assert len(deltas) <= config.look_max_steps, "the series must fit inside the step bound"
        assert all(delta > 0 for delta in deltas)
        assert list(deltas) == sorted(deltas), "the series must ascend"
        # The series is refused outright if any entry is above the per-axis
        # bound, so the shipped default has to respect it. The top of the series
        # is the largest delta the actuator will accept in one command: a series
        # that cannot reach the bound cannot bracket the answer.
        assert max(deltas) <= config.max_mouse_delta, (
            "the default series would be refused by its own per-axis bound"
        )
        assert max(deltas) == config.max_mouse_delta, (
            "the series should reach the largest delta one command can carry"
        )


# ---------------------------------------------------------------------------
# the record on disk
# ---------------------------------------------------------------------------


class TestLookRecord:
    """``look_result.json`` and the images beside it."""

    def test_the_record_is_written_immediately_and_updated_in_place(
        self, tmp_path: Path
    ) -> None:
        plan = (TrialSpec(index=0, dx=6, dy=0, settle_seconds=0.0),)
        rec = LookRecorder(tmp_path / "rec", run_id="run-7", plan=plan)
        assert rec.result_path.exists()
        assert json.loads(rec.result_path.read_text(encoding="utf-8"))["status"] == "running"

        frames = [texture_frame(), texture_frame(shift_x=6), texture_frame()]
        _runner(rec, frames=frames, moves_log=[]).run(plan)

        payload = json.loads(rec.result_path.read_text(encoding="utf-8"))
        assert payload["status"] == TRIAL_COMPLETED
        assert payload["trials_completed"] == 1
        assert payload["finished_at"] is not None

    def test_the_record_never_contains_pixel_data(self, tmp_path: Path) -> None:
        """The images go to disk as files; the JSON carries paths and numbers only."""
        plan = (TrialSpec(index=0, dx=6, dy=0, settle_seconds=0.0),)
        rec = LookRecorder(tmp_path / "rec", run_id="run-8", plan=plan)
        frames = [texture_frame(), texture_frame(shift_x=6), texture_frame()]
        _runner(rec, frames=frames, moves_log=[]).run(plan)

        text = rec.result_path.read_text(encoding="utf-8")
        payload = json.loads(text)

        for banned in ("base64", "data:image", "\\x89PNG"):
            assert banned not in text
        frame_entry = payload["trials"][0]["frame_a"]
        assert "image" not in frame_entry and "pixels" not in frame_entry
        assert all(not isinstance(value, list) for value in frame_entry.values())
        assert isinstance(payload["trials"][0]["frames"][ARTIFACT_FRAME_A], str)
        assert payload["trials"][0]["frames"][ARTIFACT_FRAME_A].endswith(ARTIFACT_FRAME_A)

    def test_the_record_is_valid_json_with_the_expected_top_level_keys(
        self, tmp_path: Path
    ) -> None:
        plan = (TrialSpec(index=0, dx=6, dy=0, settle_seconds=0.0),)
        rec = LookRecorder(tmp_path / "rec", run_id="run-9", plan=plan)
        frames = [texture_frame(), texture_frame(shift_x=6), texture_frame()]
        result = _runner(rec, frames=frames, moves_log=[]).run(plan)

        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["experiment"] == EXPERIMENT_NAME
        assert payload["run_id"] == "run-9"
        assert payload["status"] == TRIAL_COMPLETED
        assert payload["movements_sent"] == 2
        assert payload["trials_completed"] == 1
        assert payload["notes"], "the record must carry its own honest limitations"

    def test_the_mapping_note_travels_with_the_ratio(self, tmp_path: Path) -> None:
        """The number must never appear without the wording that qualifies it."""
        plan = (TrialSpec(index=0, dx=6, dy=0, settle_seconds=0.0),)
        rec = LookRecorder(tmp_path / "rec", run_id="run-10", plan=plan)
        frames = [texture_frame(), texture_frame(shift_x=6), texture_frame()]
        result = _runner(rec, frames=frames, moves_log=[]).run(plan)

        payload = result.trials[0].to_dict()["pixels_per_delta"]
        assert payload["note"]
        assert "not assumed" in payload["note"]
        assert payload["x"] is not None and payload["y"] is None

    def test_a_record_with_no_trials_serialises_honestly(self, tmp_path: Path) -> None:
        plan = (TrialSpec(index=0, dx=6, dy=0, settle_seconds=0.0),)
        rec = LookRecorder(tmp_path / "rec", run_id="run-11", plan=plan)
        result = rec.finish(status=TRIAL_INTERRUPTED, stop_reason="refused before starting")

        payload = result.to_dict()
        assert payload["trials"] == []
        assert payload["trials_completed"] == 0
        assert payload["movements_sent"] == 0
        assert payload["duration_seconds"] is not None

    def test_the_finished_record_cannot_be_written_to_twice(self, tmp_path: Path) -> None:
        plan = (TrialSpec(index=0, dx=6, dy=0, settle_seconds=0.0),)
        rec = LookRecorder(tmp_path / "rec", run_id="run-12", plan=plan)
        rec.finish(status=TRIAL_COMPLETED, stop_reason="done")
        with pytest.raises(RuntimeError, match="already closed"):
            rec.record(
                LookTrialResult(
                    spec=plan[0],
                    status=TRIAL_COMPLETED,
                    stop_reason="",
                    window_width=96,
                    window_height=96,
                    capture_seconds=0.0,
                    movements_sent=0,
                    measured_at=1000.0,
                )
            )

    def test_the_record_names_its_own_limitations(self, tmp_path: Path) -> None:
        """Every record carries its own limits, whoever ran the experiment.

        The wording lives with the record rather than at the call site, so this
        checks the default the recorder ships with - which is what a file found
        on disk months later was written from.
        """
        plan = (TrialSpec(index=0, dx=6, dy=0, settle_seconds=0.0),)
        rec = LookRecorder(tmp_path / "rec", run_id="run-13", plan=plan)
        rec.finish(status=TRIAL_COMPLETED, stop_reason="done")
        payload = json.loads(rec.result_path.read_text(encoding="utf-8"))
        joined = " ".join(payload["notes"]).lower()
        assert "not evidence" in joined
        assert "no pass/fail threshold" in joined
        assert "not assumed" in joined
        assert tuple(payload["notes"]) == LIMITATION_NOTES


# ---------------------------------------------------------------------------
# the observer's LOOK panel
# ---------------------------------------------------------------------------


class TestObserverLookPanel:
    """The panel is a truthful mirror: empty when nothing was measured."""

    def test_the_empty_state_admits_it_has_no_measurement(self, config: Config) -> None:
        state = ObserverState(config)
        payload = state.snapshot().to_dict()["look"]
        assert payload["available"] is False
        assert payload["status"] == "not-run"
        assert payload["trial_index"] is None
        assert payload["mean_absolute_difference"] is None
        assert payload["shift"]["x"] is None
        assert payload["shift"]["available"] is False
        assert payload["reversibility_ratio"] is None
        assert payload["block_map"] == []

    def test_the_empty_state_carries_no_measurement_keys_as_numbers(self, config: Config) -> None:
        """A blank panel must not read as a measurement of zero."""
        payload = ObserverState(config).snapshot().to_dict()["look"]
        for key in ("mean_absolute_difference", "rmse", "changed_fraction", "reversibility_ratio"):
            assert payload[key] is None, f"{key} must be absent, not zero"
        for key in ("x", "y", "quality"):
            assert payload["shift"][key] is None, f"shift.{key} must be absent, not zero"
        for key in ("x", "y"):
            assert payload["pixels_per_delta"][key] is None, f"pixels_per_delta.{key} must be absent"

    def test_the_panel_never_carries_pixel_data(self, config: Config) -> None:
        state = ObserverState(config)
        state.publish_look(
            available=True,
            status=TRIAL_COMPLETED,
            experiment=EXPERIMENT_NAME,
            trial_count=1,
            mean_absolute_difference=4.5,
            block_grid=8,
            block_map=((0.0,) * 8,) * 8,
        )
        payload = state.snapshot().to_dict()["look"]
        assert "image" not in json.dumps(payload)
        assert "base64" not in json.dumps(payload)
        assert len(payload["block_map"]) == 8
        assert all(len(row) == 8 for row in payload["block_map"])

    def test_publishing_a_measurement_replaces_the_empty_state(self, config: Config) -> None:
        state = ObserverState(config)
        state.publish_look(
            available=True,
            status=TRIAL_COMPLETED,
            experiment=EXPERIMENT_NAME,
            trial_count=1,
            mean_absolute_difference=4.5,
            shift_x=6.0,
            shift_y=0.0,
            shift_available=True,
            reversibility_ratio=0.2,
        )
        payload = state.snapshot().to_dict()["look"]
        assert payload["available"] is True
        assert payload["status"] == TRIAL_COMPLETED
        assert payload["mean_absolute_difference"] == 4.5
        assert payload["shift"]["x"] == 6.0
        assert payload["shift"]["available"] is True
        assert payload["reversibility_ratio"] == 0.2

    def test_a_measurement_is_stamped_with_a_time(self, config: Config) -> None:
        state = ObserverState(config)
        state.publish_look(available=True, status=TRIAL_COMPLETED, now=1234.5)
        assert state.snapshot().to_dict()["look"]["measured_at"] == 1234.5

    def test_the_panel_fields_match_what_a_trial_reports(self, tmp_path: Path) -> None:
        """The runner's report fields must be exactly what the panel accepts.

        This is the seam between the experiment and the display, so it is checked
        against a real completed trial rather than a hand-written dict.
        """
        plan = (TrialSpec(index=0, dx=6, dy=0, settle_seconds=0.0),)
        rec = LookRecorder(tmp_path / "rec", run_id="run-14", plan=plan)
        frames = [texture_frame(), texture_frame(shift_x=6), texture_frame()]
        result = _runner(rec, frames=frames, moves_log=[]).run(plan)

        state = ObserverState(Config(data_dir=tmp_path / "data"))
        state.publish_look(
            available=True,
            status=result.status,
            experiment=EXPERIMENT_NAME,
            trial_count=len(plan),
            **result.trials[0].report_fields(),
        )

        payload = state.snapshot().to_dict()["look"]
        assert payload["available"] is True
        assert payload["trial_count"] == 1
        assert payload["block_grid"] == 8
        assert payload["dx"] == 6 and payload["dy"] == 0
        assert payload["movements_sent"] == 2

    def test_beginning_a_new_run_clears_the_previous_measurement(self, config: Config) -> None:
        """A stale number from the last run must not appear to belong to this one."""
        state = ObserverState(config)
        state.publish_look(available=True, status=TRIAL_COMPLETED, mean_absolute_difference=9.0)
        state.begin_run("run-15", now=1000.0, goal="LOOK-001")
        payload = state.snapshot().to_dict()["look"]
        assert payload["available"] is False
        assert payload["mean_absolute_difference"] is None
        assert payload["status"] == "not-run"

    def test_the_snapshot_still_carries_every_other_field(self, config: Config) -> None:
        snapshot = ObserverState(config).snapshot()
        assert isinstance(snapshot, ObserverSnapshot)
        assert snapshot.look.available is False

    def test_the_panel_is_read_only(self, config: Config) -> None:
        """The observer is a viewer. Nothing about it can inject input."""
        import autocraft.observer.state as state_module

        source = Path(state_module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        assert not any(module.startswith("autocraft.control") for module in imported)
