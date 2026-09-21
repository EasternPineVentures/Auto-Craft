"""Tests for VISION-001: the learned scene model.

What is pinned here, and why:

* **The partition agrees with the rest of the project.** A cell's brightness must
  mean the same thing to the perception layer as it does to
  :meth:`Frame.block_means`, or the two halves of AutoCraft would be looking at
  different pictures. ``test_cell_luma_matches_block_means`` holds them together.
* **The model cannot be asked for a verdict before it has seen anything.** Asking
  a model that has no basis for an allowance would silently report every frame as
  normal, which is the worst possible failure mode for a detector.
* **Adaptation moves the centre only for quiet cells.** This was a real bug found
  by measuring against real frames: holding back *both* centre and spread for
  flagged cells meant the cells that most needed to learn were exactly the ones
  forbidden from doing so. ``test_adaptation_leaves_flagged_cells_alone`` and
  ``test_adaptation_moves_quiet_cells`` pin both halves of the fix.
* **The package cannot inject input.** Pinned structurally, by reading the syntax
  tree, so it keeps holding for a contributor who never reads this file.
* **Nothing here produces a pass/fail judgement.** The project's rule is that
  AutoCraft reports what it saw and the owner decides what it means.

No test touches the screen, the mouse, or the clock. Every frame below is
synthetic, and the only real-world numbers quoted are in comments.
"""

from __future__ import annotations

import ast
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from autocraft.agent.action import Action, ActionKind
from autocraft.agent.decision import DecisionPolicy
from autocraft.agent.observation import Observation, WindowStatus
from autocraft.config import Config, ConfigError
from autocraft.perception import (
    CELL_FEATURE_NAMES,
    PerceptionDecisionPolicy,
    PerceptionReport,
    PerceptionSession,
    SceneScore,
    StabilityModel,
    cell_boxes,
    cell_features,
    cell_luma,
    describe,
    feature_matrix,
    features_of,
    iter_cells,
)
from autocraft.vision.frame import Frame

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def frame_from(image: np.ndarray, *, timestamp: float = 0.0) -> Frame:
    """Wrap an RGB array in a :class:`Frame`, which validates its own input."""
    return Frame(image=np.asarray(image, dtype=np.uint8), timestamp=timestamp)


def solid(height: int, width: int, level: int = 20) -> np.ndarray:
    """A uniform RGB image, so every cell has identical features."""
    return np.full((height, width, 3), level, dtype=np.uint8)


def textured(height: int, width: int, seed: int = 0, base: int = 40) -> np.ndarray:
    """A noisy image, so cells have a non-zero spread and edge energy."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 40, size=(height, width, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def features_of_frame(image: np.ndarray, grid: int) -> np.ndarray:
    return cell_features(np.asarray(image, dtype=np.uint8), grid)


def fitted_model(
    *, grid: int = 4, fit_frames: int = 3, sigma: float = 4.0, floor: float = 2.0, **kwargs
) -> StabilityModel:
    """A model fitted on ``fit_frames`` identical textured frames."""
    model = StabilityModel(grid=grid, fit_frames=fit_frames, sigma=sigma, floor=floor, **kwargs)
    image = textured(64, 64, seed=1)
    for _ in range(fit_frames):
        model.observe(features_of_frame(image, grid))
    assert model.is_fitted
    return model


def observation_for(image: np.ndarray, *, index: int = 0) -> Observation:
    """An observation carrying ``image``, with a foreground window."""
    return Observation(
        index=index,
        timestamp=0.0,
        window=WindowStatus(
            found=True,
            title="Test Window",
            is_foreground=True,
            minimized=False,
        ),
        frame=frame_from(image),
    )


# ---------------------------------------------------------------------------
# features: the partition, and its agreement with the rest of the project
# ---------------------------------------------------------------------------


def test_cell_boxes_partition_covers_every_pixel_exactly_once() -> None:
    """The grid must be a partition, not a sampling.

    The sizes are deliberately not multiples of the grid, so the remainder has to
    be placed somewhere. Losing those pixels would mean the perception layer and
    the rest of the project were reading slightly different pictures.
    """
    width, height, grid = 37, 23, 4
    boxes = cell_boxes(width, height, grid)
    assert len(boxes) == grid * grid

    coverage = np.zeros((height, width), dtype=np.int32)
    for x0, y0, x1, y1 in boxes:
        assert 0 <= x0 < x1 <= width
        assert 0 <= y0 < y1 <= height
        coverage[y0:y1, x0:x1] += 1

    assert coverage.min() == 1
    assert coverage.max() == 1


def test_cell_boxes_spreads_the_remainder_over_the_leading_bands() -> None:
    """10 pixels into 3 bands is 4, 3, 3 - the same split block_means makes."""
    boxes = cell_boxes(10, 10, 3)
    columns = [(x0, x1) for x0, y0, x1, y1 in boxes[:3]]
    assert columns == [(0, 4), (4, 7), (7, 10)]


def test_cell_boxes_rejects_an_impossible_grid() -> None:
    with pytest.raises(ValueError, match="grid must be at least 1"):
        cell_boxes(10, 10, 0)
    with pytest.raises(ValueError, match="positive size"):
        cell_boxes(0, 10, 2)


def test_iter_cells_yields_every_cell_row_major() -> None:
    image = textured(20, 30, seed=2)
    seen = [(row, column, patch.shape) for row, column, patch in iter_cells(image, 4)]
    assert len(seen) == 16
    assert [row for row, _, _ in seen] == sorted(row for row, _, _ in seen)
    assert seen[0][:2] == (0, 0)
    assert seen[-1][:2] == (3, 3)
    assert all(shape[2] == 3 for _, _, shape in seen)


def test_iter_cells_rejects_a_non_rgb_image() -> None:
    with pytest.raises(ValueError, match="H x W x 3"):
        list(iter_cells(np.zeros((4, 4), dtype=np.uint8), 2))


def test_cell_luma_matches_block_means_on_an_even_split() -> None:
    """The perception layer and the vision layer must not disagree.

    Both partition the frame the same way and both average the colour channels
    directly, so on a frame that divides evenly into cells the two numbers are
    identical. If this fails, one of them has quietly changed its definition of a
    cell's brightness.
    """
    image = textured(96, 128, seed=3)
    frame = frame_from(image)
    for grid in (2, 4, 8, 16):
        cube = cell_features(image, grid)
        assert np.allclose(cell_luma(cube), frame.block_means(grid), atol=1e-4), grid


def test_the_two_layers_split_an_uneven_frame_differently() -> None:
    """A documented difference, not a bug to paper over.

    ``Frame.block_means`` crops the frame to a whole number of bands, so the
    trailing strip is simply not measured. ``cell_boxes`` spreads the remainder
    over the leading bands instead, so every pixel contributes to exactly one
    cell. The perception layer wants the second behaviour - an animated strip at
    the edge of the screen has to be inside a cell, not cropped out of the
    measurement - which is why the two disagree on a frame that does not divide
    evenly.
    """
    image = textured(101, 137, seed=3)
    frame = frame_from(image)

    assert not np.allclose(
        cell_luma(cell_features(image, 4)), frame.block_means(4), atol=1e-4
    )

    covered = np.zeros((101, 137), dtype=np.int64)
    for x0, y0, x1, y1 in cell_boxes(137, 101, 4):
        covered[y0:y1, x0:x1] += 1
    assert covered.min() == 1
    assert covered.max() == 1


def test_cell_luma_and_feature_matrix_read_a_feature_cube() -> None:
    """Both helpers take the cube ``cell_features`` returns, not a raw image."""
    cube = cell_features(textured(32, 32, seed=8), 4)
    assert np.allclose(cell_luma(cube), cube[:, :, CELL_FEATURE_NAMES.index("mean_luma")])
    assert feature_matrix(cube).shape == (16, len(CELL_FEATURE_NAMES))

    with pytest.raises(ValueError, match="grid x grid x features"):
        cell_luma(cube[:, :, 0])
    with pytest.raises(ValueError, match="grid x grid x features"):
        feature_matrix(np.zeros((4, 4), dtype=np.float32))


def test_cell_features_has_the_documented_shape_and_dtype() -> None:
    features = cell_features(textured(40, 60, seed=4), 8)
    assert features.shape == (8, 8, len(CELL_FEATURE_NAMES))
    assert features.dtype == np.float32


def test_cell_features_describes_a_uniform_cell_as_flat() -> None:
    """A solid patch has no texture: zero spread, zero edges, one colour."""
    features = cell_features(solid(32, 32, level=17), 4)
    index = {name: position for position, name in enumerate(CELL_FEATURE_NAMES)}
    for row in range(4):
        for column in range(4):
            cell = features[row, column]
            assert cell[index["mean_r"]] == pytest.approx(17.0)
            assert cell[index["mean_g"]] == pytest.approx(17.0)
            assert cell[index["mean_b"]] == pytest.approx(17.0)
            assert cell[index["mean_luma"]] == pytest.approx(17.0)
            assert cell[index["std_luma"]] == pytest.approx(0.0)
            assert cell[index["edge_energy"]] == pytest.approx(0.0)
            assert cell[index["colour_spread"]] == pytest.approx(0.0)


def test_cell_features_reacts_to_texture() -> None:
    """A noisy patch must look different from a solid one of the same brightness."""
    index = {name: position for position, name in enumerate(CELL_FEATURE_NAMES)}
    flat = cell_features(solid(32, 32, level=40), 4)[0, 0]
    noisy = cell_features(textured(32, 32, seed=5, base=40), 4)[0, 0]
    assert noisy[index["std_luma"]] > flat[index["std_luma"]]
    assert noisy[index["edge_energy"]] > flat[index["edge_energy"]]


def test_cell_features_rejects_a_frame_too_small_for_the_grid() -> None:
    with pytest.raises(ValueError, match="cannot be split"):
        cell_features(solid(4, 40), 8)
    with pytest.raises(ValueError, match="H x W x 3"):
        cell_features(np.zeros((8, 8), dtype=np.uint8), 2)
    with pytest.raises(ValueError, match="grid must be at least 1"):
        cell_features(solid(8, 8), 0)


def test_features_of_and_feature_matrix_agree_with_cell_features() -> None:
    image = textured(48, 48, seed=6)
    direct = cell_features(image, 4)
    assert np.allclose(features_of(frame_from(image), 4), direct)
    flat = feature_matrix(direct)
    assert flat.shape == (16, len(CELL_FEATURE_NAMES))
    assert np.allclose(flat, direct.reshape(-1, len(CELL_FEATURE_NAMES)))


def test_describe_returns_a_named_value_per_feature() -> None:
    described = describe(cell_features(textured(32, 32, seed=7), 4))
    assert set(described) == set(CELL_FEATURE_NAMES)
    assert all(isinstance(value, float) for value in described.values())
    assert json.loads(json.dumps(described)) == described


# ---------------------------------------------------------------------------
# the model: fitting
# ---------------------------------------------------------------------------


def test_scoring_before_the_fit_is_refused() -> None:
    """A model with no basis for an allowance must refuse, not guess.

    Reporting every frame as normal because nothing has been learned is the worst
    failure mode a detector can have: it looks like success.
    """
    model = StabilityModel(grid=4, fit_frames=3)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.score(features_of_frame(textured(32, 32), 4))
    assert model.warmup_remaining == 3
    assert not model.is_fitted


def test_observe_returns_none_until_the_fitting_frame() -> None:
    """The first frame only sets a baseline; it has no predecessor to compare to."""
    model = StabilityModel(grid=4, fit_frames=3)
    features = features_of_frame(textured(32, 32, seed=8), 4)

    assert model.observe(features) is None  # baseline only
    assert model.observe(features) is not None
    assert model.observe(features) is not None  # the frame that completes the fit
    assert model.is_fitted
    assert model.warmup_remaining == 0
    assert model.frames_seen == 3

    # Once fitted the warm-up window is closed, so a long run cannot drift away
    # from the scene the model was fitted against.
    assert model.observe(features) is None
    assert model.frames_seen == 3


def test_a_static_scene_gets_the_floor_as_its_allowance() -> None:
    """Identical frames have no wobble, so the floor is the whole allowance."""
    model = fitted_model(grid=4, fit_frames=3, sigma=4.0, floor=2.0)
    allowance = model.allowance(features_of_frame(textured(64, 64, seed=1), 4))
    assert np.allclose(allowance, 2.0)
    assert model.floor_share() == pytest.approx(1.0)


def test_an_identical_frame_reports_nothing_changed() -> None:
    model = fitted_model()
    score = model.score(features_of_frame(textured(64, 64, seed=1), 4))
    assert score.changed_cells == 0
    assert score.total_excess == 0.0
    assert score.max_excess == 0.0
    assert score.cell_count == 16
    assert score.changed_fraction == 0.0
    assert score.peak_cell() is None


def test_a_brightened_cell_is_the_cell_that_is_reported() -> None:
    """Localisation: the flagged cell must be the cell that changed.

    The model is fitted on a uniform scene so the changed cell is unambiguous,
    then one cell is brightened. Exactly that cell should trip.
    """
    model = StabilityModel(grid=4, fit_frames=3, sigma=4.0, floor=2.0)
    image = solid(64, 64, level=20)
    for _ in range(3):
        model.observe(features_of_frame(image, 4))

    changed = image.copy()
    changed[32:48, 48:64] = 120  # row 2, column 3 of a 4x4 grid over 64x64
    score = model.score(features_of_frame(changed, 4))

    assert score.changed_cells == 1
    assert score.peak_cell() == (2, 3)
    assert score.max_excess > 0.0
    assert score.total_excess == pytest.approx(score.max_excess)


def test_the_score_serialises_without_pixel_data() -> None:
    model = fitted_model()
    score = model.score(features_of_frame(textured(64, 64, seed=1), 4))
    payload = score.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert "excess" not in payload
    assert "deviation" not in payload
    assert set(payload) == {
        "index",
        "grid",
        "changed_cells",
        "cell_count",
        "changed_fraction",
        "total_excess",
        "max_excess",
        "mean_deviation",
        "mean_allowance",
    }


def test_the_score_renders_as_ascii() -> None:
    model = fitted_model()
    score = model.score(features_of_frame(textured(64, 64, seed=1), 4))
    lines = score.excess_ascii()
    assert len(lines) == 4
    assert all(len(line) == 4 for line in lines)
    assert all(ord(character) < 128 for line in lines for character in line)
    assert len(score.deviation_ascii()) == 4


def test_the_model_rejects_a_feature_grid_of_the_wrong_shape() -> None:
    model = fitted_model(grid=4)
    with pytest.raises(ValueError, match="expected a 4x4 feature grid"):
        model.score(cell_features(textured(64, 64), 8))
    with pytest.raises(ValueError, match="grid x grid x features"):
        model.score(np.zeros((4, 4), dtype=np.float32))


def test_the_model_validates_its_construction() -> None:
    with pytest.raises(ValueError, match="grid must be at least 1"):
        StabilityModel(grid=0)
    with pytest.raises(ValueError, match="fit_frames must be at least 2"):
        StabilityModel(fit_frames=1)
    with pytest.raises(ValueError, match="sigma must be positive"):
        StabilityModel(sigma=0.0)
    with pytest.raises(ValueError, match="floor must be non-negative"):
        StabilityModel(floor=-1.0)
    with pytest.raises(ValueError, match="adapt_rate must be between 0 and 1"):
        StabilityModel(adapt_rate=1.5)


# ---------------------------------------------------------------------------
# the model: persistence and reset
# ---------------------------------------------------------------------------


def test_saving_an_unfitted_model_is_refused() -> None:
    with pytest.raises(RuntimeError, match="has not been fitted"):
        StabilityModel(grid=4).to_dict()


def test_a_saved_model_reloads_to_the_same_scores(tmp_path: Path) -> None:
    model = fitted_model(grid=4, fit_frames=4)
    probe = features_of_frame(textured(64, 64, seed=9), 4)
    before = model.score(probe).to_dict()

    path = model.save(tmp_path / "models" / "scene.json")
    assert path.exists()
    assert path.parent.name == "models"

    reloaded = StabilityModel.load(path)
    assert reloaded.is_fitted
    assert reloaded.grid == model.grid
    assert reloaded.fit_frames == model.fit_frames
    assert reloaded.score(probe).to_dict() == before


def test_a_saved_model_is_json_and_carries_no_pixels(tmp_path: Path) -> None:
    model = fitted_model(grid=4, fit_frames=4)
    path = model.save(tmp_path / "scene.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["kind"] == "autocraft.perception.StabilityModel"
    assert payload["feature_names"] == list(CELL_FEATURE_NAMES)
    assert len(payload["centre"]) == 4
    assert len(payload["centre"][0]) == 4


def test_reset_returns_the_model_to_an_unfitted_state() -> None:
    model = fitted_model(grid=4, fit_frames=3)
    model.reset()
    assert not model.is_fitted
    assert model.frames_seen == 0
    assert model.warmup_remaining == 3
    with pytest.raises(RuntimeError, match="not fitted"):
        model.score(features_of_frame(textured(64, 64), 4))
    # Configuration survives, so the model is usable again immediately.
    assert model.grid == 4
    assert model.fit_frames == 3


def test_the_summary_reports_the_fit_and_the_floor_share() -> None:
    model = fitted_model(grid=4, fit_frames=4)
    summary = model.summary()
    assert json.loads(json.dumps(summary)) == summary
    assert summary["fitted"] is True
    assert summary["frames_seen"] == 4
    assert summary["fit"]["cells"] == 16
    assert summary["fit"]["frames"] == 3
    assert 0.0 <= summary["floor_share"] <= 1.0


# ---------------------------------------------------------------------------
# the model: adaptation
# ---------------------------------------------------------------------------


def test_adaptation_moves_the_centre_of_a_quiet_cell() -> None:
    """A scene that drifts must not be reported as changing forever.

    Measured on the real 60-frame capture: without adaptation the model reported
    ~20 changed cells on every frame from frame 28 onward, because the scene's
    ambient brightness had drifted and a frozen model never forgets that.
    """
    model = StabilityModel(grid=4, fit_frames=3, floor=2.0, adapt_rate=0.5)
    dim = solid(64, 64, level=20)
    for _ in range(3):
        model.observe(features_of_frame(dim, 4))

    brighter = solid(64, 64, level=40)
    features = features_of_frame(brighter, 4)

    first = model.score(features)
    assert first.changed_cells == 16  # a 20-level jump trips every cell

    for _ in range(10):
        model.update(features)

    after = model.score(features)
    assert after.changed_cells == 0
    assert after.total_excess == 0.0


def test_adaptation_leaves_flagged_cells_alone() -> None:
    """The centre of a cell that is *currently* surprised must not move.

    This is the half of the fix that matters: if a flagged cell's centre chased
    the surprise, a real change would be absorbed within a few frames and the
    model would go quiet exactly when something was happening.
    """
    model = StabilityModel(grid=4, fit_frames=3, floor=2.0, adapt_rate=0.5)
    for _ in range(3):
        model.observe(features_of_frame(solid(64, 64, level=20), 4))

    changed = solid(64, 64, level=20)
    changed[0:16, 0:16] = 200  # row 0, column 0
    features = features_of_frame(changed, 4)

    before = model.score(features)
    assert before.peak_cell() == (0, 0)
    centre_before = model.centre[0, 0]

    # Fold that same verdict in repeatedly, so the cell is flagged every time. Re
    # scoring each iteration would not test this rule: the widening spread
    # eventually makes the cell look ordinary, and the centre is then free to
    # move - which is the correct behaviour, but a different one.
    for _ in range(5):
        model.update(features, score=before)

    assert model.centre[0, 0] == pytest.approx(centre_before)
    assert model.score(features).peak_cell() == (0, 0)


def test_adaptation_learns_the_spread_of_a_flagged_cell() -> None:
    """The spread *is* allowed to move, even for a cell the model just flagged.

    Holding the spread back too was the original bug: the cells that needed to
    learn were exactly the ones forbidden from doing so, and adaptation measured
    as having essentially no effect.
    """
    model = StabilityModel(grid=4, fit_frames=3, floor=2.0, adapt_rate=0.5)
    for _ in range(3):
        model.observe(features_of_frame(solid(64, 64, level=20), 4))

    changed = solid(64, 64, level=20)
    changed[0:16, 0:16] = 200
    features = features_of_frame(changed, 4)

    flagged = model.score(features)
    assert flagged.peak_cell() == (0, 0)
    spread_before = model.spread[0, 0]

    for _ in range(5):
        model.update(features, score=flagged)

    assert model.spread[0, 0] > spread_before
    # But only by ``adapt_rate`` of the event each time, so a single surprise
    # cannot desensitise the cell: five folds of a 180-level jump leave the spread
    # an order of magnitude below the jump itself.
    assert model.spread[0, 0] < 20.0


def test_adaptation_is_off_when_the_rate_is_zero() -> None:
    model = StabilityModel(grid=4, fit_frames=3, floor=2.0, adapt_rate=0.0)
    for _ in range(3):
        model.observe(features_of_frame(solid(64, 64, level=20), 4))
    centre = model.centre.copy()
    spread = model.spread.copy()
    for _ in range(5):
        model.update(features_of_frame(solid(64, 64, level=40), 4))
    assert np.array_equal(model.centre, centre)
    assert np.array_equal(model.spread, spread)


# ---------------------------------------------------------------------------
# the session: driving a frame source
# ---------------------------------------------------------------------------


def test_a_session_needs_a_bound() -> None:
    """An experiment that cannot say when it stops cannot be reviewed after."""
    session = PerceptionSession(StabilityModel(grid=4, fit_frames=2), lambda: None)
    with pytest.raises(ValueError, match="at least one of seconds or steps"):
        session.run()


def test_a_session_rejects_a_non_positive_bound() -> None:
    session = PerceptionSession(StabilityModel(grid=4, fit_frames=2), lambda: None)
    with pytest.raises(ValueError, match="seconds must be positive"):
        session.run(seconds=0)
    with pytest.raises(ValueError, match="steps must be positive"):
        session.run(steps=0)


def test_a_step_bounded_session_fits_then_scores() -> None:
    model = StabilityModel(grid=4, fit_frames=3, floor=2.0)
    image = textured(64, 64, seed=10)
    frames = itertools.repeat(frame_from(image))
    session = PerceptionSession(model, lambda: next(frames))

    report = session.run(steps=6)

    assert report.warmup_frames == 3
    assert report.steady_frames == 3
    assert report.capture_failures == 0
    assert report.is_complete
    assert all(row.score is None for row in report.rows[:3])
    assert all(row.score is not None for row in report.rows[3:])
    assert report.stop_reason == "reached the 6-step limit"
    assert report.changed_counts() == [0, 0, 0]


def test_a_session_counts_capture_failures_without_crashing() -> None:
    model = StabilityModel(grid=4, fit_frames=2, floor=2.0)
    image = textured(64, 64, seed=11)
    script = [None, frame_from(image), None, frame_from(image), frame_from(image)]
    frames = iter(script)
    session = PerceptionSession(model, lambda: next(frames, None))

    report = session.run(steps=5)
    assert report.capture_failures == 2
    assert len(report.rows) == 3
    assert report.is_complete


def test_a_session_that_ends_during_warm_up_reports_no_scores() -> None:
    """A half-fitted model's output must not be presented as a measurement."""
    model = StabilityModel(grid=4, fit_frames=10, floor=2.0)
    image = textured(64, 64, seed=12)
    frames = itertools.repeat(frame_from(image))
    report = PerceptionSession(model, lambda: next(frames)).run(steps=4)

    assert not report.is_complete
    assert report.steady_frames == 0
    assert report.scores == []
    assert "complete" in report.summary()
    assert report.summary()["complete"] is False


def test_a_session_honours_the_step_limit_over_the_clock() -> None:
    model = StabilityModel(grid=4, fit_frames=2, floor=2.0)
    image = textured(64, 64, seed=13)
    frames = itertools.repeat(frame_from(image))
    report = PerceptionSession(model, lambda: next(frames)).run(seconds=3600, steps=5)
    assert len(report.rows) == 5
    assert "5-step limit" in report.stop_reason


def test_replaying_recorded_frames_needs_no_clock() -> None:
    model = StabilityModel(grid=4, fit_frames=3, floor=2.0)
    image = textured(64, 64, seed=14)
    session = PerceptionSession(model, lambda: None)
    report = session.frames(frame_from(image) for _ in range(7))

    assert report.warmup_frames == 3
    assert report.steady_frames == 4
    assert report.stop_reason == "replayed 7 recorded frame(s)"
    assert model.frames_seen == 3  # only warm-up is fed to the model


def test_the_report_serialises_and_accumulates_an_excess_map() -> None:
    model = StabilityModel(grid=4, fit_frames=2, floor=2.0)
    image = textured(64, 64, seed=15)
    session = PerceptionSession(model, lambda: None)
    report = session.frames(frame_from(image) for _ in range(4))

    payload = report.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["summary"]["steady_frames"] == 2
    assert len(payload["timeline"]) == 4

    accumulated = report.accumulated_excess()
    assert len(accumulated) == 4
    assert all(len(row) == 4 for row in accumulated)
    assert all(value == 0.0 for row in accumulated for value in row)


def test_the_accumulated_map_sums_what_each_frame_overshot() -> None:
    """The accumulated map is the answer to "where did anything happen"."""
    model = StabilityModel(grid=4, fit_frames=2, floor=2.0)
    quiet = solid(64, 64, level=20)
    for _ in range(2):
        model.observe(features_of_frame(quiet, 4))

    changed = quiet.copy()
    changed[0:16, 0:16] = 120
    session = PerceptionSession(model, lambda: None, adapt=False)
    report = session.frames(
        frame_from(changed) for _ in range(3)
    )
    accumulated = np.asarray(report.accumulated_excess())
    assert accumulated[0, 0] > 0.0
    assert accumulated[0, 0] == pytest.approx(3 * report.scores[0].max_excess, rel=1e-6)
    assert accumulated.sum() == pytest.approx(accumulated[0, 0], rel=1e-6)


def test_a_session_with_adaptation_off_leaves_the_model_frozen() -> None:
    model = StabilityModel(grid=4, fit_frames=2, floor=2.0, adapt_rate=0.9)
    quiet = solid(64, 64, level=20)
    for _ in range(2):
        model.observe(features_of_frame(quiet, 4))
    centre = model.centre.copy()

    brighter = solid(64, 64, level=90)
    session = PerceptionSession(model, lambda: None, adapt=False)
    report = session.frames(frame_from(brighter) for _ in range(5))

    assert np.array_equal(model.centre, centre)
    assert report.changed_counts() == [16] * 5


def test_a_session_stops_at_the_absolute_ceiling() -> None:
    """The ceiling covers capture time, so a slow source cannot overrun it."""
    ticks = itertools.count(0.0, 5.0)
    model = StabilityModel(grid=4, fit_frames=2, floor=2.0)
    image = textured(64, 64, seed=16)
    frames = itertools.repeat(frame_from(image))
    session = PerceptionSession(model, lambda: next(frames), clock=lambda: next(ticks))
    report = session.run(seconds=1000.0, max_seconds=12.0)
    assert "ceiling" in report.stop_reason


# ---------------------------------------------------------------------------
# the policy: the AI seam
# ---------------------------------------------------------------------------


def test_the_policy_satisfies_the_decision_policy_protocol() -> None:
    policy = PerceptionDecisionPolicy(model=StabilityModel(grid=4, fit_frames=2))
    assert isinstance(policy, DecisionPolicy)
    assert isinstance(policy.name, str) and policy.name


def test_the_policy_does_nothing_before_the_model_is_fitted() -> None:
    """Moving before you can see is noise with a mouse attached."""
    policy = PerceptionDecisionPolicy(model=StabilityModel(grid=4, fit_frames=3), max_mouse_delta=20)
    image = textured(64, 64, seed=17)
    for index in range(3):
        action = policy.decide(observation_for(image, index=index))
        assert action.kind is ActionKind.NOOP
    assert policy.moves == 0
    assert policy.last_score is None


def test_the_policy_does_nothing_when_it_cannot_see() -> None:
    policy = PerceptionDecisionPolicy(model=fitted_model(grid=4), max_mouse_delta=20)
    blind = Observation(
        index=0,
        timestamp=0.0,
        window=WindowStatus(found=True, title="Test Window", is_foreground=True),
        frame=None,
    )
    assert policy.decide(blind).kind is ActionKind.NOOP


def test_the_policy_does_nothing_when_nothing_changed() -> None:
    model = fitted_model(grid=4, fit_frames=3)
    policy = PerceptionDecisionPolicy(model=model, max_mouse_delta=20)
    image = textured(64, 64, seed=1)  # exactly what the model was fitted on
    action = policy.decide(observation_for(image))
    assert action.kind is ActionKind.NOOP
    assert policy.moves == 0


def test_the_policy_moves_toward_a_change_and_keeps_the_delta_integer() -> None:
    """A change on the right must produce a rightward move, clamped and whole."""
    model = StabilityModel(grid=4, fit_frames=3, floor=2.0)
    quiet = solid(64, 64, level=20)
    for _ in range(3):
        model.observe(features_of_frame(quiet, 4))

    changed = quiet.copy()
    changed[32:48, 48:64] = 200  # row 2, column 3: right-hand side, below centre
    policy = PerceptionDecisionPolicy(model=model, max_mouse_delta=20, gain=1.0, adapt=False)

    action = policy.decide(observation_for(changed))

    assert action.kind is ActionKind.MOUSE_MOVE
    dx = action.parameters["dx"]
    dy = action.parameters["dy"]
    assert isinstance(dx, int) and isinstance(dy, int)
    assert dx > 0, "a change on the right should move the view right"
    assert abs(dx) <= 20 and abs(dy) <= 20
    assert policy.moves == 1
    assert policy.last_target == (2, 3)


def test_the_policy_moves_left_for_a_change_on_the_left() -> None:
    model = StabilityModel(grid=4, fit_frames=3, floor=2.0)
    quiet = solid(64, 64, level=20)
    for _ in range(3):
        model.observe(features_of_frame(quiet, 4))

    changed = quiet.copy()
    changed[0:16, 0:16] = 200  # row 0, column 0
    policy = PerceptionDecisionPolicy(model=model, max_mouse_delta=20, gain=1.0, adapt=False)
    action = policy.decide(observation_for(changed))
    assert action.parameters["dx"] < 0
    assert action.parameters["dy"] < 0


def test_the_policy_never_exceeds_the_configured_delta() -> None:
    model = StabilityModel(grid=16, fit_frames=3, floor=2.0)
    quiet = solid(128, 128, level=20)
    for _ in range(3):
        model.observe(features_of_frame(quiet, 16))

    changed = quiet.copy()
    changed[0:8, 120:128] = 200
    policy = PerceptionDecisionPolicy(model=model, max_mouse_delta=7, gain=1.0, adapt=False)
    action = policy.decide(observation_for(changed))
    assert action.kind is ActionKind.MOUSE_MOVE
    assert abs(action.parameters["dx"]) <= 7
    assert abs(action.parameters["dy"]) <= 7


def test_the_policy_nudges_rather_than_stalling_one_step_short() -> None:
    """A far-off cell that rounds to nothing must still produce a nudge.

    With a timid gain every axis rounds to zero, and a policy that returns a
    zero delta is indistinguishable from one that saw nothing. The nudge goes
    along whichever axis is furthest off, and only that axis.
    """
    model = StabilityModel(grid=64, fit_frames=3, floor=2.0)
    quiet = solid(512, 512, level=20)
    for _ in range(3):
        model.observe(features_of_frame(quiet, 64))

    # Cell (0, 0): the horizontal and vertical offsets tie, so x wins.
    corner = quiet.copy()
    corner[0:8, 0:8] = 200
    policy = PerceptionDecisionPolicy(model=model, max_mouse_delta=1, gain=0.01, adapt=False)
    action = policy.decide(observation_for(corner))
    assert action.kind is ActionKind.MOUSE_MOVE
    assert (action.parameters["dx"], action.parameters["dy"]) == (-1, 0)
    assert policy.moves == 1

    # Cell (0, 31) sits on the centre column, so the vertical offset dominates.
    high = quiet.copy()
    high[0:8, 248:256] = 200
    policy = PerceptionDecisionPolicy(model=model, max_mouse_delta=1, gain=0.01, adapt=False)
    action = policy.decide(observation_for(high))
    assert action.kind is ActionKind.MOUSE_MOVE
    assert (action.parameters["dx"], action.parameters["dy"]) == (0, -1)


def test_the_policy_resets_the_model_it_learned() -> None:
    """A policy whose knowledge is the model has not reset if the model persists."""
    model = fitted_model(grid=4, fit_frames=3)
    policy = PerceptionDecisionPolicy(model=model, max_mouse_delta=20)
    policy.decide(observation_for(textured(64, 64, seed=1)))
    assert model.is_fitted

    policy.reset()
    assert not model.is_fitted
    assert policy.decisions == 0
    assert policy.moves == 0
    assert policy.last_score is None


def test_the_policy_clamps_its_own_parameters() -> None:
    policy = PerceptionDecisionPolicy(
        model=StabilityModel(grid=4, fit_frames=2), max_mouse_delta=-5, gain=9.0
    )
    assert policy.max_mouse_delta == 0
    assert policy.gain == 1.0
    # A zero delta means it can never move, however surprised it is.
    assert policy.decide(observation_for(textured(64, 64, seed=18))).kind is ActionKind.NOOP


def test_the_policy_summary_is_json_serialisable() -> None:
    model = StabilityModel(grid=4, fit_frames=2, floor=2.0)
    quiet = solid(64, 64, level=20)
    for _ in range(2):
        model.observe(features_of_frame(quiet, 4))
    changed = quiet.copy()
    changed[0:16, 0:16] = 200

    policy = PerceptionDecisionPolicy(model=model, max_mouse_delta=20, adapt=False)
    policy.decide(observation_for(changed))
    payload = policy.summary()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["moves"] == 1
    assert payload["last_target"] == [0, 0]


# ---------------------------------------------------------------------------
# config: the settings are wired into every place that needs them
# ---------------------------------------------------------------------------


def test_the_perception_settings_have_defaults() -> None:
    config = Config()
    assert config.perception_grid >= 1
    assert config.perception_sigma > 0
    assert config.perception_floor >= 0
    assert config.perception_fit_frames >= 2
    assert 0.0 <= config.perception_adapt_rate <= 1.0


def test_the_perception_settings_are_exported_and_validated() -> None:
    config = Config()
    payload = config.to_dict()
    for key in (
        "perception_grid",
        "perception_sigma",
        "perception_floor",
        "perception_fit_frames",
        "perception_adapt_rate",
    ):
        assert key in payload, f"{key} is missing from Config.to_dict()"

    for kwargs, message in (
        ({"perception_grid": 0}, "perception_grid"),
        ({"perception_grid": 999}, "perception_grid"),
        ({"perception_sigma": 0.0}, "perception_sigma"),
        ({"perception_floor": -1.0}, "perception_floor"),
        ({"perception_fit_frames": 1}, "perception_fit_frames"),
        ({"perception_adapt_rate": 2.0}, "perception_adapt_rate"),
    ):
        with pytest.raises(ConfigError, match=message):
            Config(**kwargs)


def test_the_perception_settings_round_trip_through_toml(tmp_path: Path) -> None:
    """An unknown TOML section is rejected, so the example file must stay in step."""
    from autocraft.config import load_config

    path = tmp_path / "autocraft.toml"
    path.write_text(
        "[perception]\n"
        "grid = 8\n"
        "sigma = 3.0\n"
        "floor = 1.5\n"
        "fit_frames = 5\n"
        "adapt_rate = 0.25\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.perception_grid == 8
    assert config.perception_sigma == pytest.approx(3.0)
    assert config.perception_floor == pytest.approx(1.5)
    assert config.perception_fit_frames == 5
    assert config.perception_adapt_rate == pytest.approx(0.25)


def test_an_unknown_perception_key_is_rejected(tmp_path: Path) -> None:
    from autocraft.config import load_config

    path = tmp_path / "autocraft.toml"
    path.write_text("[perception]\nnonsense = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="nonsense"):
        load_config(path)


def test_the_models_directory_sits_under_the_data_directory() -> None:
    config = Config()
    assert config.models_dir == config.data_dir / "models"


# ---------------------------------------------------------------------------
# structural: what the package is forbidden from becoming
# ---------------------------------------------------------------------------

#: The perception package's source directory, read directly so these tests need
#: no import of the code they are checking.
_PERCEPTION_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "autocraft" / "perception"

#: Modules that would let the package touch the machine. ``autocraft.control`` is
#: the input layer; the rest are the usual ways around an import ban.
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

#: The heavyweight learned models the project's DO-NOT-BUILD list forbids. Checked
#: as *imports*, because a substring search would flag ``numpy.clip`` and a
#: docstring that names the ban.
#:
#: The green light the owner gave lifted the ban on model *APIs*; it did not turn
#: this layer into a neural network. VISION-001 is a per-cell statistic fitted
#: online from the run's own frames with numpy, and this test is what keeps it
#: that way.
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

#: Vocabulary a pass/fail verdict would need.
_VERDICT_TOKENS = ("pass", "fail", "verdict", "good", "bad", "acceptable", "grade")

#: Names that carry that vocabulary without being a verdict.
#:
#: ``SceneScore`` is a *measurement*: the per-frame overshoot of each cell against
#: its learned allowance. "Score" here means "how far past the allowance did this
#: cell go", the same sense in which LOOK-001's ``DEFAULT_CHANGED_THRESHOLD`` is a
#: measurement cutoff rather than a judgement. Nothing in the package compares it
#: to a bound or calls a result acceptable.
_NOT_A_VERDICT = frozenset({"SceneScore"})

#: The only names the perception layer may take from the action module. It needs
#: ``Action`` to express what the policy decided, and it must never reach for
#: ``ActionExecutor``, which is the thing that actually injects.
_ALLOWED_ACTION_IMPORTS = frozenset({"Action", "ActionKind"})


def _perception_sources() -> list[Path]:
    sources = sorted(_PERCEPTION_PACKAGE.glob("*.py"))
    assert sources, "the perception package should have modules to check"
    return sources


#: Where the package lives, as a dotted path. Relative imports are resolved
#: against this.
_PACKAGE_DOTTED = "autocraft.perception"


def _absolute_module(node: ast.ImportFrom) -> str:
    """Resolve an ``ImportFrom`` to a fully dotted module path.

    The perception modules import their neighbours *relatively*
    (``from ..agent.action import Action``), so ``node.module`` alone is
    ``"agent.action"``. Comparing that against ``"autocraft.agent.action"`` never
    matches, which would make the forbidden-import checks below pass forever
    without ever having checked anything - the one failure mode a structural test
    must not have. ``node.level`` is how many packages to walk up: 1 is the
    package the file itself lives in, 2 its parent, and so on.
    """
    if not node.level:
        return node.module or ""
    parts = _PACKAGE_DOTTED.split(".")
    prefix = ".".join(parts[: len(parts) - (node.level - 1)])
    return f"{prefix}.{node.module}" if node.module else prefix


def _imports(source: Path) -> list[tuple[str, list[str]]]:
    """Every import in ``source`` as ``(absolute module, [names])``."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    found: list[tuple[str, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, []) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((_absolute_module(node), [alias.name for alias in node.names]))
    return found


def test_relative_imports_are_resolved_to_absolute_paths() -> None:
    """The checks below are only meaningful if they can see the imports.

    ``policy.py`` takes ``Action`` from the action module with a relative import.
    If resolution were broken the module would appear as ``"agent.action"``, every
    comparison would be against the wrong string, and the two tests that follow
    would pass while checking nothing at all.
    """
    policy = _imports(_PERCEPTION_PACKAGE / "policy.py")
    assert ("autocraft.agent.action", ["Action"]) in policy

    features = _imports(_PERCEPTION_PACKAGE / "features.py")
    assert ("autocraft.vision.frame", ["Frame"]) in features

    stability = _imports(_PERCEPTION_PACKAGE / "stability.py")
    assert ("autocraft.perception.features", ["CELL_FEATURE_NAMES", "cell_luma"]) in stability


def test_no_perception_module_can_inject_input() -> None:
    """The package reads pixels. It must have no way to touch the game.

    Read from the syntax tree rather than searched for in the text, so a docstring
    that *names* the control layer is not mistaken for an import of it.
    """
    for source in _perception_sources():
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


def test_the_perception_package_imports_no_trained_model() -> None:
    """No framework, no pretrained weights: the model is fitted from the run itself."""
    for source in _perception_sources():
        for module, _ in _imports(source):
            for forbidden in _FORBIDDEN_MODEL_IMPORTS:
                assert module != forbidden and not module.startswith(f"{forbidden}."), (
                    f"{source.name} imports the model framework {module!r}"
                )


def test_the_perception_package_opens_no_network_or_subprocess() -> None:
    """The ban on model APIs means no call out of the process either."""
    for source in _perception_sources():
        text = source.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                assert name not in {"urlopen", "Popen", "system", "run", "connect"}, (
                    f"{source.name} calls {name}()"
                )


def test_the_perception_package_exposes_no_pass_fail_verdict() -> None:
    """VISION-001 reports numbers; it does not decide whether a frame is good.

    What is checked is that no callable on the public surface turns a measurement
    into a judgement, so adding an ``is_acceptable()`` helper later fails here
    rather than quietly reintroducing a verdict.
    """
    import autocraft.perception as perception

    names = list(perception.__all__)
    assert names, "the package should declare its public surface"

    judged = [
        name
        for name in names
        if callable(getattr(perception, name))
        and name not in _NOT_A_VERDICT
        and any(token in name.lower() for token in _VERDICT_TOKENS)
    ]
    assert judged == [], f"the perception package exports verdict-shaped callables: {judged}"


def test_the_public_surface_is_declared_explicitly() -> None:
    """``__all__`` must name everything exported, so the surface is reviewable."""
    import autocraft.perception as perception

    for name in perception.__all__:
        assert hasattr(perception, name), f"{name} is in __all__ but not importable"

    assert "StabilityModel" in perception.__all__
    assert "PerceptionDecisionPolicy" in perception.__all__
    assert "PerceptionSession" in perception.__all__
