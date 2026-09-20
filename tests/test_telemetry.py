"""Telemetry: run identity, step streaming, serialisation.

The hard rule under test here is that telemetry never embeds raw pixels: frames
are referenced by path so a run log stays readable and diffable.
"""

from __future__ import annotations

import json

import pytest

from autocraft.telemetry.recorder import RunRecord, RunRecorder, new_run_id


class TestRunIdentity:
    """A run ID must be unique, sortable and filesystem-safe."""

    def test_format_is_timestamp_plus_suffix(self) -> None:
        run_id = new_run_id(1_700_000_000.0)
        stamp, _, suffix = run_id.partition("-")
        assert len(stamp) == 16 and stamp.endswith("Z")
        assert len(suffix) == 8

    def test_ids_are_unique(self) -> None:
        assert len({new_run_id() for _ in range(200)}) == 200

    def test_ids_are_filesystem_safe(self) -> None:
        run_id = new_run_id()
        assert all(char.isalnum() or char in "-_" for char in run_id)

    def test_ids_sort_chronologically(self) -> None:
        early = new_run_id(1_700_000_000.0)
        late = new_run_id(1_700_000_600.0)
        assert early[:16] < late[:16]


class TestRunRecorder:
    """Recording must stream, and must never hold pixels in memory."""

    def test_new_run_creates_its_own_directory(self, tmp_path) -> None:
        first = RunRecorder.new_run(tmp_path)
        second = RunRecorder.new_run(tmp_path)
        assert first.directory != second.directory
        assert first.directory.parent == tmp_path
        assert first.directory.is_dir()
        first.close()
        second.close()

    def test_new_run_id_can_be_supplied(self, tmp_path) -> None:
        recorder = RunRecorder.new_run(tmp_path, run_id="fixed-run-id")
        assert recorder.run_id == "fixed-run-id"
        assert recorder.directory.name == "fixed-run-id"
        recorder.close()

    def test_steps_are_streamed_to_disk(self, tmp_path) -> None:
        recorder = RunRecorder(tmp_path)
        for index in range(5):
            recorder.record_step({"index": index})
        # Readable before finish(): the run survives a crash.
        lines = recorder.steps_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5
        assert json.loads(lines[-1])["index"] == 4
        recorder.close()

    def test_finish_writes_the_run_record(self, tmp_path) -> None:
        recorder = RunRecorder(tmp_path)
        recorder.record_step({"index": 0})
        record = recorder.finish(status="finished", stop_reason="max_steps")
        assert record.step_count == 1
        assert record.status == "finished"
        assert recorder.record_path.is_file()
        payload = json.loads(recorder.record_path.read_text(encoding="utf-8"))
        assert payload["stop_reason"] == "max_steps"
        assert payload["step_count"] == 1

    def test_finish_is_idempotent(self, tmp_path) -> None:
        """The loop finishes from a ``finally``, so a second call must be a no-op.

        Letting the second call win would let a shutdown path overwrite the real
        outcome of the run.
        """
        recorder = RunRecorder(tmp_path)
        recorder.record_step({"index": 0})
        first = recorder.finish(status="finished", stop_reason="once")
        second = recorder.finish(status="error", stop_reason="twice")
        assert second.stop_reason == first.stop_reason
        assert second.status == "finished"
        assert recorder.closed is True
        payload = json.loads(recorder.record_path.read_text(encoding="utf-8"))
        assert payload["stop_reason"] == "once"

    def test_recording_after_finish_raises(self, tmp_path) -> None:
        """A closed run must not accept steps that run.json could never show."""
        recorder = RunRecorder(tmp_path)
        recorder.finish(status="finished", stop_reason="done")
        with pytest.raises(RuntimeError):
            recorder.record_step({"index": 0})
        with pytest.raises(RuntimeError):
            recorder.record_error("too late")
        with pytest.raises(RuntimeError):
            recorder.record_safety_event({"kind": "late"})

    def test_safety_events_are_recorded(self, tmp_path) -> None:
        from autocraft.control.safety import SafetyEvent

        recorder = RunRecorder(tmp_path)
        recorder.record_safety_event(SafetyEvent(timestamp=1.0, kind="focus_lost", detail="not foreground"))
        recorder.record_safety_event({"kind": "release", "detail": "released w"})
        record = recorder.finish(status="finished", stop_reason="done")
        assert len(record.safety_events) == 2
        assert record.safety_events[0]["kind"] == "focus_lost"

    def test_errors_are_recorded(self, tmp_path) -> None:
        recorder = RunRecorder(tmp_path)
        recorder.record_error("capture exploded", context="observe")
        record = recorder.finish(status="error", stop_reason="error")
        assert record.errors
        assert record.errors[0]["message"] == "capture exploded"

    def test_frame_path_lives_under_the_run(self, tmp_path) -> None:
        """Frames land in the run's own directory, not in a global captures pile."""
        recorder = RunRecorder(tmp_path)
        assert recorder.frames_dir.is_dir() is False
        path = recorder.frame_path("frame-000001.png")
        assert path.name == "frame-000001.png"
        assert path.parent == recorder.frames_dir
        assert recorder.directory in path.parents
        assert recorder.frames_dir.is_dir() is True
        recorder.close()

    def test_frames_dir_can_be_redirected(self, tmp_path) -> None:
        """A caller may point frames elsewhere; run.json must still be consistent."""
        recorder = RunRecorder(tmp_path, frames_dir=tmp_path / "elsewhere")
        assert recorder.frame_path("frame-000000.png").parent == tmp_path / "elsewhere"
        recorder.close()

    def test_context_manager_finishes_the_run(self, tmp_path) -> None:
        with RunRecorder(tmp_path) as recorder:
            recorder.record_step({"index": 0})
        assert recorder.closed is True
        assert recorder.record_path.is_file()

    def test_steps_property_mirrors_disk(self, tmp_path) -> None:
        recorder = RunRecorder(tmp_path)
        recorder.record_step({"index": 0})
        recorder.record_step({"index": 1})
        assert [step["index"] for step in recorder.steps] == [0, 1]
        recorder.close()

    def test_config_snapshot_is_persisted(self, tmp_path) -> None:
        from autocraft.config import load_config

        config = load_config(env={})
        recorder = RunRecorder.new_run(tmp_path, config=config.to_dict())
        record = recorder.finish(status="finished", stop_reason="done")
        assert record.config["target_title_patterns"]
        assert record.config["max_mouse_delta"] == config.max_mouse_delta


class TestRunRecord:
    """The summary object."""

    def _record(self) -> RunRecord:
        return RunRecord(
            run_id="r1",
            started_at=0.0,
            finished_at=2.5,
            status="finished",
            steps=(
                {"result": {"executed": True, "blocked_reason": None}},
                {"result": {"executed": False, "blocked_reason": "not foreground"}},
                {"result": {"executed": True, "blocked_reason": None}},
            ),
        )

    def test_duration(self) -> None:
        assert self._record().duration == pytest.approx(2.5)

    def test_executed_and_blocked_counts(self) -> None:
        record = self._record()
        assert record.executed_steps == 2
        assert record.blocked_steps == 1

    def test_step_count_defaults_to_len_steps(self) -> None:
        assert self._record().step_count == 3

    def test_round_trip(self) -> None:
        original = self._record()
        restored = RunRecord.from_dict(original.to_dict())
        assert restored.run_id == original.run_id
        assert restored.status == original.status
        assert restored.executed_steps == original.executed_steps
        assert len(restored.steps) == 3

    def test_to_dict_is_json_serialisable(self) -> None:
        payload = json.loads(json.dumps(self._record().to_dict()))
        assert payload["run_id"] == "r1"

    def test_unfinished_record_has_no_duration(self) -> None:
        assert RunRecord(run_id="r", started_at=1.0).duration == 0.0


class TestNoPixelsInTelemetry:
    """The explicit anti-requirement: telemetry must not carry frames."""

    def test_run_json_has_no_image_payloads(self, tmp_path) -> None:
        recorder = RunRecorder(tmp_path)
        recorder.record_step(
            {
                "index": 0,
                "observation": {"has_frame": True, "frame": {"width": 320, "height": 240}},
                "capture_path": str(tmp_path / "frames" / "step-0000.png"),
            }
        )
        recorder.finish(status="finished", stop_reason="done")
        text = recorder.record_path.read_text(encoding="utf-8")
        assert "iVBORw0KGgo" not in text  # no base64 PNG header
        assert len(text) < 20_000
        payload = json.loads(text)
        assert payload["steps"][0]["observation"]["frame"]["width"] == 320

    def test_steps_ndjson_stays_small(self, tmp_path) -> None:
        recorder = RunRecorder(tmp_path)
        for index in range(50):
            recorder.record_step({"index": index, "frame_difference": 0.0})
        recorder.finish(status="finished", stop_reason="done")
        assert recorder.steps_path.stat().st_size < 10_000
