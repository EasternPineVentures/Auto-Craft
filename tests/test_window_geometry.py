"""Window geometry and target matching.

These are the transformations that decide *where* AutoCraft looks and whether it
is allowed to act, so they are worth testing precisely and without a game
running.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from autocraft.agent.observation import Observer, WindowGeometry
from autocraft.vision.capture import ScreenCapturer
from autocraft.vision.frame import ScreenRegion
from autocraft.vision.window import (
    TargetStatus,
    WindowInfo,
    WindowLocator,
    _scaling_note,
    coordinate_scaling_note,
    title_matches,
)


class TestTitleMatching:
    """Title matching must be case-insensitive and substring-based."""

    @pytest.mark.parametrize(
        "title",
        [
            "Luanti 5.17.0 [Singleplayer] [4.6.0 NVIDIA 610.74]",
            "luanti",
            "LUANTI",
            "VoxelLibre - Singleplayer",
            "a window whose title mentions voxellibre somewhere",
        ],
    )
    def test_matches_known_titles(self, title: str) -> None:
        assert title_matches(title, ("Luanti", "VoxelLibre")) is True

    @pytest.mark.parametrize("title", ["Notepad", "AutoCraft", "SuperTuxKart", ""])
    def test_rejects_other_titles(self, title: str) -> None:
        assert title_matches(title, ("Luanti", "VoxelLibre")) is False

    def test_matching_is_substring_based(self) -> None:
        """Matching is deliberately loose, and the looseness is documented.

        A title merely *mentioning* the game still matches, so the locator also
        requires a real client area and prefers the foreground window. The
        permissive rule is asserted here so it can never drift silently.
        """
        assert title_matches("luanti-notes.txt", ("Luanti", "VoxelLibre")) is True

    def test_empty_pattern_list_matches_nothing(self) -> None:
        assert title_matches("Luanti", ()) is False

    def test_blank_pattern_is_ignored(self) -> None:
        assert title_matches("Luanti", ("", "  ")) is False
        assert title_matches("Luanti", ("", "luanti")) is True


class TestScreenRegion:
    """The client-area rectangle is the only thing AutoCraft captures."""

    def test_from_client_rect_offsets_by_screen_origin(self) -> None:
        # A window at (197, 305) whose client area covers (0, 0)..(1280, 720).
        region = ScreenRegion.from_client_rect((0, 0, 1280, 720), (197, 305))
        assert (region.left, region.top, region.width, region.height) == (197, 305, 1280, 720)

    def test_client_rect_edges_are_exclusive(self) -> None:
        """A Win32 ``GetClientRect`` returns 0..w, not 0..w-1.

        Treating the right/bottom edge as inclusive would capture one extra
        column and row of pixels outside the client area.
        """
        region = ScreenRegion.from_client_rect((8, 31, 1280, 720), (197, 305))
        assert (region.left, region.top, region.width, region.height) == (205, 336, 1272, 689)

    def test_client_rect_is_not_the_window_rect(self) -> None:
        """A decorated window's client area is strictly inside its window rect.

        Capturing the window rect would include the title bar and borders, which
        is why the geometry test exists at all.
        """
        window_rect = (100, 50, 1000, 800)
        client_rect = (8, 31, 976, 693)
        origin = (window_rect[0], window_rect[1])
        region = ScreenRegion.from_client_rect(client_rect, origin)
        assert region.left > window_rect[0]
        assert region.top > window_rect[1]
        assert region.width < window_rect[2]

    def test_right_and_bottom_are_exclusive_edges(self) -> None:
        region = ScreenRegion(10, 20, 30, 40)
        assert region.right == 40
        assert region.bottom == 60
        assert region.area == 1200

    def test_contains_uses_half_open_bounds(self) -> None:
        region = ScreenRegion(10, 20, 30, 40)
        assert region.contains(10, 20) is True
        assert region.contains(39, 59) is True
        assert region.contains(40, 59) is False
        assert region.contains(39, 60) is False

    def test_centre_is_the_middle_pixel(self) -> None:
        assert ScreenRegion(0, 0, 100, 50).centre() == (50, 25)

    def test_mss_round_trip(self) -> None:
        region = ScreenRegion(197, 305, 3454, 1593)
        monitor = region.to_mss_monitor()
        assert monitor == {"left": 197, "top": 305, "width": 3454, "height": 1593}
        assert ScreenRegion.from_mss_monitor(monitor) == region

    def test_degenerate_regions_are_empty(self) -> None:
        """Zero or negative extents are representable but never capturable.

        ``ScreenRegion`` is a value object used for arithmetic, so it does not
        raise; every capture path checks ``is_empty()`` first. That check is the
        contract worth pinning down.
        """
        zero_width = ScreenRegion(0, 0, 0, 10)
        negative_height = ScreenRegion(0, 0, 10, -1)
        assert zero_width.is_empty() is True
        assert negative_height.is_empty() is True
        assert zero_width.area == 0
        assert ScreenRegion(0, 0, 1, 1).is_empty() is False


class TestWindowLocator:
    """Locating the game window, including the foreground preference."""

    def test_prefers_the_foreground_matching_window(self, fake_windows) -> None:
        other = fake_windows.add(0x300, "Luanti 5.17.0 [Second instance]")
        fake_windows.focus(other.handle)
        locator = WindowLocator(fake_windows, ("Luanti",))
        status = locator.status(force=True)
        assert status.found is True
        assert status.window is not None
        assert status.window.handle == other.handle
        assert status.is_foreground is True

    def test_falls_back_to_the_largest_matching_window(self, fake_windows) -> None:
        fake_windows.add(0x300, "Luanti 5.17.0 [Second instance]", region=ScreenRegion(0, 0, 1920, 1080))
        fake_windows.focus(0x999)  # an unrelated window has focus
        locator = WindowLocator(fake_windows, ("Luanti",))
        status = locator.status(force=True)
        assert status.window is not None
        assert status.window.handle == 0x300
        assert status.is_foreground is False
        assert status.can_inject_input is False

    def test_prefers_a_capturable_window_over_a_degenerate_one(self, fake_windows) -> None:
        """A matching but unusable window must not shadow the real game window.

        A launcher or a minimised leftover can match the title pattern and even be
        larger in theory; picking it would leave AutoCraft capturing nothing.
        """
        fake_windows.add(0x300, "Luanti 5.17.0 [Launcher]", region=ScreenRegion(0, 0, 0, 0))
        fake_windows.focus(0x999)
        locator = WindowLocator(fake_windows, ("Luanti",))
        status = locator.status(force=True)
        assert status.window is not None
        assert status.window.handle == 0x100
        assert status.can_capture is True

    def test_reports_not_found_when_no_title_matches(self, fake_windows) -> None:
        locator = WindowLocator(fake_windows, ("NonexistentGame",))
        status = locator.status(force=True)
        assert status.found is False
        assert status.window is None
        assert status.can_capture is False
        assert status.can_inject_input is False

    def test_minimized_window_is_found_but_unusable(self, fake_windows) -> None:
        """A minimised game is still *the* target, it just cannot be acted on.

        Reporting ``found=False`` would hide a real, fixable condition (the player
        alt-tabbed away) behind the same message as "the game is not running".
        """
        fake_windows.windows.clear()
        fake_windows.add(0x400, "Luanti 5.17.0 [Minimized]", minimized=True)
        locator = WindowLocator(fake_windows, ("Luanti",))
        status = locator.status(force=True)
        assert status.found is True
        assert status.can_capture is False
        assert status.can_inject_input is False
        assert "minimi" in status.reason.lower()

    def test_ignores_invisible_windows(self, fake_windows) -> None:
        fake_windows.windows.clear()
        fake_windows.add(0x400, "Luanti 5.17.0 [Hidden]", visible=False)
        locator = WindowLocator(fake_windows, ("Luanti",))
        status = locator.status(force=True)
        assert status.found is False

    def test_caches_between_calls(self, fake_windows) -> None:
        locator = WindowLocator(fake_windows, ("Luanti",), rediscover_after=10.0)
        locator.status(force=True)
        enumerations_after_first = fake_windows.enumeration_count
        locator.status()
        locator.status()
        assert fake_windows.enumeration_count == enumerations_after_first

    def test_forget_forces_rediscovery(self, fake_windows) -> None:
        locator = WindowLocator(fake_windows, ("Luanti",), rediscover_after=10.0)
        locator.status(force=True)
        locator.forget()
        locator.status()
        assert fake_windows.enumeration_count == 2

    def test_foreground_check_uses_the_cached_handle(self, fake_windows) -> None:
        locator = WindowLocator(fake_windows, ("Luanti",))
        locator.status(force=True)
        assert locator.is_target_foreground() is True
        fake_windows.focus(0x999)
        assert locator.is_target_foreground() is False

    def test_foreground_check_discovers_the_window_lazily(self, fake_windows) -> None:
        """Asking "is the game focused?" must work before any explicit discovery.

        The guard calls this on every action, so it has to resolve the window
        itself rather than depending on some earlier call having done it.
        """
        locator = WindowLocator(fake_windows, ("Luanti",))
        assert locator.is_target_foreground() is True

    def test_target_region_returns_the_client_area(self, fake_windows) -> None:
        locator = WindowLocator(fake_windows, ("Luanti",))
        region = locator.target_region()
        assert region.width > 0
        assert region.height > 0

    def test_target_region_raises_when_nothing_matches(self, fake_windows) -> None:
        from autocraft.vision.window import WindowError

        locator = WindowLocator(fake_windows, ("NonexistentGame",))
        with pytest.raises(WindowError):
            locator.target_region()

    def test_window_disappearing_is_reported_not_raised(self, fake_windows) -> None:
        locator = WindowLocator(fake_windows, ("Luanti",))
        assert locator.status(force=True).found is True
        fake_windows.remove(0x100)
        locator.forget()
        assert locator.status(force=True).found is False


class TestTargetStatus:
    """The two permission questions must stay distinct."""

    def test_capture_allowed_while_not_foreground(self) -> None:
        window = WindowInfo(
            handle=1,
            title="Luanti",
            region=ScreenRegion(0, 0, 100, 100),
            visible=True,
            minimized=False,
        )
        status = TargetStatus(found=True, window=window, is_foreground=False)
        assert status.can_capture is True
        assert status.can_inject_input is False

    def test_minimized_window_allows_neither(self) -> None:
        window = WindowInfo(
            handle=1,
            title="Luanti",
            region=ScreenRegion(0, 0, 100, 100),
            visible=True,
            minimized=True,
        )
        status = TargetStatus(found=True, window=window, is_foreground=True)
        assert status.can_capture is False
        assert status.can_inject_input is False

    def test_serialisation_is_json_friendly(self) -> None:
        import json

        status = TargetStatus(found=False, reason="no match")
        assert json.loads(json.dumps(status.to_dict()))["reason"] == "no match"


class TestObserverGeometry:
    """Finding 1: the observer is where a window that changed size is noticed.

    Run #2 reported a 3222x1928 client area before its countdown and a 3591x1928
    frame on every step after it. The policy-level test builds that disagreement
    by hand, which proves the *event* but not the *detection*. These drive the
    real ``Observer`` - the object a live run actually uses - across a window
    that changed size underneath it, because that seam is the one a live run
    depends on and the one no test covered.
    """

    def _observer(self, fake_windows, fake_capture, clock) -> Observer:
        locator = WindowLocator(fake_windows, ("Luanti",), clock=clock, rediscover_after=0.0)
        return Observer(locator, ScreenCapturer(fake_capture, clock=clock), clock=clock)

    def test_a_window_that_changed_size_is_reported(self, fake_windows, fake_capture, clock) -> None:
        fake_windows.windows.clear()
        window = fake_windows.add(0x100, "Luanti 5.17.0", region=ScreenRegion(0, 0, 3222, 1928))
        fake_windows.focus(window.handle)
        observer = self._observer(fake_windows, fake_capture, clock)

        first = observer.observe(0)
        assert first.geometry.client_width == 3222
        assert first.geometry.frame_width == 3222, "the frame is the client area here"
        assert first.geometry_changed_from is None, "there is nothing to compare against yet"

        fake_windows.windows[:] = [
            replace(w, region=ScreenRegion(0, 0, 3591, 1928)) if w.handle == window.handle else w
            for w in fake_windows.windows
        ]

        second = observer.observe(1)
        assert second.geometry_changed_from is not None
        assert second.geometry_changed_from.client_width == 3222
        assert second.geometry_change == ("client_width", "frame_width"), (
            "the height never moved, so the change must not claim it did"
        )
        assert second.to_dict()["geometry_changed_fields"] == ["client_width", "frame_width"]

    def test_an_unchanged_window_reports_no_change(self, fake_windows, fake_capture, clock) -> None:
        fake_windows.windows.clear()
        window = fake_windows.add(0x100, "Luanti 5.17.0", region=ScreenRegion(0, 0, 1280, 720))
        fake_windows.focus(window.handle)
        observer = self._observer(fake_windows, fake_capture, clock)

        observer.observe(0)
        for index in (1, 2, 3):
            assert observer.observe(index).geometry_changed_from is None, (
                "a window that held still must not re-report its geometry every step"
            )

    def test_a_seeded_geometry_is_the_baseline_for_the_first_look(
        self, fake_windows, fake_capture, clock
    ) -> None:
        """The plan is sampled before the countdown and the run after it.

        That is exactly where run #2's two sizes appeared, so the seed has to be
        what the first observation is compared against. Without it the very
        disagreement the instrumentation exists to catch is the one it misses.

        The seed carries no frame, and a frame that was never measured is not
        news, so the change is reported once - as the client area - rather than
        three times. This test used to require ``frame_width`` and
        ``frame_height`` here as well, and that is what made the event fire on
        every run: the third live run announced a resize at step 0 with a client
        area of 2102x1061 on both sides of the comparison.
        """
        fake_windows.windows.clear()
        window = fake_windows.add(0x100, "Luanti 5.17.0", region=ScreenRegion(0, 0, 3591, 1928))
        fake_windows.focus(window.handle)
        observer = self._observer(fake_windows, fake_capture, clock)
        observer.seed_geometry(
            WindowGeometry(
                handle=window.handle,
                client_width=3222,
                client_height=1928,
                frame_width=0,
                frame_height=0,
            )
        )

        first = observer.observe(0)
        assert first.geometry_change == ("client_width",), (
            "the seed carried no frame, so the frame extents are not news; the "
            "client area is, and that is the run #2 disagreement"
        )
        assert first.geometry_changed_from is not None
        assert first.geometry_changed_from.client_width == 3222
        assert observer.last_geometry == first.geometry

    def test_a_first_frame_after_a_frameless_seed_is_not_a_resize(
        self, fake_windows, fake_capture, clock
    ) -> None:
        """The whole run, at its own scale, with nothing resized at all.

        The third live run opened with ``WINDOW_GEOMETRY_CHANGED``: a seed with
        ``frame_width 0`` against a first frame of 2102, while the client area
        read 2102x1061 on both sides. Nothing had moved. The event is not only
        printed - it rebaselines the policy and discards the view memory, the
        progress model, the repetition guard, the cooldowns and the calibration -
        so a false one costs the run its state.
        """
        fake_windows.windows.clear()
        window = fake_windows.add(0x100, "Luanti 5.17.0", region=ScreenRegion(0, 0, 2102, 1061))
        fake_windows.focus(window.handle)
        observer = self._observer(fake_windows, fake_capture, clock)
        observer.seed_geometry(
            WindowGeometry(
                handle=window.handle,
                client_width=2102,
                client_height=1061,
                frame_width=0,
                frame_height=0,
            )
        )

        first = observer.observe(0)
        assert first.geometry.frame_width == 2102, (
            "in this fixture the frame is the client area, as it was in the run"
        )
        assert first.geometry_changed_from is None, (
            "a frame that was captured for the first time is not a window that "
            "changed size"
        )
        assert first.geometry_change == ()

        for index in (1, 2, 3):
            assert observer.observe(index).geometry_changed_from is None, (
                "nothing moved, so nothing may be reported on any later step either"
            )

    def test_a_geometry_that_was_never_seen_is_not_a_change(
        self, fake_windows, fake_capture, clock
    ) -> None:
        """Losing sight of the window and getting it back has resized nothing.

        An empty geometry is a missing reading, not a change, so a step that
        failed to find the window must not bury the real change in noise.
        """
        fake_windows.windows.clear()
        window = fake_windows.add(0x100, "Luanti 5.17.0", region=ScreenRegion(0, 0, 1280, 720))
        fake_windows.focus(window.handle)
        observer = self._observer(fake_windows, fake_capture, clock)
        observer.seed_geometry(WindowGeometry())

        first = observer.observe(0)
        assert first.geometry_changed_from is None
        assert first.geometry_change == ()
        assert observer.last_geometry == first.geometry, (
            "an empty seed must not stop the real geometry being remembered"
        )


# ---------------------------------------------------------------------------
# DPI awareness
# ---------------------------------------------------------------------------

#: Run in a fresh interpreter, because DPI awareness is process-global and can
#: only be chosen once. Asking for it twice in the pytest process would measure
#: whatever the first caller had already arranged.
_AWARENESS_SCRIPT = """
import json
from autocraft.vision.window import _measured_dpi_awareness, ensure_dpi_awareness

requested = ensure_dpi_awareness()
print(json.dumps({"requested": requested, "measured": _measured_dpi_awareness()}))
"""


def _source_root() -> str:
    """Return the directory that has ``autocraft`` in it."""
    import autocraft

    return str(Path(autocraft.__file__).resolve().parents[1])


def _awareness_in_a_fresh_process() -> dict[str, object]:
    """Ask a new interpreter to arrange DPI awareness and report the result."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = _source_root()
    completed = subprocess.run(
        [sys.executable, "-c", _AWARENESS_SCRIPT],
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


class TestDpiAwareness:
    """Windows virtualises coordinates unless the process asks not to be.

    A virtualised client rectangle is still a plausible rectangle, and the
    capture backend still returns exactly the pixels it was asked for, so a
    failure here cannot be detected downstream - only prevented here.
    """

    @pytest.mark.skipif(sys.platform != "win32", reason="DPI awareness is Win32 only")
    def test_the_process_really_becomes_per_monitor_aware(self) -> None:
        """The per-monitor context must actually take effect.

        This is the assertion that catches an awareness context passed as a
        truncated integer: the call still looks like it succeeded, and the
        process quietly stays merely system aware.
        """
        report = _awareness_in_a_fresh_process()
        assert report["measured"] == "per-monitor", (
            "ensure_dpi_awareness returned "
            f"{report['requested']!r}, but the process ended up "
            f"{report['measured']!r}"
        )

    @pytest.mark.skipif(sys.platform != "win32", reason="DPI awareness is Win32 only")
    def test_the_reported_mode_is_the_measured_mode(self) -> None:
        """Report what was achieved, never what was attempted.

        The old code returned the name of the API it had just called, so a
        failure was reported as a success and ``status`` said ``system`` as if
        that had been the plan all along.
        """
        report = _awareness_in_a_fresh_process()
        assert report["requested"] == report["measured"]

    @pytest.mark.skipif(sys.platform != "win32", reason="DPI awareness is Win32 only")
    def test_the_awareness_context_is_bound_as_a_handle(self) -> None:
        """``DPI_AWARENESS_CONTEXT`` is a pointer, not an int.

        Without an ``argtypes`` declaration ctypes marshals the literal ``-4``
        as a 32-bit C ``int``; the API wants a 64-bit handle and rejects the
        call with ``ERROR_INVALID_PARAMETER``.
        """
        from autocraft.vision.window import _win32

        user32 = _win32()["user32"]
        assert user32.SetProcessDpiAwarenessContext.argtypes == [ctypes.c_void_p]


class TestCoordinateScalingNote:
    """Saying out loud when Windows is scaling a window's coordinates.

    Split from the Win32 reading so the decision itself is testable without a
    desktop to read DPI from.
    """

    def test_per_monitor_awareness_needs_no_caution(self) -> None:
        assert _scaling_note("per-monitor", 120, 288) is None

    def test_matching_scales_need_no_caution(self) -> None:
        """The same DPI everywhere means nothing is being virtualised."""
        assert _scaling_note("system", 120, 120) is None

    @pytest.mark.parametrize(("window_dpi", "system_dpi"), [(0, 288), (120, 0), (-1, 288)])
    def test_an_unreadable_dpi_is_not_a_caution(self, window_dpi: int, system_dpi: int) -> None:
        """A missing reading must not be reported as a problem."""
        assert _scaling_note("system", window_dpi, system_dpi) is None

    @pytest.mark.parametrize("mode", ["system", "unaware"])
    def test_a_scale_mismatch_is_explained_with_both_numbers(self, mode: str) -> None:
        note = _scaling_note(mode, 120, 288)
        assert note is not None
        assert mode in note
        assert "120" in note
        assert "288" in note

    def test_no_target_means_nothing_to_warn_about(self) -> None:
        assert coordinate_scaling_note(None) is None

    def test_the_note_is_a_single_line(self) -> None:
        """It is printed after a table row, so it must not wrap the layout."""
        note = _scaling_note("system", 120, 288)
        assert note is not None
        assert "\n" not in note
