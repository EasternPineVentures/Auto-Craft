"""Window geometry and target matching.

These are the transformations that decide *where* AutoCraft looks and whether it
is allowed to act, so they are worth testing precisely and without a game
running.
"""

from __future__ import annotations

import pytest

from autocraft.vision.frame import ScreenRegion
from autocraft.vision.window import TargetStatus, WindowInfo, WindowLocator, title_matches


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
