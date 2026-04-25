"""Tests for the OG image card renderer (issue #175)."""

import pytest


def test_render_card_returns_png_bytes():
    from fantasy_coach.og_image import render_card

    png = render_card(
        home_name="Sydney Roosters",
        away_name="Melbourne Storm",
        home_win_prob=0.65,
        kickoff_iso="2026-04-25T09:00:00Z",
        round_label="Round 7, 2026",
    )
    assert isinstance(png, bytes)
    assert len(png) > 1000
    assert png[:8] == b"\x89PNG\r\n\x1a\n"  # PNG file signature


def test_render_card_unknown_team_uses_default_colours():
    from fantasy_coach.og_image import render_card

    png = render_card(
        home_name="Unknown FC",
        away_name="Mystery United",
        home_win_prob=0.5,
        kickoff_iso="2026-04-25T09:00:00Z",
        round_label="Round 1, 2026",
    )
    assert isinstance(png, bytes)
    assert png[:4] == b"\x89PNG"


def test_render_card_extreme_probabilities():
    from fantasy_coach.og_image import render_card

    for prob in (0.0, 0.01, 0.99, 1.0):
        png = render_card(
            home_name="Home Team",
            away_name="Away Team",
            home_win_prob=prob,
            kickoff_iso="2026-04-25T09:00:00Z",
            round_label="Round 5, 2026",
        )
        assert png[:4] == b"\x89PNG", f"PNG header check failed for prob={prob}"


def test_render_card_bad_kickoff_iso():
    from fantasy_coach.og_image import render_card

    # Should not raise even if kickoff format is unexpected.
    png = render_card(
        home_name="Home",
        away_name="Away",
        home_win_prob=0.5,
        kickoff_iso="not-a-date",
        round_label="Round 1",
    )
    assert isinstance(png, bytes)


@pytest.mark.parametrize(
    "team",
    [
        "Brisbane Broncos",
        "Melbourne Storm",
        "Penrith Panthers",
        "South Sydney Rabbitohs",
        "Wests Tigers",
    ],
)
def test_team_colours_loaded_for_known_teams(team):
    from fantasy_coach.og_image import _team_colours

    colours = _team_colours()
    assert team in colours, f"Expected colour for {team!r}"
    assert colours[team].startswith("#"), f"Colour should be a hex string, got {colours[team]!r}"
