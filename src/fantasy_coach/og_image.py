"""Open-Graph image renderer for match prediction sharing cards (issue #175).

Renders a 1200x630 PNG using Pillow. No browser / headless dependency.

Design: dark navy background with two team-colour bands (left = home, right =
away), large team names, win-probability percentages, a filled progress bar,
round/kickoff label, and a "Fantasy Coach" wordmark.

Fonts: prefers TTF files found in (1) assets/og/ (Inter, if committed) then
(2) the system Liberation Sans font (installed via fonts-liberation apt package
in the Cloud Run runtime image) then (3) Pillow's built-in bitmap font.

Cache: callers are responsible for caching the PNG bytes externally. The
_team_colours() loader is cached with lru_cache since it reads from disk.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import UTC
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

CARD_W, CARD_H = 1200, 630
TEMPLATE_VERSION = "v1"  # bump to bust Firestore og_cache entries

_ASSETS = Path(__file__).parents[2] / "assets" / "og"
_COLOURS_PATH = _ASSETS / "team_colours.json"

# Font search order: repo-committed Inter TTF → system Liberation Sans → PIL default.
_FONT_CANDIDATES_BOLD: tuple[Path, ...] = (
    _ASSETS / "Inter-Bold.ttf",
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
)
_FONT_CANDIDATES_REG: tuple[Path, ...] = (
    _ASSETS / "Inter-Regular.ttf",
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
)


@lru_cache(maxsize=1)
def _team_colours() -> dict[str, str]:
    try:
        return json.loads(_COLOURS_PATH.read_text())
    except Exception:
        return {}


def _load_font(size: int, bold: bool = False):
    """Return an ImageFont, trying TTF paths in order then falling back to PIL default."""
    from PIL import ImageFont  # noqa: PLC0415

    candidates = _FONT_CANDIDATES_BOLD if bold else _FONT_CANDIDATES_REG
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _draw_centred(draw, text: str, cx: int, cy: int, font, fill: str) -> None:
    """Draw text with its centre at (cx, cy)."""
    try:
        bbox = font.getbbox(text)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # Older Pillow — textsize() was removed in 10.0 but keep for safety.
        try:
            w, h = draw.textsize(text, font=font)
        except Exception:
            w, h = 0, 0
    draw.text((cx - w // 2, cy - h // 2), text, font=font, fill=fill)


def _hex_to_rgba(hex_str: str, alpha: int = 255) -> tuple[int, int, int, int]:
    h = hex_str.lstrip("#")
    if len(h) != 6:
        return (30, 30, 30, alpha)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


def render_card(
    home_name: str,
    away_name: str,
    home_win_prob: float,
    kickoff_iso: str,
    round_label: str,
) -> bytes:
    """Render a 1200x630 PNG prediction card. Returns raw PNG bytes.

    Args:
        home_name: Home team display name.
        away_name: Away team display name.
        home_win_prob: Probability in [0, 1] that the home team wins.
        kickoff_iso: ISO 8601 kickoff timestamp (UTC).
        round_label: Short round description, e.g. "Round 7, 2026".

    Returns:
        PNG image bytes ready to serve with media_type="image/png".
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415

    colours = _team_colours()
    home_hex = colours.get(home_name, "#1a3a6c")
    away_hex = colours.get(away_name, "#6c1a1a")

    # Base: dark navy.
    img = Image.new("RGB", (CARD_W, CARD_H), "#0f172a")

    # Semi-transparent team colour bands overlay.
    overlay = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    ov = ImageDraw.Draw(overlay)
    ov.rectangle([(0, 0), (CARD_W // 2 - 1, CARD_H)], fill=_hex_to_rgba(home_hex, alpha=160))
    ov.rectangle([(CARD_W // 2, 0), (CARD_W, CARD_H)], fill=_hex_to_rgba(away_hex, alpha=160))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(img)

    # Vertical divider.
    draw.rectangle([(CARD_W // 2 - 1, 0), (CARD_W // 2 + 1, CARD_H)], fill="#ffffff33")

    font_team = _load_font(58, bold=True)
    font_pct = _load_font(84, bold=True)
    font_label = _load_font(30)
    font_small = _load_font(24)
    font_wordmark = _load_font(20, bold=True)

    # Team names.
    _draw_centred(draw, home_name, CARD_W // 4, 130, font_team, "#ffffff")
    _draw_centred(draw, away_name, CARD_W * 3 // 4, 130, font_team, "#ffffff")
    _draw_centred(draw, "vs", CARD_W // 2, 130, font_label, "#ffffffaa")

    # Win probability percentages.
    home_pct = round(home_win_prob * 100)
    away_pct = 100 - home_pct
    _draw_centred(draw, f"{home_pct}%", CARD_W // 4, 280, font_pct, "#ffffff")
    _draw_centred(draw, f"{away_pct}%", CARD_W * 3 // 4, 280, font_pct, "#ffffff")

    # Probability fill bar.
    bx0, bx1, by, bh = 80, CARD_W - 80, 410, 18
    bw = bx1 - bx0
    draw.rounded_rectangle([(bx0, by), (bx1, by + bh)], radius=9, fill="#ffffff22")
    fill_w = max(0, min(bw, round(bw * home_win_prob)))
    if fill_w > 0:
        draw.rounded_rectangle([(bx0, by), (bx0 + fill_w, by + bh)], radius=9, fill="#22c55e")

    # Round + kickoff labels.
    try:
        from datetime import datetime  # noqa: PLC0415

        dt = datetime.fromisoformat(kickoff_iso.replace("Z", "+00:00"))
        dt_utc = dt.astimezone(UTC)
        kickoff_str = dt_utc.strftime("%-d %b %Y  %H:%M UTC")
    except Exception:
        kickoff_str = kickoff_iso[:16].replace("T", " ")

    _draw_centred(draw, round_label, CARD_W // 2, 460, font_label, "#ffffffbb")
    _draw_centred(draw, kickoff_str, CARD_W // 2, 498, font_small, "#ffffff88")

    # Wordmark (bottom-right).
    draw.text((CARD_W - 185, CARD_H - 34), "Fantasy Coach", font=font_wordmark, fill="#ffffff44")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
