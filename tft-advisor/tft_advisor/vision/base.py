"""Perceiver contract: one screenshot in, one ``Observation`` out.

Every perceiver (Claude vision, OCR fast path, Live Client API, mock replay)
implements the same tiny protocol so the orchestrator can swap or combine them.

``purpose`` tells the perceiver why the frame was taken:

* ``auto``   : round changed, read everything (default)
* ``manual`` : the player pressed "analyze now", be thorough
* ``scout``  : the camera shows ANOTHER player's board (human in the loop)
* ``shop``   : only the shop / gold / level matter (fast, cheap)
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

from ..models import Observation

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image

PURPOSES: tuple[str, ...] = ("auto", "manual", "scout", "shop")


def normalize_purpose(purpose: Optional[str]) -> str:
    """Map any purpose string to one of :data:`PURPOSES` (unknown -> ``auto``)."""
    p = (purpose or "auto").strip().lower()
    return p if p in PURPOSES else "auto"


_EM_DASH = "\u2014"
# Two-em / three-em dashes look exactly like the forbidden double em dash; the
# horizontal bar is a common stand-in for a single em dash.
_LONG_DASHES = ("\u2e3a", "\u2e3b")
_HORIZONTAL_BAR = "\u2015"


def clean_text(text: str) -> str:
    """Player-facing text must never contain the em dash: double becomes a comma, single a hyphen."""
    out = (text or "").replace(_EM_DASH * 2, "，")
    for dash in _LONG_DASHES:
        out = out.replace(dash, "，")
    return out.replace(_EM_DASH, "-").replace(_HORIZONTAL_BAR, "-")


def clean_name(text: Any, max_len: int = 48) -> Optional[str]:
    """A short single-line name for prompts / observations (e.g. a player name typed on a phone).

    Control and separator characters (line breaks, tabs, zero-width marks)
    become spaces, whitespace is collapsed and the length capped; ``None`` when
    nothing is left. Keeps LAN input from bloating the prompt or smuggling
    extra instruction lines into it.
    """
    if text is None:
        return None
    chars = [" " if unicodedata.category(ch)[0] in ("C", "Z") else ch for ch in str(text)]
    out = " ".join("".join(chars).split())[:max_len].strip()
    return clean_text(out) or None


@dataclass
class PerceptionHint:
    """Context the tracker gives the perceiver to improve recognition.

    Everything is optional. Names are display names (any language) that are
    likely to be on screen, e.g. the units we believe are on our board.
    """

    champion_names: list[str] = field(default_factory=list)
    item_names: list[str] = field(default_factory=list)
    trait_names: list[str] = field(default_factory=list)
    self_name: Optional[str] = None  # local player's name in the player list
    expect: Optional[str] = None  # free text: what we expect to see (e.g. "augment_select")
    scouting_player: Optional[str] = None  # whose board the human says is shown


class PerceptionError(RuntimeError):
    """The perceiver could not produce an observation (API down, OCR missing, ...).

    The message may be shown to the player, so it is Simplified Chinese where
    it comes from this package. The original exception is chained as
    ``__cause__``.
    """


@runtime_checkable
class Perceiver(Protocol):
    name: str

    def perceive(
        self,
        image: "Image.Image | Any",
        purpose: str = "auto",
        hint: Optional[PerceptionHint] = None,
    ) -> Observation:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# HUD regions
# ---------------------------------------------------------------------------
# The authoritative region table lives in ``tft_advisor.capture.regions``. These
# approximate fractions of a 16:9 frame (x0, y0, x1, y1) are only a fallback so
# the vision package keeps working (and testing) without the capture package.
FALLBACK_REGIONS: dict[str, tuple[float, float, float, float]] = {
    # Same numbers as capture/regions.py (Set 18 Unreal HUD, padded to also cover
    # the "Trials" and legacy placements), copied so both tables agree.
    "stage": (0.385, 0.000, 0.475, 0.038),
    "gold": (0.450, 0.800, 0.565, 0.860),
    "level": (0.110, 0.800, 0.235, 0.860),
    "xp": (0.110, 0.800, 0.285, 0.865),
    "streak": (0.505, 0.800, 0.625, 0.860),
    "shop": (0.170, 0.845, 0.830, 1.000),  # 5 cards plus the Buy XP / Reroll buttons
    "bench": (0.150, 0.600, 0.800, 0.800),
    "board": (0.200, 0.260, 0.800, 0.700),
    "players": (0.850, 0.100, 1.000, 0.790),
    "traits": (0.028, 0.210, 0.155, 0.860),
    "items": (0.000, 0.210, 0.042, 0.820),
    "augments": (0.150, 0.150, 0.850, 0.880),
    "hud_bottom": (0.100, 0.790, 0.900, 1.000),
    "top_banner": (0.200, 0.000, 0.800, 0.200),
}


def _fallback_box(size: tuple[int, int], name: str) -> tuple[int, int, int, int]:
    if name not in FALLBACK_REGIONS:
        raise KeyError(f"unknown HUD region: {name}")
    w, h = size
    x0, y0, x1, y1 = FALLBACK_REGIONS[name]
    return (int(round(x0 * w)), int(round(y0 * h)), int(round(x1 * w)), int(round(y1 * h)))


def _capture_box(size: tuple[int, int], name: str) -> Optional[tuple[int, int, int, int]]:
    try:
        from ..capture import regions as _regions  # lazy: written by the capture module

        box = _regions.region_box(size, name)
        x0, y0, x1, y1 = (int(v) for v in box)
        if x1 > x0 and y1 > y0:
            return (x0, y0, x1, y1)
    except Exception:  # ImportError, unknown name, signature drift: use our table
        pass
    return None


def region_box(size: tuple[int, int], name: str, pad: float = 0.0) -> tuple[int, int, int, int]:
    """Pixel box ``(x0, y0, x1, y1)`` of a named HUD region, clamped to the frame.

    Uses ``tft_advisor.capture.regions`` when importable, otherwise
    :data:`FALLBACK_REGIONS`. ``pad`` grows the box by that fraction of its own
    width/height on every side.
    """
    w, h = int(size[0]), int(size[1])
    x0, y0, x1, y1 = _capture_box((w, h), name) or _fallback_box((w, h), name)
    if pad:
        px, py = (x1 - x0) * pad, (y1 - y0) * pad
        x0, y0, x1, y1 = int(x0 - px), int(y0 - py), int(round(x1 + px)), int(round(y1 + py))
    x0, y0 = min(max(0, x0), max(0, w - 1)), min(max(0, y0), max(0, h - 1))
    x1, y1 = min(max(x1, x0 + 1), w), min(max(y1, y0 + 1), h)
    return (x0, y0, max(x1, x0 + 1), max(y1, y0 + 1))


def crop_region(image: "Image.Image", name: str, pad: float = 0.0) -> "Image.Image":
    """Crop a named HUD region at full resolution."""
    return image.crop(region_box(image.size, name, pad))


__all__ = [
    "FALLBACK_REGIONS",
    "PURPOSES",
    "PerceptionError",
    "PerceptionHint",
    "Perceiver",
    "clean_name",
    "clean_text",
    "crop_region",
    "normalize_purpose",
    "region_box",
]
