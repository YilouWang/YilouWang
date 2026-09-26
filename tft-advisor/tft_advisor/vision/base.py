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
    "stage": (0.35, 0.0, 0.65, 0.06),
    "top_banner": (0.30, 0.03, 0.70, 0.14),
    "gold": (0.44, 0.805, 0.56, 0.865),
    "streak": (0.53, 0.805, 0.62, 0.865),
    "level": (0.12, 0.805, 0.25, 0.865),
    "xp": (0.12, 0.805, 0.25, 0.865),
    "shop": (0.245, 0.865, 0.775, 0.995),
    "hud_bottom": (0.10, 0.78, 0.90, 1.0),
    "bench": (0.17, 0.64, 0.80, 0.80),
    "board": (0.20, 0.28, 0.80, 0.68),
    "players": (0.83, 0.15, 1.0, 0.80),
    "traits": (0.0, 0.17, 0.15, 0.75),
    "items": (0.08, 0.42, 0.25, 0.80),
    "augments": (0.18, 0.18, 0.82, 0.78),
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
    "crop_region",
    "normalize_purpose",
    "region_box",
]
