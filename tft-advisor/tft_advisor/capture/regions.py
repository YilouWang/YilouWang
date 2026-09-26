"""Named TFT HUD regions as fractions of a 16:9 game frame.

Every region is ``(x0, y0, x1, y1)`` in fractions (0..1) of a 16:9 frame, so the
same numbers work for 1280x720, 1920x1080, 2560x1440 and 3840x2160.

Sources (measured September 2026, patch 18.x, TFT on Unreal Engine):

* Primary: the live Set 18 (Unreal client) diagnostic frame published by
  mallangmallang0619/TFT-COACH (``docs/images/live-diagnostic-set18.png``,
  2560x1440, captured 2026-09-06: level 6, 62 gold, 73 HP) measured with a
  1 % grid, cross-checked against that project's ``backend/config.py``
  ``GameROIs`` (ratios calibrated on live 2560x1440 Set 18 frames).
* Legacy reference: jfd02/TFT-OCR-BOT ``screen_coords.py`` (1920x1080, the
  old Hextech/League client used until Set 17). Kept in the comments so the
  regions can be checked if someone replays old recordings. Everything except
  the item bench also covers the legacy layout.

The regions are padded on purpose: the Claude vision perceiver is robust to
extra context, while a crop that is too tight silently loses information
(e.g. TFT-COACH notes that Set 18 shop names extend left of the card art, and
that "Trials" games shift the bottom HUD roughly 0.055 of the width to the
left, so gold / level / streak boxes cover both placements).

Non 16:9 frames (documented assumption, verify with ``tft-advisor calibrate``):

* ``layout="inscribed"`` (default): TFT draws its HUD inside the largest
  16:9 rectangle that fits the frame, centered. An ultrawide 21:9 frame is
  pillarboxed (the HUD scales with the height and is centered horizontally),
  a 16:10 frame is letterboxed (the HUD scales with the width and is centered
  vertically). This is also what TFT-COACH assumes for the Unreal client.
* ``layout="anchored"``: alternative if calibration shows the HUD glued to the
  screen edges instead: same scale as above, but left-anchored regions (trait
  tracker, item bench) stick to the left edge, the player list to the right
  edge, top regions to the top and bottom regions to the bottom.

The default layout can also be switched with the environment variable
``TFT_ADVISOR_HUD_LAYOUT=anchored``.
"""

from __future__ import annotations

import math
import os
from typing import Optional

from PIL import Image

Box = tuple[float, float, float, float]

#: Aspect ratio the region fractions refer to.
BASE_ASPECT = 16 / 9
#: Resolution the fractions were derived for (only used for documentation / pixel tables).
BASE_SIZE = (1920, 1080)

LAYOUTS = ("inscribed", "anchored")
DEFAULT_LAYOUT = "inscribed"

# fmt: off
REGIONS: dict[str, Box] = {
    # Round indicator ("3-5") at the top center.
    #   Set 18: text at x 0.397-0.416, y 0.007-0.024; TFT-COACH stage ROI
    #   (0.388, 0.000) + (0.035 x 0.032). The top bar's dark plate starts at
    #   x ~0.385 near its bottom edge (sky and clouds animate left of it).
    #   Legacy: ROUND_POS (753, 10, 870, 34) -> 0.392-0.453 x 0.009-0.031; the
    #   text moves right in stages with fewer rounds, hence x1 = 0.475.
    #   y1 stays below 0.040 where the animated round timer bar starts, so the
    #   RoundWatcher does not see the timer drain.
    "stage":      (0.385, 0.000, 0.475, 0.038),
    # Gold ("62" next to a coin) in the bottom HUD.
    #   Set 18 standard: 0.520-0.546 x 0.819-0.842 (TFT-COACH gold_standard
    #   0.530-0.552 x 0.806-0.846); Trials: 0.475-0.500 x 0.805-0.850.
    #   Legacy: GOLD_POS (870, 883, 920, 909) -> 0.453-0.479 x 0.818-0.842.
    "gold":       (0.450, 0.800, 0.565, 0.860),
    # "Lvl. 6" label.
    #   Set 18 standard: 0.182-0.212 x 0.819-0.840 (TFT-COACH level_standard
    #   0.165-0.265 x 0.795-0.855); Trials: 0.115-0.175 x 0.805-0.850.
    "level":      (0.110, 0.800, 0.235, 0.860),
    # XP text "20/36" plus the XP bar under the level label.
    #   Set 18 standard: text 0.244-0.266 x 0.819-0.840, bar 0.182-0.236 x 0.845-0.851.
    "xp":         (0.110, 0.800, 0.285, 0.865),
    # Win/loss streak (flame + number) right of the gold.
    #   Set 18 standard: 0.576-0.604 x 0.815-0.840; Trials shifted ~0.055 left.
    "streak":     (0.505, 0.800, 0.625, 0.860),
    # Shop: five cards plus the Buy XP / Reroll buttons on their left.
    #   Set 18 standard: cards 0.290-0.808 x 0.858-0.997, buttons 0.184-0.232;
    #   TFT-COACH shop ROI 0.249-0.777 x 0.855-0.998 (Trials placement).
    #   Legacy: SHOP_POS (481, 1039, 1476, 1070) -> 0.250-0.769 x 0.962-0.991,
    #   BUY_LOC y 992 (0.918), BUY_XP_LOC (364, 964), REFRESH_LOC (364, 1039).
    "shop":       (0.170, 0.845, 0.830, 1.000),
    # Champion bench (9 slots under the board, including unit models).
    #   Set 18: slots 0.184-0.750, models 0.620-0.770 (TFT-COACH
    #   champion_bench 0.183-0.748 x 0.635-0.770).
    #   Legacy: BENCH_HEALTH_POS (369..1411, 650..757) -> 0.192-0.735 x 0.602-0.701.
    "bench":      (0.150, 0.600, 0.800, 0.800),
    # Own half of the board (4 x 7 hexes) including unit models, health bars
    # and item icons above front-row units.
    #   Set 18: hex row centers y 0.411 / 0.478 / 0.547 / 0.622, x 0.256-0.710,
    #   front-row health bars up to y ~0.30 (TFT-COACH board 0.25-0.75 x 0.382-0.688).
    #   Legacy: BOARD_LOC feet 532..1349 x 423..651 -> 0.277-0.703 x 0.392-0.603.
    "board":      (0.200, 0.260, 0.800, 0.700),
    # Player list (names + HP) on the right edge, 8 rows.
    #   Set 18: rows 0.860-0.996 x 0.154-0.736 (TFT-COACH player_hp scan area
    #   0.900-0.980 x 0.120-0.750). Legacy: HEALTH_LOC (1897, 126) -> 0.988 x 0.117.
    "players":    (0.850, 0.100, 1.000, 0.790),
    # Trait tracker on the left (icon, count, name, breakpoints).
    #   Set 18: icons from x 0.034, text to 0.115, first row y 0.241, pitch
    #   ~0.047 (TFT-COACH TraitPanel: first_row_cy 0.285, row_pitch 0.0466,
    #   up to 12 rows), so 12-13 rows reach y ~0.85.
    "traits":     (0.028, 0.210, 0.155, 0.860),
    # Item bench (unequipped components): Set 18 moved it to a slot column on
    # the far left edge, filling upward: 0.002-0.029 x 0.237-0.760 (TFT-COACH
    # item_bench 0.002-0.032 x 0.240-0.820).
    #   Legacy (NOT covered): ITEM_POS triangle left of the board,
    #   (273..457, 586..753) -> 0.142-0.238 x 0.543-0.697.
    "items":      (0.000, 0.210, 0.042, 0.820),
    # Augments: the 3-card choice overlay and the owned augment icons shown
    # above the board's top-left corner (Set 18: 0.238-0.294 x 0.211-0.242).
    #   TFT-COACH augment_panel 0.18-0.82 x 0.20-0.75.
    #   Legacy: AUGMENT_LOC y 445 (0.412), AUGMENT_POS names 417..1500 x 552..582
    #   (0.217-0.781 x 0.511-0.539), AUGMENT_ROLL y 875 (0.810).
    "augments":   (0.150, 0.150, 0.850, 0.880),
    # Whole bottom HUD strip: level, XP, shop odds, gold, streak, shop.
    #   Set 18 panel frame: 0.176-0.820 x 0.805-1.000 (+ Trials shift).
    "hud_bottom": (0.100, 0.790, 0.900, 1.000),
    # Top center: round bar plus the area above the board where the board
    # owner's name plate is shown while scouting (Set 18 frame: owner name
    # plate at 0.597-0.666 x 0.108-0.150).
    "top_banner": (0.200, 0.000, 0.800, 0.200),
}
# fmt: on

#: Anchors used by ``layout="anchored"``: (horizontal l|c|r, vertical t|m|b).
ANCHORS: dict[str, tuple[str, str]] = {
    "stage": ("c", "t"),
    "gold": ("c", "b"),
    "level": ("c", "b"),
    "xp": ("c", "b"),
    "streak": ("c", "b"),
    "shop": ("c", "b"),
    "bench": ("c", "m"),
    "board": ("c", "m"),
    "players": ("r", "m"),
    "traits": ("l", "m"),
    "items": ("l", "m"),
    "augments": ("c", "m"),
    "hud_bottom": ("c", "b"),
    "top_banner": ("c", "t"),
}

#: Chinese labels (dashboard / calibration image).
REGION_LABELS: dict[str, str] = {
    "stage": "阶段回合",
    "gold": "金币",
    "level": "等级",
    "xp": "经验",
    "streak": "连胜连败",
    "shop": "商店",
    "bench": "备战席",
    "board": "棋盘",
    "players": "玩家列表",
    "traits": "羁绊",
    "items": "装备栏",
    "augments": "强化符文",
    "hud_bottom": "底部面板",
    "top_banner": "顶部横幅",
}

_EPS = 1e-6


def list_regions() -> list[str]:
    """Region names in a stable order."""
    return list(REGIONS)


def _resolve_layout(layout: Optional[str]) -> str:
    name = (layout or os.environ.get("TFT_ADVISOR_HUD_LAYOUT") or DEFAULT_LAYOUT).strip().lower()
    if name not in LAYOUTS:
        raise ValueError(f"unknown HUD layout {name!r}; expected one of {LAYOUTS}")
    return name


def _check_size(size: tuple[int, int]) -> tuple[int, int]:
    w, h = int(size[0]), int(size[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid frame size {size!r}")
    return w, h


def viewport(size: tuple[int, int]) -> tuple[float, float, float, float]:
    """The centered 16:9 rectangle inside a frame: ``(left, top, width, height)`` in pixels.

    Equal to the whole frame for 16:9 frames; pillarboxed for wider frames
    (21:9, 32:9), letterboxed for taller frames (16:10, 4:3).
    """
    w, h = _check_size(size)
    if w / h > BASE_ASPECT:
        vh = float(h)
        vw = vh * BASE_ASPECT
    else:
        vw = float(w)
        vh = vw / BASE_ASPECT
    return ((w - vw) / 2.0, (h - vh) / 2.0, vw, vh)


def _fractions(name: str) -> Box:
    try:
        return REGIONS[name]
    except KeyError:
        raise KeyError(f"unknown region {name!r}; known regions: {', '.join(REGIONS)}") from None


def region_box(
    size: tuple[int, int],
    name: str,
    *,
    pad: float = 0.0,
    layout: Optional[str] = None,
) -> tuple[int, int, int, int]:
    """Pixel box ``(x0, y0, x1, y1)`` of region ``name`` in a frame of ``size`` (w, h).

    ``pad`` grows the box on every side by that fraction of the region's own
    width / height (0.1 = 10 % more on each side). The result is clamped to
    the frame and is never empty. Edges are rounded outward (floor / ceil).
    """
    w, h = _check_size(size)
    fx0, fy0, fx1, fy1 = _fractions(name)
    ox, oy, vw, vh = viewport((w, h))
    if _resolve_layout(layout) == "anchored":
        ax, ay = ANCHORS.get(name, ("c", "m"))
        ox = {"l": 0.0, "c": ox, "r": 2.0 * ox}[ax]
        oy = {"t": 0.0, "m": oy, "b": 2.0 * oy}[ay]
    x0, x1 = ox + fx0 * vw, ox + fx1 * vw
    y0, y1 = oy + fy0 * vh, oy + fy1 * vh
    if pad:
        px, py = (x1 - x0) * pad, (y1 - y0) * pad
        x0, x1, y0, y1 = x0 - px, x1 + px, y0 - py, y1 + py
    ix0 = min(max(math.floor(x0 + _EPS), 0), w - 1)
    iy0 = min(max(math.floor(y0 + _EPS), 0), h - 1)
    ix1 = max(min(math.ceil(x1 - _EPS), w), ix0 + 1)
    iy1 = max(min(math.ceil(y1 - _EPS), h), iy0 + 1)
    return (ix0, iy0, ix1, iy1)


def crop(img: Image.Image, name: str, pad: float = 0.0, *, layout: Optional[str] = None) -> Image.Image:
    """Crop region ``name`` out of a full game frame (see ``region_box`` for ``pad``)."""
    return img.crop(region_box(img.size, name, pad=pad, layout=layout))


def all_boxes(size: tuple[int, int], *, layout: Optional[str] = None) -> dict[str, tuple[int, int, int, int]]:
    """Pixel boxes of every region for a frame size (handy for debugging / calibration)."""
    return {name: region_box(size, name, layout=layout) for name in REGIONS}
