"""Optional OCR fast path (RapidOCR, ONNX runtime, CPU): stage, gold, level, XP, shop names.

Free and local: a few hundred milliseconds per frame on a fast desktop CPU (up
to a few seconds on a slow or busy one, it runs about 8 small OCR passes), so
it is a good way to re-read the shop after every roll or to cross-check
Claude's numbers. It only returns fields it read confidently; everything else
stays ``None``.

Install with ``pip install -e ".[ocr]"``: ``rapidocr-onnxruntime`` on Python
< 3.13, its successor ``rapidocr`` 3.x (plus ``onnxruntime``) from 3.13 on.
"""

from __future__ import annotations

import difflib
import re
import threading
import time
import unicodedata
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional

from ..data.setdata import normalize_name
from ..engine.tracker import is_wisp_name
from ..models import Observation, ScreenObservation, ScreenType, ShopSlot, StageRound
from .base import FALLBACK_REGIONS, PerceptionError, PerceptionHint, _fallback_box, crop_region, normalize_purpose

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image

    from ..data.setdata import Champion, SetData

OcrEngine = Callable[[Any], Any]


class OcrLine(NamedTuple):
    """One text box recognized by the OCR engine (pixel values in the engine's input image)."""

    text: str
    score: float  # confidence 0..1
    height: float  # box height
    cy: float  # box center y
    cx: float  # box center x

MIN_CONFIDENCE = 0.5
# An unknown name on the rightmost card is only taken for a Set 18 Wisp when read this clearly.
WISP_MIN_CONFIDENCE = 0.8
# Below this similarity to every champion name an unknown text is not a misread champion
# (one wrong, missing or extra letter of any Set 18 name stays above it; CJK names are
# 2 to 4 characters, so one wrong character of two is 0.5).
_CHAMPION_NEAR_MISS = 0.6
_CHAMPION_NEAR_MISS_CJK = 0.5
_WISP_NAME_MAX = 16


def _rapidocr_class() -> tuple[Any, dict[str, Any]]:
    """The RapidOCR engine class and its constructor kwargs:
    ``rapidocr_onnxruntime`` (Python < 3.13) or its successor ``rapidocr`` 3.x
    (needs ``onnxruntime``). Raises when neither can be imported."""
    try:
        from rapidocr_onnxruntime import RapidOCR  # lazy optional dependency

        return RapidOCR, {}
    except Exception as exc:  # noqa: BLE001 - ImportError, DLL load errors, ...
        first: Exception = exc
    try:
        import onnxruntime  # noqa: F401  (rapidocr 3.x does not pull an inference engine)
        from rapidocr import RapidOCR
    except Exception:  # noqa: BLE001
        raise first from None
    # rapidocr 3.x prints model paths at INFO on every engine start and a
    # warning for every crop without text (most of them): errors only.
    return RapidOCR, {"params": {"Global.log_level": "error"}}


def ocr_available() -> bool:
    """True when a RapidOCR engine (``rapidocr_onnxruntime`` or ``rapidocr`` 3.x) can be imported."""
    try:
        _rapidocr_class()
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# text parsing helpers (pure, tested without OCR)
# ---------------------------------------------------------------------------

# Characters OCR commonly returns instead of the hyphen in "3-2" (incl. the CJK "一").
_DASHES = "-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uff0d\u4e00~\uff5e_:\uff1a"
_STAGE_RE = re.compile(r"(?<!\d)([1-9])\s*[" + re.escape(_DASHES) + r"]\s*([1-9])(?!\d)")
# Letter / digit confusions that are safe inside a purely numeric field.
_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "|": "1", "i": "1", "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2"})
_LEVEL_RES = (
    re.compile(r"(?:level|l\s*v\s*l?|等级|等級)\s*[.:：。,，]?\s*(\d{1,2})(?!\d)", re.I),
    re.compile(r"(?<!\d)(\d{1,2})\s*[级級]"),
)
_XP_RE = re.compile(r"(?<!\d)(\d{1,3})\s*[/／]\s*(\d{1,3})(?!\d)")


def _nfkc(text: Optional[str]) -> str:
    return unicodedata.normalize("NFKC", text or "").strip()


def parse_stage_text(text: Optional[str]) -> Optional[str]:
    """``"3-2"``, ``"阶段 3 - 2"``, ``"3一2"``, ``"3–2"`` -> ``"3-2"``; anything else -> ``None``."""
    t = _nfkc(text)
    if not t:
        return None
    m = _STAGE_RE.search(t)
    if not m:
        return None
    sr = StageRound.parse(f"{m.group(1)}-{m.group(2)}")
    return f"{sr.stage}-{sr.round}" if sr else None


def parse_gold_text(text: Optional[str], max_gold: int = 300) -> Optional[int]:
    """The gold counter: a single 1-3 digit number (OCR letter confusions fixed); ``None`` if implausible.

    Two separate numbers in one box (gold and the streak counter next to it,
    "12 3") are ambiguous and give ``None``: joining them would turn gold 12
    into a plausible but wrong 123.
    """
    t = _nfkc(text)
    if not t:
        return None
    # Only fix letters when the token is essentially numeric (e.g. "4O", "l5").
    compact = t.strip()
    if re.fullmatch(r"[0-9OoDQlIi|SsBZz]{1,3}", compact):
        compact = compact.translate(_DIGIT_FIX)
        return int(compact) if compact.isdigit() and int(compact) <= max_gold else None
    numbers = re.findall(r"(?<!\d)(\d+)(?!\d)", t)
    if len(numbers) != 1 or len(numbers[0]) > 4:
        return None
    value = int(numbers[0])
    return value if 0 <= value <= max_gold else None


def parse_level_text(text: Optional[str]) -> Optional[int]:
    """``"Lv. 7"``, ``"LV7"``, ``"Level 7"``, ``"等级 7"``, ``"7级"`` -> 7 (valid range 1-10)."""
    t = _nfkc(text)
    if not t:
        return None
    # "Iv. 7" / "1v. 7": OCR often reads the capital L as I or 1.
    t = re.sub(r"(?<![0-9A-Za-z])[I1|](?=\s*[vV][.\s:：]*\d)", "L", t)
    for rx in _LEVEL_RES:
        m = rx.search(t)
        if m:
            value = int(m.group(1))
            return value if 1 <= value <= 10 else None
    return None


def parse_xp_text(text: Optional[str]) -> Optional[tuple[int, int]]:
    """``"20/48"`` -> ``(20, 48)``; ``None`` when absent or implausible (current > needed)."""
    t = _nfkc(text)
    m = _XP_RE.search(t)
    if not m:
        return None
    cur, need = int(m.group(1)), int(m.group(2))
    if need <= 0 or need > 200 or cur > need:
        return None
    return cur, need


# ---------------------------------------------------------------------------
# image preprocessing
# ---------------------------------------------------------------------------


def preprocess_for_ocr(img: "Image.Image", target_height: int = 96) -> "Image.Image":
    """Upscale x2-3, grayscale, stretch contrast; invert light-on-dark text (game HUD) to dark-on-light."""
    from PIL import Image, ImageOps, ImageStat

    w, h = img.size
    factor = 3 if h * 2 < target_height else 2
    out = img.convert("L").resize((max(1, w * factor), max(1, h * factor)), Image.LANCZOS)
    out = ImageOps.autocontrast(out, cutoff=1)
    if ImageStat.Stat(out).mean[0] < 110:
        out = ImageOps.invert(out)
    return out.convert("RGB")


def _box_height(box: Any) -> float:
    try:
        ys = [float(pt[1]) for pt in box]
        return max(ys) - min(ys)
    except Exception:
        return 0.0


def _box_center(box: Any) -> tuple[float, float]:
    try:
        xs = [float(pt[0]) for pt in box]
        ys = [float(pt[1]) for pt in box]
        return (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    except Exception:
        return 0.0, 0.0


def normalize_engine_output(raw: Any) -> list[OcrLine]:
    """RapidOCR ``(result, elapse)`` (rapidocr-onnxruntime) or a ``RapidOCROutput``
    (rapidocr 3.x: ``.boxes`` / ``.txts`` / ``.scores``) -> ``OcrLine`` list
    sorted top to bottom, then left to right."""
    if not isinstance(raw, (tuple, list)) and hasattr(raw, "txts"):
        boxes, txts, scores = getattr(raw, "boxes", None), raw.txts, getattr(raw, "scores", None)
        result: Any = [] if boxes is None or txts is None or scores is None else list(zip(boxes, txts, scores))
    else:
        result = raw[0] if isinstance(raw, tuple) else raw
    lines: list[OcrLine] = []
    for entry in result or []:
        try:
            box, text, score = entry[0], entry[1], entry[2]
            cx, cy = _box_center(box)
            lines.append(OcrLine(str(text), float(score), _box_height(box), cy, cx))
        except Exception:
            continue
    lines.sort(key=lambda ln: (ln.cy, ln.cx))
    return lines


def _row_joins(lines: list[OcrLine]) -> list[str]:
    """Texts of boxes on the same row joined left to right ("Miss" + "Fortune"), bottom row first."""
    rows: list[list[OcrLine]] = []
    for ln in sorted(lines, key=lambda b: b.cy):
        if rows and abs(ln.cy - rows[-1][-1].cy) <= max(ln.height, rows[-1][-1].height, 1.0) * 0.5:
            rows[-1].append(ln)
        else:
            rows.append([ln])
    out: list[str] = []
    for row in reversed(rows):
        if len(row) > 1:
            words = [b.text.strip() for b in sorted(row, key=lambda b: b.cx)]
            out.append(" ".join(words))
            if len(words) > 2:  # name + cost digit on the same row: drop the rightmost box
                out.append(" ".join(words[:-1]))
    return out


# ---------------------------------------------------------------------------
# shop card geometry
# ---------------------------------------------------------------------------

# The 5-card strip in a 16:9 frame (fractions). The "shop" HUD region is padded
# and also contains the Buy XP / Reroll buttons, so splitting it in 5 would
# misalign the cards. Two placements are known (see capture/regions.py notes):
# the standard HUD (Set 18) and the "Trials" HUD shifted left (also close to the
# legacy client).
SHOP_CARD_STRIPS: dict[str, tuple[float, float, float, float]] = {
    "standard": (0.290, 0.855, 0.808, 0.998),
    "trials": (0.249, 0.855, 0.777, 0.998),
}
CARD_LEFT_PAD = 0.10  # names can start slightly left of the card art (fraction of a card width)


def _shop_reference(size: tuple[int, int]) -> tuple[tuple[int, int, int, int], tuple[float, float, float, float]]:
    """Pixel box of the "shop" region and the fractions it was made from (same source)."""
    try:
        from ..capture import regions as _regions  # lazy: written by the capture module

        frac = tuple(float(v) for v in _regions.REGIONS["shop"])
        box = tuple(int(v) for v in _regions.region_box(size, "shop"))
        if len(frac) == 4 and len(box) == 4 and frac[2] > frac[0] and frac[3] > frac[1]:
            return box, frac  # type: ignore[return-value]
    except Exception:
        pass
    return _fallback_box(size, "shop"), FALLBACK_REGIONS["shop"]


def shop_card_boxes(size: tuple[int, int], placement: str = "standard") -> list[tuple[int, int, int, int]]:
    """Pixel boxes of the 5 shop cards (left to right) for a frame size and HUD placement.

    Expressed relative to the "shop" region so the capture module's viewport /
    layout handling (ultrawide, 16:10) carries over.
    """
    (bx0, by0, bx1, by1), (fx0, fy0, fx1, fy1) = _shop_reference(size)
    cx0, cy0, cx1, cy1 = SHOP_CARD_STRIPS[placement]
    sx = (bx1 - bx0) / (fx1 - fx0)
    sy = (by1 - by0) / (fy1 - fy0)
    left, right = bx0 + (cx0 - fx0) * sx, bx0 + (cx1 - fx0) * sx
    top, bottom = by0 + (cy0 - fy0) * sy, by0 + (cy1 - fy0) * sy
    pitch = (right - left) / 5
    w, h = int(size[0]), int(size[1])
    boxes = []
    for i in range(5):
        x0 = left + i * pitch - CARD_LEFT_PAD * pitch
        x1 = left + (i + 1) * pitch
        box = (max(0, int(x0)), max(0, int(top)), min(w, int(round(x1))), min(h, int(round(bottom))))
        boxes.append((box[0], box[1], max(box[2], box[0] + 1), max(box[3], box[1] + 1)))
    return boxes


def _is_cjk(ch: str) -> bool:
    return (
        "\u4e00" <= ch <= "\u9fff"  # CJK unified ideographs
        or "\u3400" <= ch <= "\u4dbf"  # extension A
        or "\u3040" <= ch <= "\u30ff"  # kana
        or "\uac00" <= ch <= "\ud7a3"  # hangul syllables
    )


def _is_meaningful(text: str) -> bool:
    """Card text rather than a cost digit / stray mark: 2+ letters, or any CJK character.

    Chinese champion names can be a single character (慎, 劫, 烬, 蔚), so one
    CJK character counts; an unresolvable one makes the shop untrusted (the
    caller falls back to Claude) instead of silently reporting an empty slot.
    """
    t = _nfkc(text)
    if any(_is_cjk(ch) for ch in t):
        return True
    return sum(1 for ch in t if ch.isalpha()) >= 2


def _bottom_row(lines: list[OcrLine]) -> list[OcrLine]:
    """Boxes on the lowest text row of a card, left to right."""
    low = max(lines, key=lambda ln: ln.cy)
    row = [ln for ln in lines if abs(ln.cy - low.cy) <= max(ln.height, low.height, 1.0) * 0.5]
    return sorted(row, key=lambda ln: ln.cx)


def _strip_wisp_label(text: str) -> str:
    """'精灵 免费刷新' / 'Wisp: Grow Up' -> 'Grow Up' style Wisp name ('' when only the label was read).

    The label is the shortest head ``is_wisp_name`` accepts, so the prefixes
    live in one place; 'Wispy' keeps its letters.
    """
    t = text.strip()
    if not is_wisp_name(t):
        return t
    for i in range(1, len(t) + 1):
        if is_wisp_name(t[:i]):
            if t[i:i + 1].isascii() and t[i:i + 1].isalpha():
                return t
            return t[i:].strip(" :：·・-_|")
    return t


# ---------------------------------------------------------------------------
# perceiver
# ---------------------------------------------------------------------------


class OcrPerceiver:
    """Fast partial reader: stage, gold, level (+XP) and shop champion names
    (a Set 18 Wisp on the rightmost card as ``"Wisp: <name>"``, price unknown).

    ``engine`` is injectable (any callable taking an RGB ``numpy`` array and
    returning RapidOCR-style output) so the pipeline is testable offline.
    """

    def __init__(self, set_data: "SetData", name: str = "ocr", engine: Optional[OcrEngine] = None,
                 min_confidence: float = MIN_CONFIDENCE) -> None:
        self.set_data = set_data
        self.name = name
        self.min_confidence = min_confidence
        self._engine = engine
        self._lock = threading.Lock()
        # Placement that worked last is tried first (the HUD does not move within a game).
        self._placements: list[str] = list(SHOP_CARD_STRIPS)
        self._champion_keys: Optional[list[str]] = None  # normalized champion names (Wisp check)

    # ---- engine -------------------------------------------------------------
    def _get_engine(self) -> OcrEngine:
        if self._engine is None:
            with self._lock:  # two threads must not both load the ONNX models
                if self._engine is None:
                    try:
                        RapidOCR, kwargs = _rapidocr_class()
                    except Exception as exc:
                        raise PerceptionError("OCR 不可用") from exc
                    try:
                        self._engine = RapidOCR(**kwargs)
                    except Exception as exc:
                        raise PerceptionError(f"OCR 不可用: {exc}") from exc
        return self._engine

    def read_lines(self, img: "Image.Image") -> list[OcrLine]:
        """OCR one (already cropped) image; returns confident lines top to bottom."""
        import numpy as np

        engine = self._get_engine()
        arr = np.asarray(preprocess_for_ocr(img))
        with self._lock:  # onnxruntime sessions are not guaranteed thread-safe
            raw = engine(arr)
        return [ln for ln in normalize_engine_output(raw) if ln.score >= self.min_confidence and ln.text.strip()]

    # ---- fields -------------------------------------------------------------
    def read_stage(self, frame: "Image.Image") -> Optional[str]:
        for ln in self.read_lines(crop_region(frame, "stage")):
            stage = parse_stage_text(ln.text)
            if stage:
                return stage
        return None

    def read_gold(self, frame: "Image.Image") -> Optional[int]:
        # The streak counter sits next to the gold in a smaller font: prefer the tallest number.
        lines = sorted(self.read_lines(crop_region(frame, "gold")), key=lambda ln: -ln.height)
        for ln in lines:
            gold = parse_gold_text(ln.text)
            if gold is not None:
                return gold
        return None

    def read_level(self, frame: "Image.Image") -> tuple[Optional[int], Optional[tuple[int, int]]]:
        # The "xp" region covers the "Lvl. 6" label and the "20/36" XP text ("level" only the label).
        lines = self.read_lines(crop_region(frame, "xp"))
        level: Optional[int] = None
        xp: Optional[tuple[int, int]] = None
        for ln in lines:
            if level is None:
                level = parse_level_text(ln.text)
            if xp is None:
                xp = parse_xp_text(ln.text)
        if level is None:  # "Lv." and "7" detected as separate boxes on one row
            for joined in _row_joins(lines):
                level = parse_level_text(joined)
                if level is not None:
                    break
        return level, xp

    def _match_card(self, lines: list[OcrLine]) -> Optional["Champion"]:
        # The champion name is at the bottom of the card (trait names are above it): bottom lines
        # first, then boxes of one row joined together for multi-word names split by the detector.
        candidates = [ln.text for ln in sorted(lines, key=lambda b: -b.cy)] + _row_joins(lines)
        for text in candidates:
            champ = self.set_data.resolve_champion(text)
            if champ is not None:
                return champ
            # "Garen 1" / "盖伦 1": strip a trailing cost digit.
            stripped = re.sub(r"[\s\d]+$", "", _nfkc(text))
            if stripped and stripped != text:
                champ = self.set_data.resolve_champion(stripped)
                if champ is not None:
                    return champ
        return None

    def _near_champion(self, key: str) -> bool:
        """``key`` (normalized) is a fragment or a near miss of a champion name: a misread, not a Wisp."""
        if self._champion_keys is None:
            self._champion_keys = sorted({normalize_name(n) for n in self.set_data.champion_names()} - {""})
        keys = self._champion_keys
        cutoff = _CHAMPION_NEAR_MISS_CJK if any(_is_cjk(ch) for ch in key) else _CHAMPION_NEAR_MISS
        return any(key in k for k in keys) or bool(difflib.get_close_matches(key, keys, n=1, cutoff=cutoff))

    def _wisp_slot(self, lines: list[OcrLine]) -> Optional[ShopSlot]:
        """The rightmost card read as a Set 18 Wisp, or ``None`` when it may be a misread champion.

        In every other shop the rightmost card is a Wisp with its own name, which
        never resolves to a champion. Unlabeled text is only taken for a Wisp
        when it cannot be a champion card whose name was misread: no trait label
        on the card, read clearly, and neither a fragment nor a near miss of a
        champion name. Otherwise the shop stays untrusted (Claude fallback).
        The price is left unknown: a lone digit may belong to the neighbour card.
        """
        if any(self.set_data.resolve_trait(ln.text) is not None for ln in lines):
            return None  # trait labels: a champion card whose name was not read
        row = _bottom_row(lines)
        text = re.sub(r"[\s\d]+$", "", _nfkc(" ".join(ln.text.strip() for ln in row)))  # price digits
        if not any(is_wisp_name(_nfkc(ln.text)) for ln in lines):  # not labeled as a Wisp
            key = normalize_name(text)
            if min(ln.score for ln in row) < WISP_MIN_CONFIDENCE:
                return None
            if len(key) < (2 if any(_is_cjk(ch) for ch in key) else 3) or self._near_champion(key):
                return None
        name = _strip_wisp_label(text) or text
        return ShopSlot(name=f"Wisp: {name[:_WISP_NAME_MAX]}", cost=None) if name else None

    def _read_cards(self, frame: "Image.Image", placement: str) -> tuple[Optional[list[ShopSlot]], Optional[str]]:
        """Read 5 cards for one HUD placement: (slots or None, problem note or None)."""
        slots: list[ShopSlot] = []
        resolved = 0
        boxes = shop_card_boxes(frame.size, placement)
        for i, box in enumerate(boxes):
            lines = [ln for ln in self.read_lines(frame.crop(box)) if _is_meaningful(ln.text)]
            if not lines:
                slots.append(ShopSlot())  # nothing written on the card: bought / empty slot
                continue
            champ = self._match_card(lines)
            if champ is None and i == len(boxes) - 1:
                wisp = self._wisp_slot(lines)  # Wisps only ever take the rightmost slot
                if wisp is not None:
                    slots.append(wisp)
                    continue
            if champ is None:
                # One unreadable card makes the whole shop untrustworthy (stop early, save time).
                return None, f"商店第 {i + 1} 格无法识别 ({_nfkc(lines[-1].text)[:12]})"
            slots.append(ShopSlot(name=champ.name, cost=champ.cost))
            resolved += 1
        return (slots, None) if resolved else (None, None)

    def read_shop(self, frame: "Image.Image", notes: list[str]) -> Optional[list[ShopSlot]]:
        problem: Optional[str] = None
        with self._lock:
            order = list(self._placements)
        for placement in order:
            slots, note = self._read_cards(frame, placement)
            if slots is not None:
                with self._lock:
                    if self._placements[0] != placement:
                        self._placements = [placement] + [x for x in self._placements if x != placement]
                return slots
            problem = problem or note
        if problem:
            notes.append(problem)
        return None

    # ---- main ---------------------------------------------------------------
    def read_screen(self, image: "Image.Image", purpose: str = "auto") -> ScreenObservation:
        purpose = normalize_purpose(purpose)
        frame = image if image.mode == "RGB" else image.convert("RGB")
        notes: list[str] = []
        screen = ScreenObservation(notes=notes)
        if purpose != "shop":
            screen.stage = self.read_stage(frame)
        screen.gold = self.read_gold(frame)
        level, xp = self.read_level(frame)
        screen.level = level
        if xp is not None:
            screen.xp_current, screen.xp_needed = xp
        screen.shop = self.read_shop(frame, notes)
        if screen.shop is not None and sum(1 for s in screen.shop if s.name) >= 3:
            screen.screen_type = ScreenType.PLANNING
        screen.notes = notes
        return screen

    def perceive(self, image: "Image.Image", purpose: str = "auto", hint: Optional[PerceptionHint] = None) -> Observation:
        if image is None or not hasattr(image, "crop"):
            raise PerceptionError("没有可识别的截图")
        purpose = normalize_purpose(purpose)
        self._get_engine()  # raise PerceptionError('OCR 不可用') early
        captured_at = time.time()
        started = time.monotonic()
        try:
            screen = self.read_screen(image, purpose)
        except PerceptionError:
            raise
        except Exception as exc:
            raise PerceptionError(f"OCR 识别失败: {exc}") from exc
        # Raise instead of returning an empty result so callers can fall back to Claude.
        if purpose == "shop" and screen.shop is None:
            raise PerceptionError("OCR 没有读到商店" + (f": {screen.notes[0]}" if screen.notes else ""))
        if all(getattr(screen, f) is None for f in ("stage", "gold", "level", "shop")):
            raise PerceptionError("OCR 没有读到任何内容")
        return Observation(
            screen=screen,
            captured_at=captured_at,
            source="ocr",
            purpose=purpose,
            latency_s=round(time.monotonic() - started, 3),
        )


__all__ = [
    "OcrLine",
    "OcrPerceiver",
    "normalize_engine_output",
    "ocr_available",
    "parse_gold_text",
    "parse_level_text",
    "parse_stage_text",
    "parse_xp_text",
    "preprocess_for_ocr",
    "shop_card_boxes",
]
