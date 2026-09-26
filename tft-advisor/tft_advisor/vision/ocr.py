"""Optional OCR fast path (RapidOCR, ONNX runtime, CPU): stage, gold, level, XP, shop names.

About 100-300 ms per frame on a desktop CPU and free, so it is a good way to
re-read the shop after every roll or to cross-check Claude's numbers. It only
returns fields it read confidently; everything else stays ``None``.

Install with ``pip install rapidocr-onnxruntime`` (``pip install tft-advisor[ocr]``).
"""

from __future__ import annotations

import re
import threading
import time
import unicodedata
from typing import TYPE_CHECKING, Any, Callable, Optional

from ..models import Observation, ScreenObservation, ScreenType, ShopSlot, StageRound
from .base import PerceptionError, PerceptionHint, crop_region

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image

    from ..data.setdata import Champion, SetData

# A text line recognized by the OCR engine: (text, confidence 0..1, box height px).
TextLine = tuple[str, float, float]
OcrEngine = Callable[[Any], Any]

MIN_CONFIDENCE = 0.5


def ocr_available() -> bool:
    """True when ``rapidocr_onnxruntime`` can be imported."""
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# text parsing helpers (pure, tested without OCR)
# ---------------------------------------------------------------------------

# Characters OCR commonly returns instead of the hyphen in "3-2" (incl. the CJK "一").
_DASHES = "-‐‑‒–—―−－一~～_:："
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
    """The gold counter: the first 1-3 digit number (OCR letter confusions fixed); ``None`` if implausible."""
    t = _nfkc(text)
    if not t:
        return None
    # Only fix letters when the text is essentially numeric (e.g. "4O", "l5").
    compact = re.sub(r"\s+", "", t)
    if re.fullmatch(r"[0-9OoDQlIi|SsBZz]{1,3}", compact):
        compact = compact.translate(_DIGIT_FIX)
        return int(compact) if compact.isdigit() and int(compact) <= max_gold else None
    m = re.search(r"(?<!\d)(\d{1,4})(?!\d)", t)
    if not m:
        return None
    value = int(m.group(1))
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


def _box_center_y(box: Any) -> float:
    try:
        ys = [float(pt[1]) for pt in box]
        return (max(ys) + min(ys)) / 2
    except Exception:
        return 0.0


def normalize_engine_output(raw: Any) -> list[tuple[str, float, float, float]]:
    """RapidOCR ``(result, elapse)`` -> [(text, score, box_height, center_y)] sorted top to bottom."""
    result = raw[0] if isinstance(raw, tuple) else raw
    lines: list[tuple[str, float, float, float]] = []
    for entry in result or []:
        try:
            box, text, score = entry[0], entry[1], entry[2]
            lines.append((str(text), float(score), _box_height(box), _box_center_y(box)))
        except Exception:
            continue
    lines.sort(key=lambda x: x[3])
    return lines


# ---------------------------------------------------------------------------
# perceiver
# ---------------------------------------------------------------------------


class OcrPerceiver:
    """Fast partial reader: stage, gold, level (+XP) and shop champion names.

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

    # ---- engine -------------------------------------------------------------
    def _get_engine(self) -> OcrEngine:
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR  # lazy optional dependency
            except Exception as exc:
                raise PerceptionError("OCR 不可用") from exc
            try:
                self._engine = RapidOCR()
            except Exception as exc:
                raise PerceptionError(f"OCR 不可用: {exc}") from exc
        return self._engine

    def read_lines(self, img: "Image.Image") -> list[tuple[str, float, float, float]]:
        """OCR one (already cropped) image; returns confident lines top to bottom."""
        import numpy as np

        engine = self._get_engine()
        arr = np.asarray(preprocess_for_ocr(img))
        with self._lock:  # onnxruntime sessions are not guaranteed thread-safe
            raw = engine(arr)
        return [ln for ln in normalize_engine_output(raw) if ln[1] >= self.min_confidence and ln[0].strip()]

    # ---- fields -------------------------------------------------------------
    def read_stage(self, frame: "Image.Image") -> Optional[str]:
        for text, _score, _h, _y in self.read_lines(crop_region(frame, "stage")):
            stage = parse_stage_text(text)
            if stage:
                return stage
        return None

    def read_gold(self, frame: "Image.Image") -> Optional[int]:
        # The streak counter sits next to the gold in a smaller font: prefer the tallest number.
        lines = sorted(self.read_lines(crop_region(frame, "gold")), key=lambda ln: -ln[2])
        for text, _score, _h, _y in lines:
            gold = parse_gold_text(text)
            if gold is not None:
                return gold
        return None

    def read_level(self, frame: "Image.Image") -> tuple[Optional[int], Optional[tuple[int, int]]]:
        lines = self.read_lines(crop_region(frame, "level"))
        level: Optional[int] = None
        xp: Optional[tuple[int, int]] = None
        for text, _score, _h, _y in lines:
            if level is None:
                level = parse_level_text(text)
            if xp is None:
                xp = parse_xp_text(text)
        if level is None and len(lines) > 1:
            level = parse_level_text(" ".join(ln[0] for ln in lines))
        return level, xp

    def _match_card(self, lines: list[tuple[str, float, float, float]]) -> Optional["Champion"]:
        # The champion name is at the bottom of the card (trait names are above it): bottom lines first.
        candidates = [ln[0] for ln in reversed(lines)]
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

    def read_shop(self, frame: "Image.Image", notes: list[str]) -> Optional[list[ShopSlot]]:
        shop_img = crop_region(frame, "shop")
        w, h = shop_img.size
        if w < 5 or h < 2:
            return None
        slots: list[ShopSlot] = []
        resolved = 0
        for i in range(5):
            card = shop_img.crop((round(i * w / 5), 0, round((i + 1) * w / 5), h))
            lines = self.read_lines(card)
            if not lines:
                slots.append(ShopSlot())  # nothing written on the card: bought / empty slot
                continue
            champ = self._match_card(lines)
            if champ is None:
                notes.append(f"商店第 {i + 1} 格无法识别 ({_nfkc(lines[-1][0])[:12]})")
                return None  # one unreadable card makes the whole shop untrustworthy
            slots.append(ShopSlot(name=champ.name, cost=champ.cost))
            resolved += 1
        return slots if resolved else None

    # ---- main ---------------------------------------------------------------
    def read_screen(self, image: "Image.Image", purpose: str = "auto") -> ScreenObservation:
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
        self._get_engine()  # raise PerceptionError('OCR 不可用') early
        captured_at = time.time()
        started = time.monotonic()
        try:
            screen = self.read_screen(image, purpose)
        except PerceptionError:
            raise
        except Exception as exc:
            raise PerceptionError(f"OCR 识别失败: {exc}") from exc
        return Observation(
            screen=screen,
            captured_at=captured_at,
            source="ocr",
            purpose=purpose or "auto",
            latency_s=round(time.monotonic() - started, 3),
        )


__all__ = [
    "OcrPerceiver",
    "normalize_engine_output",
    "ocr_available",
    "parse_gold_text",
    "parse_level_text",
    "parse_stage_text",
    "parse_xp_text",
    "preprocess_for_ocr",
]
