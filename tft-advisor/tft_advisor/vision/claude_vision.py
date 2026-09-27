"""Claude vision perceiver: screenshot -> ``ScreenObservation`` via structured output.

Per call we send the full frame (downscaled) plus crops of the same frame at
native resolution: the number-heavy HUD parts (bottom HUD, player list, stage),
the board with the bench (star pips, item icons under units), the trait panel
and the item column, and for scouting the top banner. Each image is preceded
by a text label so the model knows what it sees. For ``purpose="shop"`` only
the bottom HUD crop is sent (fast and cheap).

The large system prompt (HUD guide + set name lists) is identical for every call
of a set, so the prompt cache in ``tft_advisor.llm`` makes repeated calls cheap.
"""

from __future__ import annotations

import io
import re
import time
import unicodedata
from typing import TYPE_CHECKING, Any, Optional

from pydantic import ValidationError

from ..config import AnthropicConfig
from ..engine.tracker import is_wisp_name
from ..llm import LLM, LLMError, image_block, text_block
from ..models import Observation, ScreenObservation, ScreenObservationWire, StageRound, UnitObs
from .base import (
    UNKNOWN_ITEM,
    PerceptionError,
    PerceptionHint,
    clean_name,
    clean_text,
    crop_region,
    is_unknown_item,
    normalize_purpose,
    region_box,
)
from .prompts import build_user_text, build_vision_system

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image

    from ..data.setdata import SetData

# Crops for the non-shop purposes: (region names, label shown to the model, JPEG).
# Several region names mean "crop their union". The board with the bench is a
# busy 3D scene: JPEG at high quality with full color resolution keeps the star
# pip colors and item icons while staying a few hundred KB (PNG would be MBs);
# the text-heavy crops stay lossless PNG.
_CROPS_DEFAULT: tuple[tuple[tuple[str, ...], str, bool], ...] = (
    (("hud_bottom",), "bottom HUD crop (shop, gold, level, XP, streak) of the LOCAL player", False),
    (("players",), "player list crop (names and HP, right side)", False),
    (("stage",), "stage / round indicator crop (top center)", False),
    (
        ("board", "bench"),
        "board and bench crop (units, star pips above the health bars, item icons under them) "
        "of the player whose board is shown",
        True,
    ),
    (("traits",), "trait panel crop (left side) of the player whose board is shown", False),
    (("items",), "item bench crop (column of unequipped items at the far left) of the player whose board is shown", False),
)
_CROP_SCOUT_BANNER = (("top_banner",), "top banner crop (name of the board owner while scouting)", False)
_SHOP_LABEL = "bottom HUD crop (shop, gold, level, XP, streak) of the LOCAL player"
_CROP_JPEG_QUALITY = 92

# Opus 4.7+ / Sonnet 5 read up to 2576 px on the long edge; anything larger is
# resized by the API anyway, so sending more only costs upload time.
_API_MAX_EDGE = 2576
_DEFAULT_EDGE = AnthropicConfig.max_image_edge

# Anthropic's per-image limit is 5 MB of base64; stay well below it.
_MAX_IMAGE_BYTES = 3_500_000

# Fields the model may fill from a bottom-HUD-only request.
_SHOP_FIELDS = {"screen_type", "shop", "shop_locked", "gold", "level", "xp_current", "xp_needed", "streak", "notes"}

_STAGE_STRICT = re.compile(r"^\d-\d$")
# Hyphen look-alikes the model may copy from the HUD font ("3\u20132", "3 \uff0d 2").
# A colon is NOT accepted: "2:05" style text is a timer, not a stage.
_STAGE_LENIENT = re.compile(r"^\s*([1-9])\s*[-\u2010-\u2015\u2212\ufe63\uff0d]\s*([1-9])\s*$")

# A TFT board has 4 x 7 hexes, the bench 9 slots, a lobby 8 players: more is a misreading.
_MAX_BOARD_UNITS = 28
_MAX_BENCH_UNITS = 9
_MAX_PLAYERS = 8

# Frames smaller than this (minimized window, bad crop) are not worth an API call.
_MIN_FRAME_EDGE = 64


# ---------------------------------------------------------------------------
# image helpers
# ---------------------------------------------------------------------------


def _image_edge(configured: Any) -> int:
    """Configured long-edge limit, clamped: <= 0 / garbage means the config default, never above 2576 px."""
    try:
        edge = int(configured)
    except (TypeError, ValueError):
        edge = 0
    return min(edge, _API_MAX_EDGE) if edge > 0 else min(_DEFAULT_EDGE, _API_MAX_EDGE)


def _to_rgb(img: "Image.Image") -> "Image.Image":
    return img if img.mode == "RGB" else img.convert("RGB")


def fit_long_edge(img: "Image.Image", max_edge: int) -> "Image.Image":
    """Downscale so the long edge is <= ``max_edge`` (never upscales)."""
    from PIL import Image

    w, h = img.size
    long_edge = max(w, h)
    if max_edge <= 0 or long_edge <= max_edge:
        return img
    scale = max_edge / long_edge
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def _union_box(size: tuple[int, int], regions: tuple[str, ...]) -> tuple[int, int, int, int]:
    """Pixel box covering every named HUD region (one crop instead of overlapping ones)."""
    boxes = [region_box(size, r) for r in regions]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def prepare_crop(img: "Image.Image", max_edge: int, small_edge: int = 500) -> "Image.Image":
    """Keep crops at native resolution; shrink when too large, 2x-upscale tiny ones (small text)."""
    from PIL import Image

    w, h = img.size
    long_edge = max(w, h)
    if long_edge > max_edge:
        return fit_long_edge(img, max_edge)
    if long_edge < small_edge and long_edge * 2 <= max_edge:
        return img.resize((w * 2, h * 2), Image.LANCZOS)
    return img


def encode_image(
    img: "Image.Image", optimize: bool = False, prefer_jpeg: bool = False, jpeg_quality: int = 90
) -> tuple[bytes, str]:
    """PNG (lossless, best for small HUD text); JPEG fallback if the PNG is too large for the API.

    ``prefer_jpeg`` is used for the downscaled full frame and the board crop:
    a noisy 3D scene is several MB as PNG but a few hundred KB as JPEG, which
    matters for upload latency every round. ``jpeg_quality`` >= 92 also keeps
    full color resolution (4:4:4) for small colored details like star pips.
    ``optimize`` makes PNGs only about 5 to 10 % smaller for about 3x the
    encoding time, so it is off on the real-time path.
    """
    img = _to_rgb(img)
    if prefer_jpeg:
        buf = io.BytesIO()
        extra = {"subsampling": 0} if jpeg_quality >= 92 else {}
        img.save(buf, format="JPEG", quality=jpeg_quality, **extra)
        if len(buf.getvalue()) <= _MAX_IMAGE_BYTES:
            return buf.getvalue(), "image/jpeg"
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=optimize)
    data = buf.getvalue()
    if len(data) <= _MAX_IMAGE_BYTES:
        return data, "image/png"
    for quality in (92, 85, 75):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        if len(data) <= _MAX_IMAGE_BYTES:
            break
    return data, "image/jpeg"


# ---------------------------------------------------------------------------
# post-processing
# ---------------------------------------------------------------------------


def _norm(text: object) -> str:
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


def _short_error(exc: BaseException, limit: int = 160) -> str:
    text = " ".join(str(exc).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def check_frame(image: "Image.Image") -> None:
    """Raise :class:`PerceptionError` for frames that cannot contain a readable HUD.

    Tiny frames (minimized window) and single-color frames (black capture of an
    exclusive-fullscreen game, a loading fade) would only waste an API call.
    """
    w, h = image.size
    if w < _MIN_FRAME_EDGE or h < _MIN_FRAME_EDGE:
        raise PerceptionError(f"截图尺寸异常 ({w}x{h})，游戏窗口可能被最小化")
    extrema = image.getextrema()
    if extrema and isinstance(extrema[0], (int, float)):  # single-band image
        extrema = (extrema,)
    if all(hi - lo <= 2 for lo, hi in extrema):
        raise PerceptionError("截图是纯色画面 (可能是黑屏或正在加载)，已跳过识别")


def _clean_units(units: Optional[list[UnitObs]], on_board: bool, notes: list[str]) -> Optional[list[UnitObs]]:
    if units is None:
        return None
    out: list[UnitObs] = []
    for u in units:
        if not (u.name or "").strip():
            continue
        u = u.model_copy()
        if u.star < 1 or u.star > 4:
            notes.append(f"{u.name} 的星级读数 {u.star} 不合理，按 1 星处理")
            u.star = 1
        if on_board:
            if u.row is not None and not 0 <= u.row <= 3:
                u.row = None
            if u.col is not None and not 0 <= u.col <= 6:
                u.col = None
        else:
            u.row = None
            if u.col is not None and not 0 <= u.col <= 8:
                u.col = None
        # One "?" per icon that could not be named: it still takes one of the 3 item slots.
        u.items = [UNKNOWN_ITEM if is_unknown_item(i) else i for i in u.items if (i or "").strip()][:3]
        out.append(u)
    limit = _MAX_BOARD_UNITS if on_board else _MAX_BENCH_UNITS
    if len(out) > limit:
        notes.append(f"{'棋盘' if on_board else '备战席'}读到 {len(out)} 个单位，超过上限，只保留前 {limit} 个")
        out = out[:limit]
    return out


def sanitize_screen(
    screen: ScreenObservation, purpose: str = "auto", hint: Optional[PerceptionHint] = None
) -> ScreenObservation:
    """Clamp obviously invalid readings to ``None`` and apply purpose-specific fixes.

    Returns a new object; the input is not modified.
    """
    mode = normalize_purpose(purpose)
    s = screen.model_copy(deep=True)
    notes = list(s.notes)

    if s.gold is not None and not 0 <= s.gold <= 300:
        notes.append(f"金币读数 {s.gold} 不合理，已忽略")
        s.gold = None
    if s.level is not None and not 1 <= s.level <= 10:
        notes.append(f"等级读数 {s.level} 不合理，已忽略")
        s.level = None
    if s.stage is not None:
        raw = s.stage
        m = _STAGE_LENIENT.match(unicodedata.normalize("NFKC", raw))
        parsed = StageRound.parse(f"{m.group(1)}-{m.group(2)}") if m else None
        s.stage = f"{parsed.stage}-{parsed.round}" if parsed else None
        if s.stage is None or not _STAGE_STRICT.match(s.stage):
            s.stage = None
            notes.append(f"回合读数 {raw!r} 无法识别，已忽略")
    if s.xp_current is not None and not 0 <= s.xp_current <= 200:
        s.xp_current = None
    if s.xp_needed is not None and not 1 <= s.xp_needed <= 200:
        s.xp_needed = None
    if s.xp_current is not None and s.xp_needed is not None and s.xp_current > s.xp_needed:
        notes.append(f"经验读数 {s.xp_current}/{s.xp_needed} 不合理，已忽略")
        s.xp_current = s.xp_needed = None
    if s.hp is not None and not 0 <= s.hp <= 200:
        s.hp = None
    if s.streak is not None and abs(s.streak) > 30:
        s.streak = None

    if s.shop is not None:
        if len(s.shop) > 5:
            notes.append(f"商店读到 {len(s.shop)} 个格子，只保留前 5 个")
            s.shop = s.shop[:5]
        for slot in s.shop:
            if slot.name is not None and not slot.name.strip():
                slot.name = None
            # Set 18 Wisps cost 0 up to a few dozen gold; champions 1 to 10.
            lo, hi = (0, 60) if is_wisp_name(slot.name) else (1, 10)
            if slot.cost is not None and not lo <= slot.cost <= hi:
                slot.cost = None

    s.board = _clean_units(s.board, on_board=True, notes=notes)
    s.bench = _clean_units(s.bench, on_board=False, notes=notes)

    s.viewed_player_name = clean_name(s.viewed_player_name)
    if s.players is not None:
        players = []
        for p in s.players:
            name = clean_name(p.name)
            if name:
                p.name = name
                players.append(p)
        if len(players) > _MAX_PLAYERS:
            notes.append(f"玩家列表读到 {len(players)} 行，只保留前 {_MAX_PLAYERS} 行")
            players = players[:_MAX_PLAYERS]
        for p in players:
            if p.hp is not None and not 0 <= p.hp <= 200:
                p.hp = None
        if hint is not None and hint.self_name and not any(p.is_self for p in players):
            key = _norm(hint.self_name)
            for p in players:
                if key and _norm(p.name) == key:
                    p.is_self = True
                    break
        selves = [p for p in players if p.is_self]
        if len(selves) > 1:  # at most one local player
            keep = selves[0]
            if hint is not None and hint.self_name:
                keep = next((p for p in selves if _norm(p.name) == _norm(hint.self_name)), keep)
            for p in selves:
                p.is_self = p is keep
        s.players = players

    if s.traits is not None:
        s.traits = [t for t in s.traits if (t.name or "").strip() and 0 <= t.count <= 30]

    if mode == "shop":
        # Only the bottom HUD was sent: anything else would be hallucinated.
        for field in ScreenObservation.model_fields:
            if field not in _SHOP_FIELDS:
                setattr(s, field, ScreenObservation.model_fields[field].get_default(call_default_factory=True))
    elif mode == "scout":
        if s.viewing_own_board is None:
            s.viewing_own_board = False
        named = clean_name(hint.scouting_player) if hint is not None else None
        if s.viewed_player_name is None and named:
            if s.viewing_own_board is True:
                # Pressed too early or the camera went back: this is the local
                # player's board, never file it under the named opponent.
                notes.append(f"画面仍是自己的棋盘，没有当作 {named} 的阵容")
            else:
                s.viewed_player_name = named
                notes.append("对手名字来自玩家指定，画面中未读到")

    s.notes = [clean_text(n) for n in notes if (n or "").strip()][:12]
    return s


# ---------------------------------------------------------------------------
# perceiver
# ---------------------------------------------------------------------------


class ClaudeVisionPerceiver:
    """Reads a TFT screenshot with Claude (structured output ``ScreenObservation``)."""

    def __init__(self, llm: LLM, cfg: AnthropicConfig, set_data: "SetData", name: str = "claude") -> None:
        self.llm = llm
        self.cfg = cfg
        self.set_data = set_data
        self.name = name
        self.png_optimize = False  # see encode_image: ~3x encoding time for 5 to 10 % smaller PNGs
        self.system = build_vision_system(set_data)

    # ---- images -------------------------------------------------------------
    def _image_specs(self, image: "Image.Image", purpose: str) -> list[tuple[str, "Image.Image", Optional[int]]]:
        """(label, image, JPEG quality or None for PNG) to send, in order."""
        mode = normalize_purpose(purpose)
        max_edge = _image_edge(self.cfg.max_image_edge)
        frame = _to_rgb(image)
        if mode == "shop":
            return [(_SHOP_LABEL, prepare_crop(crop_region(frame, "hud_bottom"), max_edge), None)]
        full = fit_long_edge(frame, max_edge)
        out: list[tuple[str, "Image.Image", Optional[int]]] = [
            (f"full screenshot, downscaled to {full.size[0]}x{full.size[1]}", full, 90)
        ]
        crops = list(_CROPS_DEFAULT)
        if mode == "scout":
            crops.append(_CROP_SCOUT_BANNER)
        for regions, label, jpeg in crops:
            crop = frame.crop(_union_box(frame.size, regions))
            out.append((label, prepare_crop(crop, max_edge), _CROP_JPEG_QUALITY if jpeg else None))
        return out

    def build_images(self, image: "Image.Image", purpose: str = "auto") -> list[tuple[str, "Image.Image"]]:
        """(label, image) pairs to send, in order. Labels do not include the ``IMAGE n:`` prefix."""
        return [(label, img) for label, img, _ in self._image_specs(image, purpose)]

    def build_content(
        self, image: "Image.Image", purpose: str = "auto", hint: Optional[PerceptionHint] = None
    ) -> list[dict[str, Any]]:
        specs = self._image_specs(image, purpose)
        content: list[dict[str, Any]] = []
        labels: list[str] = []
        for idx, (label, img, jpeg_quality) in enumerate(specs, start=1):
            full_label = f"IMAGE {idx}: {label}"
            labels.append(full_label)
            data, media_type = encode_image(
                img,
                optimize=self.png_optimize,
                prefer_jpeg=jpeg_quality is not None,
                jpeg_quality=jpeg_quality or 90,
            )
            content.append(text_block(full_label))
            content.append(image_block(data, media_type=media_type))
        content.append(text_block(build_user_text(purpose, hint, labels)))
        return content

    # ---- main ---------------------------------------------------------------
    def perceive(
        self, image: "Image.Image", purpose: str = "auto", hint: Optional[PerceptionHint] = None
    ) -> Observation:
        if image is None or not hasattr(image, "size") or not hasattr(image, "crop"):
            raise PerceptionError("没有可识别的截图")
        mode = normalize_purpose(purpose)
        captured_at = time.time()
        started = time.monotonic()
        try:
            frame = _to_rgb(image)
        except Exception as exc:  # exotic modes (I;16, F), corrupt image data
            raise PerceptionError(f"截图处理失败: {_short_error(exc)}") from exc
        check_frame(frame)
        try:
            content = self.build_content(frame, mode, hint)
        except Exception as exc:  # corrupt image, PIL errors
            raise PerceptionError(f"截图处理失败: {_short_error(exc)}") from exc
        try:
            # The rich ScreenObservation is over the structured-output schema
            # limits (optional / union parameters); the all-required wire model
            # carries the same information and converts back.
            wire = self.llm.parse(
                model=self.cfg.vision_model,
                effort=self.cfg.vision_effort,
                system=self.system,
                content=content,
                schema=ScreenObservationWire,
                purpose="shop" if mode == "shop" else "vision",
            )
            screen = wire.to_screen() if isinstance(wire, ScreenObservationWire) else wire
        except LLMError as exc:
            raise PerceptionError(str(exc)) from exc
        except (ValidationError, ValueError) as exc:  # malformed structured output
            raise PerceptionError(f"识别结果格式错误: {_short_error(exc)}") from exc
        except Exception as exc:  # SDK / transport surprises: callers only expect PerceptionError
            raise PerceptionError(f"识别失败: {type(exc).__name__}: {_short_error(exc)}") from exc
        if not isinstance(screen, ScreenObservation):
            raise PerceptionError("识别结果格式错误: 不是 ScreenObservation")
        screen = sanitize_screen(screen, mode, hint)
        return Observation(
            screen=screen,
            captured_at=captured_at,
            source="claude",
            purpose=mode,
            latency_s=round(time.monotonic() - started, 2),
        )


__all__ = ["ClaudeVisionPerceiver", "check_frame", "encode_image", "fit_long_edge", "prepare_crop", "sanitize_screen"]
