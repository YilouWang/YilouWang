"""Claude vision perceiver: screenshot -> ``ScreenObservation`` via structured output.

Per call we send the full frame (downscaled) plus high-resolution crops of the
small, number-heavy HUD parts (bottom HUD, player list, stage, and for scouting
the top banner), each preceded by a text label so the model knows what it sees.
For ``purpose="shop"`` only the bottom HUD crop is sent (fast and cheap).

The large system prompt (HUD guide + set name lists) is identical for every call
of a set, so the prompt cache in ``tft_advisor.llm`` makes repeated calls cheap.
"""

from __future__ import annotations

import io
import re
import time
from typing import TYPE_CHECKING, Any, Optional

from pydantic import ValidationError

from ..config import AnthropicConfig
from ..llm import LLM, LLMError, image_block, text_block
from ..models import Observation, ScreenObservation, StageRound, UnitObs
from .base import PerceptionError, PerceptionHint, clean_text, crop_region, normalize_purpose
from .prompts import build_user_text, build_vision_system

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image

    from ..data.setdata import SetData

# Crops for the non-shop purposes: (region name, label shown to the model).
_CROPS_DEFAULT: tuple[tuple[str, str], ...] = (
    ("hud_bottom", "bottom HUD crop (shop, gold, level, XP, streak) of the LOCAL player"),
    ("players", "player list crop (names and HP, right side)"),
    ("stage", "stage / round indicator crop (top center)"),
)
_CROP_SCOUT_BANNER = ("top_banner", "top banner crop (name of the board owner while scouting)")
_SHOP_LABEL = "bottom HUD crop (shop, gold, level, XP, streak) of the LOCAL player"

# Anthropic's per-image limit is 5 MB of base64; stay well below it.
_MAX_IMAGE_BYTES = 3_500_000

# Fields the model may fill from a bottom-HUD-only request.
_SHOP_FIELDS = {"screen_type", "shop", "shop_locked", "gold", "level", "xp_current", "xp_needed", "streak", "notes"}

_STAGE_STRICT = re.compile(r"^\d-\d$")


# ---------------------------------------------------------------------------
# image helpers
# ---------------------------------------------------------------------------


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


def encode_image(img: "Image.Image", optimize: bool = True) -> tuple[bytes, str]:
    """PNG (lossless, best for small HUD text); JPEG fallback if the PNG is too large for the API."""
    img = _to_rgb(img)
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


def _norm(text: Optional[str]) -> str:
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


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
        u.items = [i for i in u.items if (i or "").strip()][:3]
        out.append(u)
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
        parsed = StageRound.parse(raw)
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
            if slot.cost is not None and not 1 <= slot.cost <= 10:
                slot.cost = None

    s.board = _clean_units(s.board, on_board=True, notes=notes)
    s.bench = _clean_units(s.bench, on_board=False, notes=notes)

    if s.players is not None:
        players = [p for p in s.players if (p.name or "").strip()]
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
        if s.viewed_player_name is None and hint is not None and hint.scouting_player:
            s.viewed_player_name = hint.scouting_player
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
        self.png_optimize = True
        self.system = build_vision_system(set_data)

    def set_set_data(self, set_data: "SetData") -> None:
        """Swap set data (e.g. after a data update); rebuilds the cached system prompt."""
        self.set_data = set_data
        self.system = build_vision_system(set_data)

    # ---- images -------------------------------------------------------------
    def build_images(self, image: "Image.Image", purpose: str = "auto") -> list[tuple[str, "Image.Image"]]:
        """(label, image) pairs to send, in order. Labels do not include the ``IMAGE n:`` prefix."""
        mode = normalize_purpose(purpose)
        max_edge = int(self.cfg.max_image_edge or 1568)
        frame = _to_rgb(image)
        if mode == "shop":
            return [(_SHOP_LABEL, prepare_crop(crop_region(frame, "hud_bottom"), max_edge))]
        full = fit_long_edge(frame, max_edge)
        out: list[tuple[str, "Image.Image"]] = [
            (f"full screenshot, downscaled to {full.size[0]}x{full.size[1]}", full)
        ]
        crops = list(_CROPS_DEFAULT)
        if mode == "scout":
            crops.append(_CROP_SCOUT_BANNER)
        for region, label in crops:
            out.append((label, prepare_crop(crop_region(frame, region), max_edge)))
        return out

    def build_content(
        self, image: "Image.Image", purpose: str = "auto", hint: Optional[PerceptionHint] = None
    ) -> list[dict[str, Any]]:
        pairs = self.build_images(image, purpose)
        content: list[dict[str, Any]] = []
        labels: list[str] = []
        for idx, (label, img) in enumerate(pairs, start=1):
            full_label = f"IMAGE {idx}: {label}"
            labels.append(full_label)
            data, media_type = encode_image(img, optimize=self.png_optimize)
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
            content = self.build_content(image, mode, hint)
        except Exception as exc:  # corrupt image, PIL errors
            raise PerceptionError(f"截图处理失败: {exc}") from exc
        try:
            screen = self.llm.parse(
                model=self.cfg.vision_model,
                effort=self.cfg.vision_effort,
                system=self.system,
                content=content,
                schema=ScreenObservation,
                purpose="shop" if mode == "shop" else "vision",
            )
        except LLMError as exc:
            raise PerceptionError(str(exc)) from exc
        except (ValidationError, ValueError) as exc:  # malformed structured output
            raise PerceptionError(f"识别结果格式错误: {exc}") from exc
        screen = sanitize_screen(screen, mode, hint)
        return Observation(
            screen=screen,
            captured_at=captured_at,
            source="claude",
            purpose=mode,
            latency_s=round(time.monotonic() - started, 2),
        )


__all__ = ["ClaudeVisionPerceiver", "encode_image", "fit_long_edge", "prepare_crop", "sanitize_screen"]
