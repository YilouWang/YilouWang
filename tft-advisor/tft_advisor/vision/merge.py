"""Combine two observations of the same moment (e.g. Claude vision + OCR / Live Client).

Rules (see :func:`merge_observations`):

* ``None`` fields of the primary are filled from the secondary.
* Fields listed in ``prefer_secondary`` take the secondary value whenever it is
  not ``None``: fast exact sources (OCR digits, the Live Client API) win for
  numbers such as gold / level / stage.
* ``screen_type`` "other" means "unknown" and is filled from the secondary.
* Notes are concatenated (deduplicated), plus a short note for each overridden
  value that actually differed.
"""

from __future__ import annotations

from typing import Iterable, Optional

from ..models import Observation, ScreenObservation, ScreenType

DEFAULT_PREFER_SECONDARY: frozenset[str] = frozenset({"gold", "level", "stage"})

_FIELD_LABELS = {
    "gold": "金币",
    "level": "等级",
    "stage": "回合",
    "xp_current": "经验",
    "xp_needed": "升级所需经验",
    "hp": "血量",
    "streak": "连胜连败",
    "shop": "商店",
    "shop_locked": "商店锁定",
    "players": "玩家列表",
}


def _norm(text: Optional[str]) -> str:
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def _short(value: object) -> str:
    text = str(value)
    return text if len(text) <= 24 else text[:21] + "..."


def merge_observations(
    primary: ScreenObservation,
    secondary: ScreenObservation,
    prefer_secondary: Optional[Iterable[str]] = DEFAULT_PREFER_SECONDARY,
) -> ScreenObservation:
    """Return a new ``ScreenObservation``; inputs are not modified."""
    prefer = set(DEFAULT_PREFER_SECONDARY if prefer_secondary is None else prefer_secondary)
    out = primary.model_copy(deep=True)
    sec = secondary.model_copy(deep=True)
    notes: list[str] = list(out.notes)
    override_notes: list[str] = []

    for field in ScreenObservation.model_fields:
        if field == "notes":
            continue
        p_val = getattr(out, field)
        s_val = getattr(sec, field)
        if field == "screen_type":
            if s_val != ScreenType.OTHER and (p_val == ScreenType.OTHER or field in prefer):
                out.screen_type = s_val
            continue
        if s_val is None:
            continue
        if p_val is None:
            setattr(out, field, s_val)
        elif field in prefer and s_val != p_val:
            setattr(out, field, s_val)
            label = _FIELD_LABELS.get(field, field)
            override_notes.append(f"{label}: {_short(p_val)} 改为 {_short(s_val)} (以更精确的来源为准)")

    # Player list: keep the primary rows but learn "who am I" from the secondary.
    if out.players and sec.players and not any(p.is_self for p in out.players):
        selves = {_norm(p.name) for p in sec.players if p.is_self}
        for p in out.players:
            if _norm(p.name) in selves:
                p.is_self = True
                break

    for n in [*sec.notes, *override_notes]:
        if n and n not in notes:
            notes.append(n)
    out.notes = notes
    return out


def merge_into(primary: Observation, secondary: Observation, prefer_secondary: Optional[Iterable[str]] = None) -> Observation:
    """Observation-level helper: merged screen, ``source`` like ``"claude+ocr"``."""
    screen = merge_observations(primary.screen, secondary.screen, prefer_secondary)
    return primary.model_copy(update={"screen": screen, "source": f"{primary.source}+{secondary.source}"})


__all__ = ["DEFAULT_PREFER_SECONDARY", "merge_into", "merge_observations"]
