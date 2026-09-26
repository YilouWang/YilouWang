"""Comp definitions (team compositions) loaded from JSON / TOML.

A comp file lists target compositions for the current set::

    {"comps": [
      {"name": "枪手海盗", "units": ["Miss Fortune", "Graves", ...],
       "carry": "Miss Fortune", "carry_items": ["Infinity Edge", "Giant Slayer"],
       "traits": ["Pirate", "Gunslinger"], "style": "fast8", "tier": "A",
       "notes": "..."}
    ]}

or the TOML equivalent with ``[[comps]]`` tables. Unit, item and trait names
may be written in any language the set data knows (zh / en / api names); they
are resolved against the loaded ``SetData`` and stored as display names.

Without a user file nothing is loaded, except for the bundled synthetic sample
set (set 99, used by tests and the offline demo), which ships a few sample
comps. Real sets change every few months, so we never ship stale real comps.
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Optional

from .setdata import SetData

STYLES = ("standard", "fast8", "reroll1", "reroll2", "reroll3")
SAMPLE_SET_NUMBER = 99
_BUNDLED_SAMPLE = "comps_sample.json"

# Free-text style aliases accepted in user files (lowercased).
_STYLE_ALIASES = {
    "standard": "standard",
    "std": "standard",
    "运营": "standard",
    "标准": "standard",
    "fast8": "fast8",
    "fast 8": "fast8",
    "fast_8": "fast8",
    "速8": "fast8",
    "速八": "fast8",
    "reroll1": "reroll1",
    "reroll2": "reroll2",
    "reroll3": "reroll3",
    "1cost reroll": "reroll1",
    "2cost reroll": "reroll2",
    "3cost reroll": "reroll3",
}
_REROLL_WORDS = ("reroll", "赌", "d牌", "慢d", "slow roll", "slowroll")

LogFn = Callable[[str], None]


@dataclass
class CompDef:
    name: str
    units: list[str]
    carry: Optional[str] = None
    carry_items: list[str] = field(default_factory=list)
    traits: list[str] = field(default_factory=list)
    style: str = "standard"  # standard | fast8 | reroll1 | reroll2 | reroll3
    tier: str = ""
    notes: str = ""


def bundled_comps_path() -> Path:
    return Path(str(resources.files("tft_advisor.data").joinpath("bundled", _BUNDLED_SAMPLE)))


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace("，", ",").split(",")]
        return [p for p in parts if p]
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    return []


def normalize_style(raw: Any, carry_cost: Optional[int] = None) -> str:
    """Map a free-text style to one of ``STYLES``.

    A bare "reroll" picks the reroll level from the carry's cost (1 -> reroll1
    at level 5, 2 -> reroll2 at level 6, 3 -> reroll3 at level 7).
    """
    text = str(raw or "").strip().lower()
    if not text:
        return "standard"
    if text in _STYLE_ALIASES:
        return _STYLE_ALIASES[text]
    if any(w in text for w in _REROLL_WORDS):
        for digit, style in (("1", "reroll1"), ("2", "reroll2"), ("3", "reroll3")):
            if digit in text:
                return style
        if carry_cost in (1, 2, 3):
            return f"reroll{carry_cost}"
        return "reroll2"
    if "8" in text or "八" in text:
        return "fast8"
    return "standard"


def _read_file(path: Path) -> Any:
    if path.suffix.lower() == ".toml":
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _entries(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = raw.get("comps") or raw.get("comp") or []
    else:
        items = []
    return [e for e in items if isinstance(e, dict)]


def _build_comp(entry: dict[str, Any], set_data: SetData, log: LogFn) -> Optional[CompDef]:
    name = str(entry.get("name") or "").strip()
    raw_units = _as_list(entry.get("units") or entry.get("champions"))
    if not name or not raw_units:
        log(f"阵容配置缺少名称或英雄列表，已忽略：{name or entry}")
        return None

    units: list[str] = []
    seen: set[str] = set()
    unresolved: list[str] = []
    for raw in raw_units:
        champ = set_data.resolve_champion(raw)
        if champ is None:
            unresolved.append(raw)
            continue
        if champ.api_name not in seen:
            seen.add(champ.api_name)
            units.append(champ.name)
    if not units or len(units) * 2 < len(raw_units):
        log(f"阵容 {name} 的英雄大多不在当前赛季数据里，已忽略（无法识别：{', '.join(unresolved)}）")
        return None
    if unresolved:
        log(f"阵容 {name}：以下英雄无法识别，已跳过：{', '.join(unresolved)}")

    carry_name: Optional[str] = None
    carry_cost: Optional[int] = None
    raw_carry = entry.get("carry")
    if raw_carry:
        champ = set_data.resolve_champion(str(raw_carry))
        if champ is None:
            log(f"阵容 {name}：主C {raw_carry} 无法识别")
        else:
            carry_name, carry_cost = champ.name, champ.cost
            if champ.api_name not in seen:
                seen.add(champ.api_name)
                units.insert(0, champ.name)

    carry_items: list[str] = []
    for raw in _as_list(entry.get("carry_items") or entry.get("items")):
        item = set_data.resolve_item(raw)
        carry_items.append(item.name if item else raw)

    traits: list[str] = []
    for raw in _as_list(entry.get("traits")):
        trait = set_data.resolve_trait(raw)
        traits.append(trait.name if trait else raw)

    return CompDef(
        name=name,
        units=units,
        carry=carry_name,
        carry_items=carry_items,
        traits=traits,
        style=normalize_style(entry.get("style"), carry_cost),
        tier=str(entry.get("tier") or "").strip().upper(),
        notes=str(entry.get("notes") or "").strip(),
    )


def load_comps(path: Optional[str], set_data: SetData, log: Optional[LogFn] = None) -> list[CompDef]:
    """Load comps from ``path`` (JSON or TOML) and resolve names against ``set_data``.

    With no path, the bundled sample comps are returned only for the synthetic
    sample set (``set_number == 99``); otherwise an empty list (the engine then
    falls back to trait-based auto comps). Problems are reported through
    ``log`` and never raise.
    """
    emit: LogFn = log or (lambda _msg: None)
    if path:
        p = Path(os.path.expanduser(str(path)))
        if not p.is_file():
            emit(f"找不到阵容文件：{p}")
            return []
    elif set_data.set_number == SAMPLE_SET_NUMBER:
        p = bundled_comps_path()
    else:
        return []

    try:
        raw = _read_file(p)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        emit(f"阵容文件读取失败（{p.name}）：{exc}")
        return []

    comps: list[CompDef] = []
    names: set[str] = set()
    for entry in _entries(raw):
        comp = _build_comp(entry, set_data, emit)
        if comp is None:
            continue
        if comp.name in names:
            emit(f"阵容名称重复，只保留第一个：{comp.name}")
            continue
        names.add(comp.name)
        comps.append(comp)
    return comps
