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

import os
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Optional

from .setdata import SetData
from .textio import read_user_data

STYLES = ("standard", "fast8", "fast9", "reroll1", "reroll2", "reroll3")
SAMPLE_SET_NUMBER = 99
_BUNDLED_SAMPLE = "comps_sample.json"

# Free-text style aliases accepted in user files (lowercased).
_STYLE_ALIASES = {
    "standard": "standard",
    "std": "standard",
    "运营": "standard",
    "标准": "standard",
    "fast8": "fast8",
    "fast9": "fast9",
    "fast 9": "fast9",
    "速9": "fast9",
    "速九": "fast9",
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
    style: str = "standard"  # standard | fast8 | fast9 | reroll1 | reroll2 | reroll3
    tier: str = ""
    notes: str = ""
    early: list[str] = field(default_factory=list)  # early-game units (display names)
    item_holders: dict[str, list[str]] = field(default_factory=dict)  # unit -> recommended items
    positions: dict[str, tuple[int, int]] = field(default_factory=dict)  # unit -> (row, col), row 0 = front
    stars: dict[str, int] = field(default_factory=dict)  # units meant to be 3-star (reroll targets)
    difficulty: str = ""
    # Units a reroll line slow rolls for (display names): "reroll_units" from
    # the file, else the 3-star targets costing at most the reroll cost, else
    # the comp units at most that cost. Empty for non-reroll styles. The carry
    # is the item holder and may cost more (e.g. a 4-cost carry on a 3-cost
    # reroll).
    reroll_units: list[str] = field(default_factory=list)
    name_en: str = ""  # name as written in the file (English for the bundled library)


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
    if "9" in text or "九" in text:
        return "fast9"
    if "8" in text or "八" in text:
        return "fast8"
    return "standard"


def _read_file(path: Path) -> Any:
    return read_user_data(path, "阵容文件")


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
    # "Mostly unresolved" counts distinct names: a unit listed twice is not two misses.
    if not units or len(set(unresolved)) > len(units):
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
        carry_items.append(item_display_name(raw, set_data, log, name))

    traits: list[str] = []
    for raw in _as_list(entry.get("traits")):
        trait = set_data.resolve_trait(raw)
        traits.append(trait.name if trait else raw)

    def unit_name(raw: Any) -> Optional[str]:
        champ = set_data.resolve_champion(str(raw))
        return champ.name if champ else None

    def item_name(raw: Any) -> str:
        return item_display_name(str(raw), set_data, log, name)

    early = [n for n in (unit_name(u) for u in _as_list(entry.get("early"))) if n]
    item_holders: dict[str, list[str]] = {}
    raw_holders = entry.get("item_holders")
    if isinstance(raw_holders, dict):
        for raw_unit, raw_items in raw_holders.items():
            n = unit_name(raw_unit)
            if n:
                item_holders[n] = [item_name(x) for x in _as_list(raw_items)]
    positions: dict[str, tuple[int, int]] = {}
    raw_pos = entry.get("positions")
    if isinstance(raw_pos, dict):
        for raw_unit, rc in raw_pos.items():
            n = unit_name(raw_unit)
            if n and isinstance(rc, (list, tuple)) and len(rc) == 2:
                try:
                    r, c = int(rc[0]), int(rc[1])
                except (TypeError, ValueError):
                    continue
                if 0 <= r <= 3 and 0 <= c <= 6:
                    positions[n] = (r, c)
    stars: dict[str, int] = {}
    raw_stars = entry.get("stars")
    if isinstance(raw_stars, dict):
        for raw_unit, st in raw_stars.items():
            n = unit_name(raw_unit)
            if n and isinstance(st, int) and 1 <= st <= 4:
                stars[n] = st

    style = normalize_style(entry.get("style"), carry_cost)
    return CompDef(
        name=display_name(entry, name, set_data),
        units=units,
        carry=carry_name,
        carry_items=carry_items,
        traits=traits,
        style=style,
        tier=str(entry.get("tier") or "").strip().upper(),
        notes=str(entry.get("notes") or "").strip(),
        early=early,
        item_holders=item_holders,
        positions=positions,
        stars=stars,
        difficulty=str(entry.get("difficulty") or "").strip(),
        reroll_units=_reroll_units(entry, style, units, stars, set_data),
        name_en=name,
    )


# Items the set data may not list (artifacts, augment emblems are trimmed from
# the snapshot): normalized api id tail -> (English, Chinese) display name.
ITEM_ALIASES = {
    "navoriflickerblade": ("Navori Flickerblade", "纳沃利迅刃"),
}
_API_PREFIX = re.compile(r"^(?:DA_|TFT\d*_)?(?:\d+_)?(?:Item_)?(?:Artifact_|Ornn_|Radiant_)?", re.I)
_EMBLEM_API = re.compile(r"Emblem([A-Za-z]+?)(?:Augment|Item)?$")
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")
_API_LIKE = re.compile(r"^[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+$")


def item_display_name(raw: str, set_data: SetData, log: Optional[LogFn] = None, comp: str = "") -> str:
    """Display name of a comp item. Raw API ids the set data does not know
    (an artifact, an augment emblem) never reach the player or the prompt:
    known aliases, "<trait> emblem" from the set's traits, or a humanized
    English name (with a warning)."""
    raw = str(raw).strip()
    item = set_data.resolve_item(raw)
    if item is not None:
        return item.name
    if not _API_LIKE.match(raw):
        return raw  # already a readable name
    zh = _set_is_chinese(set_data)
    tail = _API_PREFIX.sub("", raw)
    alias = ITEM_ALIASES.get(tail.replace("_", "").lower())
    if alias:
        return alias[1] if zh else alias[0]
    m = _EMBLEM_API.search(raw)
    if m:
        words = _CAMEL.sub(" ", m.group(1))
        trait = set_data.resolve_trait(words) or set_data.resolve_trait(m.group(1))
        if trait is not None:
            return f"{trait.name}纹章" if zh else f"{trait.name_en or trait.name} Emblem"
        tail = f"{words} Emblem"
    human = _CAMEL.sub(" ", tail.replace("_", " ")).strip() or raw
    if log is not None:
        log(f"阵容 {comp}：装备 {raw} 不在赛季数据里，按 {human} 显示")
    return human


REROLL_COST = {"reroll1": 1, "reroll2": 2, "reroll3": 3}


def _reroll_units(
    entry: dict[str, Any], style: str, units: list[str], stars: dict[str, int], set_data: SetData
) -> list[str]:
    cap = REROLL_COST.get(style)
    if cap is None:
        return []

    def cost(name: str) -> int:
        champ = set_data.resolve_champion(name)
        return champ.cost if champ else 99

    explicit = []
    for raw in _as_list(entry.get("reroll_units")):
        champ = set_data.resolve_champion(raw)
        if champ is not None and champ.name not in explicit:
            explicit.append(champ.name)
    if explicit:
        return explicit
    starred = [u for u, st in stars.items() if st >= 3 and cost(u) <= cap]
    if starred:
        return starred
    return [u for u in units if cost(u) <= cap]


def _has_cjk(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text or "")


def display_name(entry: dict[str, Any], name: str, set_data: SetData) -> str:
    """Name shown to the player: ``name_zh`` from the file when the set data
    is Chinese (the default client locale), otherwise the file's name."""
    zh = str(entry.get("name_zh") or "").strip()
    if zh and _set_is_chinese(set_data):
        return zh
    return name


def _set_is_chinese(set_data: SetData) -> bool:
    names = [c.name for c in list(set_data.champions.values())[:20]]
    return any(_has_cjk(n) for n in names)


def load_comps(path: Optional[str], set_data: SetData, log: Optional[LogFn] = None) -> list[CompDef]:
    """Load comps from ``path`` (JSON or TOML) and resolve names against ``set_data``.

    With no path: the bundled comp library for the loaded set
    (``data/bundled/comps_set<N>.json``, a meta snapshot shipped with the
    tool), or the sample comps for the synthetic sample set (``set_number ==
    99``); otherwise an empty list (the engine then falls back to trait-based
    auto comps). Problems are reported through ``log`` and never raise.
    """
    emit: LogFn = log or (lambda _msg: None)
    if path:
        p = Path(os.path.expanduser(str(path)))
        if not p.is_file():
            emit(f"找不到阵容文件：{p}，改用内置阵容库")
            return load_comps(None, set_data, log)
        comps = _load_file(p, set_data, emit)
        if not comps:
            # A broken or empty custom file must not leave the advisor with
            # no comp library at all.
            emit(f"阵容文件 {p.name} 没有可用的阵容，改用内置阵容库")
            return load_comps(None, set_data, log)
        return comps
    if set_data.set_number == SAMPLE_SET_NUMBER:
        p = bundled_comps_path()
    else:
        p = Path(str(resources.files("tft_advisor.data").joinpath("bundled", f"comps_set{set_data.set_number}.json")))
        if not p.is_file():
            return []
    return _load_file(p, set_data, emit)


def _load_file(p: Path, set_data: SetData, emit: LogFn) -> list[CompDef]:
    try:
        raw = _read_file(p)
    except Exception as exc:  # OSError, bad JSON / TOML, absurd nesting: never kill startup
        emit(f"阵容文件读取失败（{p.name}）：{exc}")
        return []

    comps: list[CompDef] = []
    names: set[str] = set()
    for entry in _entries(raw):
        try:
            comp = _build_comp(entry, set_data, emit)
        except Exception as exc:  # one malformed entry must not drop the whole file
            emit(f"阵容配置有误，已忽略一条：{exc}")
            continue
        if comp is None:
            continue
        if comp.name in names:
            emit(f"阵容名称重复，只保留第一个：{comp.name}")
            continue
        names.add(comp.name)
        comps.append(comp)
    return comps


def comps_reference_text(comps: list[CompDef], limit: int = 45) -> str:
    """Compact comp library for LLM prompts (deterministic, prompt-cache friendly)."""
    if not comps:
        return ""
    lines = ["COMP LIBRARY (tier, style: units | carry: items | early units):"]
    for c in comps[:limit]:
        star = "".join(f" {u}*{n}" for u, n in c.stars.items())
        head = f"- {c.name} [{c.tier or '-'}, {c.style}]: {', '.join(c.units)}"
        if star:
            head += f" (3-star:{star})"
        carry = f" | carry {c.carry}: {', '.join(c.carry_items)}" if c.carry else ""
        early = f" | early: {', '.join(c.early)}" if c.early else ""
        lines.append(head + carry + early)
    return "\n".join(lines)
