"""Item planner: which components to combine, and who should hold the result.

* Components on the item bench are paired by brute force over all (partial)
  matchings (<= 10 components, ~10^4 matchings) to maximize a value score.
* Item value = tier from ``data/bundled/item_roles.json`` (English names)
  + a big bonus when the item is one of the target comp's carry items
  + a role fit bonus (AD items when the carry is physical, and so on).
* Holders: AD / AP items go to the matching carry (comp carry first), tank and
  utility items to a front-row unit, flex items to the main carry. A unit
  holds at most 3 items.
* Completed items still on the bench are suggested for equipping, and single
  leftover components that are half of a carry item produce a low-priority
  "still missing X" suggestion (used for carousel picks).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Iterable, Optional

from ..data.setdata import Champion, Item, SetData, normalize_name
from ..models import CompSuggestion, GameState, ItemSuggestion, StageRound, Unit

ROLES = ("ad", "ap", "tank", "utility", "flex")
ROLE_ZH = {
    "ad": "物理输出",
    "ap": "法系输出",
    "tank": "坦克",
    "utility": "功能",
    "flex": "通用",
    "emblem": "纹章",
}
DEFAULT_ROLE = ("flex", 2)
MAX_ITEMS_PER_UNIT = 3
MAX_COMPONENTS = 10
MIN_PAIR_VALUE = 0.5
SLAM_STAGE = (2, 5)

# Trait keywords (English + Chinese) hinting at a unit's damage profile.
AD_WORDS = (
    "gunslinger", "ranger", "marksman", "sniper", "blademaster", "duelist", "assassin", "slayer",
    "challenger", "deadeye", "quickstrike", "executioner", "rapidfire", "striker", "gunner", "reaper",
    "edgelord", "cannoneer", "hunter", "longshot",
    "枪手", "游侠", "狙神", "神射", "射手", "剑士", "剑圣", "决斗", "刺客", "杀手", "枪", "猎",
)
AP_WORDS = (
    "sorcerer", "mage", "arcanist", "invoker", "scholar", "spellslinger", "sage", "caster", "channeler",
    "psionic", "witch", "magus", "mystic", "visionary", "wizard", "arcana", "prodigy", "coven",
    "法师", "术士", "秘术", "巫", "咒", "奥术", "学者", "法",
)
TANK_WORDS = (
    "knight", "guardian", "brawler", "bruiser", "warden", "bastion", "vanguard", "juggernaut", "sentinel",
    "defender", "protector", "colossus", "behemoth", "tank", "heavyweight", "bulwark", "guard", "titan",
    "骑士", "守护", "斗士", "护卫", "重装", "坦克", "守卫", "哨兵", "堡垒", "卫士", "巨像", "战士",
)


# ---------------------------------------------------------------------------
# Role table
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _bundled_roles() -> dict[str, tuple[str, int]]:
    text = resources.files("tft_advisor.data").joinpath("bundled", "item_roles.json").read_text(encoding="utf-8")
    return _parse_roles(json.loads(text))


def _parse_roles(raw: dict) -> dict[str, tuple[str, int]]:
    out: dict[str, tuple[str, int]] = {}
    for name, spec in raw.items():
        if name.startswith("_") or not isinstance(spec, dict):
            continue
        role = str(spec.get("role", "flex")).lower()
        if role not in ROLES:
            role = "flex"
        try:
            tier = min(3, max(1, int(spec.get("tier", 2))))
        except (TypeError, ValueError):
            tier = 2
        out[normalize_name(name)] = (role, tier)
    return out


def item_roles() -> dict[str, tuple[str, int]]:
    """Normalized English item name -> (role, tier)."""
    return dict(_bundled_roles())


def item_role(item: Optional[Item], raw_name: str = "") -> tuple[str, int]:
    """(role, tier) of an item; emblems are ('emblem', 2); unknown -> ('flex', 2)."""
    if item is not None and item.kind == "emblem":
        return ("emblem", 2)
    table = _bundled_roles()
    names = [n for n in ((item.name_en, item.name) if item else ()) if n] + ([raw_name] if raw_name else [])
    for n in names:
        key = normalize_name(n)
        if key in table:
            return table[key]
        if key.startswith("radiant") and key[len("radiant") :] in table:
            return (table[key[len("radiant") :]][0], 3)
    return DEFAULT_ROLE


# ---------------------------------------------------------------------------
# Unit profiles
# ---------------------------------------------------------------------------


def _word_hits(texts: Iterable[str], words: tuple[str, ...]) -> int:
    hits = 0
    for t in texts:
        low = (t or "").lower()
        if any(w in low for w in words):
            hits += 1
    return hits


def champion_profile(champ: Optional[Champion], unit: Optional[Unit] = None, set_data: Optional[SetData] = None) -> str:
    """'ad' | 'ap' | 'tank' | 'unknown' from cdragon role, api suffix, traits and items."""
    score = {"ad": 0.0, "ap": 0.0, "tank": 0.0}
    role = str(getattr(champ, "role", "") or "").lower() if champ else ""
    if role:
        if "tank" in role:
            score["tank"] += 2.0
        elif role.startswith("ad"):
            score["ad"] += 2.0
        elif role.startswith("ap"):
            score["ap"] += 2.0
        if "fighter" in role:
            score["tank"] += 0.5
    api = (champ.api_name if champ else (unit.api_name if unit else "")) or ""
    if api.endswith("_AD"):
        score["ad"] += 1.0
    elif api.endswith("_AP"):
        score["ap"] += 1.0
    traits: list[str] = []
    if champ:
        traits += list(champ.traits) + list(getattr(champ, "traits_en", []) or [])
    elif unit:
        traits += list(unit.traits)
    # A trait name can contain both an AP and an AD keyword; count each trait once per class.
    score["ad"] += _word_hits(traits, AD_WORDS)
    score["ap"] += _word_hits(traits, AP_WORDS)
    score["tank"] += _word_hits(traits, TANK_WORDS)
    if unit is not None:
        for name in unit.items:
            item = set_data.resolve_item(name) if set_data else None
            r, _tier = item_role(item, name)
            if r in score:
                score[r] += 1.0
        if unit.row == 0:
            score["tank"] += 0.5
        elif unit.row == 3:
            score["ad"] += 0.1
            score["ap"] += 0.1
    best = max(score, key=lambda k: (score[k], k == "tank"))
    return best if score[best] > 0.3 else "unknown"


def _power(u: Unit) -> int:
    return (u.cost or 1) * max(1, u.star)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    set_data: SetData
    board: list[Unit]
    bench: list[Unit]
    profiles: dict[int, str] = field(default_factory=dict)
    load: dict[int, int] = field(default_factory=dict)
    carry_unit: Optional[Unit] = None
    carry_profile: Optional[str] = None
    comp_item_apis: set[str] = field(default_factory=set)
    trait_counts: dict[str, int] = field(default_factory=dict)

    @property
    def units(self) -> list[Unit]:
        return [*self.board, *self.bench]

    def profile(self, u: Unit) -> str:
        return self.profiles.get(id(u), "unknown")

    def free(self, u: Unit) -> bool:
        return self.load.get(id(u), 0) < MAX_ITEMS_PER_UNIT

    def on_board(self, u: Unit) -> bool:
        return any(u is b for b in self.board)

    def count_profile(self, role: str) -> int:
        return sum(1 for u in self.units if self.profile(u) == role)


def _build_ctx(state: GameState, set_data: SetData, comp: Optional[CompSuggestion]) -> _Ctx:
    ctx = _Ctx(set_data=set_data, board=list(state.board), bench=list(state.bench))
    for u in ctx.units:
        champ = set_data.champions.get(u.api_name)
        ctx.profiles[id(u)] = champion_profile(champ, u, set_data)
        ctx.load[id(u)] = len(u.items)
    seen_traits: dict[str, set[str]] = {}
    for u in ctx.units:
        if u.api_name.startswith("?"):
            continue
        for t in u.traits:
            seen_traits.setdefault(t, set()).add(u.api_name)
    ctx.trait_counts = {t: len(v) for t, v in seen_traits.items()}
    if comp is not None:
        for name in comp.carry_items:
            item = set_data.resolve_item(name)
            if item is not None:
                ctx.comp_item_apis.add(item.api_name)
        if comp.carry:
            champ = set_data.resolve_champion(comp.carry)
            if champ is not None:
                owned = [u for u in ctx.units if u.api_name == champ.api_name]
                if owned:
                    ctx.carry_unit = max(owned, key=lambda u: (u.star, ctx.on_board(u), len(u.items)))
                    ctx.carry_profile = ctx.profile(ctx.carry_unit)
                else:
                    ctx.carry_profile = champion_profile(champ, None, set_data)
    return ctx


def _emblem_trait(item: Item, set_data: SetData) -> Optional[str]:
    for n in (item.name, item.name_en):
        base = (n or "").replace("Emblem", "").replace("纹章", "").replace("徽章", "").strip()
        trait = set_data.resolve_trait(base) if base else None
        if trait is not None:
            return trait.name
    return None


def _item_value(item: Item, ctx: _Ctx) -> float:
    role, tier = item_role(item)
    if role == "emblem":
        value = 1.0
        trait_name = _emblem_trait(item, ctx.set_data)
        trait = ctx.set_data.resolve_trait(trait_name) if trait_name else None
        if trait is not None:
            count = ctx.trait_counts.get(trait.name, 0)
            if count and any(bp == count + 1 for bp in trait.breakpoints):
                value += 2.0
            elif count:
                value += 0.5
        if item.api_name in ctx.comp_item_apis:
            value += 3.0
        return value
    value = float(tier)
    if item.api_name in ctx.comp_item_apis:
        value += 3.0
    if role in ("ad", "ap"):
        other = "ap" if role == "ad" else "ad"
        if ctx.carry_profile == role:
            value += 1.0
        elif ctx.carry_profile == other:
            value -= 1.0
        elif ctx.count_profile(role):
            value += 0.5
        elif ctx.count_profile(other):
            value -= 1.0
    elif role == "tank" and ctx.count_profile("tank"):
        value += 0.3
    return value


def _best_matching(
    comps: list[Item], set_data: SetData, value_of: dict[str, float]
) -> tuple[float, list[tuple[int, int, Item]]]:
    """Brute force the partial matching of components with the highest value."""
    n = len(comps)
    pair_item: dict[tuple[int, int], Item] = {}
    for i in range(n):
        for j in range(i + 1, n):
            it = set_data.recipe(comps[i].api_name, comps[j].api_name)
            if it is not None and it.kind in ("completed", "emblem"):
                pair_item[(i, j)] = it

    best_score = 0.0
    best_pairs: list[tuple[int, int, Item]] = []

    def rec(used: int, pairs: list[tuple[int, int, Item]], score: float, counts: dict[str, int]) -> None:
        nonlocal best_score, best_pairs
        idx = next((k for k in range(n) if not used & (1 << k)), None)
        if idx is None:
            if score > best_score + 1e-9:
                best_score, best_pairs = score, list(pairs)
            return
        # Option 1: leave component idx unpaired.
        rec(used | (1 << idx), pairs, score, counts)
        # Option 2: pair it with a later free component.
        tried: set[str] = set()
        for j in range(idx + 1, n):
            if used & (1 << j) or (idx, j) not in pair_item:
                continue
            if comps[j].api_name in tried:  # identical component: same outcome
                continue
            tried.add(comps[j].api_name)
            it = pair_item[(idx, j)]
            dup = counts.get(it.api_name, 0)
            v = value_of[it.api_name] * (0.5 if dup else 1.0)
            if v < MIN_PAIR_VALUE:
                continue
            counts[it.api_name] = dup + 1
            pairs.append((idx, j, it))
            rec(used | (1 << idx) | (1 << j), pairs, score + v, counts)
            pairs.pop()
            counts[it.api_name] = dup

    rec(0, [], 0.0, {})
    return best_score, best_pairs


def _pick_holder(role: str, item: Item, ctx: _Ctx) -> Optional[Unit]:
    units = [u for u in ctx.units if ctx.free(u) and not u.api_name.startswith("?")]
    if not units:
        units = [u for u in ctx.units if ctx.free(u)]
    if not units:
        return None
    carry = ctx.carry_unit if ctx.carry_unit is not None and ctx.free(ctx.carry_unit) else None

    def best(cands: list[Unit]) -> Optional[Unit]:
        if not cands:
            return None
        return max(cands, key=lambda u: (_power(u), ctx.on_board(u), u.row or 0, len(u.items), u.name))

    if item.api_name in ctx.comp_item_apis and carry is not None:
        return carry
    if role in ("ad", "ap"):
        if carry is not None and ctx.profile(carry) in (role, "unknown"):
            return carry
        return (
            best([u for u in units if ctx.profile(u) == role])
            or best([u for u in units if ctx.profile(u) not in ("tank",)])
            or best(units)
        )
    if role in ("tank", "utility"):
        ids = {id(u) for u in units}
        front = [u for u in ctx.board if id(u) in ids and u.row == 0]
        tanks_front = [u for u in front if ctx.profile(u) == "tank"]
        return (
            best(tanks_front)
            or best(front)
            or best([u for u in units if ctx.profile(u) == "tank"])
            or best([u for u in ctx.board if id(u) in ids and u.row == 1])
            or best(units)
        )
    if role == "emblem":
        trait = _emblem_trait(item, ctx.set_data)
        lacking = [u for u in units if trait is None or trait not in u.traits]
        return best([u for u in lacking if ctx.on_board(u)]) or best(lacking) or best(units)
    # flex
    if carry is not None:
        return carry
    return best([u for u in units if ctx.profile(u) not in ("tank",)]) or best(units)


def _reason(a: str, b: str, item: Item, role: str, holder: Optional[Unit], comp_item: bool, slam: bool) -> str:
    parts = [f"{a}+{b} 合成{item.name}（{ROLE_ZH.get(role, '通用')}装）"]
    if holder is not None:
        parts.append(f"给{holder.name}")
    if comp_item:
        parts.append("阵容主C核心装备")
    parts.append("现在就合，早合早赢血" if slam else "先留散件，关键回合再合")
    return "，".join(parts)


def plan_items(
    state: GameState,
    set_data: SetData,
    comp: Optional[CompSuggestion] = None,
    hp_bucket: str = "unknown",
) -> list[ItemSuggestion]:
    """Suggested item combinations (priority 1 = slam now) plus equip hints."""
    ctx = _build_ctx(state, set_data, comp)

    components: list[Item] = []
    completed_on_bench: list[Item] = []
    for raw in state.item_bench:
        item = set_data.resolve_item(raw)
        if item is None:
            continue
        if item.kind == "component":
            components.append(item)
        elif item.kind in ("completed", "emblem"):
            completed_on_bench.append(item)
    components = sorted(components, key=lambda i: i.api_name)[:MAX_COMPONENTS]

    stage_key = state.stage.key if isinstance(state.stage, StageRound) else (0, 0)
    slam = (stage_key >= SLAM_STAGE and len(components) >= 2) or hp_bucket in ("low", "critical")

    out: list[tuple[int, float, ItemSuggestion]] = []

    # Completed items sitting on the bench: equip them now.
    for item in completed_on_bench:
        role, _tier = item_role(item)
        holder = _pick_holder(role, item, ctx)
        if holder is not None:
            ctx.load[id(holder)] = ctx.load.get(id(holder), 0) + 1
        text = f"{item.name} 已经合成好了还放在备战席" + (f"，装给{holder.name}" if holder else "，先上场一个英雄再装备")
        out.append(
            (1, _item_value(item, ctx), ItemSuggestion(item=item.name, components=[], holder=holder.name if holder else None, priority=1, reason=text))
        )

    # Best combination of components.
    value_of: dict[str, float] = {}
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            it = set_data.recipe(components[i].api_name, components[j].api_name)
            if it is not None and it.api_name not in value_of:
                value_of[it.api_name] = _item_value(it, ctx)
    _score, pairs = _best_matching(components, set_data, value_of)
    pairs = sorted(pairs, key=lambda p: (-value_of[p[2].api_name], p[2].name))
    used_idx: set[int] = set()
    planned: set[str] = set()
    for i, j, item in pairs:
        used_idx.update((i, j))
        planned.add(item.api_name)
        role, _tier = item_role(item)
        holder = _pick_holder(role, item, ctx)
        if holder is not None:
            ctx.load[id(holder)] = ctx.load.get(id(holder), 0) + 1
        comp_item = item.api_name in ctx.comp_item_apis
        prio = 1 if slam else 2
        out.append(
            (
                prio,
                value_of[item.api_name],
                ItemSuggestion(
                    item=item.name,
                    components=[components[i].name, components[j].name],
                    holder=holder.name if holder else None,
                    priority=prio,
                    reason=_reason(components[i].name, components[j].name, item, role, holder, comp_item, slam),
                ),
            )
        )

    # Leftover components that are half of a carry item: say what is missing.
    if ctx.comp_item_apis:
        held = {normalize_name(x) for u in ctx.units for x in u.items}
        for k, comp_item in enumerate(components):
            if k in used_idx:
                continue
            for api in sorted(ctx.comp_item_apis):
                target = set_data.items.get(api)
                if target is None or api in planned or comp_item.api_name not in target.composition:
                    continue
                if normalize_name(target.name) in held:
                    continue
                rest = list(target.composition)
                rest.remove(comp_item.api_name)
                other = set_data.items.get(rest[0]) if rest else None
                if other is None:
                    continue
                planned.add(api)
                holder_name = ctx.carry_unit.name if ctx.carry_unit is not None else (comp.carry if comp else None)
                out.append(
                    (
                        3,
                        0.0,
                        ItemSuggestion(
                            item=target.name,
                            components=[comp_item.name, other.name],
                            holder=holder_name,
                            priority=3,
                            reason=f"有{comp_item.name}，还差{other.name}就能合成{target.name}，选秀或野怪优先拿{other.name}",
                        ),
                    )
                )
                break

    out.sort(key=lambda t: (t[0], -t[1]))
    return [s for _p, _v, s in out]
