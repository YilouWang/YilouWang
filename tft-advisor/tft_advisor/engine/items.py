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
EMBLEM_NO_SYNERGY = 0.3  # an emblem for a trait nobody plays: below MIN_PAIR_VALUE, never made
# HP buckets that can wait for a carry item instead of slamming its component
# into something else (medium HP and below need the tempo).
HOLD_BUCKETS = ("healthy", "unknown")
SLAM_STAGE = (2, 5)

# Trait keywords (English + Chinese) hinting at a unit's damage profile.
# Mixed-damage traits stay out: Set 18 Executioner (裁决使) holds AD and AP
# carries, and the bare "战士" would match 魔战士 (Adaptor) and 狂战士
# (Ravager), which are not tanks. Set-specific profiles come from the comp
# library's item plans instead (profile_hints_from_comps).
AD_WORDS = (
    "gunslinger", "ranger", "marksman", "sniper", "blademaster", "duelist", "assassin", "slayer",
    "challenger", "deadeye", "quickstrike", "rapidfire", "striker", "gunner", "reaper",
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
    "骑士", "守护", "斗士", "护卫", "重装", "坦克", "守卫", "哨兵", "堡垒", "卫士", "巨像",
)
HINT_WEIGHT = 2.5  # profile known from the comp library outweighs trait keywords

# Items that are not stats for one unit (English names, normalized on use):
# Thief's Gloves takes all three slots and only goes on an itemless unit;
# +1 team size items belong on a cheap fielded unit, not on an itemized tank.
THIEFS_GLOVES = ("Thief's Gloves",)
TEAM_SIZE_ITEMS = ("Tactician's Crown", "Tactician's Cape", "Tactician's Shield")
EARLY_SLAM_STAGE = (2, 1)
LOSS_STREAK_HOLD = -3  # loss streaking into the carousel: hold components


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


def champion_profile(
    champ: Optional[Champion],
    unit: Optional[Unit] = None,
    set_data: Optional[SetData] = None,
    hint: Optional[str] = None,
) -> str:
    """'ad' | 'ap' | 'tank' | 'unknown' from cdragon role, api suffix, traits and items.

    ``hint``: profile from the comp library (the items strong players give
    this unit), weighted above trait keywords. Ties are broken by the items
    the unit already holds.
    """
    score = {"ad": 0.0, "ap": 0.0, "tank": 0.0}
    if hint in score:
        score[hint] += HINT_WEIGHT
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
    held = {"ad": 0.0, "ap": 0.0, "tank": 0.0}
    if unit is not None:
        for name in unit.items:
            item = set_data.resolve_item(name) if set_data else None
            r, _tier = item_role(item, name)
            if r in score:
                score[r] += 1.0
                held[r] += 1.0
        if unit.row == 0:
            score["tank"] += 0.5
        elif unit.row == 3:
            score["ad"] += 0.1
            score["ap"] += 0.1
    best = max(score, key=lambda k: (round(score[k], 6), held[k]))
    return best if score[best] > 0.3 else "unknown"


def profile_hints_from_comps(comps: Iterable, set_data: SetData, keep_ties: bool = False) -> dict[str, Optional[str]]:
    """Champion api name -> 'ad' / 'ap' / 'tank' from comp item plans.

    Counts the roles of the items each comp recommends for a unit
    (``item_holders``, plus ``carry_items`` for the carry) over the given
    comps; a unit gets a profile when one role clearly leads. With
    ``keep_ties`` a unit whose plan is mixed maps to None (so a single comp's
    mixed plan can cancel a library-wide guess: Set 18 Nidalee is AD in the
    Primal comps and AP in the Invoker ones).
    """
    counts: dict[str, dict[str, int]] = {}

    def add(unit_name: Optional[str], items: Iterable[str]) -> None:
        champ = set_data.resolve_champion(unit_name) if unit_name else None
        if champ is None:
            return
        c = counts.setdefault(champ.api_name, {"ad": 0, "ap": 0, "tank": 0})
        for raw in items:
            role, _tier = item_role(set_data.resolve_item(raw), raw)
            if role in c:
                c[role] += 1

    for comp in comps:
        holders = dict(getattr(comp, "item_holders", {}) or {})
        carry = getattr(comp, "carry", None)
        if carry and carry not in holders:
            holders[carry] = list(getattr(comp, "carry_items", []) or [])
        for unit_name, items in holders.items():
            add(unit_name, items)
    out: dict[str, Optional[str]] = {}
    for api, c in counts.items():
        ranked = sorted(c.items(), key=lambda kv: -kv[1])
        (role, n), (_r2, n2) = ranked[0], ranked[1]
        if n >= 1 and n > n2:
            out[api] = role
        elif keep_ties and n >= 1:
            out[api] = None
    return out


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
    comp_apis: set[str] = field(default_factory=set)
    comp_traits: set[str] = field(default_factory=set)  # traits of the target comp's units
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


def _carry_items_profile(comp: Optional[CompSuggestion], set_data: SetData) -> Optional[str]:
    """'ad' / 'ap' when the comp's carry items clearly say so."""
    if comp is None or not comp.carry_items:
        return None
    c = {"ad": 0, "ap": 0}
    for raw in comp.carry_items:
        role, _tier = item_role(set_data.resolve_item(raw), raw)
        if role in c:
            c[role] += 1
    if c["ad"] == c["ap"]:
        return None
    return "ad" if c["ad"] > c["ap"] else "ap"


def _build_ctx(
    state: GameState,
    set_data: SetData,
    comp: Optional[CompSuggestion],
    profile_hints: Optional[dict[str, str]] = None,
) -> _Ctx:
    ctx = _Ctx(set_data=set_data, board=list(state.board), bench=list(state.bench))
    hints = dict(profile_hints or {})
    carry_champ = set_data.resolve_champion(comp.carry) if comp is not None and comp.carry else None
    carry_hint = _carry_items_profile(comp, set_data)
    if carry_champ is not None and carry_hint:
        hints[carry_champ.api_name] = carry_hint  # the comp's own carry items decide
    for u in ctx.units:
        champ = set_data.champions.get(u.api_name)
        ctx.profiles[id(u)] = champion_profile(champ, u, set_data, hint=hints.get(u.api_name))
        ctx.load[id(u)] = len(u.items)
    seen_traits: dict[str, set[str]] = {}
    for u in ctx.units:
        if u.api_name.startswith("?"):
            continue
        for t in u.traits:
            seen_traits.setdefault(t, set()).add(u.api_name)
    ctx.trait_counts = {t: len(v) for t, v in seen_traits.items()}
    if comp is not None:
        for name in [*comp.core_units, *comp.have_units]:
            champ = set_data.resolve_champion(name)
            if champ is not None:
                ctx.comp_apis.add(champ.api_name)
        for name in [*comp.core_units, *comp.have_units, *comp.missing_units, comp.carry]:
            champ = set_data.resolve_champion(name) if name else None
            if champ is not None:
                ctx.comp_traits.update(champ.traits)
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
                    ctx.carry_profile = champion_profile(champ, None, set_data, hint=hints.get(champ.api_name))
    return ctx


def _is_named(item: Item, names: tuple[str, ...]) -> bool:
    keys = {normalize_name(n) for n in names}
    return any(normalize_name(n) in keys for n in (item.name_en, item.name) if n)


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
            if not count and trait.name not in ctx.comp_traits and item.api_name not in ctx.comp_item_apis:
                # Nobody on the board or in the target comp has this trait: an
                # emblem for it only burns two components.
                return EMBLEM_NO_SYNERGY
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
        # Fielded units first (items on a bench unit do nothing), then comp units, then power.
        if not cands:
            return None
        return max(
            cands,
            key=lambda u: (ctx.on_board(u), u.api_name in ctx.comp_apis, _power(u), u.row or 0, len(u.items), u.name),
        )

    if _is_named(item, THIEFS_GLOVES):
        # Uses all three slots: only a fielded unit with no items, and never
        # the carry (random items on the main carry are wasted).
        empty = [
            u
            for u in ctx.board
            if ctx.load.get(id(u), 0) == 0 and not u.api_name.startswith("?") and u is not ctx.carry_unit
        ]
        return best(empty)
    if _is_named(item, TEAM_SIZE_ITEMS):
        # +1 team size: park it on a cheap fielded unit, not on the itemized
        # tank or carry.
        spare = [u for u in units if ctx.on_board(u) and u is not ctx.carry_unit]
        if spare:
            return min(spare, key=lambda u: (ctx.load.get(id(u), 0), _power(u), u.name))
        return best(units)
    if item.api_name in ctx.comp_item_apis and carry is not None:
        return carry
    if role in ("ad", "ap"):
        if carry is not None and ctx.profile(carry) in (role, "unknown"):
            return carry
        other = "ap" if role == "ad" else "ad"
        match = best([u for u in units if ctx.profile(u) == role]) or best(
            [u for u in units if ctx.profile(u) == "unknown"]
        )
        if match is not None:
            return match
        # Never hand an AD item to an AP carry (or the reverse): on a team of
        # the other damage type the item waits for a better use.
        if any(ctx.profile(u) == other for u in ctx.units):
            return None
        return best(units)
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
        fielded = [u for u in lacking if ctx.on_board(u)]
        if fielded:
            # A fielded unit whose slots matter least: not the carry, fewest items.
            return max(
                fielded,
                key=lambda u: (
                    u is not ctx.carry_unit,
                    -ctx.load.get(id(u), 0),
                    u.api_name in ctx.comp_apis,
                    _power(u),
                    u.name,
                ),
            )
        return best(lacking) or best(units)
    # flex
    if carry is not None:
        return carry
    return best([u for u in units if ctx.profile(u) not in ("tank",)]) or best(units)


def _reason(a: str, b: str, item: Item, role: str, holder: Optional[Unit], comp_item: bool, slam: bool) -> str:
    parts = [f"{a}+{b} 合成{item.name}（{ROLE_ZH.get(role, '通用')}装）"]
    if holder is not None:
        parts.append(f"给{holder.name}")
    elif _is_named(item, THIEFS_GLOVES):
        parts.append("场上没有空装备的英雄，先留着")
    elif role in ("ad", "ap"):
        parts.append("现在的阵容没人适合用")
    if comp_item:
        parts.append("阵容主C核心装备")
    parts.append("现在就合，早成型少掉血" if slam else "先留散件，关键回合再合")
    return "，".join(parts)


def plan_items(
    state: GameState,
    set_data: SetData,
    comp: Optional[CompSuggestion] = None,
    hp_bucket: str = "unknown",
    profile_hints: Optional[dict[str, str]] = None,
) -> list[ItemSuggestion]:
    """Suggested item combinations (priority 1 = slam now) plus equip hints.

    Slam timing: from 2-5 (or at low HP) every pair is slammed. From 2-1 a
    pair is slammed early when it makes a core (tier 3) item, a carry item of
    the target comp, or a tank item for a fielded front-row unit, unless the
    player is loss streaking (then components are held for the carousel).
    ``profile_hints``: champion api -> 'ad' / 'ap' / 'tank' from the comp
    library (see ``profile_hints_from_comps``).
    """
    ctx = _build_ctx(state, set_data, comp, profile_hints)

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
    loss_streaking = state.streak is not None and state.streak <= LOSS_STREAK_HOLD
    early = not slam and stage_key >= EARLY_SLAM_STAGE and not loss_streaking

    def early_slam(item: Item, role: str, tier: int, holder: Optional[Unit]) -> bool:
        if not early:
            return False
        if tier >= 3 or item.api_name in ctx.comp_item_apis:
            return True
        return role == "tank" and holder is not None and ctx.on_board(holder) and holder.row == 0

    out: list[tuple[int, float, ItemSuggestion]] = []

    # Completed items sitting on the bench: equip them now.
    for item in completed_on_bench:
        role, _tier = item_role(item)
        holder = _pick_holder(role, item, ctx)
        thief = _is_named(item, THIEFS_GLOVES)
        if holder is not None:
            ctx.load[id(holder)] = MAX_ITEMS_PER_UNIT if thief else ctx.load.get(id(holder), 0) + 1
        if holder is not None:
            where = f"，装给{holder.name}"
        elif thief and ctx.units:
            where = "，只能给没有装备的场上英雄，先留着"
        elif ctx.units and not any(ctx.free(u) for u in ctx.units):
            where = "，英雄的装备格都满了，先留着"
        elif ctx.units:
            where = "，现在的阵容没人适合用，先留着"
        else:
            where = "，先上场一个英雄再装备"
        text = f"{item.name} 已经合成好了还放在备战席{where}"
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
    # Components the comp's carry items still need (not held by a unit, not
    # made by another pair) beyond the ones left unpaired: at healthy HP a
    # pair that eats one of them waits for the carry item.
    held_names = {normalize_name(x) for u in ctx.units for x in u.items}
    made = {p[2].api_name for p in pairs}
    reserve: dict[str, list[str]] = {}
    for api in sorted(ctx.comp_item_apis):
        target = set_data.items.get(api)
        if target is None or api in made or normalize_name(target.name) in held_names or target.kind != "completed":
            continue
        for part in target.composition:
            reserve.setdefault(part, []).append(target.name)
    paired = {k for i, j, _it in pairs for k in (i, j)}
    for k, c in enumerate(components):
        if k not in paired and reserve.get(c.api_name):
            reserve[c.api_name].pop()  # a spare copy already covers it
    used_idx: set[int] = set()
    planned: set[str] = set()
    for i, j, item in pairs:
        used_idx.update((i, j))
        planned.add(item.api_name)
        role, tier = item_role(item)
        holder = _pick_holder(role, item, ctx)
        if holder is not None:
            # Thief's Gloves fills every slot of its holder.
            ctx.load[id(holder)] = (
                MAX_ITEMS_PER_UNIT if _is_named(item, THIEFS_GLOVES) else ctx.load.get(id(holder), 0) + 1
            )
        comp_item = item.api_name in ctx.comp_item_apis
        now = slam or early_slam(item, role, tier, holder)
        if role in ("ad", "ap") and holder is None and ctx.units and hp_bucket not in ("low", "critical"):
            now = False  # nobody on this team uses it: keep the components
        saved_for: Optional[str] = None
        if not comp_item and hp_bucket in HOLD_BUCKETS:
            for c in (components[i], components[j]):
                if reserve.get(c.api_name):
                    saved_for = reserve[c.api_name].pop()
                    now = False
                    break
        prio = 1 if now else 2
        out.append(
            (
                prio,
                value_of[item.api_name],
                ItemSuggestion(
                    item=item.name,
                    components=[components[i].name, components[j].name],
                    holder=holder.name if holder else None,
                    priority=prio,
                    reason=_reason(components[i].name, components[j].name, item, role, holder, comp_item, now)
                    + (f"（散件留着做主C的{saved_for}）" if saved_for else ""),
                ),
            )
        )

    # Leftover components that are half of a carry item: say what is missing
    # (not when the carry's three slots are already taken or planned).
    carry_full = ctx.carry_unit is not None and not ctx.free(ctx.carry_unit)
    if ctx.comp_item_apis and not carry_full:
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
