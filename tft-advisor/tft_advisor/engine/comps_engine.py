"""Comp fit scoring: which target composition matches the current board best.

Score (0..1) = 0.50 * unit overlap (weighted by cost and star)
             + 0.15 * carry owned (1-star 0.6, 2-star+ 1.0)
             + 0.10 * item fit (carry items held or buildable from components)
             + 0.10 * level plan fit (reroll comps after overleveling, fast 8 at low HP)
             + 0.15 * comp tier (S/A/B/C)
             - contested penalty (0.1 per opponent holding 2+ core units or 2+ carry copies)
             + comp hint boost (the player said what they want to play).

Without comp definitions (a new set, no user file) ``auto_comps`` proposes
trait-based directions: the next breakpoint of traits that are active or
close, filled with the cheapest missing units that also share traits with
the current board.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from ..data.comps import CompDef
from ..data.setdata import Champion, SetData, Trait, normalize_name
from ..models import CompSuggestion, GameState
from .economy import hp_bucket, reroll_level
from .items import champion_profile

TIER_SCORE = {"S": 1.0, "A": 0.75, "B": 0.5, "C": 0.3}
STAR_MULT = {1: 1.0, 2: 1.35, 3: 1.6}
W_OVERLAP, W_CARRY, W_ITEMS, W_LEVEL, W_TIER = 0.50, 0.15, 0.10, 0.10, 0.15
CONTEST_PENALTY = 0.10
CONTEST_CAP = 0.30
HINT_NAME_BOOST = 0.25
HINT_CARRY_BOOST = 0.20
HINT_UNIT_BOOST = 0.10
HINT_CAP = 0.40


@dataclass
class Owned:
    api_name: str
    name: str
    copies: int
    star: int
    cost: int


def owned_units(state: GameState, set_data: SetData) -> dict[str, Owned]:
    """Resolved units on board + bench aggregated per champion."""
    out: dict[str, Owned] = {}
    for u in state.all_units():
        if u.api_name.startswith("?"):
            continue
        champ = set_data.champions.get(u.api_name)
        cost = u.cost if u.cost is not None else (champ.cost if champ else 1)
        cur = out.get(u.api_name)
        if cur is None:
            out[u.api_name] = Owned(u.api_name, u.name, u.copies, max(1, u.star), cost)
        else:
            cur.copies += u.copies
            cur.star = max(cur.star, u.star)
    return out


def _weight(cost: int) -> float:
    return 1.0 + 0.25 * max(1, cost)


def _resolve_units(names: list[str], set_data: SetData) -> list[Champion]:
    out: list[Champion] = []
    seen: set[str] = set()
    for n in names:
        champ = set_data.resolve_champion(n)
        if champ is not None and champ.api_name not in seen:
            seen.add(champ.api_name)
            out.append(champ)
    return out


def _item_fit(state: GameState, set_data: SetData, carry_items: list[str]) -> Optional[float]:
    targets = [it for it in (set_data.resolve_item(n) for n in carry_items) if it is not None]
    if not targets:
        return None
    held = Counter()
    components = Counter()
    for u in state.all_units():
        for name in u.items:
            it = set_data.resolve_item(name)
            if it is not None:
                held[it.api_name] += 1
    for name in state.item_bench:
        it = set_data.resolve_item(name)
        if it is None:
            continue
        if it.kind == "component":
            components[it.api_name] += 1
        else:
            held[it.api_name] += 1
    done = 0
    for it in targets:
        if held[it.api_name] > 0:
            held[it.api_name] -= 1
            done += 1
            continue
        need = Counter(it.composition)
        if need and all(components[c] >= k for c, k in need.items()):
            components.subtract(need)
            done += 1
    return done / len(targets)


def _level_fit(state: GameState, style: str) -> float:
    lvl = state.level
    if lvl is None:
        return 0.8
    rr = reroll_level(style)
    if rr is not None:
        if lvl > rr + 1:
            return 0.3
        if lvl == rr + 1:
            return 0.7
        return 1.0
    if style == "fast8":
        early = state.stage is None or state.stage.key < (4, 1)
        if hp_bucket(state.hp) in ("low", "critical") and early:
            return 0.4
    return 1.0


def _hint_tokens(hint: str) -> list[str]:
    return [t for t in re.split(r"[\s,，、/;；|]+", hint or "") if t.strip()]


def _hint_boost(comp_name: str, champs: list[Champion], carry: Optional[Champion], hint: str, set_data: SetData) -> float:
    h = normalize_name(hint)
    if not h:
        return 0.0
    boost = 0.0
    n = normalize_name(comp_name)
    tokens = [normalize_name(t) for t in _hint_tokens(hint)]
    tokens = [t for t in tokens if t]
    if n and (h in n or n in h or any(len(t) >= 2 and (t in n or n in t) for t in tokens)):
        boost += HINT_NAME_BOOST
    hinted: set[str] = set()
    for t in [hint, *_hint_tokens(hint)]:
        champ = set_data.resolve_champion(t)
        if champ is not None:
            hinted.add(champ.api_name)
    if carry is not None and carry.api_name in hinted:
        boost += HINT_CARRY_BOOST
    elif hinted & {c.api_name for c in champs}:
        boost += HINT_UNIT_BOOST
    return min(HINT_CAP, boost)


def _contested(core: set[str], carry_api: Optional[str], taken_by_player: dict[str, dict[str, int]]) -> list[str]:
    out: list[str] = []
    for player, counts in sorted(taken_by_player.items()):
        held = [api for api in core if counts.get(api, 0) > 0]
        if len(held) >= 2 or (carry_api is not None and counts.get(carry_api, 0) >= 2):
            out.append(player)
    return out


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_comp(
    comp: CompDef,
    state: GameState,
    set_data: SetData,
    taken_by_player: dict[str, dict[str, int]],
    hint: str = "",
    owned: Optional[dict[str, Owned]] = None,
) -> Optional[CompSuggestion]:
    champs = _resolve_units(comp.units, set_data)
    carry = set_data.resolve_champion(comp.carry) if comp.carry else None
    if carry is not None and carry.api_name not in {c.api_name for c in champs}:
        champs.insert(0, carry)
    if not champs:
        return None
    owned = owned if owned is not None else owned_units(state, set_data)

    den = sum(_weight(c.cost) for c in champs)
    num = sum(_weight(c.cost) * STAR_MULT[min(3, owned[c.api_name].star)] for c in champs if c.api_name in owned)
    overlap = _clamp(num / den) if den else 0.0

    if carry is not None:
        oc = owned.get(carry.api_name)
        carry_score = 0.0 if oc is None else (0.6 if oc.star <= 1 and oc.copies < 3 else 1.0)
    else:
        carry_score = overlap
    fit = _item_fit(state, set_data, comp.carry_items)
    item_score = 0.5 if fit is None else fit
    level_score = _level_fit(state, comp.style)
    tier_score = TIER_SCORE.get(comp.tier.upper()[:1] if comp.tier else "", 0.5)

    core = {c.api_name for c in champs if c.cost >= 3}
    if carry is not None:
        core.add(carry.api_name)
    if len(core) < 2:
        core = {c.api_name for c in champs}
    contested = _contested(core, carry.api_name if carry else None, taken_by_player)
    penalty = min(CONTEST_CAP, CONTEST_PENALTY * len(contested))
    boost = _hint_boost(comp.name, champs, carry, hint, set_data)

    score = _clamp(
        W_OVERLAP * overlap
        + W_CARRY * carry_score
        + W_ITEMS * item_score
        + W_LEVEL * level_score
        + W_TIER * tier_score
        - penalty
        + boost
    )

    have = [c.name for c in champs if c.api_name in owned]
    missing = [c.name for c in sorted((c for c in champs if c.api_name not in owned), key=lambda c: (c.cost, c.name_en))]
    parts = [f"已有 {len(have)}/{len(champs)} 个阵容英雄"]
    if carry is not None:
        oc = owned.get(carry.api_name)
        if oc is None:
            parts.append(f"还缺主C {carry.name}")
        else:
            parts.append(f"主C {carry.name} 已到手" + ("（已两星）" if oc.star >= 2 else ""))
    if fit is not None and comp.carry_items:
        parts.append(f"核心装备可做 {round(fit * len(comp.carry_items))}/{len(comp.carry_items)} 件")
    if contested:
        parts.append(f"{', '.join(contested)} 也在拿这些英雄，注意卡牌")
    if boost > 0:
        parts.append("符合你指定的方向")
    if level_score < 0.5:
        parts.append("当前等级或血量不太适合这套打法")

    return CompSuggestion(
        name=comp.name,
        score=round(score, 3),
        core_units=[c.name for c in champs],
        have_units=have,
        missing_units=missing,
        carry=carry.name if carry else None,
        carry_items=[(set_data.resolve_item(n).name if set_data.resolve_item(n) else n) for n in comp.carry_items],
        contested_by=contested,
        reason="，".join(parts),
    )


def suggest_comps(
    state: GameState,
    set_data: SetData,
    comps: Optional[list[CompDef]],
    taken_by_player: Optional[dict[str, dict[str, int]]] = None,
    top_n: int = 3,
    hint: str = "",
) -> list[CompSuggestion]:
    """Best ``top_n`` comps for the current state (auto comps when ``comps`` is empty)."""
    taken = {p: c for p, c in (taken_by_player or {}).items() if p and p != state.self_name}
    if not comps:
        return auto_comps(state, set_data, taken, top_n)
    owned = owned_units(state, set_data)
    scored = [s for s in (score_comp(c, state, set_data, taken, hint, owned) for c in comps) if s is not None]
    scored.sort(key=lambda s: (-s.score, s.name))
    return scored[: max(0, top_n)]


# ---------------------------------------------------------------------------
# Trait based auto comps
# ---------------------------------------------------------------------------


def _trait_lookup(set_data: SetData) -> dict[str, Trait]:
    out: dict[str, Trait] = {}
    for t in set_data.traits.values():
        for n in (t.name, t.name_en, t.api_name):
            if n:
                out.setdefault(n, t)
    return out


def _champ_traits(champ: Champion, lookup: dict[str, Trait]) -> list[Trait]:
    out: list[Trait] = []
    for n in [*champ.traits, *getattr(champ, "traits_en", [])]:
        t = lookup.get(n)
        if t is not None and t not in out:
            out.append(t)
    return out


def auto_comps(
    state: GameState,
    set_data: SetData,
    taken_by_player: Optional[dict[str, dict[str, int]]] = None,
    top_n: int = 3,
) -> list[CompSuggestion]:
    """Trait breakpoint suggestions built from the current board + bench."""
    taken = {p: c for p, c in (taken_by_player or {}).items() if p and p != state.self_name}
    owned = owned_units(state, set_data)
    if not owned:
        return []
    lookup = _trait_lookup(set_data)
    members: dict[str, set[str]] = {}
    owned_traits: Counter[str] = Counter()
    for api in owned:
        champ = set_data.champions.get(api)
        if champ is None:
            continue
        for t in _champ_traits(champ, lookup):
            members.setdefault(t.api_name, set()).add(api)
            owned_traits[t.api_name] += 1

    out: list[CompSuggestion] = []
    for trait_api, have_apis in members.items():
        trait = set_data.traits[trait_api]
        bps = sorted(b for b in trait.breakpoints if b > 0)
        if not bps:
            continue
        count = len(have_apis)
        nxt = next((b for b in bps if b > count), None)
        if nxt is None:
            continue
        active = count >= bps[0]
        need = nxt - count
        if not active and need > 2:
            continue
        pool = [
            c
            for c in set_data.champions.values()
            if c.api_name not in have_apis and trait in _champ_traits(c, lookup)
        ]
        if len(pool) < need:
            continue

        def synergy(c: Champion) -> int:
            return sum(owned_traits[t.api_name] for t in _champ_traits(c, lookup) if t.api_name != trait_api)

        pool.sort(key=lambda c: (c.cost, -synergy(c), c.name_en))
        missing = pool[:need]

        have_champs = [set_data.champions[a] for a in sorted(have_apis)]
        rank = bps.index(nxt) + 1
        score = 0.2 + 0.5 * (count / nxt) + 0.2 * (rank / len(bps)) - 0.05 * max(0, need - 1)
        if active:
            score += 0.1

        involved = {c.api_name for c in have_champs} | {c.api_name for c in missing}
        contested = _contested(involved, None, taken)
        score -= min(CONTEST_CAP, 0.05 * len(contested))

        # Carry: strongest owned non-tank member, else the strongest member.
        def power(c: Champion) -> tuple[int, int]:
            o = owned.get(c.api_name)
            return ((c.cost * (o.star if o else 1)), c.cost)

        dps = [c for c in have_champs if champion_profile(c) in ("ad", "ap")]
        carry = max(dps or have_champs, key=power) if have_champs else None

        missing_names = [c.name for c in missing]
        reason = f"已有 {count} 个{trait.name}，再补 {need} 个（{', '.join(missing_names)}）就能到 {nxt} 档"
        if active:
            reason += "，羁绊已激活，顺着升级"
        if contested:
            reason += f"，{', '.join(contested)} 也在玩这些英雄"
        out.append(
            CompSuggestion(
                name=f"{trait.name} {nxt}",
                score=round(_clamp(score), 3),
                core_units=[c.name for c in have_champs] + missing_names,
                have_units=[c.name for c in have_champs],
                missing_units=missing_names,
                carry=carry.name if carry else None,
                carry_items=[],
                contested_by=contested,
                reason=reason,
            )
        )
    out.sort(key=lambda s: (-s.score, s.name))
    return out[: max(0, top_n)]
