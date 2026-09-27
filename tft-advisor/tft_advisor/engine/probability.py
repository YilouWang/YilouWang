"""Shop / pool probability model.

Shop generation (per slot): a cost tier is drawn from the level's shop odds,
then a champion is drawn from that tier weighted by the copies still in the
shared pool. So for a champion X of cost c::

    P(slot shows X) = odds[level][c] * remaining(X) / remaining(all cost-c copies)

A roll shows ``shop_slots`` (5) independent slots, except that Set 18 hides
the rightmost champion of every other shop under a Wisp (``wisp_every``; a
Wisp in every shop with Blossom 5), so shops alternate 5 and 4 champion
slots (``ShopModel.schedule``; the phase of the first roll is averaged).
Inferno 3/5/7 makes 1/2/4 slots roll one cost higher (``ShopModel.upgraded``):
such a slot draws a tier from the odds and shows the next cost up.

For a roll-down we model a Markov chain over (copies bought so far, gold
left, shop phase): each roll costs ``roll_cost``; every copy of X that shows
up is bought (if gold allows) until the goal is reached. Buying removes
copies from the pool, which lowers P for the next roll. The within-shop
depletion is ignored (tiny effect).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Iterable, Optional

from ..data.mechanics import Mechanics
from ..data.setdata import SetData, normalize_name
from ..models import GameState, HitOdds, Unit

GOLD_BUDGETS = (10, 20, 30, 40, 50, 60, 80)


@dataclass(frozen=True)
class ShopModel:
    """How many champion slots each consecutive shop shows (repeating) and how
    many of them roll one cost higher (Set 18 Inferno)."""

    schedule: tuple[int, ...] = (5,)
    upgraded: int = 0


def default_shop(mech: Mechanics) -> ShopModel:
    return ShopModel(schedule=mech.shop_schedule(), upgraded=0)


def _trait_counts(state: GameState, set_data: SetData) -> dict[str, int]:
    """Normalized English trait name -> active count (fielded units or the
    trait tracker, whichever is higher)."""
    members: dict[str, set[str]] = {}
    for u in state.board:
        if u.api_name.startswith("?"):
            continue
        for t in u.traits:
            trait = set_data.resolve_trait(t)
            if trait is not None:
                members.setdefault(normalize_name(trait.name_en or trait.name), set()).add(u.api_name)
    counts = {k: len(v) for k, v in members.items()}
    for obs in state.traits:
        trait = set_data.resolve_trait(obs.name)
        if trait is not None and obs.count:
            key = normalize_name(trait.name_en or trait.name)
            counts[key] = max(counts.get(key, 0), int(obs.count))
    return counts


def shop_model(state: GameState, set_data: SetData, mech: Mechanics) -> ShopModel:
    """Shop shape for this player: Wisp schedule plus trait effects listed in
    ``mech.extra`` (``wisp_every_shop``: trait -> count; ``upgraded_slots``:
    trait -> {count: slots})."""
    extra = mech.extra if isinstance(mech.extra, dict) else {}
    schedule = mech.shop_schedule()
    upgraded = 0
    every = extra.get("wisp_every_shop") or {}
    ups = extra.get("upgraded_slots") or {}
    if not every and not ups:
        return ShopModel(schedule, 0)
    counts = _trait_counts(state, set_data)
    for trait, need in every.items():
        try:
            if counts.get(normalize_name(trait), 0) >= int(need):
                schedule = mech.shop_schedule(1)
        except (TypeError, ValueError):
            continue
    for trait, table in ups.items():
        have = counts.get(normalize_name(trait), 0)
        if not isinstance(table, dict):
            continue
        for k, v in table.items():
            try:
                if have >= int(k):
                    upgraded = max(upgraded, int(v))
            except (TypeError, ValueError):
                continue
    return ShopModel(schedule, upgraded)


def odds_upgraded(mech: Mechanics, level: int, cost: int) -> float:
    """P(an upgraded slot shows cost ``cost``): it draws a tier from the level
    odds and shows the next cost up (the top cost stays the top cost)."""
    if cost <= 1:
        return 0.0
    p = mech.odds(level, cost - 1)
    if mech.odds(level, cost + 1) == 0.0 and cost >= 5:
        p += mech.odds(level, cost)
    return p


def _slot_probs(
    mech: Mechanics, level: int, cost: int, remaining_unit: int, remaining_cost_total: int
) -> tuple[float, float]:
    """(normal slot, upgraded slot) probability of showing this champion."""
    if remaining_unit <= 0 or remaining_cost_total <= 0:
        return 0.0, 0.0
    ratio = min(1.0, remaining_unit / remaining_cost_total)
    return mech.odds(level, cost) * ratio, odds_upgraded(mech, level, cost) * ratio


def _hits_dist(slots: int, upgraded: int, p: float, p_up: float) -> list[float]:
    """P(h copies shown in one shop) for h = 0..slots."""
    n_up = max(0, min(upgraded, slots))
    n = slots - n_up
    a = [_binom_pmf(n, k, p) for k in range(n + 1)]
    if n_up == 0:
        return a
    b = [_binom_pmf(n_up, k, p_up) for k in range(n_up + 1)]
    out = [0.0] * (slots + 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def p_shop_shows(mech: Mechanics, level: int, cost: int, remaining_unit: int, remaining_cost_total: int,
                 shop: Optional[ShopModel] = None) -> float:
    """P(one fresh shop shows at least one copy), averaged over the schedule."""
    shop = shop or default_shop(mech)
    p, p_up = _slot_probs(mech, level, cost, remaining_unit, remaining_cost_total)
    vals = [1.0 - _hits_dist(n, shop.upgraded, p, p_up)[0] for n in shop.schedule]
    return sum(vals) / len(vals) if vals else 0.0


def p_unit_per_slot(mech: Mechanics, level: int, cost: int, remaining_unit: int, remaining_cost_total: int) -> float:
    if remaining_unit <= 0 or remaining_cost_total <= 0:
        return 0.0
    return mech.odds(level, cost) * min(1.0, remaining_unit / remaining_cost_total)


def p_at_least_one(p_slot: float, slots: int = 5) -> float:
    return 1.0 - (1.0 - p_slot) ** slots


def _binom_pmf(n: int, k: int, p: float) -> float:
    return comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))


def rolldown_curve(
    mech: Mechanics,
    level: int,
    cost: int,
    need: int,
    remaining_unit: int,
    remaining_cost_total: int,
    budgets: Iterable[int],
    unit_price: Optional[int] = None,
    shop: Optional[ShopModel] = None,
) -> dict[int, float]:
    """``rolldown_probability`` for several gold budgets from one table.

    The chain is solved bottom-up over gold (every roll strictly lowers the
    gold, so a state only depends on smaller gold states): one pass up to the
    largest budget answers every smaller budget, and there is no recursion
    depth limit for large budgets.
    """
    wanted = sorted({max(0, int(g)) for g in budgets})
    if not wanted:
        return {}
    if need <= 0:
        return {g: 1.0 for g in wanted}
    price = unit_price if unit_price is not None else cost
    if remaining_unit < need:
        return {g: 0.0 for g in wanted}
    shop = shop or default_shop(mech)
    schedule = tuple(max(1, int(n)) for n in shop.schedule) or (max(1, mech.shop_slots),)
    phases = len(schedule)
    roll = max(1, mech.roll_cost)  # a 0-cost override would never terminate
    top = wanted[-1]

    # Hit distribution per (copies bought, shop phase): it depends on the
    # pool left and the shop size only, not on the gold.
    dists: list[list[Optional[list[float]]]] = []
    for bought in range(need):
        p, p_up = _slot_probs(mech, level, cost, remaining_unit - bought, remaining_cost_total - bought)
        dead = p <= 0.0 and (p_up <= 0.0 or shop.upgraded <= 0)
        dists.append([None if dead else _hits_dist(schedule[ph], shop.upgraded, p, p_up) for ph in range(phases)])

    # table[bought][phase][g] = P(goal | bought so far, gold g, next shop phase)
    table = [[[0.0] * (top + 1) for _ in range(phases)] for _ in range(need + 1)]
    for ph in range(phases):
        table[need][ph] = [1.0] * (top + 1)
    for g in range(roll + price, top + 1):
        g_after_roll = g - roll
        for bought in range(need):
            for ph in range(phases):
                dist = dists[bought][ph]
                if dist is None:
                    continue
                nph = (ph + 1) % phases
                total = 0.0
                # Hits h >= 1: buy up to what's needed and affordable.
                for h in range(1, len(dist)):
                    ph_h = dist[h]
                    if ph_h < 1e-12:
                        continue
                    can_buy = min(h, need - bought, g_after_roll // price)
                    total += ph_h * table[bought + can_buy][nph][g_after_roll - can_buy * price]
                # No hit: keep rolling (a strictly smaller gold state).
                total += dist[0] * table[bought][nph][g_after_roll]
                table[bought][ph][g] = total

    out: dict[int, float] = {}
    for g in wanted:
        if g < mech.roll_cost + price:
            out[g] = 0.0
            continue
        # The phase of the first roll is unknown: average over it.
        p_goal = sum(table[0][ph][g] for ph in range(phases)) / phases
        out[g] = min(1.0, max(0.0, p_goal))
    return out


def rolldown_probability(
    mech: Mechanics,
    level: int,
    cost: int,
    need: int,
    remaining_unit: int,
    remaining_cost_total: int,
    gold: int,
    unit_price: Optional[int] = None,
    shop: Optional[ShopModel] = None,
) -> float:
    """P(buy ``need`` more copies of one champion) spending at most ``gold``.

    Every roll costs ``mech.roll_cost``; each copy costs ``unit_price``
    (defaults to ``cost``). The shop currently on screen is NOT included
    (callers account for it separately). ``shop`` defaults to the Wisp
    schedule of ``mech`` with no upgraded slots.
    """
    if need <= 0:
        return 1.0
    g = max(0, int(gold))
    return rolldown_curve(
        mech, level, cost, need, remaining_unit, remaining_cost_total, (g,), unit_price=unit_price, shop=shop
    )[g]


def expected_gold_to_goal(
    mech: Mechanics,
    level: int,
    cost: int,
    need: int,
    remaining_unit: int,
    remaining_cost_total: int,
    shop: Optional[ShopModel] = None,
) -> Optional[float]:
    """Expected gold (rolls + purchases) to find ``need`` more copies."""
    if need <= 0:
        return 0.0
    if remaining_unit < need:
        return None
    total = 0.0
    for k in range(need):
        hit = p_shop_shows(mech, level, cost, remaining_unit - k, remaining_cost_total - k, shop)
        if hit <= 1e-9:
            return None
        total += max(1, mech.roll_cost) / hit + cost
    return total


# ---------------------------------------------------------------------------
# Pool accounting
# ---------------------------------------------------------------------------


def copies_needed(owned: int, goal_star: int) -> int:
    return max(0, 3 ** (goal_star - 1) - owned)


def pool_remaining(
    set_data: SetData, mech: Mechanics, taken: dict[str, int]
) -> tuple[dict[str, int], dict[int, int]]:
    """Remaining copies per champion api name and per cost tier."""
    per_unit: dict[str, int] = {}
    per_cost: dict[int, int] = {}
    for api, champ in set_data.champions.items():
        size = mech.pool_size.get(champ.cost, 0)
        left = max(0, size - int(taken.get(api, 0)))
        per_unit[api] = left
        per_cost[champ.cost] = per_cost.get(champ.cost, 0) + left
    return per_unit, per_cost


def owned_copies(units: Iterable[Unit]) -> dict[str, int]:
    out: dict[str, int] = {}
    for u in units:
        if u.api_name.startswith("?"):
            continue
        out[u.api_name] = out.get(u.api_name, 0) + u.copies
    return out


def default_goal_star(cost: int, owned: int, level: Optional[int]) -> int:
    """2-star for most units; 3-star for cheap units once you have a pair+."""
    if cost <= 2 and owned >= 4:
        return 3
    if cost == 3 and owned >= 5 and (level or 0) <= 7:
        return 3
    return 2 if owned < 3 else 3


def compute_hit_odds(
    state: GameState,
    set_data: SetData,
    mech: Mechanics,
    taken_by_others: dict[str, int],
    targets: Optional[list[str]] = None,
    budgets: tuple[int, ...] = GOLD_BUDGETS,
    level: Optional[int] = None,
    shop: Optional[ShopModel] = None,
) -> list[HitOdds]:
    """Odds for the units that matter.

    ``taken_by_others``: known copies held by other players (from scouting).
    ``targets``: api names or display names; default = units we own that are
    not yet 3-star. ``shop``: defaults to the player's shop shape (Wisps,
    Blossom / Inferno effects from the fielded traits).
    """
    lvl = level or state.level or 1
    shop = shop or shop_model(state, set_data, mech)
    own = owned_copies(state.all_units())
    taken = dict(taken_by_others)
    for api, n in own.items():
        taken[api] = taken.get(api, 0) + n
    per_unit, per_cost = pool_remaining(set_data, mech, taken)

    wanted: list[str] = []
    if targets is not None:
        for t in targets:
            champ = set_data.champions.get(t) or set_data.resolve_champion(t)
            if champ and champ.api_name not in wanted:
                wanted.append(champ.api_name)
    else:
        for api, n in sorted(own.items(), key=lambda kv: (-kv[1], kv[0])):
            if n < 9 and api in set_data.champions:
                wanted.append(api)

    result: list[HitOdds] = []
    for api in wanted:
        champ = set_data.champions[api]
        n_own = own.get(api, 0)
        goal = default_goal_star(champ.cost, n_own, lvl)
        need = copies_needed(n_own, goal)
        rem = per_unit.get(api, 0)
        rem_cost = per_cost.get(champ.cost, 0)
        p_slot = p_unit_per_slot(mech, lvl, champ.cost, rem, rem_cost)
        curve = rolldown_curve(mech, lvl, champ.cost, need, rem, rem_cost, budgets, shop=shop)
        by_gold = {int(g): round(curve[max(0, int(g))], 4) for g in budgets}
        result.append(
            HitOdds(
                unit=champ.name,
                api_name=api,
                cost=champ.cost,
                owned_copies=n_own,
                goal_copies=3 ** (goal - 1),
                goal_star=goal,
                seen_elsewhere=int(taken_by_others.get(api, 0)),
                remaining_in_pool=rem,
                level=lvl,
                p_per_slot=round(p_slot, 5),
                p_in_shop=round(p_shop_shows(mech, lvl, champ.cost, rem, rem_cost, shop), 4),
                expected_gold_to_goal=(
                    round(e, 1)
                    if (e := expected_gold_to_goal(mech, lvl, champ.cost, need, rem, rem_cost, shop)) is not None
                    else None
                ),
                p_goal_by_gold=by_gold,
            )
        )
    return result


def best_roll_level(
    mech: Mechanics,
    level: int,
    xp_current: Optional[int],
    gold: int,
    cost: int,
    need: int,
    remaining_unit: int,
    remaining_cost_total: int,
    max_extra_levels: int = 2,
    shop: Optional[ShopModel] = None,
) -> list[tuple[int, int, float]]:
    """Compare rolling now vs leveling first.

    Returns [(level, gold_left_for_rolls, P(goal))] for the current level and
    up to ``max_extra_levels`` levels above it (leveling paid from ``gold``).
    """
    out: list[tuple[int, int, float]] = []
    for target in range(level, min(mech.max_level, level + max_extra_levels) + 1):
        spend = mech.gold_to_reach(level, xp_current, target)
        if spend is None or spend > gold:
            break
        left = gold - spend
        p = rolldown_probability(mech, target, cost, need, remaining_unit, remaining_cost_total, left, shop=shop)
        out.append((target, left, round(p, 4)))
    return out
