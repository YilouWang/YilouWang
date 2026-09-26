"""Shop / pool probability model.

Shop generation (per slot): a cost tier is drawn from the level's shop odds,
then a champion is drawn from that tier weighted by the copies still in the
shared pool. So for a champion X of cost c::

    P(slot shows X) = odds[level][c] * remaining(X) / remaining(all cost-c copies)

A roll shows ``shop_slots`` (5) independent slots. For a roll-down we model a
Markov chain over (copies bought so far, gold left): each roll costs
``roll_cost``; every copy of X that shows up is bought (if gold allows) until
the goal is reached. Buying removes copies from the pool, which lowers P for
the next roll. The within-shop depletion is ignored (tiny effect).
"""

from __future__ import annotations

from functools import lru_cache
from math import comb
from typing import Iterable, Optional

from ..data.mechanics import Mechanics
from ..data.setdata import SetData
from ..models import GameState, HitOdds, Unit

GOLD_BUDGETS = (10, 20, 30, 40, 50, 60, 80)


def p_unit_per_slot(mech: Mechanics, level: int, cost: int, remaining_unit: int, remaining_cost_total: int) -> float:
    if remaining_unit <= 0 or remaining_cost_total <= 0:
        return 0.0
    return mech.odds(level, cost) * min(1.0, remaining_unit / remaining_cost_total)


def p_at_least_one(p_slot: float, slots: int = 5) -> float:
    return 1.0 - (1.0 - p_slot) ** slots


def _binom_pmf(n: int, k: int, p: float) -> float:
    return comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))


def rolldown_probability(
    mech: Mechanics,
    level: int,
    cost: int,
    need: int,
    remaining_unit: int,
    remaining_cost_total: int,
    gold: int,
    unit_price: Optional[int] = None,
) -> float:
    """P(buy ``need`` more copies of one champion) spending at most ``gold``.

    Every roll costs ``mech.roll_cost``; each copy costs ``unit_price``
    (defaults to ``cost``). The shop currently on screen is NOT included
    (callers account for it separately).
    """
    if need <= 0:
        return 1.0
    price = unit_price if unit_price is not None else cost
    if remaining_unit < need or gold < mech.roll_cost + price:
        return 0.0
    slots = mech.shop_slots
    roll = max(1, mech.roll_cost)  # a 0-cost override would never terminate

    @lru_cache(maxsize=None)
    def solve(bought: int, g: int) -> float:
        if bought >= need:
            return 1.0
        if g < roll + price:
            return 0.0
        p = p_unit_per_slot(mech, level, cost, remaining_unit - bought, remaining_cost_total - bought)
        if p <= 0.0:
            return 0.0
        g_after_roll = g - roll
        total = 0.0
        p_none = (1.0 - p) ** slots
        # Hits h >= 1: buy up to what's needed and affordable.
        for h in range(1, slots + 1):
            ph = _binom_pmf(slots, h, p)
            if ph < 1e-12:
                continue
            can_buy = min(h, need - bought, g_after_roll // price)
            total += ph * solve(bought + can_buy, g_after_roll - can_buy * price)
        # No hit: keep rolling. Iterate instead of recursing on the same state
        # count: solve(bought, g - roll) is a strictly smaller state.
        total += p_none * solve(bought, g_after_roll)
        return total

    return min(1.0, max(0.0, solve(0, int(gold))))


def expected_gold_to_goal(
    mech: Mechanics, level: int, cost: int, need: int, remaining_unit: int, remaining_cost_total: int
) -> Optional[float]:
    """Expected gold (rolls + purchases) to find ``need`` more copies."""
    if need <= 0:
        return 0.0
    if remaining_unit < need:
        return None
    total = 0.0
    for k in range(need):
        p = p_unit_per_slot(mech, level, cost, remaining_unit - k, remaining_cost_total - k)
        hit = p_at_least_one(p, mech.shop_slots)
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
) -> list[HitOdds]:
    """Odds for the units that matter.

    ``taken_by_others``: known copies held by other players (from scouting).
    ``targets``: api names or display names; default = units we own that are
    not yet 3-star.
    """
    lvl = level or state.level or 1
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
        by_gold = {
            g: round(rolldown_probability(mech, lvl, champ.cost, need, rem, rem_cost, g), 4) for g in budgets
        }
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
                p_in_shop=round(p_at_least_one(p_slot, mech.shop_slots), 4),
                expected_gold_to_goal=(
                    round(e, 1)
                    if (e := expected_gold_to_goal(mech, lvl, champ.cost, need, rem, rem_cost)) is not None
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
        p = rolldown_probability(mech, target, cost, need, remaining_unit, remaining_cost_total, left)
        out.append((target, left, round(p, 4)))
    return out
