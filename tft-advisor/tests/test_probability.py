from __future__ import annotations

import random

import pytest

from tft_advisor.engine.probability import (
    GOLD_BUDGETS,
    best_roll_level,
    compute_hit_odds,
    copies_needed,
    default_goal_star,
    expected_gold_to_goal,
    owned_copies,
    p_at_least_one,
    p_unit_per_slot,
    pool_remaining,
    rolldown_probability,
)

from .conftest import make_state, make_unit


def four_cost_pool(mech, set_data):
    n = len(set_data.champions_by_cost(4))
    return mech.pool_size[4], mech.pool_size[4] * n


# ---------------------------------------------------------------------------
# Per-slot formula
# ---------------------------------------------------------------------------


def test_p_slot_formula(mech):
    p = p_unit_per_slot(mech, 8, 4, 10, 40)
    assert p == pytest.approx(mech.odds(8, 4) * 10 / 40)
    assert p_unit_per_slot(mech, 8, 4, 0, 40) == 0.0
    assert p_unit_per_slot(mech, 8, 4, 5, 0) == 0.0
    assert p_unit_per_slot(mech, 3, 4, 10, 40) == 0.0  # no 4 costs at level 3
    assert p_unit_per_slot(mech, 8, 4, 50, 40) == pytest.approx(mech.odds(8, 4))  # ratio capped at 1


def test_p_at_least_one():
    assert p_at_least_one(0.0) == 0.0
    assert p_at_least_one(1.0) == 1.0
    assert p_at_least_one(0.1, 5) == pytest.approx(1 - 0.9**5)
    assert p_at_least_one(0.1, 1) == pytest.approx(0.1)


def test_odds_rows_sum_to_one(mech):
    for lvl in range(1, mech.max_level + 1):
        assert sum(mech.odds(lvl, c) for c in range(1, 6)) == pytest.approx(1.0, abs=0.02)
    assert mech.odds(99, 5) == mech.odds(mech.max_level, 5)
    assert mech.odds(5, 7) == 0.0


# ---------------------------------------------------------------------------
# Roll-down chain
# ---------------------------------------------------------------------------


def test_rolldown_monotonic_in_gold(mech, set_data):
    rem, total = four_cost_pool(mech, set_data)
    probs = [rolldown_probability(mech, 8, 4, 2, rem, total, g) for g in range(0, 101, 5)]
    assert probs[0] == 0.0
    assert all(b >= a - 1e-12 for a, b in zip(probs, probs[1:]))
    assert probs[-1] > 0.5
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_rolldown_monotonic_in_level_for_four_costs(mech, set_data):
    rem, total = four_cost_pool(mech, set_data)
    levels = [lvl for lvl in range(4, mech.max_level + 1)]
    odds = [mech.odds(lvl, 4) for lvl in levels]
    probs = [rolldown_probability(mech, lvl, 4, 3, rem, total, 60) for lvl in levels]
    for (o1, p1), (o2, p2) in zip(zip(odds, probs), zip(odds[1:], probs[1:])):
        if o2 >= o1:
            assert p2 >= p1 - 1e-12
    assert probs[0] == 0.0  # level 4 cannot show 4 costs
    assert probs[-1] > probs[levels.index(7)]


def test_rolldown_zero_when_pool_exhausted(mech, set_data):
    rem, total = four_cost_pool(mech, set_data)
    assert rolldown_probability(mech, 8, 4, 3, 2, total, 100) == 0.0  # only 2 left, need 3
    assert rolldown_probability(mech, 8, 4, 1, 0, total, 100) == 0.0
    assert rolldown_probability(mech, 8, 4, 1, rem, 0, 100) == 0.0


def test_rolldown_edges(mech):
    assert rolldown_probability(mech, 8, 4, 0, 10, 30, 0) == 1.0  # nothing needed
    assert rolldown_probability(mech, 8, 4, 1, 10, 30, mech.roll_cost + 3) == 0.0  # cannot roll and buy
    assert rolldown_probability(mech, 8, 4, 1, 10, 30, mech.roll_cost + 4) > 0.0
    # More copies left in the pool -> easier.
    assert rolldown_probability(mech, 8, 4, 2, 9, 30, 50) > rolldown_probability(mech, 8, 4, 2, 4, 30, 50)


def _simulate(mech, level, cost, need, rem, total, gold, trials, seed=7):
    rng = random.Random(seed)
    hits = 0
    for _ in range(trials):
        g, bought = gold, 0
        while bought < need and g >= mech.roll_cost + cost:
            g -= mech.roll_cost
            p = p_unit_per_slot(mech, level, cost, rem - bought, total - bought)
            shown = sum(1 for _ in range(mech.shop_slots) if rng.random() < p)
            buy = min(shown, need - bought, g // cost)
            bought += buy
            g -= buy * cost
        hits += bought >= need
    return hits / trials


@pytest.mark.parametrize("level,cost,need,rem,gold", [(8, 4, 2, 8, 50), (7, 3, 3, 14, 40), (6, 2, 2, 20, 20)])
def test_rolldown_matches_monte_carlo(mech, set_data, level, cost, need, rem, gold):
    total = mech.pool_size[cost] * len(set_data.champions_by_cost(cost))
    exact = rolldown_probability(mech, level, cost, need, rem, total, gold)
    sim = _simulate(mech, level, cost, need, rem, total, gold, trials=6000)
    assert exact == pytest.approx(sim, abs=0.03)


# ---------------------------------------------------------------------------
# Expected gold / best level
# ---------------------------------------------------------------------------


def test_expected_gold(mech):
    p = p_unit_per_slot(mech, 8, 4, 10, 30)
    hit = p_at_least_one(p, mech.shop_slots)
    assert expected_gold_to_goal(mech, 8, 4, 1, 10, 30) == pytest.approx(mech.roll_cost / hit + 4)
    two = expected_gold_to_goal(mech, 8, 4, 2, 10, 30)
    assert two > 2 * expected_gold_to_goal(mech, 8, 4, 1, 10, 30) - 1e-9  # the pool shrinks
    assert expected_gold_to_goal(mech, 8, 4, 0, 10, 30) == 0.0
    assert expected_gold_to_goal(mech, 8, 4, 3, 2, 30) is None
    assert expected_gold_to_goal(mech, 3, 4, 1, 10, 30) is None  # cannot appear


def test_best_roll_level(mech, set_data):
    rem, total = four_cost_pool(mech, set_data)
    gold = mech.gold_to_reach(7, 0, 9) + 20
    rows = best_roll_level(mech, 7, 0, gold, 4, 2, rem, total)
    assert [r[0] for r in rows] == [7, 8, 9]
    assert rows[0][1] == gold
    assert rows[1][1] == gold - mech.gold_to_reach(7, 0, 8)
    assert rows[2][1] == 20
    assert rows[1][1] > rows[2][1]
    for lvl, left, p in rows:
        assert p == pytest.approx(rolldown_probability(mech, lvl, 4, 2, rem, total, left), abs=1e-4)
    # Not enough gold to level: only the current level.
    short = best_roll_level(mech, 7, 0, mech.gold_to_reach(7, 0, 8) - 1, 4, 2, rem, total)
    assert [r[0] for r in short] == [7]
    # Max level stops the list.
    assert [r[0] for r in best_roll_level(mech, mech.max_level, 0, 100, 4, 1, rem, total)] == [mech.max_level]


# ---------------------------------------------------------------------------
# Pool accounting and compute_hit_odds
# ---------------------------------------------------------------------------


def test_pool_accounting(mech, set_data):
    per_unit, per_cost = pool_remaining(set_data, mech, {"TFT99_Graves": 5, "TFT99_Kayle": 20})
    assert per_unit["TFT99_Graves"] == mech.pool_size[1] - 5
    assert per_unit["TFT99_Kayle"] == 0  # never negative
    assert per_cost[1] == mech.pool_size[1] * 5 - 5
    assert per_cost[5] == mech.pool_size[5]
    assert owned_copies([make_unit("Graves", 2), make_unit("Graves"), make_unit("Zzz")]) == {"TFT99_Graves": 4}


@pytest.mark.parametrize(
    "owned,goal,need", [(0, 2, 3), (2, 2, 1), (3, 2, 0), (4, 3, 5), (9, 3, 0)]
)
def test_copies_needed(owned, goal, need):
    assert copies_needed(owned, goal) == need


def test_default_goal_star():
    assert default_goal_star(1, 0, 3) == 2
    assert default_goal_star(1, 4, 5) == 3
    assert default_goal_star(3, 5, 7) == 3
    assert default_goal_star(3, 5, 8) == 3  # already a 2-star: next goal is 3-star
    assert default_goal_star(4, 1, 8) == 2
    assert default_goal_star(4, 3, 8) == 3


def test_compute_hit_odds_default_targets(mech, set_data):
    st = make_state("3-2", board=[("Graves", 2), "Lucian"], bench=["Graves", "Lucian"], level=6)
    odds = compute_hit_odds(st, set_data, mech, {"TFT99_Graves": 3})
    assert [o.unit for o in odds] == ["Graves", "Lucian"]  # most copies first
    g = odds[0]
    assert g.owned_copies == 4 and g.goal_star == 3 and g.goal_copies == 9
    assert g.seen_elsewhere == 3
    assert g.remaining_in_pool == mech.pool_size[1] - 4 - 3
    per_unit, per_cost = pool_remaining(set_data, mech, {"TFT99_Graves": 7, "TFT99_Lucian": 2})
    assert g.p_per_slot == pytest.approx(p_unit_per_slot(mech, 6, 1, per_unit["TFT99_Graves"], per_cost[1]), abs=1e-5)
    assert set(g.p_goal_by_gold) == set(GOLD_BUDGETS)
    budgets = [g.p_goal_by_gold[b] for b in GOLD_BUDGETS]
    assert budgets == sorted(budgets)
    lucian = odds[1]
    assert lucian.goal_star == 2 and lucian.goal_copies == 3 and lucian.expected_gold_to_goal is not None


def test_compute_hit_odds_named_targets(mech, set_data):
    st = make_state("4-1", board=["Graves"], level=8)
    odds = compute_hit_odds(st, set_data, mech, {}, targets=["Miss Fortune", "TFT99_Draven", "Nobody", "Draven"])
    assert [o.api_name for o in odds] == ["TFT99_MissFortune", "TFT99_Draven"]
    assert odds[0].level == 8 and odds[0].p_per_slot > 0
    low = compute_hit_odds(st, set_data, mech, {}, targets=["Miss Fortune"], level=4)
    assert low[0].p_per_slot == 0.0 and low[0].expected_gold_to_goal is None


def test_compute_hit_odds_skips_three_stars_and_unknowns(mech, set_data):
    st = make_state("4-1", board=[("Graves", 3), "Zzz"], level=7)
    assert compute_hit_odds(st, set_data, mech, {}) == []


def test_scouted_copies_lower_the_odds(mech, set_data):
    st = make_state("4-1", board=["Draven", "Draven"], level=8)
    free = compute_hit_odds(st, set_data, mech, {})[0]
    contested = compute_hit_odds(st, set_data, mech, {"TFT99_Draven": 5})[0]
    assert contested.remaining_in_pool == free.remaining_in_pool - 5
    assert contested.p_per_slot < free.p_per_slot
    assert contested.p_goal_by_gold[40] < free.p_goal_by_gold[40]


def test_empty_target_list_means_no_targets(mech, set_data):
    st = make_state("3-2", board=["Graves"], level=6)
    assert compute_hit_odds(st, set_data, mech, {}, targets=[]) == []
