"""Economy planner and Mechanics tests (values derived from the loaded mechanics
so a patch update of mechanics.toml does not break them)."""

from __future__ import annotations

import pytest

from tft_advisor.data.mechanics import load_mechanics
from tft_advisor.engine.economy import hp_bucket, plan_economy, reroll_level
from tft_advisor.models import EconAction, GameState, StageRound


def gs(stage, gold=0, level=None, hp=80, xp=0, streak=None) -> GameState:
    return GameState(
        stage=StageRound.parse(stage) if stage else None, gold=gold, level=level, hp=hp, xp_current=xp, streak=streak
    )


# ---------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------


def test_interest_and_streaks(mech):
    step, cap = mech.interest_step, mech.interest_cap
    assert mech.interest(0) == 0 and mech.interest(-5) == 0
    assert mech.interest(step - 1) == 0 and mech.interest(step) == 1
    assert mech.interest(step * cap + 37) == cap
    assert mech.streak_gold(None) == 0 and mech.streak_gold(0) == 0
    first_len, first_gold = mech.streak_bonus[0]
    assert mech.streak_gold(first_len - 1) == 0
    assert mech.streak_gold(first_len) == first_gold
    assert mech.streak_gold(-first_len) == first_gold  # loss streaks pay too
    assert mech.streak_gold(99) == mech.streak_bonus[-1][1]
    assert mech.income(35, 0) == mech.base_income + 3
    assert mech.income(35, 0, won=True) == mech.base_income + 3 + mech.pvp_win_gold


def test_xp_math(mech):
    need4 = mech.xp_to_level[4]
    assert mech.xp_to_next(4, 0) == need4
    assert mech.xp_to_next(4, need4 + 5) == 0
    assert mech.xp_to_next(None, 0) is None and mech.xp_to_next(mech.max_level, 0) is None
    buys = -(-need4 // mech.xp_per_buy)
    assert mech.gold_to_reach(4, 0, 5) == buys * mech.buy_xp_cost
    assert mech.gold_to_reach(4, 2, 5) == -(-(need4 - 2) // mech.xp_per_buy) * mech.buy_xp_cost
    two = mech.xp_to_level[4] + mech.xp_to_level[5]
    assert mech.gold_to_reach(4, 0, 6) == -(-two // mech.xp_per_buy) * mech.buy_xp_cost
    assert mech.gold_to_reach(6, 0, 6) == 0 and mech.gold_to_reach(6, 0, 3) == 0
    assert mech.gold_to_reach(None, 0, 6) is None
    assert mech.gold_to_reach(9, 0, mech.max_level + 1) is None


def test_round_types(mech):
    sr = StageRound.parse
    assert mech.is_pve(sr("1-3")) and mech.is_pve(sr(f"3-{mech.pve_round}")) and not mech.is_pve(sr("3-2"))
    assert mech.is_carousel(sr("1-1")) and mech.is_carousel(sr(f"4-{mech.carousel_round}"))
    assert not mech.is_carousel(sr("1-2")) and not mech.is_carousel(None)
    assert mech.is_augment(sr(mech.augment_rounds[0])) and not mech.is_augment(sr("2-2"))
    assert mech.standard_level_at(sr("3-3")) == mech.standard_levels["3-2"]
    assert mech.standard_level_at(sr("4-1")) == mech.standard_levels["4-1"]
    assert mech.standard_level_at(None) is None


def test_mechanics_tables_are_consistent(mech):
    assert set(mech.shop_odds) == set(range(1, mech.max_level + 1))
    assert set(mech.xp_to_level) == set(range(1, mech.max_level))
    assert all(v > 0 for v in mech.pool_size.values()) and set(mech.pool_size) == {1, 2, 3, 4, 5}
    lvls = [mech.standard_levels[k] for k in sorted(mech.standard_levels, key=lambda k: StageRound.parse(k).key)]
    assert lvls == sorted(lvls)


def test_mechanics_override(tmp_path):
    p = tmp_path / "mech.toml"
    p.write_text('roll_cost = 3\n[pool_size]\n4 = 12\n[shop_odds]\n8 = [10, 20, 30, 30, 10]\n', encoding="utf-8")
    m = load_mechanics(str(p))
    base = load_mechanics()
    assert m.roll_cost == 3 and m.pool_size[4] == 12 and m.pool_size[1] == base.pool_size[1]
    assert m.odds(8, 5) == pytest.approx(0.10) and m.odds(7, 1) == base.odds(7, 1)


def test_mechanics_rejects_bad_odds(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("[shop_odds]\n8 = [10, 20, 30]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_mechanics(str(p))


# ---------------------------------------------------------------------------
# Economy decisions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hp,bucket", [(None, "unknown"), (100, "healthy"), (70, "healthy"), (69, "medium"), (45, "medium"), (44, "low"), (25, "low"), (24, "critical"), (0, "critical")])
def test_hp_bucket(hp, bucket):
    assert hp_bucket(hp) == bucket


def test_reroll_levels():
    assert (reroll_level("reroll1"), reroll_level("reroll2"), reroll_level("reroll3")) == (5, 6, 7)
    assert reroll_level("standard") is None and reroll_level("fast8") is None


def test_plan_fields(mech):
    plan = plan_economy(gs("3-1", gold=34, level=6, streak=-4), mech)
    assert plan.gold == 34 and plan.interest == 3
    assert plan.streak_gold == mech.streak_gold(4)
    assert plan.expected_income == mech.base_income + 3 + mech.streak_gold(4)
    assert plan.xp_to_next == mech.xp_to_level[6]
    assert plan.gold_to_next_level == mech.gold_to_reach(6, 0, 7)


def test_unknown_stage_and_stage_one_hold(mech):
    assert plan_economy(GameState(), mech).recommendation == EconAction.HOLD
    for stage in ("1-1", "1-2", "1-4"):
        plan = plan_economy(gs(stage, gold=8, level=2), mech)
        assert plan.recommendation == EconAction.HOLD and plan.roll_budget == 0
        assert "—" not in plan.reason


def test_carousel_hold(mech):
    plan = plan_economy(gs(f"3-{mech.carousel_round}", gold=30, level=6, hp=20), mech)
    assert plan.recommendation == EconAction.HOLD


def test_critical_hp_all_in(mech):
    plan = plan_economy(gs("4-2", gold=40, level=mech.standard_levels["4-1"], hp=15), mech)
    assert plan.recommendation == EconAction.ALL_IN and plan.roll_budget == 40


def test_critical_hp_level_first_when_cheap(mech):
    need = mech.xp_to_level[6]
    xp = need - mech.xp_per_buy  # one buy from level 7
    cost = mech.gold_to_reach(6, xp, 7)
    plan = plan_economy(gs("4-1", gold=30, level=6, hp=10, xp=xp), mech)
    assert plan.recommendation == EconAction.LEVEL_AND_ROLL and plan.target_level == 7
    assert plan.roll_budget == 30 - cost


def test_standard_level_timing_3_2(mech):
    lvl = mech.standard_levels["3-2"]
    plan = plan_economy(gs("3-2", gold=50, level=lvl - 1, hp=85), mech)
    assert plan.recommendation == EconAction.LEVEL and plan.target_level == lvl
    # Medium HP at a stabilization round: level and roll, but 3-2 keeps ~20.
    spend = mech.gold_to_reach(lvl - 1, 0, lvl)
    plan = plan_economy(gs("3-2", gold=50, level=lvl - 1, hp=55), mech)
    assert plan.recommendation == EconAction.LEVEL_AND_ROLL and plan.target_level == lvl
    assert plan.roll_budget == 50 - spend - 20


def test_standard_level_timing_4_1(mech):
    lvl = mech.standard_levels["4-1"]
    spend = mech.gold_to_reach(lvl - 1, 0, lvl)
    plan = plan_economy(gs("4-1", gold=spend + 20, level=lvl - 1, hp=80), mech)
    assert plan.recommendation == EconAction.LEVEL and plan.target_level == lvl
    # Low HP: level and roll everything.
    plan = plan_economy(gs("4-1", gold=spend + 20, level=lvl - 1, hp=30), mech)
    assert plan.recommendation == EconAction.LEVEL_AND_ROLL and plan.roll_budget == 20
    # Cannot afford the level: roll to stabilize, keep 20 at medium HP.
    plan = plan_economy(gs("4-1", gold=spend - 1, level=lvl - 1, hp=50), mech)
    assert plan.recommendation == EconAction.ROLL and plan.roll_budget == spend - 1 - 20


def test_fast8_levels_to_8_at_4_1(mech):
    spend = mech.gold_to_reach(7, 0, 8)
    plan = plan_economy(gs("4-1", gold=spend + 10, level=7, hp=90), mech, style="fast8")
    assert plan.recommendation == EconAction.LEVEL and plan.target_level == 8
    std = plan_economy(gs("4-1", gold=spend + 10, level=7, hp=90), mech, style="standard")
    assert std.target_level != 8


def test_reroll_slow_roll_above_50(mech):
    cap = mech.interest_step * mech.interest_cap
    plan = plan_economy(gs("3-5", gold=cap + 14, level=6, hp=80), mech, style="reroll2")
    assert plan.recommendation == EconAction.SLOW_ROLL and plan.roll_budget == 14
    plan = plan_economy(gs("3-5", gold=cap, level=6, hp=80), mech, style="reroll2")
    assert plan.recommendation == EconAction.SAVE and plan.roll_budget == 0
    plan = plan_economy(gs("3-5", gold=35, level=6, hp=35), mech, style="reroll2")
    assert plan.recommendation == EconAction.ROLL and plan.roll_budget == 15


def test_reroll_levels_to_its_roll_level(mech):
    plan = plan_economy(gs("2-5", gold=30, level=4, hp=80), mech, style="reroll1")
    assert plan.recommendation == EconAction.LEVEL and plan.target_level == 5
    # A reroll line does not keep leveling on the standard curve at 4-1.
    plan = plan_economy(gs("4-1", gold=40, level=5, hp=80), mech, style="reroll1")
    assert plan.recommendation == EconAction.SAVE


def test_interest_step_saving(mech):
    lvl = mech.standard_levels["2-1"]
    plan = plan_economy(gs("2-3", gold=27, level=lvl, hp=90), mech)
    assert plan.recommendation == EconAction.SAVE
    assert "30" in plan.reason and plan.roll_budget == 0


def test_cheap_level_without_losing_interest(mech):
    need = mech.xp_to_level[6]
    plan = plan_economy(gs("3-3", gold=36, level=6, xp=need - 2, hp=90), mech)
    assert plan.recommendation == EconAction.LEVEL and plan.target_level == 7
    # Same, but the XP buy would drop an interest step: save instead.
    plan = plan_economy(gs("3-3", gold=31, level=6, xp=need - 2, hp=90), mech)
    assert plan.recommendation == EconAction.SAVE


def test_gold_above_cap(mech):
    cap = mech.interest_step * mech.interest_cap
    need = mech.gold_to_reach(6, 0, 7)
    plan = plan_economy(gs("3-3", gold=cap + need, level=6, hp=90), mech)
    assert plan.recommendation == EconAction.LEVEL and plan.target_level == 7
    late = plan_economy(gs("5-3", gold=cap + 10, level=9, hp=90), mech)
    assert late.recommendation == EconAction.SLOW_ROLL and late.roll_budget == 10


def test_low_hp_rolls_outside_rolldown_rounds(mech):
    plan = plan_economy(gs("3-3", gold=40, level=6, hp=30), mech)
    assert plan.recommendation == EconAction.ROLL and plan.roll_budget == 30


def test_reasons_never_use_em_dash(mech):
    for stage in ("1-2", "2-1", "2-3", "3-2", "3-4", "4-1", "4-2", "5-1", "6-3"):
        for hp in (100, 60, 30, 10):
            for gold in (0, 9, 27, 50, 75):
                for style in ("standard", "fast8", "reroll1", "reroll2", "reroll3"):
                    plan = plan_economy(gs(stage, gold=gold, level=6, hp=hp), mech, style)
                    assert "—" not in plan.reason
                    assert 0 <= plan.roll_budget <= gold


# ---------------------------------------------------------------------------
# Known bugs in economy.py (not owned by the engine tests; reported upstream)
# ---------------------------------------------------------------------------


def test_cheap_level_must_be_affordable(mech):
    need = mech.xp_to_level[6]
    plan = plan_economy(gs("3-3", gold=mech.buy_xp_cost - 2, level=6, xp=need - 2, hp=90), mech)
    assert plan.recommendation != EconAction.LEVEL


def test_two_levels_behind_buys_the_level_it_can_afford(mech):
    one = mech.gold_to_reach(5, 0, 6)
    plan = plan_economy(gs("4-1", gold=one + 25, level=5, hp=80), mech)
    assert plan.recommendation in (EconAction.LEVEL, EconAction.LEVEL_AND_ROLL) and plan.target_level == 6


def test_unknown_level_is_not_level_one(mech):
    plan = plan_economy(GameState(stage=StageRound.parse("5-1"), gold=0, hp=80), mech)
    assert plan.target_level != 2


# ---------------------------------------------------------------------------
# Regressions: reroll end, stage 2 reroll curve, fast 8 at level 8, streaks,
# low HP outside the roll-down rounds
# ---------------------------------------------------------------------------

LEVELING = (EconAction.LEVEL, EconAction.LEVEL_AND_ROLL)
ROLLING = (EconAction.ROLL, EconAction.SLOW_ROLL, EconAction.ALL_IN, EconAction.LEVEL_AND_ROLL)


def test_reroll_done_levels_like_a_standard_line(mech):
    # Carry already 3-star at 5-1, still level 5: level up, no slow roll at 5.
    plan = plan_economy(gs("5-1", gold=90, level=5, hp=70), mech, "reroll1", key_star=3)
    assert plan.recommendation in LEVELING and plan.target_level >= 7
    assert plan.style == "standard"
    assert not any(w in plan.reason for w in ("慢搜", "赌狗", "三星"))
    for stage, hp in (("5-5", 40), ("6-2", 30)):
        plan = plan_economy(gs(stage, gold=60, level=5, hp=hp), mech, "reroll1", key_star=3)
        assert plan.recommendation in LEVELING and plan.target_level > 5, stage
        assert "三星" not in plan.reason


def test_reroll_still_slow_rolls_before_the_three_star(mech):
    plan = plan_economy(gs("4-1", gold=64, level=5, hp=80), mech, "reroll1", key_star=2)
    assert plan.recommendation == EconAction.SLOW_ROLL and plan.roll_budget == 14
    plan = plan_economy(gs("4-2", gold=50, level=5, hp=60), mech, "reroll1", key_star=2)
    assert plan.recommendation == EconAction.ROLL and "还没三星" in plan.reason
    # Unknown target: no claim about the 3-star.
    plan = plan_economy(gs("4-2", gold=50, level=5, hp=60), mech, "reroll1")
    assert "还没三星" not in plan.reason


def test_reroll_stops_staying_low_from_4_5(mech):
    plan = plan_economy(gs("4-5", gold=60, level=5, hp=60), mech, "reroll1", key_star=2)
    assert plan.recommendation in LEVELING and plan.target_level > 5


def test_reroll_follows_the_standard_curve_in_stage_two(mech):
    # 2-2, level 3, 16 gold: not straight to level 5 with every coin.
    plan = plan_economy(gs("2-2", gold=16, level=3, hp=90), mech, "reroll1", key_star=2)
    assert plan.target_level in (None, mech.standard_levels["2-1"])
    # 2-5, level 4 (6 xp), 30 gold: level 6 would cost 24; stage 2 stops at 5.
    plan = plan_economy(gs("2-5", gold=30, level=4, xp=6, hp=90), mech, "reroll2", key_star=2)
    assert plan.recommendation == EconAction.LEVEL and plan.target_level == 5
    # Stage 3: the reroll level.
    plan = plan_economy(gs("3-1", gold=30, level=5, hp=80), mech, "reroll2", key_star=2)
    assert plan.recommendation == EconAction.LEVEL and plan.target_level == 6
    # A 3-cost reroll does not jump to 7 at 3-1, it goes 7 from 3-5.
    plan = plan_economy(gs("3-1", gold=60, level=5, hp=80), mech, "reroll3", key_star=2)
    assert plan.target_level != 7
    plan = plan_economy(gs("3-5", gold=60, level=6, hp=80), mech, "reroll3", key_star=2)
    assert plan.recommendation == EconAction.LEVEL and plan.target_level == 7
    # A reroll line at its level never buys the next level with spare gold.
    plan = plan_economy(gs("2-6", gold=60, level=5, xp=18, hp=90), mech, "reroll1", key_star=1)
    assert plan.target_level != 6


def test_fast8_at_level_8_stops_rolling_once_the_carry_is_two_star(mech):
    for stage, gold in (("4-5", 60), ("5-2", 70), ("5-5", 64)):
        plan = plan_economy(gs(stage, gold=gold, level=8, hp=82, streak=6), mech, "fast8", key_star=2)
        assert plan.recommendation not in ROLLING, stage
        assert plan.roll_budget == 0
    plan = plan_economy(gs("5-1", gold=90, level=8, hp=82, streak=6), mech, "fast8", key_star=2)
    assert plan.recommendation == EconAction.LEVEL and plan.target_level == 9
    # Carry still 1-star: keep rolling for it.
    plan = plan_economy(gs("4-5", gold=60, level=8, hp=82), mech, "fast8", key_star=1)
    assert plan.recommendation == EconAction.ROLL and plan.roll_budget == 50
    # Unknown carry: only the scheduled roll-down, not every round.
    plan = plan_economy(gs("5-2", gold=48, level=8, hp=82), mech, "fast8")
    assert plan.recommendation not in ROLLING


def test_loss_streak_with_enough_hp_skips_the_3_2_roll(mech):
    plan = plan_economy(gs("3-2", gold=52, level=5, hp=58, streak=-6), mech)
    assert plan.roll_budget == 0 and plan.recommendation in (EconAction.LEVEL, EconAction.SAVE)
    assert "连败" in plan.reason
    # Below the HP floor it rolls, but 3-2 still keeps about 20.
    spend = mech.gold_to_reach(5, 0, 6)
    plan = plan_economy(gs("3-2", gold=50, level=5, hp=44, streak=-7), mech)
    assert plan.recommendation == EconAction.LEVEL_AND_ROLL
    assert plan.roll_budget == 50 - spend - 20


def test_win_streak_does_not_roll_at_a_rolldown_point(mech):
    plan = plan_economy(gs("4-2", gold=36, level=7, hp=64, streak=5), mech)
    assert plan.recommendation not in ROLLING and plan.roll_budget == 0
    spend = mech.gold_to_reach(6, 0, 7)
    plan = plan_economy(gs("4-1", gold=spend + 20, level=6, hp=60, streak=4), mech)
    assert plan.recommendation == EconAction.LEVEL and plan.roll_budget == 0


def test_low_hp_levels_and_rolls_outside_the_rolldown_rounds(mech):
    spend = mech.gold_to_reach(7, 20, 8)
    plan = plan_economy(gs("4-3", gold=60, level=7, xp=20, hp=28), mech, "fast8")
    assert plan.recommendation == EconAction.LEVEL_AND_ROLL and plan.target_level == 8
    assert plan.roll_budget == 60 - spend - 10


# ---------------------------------------------------------------------------
# Regressions (domain review): fast 9 timing, critical HP levels, no-roll
# fallbacks, catching up late, the stage 2 level points
# ---------------------------------------------------------------------------


def test_fast8_ready_carry_levels_when_it_costs_no_interest(mech):
    need = mech.gold_to_reach(8, 10, 9)
    cap = mech.interest_step * mech.interest_cap
    # 4-5, 2-star carry, enough gold to go 9 and still keep 50: level now.
    plan = plan_economy(gs("4-5", gold=cap + need, level=8, xp=10, hp=80, streak=0), mech, "fast8", key_star=2)
    assert plan.recommendation == EconAction.LEVEL and plan.target_level == 9
    assert "不掉利息" in plan.reason and "主C已2星" in plan.reason
    # Only one interest step lost: level too.
    plan = plan_economy(gs("4-5", gold=cap + need - 10, level=8, xp=10, hp=80), mech, "fast8", key_star=2)
    assert plan.recommendation == EconAction.LEVEL and plan.target_level == 9
    # A 3-star carry is called a 3-star.
    plan = plan_economy(gs("4-3", gold=cap + need, level=8, xp=10, hp=80), mech, "fast8", key_star=3)
    assert "主C已3星" in plan.reason and "两星" not in plan.reason
    # 80 gold: the spare 30 goes into XP, the 50 interest stays.
    plan = plan_economy(gs("4-5", gold=cap + 30, level=8, xp=10, hp=80), mech, "fast8", key_star=2)
    assert plan.recommendation == EconAction.SAVE and plan.roll_budget == 0
    assert "多出的 30 金币买经验" in plan.reason


def test_five_cost_carry_makes_fast8_a_nine_line(mech):
    need = mech.gold_to_reach(8, 10, 9)
    cap = mech.interest_step * mech.interest_cap
    for stage, gold in (("5-1", 80), ("5-1", 100), ("5-3", 90), ("5-2", need)):
        plan = plan_economy(gs(stage, gold=gold, level=8, xp=10, hp=80), mech, "fast8", key_star=1, carry_cost=5)
        assert plan.style == "fast9", stage
        assert plan.recommendation == EconAction.LEVEL and plan.target_level == 9, (stage, gold, plan.reason)
    # Before 5-1 it never rolls at 8 for the 5 cost: it saves (or puts the
    # gold above 50 into XP) for level 9.
    for gold in (40, 60, cap + 20):
        plan = plan_economy(gs("4-5", gold=gold, level=8, xp=10, hp=80), mech, "fast8", key_star=1, carry_cost=5)
        assert plan.recommendation not in ROLLING, (gold, plan.reason)
        assert "搜主C二星" not in plan.reason
    # Enough to go 9 and keep 50: level right away.
    plan = plan_economy(gs("4-5", gold=cap + need, level=8, xp=10, hp=80), mech, "fast8", key_star=0, carry_cost=5)
    assert plan.recommendation == EconAction.LEVEL and plan.target_level == 9
    # A 4 cost carry keeps the fast 8 roll at level 8.
    plan = plan_economy(gs("4-5", gold=60, level=8, hp=82), mech, "fast8", key_star=1, carry_cost=4)
    assert plan.style == "fast8" and plan.recommendation == EconAction.ROLL


def test_critical_hp_buys_a_cheap_level_even_on_curve(mech):
    # 3-2, level 6 (on curve), 26/36 XP: level 7 costs 12 of 90 gold.
    cost = mech.gold_to_reach(6, 26, 7)
    plan = plan_economy(gs("3-2", gold=90, level=6, xp=26, hp=10), mech)
    assert plan.recommendation == EconAction.LEVEL_AND_ROLL and plan.target_level == 7
    assert plan.roll_budget == 90 - cost
    # Behind the curve, up to half the gold goes into the level.
    cost = mech.gold_to_reach(7, 20, 8)
    plan = plan_economy(gs("4-5", gold=2 * cost, level=7, xp=20, hp=18), mech)
    assert plan.recommendation == EconAction.LEVEL_AND_ROLL and plan.target_level == 8
    # A reroll line at its reroll level does not level at critical HP.
    plan = plan_economy(gs("4-1", gold=60, level=5, xp=16, hp=15), mech, "reroll1", key_star=2)
    assert plan.recommendation == EconAction.ALL_IN


def test_no_gold_to_roll_texts(mech):
    # Critical HP with 1 gold stays all-in (the rules say: reposition, sell).
    plan = plan_economy(gs("5-6", gold=1, level=8, hp=24), mech)
    assert plan.recommendation == EconAction.ALL_IN and plan.roll_budget <= 1
    # Low HP late with the reserve eating the budget: never "save to 50".
    plan = plan_economy(gs("6-2", gold=10, level=9, hp=25), mech)
    assert plan.recommendation not in (EconAction.SAVE, EconAction.ROLL)
    assert "不够搜一次" not in plan.reason and "调整站位" in plan.reason
    plan = plan_economy(gs("4-2", gold=14, level=6, hp=30), mech, "reroll2", key_star=2)
    assert plan.recommendation != EconAction.SAVE and "不够搜一次" not in plan.reason
    # "Not enough for one roll" only when the gold really is below one roll.
    for stage, gold, level, hp in (("3-3", 11, 6, 30), ("5-3", 51, 9, 90)):
        plan = plan_economy(gs(stage, gold=gold, level=level, hp=hp), mech)
        assert "不够搜一次" not in plan.reason, (stage, plan.reason)
    plan = plan_economy(gs("3-3", gold=1, level=6, hp=30), mech)
    assert plan.recommendation == EconAction.SAVE and "不够搜一次" in plan.reason
    # A roll-down that keeps nothing says so plainly.
    plan = plan_economy(gs("4-1", gold=30, level=7, hp=30), mech)
    assert "保留约 0" not in plan.reason


def test_behind_the_curve_late_buys_the_catch_up_level(mech):
    cost = mech.gold_to_reach(7, 15, 8)
    plan = plan_economy(gs("5-2", gold=cost + 7, level=7, xp=15, hp=90), mech)
    assert plan.recommendation == EconAction.LEVEL and plan.target_level == 8
    cost = mech.gold_to_reach(7, 0, 8)
    plan = plan_economy(gs("5-5", gold=cost + 2, level=7, hp=70), mech)
    assert plan.recommendation == EconAction.LEVEL and plan.target_level == 8
    # On curve in stage 3 the 10 gold floor still holds.
    cost = mech.gold_to_reach(5, 0, 6)
    plan = plan_economy(gs("3-3", gold=cost + 5, level=5, hp=90), mech)
    assert plan.target_level != 6


def test_stage_two_level_points_do_not_flip_with_gold(mech):
    one_buy = mech.xp_to_level[3] - mech.xp_per_buy  # one XP buy from level 4
    for gold in range(mech.buy_xp_cost, 25):
        plan = plan_economy(gs("2-1", gold=gold, level=3, xp=one_buy, hp=100), mech)
        assert plan.recommendation == EconAction.LEVEL and plan.target_level == 4, gold
    one_buy = mech.xp_to_level[4] - mech.xp_per_buy
    for gold in (10, 12, 15):
        plan = plan_economy(gs("2-5", gold=gold, level=4, xp=one_buy, hp=100), mech)
        assert plan.recommendation == EconAction.LEVEL and plan.target_level == 5, gold


# ---------------------------------------------------------------------------
# PvE rounds and the damage model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage, hp, gold, level", [("4-7", 35, 52, 8), ("3-7", 15, 45, 7), ("4-7", 20, 45, 8), ("3-7", 30, 40, 6)])
def test_pve_round_never_rolls(mech, stage, hp, gold, level):
    plan = plan_economy(gs(stage, gold=gold, level=level, hp=hp), mech)
    assert plan.recommendation in (EconAction.SAVE, EconAction.LEVEL) and plan.roll_budget == 0
    nxt = f"{int(stage[0]) + 1}-1"
    assert "野怪回合" in plan.reason and nxt in plan.reason
    # The same state one round later (the player fight) rolls.
    after = plan_economy(gs(nxt, gold=gold, level=level, hp=hp), mech)
    assert after.roll_budget > 0


def test_pve_round_keeps_a_cheap_level(mech):
    # Critical HP one XP buy from the next level: the level stays, the roll waits.
    need = mech.xp_to_level[6]
    plan = plan_economy(gs("3-7", gold=44, level=6, hp=15, xp=need - mech.xp_per_buy), mech)
    assert plan.recommendation == EconAction.LEVEL and plan.target_level == 7 and plan.roll_budget == 0
    assert "野怪回合" in plan.reason


def test_max_round_damage_uses_stage_damage(mech):
    assert mech.max_round_damage(4) == mech.stage_damage[4] + 20
    assert mech.max_round_damage(None) >= 20
    assert not hasattr(mech, "xp_needed")
