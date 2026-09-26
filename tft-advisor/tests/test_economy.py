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
    # Medium HP at a stabilization round: level and roll the rest down to ~10.
    spend = mech.gold_to_reach(lvl - 1, 0, lvl)
    plan = plan_economy(gs("3-2", gold=50, level=lvl - 1, hp=55), mech)
    assert plan.recommendation == EconAction.LEVEL_AND_ROLL and plan.target_level == lvl
    assert plan.roll_budget == 50 - spend - 10


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
