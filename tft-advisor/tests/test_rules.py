from __future__ import annotations

import pytest

from tft_advisor.advisor.rules import AUGMENT_GENERIC, RulesAdvisor, clip, find_carry, sanitize
from tft_advisor.config import HotkeyConfig
from tft_advisor.data.mechanics import load_mechanics
from tft_advisor.data.setdata import SetData, bundled_sample
from tft_advisor.engine.economy import plan_economy
from tft_advisor.models import (
    ActionType,
    Advice,
    Analysis,
    CompSuggestion,
    EconAction,
    EconPlan,
    GameState,
    HitOdds,
    ItemSuggestion,
    ScoutRequest,
    ScreenType,
    ShopSlot,
    StageRound,
    Unit,
)

EM_DASHES = ("\u2014", "\u2015")


@pytest.fixture(scope="module")
def mech():
    return load_mechanics()


@pytest.fixture(scope="module")
def set_data():
    return SetData.from_cdragon(bundled_sample())


@pytest.fixture()
def advisor(mech, set_data):
    return RulesAdvisor(mech=mech, set_data=set_data)


def unit(name, cost, star=1, items=(), row=None, col=None):
    return Unit(api_name=f"TFT99_{name.replace(' ', '')}", name=name, cost=cost, star=star, items=list(items), row=row, col=col)


def all_text(advice: Advice) -> list[str]:
    texts = [advice.headline, advice.plan, advice.comp, advice.items, advice.positioning, advice.augment, advice.scout_request]
    texts += [a.text for a in advice.actions]
    return [t for t in texts if t]


def check_contract(advice: Advice) -> None:
    assert advice.source == "rules"
    assert 0 < len(advice.headline) <= 30
    assert len(advice.plan) <= 120
    assert len(advice.actions) <= 6
    priorities = [a.priority for a in advice.actions]
    assert priorities == sorted(priorities)
    for a in advice.actions:
        assert 1 <= a.priority <= 3
        assert 0 < len(a.text) <= 40
    for t in all_text(advice):
        for dash in EM_DASHES:
            assert dash not in t, t


def types(advice: Advice) -> list[ActionType]:
    return [a.type for a in advice.actions]


def action(advice: Advice, kind: ActionType):
    return next(a for a in advice.actions if a.type == kind)


# ---------------------------------------------------------------------------


def test_empty_state_does_not_crash():
    adv = RulesAdvisor().advise(GameState(), Analysis())
    check_contract(adv)
    assert adv.stage is None
    assert "F6" in adv.headline
    assert types(adv) == [ActionType.OTHER]


def test_custom_hotkeys_are_used():
    adv = RulesAdvisor(HotkeyConfig(analyze="F10", scout="F11")).advise(
        GameState(stage=StageRound.parse("3-2"), gold=30, level=6, hp=80),
        Analysis(scout_requests=[ScoutRequest(id="x", text="t", target_player="Bob")]),
    )
    assert "F11" in action(adv, ActionType.SCOUT).text
    assert "F10" in RulesAdvisor(HotkeyConfig(analyze="F10")).advise(GameState(), Analysis()).headline


def test_level_and_roll_at_4_1(advisor, mech):
    state = GameState(
        stage=StageRound.parse("4-1"),
        gold=60,
        level=6,
        xp_current=0,
        hp=50,
        board=[unit("Draven", 4, 2, ["Deathblade", "Giant Slayer"], row=1, col=3), unit("Braum", 2, 2, row=0, col=3)],
        bench=[unit("Ashe", 3)],
        shop_units=[None, unit("Ashe", 3), None, unit("Garen", 1), None],
    )
    econ = plan_economy(state, mech)
    assert econ.recommendation == EconAction.LEVEL_AND_ROLL
    analysis = Analysis(
        econ=econ,
        shop_picks=["Ashe"],
        odds=[
            HitOdds(
                unit="Ashe", api_name="TFT99_Ashe", cost=3, owned_copies=1, goal_copies=3, goal_star=2,
                seen_elsewhere=0, remaining_in_pool=17, level=7, p_per_slot=0.03, p_in_shop=0.14,
                p_goal_by_gold={10: 0.12, 20: 0.35, 30: 0.55},
            )
        ],
        comps=[
            CompSuggestion(
                name="决斗大师", score=0.8, carry="Draven", core_units=["Draven", "Braum", "Ashe", "Aatrox", "Shen"],
                have_units=["Draven", "Braum", "Ashe"], missing_units=["Aatrox", "Shen"],
            )
        ],
    )
    adv = advisor.advise(state, analysis)
    check_contract(adv)
    assert adv.stage == "4-1"
    left = 60 - mech.gold_to_reach(6, 0, 7) - econ.roll_budget
    assert adv.headline == f"升到7级，搜到{left}金币"
    # Shop picks are paid out of the roll budget: the gold left counts them too.
    paid = analysis.model_copy(update={"shop_picks_cost": 3})
    assert advisor.advise(state, paid).headline == f"升到7级，搜到{left - 3}金币"
    kinds = types(adv)
    assert ActionType.LEVEL in kinds and ActionType.ROLL in kinds and ActionType.BUY in kinds
    roll = action(adv, ActionType.ROLL)
    assert roll.priority == 1
    assert f"最多花 {econ.roll_budget} 金币搜牌" in roll.text and "Ashe" in roll.text
    assert "12%" in roll.text  # uses the engine's p_goal_by_gold for the budget
    assert "第2格" in action(adv, ActionType.BUY).text
    # Carry (the comp's carry, on board) is not in the back row.
    assert "Draven" in action(adv, ActionType.POSITION).text
    assert adv.comp == "决斗大师：还缺 Aatrox、Shen"
    assert "现在升 7" in adv.plan and "4-5 升 8" in adv.plan


def test_plan_mentions_upcoming_levels(advisor, mech):
    state = GameState(stage=StageRound.parse("3-5"), gold=32, level=6, xp_current=10, hp=85)
    adv = advisor.advise(state, Analysis(econ=plan_economy(state, mech)))
    check_contract(adv)
    assert "4-1 升 7 搜牌稳血" in adv.plan
    assert "4-5 升 8 找 4 费主C" in adv.plan


def test_save_with_shop_pick_and_upgrade(advisor, mech):
    state = GameState(
        stage=StageRound.parse("2-2"),
        gold=25,
        level=4,
        xp_current=2,
        hp=90,
        board=[unit("Garen", 1), unit("Garen", 1)],
        shop_units=[unit("Vayne", 1), unit("Garen", 1), None, None, None],
    )
    econ = plan_economy(state, mech)
    assert econ.recommendation == EconAction.SAVE
    adv = advisor.advise(state, Analysis(econ=econ, shop_picks=["Garen"]))
    check_contract(adv)
    assert adv.headline == "买Garen，其余存钱"
    buy = action(adv, ActionType.BUY)
    assert buy.priority == 1 and "第2格" in buy.text and "升星" in buy.text
    save = action(adv, ActionType.SAVE)
    assert save.priority == 2


def test_save_headline_without_picks(advisor, mech):
    state = GameState(stage=StageRound.parse("2-3"), gold=18, level=4, xp_current=2, hp=95)
    adv = advisor.advise(state, Analysis(econ=plan_economy(state, mech)))
    check_contract(adv)
    assert adv.headline == "存钱到50"
    assert ActionType.SAVE in types(adv)
    assert "先存到 50" in adv.plan


def test_all_in_when_hp_critical(advisor, mech):
    state = GameState(stage=StageRound.parse("4-3"), gold=34, level=7, xp_current=20, hp=15)
    econ = plan_economy(state, mech)
    assert econ.recommendation == EconAction.ALL_IN
    adv = advisor.advise(state, Analysis(econ=econ))
    check_contract(adv)
    assert adv.headline == "血量危险，all-in搜牌"
    assert f"最多花 {econ.roll_budget} 金币搜牌" in action(adv, ActionType.ROLL).text
    assert "稳血" in adv.plan


def test_items_sell_and_scout(advisor):
    bench = [unit(n, 1) for n in ("Garen", "Graves", "Vayne", "Warwick", "Darius", "Garen", "Graves", "Vayne", "Warwick")]
    state = GameState(stage=StageRound.parse("3-3"), gold=40, level=6, hp=70, bench=bench, item_bench=["B.F. Sword", "B.F. Sword"])
    analysis = Analysis(
        econ=EconPlan(gold=40, recommendation=EconAction.SAVE, reason="存钱吃利息：下一个利息点 50（还差 10）"),
        items=[ItemSuggestion(item="Deathblade", components=["B.F. Sword", "B.F. Sword"], holder="Draven", priority=1)],
        sell_candidates=["Darius", "Warwick", "Graves"],
        shop_picks=["Ashe"],
        scout_requests=[ScoutRequest(id="scout-3-a", text="请点开右侧玩家列表里「Alice」的棋盘，然后按 F7 记录（看完点自己头像回来）", target_player="Alice")],
        warnings=["有人在抢\u2014Draven"],
    )
    adv = advisor.advise(state, analysis)
    check_contract(adv)
    item = action(adv, ActionType.ITEM)
    assert item.text == "合成 Deathblade 给 Draven" and item.priority == 1
    sell = action(adv, ActionType.SELL)
    assert sell.priority == 1 and "Darius" in sell.text and "Warwick" in sell.text and "Graves" not in sell.text
    scout = action(adv, ActionType.SCOUT)
    assert "Alice" in scout.text and "F7" in scout.text
    assert adv.scout_request and "Alice" in adv.scout_request
    assert adv.items and "Deathblade" in adv.items
    # SAVE + slam available: the slam is the headline.
    assert adv.headline == "合成Deathblade给Draven"


def test_no_sell_when_bench_not_full(advisor):
    state = GameState(stage=StageRound.parse("3-3"), gold=40, level=6, hp=70, bench=[unit("Garen", 1)])
    adv = advisor.advise(state, Analysis(sell_candidates=["Garen"]))
    assert ActionType.SELL not in types(adv)


def test_position_only_when_carry_not_in_back_row(advisor):
    carry_back = unit("Draven", 4, 2, ["Deathblade", "Giant Slayer", "Bloodthirster"], row=3, col=6)
    state = GameState(stage=StageRound.parse("3-3"), gold=20, level=6, hp=70, board=[carry_back, unit("Braum", 2, row=0, col=3)])
    adv = advisor.advise(state, Analysis())
    assert ActionType.POSITION not in types(adv)
    assert adv.positioning and "Draven" in adv.positioning
    state.board[0] = carry_back.model_copy(update={"row": 2})
    adv = advisor.advise(state, Analysis())
    assert "Draven" in action(adv, ActionType.POSITION).text
    # No carry (no unit with 2+ items and no comp) -> no position action.
    plain = GameState(stage=StageRound.parse("2-2"), board=[unit("Garen", 1, row=2, col=1)])
    assert find_carry(plain, Analysis()) is None
    assert ActionType.POSITION not in types(advisor.advise(plain, Analysis()))


def test_augment_choices(advisor):
    state = GameState(
        stage=StageRound.parse("3-2"),
        screen_type=ScreenType.AUGMENT_SELECT,
        gold=30,
        level=6,
        hp=80,
        augment_choices=["Rich Get Richer", "Blademaster Crown", "Tiny Titans"],
    )
    adv = advisor.advise(state, Analysis())
    check_contract(adv)
    aug = action(adv, ActionType.AUGMENT)
    assert aug.text == AUGMENT_GENERIC and aug.priority == 1
    assert adv.actions[0].type == ActionType.AUGMENT
    assert adv.augment and "Rich Get Richer" in adv.augment
    # Claude is off by default: never point the player to advice that will not come.
    assert "Claude" not in adv.augment
    assert "海克斯" in adv.headline and "增强" not in adv.headline
    with_claude = RulesAdvisor(claude_enabled=True).advise(state, Analysis())
    assert "Claude" in with_claude.augment


def test_carousel_picks_missing_component(advisor):
    state = GameState(
        stage=StageRound.parse("3-4"),
        gold=30,
        level=6,
        hp=80,
        item_bench=["B.F. Sword"],
        board=[unit("Draven", 4, 2, ["Deathblade"], row=3, col=0)],
    )
    analysis = Analysis(
        items=[ItemSuggestion(item="Giant Slayer", components=["B.F. Sword", "Recurve Bow"], holder="Draven", priority=2)],
    )
    adv = advisor.advise(state, analysis)
    check_contract(adv)
    assert adv.headline == "选秀：拿Recurve Bow"
    car = action(adv, ActionType.CAROUSEL)
    assert "Recurve Bow" in car.text and "Draven" in car.text and car.priority == 1
    assert ActionType.SAVE not in types(adv)


def test_carousel_uses_comp_carry_items_with_set_data(advisor):
    state = GameState(stage=StageRound.parse("2-4"), screen_type=ScreenType.CAROUSEL, item_bench=[])
    comp = CompSuggestion(name="Draven", score=0.7, carry="Draven", carry_items=["Deathblade"])
    adv = advisor.advise(state, Analysis(comps=[comp]))
    assert adv.headline == "选秀：拿B.F. Sword"
    # Without set data or suggestions: a generic pick.
    generic = RulesAdvisor().advise(state, Analysis())
    assert generic.headline == "选秀：拿主C装备散件"
    assert ActionType.CAROUSEL in types(generic)


def test_stage_one_hold(advisor, mech):
    state = GameState(stage=StageRound.parse("1-3"), gold=3, level=3, hp=100)
    adv = advisor.advise(state, Analysis(econ=plan_economy(state, mech)))
    check_contract(adv)
    assert "第一阶段" in adv.headline
    assert "2-1 升 4 级" in adv.plan


def test_actions_capped_at_six(advisor):
    bench = [unit("Garen", 1)] * 9
    state = GameState(
        stage=StageRound.parse("4-1"),
        gold=70,
        level=7,
        hp=60,
        augment_choices=["A", "B", "C"],
        board=[unit("Draven", 4, 2, ["Deathblade", "Giant Slayer"], row=1, col=3)],
        bench=bench,
    )
    analysis = Analysis(
        econ=EconPlan(gold=70, recommendation=EconAction.LEVEL_AND_ROLL, roll_budget=20, target_level=8),
        shop_picks=["Ashe", "Morgana"],
        items=[
            ItemSuggestion(item="Deathblade", components=["B.F. Sword", "B.F. Sword"], holder="Draven", priority=1),
            ItemSuggestion(item="Bloodthirster", components=["B.F. Sword", "Negatron Cloak"], holder="Draven", priority=1),
        ],
        sell_candidates=["Garen"],
        scout_requests=[
            ScoutRequest(id="a", text="t", target_player="A"),
            ScoutRequest(id="b", text="t", target_player="B"),
        ],
        warnings=["w"],
    )
    adv = advisor.advise(state, analysis)
    check_contract(adv)
    assert len(adv.actions) == 6
    assert all(a.priority == 1 for a in adv.actions)


def test_sanitize_and_clip():
    assert sanitize("先升级\u2014\u2014再搜牌") == "先升级，再搜牌"
    assert sanitize("A \u2014 B") == "A，B"
    assert "\u2014" not in sanitize("\u2014开头")
    assert clip("一二三四五六", 4) == "一二三四"
    assert clip("一二三四五六", 4, "…") == "一二三…"
    assert sanitize("第一行\n\n第二行\u2014尾", keep_newlines=True) == "第一行\n\n第二行，尾"
    assert sanitize(None) == ""


def test_econ_reason_with_dash_is_cleaned(advisor):
    state = GameState(stage=StageRound.parse("2-2"), gold=10, level=4, hp=90)
    econ = EconPlan(gold=10, recommendation=EconAction.SAVE, reason="存钱\u2014下一个利息点 20")
    adv = advisor.advise(state, Analysis(econ=econ))
    check_contract(adv)
    assert action(adv, ActionType.SAVE).text == "存钱，下一个利息点 20"


def test_zero_roll_budget_is_not_advertised(advisor):
    state = GameState(stage=StageRound.parse("4-2"), gold=1, level=7, hp=40)
    adv = advisor.advise(state, Analysis(econ=EconPlan(gold=1, recommendation=EconAction.ROLL, roll_budget=0)))
    check_contract(adv)
    assert ActionType.ROLL not in types(adv)
    assert "搜牌到" not in adv.headline
    broke = advisor.advise(
        GameState(stage=StageRound.parse("5-1"), gold=0, level=8, hp=10),
        Analysis(econ=EconPlan(gold=0, recommendation=EconAction.ALL_IN, roll_budget=0)),
    )
    check_contract(broke)
    assert ActionType.ROLL not in types(broke) and ActionType.OTHER in types(broke)


# ---------------------------------------------------------------------------
# Regression tests (adversarial review)
# ---------------------------------------------------------------------------


def test_itemized_tank_is_not_taken_for_the_carry(advisor, set_data):
    from tft_advisor.advisor.rules import is_tank_item

    tank = unit("Braum", 2, 2, ["Warmog's Armor", "Bramble Vest", "Dragon's Claw"], row=0, col=3)
    carry = unit("Ashe", 3, 2, ["Giant Slayer", "Red Buff"], row=3, col=6)
    state = GameState(stage=StageRound.parse("3-3"), gold=30, level=6, hp=70, board=[tank, carry])
    assert find_carry(state, Analysis()).name == "Ashe"
    adv = advisor.advise(state, Analysis())
    assert ActionType.POSITION not in types(adv)  # never "move the tank to the back corner"
    assert "Ashe" in adv.positioning
    # Only tank items: no carry at all.
    assert find_carry(GameState(board=[tank]), Analysis()) is None
    # Name list (en / zh) and set-data recipes (built only from vest / cloak / belt).
    assert is_tank_item("狂徒铠甲") and is_tank_item("Chain Vest") and not is_tank_item("Giant Slayer")
    assert is_tank_item("TFT_Item_GargoyleStoneplate", set_data)
    assert not is_tank_item("TFT_Item_TitansResolve", set_data)


def test_no_position_action_from_combat_frame_or_on_carousel(advisor):
    carry = unit("Draven", 4, 2, ["Deathblade", "Giant Slayer"], row=1, col=3)
    combat = GameState(stage=StageRound.parse("3-3"), screen_type=ScreenType.COMBAT, gold=20, level=6, hp=70, board=[carry])
    assert ActionType.POSITION not in types(advisor.advise(combat, Analysis()))
    planning = combat.model_copy(update={"screen_type": ScreenType.PLANNING})
    pos = action(advisor.advise(planning, Analysis()), ActionType.POSITION)
    assert "Draven" in pos.text and "近战" in pos.text
    carousel = combat.model_copy(update={"stage": StageRound.parse("3-4"), "screen_type": ScreenType.CAROUSEL})
    assert ActionType.POSITION not in types(advisor.advise(carousel, Analysis()))


def test_carousel_does_not_advise_buying_or_selling(advisor):
    bench = [unit("Garen", 1)] * 9
    state = GameState(
        stage=StageRound.parse("3-4"), gold=30, level=6, hp=70, bench=bench, shop_units=[unit("Ashe", 3)]
    )
    adv = advisor.advise(state, Analysis(shop_picks=["Ashe"], sell_candidates=["Garen"]))
    check_contract(adv)
    assert ActionType.BUY not in types(adv) and ActionType.SELL not in types(adv)
    assert ActionType.CAROUSEL in types(adv)


def test_upgrade_flag_counts_only_one_star_copies(advisor):
    shop = [unit("Ashe", 3), unit("Ashe", 3), None, None, None]
    base = dict(stage=StageRound.parse("3-3"), gold=30, level=6, hp=70)
    # A two-star Ashe plus one more copy is not an upgrade.
    two_star = GameState(board=[unit("Ashe", 3, 2)], shop_units=shop, **base)
    buy = action(advisor.advise(two_star, Analysis(shop_picks=["Ashe"])), ActionType.BUY)
    assert "升星" not in buy.text and buy.priority == 2
    # Two-star plus two one-stars plus one bought copy is.
    pair = GameState(board=[unit("Ashe", 3, 2), unit("Ashe", 3)], bench=[unit("Ashe", 3)], shop_units=shop, **base)
    assert "升星" in action(advisor.advise(pair, Analysis(shop_picks=["Ashe"])), ActionType.BUY).text
    # One copy owned plus both shop copies.
    single = GameState(bench=[unit("Ashe", 3)], shop_units=shop, **base)
    buy = action(advisor.advise(single, Analysis(shop_picks=["Ashe", "Ashe"])), ActionType.BUY)
    assert "第1格" in buy.text and "第2格" in buy.text and "升星" in buy.text


def test_roll_recommended_without_budget_says_save(advisor, mech):
    # economy.py turns an unaffordable roll into SAVE itself (medium HP, 12 gold at a roll-down round).
    state = GameState(stage=StageRound.parse("4-1"), gold=12, level=7, xp_current=10, hp=50)
    econ = plan_economy(state, mech)
    assert econ.recommendation == EconAction.SAVE and econ.roll_budget == 0
    adv = advisor.advise(state, Analysis(econ=econ))
    check_contract(adv)
    assert "搜牌" not in adv.headline
    assert ActionType.ROLL not in types(adv) and ActionType.SAVE in types(adv)
    slow = advisor.advise(state, Analysis(econ=EconPlan(recommendation=EconAction.SLOW_ROLL, roll_budget=1, reason="慢搜")))
    assert ActionType.ROLL not in types(slow) and slow.headline == "金币不够搜牌，先存钱"
    # All-in with no gold: do not promise a roll-down.
    broke = advisor.advise(
        GameState(stage=StageRound.parse("5-1"), gold=0, level=8, hp=10),
        Analysis(econ=EconPlan(recommendation=EconAction.ALL_IN, roll_budget=0)),
    )
    assert "搜牌" not in broke.headline and "血量危险" in broke.headline
    # Level and roll with a budget below one reroll: just level.
    lr = advisor.advise(
        GameState(stage=StageRound.parse("4-1"), gold=33, level=6, xp_current=0, hp=50),
        Analysis(econ=EconPlan(recommendation=EconAction.LEVEL_AND_ROLL, roll_budget=1, target_level=7)),
    )
    assert lr.headline == "升到7级" and ActionType.ROLL not in types(lr)


def test_reroll_plan_does_not_follow_the_standard_curve(advisor, mech):
    state = GameState(stage=StageRound.parse("4-1"), gold=62, level=6, xp_current=10, hp=70)
    econ = plan_economy(state, mech, "reroll2")
    assert econ.recommendation == EconAction.SLOW_ROLL
    adv = advisor.advise(state, Analysis(econ=econ))
    check_contract(adv)
    assert "停在 6 级慢搜" in adv.plan
    assert "4-5 升 8" not in adv.plan and "补到 7" not in adv.plan
    early = GameState(stage=StageRound.parse("3-5"), gold=30, level=6, xp_current=10, hp=70)
    adv = advisor.advise(early, Analysis(econ=plan_economy(early, mech, "reroll2")))
    assert "4-1 升 7" not in adv.plan and "慢搜" in adv.plan


def test_roll_target_skips_exhausted_pool(advisor):
    def odds(name, remaining):
        return HitOdds(
            unit=name, api_name=f"TFT99_{name}", cost=4, owned_copies=3, goal_copies=9, goal_star=3,
            seen_elsewhere=0, remaining_in_pool=remaining, level=8, p_per_slot=0.0, p_in_shop=0.0,
            p_goal_by_gold={20: 0.3},
        )

    state = GameState(stage=StageRound.parse("4-2"), gold=60, level=8, hp=40)
    econ = EconPlan(recommendation=EconAction.ROLL, roll_budget=40)
    adv = advisor.advise(state, Analysis(econ=econ, odds=[odds("Draven", 2), odds("Akali", 8)]))
    roll = action(adv, ActionType.ROLL)
    assert "Draven" not in roll.text and "Akali" in roll.text
    adv = advisor.advise(state, Analysis(econ=econ, odds=[odds("Draven", 0)]))
    assert action(adv, ActionType.ROLL).text == "最多花 40 金币搜牌"


def test_long_roll_target_keeps_text_whole(advisor):
    name = "VeryVeryLongChampionName"
    o = HitOdds(
        unit=name, api_name="TFT99_X", cost=4, owned_copies=1, goal_copies=3, goal_star=2, seen_elsewhere=0,
        remaining_in_pool=9, level=8, p_per_slot=0.01, p_in_shop=0.05, p_goal_by_gold={20: 0.4},
    )
    state = GameState(stage=StageRound.parse("4-2"), gold=60, level=8, hp=40)
    roll = action(advisor.advise(state, Analysis(econ=EconPlan(recommendation=EconAction.ROLL, roll_budget=20), odds=[o])), ActionType.ROLL)
    assert roll.text == f"最多花 20 金币搜牌，找 {name}"  # odds clause dropped instead of cut mid-way


def test_stale_augment_choices_are_ignored(advisor):
    base = dict(stage=StageRound.parse("2-1"), gold=10, level=4, hp=90, screen_type=ScreenType.PLANNING)
    open_choice = GameState(augment_choices=["Rich Get Richer", "Tiny Titans"], **base)
    assert ActionType.AUGMENT in types(advisor.advise(open_choice, Analysis()))
    picked = GameState(augment_choices=["Rich Get Richer", "Tiny Titans"], augments=["Tiny Titans"], **base)
    adv = advisor.advise(picked, Analysis())
    assert ActionType.AUGMENT not in types(adv) and adv.augment is None and "海克斯" not in adv.headline
    old = GameState(
        augment_choices=["Rich Get Richer"], field_age={"augment_choices": 100.0}, last_update=400.0, **base
    )
    assert ActionType.AUGMENT not in types(advisor.advise(old, Analysis()))
    # The augment screen itself always counts.
    shown = old.model_copy(update={"screen_type": ScreenType.AUGMENT_SELECT})
    assert ActionType.AUGMENT in types(advisor.advise(shown, Analysis()))


def test_headline_is_not_repeated_as_an_action(advisor, mech):
    state = GameState(stage=StageRound.parse("1-3"), gold=3, level=3, hp=100)
    adv = advisor.advise(state, Analysis(econ=plan_economy(state, mech)))
    assert all(a.text != adv.headline for a in adv.actions)


def test_dash_only_texts_never_leave_an_empty_headline(advisor):
    state = GameState(stage=StageRound.parse("3-3"), gold=20, level=6, hp=70)
    adv = advisor.advise(state, Analysis(econ=EconPlan(recommendation=EconAction.HOLD, reason="\u2014\u2014")))
    check_contract(adv)
    carry = unit("Dr\u2014aven", 4, 2, ["Deathblade", "Giant Slayer"], row=3, col=0)
    adv = advisor.advise(GameState(stage=StageRound.parse("3-3"), board=[carry]), Analysis())
    check_contract(adv)


def test_long_texts_are_cut_at_clause_boundaries(advisor):
    assert clip("备战席有散件（Recurve Bow）暂时合不成装备，选秀和野怪优先拿能配对的散件", 40) == "备战席有散件（Recurve Bow）暂时合不成装备"
    # Never an unclosed parenthesis.
    assert clip("买 LongChampionNameAAA（第1格）", 22) == "买 LongChampionNameAAA"
    # The upgrade note survives when three long names do not fit.
    shop = [unit("Morgana", 3), unit("Volibear", 3), unit("Katarina", 3), None, None]
    state = GameState(
        stage=StageRound.parse("3-3"), gold=30, level=6, hp=70, bench=[unit("Morgana", 3), unit("Morgana", 3)], shop_units=shop
    )
    buy = action(advisor.advise(state, Analysis(shop_picks=["Morgana", "Volibear", "Katarina"])), ActionType.BUY)
    # Every pick is still named (the slot labels go first).
    assert buy.text.endswith("能升星") and len(buy.text) <= 40
    assert all(n in buy.text for n in ("Morgana", "Volibear", "Katarina"))


def test_carousel_item_slam_waits_until_after(advisor):
    state = GameState(stage=StageRound.parse("3-4"), gold=30, level=6, hp=70, item_bench=["B.F. Sword", "B.F. Sword"])
    analysis = Analysis(items=[ItemSuggestion(item="Deathblade", components=["B.F. Sword", "B.F. Sword"], holder="Draven", priority=1)])
    adv = advisor.advise(state, analysis)
    assert action(adv, ActionType.ITEM).priority == 2
    assert adv.actions[0].type == ActionType.CAROUSEL


def test_roll_target_prefers_reachable_upgrades_over_hopeless_carry(advisor):
    """At 3-2 a 4-cost carry at 0% is not the thing to roll for; cheap pairs are."""

    def odds(name, cost, p40):
        return HitOdds(
            unit=name, api_name=f"TFT99_{name}", cost=cost, owned_copies=1, goal_copies=3, goal_star=2,
            seen_elsewhere=0, remaining_in_pool=9, level=6, p_per_slot=0.01, p_in_shop=0.05,
            p_goal_by_gold={20: p40 / 2, 40: p40},
        )

    state = GameState(stage=StageRound.parse("3-2"), gold=50, level=6, hp=55)
    econ = EconPlan(recommendation=EconAction.ROLL, roll_budget=40)
    comps = [CompSuggestion(name="X", score=0.8, carry="Draven", core_units=["Draven", "Garen"])]
    adv = advisor.advise(state, Analysis(econ=econ, comps=comps, odds=[odds("Draven", 4, 0.01), odds("Garen", 1, 0.8), odds("Lucian", 2, 0.4)]))
    text = action(adv, ActionType.ROLL).text
    assert "Draven" not in text and text.index("Garen") < text.index("Lucian")
    hopeless = advisor.advise(state, Analysis(econ=econ, odds=[odds("Draven", 4, 0.01)]))
    assert action(hopeless, ActionType.ROLL).text.endswith("找对子升星")


def _roll_odds(name, cost, owned, p40, remaining=20):
    return HitOdds(
        unit=name, api_name=f"TFT99_{name}", cost=cost, owned_copies=owned, goal_copies=3, goal_star=2,
        seen_elsewhere=0, remaining_in_pool=remaining, level=8, p_per_slot=0.01, p_in_shop=0.1,
        p_goal_by_gold={20: p40 / 2, 40: p40},
    )


def test_late_roll_targets_are_comp_units_ranked_by_value(advisor):
    """5-1, level 8, complete comp: a likely off-comp pair, a sell candidate or
    an unowned off-comp unit must not be named over the comp's 4-cost."""
    state = GameState(
        stage=StageRound.parse("5-1"), gold=52, level=8, hp=35,
        board=[unit("Ahri", 4, 2, row=3, col=0), unit("Morgana", 4, row=3, col=1), unit("Sentinel", 1, row=0, col=2)],
        bench=[unit("Master Yi", 3), unit("Kayle", 2), unit("Kayle", 2)],
    )
    comp = CompSuggestion(
        name="Ahri Morgana", score=0.9, carry="Ahri", core_units=["Ahri", "Morgana", "Sentinel"],
        have_units=["Ahri", "Morgana", "Sentinel"],
    )
    odds = [
        _roll_odds("Kayle", 2, 2, 0.74),
        _roll_odds("Garen", 1, 0, 0.9),
        _roll_odds("Master Yi", 3, 1, 0.54),
        _roll_odds("Morgana", 4, 1, 0.46),
        _roll_odds("Sentinel", 1, 1, 0.46),
    ]
    econ = EconPlan(recommendation=EconAction.ROLL, roll_budget=42)
    analysis = Analysis(econ=econ, comps=[comp], odds=odds, sell_candidates=["Master Yi"])
    text = action(advisor.advise(state, analysis), ActionType.ROLL).text
    assert "Kayle" not in text and "Master Yi" not in text and "Garen" not in text
    assert "Morgana" in text and "Sentinel" in text and text.index("Morgana") < text.index("Sentinel")


def test_roll_second_target_needs_real_value(advisor):
    """A cheap unowned comp filler far below the carry is not named next to it."""
    state = GameState(stage=StageRound.parse("4-1"), gold=56, level=7, hp=48, board=[unit("Yunara", 2, row=3, col=0)])
    comp = CompSuggestion(name="X", score=0.5, carry="Yunara", core_units=["Yunara", "Kayle"], missing_units=["Kayle"])
    econ = EconPlan(recommendation=EconAction.ROLL, roll_budget=34)
    odds = [_roll_odds("Yunara", 2, 1, 0.57), _roll_odds("Kayle", 2, 0, 0.27)]
    text = action(advisor.advise(state, Analysis(econ=econ, comps=[comp], odds=odds)), ActionType.ROLL).text
    assert "Yunara" in text and "Kayle" not in text


def test_early_roll_targets_allow_owned_pairs_but_not_sell_candidates(advisor):
    state = GameState(stage=StageRound.parse("3-2"), gold=50, level=6, hp=60, bench=[unit("Lucian", 2), unit("Lucian", 2), unit("Vi", 3)])
    comp = CompSuggestion(name="X", score=0.6, carry="Draven", core_units=["Draven", "Garen"], missing_units=["Draven", "Garen"])
    econ = EconPlan(recommendation=EconAction.ROLL, roll_budget=40)
    odds = [_roll_odds("Vi", 3, 1, 0.9), _roll_odds("Lucian", 2, 2, 0.6), _roll_odds("Ashe", 3, 0, 0.9)]
    analysis = Analysis(econ=econ, comps=[comp], odds=odds, sell_candidates=["Vi"])
    text = action(advisor.advise(state, analysis), ActionType.ROLL).text
    assert "Lucian" in text and "Vi" not in text and "Ashe" not in text


def test_late_roll_targets_follow_the_board_when_the_comp_is_a_loose_guess(advisor):
    state = GameState(
        stage=StageRound.parse("4-2"), gold=50, level=7, hp=60,
        board=[unit("Vi", 3, row=0, col=3)], bench=[unit("Kayle", 2), unit("Kayle", 2)],
    )
    econ = EconPlan(recommendation=EconAction.ROLL, roll_budget=40)
    odds = [_roll_odds("Kayle", 2, 2, 0.8), _roll_odds("Vi", 3, 1, 0.5)]
    weak = CompSuggestion(name="X", score=0.2, carry="Draven", core_units=["Draven"], missing_units=["Draven"])
    text = action(advisor.advise(state, Analysis(econ=econ, comps=[weak], odds=odds)), ActionType.ROLL).text
    assert "Vi" in text and "Kayle" not in text
    committed = weak.model_copy(update={"score": 0.8})
    text = action(advisor.advise(state, Analysis(econ=econ, comps=[committed], odds=odds)), ActionType.ROLL).text
    assert text == "最多花 40 金币搜牌"


def test_items_text_has_no_duplicate_label(advisor):
    state = GameState(stage=StageRound.parse("3-3"), gold=20, level=6, hp=70)
    slam = Analysis(items=[ItemSuggestion(item="Deathblade", components=["B.F. Sword", "B.F. Sword"], holder="Draven")])
    assert advisor.advise(state, slam).items == "Deathblade 给 Draven"
    comp = CompSuggestion(name="X", score=0.8, carry="Draven", carry_items=["Deathblade", "Giant Slayer"])
    adv = advisor.advise(state, Analysis(comps=[comp]))
    assert adv.items == "Draven：Deathblade、Giant Slayer"
    assert all("装备：" not in t for t in all_text(adv))


def test_augment_copy_uses_client_term_and_fullwidth_colon(advisor):
    state = GameState(stage=StageRound.parse("2-1"), screen_type=ScreenType.AUGMENT_SELECT, gold=10, level=4, hp=90)
    adv = advisor.advise(state, Analysis())
    check_contract(adv)
    texts = all_text(adv)
    assert all("增强" not in t and ": " not in t for t in texts)
    assert not action(adv, ActionType.AUGMENT).text.endswith("的")
    assert adv.augment == AUGMENT_GENERIC


def test_wisp_in_shop_gets_a_short_hint(advisor):
    from tft_advisor.models import ShopSlot

    shop = [ShopSlot(name="Vi", cost=3), ShopSlot(name="Garen", cost=1), ShopSlot(), ShopSlot(), ShopSlot(name="Wisp: Grow Up", cost=3)]
    units = [unit("Vi", 3), unit("Garen", 1), None, None, None]
    early = GameState(stage=StageRound.parse("2-5"), gold=20, level=5, hp=80, shop=shop, shop_units=units)
    adv = advisor.advise(early, Analysis())
    check_contract(adv)
    hint = [a for a in adv.actions if "精灵" in a.text]
    assert len(hint) == 1 and hint[0].priority == 3
    assert hint[0].text == "第5格精灵「Grow Up」（3金币）：前期经济或经验类值得买"
    late = early.model_copy(update={"stage": StageRound.parse("4-2"), "hp": 40})
    assert any(a.text.endswith("稳血时优先买战斗类") for a in advisor.advise(late, Analysis()).actions)
    # Cannot afford it, or on the carousel: no hint.
    poor = early.model_copy(update={"gold": 2})
    assert not any("精灵" in a.text for a in advisor.advise(poor, Analysis()).actions)
    carousel = early.model_copy(update={"stage": StageRound.parse("3-4")})
    assert not any("精灵" in a.text for a in advisor.advise(carousel, Analysis()).actions)
    # A long name falls back to a short text instead of being cut.
    long = early.model_copy(update={"shop": shop[:4] + [ShopSlot(name="Wisp: An Extremely Long Wisp Name Here", cost=12)]})
    text = next(a.text for a in advisor.advise(long, Analysis()).actions if "精灵" in a.text)
    assert text == "第5格是精灵（12金币）：前期经济或经验类值得买"


def test_plan_milestones_follow_the_line(advisor):
    base = dict(stage=StageRound.parse("4-1"), gold=40, level=7, xp_current=0, hp=80)
    std = advisor.advise(GameState(**base), Analysis(econ=EconPlan(recommendation=EconAction.SAVE, style="standard")))
    assert "4-5 升 8" in std.plan
    # Medium HP: fast 8 goes 8 at 4-2 (economy._wanted_level).
    medium = dict(base, hp=55)
    fast8 = advisor.advise(GameState(**medium), Analysis(econ=EconPlan(recommendation=EconAction.SAVE, style="fast8")))
    assert "4-2 升 8" in fast8.plan and "4-5 升 8" not in fast8.plan
    late = dict(medium, stage=StageRound.parse("4-6"), level=8)
    fast9 = advisor.advise(GameState(**late), Analysis(econ=EconPlan(recommendation=EconAction.SAVE, style="fast9")))
    assert "5-2 升 9" in fast9.plan and "5-5" not in fast9.plan


def test_plan_milestones_match_the_econ_engine_at_healthy_hp(advisor, mech):
    """Healthy fast lines level a round earlier (7 at 3-5, 8 at 4-1, 9 at
    5-1): the plan must name the rounds the econ engine will act on."""
    from tft_advisor.engine.economy import _wanted_level

    early = GameState(stage=StageRound.parse("3-2"), gold=40, level=6, xp_current=0, hp=80)
    fast8 = advisor.advise(early, Analysis(econ=EconPlan(recommendation=EconAction.SAVE, style="fast8")))
    assert "3-5 升 7" in fast8.plan and "4-1 升 8" in fast8.plan and "4-2" not in fast8.plan
    late = GameState(stage=StageRound.parse("4-6"), gold=40, level=8, xp_current=0, hp=80)
    fast9 = advisor.advise(late, Analysis(econ=EconPlan(recommendation=EconAction.SAVE, style="fast9")))
    assert "5-1 升 9" in fast9.plan and "5-2" not in fast9.plan
    # Every milestone is exactly where the engine starts wanting that level.
    for style in ("standard", "fast8", "fast9"):
        for hp, bucket in ((80, "healthy"), (55, "medium")):
            for r, lvl in advisor._milestones(style, hp):
                if r.stage < 2:
                    continue
                std = mech.standard_level_at(r)
                assert _wanted_level(r, std, style, bucket) == lvl
                prev = StageRound(stage=r.stage, round=r.round - 1) if r.round > 1 else StageRound(stage=r.stage - 1, round=7)
                assert _wanted_level(prev, mech.standard_level_at(prev), style, bucket) < lvl


# ---------------------------------------------------------------------------
# Regression tests (tft-domain review, round 2)
# ---------------------------------------------------------------------------


def test_reroll_plan_follows_econ_style_not_reason_words(advisor, mech):
    """Stage-2, roll-down and carousel reasons never say 慢搜 / 赌狗: the plan
    must still follow the reroll line (econ.style), not the standard curve."""
    stage2 = GameState(stage=StageRound.parse("2-5"), gold=30, level=5, xp_current=0, hp=80)
    econ = plan_economy(stage2, mech, "reroll2")
    assert econ.style == "reroll2" and "慢搜" not in econ.reason
    adv = advisor.advise(stage2, Analysis(econ=econ))
    check_contract(adv)
    assert "3-1 到 6 级开始慢搜" in adv.plan
    assert "4-1 升 7" not in adv.plan and "3-2 升 6" not in adv.plan
    # 4-1, medium HP, not 3-star yet: econ rolls to ~20, the plan stays at 6.
    chase = GameState(stage=StageRound.parse("4-1"), gold=34, level=6, xp_current=0, hp=50)
    econ = plan_economy(chase, mech, "reroll2", key_star=2)
    assert econ.recommendation == EconAction.ROLL and econ.style == "reroll2"
    adv = advisor.advise(chase, Analysis(econ=econ))
    assert "停在 6 级搜牌追三星" in adv.plan
    assert "补到 7" not in adv.plan and "4-5 升 8" not in adv.plan
    # Carousel: the HOLD reason is generic, the line is still a reroll.
    car = GameState(stage=StageRound.parse("3-4"), gold=30, level=6, hp=70)
    adv = advisor.advise(car, Analysis(econ=plan_economy(car, mech, "reroll2")))
    assert "停在 6 级慢搜" in adv.plan and "4-1 升 7" not in adv.plan
    # A level 7 reroll reaches its level at 3-5.
    r3 = GameState(stage=StageRound.parse("3-2"), gold=30, level=6, hp=80)
    adv = advisor.advise(r3, Analysis(econ=EconPlan(recommendation=EconAction.SAVE, style="reroll3")))
    assert "3-5 到 7 级开始慢搜" in adv.plan
    # The reroll ends at 4-5 even where the engine kept the style (carousel 5-4).
    late = GameState(stage=StageRound.parse("5-4"), gold=30, level=7, hp=70)
    adv = advisor.advise(late, Analysis(econ=plan_economy(late, mech, "reroll2")))
    assert "慢搜三星" not in adv.plan and "升 8" in adv.plan


def test_reroll_style_makes_cheap_comp_units_core_roll_targets(advisor):
    """_roll_value reads econ.style: on a reroll line a 1-cost comp unit is core."""
    state = GameState(stage=StageRound.parse("4-1"), gold=40, level=6, hp=50)
    comp = CompSuggestion(name="X", score=0.8, carry="Draven", core_units=["Draven", "Garen", "Vi"])
    odds = [_roll_odds("Garen", 1, 1, 0.65), _roll_odds("Vi", 3, 1, 0.6)]
    reroll = EconPlan(recommendation=EconAction.ROLL, roll_budget=40, style="reroll1", reason="血量 50：搜到 20")
    text = action(advisor.advise(state, Analysis(econ=reroll, comps=[comp], odds=odds)), ActionType.ROLL).text
    assert text.index("Garen") < text.index("Vi")
    std = reroll.model_copy(update={"style": "standard"})
    text = action(advisor.advise(state, Analysis(econ=std, comps=[comp], odds=odds)), ActionType.ROLL).text
    assert text.index("Vi") < text.index("Garen")


def test_fast_line_behind_schedule_still_hears_level_8(advisor):
    save = dict(recommendation=EconAction.SAVE)
    for sr in ("4-2", "4-3"):
        state = GameState(stage=StageRound.parse(sr), gold=38, level=7, hp=70)
        adv = advisor.advise(state, Analysis(econ=EconPlan(style="fast8", **save)))
        check_contract(adv)
        # Not affordable yet (38 gold): the plan names the gold for the level, not "save to 50".
        assert "升 8 找 4 费主C" in adv.plan and "5-5 升 9" in adv.plan and "先存到" not in adv.plan
    # Fast 9 at 6 after 4-2: catch up to 8, then 9 at 5-1 (healthy HP).
    state = GameState(stage=StageRound.parse("4-2"), gold=30, level=6, hp=70)
    adv = advisor.advise(state, Analysis(econ=EconPlan(style="fast9", **save)))
    assert "尽快补到 8 级" in adv.plan and "5-1 升 9" in adv.plan
    # Level 7 after 5-5: one level at a time, 9 named once.
    state = GameState(stage=StageRound.parse("5-6"), gold=38, level=7, hp=70)
    adv = advisor.advise(state, Analysis(econ=EconPlan(style="standard", **save)))
    assert "升 8" in adv.plan and "再升 9 找 5 费" in adv.plan
    assert adv.plan.count("升 9") == 1 and "补到 9" not in adv.plan
    # Econ levels to 7 now while the fast 8 line wants 8: 8 is still named.
    state = GameState(stage=StageRound.parse("4-2"), gold=40, level=6, hp=70)
    adv = advisor.advise(state, Analysis(econ=EconPlan(recommendation=EconAction.LEVEL, target_level=7, style="fast8")))
    assert "现在升 7" in adv.plan and "尽快升 8" in adv.plan
    # On schedule at 9: no "升 9" at all.
    state = GameState(stage=StageRound.parse("5-6"), gold=38, level=9, hp=70)
    adv = advisor.advise(state, Analysis(econ=EconPlan(style="standard", **save)))
    assert "升 9" not in adv.plan and "慢搜找 5 费" in adv.plan


def test_carousel_pick_does_not_reuse_slammed_components(advisor):
    """B.F. Sword + Gloves are slammed into Infinity Edge: the carousel cannot
    count that B.F. Sword again for a Deathblade."""
    state = GameState(
        stage=StageRound.parse("3-4"), gold=30, level=6, hp=70, item_bench=["B.F. Sword", "Sparring Gloves"],
        board=[unit("Draven", 4, 2, row=3, col=0)],
    )
    comp = CompSuggestion(name="X", score=0.8, carry="Draven", carry_items=["Deathblade", "Giant Slayer"])
    ie = ItemSuggestion(item="Infinity Edge", components=["B.F. Sword", "Sparring Gloves"], holder="Draven", priority=1)
    adv = advisor.advise(state, Analysis(comps=[comp], items=[ie]))
    check_contract(adv)
    car = action(adv, ActionType.CAROUSEL).text
    # Both carry items now need 2 components; the first BiS item is named, and
    # the slammed item is never "picked" again.
    assert "Infinity Edge" not in car and "凑 Deathblade" in car
    # A leftover component that completes an item beats a two-component item.
    state = state.model_copy(update={"item_bench": ["B.F. Sword", "Sparring Gloves", "Recurve Bow"]})
    adv = advisor.advise(state, Analysis(comps=[comp], items=[ie]))
    assert "做 Giant Slayer" in action(adv, ActionType.CAROUSEL).text


def test_carousel_skips_a_carry_with_three_items(advisor):
    full = unit("Draven", 4, 2, ["Red Buff", "Deathblade", "Infinity Edge"], row=3, col=0)
    state = GameState(stage=StageRound.parse("4-4"), gold=30, level=7, hp=70, item_bench=["B.F. Sword"], board=[full])
    comp = CompSuggestion(name="X", score=0.8, carry="Draven", carry_items=["Red Buff", "Deathblade", "Giant Slayer"])
    leftover = ItemSuggestion(item="Giant Slayer", components=["B.F. Sword", "Recurve Bow"], holder="Draven", priority=3)
    adv = advisor.advise(state, Analysis(comps=[comp], items=[leftover]))
    check_contract(adv)
    car = action(adv, ActionType.CAROUSEL).text
    assert "Recurve Bow" not in car and "Draven" not in car and "主C装备已满" in car
    assert "主C" not in adv.headline or "已满" in adv.headline
    assert adv.items is None or "Giant Slayer 给 Draven" not in adv.items
    # With an empty bench the comp's carry items are skipped too.
    empty = state.model_copy(update={"item_bench": []})
    adv = advisor.advise(empty, Analysis(comps=[comp]))
    assert "B.F. Sword" not in action(adv, ActionType.CAROUSEL).text
    # A free slot: the carry items are back.
    two = state.model_copy(update={"board": [full.model_copy(update={"items": ["Red Buff", "Infinity Edge"]})], "item_bench": []})
    assert "Deathblade" in action(advisor.advise(two, Analysis(comps=[comp])), ActionType.CAROUSEL).text


def test_completed_item_on_bench_is_equipped_not_combined(advisor):
    state = GameState(stage=StageRound.parse("4-3"), gold=38, level=7, hp=70, item_bench=["Red Buff"],
                      board=[unit("Draven", 4, 2, row=3, col=0), unit("Braum", 2, row=0, col=3)])
    built = ItemSuggestion(item="Red Buff", components=[], holder="Draven", priority=1)
    adv = advisor.advise(state, Analysis(items=[built]))
    check_contract(adv)
    assert adv.headline == "把Red Buff装给Draven"
    assert action(adv, ActionType.ITEM).text == "把 Red Buff 装给 Draven"
    assert all("合成" not in t for t in all_text(adv))
    # Nobody on this (AP) team uses it: keep it, never the headline, and do
    # not claim that every item slot is full.
    orphan = built.model_copy(update={"holder": None})
    adv = advisor.advise(state, Analysis(items=[orphan]))
    item = action(adv, ActionType.ITEM)
    assert item.text == "现在的阵容没人适合用 Red Buff，先留着" and item.priority == 2
    assert "Red Buff" not in adv.headline and "满" not in item.text
    # Really full board: say so.
    full = [unit("Draven", 4, 2, ["A", "B", "C"], row=3, col=0)]
    adv = advisor.advise(state.model_copy(update={"board": full}), Analysis(items=[orphan]))
    assert "装备格都满了" in action(adv, ActionType.ITEM).text


def test_raw_api_ids_never_reach_the_items_text(advisor):
    from tft_advisor.advisor.rules import is_display_name

    state = GameState(stage=StageRound.parse("3-3"), gold=20, level=6, hp=70)
    comp = CompSuggestion(
        name="Solar Kayle", score=0.8, carry="Kayle",
        carry_items=["Guinsoo's Rageblade", "DA_Artifact_NavoriFlickerblade", "Rabadon's Deathcap"],
    )
    adv = advisor.advise(state, Analysis(comps=[comp]))
    assert adv.items == "Kayle：Guinsoo's Rageblade、Rabadon's Deathcap"
    only_raw = comp.model_copy(update={"carry_items": ["DA_18_EmblemFloraFatalisAugment"]})
    assert advisor.advise(state, Analysis(comps=[only_raw])).items is None
    assert is_display_name("Warmogs Armor") and is_display_name("无尽之刃") and not is_display_name("TFT_Item_Foo")


def test_augment_pick_uses_the_bundled_reference():
    from tft_advisor.data.augments import AugmentData
    from tft_advisor.data.setdata import SetData, bundled_snapshot

    sd = SetData.from_cdragon(bundled_snapshot("zh_cn"), bundled_snapshot("en_us"))
    adv = RulesAdvisor(set_data=sd)
    adv.augments = AugmentData.load(18)
    state = GameState(
        stage=StageRound.parse("2-1"), gold=10, level=4, hp=100,
        augment_choices=["高级贷款", "休眠锻炉", "不认识的符文"],
    )
    out = adv.advise(state, Analysis())
    aug = [a for a in out.actions if a.type == ActionType.AUGMENT]
    assert aug and "休眠锻炉" in aug[0].text
    assert "休眠锻炉" in out.headline and out.augment and "推荐 休眠锻炉" in out.augment
    # Without reference data the generic hint stays.
    plain = RulesAdvisor(set_data=sd).advise(state, Analysis())
    assert "休眠锻炉" not in plain.headline


def test_strategist_message_carries_augment_effects():
    import json as _json

    from tft_advisor.advisor.prompts import build_state_message
    from tft_advisor.data.augments import AugmentData

    state = GameState(stage=StageRound.parse("3-2"), augment_choices=["Latent Forge", "Mystery Thing"])
    msg = build_state_message(state, Analysis(), None, augments=AugmentData.load(18))
    payload = _json.loads(msg[msg.index("{"): msg.rindex("}") + 1])
    info = payload["augment_info"]
    assert [x["name"] for x in info] == ["Latent Forge"] and info[0]["tier"] == "S" and info[0]["effect"]


# ---------------------------------------------------------------------------
# Regression tests (tft-domain review, round 3)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_sd():
    from tft_advisor.data.setdata import bundled_snapshot

    return SetData.from_cdragon(bundled_snapshot("zh_cn"), bundled_snapshot("en_us"))


def test_carousel_counts_items_on_their_way_to_the_carry(advisor):
    """A completed carry item on the bench, or a slam planned this round,
    fills the carry's slot: no second copy, no components for a full carry."""
    comp = CompSuggestion(name="X", score=0.8, carry="Draven", carry_items=["Red Buff", "Infinity Edge", "Deathblade"])
    two = unit("Draven", 4, 2, ["Red Buff", "Infinity Edge"], row=3, col=0)
    base = dict(stage=StageRound.parse("4-4"), gold=30, level=7, hp=60)
    # Repro 1: Deathblade is already built and waits on the bench for Draven.
    state = GameState(**base, board=[two], item_bench=["Deathblade"])
    built = ItemSuggestion(item="Deathblade", components=[], holder="Draven", priority=1)
    adv = advisor.advise(state, Analysis(comps=[comp], items=[built]))
    check_contract(adv)
    car = action(adv, ActionType.CAROUSEL).text
    assert "Deathblade" not in car and "B.F. Sword" not in car and "主C装备已满" in car
    assert adv.headline == "选秀：拿坦克装备或缺的英雄"
    # Repro 2: the plan slams Deathblade from two swords this round.
    state = GameState(**base, board=[two], item_bench=["B.F. Sword", "B.F. Sword"])
    slam = ItemSuggestion(item="Deathblade", components=["B.F. Sword", "B.F. Sword"], holder="Draven", priority=1)
    adv = advisor.advise(state, Analysis(comps=[comp], items=[slam]))
    assert adv.headline == "选秀：拿坦克装备或缺的英雄"
    assert "主C装备已满" in action(adv, ActionType.CAROUSEL).text
    # One item held, Deathblade on the bench: the pick goes to the third item.
    one = unit("Draven", 4, 2, ["Red Buff"], row=3, col=0)
    state = GameState(**base, board=[one], item_bench=["Deathblade"])
    adv = advisor.advise(state, Analysis(comps=[comp], items=[built]))
    car = action(adv, ActionType.CAROUSEL).text
    assert "Infinity Edge" in car and "Deathblade" not in car
    # Without the comp: the fielded carry (2 damage items) counts the slam too.
    adv = advisor.advise(GameState(**base, board=[two], item_bench=["Deathblade"]), Analysis(items=[built]))
    assert adv.headline == "选秀：拿坦克装备或缺的英雄"
    # Thief's Gloves fills every slot of its holder.
    gloves = ItemSuggestion(item="Thief's Gloves", components=[], holder="Draven", priority=1)
    empty = unit("Draven", 4, 2, row=3, col=0)
    adv = advisor.advise(GameState(**base, board=[empty], item_bench=["Thief's Gloves"]), Analysis(comps=[comp], items=[gloves]))
    assert adv.headline == "选秀：拿坦克装备或缺的英雄"


def test_augment_pick_uses_the_target_comp_and_board_traits(real_sd):
    from tft_advisor.data.augments import AugmentData

    adv = RulesAdvisor(set_data=real_sd)
    adv.augments = AugmentData.load(18)

    def u(name, row, col):
        ch = real_sd.resolve_champion(name)
        return Unit(api_name=ch.api_name, name=ch.name, cost=ch.cost, traits=list(ch.traits), row=row, col=col)

    base = dict(
        stage=StageRound.parse("2-1"), gold=10, level=4, hp=100, screen_type=ScreenType.AUGMENT_SELECT,
        augment_choices=["烧起来", "认知税", "手气不错"],
    )
    inferno = [u("韦鲁斯", 3, 0), u("阿卡丽", 0, 2), u("慎", 0, 3), u("约里克", 0, 4)]
    comp = CompSuggestion(
        name="日蚀骑士 近战", score=0.37, carry="韦鲁斯",
        core_units=["蕾欧娜", "阿卡丽", "卡蜜尔", "奥恩", "韦鲁斯", "凯尔", "瑟庄妮"],
    )
    # Trait panel not read: the board's own 3 Inferno count.
    out = adv.advise(GameState(**base, board=inferno), Analysis(comps=[comp]))
    assert out.headline == "海克斯选 烧起来" and "契合当前羁绊" in out.augment
    assert adv.advise(GameState(**base, board=inferno), Analysis()).headline == "海克斯选 烧起来"
    # A committed comp lends its traits even before the board has them.
    yorick = [u("约里克", 0, 4)]
    committed = comp.model_copy(update={"score": 0.8})
    assert adv.advise(GameState(**base, board=yorick), Analysis(comps=[committed])).headline == "海克斯选 烧起来"
    # A loose guess only where the board already started the trait.
    assert adv.advise(GameState(**base, board=yorick), Analysis(comps=[comp])).headline == "海克斯选 认知税"
    started = [u("约里克", 0, 4), u("韦鲁斯", 3, 0)]
    assert adv.advise(GameState(**base, board=started), Analysis(comps=[comp])).headline == "海克斯选 烧起来"


def test_tank_item_carrier_is_positioned_as_the_main_tank(advisor, real_sd):
    comp = CompSuggestion(
        name="Warwick Reroll", score=0.8, carry="Braum",
        carry_items=["Warmog's Armor", "Gargoyle Stoneplate", "Crownguard"], positions={"Braum": [0, 3]},
    )
    state = GameState(
        stage=StageRound.parse("4-5"), gold=30, level=8, hp=60,
        board=[unit("Braum", 2, 2, ["Warmog's Armor", "Gargoyle Stoneplate"], row=1, col=3), unit("Warwick", 1, 3, row=0, col=2)],
    )
    adv = advisor.advise(state, Analysis(comps=[comp]))
    check_contract(adv)
    assert adv.positioning == "Braum 是主坦：放前排第4格（阵容推荐站位），扛伤害保护后排输出"
    assert all("后排角落" not in t for t in all_text(adv))
    assert action(adv, ActionType.POSITION).text == "把 Braum 移到前排"
    # Already in front: no move; no library spot: the middle of the front row.
    front = state.model_copy(update={"board": [state.board[0].model_copy(update={"row": 0}), state.board[1]]})
    adv = advisor.advise(front, Analysis(comps=[comp.model_copy(update={"positions": {}})]))
    assert ActionType.POSITION not in types(adv)
    assert adv.positioning.startswith("Braum 是主坦：放前排中间")
    # A damage carry keeps the back-row advice.
    dmg = comp.model_copy(update={"carry_items": ["Deathblade", "Giant Slayer", "Warmog's Armor"]})
    assert "后排角落" in advisor.advise(state, Analysis(comps=[dmg])).positioning
    # Real set data: Malphite (melee) is no "近战主C" in Warwick Reroll.
    mal = real_sd.resolve_champion("墨菲特")
    board = [Unit(api_name=mal.api_name, name=mal.name, cost=mal.cost, star=2, items=["狂徒铠甲", "石像鬼石板甲"], row=0, col=3)]
    lib = CompSuggestion(
        name="沃里克 赌狗", score=0.8, carry="墨菲特", carry_items=["狂徒铠甲", "石像鬼石板甲", "冕卫"],
        positions={"墨菲特": [0, 3]},
    )
    real = RulesAdvisor(set_data=real_sd).advise(state.model_copy(update={"board": board}), Analysis(comps=[lib]))
    assert "近战主C" not in real.positioning and real.positioning.startswith("墨菲特 是主坦：放前排第4格")


def test_augment_text_has_no_double_punctuation_or_build_junk(real_sd):
    from tft_advisor.advisor.rules import AUGMENT_MORE, augment_effect
    from tft_advisor.data.augments import AugmentData

    data = AugmentData.load(18)
    assert augment_effect(data.lookup("利落保镖").desc).endswith("提升该加成2。")
    assert augment_effect(data.lookup("遥遥领先").desc).endswith("获得4经验值。")
    assert augment_effect(data.lookup("水乳交融").desc).endswith("攻击速度。")
    assert augment_effect(data.lookup("双城赢家").desc) == ""  # English text in the zh field
    adv = RulesAdvisor(set_data=real_sd, claude_enabled=True)
    adv.augments = data
    base = dict(stage=StageRound.parse("2-1"), gold=10, level=4, hp=100, screen_type=ScreenType.AUGMENT_SELECT)
    for choices in (["认知税", "手气不错"], ["双城赢家"], ["利落保镖"], ["遥遥领先"], ["烧起来"]):
        text = adv.advise(GameState(**base, augment_choices=choices), Analysis()).augment
        assert "。，" not in text and "2022" not in text and " 10" not in text and "seconds" not in text, text
        assert text.endswith(AUGMENT_MORE), text  # the Claude pointer is never cut off
        assert len(text) <= 120
    text = adv.advise(GameState(**base, augment_choices=["认知税"]), Analysis()).augment
    assert text == "推荐 认知税（经济）：获得8金币和1经验值，详细对比看 Claude 建议"


def test_level_8_tag_follows_the_carry(advisor):
    save = EconPlan(recommendation=EconAction.SAVE)
    # 5 cost carry on a fast 9 line: level 8 only stabilizes.
    fast9 = CompSuggestion(name="Kayle 9", score=0.8, carry="Kayle", style="fast9")
    for sr in ("3-5", "4-1"):
        state = GameState(stage=StageRound.parse(sr), gold=40, level=7, hp=55, board=[unit("Kayle", 5, row=3, col=0)])
        plan = advisor.advise(state, Analysis(econ=save.model_copy(update={"style": "fast9"}), comps=[fast9])).plan
        assert "找 4 费主C" not in plan and "升 8 稳血" in plan and "升 9 找 5 费" in plan, plan
    # Not owned yet: the cost comes from the set data.
    state = GameState(stage=StageRound.parse("4-1"), gold=40, level=7, hp=55)
    plan = advisor.advise(state, Analysis(econ=save, comps=[fast9.model_copy(update={"style": "standard"})])).plan
    assert "4-5 升 8 稳血" in plan
    # A loose comp guess keeps the standard tag.
    loose = fast9.model_copy(update={"style": "standard", "score": 0.2})
    assert "4-5 升 8 找 4 费主C" in advisor.advise(state, Analysis(econ=save, comps=[loose])).plan
    # A finished reroll (4-5, standard econ again) levels to add 4 costs.
    reroll = CompSuggestion(name="Warwick Reroll", score=0.8, carry="Warwick", style="reroll2")
    state = GameState(stage=StageRound.parse("4-5"), gold=30, level=7, hp=60, board=[unit("Warwick", 1, 3, row=0, col=2)])
    plan = advisor.advise(state, Analysis(econ=save, comps=[reroll])).plan
    assert "升 8 补 4 费" in plan and "找 4 费主C" not in plan
    # A 4 cost carry already 2-star: level 8 adds 4 costs, not the carry.
    std = CompSuggestion(name="Draven", score=0.8, carry="Draven", style="standard")
    state = GameState(stage=StageRound.parse("4-1"), gold=30, level=7, hp=60, board=[unit("Draven", 4, 2, row=3, col=0)])
    assert "4-5 升 8 补 4 费" in advisor.advise(state, Analysis(econ=save, comps=[std])).plan
    one = state.model_copy(update={"board": [unit("Draven", 4, 1, row=3, col=0)]})
    assert "4-5 升 8 找 4 费主C" in advisor.advise(one, Analysis(econ=save, comps=[std])).plan


def test_save_headline_names_every_unit_the_buy_action_buys(advisor):
    state = GameState(
        stage=StageRound.parse("2-5"), gold=28, level=5, hp=80,
        shop_units=[unit("Vi", 2), None, unit("Karma", 1), unit("Caitlyn", 3), None],
    )
    econ = EconPlan(recommendation=EconAction.SAVE, reason="存钱吃利息")
    adv = advisor.advise(state, Analysis(econ=econ, shop_picks=["Karma", "Vi", "Caitlyn"]))
    check_contract(adv)
    buy = action(adv, ActionType.BUY).text
    assert all(n in buy for n in ("Karma", "Vi", "Caitlyn"))
    assert adv.headline == "买Karma、Vi、Caitlyn，其余存钱"
    # Two copies of one unit, and names too long for the headline.
    adv = advisor.advise(state, Analysis(econ=econ, shop_picks=["Karma", "Karma"]))
    assert adv.headline == "买2张Karma，其余存钱"
    long = ["Aurelion Sol Prime", "Miss Fortune Prime", "Tahm Kench Prime"]
    adv = advisor.advise(state, Analysis(econ=econ, shop_picks=long))
    assert len(adv.headline) <= 30 and "其余存钱" in adv.headline
    # Neither the action nor the headline hides a pick.
    buy = action(adv, ActionType.BUY).text
    assert "等 3 张" in buy or "3 张牌" in buy or all(n in buy for n in long)
    assert "等3张" in adv.headline or "买3张牌" in adv.headline
    # One pick: unchanged; level headline lists both units.
    assert advisor.advise(state, Analysis(econ=econ, shop_picks=["Karma"])).headline == "买Karma，其余存钱"
    level = EconPlan(recommendation=EconAction.LEVEL, target_level=6)
    assert advisor.advise(state, Analysis(econ=level, shop_picks=["Karma", "Vi"])).headline == "升到6级，买Karma、Vi"


def test_sanitize_keeps_rounds_and_ranges_written_with_em_dashes():
    assert sanitize("4—1 升8") == "4-1 升8"
    assert sanitize("搜到 10——20 金币") == "搜到 10-20 金币"
    assert sanitize("3 — 2 搜牌") == "3-2 搜牌"
    assert sanitize("奥恩——前排") == "奥恩，前排"
    for t in ("4—1", "a―b", "1⸺2"):
        assert "—" not in sanitize(t) and "―" not in sanitize(t) and "⸺" not in sanitize(t)


def test_rules_share_the_econ_engine_tables():
    from tft_advisor.advisor import rules
    from tft_advisor.engine import economy

    assert rules.ROLLDOWN_POINTS is economy.ROLLDOWN_ROUNDS
    # HP 45 is "medium" in the econ buckets, 44 is "low": the rules agree.
    assert not rules._low_hp(45) and rules._low_hp(44) and not rules._low_hp(None)
    assert economy.hp_bucket(44) == "low" and economy.hp_bucket(45) == "medium"


def test_augment_pick_gets_the_level_and_round_from_the_state(real_sd):
    """The picker's timing penalty needs the real level: at level 8 on 4-2 a
    'reach level 8' payout is already spent."""
    from tft_advisor.data.augments import AugmentData

    adv = RulesAdvisor(set_data=real_sd)
    adv.augments = AugmentData.load(18)
    state = GameState(
        stage=StageRound.parse("4-2"), gold=30, level=8, hp=60,
        augment_choices=["Epic Rolldown", "Ascension", "Clockwork Accelerator"],
    )
    pick = adv._augment_pick(state, Analysis())
    assert pick is not None and pick[0] != "Epic Rolldown"


def test_augments_read_in_english_are_named_in_chinese(real_sd):
    from tft_advisor.data.augments import AugmentData

    adv = RulesAdvisor(set_data=real_sd)
    adv.augments = AugmentData.load(18)
    zh = adv.augments.lookup("Latent Forge").name
    state = GameState(stage=StageRound.parse("2-1"), gold=10, level=4, hp=100, augment_choices=["Latent Forge", "Mystery Thing"])
    out = adv.advise(state, Analysis())
    assert out.headline == f"海克斯选 {zh}"
    assert out.augment.startswith(f"推荐 {zh}（Latent Forge，")
    assert any(a.text.startswith(f"海克斯选 {zh}（Latent Forge，") for a in out.actions)
    assert not any("）（" in a.text for a in out.actions)


def test_pve_round_wait_is_not_headlined_as_saving_to_50(real_sd):
    from tft_advisor.data.comps import load_comps
    from tft_advisor.engine.analyzer import Analyzer

    mech = load_mechanics()
    analyzer = Analyzer(real_sd, mech, load_comps(None, real_sd))
    state = GameState(stage=StageRound.parse("3-7"), gold=34, hp=30, level=6, xp_current=2)
    analysis = analyzer.analyze(state, {})
    assert analysis.econ.reason.startswith("野怪回合先不搜")
    out = RulesAdvisor(set_data=real_sd, mech=mech).advise(state, analysis)
    assert out.headline == "野怪回合先存着，4-1再搜"


def test_long_augment_action_drops_english_name_instead_of_cutting():
    from tft_advisor.data.augments import AugmentData
    from tft_advisor.data.setdata import SetData, bundled_snapshot

    sd = SetData.from_cdragon(bundled_snapshot("zh_cn"), bundled_snapshot("en_us"))
    adv = RulesAdvisor(set_data=sd)
    adv.augments = AugmentData.load(18)
    state = GameState(stage=StageRound.parse("4-2"), gold=30, level=7, hp=60,
                      augment_choices=["Epic Rolldown", "Frontline Foundation", "Magic Roll"])
    act = [a for a in adv.advise(state, Analysis()).actions if a.type == ActionType.AUGMENT][0]
    assert act.text.startswith("海克斯选 ") and len(act.text) <= 40
    assert act.text.endswith("）") or "（" not in act.text  # never cut inside the parentheses


# ---------------------------------------------------------------------------
# Regression tests (tft-domain review, round 4)
# ---------------------------------------------------------------------------


def test_completed_item_for_a_benched_holder_is_never_a_p1_equip(advisor):
    """A built item the planner gives to a bench unit: field it first (p2,
    never the headline); a holder the engine wants sold keeps the item."""
    board = [unit("Draven", 4, 2, ["A", "B", "C"], row=3, col=0), unit("Braum", 2, 1, ["D", "E", "F"], row=0, col=3)]
    state = GameState(stage=StageRound.parse("4-3"), gold=38, level=7, hp=70, item_bench=["Gargoyle Stoneplate"],
                      board=board, bench=[unit("Vi", 2)])
    built = ItemSuggestion(item="Gargoyle Stoneplate", components=[], holder="Vi", priority=1)
    adv = advisor.advise(state, Analysis(items=[built]))
    check_contract(adv)
    item = action(adv, ActionType.ITEM)
    assert item.text == "先把 Vi 上场再装 Gargoyle Stoneplate" and item.priority == 2
    assert "Gargoyle" not in adv.headline and all("装给" not in t for t in all_text(adv))
    # Only a unit the engine sells: keep the item.
    adv = advisor.advise(state, Analysis(items=[built], sell_candidates=["Vi"]))
    item = action(adv, ActionType.ITEM)
    assert item.priority == 2 and item.text.endswith("先留着") and "Vi" not in item.text
    assert adv.items == "Gargoyle Stoneplate 先留着" and "Gargoyle" not in adv.headline
    # A fielded copy with a free slot takes it right away (unchanged).
    fielded = state.model_copy(update={"board": board + [unit("Vi", 2, row=0, col=4)], "bench": []})
    adv = advisor.advise(fielded, Analysis(items=[built]))
    assert adv.headline == "把Gargoyle Stoneplate装给Vi"
    assert action(adv, ActionType.ITEM).text == "把 Gargoyle Stoneplate 装给 Vi"
    # The only fielded copy is full: the bench copy is the holder.
    full_vi = unit("Vi", 2, 1, ["X", "Y", "Z"], row=0, col=4)
    both = state.model_copy(update={"board": board + [full_vi]})
    assert action(advisor.advise(both, Analysis(items=[built])), ActionType.ITEM).text.startswith("先把 Vi 上场")


def test_real_item_plan_never_equips_off_board_at_p1(real_sd):
    """The reported 5-3 state: a full board, a full bench and an AD item on
    the item bench of an AP comp."""
    from tft_advisor.data.comps import load_comps
    from tft_advisor.engine.analyzer import Analyzer
    from tft_advisor.models import Unit as U

    mech = load_mechanics()

    def u(name, star=1, items=(), row=None, col=None):
        c = real_sd.resolve_champion(name)
        return U(api_name=c.api_name, name=c.name, cost=c.cost, star=star, items=list(items), row=row, col=col,
                 traits=list(c.traits))

    board = [u("维迦", 2, ["珠光护手", "蓝霸符"], 3, 0), u("提莫", 1, [], 3, 1), u("费德提克", 1, [], 2, 2),
             u("索拉卡", 1, [], 3, 3), u("拉莫斯", 1, ["石像鬼石板甲"], 0, 3), u("雷克塞", 1, [], 0, 2),
             u("可酷伯", 1, [], 0, 4), u("纳尔", 1, [], 0, 5)]
    bench = [u(n) for n in ("绯红印记树怪", "阿利斯塔", "拉克丝", "约里克", "慎", "卡蜜尔", "阿卡丽", "暗影狼", "韦鲁斯")]
    state = GameState(stage=StageRound.parse("5-3"), gold=30, level=8, hp=55, board=board, bench=bench, item_bench=["海妖之怒"])
    analysis = Analyzer(real_sd, mech, load_comps(None, real_sd)).analyze(state, {})
    out = RulesAdvisor(mech=mech, set_data=real_sd).advise(state, analysis)
    check_contract(out)
    fielded = {x.name for x in board}
    for a in out.actions:
        if a.type == ActionType.ITEM and "装给" in a.text:
            assert a.text.split("装给", 1)[1].strip() in fielded, a.text
    if "装给" in out.headline:
        assert out.headline.split("装给", 1)[1] in fielded, out.headline
    for name in analysis.sell_candidates:
        assert all(f"装给 {name}" not in t and f"装给{name}" not in t for t in all_text(out))


def test_behind_the_curve_in_stage_4_saves_for_the_level_not_for_50(advisor, mech):
    """Stage 4+, below the line's level and the level not affordable yet: the
    gold is for the catch-up level (the econ buys it as soon as it can)."""
    state = GameState(stage=StageRound.parse("4-1"), gold=14, level=6, xp_current=14, hp=72)
    econ = plan_economy(state, mech, "fast8")
    assert econ.recommendation == EconAction.SAVE
    adv = advisor.advise(state, Analysis(econ=econ))
    check_contract(adv)
    assert adv.headline == "攒够24金币升7级"
    assert adv.plan.startswith("攒够 24 金币就升 7，尽快补到 8 级") and "先存到" not in adv.plan
    # The number is exactly where the econ engine levels (it keeps 10 gold
    # when the catch-up is not urgent: 4-3 at the standard level 7).
    for sr, lvl, xp, gold, style in (("4-1", 6, 14, 14, "fast8"), ("4-3", 7, 0, 30, "fast8"), ("5-1", 7, 10, 20, "standard")):
        st = GameState(stage=StageRound.parse(sr), gold=gold, level=lvl, xp_current=xp, hp=72)
        e = plan_economy(st, mech, style)
        head = advisor.advise(st, Analysis(econ=e)).headline
        assert head.startswith("攒够") and head.endswith(f"升{lvl + 1}级"), head
        need = int(head[2:head.index("金币")])
        levels = (EconAction.LEVEL, EconAction.LEVEL_AND_ROLL)
        assert plan_economy(st.model_copy(update={"gold": need - 1}), mech, style).recommendation not in levels
        assert plan_economy(st.model_copy(update={"gold": need}), mech, style).recommendation in levels
    # On schedule, or in stage 3: the save to 50 is unchanged.
    on_time = GameState(stage=StageRound.parse("4-3"), gold=30, level=7, hp=72)
    std = plan_economy(on_time, mech, "standard")
    adv = advisor.advise(on_time, Analysis(econ=std))
    assert adv.headline == "存钱到50" and adv.plan.startswith("先存到 50")
    early = GameState(stage=StageRound.parse("3-5"), gold=30, level=5, hp=72)
    adv = advisor.advise(early, Analysis(econ=EconPlan(recommendation=EconAction.SAVE)))
    assert adv.headline == "存钱到50" and adv.plan.startswith("先存到 50")


def test_team_size_items_and_emblems_do_not_make_a_carry(advisor, real_sd):
    settt = unit("Sett", 2, 2, ["斗士纹章", "金铲铲冠冕"], row=0, col=3)
    karma = unit("Karma", 2, 2, ["无尽之刃"], row=3, col=3)
    state = GameState(stage=StageRound.parse("3-3"), gold=30, level=7, hp=70, board=[settt, karma])
    for sd in (real_sd, None):
        assert find_carry(state, Analysis(), sd) is None
    english = settt.model_copy(update={"items": ["Brawler Emblem", "Tactician's Crown"]})
    assert find_carry(GameState(board=[english, karma]), Analysis()) is None
    krug = unit("Kobuko", 2, 1, ["金锅铲冠冕", "冕卫"], row=0, col=0)
    assert find_carry(GameState(board=[krug]), Analysis(), real_sd) is None
    assert find_carry(GameState(board=[krug]), Analysis()) is None
    # Flex damage items still make a carry.
    yunara = unit("Yunara", 2, 2, ["鬼索的狂暴之刃", "正义之手"], row=3, col=6)
    assert find_carry(GameState(board=[settt, yunara]), Analysis(), real_sd).name == "Yunara"
    adv = RulesAdvisor(set_data=real_sd).advise(state, Analysis())
    assert "Sett" not in (adv.positioning or "")


def test_wisp_hint_needs_this_rounds_shop_and_spare_gold(advisor):
    from tft_advisor.models import ShopSlot

    shop = [ShopSlot(name="Vi", cost=3), ShopSlot(name="Garen", cost=1), ShopSlot(), ShopSlot(), ShopSlot(name="Wisp: Golden Wisp", cost=3)]
    base = GameState(stage=StageRound.parse("3-3"), gold=20, level=6, hp=80, shop=shop,
                     field_age={"shop": 100.0, "round": 90.0})

    def hint(state, analysis=None):
        return [a.text for a in advisor.advise(state, analysis or Analysis()).actions if "精灵" in a.text]

    assert hint(base) == ["第5格精灵「Golden Wisp」（3金币）：前期经济或经验类值得买"]
    # Last round's shop (read before this round started), or the augment screen.
    stale = base.model_copy(update={"stage": StageRound.parse("3-5"), "field_age": {"shop": 100.0, "round": 200.0}})
    assert hint(stale) == []
    assert hint(base.model_copy(update={"screen_type": ScreenType.AUGMENT_SELECT})) == []
    # The plan already spends the gold: all-in rolls, or buys that leave 2.
    all_in = EconPlan(recommendation=EconAction.ALL_IN, roll_budget=20)
    assert hint(base.model_copy(update={"hp": 20}), Analysis(econ=all_in)) == []
    assert hint(base, Analysis(shop_picks=["Vi"], shop_picks_cost=18)) == []
    assert hint(base, Analysis(shop_picks=["Vi"], shop_picks_cost=17)) != []
    # An XP Wisp at the max level is worthless.
    xp = [*shop[:4], ShopSlot(name="精灵：经验精灵", cost=2)]
    top = GameState(stage=StageRound.parse("5-3"), gold=40, level=10, hp=60, shop=xp)
    assert hint(top) == []
    assert hint(top.model_copy(update={"level": 9})) != []


def test_pve_wait_plan_never_says_save_to_50(real_sd):
    from tft_advisor.data.comps import load_comps
    from tft_advisor.engine.analyzer import Analyzer

    mech = load_mechanics()
    analyzer = Analyzer(real_sd, mech, load_comps(None, real_sd))
    rules = RulesAdvisor(set_data=real_sd, mech=mech)
    for sr, hp, gold, level in (("3-7", 15, 12, 6), ("3-7", 30, 34, 6), ("4-7", 15, 30, 7), ("4-7", 35, 30, 7)):
        state = GameState(stage=StageRound.parse(sr), gold=gold, hp=hp, level=level, xp_current=2)
        analysis = analyzer.analyze(state, {})
        assert analysis.econ.recommendation == EconAction.SAVE
        out = rules.advise(state, analysis)
        check_contract(out)
        nxt = f"{int(sr[0]) + 1}-1"
        assert out.headline == f"野怪回合先存着，{nxt}再搜"
        assert "先存到" not in out.plan and out.plan.startswith("野怪回合先存着"), out.plan
        assert f"{nxt} " in out.plan and "搜牌稳血" in out.plan
    # A plain save on a PvE round at low HP (stage 2): no "save to 50" either.
    state = GameState(stage=StageRound.parse("2-7"), gold=20, hp=30, level=5)
    out = rules.advise(state, Analysis(econ=EconPlan(recommendation=EconAction.SAVE, reason="存钱吃利息")))
    assert "先存到" not in out.plan and out.headline == "野怪回合先存着，3-1再搜"
    # Healthy PvE save: unchanged.
    state = GameState(stage=StageRound.parse("3-7"), gold=30, hp=80, level=6)
    out = rules.advise(state, Analysis(econ=EconPlan(recommendation=EconAction.SAVE, reason="存钱吃利息")))
    assert out.headline == "存钱到50" and out.plan.startswith("先存到 50")


def test_buy_action_names_every_shop_pick(advisor):
    """Budget, roll budget and odds count every pick: the action and the
    headline must not hide any of them."""
    names = ["Varus", "Leona", "Karma", "Akali", "Camille"]
    shop = [unit(n, 1) for n in names]
    state = GameState(stage=StageRound.parse("2-3"), gold=30, level=4, hp=80,
                      bench=[unit(n, 1) for n in names] + [unit(n, 1) for n in names[:4]], shop_units=shop)
    econ = EconPlan(recommendation=EconAction.SAVE, reason="存钱吃利息")
    adv = advisor.advise(state, Analysis(econ=econ, shop_picks=names, shop_picks_cost=5))
    check_contract(adv)
    buy = action(adv, ActionType.BUY)
    assert buy.priority == 1 and buy.text.endswith("能升星")
    assert all(n in buy.text for n in names) or "第1、2、3、4、5格" in buy.text or "等 5 张" in buy.text, buy.text
    assert all(n in adv.headline for n in names) or "等5张" in adv.headline or "买5张牌" in adv.headline
    # Chinese names fit with every name and no slot label is lost for three picks.
    zh = ["韦鲁斯", "蕾欧娜", "卡尔玛", "阿卡丽", "卡蜜尔"]
    zstate = state.model_copy(update={"shop_units": [unit(n, 1) for n in zh], "bench": []})
    buy = action(advisor.advise(zstate, Analysis(econ=econ, shop_picks=zh)), ActionType.BUY).text
    assert all(n in buy for n in zh), buy
    three = advisor.advise(zstate, Analysis(econ=econ, shop_picks=zh[:3]))
    assert action(three, ActionType.BUY).text == "买 韦鲁斯（第1格）、蕾欧娜（第2格）、卡尔玛（第3格）"


def test_fill_empty_slots_leads_the_headline_and_is_not_a_footnote(advisor, mech):
    """The analyzer buys units for empty team slots: the headline says so and
    the slot warning is a priority 2 action, not the priority 3 tail."""
    shop = [ShopSlot(name="Garen", cost=1), ShopSlot(name="Ashe", cost=2)]
    state = GameState(stage=StageRound.parse("3-3"), gold=30, level=6, hp=60, shop=shop)
    warn = "场上只有 4 个英雄，还有 2 个空位，买英雄补满"
    econ = plan_economy(state, mech, "standard")
    adv = advisor.advise(state, Analysis(econ=econ, shop_picks=["Garen", "Ashe"], shop_picks_cost=3, warnings=[warn]))
    check_contract(adv)
    assert adv.headline == "买Garen、Ashe补满空位"
    assert (warn, 2) in [(a.text, a.priority) for a in adv.actions]


def test_gold_above_the_cap_headline_names_the_xp_not_saving(advisor, mech):
    state = GameState(stage=StageRound.parse("3-3"), gold=62, level=6, xp_current=0, hp=60)
    econ = plan_economy(state, mech, "standard")
    assert econ.recommendation == EconAction.SAVE and "次经验" in econ.reason
    assert advisor.advise(state, Analysis(econ=econ)).headline == "利息已满，多的钱买3次经验"
    shop = state.model_copy(update={"shop": [ShopSlot(name="Garen", cost=1)]})
    adv = advisor.advise(shop, Analysis(econ=econ, shop_picks=["Garen"], shop_picks_cost=1))
    assert adv.headline == "买Garen，多的钱买经验"


def test_level_plan_names_the_round_a_two_star_carry_goes_nine(advisor, mech):
    comp = CompSuggestion(name="金克丝", carry="Jinx", score=0.9, style="fast8", core_units=["Jinx"])
    for star, nine in ((1, "5-5 升 9"), (2, "5-1 升 9")):
        jinx = unit("Jinx", 4, star, [], row=3, col=3)
        state = GameState(stage=StageRound.parse("4-5"), gold=40, level=8, xp_current=10, hp=60, board=[jinx])
        econ = plan_economy(state, mech, "fast8", key_star=star, carry_cost=4)
        adv = advisor.advise(state, Analysis(econ=econ, comps=[comp]))
        assert nine in adv.plan, (star, adv.plan)
        if star == 2:
            assert "5-1" in econ.reason
