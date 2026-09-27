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
    assert buy.text.endswith("能升星") and "Morgana（第1格）" in buy.text and len(buy.text) <= 40


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
