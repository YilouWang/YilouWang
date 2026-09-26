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
        comps=[CompSuggestion(name="决斗大师", score=0.8, carry="Draven", missing_units=["Aatrox", "Shen"])],
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
    assert adv.augment and "Rich Get Richer" in adv.augment and "Claude" in adv.augment
    assert "增强" in adv.headline


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
