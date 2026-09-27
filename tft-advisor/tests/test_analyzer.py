from __future__ import annotations

import json

import pytest

from tft_advisor.data.comps import load_comps
from tft_advisor.engine.analyzer import Analyzer
from tft_advisor.models import Analysis, EconAction, EconPlan, GameState, ScreenType, ShopSlot, StageRound, Unit

from .conftest import make_state


@pytest.fixture(scope="module")
def comps(set_data):
    return load_comps(None, set_data)


@pytest.fixture()
def analyzer(set_data, mech, comps):
    return Analyzer(set_data, mech, comps)


def all_strings(analysis: Analysis) -> list[str]:
    out: list[str] = []

    def walk(x):
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(analysis.model_dump(mode="json"))
    return out


def no_dash(analysis: Analysis) -> None:
    for s in all_strings(analysis):
        assert "—" not in s, s


def mid_game_state() -> GameState:
    return make_state(
        "3-2",
        board=[
            ("Graves", 2, (), 3, 1),
            ("Lucian", 1, (), 3, 2),
            ("Pyke", 1, (), 2, 3),
            ("Garen", 2, (), 0, 3),
            ("Braum", 1, (), 0, 4),
        ],
        bench=["Graves", "Graves", "Warwick", "Katarina"],
        shop_names=["Graves", "Lucian", "Ahri", "Katarina", "Braum"],
        item_bench=["B.F. Sword", "Sparring Gloves", "Recurve Bow", "Chain Vest"],
        gold=32,
        level=5,
        xp_current=2,
        xp_needed=20,
        hp=60,
        streak=-2,
        self_name="Me",
    )


# ---------------------------------------------------------------------------
# Sparse / empty states never raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state",
    [
        GameState(),
        GameState(stage=StageRound(stage=1, round=1)),
        GameState(stage=StageRound(stage=4, round=2), gold=None, level=None, hp=None),
        GameState(shop=[ShopSlot(name=None), ShopSlot(name="Nobody", cost=9)]),
        GameState(board=[Unit(api_name="?x", name="x")], bench=[Unit(api_name="?y", name="y", star=0)], item_bench=["?", ""]),
        GameState(shop=[ShopSlot(name="Graves", cost=1)], shop_units=[]),
        GameState(stage=StageRound(stage=6, round=5), gold=300, level=10, hp=1, streak=-20),
    ],
)
def test_sparse_states_never_raise(set_data, mech, comps, state):
    for c in (comps, []):
        an = Analyzer(set_data, mech, c)
        out = an.analyze(state)
        assert isinstance(out, Analysis)
        assert an.last_errors == [], an.last_errors
        assert out.scout_requests == []
        no_dash(out)


def test_empty_state_is_sensible(analyzer):
    out = analyzer.analyze(GameState())
    assert out.stage is None
    assert out.econ.recommendation == EconAction.HOLD
    assert out.odds == [] and out.items == [] and out.shop_picks == [] and out.sell_candidates == []
    assert out.warnings == []
    assert [c.name for c in out.comps][:1] == ["冰川游侠"]  # tier S when nothing is owned


def test_broken_section_is_contained(set_data, mech, comps, monkeypatch):
    import tft_advisor.engine.analyzer as mod

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "plan_items", boom)
    an = Analyzer(set_data, mech, comps)
    out = an.analyze(mid_game_state())
    assert out.items == [] and out.comps  # the rest still works
    assert an.last_errors and "items" in an.last_errors[0]
    assert "部分分析出错" in out.warnings[-1]


# ---------------------------------------------------------------------------
# Realistic mid game
# ---------------------------------------------------------------------------


def test_mid_game_analysis(analyzer):
    st = mid_game_state()
    taken = {"Alice": {"TFT99_Graves": 3, "TFT99_MissFortune": 1}, "Me": {"TFT99_Graves": 99}}
    out = analyzer.analyze(st, taken)
    assert analyzer.last_errors == []
    assert out.stage == "3-2"

    # Comp: gunslinger pirates, style fast8 -> standard level timing at 3-2
    # (3-2 keeps ~20 gold, so 32 gold only pays for the level).
    assert out.comps[0].name == "枪手海盗"
    assert out.econ.recommendation in (EconAction.LEVEL, EconAction.LEVEL_AND_ROLL)
    assert out.econ.target_level == 6
    assert out.econ.roll_budget <= max(0, 32 - 20 - 20)

    # Odds: carry first when the shop can show it (5 cost: not at level 5), then comp units
    # closest to their next star; capped at 8; the player's own "Me" entry in taken is ignored.
    assert 0 < len(out.odds) <= 8
    assert out.odds[0].unit in ("Lucian", "Pyke", "Braum") and out.odds[0].goal_star == 2
    assert "Miss Fortune" not in [o.unit for o in out.odds]
    lvl7 = mid_game_state()
    lvl7.level = 7
    assert analyzer.analyze(lvl7, taken).odds[0].unit == "Miss Fortune"
    graves = next(o for o in out.odds if o.unit == "Graves")
    # 5 owned plus the shop copy the buy action takes (it completes the second 2-star).
    assert "Graves" in out.shop_picks
    assert graves.seen_elsewhere == 3 and graves.owned_copies == 6
    assert graves.remaining_in_pool == analyzer.mech.pool_size[1] - 6 - 3
    units = [o.unit for o in out.odds]
    # A 2-star upgrade of a comp unit ranks before a 3-star chase of a 1 cost (still listed:
    # cheap 3-stars are realistic at level 5).
    assert units.index("Lucian") < units.index("Graves")
    assert "Kayle" not in units  # 5 cost unreachable at level 5

    # Shop: Graves completes a second 2-star, Lucian/Braum are owned comp units.
    assert out.shop_picks[0] == "Graves"
    assert set(out.shop_picks[1:3]) == {"Lucian", "Braum"}
    assert "Ahri" not in out.shop_picks
    assert sum({"Graves": 1, "Lucian": 2, "Braum": 2, "Katarina": 3}.get(p, 0) for p in out.shop_picks) <= 32

    # Items: slam at 3-2 (2+ components), AD item to a gunslinger.
    assert out.items and all(s.priority == 1 for s in out.items)
    ad = next(s for s in out.items if s.item in ("Infinity Edge", "Giant Slayer", "Last Whisper"))
    assert ad.holder in ("Graves", "Lucian")

    # Bench has 4/9 at stage 3: nothing to sell.
    assert out.sell_candidates == []
    assert any("等级落后" in w for w in out.warnings)
    no_dash(out)


def test_style_from_top_comp_changes_econ(set_data, mech, comps):
    # Ashe reroll3 comp fits: level 7 at 4-1 with 64 gold -> slow roll above 50.
    st = make_state(
        "4-3",
        board=[("Ashe", 2, (), 3, 1), ("Braum", 2, (), 0, 2), ("Volibear", 2, (), 0, 3), "Vayne", "Warwick", "Gnar", "Aatrox"],
        gold=64,
        level=7,
        hp=70,
    )
    out = Analyzer(set_data, mech, comps).analyze(st)
    assert out.comps[0].name == "冰川游侠"
    assert out.econ.recommendation == EconAction.SLOW_ROLL
    assert out.econ.roll_budget == 14


def test_comp_hint_boosts_and_sets_style(set_data, mech, comps):
    st = make_state("3-5", board=["Graves", "Lucian", "Pyke", "Ahri"], gold=55, level=6, hp=80)
    plain = Analyzer(set_data, mech, comps).analyze(st)
    an = Analyzer(set_data, mech, comps, comp_hint="狂野法师")
    hinted = an.analyze(st)
    assert plain.comps[0].name == "枪手海盗"
    assert hinted.comps[0].name == "狂野法师"
    assert hinted.econ.recommendation in (EconAction.SLOW_ROLL, EconAction.SAVE)  # reroll2 style at level 6
    an.set_comp_hint("")
    assert an.analyze(st).comps[0].name == "枪手海盗"


def test_auto_comps_used_without_comp_defs(set_data, mech):
    st = make_state("2-5", board=["Graves", "Lucian", "Garen"], level=5, gold=20)
    out = Analyzer(set_data, mech, []).analyze(st)
    assert out.comps and out.comps[0].name.split()[0] in ("Noble", "Knight", "Pirate")


# ---------------------------------------------------------------------------
# Shop picks, sell candidates, warnings
# ---------------------------------------------------------------------------


def test_shop_picks_three_star_and_budget(set_data, mech):
    st = make_state(
        "4-1",
        board=[("Graves", 2), ("Graves", 2), "Graves", "Ahri"],
        bench=["Graves"],
        shop_names=["Graves", "Kayle", "Ahri", None, "Garen"],
        gold=3,
        level=7,
    )
    out = Analyzer(set_data, mech, []).analyze(st)
    # Graves: 2 two-stars + 2 one-stars + 1 in shop = 3-star; Ahri pair; budget 3 gold.
    assert out.shop_picks[0] == "Graves"
    assert "Kayle" not in out.shop_picks  # 5 gold > budget
    assert sum({"Graves": 1, "Ahri": 2, "Garen": 1}[p] for p in out.shop_picks) <= 3


def test_shop_picks_without_resolved_shop_units(set_data, mech):
    st = make_state("2-2", board=["Graves"], gold=10)
    st.shop = [ShopSlot(name="Graves", cost=1), ShopSlot(name="Graves", cost=1), ShopSlot(name="Zzz", cost=2)]
    st.shop_units = []
    out = Analyzer(set_data, mech, []).analyze(st)
    assert out.shop_picks == ["Graves", "Graves"]


def gun_only(comps):
    return [c for c in comps if c.name == "枪手海盗"]


def test_sell_candidates_when_bench_full(set_data, mech, comps):
    bench = ["Katarina", "Ahri", "Ahri", "Darius", "Shen", "Akali", "Ashe", "Morgana", "Zzz"]
    st = make_state("3-3", board=["Graves", "Lucian", "Pyke", ("Garen", 2), "Braum"], bench=bench, level=5, gold=10)
    out = Analyzer(set_data, mech, gun_only(comps)).analyze(st)
    assert out.comps[0].name == "枪手海盗"
    sells = out.sell_candidates
    assert "Ahri" not in sells  # pair
    assert "Zzz" not in sells  # unresolved: never suggested
    assert {"Katarina", "Darius", "Shen", "Akali", "Ashe", "Morgana"} == set(sells)
    assert sells[0] == "Darius"  # cheapest first
    assert any("备战席已满" in w for w in out.warnings)
    assert any("Zzz" in w for w in out.warnings)


def test_no_sell_early_with_room(set_data, mech, comps):
    an = Analyzer(set_data, mech, gun_only(comps))
    st = make_state("3-3", board=["Graves"], bench=["Katarina", "Darius", "Lucian"], level=5)
    assert an.analyze(st).sell_candidates == []
    late = make_state("4-3", board=["Graves"], bench=["Katarina", "Darius", "Lucian"], level=7)
    assert set(an.analyze(late).sell_candidates) == {"Katarina", "Darius"}  # Lucian is a comp unit


def test_warnings(set_data, mech, comps):
    an = Analyzer(set_data, mech, comps)
    st = make_state("4-2", board=["Graves", "Lucian"], bench=["Pyke"], item_bench=["B.F. Sword"], gold=80, level=7, hp=20)
    w = " | ".join(an.analyze(st).warnings)
    assert "场上只有 2 个英雄" in w
    assert "散件" in w
    assert "金币 80" not in w  # the critical-HP all-in already spends it
    assert "血量只剩 20" in w
    assert "等级落后" not in w  # 4-1 standard is 7
    # Too much gold is flagged only when the plan leaves it unspent.
    from tft_advisor.models import EconPlan

    saving = EconPlan(gold=80, recommendation=EconAction.SAVE)
    assert any("金币 80" in x for x in an._warnings(st, [], "standard", saving))
    rolling = EconPlan(gold=80, recommendation=EconAction.ROLL, roll_budget=30)
    assert not any("金币 80" in x for x in an._warnings(st, [], "standard", rolling))

    behind = make_state("4-2", board=["Graves"] * 6, gold=10, level=6, hp=80)
    assert any("等级落后" in x for x in an.analyze(behind).warnings)

    low = make_state("3-2", board=["Graves"], hp=40, level=1)
    assert any("偏低" in x for x in an.analyze(low).warnings)

    loose = make_state("3-2", board=["Graves"], item_bench=["Infinity Edge"], level=1)
    assert any("成装还没装备" in x for x in an.analyze(loose).warnings)

    # An item nobody on the team can use is held on purpose: no "equip it" nag.
    unusable = make_state("3-2", board=["Graves"], item_bench=["Rabadon's Deathcap"], level=1)
    result = an.analyze(unusable)
    assert not any("成装还没装备" in x for x in result.warnings)
    assert any("先留着" in s.reason and s.holder is None for s in result.items)
    assert not any("换下一件" in s.reason for s in result.items)


def test_board_warning_needs_a_seen_board(set_data, mech, comps):
    st = make_state("3-2", level=6, gold=10, hp=80)
    assert "board" not in st.field_age
    assert not any("场上只有" in w for w in Analyzer(set_data, mech, comps).analyze(st).warnings)


def test_analysis_json_roundtrip(analyzer):
    out = analyzer.analyze(mid_game_state())
    data = json.loads(out.model_dump_json())
    assert Analysis.model_validate(data).comps[0].name == out.comps[0].name


def test_chinese_locale_end_to_end(zh_set_data, mech):
    from tft_advisor.engine.tracker import GameTracker

    from .conftest import obs, uo

    comps = load_comps(None, zh_set_data)
    assert comps and comps[0].units[0] == "厄运小姐"
    tracker = GameTracker(zh_set_data, mech)
    st = tracker.ingest(
        obs(
            viewing_own_board=True,
            stage="3-2",
            gold=30,
            level=6,
            hp=70,
            board=[uo("格雷福斯", 2, row=3, col=0), uo("卢锡安", row=3, col=1), uo("派克", row=2, col=2), uo("布隆", row=0, col=3)],
            bench=[uo("格雷福斯"), uo("格雷福斯")],
            shop=["格雷福斯", "阿狸", "Katarina"],
            item_bench=["暴风大剑", "拳套"],
        )
    )
    out = Analyzer(zh_set_data, mech, comps).analyze(st, tracker.taken_by_player())
    assert out.comps[0].name == "枪手海盗"
    assert out.comps[0].carry == "厄运小姐"
    assert out.shop_picks[0] == "格雷福斯"
    assert out.items[0].item == "无尽之刃"
    no_dash(out)
    auto = Analyzer(zh_set_data, mech, []).analyze(st)
    assert auto.comps and any(c.name.startswith(("枪手", "海盗", "贵族")) for c in auto.comps)
    no_dash(auto)


def test_garbage_taken_is_ignored(analyzer):
    out = analyzer.analyze(mid_game_state(), {"Bob": {"TFT99_Graves": None, "TFT99_Lucian": "x", "TFT99_Pyke": 2}, "": {}, "Eve": None})  # type: ignore[dict-item]
    assert analyzer.last_errors == []
    pyke = [o for o in out.odds if o.unit == "Pyke"]
    assert not pyke or pyke[0].seen_elsewhere == 2


def test_unknown_level_is_estimated_not_treated_as_one(set_data, mech, comps):
    st = make_state("5-1", board=["Graves", "Lucian", "Pyke", "Garen", "Braum", "Vayne", "Ahri", "Ashe"], gold=40, hp=60)
    assert st.level is None
    an = Analyzer(set_data, mech, comps)
    out = an.analyze(st)
    assert out.econ.target_level != 2 and "只差" not in out.econ.reason
    assert all(o.level == 8 for o in out.odds)  # 8 fielded units: at least level 8
    assert any("没识别到等级" in w and "8 级" in w for w in out.warnings)
    few = make_state("4-2", board=["Graves"], gold=150, hp=80)
    out = an.analyze(few)
    assert out.econ.target_level in (None, 8)  # estimated on the curve (7): no "level to 7" from level 1
    assert any("7 级" in w for w in out.warnings if "没识别到等级" in w)


def test_odds_skip_hopeless_three_star_chases_late(set_data, mech, comps):
    # Fast 8 at level 8 with MF 1-star: roll targets are MF / Kayle / Pyke, not a 3-star Lucian.
    st = make_state(
        "4-5",
        board=[("Garen", 2), ("Darius", 2), ("Braum", 2), "Kayle", ("Graves", 2), ("Vayne", 2), ("Lucian", 2), "Miss Fortune"],
        bench=[("Ashe", 2), "Garen", "Lucian"],
        gold=32, level=8, hp=58,
    )
    out = Analyzer(set_data, mech, comps).analyze(st)
    units = [o.unit for o in out.odds]
    assert units[0] == "Miss Fortune"
    assert units.index("Kayle") < units.index("Lucian") and units.index("Pyke") < units.index("Lucian")
    # A reroll line keeps its cheap 3-star targets high.
    rr = make_state("3-5", board=[("Ahri", 2), "Morgana", "Warwick", "Gnar"], bench=["Ahri"], gold=60, level=6, hp=70)
    out = Analyzer(set_data, mech, comps, comp_hint="狂野法师").analyze(rr)
    assert out.odds[0].unit == "Ahri" and out.odds[0].goal_star == 3


def test_two_star_legendary_carry_is_not_the_first_roll_target(set_data, mech):
    from tft_advisor.data.comps import CompDef

    comp = CompDef(name="MF", units=["Miss Fortune", "Graves", "Lucian", "Pyke"], carry="Miss Fortune", style="fast8", tier="A")
    st = make_state("5-1", board=[("Miss Fortune", 2), "Graves", "Lucian"], bench=["Graves"], gold=40, level=8, hp=70)
    out = Analyzer(set_data, mech, [comp]).analyze(st)
    assert out.odds[0].unit != "Miss Fortune"
    lvl9 = make_state("5-1", board=[("Miss Fortune", 2), "Graves", "Lucian"], gold=40, level=9, hp=70)
    assert Analyzer(set_data, mech, [comp]).analyze(lvl9).odds[0].unit != "Miss Fortune"  # 9 copies of a 5 cost
    # A 4 cost carry 3-star is a real level 9 plan.
    dcomp = CompDef(name="D", units=["Draven", "Katarina", "Darius"], carry="Draven", tier="A")
    d9 = make_state("5-1", board=[("Draven", 2), "Katarina", "Darius"], bench=["Draven"], gold=40, level=9, hp=70)
    first = Analyzer(set_data, mech, [dcomp]).analyze(d9).odds[0]
    assert first.unit == "Draven" and first.goal_star == 3


# ---------------------------------------------------------------------------
# Set 18 regressions (bundled snapshot + comp library)
# ---------------------------------------------------------------------------

from .conftest import s18_comps, s18_set_data, s18_state  # noqa: E402

VEIGAR_BOARD = [
    ("Veigar", 3, ("Jeweled Gauntlet", "Blue Buff", "Hextech Gunblade")),
    ("Ornn", 3),
    "Alistar",
    "Rek'Sai",
    "LeBlanc",
]


@pytest.fixture(scope="module")
def s18(mech):
    return Analyzer(s18_set_data(), mech, s18_comps())


def test_s18_reroll_carry_three_star_levels_up(s18):
    for stage, hp, gold in (("5-1", 70, 90), ("5-5", 40, 60), ("6-2", 30, 60)):
        st = s18_state(stage, board=VEIGAR_BOARD, level=5, gold=gold, hp=hp, xp_current=0)
        out = s18.analyze(st)
        assert out.comps[0].name == "永恒之森 维迦", stage
        assert out.econ.recommendation in (EconAction.LEVEL, EconAction.LEVEL_AND_ROLL), stage
        assert out.econ.target_level and out.econ.target_level > 5
        assert not any(w in out.econ.reason for w in ("慢搜", "赌狗", "三星")), out.econ.reason
        # Level behind the curve is reported again once the reroll is done.
        assert any("等级落后" in w for w in out.warnings)


def test_s18_reroll_style_needs_the_reroll_units_not_the_item_carry(s18):
    # Ezreal pair, no Kha'Zix: Lunarwood Kha'zix (3-cost reroll, Ezreal carry) is not a reroll yet.
    ez = s18_state(
        "5-1", board=["Ezreal", "Ornn", "Hecarim", "Fiddlesticks", "Diana", "Alistar", "LeBlanc"], bench=["Ezreal"],
        level=7, gold=60, hp=60, xp_current=0,
    )
    top = s18.analyze(ez).comps[0]
    assert s18._style_for(top, ez) == "standard"
    assert "三星" not in s18.analyze(ez).econ.reason
    # Sivir pair on a Hunter board: Caitlyn Hunters (Sivir carry) is not a 2-cost reroll.
    sivir = s18_state(
        "4-5", board=["Sivir", "Caitlyn", "Sejuani", "Scuttlecrab", "Rakan", "Cinderling"], bench=["Sivir"],
        level=6, gold=60, hp=60, xp_current=0,
    )
    out = s18.analyze(sivir)
    assert s18._style_for(out.comps[0], sivir) == "standard"
    assert out.econ.recommendation in (EconAction.LEVEL, EconAction.LEVEL_AND_ROLL)
    # A real Caitlyn reroll (5 copies) is one, although it holds no Sivir.
    cait = s18_state(
        "3-5", board=[("Caitlyn", 2), "Sejuani", "Scuttlecrab", "Rakan", "Cinderling", "Sivir"],
        bench=["Caitlyn", "Caitlyn"], level=6, gold=60, hp=70, xp_current=0,
    )
    out = s18.analyze(cait)
    assert out.comps[0].name == "凯特琳 猎人" and s18._style_for(out.comps[0], cait) == "reroll2"
    assert out.econ.recommendation == EconAction.SLOW_ROLL


def test_s18_level_and_roll_odds_use_the_new_level(s18):
    st = s18_state(
        "4-2", board=["Ahri", "Morgana", "Karma", "Yorick", "Sett", "Leona", "Cassiopeia"], bench=["Ahri"],
        level=7, xp_current=40, gold=60, hp=55,
    )
    out = s18.analyze(st)
    assert out.econ.recommendation == EconAction.LEVEL_AND_ROLL and out.econ.target_level == 8
    assert out.odds and all(o.level == 8 for o in out.odds)
    ahri = next(o for o in out.odds if o.unit == "阿狸")
    assert ahri.p_goal_by_gold[30] > 0.5


def test_s18_fast8_two_star_board_does_not_roll_every_round(s18):
    comp = next(c for c in s18_comps() if c.name_en == "Ahri Morgana")
    for stage, gold in (("4-5", 60), ("5-2", 70), ("5-5", 64)):
        st = s18_state(stage, board=[(u, 2) for u in comp.units][:8], level=8, gold=gold, hp=82, xp_current=0, streak=6)
        out = s18.analyze(st)
        assert out.econ.roll_budget == 0, (stage, out.econ.reason)
    st = s18_state("5-1", board=[(u, 2) for u in comp.units][:8], level=8, gold=90, hp=82, xp_current=0, streak=6)
    out = s18.analyze(st)
    assert out.econ.recommendation == EconAction.LEVEL and out.econ.target_level == 9


def test_s18_win_streak_roll_without_a_target_is_dropped(s18):
    board = [(n, 2) for n in ("Yorick", "Rakan", "Leona", "Karma", "Caitlyn", "Sejuani", "Ahri")]
    win = s18.analyze(s18_state("4-2", board=board, level=7, gold=36, hp=64, streak=2, xp_current=0))
    assert win.econ.recommendation == EconAction.SAVE and win.econ.roll_budget == 0
    assert "连胜" in win.econ.reason
    # Without a streak a medium-HP roll-down with nothing to find is dropped too.
    plain = s18.analyze(s18_state("4-2", board=board, level=7, gold=36, hp=64, streak=-1, xp_current=0))
    assert plain.econ.recommendation == EconAction.SAVE and plain.econ.roll_budget == 0
    assert "连胜" not in plain.econ.reason
    # Low HP keeps its roll.
    low = s18.analyze(s18_state("4-2", board=board, level=7, gold=36, hp=40, streak=-1, xp_current=0))
    assert low.econ.recommendation == EconAction.ROLL


def test_s18_elder_dragon_takes_two_team_slots(s18):
    comp = next(c for c in s18_comps() if c.name_en == "Dragon 9")
    board = [(u, 2) for u in comp.units][:8]
    assert "远古巨龙" in [u if isinstance(u, str) else u[0] for u in board]
    st = s18_state("5-3", board=board, level=9, gold=40, hp=60, xp_current=0)
    assert not any("空位" in w or "还能再上" in w for w in s18.analyze(st).warnings)
    unread = s18_state("5-3", board=board, gold=40, hp=60)
    assert s18.estimate_level(unread) == 9


def test_s18_unmatched_comp_hint_warns(mech):
    st = s18_state("3-2", board=["Yorick", "Rakan", "Leona", "Karma", "Caitlyn"], level=6, gold=30, hp=70)
    warned = Analyzer(s18_set_data(), mech, s18_comps(), comp_hint="枪手").analyze(st)
    assert "没有阵容匹配「枪手」，按自动推荐" in warned.warnings
    ok = Analyzer(s18_set_data(), mech, s18_comps(), comp_hint="法师").analyze(st)
    assert not any("没有阵容匹配" in w for w in ok.warnings)
    no_dash(warned)


# ---------------------------------------------------------------------------
# Regressions (domain review): 5 cost carries, critical HP levels, shop picks
# inside the budget, rolls with nothing to find
# ---------------------------------------------------------------------------


def _names(*apis: str) -> list[str]:
    sd = s18_set_data()
    return [sd.champions[a].name for a in apis]


VANGUARDS = _names("DA_18_Diana", "DA_18_Hecarim", "DA_CrimsonRaptor18", "DA_18_Zyra", "DA_Brambleback18", "DA_Sentinel18", "DA_Taric18")


def test_s18_five_cost_carry_goes_nine_instead_of_rolling_at_eight(s18, mech):
    assert next(c for c in s18_comps() if c.name_en == "Inferno Fast 9").style == "fast9"
    board = ["Ashe"] + _names("DA_18_Rakan", "DA_Vi18", "DA_Amumu18", "DA_18_Sivir", "DA_18_Ivern", "DA_18_Maokai", "DA_Taric18")
    for stage, gold in (("5-1", 80), ("5-1", 100), ("5-3", 90)):
        out = s18.analyze(s18_state(stage, board=board, level=8, xp_current=10, hp=80, streak=0, gold=gold))
        assert out.comps[0].carry == "艾希"
        assert out.econ.recommendation == EconAction.LEVEL and out.econ.target_level == 9, (stage, out.econ.reason)
        assert not any("太多了" in w for w in out.warnings)
    # Exactly the price of level 9 at 5-2 (hinted Spirit Blossom): level, no roll at 8.
    blossom = _names("DA_Karma18", "DA_18_Yorick", "DA_Vi18", "DA_18_Sivir", "DA_18_Zyra", "DA_18_Ahri", "DA_18_Sett", "DA_18_GnarSmall")
    an = Analyzer(s18_set_data(), mech, s18_comps(), comp_hint="Spirit Blossom")
    need = mech.gold_to_reach(8, 10, 9)
    out = an.analyze(s18_state("5-2", board=blossom, level=8, xp_current=10, hp=80, gold=need))
    assert out.econ.recommendation == EconAction.LEVEL and out.econ.target_level == 9, out.econ.reason


def test_s18_ready_carry_with_spare_gold_levels_or_buys_xp(s18):
    board = [("Aphelios", 2)] + VANGUARDS
    out = s18.analyze(s18_state("4-5", board=board, level=8, xp_current=10, hp=80, streak=0, gold=110))
    assert out.econ.recommendation == EconAction.LEVEL and out.econ.target_level == 9
    out = s18.analyze(s18_state("4-5", board=board, level=8, xp_current=10, hp=80, streak=0, gold=80))
    assert out.econ.recommendation == EconAction.SAVE and "买经验" in out.econ.reason


def test_s18_critical_hp_levels_when_the_odds_hold(s18):
    # 4-5, level 7 (behind), 100 gold, 18 HP: level 8 costs 56 and the carry
    # odds are better there (plus one more unit on the board).
    st = s18_state("4-5", board=["Aphelios"] + VANGUARDS[:6], level=7, xp_current=0, hp=18, gold=100)
    out = s18.analyze(st)
    assert out.econ.recommendation == EconAction.LEVEL_AND_ROLL and out.econ.target_level == 8
    assert out.econ.roll_budget == 100 - 56 - out.shop_picks_cost
    assert all(o.level == 8 for o in out.odds)
    # 60 gold: the level would leave 4 gold for rolls; stay all-in at 7.
    out = s18.analyze(st.model_copy(update={"gold": 60}))
    assert out.econ.recommendation == EconAction.ALL_IN


def _with_shop(st: GameState, names: list[str]) -> GameState:
    sd = s18_set_data()
    champs = [sd.resolve_champion(n) for n in names]
    st.shop = [ShopSlot(name=c.name, cost=c.cost) for c in champs]
    st.shop_units = [Unit(api_name=c.api_name, name=c.name, cost=c.cost, traits=list(c.traits)) for c in champs]
    return st


def test_s18_shop_picks_fit_the_plan_and_count_in_the_odds(s18, mech):
    comp = next(c for c in s18_comps() if c.name_en == "Ahri Morgana")
    board = [u for u in comp.units if u != "阿狸"][:6] + ["Ahri"]
    st = _with_shop(s18_state("4-5", board=board, level=8, hp=40, gold=22), ["Ahri", "Karma", "Ahri", "Taric", "Leona"])
    out = s18.analyze(st)
    # Buying both Ahri copies makes her 2-star: not a roll target any more.
    assert out.shop_picks.count("阿狸") == 2
    ahri = next((o for o in out.odds if o.unit == "阿狸"), None)
    assert ahri is None or ahri.goal_star == 3
    spend = 0
    if out.econ.recommendation in (EconAction.LEVEL, EconAction.LEVEL_AND_ROLL):
        spend = mech.gold_to_reach(8, 0, out.econ.target_level)
    assert spend + out.econ.roll_budget + out.shop_picks_cost <= 22
    # A save plan at 50 does not buy filler units below the interest step.
    filler = _with_shop(
        s18_state("4-3", board=[("Aphelios", 2)] + VANGUARDS, level=8, hp=80, gold=50, streak=0),
        ["Amumu", "Varus", "Shen", "Yorick", "Leona"],
    )
    out = s18.analyze(filler)
    assert out.econ.recommendation == EconAction.SAVE
    assert 50 - out.shop_picks_cost >= 50, out.shop_picks


def test_s18_roll_down_with_nothing_to_find_is_dropped(s18, mech):
    # Fast 8 at 4-2, level 7, 40 gold, medium HP, board of 4-5 costs and
    # 2-star 3 costs: no target reaches 10%, so save for level 8.
    board = ["Aphelios", "Zyra", *_names("DA_Brambleback18", "DA_Sentinel18", "DA_Taric18"), ("Diana", 2), ("Hecarim", 2)]
    out = s18.analyze(s18_state("4-2", board=board, level=7, xp_current=10, hp=60, gold=40, streak=-1))
    assert out.econ.recommendation == EconAction.SAVE and out.econ.roll_budget == 0, out.econ.reason
    assert "升 8 级" in out.econ.reason
    # Reroll line at 4-1: 10 gold cannot find a 3-star (nor any upgrade), so
    # it saves for the slow roll. (With 34 gold the exact 14 gold budget gives
    # a 2-star Sejuani 14%, a real target: the odds are no longer rounded
    # down to the 10 gold bucket.)
    an = Analyzer(s18_set_data(), mech, s18_comps(), comp_hint="Caitlyn Hunters")
    st = s18_state("4-1", board=[("Caitlyn", 2), "Tristana", "Rakan", "Sejuani", "Vi", "Sivir"], bench=["Caitlyn"], level=6, hp=50, gold=30)
    out = an.analyze(st)
    assert out.econ.recommendation == EconAction.SAVE and "慢搜" in out.econ.reason


# ---------------------------------------------------------------------------
# Review round: shop / sell coherence, exact roll odds, PvE, stale shop, threads
# ---------------------------------------------------------------------------

APH_BOARD = ["Aphelios", "Nidalee", "Varus", "Diana", "Kog'Maw", "Vi"]


def test_s18_full_bench_buys_no_off_comp_copy_it_also_sells(s18):
    bench = ["Rakan", "Teemo", "Alistar", "Elise", "Caitlyn", "Sejuani", "Shen", "LeBlanc", "Yunara"]
    st = _with_shop(s18_state("3-3", board=APH_BOARD, bench=bench, level=6, gold=36, hp=70),
                    ["Rakan", "Gromp", "Kayle", "Murkwolf", "Karma"])
    out = s18.analyze(st)
    assert "洛" not in out.shop_picks and "洛" in out.sell_candidates
    assert not set(out.shop_picks) & set(out.sell_candidates)
    # One free bench slot: the second Rakan is bought, so it is not sold.
    st = _with_shop(s18_state("3-3", board=APH_BOARD, bench=bench[:8], level=6, gold=36, hp=70),
                    ["Rakan", "Gromp", "Kayle", "Murkwolf", "Karma"])
    out = s18.analyze(st)
    assert "洛" in out.shop_picks and "洛" not in out.sell_candidates


def test_s18_late_game_does_not_buy_random_trait_units(s18):
    board = [("Aphelios", 2), "Nidalee", "Varus", "Diana", "Kog'Maw", "Vi", "Amumu", "Sentinel"]
    st = _with_shop(s18_state("5-2", board=board, level=8, gold=30, hp=40), ["Shen", "Akali", "Kayle", "Leona", "Caitlyn"])
    out = s18.analyze(st)
    assert out.shop_picks == [] and out.shop_picks_cost == 0
    assert out.econ.roll_budget == 20  # the whole roll budget stays for rolls


def test_s18_roll_odds_use_the_exact_budget(s18):
    st = s18_state("3-2", board=["Varus", "Aphelios", "Diana", "Vi", "Kog'Maw", "Leona"], bench=["Varus"], level=6, gold=28, hp=55)
    out = s18.analyze(st)
    # 8 gold to roll: below the smallest fixed bucket, but a 24% shot at a 2-star Varus.
    assert out.econ.recommendation == EconAction.ROLL and out.econ.roll_budget == 8
    varus = next(o for o in out.odds if o.unit == "韦鲁斯")
    assert 8 in varus.p_goal_by_gold and varus.p_at(8) > 0.2
    assert varus.p_at(9) == varus.p_goal_by_gold[8] and varus.p_at(5) == 0.0


def test_s18_rich_warning_counts_the_level_on_an_unread_level(s18):
    board = ["Aphelios", "Nidalee", "Varus", "Diana", "Kog'Maw", "Vi", "Amumu"]
    out = s18.analyze(s18_state("4-2", board=board, gold=110, hp=80))
    assert out.econ.recommendation == EconAction.LEVEL and out.econ.target_level == 8
    assert not any("太多" in w for w in out.warnings)
    read = s18.analyze(s18_state("4-2", board=board, gold=110, hp=80, level=7))
    assert not any("太多" in w for w in read.warnings)


def test_s18_pve_round_waits_for_the_next_fight(s18):
    board = [("Aphelios", 2), "Nidalee", "Varus", "Diana", "Kog'Maw", "Vi", "Amumu", "Sentinel"]
    out = s18.analyze(s18_state("4-7", board=board, level=8, gold=52, hp=35))
    assert out.econ.recommendation == EconAction.SAVE and out.econ.roll_budget == 0
    assert "5-1" in out.econ.reason
    assert any("下回合 5-1" in w for w in out.warnings) and not any("别再贪经济" in w for w in out.warnings)
    crit = s18.analyze(s18_state("3-7", board=board[:7], level=7, gold=45, hp=15))
    assert crit.econ.recommendation in (EconAction.SAVE, EconAction.LEVEL) and crit.econ.roll_budget == 0
    assert not any("这回合必须" in w for w in crit.warnings)


def test_s18_carousel_or_last_round_shop_gives_no_picks(s18):
    board = ["Aphelios", "Nidalee", "Varus", "Diana", "Kog'Maw", "Vi"]
    st = _with_shop(s18_state("3-4", board=board, level=6, gold=30, hp=80), ["Varus", "Varus", "Karma", "Teemo", "Gromp"])
    assert s18.analyze(st).shop_picks  # a current shop
    st.screen_type = ScreenType.CAROUSEL
    assert s18.analyze(st).shop_picks == []
    st.screen_type = ScreenType.PLANNING
    st.field_age.update({"shop": 10.0, "round": 20.0})  # read before this round started
    assert s18.analyze(st).shop_picks == []
    st.field_age.update({"shop": 20.0})
    assert s18.analyze(st).shop_picks


def test_errors_are_kept_per_call(set_data, mech, comps, monkeypatch):
    import tft_advisor.engine.analyzer as mod

    an = Analyzer(set_data, mech, comps)
    failing = mid_game_state()
    other = make_state("2-1", board=["Garen"], gold=10, level=3, hp=100)
    real_plan_items = mod.plan_items
    real_sell = an._sell_candidates

    def plan_items(state, *a, **k):
        if state.gold == failing.gold:
            raise RuntimeError("boom")
        return real_plan_items(state, *a, **k)

    def sell(state, *a, **k):
        if state.gold == failing.gold:
            an.analyze(other)  # the ask thread runs a whole analysis meanwhile
        return real_sell(state, *a, **k)

    monkeypatch.setattr(mod, "plan_items", plan_items)
    monkeypatch.setattr(an, "_sell_candidates", sell)
    out = an.analyze(failing)
    assert "部分分析出错" in out.warnings[-1]
    assert an.last_errors and "items" in an.last_errors[0]


def test_top_comp_is_sticky_through_a_one_frame_dip(mech):
    an = Analyzer(s18_set_data(), mech, s18_comps())
    board = [("Ahri", 2), "Zyra", "Gnar", "Sett", "Yorick", "Diana"]
    full = s18_state("3-3", board=board, level=6, gold=30, hp=70, game_id="g1", last_update=1.0)
    assert an.analyze(full).comps[0].name == "阿狸 婕拉"
    dip = s18_state("3-5", board=board[1:], level=6, gold=30, hp=70, game_id="g1", last_update=2.0)
    out = an.analyze(dip)
    assert out.comps[0].name == "阿狸 婕拉"  # Ahri missed once: keep the plan
    assert an.analyze(dip).comps[0].name == "阿狸 婕拉"  # the same frame again (ask) does not count twice
    out = an.analyze(dip.model_copy(update={"last_update": 3.0}))
    assert out.comps[0].name != "阿狸 婕拉"  # second frame in a row: switch
    # States without a game id (tests, one-off analyses) are not sticky.
    assert Analyzer(s18_set_data(), mech, s18_comps()).analyze(dip.model_copy(update={"game_id": ""})).comps[0].name != "阿狸 婕拉"


# ---------------------------------------------------------------------------
# Round 4: empty slots, unpaid rolls at low HP, PvE warnings, pool overflow
# ---------------------------------------------------------------------------



def _s18_shop(st, names):
    sd = s18_set_data()
    champs = [sd.resolve_champion(n) for n in names]
    st.shop = [ShopSlot(name=c.name, cost=c.cost) for c in champs]
    st.shop_units = [Unit(api_name=c.api_name, name=c.name, cost=c.cost, traits=list(c.traits)) for c in champs]
    st.field_age["round"], st.field_age["shop"] = 1.0, 2.0
    return st


def test_s18_empty_team_slots_are_filled_from_the_shop(s18):
    # 2-1, three units at level 4, nothing in the shop for the comp: one unit
    # for the empty slot beats "save to 50".
    st = s18_state("2-1", board=[("卡尔玛", 1, (), 3, 3), ("约里克", 1, (), 0, 3), ("雷克塞", 1, (), 0, 2)],
                   gold=12, level=4, xp_current=0, hp=94, streak=0)
    out = s18.analyze(_s18_shop(st, ["蕾欧娜", "韦鲁斯", "凯特琳", "提莫", "慎"]))
    assert len(out.shop_picks) == 1 and out.shop_picks_cost <= 12
    # 4-2, six units at level 8: two units, the best trait fit first.
    st = s18_state("4-2", board=[("厄斐琉斯", 2, ("无尽之刃",)), ("韦鲁斯", 2), ("霞", 2), ("洛", 2), ("深红锋喙鸟", 2), ("阿木木", 1)],
                   gold=40, level=8, xp_current=0, hp=64, streak=0)
    out = s18.analyze(_s18_shop(st, ["瑟提", "黛安娜", "莉莉娅", "提莫", "慎"]))
    assert out.comps[0].name == "厄斐琉斯 迅捷射手"
    assert len(out.shop_picks) == 2 and out.shop_picks[0] == "黛安娜"
    # A bench unit fills the slot for free: nothing bought just to fill.
    st = s18_state("2-1", board=["卡尔玛", "约里克", "雷克塞"], bench=["蕾欧娜"], gold=12, level=4, xp_current=0, hp=94)
    out = s18.analyze(_s18_shop(st, ["韦鲁斯", "凯特琳", "提莫", "慎", "瑟提"]))
    assert out.shop_picks == []


def test_empty_slot_makes_a_missing_comp_unit_a_roll_target(analyzer):
    from tft_advisor.models import CompSuggestion, HitOdds

    def odds(api, p_shop):
        return HitOdds(unit=api, api_name=f"TFT99_{api}", cost=4, owned_copies=0, goal_copies=3, goal_star=2,
                       seen_elsewhere=0, remaining_in_pool=10, level=8, p_per_slot=p_shop / 5, p_in_shop=p_shop,
                       p_goal_by_gold={12: 0.01})

    comp = CompSuggestion(name="X", score=0.8, carry="Kayle", missing_units=["Akali", "Draven"])
    analysis = Analysis(comps=[comp], econ=EconPlan(gold=32, recommendation=EconAction.ROLL, roll_budget=12),
                        odds=[odds("Akali", 0.09), odds("Draven", 0.09)])
    # No 2-star within reach and the carry is owned: nothing to roll for...
    assert not analyzer._has_roll_target(analysis, open_slots=0)
    # ...unless a team slot is empty: one copy of either missing comp unit
    # fills it, and 6 rolls show one about two times in three.
    assert analyzer._has_roll_target(analysis, open_slots=1)
    analysis.econ.roll_budget = 4
    assert not analyzer._has_roll_target(analysis, open_slots=1)


def test_unpaid_roll_at_low_hp_late_is_hold_not_save(analyzer, mech):
    econ = EconPlan(gold=20, recommendation=EconAction.ROLL, roll_budget=10, reason="x")
    picks = [((2, 1, 0, -2, i), n, 2, None) for i, n in enumerate(["A", "B", "C", "D", "E"])]
    analyzer._fund_picks(econ, picks, hp=35, stage=StageRound.parse("4-3"))
    assert econ.recommendation == EconAction.HOLD and econ.roll_budget == 0 and "下回合再搜" in econ.reason
    econ = EconPlan(gold=20, recommendation=EconAction.ROLL, roll_budget=10, reason="x")
    analyzer._fund_picks(econ, picks, hp=80, stage=StageRound.parse("4-3"))
    assert econ.recommendation == EconAction.SAVE


def test_no_rich_gold_warning_while_the_pve_round_waits(analyzer):
    st = make_state("4-7", board=["Graves", "Ahri", "Garen", "Braum", "Shen", "Pyke", "Lucian", "Kayle"],
                    gold=74, level=8, hp=38, xp_current=10)
    out = analyzer.analyze(st)
    assert out.econ.reason.startswith("野怪回合")
    assert not any("太多了" in w for w in out.warnings)
    out = analyzer.analyze(st.model_copy(update={"stage": StageRound.parse("5-2"), "hp": 80}))
    assert out.econ.recommendation != EconAction.SAVE or any("太多了" in w for w in out.warnings)


def test_scouted_copies_over_the_pool_are_a_misread_star(analyzer, mech):
    cost = analyzer.set_data.resolve_champion("Akali").cost
    pool = mech.pool_size[cost]
    st = make_state("4-5", board=[("Akali", 2), "Graves"], bench=["Akali"], level=8, gold=20, hp=60)
    taken = {"Star": {"TFT99_Akali": 9}}  # their 2-star read as 3-star
    assert 4 + 9 > pool
    out = analyzer.analyze(st, taken)
    akali = next(o for o in out.odds if o.api_name == "TFT99_Akali")
    assert akali.seen_elsewhere == 3 and akali.remaining_in_pool == pool - 4 - 3
    assert any("Star" in w and "超过卡池" in w for w in out.warnings)
    assert not any("—" in w for w in out.warnings)
