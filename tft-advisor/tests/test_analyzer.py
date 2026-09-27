from __future__ import annotations

import json

import pytest

from tft_advisor.data.comps import load_comps
from tft_advisor.engine.analyzer import Analyzer
from tft_advisor.models import Analysis, EconAction, GameState, ShopSlot, StageRound, Unit

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
    assert graves.seen_elsewhere == 3 and graves.owned_copies == 5
    assert graves.remaining_in_pool == analyzer.mech.pool_size[1] - 5 - 3
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
    assert "金币 80" in w
    assert "血量只剩 20" in w
    assert "等级落后" not in w  # 4-1 standard is 7

    behind = make_state("4-2", board=["Graves"] * 6, gold=10, level=6, hp=80)
    assert any("等级落后" in x for x in an.analyze(behind).warnings)

    low = make_state("3-2", board=["Graves"], hp=40, level=1)
    assert any("偏低" in x for x in an.analyze(low).warnings)

    loose = make_state("3-2", board=["Graves"], item_bench=["Infinity Edge"], level=1)
    assert any("成装还没装备" in x for x in an.analyze(loose).warnings)


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
    # Without a streak the medium-HP roll-down stays.
    plain = s18.analyze(s18_state("4-2", board=board, level=7, gold=36, hp=64, streak=-1, xp_current=0))
    assert plain.econ.recommendation == EconAction.ROLL


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
