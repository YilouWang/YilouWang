from __future__ import annotations

import json

import pytest

from tft_advisor.data.comps import CompDef, load_comps, normalize_style
from tft_advisor.data.setdata import SetData
from tft_advisor.engine.comps_engine import auto_comps, score_comp, suggest_comps

from .conftest import make_state


@pytest.fixture(scope="module")
def sample_comps(set_data):
    return load_comps(None, set_data)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_bundled_sample_comps_only_for_sample_set(set_data, sample_comps):
    assert 3 <= len(sample_comps) <= 4
    names = [c.name for c in sample_comps]
    assert "枪手海盗" in names
    mf = next(c for c in sample_comps if c.name == "枪手海盗")
    assert mf.carry == "Miss Fortune" and mf.style == "fast8" and mf.tier == "A"
    assert "Infinity Edge" in mf.carry_items
    for c in sample_comps:
        assert c.style in ("standard", "fast8", "reroll1", "reroll2", "reroll3")
        assert all(set_data.resolve_champion(u) for u in c.units)
        assert "—" not in c.notes and "—" not in c.name

    other = SetData(set_number=18, set_name="x", champions=set_data.champions, traits=set_data.traits, items=set_data.items)
    assert load_comps(None, other) == []
    assert load_comps("", other) == []


def test_load_json_with_resolution_and_drop(tmp_path, set_data):
    path = tmp_path / "comps.json"
    path.write_text(
        json.dumps(
            {
                "comps": [
                    {"name": "My Ashe", "units": ["ashe", "Braum", "Volibear", "NotAUnit"], "carry": "Ashe",
                     "carry_items": ["guinsoos rageblade", "Mystery"], "traits": ["glacial"], "style": "reroll", "tier": "s"},
                    {"name": "Garbage", "units": ["Foo", "Bar", "Baz", "Graves"]},
                    {"name": "", "units": ["Graves"]},
                    {"name": "My Ashe", "units": ["Graves"]},
                    {"name": "Carry outside units", "units": "Graves, Lucian", "carry": "Miss Fortune"},
                ]
            }
        ),
        encoding="utf-8",
    )
    logs: list[str] = []
    comps = load_comps(str(path), set_data, log=logs.append)
    assert [c.name for c in comps] == ["My Ashe", "Carry outside units"]
    ashe = comps[0]
    assert ashe.units == ["Ashe", "Braum", "Volibear"]
    assert ashe.carry_items == ["Guinsoo's Rageblade", "Mystery"]
    assert ashe.traits == ["Glacial"] and ashe.tier == "S"
    assert ashe.style == "reroll3"  # bare "reroll" + 3-cost carry
    assert comps[1].units[0] == "Miss Fortune" and comps[1].carry == "Miss Fortune"
    text = "\n".join(logs)
    assert "Garbage" in text and "NotAUnit" in text and "重复" in text


def test_load_toml(tmp_path, set_data):
    path = tmp_path / "comps.toml"
    path.write_text(
        '[[comps]]\nname = "Toml Comp"\nunits = ["Ahri", "Morgana"]\ncarry = "Ahri"\nstyle = "速8"\n',
        encoding="utf-8",
    )
    comps = load_comps(str(path), set_data)
    assert len(comps) == 1 and comps[0].style == "fast8" and comps[0].units == ["Ahri", "Morgana"]


def test_load_missing_or_broken_file(tmp_path, set_data):
    logs: list[str] = []
    assert load_comps(str(tmp_path / "nope.json"), set_data, log=logs.append) == []
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_comps(str(bad), set_data, log=logs.append) == []
    assert len(logs) == 2


@pytest.mark.parametrize(
    "raw,cost,expected",
    [
        ("", None, "standard"), (None, None, "standard"), ("fast8", None, "fast8"), ("Fast 8", None, "fast8"),
        ("reroll", 1, "reroll1"), ("reroll", None, "reroll2"), ("2cost reroll", None, "reroll2"),
        ("赌狗", 3, "reroll3"), ("运营", None, "standard"), ("weird", None, "standard"),
    ],
)
def test_normalize_style(raw, cost, expected):
    assert normalize_style(raw, cost) == expected


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_empty_state_ranks_by_tier(set_data, sample_comps):
    sugs = suggest_comps(make_state(), set_data, sample_comps, {}, top_n=4)
    assert len(sugs) == 4
    assert sugs[0].name == "冰川游侠"  # tier S
    assert all(0.0 <= s.score <= 1.0 for s in sugs)
    assert sugs[0].have_units == [] and len(sugs[0].missing_units) == len(sugs[0].core_units)


def test_overlap_drives_the_ranking(set_data, sample_comps):
    st = make_state("3-2", board=["Graves", "Lucian", "Pyke", ("Garen", 2), "Braum"], bench=["Vayne"])
    sugs = suggest_comps(st, set_data, sample_comps, {}, top_n=3)
    top = sugs[0]
    assert top.name == "枪手海盗"
    assert set(top.have_units) == {"Graves", "Lucian", "Pyke", "Garen", "Braum", "Vayne"}
    assert top.missing_units == ["Kayle", "Miss Fortune"]  # sorted by cost
    assert top.carry == "Miss Fortune" and top.carry_items[0] == "Infinity Edge"
    assert "还缺主C" in top.reason
    assert len(sugs) == 3 and sugs[0].score > sugs[1].score


def test_star_and_carry_bonus(set_data, sample_comps):
    comp = next(c for c in sample_comps if c.name == "冰川游侠")
    base = score_comp(comp, make_state(board=["Braum"]), set_data, {})
    two_star = score_comp(comp, make_state(board=[("Braum", 2)]), set_data, {})
    carry = score_comp(comp, make_state(board=["Braum", "Ashe"]), set_data, {})
    carry2 = score_comp(comp, make_state(board=["Braum", ("Ashe", 2)]), set_data, {})
    assert base.score < two_star.score
    assert base.score < carry.score < carry2.score
    assert "已到手" in carry.reason and "两星" in carry2.reason


def test_item_fit(set_data, sample_comps):
    comp = next(c for c in sample_comps if c.name == "枪手海盗")
    none = score_comp(comp, make_state(item_bench=["Chain Vest", "Chain Vest"]), set_data, {})
    fit = score_comp(comp, make_state(item_bench=["B.F. Sword", "Sparring Gloves", "Recurve Bow"]), set_data, {})
    held = score_comp(comp, make_state(board=[("Graves", 1, ("Infinity Edge",))]), set_data, {})
    assert fit.score > none.score
    assert "1/3" in fit.reason
    assert "1/3" in held.reason
    # Unresolvable carry items are not counted in the denominator.
    odd = CompDef(name="odd", units=["Graves"], carry="Graves", carry_items=["Infinity Edge", "Giant Slayer", "Mystery Item"])
    half = score_comp(odd, make_state(item_bench=["B.F. Sword", "Sparring Gloves"]), set_data, {})
    assert "1/2" in half.reason


def test_contested_penalty(set_data, sample_comps):
    st = make_state("4-1", board=["Graves", "Lucian", "Pyke"])
    free = suggest_comps(st, set_data, sample_comps, {}, top_n=4)
    taken = {"Bob": {"TFT99_MissFortune": 2}, "Eve": {"TFT99_Kayle": 1, "TFT99_MissFortune": 1}, "Ann": {"TFT99_Graves": 3}}
    contested = suggest_comps(st, set_data, sample_comps, taken, top_n=4)
    f = next(s for s in free if s.name == "枪手海盗")
    c = next(s for s in contested if s.name == "枪手海盗")
    assert c.contested_by == ["Bob", "Eve"]
    assert c.score == pytest.approx(f.score - 0.2, abs=1e-3)
    assert "Bob" in c.reason


def test_self_is_never_contesting(set_data, sample_comps):
    st = make_state("4-1", board=["Graves"], self_name="Me")
    sugs = suggest_comps(st, set_data, sample_comps, {"Me": {"TFT99_MissFortune": 3}}, top_n=4)
    assert all(not s.contested_by for s in sugs)


def test_level_fit_penalizes_overleveled_reroll(set_data, sample_comps):
    comp = next(c for c in sample_comps if c.name == "狂野法师")  # reroll2 (level 6)
    ok = score_comp(comp, make_state("3-5", board=["Ahri"], level=6), set_data, {})
    over = score_comp(comp, make_state("3-5", board=["Ahri"], level=8), set_data, {})
    assert over.score < ok.score
    fast8 = next(c for c in sample_comps if c.name == "枪手海盗")
    healthy = score_comp(fast8, make_state("3-2", hp=80, level=6), set_data, {})
    dying = score_comp(fast8, make_state("3-2", hp=20, level=6), set_data, {})
    assert dying.score < healthy.score


def test_hint_boost(set_data, sample_comps):
    st = make_state("3-2", board=["Graves", "Lucian", "Pyke"])
    plain = suggest_comps(st, set_data, sample_comps, {}, top_n=4)
    hinted = suggest_comps(st, set_data, sample_comps, {}, top_n=4, hint="想玩 狂野法师")
    assert plain[0].name == "枪手海盗"
    h = next(s for s in hinted if s.name == "狂野法师")
    p = next(s for s in plain if s.name == "狂野法师")
    assert h.score > p.score and "指定" in h.reason
    by_carry = suggest_comps(st, set_data, sample_comps, {}, top_n=4, hint="Draven")
    assert next(s for s in by_carry if s.name == "帝国剑士").score > next(s for s in plain if s.name == "帝国剑士").score


def test_top_n_and_bad_comp(set_data):
    comps = [CompDef(name="empty", units=["Nobody"]), CompDef(name="ok", units=["Graves"])]
    sugs = suggest_comps(make_state(board=["Graves"]), set_data, comps, None, top_n=5)
    assert [s.name for s in sugs] == ["ok"]
    assert suggest_comps(make_state(), set_data, comps, None, top_n=0) == []


# ---------------------------------------------------------------------------
# Auto comps
# ---------------------------------------------------------------------------


def test_auto_comps_empty_state(set_data):
    assert auto_comps(make_state(), set_data) == []
    assert suggest_comps(make_state(), set_data, [], {}) == []


def test_auto_comps_trait_breakpoints(set_data):
    # Noble 3/6 (Lucian, Garen: 2 of 3), Knight 2 (Garen: 1 of 2), Pirate 3 (Graves: 1 of 3).
    # Gunslinger 4 is impossible in the sample set (only 3 Gunslingers exist).
    st = make_state("2-5", board=["Graves", "Lucian", "Garen"], level=5)
    sugs = suggest_comps(st, set_data, [], {}, top_n=5)
    names = [s.name for s in sugs]
    assert {"Noble 3", "Knight 2", "Pirate 3"} <= set(names)
    assert not any(n.startswith("Gunslinger") for n in names)
    noble = next(s for s in sugs if s.name == "Noble 3")
    assert noble.missing_units == ["Vayne"]  # cheapest Noble
    # The direction keeps the fielded board: trait members first, then the rest of the board.
    assert noble.have_units[:2] == ["Garen", "Lucian"] and set(noble.have_units) == {"Lucian", "Garen", "Graves"}
    assert noble.core_units == noble.have_units + ["Vayne"]
    knight = next(s for s in sugs if s.name == "Knight 2")
    assert knight.missing_units == ["Darius"]  # 1 cost Knight before Kayle
    pirate = next(s for s in sugs if s.name == "Pirate 3")
    assert pirate.missing_units == ["Pyke", "Miss Fortune"]
    assert noble.score > pirate.score  # closer to the breakpoint
    for s in sugs:
        assert 0 <= s.score <= 1
        assert s.reason and "—" not in s.reason
        assert s.carry in s.have_units
    assert [s.score for s in sugs] == sorted((s.score for s in sugs), reverse=True)


def test_auto_comps_prefers_cheap_and_reports_contest(set_data):
    # Wild: Warwick (1 of 2): Ahri (2 cost) before Gnar (4 cost).
    # Glacial: Volibear (1 of 2): Braum (2 cost) before Ashe (3 cost).
    # Brawler 4 would need 2 more but only Gnar is left: not proposed.
    st = make_state("3-1", board=["Warwick", "Volibear"])
    sugs = auto_comps(st, set_data, {"Bob": {"TFT99_Warwick": 2, "TFT99_Ahri": 1}}, top_n=10)
    names = [s.name for s in sugs]
    assert "Wild 2" in names and "Glacial 2" in names
    assert not any(n.startswith("Brawler") for n in names)
    wild = next(s for s in sugs if s.name == "Wild 2")
    glacial = next(s for s in sugs if s.name == "Glacial 2")
    assert wild.missing_units == ["Ahri"] and glacial.missing_units == ["Braum"]
    assert wild.contested_by == ["Bob"] and "Bob" in wild.reason
    assert glacial.contested_by == []
    assert wild.score < glacial.score


def test_auto_comps_skip_unreachable_breakpoints(set_data):
    # Assassin 3 from one Pyke (2 away) is allowed; Ninja 4 needs 3 more Ninjas but
    # only Akali exists; Sorcerer is never proposed without a Sorcerer.
    st = make_state(board=["Pyke", "Shen"])
    names = [s.name for s in auto_comps(st, set_data, top_n=10)]
    assert "Assassin 3" in names and "Blademaster 3" in names
    assert not any(n.startswith(("Ninja", "Sorcerer")) for n in names)


def test_unresolvable_comp_file_falls_back_to_auto_comps(set_data):
    stale = [CompDef(name="stale set", units=["Jinx", "Vi"])]
    sugs = suggest_comps(make_state(board=["Graves", "Lucian", "Garen"]), set_data, stale, {})
    assert sugs and all(s.name != "stale set" for s in sugs)
    assert sugs == suggest_comps(make_state(board=["Graves", "Lucian", "Garen"]), set_data, [], {})


def test_auto_comps_follow_the_fielded_board(set_data):
    # One Katarina on the bench must not drive the direction of a full board,
    # and the carry is the strongest fielded damage dealer, not the trait member.
    st = make_state(
        "4-1",
        board=[("Garen", 2), ("Darius", 2), ("Braum", 2), ("Graves", 2), ("Vayne", 2), ("Lucian", 2), "Ashe"],
        bench=["Katarina"],
        level=7,
    )
    sugs = auto_comps(st, set_data, top_n=5)
    assassin = next(s for s in sugs if s.name == "Assassin 3")
    pirate = next(s for s in sugs if s.name == "Pirate 3")
    assert assassin.score < 0.4 < pirate.score  # 0.4: the analyzer's commit threshold
    assert assassin.carry == pirate.carry == "Lucian"  # 2 cost 2-star on the board
    assert {"Garen", "Darius", "Braum", "Graves", "Vayne", "Lucian", "Ashe"} <= set(pirate.have_units)


def test_auto_comps_bench_members_weigh_half_and_hint_fielding(set_data):
    fielded = auto_comps(make_state(board=["Graves", "Lucian"], bench=["Garen"]), set_data, top_n=10)
    benched = auto_comps(make_state(board=["Graves", "Garen"], bench=["Lucian"]), set_data, top_n=10)
    noble_f = next(s for s in fielded if s.name.startswith("Noble"))
    noble_b = next(s for s in benched if s.name.startswith("Noble"))
    assert noble_f.score == noble_b.score  # one Noble fielded in both
    both = auto_comps(make_state(board=["Lucian", "Garen"]), set_data, top_n=10)
    assert next(s for s in both if s.name.startswith("Noble")).score > noble_f.score


def test_auto_comps_hint_to_field_a_benched_trait_unit():
    # Knight 2/3 (modified sample): Garen fielded + Darius benched = active once fielded.
    import copy

    from tft_advisor.data.setdata import bundled_sample

    data = copy.deepcopy(bundled_sample())
    for entry in data["setData"]:
        for t in entry.get("traits", []):
            if t["apiName"] == "TFT99_Knight":
                t["effects"] = [{"minUnits": 2}, {"minUnits": 3}]
    sd = SetData.from_cdragon(data)
    sugs = auto_comps(make_state(board=["Garen", "Graves"], bench=["Darius"]), sd, top_n=10)
    knight = next(s for s in sugs if s.name == "Knight 3")
    assert knight.missing_units == ["Kayle"]
    assert "放上场就能激活" in knight.reason and "—" not in knight.reason
    fielded = auto_comps(make_state(board=["Garen", "Graves", "Darius"]), sd, top_n=10)
    k2 = next(s for s in fielded if s.name == "Knight 3")
    assert "羁绊已激活" in k2.reason and k2.score > knight.score


def test_duplicate_units_in_comp_file_are_not_misses(tmp_path, set_data):
    path = tmp_path / "dup.json"
    path.write_text(json.dumps({"comps": [{"name": "d", "units": ["Graves", "Graves", "Graves", "Foo"]}]}), encoding="utf-8")
    comps = load_comps(str(path), set_data)
    assert [c.units for c in comps] == [["Graves"]]


def test_load_comps_never_raises(tmp_path, set_data):
    deep = tmp_path / "deep.json"
    deep.write_text("[" * 100000 + "]" * 100000, encoding="utf-8")
    logs: list[str] = []
    assert load_comps(str(deep), set_data, log=logs.append) == []
    weird = tmp_path / "weird.json"
    weird.write_text(json.dumps({"comps": [
        {"name": {"x": 1}, "units": 5}, {"name": "ok", "units": ["Graves", {"a": 1}], "carry_items": 7, "style": ["x"], "tier": 3},
    ]}), encoding="utf-8")
    comps = load_comps(str(weird), set_data, log=logs.append)
    assert [c.name for c in comps] == ["ok"] and comps[0].units == ["Graves"] and comps[0].style == "standard"
    assert load_comps(str(tmp_path), set_data, log=logs.append) == []  # a directory
