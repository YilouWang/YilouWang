from __future__ import annotations

import json
from importlib import resources

import pytest

from tft_advisor.engine.items import champion_profile, item_role, item_roles, plan_items
from tft_advisor.models import CompSuggestion

from .conftest import make_state, make_unit

CORE_ITEMS = [
    "Deathblade", "Infinity Edge", "Giant Slayer", "Hextech Gunblade", "Spear of Shojin", "Edge of Night",
    "Bloodthirster", "Sterak's Gage", "Red Buff", "Guinsoo's Rageblade", "Void Staff", "Titan's Resolve",
    "Kraken's Fury", "Runaan's Hurricane", "Nashor's Tooth", "Last Whisper", "Rabadon's Deathcap",
    "Archangel's Staff", "Crownguard", "Ionic Spark", "Morellonomicon", "Jeweled Gauntlet", "Blue Buff",
    "Protector's Vow", "Adaptive Helm", "Spirit Visage", "Hand of Justice", "Bramble Vest",
    "Gargoyle Stoneplate", "Sunfire Cape", "Steadfast Heart", "Dragon's Claw", "Evenshroud", "Quicksilver",
    "Warmog's Armor", "Guardbreaker", "Thief's Gloves", "Tactician's Crown", "Statikk Shiv", "Redemption",
    "Zeke's Herald", "Chalice of Power", "Locket of the Iron Solari", "Rapid Firecannon",
    "Shroud of Stillness", "Zephyr", "Frozen Heart", "Tactician's Cape", "Tactician's Shield",
]


def test_role_table_covers_core_items():
    table = item_roles()
    raw = json.loads(resources.files("tft_advisor.data").joinpath("bundled", "item_roles.json").read_text("utf-8"))
    for name in CORE_ITEMS:
        assert name in raw, name
    for name, spec in raw.items():
        if name.startswith("_"):
            continue
        assert spec["role"] in ("ad", "ap", "tank", "utility", "flex"), name
        assert spec["tier"] in (1, 2, 3), name
    assert len(table) == len([k for k in raw if not k.startswith("_")])


def test_item_role_lookup(set_data):
    assert item_role(set_data.resolve_item("Infinity Edge")) == ("ad", 3)
    assert item_role(set_data.resolve_item("Sunfire Cape")) == ("tank", 2)  # api TFT_Item_RedBuff
    assert item_role(set_data.resolve_item("Red Buff")) == ("ad", 2)
    assert item_role(None, "Radiant Warmog's Armor") == ("tank", 3)
    assert item_role(None, "Totally New Item") == ("flex", 2)
    assert item_role(set_data.resolve_item("Knight Emblem")) == ("emblem", 2)


def test_champion_profile(set_data):
    prof = {n: champion_profile(set_data.resolve_champion(n)) for n in ("Graves", "Ahri", "Garen", "Braum", "Draven")}
    assert prof == {"Graves": "ad", "Ahri": "ap", "Garen": "tank", "Braum": "tank", "Draven": "ad"}
    # Items held tip an unknown unit.
    u = make_unit("Qqq", items=["Rabadon's Deathcap", "Blue Buff"])
    assert champion_profile(None, u, set_data) == "ap"


def test_empty_bench_no_suggestions(set_data):
    assert plan_items(make_state("3-2"), set_data) == []
    assert plan_items(make_state(), set_data) == []


def test_single_component_nothing_to_build(set_data):
    st = make_state("3-2", board=["Graves"], item_bench=["B.F. Sword"])
    assert plan_items(st, set_data) == []


def test_pairing_and_holders(set_data):
    st = make_state(
        "3-2",
        board=[("Graves", 2, (), 3, 1), ("Ahri", 1, (), 3, 5), ("Braum", 2, (), 0, 3), ("Garen", 1, (), 1, 2)],
        item_bench=["B.F. Sword", "Sparring Gloves", "Needlessly Large Rod", "Needlessly Large Rod", "Chain Vest", "Chain Vest"],
    )
    sugs = plan_items(st, set_data)
    by_item = {s.item: s for s in sugs}
    assert set(by_item) == {"Infinity Edge", "Rabadon's Deathcap", "Bramble Vest"}
    assert by_item["Infinity Edge"].holder == "Graves"
    assert sorted(by_item["Infinity Edge"].components) == ["B.F. Sword", "Sparring Gloves"]
    assert by_item["Rabadon's Deathcap"].holder == "Ahri"
    assert by_item["Bramble Vest"].holder == "Braum"  # front row tank
    assert all(s.priority == 1 for s in sugs)  # 3-2 with 2+ components: slam
    assert all("—" not in s.reason for s in sugs)


def test_priority_two_early_unless_low_hp(set_data):
    # A tier 2 item with no special fit waits until 2-5 (or low HP).
    st = make_state("2-2", board=["Graves"], item_bench=["B.F. Sword", "Chain Vest"])
    assert [s.priority for s in plan_items(st, set_data)] == [2]
    assert [s.priority for s in plan_items(st, set_data, hp_bucket="low")] == [1]
    st25 = make_state("2-5", board=["Graves"], item_bench=["B.F. Sword", "Chain Vest"])
    assert [s.priority for s in plan_items(st25, set_data)] == [1]


def test_core_and_front_line_items_are_slammed_from_2_1(set_data):
    # Infinity Edge (tier 3) at 2-2 with healthy HP: slam it.
    st = make_state("2-2", board=["Graves"], item_bench=["B.F. Sword", "Sparring Gloves"])
    sugs = plan_items(st, set_data)
    assert [s.priority for s in sugs] == [1] and "现在就合" in sugs[0].reason
    # A tank item for a fielded front-row unit is slammed too.
    st = make_state(
        "2-2", board=[("Braum", 1, (), 0, 3), ("Graves", 1, (), 3, 0)], item_bench=["Chain Vest", "Giant's Belt"]
    )
    sugs = plan_items(st, set_data)
    assert [(s.item, s.holder, s.priority) for s in sugs] == [("Sunfire Cape", "Braum", 1)]
    # Loss streaking into the carousel: hold the components.
    st.streak = -4
    assert [s.priority for s in plan_items(st, set_data)] == [2]
    # Stage 1 never slams at healthy HP.
    st1 = make_state("1-4", board=["Graves"], item_bench=["B.F. Sword", "Sparring Gloves"])
    assert [s.priority for s in plan_items(st1, set_data)] == [2]


def test_comp_carry_items_preferred(set_data):
    # BF + Bow + Gloves: Giant Slayer (BF+Bow) or Infinity Edge (BF+Gloves) or Last Whisper (Bow+Gloves).
    st = make_state("3-2", board=[("Graves", 2, (), 3, 0), ("Lucian", 1, (), 3, 1)], item_bench=["B.F. Sword", "Recurve Bow", "Sparring Gloves"])
    comp = CompSuggestion(name="X", score=0.5, carry="Lucian", carry_items=["Last Whisper"])
    sugs = plan_items(st, set_data, comp=comp)
    assert sugs[0].item == "Last Whisper" and sugs[0].holder == "Lucian"
    assert "主C" in sugs[0].reason


def test_ad_item_avoids_ap_team(set_data):
    # AP carry: AD items lose value, AP items gain; Rabadon goes to the sorcerer carry first.
    st = make_state("3-5", board=[("Ahri", 2, (), 3, 3), ("Morgana", 1, (), 3, 4), ("Braum", 1, (), 0, 3)], item_bench=["Needlessly Large Rod", "Needlessly Large Rod"])
    sugs = plan_items(st, set_data, comp=CompSuggestion(name="AP", score=0.5, carry="Ahri"))
    assert sugs[0].item == "Rabadon's Deathcap" and sugs[0].holder == "Ahri"
    # Tear + BF: Spear of Shojin (flex) is fine; BF + Bow Giant Slayer on an AP team is still
    # built (tier 3) but ranks below the AP item.
    st2 = make_state("3-5", board=[("Ahri", 2, (), 3, 3), ("Braum", 1, (), 0, 3)], item_bench=["Needlessly Large Rod"] * 2 + ["B.F. Sword", "Recurve Bow"])
    sugs2 = plan_items(st2, set_data, comp=CompSuggestion(name="AP", score=0.5, carry="Ahri"))
    assert [s.item for s in sugs2][:2] == ["Rabadon's Deathcap", "Giant Slayer"]


def test_duplicate_items_are_discounted(set_data):
    st = make_state("3-2", board=[("Graves", 2, (), 3, 0), ("Braum", 2, (), 0, 3)], item_bench=["B.F. Sword"] * 2 + ["Sparring Gloves"] * 2)
    items = sorted(s.item for s in plan_items(st, set_data))
    # 2x Infinity Edge = 3.5 + 3.5 * 0.5 = 5.25 < Deathblade 3.5 + Thief's Gloves 2.0 = 5.5
    assert items == ["Deathblade", "Thief's Gloves"]


def test_holder_limited_to_three_items(set_data):
    st = make_state(
        "4-1",
        board=[("Graves", 2, ("Deathblade", "Giant Slayer"), 3, 0), ("Lucian", 1, (), 3, 1), ("Braum", 1, (), 0, 3)],
        item_bench=["B.F. Sword", "Sparring Gloves", "Recurve Bow", "Recurve Bow"],
    )
    sugs = plan_items(st, set_data)
    holders = [s.holder for s in sugs if s.item in ("Infinity Edge", "Red Buff", "Last Whisper", "Giant Slayer")]
    assert holders.count("Graves") <= 1
    assert "Lucian" in holders


def test_completed_item_on_bench_is_equipped(set_data):
    st = make_state("3-3", board=[("Braum", 1, (), 0, 3), ("Graves", 1, (), 3, 0)], item_bench=["Warmog's Armor"])
    sugs = plan_items(st, set_data)
    assert len(sugs) == 1 and sugs[0].components == [] and sugs[0].holder == "Braum" and sugs[0].priority == 1


def test_leftover_component_reports_missing_part(set_data):
    st = make_state("3-2", board=["Graves"], item_bench=["B.F. Sword"])
    comp = CompSuggestion(name="X", score=0.5, carry="Miss Fortune", carry_items=["Infinity Edge"])
    sugs = plan_items(st, set_data, comp=comp)
    assert len(sugs) == 1
    s = sugs[0]
    assert s.priority == 3 and s.item == "Infinity Edge" and "Sparring Gloves" in s.components
    assert s.holder == "Miss Fortune"
    assert "Sparring Gloves" in s.reason


def test_many_components_bruteforce_is_fast(set_data):
    import time

    comps = ["B.F. Sword", "Recurve Bow", "Needlessly Large Rod", "Tear of the Goddess", "Chain Vest",
             "Negatron Cloak", "Giant's Belt", "Sparring Gloves", "B.F. Sword", "Recurve Bow", "Chain Vest"]
    st = make_state("4-2", board=[("Graves", 2, (), 3, 0), ("Ahri", 2, (), 3, 5), ("Braum", 2, (), 0, 3), ("Garen", 1, (), 0, 2)], item_bench=comps)
    t0 = time.perf_counter()
    sugs = plan_items(st, set_data)
    assert time.perf_counter() - t0 < 2.0
    crafted = [s for s in sugs if s.components and s.priority <= 2]
    assert len(crafted) == 5  # 10 components considered, all paired
    used = sorted(c for s in crafted for c in s.components)
    assert len(used) == 10


def test_emblem_valued_when_trait_near_breakpoint(set_data):
    st = make_state("3-2", board=[("Garen", 1, (), 0, 0), ("Braum", 1, (), 0, 1), ("Graves", 1, (), 3, 0)], item_bench=["Spatula", "B.F. Sword"])
    sugs = plan_items(st, set_data)
    assert sugs and sugs[0].item == "Knight Emblem"
    assert sugs[0].holder in ("Braum", "Graves")  # a unit that is not already a Knight


@pytest.mark.parametrize("bucket", ["unknown", "healthy", "medium", "low", "critical"])
def test_no_crash_with_unknown_names(set_data, bucket):
    st = make_state("3-2", board=["Qqqq", "Graves"], item_bench=["???", "B.F. Sword", "B.F. Sword"])
    sugs = plan_items(st, set_data, hp_bucket=bucket)
    assert [s.item for s in sugs] == ["Deathblade"]
    assert sugs[0].holder == "Graves"


def test_chinese_locale_items(zh_set_data):
    from .conftest import chinese_set_data

    sd = chinese_set_data()
    st = make_state("3-2", item_bench=["暴风大剑", "拳套"])
    st.board = [
        make_unit("格雷福斯", 2, row=3, col=0, set_data=sd),
        make_unit("布隆", 1, row=0, col=3, set_data=sd),
    ]
    sugs = plan_items(st, sd)
    assert [(s.item, s.holder) for s in sugs] == [("无尽之刃", "格雷福斯")]
    assert sugs[0].components == ["暴风大剑", "拳套"]
    assert "物理输出" in sugs[0].reason
    assert champion_profile(sd.resolve_champion("阿狸")) == "ap"
    assert champion_profile(sd.resolve_champion("布隆")) == "tank"


def test_completed_item_when_every_unit_is_full(set_data):
    full = ("Deathblade", "Giant Slayer", "Infinity Edge")
    st = make_state("3-2", board=[("Graves", 2, full, 3, 0)], item_bench=["Warmog's Armor"])
    sugs = plan_items(st, set_data)
    assert sugs[0].holder is None and "满" in sugs[0].reason and "先上场" not in sugs[0].reason
    empty = plan_items(make_state("3-2", item_bench=["Warmog's Armor"]), set_data)
    assert "先上场" in empty[0].reason


# ---------------------------------------------------------------------------
# Set 18 regressions: profiles, Thief's Gloves, team-size items
# ---------------------------------------------------------------------------

from tft_advisor.engine.items import profile_hints_from_comps  # noqa: E402

from .conftest import s18_comps, s18_set_data, s18_state  # noqa: E402


def _s18_plan(st, comp_name=None):
    sd = s18_set_data()
    from tft_advisor.engine.analyzer import Analyzer
    from tft_advisor.data.mechanics import load_mechanics

    an = Analyzer(sd, load_mechanics(), s18_comps())
    out = an.analyze(st)
    return out


def test_s18_profiles_do_not_come_from_mixed_trait_words():
    sd = s18_set_data()
    prof = {n: champion_profile(sd.resolve_champion(n)) for n in ("Nidalee", "Master Yi", "Akali", "Soraka", "Azir")}
    # 魔战士 (Adaptor) and 狂战士 (Ravager) are not tank words; Executioner is not an AD word.
    assert prof["Nidalee"] == "ap" and prof["Master Yi"] == "ad" and prof["Akali"] == "ad"
    assert prof["Soraka"] != "ad" and prof["Azir"] != "ad"
    hints = profile_hints_from_comps(s18_comps(), sd)
    for name in ("Soraka", "Azir"):
        champ = sd.resolve_champion(name)
        assert hints[champ.api_name] == "ap"
        assert champion_profile(champ, hint=hints[champ.api_name]) == "ap"
    # Ties are broken by the items already held.
    from tft_advisor.models import Unit

    soraka = sd.resolve_champion("Soraka")
    u = Unit(api_name=soraka.api_name, name=soraka.name, cost=4, items=["灭世者的死亡之帽"], traits=list(soraka.traits))
    assert champion_profile(soraka, u, sd) == "ap"


def test_s18_ad_item_not_sent_to_an_ap_carry():
    st = s18_state(
        "4-2",
        board=[("Soraka", 1, ("Rabadon's Deathcap",), 3, 3), ("Malphite", 1, (), 0, 3), "Fiddlesticks", "Lillia", "Ornn", "Teemo", "Rammus"],
        items=["B.F. Sword", "B.F. Sword"], level=7, gold=30, hp=70,
    )
    out = _s18_plan(st)
    db = [s for s in out.items if s.item == "死亡之刃"]
    assert all(s.holder != "索拉卡" and s.priority != 1 for s in db)


def test_s18_nidalee_gets_rabadon_in_invoker_nidalee():
    st = s18_state(
        "4-2",
        board=[("Nidalee", 1, (), 1, 3), ("Morgana", 2, (), 3, 4), "Sentinel", "Teemo", "Pebbles", "Vi", "Sivir"],
        items=["Needlessly Large Rod", "Needlessly Large Rod"], level=7, gold=30, hp=70,
    )
    out = _s18_plan(st)
    assert out.comps[0].name == "神谕 奈德丽"
    rab = next(s for s in out.items if s.item == "灭世者的死亡之帽")
    assert rab.holder == "奈德丽"


def test_s18_thiefs_gloves_and_team_size_items_skip_the_itemized_carry_and_tank():
    board = [("Ahri", 2, ("Jeweled Gauntlet",), 3, 3), ("Sett", 2, ("Warmog's Armor",), 0, 3), "Morgana", "Karma", "Sentinel", "Pebbles", "Krug"]
    out = _s18_plan(s18_state("4-2", board=board, items=["Sparring Gloves", "Sparring Gloves"], level=7, gold=30, hp=70))
    tg = next(s for s in out.items if s.item == "窃贼手套")
    assert tg.holder not in ("阿狸", "瑟提", None)
    out = _s18_plan(
        s18_state("4-2", board=board, items=["Frying Pan", "Frying Pan", "Frying Pan", "Tear of the Goddess"], level=7, gold=30, hp=70)
    )
    holders = {s.item: s.holder for s in out.items}
    assert holders and all(h not in ("瑟提", "阿狸") for h in holders.values())


def test_thiefs_gloves_needs_an_itemless_fielded_unit(set_data):
    full = make_state("4-2", board=[("Graves", 2, ("Deathblade",), 3, 0)], item_bench=["Sparring Gloves", "Sparring Gloves"])
    sugs = plan_items(full, set_data)
    tg = next(s for s in sugs if s.item == "Thief's Gloves")
    assert tg.holder is None and "先留着" in tg.reason
    empty = make_state("4-2", board=[("Graves", 2, ("Deathblade",), 3, 0), ("Braum", 1, (), 0, 3)], item_bench=["Thief's Gloves"])
    sugs = plan_items(empty, set_data, comp=CompSuggestion(name="X", score=0.5, carry="Graves"))
    assert sugs[0].holder == "Braum"


# ---------------------------------------------------------------------------
# Regressions (domain review): emblems nobody can use, carry components
# ---------------------------------------------------------------------------


def _s18_comp(name_en: str):
    from tft_advisor.engine.comps_engine import suggest_comps

    from .conftest import s18_comps, s18_set_data, s18_state

    comp = next(c for c in s18_comps() if c.name_en == name_en)
    return comp, s18_set_data(), s18_state, suggest_comps


def test_s18_emblem_without_the_trait_is_not_slammed():
    comp, sd, s18_state, suggest = _s18_comp("Ahri Morgana")
    board = [("Ahri", 2, ["Jeweled Gauntlet"]), "Zyra", "Sett", "Morgana", "Karma", "Yorick", "Leona", "Taric"]
    for stage, bench, emblem in (("5-2", ["B.F. Sword", "Frying Pan"], "猎人纹章"), ("3-2", ["Spatula", "Giant's Belt"], "黑荆棘纹章")):
        st = s18_state(stage, board=board, items=bench, level=8, hp=80, gold=30)
        top = suggest(st, sd, [comp], {}, top_n=1)[0]
        out = plan_items(st, sd, comp=top, hp_bucket="healthy")
        assert not any(s.item == emblem for s in out), [s.item for s in out]
        assert not any(s.priority == 1 for s in out)
    # With the trait on the board the emblem is worth making.
    hunter = sd.resolve_trait("Hunter")
    hunters = [c.name for c in sd.champions.values() if hunter.name in c.traits][:2]
    st = s18_state("5-2", board=hunters + ["Leona"], items=["B.F. Sword", "Frying Pan"], level=8, hp=80, gold=30)
    assert any(s.item == "猎人纹章" for s in plan_items(st, sd, hp_bucket="healthy"))


def test_s18_carry_component_is_held_at_healthy_hp():
    comp, sd, s18_state, suggest = _s18_comp("Ahri Morgana")
    board = [("Ahri", 2), "Zyra", "Sett", "Morgana", "Karma", "Yorick", "Leona", "Taric"]
    st = s18_state("3-2", board=board, items=["B.F. Sword", "Giant's Belt"], level=6, hp=80, gold=30)
    top = suggest(st, sd, [comp], {}, top_n=1)[0]
    shojin = sd.resolve_item("Spear of Shojin")
    assert shojin.name in top.carry_items or "Spear of Shojin" in top.carry_items
    out = plan_items(st, sd, comp=top, hp_bucket="healthy")
    sterak = [s for s in out if s.components and "暴风大剑" in s.components]
    assert sterak and all(s.priority == 2 for s in sterak)
    # Low HP slams anyway.
    low = plan_items(st, sd, comp=top, hp_bucket="low")
    assert any(s.priority == 1 and "暴风大剑" in s.components for s in low)


def test_no_half_item_hint_for_a_carry_with_full_slots(set_data):
    # A lone B.F. Sword is half of the carry's Infinity Edge: say what is missing,
    # unless the carry already holds three items (nothing more fits).
    comp = CompSuggestion(name="X", score=0.5, carry="Lucian", carry_items=["Infinity Edge"])
    open_slots = make_state("3-2", board=[("Lucian", 2, (), 3, 1)], item_bench=["B.F. Sword"])
    assert any("还差" in s.reason for s in plan_items(open_slots, set_data, comp=comp))
    full = make_state(
        "3-2",
        board=[("Lucian", 2, ("Last Whisper", "Giant Slayer", "Deathblade"), 3, 1)],
        item_bench=["B.F. Sword"],
    )
    assert not any("还差" in s.reason for s in plan_items(full, set_data, comp=comp))


# ---------------------------------------------------------------------------
# Holders follow the comp library's item plan; no slam for a benched unit
# ---------------------------------------------------------------------------

APH_NID = [("Aphelios", 1, ("Red Buff",)), "Nidalee", "Varus", "Diana", "Kog'Maw", "Vi", "Amumu"]


@pytest.mark.parametrize(
    "components, item",
    [
        (["B.F. Sword", "Giant's Belt"], "斯特拉克的挑战护手"),
        (["B.F. Sword", "Chain Vest"], "夜之锋刃"),
        (["Negatron Cloak", "Sparring Gloves"], "水银"),
    ],
)
def test_s18_bruiser_and_flex_items_skip_the_carry_that_needs_its_slots(components, item):
    out = _s18_plan(s18_state("4-1", board=APH_NID, items=components, level=7, gold=30, hp=60))
    assert out.comps[0].name == "厄斐琉斯 奈德丽"
    made = next(s for s in out.items if s.item == item)
    # Aphelios holds Red Buff and still needs Deathblade and Giant Slayer.
    assert made.holder == "奈德丽"


def test_s18_item_holders_plan_decides_the_holder():
    board = ["Yorick", "Master Yi", "Rengar", "Vi", "Sett", ("Nidalee", 1, ("Guinsoo's Rageblade",)), "Kog'Maw"]
    out = _s18_plan(s18_state("4-1", board=board, items=["Recurve Bow", "Chain Vest"], level=7, gold=30, hp=60))
    assert out.comps[0].name == "易 雷恩加尔"
    titan = next(s for s in out.items if s.item == "泰坦的坚决")
    assert titan.holder in ("易", "雷恩加尔")


def test_s18_carry_still_takes_its_own_items_and_core_damage_items():
    out = _s18_plan(s18_state("4-1", board=APH_NID, items=["B.F. Sword", "B.F. Sword"], level=7, gold=30, hp=60))
    assert next(s for s in out.items if s.item == "死亡之刃").holder == "厄斐琉斯"  # a carry item
    # Infinity Edge is Nidalee's in the library plan.
    out = _s18_plan(s18_state("4-1", board=APH_NID, items=["B.F. Sword", "Sparring Gloves"], level=7, gold=30, hp=60))
    assert next(s for s in out.items if s.item == "无尽之刃").holder == "奈德丽"
    # Without a library plan for it, a core damage item still suits the carry.
    out = _s18_plan(s18_state("4-1", board=APH_NID, items=["Needlessly Large Rod", "Needlessly Large Rod"], level=7, gold=30, hp=60))
    assert all(s.holder != "厄斐琉斯" for s in out.items)  # AP item on an AD carry: never


def test_s18_ad_ap_item_for_a_benched_unit_is_not_slammed_now():
    st = s18_state("2-5", board=["Karma", "Morgana", "Krug", "Sett", "Pebbles"], bench=["Warwick"],
                   items=["Sparring Gloves", "Giant's Belt"], level=5, gold=20, hp=80)
    out = _s18_plan(st)
    flail = next(s for s in out.items if s.item == "强袭者的链枷")
    if flail.holder == "沃里克":
        assert flail.priority == 2 and "上场" in flail.reason
    else:
        assert flail.holder in [u.name for u in st.board]


def test_fielded_unit_of_the_right_profile_beats_a_benched_one(set_data):
    st = make_state("3-2", board=[("Graves", 1, (), 3, 1), ("Braum", 1, (), 0, 3)], bench=[("Draven", 2)],
                    item_bench=["B.F. Sword", "Recurve Bow"])
    out = plan_items(st, set_data, hp_bucket="medium")
    assert out[0].holder == "Graves" and out[0].priority == 1
    only_bench = make_state("3-2", board=[("Braum", 1, (), 0, 3)], bench=[("Draven", 2)], item_bench=["B.F. Sword", "Recurve Bow"])
    out = plan_items(only_bench, set_data, hp_bucket="medium")
    assert out[0].holder == "Draven" and out[0].priority == 2 and "上场" in out[0].reason


def test_s18_duplicate_of_a_held_carry_item_goes_to_another_unit():
    # Aphelios holds Infinity Edge + Striker's Flail and still misses Kraken's
    # Fury: a second Flail must not take the carry's last slot.
    st = s18_state(
        "4-3",
        board=[("厄斐琉斯", 2, ("无尽之刃", "强袭者的链枷"), 3, 6), ("韦鲁斯", 2), ("霞", 2), ("洛", 2, (), 0, 3),
               ("深红锋喙鸟", 2), ("苍蓝雕纹魔像", 1, (), 0, 2), ("阿木木", 1, (), 0, 4), ("凯特琳", 1)],
        items=["巨人腰带", "拳套"], gold=30, level=8, hp=60, xp_current=0,
    )
    out = _s18_plan(st)
    assert out.comps[0].carry == "厄斐琉斯"
    flail = next(s for s in out.items if s.item == "强袭者的链枷")
    assert flail.holder and flail.holder != "厄斐琉斯"
    assert "主C核心装备" not in flail.reason
    # The carry's free slot stays open for its missing carry item.
    from tft_advisor.advisor.rules import RulesAdvisor
    from tft_advisor.data.mechanics import load_mechanics
    from tft_advisor.models import ScreenType, StageRound

    carousel = st.model_copy(deep=True)
    carousel.stage, carousel.screen_type = StageRound.parse("4-4"), ScreenType.CAROUSEL
    adv = RulesAdvisor(mech=load_mechanics(), set_data=s18_set_data()).advise(carousel, _s18_plan(carousel))
    assert "装备已满" not in adv.headline and "海妖之怒" in " ".join(a.text for a in adv.actions)
