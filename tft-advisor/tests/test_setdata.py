from __future__ import annotations

import copy
import json

import pytest

from tft_advisor.config import DataConfig
from tft_advisor.data.setdata import SetData, bundled_sample, load_set_data, normalize_name

from .conftest import chinese_sample


def test_set_selection_prefers_current_standard_set(set_data):
    assert set_data.set_number == 99
    assert len(set_data.champions) == 20  # tutorial / old set entries ignored
    old = SetData.from_cdragon(bundled_sample(), set_number=98)
    assert old.set_number == 98 and len(old.champions) == 3


def test_pick_ignores_tutorial_even_if_listed_first():
    data = copy.deepcopy(bundled_sample())
    data["setData"].sort(key=lambda e: 0 if e["mutator"] == "TFTTutorial" else 1)
    assert SetData.from_cdragon(data).set_number == 99


def test_dummy_and_special_units_filtered(set_data):
    assert "TFT99_TrainingDummy" not in set_data.champions  # no traits
    assert "TFT99_ArmoryKeyA" not in set_data.champions  # cost 8
    assert all(1 <= c.cost <= 5 and c.traits for c in set_data.champions.values())
    assert set_data.champion_count_by_cost() == {1: 5, 2: 5, 3: 5, 4: 3, 5: 2}


def test_components(set_data):
    comps = set_data.components()
    assert len(comps) == 10
    assert {c.name_en for c in comps} >= {"B.F. Sword", "Sparring Gloves", "Spatula", "Frying Pan"}
    assert all(c.kind == "component" for c in comps)


def test_recipe_conflict_prefers_core_item(set_data):
    item = set_data.recipe("B.F. Sword", "B.F. Sword")
    assert item is not None and item.api_name == "TFT_Item_Deathblade"
    assert "TFT9_Item_OrnnDeathsDefiance" not in set_data.items
    # Tutorial items are blocklisted: BF + Bow is Giant Slayer.
    assert set_data.recipe("B.F. Sword", "Recurve Bow").name == "Giant Slayer"
    assert "TFT_Item_TutorialSword" not in set_data.items
    # Order and naming style do not matter.
    assert set_data.recipe("TFT_Item_SparringGloves", "bf sword").name == "Infinity Edge"
    assert set_data.recipe("B.F. Sword", "Nonsense") is None


def test_emblems(set_data):
    assert "TFT8_Item_ExoPrimeEmblemItem" not in set_data.items  # other set's emblem dropped
    assert set_data.recipe("Spatula", "Chain Vest") is None
    knight = set_data.items["TFT99_Item_KnightEmblemItem"]
    assert knight.kind == "emblem"
    assert set_data.recipe("Spatula", "B.F. Sword").api_name == "TFT99_Item_KnightEmblemItem"
    assert set_data.recipe("Frying Pan", "B.F. Sword").name == "Blademaster Emblem"
    assert set_data.recipe("Spatula", "Spatula").name == "Tactician's Crown"


def test_items_without_recipe_dropped(set_data):
    assert "TFT5_Item_DeathbladeRadiant" not in set_data.items
    assert all(len(i.composition) == 2 for i in set_data.completed_items())


@pytest.mark.parametrize(
    "raw,api",
    [
        ("Miss Fortune", "TFT99_MissFortune"),
        ("missfortune", "TFT99_MissFortune"),
        ("  MISS  FORTUNE ", "TFT99_MissFortune"),
        ("Miss Fortun", "TFT99_MissFortune"),
        ("TFT99_MissFortune", "TFT99_MissFortune"),
        ("Graves", "TFT99_Graves"),
        ("Gravse", "TFT99_Graves"),
        ("Katarina", "TFT99_Katarina"),
    ],
)
def test_resolve_champion(set_data, raw, api):
    champ = set_data.resolve_champion(raw)
    assert champ is not None and champ.api_name == api


@pytest.mark.parametrize("raw", ["", None, "x", "Zzzzzzz", "Training Dummy", "Armory Key"])
def test_resolve_champion_misses(set_data, raw):
    assert set_data.resolve_champion(raw) is None


def test_resolve_items_and_traits(set_data):
    assert set_data.resolve_item("infinity edge").api_name == "TFT_Item_InfinityEdge"
    assert set_data.resolve_item("Infinity Edg").api_name == "TFT_Item_InfinityEdge"
    assert set_data.resolve_item("Sunfire Cape").api_name == "TFT_Item_RedBuff"
    assert set_data.resolve_item("Red Buff").name == "Red Buff"
    assert set_data.resolve_item("Qqqqqqq") is None
    assert set_data.resolve_trait("gunslinger").api_name == "TFT99_Gunslinger"
    assert set_data.resolve_trait("Gunslinger").breakpoints == [2, 4]
    assert [c.name for c in set_data.trait_units("Gunslinger")] == ["Graves", "Lucian", "Miss Fortune"]


def test_display_names_beat_api_tails_regardless_of_order():
    data = copy.deepcopy(bundled_sample())
    data["items"] = list(reversed(data["items"]))
    sd = SetData.from_cdragon(data)
    assert sd.resolve_item("Red Buff").name == "Red Buff"


def test_queries(set_data):
    ones = set_data.champions_by_cost(1)
    assert [c.name_en for c in ones] == sorted(c.name_en for c in ones)
    assert set_data.champions_by_cost(9) == []
    assert "Miss Fortune" in set_data.champion_names()
    assert "Infinity Edge" in set_data.item_names()
    assert normalize_name("Miss  Fortune!") == "missfortune"
    assert normalize_name("厄运 小姐") == "厄运小姐"
    assert normalize_name("ＡＢＣ") == "abc"


# ---------------------------------------------------------------------------
# Chinese locale
# ---------------------------------------------------------------------------


def test_chinese_locale(zh_set_data):
    mf = zh_set_data.resolve_champion("厄运小姐")
    assert mf is not None and mf.api_name == "TFT99_MissFortune"
    assert mf.name == "厄运小姐" and mf.name_en == "Miss Fortune"
    assert mf.traits == ["枪手", "海盗"] and mf.traits_en == ["Gunslinger", "Pirate"]
    assert zh_set_data.resolve_champion("Miss Fortune").name == "厄运小姐"
    assert zh_set_data.resolve_champion("厄运小").api_name == "TFT99_MissFortune"  # fuzzy CJK
    assert zh_set_data.resolve_champion("格雷福斯").api_name == "TFT99_Graves"
    assert zh_set_data.resolve_champion("小姐") is None  # too different for CJK cutoff

    ie = zh_set_data.recipe("暴风大剑", "拳套")
    assert ie.name == "无尽之刃" and ie.name_en == "Infinity Edge"
    assert zh_set_data.recipe("B.F. Sword", "Sparring Gloves").name == "无尽之刃"
    assert zh_set_data.resolve_item("无尽之刃").api_name == "TFT_Item_InfinityEdge"
    assert zh_set_data.resolve_trait("枪手").name_en == "Gunslinger"
    assert zh_set_data.resolve_trait("Gunslinger").name == "枪手"
    assert [c.name for c in zh_set_data.trait_units("枪手")] == ["格雷福斯", "卢锡安", "厄运小姐"]


def test_chinese_summary_lists_both_names(zh_set_data):
    text = zh_set_data.summary_text()
    assert "厄运小姐 / Miss Fortune" in text
    assert "暴风大剑 + 拳套 = 无尽之刃 / Infinity Edge" in text
    assert "枪手 / Gunslinger: 2/4" in text


# ---------------------------------------------------------------------------
# summary_text / loading
# ---------------------------------------------------------------------------


def test_summary_text_deterministic(set_data):
    again = SetData.from_cdragon(bundled_sample())
    text = set_data.summary_text()
    assert text == set_data.summary_text() == again.summary_text()
    assert text.startswith("TFT Set 99")
    assert "B.F. Sword + B.F. Sword = Deathblade" in text
    assert "- Miss Fortune: Gunslinger, Pirate" in text
    assert text.index("[1 cost]") < text.index("[2 cost]") < text.index("[5 cost]")
    assert "Training Dummy" not in text and "Death's Defiance" not in text


def test_summary_text_independent_of_input_order():
    data = copy.deepcopy(bundled_sample())
    for entry in data["setData"]:
        entry["champions"] = list(reversed(entry["champions"]))
        entry["traits"] = list(reversed(entry["traits"]))
    assert SetData.from_cdragon(data).summary_text() == SetData.from_cdragon(bundled_sample()).summary_text()


def test_load_set_data_from_cache(tmp_path):
    (tmp_path / "cdragon_tft_zh_cn.json").write_text(json.dumps(chinese_sample(), ensure_ascii=False), encoding="utf-8")
    (tmp_path / "cdragon_tft_en_us.json").write_text(json.dumps(bundled_sample()), encoding="utf-8")
    logs: list[str] = []
    sd = load_set_data(DataConfig(cache_dir=str(tmp_path)), offline=True, log=logs.append)
    assert sd.set_number == 99
    assert sd.resolve_champion("Miss Fortune").name == "厄运小姐"
    assert sd.source.endswith("cdragon_tft_zh_cn.json")


def test_load_set_data_offline_without_cache(tmp_path):
    logs: list[str] = []
    sd = load_set_data(DataConfig(cache_dir=str(tmp_path / "empty")), offline=True, log=logs.append)
    assert sd.champions
    assert logs  # tells the player which fallback is used
    if sd.source == "bundled-sample":
        assert sd.set_number == 99
