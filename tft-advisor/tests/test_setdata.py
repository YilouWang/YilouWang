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


# ---------------------------------------------------------------------------
# CommunityDragon download: gzip, conditional requests, deadline, backoff
# ---------------------------------------------------------------------------


class _Server:
    """Loopback HTTP server: ``mode`` = "ok" (gzip JSON + ETag, 304 on a
    matching If-None-Match), "stall" (accepts, never answers) or "drip"
    (answers one byte at a time)."""

    def __init__(self, mode: str = "ok") -> None:
        import gzip
        import http.server
        import socket
        import threading

        self.mode = mode
        self.requests: list[dict] = []
        self.body = gzip.compress(json.dumps(bundled_sample()).encode("utf-8"))
        outer = self
        if mode == "stall":
            self.sock = socket.socket()
            self.sock.bind(("127.0.0.1", 0))
            self.sock.listen(8)
            self.port = self.sock.getsockname()[1]
            self.conns: list = []

            def accept() -> None:
                while True:
                    try:
                        conn, _ = self.sock.accept()
                    except OSError:
                        return
                    outer.requests.append({})
                    self.conns.append(conn)

            threading.Thread(target=accept, daemon=True).start()
            return

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a) -> None:
                pass

            def do_GET(self) -> None:
                outer.requests.append(dict(self.headers))
                if outer.mode == "ok" and self.headers.get("If-None-Match") == '"v1"':
                    self.send_response(304)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Encoding", "gzip")
                self.send_header("ETag", '"v1"')
                self.send_header("Content-Length", str(len(outer.body)))
                self.end_headers()
                import time as _t

                for i in range(0, len(outer.body), 1 if outer.mode == "drip" else 65536):
                    try:
                        self.wfile.write(outer.body[i : i + (1 if outer.mode == "drip" else 65536)])
                        self.wfile.flush()
                    except OSError:
                        return
                    if outer.mode == "drip":
                        _t.sleep(0.05)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        if self.mode == "stall":
            for c in self.conns:
                c.close()
            self.sock.close()
        else:
            self.httpd.shutdown()
            self.httpd.server_close()


def test_download_uses_gzip_and_conditional_requests(tmp_path):
    import os
    import time

    srv = _Server("ok")
    try:
        cfg = DataConfig(cache_dir=str(tmp_path), locale="en_us", cdragon_base=srv.base)
        logs: list[str] = []
        sd = load_set_data(cfg, log=logs.append)
        assert sd.set_number == 99 and srv.requests[0].get("Accept-Encoding") == "gzip"
        cached = tmp_path / "cdragon_tft_en_us.json"
        assert json.loads(cached.read_text(encoding="utf-8"))["setData"]  # stored decompressed
        old = time.time() - 3 * 86400
        os.utime(cached, (old, old))
        load_set_data(cfg, log=logs.append)  # stale: conditional request, 304
        assert srv.requests[-1].get("If-None-Match") == '"v1"'
        assert time.time() - cached.stat().st_mtime < 60  # the refresh clock restarted
        assert len(srv.requests) == 2
    finally:
        srv.close()


def test_stalled_network_fails_fast_and_backs_off(tmp_path, monkeypatch):
    import time

    import tft_advisor.data.setdata as setdata

    monkeypatch.setattr(setdata, "SOCKET_TIMEOUT_S", 0.3)
    srv = _Server("stall")
    try:
        cfg = DataConfig(cache_dir=str(tmp_path), cdragon_base=srv.base)
        logs: list[str] = []
        t0 = time.monotonic()
        sd = load_set_data(cfg, log=logs.append)
        assert time.monotonic() - t0 < 5
        assert sd.champions and sd.source != str(tmp_path / "cdragon_tft_zh_cn.json")
        assert len(srv.requests) == 1  # the second locale is not tried after a failure
        assert (tmp_path / setdata.ATTEMPT_STAMP).is_file()
        assert any("不再自动重试" in x for x in logs)
        # The next launch does not wait again.
        logs.clear()
        load_set_data(cfg, log=logs.append)
        assert len(srv.requests) == 1 and any("暂不重试" in x for x in logs)
    finally:
        srv.close()


def test_download_deadline_leaves_no_partial_cache(tmp_path, monkeypatch):
    import tft_advisor.data.setdata as setdata

    srv = _Server("drip")
    try:
        with pytest.raises(TimeoutError):
            setdata._download(f"{srv.base}/en_us.json", tmp_path / "x.json", deadline_s=0.3)
        assert not (tmp_path / "x.json").exists()
    finally:
        srv.close()


def test_background_refresh_does_not_block(tmp_path):
    import time

    srv = _Server("ok")
    try:
        cfg = DataConfig(cache_dir=str(tmp_path), locale="en_us", cdragon_base=srv.base)
        logs: list[str] = []
        sd = load_set_data(cfg, log=logs.append, background=True)
        assert sd.champions  # the bundled snapshot right away
        assert any("后台下载" in x for x in logs) and not any("data update" in x for x in logs)
        cached = tmp_path / "cdragon_tft_en_us.json"
        for _ in range(100):
            if cached.is_file():
                break
            time.sleep(0.05)
        assert cached.is_file()  # used on the next start
    finally:
        srv.close()


def test_config_example_comments_sit_on_their_own_key():
    from importlib import resources

    text = resources.files("tft_advisor.data").joinpath("bundled", "config.example.toml").read_text(encoding="utf-8")
    for line in text.splitlines():
        assert line.count("#") <= 1 or line.lstrip().startswith("#"), line
    ocr = next(line for line in text.splitlines() if line.startswith("ocr_crosscheck"))
    assert "OCR" in ocr.split("#", 1)[1]
    reserve = next(line for line in text.splitlines() if line.startswith("shop_reserve_calls"))
    assert "OCR" not in reserve


# ---------------------------------------------------------------------------
# Augment picker: timing written in the effect, HP, effect-based labels
# ---------------------------------------------------------------------------


def _aug():
    from tft_advisor.data.augments import AugmentData, pick_augment

    return AugmentData.load(18), pick_augment


@pytest.mark.parametrize(
    "choices, stage, rnd, level, hp, want, bad",
    [
        (["Epic Rolldown", "Ascension", "Clockwork Accelerator"], 4, 2, 8, 60, None, "Epic Rolldown"),  # already level 8
        (["游神的眷顾", "飞升", "星界赐福 II"], 4, 2, 8, 60, None, "游神的眷顾"),  # levels 5-8 all passed
        (["强化之能量", "认知税", "烧起来"], 4, 2, 7, 60, None, "强化之能量"),  # no augment after 4-2
        (["休眠锻炉", "集中火力", "星界赐福 I"], 4, 2, None, 25, None, "休眠锻炉"),  # 8 fights away at 25 HP
        (["认知税", "集中火力", "星界赐福 I"], 3, 2, 6, 30, None, "认知税"),  # low HP: no economy at 3-2
        (["手气不错", "拥抱 I", "应急护甲 I"], 3, 2, 6, 30, None, "手气不错"),
        (["Epic Rolldown", "Ascension", "Clockwork Accelerator"], 4, 2, 7, 60, "Epic Rolldown", None),  # level 8 ahead
        (["休眠锻炉", "集中火力", "星界赐福 I"], 4, 2, 7, 80, "休眠锻炉", None),  # healthy: the forge is fine
        (["认知税", "集中火力", "星界赐福 I"], 2, 1, 4, 100, "认知税", None),  # early and healthy: economy
    ],
)
def test_augment_pick_reads_the_effect_timing(choices, stage, rnd, level, hp, want, bad):
    data, pick = _aug()
    name, reason = pick(choices, data, stage, hp, [], level=level, round_no=rnd, augment_rounds=("2-1", "3-2", "4-2"))
    if want:
        assert name == want
    if bad:
        assert name != bad
    assert reason and "—" not in reason


def test_augment_pick_without_level_estimates_it_from_the_stage():
    data, pick = _aug()
    # "Next augment one tier higher" is dead at stage 4 even without the round.
    name, _reason = pick(["强化之能量", "认知税", "烧起来"], data, 4, 60, [])
    assert name != "强化之能量"


def test_augment_labels_come_from_the_effect():
    from tft_advisor.data.augments import effect_kinds

    data, pick = _aug()
    assert effect_kinds(data.lookup("宝宝学院")) == ["战力"]  # team AD/AP, not rolling
    assert "搜牌" in effect_kinds(data.lookup("英勇福袋"))  # duplicators
    assert "装备" in effect_kinds(data.lookup("休眠锻炉"))
    assert "装备" not in effect_kinds(data.lookup("应急护甲 I"))  # "units without items" is not an item
    name, reason = pick(["宝宝学院", "飞升", "发条增速器"], data, 3, 35, [], level=6, round_no=2)
    assert "帮助搜牌" not in reason
    _name, reason = pick(["大而有力"], data, 4, 60, [], level=7, round_no=2)
    assert reason != "彩色" and reason
