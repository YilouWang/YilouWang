"""CLI, config loading and post-game review (all offline, isolated home directory)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image

from tft_advisor import cli
from tft_advisor.config import Config, load_config
from tft_advisor.data.mechanics import load_mechanics
from tft_advisor.models import GameState, StageRound, Unit
from tft_advisor.review import (
    REVIEW_SYSTEM,
    format_summary,
    last_game,
    latest_log,
    llm_review,
    load_records,
    summarize_log,
)

from .fakeapi import FakeAnthropic

FIXTURES = Path(__file__).parent / "fixtures"
EM_DASHES = ("\u2014", "\u2015", "\u2e3a", "\u2e3b")


@pytest.fixture()
def home(tmp_path, monkeypatch) -> Path:
    """Fresh HOME / USERPROFILE / cwd, no API credentials, no config env var."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("USERPROFILE", str(h))
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.delenv("TFT_ADVISOR_CONFIG", raising=False)
    monkeypatch.delenv("TFT_ADVISOR_DEBUG", raising=False)
    for key in list(os.environ):
        if key.startswith("ANTHROPIC_"):
            monkeypatch.delenv(key, raising=False)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return h


@pytest.fixture()
def offline_data(monkeypatch):
    """Commands that normally download CommunityDragon data use the bundled copy."""
    import tft_advisor.data.setdata as setdata

    real = setdata.load_set_data
    monkeypatch.setattr(
        setdata, "load_set_data", lambda cfg, offline=False, log=print, background=False: real(cfg, offline=True, log=log)
    )


def write(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.write_bytes(text.encode(encoding))
    return path


def no_dashes(text: str) -> bool:
    return not any(d in text for d in EM_DASHES)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "toml,needle",
    [
        ("[bogus]\nx = 1\n", "[bogus]"),
        ("[ui]\nnope = 1\n", "[ui].nope"),
        ("[advisor]\nauto = 1\n", "true 或 false"),
        ('[ui]\nport = "abc"\n', "需要是数字"),
        ("[ui]\nport = 80.5\n", "需要是整数"),
        ("[hotkeys]\nanalyze = 6\n", "需要是字符串"),
        ("[ui]\nhost = 8765\n", "需要是字符串"),
        ("[data]\ncache_dir = 5\n", "需要是字符串"),
        ("[capture]\nwindow_title = 5\n", "需要是字符串"),
        ("[ui]\nport = 70000\n", "1 到 65535"),
        ("[capture]\nchange_threshold = 0\n", "[capture].change_threshold"),
        ("[capture]\npoll_interval_s = 0\n", "[capture].poll_interval_s"),
        ("[anthropic]\ntimeout_s = 0\n", "[anthropic].timeout_s"),
        ("[anthropic]\nmax_image_edge = 100\n", "[anthropic].max_image_edge"),
        ('[anthropic]\nvision_effort = "huge"\n', "low / medium"),
        ('[hotkeys]\nscout = "F99x"\n', "[hotkeys].scout"),
        ('[data]\ncdragon_base = "http://attacker.example/cdragon"\n', "https://"),
        ('[data]\ncdragon_base = "file:///etc"\n', "https://"),
        ('[capture]\nscreenshot_dir = "//attacker.example/share/caps"\n', "网络共享"),
        ('[data]\ncache_dir = "\\\\\\\\server\\\\share"\n', "网络共享"),
        ("[ui]\nport = 8765\n[ui\n", "格式错误"),
    ],
)
def test_bad_config_raises_chinese_value_error(home, tmp_path, toml, needle):
    p = write(tmp_path / "bad.toml", toml)
    with pytest.raises(ValueError) as exc:
        load_config(p)
    assert needle in str(exc.value) and str(p) in str(exc.value)


def test_bad_config_is_one_line_on_stderr_not_a_traceback(home, tmp_path, capsys):
    p = write(tmp_path / "bad.toml", '[ui]\nport = "abc"\n')
    assert cli.main(["doctor", "--offline", "--config", str(p)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("错误: ") and "Traceback" not in err and "[ui].port" in err


def test_debug_env_shows_traceback(home, tmp_path, monkeypatch):
    monkeypatch.setenv("TFT_ADVISOR_DEBUG", "1")
    p = write(tmp_path / "bad.toml", '[ui]\nport = "abc"\n')
    with pytest.raises(ValueError):
        cli.main(["doctor", "--offline", "--config", str(p)])


def test_bom_and_gbk_files(home, tmp_path):
    bom = tmp_path / "bom.toml"
    bom.write_bytes(b"\xef\xbb\xbf[ui]\nport = 9000\n")
    assert load_config(bom).ui.port == 9000
    gbk = write(tmp_path / "gbk.toml", '# 中文注释\n[advisor]\ncomp_hint = "法师"\n', encoding="gbk")
    with pytest.raises(ValueError, match="UTF-8"):
        load_config(gbk)


def test_float_port_becomes_int(home, tmp_path):
    cfg = load_config(write(tmp_path / "c.toml", "[ui]\nport = 9000.0\n"))
    assert cfg.ui.port == 9000 and isinstance(cfg.ui.port, int)


def test_missing_explicit_and_env_config_raise(home, tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="找不到配置文件"):
        load_config(tmp_path / "nope.toml")
    monkeypatch.setenv("TFT_ADVISOR_CONFIG", str(tmp_path / "missing.toml"))
    with pytest.raises(FileNotFoundError, match="TFT_ADVISOR_CONFIG"):
        load_config()


def test_lookup_order_env_then_home_and_cwd_is_ignored(home, tmp_path, monkeypatch):
    assert load_config().source_path is None
    (home / ".tft_advisor").mkdir()
    write(home / ".tft_advisor" / "config.toml", "[ui]\nport = 9001\n")
    # A config shipped in the current directory must never be picked up silently.
    write(Path.cwd() / "tft_advisor.toml", '[ui]\nhost = "0.0.0.0"\nport = 9002\n')
    cfg = load_config()
    assert cfg.ui.port == 9001 and cfg.ui.host == "127.0.0.1"
    env = write(tmp_path / "env.toml", "[ui]\nport = 9003\n")
    monkeypatch.setenv("TFT_ADVISOR_CONFIG", str(env))
    assert load_config().ui.port == 9003
    assert load_config(write(tmp_path / "x.toml", "[ui]\nport = 9004\n")).ui.port == 9004


def test_example_config_matches_defaults(home):
    from importlib import resources

    path = Path(str(resources.files("tft_advisor.data").joinpath("bundled", "config.example.toml")))
    loaded = load_config(path).to_dict()
    defaults = Config().to_dict()
    loaded.pop("source_path")
    defaults.pop("source_path")
    assert loaded == defaults
    # Every key of the dataclasses is documented in the example (except the data source URL).
    text = path.read_text(encoding="utf-8")
    for section, values in defaults.items():
        for key in values:
            if (section, key) != ("data", "cdragon_base"):
                assert f"\n{key} = " in text, f"[{section}].{key} missing from config.example.toml"


def test_cli_overrides_are_validated(home, capsys):
    assert cli.main(["run", "--port", "70000"]) == 2
    assert "1 到 65535" in capsys.readouterr().err


def test_config_flag_after_subcommand(home, tmp_path, capsys):
    p = write(tmp_path / "c.toml", "[ui]\nport = 9100\n")
    assert cli.main(["doctor", "--offline", "--config", str(p)]) == 0
    assert str(p) in capsys.readouterr().out
    assert cli.main(["--config", str(p), "doctor", "--offline"]) == 0
    assert str(p) in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "tft-advisor" in capsys.readouterr().out


def test_odds_prints_table(home, capsys):
    assert cli.main(["odds", "--level", "8", "--cost", "4", "--have", "1", "--star", "2", "--taken", "2", "--gold", "60", "--xp", "20"]) == 0
    out = capsys.readouterr().out
    assert "期望花费" in out and "当前等级搜 vs 先升级再搜" in out and no_dashes(out)


def test_init_writes_then_refuses_and_parses(home, capsys):
    assert cli.main(["init"]) == 0
    dest = home / ".tft_advisor" / "config.toml"
    assert dest.is_file()
    assert cli.main(["init"]) == 1
    assert load_config().source_path == str(dest)
    other = home / "elsewhere.toml"
    assert cli.main(["init", "--path", str(other)]) == 0
    assert "--config" in capsys.readouterr().out
    assert cli.main(["init", "--path", str(home), "--force"]) == 2
    assert "是一个目录" in capsys.readouterr().err


def test_data_show_offline(home, capsys):
    assert cli.main(["data", "show", "--offline"]) == 0
    assert "赛季" in capsys.readouterr().out


def test_doctor_offline_never_raises_and_explains_capture(home, capsys, monkeypatch):
    import tft_advisor.capture.screen as screen

    class Broken:
        def __init__(self, *a, **k):
            raise RuntimeError("no screen here")

    monkeypatch.setattr(screen, "_new_mss", lambda: Broken())
    assert cli.main(["doctor", "--offline"]) == 0
    out = capsys.readouterr().out
    assert "赛季数据" in out and "配置文件" in out
    capture_line = next(line for line in out.splitlines() if "截屏" in line)
    assert "no screen here" in capture_line


def test_calibrate_from_image_and_bad_inputs(home, tmp_path, capsys):
    img = tmp_path / "shot.png"
    Image.new("RGB", (1280, 720), (30, 30, 30)).save(img)
    out = tmp_path / "sub" / "out.png"
    assert cli.main(["calibrate", "--image", str(img), "--out", str(out)]) == 0
    assert out.is_file()
    assert cli.main(["calibrate", "--image", str(tmp_path / "missing.png")]) == 2
    bad = write(tmp_path / "bad.png", "not a png")
    assert cli.main(["calibrate", "--image", str(bad)]) == 2
    assert cli.main(["calibrate", "--image", str(img), "--out", str(tmp_path / "x.unknownext")]) == 2
    err = capsys.readouterr().err
    assert "找不到图片" in err and "无法读取图片" in err and ".png" in err and "Traceback" not in err


def test_calibrate_capture_failure_shows_reason(home, tmp_path, capsys, monkeypatch):
    import tft_advisor.capture.screen as screen

    class Broken:
        def __init__(self, *a, **k):
            raise RuntimeError("no screen here")

    monkeypatch.setattr(screen, "_new_mss", lambda: Broken())
    assert cli.main(["calibrate", "--out", str(tmp_path / "o.png")]) == 1
    assert "no screen here" in capsys.readouterr().out


def test_demo_writes_its_log_outside_the_review_dir(home, capsys):
    assert cli.main(["demo", "--steps", "10", "--interval", "0", "--no-dashboard"]) == 0
    logs = home / ".tft_advisor" / "logs"
    assert latest_log(logs) is None
    assert list((logs / "demo").glob("game-*.jsonl"))
    out = capsys.readouterr().out
    assert "配置文件" in out and no_dashes(out)
    assert cli.main(["review"]) == 1  # the demo is never reviewed as "the last game"


def test_demo_missing_fixtures(home, capsys):
    assert cli.main(["demo", "--fixtures", "missing-dir", "--steps", "1", "--no-dashboard"]) == 2
    assert "找不到演示数据" in capsys.readouterr().err


def _game_log(home: Path) -> Path:
    """A real-format game log (GameState dumps, the shape GameLogger writes)."""
    logs = home / ".tft_advisor" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    rows = [("2-1", 100, 12, 4), ("3-2", 70, 0, 6), ("3-3", 55, 8, 6), ("4-1", 40, 0, 7)]
    path = logs / "game-20260927-100000-000.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for stage, hp, gold, level in rows:
            st = GameState(game_id="g1", stage=StageRound.parse(stage), hp=hp, gold=gold, level=level)
            st.board = [Unit(api_name="TFT_Ahri", name="阿狸", star=2)]
            rec = {"game_id": "g1", "purpose": "auto", "state": st.model_dump(mode="json"), "advice": {"headline": f"建议{stage}"}}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def test_review_prints_summary(home, capsys):
    assert cli.main(["review"]) == 1
    _game_log(home)
    assert cli.main(["review"]) == 0
    out = capsys.readouterr().out
    assert "升级时间" in out and "最终阵容" in out and "阿狸★★" in out and no_dashes(out)
    row = next(line for line in out.splitlines() if line.strip().startswith("3-2"))
    assert row.split()[:4] == ["3-2", "70", "0", "6"]


def test_review_llm_without_key_is_skipped(home, capsys):
    _game_log(home)
    assert cli.main(["review", "--llm"]) == 0
    assert "跳过 Claude 复盘" in capsys.readouterr().out


def test_review_llm_unexpected_error_is_reported(home, capsys, monkeypatch):
    import tft_advisor.llm as llm_mod
    import tft_advisor.review as review_mod

    _game_log(home)
    monkeypatch.setattr(llm_mod, "has_credentials", lambda: True)

    def boom(*a, **k):
        raise TypeError("Could not resolve authentication method")

    monkeypatch.setattr(review_mod, "llm_review", boom)
    assert cli.main(["review", "--llm"]) == 0
    assert "Claude 复盘失败" in capsys.readouterr().out


def test_review_weird_logs_do_not_crash(home, tmp_path, capsys):
    weird = write(tmp_path / "weird.jsonl", "[1, 2]\n\"x\"\n")
    assert cli.main(["review", str(weird)]) == 0
    binary = tmp_path / "bin.jsonl"
    binary.write_bytes(b"\xff\xfe\x00garbage\n")
    assert cli.main(["review", str(binary)]) == 0
    assert "回合数: 0" in capsys.readouterr().out


def test_run_headless_flag_mapping(home, offline_data, monkeypatch):
    from tft_advisor.app import AdvisorApp

    seen = {}

    def fake_wait(self):
        seen.update(auto=self.auto, strategist=self.strategist, ui=self.cfg.ui, llm=self.llm, threads=[t.name for t in self._threads])

    monkeypatch.setattr(AdvisorApp, "wait", fake_wait)
    assert cli.main(["run", "--no-dashboard", "--no-vision-llm", "--no-auto", "--host", "0.0.0.0", "--port", "9123", "--voice"]) == 0
    assert seen["auto"] is False and seen["strategist"] is None and seen["llm"] is None
    assert seen["ui"].host == "0.0.0.0" and seen["ui"].port == 9123 and seen["ui"].voice is True
    assert "worker" in seen["threads"] and "capture" in seen["threads"]


def test_run_dashboard_failure_returns_2_and_stops(home, offline_data, monkeypatch, capsys):
    from tft_advisor.app import AdvisorApp
    from tft_advisor.ui import server

    stopped = []
    real_stop = AdvisorApp.stop

    def record_stop(self):
        stopped.append(True)
        real_stop(self)

    def refuse(self):
        raise OSError("看板无法监听地址 10.99.99.99")

    monkeypatch.setattr(AdvisorApp, "stop", record_stop)
    monkeypatch.setattr(server.DashboardServer, "start", refuse)
    assert cli.main(["run", "--no-vision-llm", "--host", "10.99.99.99"]) == 2
    err = capsys.readouterr().err
    assert "10.99.99.99" in err and "--no-dashboard" in err and "Traceback" not in err
    assert stopped == [True]


def test_replay_labels_follow_readable_files(home, offline_data, tmp_path, capsys):
    shots = tmp_path / "shots"
    shots.mkdir()
    Image.new("RGB", (320, 180)).save(shots / "a.png")
    write(shots / "b.png", "corrupt")
    Image.new("RGB", (320, 180)).save(shots / "c.png")
    assert cli.main(["replay", str(shots), "--mock", str(FIXTURES), "--no-llm"]) == 0
    out = capsys.readouterr().out
    blocks = out.split("=== ")[1:]
    assert [b.split(" ===")[0] for b in blocks] == ["a.png", "b.png", "c.png"]
    assert "无法读取图片" in blocks[1]
    assert "(没有结果" not in blocks[2]
    assert no_dashes(out)


def test_replay_missing_mock(home, tmp_path, capsys):
    Image.new("RGB", (32, 18)).save(tmp_path / "a.png")
    assert cli.main(["replay", str(tmp_path / "a.png"), "--mock", str(tmp_path / "none")]) == 2
    assert "找不到观察数据" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# review.py
# ---------------------------------------------------------------------------


def _rec(stage, hp=None, gold=None, level=None, game="g1", headline=None, board=()):
    st = GameState(game_id=game, stage=StageRound.parse(stage) if stage else None, hp=hp, gold=gold, level=level)
    st.board = [Unit(api_name=f"TFT_{n}", name=n, star=2) for n in board]
    return {"game_id": game, "state": st.model_dump(mode="json"), "advice": {"headline": headline or f"h{stage}"}}


def test_load_records_skips_bad_lines(tmp_path):
    p = tmp_path / "game-1.jsonl"
    lines = [json.dumps(_rec("3-2", 40, 0, 6)), "", "[1, 2]", "42", json.dumps(_rec("3-3", 30, 5, 6))[:-10]]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(p, "ab") as fh:
        fh.write(b"\xff\xfe broken bytes\n")
    recs = load_records(p)
    assert len(recs) == 1 and recs[0]["state"]["hp"] == 40


def test_summarize_log_rounds_levels_hp(tmp_path):
    mech = load_mechanics()
    recs = [
        _rec("3-3", 50, 20, 6, headline="old"),
        _rec("3-2", 70, 30, 6),
        _rec("3-3", 45, 10, 6, headline="new"),
        _rec("4-1", 40, 0, 7, board=["Ahri"]),
        {"game_id": "g1", "purpose": "strategy", "stage": "4-1", "advice": {"headline": "Claude 说升 8"}},
    ]
    s = summarize_log(recs, mech)
    assert [r["stage"] for r in s["timeline"]] == ["3-2", "3-3", "4-1"]
    assert s["timeline"][1]["advice"] == "new" and s["timeline"][1]["hp"] == 45
    assert s["timeline"][2]["advice"] == "Claude 说升 8"
    assert [x["level"] for x in s["level_timing"]] == [6, 7]
    assert s["level_timing"][0]["reached_at"] == "3-2"
    assert s["hp_lost_by_stage"] == {3: 25}
    assert s["final_board"] == ["Ahri★★"] and s["final_hp"] == 40
    text = format_summary(s)
    assert no_dashes(text)
    row = next(line for line in text.splitlines() if line.strip().startswith("4-1"))
    assert row.split()[:4] == ["4-1", "40", "0", "7"]


def test_summarize_log_keeps_only_the_last_game():
    mech = load_mechanics()
    recs = [_rec("5-1", 10, 3, 8, game="a"), _rec("1-2", 100, 2, 1, game="b"), _rec("2-1", 100, 4, 3, game="b")]
    assert [r["game_id"] for r in last_game(recs)] == ["b", "b"]
    s = summarize_log(recs, mech)
    assert s["final_stage"] == "2-1" and s["final_hp"] == 100 and s["rounds"] == 2


def test_latest_log_by_name(tmp_path):
    assert latest_log(tmp_path) is None
    for name in ("game-20260101-000000.jsonl", "game-20260102-000000-500.jsonl", "game-20260102-000000-500-1.jsonl"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    assert latest_log(tmp_path).name == "game-20260102-000000-500-1.jsonl"


def test_llm_review_sanitizes_and_sends_summary():
    from tft_advisor.config import AnthropicConfig
    from tft_advisor.llm import LLM

    summary = summarize_log([_rec("3-2", 70, 30, 6)], load_mechanics())
    with FakeAnthropic() as fake:
        fake.queue_text("做得好\u2014\u2014升级及时\n\n问题：3-2 没升级")
        llm = LLM(AnthropicConfig(), client=fake.client())
        out = llm_review(llm, "m", "medium", summary)
        assert no_dashes(out) and "\n" in out and out.startswith("做得好，升级及时")
        body = fake.requests[0]["body"]
        assert body["model"] == "m"
        assert body["system"][0]["text"] == REVIEW_SYSTEM
        assert llm.stats.by_purpose == {"review": 1}


def test_doctor_api_makes_one_vision_and_one_strategy_call(home, offline_data, monkeypatch, capsys):
    from tft_advisor.models import ScreenObservation, ScreenType

    with FakeAnthropic() as fake:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", fake.base_url)
        fake.queue_json_wire(ScreenObservation(screen_type=ScreenType.OTHER))
        fake.queue_json({"headline": "进入对局后按 F6", "actions": [{"type": "other", "text": "开始对局", "priority": 1}], "plan": "", "confidence": 0.3})
        rc = cli.main(["doctor", "--offline", "--api"])
        out = capsys.readouterr().out
        assert rc == 0, out
        assert "视觉识别" in out and "策略建议" in out and "估算" in out
        assert len(fake.requests) == 2
        vision, strategy = (r["body"] for r in fake.requests)
        assert vision["output_config"]["format"]["type"] == "json_schema"
        assert any(b["type"] == "image" for b in vision["messages"][0]["content"])
        assert "COMP LIBRARY" in strategy["system"][0]["text"]


def test_doctor_api_reports_rejected_request(home, offline_data, monkeypatch, capsys):
    with FakeAnthropic() as fake:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", fake.base_url)
        fake.queue_error(400, "Schema is too complex for compilation")
        fake.queue_error(400, "Schema is too complex for compilation")
        rc = cli.main(["doctor", "--offline", "--api"])
        out = capsys.readouterr().out
        assert rc == 1 and "!!" in out


# ---------------------------------------------------------------------------
# Windows new-user fixes
# ---------------------------------------------------------------------------


def _extra_reqs(extra: str, python_version: str) -> dict[str, str]:
    import tomllib

    from packaging.requirements import Requirement

    data = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8"))
    env = {"python_version": python_version, "python_full_version": python_version + ".0", "extra": extra}
    out = {}
    for text in data["project"]["optional-dependencies"][extra]:
        req = Requirement(text)
        if req.marker is None or req.marker.evaluate(env):
            out[req.name] = str(req.specifier)
    return out


@pytest.mark.parametrize("extra", ["ocr", "all"])
def test_ocr_extra_installs_on_every_supported_python(extra):
    # rapidocr-onnxruntime >= 1.3 declares Requires-Python < 3.13: listing it
    # unconditionally made `pip install -e ".[all]"` fail on 3.13 / 3.14.
    for py in ("3.11", "3.12"):
        reqs = _extra_reqs(extra, py)
        assert "rapidocr-onnxruntime" in reqs and "rapidocr" not in reqs
    for py in ("3.13", "3.14"):
        reqs = _extra_reqs(extra, py)
        assert "rapidocr-onnxruntime" not in reqs
        assert "rapidocr" in reqs and "onnxruntime" in reqs  # rapidocr 3 does not pull onnxruntime itself


def _fake_imports(monkeypatch, failures: dict[str, BaseException]):
    import importlib as real_importlib
    import types

    def import_module(name, *a, **k):
        if name in failures:
            raise failures[name]
        return real_importlib.import_module(name, *a, **k)

    monkeypatch.setattr(cli, "importlib", types.SimpleNamespace(import_module=import_module))


def test_doctor_explains_missing_vc_runtime_for_ocr(home, monkeypatch, capsys):
    dll = ImportError("DLL load failed while importing onnxruntime_pybind11_state: 找不到指定的模块。")
    _fake_imports(monkeypatch, {"rapidocr_onnxruntime": dll, "rapidocr": ModuleNotFoundError("No module named 'rapidocr'", name="rapidocr")})
    assert cli.main(["doctor", "--offline"]) == 0
    out = capsys.readouterr().out
    line = next(x for x in out.splitlines() if "OCR" in x)
    assert "[!!]" in line and "VC++" in line and cli.VC_REDIST_URL in line and "未安装" not in line
    assert no_dashes(out)


def test_doctor_ocr_not_installed_and_broken_optional_module(home, monkeypatch, capsys):
    missing = {m: ModuleNotFoundError(f"No module named '{m}'", name=m) for m in cli.OCR_MODULES}
    missing["pyttsx3"] = ModuleNotFoundError("No module named 'comtypes'", name="comtypes")
    _fake_imports(monkeypatch, missing)
    assert cli.main(["doctor", "--offline"]) == 0
    out = capsys.readouterr().out
    ocr = next(x for x in out.splitlines() if "OCR" in x)
    assert "[OK]" in ocr and "未安装" in ocr and '.[ocr]' in ocr
    tts = next(x for x in out.splitlines() if "pyttsx3" in x)
    assert "[!!]" in tts and "comtypes" in tts  # installed, a dependency is missing: reinstalling pyttsx3 alone will not help


def test_doctor_ocr_accepts_the_rapidocr_successor(home, monkeypatch, capsys):
    import types

    import tft_advisor.vision.ocr as ocr_mod

    fake = types.SimpleNamespace(__version__="3.9.2")
    import importlib as real_importlib

    def import_module(name, *a, **k):
        if name == "rapidocr_onnxruntime":
            raise ModuleNotFoundError("No module named 'rapidocr_onnxruntime'", name="rapidocr_onnxruntime")
        if name == "rapidocr":
            return fake
        return real_importlib.import_module(name, *a, **k)

    monkeypatch.setattr(cli, "importlib", types.SimpleNamespace(import_module=import_module))
    monkeypatch.setattr(ocr_mod, "ocr_available", lambda: True)
    assert cli.main(["doctor", "--offline"]) == 0
    line = next(x for x in capsys.readouterr().out.splitlines() if "OCR" in x)
    assert "[OK]" in line and "rapidocr 3.9.2" in line
    # Installed but the vision module cannot use it: flagged, not reported as fine.
    monkeypatch.setattr(ocr_mod, "ocr_available", lambda: False)
    assert cli.main(["doctor", "--offline"]) == 0
    line = next(x for x in capsys.readouterr().out.splitlines() if "OCR" in x)
    assert "[!!]" in line


def test_no_perceiver_message_names_the_ocr_extra():
    from tft_advisor.app import NO_PERCEIVER_MSG

    assert '.[ocr]' in NO_PERCEIVER_MSG and "rapidocr-onnxruntime" not in NO_PERCEIVER_MSG and no_dashes(NO_PERCEIVER_MSG)


class _Stream:
    def __init__(self, tty: bool):
        self.tty = tty
        self.calls: list[dict] = []

    def isatty(self):
        return self.tty

    def reconfigure(self, **kw):
        self.calls.append(kw)


def test_redirected_output_uses_the_console_code_page_on_windows(monkeypatch):
    import io

    monkeypatch.delenv("MSYSTEM", raising=False)
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)

    enc = cli._stream_encoding
    assert enc(_Stream(True), windows=True, console_cp=936) == "utf-8"  # console: WriteConsoleW anyway
    assert enc(_Stream(False), windows=True, console_cp=936) == "cp936"  # `doctor > x.txt` / `| Select-String`
    assert enc(_Stream(False), windows=True, console_cp=65001) == "utf-8"
    assert enc(_Stream(False), windows=True, console_cp=437) == "utf-8"  # cannot hold Chinese: keep bytes intact
    assert enc(_Stream(False), windows=True, console_cp=0) == "utf-8"  # no console at all
    assert enc(_Stream(False), windows=False, console_cp=0) == "utf-8"
    monkeypatch.setenv("MSYSTEM", "MINGW64")  # Git Bash: a pipe that renders UTF-8
    assert enc(_Stream(False), windows=True, console_cp=936) == "utf-8"
    monkeypatch.delenv("MSYSTEM")
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    assert enc(_Stream(False), windows=True, console_cp=936) is None
    monkeypatch.delenv("PYTHONIOENCODING")
    fake = _Stream(False)
    cli._utf8_console(streams=[fake], windows=True, console_cp=936)
    assert fake.calls == [{"encoding": "cp936", "errors": "replace"}]
    # What PowerShell reads back from the pipe is the original Chinese text.
    buf = io.BytesIO()
    out = io.TextIOWrapper(buf, encoding="utf-8")
    cli._utf8_console(streams=[out], windows=True, console_cp=936)
    out.write("[OK] 截屏: 游戏窗口★")
    out.flush()
    assert buf.getvalue().decode("cp936") == "[OK] 截屏: 游戏窗口★"


def test_run_prints_the_phone_url_in_lan_mode(home, offline_data, monkeypatch, capsys):
    import socket

    from tft_advisor.app import AdvisorApp
    from tft_advisor.ui import server

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setattr(server, "_guess_lan_ip", lambda: "192.168.1.23")
    monkeypatch.setattr(AdvisorApp, "wait", lambda self: None)
    monkeypatch.setattr(cli, "_open", lambda *a, **k: None)
    assert cli.main(["run", "--no-vision-llm", "--no-auto", "--host", "0.0.0.0", "--port", str(port)]) == 0
    out = capsys.readouterr().out
    phone = next(x for x in out.splitlines() if x.startswith("手机访问: "))
    assert f"http://192.168.1.23:{port}/?token=" in phone and no_dashes(phone)
    # The summary block (not only an earlier timestamped log line) names it.
    lines = out.splitlines()
    assert lines.index(phone) == next(i for i, x in enumerate(lines) if x.startswith("看板: ")) + 1


def test_local_run_prints_no_phone_url(home, offline_data, monkeypatch, capsys):
    import socket

    from tft_advisor.app import AdvisorApp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setattr(AdvisorApp, "wait", lambda self: None)
    monkeypatch.setattr(cli, "_open", lambda *a, **k: None)
    assert cli.main(["run", "--no-vision-llm", "--no-auto", "--port", str(port)]) == 0
    assert "手机访问" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "line",
    [
        'comps_file = "C:\\Users\\王一楼\\Documents\\我的阵容.json"',  # \U -> "Invalid hex value"
        'comps_file = "D:\\tft\\my.json"',  # \m -> "Unescaped '\\'"
    ],
)
def test_windows_path_in_double_quotes_gets_a_chinese_hint(home, tmp_path, line):
    p = write(tmp_path / "c.toml", "[data]\n" + line + "\n")
    with pytest.raises(ValueError) as exc:
        load_config(p)
    msg = str(exc.value)
    assert "单引号" in msg and "Windows 路径" in msg and str(p) in msg and no_dashes(msg)


def test_unrelated_toml_error_has_no_path_hint(home, tmp_path):
    p = write(tmp_path / "c.toml", "[ui]\nport = = 3\n")
    with pytest.raises(ValueError) as exc:
        load_config(p)
    assert "单引号" not in str(exc.value)


def test_silently_mangled_windows_path_is_rejected(home, tmp_path, capsys):
    # "D:\tft\new_comps.json" is valid TOML: D:<TAB>ft<LF>ew_comps.json. It used
    # to load, then "找不到阵容文件" dropped the whole comp library.
    p = write(tmp_path / "c.toml", '[data]\ncomps_file = "D:\\tft\\new_comps.json"\n')
    with pytest.raises(ValueError) as exc:
        load_config(p)
    msg = str(exc.value)
    assert "[data].comps_file" in msg and "制表符" in msg
    assert "comps_file = 'D:\\tft\\new_comps.json'" in msg  # the fixed line, ready to paste
    assert "\t" not in msg and "\n" not in msg
    assert cli.main(["doctor", "--offline", "--config", str(p)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("错误: ") and len(err.strip().splitlines()) == 1
    cfg = Config()
    cfg.capture.screenshot_dir = "C:\new\\caps"  # what TOML makes of "C:\new\caps": a real newline
    with pytest.raises(ValueError, match="screenshot_dir"):
        cfg.validate()


def test_single_quoted_windows_path_loads(home, tmp_path):
    cfg = load_config(write(tmp_path / "c.toml", "[data]\ncomps_file = 'C:\\Users\\王一楼\\我的阵容.json'\n"))
    assert cfg.data.comps_file.endswith("C:\\Users\\王一楼\\我的阵容.json")


def test_relative_paths_are_relative_to_the_config_file(home, tmp_path):
    cfg_dir = tmp_path / "cfgdir"
    cfg_dir.mkdir()
    p = write(
        cfg_dir / "config.toml",
        "[data]\ncomps_file = '我的阵容.json'\nmechanics_file = 'patch/m.toml'\ncache_dir = 'cache'\n"
        "[capture]\nscreenshot_dir = '~/shots'\n",
    )
    assert Path.cwd() != cfg_dir  # the home fixture runs from another directory
    cfg = load_config(p)
    assert Path(cfg.data.comps_file) == cfg_dir / "我的阵容.json"
    assert Path(cfg.data.mechanics_file) == cfg_dir / "patch" / "m.toml"
    assert Path(cfg.data.cache_dir) == cfg_dir / "cache" and cfg.cache_dir == cfg_dir / "cache"
    assert cfg.capture.screenshot_dir == "~/shots"  # home-relative stays as written
    absolute = tmp_path / "abs.json"
    cfg = load_config(write(cfg_dir / "c2.toml", f"[data]\ncomps_file = '{absolute}'\n"))
    assert cfg.data.comps_file == str(absolute)
    # A relative --config path works from the current directory too.
    write(Path.cwd() / "local.toml", "[data]\ncomps_file = 'x.json'\n")
    assert Path(load_config("local.toml").data.comps_file) == Path.cwd() / "x.json"


def test_relative_comps_file_is_found_next_to_the_default_config(home, offline_data, capsys):
    from tft_advisor.app import AdvisorApp

    cfg_dir = home / ".tft_advisor"
    cfg_dir.mkdir()
    (cfg_dir / "我的阵容.json").write_text(
        json.dumps({"comps": [{"name": "我的阿狸", "style": "fast8", "units": ["Ahri", "Morgana"], "carry": "Ahri"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    write(cfg_dir / "config.toml", "[data]\ncomps_file = '我的阵容.json'\n")
    app = AdvisorApp(load_config(), perceiver=None, fast_perceiver=None, use_llm=False, console=False, offline_data=True)
    assert [c.name for c in app.comps] == ["我的阿狸"]


# ---------------------------------------------------------------------------
# review.py: metrics the review model can trust
# ---------------------------------------------------------------------------


def _full_rec(stage, gold=None, level=None, econ=None, style="standard", hp=80, streak=None, board=(), augments=(), item_bench=(), comp=None):
    st = GameState(game_id="g1", stage=StageRound.parse(stage), hp=hp, gold=gold, level=level, streak=streak)
    st.board = [Unit(api_name=f"TFT_{n}", name=n, star=s, items=list(items)) for n, s, items in board]
    st.augments = list(augments)
    st.item_bench = list(item_bench)
    analysis = {"econ": {"recommendation": econ, "style": style} if econ else None, "comps": [{"name": comp}] if comp else []}
    return {"game_id": "g1", "state": st.model_dump(mode="json"), "analysis": analysis, "advice": {"headline": f"h{stage}"}}


def test_level_timing_reports_late_early_and_unknown():
    mech = load_mechanics()
    recs = [
        _full_rec("2-1", 10, 4),
        _full_rec("3-2", 34, 5),  # level 5 first seen here; 2-5 was not logged
        _full_rec("3-3", 18, 5),
        _full_rec("3-4", 24, 6),
        _full_rec("4-1", 50, 6),
        _full_rec("4-2", 30, 7),
        _full_rec("4-3", 20, 8),
    ]
    timing = {x["level"]: x for x in summarize_log(recs, mech)["level_timing"]}
    assert timing[4]["standard_round"] == "2-1" and timing[4]["rounds_late"] == 0
    # A gap in the log hides when level 5 came: not called late.
    assert timing[5]["standard_round"] == "2-5" and timing[5]["rounds_late"] is None
    # Regression: this used to read "6级@3-4(标准6)", i.e. on curve.
    assert timing[6]["standard_round"] == "3-2" and timing[6]["rounds_late"] == 2 and timing[6]["below_at"] == ["3-2", "3-3"]
    assert timing[7]["standard_round"] == "4-1" and timing[7]["rounds_late"] == 1
    assert timing[8]["standard_round"] == "4-5" and timing[8]["rounds_late"] == -2
    text = format_summary(summarize_log(recs, mech))
    assert "6级@3-4(标准3-2，至少晚2回合)" in text and "8级@4-3(标准4-5，早2回合)" in text and "5级@3-2(标准2-5)" in text
    assert no_dashes(text)


def test_interest_metric_counts_only_rounds_the_assistant_said_to_save():
    mech = load_mechanics()
    recs = [
        _full_rec("2-1", 12, 4, econ="save"),  # stage 2 cannot be at the cap
        _full_rec("3-2", 34, 5, econ="level"),  # the assistant asked to spend
        _full_rec("3-5", 30, 6, econ="save"),  # 3 interest instead of 5
        _full_rec("4-1", 55, 6, econ="save"),
        _full_rec("4-2", 8, 7, econ="roll"),
    ]
    s = summarize_log(recs, mech)
    assert s["interest_short_on_save_rounds"] == 2 and s["interest_short_rounds"] == ["3-5"]
    assert "interest_missed_stage2_3" not in s
    assert "(第3阶段起，累计): 2（3-5）" in format_summary(s)


def test_interest_metric_skips_saves_that_are_not_about_interest():
    """The econ also says "save" on a PvE round (wait to roll at the next player
    round), at low HP, and when a roll budget went to shop units: following
    that plan is not an interest mistake."""
    mech = load_mechanics()

    def rec(stage, gold, econ, hp=80, picks_cost=0):
        r = _full_rec(stage, gold, 8, econ=econ, hp=hp)
        r["analysis"]["shop_picks_cost"] = picks_cost
        return r

    recs = [
        rec("4-5", 40, "roll", hp=30),
        rec("4-6", 30, "all_in", hp=22),
        rec("4-7", 22, "save", hp=18),  # PvE wait after the all-in plan (low HP too)
    ]
    s = summarize_log(recs, mech)
    assert s["interest_short_on_save_rounds"] == 0 and s["interest_short_rounds"] == []
    for row in (
        rec("3-7", 30, "save", hp=90),  # PvE round, any HP
        rec("4-2", 30, "save", hp=40),  # low HP: the gold is meant for the board
        rec("4-1", 30, "save", hp=90, picks_cost=6),  # the plan buys shop units
    ):
        s = summarize_log([row], mech)
        assert s["interest_short_on_save_rounds"] == 0, row["state"]["stage"]
    # A plain interest save at healthy HP still counts.
    s = summarize_log([rec("4-3", 30, "save", hp=60)], mech)
    assert s["interest_short_on_save_rounds"] == 2 and s["interest_short_rounds"] == ["4-3"]
    assert "PvE" in REVIEW_SYSTEM and "low HP" in REVIEW_SYSTEM


def test_review_payload_has_items_augments_comp_and_explains_fields():
    from tft_advisor.config import AnthropicConfig
    from tft_advisor.llm import LLM

    mech = load_mechanics()
    recs = [
        _full_rec("2-1", 10, 4, econ="save", streak=-2, augments=["潜在锻造"], comp="日蚀骑士 芸阿娜", style="reroll2"),
        _full_rec(
            "3-2", 30, 5, econ="slow_roll", streak=-3, augments=["潜在锻造", "史诗搜牌"], item_bench=["女神之泪"],
            board=[("阿狸", 2, ["珠光护手", "朔极之矛"]), ("瑟提", 1, [])], comp="日蚀骑士 芸阿娜", style="reroll2",
        ),
    ]
    s = summarize_log(recs, mech)
    row = s["timeline"][-1]
    assert row["board"] == ["阿狸★★[珠光护手+朔极之矛]", "瑟提★"]
    assert row["streak"] == -3 and row["comp"] == "日蚀骑士 芸阿娜" and row["style"] == "reroll2" and row["econ"] == "slow_roll"
    assert s["final_item_bench"] == ["女神之泪"]
    assert s["augments"] == [{"name": "潜在锻造", "seen_at": "2-1"}, {"name": "史诗搜牌", "seen_at": "3-2"}]
    text = format_summary(s)
    assert "海克斯: 潜在锻造@2-1, 史诗搜牌@3-2" in text and "阿狸★★[珠光护手+朔极之矛]" in text
    # The prompt explains the derived metrics and forbids criticising what the data does not show.
    for needle in ("rounds_late", "interest_short_on_save_rounds", "reroll", "skip", "up to 3"):
        assert needle in REVIEW_SYSTEM
    with FakeAnthropic() as fake:
        fake.queue_text("做得好：前期连败攒钱")
        llm_review(LLM(AnthropicConfig(), client=fake.client()), "m", "medium", s)
        payload = json.loads(fake.requests[0]["body"]["messages"][0]["content"][0]["text"])
    assert payload["timeline"][-1]["board"][0] == "阿狸★★[珠光护手+朔极之矛]"
    assert payload["augments"][1]["name"] == "史诗搜牌" and payload["final_item_bench"] == ["女神之泪"]


def test_replay_mock_and_demo_never_use_an_installed_ocr(home, offline_data, tmp_path, monkeypatch, capsys):
    import tft_advisor.vision.ocr as ocr_mod

    made = []

    class FakeOcr:
        name = "ocr"

        def __init__(self, set_data):
            made.append(self)

        def perceive(self, image, purpose="auto", hint=None):  # pragma: no cover - must not run
            raise AssertionError("the real OCR was used instead of the mock")

    monkeypatch.setattr(ocr_mod, "ocr_available", lambda: True)
    monkeypatch.setattr(ocr_mod, "OcrPerceiver", FakeOcr)
    Image.new("RGB", (320, 180)).save(tmp_path / "a.png")
    assert cli.main(["replay", str(tmp_path / "a.png"), "--mock", str(FIXTURES), "--purpose", "shop", "--no-llm"]) == 0
    assert cli.main(["demo", "--steps", "2", "--interval", "0", "--no-dashboard"]) == 0
    assert made == []
    assert "(没有结果" not in capsys.readouterr().out


def test_run_names_only_hotkeys_that_work(home, offline_data, monkeypatch, capsys):
    import socket

    import tft_advisor.capture.hotkeys as hk
    from tft_advisor.app import AdvisorApp

    class Stub:
        def __init__(self, bindings, log=print):
            self.registered = [k for k in bindings if k != "F7"]  # F7 taken by another program
            self.failed = {"F7": "taken"}

        def start(self):
            return False

        def stop(self):
            pass

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setattr(hk, "HotkeyManager", Stub)
    monkeypatch.setattr(AdvisorApp, "wait", lambda self: None)
    monkeypatch.setattr(cli, "_open", lambda *a, **k: None)
    assert cli.main(["run", "--no-vision-llm", "--no-auto", "--port", str(port)]) == 0
    line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("热键"))
    assert "F6=分析" in line and "F9=读商店" in line and "F7" not in line
