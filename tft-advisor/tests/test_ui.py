"""Tests for the dashboard server, the static page, the overlay and the voice (all offline)."""

from __future__ import annotations

import http.client
import json
import math
import re
import socket
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import pytest

from tft_advisor.bus import EventBus
from tft_advisor.config import UIConfig
from tft_advisor.models import ActionType, Advice, AdviceAction, ScoutRequest, ScreenType
from tft_advisor.ui.overlay import Overlay, compact_line, display_available, overlay_lines
from tft_advisor.ui.server import (
    ALLOWED_COMMANDS,
    MAX_BODY_BYTES,
    CommandError,
    DashboardServer,
    dumps,
    is_local_host,
    jsonable,
    validate_command,
)
from tft_advisor.ui.voice import Speaker, advice_speech, clean_text, pick_voice

UI_DIR = Path(__file__).resolve().parents[1] / "tft_advisor" / "ui"
INDEX = UI_DIR / "static" / "index.html"
EM_DASHES = ("\u2014", "\u2014\u2014")


def quiet(_msg: str) -> None:
    pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Nothing in these tests may write to the real ~/.tft_advisor (token, overlay position)."""
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def server(bus: EventBus):
    srv = DashboardServer(bus, UIConfig(host="127.0.0.1", port=0, open_browser=False), log=quiet, keepalive_s=0.2)
    srv.start()
    yield srv
    srv.stop()


def base(srv: DashboardServer) -> str:
    return f"http://127.0.0.1:{srv.port}"


def request(
    method: str, url: str, body: Any = None, headers: Optional[dict[str, str]] = None, raw: Optional[bytes] = None
) -> tuple[int, bytes]:
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    hdrs = {"Content-Type": "application/json"} if data is not None else {}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def raw_request(port: int, method: str, path: str, headers: dict[str, str], body: bytes = b"") -> tuple[int, bytes]:
    """http.client request with full control over Host / Origin headers."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        for k, v in headers.items():
            conn.putheader(k, v)
        if body:
            conn.putheader("Content-Length", str(len(body)))
        conn.endheaders(body or None)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


class SSEClient:
    def __init__(self, port: int, path: str = "/api/events", headers: Optional[dict[str, str]] = None) -> None:
        self.conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.conn.request("GET", path, headers=headers or {})
        self.resp = self.conn.getresponse()
        self.comments = 0

    @property
    def status(self) -> int:
        return self.resp.status

    def next_event(self) -> tuple[str, Any]:
        event: Optional[str] = None
        data: list[str] = []
        while True:
            line = self.resp.fp.readline()
            if not line:
                raise EOFError("stream closed")
            text = line.decode("utf-8").rstrip("\r\n")
            if text == "":
                if event is not None or data:
                    return event or "message", json.loads("\n".join(data))
                continue
            if text.startswith(":"):
                self.comments += 1
                continue
            field, _, value = text.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                event = value
            elif field == "data":
                data.append(value)

    def next_named(self, name: str, limit: int = 20) -> Any:
        for _ in range(limit):
            ev, data = self.next_event()
            if ev == name:
                return data
        raise AssertionError(f"event {name!r} not received")

    def close(self) -> None:
        self.resp.close()
        self.conn.close()


def wait_for(cond, timeout: float = 5.0, step: float = 0.02) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(step)
    return cond()


def sample_advice(**kw: Any) -> Advice:
    base_kw: dict[str, Any] = dict(
        headline="稳住血量，准备 4-1 升 7",
        actions=[
            AdviceAction(type=ActionType.BUY, text="买下商店里的卡特琳娜", priority=2),
            AdviceAction(type=ActionType.LEVEL, text="现在升到 7 级", priority=1),
            AdviceAction(type=ActionType.ITEM, text="合成无尽之刃给薇恩", priority=3),
            AdviceAction(type=ActionType.SAVE, text="保留 50 金利息", priority=2),
        ],
        plan="4-1 升 7 后搜 4 费",
        source="rules",
        stage="3-5",
    )
    base_kw.update(kw)
    return Advice(**base_kw)


# ---------------------------------------------------------------------------
# server: pages and snapshot
# ---------------------------------------------------------------------------


def test_index_served_with_chinese_title(server: DashboardServer) -> None:
    status, body = request("GET", base(server) + "/")
    assert status == 200
    html = body.decode("utf-8")
    assert "<title>TFT 实时助手</title>" in html
    assert "EventSource" in html


def test_start_returns_url_and_is_idempotent(server: DashboardServer) -> None:
    assert server.url == f"http://127.0.0.1:{server.port}/"
    assert server.start() == server.url
    assert server.token is None and not server.lan_mode


def test_health(server: DashboardServer) -> None:
    status, body = request("GET", base(server) + "/api/health")
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is True and data["lan"] is False and data["port"] == server.port


class Color(Enum):
    RED = "red"


def test_snapshot_is_json_after_publishing_models(server: DashboardServer, bus: EventBus) -> None:
    bus.publish("state", {"stage": {"stage": 3, "round": 2}, "gold": 42, "screen_type": ScreenType.PLANNING})
    bus.publish("advice", sample_advice())  # a raw pydantic model, not dumped
    bus.publish("requests", [ScoutRequest(id="r1", text="请点开 小明 的棋盘后按 F7", target_player="小明")])
    bus.publish("misc", {"s": {3, 1}, "e": Color.RED, "nan": float("nan"), (1, 2): "tuple key"})
    bus.log("看板测试")
    status, body = request("GET", base(server) + "/api/snapshot")
    assert status == 200
    snap = json.loads(body)
    assert snap["state"]["gold"] == 42 and snap["state"]["screen_type"] == "planning"
    assert snap["advice"]["headline"].startswith("稳住血量")
    assert snap["advice"]["actions"][1]["type"] == "level"
    assert snap["requests"][0]["target_player"] == "小明"
    assert snap["misc"] == {"s": [1, 3], "e": "red", "nan": None, "(1, 2)": "tuple key"}
    assert snap["log"][-1]["text"] == "看板测试"


def test_unknown_path_is_404(server: DashboardServer) -> None:
    assert request("GET", base(server) + "/nope")[0] == 404
    assert request("GET", base(server) + "/api/command")[0] == 404


# ---------------------------------------------------------------------------
# server: commands
# ---------------------------------------------------------------------------


def test_command_validates_and_publishes(server: DashboardServer, bus: EventBus) -> None:
    got: list[dict[str, Any]] = []
    bus.subscribe("command", lambda _t, p: got.append(p))
    url = base(server) + "/api/command"

    status, body = request("POST", url, {"cmd": "scout", "player": "小明"})
    assert status == 200 and json.loads(body)["ok"] is True
    status, _ = request("POST", url, {"cmd": "ask", "question": "  现在该 D 牌吗？  "})
    assert status == 200
    status, _ = request("POST", url, {"cmd": "set_field", "field": "gold", "value": 37})
    assert status == 200
    status, _ = request("POST", url, {"cmd": "analyze"})
    assert status == 200

    assert got == [
        {"cmd": "scout", "player": "小明"},
        {"cmd": "ask", "question": "现在该 D 牌吗？"},
        {"cmd": "set_field", "field": "gold", "value": 37},
        {"cmd": "analyze"},
    ]


def test_every_allowed_command_is_accepted(server: DashboardServer, bus: EventBus) -> None:
    got: list[str] = []
    bus.subscribe("command", lambda _t, p: got.append(p["cmd"]))
    args = {
        "ask": {"question": "q"},
        "dismiss": {"id": "r1"},
        "set_field": {"field": "stage", "value": "3-2"},
        "set_comp": {"comp": ""},
    }
    for cmd in sorted(ALLOWED_COMMANDS):
        status, body = request("POST", base(server) + "/api/command", {"cmd": cmd, **args.get(cmd, {})})
        assert status == 200, (cmd, body)
    assert sorted(got) == sorted(ALLOWED_COMMANDS)


def test_invalid_commands_are_rejected(server: DashboardServer, bus: EventBus) -> None:
    got: list[Any] = []
    bus.subscribe("command", lambda _t, p: got.append(p))
    url = base(server) + "/api/command"
    assert request("POST", url, {"cmd": "shutdown"})[0] == 400
    assert request("POST", url, {"nocmd": 1})[0] == 400
    assert request("POST", url, ["analyze"])[0] == 400
    assert request("POST", url, raw=b"{not json")[0] == 400
    assert request("POST", url, raw=b"")[0] == 400
    assert request("POST", url, {"cmd": "ask", "question": "   "})[0] == 400
    assert request("POST", url, {"cmd": "dismiss"})[0] == 400
    assert request("POST", url, {"cmd": "set_field", "field": "gold"})[0] == 400
    assert request("POST", url, {"cmd": "set_field", "field": "gold", "value": [1]})[0] == 400
    status, body = request("POST", url, {"cmd": "shutdown"})
    assert "未知命令" in json.loads(body)["error"]
    assert got == []


def test_oversized_body_is_413(server: DashboardServer, bus: EventBus) -> None:
    got: list[Any] = []
    bus.subscribe("command", lambda _t, p: got.append(p))
    big = json.dumps({"cmd": "ask", "question": "长" * (MAX_BODY_BYTES)}).encode()
    assert len(big) > MAX_BODY_BYTES
    status, body = request("POST", base(server) + "/api/command", raw=big)
    assert status == 413
    assert json.loads(body)["ok"] is False
    assert got == []


def test_validate_command_unit() -> None:
    cmd, args = validate_command({"cmd": "ask", "question": "问" * 900, "token": "x", "extra": 1, "bad": {"a": 1}, "_p": 1})
    assert cmd == "ask" and len(args["question"]) == 500
    assert args["extra"] == 1 and "bad" not in args and "token" not in args and "_p" not in args
    assert validate_command({"cmd": "scout"}) == ("scout", {"player": None})
    assert validate_command({"cmd": "set_comp", "comp": None}) == ("set_comp", {"comp": ""})
    assert validate_command({"cmd": "set_field", "field": "hp", "value": None}) == ("set_field", {"field": "hp", "value": None})
    with pytest.raises(CommandError):
        validate_command({"cmd": "set_field", "field": "hp", "value": float("nan")})
    with pytest.raises(CommandError):
        validate_command({"cmd": 3})


# ---------------------------------------------------------------------------
# server: Server-Sent Events
# ---------------------------------------------------------------------------


def test_sse_snapshot_then_events(server: DashboardServer, bus: EventBus) -> None:
    bus.publish("advice", sample_advice(headline="先存钱"))
    client = SSEClient(server.port)
    try:
        assert client.status == 200
        assert client.resp.getheader("Content-Type").startswith("text/event-stream")
        ev, snap = client.next_event()
        assert ev == "snapshot"
        assert snap["advice"]["headline"] == "先存钱"
        assert wait_for(lambda: server.sse_clients == 1)

        bus.publish("status", {"auto": True, "calls": 3, "busy": False})
        bus.publish("advice", sample_advice(headline="升 7 搜牌"))
        assert client.next_named("status")["calls"] == 3
        assert client.next_named("advice")["headline"] == "升 7 搜牌"
        bus.log("第二条\n换行")
        assert client.next_named("log")["text"] == "第二条\n换行"
        # keepalive comments arrive when idle (0.2 s in this fixture)
        time.sleep(0.5)
        bus.publish("answer", {"question": "q", "answer": "a", "ts": 1.0})
        assert client.next_named("answer")["answer"] == "a"
        assert client.comments >= 1
    finally:
        client.close()
    # the handler notices the disconnect on its next write and releases the bus queue
    assert wait_for(lambda: server.sse_clients == 0, timeout=5)
    assert wait_for(lambda: len(bus._queues) == 0, timeout=2)


def test_stop_ends_sse_stream(bus: EventBus) -> None:
    srv = DashboardServer(bus, UIConfig(host="127.0.0.1", port=0), log=quiet)
    srv.start()
    client = SSEClient(srv.port)
    try:
        assert client.next_event()[0] == "snapshot"
        t0 = time.time()
        srv.stop()
        with pytest.raises(EOFError):
            for _ in range(50):
                client.next_event()
        assert time.time() - t0 < 4
    finally:
        client.close()
    assert wait_for(lambda: srv.sse_clients == 0)


def test_sse_frame_never_breaks_on_bad_payload(server: DashboardServer, bus: EventBus) -> None:
    class Weird:
        def __str__(self) -> str:
            return "奇怪\n对象"

    client = SSEClient(server.port)
    try:
        client.next_event()
        bus.publish("state", {"obj": Weird(), "inf": math.inf})
        data = client.next_named("state")
        assert data == {"obj": "奇怪\n对象", "inf": None}
    finally:
        client.close()


# ---------------------------------------------------------------------------
# server: access control
# ---------------------------------------------------------------------------


def test_lan_mode_requires_token(bus: EventBus, tmp_path: Path) -> None:
    logs: list[str] = []
    srv = DashboardServer(bus, UIConfig(host="0.0.0.0", port=0), log=logs.append, token_file=tmp_path / "token")
    try:
        try:
            url = srv.start()
        except OSError as exc:  # pragma: no cover - sandbox without 0.0.0.0
            pytest.skip(f"cannot bind 0.0.0.0: {exc}")
        assert srv.lan_mode and srv.token
        assert url.endswith(f"/?token={srv.token}")
        assert srv.lan_url and srv.token in srv.lan_url
        assert any(srv.token in line for line in logs)  # the printed URL carries the token
        b = f"http://127.0.0.1:{srv.port}"

        assert request("GET", b + "/")[0] == 200  # the page itself holds no data
        assert request("GET", b + "/api/snapshot")[0] == 401
        assert request("GET", b + "/api/health")[0] == 401
        assert request("GET", b + "/api/snapshot?token=wrong")[0] == 401
        assert request("GET", b + f"/api/snapshot?token={srv.token}")[0] == 200
        assert request("GET", b + "/api/snapshot", headers={"X-Token": srv.token})[0] == 200
        assert request("POST", b + "/api/command", {"cmd": "analyze"})[0] == 401
        got: list[Any] = []
        bus.subscribe("command", lambda _t, p: got.append(p))
        assert request("POST", b + "/api/command", {"cmd": "analyze"}, headers={"X-Token": srv.token})[0] == 200
        assert got == [{"cmd": "analyze"}]

        denied = SSEClient(srv.port)
        assert denied.status == 401
        denied.close()
        client = SSEClient(srv.port, f"/api/events?token={srv.token}")
        try:
            assert client.status == 200 and client.next_event()[0] == "snapshot"
        finally:
            client.close()
    finally:
        srv.stop()


def test_localhost_mode_blocks_foreign_host_and_origin(server: DashboardServer) -> None:
    port = server.port
    ok_host = {"Host": f"127.0.0.1:{port}"}
    assert raw_request(port, "GET", "/api/snapshot", ok_host)[0] == 200
    assert raw_request(port, "GET", "/api/snapshot", {"Host": f"localhost:{port}"})[0] == 200
    # DNS rebinding: attacker domain resolving to 127.0.0.1
    assert raw_request(port, "GET", "/api/snapshot", {"Host": f"evil.example:{port}"})[0] == 403
    assert raw_request(port, "GET", "/", {"Host": "evil.example"})[0] == 403
    body = json.dumps({"cmd": "analyze"}).encode()
    json_hdr = {"Content-Type": "application/json"}
    # CSRF from another web page
    assert raw_request(port, "POST", "/api/command", {**ok_host, **json_hdr, "Origin": "http://evil.example"}, body)[0] == 403
    assert raw_request(port, "POST", "/api/command", {**ok_host, **json_hdr, "Origin": "null"}, body)[0] == 403
    assert raw_request(port, "POST", "/api/command", {**ok_host, **json_hdr, "Origin": f"http://127.0.0.1:{port}"}, body)[0] == 200


def test_busy_port_falls_back_to_next(bus: EventBus) -> None:
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    busy = holder.getsockname()[1]
    if busy > 65525:  # pragma: no cover - keep the +10 window inside the port range
        holder.close()
        pytest.skip("ephemeral port too high")
    logs: list[str] = []
    srv = DashboardServer(bus, UIConfig(host="127.0.0.1", port=busy), log=logs.append)
    try:
        srv.start()
        assert busy < srv.port <= busy + 10
        assert any(str(busy) in line for line in logs)
        assert request("GET", base(srv) + "/api/health")[0] == 200
    finally:
        srv.stop()
        holder.close()


def test_is_local_host() -> None:
    assert is_local_host("127.0.0.1") and is_local_host("localhost") and is_local_host("::1") and is_local_host("[::1]")
    assert not is_local_host("0.0.0.0") and not is_local_host("192.168.1.5") and not is_local_host("")


def test_json_helpers() -> None:
    assert jsonable({"a": {1, 2}, "b": float("inf"), "c": Color.RED, 4: None}) == {"a": [1, 2], "b": None, "c": "red", "4": None}
    assert jsonable(sample_advice())["actions"][0]["type"] == "buy"
    out = dumps({"t": "多行\n文本", "f": float("nan")})
    assert "\n" not in out and json.loads(out) == {"t": "多行\n文本", "f": None}


# ---------------------------------------------------------------------------
# static page
# ---------------------------------------------------------------------------


def test_index_html_static_checks() -> None:
    html = INDEX.read_text(encoding="utf-8")
    for dash in EM_DASHES:
        assert dash not in html
    assert '<html lang="zh-CN">' in html and 'name="viewport"' in html
    for needle in ("/api/events", "/api/snapshot", "/api/command", "EventSource", "我已切到他的棋盘，记录", "set_field", "新对局"):
        assert needle in html, needle
    # self contained: no external scripts, styles or fonts
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", html)
    assert "@import" not in html
    # every command the page can send is accepted by the server
    sent = set(re.findall(r'data-cmd="([a-z_]+)"', html)) | set(re.findall(r'cmd:\s*"([a-z_]+)"', html))
    assert sent and sent <= ALLOWED_COMMANDS
    assert {"analyze", "scout", "shop", "toggle_auto", "new_game", "ask", "dismiss", "set_field", "set_comp"} <= sent


def test_ui_python_sources_have_no_em_dash() -> None:
    for name in ("server.py", "overlay.py", "voice.py"):
        text = (UI_DIR / name).read_text(encoding="utf-8")
        for dash in EM_DASHES:
            assert dash not in text, name


def test_index_served_from_package_resources(server: DashboardServer) -> None:
    assert server.index_html() == INDEX.read_bytes()


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------


def test_overlay_run_returns_false_without_display(bus: EventBus, monkeypatch: pytest.MonkeyPatch) -> None:
    if sys.platform in ("win32", "darwin"):
        pytest.skip("a display always exists on this platform")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert not display_available()
    logs: list[str] = []
    ov = Overlay(bus, UIConfig(overlay=True), log=logs.append)
    t0 = time.time()
    assert ov.run() is False
    assert time.time() - t0 < 1.0
    assert logs and "置顶小窗不可用" in logs[0]
    ov.close()  # harmless when never shown


def test_overlay_run_returns_false_on_this_box(bus: EventBus) -> None:
    if display_available():
        pytest.skip("a display is available here")
    t0 = time.time()
    assert Overlay(bus, UIConfig(), log=quiet).run() is False
    assert time.time() - t0 < 1.0


def test_overlay_shows_and_closes_when_a_display_exists(bus: EventBus, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    if not display_available():
        pytest.skip("no display")
    pytest.importorskip("tkinter")
    import tft_advisor.ui.overlay as overlay_mod

    monkeypatch.setattr(overlay_mod, "_pos_file", lambda: tmp_path / "overlay_pos.json")
    ov = Overlay(bus, UIConfig(overlay=True), log=quiet)

    def feed() -> None:
        time.sleep(0.3)
        bus.publish("advice", sample_advice())
        time.sleep(0.5)
        ov.close()

    threading.Thread(target=feed, daemon=True).start()
    try:
        result = ov.run()
    except Exception as exc:  # pragma: no cover - broken Tk install
        pytest.skip(f"Tk not usable: {exc}")
    if result is False:  # pragma: no cover - Tk could not open a window
        pytest.skip("Tk could not open a window")
    assert result is True and ov.closed
    assert (tmp_path / "overlay_pos.json").exists()


def test_overlay_lines_formatting() -> None:
    lines = overlay_lines(
        sample_advice(),
        [ScoutRequest(id="r1", text="请点开 小明 的棋盘后按 F7", target_player="小明"), {"id": "r2", "target_player": "阿强"}, {"id": "r3", "text": "第三条"}],
        {"stage": {"stage": 3, "round": 5}, "gold": 50, "level": 6, "hp": 72},
        {"busy": True, "auto": True},
    )
    styles = [s for _t, s in lines]
    texts = [t for t, _s in lines]
    assert lines[0] == ("3-5  金币 50  6 级  血量 72", "info")
    assert ("分析中...", "busy") in lines
    assert ("稳住血量，准备 4-1 升 7", "headline") in lines
    # priority 1 first, at most 3 actions
    idx = styles.index("headline")
    assert styles[idx + 1] == "p1" and "升到 7 级" in texts[idx + 1]
    assert styles.count("p1") + styles.count("action") == 3
    reqs = [t for t, s in lines if s == "request"]
    assert len(reqs) == 2 and "小明" in reqs[0] and "阿强" in reqs[1]
    for t in texts:
        assert "\u2014" not in t


def test_overlay_lines_without_advice() -> None:
    lines = overlay_lines(None, None, None, None)
    assert lines == [("等待分析... 按 F6 立即分析", "dim")]
    lines = overlay_lines({"headline": "存钱", "scout_request": "看看 小红 的棋盘"}, [], {}, {"auto": False})
    assert ("自动关", "info") in lines
    assert lines[-1] == ("» 看看 小红 的棋盘", "request")


# ---------------------------------------------------------------------------
# voice
# ---------------------------------------------------------------------------


def test_speaker_without_pyttsx3(monkeypatch: pytest.MonkeyPatch, bus: EventBus) -> None:
    monkeypatch.setitem(sys.modules, "pyttsx3", None)  # import pyttsx3 -> ImportError
    logs: list[str] = []
    sp = Speaker(UIConfig(voice=True), log=logs.append)
    assert sp.start() is False
    assert sp.say("升级") is False
    assert logs and "pyttsx3" in logs[0]
    sp.attach(bus)
    bus.publish("advice", sample_advice())  # no crash, no speech
    sp.stop()
    assert not sp.running


class FakeVoice:
    def __init__(self, vid: str, name: str, languages: Any = ()) -> None:
        self.id = vid
        self.name = name
        self.languages = list(languages)


VOICES = [
    FakeVoice(r"HKEY\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0", "Microsoft Zira Desktop", ["en_US"]),
    FakeVoice(r"HKEY\Voices\Tokens\TTS_MS_ZH-CN_HUIHUI_11.0", "Microsoft Huihui Desktop"),
    FakeVoice("english", "English"),
]


class FakeEngine:
    def __init__(self) -> None:
        self.threads: set[int] = set()
        self.props: dict[str, Any] = {}
        self.said: list[str] = []
        self._buffer: list[str] = []
        self.gate = threading.Event()
        self.gate.set()
        self.entered = threading.Event()
        self.stopped = False
        self.threads.add(threading.get_ident())

    def getProperty(self, name: str) -> Any:  # noqa: N802 - pyttsx3 API
        self.threads.add(threading.get_ident())
        return VOICES if name == "voices" else self.props.get(name)

    def setProperty(self, name: str, value: Any) -> None:  # noqa: N802
        self.threads.add(threading.get_ident())
        self.props[name] = value

    def say(self, text: str) -> None:
        self.threads.add(threading.get_ident())
        self._buffer.append(text)

    def runAndWait(self) -> None:  # noqa: N802
        self.threads.add(threading.get_ident())
        self.entered.set()
        self.gate.wait(5)
        self.said.extend(self._buffer)
        self._buffer.clear()

    def endLoop(self) -> None:  # noqa: N802
        pass

    def stop(self) -> None:
        self.threads.add(threading.get_ident())
        self.stopped = True


@pytest.fixture()
def fake_tts(monkeypatch: pytest.MonkeyPatch) -> FakeEngine:
    engine = FakeEngine()
    engine.threads.clear()
    mod = types.ModuleType("pyttsx3")
    mod.init = lambda *a, **k: engine  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyttsx3", mod)
    return engine


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_speaker_speaks_on_dedicated_thread_with_chinese_voice(fake_tts: FakeEngine) -> None:
    clock = Clock()
    sp = Speaker(UIConfig(voice=True, voice_rate=210), log=quiet, clock=clock)
    assert sp.start() is True
    try:
        assert fake_tts.props["voice"].endswith("HUIHUI_11.0")
        assert fake_tts.props["rate"] == 210
        assert sp.voice_name == "Microsoft Huihui Desktop"
        assert sp.say("追卡特★★") is True
        assert wait_for(lambda: fake_tts.said == ["追卡特2星"])
        # engine created and used on one thread, and not on the caller's thread
        assert len(fake_tts.threads) == 1 and threading.get_ident() not in fake_tts.threads
        # same text within 20 s is dropped, after 20 s it is spoken again
        assert sp.say("追卡特★★") is False
        clock.t += 21
        assert sp.say("追卡特★★") is True
        assert wait_for(lambda: len(fake_tts.said) == 2)
        assert sp.say("") is False and sp.say(None) is False
    finally:
        sp.stop()
    assert fake_tts.stopped
    assert sp.say("停了") is False


def test_speaker_keeps_only_latest_message_per_key(fake_tts: FakeEngine) -> None:
    sp = Speaker(UIConfig(voice=True), log=quiet)
    assert sp.start()
    try:
        fake_tts.gate.clear()
        assert sp.say("A", key="advice")
        assert fake_tts.entered.wait(2)  # "A" is being spoken, the engine is busy
        sp.say("B", key="advice")
        sp.say("C", key="advice")
        sp.say("R", priority=2, key="requests")
        sp.say("D", priority=1, key="advice")  # replaces B and C, and is more urgent than R
        fake_tts.gate.set()
        assert wait_for(lambda: len(fake_tts.said) >= 3)
        time.sleep(0.2)
        assert fake_tts.said == ["A", "D", "R"]
    finally:
        fake_tts.gate.set()
        sp.stop()


def test_speaker_attach_speaks_advice_and_new_requests(fake_tts: FakeEngine, bus: EventBus) -> None:
    sp = Speaker(UIConfig(voice=True), log=quiet)
    assert sp.start()
    sp.attach(bus)
    try:
        bus.publish("advice", sample_advice().model_dump(mode="json"))
        assert wait_for(lambda: fake_tts.said == ["稳住血量，准备 四一 升 7。现在升到 7 级"])  # "4-1" read as 四一
        bus.publish("advice", sample_advice().model_dump(mode="json"))  # unchanged: silent
        req = ScoutRequest(id="r1", text="请点开 小明 的棋盘后按 F7", target_player="小明")
        bus.publish("requests", [req.model_dump(mode="json")])
        assert wait_for(lambda: len(fake_tts.said) == 2)
        assert fake_tts.said[1] == "请点开 小明 的棋盘后按 F7"
        bus.publish("requests", [req.model_dump(mode="json")])  # already announced
        bus.publish("advice", sample_advice(headline="改玩枪手", actions=[]))
        assert wait_for(lambda: len(fake_tts.said) == 3)
        time.sleep(0.2)
        assert fake_tts.said[2] == "改玩枪手"
    finally:
        sp.stop()
    bus.publish("advice", sample_advice(headline="停止后不再播报"))
    time.sleep(0.1)
    assert "停止后不再播报" not in fake_tts.said


def test_voice_helpers() -> None:
    assert pick_voice(VOICES) is VOICES[1]
    assert pick_voice([FakeVoice("v1", "Voice", [b"\x05zh"])]).id == "v1"
    assert pick_voice([FakeVoice("x", "Some Chinese (Taiwan)")]).id == "x"
    assert pick_voice([VOICES[0], VOICES[2]]) is None
    assert pick_voice([]) is None
    assert clean_text("  追  三星\n卡特 \u2014 快 ") == "追 三星 卡特 ， 快"
    assert advice_speech({"headline": "现在升到 7 级", "actions": [{"text": "现在升到 7 级", "priority": 1}]}) == "现在升到 7 级"
    assert advice_speech({"headline": "", "actions": []}) == ""


# ---------------------------------------------------------------------------
# regression tests (adversarial review)
# ---------------------------------------------------------------------------


def test_reserved_extra_args_do_not_crash_the_command(server: DashboardServer, bus: EventBus) -> None:
    # "self" used to reach EventBus.command(self, cmd, **kw) -> TypeError -> 500
    got: list[Any] = []
    bus.subscribe("command", lambda _t, p: got.append(p))
    status, body = request("POST", base(server) + "/api/command", {"cmd": "analyze", "self": 1, "cmd2": "x"})
    assert status == 200, body
    assert got == [{"cmd": "analyze", "cmd2": "x"}]
    assert validate_command({"cmd": "shop", "self": 1, "token": "t"}) == ("shop", {})


def test_hostile_json_bodies_are_400_not_500(server: DashboardServer, bus: EventBus) -> None:
    logs: list[str] = []
    server.log = logs.append
    got: list[Any] = []
    bus.subscribe("command", lambda _t, p: got.append(p))
    url = base(server) + "/api/command"
    assert request("POST", url, raw=b"[" * 5000 + b"]" * 5000)[0] == 400  # RecursionError in json
    assert request("POST", url, raw=b'{"cmd":"analyze","n":' + b"9" * 5000 + b"}")[0] == 400  # int digit limit
    assert request("POST", url, {"cmd": "set_field", "field": "gold", "value": 10**30})[0] == 400
    assert request("POST", url, {"cmd": "set_field", "field": "gold", "value": -(10**30)})[0] == 400
    assert request("POST", url, raw=b'{"cmd":"set_field","field":"gold","value":1e999}')[0] == 400
    assert got == [] and not any("出错" in line for line in logs)


def test_jsonable_handles_numpy_values() -> None:
    np = pytest.importorskip("numpy")
    out = json.loads(dumps({"i": np.int64(7), "b": np.bool_(True), "f": np.float32(0.5), "a": np.array([[1, 2], [3, np.nan]])}))
    assert out == {"i": 7, "b": True, "f": 0.5, "a": [[1.0, 2.0], [3.0, None]]}


def test_sse_slow_client_is_resynced_with_a_snapshot(bus: EventBus, monkeypatch: pytest.MonkeyPatch) -> None:
    """A client that stalls long enough for the bus queue to overflow must not
    silently miss the newest payloads: it gets a fresh snapshot instead."""
    import tft_advisor.ui.server as server_mod

    gate = threading.Event()
    stalled = threading.Event()
    original = server_mod._Handler._write
    writes = {"n": 0}

    def slow_write(self: Any, text: str) -> None:
        writes["n"] += 1
        if writes["n"] == 3:  # retry hint, snapshot, then the first event stalls
            stalled.set()
            gate.wait(5)
        original(self, text)

    monkeypatch.setattr(server_mod._Handler, "_write", slow_write)
    srv = DashboardServer(bus, UIConfig(host="127.0.0.1", port=0), log=quiet, keepalive_s=5)
    srv.start()
    client = SSEClient(srv.port)
    try:
        assert client.next_event()[0] == "snapshot"
        bus.publish("status", {"auto": True})
        assert stalled.wait(3)
        for i in range(400):  # far more than the 256-slot bus queue
            bus.log(f"日志 {i}")
        bus.publish("advice", sample_advice(headline="最新建议").model_dump(mode="json"))
        gate.set()
        snap = client.next_named("snapshot", limit=10)
        assert snap["advice"]["headline"] == "最新建议"
    finally:
        gate.set()
        client.close()
        srv.stop()


def test_sse_keepalive_sends_comment_and_ping_event(server: DashboardServer) -> None:
    client = SSEClient(server.port)
    try:
        assert client.next_event()[0] == "snapshot"
        ping = client.next_named("ping")  # keepalive_s=0.2 in the fixture
        assert isinstance(ping["ts"], float)
        assert client.comments >= 1
    finally:
        client.close()


def test_unusable_host_fails_fast_with_clear_message(bus: EventBus) -> None:
    srv = DashboardServer(bus, UIConfig(host="203.0.113.5", port=18765), log=quiet)  # TEST-NET, not ours
    t0 = time.time()
    with pytest.raises(OSError) as info:
        srv.start()
    assert time.time() - t0 < 2
    assert "[ui] host" in str(info.value) or "[ui] port" in str(info.value)
    srv.stop()  # harmless after a failed start


def test_index_html_regressions() -> None:
    html = INDEX.read_text(encoding="utf-8")
    js = re.search(r"<script>(.*)</script>", html, re.S).group(1)
    # request buttons carry their own player / id, not an index into a list that may change
    assert 'data-player="' in js and 'data-id="' in js
    assert "currentRequests()[" not in js and "reqs[Number(" not in js
    # the DOM is only rewritten through the diffing helper (keeps scroll / taps on phones)
    assert js.count(".innerHTML =") == 1 and "function setHTML" in js
    # dead stream watchdog, poll never overwrites a live stream, answer spinner times out
    assert 'addEventListener("ping"' in js and "STALE_STREAM_MS" in js
    assert 'connMode === "live") return' in js
    assert "ASK_TIMEOUT_S" in js


def test_voice_reads_rounds_the_way_players_say_them() -> None:
    assert clean_text("准备 4-1 升 8，3 - 2 D 牌") == "准备 四一 升 8，三二 D 牌"
    assert clean_text("版本 14.1-2") == "版本 14.1-2" and clean_text("12-3") == "12-3"
    assert clean_text("血量 -5") == "血量 -5"


def test_compact_line() -> None:
    assert compact_line(None, None) == ("等待分析", "dim")
    text, style = compact_line(sample_advice(), [{"id": "r1", "text": "x"}])
    assert style == "p1" and text.startswith("现在升到 7 级") and text.endswith("» 1 条请求")
    text, style = compact_line({"headline": "一" * 40, "actions": []}, [])
    assert style == "compact" and len(text) == 24 and text.endswith("…")
    assert compact_line({"headline": "存钱"}, "not a list") == ("存钱", "compact")


class _FakeWidget:
    def __init__(self, parent: Any = None, **kw: Any) -> None:
        self.kw = dict(kw)
        self.children: list[_FakeWidget] = []
        if parent is not None:
            parent.children.append(self)
        self.parent = parent

    def pack(self, **_kw: Any) -> None:
        pass

    def bind(self, *_a: Any, **_kw: Any) -> None:
        pass

    def destroy(self) -> None:
        if self.parent is not None:
            self.parent.children.remove(self)

    def winfo_children(self) -> list["_FakeWidget"]:
        return list(self.children)

    def configure(self, **kw: Any) -> None:
        self.kw.update(kw)

    def winfo_reqheight(self) -> int:
        return 20 * len(self.children[0].children) if self.children else 20


class _FakeRoot:
    def __init__(self) -> None:
        self.after_calls: list[Any] = []
        self.geometries: list[str] = []
        self.quit_called = False

    def after(self, ms: int, fn: Any) -> None:
        self.after_calls.append((ms, fn))

    def quit(self) -> None:
        self.quit_called = True

    def update_idletasks(self) -> None:
        pass

    def geometry(self, spec: str) -> None:
        self.geometries.append(spec)

    def winfo_x(self) -> int:
        return 100

    def winfo_y(self) -> int:
        return 50


def _fake_overlay(bus: EventBus, logs: list[str]) -> Overlay:
    ov = Overlay(bus, UIConfig(overlay=True), log=logs.append)
    root = _FakeRoot()
    frame = _FakeWidget()
    body = _FakeWidget(frame)
    ov._root, ov._frame, ov._body, ov._toggle = root, frame, body, _FakeWidget()
    ov._tk = types.SimpleNamespace(Label=_FakeWidget)
    ov._fonts = {k: ("f", 10) for k in ("info", "busy", "headline", "p1", "action", "request", "dim", "compact")}
    ov._colors = {}
    ov._scale = 1.5
    ov._bounds = (0, 0, 1920, 1080)
    return ov


def test_overlay_poll_rearms_after_an_error_and_notices_close(bus: EventBus) -> None:
    logs: list[str] = []
    ov = _fake_overlay(bus, logs)

    def boom() -> None:
        raise RuntimeError("render broke")

    ov._render = boom  # type: ignore[method-assign]
    bus.subscribe("advice", ov._on_event)
    bus.publish("advice", sample_advice())
    ov._poll()
    assert ov._root.after_calls, "polling must be re-armed even after an exception"
    assert any("render broke" in line for line in logs)
    ov.close()
    ov._poll()
    assert ov._root.quit_called


def test_overlay_render_scales_fits_height_and_collapses(bus: EventBus) -> None:
    ov = _fake_overlay(bus, [])
    ov._data = {"advice": sample_advice(), "requests": [{"id": "r1", "text": "请点开 小明 的棋盘后按 F7"}], "state": {"gold": 30}}
    ov._render()
    labels = [w.kw["text"] for w in ov._body.children]
    assert "稳住血量，准备 4-1 升 7" in labels and any("小明" in t for t in labels)
    width = int(round(WIDTH_PX * 1.5))
    assert all(w.kw["wraplength"] < width for w in ov._body.children)
    # window follows the content height at 150 % scaling, keeping its position
    assert ov._root.geometries[-1] == f"{width}x{max(int(round(30 * 1.5)), 20 * len(labels))}+100+50"
    ov.toggle_collapsed()
    assert ov._toggle.kw["text"] == "展开" and len(ov._body.children) == 1
    assert ov._body.children[0].kw["text"].startswith("现在升到 7 级")
    ov.toggle_collapsed()
    assert ov._toggle.kw["text"] == "收起" and len(ov._body.children) == len(labels)
    assert not ov.closed  # collapsing never closes the overlay (or the app)


WIDTH_PX = 380


def test_overlay_windows_helpers_never_raise_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    import tft_advisor.ui.overlay as overlay_mod

    class Root(_FakeRoot):
        def winfo_screenwidth(self) -> int:
            return 1600

        def winfo_screenheight(self) -> int:
            return 900

        def winfo_id(self) -> int:
            return 1

        def winfo_fpixels(self, _spec: str) -> float:
            return 144.0

    root = Root()
    assert overlay_mod._tk_scale(root) == 1.5
    monkeypatch.setattr(overlay_mod.sys, "platform", "win32")
    # ctypes.WinDLL does not exist here: both helpers must degrade, not crash run()
    assert overlay_mod._make_non_activating(root) is False
    assert overlay_mod._screen_bounds(root) == (0, 0, 1600, 900)
    assert overlay_mod._exclude_from_capture(root) is False


# ---------------------------------------------------------------------------
# overlay vs screen capture: mss grabs with CAPTUREBLT, so a topmost layered
# window is in every frame sent to Claude unless it is excluded from capture
# ---------------------------------------------------------------------------


class _FakeUser32:
    def __init__(self, set_ok: int = 1, reported: Optional[int] = None) -> None:
        self.set_calls: list[tuple[int, int]] = []
        self.set_ok = set_ok
        self.reported = reported

        def get_parent(hwnd: int) -> int:
            return 5000 + hwnd  # Tk's top-level wrapper around its child window

        def set_affinity(hwnd: int, flag: int) -> int:
            self.set_calls.append((hwnd, flag))
            return self.set_ok

        def get_affinity(_hwnd: int, ref: Any) -> int:
            ref._obj.value = self.reported if self.reported is not None else self.set_calls[-1][1]
            return 1

        self.GetParent, self.SetWindowDisplayAffinity, self.GetWindowDisplayAffinity = get_parent, set_affinity, get_affinity


class _IdRoot(_FakeRoot):
    def winfo_id(self) -> int:
        return 7


def test_overlay_is_excluded_from_capture_on_windows_2004_and_later(monkeypatch: pytest.MonkeyPatch) -> None:
    import tft_advisor.ui.overlay as overlay_mod

    user32 = _FakeUser32()
    monkeypatch.setattr(overlay_mod.sys, "platform", "win32")
    monkeypatch.setattr(overlay_mod, "_user32", lambda: user32)
    monkeypatch.setattr(overlay_mod, "_windows_build", lambda: 22631)
    assert overlay_mod._exclude_from_capture(_IdRoot()) is True
    # the top-level wrapper, not Tk's child window, with WDA_EXCLUDEFROMCAPTURE
    assert user32.set_calls == [(5007, 0x11)]

    # before Windows 10 2004 the flag acts as WDA_MONITOR (a black box on the board): not used
    user32.set_calls.clear()
    monkeypatch.setattr(overlay_mod, "_windows_build", lambda: 18363)
    assert overlay_mod._exclude_from_capture(_IdRoot()) is False
    assert user32.set_calls == []

    monkeypatch.setattr(overlay_mod, "_windows_build", lambda: 19041)
    assert overlay_mod._exclude_from_capture(_IdRoot()) is True
    assert overlay_mod._exclude_from_capture(_IdRoot()) is True
    monkeypatch.setattr(overlay_mod, "_user32", lambda: _FakeUser32(set_ok=0))
    assert overlay_mod._exclude_from_capture(_IdRoot()) is False
    monkeypatch.setattr(overlay_mod, "_user32", lambda: _FakeUser32(reported=1))
    assert overlay_mod._exclude_from_capture(_IdRoot()) is False

    def no_user32() -> Any:
        raise OSError("no user32")

    monkeypatch.setattr(overlay_mod, "_user32", no_user32)
    assert overlay_mod._exclude_from_capture(_IdRoot()) is False
    monkeypatch.setattr(overlay_mod.sys, "platform", "linux")
    monkeypatch.setattr(overlay_mod, "_user32", lambda: _FakeUser32())
    assert overlay_mod._exclude_from_capture(_IdRoot()) is False


def _build_overlay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    supported: bool,
    excluded: bool,
    saved: Optional[tuple[int, int]] = None,
) -> tuple[Overlay, Any, list[str]]:
    """Run Overlay._build against a fake tkinter at 1920x1080, 150 % scaling."""
    import tft_advisor.ui.overlay as overlay_mod

    class Widget:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        def __getattr__(self, _name: str) -> Any:
            return lambda *_a, **_kw: None

    class Root(Widget):
        def __init__(self) -> None:
            self.geometries: list[str] = []

        def geometry(self, spec: str) -> None:
            self.geometries.append(spec)

        def winfo_fpixels(self, _spec: str) -> float:
            return 144.0

        def winfo_screenwidth(self) -> int:
            return 1920

        def winfo_screenheight(self) -> int:
            return 1080

    pos = tmp_path / "overlay_pos.json"
    if saved is not None:
        pos.write_text(json.dumps({"x": saved[0], "y": saved[1]}), encoding="utf-8")
    monkeypatch.setattr(overlay_mod, "_pos_file", lambda: pos)
    monkeypatch.setattr(overlay_mod, "_make_non_activating", lambda _root: False)
    monkeypatch.setattr(overlay_mod, "_capture_exclusion_supported", lambda: supported)
    monkeypatch.setattr(overlay_mod, "_exclude_from_capture", lambda _root: excluded)
    tk = types.SimpleNamespace(Frame=Widget, Label=Widget)
    tkfont = types.SimpleNamespace(families=lambda _root=None: [])
    logs: list[str] = []
    ov = Overlay(EventBus(), UIConfig(overlay=True), log=logs.append)
    root = Root()
    ov._build(root, tk, tkfont)
    return ov, root, logs


def test_overlay_hidden_from_capture_keeps_its_place_on_the_right(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ov, root, logs = _build_overlay(monkeypatch, tmp_path, supported=True, excluded=True)
    assert ov.capture_hidden
    assert root.geometries == ["570x330+960+129"]
    assert not any("截图" in line for line in logs)


def test_overlay_visible_to_capture_moves_off_the_board_and_warns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Windows 10 2004+ but the call failed: move after the window was created
    ov, root, logs = _build_overlay(monkeypatch, tmp_path, supported=True, excluded=False)
    assert not ov.capture_hidden
    assert root.geometries[-1] == "+86+0"
    assert len(logs) == 1 and "截图" in logs[0] and "左上角" in logs[0] and "棋盘" in logs[0]
    assert not any(d in logs[0] for d in EM_DASHES)

    # older Windows / other OS: created in the safe corner, height capped above the board
    ov, root, logs = _build_overlay(monkeypatch, tmp_path, supported=False, excluded=False)
    assert not ov.capture_hidden
    assert root.geometries[0] == "570x270+86+0"
    assert "截图" in logs[0]

    # a position the player chose is kept (only the warning)
    ov, root, logs = _build_overlay(monkeypatch, tmp_path, supported=False, excluded=False, saved=(300, 600))
    assert root.geometries == ["570x270+300+600"]
    assert "截图" in logs[0] and "左上角" not in logs[0]


@pytest.mark.parametrize(
    ("size", "scale"),
    [((1920, 1080), 1.0), ((1920, 1080), 1.25), ((1920, 1080), 1.5), ((2560, 1440), 1.5), ((3840, 2160), 1.5), ((3840, 2160), 2.0)],
)
def test_capture_visible_overlay_stays_off_what_vision_reads(size: tuple[int, int], scale: float) -> None:
    from tft_advisor.capture.regions import region_box

    class Root:
        def winfo_screenwidth(self) -> int:
            return size[0]

        def winfo_screenheight(self) -> int:
            return size[1]

    ov = Overlay(EventBus(), UIConfig(overlay=True), log=quiet)
    ov._scale, ov._screen_size, ov._bounds = scale, size, (0, 0, *size)

    def hits(rect: tuple[int, int, int, int], region: str) -> bool:
        rx0, ry0, rx1, ry1 = region_box(size, region, layout="inscribed")
        return rect[0] < rx1 and rx0 < rect[2] and rect[1] < ry1 and ry0 < rect[3]

    ov._capture_hidden = False
    x, y = ov._default_position(Root(), capture_safe=True)
    rect = (x, y, x + ov._px(ov.width), y + ov._max_height())
    for region in ("board", "bench", "shop", "hud_bottom", "gold", "level", "stage", "players", "items"):
        assert not hits(rect, region), (region, rect)
    assert ov._max_height() >= int(size[1] * 0.2)  # still room for the advice

    if size == (1920, 1080) and scale == 1.5:
        # the finding: the old default spot covers the board's front row in every frame
        ov._capture_hidden = True
        x, y = ov._default_position(Root())
        assert hits((x, y, x + ov._px(ov.width), y + ov._max_height()), "board")


def test_overlay_height_is_capped_above_the_board_only_when_visible_to_capture(bus: EventBus) -> None:
    ov = _fake_overlay(bus, [])
    tall = _FakeWidget()
    tall.winfo_reqheight = lambda: 900  # type: ignore[method-assign]
    ov._frame = tall
    ov._fit_height()
    assert ov._root.geometries[-1] == "570x270+100+50"  # 0.25 of 1080: the board starts at 0.26
    ov._capture_hidden = True
    ov._fit_height()
    assert ov._root.geometries[-1] == "570x486+100+50"


# ---------------------------------------------------------------------------
# review regressions: Ctrl+C in the overlay, CSRF, token, command errors, slow clients
# ---------------------------------------------------------------------------


def test_overlay_ctrl_c_during_mainloop_stops_the_whole_app(bus: EventBus, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tk raises KeyboardInterrupt from mainloop() (not a callback): the shared
    stop event must be set, or AdvisorApp.wait keeps the app running."""
    import tft_advisor.ui.overlay as overlay_mod

    class Widget:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        def __getattr__(self, _name: str) -> Any:
            return lambda *_a, **_kw: None

        def winfo_reqheight(self) -> int:
            return 20

        def winfo_children(self) -> list[Any]:
            return []

    class Root(Widget):
        def mainloop(self) -> None:
            raise KeyboardInterrupt

        def winfo_fpixels(self, _spec: str) -> float:
            return 96.0

        def winfo_screenwidth(self) -> int:
            return 1920

        def winfo_screenheight(self) -> int:
            return 1080

        def winfo_x(self) -> int:
            return 10

        def winfo_y(self) -> int:
            return 10

    tk = types.ModuleType("tkinter")
    tk.Tk, tk.Frame, tk.Label = Root, Widget, Widget  # type: ignore[attr-defined]
    tkfont = types.ModuleType("tkinter.font")
    tkfont.families = lambda _root=None: []  # type: ignore[attr-defined]
    tk.font = tkfont  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tkinter", tk)
    monkeypatch.setitem(sys.modules, "tkinter.font", tkfont)
    monkeypatch.setattr(overlay_mod, "display_available", lambda: True)
    monkeypatch.setattr(overlay_mod, "_make_non_activating", lambda _root: False)

    stop = threading.Event()
    ov = Overlay(bus, UIConfig(overlay=True), log=quiet, stop_event=stop)
    assert ov.run() is True
    assert ov.interrupted and ov.closed
    assert stop.is_set(), "Ctrl+C must stop capture / hotkeys / Claude calls, not only hide the window"
    assert not bus._handlers.get("advice"), "bus subscriptions are released"


def test_localhost_mode_rejects_pages_on_other_local_ports(server: DashboardServer, bus: EventBus) -> None:
    port = server.port
    got: list[Any] = []
    bus.subscribe("command", lambda _t, p: got.append(p))
    body = json.dumps({"cmd": "new_game"}).encode()
    host = {"Host": f"127.0.0.1:{port}"}
    as_json = {"Content-Type": "application/json"}

    def post(extra: dict[str, str]) -> int:
        return raw_request(port, "POST", "/api/command", {**host, **extra}, body)[0]

    # another local web page (dev server, Jupyter, a downloaded HTML file) is not the dashboard
    for origin in ("http://localhost:3000", "http://127.0.0.1:3000", f"https://127.0.0.1:{port}", "http://[::1]:3000"):
        assert post({**as_json, "Origin": origin}) == 403, origin
    # no-preflight "simple" requests (text/plain, form bodies) are refused outright
    assert post({"Content-Type": "text/plain", "Origin": f"http://127.0.0.1:{port}"}) == 415
    assert post({"Content-Type": "application/x-www-form-urlencoded"}) == 415
    assert post({}) == 415
    # fetch metadata: only the page itself (same-origin) or a typed URL (none)
    assert post({**as_json, "Sec-Fetch-Site": "same-site"}) == 403
    assert post({**as_json, "Sec-Fetch-Site": "cross-site"}) == 403
    assert raw_request(port, "GET", "/api/snapshot", {**host, "Sec-Fetch-Site": "cross-site"})[0] == 403
    assert raw_request(port, "GET", "/api/snapshot", {**host, "Origin": "http://localhost:3000"})[0] == 403
    assert raw_request(port, "GET", "/", {**host, "Sec-Fetch-Site": "cross-site"})[0] == 200  # a link to the page is fine
    assert got == []
    # the dashboard itself, also when reached through another loopback name on the same port
    assert post({**as_json, "Origin": f"http://127.0.0.1:{port}", "Sec-Fetch-Site": "same-origin"}) == 200
    assert post({**as_json, "Origin": f"http://localhost:{port}"}) == 200
    assert post({**as_json, "Content-Type": "application/json; charset=utf-8"}) == 200
    assert len(got) == 3


def test_lan_token_is_kept_between_runs_and_can_be_rotated(bus: EventBus, tmp_path: Path) -> None:
    token_file = tmp_path / "cache" / "dashboard_token"

    def run(**kw: Any) -> tuple[str, list[str]]:
        logs: list[str] = []
        srv = DashboardServer(bus, UIConfig(host="0.0.0.0", port=0), log=logs.append, token_file=token_file, **kw)
        try:
            try:
                srv.start()
            except OSError as exc:  # pragma: no cover - sandbox without 0.0.0.0
                pytest.skip(f"cannot bind 0.0.0.0: {exc}")
            assert srv.token
            assert request("GET", f"http://127.0.0.1:{srv.port}/api/health?token={srv.token}")[0] == 200
            return srv.token, logs
        finally:
            srv.stop()

    first, logs = run()
    assert token_file.read_text(encoding="utf-8").strip() == first
    assert any(str(token_file) in line for line in logs), "the log says where the token lives"
    if sys.platform != "win32":
        assert token_file.stat().st_mode & 0o077 == 0, "token file is private"
    assert run()[0] == first, "a phone bookmark must keep working after a restart"
    rotated = run(new_token=True)[0]
    assert rotated != first and token_file.read_text(encoding="utf-8").strip() == rotated
    token_file.write_text("short\n", encoding="utf-8")  # damaged file: replaced by a fresh token
    fresh = run()[0]
    assert fresh not in (first, rotated, "short") and len(fresh) >= 22
    # no file: a new token every run (old behaviour)
    srv = DashboardServer(bus, UIConfig(host="0.0.0.0", port=0), log=quiet, token_file=None)
    try:
        srv.start()
        assert srv.token and srv.token != fresh
    finally:
        srv.stop()


def test_load_or_create_token_survives_an_unwritable_location(tmp_path: Path) -> None:
    from tft_advisor.ui.server import load_or_create_token

    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    logs: list[str] = []
    token = load_or_create_token(blocker / "sub" / "token", log=logs.append)  # parent is a file
    assert len(token) >= 22 and logs and "临时令牌" in logs[0]


def test_rejected_command_is_reported_to_the_page(server: DashboardServer, bus: EventBus) -> None:
    """The app refuses a command by logging a warning from the handler; the POST must say so."""

    def fake_app(_topic: str, payload: Any) -> None:
        if payload.get("cmd") == "set_field" and payload.get("value") == 999:
            bus.log("命令 set_field 失败: 金币 999 不合理（范围 0 到 300）", "warn")
        elif payload.get("cmd") == "set_comp" and payload.get("comp") == "bad":
            bus.log("命令 set_comp 失败: 阵容名太长", "warn")
        elif payload.get("cmd") == "set_comp":
            bus.log("没有找到阵容 xyz，按关键词匹配", "warn")  # a hint: the comp was still set
        elif payload.get("cmd") == "set_field" and payload.get("value") == 7:
            bus.log("修正失败：等级 7 不合理", "warn")  # other wording, same meaning
        elif payload.get("cmd") == "analyze":
            bus.log("截图失败: 找不到游戏窗口", "warn")  # the job still runs: only a warning
        elif payload.get("cmd") == "toggle_auto":
            threading.Thread(target=bus.log, args=("另一个线程的警告", "warn")).start()
            time.sleep(0.05)

    bus.subscribe("command", fake_app)
    url = base(server) + "/api/command"
    status, raw = request("POST", url, {"cmd": "set_field", "field": "gold", "value": 999})
    reply = json.loads(raw)
    assert status == 400 and reply["ok"] is False and reply["rejected"] is True
    assert reply["error"] == "金币 999 不合理（范围 0 到 300）", "no internal command name in the message"
    status, raw = request("POST", url, {"cmd": "set_field", "field": "level", "value": 7})
    assert status == 400 and json.loads(raw)["error"] == "等级 7 不合理"
    status, raw = request("POST", url, {"cmd": "set_comp", "comp": "bad"})
    assert status == 400 and json.loads(raw)["error"] == "阵容名太长"
    status, raw = request("POST", url, {"cmd": "set_comp", "comp": "xyz"})
    assert status == 200 and json.loads(raw)["warnings"] == ["没有找到阵容 xyz，按关键词匹配"]
    status, raw = request("POST", url, {"cmd": "set_field", "field": "gold", "value": 42})
    assert status == 200 and json.loads(raw) == {"ok": True, "cmd": "set_field"}
    status, raw = request("POST", url, {"cmd": "analyze"})
    assert status == 200 and json.loads(raw)["warnings"] == ["截图失败: 找不到游戏窗口"]
    status, raw = request("POST", url, {"cmd": "toggle_auto"})  # other threads' logs are not ours
    assert status == 200 and "warnings" not in json.loads(raw)
    assert not [h for h in bus._handlers.get("log", [])], "the capture handler is removed"


def test_app_manual_correction_out_of_range_is_rejected_over_http(tmp_path: Path) -> None:
    from tft_advisor.app import AdvisorApp
    from tft_advisor.config import Config
    from tft_advisor.data.setdata import SetData, bundled_sample

    cfg = Config()
    cfg.data.cache_dir = str(tmp_path / "cache")
    cfg.capture.screenshot_dir = str(tmp_path / "shots")
    cfg.ui.open_browser = False
    cfg.ui.port = 0
    cfg.hotkeys.enabled = False
    sample = SetData.from_cdragon(bundled_sample(), source="bundled-sample")
    app = AdvisorApp(cfg, set_data=sample, perceiver=None, fast_perceiver=None, use_llm=False, console=False)
    url = app.start(dashboard=True, hotkeys=False, voice=False, capture=False)
    try:
        assert url
        cmd_url = url.split("?")[0].rstrip("/") + "/api/command"
        status, raw = request("POST", cmd_url, {"cmd": "set_field", "field": "gold", "value": 999})
        reply = json.loads(raw)
        assert status == 400 and reply["rejected"] and "999" in reply["error"] and "300" in reply["error"]
        assert "set_field" not in reply["error"]
        assert app.tracker.state.gold != 999
        status, raw = request("POST", cmd_url, {"cmd": "set_field", "field": "gold", "value": 42})
        assert status == 200 and json.loads(raw)["ok"] is True
        assert app.tracker.state.gold == 42
    finally:
        app.stop()


def _slow_sockets(port: int, n: int) -> list[socket.socket]:
    socks = []
    for _ in range(n):
        s = socket.create_connection(("127.0.0.1", port), timeout=5)
        s.sendall(b"GET / HTTP/1.1\r\n")  # headers never finish
        socks.append(s)
    return socks


def _closed_by_server(s: socket.socket, timeout: float) -> bool:
    s.settimeout(timeout)
    try:
        return s.recv(1024) == b""
    except (ConnectionResetError, ConnectionAbortedError):
        return True
    except (socket.timeout, TimeoutError):
        return False


def test_slow_header_clients_are_cut_off_by_a_total_deadline(bus: EventBus, monkeypatch: pytest.MonkeyPatch) -> None:
    import tft_advisor.ui.server as server_mod

    monkeypatch.setattr(server_mod, "HEADER_TIMEOUT_S", 0.6)
    srv = DashboardServer(bus, UIConfig(host="127.0.0.1", port=0), log=quiet)
    srv.start()
    try:
        s = _slow_sockets(srv.port, 1)[0]
        t0 = time.time()
        # one header byte every 0.2 s: each recv succeeds, only a total deadline stops it
        for _ in range(10):
            try:
                s.sendall(b"X")
            except OSError:
                break
            time.sleep(0.2)
        assert _closed_by_server(s, 3.0)
        assert time.time() - t0 < 4.0
        s.close()
        assert wait_for(lambda: srv._httpd.open_connections == 0, 3.0)
        assert request("GET", base(srv) + "/api/health")[0] == 200  # normal requests unaffected
    finally:
        srv.stop()


def test_connection_cap_drops_extra_connections_without_threads(bus: EventBus, monkeypatch: pytest.MonkeyPatch) -> None:
    import tft_advisor.ui.server as server_mod

    monkeypatch.setattr(server_mod, "MAX_CONNECTIONS", 4)
    srv = DashboardServer(bus, UIConfig(host="127.0.0.1", port=0), log=quiet)
    srv.start()
    held: list[socket.socket] = []
    try:
        threads_before = threading.active_count()
        held = _slow_sockets(srv.port, 4)
        assert wait_for(lambda: srv._httpd.open_connections == 4, 3.0)
        extra = _slow_sockets(srv.port, 6)
        assert all(_closed_by_server(s, 2.0) for s in extra), "over the cap: closed at once"
        for s in extra:
            s.close()
        assert threading.active_count() <= threads_before + 4 + 1
        assert srv._httpd.open_connections == 4
        for s in held:
            s.close()
        held = []
        assert wait_for(lambda: srv._httpd.open_connections == 0, 12.0)
        assert request("GET", base(srv) + "/api/health")[0] == 200
    finally:
        for s in held:
            s.close()
        srv.stop()


def test_per_client_cap_applies_to_lan_peers_only(bus: EventBus) -> None:
    import tft_advisor.ui.server as server_mod

    srv = DashboardServer(bus, UIConfig(host="127.0.0.1", port=0), log=quiet)
    srv.start()
    try:
        httpd = srv._httpd
        phone = ("192.168.1.23", 50000)
        admitted = [httpd._admit(phone) for _ in range(server_mod.MAX_CONNECTIONS_PER_IP + 3)]
        assert admitted.count(True) == server_mod.MAX_CONNECTIONS_PER_IP
        assert httpd._admit(("192.168.1.24", 1)), "another device still gets in"
        assert httpd._admit(("127.0.0.1", 1)) and httpd._admit(("::ffff:127.0.0.1", 1, 0, 0))
        for _ in range(server_mod.MAX_CONNECTIONS_PER_IP):
            httpd._release(phone)
        for addr in (("192.168.1.24", 1), ("127.0.0.1", 1), ("::ffff:127.0.0.1", 1, 0, 0)):
            httpd._release(addr)
        assert httpd.open_connections == 0 and httpd._admit(phone)
        httpd._release(phone)
    finally:
        srv.stop()


def test_index_html_review_regressions() -> None:
    html = INDEX.read_text(encoding="utf-8")
    js = re.search(r"<script>(.*)</script>", html, re.S).group(1)
    # a snapshot without advice (fresh app after a restart) really shows the placeholder
    body = js.split("function renderHeadline()", 1)[1].split("function currentRequests()", 1)[0]
    assert "setHTML(el, waitingHTML())" in body and "keep the waiting placeholder" not in body
    # lost connection: banner with the data's age, faded page, cleared when live again
    assert 'classList.toggle("stale"' in js and "main.grid.stale" in html
    assert "与助手的连接已断开" in js and "refreshBanner()" in js
    # hidden tabs give their stream back; commands time out instead of queueing forever
    assert "function pauseStream()" in js and "HIDDEN_PAUSE_MS" in js and "pageHidden()" in js
    assert "AbortController" in js and "连接超时" in js
    # refused commands are shown as refused; client-side ranges match the tracker
    assert "j.rejected" in js and "修正失败" in js and "FIELD_RANGE" in js
    from tft_advisor.engine import tracker

    rng = re.search(r"var FIELD_RANGE = (\{.*?\});", js).group(1)
    assert f"gold: [{tracker.GOLD_RANGE[0]}, {tracker.GOLD_RANGE[1]}]" in rng
    assert f"hp: [{tracker.HP_RANGE[0]}, {tracker.HP_RANGE[1]}]" in rng
    assert f"streak: [{tracker.STREAK_RANGE[0]}, {tracker.STREAK_RANGE[1]}]" in rng
    assert f"xp_current: [0, {tracker.XP_MAX}]" in rng
    # small layout / copy fixes
    assert "(pointer: coarse)" in js and "Ctrl+Enter" not in re.search(r'<textarea id="ask-q"[^>]*>', html).group(0)
    phone_css = html.split("@media (max-width: 720px)", 1)[1].split("</style>", 1)[0]
    assert re.search(r"\.pl \{[^}]*min-height: 44px", phone_css)
    assert 'g(S, "status.strategist", null) === false' in js and "需要 API Key" in js
    for opt in re.findall(r"<option value=\"[a-z_]+\">([^<]*)</option>", html):
        assert len(opt) <= 5, opt  # long labels get cut off in the narrow select


def _chromium():
    sync_api = pytest.importorskip("playwright.sync_api")
    pw = sync_api.sync_playwright().start()
    try:
        browser = pw.chromium.launch()
    except Exception as exc:  # pragma: no cover - no browser installed here
        pw.stop()
        pytest.skip(f"no Chromium for Playwright: {exc}")
    return pw, browser


def test_page_in_a_real_browser_restart_disconnect_and_rejections(bus: EventBus) -> None:
    """Needs Playwright + Chromium (skipped otherwise)."""
    pw, browser = _chromium()
    srv = DashboardServer(bus, UIConfig(host="127.0.0.1", port=0), log=quiet, keepalive_s=0.5)
    srv2: Optional[DashboardServer] = None
    try:
        srv.start()
        port = srv.port
        bus.publish("status", {"auto": True, "strategist": False, "hotkeys": {"analyze": "F6"}})
        bus.publish("advice", sample_advice(stage="3-2"))
        bus.publish("state", {"stage": {"stage": 3, "round": 2}, "gold": 38})

        def fake_app(_topic: str, payload: Any) -> None:
            if payload.get("cmd") == "set_field" and payload.get("field") == "stage":
                bus.log(f"命令 set_field 失败: 回合格式不对：{payload.get('value')!r}（例如 3-2）", "warn")

        bus.subscribe("command", fake_app)
        page = browser.new_context(viewport={"width": 1440, "height": 900}, bypass_csp=True).new_page()
        page.goto(srv.url)
        page.wait_for_function("() => document.querySelector('#s-conn').textContent === '实时'", timeout=10000)
        page.wait_for_function("() => document.querySelector('#headline').textContent.indexOf('稳住血量') >= 0", timeout=5000)
        assert page.eval_on_selector("#ask-q", "e => e.disabled") is True  # no Claude: asking is pointless

        # a refused correction says so (and never says "sent")
        page.select_option("#fix-field", "stage")
        page.fill("#fix-value", "0-0")
        page.click("#fix-btn")
        page.wait_for_function("() => document.querySelector('#toast').textContent.indexOf('修正失败') >= 0", timeout=5000)
        assert "回合格式不对" in page.inner_text("#toast") and "set_field" not in page.inner_text("#toast")
        page.select_option("#fix-field", "gold")
        page.fill("#fix-value", "999")
        page.click("#fix-btn")
        assert "范围 0 到 300" in page.inner_text("#toast")

        # the app dies: banner + faded page, not "live looking" data
        srv.stop()
        page.wait_for_function("() => document.querySelector('main.grid').classList.contains('stale')", timeout=8000)
        assert "连接已断开" in page.inner_text("#banner") and page.inner_text("#s-conn") == "断开"

        # a fresh app on the same port without advice: the old advice must go away
        bus2 = EventBus()
        bus2.publish("status", {"auto": True, "strategist": True})
        srv2 = DashboardServer(bus2, UIConfig(host="127.0.0.1", port=port), log=quiet, keepalive_s=0.5)
        srv2.start()
        page.wait_for_function("() => !document.querySelector('main.grid').classList.contains('stale')", timeout=20000)
        page.wait_for_function("() => document.querySelector('#headline .hl-text').textContent === '等待第一次分析'", timeout=5000)
        assert page.inner_text("#banner") == "" and "稳住血量" not in page.inner_text("#headline")
        assert page.eval_on_selector("#ask-q", "e => e.disabled") is False

        # a hidden tab pauses its stream and resumes when shown again
        page.wait_for_function("() => document.querySelector('#s-conn').textContent === '实时'", timeout=15000)
        page.evaluate(
            "() => { Object.defineProperty(document, 'visibilityState', {get: () => 'hidden', configurable: true});"
            " document.dispatchEvent(new Event('visibilitychange')); }"
        )
        page.wait_for_function("() => document.querySelector('#s-conn').textContent === '暂停'", timeout=8000)
        assert wait_for(lambda: srv2.sse_clients == 0, 5.0), "the server sees the stream closed"
        page.evaluate(
            "() => { Object.defineProperty(document, 'visibilityState', {get: () => 'visible', configurable: true});"
            " document.dispatchEvent(new Event('visibilitychange')); }"
        )
        page.wait_for_function("() => document.querySelector('#s-conn').textContent === '实时'", timeout=8000)
    finally:
        srv.stop()
        if srv2 is not None:
            srv2.stop()
        browser.close()
        pw.stop()


def test_validate_command_strips_terminal_control_characters() -> None:
    assert validate_command({"cmd": "set_comp", "comp": "阿狸\x1b[2J‮法师"}) == ("set_comp", {"comp": "阿狸[2J法师"})
    assert validate_command({"cmd": "scout", "player": "Opp\x1b]0;x\x07\nA"})[1]["player"] == "Opp]0;x A"
    # questions keep their line breaks, lose the controls
    assert validate_command({"cmd": "ask", "question": "该不该\n转法师\x07？"})[1]["question"] == "该不该\n转法师？"
    assert validate_command({"cmd": "set_field", "field": "gold", "value": "4\x1b2"})[1]["value"] == "42"
