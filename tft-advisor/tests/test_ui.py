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
from tft_advisor.ui.overlay import Overlay, display_available, overlay_lines
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


def test_lan_mode_requires_token(bus: EventBus) -> None:
    logs: list[str] = []
    srv = DashboardServer(bus, UIConfig(host="0.0.0.0", port=0), log=logs.append)
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
        assert wait_for(lambda: fake_tts.said == ["稳住血量，准备 4-1 升 7。现在升到 7 级"])
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
