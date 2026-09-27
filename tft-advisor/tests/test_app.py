"""End-to-end tests of the orchestrator with mock perception and a fake Claude API."""

from __future__ import annotations

import json
import socket
import time
import urllib.request
from pathlib import Path

import pytest

from tft_advisor.app import AdvisorApp, Job, JobSlot
from tft_advisor.bus import EventBus
from tft_advisor.config import Config
from tft_advisor.data.setdata import SetData, bundled_sample
from tft_advisor.models import ScreenObservation, ScreenType

from .fakeapi import FakeAnthropic

FIXTURES = Path(__file__).parent / "fixtures"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def cfg(tmp_path) -> Config:
    c = Config()
    c.data.cache_dir = str(tmp_path / "cache")
    c.capture.screenshot_dir = str(tmp_path / "shots")
    c.ui.open_browser = False
    c.ui.port = free_port()
    c.hotkeys.enabled = False
    return c


@pytest.fixture()
def sample() -> SetData:
    return SetData.from_cdragon(bundled_sample(), source="bundled-sample")


def fixture_obs() -> list[ScreenObservation]:
    files = sorted(FIXTURES.glob("obs_*.json"))
    assert files, "vision fixtures missing"
    out = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]
        out += [ScreenObservation.model_validate(x) for x in items]
    return out


class ListPerceiver:
    name = "list"

    def __init__(self, observations):
        self.observations = list(observations)

    def perceive(self, image, purpose="auto", hint=None):
        from tft_advisor.models import Observation

        return Observation(screen=self.observations.pop(0), source="mock", purpose=purpose)


def make_app(cfg, sample, observations, **kw) -> AdvisorApp:
    return AdvisorApp(
        cfg,
        set_data=sample,
        perceiver=ListPerceiver(observations),
        fast_perceiver=None,
        use_llm=False,
        console=False,
        **kw,
    )


def test_job_slot_priority():
    slot = JobSlot()
    assert slot.put(Job(1, "shop"))
    assert slot.put(Job(3, "auto"))  # replaces lower priority
    assert not slot.put(Job(1, "shop"))  # does not replace a more important job
    job = slot.get(timeout=0.01)
    assert job is not None and job.purpose == "auto"
    assert slot.get(timeout=0.01) is None


def test_full_game_replay_publishes_everything(cfg, sample):
    obs = fixture_obs()
    app = make_app(cfg, sample, obs)
    seen: dict[str, list] = {}
    app.bus.subscribe("*", lambda topic, payload: seen.setdefault(topic, []).append(payload))
    advices = []
    for _ in range(len(obs)):
        adv = app.run_job(Job(4, "manual"))
        assert adv is not None, app.last_error
        advices.append(adv)
        assert adv.headline and "—" not in adv.headline
    for topic in ("state", "analysis", "advice", "status", "requests"):
        assert seen.get(topic), f"no {topic} published"
    state = app.tracker.state
    assert state.stage is not None and state.stage.stage >= 3
    # the scout fixture must have produced an opponent snapshot
    assert state.opponents, "scout observation not recorded"
    logs = list((cfg.cache_dir / "logs").glob("game-*.jsonl"))
    assert logs and len(logs[0].read_text(encoding="utf-8").splitlines()) == len(obs)


def test_commands_set_field_and_new_game(cfg, sample):
    obs = fixture_obs()
    app = make_app(cfg, sample, obs)
    app.run_job(Job(4, "manual"))
    app.bus.command("set_field", field="gold", value=37)
    job = app._slot.get(timeout=0.1)
    assert job is not None and job.purpose == "reanalyze"
    app.run_job(job)
    assert app.tracker.state.gold == 37
    before = app.auto
    app.bus.command("toggle_auto")
    assert app.auto is (not before)
    app.bus.command("new_game")
    assert app.tracker.state.stage is None


def test_no_perceiver_warns_instead_of_crashing(cfg, sample):
    app = AdvisorApp(cfg, set_data=sample, perceiver=None, fast_perceiver=None, use_llm=False, console=False)
    app.perceiver = None
    app.fast_perceiver = None
    assert app.run_job(Job(4, "manual")) is None
    assert app.run_job(Job(2, "reanalyze")) is not None  # rules still work on the empty state


def test_llm_strategist_upgrades_advice(cfg, sample):
    from tft_advisor.advisor.strategist import ClaudeStrategist
    from tft_advisor.llm import LLM

    obs = fixture_obs()
    with FakeAnthropic() as fake:
        llm = LLM(cfg.anthropic, client=fake.client())
        strategist = ClaudeStrategist(llm, cfg.anthropic, sample)
        app = make_app(cfg, sample, obs, llm=llm, strategist=strategist)
        fake.queue_json(
            {
                "headline": "升到6级，搜到30",
                "actions": [{"type": "level", "text": "买经验升6级", "priority": 1}],
                "plan": "3-2 升 6 搜到 30 稳血",
                "confidence": 0.7,
            }
        )
        adv = app.run_job(Job(4, "manual"))
        assert adv is not None and adv.source == "llm"
        assert adv.headline == "升到6级，搜到30"
        body = fake.requests[-1]["body"]
        assert body["model"] == cfg.anthropic.strategy_model
        assert app.bus.latest("advice")["source"] == "llm"


def test_app_strategist_follows_scout_and_hotkey_settings(cfg, sample):
    # [advisor] scout_prompts = false must reach Claude, and so must the real keys
    # (or "off" when no global hotkey is registered, e.g. off Windows).
    from tft_advisor.llm import LLM

    cfg.advisor.scout_prompts = False
    cfg.hotkeys.enabled = True
    cfg.hotkeys.scout = "F10"
    with FakeAnthropic() as fake:
        app = AdvisorApp(
            cfg, set_data=sample, perceiver=None, fast_perceiver=None, llm=LLM(cfg.anthropic, client=fake.client()), console=False
        )
        assert app.strategist is not None
        assert app.strategist.scout_enabled is False
        assert app.strategist.hotkeys.scout == "F10" and app.strategist.hotkeys.enabled
        app.start(dashboard=False, hotkeys=False, capture=False)
        try:
            assert app.strategist.hotkeys.enabled is False  # nothing registered: point to the buttons
        finally:
            app.stop()


def test_llm_failure_keeps_rules_advice(cfg, sample):
    from tft_advisor.advisor.strategist import ClaudeStrategist
    from tft_advisor.llm import LLM

    obs = fixture_obs()
    with FakeAnthropic() as fake:
        llm = LLM(cfg.anthropic, client=fake.client())
        app = make_app(cfg, sample, obs, llm=llm, strategist=ClaudeStrategist(llm, cfg.anthropic, sample))
        fake.queue_error(500)
        adv = app.run_job(Job(4, "manual"))
        assert adv is not None and adv.source == "rules"


def test_dashboard_command_roundtrip(cfg, sample):
    obs = fixture_obs()
    app = make_app(cfg, sample, obs)
    url = app.start(dashboard=True, hotkeys=False, voice=False, capture=False)
    try:
        assert url
        base = url.split("?")[0].rstrip("/")
        req = urllib.request.Request(
            base + "/api/command",
            data=json.dumps({"cmd": "analyze"}).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
        deadline = time.time() + 10
        while time.time() < deadline and app.bus.latest("advice") is None:
            time.sleep(0.05)
        assert app.bus.latest("advice") is not None
        with urllib.request.urlopen(base + "/api/snapshot", timeout=5) as resp:
            snap = json.loads(resp.read())
        assert "advice" in snap and "state" in snap
    finally:
        app.stop()


def test_scout_purpose_records_opponent(cfg, sample):
    opp = ScreenObservation(
        screen_type=ScreenType.PLANNING,
        viewing_own_board=False,
        viewed_player_name="Enemy",
        stage="3-3",
        board=[{"name": "Draven", "star": 2}, {"name": "Katarina", "star": 1}],
    )
    app = make_app(cfg, sample, [opp])
    app.run_job(Job(4, "scout", extra={"player": "Enemy"}))
    snap = app.tracker.state.opponents
    assert snap and any(u.name == "Draven" for s in snap.values() for u in s.board)


# ---------------------------------------------------------------------------
# Helpers for the orchestration tests below
# ---------------------------------------------------------------------------

import threading  # noqa: E402

from PIL import Image, ImageDraw  # noqa: E402

from tft_advisor.app import GameLogger  # noqa: E402
from tft_advisor.models import Observation  # noqa: E402
from tft_advisor.vision.base import PerceptionError  # noqa: E402

EM_DASHES = ("\u2014", "\u2015", "\u2e3a", "\u2e3b")


def wait_for(pred, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return bool(pred())


def hud_frame(stage: str = "3-5", shop_seed: int = 0, size=(1280, 720)) -> Image.Image:
    """Synthetic HUD: one distinct colour block per stage character, 5 shop cards."""
    import numpy as np

    from tft_advisor.capture.regions import region_box

    img = Image.new("RGB", size, (35, 40, 55))
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = region_box(size, "stage")
    w = max(3, (x1 - x0) // max(1, len(stage)))
    for i, ch in enumerate(stage):
        v = (ord(ch) * 67) % 256
        draw.rectangle([x0 + i * w + 1, y0 + 1, x0 + (i + 1) * w - 2, y1 - 2], fill=(v, 255 - v, (v * 5) % 256))
    sx0, sy0, sx1, sy1 = region_box(size, "shop")
    rng = np.random.default_rng(shop_seed)
    card = (sx1 - sx0) // 5
    for i in range(5):
        color = tuple(int(c) for c in rng.integers(40, 255, 3))
        draw.rectangle([sx0 + i * card + 3, sy0 + 3, sx0 + (i + 1) * card - 3, sy1 - 3], fill=color)
    return img


class FrameCapturer:
    """Capturer double: returns the queued frames in order (then None)."""

    def __init__(self, frames, last_source=None, foreground=None):
        self.frames_left = list(frames)
        self.grabs = 0
        self.last_error = None
        if last_source is not None:
            self.last_source = last_source
        if foreground is not None:
            self.game_foreground = lambda: foreground

    def grab(self):
        self.grabs += 1
        return self.frames_left.pop(0) if self.frames_left else None


class StepStop(threading.Event):
    """Stop event for _capture_loop: each wait() advances the fake clock by 1 s
    and the loop ends after ``n`` iterations (no real sleeping)."""

    def __init__(self, n: int, clock) -> None:
        super().__init__()
        self.n, self.clock, self.i = n, clock, 0

    def wait(self, timeout=None):
        self.i += 1
        self.clock.tick(1.0)
        if self.i > self.n:
            self.set()
        return self.is_set()


class RecordingPerceiver:
    """Records (purpose, image) of each call; ``fail`` raises PerceptionError."""

    def __init__(self, name="main", screen=None, fail=False, delay=0.0):
        self.name = name
        self.calls: list[str] = []
        self.images: list = []
        self.fail = fail
        self.delay = delay
        self.screen = screen or ScreenObservation(screen_type=ScreenType.PLANNING, stage="3-2", gold=31, level=6, hp=70)

    def perceive(self, image, purpose="auto", hint=None):
        self.calls.append(purpose)
        self.images.append(image)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise PerceptionError(f"{self.name} 读不出来")
        return Observation(screen=self.screen.model_copy(deep=True), source=self.name, purpose=purpose)


class StubStrategist:
    """advise()/ask() doubles; ``gate`` blocks advise until set."""

    def __init__(self, headline="Claude 建议", gate=None, ask_gate=None, ask_error=None):
        self.headline = headline
        self.gate = gate
        self.ask_gate = ask_gate
        self.ask_error = ask_error
        self.advise_calls = 0
        self.asked: list = []
        self.last_error = None

    def advise(self, state, analysis, rules_advice, question=None, recent_history=None):
        from tft_advisor.models import Advice

        self.advise_calls += 1
        if self.gate is not None:
            self.gate.wait(5)
        return Advice(headline=self.headline, source="llm", stage=str(state.stage) if state.stage else None)

    def ask(self, question, state, analysis, rules_advice=None):
        self.asked.append((question, rules_advice))
        if self.ask_gate is not None:
            self.ask_gate.wait(5)
        if self.ask_error is not None:
            raise self.ask_error
        return f"回答: {question}"


def app_with(cfg, sample, perceiver, **kw) -> AdvisorApp:
    kw.setdefault("fast_perceiver", None)
    kw.setdefault("use_llm", False)
    return AdvisorApp(cfg, set_data=sample, perceiver=perceiver, console=False, **kw)


def log_texts(app) -> list[str]:
    return [e["text"] for e in app.bus.snapshot()["log"]]


# ---------------------------------------------------------------------------
# Auto mode: capture loop -> RoundWatcher -> auto / shop jobs
# ---------------------------------------------------------------------------


def _loop_app(cfg, sample, clock, frames, **cap_kw):
    cap = FrameCapturer(frames, **cap_kw)
    app = app_with(cfg, sample, RecordingPerceiver(), capturer=cap, clock=clock)
    return app, cap


def test_capture_loop_round_change_queues_auto_after_settle(cfg, sample, clock):
    cfg.capture.settle_delay_s = 1.5
    frames = [hud_frame("3-5")] * 3 + [hud_frame("3-6")] * 4
    app, _cap = _loop_app(cfg, sample, clock, frames)
    app._stop = StepStop(12, clock)
    app._capture_loop()
    job = app._slot.get(timeout=0.01)
    assert job is not None and job.purpose == "auto" and job.gated
    assert job.image is None  # grabbed by the worker, freshest frame


def test_capture_loop_shop_change_queues_shop(cfg, sample, clock):
    frames = [hud_frame("3-5", 0)] * 3 + [hud_frame("3-5", 42)] * 4
    app, _cap = _loop_app(cfg, sample, clock, frames)
    app._stop = StepStop(12, clock)
    app._capture_loop()
    job = app._slot.get(timeout=0.01)
    assert job is not None and job.purpose == "shop"


def test_capture_loop_shop_watch_off_and_combat_queue_nothing(cfg, sample, clock):
    frames = [hud_frame("3-5", 0)] * 3 + [hud_frame("3-5", 42)] * 4
    cfg.advisor.shop_watch = False
    app, _ = _loop_app(cfg, sample, clock, list(frames))
    app._stop = StepStop(12, clock)
    app._capture_loop()
    assert app._slot.get(timeout=0.01) is None

    cfg.advisor.shop_watch = True
    app2, _ = _loop_app(cfg, sample, clock, list(frames))
    app2.tracker.ingest(Observation(screen=ScreenObservation(screen_type=ScreenType.COMBAT, stage="3-5"), source="t"))
    app2._stop = StepStop(12, clock)
    app2._capture_loop()
    assert app2._slot.get(timeout=0.01) is None


def test_capture_loop_auto_off_never_grabs(cfg, sample, clock):
    cfg.advisor.auto = False
    app, cap = _loop_app(cfg, sample, clock, [hud_frame("3-5")] * 3 + [hud_frame("3-6")] * 4)
    app._stop = StepStop(12, clock)
    app._capture_loop()
    assert cap.grabs == 0 and app._slot.get(timeout=0.01) is None


def test_capture_loop_round_change_beats_shop_change(cfg, sample, clock):
    cfg.capture.settle_delay_s = 3.0
    frames = [hud_frame("3-5", 0)] * 3 + [hud_frame("3-6", 0)] * 3 + [hud_frame("3-6", 42)] * 3
    app, _ = _loop_app(cfg, sample, clock, frames)
    app._stop = StepStop(14, clock)
    app._capture_loop()
    job = app._slot.get(timeout=0.01)
    assert job is not None and job.purpose == "auto"


@pytest.mark.parametrize("cap_kw", [{"last_source": "monitor"}, {"last_source": "minimized"}, {"last_source": "window", "foreground": False}])
def test_capture_loop_ignores_frames_that_are_not_the_foreground_game(cfg, sample, clock, cap_kw):
    """Desktop / another app on screen (window missing or covered): never queue auto jobs."""
    frames = [hud_frame("3-5", 0)] * 3 + [hud_frame("3-6", 42)] * 4
    app, cap = _loop_app(cfg, sample, clock, frames, **cap_kw)
    app._stop = StepStop(12, clock)
    app._capture_loop()
    assert cap.grabs > 0
    assert app._slot.get(timeout=0.01) is None
    assert app._auto_paused
    assert log_texts(app).count("没找到游戏窗口或游戏不在前台，自动分析暂停") == 1
    assert app.bus.latest("status")["auto_paused"] is True


def test_capture_loop_window_in_foreground_still_fires(cfg, sample, clock):
    frames = [hud_frame("3-5")] * 3 + [hud_frame("3-6")] * 4
    app, _ = _loop_app(cfg, sample, clock, frames, last_source="window", foreground=True)
    app._stop = StepStop(12, clock)
    app._capture_loop()
    job = app._slot.get(timeout=0.01)
    assert job is not None and job.purpose == "auto"


def test_monitor_frames_allowed_when_window_capture_is_off(cfg, sample, clock):
    cfg.capture.use_window = False  # the player explicitly chose whole-monitor capture
    frames = [hud_frame("3-5")] * 3 + [hud_frame("3-6")] * 4
    app, _ = _loop_app(cfg, sample, clock, frames, last_source="monitor")
    app._stop = StepStop(12, clock)
    app._capture_loop()
    job = app._slot.get(timeout=0.01)
    assert job is not None and job.purpose == "auto"


def test_gated_auto_job_rechecks_frame_but_hotkey_job_uses_monitor(cfg, sample):
    perceiver = RecordingPerceiver()
    cap = FrameCapturer([hud_frame(), hud_frame()], last_source="monitor")
    app = app_with(cfg, sample, perceiver, capturer=cap)
    assert app.run_job(Job(3, "auto", gated=True)) is None
    assert perceiver.calls == []
    # An explicit F6 press keeps the whole-monitor fallback.
    assert app.run_job(Job(4, "manual")) is not None
    assert perceiver.calls == ["manual"]


# ---------------------------------------------------------------------------
# Trigger-time frames and the strategy thread
# ---------------------------------------------------------------------------


def test_hotkey_jobs_grab_the_frame_when_pressed(cfg, sample):
    perceiver = RecordingPerceiver()
    at_press = hud_frame("3-3", 1)
    later = hud_frame("3-3", 2)
    cap = FrameCapturer([at_press, later])
    app = app_with(cfg, sample, perceiver, capturer=cap)
    for purpose in ("scout", "shop", "manual"):
        cap.frames_left = [at_press, later]
        assert app.request_analysis(purpose)
        job = app._slot.get(timeout=0.01)
        assert job is not None and job.image is at_press, purpose
        app.run_job(job)
        assert perceiver.images[-1] is at_press, purpose


def test_strategy_runs_on_its_own_thread(cfg, sample):
    """A slow Claude strategy call must not delay the next perception job."""
    gate = threading.Event()
    strat = StubStrategist(gate=gate)
    perceiver = RecordingPerceiver()
    app = app_with(cfg, sample, perceiver, strategist=strat)
    app.start(dashboard=False, hotkeys=False, voice=False, capture=False)
    try:
        app.request_analysis("manual")
        assert wait_for(lambda: strat.advise_calls == 1)
        assert app.bus.latest("advice")["source"] == "rules"
        assert app.bus.latest("status")["thinking"] is True
        app.request_analysis("shop")
        assert wait_for(lambda: perceiver.calls == ["manual", "shop"]), perceiver.calls
        gate.set()
        assert wait_for(lambda: (app.bus.latest("advice") or {}).get("source") == "llm")
        assert app.bus.latest("advice")["headline"] == "Claude 建议"
        assert wait_for(lambda: app.bus.latest("status")["thinking"] is False)
    finally:
        gate.set()
        app.stop()
    log = next((cfg.cache_dir / "logs").glob("game-*.jsonl"))
    recs = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert [r["purpose"] for r in recs] == ["manual", "shop", "strategy"]
    from tft_advisor.data.mechanics import load_mechanics
    from tft_advisor.review import summarize_log

    assert summarize_log(recs, load_mechanics())["timeline"][-1]["advice"] == "Claude 建议"


def test_stale_strategy_result_is_dropped_after_new_game(cfg, sample):
    gate = threading.Event()
    strat = StubStrategist(gate=gate)
    app = app_with(cfg, sample, RecordingPerceiver(), strategist=strat)
    app.start(dashboard=False, hotkeys=False, voice=False, capture=False)
    try:
        app.request_analysis("manual")
        assert wait_for(lambda: strat.advise_calls == 1)
        app.new_game()
        gate.set()
        assert wait_for(lambda: app.bus.latest("status")["thinking"] is False)
        time.sleep(0.05)
        assert app.bus.latest("advice")["headline"] == "新对局，等待第一次分析"
    finally:
        gate.set()
        app.stop()


# ---------------------------------------------------------------------------
# New games: manual reset during a job, and games detected by the tracker
# ---------------------------------------------------------------------------


def test_new_game_during_inflight_job_drops_old_result(cfg, sample):
    old = ScreenObservation(screen_type=ScreenType.PLANNING, stage="5-1", hp=12, level=8, gold=20)
    perceiver = RecordingPerceiver(screen=old, delay=0.3)
    app = app_with(cfg, sample, perceiver)
    app.start(dashboard=False, hotkeys=False, voice=False, capture=False)
    try:
        app.request_analysis("manual")
        assert wait_for(lambda: perceiver.calls == ["manual"])
        app.new_game()
        assert wait_for(lambda: not app.busy and app._slot.pending() is None)
        time.sleep(0.4)
    finally:
        app.stop()
    st = app.tracker.state
    assert st.stage is None and st.hp is None
    assert app.bus.latest("advice")["headline"] == "新对局，等待第一次分析"
    assert not list((cfg.cache_dir / "logs").glob("game-*.jsonl"))


def test_new_game_clears_pending_job_and_placeholder_has_no_fake_plan(cfg, sample):
    cfg.advisor.comp_hint = "法师"
    app = app_with(cfg, sample, RecordingPerceiver())
    app.bus.command("set_comp", comp="Ahri\x1b[2J Morgana")
    assert app.analyzer.comp_hint == "Ahri[2J Morgana"
    assert app.bus.latest("status")["comp_hint"] == "Ahri[2J Morgana"
    app.request_analysis("manual")
    app.new_game()
    assert app._slot.pending() is None
    assert app.analyzer.comp_hint == "法师"  # back to the configured default
    assert app.bus.latest("analysis")["econ"] is None
    assert app.bus.latest("advice")["confidence"] is None
    assert app.bus.latest("status")["comp_hint"] == "法师"


def test_tracker_detected_new_game_rotates_log(cfg, sample):
    def o(stage, hp, level):
        return ScreenObservation(screen_type=ScreenType.PLANNING, stage=stage, hp=hp, level=level, gold=10)

    seq = [o("3-2", 60, 6), o("4-1", 40, 7), o("5-1", 10, 8), o("1-2", 100, 1), o("1-3", 100, 2), o("2-1", 100, 3)]
    app = make_app(cfg, sample, seq)
    ids = set()
    for _ in seq:
        assert app.run_job(Job(4, "manual")) is not None
        ids.add(app.tracker.state.game_id)
    assert len(ids) == 2
    logs = sorted((cfg.cache_dir / "logs").glob("game-*.jsonl"), key=lambda p: p.stem)
    assert len(logs) == 2
    first = [json.loads(x) for x in logs[0].read_text(encoding="utf-8").splitlines()]
    second = [json.loads(x) for x in logs[1].read_text(encoding="utf-8").splitlines()]
    assert [r["state"]["stage"]["stage"] for r in first] == [3, 4, 5]
    assert len(second) == 3 and len({r["game_id"] for r in second}) == 1
    from tft_advisor.data.mechanics import load_mechanics
    from tft_advisor.review import latest_log, load_records, summarize_log

    summary = summarize_log(load_records(latest_log(cfg.cache_dir / "logs")), load_mechanics())
    assert summary["final_stage"] == "2-1" and summary["final_hp"] == 100
    assert any("检测到新对局" in t for t in log_texts(app))


def test_game_logger_prunes_old_logs(tmp_path):
    d = tmp_path / "logs"
    d.mkdir()
    for i in range(5):
        (d / f"game-20260101-00000{i}.jsonl").write_text("{}\n", encoding="utf-8")
    gl = GameLogger(d, keep=3)
    gl.write({"a": 1})
    names = sorted(p.name for p in d.glob("game-*.jsonl"))
    assert len(names) == 3 and gl.path is not None and gl.path.name in names
    assert "game-20260101-000000.jsonl" not in names
    gl.new_game()
    gl.write({"b": 2})
    assert len(list(d.glob("game-*.jsonl"))) == 3


# ---------------------------------------------------------------------------
# Scouting feedback and hotkeys
# ---------------------------------------------------------------------------


def test_scout_of_own_board_is_not_reported_as_recorded(cfg, sample):
    own = ScreenObservation(
        screen_type=ScreenType.PLANNING, stage="3-3", viewing_own_board=True, board=[{"name": "Draven", "star": 2}]
    )
    app = make_app(cfg, sample, [own, own.model_copy(update={"viewing_own_board": True})])
    app.run_job(Job(4, "manual"))
    app.run_job(Job(4, "scout"))
    assert not app.tracker.state.opponents
    texts = log_texts(app)
    assert not any(t.startswith("已记录") for t in texts)
    assert any("自己的棋盘" in t for t in texts)


def test_named_scout_of_own_board_is_not_filed_under_that_opponent(cfg, sample):
    players = [{"name": "Me", "hp": 80, "is_self": True}, {"name": "Enemy", "hp": 70}]
    own = ScreenObservation(
        screen_type=ScreenType.PLANNING,
        stage="3-3",
        viewing_own_board=True,
        players=players,
        board=[{"name": "Draven", "star": 2}, {"name": "Katarina", "star": 1}],
    )
    # The dashboard's "record Enemy" button, pressed before the camera moved.
    app = make_app(cfg, sample, [own, own.model_copy(deep=True)])
    app.run_job(Job(4, "manual"))
    app.run_job(Job(4, "scout", extra={"player": "Enemy"}))
    assert not app.tracker.state.opponents
    assert not any(t.startswith("已记录") for t in log_texts(app))


def test_nameless_scout_recorded_as_unknown(cfg, sample):
    opp = ScreenObservation(screen_type=ScreenType.PLANNING, stage="3-3", board=[{"name": "Katarina", "star": 1}])
    app = make_app(cfg, sample, [opp])
    app.run_job(Job(4, "scout"))
    assert "unknown" in app.tracker.state.opponents
    assert "已记录 对手（没读到名字） 的棋盘" in log_texts(app)


class RecordingHotkeys:
    instances: list = []
    registered_keys: list = []
    failed_keys: dict = {}

    def __init__(self, bindings, log=print):
        self.bindings = dict(bindings)
        self.registered = [k for k in bindings if k not in self.failed_keys]
        self.failed = {k: v for k, v in self.failed_keys.items() if k in bindings}
        RecordingHotkeys.instances.append(self)

    def start(self):
        return not self.failed

    def stop(self):
        pass


@pytest.fixture()
def hotkeys_stub(monkeypatch, cfg):
    import tft_advisor.capture.hotkeys as hk

    RecordingHotkeys.instances = []
    RecordingHotkeys.failed_keys = {}
    monkeypatch.setattr(hk, "HotkeyManager", RecordingHotkeys)
    cfg.hotkeys.enabled = True
    return RecordingHotkeys


def test_hotkey_bindings_and_f7_queues_scout(cfg, sample, hotkeys_stub):
    opp = ScreenObservation(
        screen_type=ScreenType.PLANNING, stage="3-3", viewing_own_board=False, viewed_player_name="Enemy",
        board=[{"name": "Draven", "star": 2}],
    )
    app = make_app(cfg, sample, [opp])
    app.start(dashboard=False, hotkeys=True, voice=False, capture=False)
    app._stop.set()
    for t in app._threads:
        t.join(2)
    try:
        hk = hotkeys_stub.instances[-1]
        assert set(hk.bindings) == {"F6", "F7", "F8", "F9"}
        hk.bindings["F7"]()
        job = app._slot.get(timeout=0.01)
        assert job is not None and job.purpose == "scout" and job.extra == {}
        app.run_job(job)
        assert "Enemy" in app.tracker.state.opponents
        before = app.auto
        hk.bindings["F8"]()
        assert app.auto is (not before)
    finally:
        app.stop()


def test_f7_passes_the_single_open_request_target(cfg, sample, hotkeys_stub):
    from tft_advisor.models import ScoutRequest

    app = make_app(cfg, sample, [])
    app.start(dashboard=False, hotkeys=True, voice=False, capture=False)
    app._stop.set()
    for t in app._threads:
        t.join(2)
    try:
        req = ScoutRequest(id="r1", text="看看 Kaiser", target_player="Kaiser")
        app.scout_planner.open_requests = lambda: [req]
        hotkeys_stub.instances[-1].bindings["F7"]()
        job = app._slot.get(timeout=0.01)
        assert job is not None and job.extra == {"player": "Kaiser"}
    finally:
        app.stop()


def test_duplicate_hotkeys_warn(cfg, sample, hotkeys_stub):
    cfg.hotkeys.shop = "f6"
    app = make_app(cfg, sample, [])
    app.start(dashboard=False, hotkeys=True, voice=False, capture=False)
    try:
        assert len(hotkeys_stub.instances[-1].bindings) == 3
        assert any("重复" in t for t in log_texts(app))
    finally:
        app.stop()


def test_partial_hotkey_failure_message(cfg, sample, hotkeys_stub):
    hotkeys_stub.failed_keys = {"F9": "已被其他程序占用"}
    app = make_app(cfg, sample, [])
    app.start(dashboard=False, hotkeys=True, voice=False, capture=False)
    try:
        texts = log_texts(app)
        assert any(t.startswith("部分热键不可用（F9）") for t in texts)
        assert "全局热键不可用：请用网页上的按钮" not in texts
    finally:
        app.stop()


# ---------------------------------------------------------------------------
# Shop path, LLM gating, OCR / live-client merges
# ---------------------------------------------------------------------------


def _fake_strategist(fake, cfg, sample):
    from tft_advisor.advisor.strategist import ClaudeStrategist
    from tft_advisor.llm import LLM

    llm = LLM(cfg.anthropic, client=fake.client())
    return llm, ClaudeStrategist(llm, cfg.anthropic, sample)


def test_shop_job_without_ocr_uses_vision_and_skips_strategy(cfg, sample):
    main = RecordingPerceiver("vision")
    with FakeAnthropic() as fake:
        llm, strat = _fake_strategist(fake, cfg, sample)
        app = app_with(cfg, sample, main, llm=llm, strategist=strat)
        adv = app.run_job(Job(1, "shop"))
        assert adv is not None and adv.source == "rules"
        assert main.calls == ["shop"] and fake.requests == []


def test_shop_job_prefers_ocr_and_falls_back_to_vision(cfg, sample):
    main, ocr = RecordingPerceiver("vision"), RecordingPerceiver("ocr")
    app = app_with(cfg, sample, main, fast_perceiver=ocr)
    assert app.run_job(Job(1, "shop")) is not None
    assert ocr.calls == ["shop"] and main.calls == []
    ocr.fail = True
    assert app.run_job(Job(1, "shop")) is not None
    assert main.calls == ["shop"] and "ocr" in (app.last_error or "")


def test_no_perceiver_sets_last_error_for_dashboard(cfg, sample):
    app = app_with(cfg, sample, None)
    app.perceiver = app.fast_perceiver = None
    assert app.perception_mode() == "manual"
    assert app.run_job(Job(1, "shop")) is None
    assert "ANTHROPIC_API_KEY" in (app.last_error or "")
    assert "ANTHROPIC_API_KEY" in (app.bus.latest("status")["last_error"] or "")


def test_save_screenshots_skips_shop_frames(cfg, sample):
    cfg.capture.save_screenshots = True
    app = app_with(cfg, sample, RecordingPerceiver())
    app.run_job(Job(4, "manual", image=hud_frame()))
    app.run_job(Job(1, "shop", image=hud_frame()))
    shots = list(Path(cfg.capture.screenshot_dir).glob("*.png"))
    assert len(shots) == 1 and shots[0].name.endswith("_manual.png")


def test_llm_throttle_manual_bypass_and_no_strategy_for_shop_scout_postgame(cfg, sample, clock):
    cfg.advisor.min_seconds_between_llm = 8.0
    strat = StubStrategist()
    app = app_with(cfg, sample, RecordingPerceiver(), strategist=strat, clock=clock)
    app.run_job(Job(3, "auto"))
    assert strat.advise_calls == 1
    clock.tick(1)
    app.run_job(Job(3, "auto"))
    assert strat.advise_calls == 1  # throttled
    app.run_job(Job(4, "manual"))
    assert strat.advise_calls == 2  # manual always asks
    clock.tick(60)
    app.run_job(Job(1, "shop"))
    app.run_job(Job(4, "scout"))
    assert strat.advise_calls == 2
    post = RecordingPerceiver(screen=ScreenObservation(screen_type=ScreenType.POST_GAME))
    app.perceiver = post
    app.run_job(Job(4, "manual"))
    assert strat.advise_calls == 2


def test_ocr_crosscheck_prefers_ocr_numbers(cfg, sample):
    cfg.advisor.ocr_crosscheck = True
    vision = RecordingPerceiver("vision", ScreenObservation(screen_type=ScreenType.PLANNING, stage="3-2", gold=31, level=6))
    ocr = RecordingPerceiver("ocr", ScreenObservation(screen_type=ScreenType.PLANNING, stage="3-2", gold=13, level=None))
    app = app_with(cfg, sample, vision, fast_perceiver=ocr)
    app.run_job(Job(4, "manual"))
    assert ocr.calls == ["manual"]
    assert app.tracker.state.gold == 13 and app.tracker.state.level == 6


class FakeLive:
    def __init__(self, level=8, tft=True, boom=False):
        self.level, self.tft, self.boom = level, tft, boom

    def fetch(self):
        if self.boom:
            raise RuntimeError("no client")
        return {"x": 1}

    def is_tft(self, data):
        return self.tft

    def to_screen_observation(self, data):
        return ScreenObservation(level=self.level)


@pytest.mark.parametrize("live,expected", [(FakeLive(8), 8), (FakeLive(8, tft=False), 6), (FakeLive(boom=True), 6)])
def test_live_client_merge(cfg, sample, live, expected):
    app = app_with(cfg, sample, RecordingPerceiver(), live_client=live)
    app.run_job(Job(4, "manual"))
    assert app.tracker.state.level == expected


# ---------------------------------------------------------------------------
# Free-form questions
# ---------------------------------------------------------------------------


def test_ask_without_strategist(cfg, sample):
    app = app_with(cfg, sample, RecordingPerceiver())
    assert "未启用" in app.ask("q")


def test_ask_async_busy_then_answer(cfg, sample):
    gate = threading.Event()
    strat = StubStrategist(ask_gate=gate)
    app = app_with(cfg, sample, RecordingPerceiver(), strategist=strat)
    answers: list = []
    app.bus.subscribe("answer", lambda _t, p: answers.append(p["answer"]))
    app.bus.command("ask", question="q1")
    assert wait_for(lambda: len(strat.asked) == 1)
    app.bus.command("ask", question="q2")
    assert wait_for(lambda: answers == ["上一个问题还在处理中，请稍等"])
    gate.set()
    assert wait_for(lambda: len(answers) == 2)
    assert answers[1] == "回答: q1"


def test_ask_async_exception_still_answers(cfg, sample):
    strat = StubStrategist(ask_error=RuntimeError("boom"))
    app = app_with(cfg, sample, RecordingPerceiver(), strategist=strat)
    answers: list = []
    app.bus.subscribe("answer", lambda _t, p: answers.append(p["answer"]))
    app.ask_async("q")
    assert wait_for(lambda: answers)
    assert answers[0].startswith("出错了") and not any(d in answers[0] for d in EM_DASHES)


def test_ask_includes_current_advice(cfg, sample, clock):
    with FakeAnthropic() as fake:
        llm, strat = _fake_strategist(fake, cfg, sample)
        app = app_with(cfg, sample, RecordingPerceiver(), llm=llm, strategist=strat, clock=clock)
        app._last_llm_ts = clock()  # throttled: the job stays rules-only
        adv = app.run_job(Job(3, "auto"))
        assert adv is not None and adv.source == "rules" and fake.requests == []
        fake.queue_text("可以")
        assert app.ask("为什么这样打？") == "可以"
        text = fake.requests[-1]["body"]["messages"][0]["content"][0]["text"]
        assert adv.headline in text


# ---------------------------------------------------------------------------
# Console safety
# ---------------------------------------------------------------------------


def test_console_log_escapes_terminal_sequences(cfg, sample, capsys):
    app = AdvisorApp(cfg, set_data=sample, perceiver=RecordingPerceiver(), fast_perceiver=None, use_llm=False, console=True)
    app.bus.command("set_field", field="\x1b[2J\x1b[31mFAKE", value=1)
    app.info("x\x1b]52;c;Y2FsYw==\x07y")
    out = capsys.readouterr().out
    assert "\x1b" not in out and "\x07" not in out
    assert "\\x1b[2J" in out


def test_shop_reads_are_throttled_and_keep_a_call_reserve(cfg, sample):
    from tft_advisor.llm import LLM, RateLimiter

    now = [100.0]
    cfg.anthropic.max_calls_per_minute = 5
    cfg.advisor.shop_reserve_calls = 3
    app = make_app(cfg, sample, [], llm=LLM(cfg.anthropic, client=object(), limiter=RateLimiter(5, clock=lambda: now[0])))
    app.fast_perceiver = None
    assert app._shop_read_allowed()
    app.llm.limiter.try_acquire()
    app.llm.limiter.try_acquire()
    assert not app._shop_read_allowed()  # only 3 calls left: keep them for round analysis
    now[0] += 61
    assert app._shop_read_allowed()


def test_require_foreground_can_be_disabled(cfg, sample):
    class Cap:
        last_source = "window"

        def game_foreground(self):
            return False

        def grab(self):
            return None

    app = make_app(cfg, sample, [], capturer=Cap())
    assert app._game_frame_ok() is False
    cfg.capture.require_foreground = False
    assert app._game_frame_ok() is True


def test_live_threads_auto_mode_end_to_end(cfg, sample):
    """Capture thread -> RoundWatcher -> auto job -> Claude vision (fake API)
    -> tracker -> rules -> Claude strategy thread -> bus, all on real threads."""
    import threading

    from PIL import Image, ImageDraw

    from tft_advisor.advisor.strategist import ClaudeStrategist
    from tft_advisor.llm import LLM
    from tft_advisor.vision.claude_vision import ClaudeVisionPerceiver

    def frame(stage: str) -> Image.Image:
        img = Image.new("RGB", (1920, 1080), (20, 24, 32))
        d = ImageDraw.Draw(img)
        d.rectangle((0, 820, 1920, 1080), fill=(40, 40, 50))  # static bottom HUD
        for i, ch in enumerate(stage):  # big blocky digits in the stage region
            x = 760 + i * 40
            if ch == "-":
                d.rectangle((x + 8, 18, x + 28, 22), fill=(240, 230, 200))
            else:
                n = int(ch)
                d.rectangle((x, 4, x + 30, 36), outline=(240, 230, 200), width=4)
                for k in range(n):
                    d.rectangle((x + 5 + 4 * k, 10, x + 7 + 4 * k, 30), fill=(240, 230, 200))
        return img

    frames = [frame("3-1")] * 6 + [frame("3-2")] * 400
    lock = threading.Lock()

    class Cap:
        last_source = None  # not a window capturer: no foreground gating

        def grab(self):
            with lock:
                return frames.pop(0) if len(frames) > 1 else frames[0]

        def close(self):
            pass

    cfg.capture.poll_interval_s = 0.2
    cfg.capture.settle_delay_s = 0.1
    cfg.advisor.auto = True
    cfg.advisor.shop_watch = False
    with FakeAnthropic() as fake:
        llm = LLM(cfg.anthropic, client=fake.client())
        fake.queue_json_wire(
            ScreenObservation(
                screen_type=ScreenType.PLANNING, viewing_own_board=True, stage="3-2", gold=34, level=5,
                xp_current=12, xp_needed=20, hp=62, board=[{"name": "Garen", "star": 2, "row": 0, "col": 3}], bench=[],
                shop=[{"name": "Garen", "cost": 1}, {"name": "Vayne", "cost": 1}, {}, {}, {}],
            )
        )
        fake.queue_json({"headline": "升到6级，搜到20", "actions": [{"type": "level", "text": "买经验升6级", "priority": 1}], "plan": "3-2 稳血", "confidence": 0.6})
        app = AdvisorApp(
            cfg, set_data=sample, capturer=Cap(), perceiver=ClaudeVisionPerceiver(llm, cfg.anthropic, sample),
            fast_perceiver=None, llm=llm, strategist=ClaudeStrategist(llm, cfg.anthropic, sample), console=False,
        )
        app.start(dashboard=False, hotkeys=False, voice=False, capture=True)
        try:
            deadline = time.time() + 20
            while time.time() < deadline:
                adv = app.bus.latest("advice")
                if adv and adv.get("source") == "llm":
                    break
                time.sleep(0.05)
            adv = app.bus.latest("advice")
            assert adv and adv["source"] == "llm" and adv["headline"] == "升到6级，搜到20", (adv, app.last_error)
            st = app.tracker.state
            assert str(st.stage) == "3-2" and st.gold == 34 and st.level == 5
            purposes = [r["body"]["messages"][0]["content"][-1]["text"] for r in fake.requests[:1]]
            assert "automatic capture" in purposes[0]
        finally:
            app.stop()


@pytest.fixture()
def installed_ocr(monkeypatch):
    """Pretend a local OCR engine is installed (as after `pip install -e ".[all]"`)."""
    import tft_advisor.vision.ocr as ocr_mod

    made: list = []

    class FakeOcr(RecordingPerceiver):
        def __init__(self, set_data):
            super().__init__("ocr")
            made.append(self)

    monkeypatch.setattr(ocr_mod, "ocr_available", lambda: True)
    monkeypatch.setattr(ocr_mod, "OcrPerceiver", FakeOcr)
    return made


def test_none_perceivers_mean_none_even_with_ocr_installed(cfg, sample, installed_ocr):
    # Regression: None used to mean "auto-detect", so with OCR installed the
    # shop job of a test / demo / replay --mock went to the real OCR engine.
    main = RecordingPerceiver("vision")
    app = app_with(cfg, sample, main)  # fast_perceiver=None
    assert app.fast_perceiver is None and installed_ocr == []
    assert app.run_job(Job(1, "shop")) is not None
    assert main.calls == ["shop"]
    bare = AdvisorApp(cfg, set_data=sample, perceiver=None, fast_perceiver=None, use_llm=False, console=False)
    assert bare.perceiver is None and bare.fast_perceiver is None and bare.perception_mode() == "manual"
    assert installed_ocr == []


def test_default_perceivers_still_auto_detect_ocr(cfg, sample, installed_ocr):
    app = AdvisorApp(cfg, set_data=sample, use_llm=False, console=False)
    assert app.fast_perceiver is installed_ocr[-1]
    assert app.perceiver is installed_ocr[0]  # no Claude: OCR is the main perceiver too
    main = RecordingPerceiver("vision")
    app = AdvisorApp(cfg, set_data=sample, perceiver=main, use_llm=False, console=False)
    assert app.perceiver is main and app.fast_perceiver is installed_ocr[-1]
    assert app.run_job(Job(1, "shop")) is not None
    assert installed_ocr[-1].calls == ["shop"] and main.calls == []
