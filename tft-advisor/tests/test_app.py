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
