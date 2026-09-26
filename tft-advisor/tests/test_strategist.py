from __future__ import annotations

import json

import pytest

from tft_advisor.advisor.prompts import (
    DEFAULT_QUESTION,
    MAX_STATE_CHARS,
    SYSTEM_PROMPT_STRATEGIST,
    build_state_message,
    build_strategist_system,
)
from tft_advisor.advisor.rules import RulesAdvisor
from tft_advisor.advisor.strategist import ClaudeStrategist
from tft_advisor.config import AnthropicConfig
from tft_advisor.data.mechanics import load_mechanics
from tft_advisor.data.setdata import SetData, bundled_sample
from tft_advisor.engine.economy import plan_economy
from tft_advisor.llm import LLM
from tft_advisor.models import (
    ActionType,
    Analysis,
    CompSuggestion,
    GameState,
    HistoryPoint,
    HitOdds,
    ItemSuggestion,
    OpponentSnapshot,
    PlayerObs,
    ShopSlot,
    StageRound,
    TraitObs,
    Unit,
)

from .fakeapi import FakeAnthropic


@pytest.fixture(scope="module")
def set_data():
    return SetData.from_cdragon(bundled_sample())


@pytest.fixture(scope="module")
def mech():
    return load_mechanics()


def make_state() -> GameState:
    return GameState(
        stage=StageRound.parse("4-1"),
        gold=54,
        level=7,
        xp_current=12,
        xp_needed=48,
        hp=62,
        streak=-2,
        board=[
            Unit(api_name="TFT99_Draven", name="Draven", cost=4, star=2, items=["Deathblade"], row=3, col=0),
            Unit(api_name="TFT99_Braum", name="Braum", cost=2, star=2, row=0, col=3),
        ],
        bench=[Unit(api_name="TFT99_Ashe", name="Ashe", cost=3)],
        item_bench=["Recurve Bow"],
        shop=[ShopSlot(name="Ashe", cost=3), ShopSlot(), ShopSlot(name="Garen", cost=1)],
        traits=[TraitObs(name="Blademaster", count=2, active=False, next_breakpoint=3)],
        augment_choices=["Rich Get Richer", "Tiny Titans"],
        players=[PlayerObs(name="Me", hp=62, is_self=True), PlayerObs(name="Alice", hp=80)],
        self_name="Me",
        opponents={
            "Alice": OpponentSnapshot(
                player="Alice",
                stage="3-5",
                board=[Unit(api_name="TFT99_Draven", name="Draven", cost=4, star=1)],
            )
        },
        history=[HistoryPoint(stage="3-5", hp=70, gold=40, level=6)],
    )


def make_analysis(state: GameState, mech) -> Analysis:
    return Analysis(
        econ=plan_economy(state, mech),
        odds=[
            HitOdds(
                unit="Draven", api_name="TFT99_Draven", cost=4, owned_copies=3, goal_copies=9, goal_star=3,
                seen_elsewhere=1, remaining_in_pool=6, level=7, p_per_slot=0.01, p_in_shop=0.05,
                p_goal_by_gold={10: 0.0, 50: 0.02},
            )
        ],
        comps=[CompSuggestion(name="Draven Blademaster", score=0.8, carry="Draven", contested_by=["Alice"])],
        items=[ItemSuggestion(item="Giant Slayer", components=["B.F. Sword", "Recurve Bow"], holder="Draven")],
        warnings=["Alice 也在玩 Draven"],
    )


REPLY = {
    "headline": "升8级\u2014找Draven",
    "actions": [
        {"type": "roll", "text": "最多花 20 金币搜牌", "priority": 2},
        {"type": "level", "text": "买经验升到 8 级", "priority": 1},
        {"type": "augment", "text": "选 Rich Get Richer", "priority": 1},
    ],
    "plan": "4-2 搜牌稳血，4-5 升 8",
    "comp": "Draven 决斗大师",
    "items": None,
    "positioning": None,
    "augment": "选 Rich Get Richer：经济好",
    "scout_request": "请点开「Alice」的棋盘后按 F7",
    "confidence": 0.8,
}


def make_strategist(fake: FakeAnthropic, set_data, **cfg_kw) -> tuple[ClaudeStrategist, LLM, AnthropicConfig]:
    cfg = AnthropicConfig(strategy_model="claude-test-strategy", strategy_effort="high", **cfg_kw)
    llm = LLM(cfg, client=fake.client())
    return ClaudeStrategist(llm, cfg, set_data, clock=lambda: 1234.0), llm, cfg


def test_advise_happy_path(set_data, mech):
    state = make_state()
    analysis = make_analysis(state, mech)
    rules = RulesAdvisor(mech=mech).advise(state, analysis)
    with FakeAnthropic() as fake:
        fake.queue_json(REPLY)
        fake.queue_json(REPLY)
        strat, llm, cfg = make_strategist(fake, set_data)
        adv = strat.advise(state, analysis, rules)
        adv2 = strat.advise(state, analysis, rules, question="要不要升8？")

        assert strat.last_error is None
        assert adv.source == "llm" and adv.stage == "4-1" and adv.created_at == 1234.0
        assert adv.headline == "升8级，找Draven"  # em dash removed
        assert [a.type for a in adv.actions] == [ActionType.LEVEL, ActionType.AUGMENT, ActionType.ROLL]
        assert [a.priority for a in adv.actions] == [1, 1, 2]
        assert adv.augment == "选 Rich Get Richer：经济好"
        assert adv.scout_request == "请点开「Alice」的棋盘后按 F7"
        assert adv.confidence == 0.8
        # Fields the model left empty fall back to the rules advice.
        assert adv.positioning == rules.positioning and adv.items == rules.items
        assert adv2.headline == adv.headline

        assert len(fake.requests) == 2
        body = fake.requests[0]["body"]
        assert body["model"] == "claude-test-strategy"
        assert body["output_config"]["effort"] == "high"
        assert body["output_config"]["format"]["type"] == "json_schema"
        system = body["system"][0]
        assert system["cache_control"] == {"type": "ephemeral"}
        assert system["text"].startswith(SYSTEM_PROMPT_STRATEGIST)
        assert "TFT Set 99" in system["text"] and "Draven" in system["text"]
        assert system["text"] == build_strategist_system(set_data.summary_text())
        # Identical system prompt bytes on every call (prompt cache).
        assert fake.requests[1]["body"]["system"] == body["system"]
        assert strat.system_prompt is strat.system_prompt

        user = body["messages"][0]["content"][0]["text"]
        assert '"gold":54' in user and '"stage":"4-1"' in user
        assert "Rich Get Richer" in user and DEFAULT_QUESTION in user
        user2 = fake.requests[1]["body"]["messages"][0]["content"][0]["text"]
        assert "要不要升8？" in user2
        assert llm.stats.by_purpose == {"strategy": 2}


def test_accepts_raw_client(set_data, mech):
    state = make_state()
    analysis = make_analysis(state, mech)
    rules = RulesAdvisor(mech=mech).advise(state, analysis)
    with FakeAnthropic() as fake:
        fake.queue_json(REPLY)
        cfg = AnthropicConfig()
        strat = ClaudeStrategist(fake.client(), cfg, set_data)
        assert isinstance(strat.llm, LLM)
        assert strat.advise(state, analysis, rules).source == "llm"


@pytest.mark.parametrize("failure", ["refusal", "500", "429", "truncated"])
def test_fallback_to_rules(set_data, mech, failure):
    state = make_state()
    analysis = make_analysis(state, mech)
    rules = RulesAdvisor(mech=mech).advise(state, analysis)
    with FakeAnthropic() as fake:
        if failure == "refusal":
            fake.queue_refusal()
        elif failure == "truncated":
            fake.queue_text('{"headline": "x"', stop_reason="max_tokens")
        else:
            fake.queue_error(int(failure))
        strat, _, _ = make_strategist(fake, set_data)
        adv = strat.advise(state, analysis, rules)
        assert adv is rules
        assert strat.last_error


def test_fallback_on_local_rate_limit(set_data, mech):
    state = make_state()
    analysis = make_analysis(state, mech)
    rules = RulesAdvisor(mech=mech).advise(state, analysis)
    with FakeAnthropic() as fake:
        fake.queue_json(REPLY)
        strat, _, _ = make_strategist(fake, set_data, max_calls_per_minute=1)
        assert strat.advise(state, analysis, rules).source == "llm"
        assert strat.advise(state, analysis, rules) is rules
        assert "频率" in strat.last_error
        assert len(fake.requests) == 1


def test_empty_actions_fall_back_to_rules_actions(set_data, mech):
    state = make_state()
    analysis = make_analysis(state, mech)
    rules = RulesAdvisor(mech=mech).advise(state, analysis)
    reply = dict(REPLY, actions=[], confidence=3.0)
    with FakeAnthropic() as fake:
        fake.queue_json(reply)
        strat, _, _ = make_strategist(fake, set_data)
        adv = strat.advise(state, analysis, rules)
        assert adv.actions == rules.actions
        assert adv.confidence == 1.0


def test_ask_happy_path(set_data, mech):
    state = make_state()
    analysis = make_analysis(state, mech)
    with FakeAnthropic() as fake:
        fake.queue_text("现在升 8 级\u2014\u2014然后搜牌找 Draven。")
        strat, llm, _ = make_strategist(fake, set_data)
        answer = strat.ask("要不要升8？", state, analysis)
        assert answer == "现在升 8 级，然后搜牌找 Draven。"
        body = fake.requests[0]["body"]
        assert "output_config" in body and "format" not in body["output_config"]
        text = body["messages"][0]["content"][0]["text"]
        assert "PLAYER QUESTION: 要不要升8？" in text and "Simplified Chinese" in text
        assert body["system"][0]["text"] == strat.system_prompt
        assert llm.stats.by_purpose == {"ask": 1}
        assert strat.last_error is None


def test_ask_error_returns_chinese_message(set_data, mech):
    state = make_state()
    analysis = make_analysis(state, mech)
    with FakeAnthropic() as fake:
        fake.queue_error(500)
        strat, _, _ = make_strategist(fake, set_data)
        answer = strat.ask("要不要升8？", state, analysis)
        assert answer.startswith("Claude 暂时无法回答")
        assert "\u2014" not in answer
        assert strat.last_error
        assert strat.ask("   ", state, analysis) == "请输入要问的问题"
        assert len(fake.requests) == 1


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


def _payload(message: str) -> dict:
    head = "GAME SNAPSHOT (JSON):\n"
    assert message.startswith(head)
    return json.loads(message[len(head) :].split("\n\n", 1)[0])


def test_state_message_contents(set_data, mech):
    state = make_state()
    analysis = make_analysis(state, mech)
    rules = RulesAdvisor(mech=mech).advise(state, analysis)
    msg = build_state_message(state, analysis, rules)
    assert msg.endswith(DEFAULT_QUESTION)
    data = _payload(msg)
    assert data["stage"] == "4-1" and data["gold"] == 54 and data["xp"] == "12/48"
    assert data["board"][0] == {"name": "Draven", "star": 2, "cost": 4, "items": ["Deathblade"], "row": 3, "col": 0}
    assert data["shop"] == ["Ashe(3)", None, "Garen(1)"]
    assert data["opponents"]["Alice"] == {"stage": "3-5", "board": ["Draven*1"]}
    assert data["players"][0] == {"name": "Me", "hp": 62, "self": True}
    assert data["econ"]["recommendation"] == analysis.econ.recommendation.value
    assert data["odds"][0]["p_goal_by_gold"] == {"10": 0.0, "50": 0.02}
    assert data["comps"][0]["contested_by"] == ["Alice"]
    assert data["rules"]["headline"] == rules.headline
    assert data["history"] == [{"stage": "3-5", "hp": 70, "gold": 40, "level": 6}]
    assert data["augment_choices"] == ["Rich Get Richer", "Tiny Titans"]
    # Deterministic: same input -> identical bytes.
    assert build_state_message(state, analysis, rules) == msg


def test_state_message_question_and_history_override(mech):
    state = GameState()
    msg = build_state_message(state, Analysis(), None, question="选哪个增强？", recent_history=[{"stage": "2-1", "hp": 90, "ts": 5}])
    assert msg.endswith("PLAYER QUESTION: 选哪个增强？")
    data = _payload(msg)
    assert data["history"] == [{"stage": "2-1", "hp": 90}]
    assert "rules" not in data


def test_state_message_is_trimmed_to_budget(monkeypatch):
    big_units = [Unit(api_name=f"TFT99_U{i}", name=f"VeryLongChampionName{i}", cost=3, star=2, items=["Deathblade", "Giant Slayer", "Bloodthirster"], row=i % 4, col=i % 7) for i in range(14)]
    opponents = {
        f"Player{p}": OpponentSnapshot(player=f"Player{p}", stage="4-1", board=big_units, bench=big_units[:9])
        for p in range(7)
    }
    odds = [
        HitOdds(unit=f"VeryLongChampionName{i}", api_name=f"TFT99_U{i}", cost=3, owned_copies=1, goal_copies=3, goal_star=2,
                seen_elsewhere=0, remaining_in_pool=10, level=7, p_per_slot=0.01, p_in_shop=0.05,
                p_goal_by_gold={g: 0.5 for g in (10, 20, 30, 40, 50, 60, 80)})
        for i in range(20)
    ]
    history = [HistoryPoint(stage=f"{s}-{r}", hp=50, gold=30, level=7) for s in range(2, 6) for r in range(1, 8)]
    state = GameState(stage=StageRound.parse("5-1"), board=big_units, bench=big_units[:9], opponents=opponents, history=history)
    full = build_state_message(state, Analysis(odds=odds), None)
    assert len(full) <= MAX_STATE_CHARS + 200
    data = _payload(full)
    assert len(data["history"]) == 8 and len(data["odds"]) == 8 and "bench" in data["opponents"]["Player0"]

    import tft_advisor.advisor.prompts as prompts

    monkeypatch.setattr(prompts, "MAX_STATE_CHARS", 5000)
    trimmed = build_state_message(state, Analysis(odds=odds), None)
    data = _payload(trimmed)
    assert len(trimmed) < len(full)
    assert len(data["board"]) == 14  # our own board is never trimmed away
    assert len(data.get("history", [])) <= 3
    assert len(data.get("odds", [])) <= 5
    assert all("p_goal_by_gold" not in o for o in data.get("odds", []))
    assert all("bench" not in o for o in data.get("opponents", {}).values())
