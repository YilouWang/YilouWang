from __future__ import annotations

import json

import pytest

from tft_advisor.advisor.prompts import (
    DEFAULT_QUESTION,
    MAX_STATE_CHARS,
    SYSTEM_PROMPT_STRATEGIST,
    build_state_message,
    build_strategist_system,
    trait_effects_text,
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
    from tft_advisor.models import ScoutRequest

    state = make_state()
    analysis = make_analysis(state, mech)
    # The model may restate a request the planner already made.
    analysis.scout_requests = [ScoutRequest(id="s", text="请点开「Alice」的棋盘后按 F7", target_player="Alice")]
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
        assert system["text"] == build_strategist_system(set_data.summary_text() + "\n\n" + trait_effects_text(set_data))
        assert "TRAIT EFFECTS" in system["text"]
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


# ---------------------------------------------------------------------------
# Regression tests (adversarial review)
# ---------------------------------------------------------------------------


def test_priority_zero_is_most_urgent_and_scout_request_falls_back(set_data, mech):
    from tft_advisor.models import ScoutRequest

    state = make_state()
    analysis = make_analysis(state, mech)
    analysis.scout_requests = [ScoutRequest(id="s", text="请点开右侧玩家列表里「Alice」的棋盘，然后按 F7 记录", target_player="Alice")]
    rules = RulesAdvisor(mech=mech).advise(state, analysis)
    assert rules.scout_request
    reply = dict(
        REPLY,
        actions=[
            {"type": "roll", "text": "搜牌", "priority": 2},
            {"type": "level", "text": "马上升 8", "priority": 0},
        ],
        scout_request=None,
    )
    with FakeAnthropic() as fake:
        fake.queue_json(reply)
        strat, _, _ = make_strategist(fake, set_data)
        adv = strat.advise(state, analysis, rules)
    assert [(a.type, a.priority) for a in adv.actions] == [(ActionType.LEVEL, 1), (ActionType.ROLL, 2)]
    assert adv.scout_request == rules.scout_request


def test_wisp_in_shop_reaches_the_strategist():
    """The tracker stores a Wisp slot as shop_units[i] = None: the raw slot
    (name and price) must still be sent, not a null like an empty slot."""
    shop = [ShopSlot(name="Vi", cost=3), ShopSlot(), ShopSlot(name="Wisp: Grow Up", cost=3), ShopSlot(name="Wisp: Golden Wisp")]
    units = [Unit(api_name="TFT99_Vi", name="Vi", cost=3), None, None, None]
    state = GameState(stage=StageRound.parse("2-5"), shop=shop, shop_units=units)
    data = _payload(build_state_message(state, Analysis(), None))
    assert data["shop"] == ["Vi(3)", None, "Wisp: Grow Up(3)", "Wisp: Golden Wisp"]
    assert "Wisp: " in SYSTEM_PROMPT_STRATEGIST
    # More slots than resolved units: the extra slots are not dropped.
    state = GameState(shop=shop, shop_units=units[:1])
    assert _payload(build_state_message(state, Analysis(), None))["shop"][2] == "Wisp: Grow Up(3)"


def test_duplicate_field_labels_are_stripped(set_data, mech):
    state = make_state()
    analysis = make_analysis(state, mech)
    rules = RulesAdvisor(mech=mech).advise(state, analysis)
    reply = dict(
        REPLY,
        items="装备：巨人杀手 给 Draven",
        augment="海克斯：选 Rich Get Richer，经济好",
        positioning="站位: Draven 放后排角落",
        comp="阵容：Draven 决斗大师",
    )
    with FakeAnthropic() as fake:
        fake.queue_json(reply)
        strat, _, _ = make_strategist(fake, set_data)
        adv = strat.advise(state, analysis, rules)
    assert adv.items == "巨人杀手 给 Draven"
    assert adv.augment == "选 Rich Get Richer，经济好"
    assert adv.positioning == "Draven 放后排角落"
    assert adv.comp == "Draven 决斗大师"
    assert rules.items and not rules.items.startswith("装备")


# ---------------------------------------------------------------------------
# Regression tests (strategist prompt review)
# ---------------------------------------------------------------------------


def _scout_reply(**kw):
    reply = dict(
        REPLY,
        actions=[
            {"type": "level", "text": "买经验升到 8 级", "priority": 1},
            {"type": "scout", "text": "点开「Nina」的棋盘后按 F7", "priority": 2},
            {"type": "other", "text": "去看一下 Alice 的棋盘", "priority": 3},
            {"type": "other", "text": "Alice 在抢 Draven，准备转型", "priority": 3},
        ],
        scout_request="请点开「Nina」的棋盘后按 F7",
    )
    reply.update(kw)
    return reply


def test_scouting_off_never_reaches_the_player(set_data, mech):
    """[advisor] scout_prompts = false: the prompt says so and the reply's
    scout request / scout actions are dropped in code as well."""
    state = make_state()
    state.players.append(PlayerObs(name="Nina", hp=50))
    analysis = make_analysis(state, mech)
    rules = RulesAdvisor(mech=mech).advise(state, analysis)
    assert "scouting false" in SYSTEM_PROMPT_STRATEGIST
    with FakeAnthropic() as fake:
        fake.queue_json(_scout_reply())
        cfg = AnthropicConfig(strategy_model="m")
        strat = ClaudeStrategist(LLM(cfg, client=fake.client()), cfg, set_data, scout_enabled=False)
        adv = strat.advise(state, analysis, rules)
        data = _payload(fake.requests[0]["body"]["messages"][0]["content"][0]["text"])
    assert adv.source == "llm"
    assert adv.scout_request is None
    texts = [a.text for a in adv.actions]
    assert ActionType.SCOUT not in [a.type for a in adv.actions]
    assert "去看一下 Alice 的棋盘" not in texts and "Alice 在抢 Draven，准备转型" in texts
    assert data["scouting"] is False and "open_scout_requests" not in data


def test_scouting_on_keeps_only_planned_targets(set_data, mech):
    from tft_advisor.models import ScoutRequest

    state = make_state()
    state.players.append(PlayerObs(name="Nina", hp=50))
    analysis = make_analysis(state, mech)
    analysis.scout_requests = [ScoutRequest(id="s", text="请点开「Alice」的棋盘后按 F7", target_player="Alice")]
    rules = RulesAdvisor(mech=mech).advise(state, analysis)
    with FakeAnthropic() as fake:
        fake.queue_json(_scout_reply())
        fake.queue_json(_scout_reply(scout_request="请点开「Alice」的棋盘确认她在不在抢"))
        fake.queue_json(_scout_reply(scout_request="看不清金币，请切回自己的棋盘后按 F6"))
        strat, _, _ = make_strategist(fake, set_data)
        off_budget = strat.advise(state, analysis, rules)
        planned = strat.advise(state, analysis, rules)
        own = strat.advise(state, analysis, rules)
        data = _payload(fake.requests[0]["body"]["messages"][0]["content"][0]["text"])
    # Nina is not a planned request: dropped, the planner's request stays.
    assert off_budget.scout_request == rules.scout_request and "Alice" in off_budget.scout_request
    assert all("Nina" not in a.text for a in off_budget.actions)
    assert "去看一下 Alice 的棋盘" in [a.text for a in off_budget.actions]
    assert planned.scout_request == "请点开「Alice」的棋盘确认她在不在抢"
    assert own.scout_request == "看不清金币，请切回自己的棋盘后按 F6"
    assert data["scouting"] is True and data["open_scout_requests"] == ["Alice"]
    assert data["rules"]["scout_request"] == rules.scout_request


def test_state_carries_econ_style_and_comp_style(set_data, mech):
    state = make_state()
    analysis = make_analysis(state, mech)
    analysis.econ = analysis.econ.model_copy(update={"style": "reroll2", "level_estimated": True})
    data = _payload(build_state_message(state, analysis, None))
    assert data["econ"]["style"] == "reroll2" and data["econ"]["level_estimated"] is True
    assert "econ.style is the line the engine plans for" in SYSTEM_PROMPT_STRATEGIST
    assert "from 4-5" in SYSTEM_PROMPT_STRATEGIST and "streaks can skip a roll-down" in SYSTEM_PROMPT_STRATEGIST
    analysis.econ = analysis.econ.model_copy(update={"style": "standard", "level_estimated": False})
    data = _payload(build_state_message(state, analysis, None))
    assert data["econ"]["style"] == "standard" and "level_estimated" not in data["econ"]

    # A comp entry passes the library style / tier through when the engine sets them.
    class StyledComp(CompSuggestion):
        style: str = "reroll2"
        tier: str = "B"

    analysis.comps = [StyledComp(name="X", score=0.5, carry_items=["Deathblade", "DA_Artifact_NavoriFlickerblade"])]
    comp = _payload(build_state_message(state, analysis, None))["comps"][0]
    assert comp["style"] == "reroll2" and comp["tier"] == "B"
    assert comp["carry_items"] == ["Deathblade"]  # raw API ids never reach the model


def test_null_fields_do_not_backfill_contradicting_rules_text(set_data, mech):
    from tft_advisor.advisor.rules import AUGMENT_MORE

    state = make_state()
    analysis = make_analysis(state, mech)
    rules = RulesAdvisor(mech=mech, claude_enabled=True).advise(state, analysis)
    assert AUGMENT_MORE in rules.augment and rules.comp
    data = _payload(build_state_message(state, analysis, rules))
    for field in ("comp", "items", "positioning", "augment"):
        assert data["rules"][field] == getattr(rules, field)
    assert "keeps the rules text" in SYSTEM_PROMPT_STRATEGIST
    assert "must name one of the choices" in SYSTEM_PROMPT_STRATEGIST
    pivot = dict(
        REPLY,
        comp=None,
        augment=None,
        actions=[
            {"type": "pivot", "text": "转 Garen 骑士，Draven 被抢", "priority": 1},
            {"type": "augment", "text": "选 Tiny Titans 保血", "priority": 1},
        ],
    )
    no_aug_action = dict(REPLY, augment=None, actions=[{"type": "level", "text": "升 8", "priority": 1}])
    with FakeAnthropic() as fake:
        fake.queue_json(pivot)
        fake.queue_json(no_aug_action)
        strat, _, _ = make_strategist(fake, set_data)
        adv = strat.advise(state, analysis, rules)
        adv2 = strat.advise(state, analysis, rules)
    assert adv.comp is None  # the rules comp is the line the model just left
    assert adv.augment == "选 Tiny Titans 保血"
    assert "Claude" not in adv2.augment and "Rich Get Richer" in adv2.augment


def test_system_prompt_grounds_augments_wisps_and_language(set_data):
    text = SYSTEM_PROMPT_STRATEGIST
    assert "不确定" in text and "never invent numbers or effects" in text
    assert "精灵" in text and "灵魂莲华" in text and "主宰" in text
    assert "at most 60 characters" in text
    assert "+N is an N-round win" in text and "1-star copies" in text and "exp_gold" in text
    effects = trait_effects_text(set_data)
    assert effects.startswith("TRAIT EFFECTS") and "%i:" not in effects
    assert all(len(line) < 200 for line in effects.splitlines())


def test_state_message_carries_configured_hotkeys(set_data, mech):
    from tft_advisor.config import HotkeyConfig

    state = make_state()
    analysis = make_analysis(state, mech)
    data = _payload(build_state_message(state, analysis, None))
    assert data["hotkeys"] == {"analyze": "F6", "scout": "F7"}
    data = _payload(build_state_message(state, analysis, None, hotkeys=HotkeyConfig(analyze="F10", scout="F11")))
    assert data["hotkeys"] == {"analyze": "F10", "scout": "F11"}
    data = _payload(build_state_message(state, analysis, None, hotkeys=HotkeyConfig(enabled=False)))
    assert data["hotkeys"] == "off"
    assert "分析" in SYSTEM_PROMPT_STRATEGIST and "记录对手" in SYSTEM_PROMPT_STRATEGIST
    assert "F7 (record" not in SYSTEM_PROMPT_STRATEGIST  # no hard-coded default keys
    with FakeAnthropic() as fake:
        fake.queue_json(REPLY)
        cfg = AnthropicConfig(strategy_model="m")
        strat = ClaudeStrategist(LLM(cfg, client=fake.client()), cfg, set_data, hotkeys=HotkeyConfig(scout="F11"))
        strat.advise(state, analysis, RulesAdvisor(mech=mech).advise(state, analysis))
        sent = _payload(fake.requests[0]["body"]["messages"][0]["content"][0]["text"])
    assert sent["hotkeys"]["scout"] == "F11"


def test_odds_rows_trimmed_to_useful_signal():
    def row(name, best, goal_star=2):
        return HitOdds(
            unit=name, api_name=f"TFT99_{name}", cost=5, owned_copies=1, goal_copies=3 if goal_star == 2 else 9,
            goal_star=goal_star, seen_elsewhere=0, remaining_in_pool=8, level=8, p_per_slot=0.001, p_in_shop=0.01,
            p_goal_by_gold={g: best * g / 80 for g in (10, 20, 30, 40, 50, 60, 80)},
        )

    from tft_advisor.models import EconPlan

    odds = [row("Taric", 0.005), row("Draven", 0.004), row("Garen", 0.6), row("Vi", 0.01, goal_star=3)]
    comps = [CompSuggestion(name="X", score=0.8, carry="Draven")]
    analysis = Analysis(odds=odds, comps=comps, econ=EconPlan(roll_budget=34))
    data = _payload(build_state_message(GameState(stage=StageRound.parse("4-2")), analysis, None))
    units = [o["unit"] for o in data["odds"]]
    assert units == ["Draven", "Garen"]  # hopeless Taric dropped, the carry kept
    assert list(data["odds"][1]["p_goal_by_gold"]) == ["20", "30", "40", "60", "80"]
    # A reroll line keeps its 3-star target even when it is far away.
    analysis.econ = EconPlan(roll_budget=34, style="reroll1")
    units = [o["unit"] for o in _payload(build_state_message(GameState(), analysis, None))["odds"]]
    assert units == ["Draven", "Garen", "Vi"]


def test_optional_constructor_arguments_are_keyword_only(set_data):
    """ClaudeStrategist(llm, cfg, sd, "notes") used to bind the notes to
    ``clock``: every advise() then fell back to the rules with a TypeError.
    Now the mistake fails at construction, and the keyword form works."""
    cfg = AnthropicConfig()
    with FakeAnthropic() as fake:
        llm = LLM(cfg, client=fake.client())
        with pytest.raises(TypeError):
            ClaudeStrategist(llm, cfg, set_data, "set notes here")
        strat = ClaudeStrategist(llm, cfg, set_data, extra_reference="set notes here")
        assert "set notes here" in strat.system_prompt
        assert isinstance(strat.clock(), float)
