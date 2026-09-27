from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from tft_advisor.config import AnthropicConfig
from tft_advisor.llm import (
    FALLBACK_BETA,
    LLM,
    LLMError,
    LLMRateLimited,
    LLMRefusal,
    LLMUnavailable,
    RateLimiter,
    has_credentials,
    image_block,
    model_price,
    text_block,
)

from .fakeapi import FakeAnthropic


class Pick(BaseModel):
    headline: str
    score: int


def make_llm(fake: FakeAnthropic, **cfg_kw) -> LLM:
    cfg = AnthropicConfig(**cfg_kw)
    return LLM(cfg, client=fake.client())


def test_parse_builds_expected_request_and_parses():
    with FakeAnthropic() as fake:
        fake.queue_json({"headline": "升级", "score": 3})
        llm = make_llm(fake)
        out = llm.parse(
            model="claude-opus-5",
            effort="low",
            system="SYSTEM",
            content=[image_block(b"\x89PNG fake"), text_block("read it")],
            schema=Pick,
            purpose="vision",
        )
        assert out == Pick(headline="升级", score=3)
        req = fake.requests[0]
        body = req["body"]
        assert req["path"].startswith("/v1/messages")
        assert body["model"] == "claude-opus-5"
        assert body["fallbacks"] == "default"
        assert FALLBACK_BETA in req["headers"].get("anthropic-beta", "")
        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"]["effort"] == "low"
        assert body["output_config"]["format"]["type"] == "json_schema"
        assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert body["messages"][0]["content"][0]["type"] == "image"
        assert llm.stats.calls == 1 and llm.stats.cache_read_tokens == 80
        assert llm.stats.by_purpose == {"vision": 1}


def test_no_fallbacks_when_disabled():
    with FakeAnthropic() as fake:
        fake.queue_json({"headline": "x", "score": 1})
        llm = make_llm(fake, use_fallbacks=False)
        llm.parse(model="m", effort="low", system="s", content=[text_block("t")], schema=Pick)
        body = fake.requests[0]["body"]
        assert "fallbacks" not in body
        assert FALLBACK_BETA not in fake.requests[0]["headers"].get("anthropic-beta", "")


def test_rejected_fallback_beta_is_dropped_and_the_request_retried():
    # An account or model without the server-side fallback beta answers 400:
    # the call is sent again without it, and later calls skip it.
    with FakeAnthropic() as fake:
        fake.queue_error(400, "fallbacks: this beta is not available for your organization")
        fake.queue_json({"headline": "x", "score": 1})
        fake.queue_json({"headline": "y", "score": 2})
        llm = make_llm(fake)
        assert llm.parse(model="m", effort="low", system="s", content=[text_block("t")], schema=Pick).score == 1
        assert llm.parse(model="m", effort="low", system="s", content=[text_block("t")], schema=Pick).score == 2
        assert llm.fallbacks_rejected
        assert [("fallbacks" in r["body"]) for r in fake.requests] == [True, False, False]
        assert FALLBACK_BETA not in fake.requests[2]["headers"].get("anthropic-beta", "")
        assert llm.stats.calls == 2


def test_other_bad_requests_are_not_retried():
    with FakeAnthropic() as fake:
        fake.queue_error(400, "messages: image too large")
        llm = make_llm(fake)
        with pytest.raises(LLMError):
            llm.parse(model="m", effort="low", system="s", content=[text_block("t")], schema=Pick)
        assert len(fake.requests) == 1 and not llm.fallbacks_rejected


def test_text_call():
    with FakeAnthropic() as fake:
        fake.queue_text("你好")
        llm = make_llm(fake)
        assert llm.text(model="m", effort="medium", system="s", content=[text_block("hi")]) == "你好"


def test_refusal_and_truncation_raise():
    with FakeAnthropic() as fake:
        fake.queue_refusal()
        fake.queue_text('{"headline": "x"', stop_reason="max_tokens")
        llm = make_llm(fake)
        with pytest.raises(LLMRefusal):
            llm.text(model="m", effort="low", system="s", content=[text_block("t")])
        with pytest.raises(LLMError):
            llm.text(model="m", effort="low", system="s", content=[text_block("t")])
        assert llm.stats.failures == 2


@pytest.mark.parametrize(
    "status,exc",
    [(401, LLMUnavailable), (404, LLMUnavailable), (400, LLMError), (429, LLMRateLimited), (500, LLMError)],
)
def test_http_errors_map_to_llm_errors(status, exc):
    with FakeAnthropic() as fake:
        fake.queue_error(status)
        llm = make_llm(fake)
        with pytest.raises(exc):
            llm.text(model="m", effort="low", system="s", content=[text_block("t")])


def test_rate_limiter_window():
    now = [0.0]
    rl = RateLimiter(2, window_s=60, clock=lambda: now[0])
    assert rl.try_acquire() and rl.try_acquire()
    assert not rl.try_acquire()
    now[0] = 61.0
    assert rl.try_acquire()


def test_local_rate_limit_blocks_call():
    with FakeAnthropic() as fake:
        llm = make_llm(fake, max_calls_per_minute=1)
        fake.queue_text("a")
        llm.text(model="m", effort="low", system="s", content=[text_block("t")])
        with pytest.raises(LLMRateLimited):
            llm.text(model="m", effort="low", system="s", content=[text_block("t")])
        assert len(fake.requests) == 1


# ---------------------------------------------------------------------------
# regressions
# ---------------------------------------------------------------------------


def test_parse_reports_truncation_and_refusal_and_counts_every_billed_call():
    with FakeAnthropic() as fake:
        fake.queue_json({"headline": "ok", "score": 1})
        fake.queue_text('{"headline": "x"', stop_reason="max_tokens")  # truncated JSON
        fake._replies.append((200, fake._message_text('{"headl', "refusal")))  # declined mid-output
        fake.queue_refusal()  # declined before output
        fake.queue_json({"headline": "no score"})  # valid JSON, wrong structure
        fake.queue_text("not json at all")
        llm = make_llm(fake)

        def call():
            return llm.parse(model="claude-opus-5", effort="low", system="s", content=[text_block("t")], schema=Pick)

        assert call() == Pick(headline="ok", score=1)
        with pytest.raises(LLMError, match="max_tokens") as info:
            call()
        assert not isinstance(info.value, LLMRefusal)
        with pytest.raises(LLMRefusal):
            call()
        with pytest.raises(LLMRefusal):
            call()
        with pytest.raises(LLMError, match="结构不符合要求") as info:
            call()
        assert "score" in str(info.value)
        with pytest.raises(LLMError, match="JSON"):
            call()
        # Short Chinese messages, no multi-line pydantic dumps.
        assert "\n" not in str(info.value) and "validation error" not in str(info.value)
        # Every reply was billed, so every reply is counted.
        assert llm.stats.calls == 6 and llm.stats.failures == 5
        assert llm.stats.input_tokens == 600 and llm.stats.output_tokens == 300
        # The request still asks for the JSON schema of the model.
        fmt = fake.requests[0]["body"]["output_config"]["format"]
        assert fmt["type"] == "json_schema" and set(fmt["schema"]["properties"]) == {"headline", "score"}
        assert fake.requests[0]["body"]["output_config"]["effort"] == "low"


def test_usage_iterations_are_summed_and_cost_is_estimated():
    with FakeAnthropic() as fake:
        msg = fake._message_text(json.dumps({"headline": "救回来了", "score": 2}), model="claude-opus-4-8")
        msg["content"].insert(0, {"type": "fallback", "from": {"model": "claude-opus-5"}, "to": {"model": "claude-opus-4-8"}})
        declined = {"type": "message", "model": "claude-opus-5", "input_tokens": 1000, "output_tokens": 40,
                    "cache_read_input_tokens": 0, "cache_creation_input_tokens": 2000}
        served = {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 1000, "output_tokens": 200,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 2000}
        # Top-level usage covers only the attempt that produced the message.
        msg["usage"] = {k: served[k] for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
        msg["usage"]["iterations"] = [declined, served]
        fake._replies.append((200, msg))
        llm = make_llm(fake)
        out = llm.parse(model="claude-opus-5", effort="low", system="s", content=[text_block("t")], schema=Pick)
        assert out == Pick(headline="救回来了", score=2)
        s = llm.stats
        assert (s.calls, s.input_tokens, s.output_tokens, s.cache_write_tokens) == (1, 2000, 240, 4000)
        expected = (2000 * 5 + 4000 * 5 * 1.25 + 240 * 25) / 1e6
        assert s.cost_usd == pytest.approx(expected)
        assert llm.stats.as_dict()["cost_usd"] == round(expected, 4)


def test_cost_without_iterations_uses_top_level_usage():
    with FakeAnthropic() as fake:
        fake.queue_text("好")
        llm = make_llm(fake)
        llm.text(model="claude-opus-5", effort="low", system="s", content=[text_block("t")])
        # 100 input, 80 cache read, 50 output at Opus 5 prices.
        assert llm.stats.cost_usd == pytest.approx((100 * 5 + 80 * 0.5 + 50 * 25) / 1e6)


def test_model_price_prefixes():
    assert model_price("claude-opus-5") == (5.0, 25.0, 0.5)
    assert model_price("claude-opus-5-5") == (4.0, 20.0, 0.2)  # longest prefix wins
    assert model_price("claude-fable-5-1")[2] == 0.25 and model_price("claude-fable-5")[2] == 1.0
    assert model_price("anthropic.claude-sonnet-5") == (2.0, 10.0, 0.2)
    assert model_price("claude-sonnet-4-6")[0] == 3.0 and model_price("claude-haiku-4-5")[0] == 1.0
    assert model_price(None) == model_price("something-new") == model_price("claude-opus-5")


def _clear_auth_env(monkeypatch, home):
    for var in (
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR",
        "ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID", "ANTHROPIC_IDENTITY_TOKEN",
        "ANTHROPIC_IDENTITY_TOKEN_FILE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("APPDATA", str(home))


def test_missing_credentials_is_llm_unavailable_not_a_type_error(monkeypatch, tmp_path):
    _clear_auth_env(monkeypatch, tmp_path)
    with FakeAnthropic() as fake:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", fake.base_url)
        llm = LLM(AnthropicConfig())  # real SDK client, no key, no profile
        with pytest.raises(LLMUnavailable, match="ANTHROPIC_API_KEY") as info:
            llm.text(model="claude-opus-5", effort="low", system="s", content=[text_block("t")])
        assert isinstance(info.value.__cause__, TypeError)
        with pytest.raises(LLMUnavailable):
            llm.parse(model="claude-opus-5", effort="low", system="s", content=[text_block("t")], schema=Pick)
        assert fake.requests == [] and llm.stats.failures == 2 and llm.stats.calls == 0
    # A profile that was selected but does not exist is also "unavailable".
    monkeypatch.setenv("ANTHROPIC_PROFILE", "missing")
    with pytest.raises(LLMUnavailable):
        LLM(AnthropicConfig()).text(model="claude-opus-5", effort="low", system="s", content=[text_block("t")])


def test_has_credentials_ignores_an_empty_config_dir(monkeypatch, tmp_path):
    _clear_auth_env(monkeypatch, tmp_path)
    assert not has_credentials()
    cfg_dir = tmp_path / ".config" / "anthropic"
    (cfg_dir / "configs").mkdir(parents=True)
    assert not has_credentials()  # stale / empty directory
    (cfg_dir / "configs" / "default.json").write_text("{}")
    assert has_credentials()
    (cfg_dir / "configs" / "default.json").unlink()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert has_credentials()


def test_unexpected_sdk_exception_becomes_llm_error():
    class Messages:
        def create(self, **kwargs):
            raise RuntimeError("socket exploded\nsecond line")

    client = SimpleNamespace(beta=SimpleNamespace(messages=Messages()))
    llm = LLM(AnthropicConfig(), client=client)
    with pytest.raises(LLMError, match="RuntimeError") as info:
        llm.text(model="m", effort="low", system="s", content=[text_block("t")])
    assert "\n" not in str(info.value)
    with pytest.raises(LLMError):
        llm.parse(model="m", effort="low", system="s", content=[text_block("t")], schema=Pick)
    assert llm.stats.failures == 2 and llm.stats.calls == 0


def test_timeout_is_reported_as_a_timeout_not_a_network_problem():
    import anthropic
    import httpx2 as httpx

    class Messages:
        def create(self, **kwargs):
            raise anthropic.APITimeoutError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))

    client = SimpleNamespace(beta=SimpleNamespace(messages=Messages()))
    llm = LLM(AnthropicConfig(timeout_s=30), client=client)
    with pytest.raises(LLMError, match="超时") as info:
        llm.text(model="m", effort="low", system="s", content=[text_block("t")])
    assert "30" in str(info.value) and llm.stats.last_error == "timeout"


def test_system_prompt_cache_uses_1h_ttl_once_calls_are_more_than_5_minutes_apart():
    # auto = false: the player presses F6 at key rounds (2-1, 3-2, 4-1), often
    # more than 5 minutes apart. A 5 minute entry would be rewritten every time.
    now = [1000.0]
    with FakeAnthropic() as fake:
        llm = LLM(AnthropicConfig(max_calls_per_minute=100), client=fake.client(), clock=lambda: now[0])

        def call(system="BIG SET PROMPT", model="claude-opus-5"):
            fake.queue_text("好")
            llm.text(model=model, effort="low", system=system, content=[text_block("t")])
            return fake.requests[-1]["body"]["system"][0]["cache_control"]

        assert call() == {"type": "ephemeral"}  # first call: 5 minute TTL
        now[0] += 60
        assert call() == {"type": "ephemeral"}  # auto mode cadence: the 5m entry is read and refreshed
        now[0] += 420  # 7 minutes idle: the 5m entry expired, this call writes anyway
        assert call() == {"type": "ephemeral", "ttl": "1h"}
        now[0] += 900
        assert call() == {"type": "ephemeral", "ttl": "1h"}  # the 1h entry is alive: keep reading it
        now[0] += 60
        assert call() == {"type": "ephemeral", "ttl": "1h"}
        # Other prompts (vision vs strategy) and other models are tracked separately.
        assert call(system="VISION PROMPT") == {"type": "ephemeral"}
        assert call(model="claude-sonnet-5") == {"type": "ephemeral"}
        now[0] += 2 * 3600  # idle for hours: nothing survives, the cheaper 5m write
        assert call() == {"type": "ephemeral"}
        # A call stopped by the local rate limit never reached the API: it neither
        # refreshed the entry nor counts as the last use of the prompt.
        llm.limiter = RateLimiter(1)
        assert llm.limiter.try_acquire()
        now[0] += 200
        with pytest.raises(LLMRateLimited):
            llm.text(model="claude-opus-5", effort="low", system="BIG SET PROMPT", content=[text_block("t")])
        llm.limiter = RateLimiter(100)
        now[0] += 200  # 400 s after the last call that reached the API
        assert call() == {"type": "ephemeral", "ttl": "1h"}


def test_one_hour_cache_writes_are_priced_at_2x_input():
    with FakeAnthropic() as fake:
        msg = fake._message_text("好")
        msg["usage"] = {
            "input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 9000,
            "cache_creation": {"ephemeral_1h_input_tokens": 8000, "ephemeral_5m_input_tokens": 1000},
        }
        fake._replies.append((200, msg))
        llm = make_llm(fake)
        llm.text(model="claude-opus-5", effort="low", system="s", content=[text_block("t")])
        assert llm.stats.cache_write_tokens == 9000
        assert llm.stats.cost_usd == pytest.approx((100 * 5 + 1000 * 5 * 1.25 + 8000 * 5 * 2 + 50 * 25) / 1e6)


def test_one_hour_write_without_breakdown_uses_the_requested_ttl():
    now = [0.0]
    with FakeAnthropic() as fake:
        llm = LLM(AnthropicConfig(), client=fake.client(), clock=lambda: now[0])
        fake.queue_text("a")
        llm.text(model="claude-opus-5", effort="low", system="s", content=[text_block("t")])
        now[0] += 600
        msg = fake._message_text("b")
        msg["usage"] = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 1000}
        fake._replies.append((200, msg))
        before = llm.stats.cost_usd
        llm.text(model="claude-opus-5", effort="low", system="s", content=[text_block("t")])
        assert fake.requests[-1]["body"]["system"][0]["cache_control"]["ttl"] == "1h"
        assert llm.stats.cost_usd - before == pytest.approx(1000 * 5 * 2 / 1e6)
