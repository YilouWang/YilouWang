from __future__ import annotations

import pytest
from pydantic import BaseModel

from tft_advisor.config import AnthropicConfig
from tft_advisor.llm import FALLBACK_BETA, LLM, LLMError, LLMRateLimited, LLMRefusal, LLMUnavailable, RateLimiter, image_block, text_block

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
