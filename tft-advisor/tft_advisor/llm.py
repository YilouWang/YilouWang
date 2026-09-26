"""Single place that talks to the Claude API (vision + strategy share it).

* Structured output via ``client.beta.messages.parse(output_format=Model)``.
* Adaptive thinking with a per-call effort level.
* Server-side refusal fallbacks (``fallbacks="default"``) when enabled.
* The big, stable system prompt (set data) is marked for prompt caching.
* A client-side sliding-window rate limit keeps the per-game cost bounded.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from .config import AnthropicConfig

FALLBACK_BETA = "server-side-fallback-2026-07-01"

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Any failure that should make the caller fall back to offline logic."""


class LLMUnavailable(LLMError):
    """No credentials / SDK problem: stop trying until the user fixes it."""


class LLMRefusal(LLMError):
    pass


class LLMRateLimited(LLMError):
    pass


def has_credentials() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    # `ant auth login` stores a profile the SDK picks up automatically.
    return Path(os.path.expanduser("~/.config/anthropic")).is_dir()


class RateLimiter:
    """At most ``max_calls`` calls per ``window_s`` seconds (thread-safe)."""

    def __init__(self, max_calls: int, window_s: float = 60.0, clock=time.monotonic) -> None:
        self.max_calls = max(1, int(max_calls))
        self.window_s = window_s
        self._clock = clock
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            now = self._clock()
            while self._calls and now - self._calls[0] >= self.window_s:
                self._calls.popleft()
            if len(self._calls) >= self.max_calls:
                return False
            self._calls.append(now)
            return True


@dataclass
class LLMStats:
    calls: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    last_latency_s: Optional[float] = None
    last_error: Optional[str] = None
    by_purpose: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "last_latency_s": self.last_latency_s,
            "last_error": self.last_error,
            "by_purpose": dict(self.by_purpose),
        }


class LLM:
    def __init__(self, cfg: AnthropicConfig, client: Any = None, limiter: Optional[RateLimiter] = None) -> None:
        self.cfg = cfg
        self._client = client
        self.limiter = limiter or RateLimiter(cfg.max_calls_per_minute)
        self.stats = LLMStats()
        self._lock = threading.Lock()

    # ---- client -------------------------------------------------------------
    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - dependency is required
                raise LLMUnavailable("anthropic SDK not installed: pip install anthropic") from exc
            try:
                self._client = anthropic.Anthropic(timeout=self.cfg.timeout_s, max_retries=2)
            except Exception as exc:
                raise LLMUnavailable(f"无法创建 Claude 客户端: {exc}") from exc
        return self._client

    def _request_kwargs(self, model: str, effort: str, system: str, content: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": content}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
        }
        if self.cfg.use_fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def _account(self, resp: Any, started: float, purpose: str) -> None:
        with self._lock:
            self.stats.calls += 1
            self.stats.by_purpose[purpose] = self.stats.by_purpose.get(purpose, 0) + 1
            self.stats.last_latency_s = round(time.monotonic() - started, 2)
            usage = getattr(resp, "usage", None)
            if usage is not None:
                self.stats.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                self.stats.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
                self.stats.cache_read_tokens += int(getattr(usage, "cache_read_input_tokens", 0) or 0)
                self.stats.cache_write_tokens += int(getattr(usage, "cache_creation_input_tokens", 0) or 0)

    def _fail(self, msg: str) -> None:
        with self._lock:
            self.stats.failures += 1
            self.stats.last_error = msg

    def _call(self, fn_name: str, purpose: str, kwargs: dict[str, Any]) -> Any:
        if not self.limiter.try_acquire():
            self._fail("rate limited")
            raise LLMRateLimited("本地调用频率上限，跳过这次 Claude 调用")
        import anthropic

        started = time.monotonic()
        try:
            resp = getattr(self.client.beta.messages, fn_name)(**kwargs)
        except anthropic.AuthenticationError as exc:
            self._fail("auth")
            raise LLMUnavailable("Claude API 密钥无效，请检查 ANTHROPIC_API_KEY") from exc
        except anthropic.PermissionDeniedError as exc:
            self._fail("permission")
            raise LLMUnavailable(f"Claude API 权限不足: {exc}") from exc
        except anthropic.NotFoundError as exc:
            self._fail("not found")
            raise LLMUnavailable(f"模型不存在或不可用 ({kwargs.get('model')}): {exc}") from exc
        except anthropic.BadRequestError as exc:
            self._fail(f"bad request: {exc}")
            raise LLMError(f"请求被拒绝 (400): {exc}") from exc
        except anthropic.RateLimitError as exc:
            self._fail("429")
            raise LLMRateLimited("Claude API 限流，稍后再试") from exc
        except anthropic.APIStatusError as exc:
            self._fail(f"status {exc.status_code}")
            raise LLMError(f"Claude API 错误 {exc.status_code}") from exc
        except anthropic.APIConnectionError as exc:
            self._fail("connection")
            raise LLMError("连接 Claude API 失败，检查网络") from exc
        self._account(resp, started, purpose)
        stop = getattr(resp, "stop_reason", None)
        if stop == "refusal":
            self._fail("refusal")
            raise LLMRefusal("Claude 拒绝了这次请求")
        if stop == "max_tokens":
            self._fail("max_tokens")
            raise LLMError("Claude 输出被截断 (max_tokens)")
        return resp

    # ---- public -------------------------------------------------------------
    def parse(
        self,
        *,
        model: str,
        effort: str,
        system: str,
        content: list[dict[str, Any]],
        schema: Type[T],
        purpose: str = "parse",
        max_tokens: int = 16000,
    ) -> T:
        kwargs = self._request_kwargs(model, effort, system, content, max_tokens)
        kwargs["output_format"] = schema
        resp = self._call("parse", purpose, kwargs)
        out = getattr(resp, "parsed_output", None)
        if out is None:
            self._fail("no parsed output")
            raise LLMError("Claude 没有返回可解析的结构化结果")
        return out

    def text(
        self,
        *,
        model: str,
        effort: str,
        system: str,
        content: list[dict[str, Any]],
        purpose: str = "text",
        max_tokens: int = 8000,
    ) -> str:
        kwargs = self._request_kwargs(model, effort, system, content, max_tokens)
        resp = self._call("create", purpose, kwargs)
        parts = [b.text for b in getattr(resp, "content", []) if getattr(b, "type", None) == "text"]
        text = "\n".join(p for p in parts if p).strip()
        if not text:
            raise LLMError("Claude 返回了空回答")
        return text


def image_block(png_bytes: bytes, media_type: str = "image/png") -> dict[str, Any]:
    import base64

    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": base64.standard_b64encode(png_bytes).decode("ascii")},
    }


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}
