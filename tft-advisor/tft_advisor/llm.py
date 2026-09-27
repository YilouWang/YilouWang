"""Single place that talks to the Claude API (vision + strategy share it).

* Structured output via ``output_config.format`` (JSON schema of a pydantic
  model), validated here only after usage and ``stop_reason`` were checked, so a
  truncated or refused reply is reported as such and its tokens are counted.
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

from pydantic import BaseModel, ValidationError

from .config import AnthropicConfig

FALLBACK_BETA = "server-side-fallback-2026-07-01"

# USD per million tokens: (input, output, cache read). Cache writes (5 minute
# TTL) cost 1.25x input. Longest matching prefix wins; unknown models are
# priced like Claude Opus 5. Only an estimate for the dashboard, not a bill.
_PRICES: dict[str, tuple[float, float, float]] = {
    "claude-fable-5-1": (10.0, 50.0, 0.25),
    "claude-mythos-5-1": (10.0, 50.0, 0.25),
    "claude-fable-5": (10.0, 50.0, 1.0),
    "claude-mythos-5": (10.0, 50.0, 1.0),
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 0.50),
    "claude-opus-4": (5.0, 25.0, 0.50),
    "claude-sonnet-5": (2.0, 10.0, 0.20),
    "claude-sonnet-4": (3.0, 15.0, 0.30),
    "claude-haiku-4": (1.0, 5.0, 0.10),
}
_DEFAULT_PRICE = _PRICES["claude-opus-5"]
_CACHE_WRITE_FACTOR = 1.25


def model_price(model: Optional[str]) -> tuple[float, float, float]:
    """(input, output, cache read) USD per million tokens for a model id (any prefix like ``anthropic.``)."""
    name = str(model or "").lower()
    at = name.find("claude-")
    name = name[at:] if at >= 0 else name
    for prefix in sorted(_PRICES, key=len, reverse=True):
        if name.startswith(prefix):
            return _PRICES[prefix]
    return _DEFAULT_PRICE


def _short(exc: BaseException, limit: int = 120) -> str:
    """One-line, length-capped exception text (SDK and pydantic messages span lines)."""
    text = " ".join(str(exc).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _num(obj: Any, name: str) -> int:
    value = _field(obj, name)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Any failure that should make the caller fall back to offline logic."""


class LLMUnavailable(LLMError):
    """No credentials / SDK problem: stop trying until the user fixes it."""


class LLMRefusal(LLMError):
    pass


class LLMRateLimited(LLMError):
    pass


NO_CREDENTIALS_MESSAGE = "没有可用的 Claude 凭证，请设置 ANTHROPIC_API_KEY"


def _anthropic_config_dir() -> Path:
    """Where the SDK looks for `ant auth login` profiles (same rule as the SDK)."""
    env = os.environ.get("ANTHROPIC_CONFIG_DIR")
    if env:
        return Path(env)
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        return (Path(appdata) if appdata else Path.home() / "AppData" / "Roaming") / "Anthropic"
    return Path.home() / ".config" / "anthropic"


def has_credentials() -> bool:
    """Best-effort check that the SDK can authenticate (no network call)."""
    env = os.environ
    if env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN") or env.get("ANTHROPIC_PROFILE"):
        return True
    if env.get("ANTHROPIC_FEDERATION_RULE_ID") and env.get("ANTHROPIC_ORGANIZATION_ID") and (
        env.get("ANTHROPIC_IDENTITY_TOKEN") or env.get("ANTHROPIC_IDENTITY_TOKEN_FILE")
    ):
        return True
    # `ant auth login` stores a profile the SDK picks up automatically. An
    # empty or stale directory is not a credential.
    cfg_dir = _anthropic_config_dir()
    try:
        if (cfg_dir / "active_config").is_file():
            return True
        configs = cfg_dir / "configs"
        return configs.is_dir() and any(configs.glob("*.json"))
    except OSError:
        return False


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
    cost_usd: float = 0.0  # estimate from the price table, every billed attempt included
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
            "cost_usd": round(self.cost_usd, 4),
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
        usage = getattr(resp, "usage", None)
        # With server-side fallbacks the top-level usage covers only the attempt
        # that produced the message; ``usage.iterations`` lists every billed attempt.
        iterations = _field(usage, "iterations") if usage is not None else None
        entries = list(iterations) if iterations else ([usage] if usage is not None else [])
        resp_model = getattr(resp, "model", None)
        with self._lock:
            self.stats.calls += 1
            self.stats.by_purpose[purpose] = self.stats.by_purpose.get(purpose, 0) + 1
            self.stats.last_latency_s = round(time.monotonic() - started, 2)
            for entry in entries:
                inp, out = _num(entry, "input_tokens"), _num(entry, "output_tokens")
                read, write = _num(entry, "cache_read_input_tokens"), _num(entry, "cache_creation_input_tokens")
                self.stats.input_tokens += inp
                self.stats.output_tokens += out
                self.stats.cache_read_tokens += read
                self.stats.cache_write_tokens += write
                p_in, p_out, p_read = model_price(_field(entry, "model") or resp_model)
                self.stats.cost_usd += (inp * p_in + write * p_in * _CACHE_WRITE_FACTOR + read * p_read + out * p_out) / 1e6

    def _fail(self, msg: str) -> None:
        with self._lock:
            self.stats.failures += 1
            self.stats.last_error = msg

    def _call(self, fn_name: str, purpose: str, kwargs: dict[str, Any]) -> Any:
        if not self.limiter.try_acquire():
            self._fail("rate limited")
            raise LLMRateLimited("本地调用频率上限，跳过这次 Claude 调用")
        import anthropic

        client = self.client  # LLMUnavailable when no client can be built
        started = time.monotonic()
        try:
            resp = getattr(client.beta.messages, fn_name)(**kwargs)
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
        except anthropic.APITimeoutError as exc:  # subclass of APIConnectionError
            self._fail("timeout")
            raise LLMError(f"Claude 响应超时（超过 {self.cfg.timeout_s:.0f} 秒）") from exc
        except anthropic.APIConnectionError as exc:
            self._fail("connection")
            raise LLMError("连接 Claude API 失败，检查网络") from exc
        except anthropic.CredentialsError as exc:  # broken `ant auth login` profile, expired refresh token
            self._fail("credentials")
            raise LLMUnavailable(f"{NO_CREDENTIALS_MESSAGE} ({_short(exc, 80)})") from exc
        except TypeError as exc:
            if "authentication" in str(exc).lower():  # SDK: no api key, token or profile at request time
                self._fail("no credentials")
                raise LLMUnavailable(NO_CREDENTIALS_MESSAGE) from exc
            self._fail(f"TypeError: {_short(exc)}")
            raise LLMError(f"调用 Claude 失败: TypeError: {_short(exc)}") from exc
        except Exception as exc:  # credential refresh, response validation, SDK surprises
            self._fail(f"{type(exc).__name__}: {_short(exc)}")
            raise LLMError(f"调用 Claude 失败: {type(exc).__name__}: {_short(exc)}") from exc
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
        from anthropic import transform_schema

        kwargs = self._request_kwargs(model, effort, system, content, max_tokens)
        kwargs["output_config"] = {
            **kwargs["output_config"],
            "format": {"type": "json_schema", "schema": transform_schema(schema)},
        }
        # ``messages.create`` (not ``messages.parse``): the SDK's parse helper
        # validates inside the HTTP call, so a truncated or refused reply would
        # surface as a validation error before usage and stop_reason are seen.
        resp = self._call("create", purpose, kwargs)
        texts = [b.text for b in getattr(resp, "content", None) or [] if getattr(b, "type", None) == "text" and b.text]
        if not texts:
            self._fail("no structured output")
            raise LLMError("Claude 没有返回可解析的结构化结果")
        try:
            return schema.model_validate_json(texts[-1])
        except ValidationError as exc:
            self._fail(f"invalid structured output: {_short(exc)}")
            first = exc.errors()[0] if exc.errors() else {}
            if first.get("type") == "json_invalid":
                raise LLMError("Claude 返回的 JSON 不完整，无法解析") from exc
            where = ".".join(str(p) for p in first.get("loc", ())) or "-"
            raise LLMError(f"Claude 返回的结构不符合要求 ({exc.error_count()} 处错误，第一处在 {where})") from exc

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
