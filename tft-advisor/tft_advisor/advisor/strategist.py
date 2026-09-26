"""Claude strategist: refines the offline rules advice with an LLM call.

Never raises: on any failure ``advise`` returns the rules advice unchanged and
``ask`` returns a Chinese error message. The failure reason is kept in
``last_error`` so the app can show it in the status bar.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Iterable, Optional

from ..config import AnthropicConfig
from ..data.setdata import SetData
from ..llm import LLM, LLMError, text_block
from ..models import ActionType, Advice, AdviceAction, Analysis, GameState, StrategistAdvice
from .prompts import ASK_INSTRUCTIONS, build_state_message, build_strategist_system
from .rules import ACTION_MAX, FIELD_MAX, HEADLINE_MAX, MAX_ACTIONS, PLAN_MAX, clip, sanitize

ASK_MAX_CHARS = 600


def _looks_like_llm(obj: Any) -> bool:
    return callable(getattr(obj, "parse", None)) and callable(getattr(obj, "text", None))


class ClaudeStrategist:
    """``llm`` is a ``tft_advisor.llm.LLM``. For compatibility with the
    architecture doc a raw ``anthropic.Anthropic`` client is also accepted and
    wrapped in an ``LLM``."""

    def __init__(
        self,
        llm: Any,
        cfg: AnthropicConfig,
        set_data: SetData,
        clock: Callable[[], float] = time.time,
        extra_reference: str = "",
    ) -> None:
        self.llm: LLM = llm if _looks_like_llm(llm) else LLM(cfg, client=llm)
        self.cfg = cfg
        self.set_data = set_data
        self.extra_reference = extra_reference
        self.clock = clock
        self.last_error: Optional[str] = None
        self.last_advice: Optional[Advice] = None
        self._system: Optional[str] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ prompt
    @property
    def system_prompt(self) -> str:
        """Built once per instance: identical bytes every call (prompt cache)."""
        with self._lock:
            if self._system is None:
                reference = self.set_data.summary_text()
                if self.extra_reference.strip():
                    reference += "\n\n" + self.extra_reference.strip()
                self._system = build_strategist_system(reference)
            return self._system

    # ------------------------------------------------------------------ advise
    def advise(
        self,
        state: GameState,
        analysis: Analysis,
        rules_advice: Advice,
        question: Optional[str] = None,
        recent_history: Optional[Iterable[Any]] = None,
    ) -> Advice:
        try:
            message = build_state_message(state, analysis, rules_advice, question, recent_history)
            out = self.llm.parse(
                model=self.cfg.strategy_model,
                effort=self.cfg.strategy_effort,
                system=self.system_prompt,
                content=[text_block(message)],
                schema=StrategistAdvice,
                purpose="strategy",
            )
            advice = self._convert(out, state, rules_advice)
        except LLMError as exc:
            self.last_error = str(exc) or exc.__class__.__name__
            return rules_advice
        except Exception as exc:  # SDK / validation surprises must not kill the pipeline
            self.last_error = f"{exc.__class__.__name__}: {exc}"
            return rules_advice
        self.last_error = None
        self.last_advice = advice
        return advice

    def _convert(self, out: StrategistAdvice, state: GameState, rules_advice: Advice) -> Advice:
        actions: list[AdviceAction] = []
        for a in out.actions:
            text = clip(a.text, ACTION_MAX + 8, "…")
            if not text:
                continue
            kind = a.type if isinstance(a.type, ActionType) else ActionType.OTHER
            try:
                prio = int(a.priority)
            except (TypeError, ValueError):
                prio = 2
            # 0 (or less) means "most urgent" to the model: clamp, never demote to 2.
            actions.append(AdviceAction(type=kind, text=text, priority=max(1, min(3, prio))))
        actions.sort(key=lambda x: x.priority)
        actions = actions[:MAX_ACTIONS] or list(rules_advice.actions)

        headline = clip(out.headline, HEADLINE_MAX, "…") or rules_advice.headline
        try:
            confidence = float(out.confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        def opt(value: Optional[str], fallback: Optional[str]) -> Optional[str]:
            text = clip(value, FIELD_MAX, "…") if value else ""
            return text or fallback

        stage = str(state.stage) if state.stage is not None else rules_advice.stage
        return Advice(
            headline=headline,
            actions=actions,
            plan=clip(out.plan, PLAN_MAX, "…") or rules_advice.plan,
            comp=opt(out.comp, rules_advice.comp),
            items=opt(out.items, rules_advice.items),
            positioning=opt(out.positioning, rules_advice.positioning),
            augment=opt(out.augment, rules_advice.augment),
            # Keep the planner's open request when the model does not ask for one.
            scout_request=opt(out.scout_request, rules_advice.scout_request),
            confidence=confidence,
            source="llm",
            stage=stage,
            created_at=self.clock(),
        )

    # --------------------------------------------------------------------- ask
    def ask(
        self,
        question: str,
        state: GameState,
        analysis: Analysis,
        rules_advice: Optional[Advice] = None,
    ) -> str:
        q = (question or "").strip()
        if not q:
            return "请输入要问的问题"
        try:
            message = build_state_message(state, analysis, rules_advice, question=q)
            text = self.llm.text(
                model=self.cfg.strategy_model,
                effort=self.cfg.strategy_effort,
                system=self.system_prompt,
                content=[text_block(f"{message}\n\n{ASK_INSTRUCTIONS}")],
                purpose="ask",
            )
        except LLMError as exc:
            self.last_error = str(exc) or exc.__class__.__name__
            return f"Claude 暂时无法回答：{sanitize(self.last_error)}"
        except Exception as exc:
            self.last_error = f"{exc.__class__.__name__}: {exc}"
            return f"Claude 暂时无法回答：{sanitize(self.last_error)}"
        self.last_error = None
        answer = sanitize(text, keep_newlines=True)
        if len(answer) > ASK_MAX_CHARS:
            answer = answer[: ASK_MAX_CHARS - 1] + "…"
        return answer
