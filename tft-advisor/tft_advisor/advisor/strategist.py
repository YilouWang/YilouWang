"""Claude strategist: refines the offline rules advice with an LLM call.

Never raises: on any failure ``advise`` returns the rules advice unchanged and
``ask`` returns a Chinese error message. The failure reason is kept in
``last_error`` so the app can show it in the status bar.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Callable, Iterable, Optional

from ..config import AnthropicConfig, HotkeyConfig
from ..data.setdata import SetData
from ..llm import LLM, LLMError, text_block
from ..models import ActionType, Advice, AdviceAction, Analysis, GameState, StrategistAdvice
from .prompts import ASK_INSTRUCTIONS, build_state_message, build_strategist_system, trait_effects_text
from .rules import ACTION_MAX, AUGMENT_MORE, FIELD_MAX, HEADLINE_MAX, MAX_ACTIONS, PLAN_MAX, clip, sanitize

ASK_MAX_CHARS = 600

# Labels the dashboard already prints in front of these fields: a reply that
# repeats one would read "装备 装备：...".
_FIELD_LABELS = {
    "comp": ("阵容",),
    "items": ("装备",),
    "positioning": ("站位",),
    "augment": ("海克斯强化", "海克斯", "增强"),
}


def _strip_label(text: str, field: str) -> str:
    for label in _FIELD_LABELS.get(field, ()):
        m = re.match(rf"\s*{re.escape(label)}\s*[:：]\s*", text)
        if m:
            return text[m.end() :]
    return text


# "请点开「Nina」的棋盘…": a request to look at another player's board.
_QUOTED = re.compile(r"「([^」]+)」")
_BOARD_ASK = re.compile(r"(点开|点一下|切到|切换到|打开|查看|看看|去看|看一下).{0,16}棋盘")


def _named_players(text: str, state: GameState) -> set[str]:
    """Other players a text names (quoted with 「」 or by a known name)."""
    me = {state.self_name} | {p.name for p in state.players if p.is_self}
    known = ({p.name for p in state.players} | set(state.opponents)) - me
    names = {n.strip() for n in _QUOTED.findall(text or "")} - me
    names |= {n for n in known if n and len(n) >= 2 and n in (text or "")}
    names.discard("")
    return names


def _looks_like_llm(obj: Any) -> bool:
    return callable(getattr(obj, "parse", None)) and callable(getattr(obj, "text", None))


class ClaudeStrategist:
    """``llm`` is a ``tft_advisor.llm.LLM``. For compatibility with the
    architecture doc a raw ``anthropic.Anthropic`` client is also accepted and
    wrapped in an ``LLM``. Everything after ``set_data`` is keyword-only: a
    reference text passed where ``clock`` was expected would otherwise fail
    every call silently."""

    def __init__(
        self,
        llm: Any,
        cfg: AnthropicConfig,
        set_data: SetData,
        *,
        clock: Callable[[], float] = time.time,
        extra_reference: str = "",
        hotkeys: Optional[HotkeyConfig] = None,
        scout_enabled: bool = True,
        augments: Any = None,
    ) -> None:
        self.llm: LLM = llm if _looks_like_llm(llm) else LLM(cfg, client=llm)
        self.cfg = cfg
        self.set_data = set_data
        self.extra_reference = extra_reference
        self.augments = augments  # Optional[AugmentData]: effect text for offered augments
        # [hotkeys] (keys named in the advice) and [advisor] scout_prompts
        # (False: never ask the player to open another board).
        self.hotkeys = hotkeys or HotkeyConfig()
        self.scout_enabled = scout_enabled
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
                effects = trait_effects_text(self.set_data)
                if effects:
                    reference += "\n\n" + effects
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
            message = build_state_message(
                state, analysis, rules_advice, question, recent_history, scouting=self.scout_enabled, hotkeys=self.hotkeys,
                augments=self.augments,
            )
            out = self.llm.parse(
                model=self.cfg.strategy_model,
                effort=self.cfg.strategy_effort,
                system=self.system_prompt,
                content=[text_block(message)],
                schema=StrategistAdvice,
                purpose="strategy",
            )
            advice = self._convert(out, state, rules_advice, analysis)
        except LLMError as exc:
            self.last_error = str(exc) or exc.__class__.__name__
            return rules_advice
        except Exception as exc:  # SDK / validation surprises must not kill the pipeline
            self.last_error = f"{exc.__class__.__name__}: {exc}"
            return rules_advice
        self.last_error = None
        self.last_advice = advice
        return advice

    def _scout_allowed(self, text: str, kind: Optional[ActionType], state: GameState, analysis: Optional[Analysis]) -> bool:
        """Scouting prompts off: no request to open another board at all. On:
        only boards the planner already asked for (it owns the per-stage cap)."""
        board_ask = kind == ActionType.SCOUT or bool(_BOARD_ASK.search(text or ""))
        named = _named_players(text, state)
        if not self.scout_enabled:
            # kind None = the scout_request field itself: shown as "需要信息".
            return kind is not None and not board_ask
        if not (board_ask or kind is None) or not named:
            return True
        open_targets = {r.target_player for r in (analysis.scout_requests if analysis else []) if r.target_player}
        return named <= open_targets

    def _convert(
        self, out: StrategistAdvice, state: GameState, rules_advice: Advice, analysis: Optional[Analysis] = None
    ) -> Advice:
        actions: list[AdviceAction] = []
        for a in out.actions:
            text = clip(a.text, ACTION_MAX + 8, "…")
            if not text:
                continue
            kind = a.type if isinstance(a.type, ActionType) else ActionType.OTHER
            if not self._scout_allowed(text, kind, state, analysis):
                continue
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

        def opt(value: Optional[str], fallback: Optional[str], field: str = "") -> Optional[str]:
            text = clip(_strip_label(value, field), FIELD_MAX, "…") if value else ""
            return text or fallback

        # A pivot in the actions makes the rules comp (the old line) wrong.
        pivot = any(a.type == ActionType.PIVOT for a in actions)
        comp_fallback = None if pivot and not out.comp else rules_advice.comp
        # Augment left null: the model's own augment action, else the rules
        # text without the "see Claude's advice" pointer (this is that advice).
        aug_fallback = rules_advice.augment
        if aug_fallback and not out.augment:
            picked = next((a.text for a in actions if a.type == ActionType.AUGMENT), None)
            aug_fallback = picked or clip(aug_fallback.replace(AUGMENT_MORE, ""), FIELD_MAX)
        scout = out.scout_request
        if scout and not self._scout_allowed(scout, None, state, analysis):
            scout = None
        scout_fallback = rules_advice.scout_request if self.scout_enabled else None

        stage = str(state.stage) if state.stage is not None else rules_advice.stage
        return Advice(
            headline=headline,
            actions=actions,
            plan=clip(out.plan, PLAN_MAX, "…") or rules_advice.plan,
            comp=opt(out.comp, comp_fallback, "comp"),
            items=opt(out.items, rules_advice.items, "items"),
            positioning=opt(out.positioning, rules_advice.positioning, "positioning"),
            augment=opt(out.augment, aug_fallback, "augment"),
            # Keep the planner's open request when the model does not ask for one.
            scout_request=opt(scout, scout_fallback),
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
            message = build_state_message(
                state, analysis, rules_advice, question=q, scouting=self.scout_enabled, hotkeys=self.hotkeys,
                augments=self.augments,
            )
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
