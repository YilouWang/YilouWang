"""Prompts for the Claude strategist.

The system prompt is large and stable (instructions + set reference) so it is
prompt-cache friendly: no timestamps, no per-game data. Everything that
changes goes into the user message built by ``build_state_message``.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from ..models import Advice, Analysis, GameState, HistoryPoint, OpponentSnapshot, Unit

SYSTEM_PROMPT_STRATEGIST = """\
You are an expert Teamfight Tactics (TFT) coach sitting next to the player during a live game. \
You receive a compact JSON snapshot of the game (read from screenshots by a vision model, so it can be \
incomplete or slightly wrong) plus deterministic math from an offline engine, and you tell the player \
the best thing to do right now.

LANGUAGE AND STYLE
- Answer in Simplified Chinese. Keep champion, trait, item and augment names exactly as they appear in the \
snapshot (the player's client language).
- Be concise and imperative, like a coach talking during the game. No filler, no greetings.
- headline: one line, at most 30 characters, the single most important move.
- actions: 1 to 6 concrete actions ordered by priority; each text at most 40 characters. priority 1 = do it \
now, 2 = this round, 3 = nice to have. Use the action type that fits (buy, sell, level, roll, save, item, \
position, pivot, scout, augment, carousel, other).
- plan: the next 2 to 3 rounds (levels, roll-downs, econ targets), at most 120 characters.
- Never output the em dash character (U+2014) or the doubled Chinese dash (two U+2014 in a row). Use \
commas, colons or parentheses instead.

WHAT TO COACH
- Economy: interest is 1 gold per 10 held, capped at 50 gold. Win/loss streak gold. Do not break an interest \
threshold for low-value spending. Loss streaking is fine only when it is deliberate and HP allows it.
- Tempo: the standard leveling curve (2-1 L4, 2-5 L5, 3-2 L6, 4-1 L7, 4-5 L8, 5-5 L9), fast 8 lines, reroll \
lines that stay low and slow roll above 50 gold.
- Roll-downs: stabilize at 3-2 / 4-1 / 4-2 or whenever HP is low; at critical HP spend everything. Say how much \
gold to roll and which units to look for.
- Items: slam strong items early when they fit the carry or the tank; do not hold components for too long; \
best-in-slot items for the carry; emblems only when they complete a strong breakpoint.
- Positioning: carry in a back corner, tanks in front, spread against AoE, counter-position against assassins \
or hooks when the snapshot of an opponent shows them.
- Augments: when augment_choices is present, fill the augment field: name the best pick and one short reason \
(economy vs combat power, fit with the current board and items, stage of the game).
- Carousel: the component (or unit) that completes the carry's items.
- Pivots: if the top comp is contested (contested_by) or the key units are drained from the pool, suggest the \
best alternative from comps.

USE THE MATH
- econ is a deterministic plan (save / level / roll, roll_budget, target_level). odds give, per unit, \
p_in_shop (chance one fresh shop shows it) and p_goal_by_gold (gold budget -> probability of reaching the \
goal star level at the current level). comps, items, shop_picks and sell_candidates come from the same engine.
- Prefer these numbers over your own estimates. Do not invent probabilities. You may overrule the engine when \
the game situation clearly calls for it (for example HP, a strong board, a contested line); say why in one \
short clause.
- rules is the offline baseline advice already shown to the player. Improve on it; correct it when it is wrong.

MISSING INFORMATION
- If a decision depends on something you cannot see (another player's board for a contest check, a missing \
shop, unreadable gold or level), do not guess: set scout_request to a short Chinese instruction that asks the \
player to open that board or screen (for example: 请点开「玩家名」的棋盘后按 F7) and lower confidence. \
F7 (record a scouted board) and F6 (analyse now) are the default hotkeys; when the rules advice names \
different keys, use those.
- If the snapshot looks unreadable or not in a game, say so briefly and ask the player to press the analyse \
hotkey (F6 by default) again.

RULES YOU MUST FOLLOW
- You only advise. Never recommend macros, scripts, auto-buy, input automation, memory reading or any other \
third-party automation. The player performs every action manually.
- Do not try to predict or reveal which opponent the player will fight next.
- confidence: 0..1, lower it when the snapshot is incomplete or contradictory.

SNAPSHOT FORMAT
- Board rows: row 0 is the front row, row 3 the back row; col 0..6 left to right. star is 1..3.
- Opponent unit strings look like "Name*2" (a two-star Name). opponents.*.stage is the round when that board \
was recorded; older snapshots may be outdated.
- players lists every player's HP; self marks the player you are coaching.
"""

ASK_INSTRUCTIONS = (
    "The player asked the question above. Answer it directly in Simplified Chinese plain text: "
    "at most 5 short sentences, no JSON, no markdown headings, no em dash characters. "
    "Use the snapshot and the engine math; if you need another player's board, say which one to open."
)

DEFAULT_QUESTION = "Give the best advice for this moment."

# Soft budget for the JSON part of the user message (about 6k tokens worst case).
MAX_STATE_CHARS = 14000

LIMITS = {
    "board": 14,
    "bench": 9,
    "odds": 8,
    "comps": 3,
    "items": 6,
    "warnings": 6,
    "history": 8,
    "opponents": 7,
    "opp_units": 16,
}


def build_strategist_system(set_summary: str) -> str:
    """System prompt = stable instructions + set reference (cacheable)."""
    summary = (set_summary or "").strip()
    if not summary:
        return SYSTEM_PROMPT_STRATEGIST
    return f"{SYSTEM_PROMPT_STRATEGIST}\n<set_reference>\n{summary}\n</set_reference>\n"


# ---------------------------------------------------------------------------
# State message
# ---------------------------------------------------------------------------


def _prune(obj: Any) -> Any:
    """Drop None / empty values recursively to save tokens."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            v = _prune(v)
            if v is None or v == [] or v == {} or v == "":
                continue
            out[k] = v
        return out
    if isinstance(obj, list):
        return [_prune(v) for v in obj]
    return obj


def _unit(u: Unit, with_pos: bool) -> dict[str, Any]:
    d: dict[str, Any] = {"name": u.name or u.api_name, "star": u.star, "cost": u.cost, "items": list(u.items)}
    if with_pos:
        d["row"] = u.row
        d["col"] = u.col
    return d


def _unit_str(u: Unit) -> str:
    return f"{u.name or u.api_name}*{u.star}"


def _opponent(snap: OpponentSnapshot, limit: int, with_bench: bool) -> dict[str, Any]:
    d: dict[str, Any] = {
        "stage": snap.stage,
        "level": snap.level,
        "hp": snap.hp,
        "board": [_unit_str(u) for u in snap.board[:limit]],
    }
    if with_bench:
        d["bench"] = [_unit_str(u) for u in snap.bench[:limit]]
    return d


def _history_entry(h: Any) -> Any:
    if isinstance(h, HistoryPoint):
        return {"stage": h.stage, "hp": h.hp, "gold": h.gold, "level": h.level}
    if isinstance(h, Advice):
        return {"stage": h.stage, "advice": h.headline}
    if isinstance(h, dict):
        return {k: v for k, v in h.items() if k not in ("captured_at", "created_at", "ts")}
    return str(h)


def _shop(state: GameState) -> list[Optional[str]]:
    if state.shop_units:
        return [
            (f"{u.name}({u.cost})" if u.cost is not None else u.name) if u is not None else None
            for u in state.shop_units
        ]
    return [
        (f"{s.name}({s.cost})" if s.cost is not None else s.name) if s.name else None for s in state.shop
    ]


def _state_dict(
    state: GameState,
    analysis: Analysis,
    rules_advice: Optional[Advice],
    recent_history: Optional[Iterable[Any]],
    lim: dict[str, int],
    opp_bench: bool,
    odds_detail: bool,
) -> dict[str, Any]:
    xp = None
    if state.xp_current is not None or state.xp_needed is not None:
        xp = f"{state.xp_current if state.xp_current is not None else '?'}/{state.xp_needed if state.xp_needed is not None else '?'}"

    players = [{"name": p.name, "hp": p.hp, "self": True if p.is_self else None} for p in state.players]

    opponents: dict[str, Any] = {}
    snaps = sorted(state.opponents.values(), key=lambda s: -(s.captured_at or 0))[: lim["opponents"]]
    for snap in snaps:
        opponents[snap.player] = _opponent(snap, lim["opp_units"], opp_bench)

    odds = []
    for o in analysis.odds[: lim["odds"]]:
        d: dict[str, Any] = {
            "unit": o.unit,
            "cost": o.cost,
            "owned": o.owned_copies,
            "goal_star": o.goal_star,
            "goal_copies": o.goal_copies,
            "remaining": o.remaining_in_pool,
            "seen_elsewhere": o.seen_elsewhere,
            "p_in_shop": round(o.p_in_shop, 3),
            "exp_gold": o.expected_gold_to_goal,
        }
        if odds_detail and o.p_goal_by_gold:
            d["p_goal_by_gold"] = {str(k): round(v, 3) for k, v in sorted(o.p_goal_by_gold.items())}
        odds.append(d)

    comps = []
    for c in analysis.comps[: lim["comps"]]:
        comps.append(
            {
                "name": c.name,
                "score": round(c.score, 2),
                "have": c.have_units,
                "missing": c.missing_units,
                "carry": c.carry,
                "carry_items": c.carry_items,
                "contested_by": c.contested_by,
                "reason": c.reason,
            }
        )

    items = [
        {"item": s.item, "components": s.components, "holder": s.holder, "priority": s.priority, "reason": s.reason}
        for s in analysis.items[: lim["items"]]
    ]

    econ = analysis.econ
    econ_d = {
        "gold": econ.gold,
        "interest": econ.interest,
        "streak_gold": econ.streak_gold,
        "expected_income": econ.expected_income,
        "xp_to_next": econ.xp_to_next,
        "gold_to_next_level": econ.gold_to_next_level,
        "recommendation": econ.recommendation.value,
        "roll_budget": econ.roll_budget,
        "target_level": econ.target_level,
        "reason": econ.reason,
    }

    source = state.history if recent_history is None else list(recent_history)
    n_hist = lim["history"]
    history = [_history_entry(h) for h in source[-n_hist:]] if n_hist > 0 else []

    rules = None
    if rules_advice is not None:
        rules = {
            "headline": rules_advice.headline,
            "actions": [{"type": a.type.value, "text": a.text, "priority": a.priority} for a in rules_advice.actions],
            "plan": rules_advice.plan,
        }

    return {
        "stage": str(state.stage) if state.stage is not None else None,
        "screen": state.screen_type.value,
        "gold": state.gold,
        "level": state.level,
        "xp": xp,
        "hp": state.hp,
        "streak": state.streak,
        "self_name": state.self_name,
        "board": [_unit(u, True) for u in state.board[: lim["board"]]],
        "bench": [_unit(u, False) for u in state.bench[: lim["bench"]]],
        "item_bench": list(state.item_bench),
        "shop": _shop(state),
        "traits": [
            {"name": t.name, "count": t.count, "active": t.active or None, "next": t.next_breakpoint}
            for t in state.traits
        ],
        "augments": list(state.augments),
        "augment_choices": list(state.augment_choices),
        "players": players,
        "opponents": opponents,
        "econ": econ_d,
        "odds": odds,
        "comps": comps,
        "items": items,
        "shop_picks": list(analysis.shop_picks),
        "sell_candidates": list(analysis.sell_candidates),
        "warnings": list(analysis.warnings[: lim["warnings"]]),
        "open_scout_requests": [r.target_player for r in analysis.scout_requests if r.target_player],
        "history": history,
        "rules": rules,
    }


def _dumps(obj: Any) -> str:
    return json.dumps(_prune(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_state_message(
    state: GameState,
    analysis: Analysis,
    rules_advice: Optional[Advice],
    question: Optional[str] = None,
    recent_history: Optional[Iterable[Any]] = None,
) -> str:
    """Compact, deterministic user message for the strategist."""
    history_list = list(recent_history) if recent_history is not None else None
    lim = dict(LIMITS)
    opp_bench, odds_detail = True, True
    # Progressive trimming until the JSON fits the soft budget.
    reducers = [
        lambda: lim.update(history=3),
        lambda: None,  # drop opponents' benches (flag below)
        lambda: lim.update(odds=5, items=4, warnings=3),
        lambda: None,  # drop per-budget odds detail (flag below)
        lambda: lim.update(opponents=4, opp_units=10, comps=2),
        lambda: lim.update(history=0, opponents=0, odds=3),
    ]
    payload = _dumps(_state_dict(state, analysis, rules_advice, history_list, lim, opp_bench, odds_detail))
    step = 0
    while len(payload) > MAX_STATE_CHARS and step < len(reducers):
        reducers[step]()
        if step == 1:
            opp_bench = False
        if step == 3:
            odds_detail = False
        step += 1
        payload = _dumps(_state_dict(state, analysis, rules_advice, history_list, lim, opp_bench, odds_detail))

    q = (question or "").strip()
    tail = f"PLAYER QUESTION: {q}" if q else DEFAULT_QUESTION
    return f"GAME SNAPSHOT (JSON):\n{payload}\n\n{tail}"
