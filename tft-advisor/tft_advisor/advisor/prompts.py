"""Prompts for the Claude strategist.

The system prompt is large and stable (instructions + set reference) so it is
prompt-cache friendly: no timestamps, no per-game data. Everything that
changes goes into the user message built by ``build_state_message``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional

from ..config import HotkeyConfig
from ..models import Advice, Analysis, GameState, HistoryPoint, OpponentSnapshot, Unit
from .rules import is_display_name

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
- comp, items, positioning, augment, scout_request: each at most 60 characters, one sentence.
- Call a Wisp 精灵 (keep the Wisp's own name as the shop shows it). When you use a term from the set notes or the \
comp library, write the Chinese name from the set reference (for example Blossom -> 灵魂莲华, Juggernaut -> 主宰, \
fast 8 -> 速8, reroll -> 赌狗, roll-down -> 搜牌). Never write English words in the advice except names that appear in \
English in the snapshot.
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
- Augments: when augment_choices is non-empty, the augment field must name one of the choices as the best pick \
with one short reason (economy vs combat power, fit with the current board and items, stage of the game).
- augment_info (when present) gives each offered or owned augment's effect text, rarity, category and tier \
("S" = rated S by a public tier list when this tool was built: a static hint, not live data). Use the effect \
text; still judge the fit with the board, items, HP and stage.
- Unknown augments and Wisps: this set may be newer than what you know. If an augment or Wisp has no \
augment_info and you do not know it, judge it from its name and category, write 不确定 in the reason and lower \
confidence; never invent numbers or effects.
- Carousel: the component (or unit) that completes the carry's items.
- Pivots: if the top comp is contested (contested_by) or the key units are drained from the pool, suggest the \
best alternative from comps.

USE THE MATH
- econ is a deterministic plan (save / level / roll, roll_budget, target_level). odds give, per unit, \
p_in_shop (chance one fresh shop shows it) and p_goal_by_gold (gold budget -> probability of reaching the \
goal star level), computed at that entry's level (the level after the planned level up when econ levels first). comps, items, shop_picks and sell_candidates come from the same engine.
- Prefer these numbers over your own estimates. Do not invent probabilities. You may overrule the engine when \
the game situation clearly calls for it (for example HP, a strong board, a contested line); say why in one \
short clause.
- econ.style is the line the engine plans for (standard, fast8, fast9, reroll1/2/3). If it differs from the comp's \
library style, the player has not committed to that line yet (for example no reroll pairs): follow econ.style. \
Reroll lines stop slow rolling once the target is 3-star or from 4-5. Win/loss streaks can skip a roll-down point; \
econ already accounts for that. econ.level_estimated: the level was not read, a guess was used.
- rules is the offline baseline advice already shown to the player. Improve on it; correct it when it is wrong. \
A null comp, items, positioning or augment in your answer keeps the rules text for that field (shown in rules); \
fill the field whenever you disagree with it.

MISSING INFORMATION
- hotkeys gives the player's keys: analyze (analyse now) and scout (record the board on screen). Use exactly \
these keys. When hotkeys is "off" (no global hotkeys, for example on a phone), tell the player to tap the \
dashboard button 分析 or 记录对手 instead of a key.
- scouting false: the player turned scouting prompts off. Never ask to open another player's board: \
scout_request null, no scout actions.
- scouting true: open_scout_requests lists the players the player has already been asked to scout (the \
request is on screen). scout_request may only name one of them (or stay null, which keeps that request); \
never ask for any other board (the engine picks who is worth scouting and caps requests per stage). If a \
decision depends on a board you cannot see, say so briefly and lower confidence.
- If the snapshot looks unreadable or not in a game, say so briefly and ask the player to press the analyze \
key again.

RULES YOU MUST FOLLOW
- You only advise. Never recommend macros, scripts, auto-buy, input automation, memory reading or any other \
third-party automation. The player performs every action manually.
- Do not try to predict or reveal which opponent the player will fight next.
- confidence: 0..1, lower it when the snapshot is incomplete or contradictory.

SNAPSHOT FORMAT
- Board rows: row 0 is the front row, row 3 the back row; col 0..6 left to right. star is 1..4 \
(4 = a Set 18 four-star, same 9 copies as a three-star).
- Opponent unit strings look like "Name*2" (a two-star Name). opponents.*.stage is the round when that board \
was recorded; older snapshots may be outdated.
- players lists every player's HP; self marks the player you are coaching. streak: +N is an N-round win \
streak, -N a loss streak.
- odds: owned and goal_copies count 1-star copies (a 2-star = 3, a 3-star = 9); exp_gold is the expected gold to \
reach goal_star; p_goal_by_gold keys are gold budgets.
- shop entries are "Name(cost)"; null is an empty or unread slot. An entry starting with "Wisp: " is a Set 18 \
Wisp, not a champion ("Wisp: X(n)" = the Wisp X for n gold; no (n) when the price was not read): judge it like a \
small augment and say in the buy advice whether to take it.
- comps[0] may carry positions (the comp library's spot per unit as [row, col], rows as on the board) and \
item_holders (the library's items per unit, tanks included): use them for positioning and item advice.
- A "?" in a unit's items is an item icon the vision model could not name (未识别的装备): the slot is taken, but \
do not guess which item it is.
- comp, items, positioning and augment are shown under their own labels (阵容, 装备, 站位, 海克斯): write only \
the content, no leading label. items example: 无尽之刃 给 芸阿娜.
"""

ASK_INSTRUCTIONS = (
    "The player asked the question above. Answer it directly in Simplified Chinese plain text: "
    "at most 5 short sentences, no JSON, no markdown headings, no em dash characters. "
    "Use the snapshot and the engine math; if you need another player's board and scouting is true, "
    "say which one to open (use the snapshot's hotkeys)."
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


# Odds rows: budgets sent to the model (plus the largest one within the
# current roll budget), and rows below this best chance are noise unless the
# unit is the carry or a reroll 3-star target.
ODDS_BUDGETS = (20, 40, 60, 80)
ODDS_MIN_P = 0.02

TRAIT_DESC_MAX = 110
_TRAIT_ICON = re.compile(r"%i:\w+%|<[^>]*>|@[^@\s]*@")
_TRAIT_BREAK = re.compile(r"\(X\)")
_TRAIT_WORDS = re.compile(r"[A-Za-z]{3,}|[\u4e00-\u9fff]")


def trait_effects_text(set_data: Any) -> str:
    """TRAIT EFFECTS section for the cached reference: each trait's effect
    text (numbers are X in the export), clipped. Deterministic."""
    traits = getattr(set_data, "traits", None) or {}
    lines = []
    for t in sorted(traits.values(), key=lambda t: (t.name_en, t.api_name)):
        # Breakpoint rows made only of placeholders ("X% | X% | X%") are noise.
        segs = [re.sub(r"\s+", " ", x).strip() for x in _TRAIT_BREAK.split(_TRAIT_ICON.sub("", t.desc or ""))]
        desc = " | ".join(x for x in segs if _TRAIT_WORDS.search(x))
        if not desc:
            continue
        if len(desc) > TRAIT_DESC_MAX:
            desc = desc[: TRAIT_DESC_MAX - 1].rstrip(" |,.") + "…"
        nm = t.name if t.name == t.name_en else f"{t.name} / {t.name_en}"
        lines.append(f"- {nm}: {desc}")
    if not lines:
        return ""
    return "TRAIT EFFECTS (name / english: effect; X = a number):\n" + "\n".join(lines)


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


def _slot_str(name: Optional[str], cost: Optional[int]) -> Optional[str]:
    if not name:
        return None
    return f"{name}({cost})" if cost is not None else name


def _shop(state: GameState) -> list[Optional[str]]:
    """Resolved champions when available; slots without a unit (a Set 18
    Wisp is stored as ``shop_units[i] = None``) fall back to the raw slot so
    the Wisp and its price still reach the model."""
    slots = list(state.shop)
    if not state.shop_units:
        return [_slot_str(s.name, s.cost) for s in slots]
    out: list[Optional[str]] = []
    for i in range(max(len(state.shop_units), len(slots))):
        u = state.shop_units[i] if i < len(state.shop_units) else None
        if u is not None:
            out.append(_slot_str(u.name or u.api_name, u.cost))
        elif i < len(slots):
            out.append(_slot_str(slots[i].name, slots[i].cost))
        else:
            out.append(None)
    return out


def _state_dict(
    state: GameState,
    analysis: Analysis,
    rules_advice: Optional[Advice],
    recent_history: Optional[Iterable[Any]],
    lim: dict[str, int],
    opp_bench: bool,
    odds_detail: bool,
    scouting: bool = True,
    hotkeys: Optional[HotkeyConfig] = None,
    augments: Any = None,
) -> dict[str, Any]:
    xp = None
    if state.xp_current is not None or state.xp_needed is not None:
        xp = f"{state.xp_current if state.xp_current is not None else '?'}/{state.xp_needed if state.xp_needed is not None else '?'}"

    players = [{"name": p.name, "hp": p.hp, "self": True if p.is_self else None} for p in state.players]

    opponents: dict[str, Any] = {}
    snaps = sorted(state.opponents.values(), key=lambda s: -(s.captured_at or 0))[: lim["opponents"]]
    for snap in snaps:
        opponents[snap.player] = _opponent(snap, lim["opp_units"], opp_bench)

    econ = analysis.econ
    top = analysis.comps[0] if analysis.comps else None
    carry_key = (top.carry or "").strip().lower() if top else ""
    reroll = (econ.style or "").startswith("reroll")

    def keep_row(o: Any) -> bool:
        if not o.p_goal_by_gold or max(o.p_goal_by_gold.values()) >= ODDS_MIN_P:
            return True
        if carry_key and carry_key in (o.unit.strip().lower(), o.api_name.strip().lower()):
            return True
        return reroll and o.goal_star >= 3

    def budgets(p: dict[int, float]) -> dict[str, float]:
        keys = sorted(p)
        wanted = set(ODDS_BUDGETS)
        within = [k for k in keys if k <= econ.roll_budget]
        if within:
            wanted.add(within[-1])
        picked = [k for k in keys if k in wanted]
        if len(picked) < 2:
            picked = keys
        return {str(k): round(p[k], 3) for k in picked}

    odds = []
    for o in [o for o in analysis.odds if keep_row(o)][: lim["odds"]]:
        d: dict[str, Any] = {
            "unit": o.unit,
            "cost": o.cost,
            "owned": o.owned_copies,
            "goal_star": o.goal_star,
            "goal_copies": o.goal_copies,
            "level": o.level,
            "remaining": o.remaining_in_pool,
            "seen_elsewhere": o.seen_elsewhere,
            "p_in_shop": round(o.p_in_shop, 3),
            "exp_gold": o.expected_gold_to_goal,
        }
        if odds_detail and o.p_goal_by_gold:
            d["p_goal_by_gold"] = budgets(o.p_goal_by_gold)
        odds.append(d)

    comps = []
    for n, c in enumerate(analysis.comps[: lim["comps"]]):
        entry = {
            "name": c.name,
            "score": round(c.score, 2),
            "have": c.have_units,
            "missing": c.missing_units,
            "style": getattr(c, "style", None),
            "tier": getattr(c, "tier", None),
            "carry": c.carry,
            "carry_items": [i for i in c.carry_items if is_display_name(i)],
            "contested_by": c.contested_by,
            "reason": c.reason,
        }
        if n == 0:
            # The library's board layout and item holders, for the top comp only
            # (keeps the per-call payload small).
            positions = {
                u: list(rc) for u, rc in (getattr(c, "positions", None) or {}).items() if is_display_name(u) and len(rc) == 2
            }
            holders = {
                u: [i for i in its if is_display_name(i)]
                for u, its in (getattr(c, "item_holders", None) or {}).items()
                if is_display_name(u)
            }
            holders = {u: its for u, its in holders.items() if its}
            if positions:
                entry["positions"] = positions
            if holders:
                entry["item_holders"] = holders
        comps.append(entry)

    items = [
        {"item": s.item, "components": s.components, "holder": s.holder, "priority": s.priority, "reason": s.reason}
        for s in analysis.items[: lim["items"]]
    ]

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
        "style": econ.style,
        "level_estimated": True if econ.level_estimated else None,
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
            # Shown under their own labels; a null field in the reply keeps them.
            "comp": rules_advice.comp,
            "items": rules_advice.items,
            "positioning": rules_advice.positioning,
            "augment": rules_advice.augment,
            "scout_request": rules_advice.scout_request if scouting else None,
        }

    keys = hotkeys or HotkeyConfig()
    hk: Any = {"analyze": keys.analyze, "scout": keys.scout} if keys.enabled else "off"

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
        "augment_info": _augment_info(augments, [*state.augment_choices, *state.augments]),
        "players": players,
        "opponents": opponents,
        "econ": econ_d,
        "odds": odds,
        "comps": comps,
        "items": items,
        "shop_picks": list(analysis.shop_picks),
        "sell_candidates": list(analysis.sell_candidates),
        "warnings": list(analysis.warnings[: lim["warnings"]]),
        "scouting": scouting,
        "hotkeys": hk,
        "open_scout_requests": (
            [r.target_player for r in analysis.scout_requests if r.target_player] if scouting else []
        ),
        "history": history,
        "rules": rules,
    }


def _dumps(obj: Any) -> str:
    return json.dumps(_prune(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _augment_info(augments: Any, names: list[str]) -> Optional[list[dict[str, Any]]]:
    """Effect text, rarity, category and tier snapshot for offered / owned augments."""
    if not augments or not names:
        return None
    out: list[dict[str, Any]] = []
    for raw in dict.fromkeys(n for n in names if n):
        a = augments.lookup(raw)
        if a is None:
            continue
        out.append(
            {
                "name": raw,
                "effect": a.desc_en or a.desc,
                "rarity": a.rarity or None,
                "category": a.category or None,
                "tier": a.meta_tier or None,
            }
        )
    return out or None


def build_state_message(
    state: GameState,
    analysis: Analysis,
    rules_advice: Optional[Advice],
    question: Optional[str] = None,
    recent_history: Optional[Iterable[Any]] = None,
    scouting: bool = True,
    hotkeys: Optional[HotkeyConfig] = None,
    augments: Any = None,
) -> str:
    """Compact, deterministic user message for the strategist.

    ``scouting``: whether the player wants scouting prompts
    (``[advisor] scout_prompts``); ``hotkeys``: the configured keys."""
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
    payload = _dumps(_state_dict(state, analysis, rules_advice, history_list, lim, opp_bench, odds_detail, scouting, hotkeys, augments))
    step = 0
    while len(payload) > MAX_STATE_CHARS and step < len(reducers):
        reducers[step]()
        if step == 1:
            opp_bench = False
        if step == 3:
            odds_detail = False
        step += 1
        payload = _dumps(_state_dict(state, analysis, rules_advice, history_list, lim, opp_bench, odds_detail, scouting, hotkeys, augments))

    q = (question or "").strip()
    tail = f"PLAYER QUESTION: {q}" if q else DEFAULT_QUESTION
    return f"GAME SNAPSHOT (JSON):\n{payload}\n\n{tail}"
