"""Deterministic, offline advice (Simplified Chinese).

``RulesAdvisor.advise(state, analysis)`` turns the tracker state plus the
engine's math into a short list of imperative actions. It never calls the
network and never fails on partial / empty state: every field of ``GameState``
and ``Analysis`` may be missing.

Board rows: 0 is the front row, 3 the back row (see ``models.py``).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Optional

from ..config import HotkeyConfig
from ..data.mechanics import Mechanics, load_mechanics
from ..data.setdata import SetData, normalize_name
from ..models import (
    ActionType,
    Advice,
    AdviceAction,
    Analysis,
    CompSuggestion,
    EconAction,
    GameState,
    ScreenType,
    StageRound,
    Unit,
)

HEADLINE_MAX = 30
ACTION_MAX = 40
PLAN_MAX = 120
FIELD_MAX = 120
MAX_ACTIONS = 6

# Stabilization rounds where a roll-down is standard.
ROLLDOWN_POINTS = {(3, 2), (4, 1), (4, 2)}

AUGMENT_GENERIC = "增强选择: 优先经济/战力符合当前阵容的"
NO_ROLL_GOLD = "金币不够搜牌，先存钱"
DEFAULT_HEADLINE = "按节奏运营，稳住血量"
NO_ROLL_GOLD_ACTION = "金币不够搜牌，这回合先存钱吃利息"

_DASHES = re.compile(r"\s*[\u2014\u2015\u2E3A\u2E3B]+\s*")
_DUP_COMMA = re.compile(r"[，,]\s*[，,]+")


# ---------------------------------------------------------------------------
# Text helpers (shared with strategist.py)
# ---------------------------------------------------------------------------


def sanitize(text: Optional[str], keep_newlines: bool = False) -> str:
    """Remove em dashes (player-facing text must never contain them) and
    collapse whitespace (line breaks survive when ``keep_newlines``)."""
    if not text:
        return ""
    t = _DASHES.sub("，", str(text))
    t = _DUP_COMMA.sub("，", t)
    if keep_newlines:
        lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip().strip("，").strip() for ln in t.split("\n")]
        t = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        return t
    t = re.sub(r"\s+", " ", t).strip()
    return t.strip("，").strip()


_CLAUSE_ENDS = "，。；！？,;"
_OPEN_PARENS = "（("
_CLOSE_PARENS = "）)"


def _cut(t: str, limit: int) -> str:
    """Cut ``t`` to at most ``limit`` chars, preferably at a clause boundary
    outside parentheses (never leaves a half word like "散" of "散件" when a
    whole clause fits) and never with an unclosed parenthesis."""
    head = t[:limit]
    depth = 0
    best = -1
    for i, ch in enumerate(head):
        if ch in _OPEN_PARENS:
            depth += 1
        elif ch in _CLOSE_PARENS:
            depth = max(0, depth - 1)
        elif ch in _CLAUSE_ENDS and depth == 0:
            best = i
    if best >= int(limit * 0.6):
        return head[:best].rstrip("，、,： ")
    opened = max(head.rfind(c) for c in _OPEN_PARENS)
    closed = max(head.rfind(c) for c in _CLOSE_PARENS)
    if opened > closed and opened >= int(limit * 0.5):
        head = head[:opened]
    return head.rstrip("，、,： ")


def clip(text: Optional[str], limit: int, ellipsis: str = "") -> str:
    """Sanitize and cut to ``limit`` characters."""
    t = sanitize(text)
    if len(t) <= limit:
        return t
    if ellipsis:
        return _cut(t, max(0, limit - len(ellipsis))) + ellipsis
    return _cut(t, limit)


def _join(names: Iterable[str], sep: str = "、") -> str:
    return sep.join(n for n in names if n)


def _norm(name: Optional[str]) -> str:
    return normalize_name(name or "")


def _unit_keys(u: Unit) -> set[str]:
    keys = {_norm(u.name), _norm(u.api_name), _norm(u.api_name.split("_", 1)[-1])}
    keys.discard("")
    return keys


# Defensive components: an item built only from these is a tank item.
_DEFENSIVE_COMPONENT_APIS = {"TFT_Item_ChainVest", "TFT_Item_NegatronCloak", "TFT_Item_GiantsBelt"}
# Fallback when set data is not available (en + zh display names, normalized).
_TANK_ITEM_NAMES = {
    normalize_name(n)
    for n in (
        "Chain Vest", "Negatron Cloak", "Giant's Belt", "Warmog's Armor", "Bramble Vest", "Dragon's Claw",
        "Gargoyle Stoneplate", "Sunfire Cape", "Evenshroud", "Redemption", "Spirit Visage", "Protector's Vow",
        "Steadfast Heart",
        "锁子甲", "负极斗篷", "巨人腰带", "狂徒铠甲", "棘刺背心", "巨龙之爪", "石像鬼石板甲", "日炎斗篷",
        "薄暮法袍", "救赎", "圣盾使的誓约", "坚定之心",
    )
}


def is_tank_item(name: str, set_data: Optional[SetData] = None) -> bool:
    """True for defensive items (built only from vest / cloak / belt)."""
    if _norm(name) in _TANK_ITEM_NAMES:
        return True
    if set_data is not None:
        try:
            item = set_data.resolve_item(name)
        except Exception:
            item = None
        if item is not None:
            if item.api_name in _DEFENSIVE_COMPONENT_APIS:
                return True
            comp = list(item.composition or [])
            return bool(comp) and all(c in _DEFENSIVE_COMPONENT_APIS for c in comp)
    return False


def _carry_items(u: Unit, set_data: Optional[SetData]) -> int:
    return sum(1 for i in u.items if i and not is_tank_item(i, set_data))


def find_carry(state: GameState, analysis: Analysis, set_data: Optional[SetData] = None) -> Optional[Unit]:
    """The main carry on the board: the top comp's carry if fielded, else the
    board unit holding the most damage items (at least 2; tank items such as
    Warmog's or Bramble Vest do not count, so an itemized tank is never taken
    for the carry). Ties: higher cost, then star."""
    board = list(state.board)
    if not board:
        return None
    comp = analysis.comps[0] if analysis.comps else None
    if comp and comp.carry:
        key = _norm(comp.carry)
        for u in board:
            if key and key in _unit_keys(u):
                return u
    best = max(board, key=lambda u: (_carry_items(u, set_data), u.cost or 0, u.star))
    if _carry_items(best, set_data) >= 2:
        return best
    return None


AUGMENT_STALE_S = 90.0

# Words the econ engine uses in its reason for reroll (slow roll) lines.
_REROLL_REASON_WORDS = ("慢搜", "赌狗")


def _is_reroll_plan(econ) -> bool:
    return any(w in (econ.reason or "") for w in _REROLL_REASON_WORDS)


def augment_pending(state: GameState) -> bool:
    """True while an augment choice is (still) open.

    The tracker keeps ``augment_choices`` until the round changes, so after the
    pick they linger for the rest of the round. Treat them as done when one of
    them already shows up in ``augments`` or when they were last confirmed long
    before the latest update.
    """
    if state.screen_type == ScreenType.AUGMENT_SELECT:
        return True
    choices = [c for c in state.augment_choices if c and c.strip()]
    if not choices:
        return False
    owned = {_norm(a) for a in state.augments}
    if any(_norm(c) in owned for c in choices):
        return False
    seen = state.field_age.get("augment_choices")
    if seen is not None and state.last_update and state.last_update - seen > AUGMENT_STALE_S:
        return False
    return True


def _unit_label(u: Unit) -> str:
    return u.name or u.api_name.lstrip("?")


class RulesAdvisor:
    """Offline advisor. ``mech`` / ``set_data`` are optional extras: without
    them the bundled mechanics are used and carousel picks rely only on the
    engine's item suggestions."""

    def __init__(
        self,
        hotkeys: Optional[HotkeyConfig] = None,
        mech: Optional[Mechanics] = None,
        set_data: Optional[SetData] = None,
    ) -> None:
        self.hotkeys = hotkeys or HotkeyConfig()
        self.mech = mech or load_mechanics()
        self.set_data = set_data

    # ------------------------------------------------------------------ public
    def advise(self, state: GameState, analysis: Analysis) -> Advice:
        sr = state.stage
        econ = analysis.econ
        carry = find_carry(state, analysis, self.set_data)
        comp = analysis.comps[0] if analysis.comps else None
        carousel = self._is_carousel(state)
        augment = augment_pending(state)
        # Positions read from a combat frame are mid-fight positions, not the
        # player's setup; on the carousel the board / shop cannot be used.
        can_position = not carousel and state.screen_type != ScreenType.COMBAT
        carousel_pick = self._carousel_pick(state, analysis, carry) if carousel else None

        actions: list[AdviceAction] = []

        def add(kind: ActionType, text: str, priority: int) -> None:
            text = clip(text, ACTION_MAX)
            if text:
                actions.append(AdviceAction(type=kind, text=text, priority=max(1, min(3, priority))))

        if augment:
            add(ActionType.AUGMENT, AUGMENT_GENERIC, 1)
        if carousel:
            add(ActionType.CAROUSEL, self._carousel_text(carousel_pick), 1)

        buy_text, buy_priority = ("", 2) if carousel else self._buy(state, analysis)
        if buy_text:
            add(ActionType.BUY, buy_text, buy_priority)

        for kind, text, prio in self._econ_actions(state, analysis, carousel):
            add(kind, text, prio)

        for text, prio in self._item_actions(analysis):
            # Items cannot be moved while on the carousel: do it right after.
            add(ActionType.ITEM, text, max(prio, 2) if carousel else prio)

        sell = "" if carousel else self._sell(state, analysis)
        if sell:
            add(ActionType.SELL, sell, 1 if analysis.shop_picks else 2)

        if can_position and carry is not None and carry.row is not None and carry.row != 3:
            move = f"把 {_unit_label(carry)} 移到后排角落"
            hedged = move + "（近战主C除外）"
            add(ActionType.POSITION, hedged if len(hedged) <= ACTION_MAX else move, 2)

        for i, req in enumerate(analysis.scout_requests[:2]):
            who = req.target_player or "对手"
            add(ActionType.SCOUT, f"点开「{who}」的棋盘后按 {self.hotkeys.scout}", 2 if i == 0 else 3)

        if analysis.warnings:
            add(ActionType.OTHER, analysis.warnings[0], 3)

        headline = clip(self._headline(state, analysis, carousel, carousel_pick, augment), HEADLINE_MAX)
        headline = headline or DEFAULT_HEADLINE  # e.g. an econ reason made only of dashes
        # Do not repeat the headline as a generic action (e.g. the stage-1 econ reason).
        actions = [a for a in actions if not (a.type == ActionType.OTHER and a.text == headline)]
        actions.sort(key=lambda a: a.priority)  # stable: keeps insertion order within a priority
        actions = actions[:MAX_ACTIONS]

        return Advice(
            headline=headline,
            actions=actions,
            plan=clip(self._plan(state, analysis), PLAN_MAX),
            comp=self._comp_text(comp) or None,
            items=self._items_text(analysis, comp, carry) or None,
            positioning=clip(self._positioning_text(carry, state), FIELD_MAX) or None,
            augment=self._augment_text(state) if augment else None,
            scout_request=sanitize(analysis.scout_requests[0].text) if analysis.scout_requests else None,
            confidence=0.3 if sr is None else 0.55,
            source="rules",
            stage=str(sr) if sr is not None else None,
        )

    # ------------------------------------------------------------ situations
    def _is_carousel(self, state: GameState) -> bool:
        if state.screen_type == ScreenType.CAROUSEL:
            return True
        return state.stage is not None and self.mech.is_carousel(state.stage)

    def _cap_gold(self) -> int:
        return self.mech.interest_step * self.mech.interest_cap

    def _level_spend(self, state: GameState, analysis: Analysis) -> Optional[int]:
        econ = analysis.econ
        target = econ.target_level
        if state.level is not None and target is not None:
            spend = self.mech.gold_to_reach(state.level, state.xp_current, target)
            if spend is not None:
                return spend
        return econ.gold_to_next_level

    def _gold(self, state: GameState, analysis: Analysis) -> int:
        return state.gold if state.gold is not None else analysis.econ.gold

    def _gold_left_after(self, state: GameState, analysis: Analysis, level_first: bool) -> int:
        gold = self._gold(state, analysis)
        spend = (self._level_spend(state, analysis) or 0) if level_first else 0
        return max(0, gold - spend - analysis.econ.roll_budget)

    # --------------------------------------------------------------- headline
    def _headline(
        self,
        state: GameState,
        analysis: Analysis,
        carousel: bool,
        carousel_pick: Optional[tuple[str, Optional[str], Optional[str]]],
        augment: bool,
    ) -> str:
        econ = analysis.econ
        rec = econ.recommendation
        pick = analysis.shop_picks[0] if analysis.shop_picks else None
        if augment:
            return "选增强：优先契合当前阵容"
        if carousel:
            if carousel_pick:
                return f"选秀：拿{carousel_pick[0]}"
            return "选秀：拿主C装备散件"
        if state.stage is None:
            return f"按 {self.hotkeys.analyze} 分析当前局面"
        can_roll = econ.roll_budget >= self.mech.roll_cost
        if rec == EconAction.ALL_IN:
            return "血量危险，all-in搜牌" if can_roll else "血量危险：没钱了，调站位卖闲置"
        if rec == EconAction.LEVEL_AND_ROLL:
            target = econ.target_level or ((state.level or 0) + 1)
            if not can_roll:
                return f"升到{target}级"
            left = self._gold_left_after(state, analysis, level_first=True)
            return f"升到{target}级，搜到{left}金币" if left > 0 else f"升到{target}级，搜光金币"
        if rec == EconAction.ROLL and can_roll:
            left = self._gold_left_after(state, analysis, level_first=False)
            low = state.hp is not None and state.hp < 45
            return f"搜牌到{left}金币" + ("稳血" if low else "")
        if rec == EconAction.LEVEL:
            target = econ.target_level or ((state.level or 0) + 1)
            return f"升到{target}级" + (f"，买{pick}" if pick else "")
        slam = next((s for s in analysis.items if s.priority == 1), None)
        if slam:
            return f"合成{slam.item}给{slam.holder}" if slam.holder else f"合成{slam.item}"
        if rec == EconAction.SLOW_ROLL and can_roll:
            return f"慢搜：只花{econ.roll_budget}金币"
        if rec == EconAction.SAVE:
            gold = self._gold(state, analysis)
            cap = self._cap_gold()
            if pick:
                return f"买{pick}，其余存钱"
            return f"存钱到{cap}" if gold < cap else f"保持{cap}金币吃满利息"
        if pick:
            return f"买{pick}"
        if rec in (EconAction.ROLL, EconAction.SLOW_ROLL) and not can_roll:
            # The engine wants a roll but the budget cannot pay for one: its
            # reason text would tell the player to roll, so do not echo it.
            return NO_ROLL_GOLD
        if econ.reason:
            return econ.reason
        return DEFAULT_HEADLINE

    # ---------------------------------------------------------------- actions
    def _slot_of(self, state: GameState, name: str, used: set[int]) -> Optional[int]:
        key = _norm(name)
        names: list[Optional[str]] = []
        if state.shop_units:
            names = [(u.name if u else None) for u in state.shop_units]
            apis = [(u.api_name if u else None) for u in state.shop_units]
        else:
            names = [s.name for s in state.shop]
            apis = [None] * len(names)
        for i, (n, api) in enumerate(zip(names, apis)):
            if i in used:
                continue
            if key and (key == _norm(n) or (api and key == _norm(api))):
                used.add(i)
                return i + 1
        return None

    def _buy(self, state: GameState, analysis: Analysis) -> tuple[str, int]:
        picks = [p for p in analysis.shop_picks if p]
        if not picks:
            return "", 2
        used: set[int] = set()
        parts: list[str] = []
        for name in picks[:3]:
            slot = self._slot_of(state, name, used)
            part = f"{name}（第{slot}格）" if slot else name
            candidate = "买 " + _join(parts + [part])
            if parts and len(candidate) > ACTION_MAX:
                break
            parts.append(part)
        owned: Counter[str] = Counter()
        for u in state.all_units():
            for k in _unit_keys(u):
                owned[k] += u.copies
        rolling = analysis.econ.recommendation in (
            EconAction.ROLL,
            EconAction.LEVEL_AND_ROLL,
            EconAction.ALL_IN,
        )
        while True:
            bought = Counter(_norm(p) for p in picks[: len(parts)])
            # Three copies of the same star level merge: only the leftover
            # one-star copies (total copies mod 3) count toward the next upgrade.
            makes_upgrade = any(k and owned.get(k, 0) % 3 + n >= 3 for k, n in bought.items())
            text = "买 " + _join(parts) + ("，能升星" if makes_upgrade else "")
            if len(text) <= ACTION_MAX or len(parts) <= 1:
                break
            parts.pop()  # keep the upgrade note rather than a low-priority third unit
        return text, 1 if (makes_upgrade or rolling) else 2

    def _roll_text(self, analysis: Analysis) -> str:
        base = f"最多花 {analysis.econ.roll_budget} 金币搜牌"
        full = base + self._roll_targets(analysis, with_odds=True)
        if len(full) <= ACTION_MAX:
            return full
        short = base + self._roll_targets(analysis, with_odds=False)
        return short if len(short) <= ACTION_MAX else base

    def _roll_targets(self, analysis: Analysis, with_odds: bool = True) -> str:
        econ = analysis.econ
        # Units whose remaining pool cannot complete the goal are not worth rolling for.
        wanted = [
            o
            for o in analysis.odds
            if o.owned_copies < o.goal_copies and o.remaining_in_pool >= o.goal_copies - o.owned_copies
        ]
        if not wanted:
            return ""
        comp = analysis.comps[0] if analysis.comps else None
        if comp and comp.carry:
            key = _norm(comp.carry)
            wanted.sort(key=lambda o: 0 if key in (_norm(o.unit), _norm(o.api_name)) else 1)
        first = wanted[0]
        text = f"，找 {first.unit}"
        budget = econ.roll_budget
        options = sorted(g for g in first.p_goal_by_gold if g <= budget)
        if options and with_odds:
            p = first.p_goal_by_gold[options[-1]]
            text += f"（{first.goal_star}星概率 {round(p * 100)}%）"
        return text

    def _econ_actions(
        self, state: GameState, analysis: Analysis, carousel: bool
    ) -> list[tuple[ActionType, str, int]]:
        econ = analysis.econ
        rec = econ.recommendation
        budget = econ.roll_budget
        can_roll = budget >= self.mech.roll_cost
        out: list[tuple[ActionType, str, int]] = []
        if state.stage is None:
            out.append((ActionType.OTHER, f"进入对局后按 {self.hotkeys.analyze} 分析", 1))
            return out
        if carousel:
            return out
        if rec in (EconAction.LEVEL, EconAction.LEVEL_AND_ROLL):
            target = econ.target_level or ((state.level or 0) + 1)
            spend = self._level_spend(state, analysis)
            cost = f"（约 {spend} 金币）" if spend else ""
            out.append((ActionType.LEVEL, f"买经验升到 {target} 级{cost}", 1))
        if rec in (EconAction.LEVEL_AND_ROLL, EconAction.ROLL):
            if can_roll:
                out.append((ActionType.ROLL, self._roll_text(analysis), 1))
            elif rec == EconAction.ROLL:
                out.append((ActionType.SAVE, NO_ROLL_GOLD_ACTION, 2))
        elif rec == EconAction.ALL_IN:
            if can_roll:
                out.append((ActionType.ROLL, f"最多花 {budget} 金币搜牌，全力补强", 1))
            else:
                out.append((ActionType.OTHER, "没钱搜牌了：调整站位，卖掉没用的牌", 1))
        elif rec == EconAction.SLOW_ROLL:
            if can_roll:
                out.append((ActionType.ROLL, f"最多花 {budget} 金币搜牌，保持 {self._cap_gold()} 利息", 2))
            else:
                out.append((ActionType.SAVE, NO_ROLL_GOLD_ACTION, 2))
        elif rec == EconAction.SAVE:
            out.append((ActionType.SAVE, econ.reason or "不搜牌，存钱吃利息", 2))
        elif rec == EconAction.HOLD and econ.reason:
            out.append((ActionType.OTHER, econ.reason, 2))
        return out

    def _item_actions(self, analysis: Analysis) -> list[tuple[str, int]]:
        slams = [s for s in analysis.items if s.priority == 1][:2]
        out = [(f"合成 {s.item} 给 {s.holder}" if s.holder else f"合成 {s.item}", 1) for s in slams]
        if not out:
            nxt = next((s for s in analysis.items if s.priority == 2), None)
            if nxt:
                out.append((f"准备合成 {nxt.item}" + (f" 给 {nxt.holder}" if nxt.holder else ""), 3))
        return out

    def _sell(self, state: GameState, analysis: Analysis) -> str:
        if len(state.bench) < self.mech.bench_size or not analysis.sell_candidates:
            return ""
        return "备战席满了，卖掉 " + _join(analysis.sell_candidates[:2])

    # --------------------------------------------------------------- carousel
    def _carousel_pick(
        self, state: GameState, analysis: Analysis, carry: Optional[Unit]
    ) -> Optional[tuple[str, Optional[str], Optional[str]]]:
        """(component, completed item, holder) worth taking on the carousel."""
        bench = Counter(_norm(x) for x in state.item_bench)

        def missing(components: list[str]) -> list[str]:
            have = bench.copy()
            out = []
            for c in components:
                k = _norm(c)
                if have.get(k, 0) > 0:
                    have[k] -= 1
                else:
                    out.append(c)
            return out

        for sug in sorted(analysis.items, key=lambda s: s.priority):
            miss = missing(list(sug.components))
            if miss:
                return miss[0], sug.item, sug.holder
        comp = analysis.comps[0] if analysis.comps else None
        if comp and comp.carry_items and self.set_data is not None:
            held = {_norm(i) for i in (carry.items if carry else [])}
            for name in comp.carry_items:
                item = self.set_data.resolve_item(name)
                if item is None or not item.composition or _norm(item.name) in held or _norm(name) in held:
                    continue
                parts = [self.set_data.items[c].name for c in item.composition if c in self.set_data.items]
                miss = missing(parts)
                if miss:
                    return miss[0], item.name, comp.carry
        return None

    def _carousel_text(self, pick: Optional[tuple[str, Optional[str], Optional[str]]]) -> str:
        if not pick:
            return "选秀优先拿主C需要的装备散件"
        component, item, holder = pick
        if item and holder:
            return f"选秀拿 {component}（给 {holder} 做 {item}）"
        if item:
            return f"选秀拿 {component}（做 {item}）"
        return f"选秀拿 {component}"

    # ------------------------------------------------------------ text fields
    def _plan(self, state: GameState, analysis: Analysis) -> str:
        sr = state.stage
        if sr is None:
            return ""
        econ = analysis.econ
        level = state.level or 0
        parts: list[str] = []
        if econ.recommendation == EconAction.ALL_IN:
            parts.append("接下来每回合搜到没钱，优先稳血保名次")
        elif sr.stage == 1:
            parts.append("2-1 升 4 级，前期买对子凑羁绊，之后开始存钱")
        elif _is_reroll_plan(econ):
            # Reroll lines stay low and slow roll; the standard level curve
            # (4-1 L7, 4-5 L8) would contradict the econ advice.
            stay = level
            if econ.recommendation == EconAction.LEVEL and econ.target_level:
                parts.append(f"现在升 {econ.target_level}")
                stay = max(level, econ.target_level)
            cap = self._cap_gold()
            parts.append(f"停在 {stay} 级慢搜三星，只花 {cap} 以上的钱" if stay else f"慢搜三星，只花 {cap} 以上的钱")
            parts.append("主C三星后再升级补强")
        else:
            std = self.mech.standard_level_at(sr)
            if econ.recommendation in (EconAction.LEVEL, EconAction.LEVEL_AND_ROLL) and econ.target_level:
                now = f"现在升 {econ.target_level}"
                if econ.recommendation == EconAction.LEVEL_AND_ROLL:
                    now += " 搜牌稳血"
                parts.append(now)
                level = max(level, econ.target_level)
            elif std is not None and level and level < std:
                parts.append(f"尽快补到 {std} 级")
            upcoming: list[tuple[StageRound, int]] = []
            for key, lvl in self.mech.standard_levels.items():
                r = StageRound.parse(key)
                if r is not None and r.key > sr.key and lvl > max(level, std or 0):
                    upcoming.append((r, lvl))
            upcoming.sort(key=lambda x: x[0].key)
            for r, lvl in upcoming[:2]:
                text = f"{r.stage}-{r.round} 升 {lvl}"
                if lvl == 8:
                    text += " 找 4 费主C"
                elif lvl >= 9:
                    text += " 找 5 费"
                elif r.key in ROLLDOWN_POINTS:
                    text += " 搜牌稳血"
                parts.append(text)
            if not upcoming:
                if level >= 9:
                    parts.append("慢搜找 5 费和三星，补满最强阵容")
                else:
                    parts.append("利息满后升 9 找 5 费")
            gold = self._gold(state, analysis)
            if econ.recommendation == EconAction.SAVE and gold < self._cap_gold() and sr.stage <= 4:
                parts.insert(0, f"先存到 {self._cap_gold()}")
        if state.hp is not None and 0 < state.hp < 45 and econ.recommendation != EconAction.ALL_IN:
            parts.append("血量偏低，优先稳血别贪经济")
        comp = analysis.comps[0] if analysis.comps else None
        if comp:
            parts.append(f"目标 {comp.name}")
        return "，".join(parts)

    def _comp_text(self, comp: Optional[CompSuggestion]) -> str:
        if comp is None:
            return ""
        if comp.missing_units:
            text = f"{comp.name}：还缺 {_join(comp.missing_units[:4])}"
        else:
            text = f"{comp.name}：核心已齐"
        if comp.contested_by:
            text += f"（{_join(comp.contested_by[:2])} 在抢）"
        return clip(text, FIELD_MAX)

    def _items_text(self, analysis: Analysis, comp: Optional[CompSuggestion], carry: Optional[Unit]) -> str:
        if analysis.items:
            parts = [f"{s.item} 给 {s.holder}" if s.holder else s.item for s in analysis.items[:3]]
            return clip("装备：" + _join(parts, "，"), FIELD_MAX)
        if comp and comp.carry_items:
            who = comp.carry or (carry.name if carry else "主C")
            return clip(f"{who} 装备：{_join(comp.carry_items[:3])}", FIELD_MAX)
        return ""

    def _positioning_text(self, carry: Optional[Unit], state: GameState) -> str:
        if not state.board:
            return ""
        if carry is not None:
            return f"{_unit_label(carry)} 放后排角落（近战主C除外），坦克放前排挡伤害，注意防刺客"
        return "主C放后排角落，坦克放前排挡伤害"

    def _augment_text(self, state: GameState) -> str:
        choices = [c for c in state.augment_choices if c]
        if choices:
            return clip(f"可选：{_join(choices)}。优先经济/战力符合当前阵容的，详细对比看 Claude 建议", FIELD_MAX)
        return f"{AUGMENT_GENERIC}，详细对比看 Claude 建议"
