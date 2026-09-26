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


def clip(text: Optional[str], limit: int, ellipsis: str = "") -> str:
    """Sanitize and cut to ``limit`` characters."""
    t = sanitize(text)
    if len(t) <= limit:
        return t
    if ellipsis:
        return t[: max(0, limit - len(ellipsis))].rstrip("，、,： ") + ellipsis
    return t[:limit].rstrip("，、,： ")


def _join(names: Iterable[str], sep: str = "、") -> str:
    return sep.join(n for n in names if n)


def _norm(name: Optional[str]) -> str:
    return normalize_name(name or "")


def _unit_keys(u: Unit) -> set[str]:
    keys = {_norm(u.name), _norm(u.api_name), _norm(u.api_name.split("_", 1)[-1])}
    keys.discard("")
    return keys


def find_carry(state: GameState, analysis: Analysis) -> Optional[Unit]:
    """The main carry on the board: the top comp's carry if fielded, else the
    board unit holding the most items (ties: higher cost, then star)."""
    board = list(state.board)
    if not board:
        return None
    comp = analysis.comps[0] if analysis.comps else None
    if comp and comp.carry:
        key = _norm(comp.carry)
        for u in board:
            if key and key in _unit_keys(u):
                return u
    best = max(board, key=lambda u: (len(u.items), u.cost or 0, u.star))
    if len(best.items) >= 2:
        return best
    return None


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
        carry = find_carry(state, analysis)
        comp = analysis.comps[0] if analysis.comps else None
        carousel = self._is_carousel(state)
        augment = bool(state.augment_choices) or state.screen_type == ScreenType.AUGMENT_SELECT
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

        buy_text, buy_priority = self._buy(state, analysis)
        if buy_text:
            add(ActionType.BUY, buy_text, buy_priority)

        for kind, text, prio in self._econ_actions(state, analysis, carousel):
            add(kind, text, prio)

        for text, prio in self._item_actions(analysis):
            add(ActionType.ITEM, text, prio)

        sell = self._sell(state, analysis)
        if sell:
            add(ActionType.SELL, sell, 1 if analysis.shop_picks else 2)

        if carry is not None and carry.row is not None and carry.row != 3:
            add(ActionType.POSITION, f"把 {_unit_label(carry)} 移到后排角落", 2)

        for i, req in enumerate(analysis.scout_requests[:2]):
            who = req.target_player or "对手"
            add(ActionType.SCOUT, f"点开「{who}」的棋盘后按 {self.hotkeys.scout}", 2 if i == 0 else 3)

        if analysis.warnings:
            add(ActionType.OTHER, analysis.warnings[0], 3)

        actions.sort(key=lambda a: a.priority)  # stable: keeps insertion order within a priority
        actions = actions[:MAX_ACTIONS]

        headline = self._headline(state, analysis, carousel, carousel_pick, augment)
        return Advice(
            headline=clip(headline, HEADLINE_MAX),
            actions=actions,
            plan=clip(self._plan(state, analysis), PLAN_MAX),
            comp=self._comp_text(comp) or None,
            items=self._items_text(analysis, comp, carry) or None,
            positioning=self._positioning_text(carry, state) or None,
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
        if rec == EconAction.ALL_IN:
            return "血量危险，all-in搜牌"
        if rec == EconAction.LEVEL_AND_ROLL:
            target = econ.target_level or ((state.level or 0) + 1)
            left = self._gold_left_after(state, analysis, level_first=True)
            return f"升到{target}级，搜到{left}金币" if left > 0 else f"升到{target}级，搜光金币"
        can_roll = econ.roll_budget >= self.mech.roll_cost
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
        if econ.reason:
            return econ.reason
        return "按节奏运营，稳住血量"

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
        owned = Counter()
        for u in state.all_units():
            for k in _unit_keys(u):
                owned[k] += u.copies
        makes_upgrade = any(owned.get(_norm(p), 0) >= 2 for p in picks[: len(parts)])
        rolling = analysis.econ.recommendation in (
            EconAction.ROLL,
            EconAction.LEVEL_AND_ROLL,
            EconAction.ALL_IN,
        )
        text = "买 " + _join(parts)
        if makes_upgrade:
            text += "，能升星"
        return text, 1 if (makes_upgrade or rolling) else 2

    def _roll_targets(self, analysis: Analysis) -> str:
        econ = analysis.econ
        wanted = [o for o in analysis.odds if o.owned_copies < o.goal_copies]
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
        if options:
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
                out.append((ActionType.ROLL, f"最多花 {budget} 金币搜牌" + self._roll_targets(analysis), 1))
        elif rec == EconAction.ALL_IN:
            if can_roll:
                out.append((ActionType.ROLL, f"最多花 {budget} 金币搜牌，全力补强", 1))
            else:
                out.append((ActionType.OTHER, "没钱搜牌了：调整站位，卖掉没用的牌", 1))
        elif rec == EconAction.SLOW_ROLL and can_roll:
            out.append((ActionType.ROLL, f"最多花 {budget} 金币搜牌，保持 {self._cap_gold()} 利息", 2))
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
            return f"{_unit_label(carry)} 放后排角落，坦克放前排挡伤害，注意防刺客"
        return "主C放后排角落，坦克放前排挡伤害"

    def _augment_text(self, state: GameState) -> str:
        choices = [c for c in state.augment_choices if c]
        if choices:
            return clip(f"可选：{_join(choices)}。优先经济/战力符合当前阵容的，详细对比看 Claude 建议", FIELD_MAX)
        return f"{AUGMENT_GENERIC}，详细对比看 Claude 建议"
