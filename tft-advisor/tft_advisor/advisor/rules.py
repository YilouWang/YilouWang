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
from typing import Iterable, NamedTuple, Optional

from ..config import HotkeyConfig
from ..data.mechanics import Mechanics, load_mechanics
from ..data.setdata import SetData, normalize_name
from ..engine.economy import REROLL_END, reroll_level
from ..engine.tracker import WISP_PREFIXES, is_wisp_name
from ..vision.base import is_unknown_item
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
MAX_UNIT_ITEMS = 3  # item slots per champion

# Stabilization rounds where a roll-down is standard.
ROLLDOWN_POINTS = {(3, 2), (4, 1), (4, 2)}

# Roll targets: from this stage on only the top comp's units are named (the
# early game holds any pair for tempo, the mid game rolls for the comp).
ROLL_TARGET_LATE_STAGE = 4
# A comp suggestion below this score is a loose guess (same cut as the
# analyzer's STYLE_MIN_SCORE): the fielded board is then the real plan.
ROLL_TARGET_COMMITTED_SCORE = 0.4
ROLL_TARGET_MIN_P = 0.1
# A second target is named only when it is worth this share of the first.
ROLL_TARGET_SECOND_SHARE = 0.25
# Value weights: carry > comp 3+ costs > cheap comp units > off-comp units.
ROLL_VALUE_CARRY = 1.0
ROLL_VALUE_CORE = 0.7
ROLL_VALUE_FILLER = 0.45
ROLL_VALUE_OFF_COMP = 0.3

# "海克斯强化" is the client's term for augments ("增强" is not used in game).
AUGMENT_HINT = "优先选契合当前阵容的经济或战力类"
AUGMENT_GENERIC = f"海克斯强化：{AUGMENT_HINT}"
AUGMENT_HEADLINE = "选海克斯：优先契合当前阵容"
AUGMENT_MORE = "，详细对比看 Claude 建议"
NO_ROLL_GOLD = "金币不够搜牌，先存钱"
DEFAULT_HEADLINE = "按节奏运营，稳住血量"
NO_ROLL_GOLD_ACTION = "金币不够搜牌，这回合先存钱吃利息"

_WISP_PREFIX = re.compile(r"^\s*(?:" + "|".join(re.escape(p) for p in WISP_PREFIXES) + r")\s*[:：]?\s*", re.I)
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


_THIEFS_GLOVES = {normalize_name(n) for n in ("Thief's Gloves", "窃贼手套")}
# Raw API ids (e.g. "DA_Artifact_NavoriFlickerblade") left by a comp library
# entry the set data could not resolve: never shown to the player.
_API_ID = re.compile(r"^[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+$")


def is_display_name(name: Optional[str]) -> bool:
    """False for empty names and raw API ids."""
    return bool(name and name.strip()) and not _API_ID.match(name.strip())


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
    # "?" (an icon vision could not name) is neutral: it may be a tank item.
    return sum(1 for i in u.items if i and not is_unknown_item(i) and not is_tank_item(i, set_data))


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

# Rounds a reroll line reaches its reroll level (economy.py: stage 3, 3-5 for
# 3-cost rerolls that sit at level 7).
REROLL_START = (3, 1)
REROLL_START_L7 = (3, 5)


def _is_reroll_plan(econ, sr: Optional[StageRound] = None) -> bool:
    """True while the econ plan follows a reroll line (stay low, slow roll).

    Decided from ``econ.style``, never from the reason text (stage-2, carousel
    and roll-down reasons do not mention the reroll). The engine resets the
    style to standard once the reroll ends (target 3-star or 4-5); its early
    exits (carousel, stage 1) keep the comp's style, so 4-5 is checked here.
    """
    if not (getattr(econ, "style", "") or "").startswith("reroll"):
        return False
    return sr is None or sr.key < REROLL_END


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


class CarouselPick(NamedTuple):
    component: str
    item: Optional[str]
    holder: Optional[str]
    completes: bool = True  # False: the item still needs another component after this one


def _unit_label(u: Unit) -> str:
    return u.name or u.api_name.lstrip("?")


class RulesAdvisor:
    """Offline advisor. ``mech`` / ``set_data`` are optional extras: without
    them the bundled mechanics are used and carousel picks rely only on the
    engine's item suggestions. ``claude_enabled`` says whether a Claude
    strategist follows up (texts only point to Claude advice when it does)."""

    def __init__(
        self,
        hotkeys: Optional[HotkeyConfig] = None,
        mech: Optional[Mechanics] = None,
        set_data: Optional[SetData] = None,
        claude_enabled: bool = False,
    ) -> None:
        self.hotkeys = hotkeys or HotkeyConfig()
        self.mech = mech or load_mechanics()
        self.set_data = set_data
        self.claude_enabled = claude_enabled

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
        carry_full = carousel and carousel_pick is None and self._carry_full(state, analysis, carry)

        actions: list[AdviceAction] = []

        def add(kind: ActionType, text: str, priority: int) -> None:
            text = clip(text, ACTION_MAX)
            if text:
                actions.append(AdviceAction(type=kind, text=text, priority=max(1, min(3, priority))))

        if augment:
            add(ActionType.AUGMENT, AUGMENT_GENERIC, 1)
        if carousel:
            add(ActionType.CAROUSEL, self._carousel_text(carousel_pick, carry_full), 1)

        buy_text, buy_priority = ("", 2) if carousel else self._buy(state, analysis)
        if buy_text:
            add(ActionType.BUY, buy_text, buy_priority)

        for kind, text, prio in self._econ_actions(state, analysis, carousel):
            add(kind, text, prio)

        for text, prio in self._item_actions(state, analysis):
            # Items cannot be moved while on the carousel: do it right after.
            add(ActionType.ITEM, text, max(prio, 2) if carousel else prio)

        sell = "" if carousel else self._sell(state, analysis)
        if sell:
            add(ActionType.SELL, sell, 1 if analysis.shop_picks else 2)

        melee = self._carry_is_melee(carry)
        if can_position and carry is not None and carry.row is not None and carry.row != 3 and melee is not True:
            move = f"把 {_unit_label(carry)} 移到后排角落"
            hedged = move + "（近战主C除外）"
            add(ActionType.POSITION, move if melee is False or len(hedged) > ACTION_MAX else hedged, 2)

        for i, req in enumerate(analysis.scout_requests[:2]):
            who = req.target_player or "对手"
            add(ActionType.SCOUT, f"点开「{who}」的棋盘后按 {self.hotkeys.scout}", 2 if i == 0 else 3)

        if analysis.warnings:
            add(ActionType.OTHER, analysis.warnings[0], 3)

        wisp = "" if carousel else self._wisp(state)
        if wisp:
            add(ActionType.BUY, wisp, 3)

        headline = clip(self._headline(state, analysis, carousel, carousel_pick, augment, carry_full), HEADLINE_MAX)
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
            items=self._items_text(analysis, comp, carry, state) or None,
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
        # The analyzer pays shop picks out of the roll budget (shop_picks_cost).
        picks = getattr(analysis, "shop_picks_cost", 0) or 0
        return max(0, gold - spend - picks - analysis.econ.roll_budget)

    # --------------------------------------------------------------- headline
    def _headline(
        self,
        state: GameState,
        analysis: Analysis,
        carousel: bool,
        carousel_pick: Optional[CarouselPick],
        augment: bool,
        carry_full: bool = False,
    ) -> str:
        econ = analysis.econ
        rec = econ.recommendation
        pick = analysis.shop_picks[0] if analysis.shop_picks else None
        if augment:
            return AUGMENT_HEADLINE
        if carousel:
            if carousel_pick:
                return f"选秀：拿{carousel_pick[0]}"
            return "选秀：拿坦克装备或缺的英雄" if carry_full else "选秀：拿主C装备散件"
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
        # A built item nobody can take is not headline material.
        slam = next((s for s in analysis.items if s.priority == 1 and (s.components or s.holder)), None)
        if slam:
            if not slam.components:
                return f"把{slam.item}装给{slam.holder}"
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

    def _roll_text(self, state: GameState, analysis: Analysis) -> str:
        base = f"最多花 {analysis.econ.roll_budget} 金币搜牌"
        full = base + self._roll_targets(state, analysis, with_odds=True)
        if len(full) <= ACTION_MAX:
            return full
        short = base + self._roll_targets(state, analysis, with_odds=False)
        return short if len(short) <= ACTION_MAX else base

    def _roll_value(self, o, state: GameState, analysis: Analysis) -> float:
        """How much hitting this unit's goal is worth (0 = not a roll target).

        The carry comes first, then the top comp's 3+ cost units (all comp
        units on a reroll line), then its cheap units. Units the engine wants
        sold are never targets, and neither are unowned off-comp units. From
        stage 4 on, off-comp units only count while there is no committed comp
        (then the fielded board is the plan); before that any owned pair is a
        fine tempo upgrade.
        """
        keys = {_norm(o.unit), _norm(o.api_name), _norm(o.api_name.split("_", 1)[-1])}
        keys.discard("")
        comp = analysis.comps[0] if analysis.comps else None
        carry = _norm(comp.carry) if comp and comp.carry else ""
        if carry and carry in keys:
            return ROLL_VALUE_CARRY
        if keys & {_norm(n) for n in analysis.sell_candidates}:
            return 0.0
        comp_keys = {_norm(n) for n in (comp.core_units + comp.have_units + comp.missing_units)} if comp else set()
        if keys & comp_keys:
            core = (o.cost or 0) >= 3 or _is_reroll_plan(analysis.econ, state.stage)
            return ROLL_VALUE_CORE if core else ROLL_VALUE_FILLER
        if o.owned_copies <= 0:
            return 0.0
        late = state.stage is not None and state.stage.stage >= ROLL_TARGET_LATE_STAGE
        if comp is None or not late:
            return ROLL_VALUE_OFF_COMP
        if comp.score < ROLL_TARGET_COMMITTED_SCORE:
            board = set().union(*(_unit_keys(u) for u in state.board)) if state.board else set()
            if keys & board:
                return ROLL_VALUE_OFF_COMP
        return 0.0

    def _roll_targets(self, state: GameState, analysis: Analysis, with_odds: bool = True) -> str:
        """Name the 1-2 units most worth rolling for with this budget.

        Rolling for a unit that the budget cannot realistically upgrade (for
        example a 4-cost at level 6) is bad advice, and so is rolling for a
        cheap off-comp unit just because it is likely: units are ranked by
        value (carry > comp core > filler, see ``_roll_value``) times the
        probability of reaching their goal star with ``roll_budget`` gold.
        """
        econ = analysis.econ
        budget = econ.roll_budget
        # Units whose remaining pool cannot complete the goal are not worth rolling for.
        wanted = [
            (o, v)
            for o in analysis.odds
            if o.owned_copies < o.goal_copies and o.remaining_in_pool >= o.goal_copies - o.owned_copies
            for v in (self._roll_value(o, state, analysis),)
            if v > 0
        ]
        if not wanted:
            return ""

        def p_at(o) -> float:
            options = sorted(g for g in o.p_goal_by_gold if g <= budget)
            return o.p_goal_by_gold[options[-1]] if options else 0.0

        ranked = sorted(
            ((o, v * p_at(o)) for o, v in wanted if p_at(o) >= ROLL_TARGET_MIN_P),
            key=lambda t: t[1],
            reverse=True,
        )
        if not ranked:
            return "，找对子升星"
        best = ranked[0][1]
        good = [ranked[0][0]] + [o for o, s in ranked[1:2] if s >= ROLL_TARGET_SECOND_SHARE * best]
        if not with_odds:
            return "，找 " + "、".join(o.unit for o in good)
        parts = []
        for o in good:
            p = p_at(o)
            parts.append(f"{o.unit}（{o.goal_star}星{round(p * 100)}%）" if p > 0 else o.unit)
        return "，找 " + "、".join(parts)

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
                out.append((ActionType.ROLL, self._roll_text(state, analysis), 1))
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

    def _stuck_item_text(self, state: GameState, item: str) -> str:
        """A built item on the bench that no unit takes right now: say why."""
        board = list(state.board)
        if not board:
            return f"先上场英雄再装 {item}"
        if _norm(item) in _THIEFS_GLOVES:
            return f"{item} 只能给没装备的英雄，先留着"
        if all(len([i for i in u.items if i]) >= MAX_UNIT_ITEMS for u in board):
            return f"装备格都满了，{item} 先留着"
        return f"现在的阵容没人适合用 {item}，先留着"

    def _item_actions(self, state: GameState, analysis: Analysis) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []
        for s in [s for s in analysis.items if s.priority == 1][:2]:
            if s.components:
                out.append((f"合成 {s.item} 给 {s.holder}" if s.holder else f"合成 {s.item}", 1))
            elif s.holder:
                # Already built (sitting on the bench): equip, do not "combine".
                out.append((f"把 {s.item} 装给 {s.holder}", 1))
            else:
                out.append((self._stuck_item_text(state, s.item), 2))
        if not out:
            nxt = next((s for s in analysis.items if s.priority == 2 and s.components), None)
            if nxt:
                out.append((f"准备合成 {nxt.item}" + (f" 给 {nxt.holder}" if nxt.holder else ""), 3))
        return out

    def _wisp(self, state: GameState) -> str:
        """Hint for a Set 18 Wisp in the shop. Its effect is unknown here, so
        judge it like a small augment (set notes): cheap econ / XP Wisps
        early, combat Wisps when stabilizing."""
        for i, slot in enumerate(state.shop):
            if not (slot.name and is_wisp_name(slot.name)):
                continue
            if slot.cost is not None and state.gold is not None and state.gold < slot.cost:
                return ""
            name = _WISP_PREFIX.sub("", slot.name).strip()
            price = f"（{slot.cost}金币）" if slot.cost is not None else ""
            sr = state.stage
            early = sr is not None and sr.stage <= 3 and (state.hp is None or state.hp >= 45)
            tip = "前期经济或经验类值得买" if early else "稳血时优先买战斗类"
            text = f"第{i + 1}格精灵「{name}」{price}：{tip}" if name else ""
            if not text or len(text) > ACTION_MAX:
                text = f"第{i + 1}格是精灵{price}：{tip}"
            return text
        return ""

    def _sell(self, state: GameState, analysis: Analysis) -> str:
        if len(state.bench) < self.mech.bench_size or not analysis.sell_candidates:
            return ""
        return "备战席满了，卖掉 " + _join(analysis.sell_candidates[:2])

    # --------------------------------------------------------------- carousel
    @staticmethod
    def _holder_full(state: GameState, name: Optional[str]) -> bool:
        """True when the unit called ``name`` cannot take another item (3 slots
        used). Board copies decide; a bench copy only when none is fielded."""
        key = _norm(name)
        if not key:
            return False
        for units in (state.board, state.bench):
            found = [u for u in units if key in _unit_keys(u)]
            if found:
                return all(len([i for i in u.items if i]) >= MAX_UNIT_ITEMS for u in found)
        return False

    def _carry_full(self, state: GameState, analysis: Analysis, carry: Optional[Unit]) -> bool:
        comp = analysis.comps[0] if analysis.comps else None
        if comp and comp.carry:
            # Not owned yet: its items are still worth collecting.
            return self._holder_full(state, comp.carry)
        return carry is not None and len([i for i in carry.items if i]) >= MAX_UNIT_ITEMS

    def _carousel_pick(
        self, state: GameState, analysis: Analysis, carry: Optional[Unit]
    ) -> Optional[CarouselPick]:
        """(component, completed item, holder) worth taking on the carousel.

        Components the item plan already combines (priority 1-2) are spoken
        for, and a holder with 3 items cannot take another one. A pick that
        completes an item beats one that is still a component short."""
        bench = Counter(_norm(x) for x in state.item_bench)
        reserved: set[int] = set()
        for i, sug in enumerate(analysis.items):
            need = Counter(_norm(c) for c in sug.components)
            if sug.priority <= 2 and need and all(bench.get(k, 0) >= n for k, n in need.items()):
                bench.subtract(need)
                reserved.add(i)

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

        candidates: list[tuple[list[str], str, Optional[str]]] = []
        order = sorted(range(len(analysis.items)), key=lambda i: analysis.items[i].priority)
        for i in order:
            sug = analysis.items[i]
            if i in reserved or not sug.components or self._holder_full(state, sug.holder):
                continue
            miss = missing(list(sug.components))
            if miss:
                candidates.append((miss, sug.item, sug.holder))
        comp = analysis.comps[0] if analysis.comps else None
        if comp and comp.carry_items and self.set_data is not None and not self._carry_full(state, analysis, carry):
            held = {_norm(i) for i in (carry.items if carry else [])}
            planned = {_norm(s.item) for s in analysis.items if s.components}
            for name in comp.carry_items:
                item = self.set_data.resolve_item(name)
                if item is None or not item.composition or _norm(item.name) in held or _norm(name) in held:
                    continue
                if _norm(item.name) in planned:
                    continue  # already built from the bench or named above
                parts = [self.set_data.items[c].name for c in item.composition if c in self.set_data.items]
                miss = missing(parts)
                if miss:
                    candidates.append((miss, item.name, comp.carry))
        if not candidates:
            return None
        miss, item_name, holder = next((c for c in candidates if len(c[0]) == 1), candidates[0])
        return CarouselPick(miss[0], item_name, holder, len(miss) == 1)

    def _carousel_text(
        self, pick: Optional[CarouselPick], carry_full: bool = False
    ) -> str:
        if not pick:
            if carry_full:
                return "主C装备已满，选秀拿坦克装备散件或缺的英雄"
            return "选秀优先拿主C需要的装备散件"
        component, item, holder, completes = pick
        verb = "做" if completes else "凑"  # 凑: one more component still needed
        if item and holder:
            return f"选秀拿 {component}（给 {holder} {verb} {item}）"
        if item:
            return f"选秀拿 {component}（{verb} {item}）"
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
        elif _is_reroll_plan(econ, sr):
            # Reroll lines stay low and slow roll; the standard level curve
            # (4-1 L7, 4-5 L8) would contradict the econ advice.
            stay = level
            if econ.recommendation in (EconAction.LEVEL, EconAction.LEVEL_AND_ROLL) and econ.target_level:
                parts.append(f"现在升 {econ.target_level}")
                stay = max(level, econ.target_level)
            cap = self._cap_gold()
            rr = reroll_level(econ.style)
            if rr and (not stay or stay < rr):
                # Stage 2 follows the standard curve; the reroll level comes in
                # stage 3 (3-5 for a level 7 reroll).
                start = REROLL_START_L7 if rr >= 7 else REROLL_START
                when = f"{start[0]}-{start[1]} " if sr.key < start else ""
                parts.append(f"{when}到 {rr} 级开始慢搜三星，只花 {cap} 以上的钱")
            elif econ.recommendation in (EconAction.ROLL, EconAction.LEVEL_AND_ROLL):
                # HP forces a real roll this round (econ: "搜到 20 左右追三星").
                parts.append(f"停在 {stay} 级搜牌追三星")
            else:
                parts.append(f"停在 {stay} 级慢搜三星，只花 {cap} 以上的钱")
            parts.append("三星成型或 4-5 后再升级补强")
        else:
            parts.extend(self._level_steps(sr, level, econ))
            gold = self._gold(state, analysis)
            if econ.recommendation == EconAction.SAVE and gold < self._cap_gold() and sr.stage <= 4:
                parts.insert(0, f"先存到 {self._cap_gold()}")
        if state.hp is not None and 0 < state.hp < 45 and econ.recommendation != EconAction.ALL_IN:
            parts.append("血量偏低，优先稳血别贪经济")
        comp = analysis.comps[0] if analysis.comps else None
        if comp:
            parts.append(f"目标 {comp.name}")
        return "，".join(parts)

    def _milestones(self, style: str) -> list[tuple[StageRound, int]]:
        """(round, level) milestones of the line being played, sorted (fast 8 /
        fast 9 level earlier than the standard curve, matching economy.py)."""
        table = dict(self.mech.standard_levels)
        if style in ("fast8", "fast9"):
            table.pop("4-5", None)
            table["4-2"] = 8
        if style == "fast9":
            table.pop("5-5", None)
            table["5-2"] = 9
        out = [(r, lvl) for key, lvl in table.items() for r in (StageRound.parse(key),) if r is not None]
        return sorted(out, key=lambda x: x[0].key)

    @staticmethod
    def _level_tag(lvl: int) -> str:
        if lvl == 8:
            return " 找 4 费主C"
        if lvl >= 9:
            return " 找 5 费"
        return ""

    def _level_steps(self, sr: StageRound, level: int, econ) -> list[str]:
        """Plan steps for a standard / fast 8 / fast 9 line: the level up the
        econ plans now, a catch-up when the line is behind (a fast 8 player
        still at 7 after 4-2 must hear "升 8"), then the next milestones."""
        steps: list[str] = []
        sched = self._milestones(econ.style)
        # Level the line should already have (milestones passed so far).
        due = max((lvl for r, lvl in sched if r.key <= sr.key), default=0)
        now = level or due  # unknown level: assume the line is on schedule
        if econ.recommendation in (EconAction.LEVEL, EconAction.LEVEL_AND_ROLL) and econ.target_level:
            text = f"现在升 {econ.target_level}"
            if econ.recommendation == EconAction.LEVEL_AND_ROLL:
                text += " 搜牌稳血"
            steps.append(text)
            now = max(now, econ.target_level)
        nine_named = False
        if now < due:
            # Behind: never jump from 7 or lower straight to 9 in one step.
            targets = [8, due] if due > 8 and now < 8 else [due]
            for i, lvl in enumerate(targets):
                if i:
                    verb = f"再升 {lvl}"
                else:
                    verb = f"尽快升 {lvl}" if lvl == now + 1 else f"尽快补到 {lvl} 级"
                steps.append(verb + (self._level_tag(lvl).lstrip() if verb.endswith("级") else self._level_tag(lvl)))
            nine_named = due >= 9
            now = due
        upcoming = [(r, lvl) for r, lvl in sched if r.key > sr.key and lvl > now]
        for r, lvl in upcoming[:2]:
            tag = self._level_tag(lvl) or (" 搜牌稳血" if r.key in ROLLDOWN_POINTS else "")
            steps.append(f"{r.stage}-{r.round} 升 {lvl}{tag}")
            nine_named = nine_named or lvl >= 9
        if not upcoming and not nine_named:
            if now >= 9:
                steps.append("慢搜找 5 费和三星，补满最强阵容")
            else:
                steps.append("利息满后升 9 找 5 费")
        return steps

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

    def _items_text(
        self,
        analysis: Analysis,
        comp: Optional[CompSuggestion],
        carry: Optional[Unit],
        state: Optional[GameState] = None,
    ) -> str:
        # No leading "装备：" label: the dashboard and overlay show their own.
        # A component still to find is no plan for a unit with 3 items.
        sugs = [
            s
            for s in analysis.items
            if not (state is not None and s.components and s.priority >= 3 and self._holder_full(state, s.holder))
        ]
        if sugs:
            parts = [
                f"{s.item} 给 {s.holder}" if s.holder else (s.item if s.components else f"{s.item} 先留着")
                for s in sugs[:3]
            ]
            return clip(_join(parts, "，"), FIELD_MAX)
        names = [n for n in (comp.carry_items if comp else []) if is_display_name(n)]
        if names:
            who = comp.carry or (carry.name if carry else "主C")
            return clip(f"{who}：{_join(names[:3])}", FIELD_MAX)
        return ""

    def _carry_is_melee(self, carry: Optional[Unit]) -> Optional[bool]:
        """True / False from the set data's attack range, None when unknown."""
        if carry is None or self.set_data is None:
            return None
        champ = self.set_data.champions.get(carry.api_name) or self.set_data.resolve_champion(carry.name)
        return None if champ is None else getattr(champ, "is_melee", None)

    def _positioning_text(self, carry: Optional[Unit], state: GameState) -> str:
        if not state.board:
            return ""
        if carry is not None:
            melee = self._carry_is_melee(carry)
            if melee is True:
                return f"{_unit_label(carry)} 是近战主C：放前排侧翼，旁边放坦克分担伤害"
            if melee is False:
                return f"{_unit_label(carry)} 放后排角落，坦克放前排挡伤害，注意防刺客"
            return f"{_unit_label(carry)} 放后排角落（近战主C除外），坦克放前排挡伤害，注意防刺客"
        return "主C放后排角落，坦克放前排挡伤害"

    def _augment_text(self, state: GameState) -> str:
        # Point to Claude only when a strategist will actually answer.
        more = AUGMENT_MORE if self.claude_enabled else ""
        choices = [c for c in state.augment_choices if c]
        if choices:
            return clip(f"可选：{_join(choices)}。{AUGMENT_HINT}{more}", FIELD_MAX)
        return f"{AUGMENT_GENERIC}{more}"
