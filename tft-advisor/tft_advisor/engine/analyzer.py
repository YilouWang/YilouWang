"""Analyzer: one deterministic pass over a GameState -> Analysis.

Combines comp fit, economy, shop odds, item plan, shop picks, sell candidates
and warnings. It must never raise: every section is guarded, a failing section
is skipped (and recorded in ``last_errors`` for debugging) so the pipeline
keeps producing advice from partial or empty states.
"""

from __future__ import annotations

import threading
import traceback
from collections import Counter
from typing import Callable, Optional, TypeVar

from ..data.comps import CompDef
from ..data.mechanics import Mechanics
from ..data.setdata import SetData
from ..models import (
    Analysis,
    CompSuggestion,
    EconAction,
    EconPlan,
    GameState,
    HitOdds,
    ItemSuggestion,
    ScreenType,
    StageRound,
    Unit,
)
from .comps_engine import (
    Owned,
    comp_hint_boost,
    owned_units,
    reroll_key_copies,
    star_of_copies,
    suggest_comps,
)
from .economy import ROLLDOWN_ROUNDS, STYLES, hp_bucket, no_roll, plan_economy, reroll_cost, reroll_level
from .items import plan_items, profile_hints_from_comps
from .probability import GOLD_BUDGETS, compute_hit_odds

MAX_ODDS = 8
STYLE_MIN_SCORE = 0.4  # commit to a comp's econ style only when it fits reasonably
LATE_STAGE = 4
RICH_GOLD = 70
ROLL_TARGET_FLOOR = 0.10  # a roll needs a target with at least this chance at its budget
FIND_ONE_FLOOR = 0.5  # ... or a missing unit it will probably find
WIN_STREAK_CHECK = 2  # win streaks from this length only roll for a real target
CRITICAL_TARGETS = 3  # odds targets compared when a critical-HP all-in could level first
HOPELESS_P = 0.05  # a target below this at both levels does not decide the level
LEVEL_SLOT_BONUS = 0.05  # odds a level may cost at critical HP (it also adds a unit)
COMP_POOL = 6  # comps scored per frame (the shown top 3 plus candidates for the sticky top)
STICKY_MARGIN = 0.15  # a new top comp replaces the current one at once only with this lead
FILL_TIER = 5  # shop pick only to fill an empty team slot
UNKNOWN_OWNER = "unknown"  # tracker.UNKNOWN_PLAYER: a scouted board with no readable owner

T = TypeVar("T")


def _clean_taken(
    taken_by_player: Optional[dict[str, dict[str, int]]], self_name: Optional[str]
) -> dict[str, dict[str, int]]:
    """Drop our own entry and non-numeric / negative counts."""
    out: dict[str, dict[str, int]] = {}
    for player, counts in (taken_by_player or {}).items():
        if not player or player == self_name or not isinstance(counts, dict):
            continue
        clean: dict[str, int] = {}
        for api, n in counts.items():
            if isinstance(n, bool) or not isinstance(n, (int, float)) or n <= 0:
                continue
            clean[str(api)] = int(n)
        if clean:
            out[str(player)] = clean
    return out


class Analyzer:
    def __init__(
        self,
        set_data: SetData,
        mech: Mechanics,
        comps: Optional[list[CompDef]] = None,
        comp_hint: str = "",
    ) -> None:
        self.set_data = set_data
        self.mech = mech
        self.comps: list[CompDef] = list(comps or [])
        self.comp_hint = comp_hint or ""
        self.last_errors: list[str] = []
        self._hint_match: tuple[str, bool] = ("", True)
        self._lib_profiles: Optional[dict[str, Optional[str]]] = None
        # Sticky top comp per game: (game id, hint) -> current top, challenger, last frame seen.
        self._sticky: Optional[dict] = None
        self._sticky_lock = threading.Lock()

    def set_comp_hint(self, hint: str) -> None:
        self.comp_hint = (hint or "").strip()

    # ------------------------------------------------------------------ main
    def analyze(self, state: GameState, taken_by_player: Optional[dict[str, dict[str, int]]] = None) -> Analysis:
        # Errors are collected per call: the ask thread may analyze while the
        # worker does, and one call must not clear or steal the other's.
        errors: list[str] = []

        def guard(name: str, fn: Callable[[], T], default: T) -> T:
            return self._guard(name, fn, default, errors)

        analysis = self._analyze(state, taken_by_player, guard)
        if errors:
            analysis.warnings.append("部分分析出错已跳过，建议可能不完整")
        self.last_errors = errors
        return analysis

    def _analyze(
        self,
        state: GameState,
        taken_by_player: Optional[dict[str, dict[str, int]]],
        guard: Callable,
    ) -> Analysis:
        taken_by_player = guard("taken", lambda: _clean_taken(taken_by_player, state.self_name), {})
        pool_notes: list[str] = []
        taken_by_player = guard("pool", lambda: self._fit_pool(state, taken_by_player, pool_notes), taken_by_player)
        taken_total: dict[str, int] = {}
        for counts in taken_by_player.values():
            for api, n in counts.items():
                taken_total[api] = taken_total.get(api, 0) + n

        analysis = Analysis(stage=str(state.stage) if state.stage else None)

        comps = guard("comps", lambda: suggest_comps(
            state, self.set_data, self.comps, taken_by_player, top_n=COMP_POOL, hint=self.comp_hint
        ), [])
        comps = guard("sticky", lambda: self._sticky_top(state, comps), comps)[:3]
        analysis.comps = comps
        top = comps[0] if comps else None
        style = guard("style", lambda: self._style_for(top, state), "standard")
        key_star = guard("key", lambda: self._key_star(top, state, style), None)
        if style.startswith("reroll") and key_star is not None and key_star >= 3:
            # The reroll target is 3-star: level and chase upgrades like a
            # standard line from here on.
            style = "standard"

        # Econ and odds need a level: an unread level would be treated as 1
        # ("buy XP to reach level 2" at stage 5), so estimate it instead.
        est = guard("level", lambda: self._with_level_estimate(state), state)
        carry_cost = guard("carry_cost", lambda: self._carry_cost(top), None)
        analysis.econ = guard(
            "econ",
            lambda: plan_economy(est, self.mech, style, key_star=key_star, carry_cost=carry_cost),
            EconPlan(gold=state.gold or 0),
        )
        guard("critical_level", lambda: self._critical_level(est, analysis, top, taken_total, style), None)
        # Shop picks come first (the cheapest copies there are): they are paid
        # from the roll budget, and the odds count the copies they buy.
        open_slots = guard("slots", lambda: self.open_slots(state, est), 0)
        picks = guard("shop", lambda: self._pick_shop(state, est, top, analysis.econ), [])
        guard("fund_picks", lambda: self._fund_picks(analysis.econ, picks, state.hp, state.stage), None)
        # Odds at the level the next shops will be rolled at (after leveling).
        odds_state = guard("odds_level", lambda: self._odds_state(est, analysis.econ, picks), est)
        analysis.odds = guard(
            "odds", lambda: self._odds(odds_state, top, taken_total, style, analysis.econ.roll_budget), []
        )
        before = (analysis.econ.recommendation, analysis.econ.target_level)
        guard("roll_check", lambda: self._drop_pointless_roll(est, analysis, style, open_slots), None)
        if (analysis.econ.recommendation, analysis.econ.target_level) != before:
            # No roll after all: the shop budget and the roll level changed.
            new_picks = guard("shop", lambda: self._pick_shop(state, est, top, analysis.econ), picks)
            if new_picks != picks or analysis.econ.target_level != before[1]:
                picks = new_picks
                odds_state = guard("odds_level", lambda: self._odds_state(est, analysis.econ, picks), est)
                analysis.odds = guard(
                    "odds",
                    lambda: self._odds(odds_state, top, taken_total, style, analysis.econ.roll_budget),
                    analysis.odds,
                )
        analysis.shop_picks = [p[1] for p in picks]
        analysis.shop_picks_cost = sum(p[2] for p in picks)
        analysis.items = guard(
            "items",
            lambda: plan_items(
                state,
                self.set_data,
                comp=top,
                hp_bucket=hp_bucket(state.hp),
                profile_hints=self._profile_hints(top),
            ),
            [],
        )
        # A unit the shop picks buy another copy of is kept, not sold.
        keep = {p[3].api_name for p in picks if p[3] is not None}
        analysis.sell_candidates = guard("sell", lambda: self._sell_candidates(state, top, keep), [])
        # The econ plan's style: a reroll that ended (4-5 or later) levels normally.
        econ_style = analysis.econ.style if analysis.econ.style in STYLES else style
        analysis.warnings = guard(
            "warnings", lambda: self._warnings(state, analysis.items, econ_style, analysis.econ, est), []
        )
        analysis.warnings.extend(pool_notes)
        return analysis

    # --------------------------------------------------------------- helpers
    def _guard(self, name: str, fn: Callable[[], T], default: T, errors: Optional[list[str]] = None) -> T:
        try:
            return fn()
        except Exception as exc:  # never let one section kill the advice
            (self.last_errors if errors is None else errors).append(
                f"{name}: {exc!r}\n{traceback.format_exc(limit=4)}"
            )
            return default

    def _sticky_top(self, state: GameState, comps: list[CompSuggestion]) -> list[CompSuggestion]:
        """Keep the current top comp through a one-frame dip (a unit the vision
        missed once): a new top replaces it only with a clear lead
        (STICKY_MARGIN), when the old one dropped out, or on the second frame
        in a row. Per game and comp hint; states without a game id (tests,
        one-off analyses) are not sticky."""
        if not comps or not state.game_id:
            return comps
        key = (state.game_id, self.comp_hint)
        frame = state.last_update
        with self._sticky_lock:
            prev = self._sticky
            if prev is None or prev["key"] != key:
                self._sticky = {"key": key, "top": comps[0].name, "challenger": None, "frame": frame}
                return comps
            cur = prev["top"]
            new = comps[0]
            if new.name == cur:
                if frame != prev["frame"]:
                    self._sticky = {"key": key, "top": cur, "challenger": None, "frame": frame}
                return comps
            old = next((c for c in comps if c.name == cur), None)
            repeat = prev["challenger"] == new.name and frame != prev["frame"]
            if old is None or new.score - old.score >= STICKY_MARGIN or repeat:
                self._sticky = {"key": key, "top": new.name, "challenger": None, "frame": frame}
                return comps
            if frame != prev["frame"] or prev["challenger"] is None:
                self._sticky = {"key": key, "top": cur, "challenger": new.name, "frame": frame}
            return [old, *[c for c in comps if c is not old]]

    def _fit_pool(
        self, state: GameState, taken: dict[str, dict[str, int]], notes: list[str]
    ) -> dict[str, dict[str, int]]:
        """Our copies plus the scouted ones can never exceed a champion's pool:
        when they do, a scouted star was misread (a 2-star read as 3-star).
        The largest holder's count is taken one star lower (a third), or down
        to what the pool leaves when that is lower still, and the player is
        asked to scout that board again."""
        ours = {api: o.copies for api, o in owned_units(state, self.set_data).items()}
        out = {p: dict(c) for p, c in taken.items()}
        apis = {api for counts in out.values() for api in counts}
        for api in sorted(apis):
            champ = self.set_data.champions.get(api)
            pool = self.mech.pool_size.get(champ.cost) if champ is not None else None
            if not pool:
                continue
            excess = ours.get(api, 0) + sum(c.get(api, 0) for c in out.values()) - pool
            for player in sorted(out, key=lambda p: (-out[p].get(api, 0), p)):
                if excess <= 0:
                    break
                have = out[player].get(api, 0)
                if have <= 0:
                    continue
                left = max(0, have - excess)
                if have >= 3:
                    left = min(left, have // 3)  # one star lower
                out[player][api] = left
                excess -= have - left
                who = "未知对手" if player == UNKNOWN_OWNER else player
                notes.append(f"{who}的{champ.name}张数超过卡池，星级可能读错了，请再看一次{who}的棋盘")
        return {p: {a: n for a, n in c.items() if n > 0} for p, c in out.items() if any(n > 0 for n in c.values())}

    def estimate_level(self, state: GameState) -> Optional[int]:
        """Best guess of an unread level: the fielded unit count is a lower
        bound, the standard curve for the round a typical value."""
        if state.level is not None:
            return state.level
        guess = max(self.team_slots_used(state), self.mech.standard_level_at(state.stage) or 0)
        return min(self.mech.max_level, guess) if guess > 0 else None

    def slot_weight(self, u: Unit) -> int:
        """Team slots a fielded unit takes (Set 18 Elder Dragon takes 2)."""
        table = self.mech.extra.get("team_slots") if isinstance(self.mech.extra, dict) else None
        if not isinstance(table, dict) or not table:
            return 1
        champ = self.set_data.champions.get(u.api_name)
        for key in (u.api_name, champ.name_en if champ else None, champ.name if champ else None, u.name):
            if key and key in table:
                try:
                    return max(1, int(table[key]))
                except (TypeError, ValueError):
                    return 1
        return 1

    def team_slots_used(self, state: GameState) -> int:
        return sum(self.slot_weight(u) for u in state.board)

    def _with_level_estimate(self, state: GameState) -> GameState:
        if state.level is not None:
            return state
        lvl = self.estimate_level(state)
        return state.model_copy(update={"level": lvl}) if lvl is not None else state

    def _comp_def(self, name: Optional[str]) -> Optional[CompDef]:
        if not name:
            return None
        return next((c for c in self.comps if c.name == name), None)

    def _style_for(self, top: Optional[CompSuggestion], state: Optional[GameState] = None) -> str:
        comp = self._comp_def(top.name if top else None)
        if comp is None or top is None or comp.style not in STYLES:
            return "standard"
        hinted = False
        if self.comp_hint:
            hinted = comp_hint_boost(comp, self.comp_hint, self.set_data) > 0
        if top.score < STYLE_MIN_SCORE and not hinted:
            return "standard"
        # Staying low to slow roll is a commitment: players make it when they
        # already hold a pair (or a 2-star) of a unit they reroll for, not
        # because a few generic early units overlap with a reroll comp. The
        # comp's carry is only the item holder and often costs more than the
        # reroll (a 4-cost carry on a 3-cost reroll), so it does not count.
        if comp.style.startswith("reroll") and not hinted and state is not None:
            copies = reroll_key_copies(comp, owned_units(state, self.set_data), self.set_data)
            if copies is None or copies < 2:
                return "standard"
        return comp.style

    def _key_star(self, top: Optional[CompSuggestion], state: GameState, style: str) -> Optional[int]:
        """Star level of the unit the line is built around: the reroll target on
        reroll lines, the carry otherwise (0 = not owned, None = no target)."""
        if top is None:
            return None
        owned = owned_units(state, self.set_data)
        comp = self._comp_def(top.name)
        if comp is not None and style.startswith("reroll"):
            copies = reroll_key_copies(comp, owned, self.set_data)
            if copies is not None:
                return star_of_copies(copies)
        carry = self._resolve_api(top.carry)
        if carry is None:
            return None
        o = owned.get(carry)
        return star_of_copies(o.copies) if o else 0

    def _profile_hints(self, top: Optional[CompSuggestion]) -> dict[str, str]:
        """Champion damage profiles from the comp library's item plans (AD / AP /
        tank by the items strong players give each unit); the top comp wins."""
        if self._lib_profiles is None:
            self._lib_profiles = {
                k: v for k, v in profile_hints_from_comps(self.comps, self.set_data).items() if v
            }
        hints: dict[str, Optional[str]] = dict(self._lib_profiles)
        comp = self._comp_def(top.name if top else None)
        if comp is not None:
            # The target comp's own plan wins, a mixed plan included (None).
            hints.update(profile_hints_from_comps([comp], self.set_data, keep_ties=True))
        return {k: v for k, v in hints.items() if v}

    def _hint_matches_any(self) -> bool:
        """Does the comp hint boost at least one loaded comp? (cached per hint)"""
        hint = self.comp_hint
        if self._hint_match[0] != hint:
            ok = any(comp_hint_boost(c, hint, self.set_data) > 0 for c in self.comps)
            self._hint_match = (hint, ok)
        return self._hint_match[1]

    def _resolve_api(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        champ = self.set_data.resolve_champion(name)
        return champ.api_name if champ else None

    def _comp_apis(self, top: Optional[CompSuggestion]) -> set[str]:
        if top is None:
            return set()
        apis = {self._resolve_api(n) for n in [*top.core_units, *top.have_units, *top.missing_units]}
        carry = self._resolve_api(top.carry)
        if carry:
            apis.add(carry)
        return {a for a in apis if a}

    def _carry_cost(self, top: Optional[CompSuggestion]) -> Optional[int]:
        api = self._resolve_api(top.carry) if top else None
        champ = self.set_data.champions.get(api) if api else None
        return champ.cost if champ else None

    # ------------------------------------------------------------------ odds
    def _odds_state(self, state: GameState, econ: EconPlan, picks: Optional[list[tuple]] = None) -> GameState:
        """The state the next shops roll from: after a LEVEL or LEVEL_AND_ROLL
        plan the level is the target level, and the shop picks are already
        bought (their copies count as owned, so a goal the shop completes is
        not a roll target any more)."""
        update: dict = {}
        target = econ.target_level
        if (
            econ.recommendation in (EconAction.LEVEL, EconAction.LEVEL_AND_ROLL)
            and target is not None
            and state.level is not None
            and target > state.level
        ):
            update.update(level=target, xp_current=0)
        bought = [p[3] for p in picks or [] if p[3] is not None]
        if bought:
            update["bench"] = [*state.bench, *bought]
        return state.model_copy(update=update) if update else state

    def _critical_level(
        self,
        state: GameState,
        analysis: Analysis,
        top: Optional[CompSuggestion],
        taken_total: dict[str, int],
        style: str,
    ) -> None:
        """Critical HP all-in: buy the next level first when the roll-down odds
        of the main target are about as good there (the level also adds a unit)."""
        econ = analysis.econ
        level = state.level
        if econ.recommendation != EconAction.ALL_IN or econ.level_estimated or level is None:
            return
        if level >= self.mech.max_level:
            return
        rr = reroll_level(econ.style)
        if rr is not None and level >= rr:
            return  # a reroll line does not level past its reroll level
        gold = econ.gold
        spend = self.mech.gold_to_reach(level, state.xp_current, level + 1)
        if spend is None or spend <= 0 or spend > gold:
            return
        targets = self._odds_targets(state, top, style)[:CRITICAL_TARGETS]
        if not targets:
            return
        left = gold - spend
        now = compute_hit_odds(state, self.set_data, self.mech, taken_total, targets=targets, budgets=(gold,))
        up = compute_hit_odds(
            state, self.set_data, self.mech, taken_total, targets=targets, budgets=(left,), level=level + 1
        )
        pairs = []
        for a, b in zip(now, up):
            if a.goal_copies - a.owned_copies <= 0 or a.remaining_in_pool < a.goal_copies - a.owned_copies:
                continue
            pairs.append((a.p_goal_by_gold.get(gold, 0.0), b.p_goal_by_gold.get(left, 0.0)))
        if not pairs:
            return
        # The main target decides; one nobody can hit at either level does not.
        p_now, p_up = next(((a, b) for a, b in pairs if max(a, b) >= HOPELESS_P), pairs[0])
        if p_up + LEVEL_SLOT_BONUS < p_now:
            return
        roll = max(1, self.mech.roll_cost)
        econ.target_level = level + 1
        if left >= roll:
            econ.recommendation = EconAction.LEVEL_AND_ROLL
            econ.roll_budget = left
            econ.reason = f"血量 {state.hp}，危险：先升 {level + 1} 级多上一个人，再 all-in 搜牌"
        else:
            econ.recommendation = EconAction.LEVEL
            econ.roll_budget = 0
            econ.reason = f"血量 {state.hp}，危险：升到 {level + 1} 级多上一个人"

    def _has_roll_target(self, analysis: Analysis, open_slots: int = 0) -> bool:
        """Does the roll budget have something realistic to find? A unit whose
        next star is reachable with at least ROLL_TARGET_FLOOR (the same bar
        the rules use to name a roll target), a missing comp carry the rolls
        will probably show, or, with a team slot empty, any missing comp unit
        the rolls will probably show (one copy fills the slot)."""
        budget = analysis.econ.roll_budget
        roll = max(1, self.mech.roll_cost)
        rolls = budget // roll
        top = analysis.comps[0] if analysis.comps else None
        carry = self._resolve_api(top.carry) if top else None
        missing = {self._resolve_api(n) for n in top.missing_units} if top is not None and open_slots > 0 else set()
        none_found = 1.0  # chance the rolls show none of the missing comp units
        for o in analysis.odds:
            need = o.goal_copies - o.owned_copies
            if need <= 0 or o.remaining_in_pool < need:
                continue
            if o.p_at(budget) >= ROLL_TARGET_FLOOR:
                return True
            if o.owned_copies == 0 and o.api_name == carry:
                if 1.0 - (1.0 - o.p_in_shop) ** rolls >= FIND_ONE_FLOOR:
                    return True  # the carry is missing: finding one copy is the point
            if o.owned_copies == 0 and o.api_name in missing:
                none_found *= (1.0 - o.p_in_shop) ** rolls
        return bool(missing) and 1.0 - none_found >= FIND_ONE_FLOOR

    def _drop_pointless_roll(self, state: GameState, analysis: Analysis, style: str, open_slots: int = 0) -> None:
        """Turn a roll with no realistic target into save / level.

        Only where rolling is optional: a win streak (the board already wins),
        a fast 8 / fast 9 line at level 8+, a roll-down round without low HP,
        and the stage 4+ "chase the 3-star" roll of a reroll line. Low HP,
        reroll slow rolls and all-ins keep their roll. Gold that stays above
        the interest cap after the level goes into the level instead.
        """
        econ = analysis.econ
        if econ.recommendation not in (EconAction.ROLL, EconAction.LEVEL_AND_ROLL):
            return
        if hp_bucket(state.hp) in ("low", "critical"):
            return
        sr = state.stage
        stage = sr.stage if sr else 0
        line = econ.style if econ.style in STYLES else style
        reroll = line.startswith("reroll")
        reroll_chase = reroll and stage >= LATE_STAGE and econ.recommendation == EconAction.ROLL
        if reroll and not reroll_chase:
            return
        streak = state.streak or 0
        level = econ.target_level if econ.recommendation == EconAction.LEVEL_AND_ROLL else state.level
        fast = line in ("fast8", "fast9")
        late_fast = fast and (level or 0) >= 8
        rolldown = sr is not None and sr.key in ROLLDOWN_ROUNDS
        if streak < WIN_STREAK_CHECK and not late_fast and not rolldown and not reroll_chase:
            return
        if self._has_roll_target(analysis, open_slots):
            return
        econ.roll_budget = 0
        if econ.recommendation == EconAction.LEVEL_AND_ROLL and econ.target_level:
            econ.recommendation = EconAction.LEVEL
            econ.reason = f"升到 {econ.target_level} 级；这回合搜到关键牌的概率很低，先不搜"
            return
        cap = self.mech.interest_step * self.mech.interest_cap
        to_next = econ.gold_to_next_level
        cur = state.level
        if (
            not reroll
            and cur is not None
            and cur < self.mech.max_level
            and to_next is not None
            and econ.gold - to_next >= cap
        ):
            econ.recommendation = EconAction.LEVEL
            econ.target_level = cur + 1
            econ.reason = f"搜到关键牌的概率很低：升到 {cur + 1} 级也不掉利息，先升级"
            return
        econ.recommendation = EconAction.SAVE
        econ.target_level = None
        if streak >= WIN_STREAK_CHECK:
            econ.reason = f"连胜 {streak} 场，这回合搜到关键牌的概率很低：不搜，存钱"
        elif reroll_chase:
            econ.reason = f"搜到三星的概率很低：先存钱，攒到 {cap} 以上再慢搜"
        elif fast and (cur or 0) < 8:
            econ.reason = "搜到升星的概率很低：先不搜，存钱升 8 级"
        else:
            econ.reason = "搜到升星的概率很低：先不搜，存钱准备升级"

    def _chase_three_star(self, o: Owned, level: int, style: str) -> bool:
        """Is rolling for a 3-star of this (already 2-star) unit a real plan?

        Strong players chase 3-stars on reroll lines (units up to the reroll
        cost), for cheap units while still low level, or when one or two
        copies are missing. A 2-star 1-2 cost at level 8 on a fast 8 line is
        not a roll target, the 4-5 cost upgrades are.
        """
        cap = reroll_cost(style)
        if cap is not None and o.cost <= cap:
            return True
        if o.copies >= 7:
            return True
        if o.cost <= 2 and level <= 6:
            return True
        return o.cost <= 3 and level <= 7 and o.copies >= 5

    def _odds(
        self,
        state: GameState,
        top: Optional[CompSuggestion],
        taken_total: dict[str, int],
        style: str = "standard",
        roll_budget: int = 0,
    ) -> list[HitOdds]:
        targets = self._odds_targets(state, top, style)
        if not targets:
            return []
        # The plan's own budget is computed exactly (HitOdds.p_at), not
        # rounded down to the nearest fixed bucket.
        budgets = tuple(sorted(set(GOLD_BUDGETS) | ({int(roll_budget)} if roll_budget > 0 else set())))
        odds = compute_hit_odds(state, self.set_data, self.mech, taken_total, targets=targets, budgets=budgets)
        return odds[:MAX_ODDS]

    def _odds_targets(self, state: GameState, top: Optional[CompSuggestion], style: str = "standard") -> list[str]:
        """Units worth odds, most important first (the carry, then one copy
        from the next star, comp units, missing comp units, the rest)."""
        owned = owned_units(state, self.set_data)
        comp_apis = self._comp_apis(top)
        carry = self._resolve_api(top.carry) if top else None
        level = state.level or 1

        ranked: list[tuple[tuple, str]] = []

        def add(api: Optional[str], key: tuple) -> None:
            if api and api in self.set_data.champions and all(a != api for _k, a in ranked):
                ranked.append((key, api))

        def need(o: Owned) -> int:
            return (3 if o.copies < 3 else 9) - o.copies

        def relevant(o: Owned) -> bool:
            return o.copies < 3 or self._chase_three_star(o, level, style)

        committed = top is not None and top.score >= STYLE_MIN_SCORE
        if carry and carry in self.set_data.champions:
            oc = owned.get(carry)
            reachable = self.mech.odds(level, self.set_data.champions[carry].cost) > 0
            # A 2-star carry is a roll target again only for a 3-star plan: cheap
            # carries, 4 costs at level 9+, 5 costs at level 10.
            late_three = oc is not None and ((oc.cost <= 4 and level >= 9) or level >= 10)
            carry_goal = oc is not None and oc.copies < 9 and (relevant(oc) or late_three)
            if carry_goal or (oc is None and committed and reachable):
                add(carry, (0, 0, 0, ""))
        # Tiers: one copy from the next star (comp units first), comp units,
        # missing comp units the shop can show, other owned units, and last
        # 3-stars nobody would roll for.
        for api, o in sorted(owned.items(), key=lambda kv: (need(kv[1]), -kv[1].cost, kv[0])):
            if o.copies >= 9:
                continue
            in_comp = api in comp_apis
            if not relevant(o):
                add(api, (6, need(o), -o.cost, api))
            elif need(o) == 1:
                add(api, (1 if in_comp else 2, 1, -o.cost, api))
            elif in_comp:
                add(api, (3, need(o), -o.cost, api))
        if top is not None and committed:
            for name in top.missing_units:
                api = self._resolve_api(name)
                champ = self.set_data.champions.get(api) if api else None
                # Missing units the shop cannot show at this level are noise.
                if champ is not None and self.mech.odds(level, champ.cost) > 0:
                    add(api, (4, 3, -champ.cost, api))
        for api, o in owned.items():
            if o.copies < 9:
                add(api, (5, need(o), -o.cost, api))
        ranked.sort(key=lambda t: t[0])
        return [api for _k, api in ranked[:MAX_ODDS]]

    # ------------------------------------------------------------------ shop
    def _shop_units(self, state: GameState) -> list[Optional[Unit]]:
        if state.shop_units and len(state.shop_units) >= len(state.shop):
            return list(state.shop_units)
        out: list[Optional[Unit]] = []
        for slot in state.shop:
            champ = self.set_data.resolve_champion(slot.name) if slot.name else None
            if champ is None:
                out.append(None)
            else:
                out.append(Unit(api_name=champ.api_name, name=champ.name, cost=champ.cost, traits=list(champ.traits)))
        return out

    def _active_traits(self, state: GameState) -> set[str]:
        active = {t.name for t in state.traits if t.active}
        counts: dict[str, set[str]] = {}
        for u in state.board:
            if u.api_name.startswith("?"):
                continue
            for t in u.traits:
                counts.setdefault(t, set()).add(u.api_name)
        for name, apis in counts.items():
            trait = self.set_data.resolve_trait(name)
            if trait and trait.breakpoints and len(apis) >= min(trait.breakpoints):
                active.add(name)
        return active

    def _pick_budget(self, state: GameState, econ: EconPlan) -> tuple[Optional[int], Optional[int]]:
        """(hard, soft) gold limits for shop picks under the econ plan.

        hard: gold minus the level the plan buys. soft (everything but an
        upgrade now or an owned comp unit): a rolling plan pays the picks from
        its roll budget; a save or level plan keeps its interest step.
        """
        gold = state.gold
        if gold is None:
            return None, None
        rec = econ.recommendation
        hard = max(0, gold - self._level_spend(state, econ))
        if rec in (EconAction.ROLL, EconAction.SLOW_ROLL, EconAction.LEVEL_AND_ROLL, EconAction.ALL_IN):
            return hard, min(hard, econ.roll_budget)
        if rec in (EconAction.SAVE, EconAction.LEVEL):
            cap = self.mech.interest_step * self.mech.interest_cap
            floor = min(cap, (hard // max(1, self.mech.interest_step)) * self.mech.interest_step)
            return hard, hard - floor
        return hard, hard

    @staticmethod
    def shop_is_stale(state: GameState) -> bool:
        """The known shop is not this round's: a carousel frame (no shop on
        screen), or a shop read before the current round started (the shop
        refreshes every round)."""
        if state.screen_type == ScreenType.CAROUSEL:
            return True
        shop_t = state.field_age.get("shop")
        round_t = state.field_age.get("round")
        return shop_t is not None and round_t is not None and shop_t < round_t

    def _adds_breakpoint(self, state: GameState, u: Unit) -> bool:
        """Does fielding ``u`` reach a new breakpoint of one of its traits?"""
        for t in u.traits:
            trait = self.set_data.resolve_trait(t)
            if trait is None or not trait.breakpoints:
                continue
            have = {x.api_name for x in state.board if t in x.traits and x.api_name != u.api_name}
            if len(have) + 1 in trait.breakpoints:
                return True
        return False

    def _pick_shop(
        self, state: GameState, est: GameState, top: Optional[CompSuggestion], econ: EconPlan
    ) -> list[tuple[tuple, str, int, Optional[Unit]]]:
        """Shop units worth buying, in priority order: (key, name, cost, unit)."""
        if self.shop_is_stale(state):
            return []
        shop = self._shop_units(state)
        if not shop:
            return []
        owned = owned_units(state, self.set_data)
        comp_apis = self._comp_apis(top)
        active = self._active_traits(state)
        board_apis = {u.api_name for u in state.board}
        in_shop = Counter(u.api_name for u in shop if u is not None)
        early = state.stage is None or state.stage.stage <= 3
        level = est.level or 1
        style = econ.style if econ.style in STYLES else "standard"
        committed = top is not None and top.score >= STYLE_MIN_SCORE
        # From stage 4 an off-comp unit is only worth a free team slot when it
        # reaches a new breakpoint (never with a committed comp).
        free_slots = max(0, (est.level or 0) - self.team_slots_used(state))
        # Team slots neither the board nor the bench can fill: playing a unit
        # short is the worst tempo loss, so any unit beats an empty slot.
        open_slots = self.open_slots(state, est)
        board_traits = {t for u in state.board for t in u.traits}
        comp_traits = self._comp_traits(top)
        upgrade_left: dict[str, int] = {}  # copies still needed for the next star

        picks: list[tuple[tuple, str, int, Optional[Unit]]] = []
        for idx, u in enumerate(shop):
            if u is None or u.api_name.startswith("?"):
                continue
            o = owned.get(u.api_name)
            copies = o.copies if o else 0
            if copies >= 9:
                continue
            cost = u.cost or 1
            ones = copies % 3
            twos = (copies // 3) % 3
            if u.api_name not in upgrade_left:
                upgrade_left[u.api_name] = 3 - ones if ones + in_shop[u.api_name] >= 3 else 0
            upgrade = upgrade_left[u.api_name] > 0
            three_star = upgrade and twos == 2
            fit = 0
            if upgrade:
                upgrade_left[u.api_name] -= 1
                tier = 0
            elif copies and u.api_name in comp_apis and (copies < 3 or self._chase_three_star(o, level, style)):
                tier = 1
            elif u.api_name in comp_apis:
                tier = 2  # a first copy, or a 3-star nobody would roll for
            elif copies and early:
                tier = 3
            elif (
                u.api_name not in board_apis
                and any(t in active for t in u.traits)
                and (early or (not committed and free_slots > 0 and self._adds_breakpoint(state, u)))
            ):
                if not early:
                    free_slots -= 1
                tier = 4
            elif not copies and open_slots > 0:
                # Only to fill an empty slot: the best fit with the board and
                # the target comp's traits, then the highest cost.
                tier = FILL_TIER
                fit = len(set(u.traits) & board_traits) + len(set(u.traits) & comp_traits)
            else:
                continue
            bought = Unit(api_name=u.api_name, name=u.name, cost=u.cost, traits=list(u.traits))
            picks.append(((tier, 0 if three_star else 1, -fit, -cost, idx), u.name, cost, bought))
        picks.sort(key=lambda p: p[0])

        hard, soft = self._pick_budget(est, econ)
        # Bench space: an upgrade combines on purchase; a comp unit is worth
        # selling junk for; anything else needs a free bench slot.
        bench_free = self.mech.bench_size - len(state.bench)
        out: list[tuple[tuple, str, int, Optional[Unit]]] = []
        spent = 0
        filled = 0
        new_units: set[str] = set()
        for pick in picks:
            cost = pick[2]
            tier = pick[0][0]
            api = pick[3].api_name if pick[3] is not None else ""
            fills = api not in owned and api not in new_units and filled < open_slots
            if tier == FILL_TIER and not fills:
                continue
            if hard is not None and spent + cost > hard:
                continue
            # An upgrade now, a copy of an owned comp unit or a unit for an
            # empty slot may dip into the reserve; anything else must fit the
            # plan's spare gold.
            exempt = tier <= 1 or fills
            if not exempt and soft is not None and spent + cost > soft:
                continue
            if tier >= 3 and bench_free <= 0:
                continue
            if tier > 0:
                bench_free -= 1
            if fills:
                filled += 1
                new_units.add(api)
            spent += cost
            out.append(pick)
        return out

    def open_slots(self, state: GameState, est: Optional[GameState] = None) -> int:
        """Team slots neither the board nor the bench can fill (from stage 2,
        a known board): units to buy just to play a full team."""
        level = (est or state).level
        if not level or state.stage is None or state.stage.stage < 2 or "board" not in state.field_age:
            return 0
        if state.screen_type not in (ScreenType.PLANNING, ScreenType.OTHER, ScreenType.COMBAT):
            return 0
        return max(0, level - self.team_slots_used(state) - len(state.bench))

    def _comp_traits(self, top: Optional[CompSuggestion]) -> set[str]:
        out: set[str] = set()
        for api in self._comp_apis(top):
            champ = self.set_data.champions.get(api)
            if champ is not None:
                out.update(champ.traits)
        return out

    def _fund_picks(
        self, econ: EconPlan, picks: list[tuple], hp: Optional[int] = None, stage: Optional[StageRound] = None
    ) -> None:
        """Shop picks are paid from the roll budget of a rolling plan. A roll
        the picks leave no budget for becomes what the econ engine makes of
        an unpaid roll (``economy.no_roll``): at low HP late never "save",
        buy the shop and reposition instead."""
        cost = sum(p[2] for p in picks)
        if not cost or econ.recommendation not in (
            EconAction.ROLL,
            EconAction.SLOW_ROLL,
            EconAction.LEVEL_AND_ROLL,
            EconAction.ALL_IN,
        ):
            return
        econ.roll_budget = max(0, econ.roll_budget - cost)
        if econ.roll_budget >= max(1, self.mech.roll_cost) or econ.recommendation == EconAction.ALL_IN:
            return
        econ.roll_budget = 0
        if econ.recommendation == EconAction.LEVEL_AND_ROLL and econ.target_level:
            econ.recommendation = EconAction.LEVEL
            econ.reason = f"升到 {econ.target_level} 级，买完商店里的牌，剩下的钱不够再搜"
            return
        action, _reason = no_roll(econ.recommendation, hp, stage, max(0, econ.gold - cost), self.mech)
        econ.target_level = None
        if action == EconAction.HOLD:
            econ.recommendation = EconAction.HOLD
            econ.reason = "先买商店里的牌，调整站位卖闲置，下回合再搜"
        else:
            econ.recommendation = EconAction.SAVE
            econ.reason = "先买商店里的牌，剩下的金币不够再搜"

    # ------------------------------------------------------------------ sell
    def _sell_candidates(
        self, state: GameState, top: Optional[CompSuggestion], keep: Optional[set[str]] = None
    ) -> list[str]:
        """Bench units to sell; ``keep``: api names the shop picks buy more of."""
        if not state.bench:
            return []
        stage = state.stage.stage if state.stage else 0
        near_full = len(state.bench) >= self.mech.bench_size - 1
        if not (near_full or stage >= LATE_STAGE):
            return []
        comp_apis = self._comp_apis(top)
        ones = Counter(u.api_name for u in state.all_units() if u.star <= 1)
        cands: list[tuple[int, str]] = []
        for u in state.bench:
            if u.api_name.startswith("?") or u.api_name in comp_apis or u.api_name in (keep or ()):
                continue
            if ones[u.api_name] >= 2:
                continue  # pair: one more copy upgrades it
            value = (u.cost or 1) * u.copies
            cands.append((value, u.name))
        cands.sort(key=lambda c: (c[0], c[1]))
        return [name for _v, name in cands]

    # -------------------------------------------------------------- warnings
    def _level_spend(self, state: GameState, econ: EconPlan) -> int:
        """Gold the econ plan's level costs. ``state`` must carry the level the
        plan used (the estimate when the level was not read)."""
        if econ.recommendation not in (EconAction.LEVEL, EconAction.LEVEL_AND_ROLL) or not econ.target_level:
            return 0
        if state.level is None:
            return econ.gold_to_next_level or 0
        return self.mech.gold_to_reach(state.level, state.xp_current, econ.target_level) or 0

    def _gold_after(self, state: GameState, econ: Optional[EconPlan]) -> int:
        """Gold left once the econ plan's level and rolls are paid."""
        gold = state.gold or 0
        if econ is None:
            return gold
        return gold - self._level_spend(self._with_level_estimate(state), econ) - econ.roll_budget

    def _warnings(
        self,
        state: GameState,
        items: list[ItemSuggestion],
        style: str,
        econ: Optional[EconPlan] = None,
        est: Optional[GameState] = None,
    ) -> list[str]:
        out: list[str] = []
        mech = self.mech
        sr = state.stage
        stage = sr.stage if sr else 0
        planning = state.screen_type in (ScreenType.PLANNING, ScreenType.OTHER, ScreenType.COMBAT)

        if len(state.bench) >= mech.bench_size:
            out.append("备战席已满，先卖掉用不上的棋子，不然买不了新英雄")

        used = self.team_slots_used(state)  # Elder Dragon takes 2 slots
        if (
            state.level
            and "board" in state.field_age
            and planning
            and stage >= 2
            and used < state.level
        ):
            free = state.level - used
            if state.bench:
                out.append(f"场上只有 {len(state.board)} 个英雄，还能再上 {free} 个，把备战席的棋子放上去")
            else:
                out.append(f"场上只有 {len(state.board)} 个英雄，还有 {free} 个空位，买英雄补满")

        std = mech.standard_level_at(sr) if sr else None
        rr = reroll_level(style)
        if std and state.level and stage >= 2 and state.level < std:
            if rr is None or state.level < min(rr, std):
                out.append(f"等级落后标准节奏：{sr} 一般已经 {std} 级，你现在 {state.level} 级")

        if stage >= 3:
            comps_on_bench = []
            for name in state.item_bench:
                it = self.set_data.resolve_item(name)
                if it is not None and it.kind == "component":
                    comps_on_bench.append(it.name)
            crafted = [s for s in items if s.components and s.priority <= 2]
            if comps_on_bench and not crafted:
                out.append(f"备战席有散件（{', '.join(comps_on_bench)}）暂时合不成装备，选秀和野怪优先拿能配对的散件")
            # Only items someone can take: the rest are held on purpose ("先留着").
            loose = [s.item for s in items if not s.components and s.holder]
            if loose:
                out.append(f"成装还没装备：{', '.join(loose)}，记得给英雄装上")

        # PvE round (x-7): the plan waits for the next player fight to roll.
        pve = sr is not None and stage >= 2 and mech.is_pve(sr)
        rich = state.gold is not None and state.gold >= RICH_GOLD and stage >= LATE_STAGE
        # The plan itself leaves the gold unspent (the level it pays for counted
        # at the estimated level when the level was not read). Not on a PvE
        # round: the plan spends it at (x+1)-1 on purpose.
        if rich and not pve and self._gold_after(est or state, econ) >= RICH_GOLD:
            cap = mech.interest_step * mech.interest_cap
            out.append(f"金币 {state.gold} 太多了，利息最多只算到 {cap}，多出来的钱拿去升级或搜牌")

        if state.level is None and stage >= 2:
            est = self.estimate_level(state)
            if est is not None:
                out.append(f"没识别到等级，暂按 {est} 级估算，可以在面板里手动修正")

        bucket = hp_bucket(state.hp)
        when = f"下回合 {stage + 1}-1 " if pve else "这回合"
        if bucket == "critical":
            out.append(f"血量只剩 {state.hp}，非常危险，{when}必须全力补强阵容")
        elif bucket == "low":
            if pve:
                out.append(f"血量 {state.hp} 偏低，{when}对战前搜牌稳血")
            else:
                out.append(f"血量 {state.hp} 偏低，别再贪经济，优先稳血")

        if self.comp_hint and not self._hint_matches_any():
            out.append(f"没有阵容匹配「{self.comp_hint}」，按自动推荐")

        unresolved = [u.name for u in state.all_units() if u.api_name.startswith("?")]
        unresolved += [u.name for u in state.shop_units if u is not None and u.api_name.startswith("?")]
        if unresolved:
            uniq = list(dict.fromkeys(unresolved))
            names = ", ".join(uniq[:3]) + (" 等" if len(uniq) > 3 else "")
            out.append(f"有英雄名字没识别准（{names}），建议可能不准，可以重新分析一次")
        return out
