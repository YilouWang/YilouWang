"""Analyzer: one deterministic pass over a GameState -> Analysis.

Combines comp fit, economy, shop odds, item plan, shop picks, sell candidates
and warnings. It must never raise: every section is guarded, a failing section
is skipped (and recorded in ``last_errors`` for debugging) so the pipeline
keeps producing advice from partial or empty states.
"""

from __future__ import annotations

import traceback
from collections import Counter
from typing import Callable, Optional, TypeVar

from ..data.comps import CompDef
from ..data.mechanics import Mechanics
from ..data.setdata import SetData
from ..models import Analysis, CompSuggestion, EconPlan, GameState, HitOdds, ItemSuggestion, ScreenType, Unit
from .comps_engine import Owned, hint_boost, owned_units, suggest_comps
from .economy import STYLES, hp_bucket, plan_economy, reroll_level
from .items import plan_items
from .probability import compute_hit_odds

MAX_ODDS = 8
STYLE_MIN_SCORE = 0.4  # commit to a comp's econ style only when it fits reasonably
LATE_STAGE = 4
RICH_GOLD = 70

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

    def set_comp_hint(self, hint: str) -> None:
        self.comp_hint = (hint or "").strip()

    # ------------------------------------------------------------------ main
    def analyze(self, state: GameState, taken_by_player: Optional[dict[str, dict[str, int]]] = None) -> Analysis:
        self.last_errors = []
        taken_by_player = self._guard("taken", lambda: _clean_taken(taken_by_player, state.self_name), {})
        taken_total: dict[str, int] = {}
        for counts in taken_by_player.values():
            for api, n in counts.items():
                taken_total[api] = taken_total.get(api, 0) + n

        analysis = Analysis(stage=str(state.stage) if state.stage else None)

        comps = self._guard("comps", lambda: suggest_comps(
            state, self.set_data, self.comps, taken_by_player, top_n=3, hint=self.comp_hint
        ), [])
        analysis.comps = comps
        top = comps[0] if comps else None
        style = self._guard("style", lambda: self._style_for(top), "standard")

        # Econ and odds need a level: an unread level would be treated as 1
        # ("buy XP to reach level 2" at stage 5), so estimate it instead.
        est = self._guard("level", lambda: self._with_level_estimate(state), state)
        analysis.econ = self._guard("econ", lambda: plan_economy(est, self.mech, style), EconPlan(gold=state.gold or 0))
        analysis.odds = self._guard("odds", lambda: self._odds(est, top, taken_total, style), [])
        analysis.items = self._guard(
            "items", lambda: plan_items(state, self.set_data, comp=top, hp_bucket=hp_bucket(state.hp)), []
        )
        analysis.shop_picks = self._guard("shop", lambda: self._shop_picks(state, top), [])
        analysis.sell_candidates = self._guard("sell", lambda: self._sell_candidates(state, top), [])
        analysis.warnings = self._guard("warnings", lambda: self._warnings(state, analysis.items, style), [])
        if self.last_errors:
            analysis.warnings.append("部分分析出错已跳过，建议可能不完整")
        return analysis

    # --------------------------------------------------------------- helpers
    def _guard(self, name: str, fn: Callable[[], T], default: T) -> T:
        try:
            return fn()
        except Exception as exc:  # never let one section kill the advice
            self.last_errors.append(f"{name}: {exc!r}\n{traceback.format_exc(limit=4)}")
            return default

    def estimate_level(self, state: GameState) -> Optional[int]:
        """Best guess of an unread level: the fielded unit count is a lower
        bound, the standard curve for the round a typical value."""
        if state.level is not None:
            return state.level
        guess = max(len(state.board), self.mech.standard_level_at(state.stage) or 0)
        return min(self.mech.max_level, guess) if guess > 0 else None

    def _with_level_estimate(self, state: GameState) -> GameState:
        if state.level is not None:
            return state
        lvl = self.estimate_level(state)
        return state.model_copy(update={"level": lvl}) if lvl is not None else state

    def _comp_def(self, name: Optional[str]) -> Optional[CompDef]:
        if not name:
            return None
        return next((c for c in self.comps if c.name == name), None)

    def _style_for(self, top: Optional[CompSuggestion]) -> str:
        comp = self._comp_def(top.name if top else None)
        if comp is None or top is None or comp.style not in STYLES:
            return "standard"
        hinted = False
        if self.comp_hint:
            champs = [c for c in (self.set_data.resolve_champion(n) for n in comp.units) if c is not None]
            carry = self.set_data.resolve_champion(comp.carry) if comp.carry else None
            hinted = hint_boost(comp.name, champs, carry, self.comp_hint, self.set_data) > 0
        if top.score >= STYLE_MIN_SCORE or hinted:
            return comp.style
        return "standard"

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

    # ------------------------------------------------------------------ odds
    def _chase_three_star(self, o: Owned, level: int, style: str) -> bool:
        """Is rolling for a 3-star of this (already 2-star) unit a real plan?

        Strong players chase 3-stars on reroll lines (units up to the reroll
        cost), for cheap units while still low level, or when one or two
        copies are missing. A 2-star 1-2 cost at level 8 on a fast 8 line is
        not a roll target, the 4-5 cost upgrades are.
        """
        rr = reroll_level(style)
        reroll_cost = {5: 1, 6: 2, 7: 3}.get(rr) if rr is not None else None
        if reroll_cost is not None and o.cost <= reroll_cost:
            return True
        if o.copies >= 7:
            return True
        if o.cost <= 2 and level <= 6:
            return True
        return o.cost <= 3 and level <= 7 and o.copies >= 5

    def _odds(
        self, state: GameState, top: Optional[CompSuggestion], taken_total: dict[str, int], style: str = "standard"
    ) -> list[HitOdds]:
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
        targets = [api for _k, api in ranked[:MAX_ODDS]]
        if not targets:
            return []
        odds = compute_hit_odds(state, self.set_data, self.mech, taken_total, targets=targets)
        return odds[:MAX_ODDS]

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

    def _shop_picks(self, state: GameState, top: Optional[CompSuggestion]) -> list[str]:
        shop = self._shop_units(state)
        if not shop:
            return []
        owned = owned_units(state, self.set_data)
        comp_apis = self._comp_apis(top)
        active = self._active_traits(state)
        board_apis = {u.api_name for u in state.board}
        in_shop = Counter(u.api_name for u in shop if u is not None)
        early = state.stage is None or state.stage.stage <= 3

        picks: list[tuple[tuple, str, int]] = []
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
            upgrade = ones + in_shop[u.api_name] >= 3
            three_star = upgrade and twos == 2
            if upgrade or three_star:
                tier = 0
            elif copies and u.api_name in comp_apis:
                tier = 1
            elif u.api_name in comp_apis:
                tier = 2
            elif copies and early:
                tier = 3
            elif u.api_name not in board_apis and any(t in active for t in u.traits):
                tier = 4
            else:
                continue
            picks.append(((tier, 0 if three_star else 1, -cost, idx), u.name, cost))
        picks.sort(key=lambda p: p[0])

        out: list[str] = []
        budget = state.gold
        spent = 0
        for _key, name, cost in picks:
            if budget is not None and spent + cost > budget:
                continue
            spent += cost
            out.append(name)
        return out

    # ------------------------------------------------------------------ sell
    def _sell_candidates(self, state: GameState, top: Optional[CompSuggestion]) -> list[str]:
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
            if u.api_name.startswith("?") or u.api_name in comp_apis:
                continue
            if ones[u.api_name] >= 2:
                continue  # pair: one more copy upgrades it
            value = (u.cost or 1) * u.copies
            cands.append((value, u.name))
        cands.sort(key=lambda c: (c[0], c[1]))
        return [name for _v, name in cands]

    # -------------------------------------------------------------- warnings
    def _warnings(self, state: GameState, items: list[ItemSuggestion], style: str) -> list[str]:
        out: list[str] = []
        mech = self.mech
        sr = state.stage
        stage = sr.stage if sr else 0
        planning = state.screen_type in (ScreenType.PLANNING, ScreenType.OTHER, ScreenType.COMBAT)

        if len(state.bench) >= mech.bench_size:
            out.append("备战席已满，先卖掉用不上的棋子，不然买不了新英雄")

        if (
            state.level
            and "board" in state.field_age
            and planning
            and stage >= 2
            and len(state.board) < state.level
        ):
            free = state.level - len(state.board)
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
            loose = [s.item for s in items if not s.components]
            if loose:
                out.append(f"成装还没装备：{', '.join(loose)}，记得给英雄装上")

        if state.gold is not None and state.gold >= RICH_GOLD and stage >= LATE_STAGE:
            cap = mech.interest_step * mech.interest_cap
            out.append(f"金币 {state.gold} 太多了，利息最多只算到 {cap}，多出来的钱拿去升级或搜牌")

        if state.level is None and stage >= 2:
            est = self.estimate_level(state)
            if est is not None:
                out.append(f"没识别到等级，暂按 {est} 级估算，可以在面板里手动修正")

        bucket = hp_bucket(state.hp)
        if bucket == "critical":
            out.append(f"血量只剩 {state.hp}，非常危险，这回合必须全力补强阵容")
        elif bucket == "low":
            out.append(f"血量 {state.hp} 偏低，别再贪经济，优先稳血")

        unresolved = [u.name for u in state.all_units() if u.api_name.startswith("?")]
        unresolved += [u.name for u in state.shop_units if u is not None and u.api_name.startswith("?")]
        if unresolved:
            uniq = list(dict.fromkeys(unresolved))
            names = ", ".join(uniq[:3]) + (" 等" if len(uniq) > 3 else "")
            out.append(f"有英雄名字没识别准（{names}），建议可能不准，可以重新分析一次")
        return out
