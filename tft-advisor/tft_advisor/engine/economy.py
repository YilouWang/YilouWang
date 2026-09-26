"""Economy / tempo planner: save, level, roll, or all-in.

Encodes the standard TFT playbook:
  * 50 gold = max interest; gold above 50 is "free" to spend (XP or rolls).
  * Level on the standard curve (2-1 L4, 2-5 L5, 3-2 L6, 4-1 L7, 4-5 L8, 5-5 L9)
    unless the line is reroll (stay low and slow roll above 50).
  * Big roll-downs happen at fixed stabilization points (3-2 / 4-1 / 4-2)
    and whenever HP is low; at critical HP spend everything.
"""

from __future__ import annotations

from typing import Optional

from ..data.mechanics import Mechanics
from ..models import EconAction, EconPlan, GameState, StageRound

STYLES = ("standard", "fast8", "reroll1", "reroll2", "reroll3")


def hp_bucket(hp: Optional[int]) -> str:
    if hp is None:
        return "unknown"
    if hp >= 70:
        return "healthy"
    if hp >= 45:
        return "medium"
    if hp >= 25:
        return "low"
    return "critical"


def reroll_level(style: str) -> Optional[int]:
    return {"reroll1": 5, "reroll2": 6, "reroll3": 7}.get(style)


def _next_interest_step(gold: int, mech: Mechanics) -> Optional[int]:
    if gold >= mech.interest_step * mech.interest_cap:
        return None
    return (gold // mech.interest_step + 1) * mech.interest_step


def plan_economy(state: GameState, mech: Mechanics, style: str = "standard") -> EconPlan:
    gold = max(0, state.gold or 0)
    level = state.level or 1
    sr: Optional[StageRound] = state.stage
    hp = state.hp
    bucket = hp_bucket(hp)
    cap_gold = mech.interest_step * mech.interest_cap

    plan = EconPlan(
        gold=gold,
        interest=mech.interest(gold),
        streak_gold=mech.streak_gold(state.streak),
        base_income=mech.base_income,
        expected_income=mech.income(gold, state.streak),
        xp_to_next=mech.xp_to_next(level, state.xp_current),
    )
    to_next = mech.gold_to_reach(level, state.xp_current, level + 1) if level < mech.max_level else None
    plan.gold_to_next_level = to_next

    def done(action: EconAction, reason: str, budget: int = 0, target: Optional[int] = None) -> EconPlan:
        plan.recommendation = action
        plan.reason = reason
        plan.roll_budget = max(0, min(gold, budget))
        plan.target_level = target
        return plan

    if sr is None:
        return done(EconAction.HOLD, "还没识别到回合，先按 F6 分析一次")

    stage = sr.stage
    if stage == 1:
        return done(EconAction.HOLD, "第一阶段：买对子，别搜牌，留钱到 2-1")
    if mech.is_carousel(sr):
        return done(EconAction.HOLD, "选秀回合：优先拿核心装备散件或主C")

    std_level = mech.standard_level_at(sr) or level
    rr = reroll_level(style)

    # --- Critical HP: everything into the board -----------------------------
    if bucket == "critical" and stage >= 3:
        if level < mech.max_level and to_next is not None and to_next <= gold // 3 and level < std_level:
            return done(EconAction.LEVEL_AND_ROLL, f"血量 {hp}，危险：先升 {level + 1} 级再 all-in 搜牌", gold - to_next, level + 1)
        return done(EconAction.ALL_IN, f"血量 {hp}，危险：all-in 搜牌补强阵容，不要存钱", gold)

    # --- Reroll lines --------------------------------------------------------
    if rr is not None:
        if level < rr and to_next is not None and to_next <= gold and (stage >= 3 or level + 1 <= std_level):
            return done(EconAction.LEVEL, f"赌狗阵容：先升到 {rr} 级再开始慢搜", 0, rr)
        if level >= rr and stage >= 3:
            if bucket in ("low",):
                return done(EconAction.ROLL, f"血量 {hp} 偏低：搜到 20 左右保血", gold - 20)
            if gold > cap_gold:
                return done(EconAction.SLOW_ROLL, f"慢搜：只花 {cap_gold} 以上的部分", gold - cap_gold)
            return done(EconAction.SAVE, "慢搜节奏：攒回 50 再搜")

    # --- Standard / fast 8 ---------------------------------------------------
    rolldown_rounds = {(3, 2), (4, 1), (4, 2)}
    is_rolldown_point = sr.key in rolldown_rounds

    want_level = std_level
    if style == "fast8" and sr.key >= (4, 1):
        want_level = max(want_level, 8)
    if style == "fast8" and sr.key >= (3, 2) and bucket == "healthy":
        want_level = max(want_level, 7 if sr.key >= (3, 5) else want_level)

    if level < want_level and to_next is not None and level < mech.max_level:
        spend = mech.gold_to_reach(level, state.xp_current, want_level) or to_next
        keep = 0 if bucket in ("low",) or is_rolldown_point else 10
        if spend <= gold - keep:
            if is_rolldown_point and bucket in ("low", "medium"):
                return done(
                    EconAction.LEVEL_AND_ROLL,
                    f"{sr.stage}-{sr.round} 是关键节点：升到 {want_level} 级后搜牌稳血",
                    gold - spend - (0 if bucket == "low" else 10),
                    want_level,
                )
            return done(EconAction.LEVEL, f"按标准节奏升 {want_level} 级", 0, want_level)

    if is_rolldown_point and bucket in ("low", "medium"):
        keep = 0 if bucket == "low" else 20
        return done(EconAction.ROLL, f"血量 {hp}：这回合搜牌稳住，保留约 {keep} 金币", gold - keep)

    if bucket == "low" and stage >= 3:
        return done(EconAction.ROLL, f"血量 {hp} 偏低：搜到 10-20 金币补强", gold - 10)

    # Spare gold above the interest cap goes into XP (tempo) or rolls late game.
    if gold > cap_gold:
        extra = gold - cap_gold
        if level < mech.max_level and to_next is not None and to_next <= extra and (stage <= 4 or level < 9):
            return done(EconAction.LEVEL, f"利息已满：多出的 {extra} 金币拿来升 {level + 1} 级", 0, level + 1)
        if stage >= 4:
            return done(EconAction.SLOW_ROLL, f"利息已满：用多出的 {extra} 金币搜牌", extra)
        return done(EconAction.SAVE, f"利息已满 ({cap_gold})，可以用多余的金币买经验/补强")

    # Level with cheap XP: if only one buy away and we stay above an interest step.
    if (
        to_next is not None
        and to_next <= mech.buy_xp_cost
        and level < want_level + 1
        and mech.interest(gold - to_next) == mech.interest(gold)
        and stage >= 2
    ):
        return done(EconAction.LEVEL, f"只差 {plan.xp_to_next} 经验，买一次就升级且不掉利息", 0, level + 1)

    nxt = _next_interest_step(gold, mech)
    if nxt is not None:
        return done(EconAction.SAVE, f"存钱吃利息：下一个利息点 {nxt}（还差 {nxt - gold}）")
    return done(EconAction.SAVE, "保持 50 金币利息")
