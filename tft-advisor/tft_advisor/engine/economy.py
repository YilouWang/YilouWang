"""Economy / tempo planner: save, level, roll, or all-in.

Encodes the standard TFT playbook:
  * 50 gold = max interest; gold above 50 is "free" to spend (XP or rolls).
  * Level on the standard curve (2-1 L4, 2-5 L5, 3-2 L6, 4-1 L7, 4-5 L8, 5-5 L9)
    unless the line is reroll (stay low and slow roll above 50).
  * Fast 8 reaches level 8 at 4-1 / 4-2; fast 9 then goes 9 around 5-1 / 5-2.
  * Big roll-downs happen at fixed stabilization points (3-2 / 4-1 / 4-2)
    and whenever HP is low; at critical HP spend everything.
Budgets are always affordable: a plan never says "roll" with less gold than
one reroll, and never says "level" when the XP cannot be bought.
"""

from __future__ import annotations

from typing import Optional

from ..data.mechanics import Mechanics
from ..models import EconAction, EconPlan, GameState, StageRound

STYLES = ("standard", "fast8", "fast9", "reroll1", "reroll2", "reroll3")

ROLLDOWN_ROUNDS = {(3, 2), (4, 1), (4, 2)}


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


def _wanted_level(sr: StageRound, std_level: int, style: str, bucket: str) -> int:
    want = std_level
    if style in ("fast8", "fast9"):
        if sr.key >= (4, 2) or (sr.key >= (4, 1) and bucket in ("healthy", "unknown")):
            want = max(want, 8)
        elif sr.key >= (3, 5) and bucket == "healthy":
            want = max(want, 7)
    if style == "fast9":
        if sr.key >= (5, 2) or (sr.key >= (5, 1) and bucket in ("healthy", "unknown")):
            want = max(want, 9)
    return want


def plan_economy(state: GameState, mech: Mechanics, style: str = "standard") -> EconPlan:
    gold = max(0, state.gold or 0)
    sr: Optional[StageRound] = state.stage
    level_estimated = state.level is None
    level = state.level or (mech.standard_level_at(sr) if sr else None) or 1
    xp_cur = state.xp_current if not level_estimated else 0
    hp = state.hp
    bucket = hp_bucket(hp)
    cap_gold = mech.interest_step * mech.interest_cap
    roll = max(1, mech.roll_cost)
    style = style if style in STYLES else "standard"

    plan = EconPlan(
        gold=gold,
        interest=mech.interest(gold),
        streak_gold=mech.streak_gold(state.streak),
        base_income=mech.base_income,
        expected_income=mech.income(gold, state.streak),
        xp_to_next=mech.xp_to_next(level, xp_cur),
        style=style,
        level_estimated=level_estimated,
    )
    to_next = mech.gold_to_reach(level, xp_cur, level + 1) if level < mech.max_level else None
    plan.gold_to_next_level = to_next

    def done(action: EconAction, reason: str, budget: int = 0, target: Optional[int] = None) -> EconPlan:
        budget = max(0, min(gold, budget))
        if action in (EconAction.ROLL, EconAction.SLOW_ROLL, EconAction.ALL_IN) and budget < roll:
            action, reason, budget = EconAction.SAVE, "金币不够搜一次，先存钱", 0
        elif action == EconAction.LEVEL_AND_ROLL and budget < roll:
            action, budget = EconAction.LEVEL, 0
            if target:
                reason = f"升到 {target} 级（剩下的钱不够再搜）"
        plan.recommendation = action
        plan.reason = reason
        plan.roll_budget = budget
        plan.target_level = target
        return plan

    def affordable_target(target: int, keep: int) -> tuple[Optional[int], int]:
        """Highest level <= target reachable now while keeping ``keep`` gold."""
        best: tuple[Optional[int], int] = (None, 0)
        for lvl in range(level + 1, min(target, mech.max_level) + 1):
            spend = mech.gold_to_reach(level, xp_cur, lvl)
            if spend is None or spend > gold - keep:
                break
            best = (lvl, spend)
        return best

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
        if level < std_level and to_next is not None and to_next <= gold // 3:
            return done(EconAction.LEVEL_AND_ROLL, f"血量 {hp}，危险：先升 {level + 1} 级再 all-in 搜牌", gold - to_next, level + 1)
        return done(EconAction.ALL_IN, f"血量 {hp}，危险：all-in 搜牌补强阵容，不要存钱", gold)

    # --- Reroll lines --------------------------------------------------------
    if rr is not None:
        if level < rr and (stage >= 3 or level + 1 <= std_level):
            target, spend = affordable_target(rr, 0)
            if target is not None:
                return done(EconAction.LEVEL, f"赌狗阵容：先升到 {target} 级再开始慢搜", 0, target)
        if level >= rr and stage >= 3:
            if bucket == "low":
                return done(EconAction.ROLL, f"血量 {hp} 偏低：搜到 20 左右保血", gold - 20)
            if bucket == "medium" and stage >= 4:
                return done(EconAction.ROLL, f"血量 {hp}，第{stage}阶段还没三星：搜到 20 左右追三星", gold - 20)
            if gold > cap_gold:
                return done(EconAction.SLOW_ROLL, f"慢搜：只花 {cap_gold} 以上的部分", gold - cap_gold)
            return done(EconAction.SAVE, f"慢搜节奏：攒回 {cap_gold} 再搜")

    # --- Standard / fast 8 / fast 9 -----------------------------------------
    is_rolldown_point = sr.key in ROLLDOWN_ROUNDS
    want_level = _wanted_level(sr, std_level, style, bucket)

    if level < want_level and level < mech.max_level:
        keep = 0 if bucket == "low" or is_rolldown_point else 10
        target, spend = affordable_target(want_level, keep)
        if target is not None:
            behind = f"（标准是 {want_level} 级）" if target < want_level else ""
            if is_rolldown_point and bucket in ("low", "medium"):
                reserve = 0 if bucket == "low" else 10
                return done(
                    EconAction.LEVEL_AND_ROLL,
                    f"{sr.stage}-{sr.round} 是关键节点：升到 {target} 级后搜牌稳血{behind}",
                    gold - spend - reserve,
                    target,
                )
            return done(EconAction.LEVEL, f"按节奏升到 {target} 级{behind}", 0, target)

    if is_rolldown_point and bucket in ("low", "medium"):
        keep = 0 if bucket == "low" else 20
        return done(EconAction.ROLL, f"血量 {hp}：这回合搜牌稳住，保留约 {keep} 金币", gold - keep)

    if bucket == "low" and stage >= 3:
        return done(EconAction.ROLL, f"血量 {hp} 偏低：搜到 10-20 金币补强", gold - 10)

    # Fast 8 / fast 9 roll-down once the target level is reached.
    if style in ("fast8", "fast9") and level >= 8 and sr.key >= (4, 2) and gold > 10:
        if style == "fast9" and level < 9 and bucket in ("healthy", "medium") and sr.key < (5, 2):
            return done(EconAction.SLOW_ROLL, "速9：8级小搜稳血，留钱冲 9 级", gold - cap_gold)
        return done(EconAction.ROLL, f"{level} 级搜 4 费主C和二星，搜到 10 左右", gold - 10)

    # Spare gold above the interest cap goes into XP (tempo) or rolls late game.
    if gold > cap_gold:
        extra = gold - cap_gold
        if level < mech.max_level and to_next is not None and to_next <= extra and (stage <= 4 or level < 9):
            return done(EconAction.LEVEL, f"利息已满：多出的 {extra} 金币拿来升 {level + 1} 级", 0, level + 1)
        if stage >= 4:
            return done(EconAction.SLOW_ROLL, f"利息已满：用多出的 {extra} 金币搜牌", extra)
        return done(EconAction.SAVE, f"利息已满 ({cap_gold})，可以用多余的金币买经验/补强")

    # Level with cheap XP: one buy away, affordable, and no interest lost.
    if (
        to_next is not None
        and to_next <= mech.buy_xp_cost
        and to_next <= gold
        and not level_estimated
        and level < want_level + 1
        and mech.interest(gold - to_next) == mech.interest(gold)
        and stage >= 2
    ):
        return done(EconAction.LEVEL, f"只差 {plan.xp_to_next} 经验，买一次就升级且不掉利息", 0, level + 1)

    nxt = _next_interest_step(gold, mech)
    if nxt is not None:
        return done(EconAction.SAVE, f"存钱吃利息：下一个利息点 {nxt}（还差 {nxt - gold}）")
    return done(EconAction.SAVE, f"保持 {cap_gold} 金币利息")
