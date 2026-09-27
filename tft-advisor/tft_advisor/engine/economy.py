"""Economy / tempo planner: save, level, roll, or all-in.

Encodes the standard TFT playbook:
  * 50 gold = max interest; gold above 50 is "free" to spend (XP or rolls).
  * Level on the standard curve (2-1 L4, 2-5 L5, 3-2 L6, 4-1 L7, 4-5 L8, 5-5 L9)
    unless the line is reroll (stay low and slow roll above 50).
  * Fast 8 reaches level 8 at 4-1 / 4-2; fast 9 then goes 9 around 5-1 / 5-2.
  * Big roll-downs happen at fixed stabilization points (3-2 / 4-1 / 4-2)
    and whenever HP is low; at critical HP spend everything. 3-2 keeps about
    20 gold unless HP is critical.
  * Streaks: a loss streak with enough HP keeps the streak (no 3-2 roll);
    a win streak means the board is stable, so roll-down points only level.
  * Reroll lines level on the standard curve in stage 2, reach the reroll
    level in stage 3 (3-5 for 3-cost rerolls) and slow roll above 50. Once
    the reroll target is 3-star (``key_star >= 3``) or from 4-5 on, they
    level like a standard line again.
  * Fast 8 / fast 9 at level 8+ roll only while the carry is below 2-star
    (``key_star``); with a 2-star carry they put the gold above 50 into XP
    and go 9 (right away when it costs no interest, otherwise from 5-1).
  * A carry the current level barely shows (``carry_cost``: a 5 cost at
    level 8) makes a fast 8 line a fast 9 line: never roll at 8 for it, level
    to 9 instead.
  * Players behind the standard curve in stage 4+ buy the catch-up level as
    soon as it is affordable (no 10 gold floor); the stage 2 level points
    (2-1, 2-5) take the level when one XP buy reaches it.
  * Critical HP buys the next level when it is cheap (a third of the gold, or
    half when behind the curve); the analyzer also compares roll-down odds.
  * PvE rounds (x-7, ``Mechanics.is_pve``) never roll or all-in: the next
    player fight is (x+1)-1, and rolling there gives the same board without
    losing this round's interest. A cheap level (no interest lost) stays.
Budgets are always affordable: a plan never says "roll" with less gold than
one reroll, and never says "level" when the XP cannot be bought.
"""

from __future__ import annotations

from typing import Optional

from ..data.mechanics import Mechanics
from ..models import EconAction, EconPlan, GameState, StageRound

STYLES = ("standard", "fast8", "fast9", "reroll1", "reroll2", "reroll3")

ROLLDOWN_ROUNDS = {(3, 2), (4, 1), (4, 2)}
REROLL_END = (4, 5)  # reroll lines stop staying low from here on, 3-star or not
STREAK_LEN = 3  # a streak worth playing around (3+ wins / losses)
LOSS_STREAK_HP_FLOOR = {3: 45}  # stage -> HP above which a loss streak skips the roll-down
EARLY_RESERVE_ROUND = (3, 2)  # roll-down that keeps ~20 gold unless HP is critical
EARLY_RESERVE = 20
CARRY_ODDS_FLOOR = 0.10  # below this shop odds the carry needs a higher level before rolling for it
FAST_NINE_ROUND = (5, 1)  # fast lines go 9 from here even when it costs interest


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


def reroll_cost(style: str) -> Optional[int]:
    """Highest unit cost a reroll line wants to 3-star (reroll1 -> 1, ...)."""
    return {"reroll1": 1, "reroll2": 2, "reroll3": 3}.get(style)


def carry_needs_level(mech: Mechanics, level: int, carry_cost: Optional[int]) -> bool:
    """True when the carry barely shows at ``level`` and a higher level shows it
    more often (a 5 cost at level 8): level up before rolling for it."""
    if not carry_cost or level >= mech.max_level:
        return False
    now = mech.odds(level, carry_cost)
    return now < CARRY_ODDS_FLOOR and mech.odds(level + 1, carry_cost) > now


def plan_economy(
    state: GameState,
    mech: Mechanics,
    style: str = "standard",
    key_star: Optional[int] = None,
    carry_cost: Optional[int] = None,
) -> EconPlan:
    """Save / level / roll plan for this round.

    ``key_star``: star level of the unit the line is built around (the reroll
    target on reroll lines, the carry otherwise); 0 = not owned yet, None =
    unknown (no target comp). ``carry_cost``: cost of the comp's carry (fast
    lines: a carry level 8 barely shows turns fast 8 into fast 9).
    """
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
    if style == "fast8" and carry_needs_level(mech, 8, carry_cost):
        # "Fast 8" with a carry level 8 barely shows is a fast 9 line.
        style = "fast9"
    streak = state.streak or 0
    pve = sr is not None and sr.stage >= 2 and mech.is_pve(sr)

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

    def save_reason() -> str:
        nxt = _next_interest_step(gold, mech)
        if nxt is not None:
            return f"存钱吃利息：下一个利息点 {nxt}（还差 {nxt - gold}）"
        return f"保持 {cap_gold} 金币利息"

    def no_roll(action: EconAction) -> tuple[EconAction, str]:
        """What a roll whose budget cannot pay for one reroll becomes."""
        if bucket in ("low", "critical") and sr is not None and sr.stage >= 4:
            # Never "save to 50" at low HP late: the gold is just short.
            return EconAction.HOLD, "金币不多：调整站位，卖掉闲置棋子，下回合再搜"
        if gold < roll:
            return EconAction.SAVE, "金币不够搜一次，先存钱"
        if action == EconAction.SLOW_ROLL:
            return EconAction.SAVE, save_reason()
        return EconAction.SAVE, "金币不多：先存钱吃利息，下回合再搜"

    def done(action: EconAction, reason: str, budget: int = 0, target: Optional[int] = None) -> EconPlan:
        budget = max(0, min(gold, budget))
        if pve and action in (EconAction.ROLL, EconAction.ALL_IN, EconAction.LEVEL_AND_ROLL):
            # PvE round (x-7): the next player fight is (x+1)-1 and rolling
            # now gives the same board for it, minus this round's interest.
            wait = f"野怪回合先不搜，下回合 {sr.stage + 1}-1 再搜稳血"
            spend = mech.gold_to_reach(level, xp_cur, target) if target and not level_estimated else None
            cheap = spend is not None and 0 < spend <= gold and (
                spend <= mech.buy_xp_cost or mech.interest(gold - spend) == mech.interest(gold)
            )
            if action == EconAction.LEVEL_AND_ROLL and cheap:
                action, reason, budget = EconAction.LEVEL, f"升到 {target} 级；{wait}", 0
            else:
                action, reason, budget, target = EconAction.SAVE, wait, 0, None
        if action == EconAction.ALL_IN:
            pass  # stays all-in with no gold: the advice is to reposition and sell
        elif action in (EconAction.ROLL, EconAction.SLOW_ROLL) and budget < roll:
            (action, reason), budget = no_roll(action), 0
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
    if rr is not None and ((key_star or 0) >= 3 or sr.key >= REROLL_END):
        # The reroll is over (target already 3-star, or too late to keep
        # staying low): level like a standard line from the current level.
        rr = None
        style = "standard"
        plan.style = style

    # --- Critical HP: everything into the board -----------------------------
    if bucket == "critical" and stage >= 3:
        # One more unit on the board is worth a lot at critical HP: buy the
        # level when it is cheap (a reroll line only up to its reroll level).
        if (
            to_next is not None
            and 0 < to_next <= gold
            and not level_estimated
            and (rr is None or level < rr)
            and (to_next <= gold // 3 or (level < std_level and to_next <= gold // 2))
        ):
            return done(
                EconAction.LEVEL_AND_ROLL,
                f"血量 {hp}，危险：先升 {level + 1} 级多上一个人，再 all-in 搜牌",
                gold - to_next,
                level + 1,
            )
        return done(EconAction.ALL_IN, f"血量 {hp}，危险：all-in 搜牌补强阵容，不要存钱", gold)

    # --- Reroll lines --------------------------------------------------------
    # Stage 2 follows the standard curve (with an interest floor, below);
    # the reroll level comes in stage 3 (3-5 for 3-cost rerolls).
    if rr is not None and stage >= 3:
        goal = rr if rr < 7 or sr.key >= (3, 5) else min(rr, max(level, std_level))
        if level < goal:
            target, spend = affordable_target(goal, 0)
            if target is not None:
                tail = "再开始慢搜" if target >= rr else f"，之后到 {rr} 级慢搜"
                return done(EconAction.LEVEL, f"赌狗阵容：先升到 {target} 级{tail}", 0, target)
        if level >= rr:
            if bucket == "low":
                return done(EconAction.ROLL, f"血量 {hp} 偏低：搜到 20 左右保血", gold - 20)
            if bucket == "medium" and stage >= 4:
                chase = "还没三星：" if key_star is not None else "："
                return done(EconAction.ROLL, f"血量 {hp}，第{stage}阶段{chase}搜到 20 左右追三星", gold - 20)
            if gold > cap_gold:
                return done(EconAction.SLOW_ROLL, f"慢搜：只花 {cap_gold} 以上的部分", gold - cap_gold)
            return done(EconAction.SAVE, f"慢搜节奏：攒回 {cap_gold} 再搜")

    # --- Standard / fast 8 / fast 9 -----------------------------------------
    is_rolldown_point = sr.key in ROLLDOWN_ROUNDS
    want_level = _wanted_level(sr, std_level, style, bucket)
    # Streaks: losing on purpose with enough HP keeps the streak gold; a win
    # streak means the board already holds, so roll-down points only level.
    floor = LOSS_STREAK_HP_FLOOR.get(stage)
    keep_streak = streak <= -STREAK_LEN and floor is not None and hp is not None and hp >= floor
    stable = streak >= STREAK_LEN and bucket in ("healthy", "medium", "unknown")
    rolldown = is_rolldown_point and bucket in ("low", "medium") and not keep_streak and not stable
    low_hp = bucket == "low" and stage >= 3
    early_reserve = sr.key == EARLY_RESERVE_ROUND

    if level < want_level and level < mech.max_level:
        # Behind the standard curve late: buy the catch-up level as soon as it
        # is affordable. The stage 2 level points (2-1 L4, 2-5 L5) take the
        # level when one XP buy reaches it, even if that costs 1 interest.
        behind_late = stage >= 4 and level < std_level
        level_point = (
            stage == 2
            and f"{sr.stage}-{sr.round}" in mech.standard_levels
            and (mech.gold_to_reach(level, xp_cur, want_level) or 0) <= mech.buy_xp_cost
        )
        keep = 0 if bucket == "low" or is_rolldown_point or behind_late or level_point else 10
        target, spend = affordable_target(want_level, keep)
        if target is not None:
            behind = f"（标准是 {want_level} 级）" if target < want_level else ""
            if rolldown or low_hp:
                if early_reserve:
                    reserve = EARLY_RESERVE
                elif rolldown:
                    reserve = 0 if bucket == "low" else 10
                else:
                    reserve = 10  # low HP outside the roll-down rounds
                if rolldown:
                    reason = f"{sr.stage}-{sr.round} 是关键节点：升到 {target} 级后搜牌稳血{behind}"
                else:
                    reason = f"血量 {hp} 偏低：升到 {target} 级后搜牌补强{behind}"
                return done(EconAction.LEVEL_AND_ROLL, reason, gold - spend - reserve, target)
            if keep_streak and is_rolldown_point:
                reason = f"连败 {-streak} 场且血量还够：升到 {target} 级但别搜牌，继续吃连败金币"
            elif stable and is_rolldown_point:
                reason = f"连胜 {streak} 场阵容够强：升到 {target} 级，不用搜牌"
            else:
                reason = f"按节奏升到 {target} 级{behind}"
            return done(EconAction.LEVEL, reason, 0, target)

    if rolldown:
        keep = EARLY_RESERVE if early_reserve else (0 if bucket == "low" else 20)
        tail = f"保留约 {keep} 金币" if keep else "搜光金币"
        return done(EconAction.ROLL, f"血量 {hp}：这回合搜牌稳住，{tail}", gold - keep)

    if low_hp:
        return done(EconAction.ROLL, f"血量 {hp} 偏低：搜到 10-20 金币补强", gold - 10)

    # Fast 8 / fast 9 at level 8+: roll only while the carry is not 2-star.
    if style in ("fast8", "fast9") and level >= 8 and sr.key >= (4, 2) and gold > 10:
        carry_ready = key_star is not None and key_star >= 2
        nine = min(9, mech.max_level)
        nine_round = f"{FAST_NINE_ROUND[0]}-{FAST_NINE_ROUND[1]}"
        extra = gold - cap_gold
        if not carry_ready and level < nine and to_next is not None and carry_needs_level(mech, level, carry_cost):
            # The carry barely shows at this level (a 5 cost at 8): never roll
            # here for it, level first.
            hard = f"主C是 {carry_cost} 费，{level} 级很难搜到"
            if gold - to_next >= cap_gold or (sr.key >= FAST_NINE_ROUND and to_next <= gold):
                return done(EconAction.LEVEL, f"{hard}：先升 {level + 1} 级再找", 0, level + 1)
            if extra > 0 and bucket == "medium":
                return done(EconAction.SLOW_ROLL, f"{hard}：只花 {cap_gold} 以上的钱补强，留钱升 {level + 1} 级", extra)
            if extra >= mech.buy_xp_cost:
                return done(EconAction.SAVE, f"{hard}：多出的 {extra} 金币买经验，保持 {cap_gold} 利息，准备升 {level + 1} 级")
            return done(EconAction.SAVE, f"{hard}：存钱升 {level + 1} 级（约 {to_next} 金币）")
        if not carry_ready and (key_star is not None or is_rolldown_point):
            if style == "fast9" and level < 9 and bucket in ("healthy", "medium") and sr.key < (5, 2):
                return done(EconAction.SLOW_ROLL, "速9：8级小搜稳血，留钱冲 9 级", gold - cap_gold)
            return done(EconAction.ROLL, f"{level} 级搜主C二星，搜到 10 左右", gold - 10)
        if carry_ready and level < nine and to_next is not None:
            star = f"主C已{key_star}星"
            left = gold - to_next
            if left >= cap_gold:
                return done(EconAction.LEVEL, f"{star}，升 {level + 1} 级不掉利息：现在升级找 5 费", 0, level + 1)
            if left >= cap_gold - mech.interest_step:
                return done(EconAction.LEVEL, f"{star}，升 {level + 1} 级只少 1 利息：现在升级找 5 费", 0, level + 1)
            if sr.key < FAST_NINE_ROUND:
                if extra >= mech.buy_xp_cost:
                    return done(
                        EconAction.SAVE,
                        f"{star}，不用再搜：多出的 {extra} 金币买经验，保持 {cap_gold} 利息，{nine_round} 后升 {level + 1} 级",
                    )
                return done(EconAction.SAVE, f"{star}，不用再搜：存钱，{nine_round} 后升 {level + 1} 级找 5 费")
            if left >= 10:
                return done(EconAction.LEVEL, f"{star}：升 {level + 1} 级找 5 费", 0, level + 1)
            return done(EconAction.SAVE, f"{star}，不用再搜：存够 {to_next + 10} 金币就升 {level + 1} 级")

    # Spare gold above the interest cap goes into XP (tempo) or rolls late game.
    reroll_cap = rr is not None and level + 1 > rr  # a reroll line does not overlevel
    if gold > cap_gold:
        extra = gold - cap_gold
        if (
            level < mech.max_level
            and to_next is not None
            and to_next <= extra
            and (stage <= 4 or level < 9)
            and not reroll_cap
        ):
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
        and not reroll_cap
        and mech.interest(gold - to_next) == mech.interest(gold)
        and stage >= 2
    ):
        return done(EconAction.LEVEL, f"只差 {plan.xp_to_next} 经验，买一次就升级且不掉利息", 0, level + 1)

    if keep_streak and is_rolldown_point:
        return done(EconAction.SAVE, f"连败 {-streak} 场且血量还够：别搜牌，继续吃连败金币和利息")
    return done(EconAction.SAVE, save_reason())
