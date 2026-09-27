"""Post-game review from the JSONL game log written by the app.

``summarize_log`` is deterministic (no API): per-round HP / gold / level /
streak / board (with items) / target comp timeline, level timing vs the
standard curve, interest below the cap on rounds the assistant said to save,
augments, HP lost per stage. ``llm_review`` asks Claude for a short Chinese
coaching summary on top of it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .data.mechanics import Mechanics
from .models import StageRound


def latest_log(directory: Path) -> Optional[Path]:
    """Newest ``game-*.jsonl`` by name (names start with the local start time).

    Sorted by stem so ``game-...-123-1.jsonl`` (same-millisecond collision)
    comes after ``game-...-123.jsonl``.
    """
    try:
        logs = sorted((p for p in directory.glob("game-*.jsonl") if p.is_file()), key=lambda p: p.stem)
    except OSError:
        return None
    return logs[-1] if logs else None


def load_records(path: Path) -> list[dict[str, Any]]:
    """JSON objects of a log file; blank, broken, non-object and undecodable lines are skipped."""
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _game_of(rec: dict[str, Any]) -> Optional[str]:
    gid = rec.get("game_id")
    if not gid and isinstance(rec.get("state"), dict):
        gid = rec["state"].get("game_id")
    return str(gid) if gid else None


def last_game(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only the records of the last game in the file (older logs may mix games)."""
    games = [g for g in (_game_of(r) for r in records) if g]
    if not games:
        return records
    last = games[-1]
    return [r for r in records if _game_of(r) in (None, last)]


def _stage_of(state: dict[str, Any]) -> Optional[StageRound]:
    st = state.get("stage")
    if isinstance(st, dict) and "stage" in st and "round" in st:
        try:
            return StageRound(stage=int(st["stage"]), round=int(st["round"]))
        except (TypeError, ValueError):
            return None
    if isinstance(st, str):
        return StageRound.parse(st)
    return None


def _round_index(sr: StageRound) -> int:
    """Rounds since 1-1 (stage 1 has 4 rounds, later stages 7)."""
    if sr.stage <= 1:
        return sr.round - 1
    return 4 + (sr.stage - 2) * 7 + (sr.round - 1)


def _standard_round(mech: Mechanics, level: int) -> Optional[StageRound]:
    """First round at which a standard line has reached ``level``."""
    best: Optional[StageRound] = None
    for key, lvl in mech.standard_levels.items():
        sr = StageRound.parse(key)
        if sr is not None and lvl >= level and (best is None or sr.key < best.key):
            best = sr
    return best


def _unit_text(u: dict[str, Any]) -> str:
    try:
        star = max(1, int(u.get("star") or 1))
    except (TypeError, ValueError):
        star = 1
    items = [str(i) for i in (u.get("items") or []) if i] if isinstance(u.get("items"), list) else []
    return f"{u.get('name')}{'★' * star}" + (f"[{'+'.join(items)}]" if items else "")


def _level_timing(timeline: list[dict[str, Any]], mech: Mechanics) -> list[dict[str, Any]]:
    """When each level was first logged, against the standard curve.

    ``rounds_late`` is a lower bound backed by the log: > 0 only when a logged
    round at or after the standard round still shows a lower level; <= 0 when
    the level was already reached by the standard round (negative = early);
    None when a gap in the log hides when the level up happened.
    """
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    rows = [(StageRound.parse(r["stage"]), r) for r in timeline]
    for i, (sr, row) in enumerate(rows):
        lvl = row.get("level")
        if not isinstance(lvl, int) or isinstance(lvl, bool) or lvl in seen or sr is None:
            continue
        seen.add(lvl)
        std = _standard_round(mech, lvl)
        entry: dict[str, Any] = {"level": lvl, "reached_at": row["stage"], "standard_round": None, "rounds_late": None, "below_at": []}
        if std is not None:
            entry["standard_round"] = f"{std.stage}-{std.round}"
            below = [
                (psr, prow["stage"])
                for psr, prow in rows[:i]
                if psr is not None and psr.key >= std.key and isinstance(prow.get("level"), int) and prow["level"] < lvl
            ]
            entry["below_at"] = [stage for _, stage in below]
            delta = _round_index(sr) - _round_index(std)
            if delta <= 0:
                entry["rounds_late"] = delta
            elif below:
                entry["rounds_late"] = _round_index(below[-1][0]) - _round_index(std) + 1
        out.append(entry)
    return out


def summarize_log(records: list[dict[str, Any]], mech: Mechanics) -> dict[str, Any]:
    """Collapse the log to one row per round (last record of each round wins)."""
    rounds: dict[tuple[int, int], dict[str, Any]] = {}
    item_bench: dict[tuple[int, int], list[str]] = {}
    augments: dict[tuple[int, int], list[str]] = {}
    for rec in last_game(records):
        state = rec.get("state")
        if not isinstance(state, dict):
            # Advice-only record (Claude strategy that arrived after the rules advice).
            sr = StageRound.parse(rec.get("stage")) if isinstance(rec.get("stage"), str) else None
            advice = rec.get("advice")
            if sr is not None and sr.key in rounds and isinstance(advice, dict) and advice.get("headline"):
                rounds[sr.key]["advice"] = advice.get("headline")
            continue
        sr = _stage_of(state)
        if sr is None:
            continue
        analysis = rec.get("analysis") if isinstance(rec.get("analysis"), dict) else {}
        econ = analysis.get("econ") if isinstance(analysis.get("econ"), dict) else {}
        comps = analysis.get("comps") if isinstance(analysis.get("comps"), list) else []
        top = comps[0] if comps and isinstance(comps[0], dict) else {}
        advice = rec.get("advice") if isinstance(rec.get("advice"), dict) else {}
        rounds[sr.key] = {
            "stage": f"{sr.stage}-{sr.round}",
            "hp": state.get("hp"),
            "gold": state.get("gold"),
            "level": state.get("level"),
            "streak": state.get("streak"),
            "board": [_unit_text(u) for u in state.get("board") or [] if isinstance(u, dict)],
            "comp": top.get("name"),
            "style": econ.get("style"),
            "advice": advice.get("headline"),
            "econ": econ.get("recommendation"),
        }
        bench = state.get("item_bench")
        item_bench[sr.key] = [str(x) for x in bench if x] if isinstance(bench, list) else []
        augs = state.get("augments")
        augments[sr.key] = [str(x) for x in augs if x] if isinstance(augs, list) else []
    keys = sorted(rounds)
    timeline = [rounds[k] for k in keys]

    # Interest below the cap, only where the assistant itself said to save from
    # stage 3 on: stage 2 cannot reach the cap, and gold spent on a level or a
    # roll the assistant asked for is not an econ mistake.
    cap_gold = mech.interest_step * mech.interest_cap
    interest_short = 0
    short_rounds: list[str] = []
    for row in timeline:
        g = row.get("gold")
        sr = StageRound.parse(row["stage"])
        if row.get("econ") != "save" or not isinstance(g, int) or sr is None or sr.stage < 3 or g >= cap_gold:
            continue
        miss = max(0, mech.interest_cap - mech.interest(g))
        if miss:
            interest_short += miss
            short_rounds.append(row["stage"])

    hp_by_stage: dict[int, list[int]] = {}
    for row in timeline:
        sr = StageRound.parse(row["stage"])
        if sr is not None and isinstance(row.get("hp"), int):
            hp_by_stage.setdefault(sr.stage, []).append(row["hp"])
    hp_lost = {s: (v[0] - v[-1]) for s, v in hp_by_stage.items() if len(v) >= 2}

    augment_seen: list[dict[str, Any]] = []
    for k in keys:
        for name in augments[k]:
            if name not in {a["name"] for a in augment_seen}:
                augment_seen.append({"name": name, "seen_at": rounds[k]["stage"]})

    final = timeline[-1] if timeline else {}
    return {
        "rounds": len(timeline),
        "final_stage": final.get("stage"),
        "final_hp": final.get("hp"),
        "final_board": final.get("board"),
        "final_item_bench": item_bench[keys[-1]] if keys else [],
        "augments": augment_seen,
        "level_timing": _level_timing(timeline, mech),
        "interest_short_on_save_rounds": interest_short,
        "interest_short_rounds": short_rounds,
        "hp_lost_by_stage": hp_lost,
        "timeline": timeline,
    }


def _cell(v: Any) -> str:
    return "-" if v is None else str(v)  # 0 gold / 0 hp are real values, not "unknown"


def _level_text(x: dict[str, Any]) -> str:
    text = f"{x['level']}级@{x['reached_at']}"
    std, late = x.get("standard_round"), x.get("rounds_late")
    if not std:
        return text
    if isinstance(late, int) and late > 0:
        return text + f"(标准{std}，至少晚{late}回合)"
    if isinstance(late, int) and late < 0:
        return text + f"(标准{std}，早{-late}回合)"
    return text + f"(标准{std})"


def format_summary(summary: dict[str, Any]) -> str:
    short_rounds = summary.get("interest_short_rounds") or []
    lines = [
        f"回合数: {summary['rounds']}  最后回合: {_cell(summary.get('final_stage'))}  最后血量: {_cell(summary.get('final_hp'))}",
        f"最终阵容: {' '.join(summary.get('final_board') or []) or '未知'}",
        f"海克斯: {', '.join(a['name'] + '@' + a['seen_at'] for a in summary.get('augments') or []) or '无记录'}",
        "升级时间: " + (", ".join(_level_text(x) for x in summary["level_timing"]) or "无记录"),
        f"建议存钱时离利息上限差的利息(第3阶段起，累计): {summary['interest_short_on_save_rounds']}"
        + (f"（{', '.join(short_rounds)}）" if short_rounds else ""),
        "各阶段掉血: " + (", ".join(f"阶段{s}: {v}" for s, v in sorted(summary["hp_lost_by_stage"].items())) or "无记录"),
        "",
        "回合  血量  金币  等级  建议",
    ]
    for row in summary["timeline"]:
        lines.append(f"{row['stage']:>4}  {_cell(row.get('hp')):>4}  {_cell(row.get('gold')):>4}  {_cell(row.get('level')):>4}  {row.get('advice') or ''}")
    return "\n".join(lines)


REVIEW_SYSTEM = """You are a Teamfight Tactics coach reviewing one of your student's games.
The data is a JSON summary of the game log. A round is only logged when the assistant
analysed it, so gaps between rounds are normal.
- timeline: per round HP, gold, level, streak (+N win streak, -N loss streak), board
  (units with stars and [items]), comp (the assistant's target comp), style (econ
  line: standard, fast8, fast9, reroll1/2/3), econ (the assistant's econ call: save,
  level, roll, level_and_roll, slow_roll, all_in, hold) and advice (its headline).
- level_timing: for each level, reached_at is the first logged round at that level and
  standard_round is when a standard line gets there. rounds_late > 0 means at least
  that many rounds late (below_at lists logged rounds still under that level), <= 0
  means on time or early, null means the log cannot tell. Reroll styles stay at a low
  level on purpose, so late levels on a reroll line are not a mistake.
- interest_short_on_save_rounds: from stage 3 on, only on rounds where the assistant
  said "save", how much interest the player was short of the cap. Rounds where the
  assistant said level or roll are not econ mistakes. Right after a planned level or
  roll a small number is normal.
- augments (with the round first seen), final_item_bench, hp_lost_by_stage.
Write a short post-game review in Simplified Chinese: up to 3 things done well, up to 3
biggest mistakes, and up to 3 concrete habits to practice next game. Only criticise
what the data shows (econ, level timing, rolldowns, comp choice, items, augments); if a
category has no data, skip it rather than guess. Be specific to the rounds in the
data. Do not use em dashes. Keep it under 400 Chinese characters."""


def llm_review(llm: Any, model: str, effort: str, summary: dict[str, Any]) -> str:
    from .advisor.rules import sanitize
    from .llm import text_block

    payload = json.dumps({k: v for k, v in summary.items()}, ensure_ascii=False, sort_keys=True)
    text = llm.text(model=model, effort=effort, system=REVIEW_SYSTEM, content=[text_block(payload)], purpose="review")
    # Player-facing text never contains em dashes, whatever the model wrote.
    return sanitize(text, keep_newlines=True)
