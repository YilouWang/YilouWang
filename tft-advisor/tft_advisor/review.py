"""Post-game review from the JSONL game log written by the app.

``summarize_log`` is deterministic (no API): per-round HP / gold / level
timeline, level timing vs the standard curve, interest lost, HP lost per stage.
``llm_review`` asks Claude for a short Chinese coaching summary on top of it.
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


def summarize_log(records: list[dict[str, Any]], mech: Mechanics) -> dict[str, Any]:
    """Collapse the log to one row per round (last record of each round wins)."""
    rounds: dict[tuple[int, int], dict[str, Any]] = {}
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
        advice = rec.get("advice") if isinstance(rec.get("advice"), dict) else {}
        rounds[sr.key] = {
            "stage": f"{sr.stage}-{sr.round}",
            "hp": state.get("hp"),
            "gold": state.get("gold"),
            "level": state.get("level"),
            "board": [f"{u.get('name')}{'★' * int(u.get('star') or 1)}" for u in state.get("board") or [] if isinstance(u, dict)],
            "advice": advice.get("headline"),
            "econ": econ.get("recommendation"),
        }
    timeline = [rounds[k] for k in sorted(rounds)]

    level_timing: list[dict[str, Any]] = []
    seen_levels: set[int] = set()
    for row in timeline:
        lvl = row.get("level")
        if isinstance(lvl, int) and lvl not in seen_levels:
            seen_levels.add(lvl)
            sr = StageRound.parse(row["stage"])
            std = mech.standard_level_at(sr)
            level_timing.append({"level": lvl, "reached_at": row["stage"], "standard_level_then": std})

    interest_lost = 0
    for row in timeline:
        g = row.get("gold")
        sr = StageRound.parse(row["stage"])
        if isinstance(g, int) and sr is not None and sr.stage >= 2 and g < 50:
            # Gold short of the next interest step is not "lost"; count only
            # how far below the 50 cap the player sat in stages 2-3 (econ phase).
            if sr.stage <= 3:
                interest_lost += mech.interest_cap - mech.interest(g)

    hp_by_stage: dict[int, list[int]] = {}
    for row in timeline:
        sr = StageRound.parse(row["stage"])
        if sr is not None and isinstance(row.get("hp"), int):
            hp_by_stage.setdefault(sr.stage, []).append(row["hp"])
    hp_lost = {s: (v[0] - v[-1]) for s, v in hp_by_stage.items() if len(v) >= 2}

    final = timeline[-1] if timeline else {}
    return {
        "rounds": len(timeline),
        "final_stage": final.get("stage"),
        "final_hp": final.get("hp"),
        "final_board": final.get("board"),
        "level_timing": level_timing,
        "interest_missed_stage2_3": interest_lost,
        "hp_lost_by_stage": hp_lost,
        "timeline": timeline,
    }


def _cell(v: Any) -> str:
    return "-" if v is None else str(v)  # 0 gold / 0 hp are real values, not "unknown"


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"回合数: {summary['rounds']}  最后回合: {_cell(summary.get('final_stage'))}  最后血量: {_cell(summary.get('final_hp'))}",
        f"最终阵容: {' '.join(summary.get('final_board') or []) or '未知'}",
        "升级时间: "
        + (", ".join(f"{x['level']}级@{x['reached_at']}" + (f"(标准{x['standard_level_then']})" if x.get("standard_level_then") else "") for x in summary["level_timing"]) or "无记录"),
        f"第2-3阶段少拿的利息(累计): {summary['interest_missed_stage2_3']}",
        "各阶段掉血: " + (", ".join(f"阶段{s}: {v}" for s, v in sorted(summary["hp_lost_by_stage"].items())) or "无记录"),
        "",
        "回合  血量  金币  等级  建议",
    ]
    for row in summary["timeline"]:
        lines.append(f"{row['stage']:>4}  {_cell(row.get('hp')):>4}  {_cell(row.get('gold')):>4}  {_cell(row.get('level')):>4}  {row.get('advice') or ''}")
    return "\n".join(lines)


REVIEW_SYSTEM = """You are a Teamfight Tactics coach reviewing one of your student's games.
You get a per-round timeline (HP, gold, level, board, the advice the assistant gave).
Write a short post-game review in Simplified Chinese: 3 things done well, 3 biggest
mistakes (econ, leveling timing, rolldowns, comp choice, items), and 3 concrete habits
to practice next game. Be specific to the rounds in the data. Do not use em dashes.
Keep it under 350 Chinese characters."""


def llm_review(llm: Any, model: str, effort: str, summary: dict[str, Any]) -> str:
    from .advisor.rules import sanitize
    from .llm import text_block

    payload = json.dumps({k: v for k, v in summary.items()}, ensure_ascii=False, sort_keys=True)
    text = llm.text(model=model, effort=effort, system=REVIEW_SYSTEM, content=[text_block(payload)], purpose="review")
    # Player-facing text never contains em dashes, whatever the model wrote.
    return sanitize(text, keep_newlines=True)
