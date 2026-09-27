"""Static augment reference for the current set (names, effects, rarity, category).

Bundled as ``data/bundled/augments_set<N>.json`` (built by
``tools/build_augments.py`` from Data Dragon + MetaTFT lookup data). The
optional ``meta_tier`` is a snapshot of a public tier list at build time:
static, pre-game information, refreshed with the tool.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, Optional

from .setdata import normalize_name


@dataclass
class Augment:
    api_name: str
    name: str
    name_en: str
    desc: str = ""
    desc_en: str = ""
    rarity: str = ""  # Silver | Gold | Prismatic
    category: list[str] = field(default_factory=list)  # 经济 / 装备 / 羁绊 / 搜牌 / ...
    meta_tier: str = ""  # "S" when a public tier list rated it S at build time

    @property
    def rarity_zh(self) -> str:
        return {"Silver": "银色", "Gold": "金色", "Prismatic": "彩色"}.get(self.rarity, "")


class AugmentData:
    def __init__(self, augments: dict[str, Augment], note: str = "") -> None:
        self.augments = augments
        self.note = note
        self._index: dict[str, str] = {}
        for tier in (lambda a: [a.name, a.name_en], lambda a: [a.api_name]):
            for a in augments.values():
                for n in tier(a):
                    key = normalize_name(n) + ("+" * n.count("+"))  # keep "+" / "++" variants apart
                    if key.strip("+") and key not in self._index:
                        self._index[key] = a.api_name

    def __len__(self) -> int:
        return len(self.augments)

    @classmethod
    def load(cls, set_number: int) -> "AugmentData":
        try:
            text = resources.files("tft_advisor.data").joinpath("bundled", f"augments_set{set_number}.json").read_text(encoding="utf-8")
            raw: dict[str, Any] = json.loads(text)
        except (FileNotFoundError, OSError, ValueError):
            return cls({})
        out: dict[str, Augment] = {}
        for api, a in (raw.get("augments") or {}).items():
            if not isinstance(a, dict):
                continue
            out[api] = Augment(
                api_name=api,
                name=str(a.get("name") or api),
                name_en=str(a.get("name_en") or a.get("name") or api),
                desc=str(a.get("desc") or ""),
                desc_en=str(a.get("desc_en") or ""),
                rarity=str(a.get("rarity") or ""),
                category=[str(c) for c in a.get("category") or []],
                meta_tier=str(a.get("meta_tier") or ""),
            )
        return cls(out, note=str((raw.get("meta") or {}).get("note") or ""))

    def lookup(self, name: Optional[str]) -> Optional[Augment]:
        if not name or not self._index:
            return None
        key = normalize_name(name) + ("+" * str(name).count("+"))
        if key in self._index:
            return self.augments[self._index[key]]
        if len(key.strip("+")) < 2:
            return None
        cjk = any("一" <= ch <= "鿿" for ch in key)
        same_plus = [k for k in self._index if k.count("+") == key.count("+")]
        match = difflib.get_close_matches(key, same_plus, n=1, cutoff=0.86 if cjk else 0.82)
        return self.augments[self._index[match[0]]] if match else None


# Effect text patterns (Chinese client text, the English text as a fallback).
_LEVEL_TRIGGER = re.compile(r"(?:到达|在)\s*((?:\d+\s*[、,，和及与]\s*)*\d+)\s*级")
_LEVEL_TRIGGER_EN = re.compile(r"(?:reach|at)\s+levels?\s+((?:\d+\s*(?:,|and|&)?\s*)+)", re.IGNORECASE)
_ON_LEVEL_UP = re.compile(r"升级时|level up", re.IGNORECASE)
_NEXT_AUGMENT = re.compile(r"下一个强化符文|next augment", re.IGNORECASE)
_NEXT_UPGRADE = re.compile(r"提高|higher", re.IGNORECASE)
_DELAY = re.compile(r"(\d+)\s*(?:场玩家对战回合|场玩家对战|个回合)(?:后|来孵化)")
_DELAY_EN = re.compile(r"after\s+(\d+)\s+(?:player\s+combats?|rounds?)|(\d+)\s+rounds?\s+to\s+hatch", re.IGNORECASE)
_AT_ROUND = re.compile(r"(?:阶段|在)\s*(\d)\s*-\s*(\d)")
_REROLL_WORDS = re.compile(r"刷新|复制器|重随|reroll|duplicator", re.IGNORECASE)
_COMBAT_WORDS = re.compile(
    r"生命值|伤害增幅|攻击速度|物理加成|法术加成|护甲|魔法抗性|魔抗|全能吸血|暴击|伤害减免|护盾|法力回复|治疗|"
    r"真实伤害|魔法伤害|眩晕|体型变大"
)
_ITEM_WORDS = re.compile(r"获得[^。]*?(?:装备|锻造器|纹章|神器|光明武器|窃贼手套)")
_ECON_WORDS = re.compile(r"获得\d+金币|金币|经验值")
_UNIT_WORDS = re.compile(r"获得[^。]*弈子")
DEFAULT_AUGMENT_ROUNDS = ("2-1", "3-2", "4-2")
# Typical level by stage when the level was not read (a standard line).
_STAGE_LEVEL = {1: 3, 2: 4, 3: 6, 4: 7}


def _levels(text: str, text_en: str) -> list[int]:
    found = [int(n) for m in _LEVEL_TRIGGER.finditer(text) for n in re.findall(r"\d+", m.group(1))]
    if not found and text_en:
        found = [int(n) for m in _LEVEL_TRIGGER_EN.finditer(text_en) for n in re.findall(r"\d+", m.group(1))]
    return [n for n in found if 1 <= n <= 10]


def effect_kinds(a: Augment) -> list[str]:
    """Categories of an augment, checked against its effect text.

    The bundled categories come from a public tag list: ``搜牌`` is kept only
    when the effect gives rerolls or duplicators (宝宝学院 is team AD/AP), and
    an augment without a category gets one from its effect words
    (战力 for stats, 装备, 经济, 搜牌, 弈子).
    """
    text = f"{a.desc} {a.desc_en}"
    cats = [c for c in a.category if c != "搜牌" or _REROLL_WORDS.search(text)]
    if cats:
        return cats
    out: list[str] = []
    if _COMBAT_WORDS.search(a.desc or a.desc_en):
        out.append("战力")
    if _ITEM_WORDS.search(a.desc):
        out.append("装备")
    if _REROLL_WORDS.search(text):
        out.append("搜牌")
    if _ECON_WORDS.search(a.desc):
        out.append("经济")
    if not out and _UNIT_WORDS.search(a.desc):
        out.append("弈子")
    return out


def _timing_penalty(
    a: Augment,
    stage: Optional[int],
    round_no: Optional[int],
    level: Optional[int],
    hp: Optional[int],
    augment_rounds: tuple[str, ...],
) -> float:
    """Score lost to the timing written in the effect: a level trigger the
    player already passed, a "next augment" effect on the last augment round,
    a round already past, or a payout many player combats away at low HP or
    late in the game."""
    text, text_en = a.desc or "", a.desc_en or ""
    penalty = 0.0
    lvl = level if level is not None else _STAGE_LEVEL.get(stage or 0, 8 if (stage or 0) >= 5 else None)
    levels = _levels(text, text_en)
    if levels and lvl is not None:
        ahead = [n for n in levels if n > lvl]
        penalty += 3.0 * (1 - len(ahead) / len(levels))
    if _ON_LEVEL_UP.search(text or text_en) and lvl is not None and lvl >= 8:
        penalty += 1.0 if lvl == 8 else 2.0  # one or no level up left to pay out
    # "Your next augment ...": nothing comes after the last augment round.
    rounds = [r for r in (augment_rounds or DEFAULT_AUGMENT_ROUNDS) if isinstance(r, str) and "-" in r]
    last = max((tuple(int(x) for x in r.split("-", 1)) for r in rounds if r.replace("-", "").isdigit()), default=None)
    if stage is not None and last is not None:
        now = (stage, round_no if round_no is not None else last[1] if stage == last[0] else 0)
        if now >= last and _NEXT_AUGMENT.search(text or text_en) and _NEXT_UPGRADE.search(text or text_en):
            penalty += 3.0
        for m in _AT_ROUND.finditer(text):
            if (int(m.group(1)), int(m.group(2))) <= now:
                penalty += 2.0  # the round it pays out on is already here or past
    # Delayed payouts: worth less the fewer rounds the player can afford.
    matches = list(_DELAY.finditer(text)) or list(_DELAY_EN.finditer(text_en))
    delay = max((int(next(g for g in m.groups() if g)) for m in matches), default=0)
    if delay >= 3 and stage is not None:
        if hp is not None and hp < 45:
            urgency = 1.0
        elif stage >= 4:
            urgency = 0.5
        elif stage == 3:
            urgency = 0.25
        else:
            urgency = 0.0
        first = matches[0]
        before = first.string[: first.start()]
        immediate = bool(re.search(r"获得|立刻|gain", before, re.IGNORECASE)) and "孵化" not in text
        penalty += min(4.0, 0.5 * delay) * urgency * (0.4 if immediate else 1.0)
    return penalty


def pick_augment(
    choices: list[str],
    data: AugmentData,
    stage: Optional[int],
    hp: Optional[int],
    comp_traits: list[str],
    level: Optional[int] = None,
    round_no: Optional[int] = None,
    augment_rounds: Optional[tuple[str, ...]] = None,
) -> Optional[tuple[str, str]]:
    """(choice as displayed, short Chinese reason) for the best of the offered augments.

    Heuristic only (Claude refines it): a tier-list S augment first, then
    economy / items early and combat later (economy is avoided at low HP from
    stage 3 on), and trait augments that fit the current comp. The effect's
    own timing counts (``_timing_penalty``): pass ``level``, ``round_no`` and
    ``augment_rounds`` (``Mechanics.augment_rounds``); without them the level
    is estimated from the stage and the rounds default to 2-1 / 3-2 / 4-2.
    """
    best: Optional[tuple[float, str, str]] = None
    st = stage or 0
    low = hp is not None and hp < 45
    mid = hp is not None and hp < 60
    for i, raw in enumerate(choices):
        a = data.lookup(raw)
        if a is None:
            continue
        score = 0.0
        why: list[str] = []
        if a.meta_tier == "S":
            score += 3
            why.append("当前版本强势")
        cats = set(effect_kinds(a))
        if "经济" in cats:
            if stage is not None and st <= 2 and (hp is None or hp >= 60):
                score += 1.2
            elif st >= 4:
                score -= 0.8
            elif low or (st >= 3 and mid):
                score -= 0.6  # low HP: the gold should already be going into the board
            else:
                score += 0.2
            why.append("经济")
        if "装备" in cats or "纹章" in cats:
            score += 1.0
            why.append("给装备")
        if "搜牌" in cats and st >= 3:
            score += 0.5
            why.append("帮助搜牌")
        if "羁绊" in cats:
            text = f"{a.name}{a.name_en}{a.desc}{a.desc_en}"
            if any(t and t in text for t in comp_traits):
                score += 1.5
                why.append("契合当前羁绊")
            else:
                score -= 1.0
        if "战力" in cats:
            if st >= 4:
                score += 0.6 + (0.8 if mid else 0.0)
            elif st == 3:
                score += 0.2 + (0.8 if mid else 0.0)
            elif low:
                score += 0.5
            why.append("战力")
        if "弈子" in cats:
            score += 0.5 if st >= 3 else 0.3
            why.append("给弈子")
        score -= _timing_penalty(a, stage, round_no, level, hp, tuple(augment_rounds or ()))
        score -= i * 0.01  # stable order on ties
        reason = "、".join(dict.fromkeys(why)) or "综合"
        if best is None or score > best[0]:
            best = (score, raw, reason)
    return (best[1], best[2]) if best else None
