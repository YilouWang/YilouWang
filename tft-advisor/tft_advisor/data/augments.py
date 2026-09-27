"""Static augment reference for the current set (names, effects, rarity, category).

Bundled as ``data/bundled/augments_set<N>.json`` (built by
``tools/build_augments.py`` from Data Dragon + MetaTFT lookup data). The
optional ``meta_tier`` is a snapshot of a public tier list at build time:
static, pre-game information, refreshed with the tool.
"""

from __future__ import annotations

import difflib
import json
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


def pick_augment(
    choices: list[str], data: AugmentData, stage: Optional[int], hp: Optional[int], comp_traits: list[str]
) -> Optional[tuple[str, str]]:
    """(choice as displayed, short Chinese reason) for the best of the offered augments.

    Heuristic only (Claude refines it): a tier-list S augment first, then
    economy / items early and combat later, and trait augments that fit the
    current comp.
    """
    best: Optional[tuple[float, str, str]] = None
    for i, raw in enumerate(choices):
        a = data.lookup(raw)
        if a is None:
            continue
        score = 0.0
        why: list[str] = []
        if a.meta_tier == "S":
            score += 3
            why.append("当前版本强势")
        cats = set(a.category)
        early = stage is not None and stage <= 2
        healthy = hp is None or hp >= 60
        if "经济" in cats:
            score += 1.2 if early and healthy else (-0.8 if (stage or 0) >= 4 else 0.2)
            why.append("经济")
        if "装备" in cats or "纹章" in cats:
            score += 1.0
            why.append("给装备")
        if "搜牌" in cats and (stage or 0) >= 3:
            score += 0.5
            why.append("帮助搜牌")
        if "羁绊" in cats:
            text = f"{a.name}{a.name_en}{a.desc}{a.desc_en}"
            if any(t and t in text for t in comp_traits):
                score += 1.5
                why.append("契合当前羁绊")
            else:
                score -= 1.0
        if not cats and (stage or 0) >= 4:
            score += 0.5  # uncategorized augments are mostly combat power
            why.append("战力")
        score -= i * 0.01  # stable order on ties
        reason = "、".join(dict.fromkeys(why)) or a.rarity_zh or "综合"
        if best is None or score > best[0]:
            best = (score, raw, reason)
    return (best[1], best[2]) if best else None
