"""Game rules that change between sets/patches: shop odds, pool sizes, XP, income.

Defaults are in ``data/bundled/mechanics.toml``. Riot tweaks these numbers from
patch to patch, so a user file (``[data].mechanics_file``) can override any key
without touching code.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Optional

from ..models import StageRound
from .textio import read_user_data

_BUNDLED = "mechanics.toml"


@dataclass(frozen=True)
class Mechanics:
    shop_odds: dict[int, tuple[float, float, float, float, float]]
    pool_size: dict[int, int]
    xp_to_level: dict[int, int]
    streak_bonus: tuple[tuple[int, int], ...]
    augment_rounds: tuple[str, ...]
    stage_damage: dict[int, int]
    standard_levels: dict[str, int]
    xp_per_buy: int = 4
    buy_xp_cost: int = 4
    roll_cost: int = 2
    interest_step: int = 10
    interest_cap: int = 5
    base_income: int = 5
    pvp_win_gold: int = 1
    passive_xp_per_round: int = 2
    bench_size: int = 9
    shop_slots: int = 5
    wisp_every: int = 0  # 1 shop in N hides one champion slot under a Wisp (0 = none)
    max_level: int = 10
    carousel_round: int = 4
    pve_round: int = 7  # x-7 is PvE from stage 2 on; every round of stage 1 is PvE/carousel
    patch: str = ""
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # ---- income -------------------------------------------------------------
    def interest(self, gold: int) -> int:
        if gold <= 0:
            return 0
        return min(gold // self.interest_step, self.interest_cap)

    def streak_gold(self, streak: Optional[int]) -> int:
        if not streak:
            return 0
        n = abs(streak)
        bonus = 0
        for min_len, gold in self.streak_bonus:
            if n >= min_len:
                bonus = gold
        return bonus

    def income(self, gold: int, streak: Optional[int], won: Optional[bool] = None) -> int:
        """Gold gained at the start of the next round (after this round's fight).

        ``won`` adds PvP win gold when known. ``streak`` is the streak *after*
        the fight if known; callers usually pass the current streak.
        """
        total = self.base_income + self.interest(gold) + self.streak_gold(streak)
        if won:
            total += self.pvp_win_gold
        return total

    # ---- experience ---------------------------------------------------------
    def xp_needed(self, level: int) -> Optional[int]:
        return self.xp_to_level.get(level)

    def xp_to_next(self, level: Optional[int], xp_current: Optional[int]) -> Optional[int]:
        if level is None or level >= self.max_level:
            return None
        need = self.xp_to_level.get(level)
        if need is None:
            return None
        return max(0, need - (xp_current or 0))

    def gold_to_reach(self, level: Optional[int], xp_current: Optional[int], target_level: int) -> Optional[int]:
        """Gold spent on XP buys to go from (level, xp_current) to target_level now."""
        if level is None:
            return None
        if target_level <= level:
            return 0
        if target_level > self.max_level:
            return None
        xp = 0
        cur = xp_current or 0
        for lvl in range(level, target_level):
            need = self.xp_to_level.get(lvl)
            if need is None:
                return None
            xp += max(0, need - cur)
            cur = 0
        buys = -(-xp // self.xp_per_buy)  # ceil
        return buys * self.buy_xp_cost

    # ---- rounds -------------------------------------------------------------
    def is_pve(self, sr: Optional[StageRound]) -> bool:
        if sr is None:
            return False
        if sr.stage == 1:
            return True
        return sr.round == self.pve_round

    def is_carousel(self, sr: Optional[StageRound]) -> bool:
        if sr is None:
            return False
        if sr.stage == 1:
            return sr.round == 1
        return sr.round == self.carousel_round

    def is_augment(self, sr: Optional[StageRound]) -> bool:
        return sr is not None and f"{sr.stage}-{sr.round}" in self.augment_rounds

    def standard_level_at(self, sr: Optional[StageRound]) -> Optional[int]:
        """Level a standard line should have reached by this round."""
        if sr is None:
            return None
        best: Optional[int] = None
        best_key: tuple[int, int] = (0, 0)
        for key, lvl in self.standard_levels.items():
            parsed = StageRound.parse(key)
            if parsed and parsed.key <= sr.key and parsed.key >= best_key:
                best, best_key = lvl, parsed.key
        return best

    def shop_schedule(self, wisp_every: Optional[int] = None) -> tuple[int, ...]:
        """Champion slots per consecutive shop, repeating: with a Wisp in
        every other shop (5, 4); a Wisp in every shop (4,); no Wisps (5,)."""
        n = self.wisp_every if wisp_every is None else wisp_every
        slots = max(1, self.shop_slots)
        if n is None or n <= 0 or slots <= 1:
            return (slots,)
        return tuple([slots] * (n - 1) + [slots - 1])

    def odds(self, level: int, cost: int) -> float:
        lvl = min(max(level, 1), self.max_level)
        row = self.shop_odds.get(lvl)
        if row is None or not 1 <= cost <= len(row):
            return 0.0
        return row[cost - 1]


def _int_keys(d: dict[str, Any]) -> dict[int, Any]:
    return {int(k): v for k, v in d.items()}


def _from_dict(raw: dict[str, Any]) -> Mechanics:
    odds = {int(k): tuple(float(x) / (100.0 if sum(v) > 1.5 else 1.0) for x in v) for k, v in raw["shop_odds"].items()}
    for lvl, row in odds.items():
        if len(row) != 5 or abs(sum(row) - 1.0) > 0.02:
            raise ValueError(f"shop_odds for level {lvl} must have 5 entries summing to 100")
    scalars = {
        k: raw[k]
        for k in (
            "xp_per_buy",
            "buy_xp_cost",
            "roll_cost",
            "interest_step",
            "interest_cap",
            "base_income",
            "pvp_win_gold",
            "passive_xp_per_round",
            "bench_size",
            "shop_slots",
            "wisp_every",
            "max_level",
            "carousel_round",
            "pve_round",
            "patch",
            "notes",
        )
        if k in raw
    }
    return Mechanics(
        shop_odds=odds,  # type: ignore[arg-type]
        pool_size={int(k): int(v) for k, v in raw["pool_size"].items()},
        xp_to_level={int(k): int(v) for k, v in raw["xp_to_level"].items()},
        streak_bonus=tuple(sorted((int(a), int(b)) for a, b in raw["streak_bonus"])),
        augment_rounds=tuple(raw.get("augment_rounds", ())),
        stage_damage={int(k): int(v) for k, v in raw.get("stage_damage", {}).items()},
        standard_levels={str(k): int(v) for k, v in raw.get("standard_levels", {}).items()},
        extra=dict(raw.get("extra", {})),
        **scalars,
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def bundled_mechanics_raw() -> dict[str, Any]:
    text = resources.files("tft_advisor.data").joinpath("bundled", _BUNDLED).read_text(encoding="utf-8")
    return tomllib.loads(text)


def load_mechanics(override_path: Optional[str] = None) -> Mechanics:
    """Bundled mechanics, with a user TOML (UTF-8, BOM allowed) merged over them."""
    raw = bundled_mechanics_raw()
    if override_path:
        p = Path(override_path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"找不到机制文件: {p}")
        user = read_user_data(p, "机制文件", fmt="toml")
        raw = _deep_merge(raw, user)
        try:
            return _from_dict(raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"机制文件 {p} 内容有误: {exc}") from None
    return _from_dict(raw)
