"""Shared fixtures and builders for engine / data tests (fully offline).

Fixtures: ``set_data`` (bundled synthetic TFTSet99), ``mech`` (bundled
mechanics), ``make_obs`` / ``make_state`` / ``make_unit`` factories.
The plain builder functions can also be imported: ``from .conftest import obs``.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import pytest

from tft_advisor.data.mechanics import Mechanics, load_mechanics
from tft_advisor.data.setdata import SetData, bundled_sample
from tft_advisor.models import (
    GameState,
    Observation,
    PlayerObs,
    ScreenObservation,
    ScreenType,
    ShopSlot,
    StageRound,
    TraitObs,
    Unit,
    UnitObs,
)

_SET_DATA: Optional[SetData] = None


def sample_set_data() -> SetData:
    global _SET_DATA
    if _SET_DATA is None:
        _SET_DATA = SetData.from_cdragon(bundled_sample())
    return _SET_DATA


# Chinese-locale fixture: the sample export with display names translated, the
# way CommunityDragon's zh_cn.json differs from en_us.json (same apiNames).
CHAMP_ZH = {
    "Garen": "盖伦", "Graves": "格雷福斯", "Vayne": "薇恩", "Warwick": "沃里克", "Darius": "德莱厄斯",
    "Lucian": "卢锡安", "Ahri": "阿狸", "Braum": "布隆", "Shen": "慎", "Pyke": "派克", "Katarina": "卡特琳娜",
    "Ashe": "艾希", "Morgana": "莫甘娜", "Aatrox": "亚托克斯", "Volibear": "沃利贝尔", "Draven": "德莱文",
    "Gnar": "纳尔", "Akali": "阿卡丽", "Miss Fortune": "厄运小姐", "Kayle": "凯尔",
    "Training Dummy": "训练假人", "Armory Key": "军械库钥匙",
}
TRAIT_ZH = {
    "Knight": "骑士", "Noble": "贵族", "Gunslinger": "枪手", "Pirate": "海盗", "Ranger": "游侠", "Wild": "狂野",
    "Brawler": "斗士", "Imperial": "帝国", "Sorcerer": "法师", "Glacial": "冰川", "Guardian": "守护者",
    "Blademaster": "剑士", "Ninja": "忍者", "Assassin": "刺客", "Demon": "恶魔",
}
ITEM_ZH = {
    "B.F. Sword": "暴风大剑", "Recurve Bow": "反曲之弓", "Needlessly Large Rod": "无用大棒",
    "Tear of the Goddess": "女神之泪", "Chain Vest": "锁子甲", "Negatron Cloak": "负极斗篷",
    "Giant's Belt": "巨人腰带", "Sparring Gloves": "拳套", "Spatula": "金铲铲", "Frying Pan": "金锅锅",
    "Deathblade": "死亡之刃", "Infinity Edge": "无尽之刃", "Giant Slayer": "巨人杀手",
    "Rabadon's Deathcap": "灭世者的死亡之帽", "Warmog's Armor": "狂徒铠甲", "Bramble Vest": "棘刺背心",
    "Blue Buff": "蓝霸符", "Guinsoo's Rageblade": "鬼索的狂暴之刃", "Knight Emblem": "骑士纹章",
}


def chinese_sample() -> dict[str, Any]:
    """The bundled sample export with zh_cn display names."""
    import copy

    data = copy.deepcopy(bundled_sample())
    for entry in data["setData"]:
        for c in entry.get("champions", []):
            c["name"] = CHAMP_ZH.get(c["name"], c["name"])
            c["traits"] = [TRAIT_ZH.get(t, t) for t in c.get("traits") or []]
        for t in entry.get("traits", []):
            t["name"] = TRAIT_ZH.get(t["name"], t["name"])
    for it in data["items"]:
        it["name"] = ITEM_ZH.get(it["name"], it["name"])
    return data


_ZH_SET_DATA: Optional[SetData] = None


def chinese_set_data() -> SetData:
    global _ZH_SET_DATA
    if _ZH_SET_DATA is None:
        _ZH_SET_DATA = SetData.from_cdragon(chinese_sample(), bundled_sample())
    return _ZH_SET_DATA


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def uo(name: str, star: int = 1, items: Iterable[str] = (), row: Optional[int] = None, col: Optional[int] = None) -> UnitObs:
    """UnitObs shorthand."""
    return UnitObs(name=name, star=star, items=list(items), row=row, col=col)


def shop(*names: Optional[str]) -> list[ShopSlot]:
    sd = sample_set_data()
    out = []
    for n in names:
        champ = sd.resolve_champion(n) if n else None
        out.append(ShopSlot(name=n, cost=champ.cost if champ else None))
    return out


def players(*rows: tuple[str, Optional[int]], me: Optional[str] = None) -> list[PlayerObs]:
    return [PlayerObs(name=n, hp=hp, is_self=(n == me)) for n, hp in rows]


def obs(purpose: str = "auto", captured_at: float = 0.0, **screen: Any) -> Observation:
    """Observation with a ScreenObservation built from keyword fields.

    ``board``/``bench`` accept UnitObs or plain names; ``shop`` accepts
    ShopSlot or plain names; ``screen_type`` defaults to planning.
    """
    screen.setdefault("screen_type", ScreenType.PLANNING)
    for key in ("board", "bench"):
        if screen.get(key) is not None:
            screen[key] = [u if isinstance(u, UnitObs) else uo(u) for u in screen[key]]
    if screen.get("shop") is not None:
        screen["shop"] = [s if isinstance(s, ShopSlot) else shop(s)[0] for s in screen["shop"]]
    return Observation(screen=ScreenObservation(**screen), purpose=purpose, source="test", captured_at=captured_at)


def make_unit(
    name: str,
    star: int = 1,
    items: Iterable[str] = (),
    row: Optional[int] = None,
    col: Optional[int] = None,
    set_data: Optional[SetData] = None,
) -> Unit:
    """Resolved Unit (unresolvable names become '?name' like the tracker does)."""
    sd = set_data or sample_set_data()
    champ = sd.resolve_champion(name)
    if champ is None:
        return Unit(api_name=f"?{name}", name=name, star=star, items=list(items), row=row, col=col)
    return Unit(
        api_name=champ.api_name,
        name=champ.name,
        cost=champ.cost,
        star=star,
        items=list(items),
        row=row,
        col=col,
        traits=list(champ.traits),
    )


def _as_unit(u: Any, on_board: bool, i: int) -> Unit:
    if isinstance(u, Unit):
        return u
    if isinstance(u, tuple):
        name, star = u[0], (u[1] if len(u) > 1 else 1)
        items = u[2] if len(u) > 2 else ()
        row = u[3] if len(u) > 3 else (None if not on_board else 3 - (i // 7) % 4)
        col = u[4] if len(u) > 4 else i % 7
        return make_unit(name, star, items, row if on_board else None, col)
    return make_unit(str(u), 1, (), (3 - (i // 7) % 4) if on_board else None, i % 7 if on_board else i)


def make_state(
    stage: Optional[str] = None,
    board: Iterable[Any] = (),
    bench: Iterable[Any] = (),
    shop_names: Optional[Iterable[Optional[str]]] = None,
    item_bench: Iterable[str] = (),
    traits: Iterable[TraitObs] = (),
    **fields: Any,
) -> GameState:
    """GameState with resolved units.

    Units are names, ``Unit`` objects or tuples ``(name, star, items, row, col)``.
    """
    sd = sample_set_data()
    st = GameState(stage=StageRound.parse(stage) if stage else None, screen_type=ScreenType.PLANNING, **fields)
    st.board = [_as_unit(u, True, i) for i, u in enumerate(board)]
    st.bench = [_as_unit(u, False, i) for i, u in enumerate(bench)]
    st.item_bench = list(item_bench)
    st.traits = list(traits)
    if shop_names is not None:
        names = list(shop_names)
        st.shop = shop(*names)
        st.shop_units = [make_unit(n) if n and sd.resolve_champion(n) else None for n in names]
        for i, n in enumerate(names):
            if n and st.shop_units[i] is None:
                st.shop_units[i] = Unit(api_name=f"?{n}", name=n)
    if st.board:
        st.field_age["board"] = 1.0
    return st


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def set_data() -> SetData:
    return sample_set_data()


@pytest.fixture(scope="session")
def zh_set_data() -> SetData:
    return chinese_set_data()


@pytest.fixture(scope="session")
def mech() -> Mechanics:
    return load_mechanics()


@pytest.fixture()
def make_obs():
    return obs


@pytest.fixture(name="make_state")
def make_state_fixture():
    return make_state


@pytest.fixture(name="make_unit")
def make_unit_fixture():
    return make_unit


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float = 1.0) -> float:
        self.t += dt
        return self.t


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()
