"""Shared fixtures and builders for engine / data tests (fully offline).

Fixtures: ``set_data`` (bundled synthetic TFTSet99), ``mech`` (bundled
mechanics), ``make_obs`` / ``make_state`` / ``make_unit`` factories.
The plain builder functions can also be imported: ``from .conftest import obs``.
"""

from __future__ import annotations

import errno
import ipaddress
import os
import socket
import urllib.request
from dataclasses import dataclass, field
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


# Real Set 18 data: the bundled CommunityDragon snapshot (zh_cn names, en_us
# for English aliases) and the bundled Set 18 comp library.
_S18: Optional[SetData] = None
_S18_COMPS: Optional[list] = None


def s18_set_data() -> SetData:
    global _S18
    if _S18 is None:
        from tft_advisor.data.setdata import bundled_snapshot

        _S18 = SetData.from_cdragon(
            bundled_snapshot("zh_cn"), bundled_snapshot("en_us"), 18, source="bundled-snapshot"
        )
    return _S18


def s18_comps() -> list:
    global _S18_COMPS
    if _S18_COMPS is None:
        from tft_advisor.data.comps import load_comps

        _S18_COMPS = load_comps(None, s18_set_data())
    return _S18_COMPS


def s18_unit(name: str, star: int = 1, items: Iterable[str] = (), row: Optional[int] = None, col: Optional[int] = None) -> Unit:
    """A resolved Set 18 unit; item names may be English or Chinese."""
    sd = s18_set_data()
    champ = sd.resolve_champion(name)
    assert champ is not None, name
    held = []
    for raw in items:
        it = sd.resolve_item(raw)
        assert it is not None, raw
        held.append(it.name)
    return Unit(
        api_name=champ.api_name, name=champ.name, cost=champ.cost, star=star, items=held, row=row, col=col,
        traits=list(champ.traits),
    )


def s18_state(stage: str, board: Iterable[Any] = (), bench: Iterable[Any] = (), items: Iterable[str] = (), **fields: Any) -> GameState:
    """Set 18 GameState. Units are names or tuples ``(name, star, items, row, col)``;
    fielded units without a row fill the back rows."""
    sd = s18_set_data()

    def build(u: Any, i: int, on_board: bool) -> Unit:
        if isinstance(u, Unit):
            return u
        if isinstance(u, tuple):
            name, star = u[0], (u[1] if len(u) > 1 else 1)
            its = u[2] if len(u) > 2 else ()
            row = u[3] if len(u) > 3 else None
            col = u[4] if len(u) > 4 else None
        else:
            name, star, its, row, col = u, 1, (), None, None
        if on_board and row is None:
            row, col = 3 - (i // 7) % 4, i % 7
        return s18_unit(name, star, its, row if on_board else None, col if on_board else i)

    st = GameState(stage=StageRound.parse(stage), screen_type=ScreenType.PLANNING, **fields)
    st.board = [build(u, i, True) for i, u in enumerate(board)]
    st.bench = [build(u, i, False) for i, u in enumerate(bench)]
    st.item_bench = [sd.resolve_item(x).name if sd.resolve_item(x) else x for x in items]
    if st.board:
        st.field_age["board"] = 1.0
    return st


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
# Isolation: every test runs as a fresh, offline user with no game running
# ---------------------------------------------------------------------------

# The real Riot Live Client Data API port (https://127.0.0.1:2999). The suite
# runs on the user's gaming PC, so a live match must not change any outcome.
LIVECLIENT_PORT = 2999
_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def _is_local_host(host: Any) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    host = str(host)
    if host.lower() == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False  # a host name: resolving it is already egress intent
    mapped = getattr(ip, "ipv4_mapped", None)
    ip = mapped or ip
    return ip.is_loopback or ip.is_unspecified


@dataclass
class Isolation:
    """What the autouse ``isolation`` fixture set up (tests may inspect it)."""

    home: Any
    egress: list = field(default_factory=list)  # non-loopback TCP connects that were blocked


@pytest.fixture(autouse=True, name="isolation")
def _isolation(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    """Hermetic by construction, not by convention.

    * HOME / USERPROFILE / APPDATA / LOCALAPPDATA and the cwd point into a fresh
      temp dir, so ``~/.tft_advisor`` (game logs, review history, token, config)
      and ``~/.config/anthropic`` are never the user's real ones.
    * No ANTHROPIC_* (real API key, base URL), TFT_ADVISOR_* (config path, HUD
      layout, debug) or proxy variables leak in; system (registry) proxies are
      ignored too, so a local proxy cannot hide real egress from the guard.
    * TCP connects to non-loopback addresses fail and fail the test; the Live
      Client port is always refused, as if no match were running.
    """
    home = tmp_path_factory.mktemp("home")
    for var in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        monkeypatch.setenv(var, str(home))
    for var in list(os.environ):
        if var.startswith(("ANTHROPIC_", "TFT_ADVISOR_")) or var in _PROXY_VARS:
            monkeypatch.delenv(var, raising=False)
    for name in ("getproxies_registry", "getproxies_macosx_sysconf"):  # Windows / macOS system proxy
        if hasattr(urllib.request, name):
            monkeypatch.setattr(urllib.request, name, lambda: {})
    monkeypatch.chdir(home)

    state = Isolation(home=home)
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def blocked(sock: socket.socket, address: Any) -> Optional[OSError]:
        if sock.family not in (socket.AF_INET, socket.AF_INET6) or not isinstance(address, tuple) or len(address) < 2:
            return None
        host, port = address[0], address[1]
        if port == LIVECLIENT_PORT and _is_local_host(host):
            return ConnectionRefusedError(errno.ECONNREFUSED, "tests: no game is running (Live Client port refused)")
        if sock.type == socket.SOCK_STREAM and not _is_local_host(host):
            state.egress.append(address)
            return OSError(errno.ENETUNREACH, f"tests: network egress blocked by tests/conftest.py ({host}:{port})")
        return None  # loopback, or UDP connect (sends nothing, e.g. LAN IP guess)

    def connect(sock: socket.socket, address: Any) -> None:
        err = blocked(sock, address)
        if err is not None:
            raise err
        return real_connect(sock, address)

    def connect_ex(sock: socket.socket, address: Any) -> int:
        err = blocked(sock, address)
        return err.errno if err is not None else real_connect_ex(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    yield state
    if state.egress:
        pytest.fail(f"test tried to reach the network: {state.egress}", pytrace=False)


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
