"""Game tracker: merges noisy per-screenshot observations into one GameState.

Merge rules (every field of a ``ScreenObservation`` is optional):

* ``None`` means "not visible": the old value is kept. An empty list means
  "visible and empty" and replaces the old value.
* Numbers go through plausibility filters (gold 0..300, level 1..max_level,
  xp >= 0, hp 0..200, streak -20..20); an implausible value is dropped and the
  old one kept.
* Stage: a later stage is accepted, except a jump over a whole stage (``3-2``
  then ``5-x``), which needs a second reading of that stage (digit misreads
  are far more common than a paused advisor). An earlier stage means either a
  new game (``1-x`` while we are at ``2-1`` or later, or any earlier stage
  after a post-game screen: full reset) or a misread (ignored). A ``1-x``
  reading on a frame that contradicts stage 1 (HP below 100, level above 4)
  only resets when it is read twice in a row.
  If the same earlier stage is read twice in a row, the previous stage was the
  misread: the stage is corrected and history points after it are dropped.
* Scouting (``purpose == "scout"`` or ``viewing_own_board is False``): the
  board, bench, traits, HP, level and item bench on screen belong to the viewed
  player and are stored as an ``OpponentSnapshot``. Only the stage, the player
  list, our gold and our shop are applied to our own state, because in TFT the
  bottom HUD shop and gold always show the local player's values. Our level /
  XP / streak are NOT taken from a scouting frame (conservative: some clients
  show the viewed player's level near the board, and a wrong own level is
  worse than a slightly stale one). The board owner is the named opponent
  when the banner names a known opponent; a scout frame that the vision marks
  ``viewing_own_board=True`` without another player's name, or whose board
  is identical to ours with no known opponent named, is our own board.
* Carousel and loading frames never carry our board, bench, traits or item
  bench (the carousel ring is not a board): those fields are ignored.
* Eliminated players (HP 0 in the player list) do not count in
  ``taken_copies`` because their units go back to the pool.

Timestamps (``last_update``, ``field_age``, snapshot / history times) come
from the injected ``clock`` so tests are deterministic.
"""

from __future__ import annotations

import difflib
import threading
import time
from collections import Counter
from typing import Any, Callable, Optional

from ..data.mechanics import Mechanics
from ..data.setdata import Champion, SetData, normalize_name
from ..models import (
    GameState,
    HistoryPoint,
    Observation,
    OpponentSnapshot,
    PlayerObs,
    ScreenObservation,
    ScreenType,
    ShopSlot,
    StageRound,
    TraitObs,
    Unit,
    UnitObs,
)

UNKNOWN_PLAYER = "unknown"
MAX_PLAYERS = 8

GOLD_RANGE = (0, 300)
HP_RANGE = (0, 200)
STREAK_RANGE = (-20, 20)
XP_MAX = 200
SETTABLE_FIELDS = ("gold", "level", "hp", "xp_current", "xp_needed", "streak", "stage")
FIELD_ZH = {
    "gold": "金币",
    "level": "等级",
    "hp": "血量",
    "xp_current": "当前经验",
    "xp_needed": "升级所需经验",
    "streak": "连胜/连败",
    "stage": "回合",
}
# Screens that never show a player's board / bench / trait tracker / item bench.
NO_BOARD_SCREENS = (ScreenType.CAROUSEL, ScreenType.LOADING)


DUPLICATE_OVERLAP = 0.7  # share of copies two unit lists must have in common to be "the same board"



WISP_PREFIXES = ("wisp", "精灵", "灵火")


def is_wisp_name(name: Optional[str]) -> bool:
    """Shop slots holding a Set 18 Wisp are reported as 'Wisp: <name>' by the vision layer."""
    n = (name or "").strip().lower()
    return n.startswith(WISP_PREFIXES)

def _overlap(a: dict[str, int], b: dict[str, int]) -> float:
    """Shared copies / copies of the larger list (0..1)."""
    size = max(sum(a.values()), sum(b.values()))
    if size <= 0:
        return 0.0
    return sum(min(n, b.get(api, 0)) for api, n in a.items()) / size


def _to_int(value: Any) -> Optional[int]:
    """Lenient int conversion for vision / manual input; None if not a number."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        try:
            return int(value)  # NaN -> ValueError, inf -> OverflowError
        except (ValueError, OverflowError):
            return None
    if isinstance(value, str):
        text = value.strip().replace("+", "")
        try:
            return int(float(text))
        except (ValueError, OverflowError):
            return None
    return None


class GameTracker:
    """Thread-safe owner of the current ``GameState``."""

    def __init__(self, set_data: SetData, mech: Mechanics, clock: Callable[[], float] = time.time) -> None:
        self._sd = set_data
        self._mech = mech
        self._clock = clock
        self._lock = threading.RLock()
        self._pending_lower: Optional[tuple[int, int]] = None
        self._pending_jump: Optional[int] = None  # stage number of an unconfirmed big jump
        self._game_over = False  # a post-game screen was seen: the next lower stage is a new game
        self._games = 0
        self._state = GameState(game_id=self._new_game_id())

    # ------------------------------------------------------------------ public
    @property
    def state(self) -> GameState:
        with self._lock:
            return self._state.model_copy(deep=True)

    @property
    def set_data(self) -> SetData:
        return self._sd

    def reset(self, game_id: Optional[str] = None) -> GameState:
        with self._lock:
            self._state = GameState(game_id=game_id or self._new_game_id())
            self._pending_lower = None
            self._pending_jump = None
            self._game_over = False
            return self._state.model_copy(deep=True)

    def ingest(self, obs: Observation) -> GameState:
        with self._lock:
            now = self._clock()
            scr = obs.screen
            st = self._state

            if scr.screen_type == ScreenType.POST_GAME:
                st.screen_type = ScreenType.POST_GAME
                st.last_update = now
                self._game_over = True
                return st.model_copy(deep=True)

            stage_changed = self._merge_stage(scr.stage, now, scr)
            st = self._state  # may have been replaced by a new-game reset
            st.screen_type = scr.screen_type

            if scr.players is not None:
                self._merge_players(scr.players, now)

            if scr.screen_type in NO_BOARD_SCREENS:
                # The carousel ring / loading screen is not anyone's board.
                scr = scr.model_copy(update={"board": None, "bench": None, "traits": None, "item_bench": None})

            scouting = self._is_scouting(obs)
            if scouting:
                key = self._opponent_key(scr)
                if key is None:  # the "scouted" board is our own board
                    scouting = False
                else:
                    self._store_opponent(key, scr, now)

            # Gold and shop always belong to the local player.
            self._set_int("gold", scr.gold, GOLD_RANGE, now)
            if scr.shop is not None:
                self._merge_shop(scr.shop, now)

            if not scouting:
                self._merge_own(scr, now)

            if stage_changed:
                st.augment_choices = [] if scr.augment_choices is None else st.augment_choices
                self._append_history(now)
            else:
                self._fill_history()
            st.last_update = now
            return st.model_copy(deep=True)

    def taken_by_player(self) -> dict[str, dict[str, int]]:
        """Copies (1-star equivalents) held by each alive opponent, from the latest scout.

        The ``unknown`` snapshot (a scout frame with no readable owner) is
        dropped when it is mostly the same units as a named snapshot or as our
        own board, so the same copies are not subtracted from the pool twice.
        """
        with self._lock:
            st = self._state
            out: dict[str, dict[str, int]] = {}
            for name, snap in st.opponents.items():
                if not self._is_alive_opponent(name, snap):
                    continue
                counts = self._unit_counts([*snap.board, *snap.bench])
                if counts:
                    out[name] = counts
            unknown = out.get(UNKNOWN_PLAYER)
            if unknown:
                others = [c for n, c in out.items() if n != UNKNOWN_PLAYER]
                others.append(self._unit_counts(st.all_units()))
                if any(_overlap(unknown, c) >= DUPLICATE_OVERLAP for c in others):
                    del out[UNKNOWN_PLAYER]
            return out

    @staticmethod
    def _unit_counts(units: list[Unit]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for u in units:
            if u.api_name.startswith("?"):
                continue
            counts[u.api_name] = counts.get(u.api_name, 0) + u.copies
        return counts

    def taken_copies(self) -> dict[str, int]:
        """Copies per champion held by alive opponents (sum over players)."""
        total: dict[str, int] = {}
        for counts in self.taken_by_player().values():
            for api, n in counts.items():
                total[api] = total.get(api, 0) + n
        return total

    def set_field(self, name: str, value: Any) -> GameState:
        """Manual correction from the dashboard. Raises ``ValueError`` when invalid."""
        if name not in SETTABLE_FIELDS:
            raise ValueError(f"不能修改字段 {name}（可修改：{'、'.join(FIELD_ZH[f] for f in SETTABLE_FIELDS)}）")
        with self._lock:
            now = self._clock()
            st = self._state
            if name == "stage":
                sr = value if isinstance(value, StageRound) else StageRound.parse(str(value) if value is not None else None)
                if sr is None:
                    raise ValueError(f"回合格式不对：{value!r}（例如 3-2）")
                advanced = st.stage is None or sr.key > st.stage.key
                if st.stage is not None and sr.key < st.stage.key:
                    st.history = [h for h in st.history if (StageRound.parse(h.stage) or sr).key <= sr.key]
                st.stage = sr
                self._pending_lower = None
                self._pending_jump = None
                self._game_over = False
                st.field_age["stage"] = now
                if advanced:
                    self._append_history(now)
            else:
                if value is None or (isinstance(value, str) and not value.strip()):
                    setattr(st, name, None)
                    st.field_age.pop(name, None)
                    return st.model_copy(deep=True)
                num = _to_int(value)
                if num is None:
                    raise ValueError(f"{FIELD_ZH[name]}需要一个整数，收到 {value!r}")
                lo, hi = self._range_for(name)
                if not lo <= num <= hi:
                    raise ValueError(f"{FIELD_ZH[name]} {num} 不合理（范围 {lo} 到 {hi}）")
                setattr(st, name, num)
                st.field_age[name] = now
                if name == "hp" and st.self_name:
                    for p in st.players:
                        if p.name == st.self_name:
                            p.hp = num
            st.last_update = now
            return st.model_copy(deep=True)

    # ------------------------------------------------------------- internals
    def _new_game_id(self) -> str:
        self._games += 1
        return f"game-{int(self._clock() * 1000)}-{self._games}"

    def _range_for(self, name: str) -> tuple[int, int]:
        if name == "gold":
            return GOLD_RANGE
        if name == "level":
            return (1, self._mech.max_level)
        if name == "hp":
            return HP_RANGE
        if name == "streak":
            return STREAK_RANGE
        return (0, XP_MAX)  # xp_current / xp_needed

    def _set_int(self, name: str, value: Optional[int], rng: tuple[int, int], now: float) -> bool:
        num = _to_int(value)
        if num is None or not rng[0] <= num <= rng[1]:
            return False
        setattr(self._state, name, num)
        self._state.field_age[name] = now
        return True

    # ---- stage ----------------------------------------------------------------
    @staticmethod
    def _looks_like_stage_one(scr: Optional[ScreenObservation]) -> bool:
        """No player damage happens in stage 1 and levels stay low: a frame that
        shows HP below 100 or a high level cannot be stage 1 (stage misread)."""
        if scr is None:
            return True
        hp = _to_int(scr.hp)
        if hp is not None and hp < 100:
            return False
        lvl = _to_int(scr.level)
        if lvl is not None and lvl > 4:
            return False
        for p in scr.players or []:
            php = _to_int(p.hp)
            if php is not None and 0 <= php < 100:
                return False
        return True

    def _merge_stage(self, text: Optional[str], now: float, scr: Optional[ScreenObservation] = None) -> bool:
        """Apply a stage reading. Returns True when the stage advanced / changed."""
        sr = StageRound.parse(text)
        if sr is None:
            return False
        st = self._state
        cur = st.stage
        if cur is not None and sr.stage > cur.stage + 1 and self._pending_jump != sr.stage:
            # Skipping a whole stage between two reads is almost always a digit
            # misread ("3-2" read as "8-2"): wait for a second reading.
            self._pending_jump = sr.stage
            return False
        if cur is None or sr.key > cur.key:
            st.stage = sr
            st.field_age["stage"] = now
            self._pending_lower = None
            self._pending_jump = None
            self._game_over = False  # the game (or a spectated one) goes on
            return True
        self._pending_jump = None
        if sr.key == cur.key:
            st.field_age["stage"] = now
            self._pending_lower = None
            self._game_over = False  # a post-game frame was a misclassification
            return False
        # Lower than the current stage: a new game (1-x, corroborated by the
        # frame or read twice in a row; or anything lower after a post-game
        # screen) or a misread.
        if self._game_over and cur.key >= (2, 1):
            self.reset()
            self._state.stage = sr
            self._state.field_age["stage"] = now
            return True
        if sr.stage == 1 and cur.key >= (2, 1) and (self._looks_like_stage_one(scr) or self._pending_lower == sr.key):
            self.reset()
            self._state.stage = sr
            self._state.field_age["stage"] = now
            return True
        if self._pending_lower == sr.key:
            # Two consecutive readings agree: the current stage was a misread.
            st.history = [h for h in st.history if (StageRound.parse(h.stage) or sr).key < sr.key]
            st.stage = sr
            st.field_age["stage"] = now
            self._pending_lower = None
            return True
        self._pending_lower = sr.key
        return False

    def _append_history(self, now: float) -> None:
        st = self._state
        if st.stage is None:
            return
        key = str(st.stage)
        if st.history and st.history[-1].stage == key:
            self._fill_history()
            return
        st.history.append(HistoryPoint(stage=key, hp=st.hp, gold=st.gold, level=st.level, captured_at=now))

    def _fill_history(self) -> None:
        st = self._state
        if not st.history or st.stage is None or st.history[-1].stage != str(st.stage):
            return
        last = st.history[-1]
        if last.hp is None:
            last.hp = st.hp
        if last.gold is None:
            last.gold = st.gold
        if last.level is None:
            last.level = st.level

    # ---- players --------------------------------------------------------------
    def _known_player_names(self) -> list[str]:
        st = self._state
        names = [p.name for p in st.players]
        names += [n for n in st.opponents if n not in names and n != UNKNOWN_PLAYER]
        if st.self_name and st.self_name not in names:
            names.append(st.self_name)
        return names

    def _canonical_player(self, raw: Optional[str], candidates: Optional[list[str]] = None) -> Optional[str]:
        """Map an OCR'd player name onto a known one (names wobble between frames)."""
        if raw is None:
            return None
        name = raw.strip()
        if not name:
            return None
        known = candidates if candidates is not None else self._known_player_names()
        key = normalize_name(name)
        if not key:
            return name
        by_key = {normalize_name(k): k for k in known if normalize_name(k)}
        if key in by_key:
            return by_key[key]
        if len(key) >= 3:
            match = difflib.get_close_matches(key, list(by_key), n=1, cutoff=0.75)
            if match:
                return by_key[match[0]]
        return name

    def _merge_players(self, players: list[PlayerObs], now: float) -> None:
        st = self._state
        old = {p.name: p for p in st.players}
        known = self._known_player_names()
        # Names made only of symbols normalize to "": never match those by key.
        known_keys = {normalize_name(k): k for k in known if normalize_name(k)}
        # Exact matches first; fuzzy matching may only claim names nobody in
        # this frame matched exactly ("Player1" must not swallow "Player2").
        exact = {known_keys[normalize_name(p.name)] for p in players if normalize_name(p.name) in known_keys}
        raw_keys = {normalize_name(p.name) for p in players}
        free = [k for k in known if k not in exact and normalize_name(k) not in raw_keys]
        merged: list[PlayerObs] = []
        seen: set[str] = set()
        for p in players:
            if len(merged) >= MAX_PLAYERS:
                break  # a garbled read with more rows than a lobby can have
            key = normalize_name(p.name)
            if key and key in known_keys:
                name: Optional[str] = known_keys[key]
            elif not key:
                name = p.name.strip() or None
            else:
                name = self._canonical_player(p.name, free)
                if name in free:
                    free.remove(name)
            if not name or name in seen:
                continue
            seen.add(name)
            hp = _to_int(p.hp)
            if hp is not None and not HP_RANGE[0] <= hp <= HP_RANGE[1]:
                hp = None
            prev = old.get(name)
            if hp is None and prev is not None:
                hp = prev.hp
            is_self = bool(p.is_self) or (st.self_name is not None and name == st.self_name)
            merged.append(PlayerObs(name=name, hp=hp, is_self=is_self))
        # Keep players the frame missed (partly hidden list), up to 8 total.
        for p in st.players:
            if p.name not in seen and len(merged) < MAX_PLAYERS:
                merged.append(p.model_copy())
                seen.add(p.name)
        selves = [p for p in merged if p.is_self]
        if selves:
            # A known self wins over a row the vision flagged by mistake.
            me = next((p for p in selves if p.name == st.self_name), selves[0])
            st.self_name = me.name
            for p in merged:
                p.is_self = p.name == st.self_name
            if me.hp is not None:
                st.hp = me.hp
                st.field_age["hp"] = now
        st.players = merged
        st.field_age["players"] = now

    def _player_hp(self, name: str) -> Optional[int]:
        for p in self._state.players:
            if p.name == name:
                return p.hp
        return None

    def _is_alive_opponent(self, name: str, snap: OpponentSnapshot) -> bool:
        st = self._state
        if st.self_name and name == st.self_name:
            return False
        listed = next((p for p in st.players if p.name == name), None)
        if listed is not None and listed.hp is not None:
            return listed.hp > 0
        return snap.hp is None or snap.hp > 0

    # ---- scouting -------------------------------------------------------------
    @staticmethod
    def _is_scouting(obs: Observation) -> bool:
        return obs.purpose == "scout" or obs.screen.viewing_own_board is False

    def _opponent_key(self, scr: ScreenObservation) -> Optional[str]:
        """Snapshot key for a scouted board, or None when it is our own board.

        Order matters: a readable banner naming a known opponent wins over the
        "identical board" heuristic (early boards of 1-3 cheap units often
        match by chance). The vision layer defaults ``viewing_own_board`` to
        False on scout frames, so an explicit True without another player's
        name means the camera never left home (the player pressed the scout
        key too early).
        """
        st = self._state
        name = self._canonical_player(scr.viewed_player_name)
        if name and st.self_name and normalize_name(name) == normalize_name(st.self_name):
            return None
        opponents = {p.name for p in st.players if p.name != st.self_name and not p.is_self}
        if name and name in opponents:
            return name
        if scr.viewing_own_board is True and not name:
            return None
        if scr.board and st.board and self._same_units(scr.board, st.board):
            return None  # identical to our board: the camera never left home
        if name:
            return name
        # Fallback: a unique opponent with the HP shown on the board.
        hp = _to_int(scr.hp)
        if hp is not None:
            matches = [p.name for p in st.players if not p.is_self and p.hp == hp and p.name != st.self_name]
            if len(matches) == 1:
                return matches[0]
        return UNKNOWN_PLAYER

    def _same_units(self, seen: list[UnitObs], own: list[Unit]) -> bool:
        a = Counter((self._resolve_unit(u).api_name, max(1, u.star)) for u in seen if u.name.strip())
        b = Counter((u.api_name, max(1, u.star)) for u in own)
        return bool(a) and a == b

    def _store_opponent(self, key: str, scr: ScreenObservation, now: float) -> None:
        st = self._state
        if scr.board is None and scr.bench is None and scr.traits is None and scr.item_bench is None:
            return  # e.g. a shop-only read while the camera is away: nothing about the opponent
        old = st.opponents.get(key)
        snap = old.model_copy(deep=True) if old else OpponentSnapshot(player=key, captured_at=now)
        snap.player = key
        if scr.board is not None or scr.bench is not None or old is None:
            snap.captured_at = now
            snap.stage = str(st.stage) if st.stage else snap.stage
        hp = _to_int(scr.hp)
        if hp is not None and HP_RANGE[0] <= hp <= HP_RANGE[1]:
            snap.hp = hp
        elif (listed := self._player_hp(key)) is not None:
            snap.hp = listed
        lvl = _to_int(scr.level)
        if lvl is not None and 1 <= lvl <= self._mech.max_level:
            snap.level = lvl
        if scr.board is not None:
            snap.board = self._resolve_units(scr.board, on_board=True)
        if scr.bench is not None:
            snap.bench = self._resolve_units(scr.bench, on_board=False)[: self._mech.bench_size]
        if scr.traits is not None:
            snap.traits = self._resolve_traits(scr.traits)
        if scr.item_bench is not None:
            snap.items = [self._item_name(x) for x in scr.item_bench if x and x.strip()]
        st.opponents[key] = snap

    # ---- own board ------------------------------------------------------------
    def _merge_own(self, scr: ScreenObservation, now: float) -> None:
        st = self._state
        self._set_int("level", scr.level, (1, self._mech.max_level), now)
        self._set_int("xp_current", scr.xp_current, (0, XP_MAX), now)
        self._set_int("xp_needed", scr.xp_needed, (0, XP_MAX), now)
        if self._set_int("hp", scr.hp, HP_RANGE, now) and st.self_name:
            for p in st.players:
                if p.name == st.self_name:
                    p.hp = st.hp
        self._set_int("streak", scr.streak, STREAK_RANGE, now)
        if st.self_name is None and scr.viewing_own_board is True and scr.viewed_player_name:
            name = scr.viewed_player_name.strip()
            if name:
                st.self_name = self._canonical_player(name) or name
                for p in st.players:
                    p.is_self = p.name == st.self_name
        if scr.board is not None:
            st.board = self._keep_positions(self._resolve_units(scr.board, on_board=True), st.board)
            st.field_age["board"] = now
        if scr.bench is not None:
            st.bench = self._resolve_units(scr.bench, on_board=False)[: self._mech.bench_size]
            st.field_age["bench"] = now
        if scr.item_bench is not None:
            st.item_bench = [self._item_name(x) for x in scr.item_bench if x and x.strip()]
            st.field_age["item_bench"] = now
        if scr.traits is not None:
            st.traits = self._resolve_traits(scr.traits)
            st.field_age["traits"] = now
        if scr.augments is not None:
            st.augments = [a.strip() for a in scr.augments if a and a.strip()]
            st.field_age["augments"] = now
        if scr.augment_choices is not None:
            st.augment_choices = [a.strip() for a in scr.augment_choices if a and a.strip()]
            st.field_age["augment_choices"] = now

    @staticmethod
    def _keep_positions(new: list[Unit], old: list[Unit]) -> list[Unit]:
        """Units read without a hex keep the position of the same unit last time."""
        pool: dict[tuple[str, int], list[Unit]] = {}
        for u in old:
            if u.row is not None and u.col is not None:
                pool.setdefault((u.api_name, u.star), []).append(u)
        taken = {(u.row, u.col) for u in new if u.row is not None and u.col is not None}
        for u in new:
            if u.row is not None and u.col is not None:
                continue
            for cand in pool.get((u.api_name, u.star), []):
                if (cand.row, cand.col) not in taken:
                    u.row, u.col = cand.row, cand.col
                    taken.add((cand.row, cand.col))
                    break
        return new

    def _merge_shop(self, slots: list[ShopSlot], now: float) -> None:
        st = self._state
        shop: list[ShopSlot] = []
        units: list[Optional[Unit]] = []
        for slot in slots[: self._mech.shop_slots]:
            name = (slot.name or "").strip() or None
            cost = _to_int(slot.cost)
            lo, hi = (0, 60) if is_wisp_name(name) else (1, 10)  # Wisps can be free or cost more than 10
            if cost is not None and not lo <= cost <= hi:
                cost = None
            if name is None:
                shop.append(ShopSlot(name=None, cost=cost))
                units.append(None)
                continue
            if is_wisp_name(name):
                # Set 18 Wisp consumable in the shop: not a champion, not a misread.
                shop.append(ShopSlot(name=name, cost=cost))
                units.append(None)
                continue
            champ = self._resolve_champion(name, cost)
            if champ is not None:
                units.append(
                    Unit(api_name=champ.api_name, name=champ.name, cost=champ.cost, traits=list(champ.traits))
                )
                shop.append(ShopSlot(name=name, cost=cost if cost is not None else champ.cost))
            else:
                units.append(Unit(api_name=f"?{name}", name=name, cost=cost))
                shop.append(ShopSlot(name=name, cost=cost))
        st.shop = shop
        st.shop_units = units
        st.field_age["shop"] = now

    # ---- resolution -----------------------------------------------------------
    def _resolve_champion(self, name: str, cost: Optional[int] = None) -> Optional[Champion]:
        champ = self._sd.resolve_champion(name)
        if cost is None or (champ is not None and champ.cost == cost):
            return champ
        key = normalize_name(name)
        if champ is not None and key in {normalize_name(champ.name), normalize_name(champ.name_en)}:
            return champ  # exact name beats a misread cost
        # The shop shows the cost: prefer a same-cost champion with a close name.
        names: dict[str, Champion] = {}
        for c in self._sd.champions_by_cost(cost):
            for n in (c.name, c.name_en):
                if normalize_name(n):
                    names.setdefault(normalize_name(n), c)
        if key in names:
            return names[key]
        match = difflib.get_close_matches(key, list(names), n=1, cutoff=0.6) if key else []
        if match:
            return names[match[0]]
        return champ

    def _resolve_unit(self, obs: UnitObs, on_board: bool = True) -> Unit:
        raw = obs.name.strip()
        star = _to_int(obs.star) or 1
        star = min(max(star, 1), 4)
        items = [self._item_name(i) for i in obs.items if i and i.strip()][:3]
        row = _to_int(obs.row)
        col = _to_int(obs.col)
        if on_board:
            if row is None or not 0 <= row <= 3:
                row = None
            if col is None or not 0 <= col <= 6:
                col = None
        else:
            row = None
            if col is not None and not 0 <= col < self._mech.bench_size:
                col = None
        champ = self._sd.resolve_champion(raw)
        if champ is None:
            return Unit(api_name=f"?{raw}", name=raw, cost=None, star=star, items=items, row=row, col=col)
        return Unit(
            api_name=champ.api_name,
            name=champ.name,
            cost=champ.cost,
            star=star,
            items=items,
            row=row,
            col=col,
            traits=list(champ.traits),
        )

    def _resolve_units(self, units: list[UnitObs], on_board: bool) -> list[Unit]:
        return [self._resolve_unit(u, on_board) for u in units if u.name and u.name.strip()]

    def _item_name(self, raw: str) -> str:
        item = self._sd.resolve_item(raw)
        return item.name if item else raw.strip()

    def _resolve_traits(self, traits: list[TraitObs]) -> list[TraitObs]:
        out: list[TraitObs] = []
        for t in traits:
            if not t.name or not t.name.strip():
                continue
            resolved = self._sd.resolve_trait(t.name)
            count = max(0, _to_int(t.count) or 0)
            out.append(
                TraitObs(
                    name=resolved.name if resolved else t.name.strip(),
                    count=count,
                    active=bool(t.active),
                    next_breakpoint=t.next_breakpoint,
                )
            )
        return out
