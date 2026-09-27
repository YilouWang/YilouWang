"""Game tracker: merges noisy per-screenshot observations into one GameState.

Merge rules (every field of a ``ScreenObservation`` is optional):

* ``None`` means "not visible": the old value is kept. An empty list means
  "visible and empty" and replaces the old value.
* Numbers go through plausibility filters (gold 0..300, level 1..max_level,
  xp >= 0, hp 0..200, streak -20..20); an implausible value is dropped and the
  old one kept.
* Stage: a later stage is accepted, except a jump over a whole stage (``3-2``
  then ``5-x``), which needs a second reading of that stage (digit misreads
  are far more common than a paused advisor). An earlier stage is ignored
  once; two consecutive earlier readings that agree with each other (the
  second the same round or later, e.g. ``3-5`` then ``3-6``) confirm it. Then
  it is either a new game (full reset) or a correction of a misread current
  stage (history points after it are dropped). A lower reading not below the
  round we were in before the current one is always a correction (the
  current stage was a forward misread). New-game evidence: a new lobby (most
  names unknown, or a full-HP list while the tracked players had lost HP), or
  our HP back up (read on our row by name, or above our verified own board;
  at most 100) on both readings: one HP reading is one misread digit away.
  Stage 1 read twice is a new game unless a frame showed this game going on
  (our lost HP, opponents at their tracked HP, a level or board stage 1
  cannot have). Auto mode reads once per round, so a new game is found even
  when stage 1 and the post-game screen were missed. A stage 1 / 2 reading
  in a new lobby, or after a plausible post-game screen, resets at once. The
  stage a reset starts from is not trusted for the jump check.
* Suspicious numbers wait for a second reading: gold up by more than a
  round's income can explain (confirmed by a second big reading not above
  the first by more than the income since), gold dropping to the old value
  with a digit lost (69 read as 6: any lower next reading confirms a drop), a
  lower level (levels never go down; two lower readings confirm the later),
  a level rise above one per round or, within a round, above what the gold
  spent since can buy, and an HP rise or a drop bigger than the fights since
  the last reading can deal (``Mechanics.max_round_damage``). The HP in our
  player-list row and the HP above our board are one reading when they agree
  (the vision reads the top HP from our row); when they disagree the
  plausible one wins. A scout frame's list row counts as a reading.
  Opponents' HP: a 0 applies at once and is final when read again (a single
  0 is undone by a next reading of the HP from before it); a confirmed
  elimination never comes back, and a rise needs the next reading to agree.
* The local player (``self_name``) is learned from the flagged player-list
  row and switches when another row is flagged on two frames in a row (or
  once, while the name is unconfirmed, when that row's HP matches the HP
  above our own board). Once known, our row is found by name, never by a
  flag. A frame whose flagged row carries an opponent's name with our HP
  while our name carries that opponent's HP (two names swapped) keeps the
  list we have.
* Scouting (``purpose == "scout"`` or ``viewing_own_board is False`` outside
  the carousel / loading / augment screens): the
  board, bench, traits, HP, level and item bench on screen belong to the viewed
  player and are stored as an ``OpponentSnapshot``. Only the stage, the player
  list, our gold and our shop are applied to our own state, because in TFT the
  bottom HUD shop and gold always show the local player's values. A level on
  a frame whose arena is not ours only confirms a pending reading (some
  clients show the viewed player's level near the board, and a wrong own
  level is worse than a slightly stale one); XP is not taken there, the
  streak only on a non-scout frame. The board owner is the named opponent
  when the banner names a known opponent whose listed HP is the HP above the
  board (otherwise the player that HP belongs to, or ``unknown``), except
  that on a non-scout frame a board that looks like ours (same champions,
  stars ignored) is ours whatever name was read. A scout frame the vision
  marks ``viewing_own_board=True`` without another player's name is our own
  board only when the board looks like ours; otherwise it is filed under the
  opponent with that HP (provisionally: dropped when it turns out to be a
  board we know), or ``unknown``. A non-scout frame
  called ours whose top HP contradicts our row while the board is not ours
  is unsure: bottom HUD only, no snapshot. A scout frame whose board is
  identical to ours with no known opponent named is our own board. The
  opponent's level comes from ``viewed_player_level`` (or the older note
  "对手等级 N"), never from ``level`` (the local HUD). Without a scout request
  and without a readable owner, ``viewing_own_board=False`` with our own HP
  above the board (and no other player's) is our arena, with an HP another
  player shares it is unsure, otherwise it is filed under ``unknown`` when
  the board is clearly not ours (no HP-match guess).
* ``viewing_own_board=None`` (unsure whose arena it is) on an auto / manual
  frame: the bottom HUD (level, XP, streak) applies; the arena fields (board,
  bench, traits, item bench, top level HP) only when the board looks like
  ours or no own board is known yet. A board called ours that shares no
  champion with our board (3+ units) and names nobody replaces it only when
  the next frame shows it again.
* Gold, shop and augment cards are local UI and apply on every frame. An
  empty shop list means the shop was not seen (a visible shop has slots).
  The streak only changes with the round: a 0 read within the round a known
  streak was read in (the placeholder of an unread streak) is ignored. A level
  change without an XP reading drops the old level's XP.
* Carousel and loading frames never carry our board, bench, traits or item
  bench (the carousel ring is not a board): those fields are ignored, and
  ``viewing_own_board=False`` there is not scouting. On the augment screen
  the cards cover the arena: an empty or much smaller board / bench read is
  ignored when ours is known.
* Eliminated players (HP 0 in the player list) do not count in
  ``taken_copies`` because their units go back to the pool, nor do snapshots
  of names missing from a full 8-row list. The ``unknown`` snapshot and the
  provisional ones are dropped when their units are mostly ours or another
  snapshot's (containment, stars ignored), ``unknown`` also once the stage
  moves on; a named scout replaces ``unknown`` only when it holds the same
  units.
* Combat frames: once our board is known a combat frame never changes it (no
  unit joins or leaves the board mid-fight, fighters walk off their hexes and
  surviving enemy units stand on our arena); a combat board that does not
  look like ours is not our arena. A fielded unit that vanishes for one
  frame (not moved to the bench, not sold) is kept for that frame, and hex
  positions are remembered for a few rounds.

Timestamps (``last_update``, ``field_age``, snapshot / history times) come
from the injected ``clock`` so tests are deterministic. ``field_age["round"]``
is when the current round started: a shop read before it is last round's
shop (the shop refreshes every round).

Manual corrections (``set_field``) accept full-width digits and dashes
(``３－２``), bound ``xp_current`` by the level's XP table and stage / round
by 1..7, and are taken at once (no second reading). Correcting the stage back
to the round the current shop was read in keeps that shop current.
"""

from __future__ import annotations

import difflib
import re
import threading
import time
import unicodedata
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
MAX_STAGE = 7  # manual stage / round corrections above this are typos
ROUNDS_PER_STAGE = 7
GOLD_JUMP_ROUND = 30  # most gold one round's selling / augments plausibly add without a second reading
GOLD_JUMP_PER_ROUND = 30  # ... plus this per round since the last gold reading
GOLD_CONFIRM_TOL = 3  # a repeated suspicious gold reading may differ by a few gold (a buy)
HP_RISE_MAX = 6  # HP never rises in TFT except a few points from augments
HP_MISMATCH = 2  # top HP and our player-list row further apart than this disagree
SELF_SWITCH_VOTES = 2  # frames flagging another row as the local player before switching
POS_MEMORY_ROUNDS = 4  # rounds a unit's last hex is remembered after it left the board
NEW_LOBBY_HP_RISE = 15  # our HP read this much above the tracked HP: a new game
START_HP = 100  # every game starts at 100 HP: a "new game" HP reading is never above it
STAGE_ONE_MAX_LEVEL = 4  # a stage 1 frame never shows a higher level ...
STAGE_ONE_MAX_BOARD = 4  # ... or more fielded units
SAME_GAME_ROWS = 2  # listed opponents still at their tracked (lost) HP: the same lobby
MIN_BOARD_FOR_HOLD = 3  # a board this big replaced by one sharing no champion waits for a second frame

# ``_opponent_key`` results that are not snapshot keys: the frame is our own
# arena after all (the vision's "not own" was wrong), or nobody's for sure
# (apply the bottom HUD only, store no snapshot).
_OWN_ARENA = "\x00own"
_UNSURE = "\x00unsure"



WISP_PREFIXES = ("wisp", "精灵", "灵火")
_OPP_LEVEL_NOTE = re.compile(r"对手等级\s*[:：]?\s*(\d{1,2})")


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


def _containment(a: dict[str, int], b: dict[str, int]) -> float:
    """Share of ``a`` found in ``b`` (0..1): a subset of ``b`` scores 1."""
    size = sum(a.values())
    if size <= 0:
        return 0.0
    return sum(min(n, b.get(api, 0)) for api, n in a.items()) / size


def _identity(units: list[Unit]) -> dict[str, int]:
    """Units per champion, stars ignored (a star-up is the same board)."""
    out: dict[str, int] = {}
    for u in units:
        if not u.api_name.startswith("?"):
            out[u.api_name] = out.get(u.api_name, 0) + 1
    return out


def _rounds_between(a: Optional[tuple[int, int]], b: Optional[tuple[int, int]]) -> int:
    """Approximate rounds from stage key ``a`` to ``b`` (0 when unknown or not later)."""
    if a is None or b is None or b <= a:
        return 0
    return (b[0] - a[0]) * ROUNDS_PER_STAGE + (b[1] - a[1])


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
        # Full-width digits and signs (a Chinese IME in full-width mode).
        text = unicodedata.normalize("NFKC", value).strip().replace("+", "")
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
        self._streak_stage: Optional[tuple[int, int]] = None  # round the streak was last read in
        self._games = 0
        self._state = GameState(game_id=self._new_game_id())
        self._clear_game_memory()

    def _clear_game_memory(self) -> None:
        """Per-game bookkeeping that is not part of the published state."""
        self._pending_evidence: Optional[str] = None  # new-lobby evidence of the pending lower reading
        self._pending_same = False  # the pending lower reading's frame showed the current game going on
        self._fresh_reset = False  # the stage came from a new-game reset: not trusted for the jump check
        self._suspect: dict[str, int] = {}  # field -> suspicious reading waiting for a second one
        self._suspect_stage: dict[str, Optional[tuple[int, int]]] = {}  # round a suspect was read in
        self._opp_suspect: dict[str, int] = {}  # opponent -> suspicious HP rise waiting for a second reading
        self._opp_zero: dict[str, int] = {}  # opponent read at 0 once -> the HP before (not final yet)
        self._level_mark: Optional[tuple[Optional[tuple[int, int]], Optional[int]]] = None  # (round, gold) of the last level reading
        self._level_undo: Optional[int] = None  # our level before the last accepted drop
        self._shop_stage: Optional[tuple[int, int]] = None  # round the current shop was read in
        self._provisional: set[str] = set()  # own-labeled scouts filed by an HP match (no readable owner)
        self._pending_board: Optional[dict[str, int]] = None  # an own board sharing nothing with ours, once
        self._listed_before: dict[str, Optional[int]] = {}  # player HP as listed before the current frame
        self._gold_stage: Optional[tuple[int, int]] = None  # round gold / HP were last accepted in
        self._hp_stage: Optional[tuple[int, int]] = None
        self._hp_undo: Optional[int] = None  # our HP before the last accepted drop
        self._xp_memo: dict[int, tuple[Optional[int], Optional[int], Optional[tuple[int, int]]]] = {}
        self._self_challenger: Optional[tuple[str, int]] = None  # another flagged row and its votes
        self._self_seen = 0  # frames that flagged the current self_name
        self._pos_memory: dict[tuple[str, int], list[tuple[int, int, Optional[tuple[int, int]]]]] = {}
        self._missing_once: set[tuple[str, int]] = set()  # board units kept through one missed frame

    @property
    def self_name_confirmed(self) -> bool:
        """The local player's name was flagged on at least two frames (or set
        from our own board's banner): safe to send as a vision hint."""
        with self._lock:
            return self._state.self_name is not None and self._self_seen >= 2

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
            self._streak_stage = None
            self._clear_game_memory()
            return self._state.model_copy(deep=True)

    def ingest(self, obs: Observation) -> GameState:
        with self._lock:
            now = self._clock()
            scr = obs.screen
            st = self._state

            if scr.screen_type == ScreenType.POST_GAME:
                st.screen_type = ScreenType.POST_GAME
                st.last_update = now
                # Only a plausible end screen arms the new-game reset (a round
                # result banner misread as post-game must not).
                if self._post_game_plausible(scr):
                    self._game_over = True
                return st.model_copy(deep=True)

            stage_changed = self._merge_stage(scr.stage, now, scr, obs.purpose)
            st = self._state  # may have been replaced by a new-game reset
            st.screen_type = scr.screen_type
            gold_before = st.gold
            # Listed HP before this frame's list (a name swap on this frame
            # must not make a correct banner look wrong).
            self._listed_before = {p.name: p.hp for p in st.players}

            row_hp: Optional[int] = None
            if scr.players is not None:
                # While the camera is away TFT highlights the viewed player's
                # row too: flags there do not teach who we are.
                away = obs.purpose == "scout" or scr.viewing_own_board is False
                row_hp = self._merge_players(scr.players, now, scr, away=away)

            if scr.screen_type in NO_BOARD_SCREENS:
                # The carousel ring / loading screen is not anyone's board.
                scr = scr.model_copy(update={"board": None, "bench": None, "traits": None, "item_bench": None})
            elif scr.screen_type == ScreenType.AUGMENT_SELECT:
                scr = self._drop_covered_board(scr)

            scouting = self._is_scouting(obs) or self._foreign_arena(scr, obs.purpose)
            force_own = False
            key: Optional[str] = None
            if scouting:
                key = self._opponent_key(scr, obs.purpose)
                if key is None:  # the "scouted" board is our own board
                    scouting = False
                elif key == _OWN_ARENA:  # "not own" read on our own arena (our HP above it)
                    scouting = False
                    force_own = True
                elif key != _UNSURE:
                    self._store_opponent(key, scr, now)

            # Gold, shop and the augment cards always belong to the local player.
            self._merge_gold(scr.gold, now)
            if scr.shop:  # a visible shop has its slots: [] means it was not seen
                self._merge_shop(scr.shop, now)
            if scr.augment_choices is not None:
                st.augment_choices = [a.strip() for a in scr.augment_choices if a and a.strip()]
                st.field_age["augment_choices"] = now

            top_hp: Optional[int] = None
            if not scouting:
                top_hp = self._merge_own(scr, now, gold_before, force=force_own)
            else:
                # A board that is not ours: the bottom HUD is still the local
                # player's, but some clients show the viewed player's level
                # near the board, so its level only confirms a pending reading
                # (never sets one). A scout frame keeps our streak / XP. An
                # unsure frame of our own capture (maybe our arena) applies
                # the HUD; its level still goes through the plausibility check.
                unsure = key == _UNSURE and obs.purpose != "scout"
                self._merge_hud(scr, now, trusted=unsure, streak=obs.purpose != "scout")
            self._merge_hp(row_hp, top_hp, now)

            if stage_changed:
                st.augment_choices = [] if scr.augment_choices is None else st.augment_choices
                self._append_history(now)
            else:
                self._fill_history()
            st.last_update = now
            return st.model_copy(deep=True)

    def taken_by_player(self) -> dict[str, dict[str, int]]:
        """Copies (1-star equivalents) held by each alive opponent, from the latest scout.

        The ``unknown`` snapshot (a scout frame with no readable owner) and
        the provisional ones (a scout the vision called our own board, filed
        under a player only because the HP above the board matched their
        listed HP) are dropped when their units are mostly found in another
        snapshot or in our own units (containment, stars ignored: a subset of
        our bench read on the augment screen is ours), ``unknown`` also when
        it is from an earlier stage, so the same copies are not subtracted
        from the pool twice.
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
            dropped: set[str] = set()
            for key in [UNKNOWN_PLAYER, *sorted(self._provisional)]:
                snap = st.opponents.get(key)
                if key not in out or snap is None:
                    continue
                seen_at = StageRound.parse(snap.stage) if snap.stage else None
                ident = _identity([*snap.board, *snap.bench])
                others = [
                    _identity([*o.board, *o.bench]) for n, o in st.opponents.items() if n != key and n not in dropped
                ]
                others.append(_identity(st.all_units()))
                # The ownerless board expires with its stage; an HP-matched
                # scout (the player pressed F7 on someone) stays until it
                # turns out to be a board we know.
                stale = (
                    key == UNKNOWN_PLAYER
                    and seen_at is not None
                    and st.stage is not None
                    and seen_at.stage < st.stage.stage
                )
                if stale or any(_containment(ident, c) >= DUPLICATE_OVERLAP for c in others):
                    del out[key]
                    dropped.add(key)
                    continue
                if key == UNKNOWN_PLAYER:
                    # A named board found inside a newer ownerless one: the
                    # same player seen later (bigger board). Count it once,
                    # with the newer units.
                    for n, o in st.opponents.items():
                        if n == key or n not in out or o.captured_at > snap.captured_at:
                            continue
                        if _containment(_identity([*o.board, *o.bench]), ident) >= DUPLICATE_OVERLAP:
                            del out[n]
                            dropped.add(n)
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
        if isinstance(value, str):
            # Full-width digits / dashes typed with a Chinese IME ("３－２", "５５").
            value = unicodedata.normalize("NFKC", value)
        with self._lock:
            now = self._clock()
            st = self._state
            if name == "stage":
                sr = value if isinstance(value, StageRound) else StageRound.parse(str(value) if value is not None else None)
                if sr is None:
                    raise ValueError(f"回合格式不对：{value!r}（例如 3-2）")
                if not (1 <= sr.stage <= MAX_STAGE and 1 <= sr.round <= ROUNDS_PER_STAGE):
                    raise ValueError(f"回合 {sr} 不合理（阶段和回合都在 1 到 {MAX_STAGE} 之间，例如 3-2）")
                advanced = st.stage is None or sr.key > st.stage.key
                if st.stage is not None and sr.key < st.stage.key:
                    st.history = [h for h in st.history if (StageRound.parse(h.stage) or sr).key <= sr.key]
                if st.stage is None or sr.key != st.stage.key:
                    shop_t = st.field_age.get("shop")
                    if self._shop_stage is not None and shop_t is not None and sr.key <= self._shop_stage:
                        # Back to the round the shop was read in (the shop was
                        # read under the misread stage): it is this round's shop.
                        st.field_age["round"] = min(st.field_age.get("round", shop_t), shop_t)
                    else:
                        st.field_age["round"] = now
                st.stage = sr
                self._pending_lower = None
                self._pending_jump = None
                self._fresh_reset = False
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
                if name == "xp_current" and st.level is not None:
                    need = self._mech.xp_to_level.get(st.level)
                    if need is not None and num >= need:
                        raise ValueError(f"{st.level} 级升级只要 {need} 经验，当前经验应在 0 到 {need - 1} 之间")
                old_level = st.level
                setattr(st, name, num)
                st.field_age[name] = now
                # A correction is the truth: no second reading needed.
                self._suspect.pop(name, None)
                key = st.stage.key if st.stage is not None else None
                if name == "gold":
                    self._gold_stage = key
                if name == "level":
                    self._level_mark = (key, st.gold)
                    self._level_undo = None
                    if num != old_level:
                        self._reset_xp(False, False)
                if name == "streak":
                    self._streak_stage = key
                if name == "hp":
                    self._hp_stage = key
                    self._hp_undo = None
                    if st.self_name:
                        for p in st.players:
                            if p.name == st.self_name:
                                p.hp = num
                self._remember_xp()
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
    def _post_game_plausible(self, scr: ScreenObservation) -> bool:
        """An end screen: our HP at 0 (in the list or above the board), or no
        stage and no shop visible. A round result banner the model called
        post-game shows the stage or the shop."""
        if _to_int(scr.hp) == 0:
            return True
        st = self._state
        for p in scr.players or []:
            if (p.is_self or (st.self_name and p.name == st.self_name)) and _to_int(p.hp) == 0:
                return True
        return StageRound.parse(scr.stage) is None and not scr.shop

    def _own_hp_rows(self, scr: ScreenObservation, purpose: str) -> list[int]:
        """Our HP as listed in this frame's player list: our row by name once
        the local name is known (a flag can sit on the viewed player's row, or
        on a row whose name was swapped), the flagged row only before that and
        never on a frame whose camera is away."""
        st = self._state
        away = purpose == "scout" or scr.viewing_own_board is False
        out: list[int] = []
        for p in scr.players or []:
            if not p.name or not p.name.strip():
                continue
            if st.self_name:
                ours = self._canonical_player(p.name) == st.self_name
            else:
                ours = bool(p.is_self) and not away
            if ours and (h := _to_int(p.hp)) is not None and HP_RANGE[0] <= h <= HP_RANGE[1]:
                out.append(h)
        return out

    def _top_hp_is_ours(self, scr: ScreenObservation, purpose: str) -> bool:
        """Is the HP above the board ours? Only on a frame of our own arena:
        not a scout, not a "not own" frame, the HP agrees with our row, and
        the vision called the arena ours or the board looks like ours (a new
        game's board never looks like the old one)."""
        if purpose == "scout" or scr.viewing_own_board is False:
            return False
        if scr.viewing_own_board is None and scr.board is not None and not self._board_is_ours(scr):
            return False
        return not self._hp_contradicts(scr)

    def _new_lobby_evidence(self, scr: Optional[ScreenObservation], purpose: str = "auto") -> Optional[str]:
        """Does this frame look like a different game?

        ``"names"``: most listed names are unknown, or a full-HP list while
        the tracked players had lost HP (a new lobby). ``"hp"``: our HP read
        well above the tracked HP (HP never comes back) and at most the start
        HP: one reading, which a misread digit can fake, so it counts only
        when two frames show it (``"start"``: exactly the start HP, enough
        with an early stage on a frame showing nothing of this game). None:
        no evidence.
        """
        if scr is None:
            return None
        st = self._state
        rows = [p for p in scr.players or [] if p.name and p.name.strip()]
        if len(rows) >= 4:
            known = set(self._known_player_names())
            if known:
                unknown = [p for p in rows if self._canonical_player(p.name) not in known]
                if len(unknown) * 2 > len(rows):
                    return "names"
            listed = [h for h in (_to_int(p.hp) for p in rows) if h is not None]
            hurt = any(p.hp is not None and p.hp < START_HP for p in st.players)
            if hurt and len(listed) >= 4 and all(h == START_HP for h in listed):
                return "names"
        mine = self._own_hp_rows(scr, purpose)
        if self._top_hp_is_ours(scr, purpose) and (h := _to_int(scr.hp)) is not None:
            mine.append(h)
        risen = [h for h in mine if st.hp is not None and st.hp + NEW_LOBBY_HP_RISE <= h <= START_HP]
        if risen:
            # Exactly the start HP ("start") is stronger than any other rise.
            return "start" if START_HP in risen else "hp"
        return None

    def _same_game_evidence(self, scr: Optional[ScreenObservation], purpose: str = "auto") -> bool:
        """Does this frame show the current game going on? Our HP (well below
        the start HP) read again, several opponents listed at their tracked
        lost HP, or a level / board no stage 1 frame can show."""
        if scr is None:
            return False
        st = self._state
        if st.hp is not None and st.hp <= START_HP - NEW_LOBBY_HP_RISE:
            mine = self._own_hp_rows(scr, purpose)
            if self._top_hp_is_ours(scr, purpose) and (h := _to_int(scr.hp)) is not None:
                mine.append(h)
            if any(abs(h - st.hp) <= HP_MISMATCH for h in mine):
                return True
        old = {p.name: p.hp for p in st.players if p.name != st.self_name}
        same = 0
        for p in scr.players or []:
            name = self._canonical_player(p.name)
            hp = _to_int(p.hp)
            if name in old and hp is not None and old[name] is not None and old[name] < START_HP and hp == old[name]:
                same += 1
        if same >= SAME_GAME_ROWS:
            return True
        level = _to_int(scr.level)
        if level is not None and STAGE_ONE_MAX_LEVEL < level <= self._mech.max_level:
            return True
        units = [u for u in scr.board or [] if u.name and u.name.strip()]
        return purpose != "scout" and scr.viewing_own_board is not False and len(units) > STAGE_ONE_MAX_BOARD

    def _start_new_game(self, sr: StageRound, now: float) -> None:
        self.reset()
        self._state.stage = sr
        self._state.field_age["stage"] = now
        self._state.field_age["round"] = now
        # The stage came from the frames that looked like a new game: the next
        # reading is not judged against it (no "jump over a stage" wait).
        self._fresh_reset = True

    def _last_stage_before(self, key: tuple[int, int]) -> Optional[tuple[int, int]]:
        """The latest history round before ``key`` (the round we were in
        before a jump to ``key``)."""
        best: Optional[tuple[int, int]] = None
        for h in self._state.history:
            sr = StageRound.parse(h.stage)
            if sr is not None and sr.key < key and (best is None or sr.key > best):
                best = sr.key
        return best

    def _merge_stage(
        self, text: Optional[str], now: float, scr: Optional[ScreenObservation] = None, purpose: str = "auto"
    ) -> bool:
        """Apply a stage reading. Returns True when the stage advanced / changed."""
        sr = StageRound.parse(text)
        if sr is None:
            return False
        st = self._state
        cur = st.stage
        if (
            cur is not None
            and sr.stage > cur.stage + 1
            and self._pending_jump != sr.stage
            and not self._fresh_reset
        ):
            # Skipping a whole stage between two reads is almost always a digit
            # misread ("3-2" read as "8-2"): wait for a second reading.
            self._pending_jump = sr.stage
            return False
        if cur is None or sr.key > cur.key:
            st.stage = sr
            st.field_age["stage"] = now
            st.field_age["round"] = now
            self._pending_lower = None
            self._pending_jump = None
            self._fresh_reset = False
            self._game_over = False  # the game (or a spectated one) goes on
            return True
        self._pending_jump = None
        if sr.key == cur.key:
            st.field_age["stage"] = now
            self._pending_lower = None
            self._fresh_reset = False
            self._game_over = False  # a post-game frame was a misclassification
            return False
        # Lower than the current stage: a misread (ignored once), a new game or
        # a correction of a misread current stage.
        evidence = self._new_lobby_evidence(scr, purpose)
        same = self._same_game_evidence(scr, purpose)
        # Not below the round we were in before the current one: the current
        # stage was a forward misread ("5-1" read as "5-5") and this is the
        # game going on, never a new game.
        before = self._last_stage_before(cur.key)
        correction = before is not None and sr.key >= before
        fresh_start = evidence == "start" and not same
        if (
            not correction
            and cur.key >= (2, 1)
            and sr.stage <= 2
            and (evidence == "names" or fresh_start or self._game_over)
        ):
            # An early round in a new lobby, or with our HP at the start HP
            # and nothing of this game on screen (or after the end screen): a
            # new game from its first frame.
            self._start_new_game(sr, now)
            return True
        if not correction and self._game_over and cur.key >= (2, 1) and evidence:
            self._start_new_game(sr, now)
            return True
        pending = self._pending_lower
        if pending is not None and pending <= sr.key:
            # Two consecutive lower readings that agree (the same round, or a
            # later one: auto mode reads once per round).
            hp_now = evidence in ("hp", "start")
            hp_before = self._pending_evidence in ("hp", "start")
            new_game = cur.key >= (2, 1) and not correction and (
                "names" in (evidence, self._pending_evidence)
                or (hp_now and hp_before)
                # Stage 1 twice: a new game unless either frame showed this
                # game going on (the same stylized digit is often misread
                # twice on the same pixels).
                or (pending[0] == 1 and not same and not self._pending_same)
            )
            if new_game:
                self._start_new_game(sr, now)
                return True
            if sr.stage == 1 and cur.stage >= 2 and (same or self._pending_same):
                # Stage 1 on frames showing this game going on (our lost HP,
                # a level or board stage 1 cannot have): a misread, twice.
                self._pending_lower = None
                return False
            st.history = [h for h in st.history if (StageRound.parse(h.stage) or sr).key < sr.key]
            st.stage = sr
            st.field_age["stage"] = now
            st.field_age["round"] = now
            self._pending_lower = None
            return True
        self._pending_lower = sr.key
        self._pending_evidence = evidence
        self._pending_same = same
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
        cjk = any("\u4e00" <= ch <= "\u9fff" for ch in key)
        if 3 <= len(key) <= 4 or (len(key) == 2 and cjk):
            # Short names (typical Chinese IDs) score below the cutoff with one
            # wrong character: accept a single substitution when exactly one
            # known name of the same length fits. Two different digits are
            # different players ("P1" / "P2"), not a misread.
            def one_off(k: str) -> bool:
                diff = [(a, b) for a, b in zip(k, key) if a != b]
                return len(k) == len(key) and len(diff) == 1 and not (diff[0][0].isdigit() and diff[0][1].isdigit())

            close = [k for k in by_key if one_off(k)]
            if len(close) == 1:
                return by_key[close[0]]
        return name

    def _merge_players(
        self, players: list[PlayerObs], now: float, scr: Optional[ScreenObservation] = None, away: bool = False
    ) -> Optional[int]:
        """Merge the player list; returns our own row's HP as read on this
        frame (None when not read). The HP itself is applied by ``_merge_hp``."""
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
        rows: list[tuple[str, Optional[int], bool]] = []
        seen: set[str] = set()
        flagged: list[str] = []
        read_hp: dict[str, Optional[int]] = {}
        for p in players:
            if len(rows) >= MAX_PLAYERS:
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
            read_hp[name] = hp
            if p.is_self:
                flagged.append(name)
            rows.append((name, hp, bool(p.is_self)))
        if not away and self._names_swapped(flagged, read_hp):
            # Two rows' names swapped (the HP stays with the row): every name
            # on this frame is suspect, keep the list we have.
            return None
        merged: list[PlayerObs] = []
        for name, hp, _flag in rows:
            prev = old.get(name)
            if prev is not None and name != st.self_name:
                hp = self._opponent_hp(name, hp, prev.hp)
            if hp is None and prev is not None:
                hp = prev.hp
            merged.append(PlayerObs(name=name, hp=hp, is_self=False))
        # Keep players the frame missed (partly hidden list), up to 8 total.
        for p in st.players:
            if p.name not in seen and len(merged) < MAX_PLAYERS:
                merged.append(p.model_copy())
                seen.add(p.name)
        if not away:
            self._update_self_name(flagged, read_hp, scr)
        for p in merged:
            p.is_self = st.self_name is not None and p.name == st.self_name
        st.players = merged
        st.field_age["players"] = now
        return read_hp.get(st.self_name) if st.self_name else None

    def _names_swapped(self, flagged: list[str], read_hp: dict[str, Optional[int]]) -> bool:
        """The flagged row carries a known opponent's name with our HP, and our
        name sits on an unflagged row with that opponent's HP: the vision
        swapped two names (the highlight and the HP stay with the row)."""
        st = self._state
        rows = list(dict.fromkeys(flagged))
        if len(rows) != 1 or not st.self_name or self._self_seen < 2 or st.hp is None:
            return False
        flag = rows[0]
        if flag == st.self_name or st.self_name not in read_hp:
            return False
        theirs = self._player_hp(flag)
        at_flag, at_ours = read_hp.get(flag), read_hp.get(st.self_name)
        if theirs is None or at_flag is None or at_ours is None or abs(st.hp - theirs) <= HP_MISMATCH:
            return False
        return abs(at_flag - st.hp) <= HP_MISMATCH and abs(at_ours - theirs) <= HP_MISMATCH

    def _opponent_hp(self, name: str, hp: Optional[int], prev: Optional[int]) -> Optional[int]:
        """An opponent's HP reading. A drop to 0 applies at once (their units
        go back to the pool now) but is final only when read again: a player
        whose single 0 was a misread or a name swap comes back when the next
        reading shows the HP they had before it. A confirmed elimination
        never comes back, and an HP rise (HP never comes back) counts only
        when the next reading agrees."""
        if hp is None or prev is None:
            return hp
        if prev == 0:
            before = self._opp_zero.get(name)
            if before is None:
                return 0  # confirmed (or read as 0 from the start)
            if hp == 0:
                del self._opp_zero[name]
                return 0
            if abs(hp - before) <= HP_MISMATCH:
                del self._opp_zero[name]
                return hp  # the 0 was a misread
            return 0
        if hp == 0:
            self._opp_zero[name] = prev
            self._opp_suspect.pop(name, None)
            return 0
        if hp > prev + HP_RISE_MAX:
            last = self._opp_suspect.get(name)
            if last is not None and abs(last - hp) <= 1:
                self._opp_suspect.pop(name, None)
                return hp
            self._opp_suspect[name] = hp
            return prev
        self._opp_suspect.pop(name, None)
        return hp

    def _update_self_name(
        self, flagged: list[str], read_hp: dict[str, Optional[int]], scr: Optional[ScreenObservation]
    ) -> None:
        """Learn / correct the local player's name from the flagged row.

        A known name is not replaced by one stray flag (a hovered row): another
        row must be flagged on SELF_SWITCH_VOTES frames in a row. While the
        name is not confirmed (one flag so far), the HP above our own board
        matching that row (and not the current one) counts as a second vote
        on the same frame; once confirmed it does not (the vision reads our
        top HP from the flagged row, so a swapped name carries it along).
        """
        st = self._state
        rows = list(dict.fromkeys(flagged))
        flag = rows[0] if len(rows) == 1 else None
        if flag is None:
            return
        if st.self_name is None:
            st.self_name = flag
            self._self_seen = 1
            self._self_challenger = None
            return
        if flag == st.self_name:
            self._self_seen += 1
            self._self_challenger = None
            return
        votes = 1
        own = scr is not None and scr.viewing_own_board is True and scr.board is not None and self._board_is_ours(scr)
        top = _to_int(scr.hp) if own and scr is not None and self._self_seen < 2 else None
        if top is not None and read_hp.get(flag) == top and read_hp.get(st.self_name) != top:
            votes += 1
        prev = self._self_challenger
        total = (prev[1] if prev is not None and prev[0] == flag else 0) + votes
        if total >= SELF_SWITCH_VOTES:
            st.self_name = flag
            self._self_seen = 1
            self._self_challenger = None
        else:
            self._self_challenger = (flag, total)

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
        if listed is None and name != UNKNOWN_PLAYER and len(st.players) >= MAX_PLAYERS:
            return False  # not in a full lobby list: a misread name or another game's player
        return snap.hp is None or snap.hp > 0

    # ---- scouting -------------------------------------------------------------
    @staticmethod
    def _is_scouting(obs: Observation) -> bool:
        if obs.purpose == "scout":
            return True
        # False on a carousel / loading frame is the "no arena" answer (or the
        # false placeholder), not a scouted board: the HUD there is ours. On
        # the augment screen the cards cover the arena, so False there is not
        # a scouted board either.
        return (
            obs.screen.viewing_own_board is False
            and obs.screen.screen_type not in NO_BOARD_SCREENS
            and obs.screen.screen_type != ScreenType.AUGMENT_SELECT
        )

    def _frame_self_hp(self, scr: ScreenObservation) -> Optional[int]:
        """Our HP in this frame's player list: our row by name once the local
        name is known (a flag may sit on a swapped or viewed row), the
        flagged row before that."""
        st = self._state
        for p in scr.players or []:
            if st.self_name:
                if self._canonical_player(p.name) == st.self_name:
                    return _to_int(p.hp)
            elif p.is_self:
                return _to_int(p.hp)
        return None

    def _hp_contradicts(self, scr: ScreenObservation) -> bool:
        """The HP above the board differs from our own row in the same frame."""
        top = _to_int(scr.hp)
        row = self._frame_self_hp(scr)
        return top is not None and row is not None and abs(top - row) > HP_MISMATCH

    def _foreign_arena(self, scr: ScreenObservation, purpose: str) -> bool:
        """A frame the vision calls our own board that is someone else's: the
        HP above it is not ours and the board does not look like ours."""
        if purpose == "scout" or scr.viewing_own_board is not True:
            return False
        if scr.screen_type in NO_BOARD_SCREENS or scr.screen_type == ScreenType.AUGMENT_SELECT:
            return False
        st = self._state
        return (
            bool(st.board) and bool(scr.board) and self._hp_contradicts(scr) and not self._board_is_ours(scr, lenient=True)
        )

    def _hp_match(self, scr: ScreenObservation, exclude: Optional[str] = None) -> Optional[str]:
        """The unique opponent whose listed HP is the HP shown above the board."""
        st = self._state
        hp = _to_int(scr.hp)
        if hp is None:
            return None
        matches = [
            p.name for p in st.players if not p.is_self and p.hp == hp and p.name not in (st.self_name, exclude)
        ]
        return matches[0] if len(matches) == 1 else None

    def _opponent_key(self, scr: ScreenObservation, purpose: str = "scout") -> Optional[str]:
        """Snapshot key for a scouted board, or None when it is our own board.

        Order matters: on a scout frame a readable banner naming a known
        opponent wins over the "identical board" heuristic (early boards of
        1-3 cheap units often match by chance). On a frame the player did not
        ask to scout, a board that looks like ours is ours whatever name the
        model read. ``viewing_own_board=True`` without another player's name
        is our own board only when the board looks like ours.
        """
        st = self._state
        name = self._canonical_player(scr.viewed_player_name)
        looks_ours = bool(st.board) and scr.board is not None and self._board_is_ours(scr)
        if name and st.self_name and normalize_name(name) == normalize_name(st.self_name):
            if purpose == "scout" and st.board and scr.board and not self._board_is_ours(scr, lenient=True):
                # Our name on a clearly different board: the local name is
                # probably a mis-flagged row. Keep the board under that name;
                # it counts once the local player's name is corrected.
                return self._named_key(name)
            return None
        if purpose != "scout" and looks_ours:
            return None  # our own board with a model-read name or a false placeholder
        opponents = {p.name for p in st.players if p.name != st.self_name and not p.is_self}
        if name and name in opponents:
            listed = [h for h in (self._player_hp(name), self._listed_before.get(name)) if h is not None]
            top = _to_int(scr.hp)
            mine = self._frame_self_hp(scr)
            if mine is None:
                mine = st.hp
            if (
                listed
                and top is not None
                and all(abs(top - h) > HP_MISMATCH for h in listed)
                and top != mine  # our own HP copied from our row: says nothing about the board
            ):
                # The HP above the board is not the named player's: the banner
                # name was misread. File it under the player that HP belongs
                # to, never over the named player's snapshot.
                return self._hp_match(scr, exclude=name) or UNKNOWN_PLAYER
            return self._named_key(name)
        if scr.viewing_own_board is True and not name:
            if not st.board or not scr.board or self._board_is_ours(scr, lenient=True):
                return None  # the camera never left home (scout key pressed too early)
            if purpose != "scout":
                # Our own capture, called our arena, whose top HP disagrees
                # with our row: a misread HP after a roll-down is as likely as
                # another arena. Unsure: bottom HUD only, no snapshot.
                return _UNSURE
            return self._provisional_key(self._hp_match(scr))
        if scr.board and st.board and self._same_units(scr.board, st.board):
            return None  # identical to our board: the camera never left home
        if name:
            return self._named_key(name)
        if purpose != "scout":
            # viewing_own_board=False without a scout request or a readable
            # owner, and a board that does not look like ours: our own arena
            # when the HP above it is ours (and nobody else's), unsure when an
            # opponent has the same HP, otherwise an unknown opponent.
            top = _to_int(scr.hp)
            mine = self._frame_self_hp(scr)
            if mine is None:
                mine = st.hp
            if top is not None and mine is not None and top == mine:
                return _UNSURE if any(p.hp == top for p in st.players if p.name != st.self_name) else _OWN_ARENA
            return UNKNOWN_PLAYER
        # Fallback: a unique opponent with the HP shown on the board.
        return self._hp_match(scr) or UNKNOWN_PLAYER

    def _named_key(self, name: str) -> str:
        """Snapshot key read from the banner: a board filed there earlier by
        its HP only is confirmed (or replaced) now."""
        self._provisional.discard(name)
        return name

    def _provisional_key(self, name: Optional[str]) -> str:
        """Snapshot key for a scout the vision called our own board that is
        not ours, filed by its HP only: provisional (dropped like ``unknown``
        when it turns out to be a board we know), or ``unknown`` when no
        single player has that HP."""
        if name is None:
            return UNKNOWN_PLAYER
        self._provisional.add(name)
        return name

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
        # ``scr.level`` is the local player's HUD level, never the opponent's.
        lvl = self._viewed_level(scr)
        if lvl is not None:
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
        unknown = st.opponents.get(UNKNOWN_PLAYER)
        if key != UNKNOWN_PLAYER and unknown is not None:
            # A named scout supersedes the ownerless one when it is the same
            # board (a different board is another player's: keep it).
            ident = _identity([*unknown.board, *unknown.bench])
            if _containment(ident, _identity([*snap.board, *snap.bench])) >= DUPLICATE_OVERLAP:
                st.opponents.pop(UNKNOWN_PLAYER, None)

    def _viewed_level(self, scr: ScreenObservation) -> Optional[int]:
        """The viewed player's level: the dedicated field, or the older
        free-text note "对手等级 N"."""
        lvl = _to_int(scr.viewed_player_level)
        if lvl is None:
            for note in scr.notes:
                m = _OPP_LEVEL_NOTE.search(note or "")
                if m:
                    lvl = int(m.group(1))
                    break
        if lvl is not None and 1 <= lvl <= self._mech.max_level:
            return lvl
        return None

    # ---- own board ------------------------------------------------------------
    def _confirmed(self, name: str, value: int, tol: int = 0) -> bool:
        """A suspicious reading counts once the next reading of the field agrees."""
        prev = self._suspect.get(name)
        if prev is not None and abs(prev - value) <= tol:
            self._suspect.pop(name, None)
            return True
        self._suspect[name] = value
        return False

    def _stage_key(self) -> Optional[tuple[int, int]]:
        st = self._state
        return st.stage.key if st.stage is not None else None

    def _merge_gold(self, value: Any, now: float) -> None:
        """Gold cannot jump by much more than a round's income: a big rise
        (a digit misread, 42 read as 142) waits for a second reading, which
        confirms it when it is also a big rise and not above the first one by
        more than the income since (gold only goes down within a round). A
        drop to the old value with one digit lost (69 read as 6) waits too:
        any next reading below the old value confirms a drop."""
        st = self._state
        num = _to_int(value)
        if num is None or not GOLD_RANGE[0] <= num <= GOLD_RANGE[1]:
            return
        key = self._stage_key()
        if st.gold is not None and self._gold_stage is not None:
            cap = GOLD_JUMP_ROUND + GOLD_JUMP_PER_ROUND * _rounds_between(self._gold_stage, key)
            if self._mech.is_augment(st.stage):
                cap += GOLD_JUMP_ROUND  # gold augments (Invested++ gives 45)
            if num > st.gold + cap:
                prev = self._suspect.get("gold")
                income = GOLD_JUMP_PER_ROUND * _rounds_between(self._suspect_stage.get("gold"), key)
                if prev is None or num > prev + income + GOLD_CONFIRM_TOL:
                    self._suspect["gold"] = num
                    self._suspect_stage["gold"] = key
                    return
            elif self._digit_lost(st.gold, num) and "gold_drop" not in self._suspect:
                self._suspect["gold_drop"] = num
                return
        self._suspect.pop("gold", None)
        self._suspect.pop("gold_drop", None)
        st.gold = num
        st.field_age["gold"] = now
        self._gold_stage = key

    @staticmethod
    def _digit_lost(old: int, new: int) -> bool:
        """``new`` is ``old`` with one digit dropped (69 -> 6 or 9)."""
        a, b = str(old), str(new)
        return len(a) >= 2 and len(b) == len(a) - 1 and any(a[:i] + a[i + 1 :] == b for i in range(len(a)))

    def _level_plausible(self, num: int) -> bool:
        """A level reading above the tracked level: at most one level per
        round since the last level reading, and within the round no more
        than the gold spent since then can buy (augment rounds excepted:
        some augments grant XP)."""
        st = self._state
        level = st.level
        if level is None or num <= level:
            return num == level or level is None
        if num == self._level_undo:
            return True  # back to the level before the last drop: that drop was the misread
        key = self._stage_key()
        mark = self._level_mark
        if mark is None or mark[0] is None or key is None or mark[0] != key:
            # Another round (or a stage corrected since): a level per round.
            return num - level <= max(1, _rounds_between(mark[0] if mark else None, key))
        if mark is not None and mark[1] is not None and st.gold is not None and st.xp_current is not None:
            if not self._mech.is_augment(st.stage):
                cost = self._mech.gold_to_reach(level, st.xp_current, num)
                return cost is not None and mark[1] - st.gold >= cost
        return num - level <= 1

    def _merge_level(self, value: Any, now: float, trusted: bool = True) -> Optional[bool]:
        """True: applied, False: held back, None: nothing readable.

        Levels never go down: a lower reading counts once a second reading is
        lower too (the later one is taken). A rise more than one level per
        round (or more than the gold spent within the round buys) waits for
        the same reading again. ``trusted=False`` (a frame whose arena is not
        ours): the reading only confirms a pending one, it never sets one.
        """
        st = self._state
        num = _to_int(value)
        if num is None or not 1 <= num <= self._mech.max_level:
            return None
        prev = self._suspect.get("level")
        if not trusted:
            if st.level is not None and num == st.level:
                self._suspect.pop("level", None)  # the pending reading was the misread
                return True
            if prev is None or prev != num:
                return False
        elif not self._level_plausible(num):
            lower = st.level is not None and num < st.level
            if prev is None or not (prev == num or (lower and prev < st.level)):
                self._suspect["level"] = num
                return False
        self._suspect.pop("level", None)
        if st.level is not None and num != st.level:
            self._level_undo = st.level if num < st.level else None
        st.level = num
        st.field_age["level"] = now
        self._level_mark = (self._stage_key(), st.gold)
        return True

    def _hp_plausible(self, hp: int) -> bool:
        st = self._state
        if st.hp is None:
            return True
        if self._hp_undo is not None and abs(hp - self._hp_undo) <= 1 and hp > st.hp:
            return True  # back to the HP before the last drop: that drop was the misread
        if hp > st.hp + HP_RISE_MAX:
            return False  # HP never comes back in TFT (a few points from augments at most)
        rounds = max(1, _rounds_between(self._hp_stage, self._stage_key()))
        stage = st.stage.stage if st.stage is not None else None
        return st.hp - hp <= self._mech.max_round_damage(stage) * rounds

    def _merge_hp(self, row_hp: Optional[int], top_hp: Optional[int], now: float) -> None:
        """Apply our HP from this frame: our player-list row and the HP above
        our own board. When they agree they are one reading (the vision reads
        the top HP from our row); otherwise the first plausible one is taken.
        An implausible value counts only when the next frame reads it again
        (a scout frame's list row counts as that reading)."""
        st = self._state
        cands = [h for h in (row_hp, top_hp) if h is not None and HP_RANGE[0] <= h <= HP_RANGE[1]]
        if row_hp is not None and top_hp is not None and abs(row_hp - top_hp) <= 1:
            # The vision reads our top HP from our list row: the same pixels,
            # so an agreeing pair is one reading (a misread lands in both).
            cands = [row_hp]
        value: Optional[int] = None
        single = False  # taken on one plausible reading (a drop that one reading of the old HP undoes)
        if cands:
            value = next((h for h in cands if self._hp_plausible(h)), None)
            single = value is not None
            if value is None and self._confirmed("hp", cands[0], 1):
                value = cands[0]
        if value is not None:
            self._suspect.pop("hp", None)
            undo = self._hp_undo is not None and abs(value - self._hp_undo) <= 1 and st.hp is not None and value > st.hp
            self._hp_undo = st.hp if single and not undo and st.hp is not None and value < st.hp else None
            st.hp = value
            st.field_age["hp"] = now
            self._hp_stage = self._stage_key()
        if st.self_name and st.hp is not None:
            for p in st.players:
                if p.name == st.self_name:
                    p.hp = st.hp

    def _merge_hud(self, scr: ScreenObservation, now: float, trusted: bool = True, streak: bool = True) -> None:
        """The bottom HUD of our own screen: level, XP and streak.

        ``trusted=False`` (a frame whose arena is someone else's): the level
        only confirms a pending reading and the XP is not taken; ``streak``
        False skips the streak too (scout frames)."""
        st = self._state
        old_level = st.level
        level_set = self._merge_level(scr.level, now, trusted)
        if level_set is False and trusted:
            return  # the XP shown next to a rejected level belongs to that reading
        if trusted:
            xp_set = self._set_int("xp_current", scr.xp_current, (0, XP_MAX), now)
            need_set = self._set_int("xp_needed", scr.xp_needed, (0, XP_MAX), now)
        else:
            xp_set = need_set = False
        if level_set and st.level != old_level:
            self._reset_xp(xp_set, need_set)
        if streak:
            self._merge_streak(scr.streak, now)
        self._remember_xp()

    def _arena_is_ours(self, scr: ScreenObservation) -> bool:
        st = self._state
        if scr.screen_type == ScreenType.COMBAT and st.board and scr.board is not None and not self._board_is_ours(scr):
            return False  # enemy units stand on our arena during a home fight
        if scr.viewing_own_board is True:
            return True  # (a foreign arena labeled as ours was filed as scouting)
        return self._board_is_ours(scr)

    def _drop_oversized_board(self, scr: ScreenObservation) -> ScreenObservation:
        """More fielded units than the level allows (plus team size items) is
        a combat frame with enemies on our arena, not our board."""
        if scr.board is None:
            return scr
        cap = (self._state.level or self._mech.max_level) + 2
        if len([u for u in scr.board if u.name and u.name.strip()]) > cap:
            return scr.model_copy(update={"board": None})
        return scr

    def _combat_board(self, scr: ScreenObservation) -> ScreenObservation:
        """During a fight no unit joins or leaves our board (units cannot be
        placed, the dead come back) and fighters walk off their hexes, while
        surviving enemy units stand on our arena: once our board is known a
        combat frame never changes it (bench and items still apply)."""
        if scr.screen_type != ScreenType.COMBAT or not self._state.board or scr.board is None:
            return scr
        return scr.model_copy(update={"board": None})

    def _hold_foreign_looking_board(self, scr: ScreenObservation) -> bool:
        """A board called ours that shares no champion with our (sizable)
        board and names nobody: often the camera on an opponent whose HP is
        close to ours. It replaces our board only when the next frame shows
        the same board again."""
        st = self._state
        if scr.board is None or scr.viewed_player_name or len(st.board) < MIN_BOARD_FOR_HOLD:
            self._pending_board = None
            return False
        seen = _identity(self._resolve_units(scr.board, on_board=True))
        if not seen or _overlap(seen, _identity(st.board)) > 0:
            self._pending_board = None
            return False
        if self._pending_board is not None and _overlap(seen, self._pending_board) >= DUPLICATE_OVERLAP:
            self._pending_board = None
            return False
        self._pending_board = seen
        return True

    def _drop_own_snapshots(self) -> None:
        """Ownerless / HP-matched snapshots that turn out to be our own board
        (read with a wrong "not own" flag before our board was updated)."""
        st = self._state
        own = _identity(st.all_units())
        for key in [UNKNOWN_PLAYER, *sorted(self._provisional)]:
            snap = st.opponents.get(key)
            if snap is None:
                continue
            ident = _identity([*snap.board, *snap.bench])
            if ident and _containment(ident, own) >= DUPLICATE_OVERLAP:
                del st.opponents[key]
                self._provisional.discard(key)

    def _merge_own(
        self, scr: ScreenObservation, now: float, gold_before: Optional[int] = None, force: bool = False
    ) -> Optional[int]:
        """Apply a frame of our own screen. The bottom HUD (level, XP, streak)
        applies (the level only as a confirming reading when the frame shows
        a board that is not ours); the arena fields (board, bench, traits,
        item bench and the top level HP, which belongs to the viewed player)
        only when the vision said the arena is ours or the board looks like
        ours (or none is known yet), or ``force`` (our HP above the board).
        Returns the top HP to use as a reading of our HP."""
        st = self._state
        arena = self._drop_oversized_board(scr)
        ours = force or self._arena_is_ours(arena)
        if ours and not force and self._hold_foreign_looking_board(arena):
            ours = False
        self._merge_hud(scr, now, trusted=ours or arena.board is None)
        scr = self._combat_board(arena)
        if not ours:
            return None
        top_hp = _to_int(scr.hp)
        if st.self_name is None and scr.viewing_own_board is True and scr.viewed_player_name:
            name = scr.viewed_player_name.strip()
            if name:
                st.self_name = self._canonical_player(name) or name
                self._self_seen = 2  # our own board's banner: as good as two flags
                for p in st.players:
                    p.is_self = p.name == st.self_name
        if scr.board is not None:
            board = self._keep_missing_unit(self._resolve_units(scr.board, on_board=True), scr, gold_before)
            st.board = self._keep_positions(board, st.board)
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
        if scr.augments is not None and scr.viewing_own_board is True:
            st.augments = [a.strip() for a in scr.augments if a and a.strip()]
            st.field_age["augments"] = now
        if scr.board is not None or scr.bench is not None:
            self._drop_own_snapshots()
        return top_hp

    def _remember_xp(self) -> None:
        st = self._state
        if st.level is not None and st.xp_current is not None:
            self._xp_memo[st.level] = (st.xp_current, st.xp_needed, self._stage_key())

    def _reset_xp(self, xp_set: bool, need_set: bool) -> None:
        """The level changed: an XP pair read at the old level means nothing
        now. A pair read at the new level earlier this round comes back (a
        level flicker must not lose the XP progress)."""
        st = self._state
        memo = self._xp_memo.get(st.level) if st.level is not None else None
        if memo is not None and memo[2] != self._stage_key():
            memo = None
        if not xp_set:
            st.xp_current = memo[0] if memo is not None else None
            if st.xp_current is None:
                st.field_age.pop("xp_current", None)
        if not need_set:
            if memo is not None and memo[1] is not None:
                st.xp_needed = memo[1]
            else:
                st.xp_needed = self._mech.xp_to_level.get(st.level) if st.level is not None else None

    def _merge_streak(self, value: Optional[int], now: float) -> None:
        """The streak only changes with a combat result, i.e. with the round: a
        0 read within the round a known streak was read in is the placeholder
        of an unread streak, not a reset (a real value still replaces a 0)."""
        st = self._state
        num = _to_int(value)
        if num is None or not STREAK_RANGE[0] <= num <= STREAK_RANGE[1]:
            return
        key = st.stage.key if st.stage is not None else None
        if num == 0 and st.streak and key is not None and key == self._streak_stage:
            return
        st.streak = num
        st.field_age["streak"] = now
        self._streak_stage = key

    def _board_is_ours(self, scr: ScreenObservation, lenient: bool = False) -> bool:
        """Does the board on an uncertain frame look like our own? Same units,
        or mostly the same champions (stars ignored: a star-up after a roll
        is still our board). ``lenient`` (the vision said "own board"): also
        our board plus or minus a unit or two."""
        st = self._state
        if not st.board:
            return True  # nothing known yet to protect
        if scr.board is None:
            return False
        if self._same_units(scr.board, st.board):
            return True
        units = self._resolve_units(scr.board, on_board=True)
        if _overlap(self._unit_counts(units), self._unit_counts(st.board)) >= DUPLICATE_OVERLAP:
            return True
        seen, own = _identity(units), _identity(st.board)
        if _overlap(seen, own) >= DUPLICATE_OVERLAP:
            return True
        # Our board plus a unit or two (bought / leveled), or minus a unit or
        # two (benched / sold): one list holds the other.
        n_seen, n_own = sum(seen.values()), sum(own.values())
        if lenient and seen and abs(n_seen - n_own) <= 2:
            small, big = (seen, own) if n_seen <= n_own else (own, seen)
            return _containment(small, big) >= 1.0
        return False

    def _keep_missing_unit(
        self, board: list[Unit], scr: ScreenObservation, gold_before: Optional[int]
    ) -> list[Unit]:
        """A fielded unit the vision missed once: exactly one unit gone, nothing
        new, not on the bench now and no gold from selling it. It stays for
        this frame; missing again on the next frame, it is gone."""
        st = self._state
        old = st.board
        if not old or len(board) != len(old) - 1:
            self._missing_once.clear()
            return board
        old_ids = Counter((u.api_name, u.star) for u in old)
        new_ids = Counter((u.api_name, u.star) for u in board)
        gone = old_ids - new_ids
        if new_ids - old_ids or sum(gone.values()) != 1:
            self._missing_once.clear()
            return board
        key = next(iter(gone))
        if key[0].startswith("?") or key in self._missing_once:
            self._missing_once.clear()
            return board
        if scr.bench is not None and any(u.api_name == key[0] for u in self._resolve_units(scr.bench, on_board=False)):
            return board  # moved to the bench
        unit = next(u for u in old if (u.api_name, u.star) == key)
        sell = max(1, (unit.cost or 1) * unit.copies - (1 if unit.star > 1 else 0))
        if gold_before is not None and st.gold is not None and st.gold - gold_before >= sell:
            return board  # sold
        self._missing_once = {key}
        return [*board, unit.model_copy(deep=True)]

    def _drop_covered_board(self, scr: ScreenObservation) -> ScreenObservation:
        """Augment cards cover the arena: an empty or much smaller board / bench
        read there is the cards, not a sold board."""
        st = self._state
        update: dict[str, Any] = {}
        for field, own in (("board", st.board), ("bench", st.bench)):
            seen = getattr(scr, field)
            if seen is not None and own and len([u for u in seen if u.name and u.name.strip()]) * 2 < len(own):
                update[field] = None
        if scr.traits is not None and not scr.traits and st.traits:
            update["traits"] = None
        return scr.model_copy(update=update) if update else scr

    def _keep_positions(self, new: list[Unit], old: list[Unit]) -> list[Unit]:
        """Units read without a hex keep the position of the same unit last
        time, or the hex it last stood on within POS_MEMORY_ROUNDS rounds."""
        key_now = self._stage_key()
        pool: dict[tuple[str, int], list[tuple[int, int]]] = {}
        for u in old:
            if u.row is not None and u.col is not None:
                pool.setdefault((u.api_name, u.star), []).append((u.row, u.col))
        for k, spots in self._pos_memory.items():
            if k not in pool:
                pool[k] = [(r, c) for r, c, at in spots if _rounds_between(at, key_now) <= POS_MEMORY_ROUNDS]
        taken = {(u.row, u.col) for u in new if u.row is not None and u.col is not None}
        for u in new:
            if u.row is not None and u.col is not None:
                continue
            for row, col in pool.get((u.api_name, u.star), []):
                if (row, col) not in taken:
                    u.row, u.col = row, col
                    taken.add((row, col))
                    break
        placed: dict[tuple[str, int], list[tuple[int, int, Optional[tuple[int, int]]]]] = {}
        for u in new:
            if u.row is not None and u.col is not None:
                placed.setdefault((u.api_name, u.star), []).append((u.row, u.col, key_now))
        self._pos_memory.update(placed)
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
        self._shop_stage = self._stage_key()

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
