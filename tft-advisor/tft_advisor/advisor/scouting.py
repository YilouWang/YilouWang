"""Human-in-the-loop scouting: ask the player to open other players' boards.

The advisor cannot see other boards on its own. ``ScoutPlanner.plan`` decides
which opponents are worth a look (contest check on our comp / carry, strong
high-HP players, boards never recorded) and phrases a short Chinese request.
The player clicks that player in the list and presses the scout hotkey; the
frame becomes an ``OpponentSnapshot`` and the request closes itself.

Policy note: we deliberately never predict the next opponent (Riot's
third-party tool policy forbids it). Targets are chosen only from the public
player list and already recorded boards.
"""

from __future__ import annotations

import difflib
import hashlib
import threading
import time
from typing import Callable, Optional

from ..data.mechanics import Mechanics
from ..data.setdata import normalize_name
from ..models import Analysis, GameState, OpponentSnapshot, ScoutRequest, ScreenType, StageRound, Unit
from .rules import find_carry

STAGE1_ROUNDS = 4
ROUNDS_PER_STAGE = 7
FIRST_SCOUT_ROUND = (2, 5)
UNKNOWN_BOARD_BONUS = 15

REQUEST_TEXT = "请点开右侧玩家列表里「{name}」的棋盘，然后按 {hotkey} 记录（看完点自己头像回来）"

_NO_NEW_REQUEST_SCREENS = {ScreenType.CAROUSEL, ScreenType.AUGMENT_SELECT, ScreenType.LOADING, ScreenType.POST_GAME}


def round_index(sr: StageRound) -> int:
    """Monotonic round counter: 1-1 -> 0, 2-1 -> 4, 3-1 -> 11, ..."""
    if sr.stage <= 1:
        return sr.round - 1
    return STAGE1_ROUNDS + (sr.stage - 2) * ROUNDS_PER_STAGE + (sr.round - 1)


def _norm(name: Optional[str]) -> str:
    return normalize_name(name or "")


def _name_in(key: str, names: set[str]) -> bool:
    if not key:
        return False
    return key in names or bool(difflib.get_close_matches(key, list(names), n=1, cutoff=0.85))


def request_id(stage: int, player: str) -> str:
    """Stable id per (stage number, player)."""
    digest = hashlib.sha1(_norm(player).encode("utf-8")).hexdigest()[:10]
    return f"scout-{stage}-{digest}"


def _unit_keys(u: Unit) -> set[str]:
    keys = {_norm(u.name), _norm(u.api_name), _norm(u.api_name.split("_", 1)[-1])}
    keys.discard("")
    return keys


class ScoutPlanner:
    def __init__(
        self,
        max_per_stage: int = 2,
        hotkey: str = "F7",
        clock: Callable[[], float] = time.time,
        mech: Optional[Mechanics] = None,
        expire_after_rounds: int = 2,
        stale_rounds: int = ROUNDS_PER_STAGE,
        stale_seconds: float = 300.0,
    ) -> None:
        self.max_per_stage = max(0, int(max_per_stage))
        self.hotkey = hotkey
        self.clock = clock
        self.carousel_round = mech.carousel_round if mech is not None else 4
        self.expire_after_rounds = max(1, int(expire_after_rounds))
        self.stale_rounds = max(1, int(stale_rounds))
        self.stale_seconds = stale_seconds
        self._lock = threading.RLock()
        self.reset()

    # --------------------------------------------------------------- helpers
    def reset(self) -> None:
        with self._lock:
            self._open: dict[str, ScoutRequest] = {}
            self._open_meta: dict[str, tuple[int, str]] = {}  # id -> (round index, normalized player)
            self._requested: dict[int, set[str]] = {}  # stage number -> normalized players asked
            self._game_id: Optional[str] = None
            self._last_index: Optional[int] = None

    def open_requests(self) -> list[ScoutRequest]:
        with self._lock:
            return [r.model_copy() for r in self._open.values()]

    def dismiss(self, request_id: str) -> bool:
        """Close a request (the player does not want to do it). It still counts
        toward the stage quota and is not asked again this stage."""
        with self._lock:
            self._open_meta.pop(request_id, None)
            return self._open.pop(request_id, None) is not None

    def mark_scouted(self, player: Optional[str]) -> None:
        """Close open requests for ``player`` right after a scout capture."""
        key = _norm(player)
        if not key:
            return
        with self._lock:
            for rid, (_, who) in list(self._open_meta.items()):
                if who == key:
                    self._open.pop(rid, None)
                    self._open_meta.pop(rid, None)

    @staticmethod
    def _find_snapshot(state: GameState, player: str) -> Optional[OpponentSnapshot]:
        key = _norm(player)
        if not key:
            return None
        by_norm = {_norm(name): snap for name, snap in state.opponents.items()}
        if key in by_norm:
            return by_norm[key]
        # Fuzzy match absorbs OCR noise, but a snapshot whose name exactly
        # matches another listed player belongs to that player ("Player1" must
        # never pick up "Player2"'s board).
        others = {_norm(p.name) for p in state.players} - {key}
        pool = [n for n in by_norm if n not in others]
        match = difflib.get_close_matches(key, pool, n=1, cutoff=0.85)
        return by_norm[match[0]] if match else None

    def _snapshot_index(self, snap: OpponentSnapshot) -> Optional[int]:
        sr = StageRound.parse(snap.stage) if snap.stage else None
        return round_index(sr) if sr is not None else None

    def _is_stale(self, snap: Optional[OpponentSnapshot], now_index: int) -> bool:
        if snap is None:
            return True
        idx = self._snapshot_index(snap)
        if idx is not None:
            return now_index - idx >= self.stale_rounds
        return self.clock() - (snap.captured_at or 0.0) >= self.stale_seconds

    def _scouted_this_stage(self, snap: Optional[OpponentSnapshot], stage: int) -> bool:
        if snap is None or not snap.stage:
            return False
        sr = StageRound.parse(snap.stage)
        return sr is not None and sr.stage == stage

    # ------------------------------------------------------------------ plan
    def plan(self, state: GameState, analysis: Analysis) -> list[ScoutRequest]:
        """Update and return the open requests (new ones included)."""
        with self._lock:
            sr = state.stage
            self._maybe_new_game(state)
            if sr is None:
                return self.open_requests()
            now_index = round_index(sr)
            self._last_index = now_index
            self._expire(state, now_index)

            if self._may_request(state, sr):
                self._create(state, analysis, sr, now_index)
            return self.open_requests()

    def _maybe_new_game(self, state: GameState) -> None:
        if state.game_id and self._game_id is not None and state.game_id != self._game_id:
            self.reset()
        if state.game_id:
            self._game_id = state.game_id
        if state.stage is not None and self._last_index is not None:
            # The round counter jumping far back means a new game started.
            if round_index(state.stage) + 3 < self._last_index:
                gid = self._game_id
                self.reset()
                self._game_id = gid

    def _expire(self, state: GameState, now_index: int) -> None:
        alive = {_norm(p.name) for p in state.alive_players()}
        known = {_norm(p.name) for p in state.players}
        for rid, req in list(self._open.items()):
            created_index, who = self._open_meta.get(rid, (now_index, _norm(req.target_player)))
            expired = now_index - created_index >= self.expire_after_rounds
            snap = self._find_snapshot(state, req.target_player or "")
            if snap is not None and not expired:
                idx = self._snapshot_index(snap)
                if idx is not None:
                    expired = idx >= created_index
                else:
                    expired = (snap.captured_at or 0.0) >= req.created_at
            if not expired and known:
                if who in known:
                    # Listed under the exact name: dead when not alive.
                    expired = who not in alive
                elif not _name_in(who, alive):
                    # Player left the list (fuzzy: OCR noise must not close requests).
                    expired = True
            if expired:
                self._open.pop(rid, None)
                self._open_meta.pop(rid, None)

    def _may_request(self, state: GameState, sr: StageRound) -> bool:
        if self.max_per_stage <= 0 or sr.stage <= 1:
            return False
        if sr.round == self.carousel_round:
            return False
        if state.screen_type in _NO_NEW_REQUEST_SCREENS:
            return False
        return sr.key >= FIRST_SCOUT_ROUND

    def _targets(self, state: GameState, analysis: Analysis) -> tuple[set[str], set[str], set[str]]:
        """(comp unit keys, carry keys, players flagged as contesting)."""
        comp_keys: set[str] = set()
        carry_keys: set[str] = set()
        contested: set[str] = set()
        comp = analysis.comps[0] if analysis.comps else None
        if comp is not None:
            comp_keys |= {_norm(n) for n in comp.core_units}
            if comp.carry:
                carry_keys.add(_norm(comp.carry))
            contested |= {_norm(p) for p in comp.contested_by}
        if not carry_keys and state.board:
            best = find_carry(state, Analysis())  # most damage items; itemized tanks do not count
            if best is not None:
                carry_keys |= _unit_keys(best)
        comp_keys |= carry_keys
        comp_keys.discard("")
        carry_keys.discard("")
        return comp_keys, carry_keys, contested

    def _create(self, state: GameState, analysis: Analysis, sr: StageRound, now_index: int) -> None:
        asked = self._requested.setdefault(sr.stage, set())
        quota = self.max_per_stage - len(asked)
        if quota <= 0:
            return
        self_key = _norm(state.self_name)
        comp_keys, carry_keys, contested = self._targets(state, analysis)

        candidates = []
        for p in state.alive_players():
            key = _norm(p.name)
            if not key or p.is_self or (self_key and key == self_key) or key in asked:
                continue
            snap = self._find_snapshot(state, p.name)
            if self._scouted_this_stage(snap, sr.stage) or not self._is_stale(snap, now_index):
                continue
            hits: list[str] = []
            holds_carry = False
            if snap is not None:
                for u in [*snap.board, *snap.bench]:
                    keys = _unit_keys(u)
                    if keys & comp_keys:
                        label = u.name or u.api_name
                        if label not in hits:
                            hits.append(label)
                    if keys & carry_keys:
                        holds_carry = True
            contest = len(hits) + (2 if holds_carry else 0) + (3 if key in contested else 0)
            # (a) contest dominates; then (b) HP, where (c) a never-recorded
            # board is worth UNKNOWN_BOARD_BONUS HP (unknown HP counts as 50).
            hp = p.hp if p.hp is not None else 50
            value = hp + (UNKNOWN_BOARD_BONUS if snap is None else 0)
            sort_key = (-contest, -value, key)
            candidates.append((sort_key, p, snap, hits, key in contested))

        candidates.sort(key=lambda c: c[0])
        for _, p, snap, hits, flagged in candidates[:quota]:
            rid = request_id(sr.stage, p.name)
            req = ScoutRequest(
                id=rid,
                text=REQUEST_TEXT.format(name=p.name, hotkey=self.hotkey),
                target_player=p.name,
                reason=self._reason(p.hp, snap, hits, flagged),
                created_at=self.clock(),
                stage=str(sr),
            )
            self._open[rid] = req
            self._open_meta[rid] = (now_index, _norm(p.name))
            asked.add(_norm(p.name))

    @staticmethod
    def _reason(hp: Optional[int], snap: Optional[OpponentSnapshot], hits: list[str], flagged: bool) -> str:
        if hits:
            reason = f"检查是否有人和你抢 {'、'.join(hits[:2])}"
        elif flagged:
            reason = "这名玩家可能在玩和你相同的阵容，确认一下"
        elif snap is None:
            reason = "还没记录过这个棋盘，记录后搜牌概率更准"
        else:
            reason = f"上次记录在 {snap.stage or '较早'}，更新一下牌池信息"
        if hp is not None and hp >= 0:
            reason += f"（血量 {hp}）"
        return reason
