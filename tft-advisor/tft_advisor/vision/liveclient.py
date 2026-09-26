"""Riot's local Live Client Data API (read-only HTTP on the player's own machine).

While a match runs, the game client serves ``https://127.0.0.1:2999/liveclientdata/*``.
It is the same interface overlays and stream tools use: passive, documented
by Riot, no memory reading. For TFT it returns the League of Legends JSON shape
with very little TFT data, so we only use what is known to be reliable:

* ``gameData.gameMode == "TFT"``: we are in a TFT match,
* ``allPlayers`` names and ``activePlayer`` identity: exact player names and
  which one is the local player (great for matching the HUD player list),
* ``gameData.gameTime``.

``activePlayer.currentGold`` is NOT the TFT gold (it grows like LoL passive gold,
see RiotGames/developer-relations#865), so it is ignored unless ``trust_gold``
is set. ``activePlayer.level`` is unverified for TFT and ignored unless
``trust_level`` is set.
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.parse
import urllib.request
from typing import Any, Optional

from ..models import Observation, PlayerObs, ScreenObservation
from .base import PerceptionError, PerceptionHint

DEFAULT_BASE = "https://127.0.0.1:2999"
ALL_GAME_DATA = "/liveclientdata/allgamedata"
_MAX_BYTES = 8_000_000


def _loopback_insecure_context() -> ssl.SSLContext:
    # The game client serves this API on 127.0.0.1 with a self-signed certificate
    # (issued by Riot's own root, not for the hostname "127.0.0.1"), so normal
    # verification always fails. Traffic never leaves this machine, so skipping
    # verification here is safe. This context is used ONLY for 127.0.0.1.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """The Live Client API never redirects. Following a redirect would carry the
    unverified TLS context (meant for 127.0.0.1 only) to another host, so any 3xx
    becomes an error (and ``fetch`` returns None)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401, ANN001
        return None


def _player_display_name(p: dict[str, Any]) -> Optional[str]:
    for key in ("riotIdGameName", "riotId", "summonerName"):
        val = p.get(key)
        if isinstance(val, str) and val.strip():
            name = val.strip()
            # Since Riot IDs, summonerName is often "GameName#TAG" too; the HUD shows only GameName.
            if "#" in name:
                name = name.split("#", 1)[0].strip()
            if name:
                return name
    return None


def _full_riot_id(p: dict[str, Any]) -> Optional[str]:
    rid = p.get("riotId")
    if isinstance(rid, str) and "#" in rid:
        return rid.strip().lower()
    name, tag = p.get("riotIdGameName"), p.get("riotIdTagLine")
    if isinstance(name, str) and isinstance(tag, str) and name.strip() and tag.strip():
        return f"{name.strip()}#{tag.strip()}".lower()
    return None


def _identity_keys(p: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for key in ("riotId", "riotIdGameName", "summonerName"):
        val = p.get(key)
        if isinstance(val, str) and val.strip():
            keys.add(val.strip().lower())
            if "#" in val:
                keys.add(val.split("#", 1)[0].strip().lower())
    return keys


def _same_player(active: dict[str, Any], p: dict[str, Any]) -> bool:
    a_full, p_full = _full_riot_id(active), _full_riot_id(p)
    if a_full and p_full:  # full Riot IDs are unique: never fall back to bare names
        return a_full == p_full
    return bool(_identity_keys(active) & _identity_keys(p))


class LiveClient:
    """Tiny poller for the Live Client Data API. ``fetch`` never raises."""

    def __init__(self, base: str = DEFAULT_BASE, timeout: float = 0.5, trust_level: bool = False,
                 trust_gold: bool = False, name: str = "liveclient") -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.trust_level = trust_level
        self.trust_gold = trust_gold
        self.name = name
        parsed = urllib.parse.urlsplit(self.base)
        handlers: list[Any] = [urllib.request.ProxyHandler({}), _NoRedirect()]  # never route localhost through a proxy
        if parsed.scheme == "https" and parsed.hostname == "127.0.0.1":
            handlers.append(urllib.request.HTTPSHandler(context=_loopback_insecure_context()))
        self._opener = urllib.request.build_opener(*handlers)

    # ---- transport ----------------------------------------------------------
    def fetch(self, path: str = ALL_GAME_DATA) -> Optional[dict[str, Any]]:
        """GET ``path`` and return the JSON object, or ``None`` on any problem (not running, 404, bad JSON)."""
        url = self.base + (path if path.startswith("/") else "/" + path)
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with self._opener.open(req, timeout=self.timeout) as resp:
                if getattr(resp, "status", 200) != 200:
                    return None
                payload = resp.read(_MAX_BYTES)
            data = json.loads(payload.decode("utf-8", errors="replace"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def available(self) -> bool:
        return self.fetch() is not None

    # ---- parsing ------------------------------------------------------------
    @staticmethod
    def is_tft(data: Optional[dict[str, Any]]) -> bool:
        try:
            mode = (data or {}).get("gameData", {}).get("gameMode")
        except AttributeError:
            return False
        return isinstance(mode, str) and mode.strip().upper() == "TFT"

    @staticmethod
    def game_time(data: Optional[dict[str, Any]]) -> Optional[float]:
        try:
            value = (data or {}).get("gameData", {}).get("gameTime")
            return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
        except (AttributeError, TypeError, ValueError):
            return None

    @staticmethod
    def self_name(data: Optional[dict[str, Any]]) -> Optional[str]:
        active = (data or {}).get("activePlayer") if isinstance(data, dict) else None
        return _player_display_name(active) if isinstance(active, dict) else None

    def to_screen_observation(self, data: Optional[dict[str, Any]]) -> ScreenObservation:
        """Whatever is reliable in ``data`` as a partial observation (unknown fields are ignored)."""
        screen = ScreenObservation()
        if not isinstance(data, dict):
            return screen
        notes: list[str] = []
        active = data.get("activePlayer") if isinstance(data.get("activePlayer"), dict) else {}

        raw_players = data.get("allPlayers")
        if isinstance(raw_players, list):
            players: list[PlayerObs] = []
            seen: set[str] = set()
            for p in raw_players:
                if not isinstance(p, dict):
                    continue
                name = _player_display_name(p)
                if not name or name in seen:
                    continue
                seen.add(name)
                is_self = bool(active) and _same_player(active, p)
                players.append(PlayerObs(name=name, hp=None, is_self=is_self))
            if players:
                selves = [p for p in players if p.is_self]
                for extra in selves[1:]:  # at most one local player
                    extra.is_self = False
                if not selves:
                    mine = self.self_name(data)
                    for p in players:
                        if mine and p.name == mine:
                            p.is_self = True
                            break
                screen.players = players

        if self.trust_level:
            level = active.get("level")
            if isinstance(level, int) and not isinstance(level, bool) and 1 <= level <= 10:
                screen.level = level
        if self.trust_gold:
            gold = active.get("currentGold")
            if isinstance(gold, (int, float)) and not isinstance(gold, bool) and 0 <= gold <= 300:
                screen.gold = int(gold)

        gt = self.game_time(data)
        if gt is not None:
            notes.append(f"本地实时接口: 对局时间 {int(gt)} 秒")
        screen.notes = notes
        return screen

    # ---- perceiver-style API -----------------------------------------------
    def observe(self, purpose: str = "auto") -> Optional[Observation]:
        """Fetch + parse in one go; ``None`` when no TFT match is running."""
        started = time.monotonic()
        data = self.fetch()
        if data is None or not self.is_tft(data):
            return None
        return Observation(
            screen=self.to_screen_observation(data),
            source="liveclient",
            purpose=purpose or "auto",
            latency_s=round(time.monotonic() - started, 3),
        )

    def perceive(self, image: Any = None, purpose: str = "auto", hint: Optional[PerceptionHint] = None) -> Observation:
        """Perceiver protocol (the image is ignored)."""
        obs = self.observe(purpose)
        if obs is None:
            raise PerceptionError("本地实时接口不可用 (未在云顶对局中)")
        return obs


__all__ = ["ALL_GAME_DATA", "DEFAULT_BASE", "LiveClient"]
