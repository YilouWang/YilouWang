"""Tiny thread-safe publish/subscribe bus between the pipeline and the UIs.

Topics used across the app:
  "state"     GameState.model_dump(mode="json")
  "analysis"  Analysis.model_dump(mode="json")
  "advice"    Advice.model_dump(mode="json")
  "requests"  list of ScoutRequest dicts (current open human requests)
  "status"    {"auto": bool, "busy": bool, "last_error": str|None, "calls": int,
               "last_error_ts": float|None (when last_error happened; a later
               success of the same source clears both),
               "thinking": bool (Claude strategy call running), "auto_paused": bool
               (game window missing / not in the foreground), "capture_error": str|None,
               "comp_hint": str (current target comp, "" = auto), "perceiver": str
               ("manual" = no way to read the screen), "hotkeys": {...}, ...}
               "calls" counts Claude calls since the program started (not per game).
  "log"       {"level": "info"|"warn"|"error", "text": str, "ts": float}
  "answer"    {"question": str, "answer": str, "ts": float}   (reply to an "ask" command)

Commands flow the other way (UI -> app) through ``command`` topic:
  {"cmd": "analyze"}                      full analysis now
  {"cmd": "scout", "player": str|None}     current screen shows another player's board
  {"cmd": "shop"}                          read the shop and say what to buy
  {"cmd": "toggle_auto"}
  {"cmd": "ask", "question": str}          free-form question to the strategist
  {"cmd": "dismiss", "id": str}            close a human request
  {"cmd": "set_field", "field": str, "value": any}   manual correction (gold, level, hp, ...)
  {"cmd": "set_comp", "comp": str}         the comp the player wants to go for
  {"cmd": "new_game"}                      reset tracker
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any, Callable

Handler = Callable[[str, Any], None]


class EventBus:
    def __init__(self, log_size: int = 200) -> None:
        self._lock = threading.RLock()
        self._handlers: dict[str, list[Handler]] = {}
        self._latest: dict[str, Any] = {}
        self._log: deque[dict[str, Any]] = deque(maxlen=log_size)
        self._queues: list[queue.Queue[tuple[str, Any]]] = []

    # ----- publish / subscribe -------------------------------------------------
    def publish(self, topic: str, payload: Any) -> None:
        with self._lock:
            if topic == "log":
                self._log.append(payload)
            else:
                self._latest[topic] = payload
            handlers = list(self._handlers.get(topic, ())) + list(self._handlers.get("*", ()))
            queues = list(self._queues)
        for h in handlers:
            try:
                h(topic, payload)
            except Exception:  # a broken UI must never kill the pipeline
                pass
        for q in queues:
            try:
                q.put_nowait((topic, payload))
            except queue.Full:
                pass

    def subscribe(self, topic: str, handler: Handler) -> Callable[[], None]:
        with self._lock:
            self._handlers.setdefault(topic, []).append(handler)

        def unsubscribe() -> None:
            with self._lock:
                lst = self._handlers.get(topic, [])
                if handler in lst:
                    lst.remove(handler)

        return unsubscribe

    def open_queue(self, maxsize: int = 256) -> queue.Queue[tuple[str, Any]]:
        """A queue receiving every event (used by SSE connections)."""
        q: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._queues.append(q)
        return q

    def close_queue(self, q: queue.Queue[tuple[str, Any]]) -> None:
        with self._lock:
            if q in self._queues:
                self._queues.remove(q)

    # ----- snapshots -------------------------------------------------------------
    def latest(self, topic: str, default: Any = None) -> Any:
        with self._lock:
            return self._latest.get(topic, default)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            snap = dict(self._latest)
            snap["log"] = list(self._log)
            return snap

    # ----- helpers ---------------------------------------------------------------
    def log(self, text: str, level: str = "info") -> None:
        self.publish("log", {"level": level, "text": text, "ts": time.time()})

    def command(self, cmd: str, **kwargs: Any) -> None:
        self.publish("command", {"cmd": cmd, **kwargs})
