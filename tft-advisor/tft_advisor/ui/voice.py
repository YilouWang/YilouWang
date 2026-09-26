"""Optional text-to-speech of the advice (pyttsx3, Windows SAPI5 on the target PC).

``Speaker`` owns one dedicated thread: the pyttsx3 engine is created, used and
stopped on that same thread (SAPI5 / COM objects must not cross threads).

Queue policy: there is no backlog. Each message has a ``key`` ("advice",
"requests", ...) and only the latest message per key waits to be spoken; the
most urgent waiting message (lowest priority number, then oldest) goes first.
The same text is not repeated within 20 s.

Without pyttsx3 or without a working audio backend ``start()`` returns False
and ``say()`` is a no-op, so the rest of the app never has to care.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from typing import Any, Callable, Optional

from ..config import UIConfig

DUP_WINDOW_S = 20.0
STALE_AFTER_S = 30.0  # a message that waited this long is no longer worth saying
INIT_TIMEOUT_S = 10.0
MAX_TEXT = 150

_NAMED_ZH_VOICES = ("huihui", "kangkang", "yaoyao", "xiaoxiao", "xiaoyi", "yunxi", "yunyang", "tingting", "meijia")
_ZH_TOKEN = re.compile(r"(?<![a-z])(zh|cmn|yue)(?![a-z])")
_STARS = re.compile(r"[★☆]+")
_DASHES = re.compile(r"[‒-―⸺⸻]+")
_SPACES = re.compile(r"\s+")


def _voice_blob(voice: Any) -> str:
    parts = [str(getattr(voice, "id", "") or ""), str(getattr(voice, "name", "") or "")]
    for lang in getattr(voice, "languages", None) or []:
        if isinstance(lang, bytes):
            lang = lang.decode("utf-8", "ignore")
        parts.append(str(lang))
    return " ".join(parts).lower()


def voice_score(voice: Any) -> int:
    """How Chinese a pyttsx3 voice looks: 3 named zh voice, 2 'Chinese', 1 zh code, 0 none."""
    blob = _voice_blob(voice)
    if any(n in blob for n in _NAMED_ZH_VOICES):
        return 3
    if "chinese" in blob or "中文" in blob or "mandarin" in blob:
        return 2
    if _ZH_TOKEN.search(blob.replace("_", " ").replace("-", " ")):
        return 1
    return 0


def pick_voice(voices: Any) -> Optional[Any]:
    """The most Chinese-looking voice, or None when there is none."""
    best, best_score = None, 0
    for v in list(voices or []):
        s = voice_score(v)
        if s > best_score:
            best, best_score = v, s
    return best


def clean_text(text: Any) -> str:
    """Make advice text speakable: stars become '2星', no dashes, one line, bounded."""
    if text is None:
        return ""
    s = str(text)
    s = _STARS.sub(lambda m: f"{len(m.group(0))}星", s)
    s = _DASHES.sub("，", s)
    s = _SPACES.sub(" ", s).strip()
    return s[:MAX_TEXT]


def _com_init() -> bool:
    """Initialise COM on this thread (SAPI5 on Windows). Returns True if we must uninit."""
    if not _is_windows():
        return False
    try:
        import pythoncom  # type: ignore[import-not-found]

        pythoncom.CoInitialize()
        return True
    except Exception:
        pass
    try:
        import comtypes  # type: ignore[import-not-found]

        comtypes.CoInitialize()
        return True
    except Exception:
        return False


def _com_uninit() -> None:
    try:
        import pythoncom  # type: ignore[import-not-found]

        pythoncom.CoUninitialize()
        return
    except Exception:
        pass
    try:
        import comtypes  # type: ignore[import-not-found]

        comtypes.CoUninitialize()
    except Exception:
        pass


def _is_windows() -> bool:
    import sys

    return sys.platform == "win32"


def _as_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            return {}
    return obj if isinstance(obj, dict) else {}


class Speaker:
    """Speak short Chinese messages on a dedicated TTS thread."""

    def __init__(self, cfg: UIConfig, log: Callable[[str], Any] = print, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg
        self.log = log
        self.clock = clock
        self.voice_name: Optional[str] = None
        self.spoken: deque[str] = deque(maxlen=50)  # recently spoken texts (diagnostics / tests)
        self._cond = threading.Condition()
        self._pending: dict[str, tuple[str, int, int, float]] = {}  # key -> (text, priority, seq, queued_at)
        self._seq = 0
        self._gen = 0  # bumped on every start(): a stale thread from an old start() exits
        self._last_spoken: tuple[str, float] = ("", -1e18)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._stopping = False
        self._ready = threading.Event()
        self._init_ok = False
        self._init_error: Optional[BaseException] = None
        self._unsubs: list[Callable[[], None]] = []
        self._last_advice_text = ""
        self._seen_requests: set[str] = set()

    # ------------------------------------------------------------------ public
    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> bool:
        """Start the TTS thread. False (and ``say`` stays a no-op) when TTS is unavailable."""
        if self._running:
            return True
        try:
            import pyttsx3  # noqa: F401  (lazy optional dependency)
        except Exception:
            self.log("语音播报不可用：没有安装 pyttsx3（pip install pyttsx3）")
            return False
        with self._cond:
            self._stopping = False
            self._pending.clear()
            self._gen += 1
            gen = self._gen
        self._ready.clear()
        self._init_ok = False
        self._init_error = None
        thread = threading.Thread(target=self._run, args=(gen,), name="voice", daemon=True)
        self._thread = thread
        thread.start()
        if not self._ready.wait(INIT_TIMEOUT_S):
            self.log("语音播报不可用：语音引擎初始化超时")
            self._halt_thread()
            return False
        if not self._init_ok:
            self.log(f"语音播报不可用：{self._init_error or '没有可用的语音引擎'}")
            self._halt_thread()
            return False
        self._running = True
        if self.voice_name:
            self.log(f"语音播报已开启（{self.voice_name}）")
        else:
            self.log("语音播报已开启，但没有找到中文语音：请在 Windows 设置 > 时间和语言 > 语音 中添加中文语音包")
        return True

    def say(self, text: Any, priority: int = 2, key: str = "default") -> bool:
        """Queue ``text``. Returns True if it was queued, False if dropped or disabled."""
        if not self._running:
            return False
        text = clean_text(text)
        if not text:
            return False
        try:
            priority = max(1, min(3, int(priority)))
        except (TypeError, ValueError):
            priority = 2
        now = self.clock()
        with self._cond:
            if self._stopping:
                return False
            last_text, last_ts = self._last_spoken
            if text == last_text and now - last_ts < DUP_WINDOW_S:
                return False
            if any(p[0] == text for p in self._pending.values()):
                return False
            self._seq += 1
            self._pending[str(key)] = (text, priority, self._seq, now)  # replaces older message of this key
            self._cond.notify_all()
        return True

    def stop(self) -> None:
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._unsubs.clear()
        self._halt_thread()
        self._running = False

    def attach(self, bus: Any) -> Callable[[], None]:
        """Speak new advice (headline + first priority-1 action) and new human requests."""
        self._unsubs.append(bus.subscribe("advice", self._on_advice))
        self._unsubs.append(bus.subscribe("requests", self._on_requests))
        return self.stop

    # --------------------------------------------------------------- bus hooks
    def _on_advice(self, _topic: str, payload: Any) -> None:
        adv = _as_dict(payload)
        text = advice_speech(adv)
        if not text or text == self._last_advice_text:
            return
        self._last_advice_text = text
        has_p1 = any(_prio(a) == 1 for a in adv.get("actions") or [] if isinstance(a, dict))
        self.say(text, priority=1 if has_p1 else 2, key="advice")

    def _on_requests(self, _topic: str, payload: Any) -> None:
        reqs = [_as_dict(r) for r in (payload or []) if r is not None] if isinstance(payload, (list, tuple)) else []
        fresh = []
        for r in reqs:
            rid = str(r.get("id") or r.get("text") or "")
            if not rid or rid in self._seen_requests:
                continue
            if len(self._seen_requests) > 500:
                self._seen_requests.clear()
            self._seen_requests.add(rid)
            text = str(r.get("text") or "").strip()
            if not text and r.get("target_player"):
                text = f"请切到 {r['target_player']} 的棋盘后按 F7"
            if text:
                fresh.append(text)
        if fresh:
            self.say("。".join(fresh[:2]), priority=2, key="requests")

    # ------------------------------------------------------------------ thread
    def _halt_thread(self) -> None:
        with self._cond:
            self._stopping = True
            self._pending.clear()
            self._cond.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None

    def _next_message(self, gen: int) -> Optional[str]:
        """Block until a message is due (None when stopping). Caller must not hold the lock."""
        with self._cond:
            while not self._stopping and gen == self._gen:
                now = self.clock()
                for key in [k for k, p in self._pending.items() if now - p[3] > STALE_AFTER_S]:
                    del self._pending[key]
                if self._pending:
                    key = min(self._pending, key=lambda k: (self._pending[k][1], self._pending[k][2]))
                    text = self._pending.pop(key)[0]
                    self._last_spoken = (text, now)
                    return text
                self._cond.wait(0.5)
            return None

    def _run(self, gen: int) -> None:
        com = _com_init()
        engine = None
        try:
            try:
                import pyttsx3

                engine = pyttsx3.init()
                try:
                    engine.setProperty("rate", int(self.cfg.voice_rate))
                except Exception:
                    pass
                try:
                    voice = pick_voice(engine.getProperty("voices"))
                except Exception:
                    voice = None
                if voice is not None:
                    engine.setProperty("voice", voice.id)
                    self.voice_name = str(getattr(voice, "name", None) or voice.id)
                self._init_ok = True
            except Exception as exc:  # no espeak / SAPI / audio device
                self._init_error = exc
                return
            finally:
                self._ready.set()

            errors = 0
            while True:
                text = self._next_message(gen)
                if text is None:
                    break
                try:
                    engine.say(text)
                    engine.runAndWait()
                    self.spoken.append(text)
                    errors = 0
                except Exception as exc:
                    errors += 1
                    try:
                        engine.endLoop()  # "run loop already started" recovery
                    except Exception:
                        pass
                    if errors == 1:
                        self.log(f"语音播报出错：{exc!r}")
                    if errors >= 5:
                        self.log("语音播报连续出错，已关闭")
                        break
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
            if gen == self._gen:
                self._running = False  # say() becomes a no-op once the thread is gone
            if com:
                _com_uninit()


def _prio(action: Any) -> int:
    try:
        return int(action.get("priority", 2))
    except (TypeError, ValueError, AttributeError):
        return 2


def advice_speech(adv: dict[str, Any]) -> str:
    """Headline plus the first priority-1 action (if not already in the headline)."""
    headline = str(adv.get("headline") or "").strip()
    if not headline:
        return ""
    actions = [a for a in adv.get("actions") or [] if isinstance(a, dict) and str(a.get("text") or "").strip()]
    first = next((str(a["text"]).strip() for a in actions if _prio(a) == 1), "")
    if first and first not in headline:
        return f"{headline}。{first}"
    return headline


__all__ = ["Speaker", "advice_speech", "clean_text", "pick_voice", "voice_score"]
