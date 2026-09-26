"""Global hotkeys via Win32 ``RegisterHotKey`` (ctypes, no extra dependencies).

``RegisterHotKey`` only *listens* for a key combination system wide: it never
sends input to the game. Hotkeys are registered on a dedicated thread that
runs the ``GetMessageW`` loop (Win32 requires both on the same thread);
``WM_HOTKEY`` messages are handed to a small worker thread so a slow callback
never blocks the message loop. Callback exceptions are logged, never fatal.

On other platforms ``start()`` logs a hint and returns False (use the
dashboard buttons instead, e.g. from a phone).
"""

from __future__ import annotations

import queue
import sys
import threading
from typing import Any, Callable, Optional

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
WM_USER = 0x0400
PM_NOREMOVE = 0x0000

_MODIFIERS: dict[str, int] = {
    "ctrl": MOD_CONTROL,
    "control": MOD_CONTROL,
    "ctl": MOD_CONTROL,
    "alt": MOD_ALT,
    "shift": MOD_SHIFT,
    "win": MOD_WIN,
    "windows": MOD_WIN,
    "super": MOD_WIN,
    "meta": MOD_WIN,
}

_NAMED_KEYS: dict[str, int] = {
    "backspace": 0x08,
    "tab": 0x09,
    "enter": 0x0D,
    "return": 0x0D,
    "pause": 0x13,
    "esc": 0x1B,
    "escape": 0x1B,
    "space": 0x20,
    "pageup": 0x21,
    "pgup": 0x21,
    "pagedown": 0x22,
    "pgdn": 0x22,
    "end": 0x23,
    "home": 0x24,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "printscreen": 0x2C,
    "prtsc": 0x2C,
    "insert": 0x2D,
    "ins": 0x2D,
    "delete": 0x2E,
    "del": 0x2E,
    "multiply": 0x6A,
    "add": 0x6B,
    "subtract": 0x6D,
    "decimal": 0x6E,
    "divide": 0x6F,
    "scrolllock": 0x91,
    "semicolon": 0xBA,
    ";": 0xBA,
    "plus": 0xBB,
    "equals": 0xBB,
    "=": 0xBB,
    "comma": 0xBC,
    ",": 0xBC,
    "minus": 0xBD,
    "-": 0xBD,
    "period": 0xBE,
    ".": 0xBE,
    "slash": 0xBF,
    "/": 0xBF,
    "backquote": 0xC0,
    "grave": 0xC0,
    "`": 0xC0,
    "[": 0xDB,
    "backslash": 0xDC,
    "\\": 0xDC,
    "]": 0xDD,
    "quote": 0xDE,
    "'": 0xDE,
}
for _i in range(10):
    _NAMED_KEYS[f"num{_i}"] = 0x60 + _i
    _NAMED_KEYS[f"numpad{_i}"] = 0x60 + _i


def parse_hotkey(text: str) -> tuple[int, int]:
    """``'F6'`` / ``'ctrl+shift+a'`` / ``'alt+F1'`` -> ``(modifiers, virtual_key)``.

    Case and spaces are ignored. ``modifiers`` is a combination of
    ``MOD_ALT | MOD_CONTROL | MOD_SHIFT | MOD_WIN`` (``MOD_NOREPEAT`` is added
    at registration). Keys: F1..F24, A..Z, 0..9, numpad0..9 and common named
    keys (space, tab, home, pageup, ...). Raises ValueError otherwise.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"empty hotkey: {text!r}")
    parts = [p.strip().lower() for p in text.strip().split("+")]
    if any(not p for p in parts):
        raise ValueError(f"malformed hotkey {text!r} (use e.g. 'ctrl+shift+a')")
    modifiers = 0
    key: Optional[int] = None
    for part in parts:
        if part in _MODIFIERS:
            modifiers |= _MODIFIERS[part]
            continue
        if key is not None:
            raise ValueError(f"hotkey {text!r} has more than one non-modifier key")
        key = _vk_for(part, text)
    if key is None:
        raise ValueError(f"hotkey {text!r} has no key besides modifiers")
    return modifiers, key


def _vk_for(part: str, original: str) -> int:
    if len(part) >= 2 and part[0] == "f" and part[1:].isascii() and part[1:].isdigit():
        n = int(part[1:])
        if 1 <= n <= 24:
            return 0x70 + n - 1  # VK_F1 = 0x70 ... VK_F24 = 0x87
        raise ValueError(f"unknown function key {part!r} in {original!r} (F1..F24)")
    if len(part) == 1 and ("a" <= part <= "z" or "0" <= part <= "9"):
        return ord(part.upper())
    if part in _NAMED_KEYS:
        return _NAMED_KEYS[part]
    raise ValueError(f"unknown key {part!r} in hotkey {original!r}")


class HotkeyManager:
    """Registers ``{hotkey: callback}`` bindings system wide (Windows only).

    ``start()`` returns True when every binding was registered. If some keys
    are taken by another program they are logged, the others stay active and
    ``start()`` returns False (see ``registered`` / ``failed``). Call
    ``stop()`` on shutdown in any case.
    """

    def __init__(self, bindings: dict[str, Callable[[], None]], log: Callable[[str], None] = print) -> None:
        self.bindings = dict(bindings)
        self._log_fn = log
        self.registered: list[str] = []
        self.failed: dict[str, str] = {}
        self._thread: Optional[threading.Thread] = None
        self._thread_id: Optional[int] = None
        self._worker: Optional[threading.Thread] = None
        self._jobs: "queue.Queue[Optional[Callable[[], None]]]" = queue.Queue()
        self._ready = threading.Event()
        self._ids: dict[int, str] = {}
        self._api: Optional[dict[str, Any]] = None

    # ---- public API --------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        if self.running:
            return not self.failed
        if sys.platform != "win32":
            self._log("全局热键只支持 Windows，已跳过（可以用网页面板或手机上的按钮代替）")
            return False
        if not self.bindings:
            return False

        parsed: dict[str, tuple[int, int]] = {}
        owner: dict[tuple[int, int], str] = {}
        self.failed = {}
        self.registered = []
        for combo in self.bindings:
            try:
                key = parse_hotkey(combo)
            except ValueError as exc:
                self.failed[combo] = str(exc)
                self._log(f"热键 {combo} 无法识别：{exc}")
                continue
            if key in owner:
                # "F6" and "f6" are the same key: a second RegisterHotKey would fail
                # with "already registered" and be misreported as taken by another program.
                self.failed[combo] = f"与 {owner[key]} 是同一个按键"
                self._log(f"热键 {combo} 与 {owner[key]} 是同一个按键，已忽略，请在配置里换一个按键")
                continue
            owner[key] = combo
            parsed[combo] = key
            if key == (0, 0x7B):
                self._log("提示：F12 被 Windows 保留给调试器，可能不起作用，建议换一个按键")
        if not parsed:
            return False
        try:
            self._api = _load_api()
        except Exception as exc:  # noqa: BLE001
            self._log(f"无法加载 Win32 热键接口：{exc}")
            return False

        self._ready.clear()
        self._jobs = queue.Queue()
        self._worker = threading.Thread(target=self._work, name="hotkey-callbacks", daemon=True)
        self._worker.start()
        self._thread = threading.Thread(target=self._loop, args=(parsed,), name="hotkey-loop", daemon=True)
        self._thread.start()
        if not self._ready.wait(5.0):
            self._log("热键线程启动超时")
            self.stop()
            return False
        if not self.registered:
            self._stop_worker()
            return False
        self._log(f"全局热键已启用：{'、'.join(self.registered)}")
        return not self.failed

    def stop(self) -> None:
        thread, tid, api = self._thread, self._thread_id, self._api
        if thread is not None and thread.is_alive() and tid and api is not None:
            try:
                api["user32"].PostThreadMessageW(tid, WM_QUIT, 0, 0)
            except Exception:
                pass
            thread.join(2.0)
        self._thread = None
        self._thread_id = None
        self._stop_worker()

    # ---- threads -----------------------------------------------------------------
    def _loop(self, parsed: dict[str, tuple[int, int]]) -> None:
        api = self._api
        assert api is not None
        user32, kernel32, ctypes, wintypes = api["user32"], api["kernel32"], api["ctypes"], api["wintypes"]
        ids: dict[int, str] = {}
        try:
            self._thread_id = int(kernel32.GetCurrentThreadId())
            msg = wintypes.MSG()
            # Make sure this thread has a message queue before anyone posts WM_QUIT to it.
            user32.PeekMessageW(ctypes.byref(msg), None, WM_USER, WM_USER, PM_NOREMOVE)
            for n, (combo, (mods, vk)) in enumerate(parsed.items(), start=1):
                if user32.RegisterHotKey(None, n, mods | MOD_NOREPEAT, vk):
                    ids[n] = combo
                    self.registered.append(combo)
                else:
                    err = ctypes.get_last_error()
                    reason = "已被其他程序占用" if err == 1409 else f"错误码 {err}"
                    self.failed[combo] = reason
                    self._log(f"热键 {combo} 注册失败（{reason}），请在配置里换一个按键")
            self._ids = ids
        except Exception as exc:  # noqa: BLE001
            self._log(f"热键线程出错：{exc}")
        finally:
            self._ready.set()
        if not ids:
            return
        try:
            while True:
                r = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if r == 0 or r == -1:  # WM_QUIT or error
                    break
                if msg.message == WM_HOTKEY:
                    combo = ids.get(int(msg.wParam))
                    cb = self.bindings.get(combo) if combo else None
                    if cb is not None:
                        self._jobs.put(cb)
        except Exception as exc:  # noqa: BLE001
            self._log(f"热键消息循环出错：{exc}")
        finally:
            for n in ids:
                try:
                    user32.UnregisterHotKey(None, n)
                except Exception:
                    pass

    def _work(self) -> None:
        while True:
            cb = self._jobs.get()
            if cb is None:
                return
            try:
                cb()
            except Exception as exc:  # noqa: BLE001 - a broken callback must not kill hotkeys
                self._log(f"热键回调出错：{type(exc).__name__}: {exc}")

    def _stop_worker(self) -> None:
        worker = self._worker
        if worker is not None and worker.is_alive():
            self._jobs.put(None)
            if worker is not threading.current_thread():
                worker.join(2.0)
        self._worker = None

    def _log(self, text: str) -> None:
        try:
            self._log_fn(text)
        except Exception:
            pass


def _load_api() -> dict[str, Any]:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
    user32.RegisterHotKey.restype = wintypes.BOOL
    user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.UnregisterHotKey.restype = wintypes.BOOL
    user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
    user32.GetMessageW.restype = wintypes.BOOL
    user32.PeekMessageW.argtypes = [
        ctypes.POINTER(wintypes.MSG),
        wintypes.HWND,
        wintypes.UINT,
        wintypes.UINT,
        wintypes.UINT,
    ]
    user32.PeekMessageW.restype = wintypes.BOOL
    user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.PostThreadMessageW.restype = wintypes.BOOL
    kernel32.GetCurrentThreadId.argtypes = []
    kernel32.GetCurrentThreadId.restype = wintypes.DWORD
    return {"user32": user32, "kernel32": kernel32, "ctypes": ctypes, "wintypes": wintypes}
