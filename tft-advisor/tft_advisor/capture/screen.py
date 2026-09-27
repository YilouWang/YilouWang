"""Screen capture: passive screenshots of the game (like OBS would take).

* ``ScreenCapturer(cfg)`` grabs the game window's client area (Windows) or a
  whole monitor with ``mss``. One ``mss`` instance per thread (mss is not
  thread safe). ``grab()`` never raises: it returns ``None`` on any failure
  and stores a short Chinese explanation in ``last_error``.
* ``FileCapturer(paths)`` has the same ``grab()`` API and replays image files.
* ``save_frame(img, directory, tag)`` stores a timestamped PNG.

Only screenshots are taken: nothing here sends input to the game or reads its memory.

Game window lookup (Windows): TFT moved to Unreal Engine with Set 18
(August 2026). The Unreal build runs as ``TFT.exe`` /
``TFTClient-Win64-Shipping.exe`` and no longer uses the old
"League of Legends (TM) Client" title, so ``find_window_rect`` first tries the
configured title, then any visible top level window whose title is a known TFT
title or whose owning executable is a known TFT game process, and keeps the
largest one. Executable names come from a Toolhelp process snapshot (the
system's process list, cached a few seconds) and windows are matched by PID:
no handle to the game process (anti-cheat protected) or any other process is
ever opened.

Window frames are screen pixels of the game's client area, so another app in
front of the game would be captured too. ``ScreenCapturer.game_foreground()``
tells whether the game had focus when the calling thread's last frame was
taken; automatic analysis skips frames where it is False.
"""

from __future__ import annotations

import glob
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Union

from PIL import Image

from ..config import CaptureConfig

Rect = tuple[int, int, int, int]  # left, top, width, height (screen pixels)
PathLike = Union[str, Path]

#: Window titles of the in-game (not lobby) TFT window, old and new client.
KNOWN_WINDOW_TITLES: tuple[str, ...] = ("League of Legends (TM) Client", "Teamfight Tactics")
#: Executables that own the in-game window (Unreal client first, legacy last).
KNOWN_PROCESS_NAMES: tuple[str, ...] = ("TFT.exe", "TFTClient-Win64-Shipping.exe", "League of Legends.exe")
#: Client areas smaller than this are splash / helper windows, not the game.
MIN_WINDOW_SIZE = (320, 200)
#: Seconds a process snapshot (PIDs of the game executables) is reused.
PROCESS_SNAPSHOT_TTL_S = 5.0

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# --------------------------------------------------------------------------
# Win32 helpers (ctypes, Windows only, everything imported lazily)
# --------------------------------------------------------------------------

_dpi_lock = threading.Lock()
_dpi_done: Optional[bool] = None
_win32_cache: dict[str, Any] = {}
_pid_lock = threading.Lock()
_pid_cache: dict[str, Any] = {}  # {"key": frozenset of names, "at": monotonic, "pids": frozenset}


def set_dpi_awareness() -> bool:
    """Make this process DPI aware so window rects match mss's physical pixels.

    Without it, with Windows display scaling != 100 %, ``GetClientRect``
    returns virtualized (scaled) coordinates and every crop is shifted.
    Idempotent. No-op (returns False) outside Windows.
    """
    global _dpi_done
    if sys.platform != "win32":
        return False
    with _dpi_lock:
        if _dpi_done is not None:
            return _dpi_done
        ok = False
        try:
            import ctypes

            try:
                # 2 = PROCESS_PER_MONITOR_DPI_AWARE. S_OK (0) or E_ACCESSDENIED
                # (already set, e.g. by mss or a manifest) both mean "aware".
                hr = ctypes.windll.shcore.SetProcessDpiAwareness(2)
                ok = hr in (0, -2147024891, 0x80070005)
            except (AttributeError, OSError):
                ok = False
            if not ok:
                try:
                    ok = bool(ctypes.windll.user32.SetProcessDPIAware())
                except (AttributeError, OSError):
                    ok = False
        except Exception:
            ok = False
        _dpi_done = ok
        return ok


def _win32() -> dict[str, Any]:
    """Private WinDLL handles with argtypes set (does not touch ``ctypes.windll``)."""
    if _win32_cache:
        return _win32_cache
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    user32.FindWindowW.restype = wintypes.HWND
    user32.EnumWindows.argtypes = [enum_proc, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetClientRect.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.ClientToScreen.restype = wintypes.BOOL
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wintypes.HWND

    entry_type = _processentry32w(ctypes, wintypes)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(entry_type)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(entry_type)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    _win32_cache.update(
        ctypes=ctypes,
        wintypes=wintypes,
        user32=user32,
        kernel32=kernel32,
        enum_proc=enum_proc,
        PROCESSENTRY32W=entry_type,
        INVALID_HANDLE_VALUE=ctypes.c_void_p(-1).value,
    )
    return _win32_cache


def _processentry32w(ctypes: Any, wintypes: Any) -> Any:
    """``PROCESSENTRY32W`` (tlhelp32.h)."""

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),  # ULONG_PTR
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    return PROCESSENTRY32W


def _window_title(hwnd: int) -> str:
    w = _win32()
    n = w["user32"].GetWindowTextLengthW(hwnd)
    if n <= 0:
        return ""
    buf = w["ctypes"].create_unicode_buffer(n + 1)
    w["user32"].GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def _window_pid(hwnd: Any) -> Optional[int]:
    """PID of the process owning a window (a window query, no process handle)."""
    w = _win32()
    pid = w["wintypes"].DWORD()
    if not w["user32"].GetWindowThreadProcessId(hwnd, w["ctypes"].byref(pid)) or not pid.value:
        return None
    return int(pid.value)


def _running_processes() -> list[tuple[int, str]]:
    """``(pid, executable name)`` of every running process.

    Read from a Toolhelp snapshot of the system's process list: no handle to
    any process is opened, so the anti-cheat protected game is never touched.
    """
    w = _win32()
    ctypes, kernel32 = w["ctypes"], w["kernel32"]
    snap = kernel32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if not snap or snap == w["INVALID_HANDLE_VALUE"]:
        return []
    out: list[tuple[int, str]] = []
    try:
        entry = w["PROCESSENTRY32W"]()
        entry.dwSize = ctypes.sizeof(entry)
        ok = kernel32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            out.append((int(entry.th32ProcessID), str(entry.szExeFile)))
            ok = kernel32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snap)
    return out


def _game_pids(process_names: Sequence[str]) -> frozenset[int]:
    """PIDs of running processes whose executable is one of ``process_names``.

    Cached for ``PROCESS_SNAPSHOT_TTL_S`` (the capture and worker threads both
    search while the game window is not found). Empty on any failure."""
    key = frozenset(p.casefold() for p in process_names if p)
    if not key:
        return frozenset()
    now = time.monotonic()
    with _pid_lock:
        if _pid_cache.get("key") == key and now - _pid_cache.get("at", float("-inf")) < PROCESS_SNAPSHOT_TTL_S:
            return _pid_cache["pids"]
    try:
        pids = frozenset(pid for pid, name in _running_processes() if pid and name.casefold() in key)
    except Exception:
        return frozenset()
    with _pid_lock:
        _pid_cache.update(key=key, at=now, pids=pids)
    return pids


def _is_game_foreground(hwnd: Optional[int]) -> bool:
    """True when ``hwnd`` or another window of its process has the focus."""
    if not hwnd:
        return False
    fg = _win32()["user32"].GetForegroundWindow()
    if not fg:  # focus in transition (alt-tab) or a secure desktop
        return False
    if int(fg) == int(hwnd):
        return True
    pid = _window_pid(hwnd)
    return pid is not None and _window_pid(fg) == pid


def _on_windows() -> bool:
    return sys.platform == "win32"


def _is_minimized(hwnd: Optional[int]) -> bool:
    """True when ``hwnd`` is an existing window that is minimized (iconic)."""
    if not hwnd:
        return False
    user32 = _win32()["user32"]
    return bool(user32.IsWindow(hwnd) and user32.IsIconic(hwnd))


def _client_rect(hwnd: int) -> Optional[Rect]:
    """Screen-space client area (no title bar / borders) of a visible, non minimized window."""
    w = _win32()
    user32, wintypes, ctypes = w["user32"], w["wintypes"], w["ctypes"]
    if not user32.IsWindow(hwnd) or user32.IsIconic(hwnd):
        return None
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    origin = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        return None
    width, height = rect.right - rect.left, rect.bottom - rect.top
    if width < MIN_WINDOW_SIZE[0] or height < MIN_WINDOW_SIZE[1]:
        return None
    return (int(origin.x), int(origin.y), int(width), int(height))


def _find_game_hwnd(
    title: Optional[str],
    process_names: Sequence[str] = KNOWN_PROCESS_NAMES,
    *,
    include_minimized: bool = False,
) -> Optional[int]:
    """Handle of the game window. With ``include_minimized`` a minimized game
    window is returned too (ranked below any restored one), so callers can
    tell "game minimized" from "game not running"."""
    w = _win32()
    user32 = w["user32"]
    if title:
        hwnd = user32.FindWindowW(None, title)
        if hwnd and user32.IsWindowVisible(hwnd):
            if _client_rect(hwnd) or (include_minimized and _is_minimized(hwnd)):
                return int(hwnd)

    wanted_titles = {t.casefold() for t in ([title] if title else []) + list(KNOWN_WINDOW_TITLES)}
    game_pids = _game_pids(process_names)
    candidates: list[tuple[int, int]] = []  # (area, hwnd)

    def visit(hwnd: Any, _lparam: Any) -> bool:
        try:
            if not hwnd or not user32.IsWindowVisible(hwnd):
                return True
            match = _window_title(hwnd).strip().casefold() in wanted_titles
            if not match and game_pids:
                match = _window_pid(hwnd) in game_pids
            if match:
                rect = _client_rect(hwnd)
                if rect:
                    candidates.append((rect[2] * rect[3], int(hwnd)))
                elif include_minimized and _is_minimized(hwnd):
                    candidates.append((0, int(hwnd)))
        except Exception:
            pass
        return True  # keep enumerating

    user32.EnumWindows(w["enum_proc"](visit), 0)
    if not candidates:
        return None
    # A game process can briefly expose helper windows: the largest surface is the game.
    return max(candidates)[1]


def find_window_rect(
    title: Optional[str] = KNOWN_WINDOW_TITLES[0],
    *,
    process_names: Sequence[str] = KNOWN_PROCESS_NAMES,
) -> Optional[Rect]:
    """``(left, top, width, height)`` of the game window's client area, or None.

    Returns None outside Windows, when the window is not found, minimized, or
    too small. Never raises.
    """
    if sys.platform != "win32":
        return None
    try:
        set_dpi_awareness()
        hwnd = _find_game_hwnd(title, process_names)
        return _client_rect(hwnd) if hwnd else None
    except Exception:
        return None


# --------------------------------------------------------------------------
# Capturers
# --------------------------------------------------------------------------


def _new_mss() -> Any:
    import mss  # lazy: optional on CI, may fail without a display

    factory = getattr(mss, "MSS", None) or mss.mss  # mss >= 10.2 deprecates mss.mss()
    return factory()


_UNSET = object()


def _is_black(img: Image.Image, level: int = 8) -> bool:
    small = img.resize((64, 36), Image.Resampling.BOX)
    return all(hi <= level for _lo, hi in small.getextrema())


class ScreenCapturer:
    """Grabs the game window (Windows) or a monitor as an RGB ``PIL.Image``.

    Attributes useful for diagnostics: ``last_error`` (Chinese text or None),
    ``last_source`` ("window" / "monitor" / "minimized"), ``last_rect``
    (left, top, w, h), ``frames`` and ``failures`` counters.
    ``game_foreground()`` says whether the game had the focus when the calling
    thread's last frame was taken.
    """

    #: Seconds between window searches while the game window is not found.
    SEARCH_INTERVAL_S = 3.0
    #: Consecutive black frames before ``last_error`` explains fullscreen capture issues.
    BLACK_FRAMES_HINT = 3

    def __init__(self, cfg: Optional[CaptureConfig] = None, log: Optional[Callable[[str], None]] = None) -> None:
        self.cfg = cfg or CaptureConfig()
        self._log = log
        self._local = threading.local()
        self._instances: list[tuple[threading.Thread, Any]] = []  # (owner thread, mss instance)
        self._lock = threading.Lock()
        self._hwnd: Optional[int] = None
        self._next_search = 0.0
        self._closed = False
        self._black_run = 0
        self.last_error: Optional[str] = None
        self.last_source: str = ""
        self.last_rect: Optional[Rect] = None
        self.frames = 0
        self.failures = 0
        set_dpi_awareness()

    # ---- public API --------------------------------------------------------------
    @property
    def window_found(self) -> bool:
        return self.last_source == "window"

    def game_foreground(self) -> Optional[bool]:
        """Did the game have the focus when this thread's last frame was taken?

        True: the frame is the game window's client area and the game (or
        another window of its process) was the foreground window just before
        and just after the grab. False: another app had the focus, so its
        window may cover the game in the frame, or the frame was not the game
        window (monitor fallback, minimized, failure). None: unknown (not
        Windows, or window capture turned off).

        Per thread: the capture and worker threads grab independently. A
        thread that has not grabbed yet gets a live check.
        """
        if not self.cfg.use_window or not _on_windows():
            return None
        recorded = getattr(self._local, "foreground", _UNSET)
        if recorded is not _UNSET:
            return recorded
        return self._foreground(self._hwnd)

    def grab(self) -> Optional[Image.Image]:
        """One RGB frame, or None on any failure (never raises).

        Also None while the game window is minimized: falling back to the
        monitor would feed the desktop (or the dashboard) to the vision
        pipeline and fire bogus round changes. Other cases where the frame
        may not show the game (window not found: monitor fallback; another
        app in front of the game) still return a frame for explicit hotkey
        requests: check ``last_source`` and ``game_foreground()`` before
        using it automatically.
        """
        if self._closed:
            return None
        window_mode = bool(self.cfg.use_window) and _on_windows()
        self._local.foreground = False if window_mode else None
        try:
            window, minimized = self._window_region()
            if minimized:
                self.last_source = "minimized"
                self.last_rect = None
                self._fail("游戏窗口已最小化，恢复游戏窗口后会继续截图")
                return None
            sct = self._sct()
            hwnd = self._hwnd
            if window is not None:
                region, source = window, "window"
            else:
                region, source = self._monitor_region(sct), "monitor"
            focused = source == "window" and self._foreground(hwnd)
            try:
                shot = sct.grab(region)
            except Exception:
                if source != "window":
                    raise
                # The window moved / closed between lookup and grab: use the monitor.
                self._hwnd = None
                region, source = self._monitor_region(sct), "monitor"
                shot = sct.grab(region)
            # Focus must hold for the whole grab (an alt-tab mid-grab is not the game).
            focused = focused and source == "window" and self._foreground(hwnd)
            img = Image.frombytes("RGB", (int(shot.size[0]), int(shot.size[1])), shot.bgra, "raw", "BGRX")
            black = _is_black(img)
        except Exception as exc:  # noqa: BLE001 - the capture loop must never die
            self._drop_sct()
            if self._closed:  # close() raced with this grab: not an error worth reporting
                return None
            self._fail(self._explain(exc))
            return None

        if window_mode:
            self._local.foreground = bool(focused)
        self.last_source = source
        self.last_rect = (int(region["left"]), int(region["top"]), int(region["width"]), int(region["height"]))
        if black:
            self._black_run += 1
            if self._black_run >= self.BLACK_FRAMES_HINT:
                self._fail("截图一直是全黑的：游戏可能处于独占全屏模式，请在游戏设置里改成无边框窗口")
            else:
                self.failures += 1
            return None
        self._black_run = 0
        self.frames += 1
        self.last_error = None
        return img

    def close(self) -> None:
        self._closed = True
        with self._lock:
            instances, self._instances = self._instances, []
        for _owner, inst in instances:
            try:
                inst.close()
            except Exception:
                pass

    def __enter__(self) -> "ScreenCapturer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- internals -----------------------------------------------------------------
    def _sct(self) -> Any:
        inst = getattr(self._local, "sct", None)
        if inst is None:
            inst = _new_mss()
            with self._lock:
                closed = self._closed
                # Instances of threads that have exited can never be used again:
                # release them instead of keeping them (and their GDI handles) forever.
                dead = [pair for pair in self._instances if not pair[0].is_alive()]
                self._instances = [pair for pair in self._instances if pair[0].is_alive()]
                if not closed:
                    self._instances.append((threading.current_thread(), inst))
            for _owner, old in dead:
                try:
                    old.close()
                except Exception:
                    pass
            if closed:  # close() ran while this instance was being created
                try:
                    inst.close()
                except Exception:
                    pass
                raise RuntimeError("capturer closed")
            self._local.sct = inst
        return inst

    def _drop_sct(self) -> None:
        inst = getattr(self._local, "sct", None)
        self._local.sct = None
        if inst is None:
            return
        with self._lock:
            self._instances = [pair for pair in self._instances if pair[1] is not inst]
        try:
            inst.close()
        except Exception:
            pass

    def _monitor_region(self, sct: Any) -> dict[str, int]:
        monitors = sct.monitors
        idx = int(self.cfg.monitor)
        if idx < 0 or idx >= len(monitors):
            idx = 1 if len(monitors) > 1 else 0
        mon = monitors[idx]
        return {"left": int(mon["left"]), "top": int(mon["top"]), "width": int(mon["width"]), "height": int(mon["height"])}

    def _window_region(self) -> tuple[Optional[dict[str, int]], bool]:
        """(client area of the game window or None, game window minimized)."""
        if not self.cfg.use_window or not _on_windows():
            return None, False
        try:
            if self._hwnd and _is_minimized(self._hwnd):
                return None, True
            rect = _client_rect(self._hwnd) if self._hwnd else None
            if rect is None:
                now = time.monotonic()
                if now < self._next_search:
                    return None, False
                self._hwnd = _find_game_hwnd(self.cfg.window_title, include_minimized=True)
                self._next_search = now + (0.0 if self._hwnd else self.SEARCH_INTERVAL_S)
                if self._hwnd and _is_minimized(self._hwnd):
                    return None, True
                rect = _client_rect(self._hwnd) if self._hwnd else None
        except Exception:
            self._hwnd = None
            return None, False
        if rect is None:
            return None, False
        left, top, width, height = rect
        return {"left": left, "top": top, "width": width, "height": height}, False

    @staticmethod
    def _foreground(hwnd: Optional[int]) -> bool:
        try:
            return _is_game_foreground(hwnd)
        except Exception:
            return False

    @staticmethod
    def _explain(exc: Exception) -> str:
        text = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, ImportError):
            return f"截图组件 mss 不可用：{text}"
        if sys.platform != "win32" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return f"没有可用的显示器，无法截图（{text}）"
        return f"截图失败：{text}"

    def _fail(self, message: str) -> None:
        self.failures += 1
        if message != self.last_error and self._log is not None:
            try:
                self._log(message)
            except Exception:
                pass
        self.last_error = message


def _expand_paths(paths: Iterable[PathLike]) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    missing: list[str] = []
    for raw in paths:
        text = os.path.expanduser(str(raw))
        # A literal path wins over glob syntax ("D:/录像[旧]/a.png" is a real file, not a pattern).
        if any(ch in text for ch in "*?[") and not os.path.exists(text):
            matches = sorted(Path(p) for p in glob.glob(text))
            files.extend(p for p in matches if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            if not matches:
                missing.append(str(raw))
            continue
        p = Path(text)
        if p.is_dir():
            files.extend(sorted(c for c in p.iterdir() if c.is_file() and c.suffix.lower() in IMAGE_EXTENSIONS))
        elif p.is_file():
            files.append(p)
        else:
            missing.append(str(raw))
    return files, missing


class FileCapturer:
    """Replays image files with the ``ScreenCapturer.grab()`` API (tests / replay).

    ``paths`` may contain files, directories (their images, sorted by name) and
    glob patterns. ``grab()`` returns the next image as RGB, cycling to the
    start when ``loop`` is True, or None when exhausted. Unreadable files are
    skipped. ``current_path`` is the file of the last returned frame.
    """

    def __init__(self, paths: Iterable[PathLike], loop: bool = False) -> None:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths, self.missing = _expand_paths(paths)
        self.loop = loop
        self.index = 0
        self.current_path: Optional[Path] = None
        self.last_error: Optional[str] = None
        self.frames = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def remaining(self) -> int:
        return max(0, len(self.paths) - self.index)

    def grab(self) -> Optional[Image.Image]:
        with self._lock:
            for _ in range(len(self.paths)):
                if self.index >= len(self.paths):
                    if not self.loop:
                        return None
                    self.index = 0
                path = self.paths[self.index]
                self.index += 1
                try:
                    with Image.open(path) as im:
                        img = im.convert("RGB")
                        img.load()
                except Exception as exc:  # noqa: BLE001
                    self.last_error = f"无法读取图片 {path.name}：{exc}"
                    continue
                self.current_path = path
                self.frames += 1
                return img
            return None

    def reset(self) -> None:
        with self._lock:
            self.index = 0
            self.current_path = None

    def close(self) -> None:
        pass


_TAG_RE = re.compile(r"[^\w\-]+", re.UNICODE)


def save_frame(img: Image.Image, directory: PathLike, tag: str = "frame") -> Path:
    """Save ``img`` as ``<directory>/<YYYYmmdd-HHMMSS-mmm>_<tag>.png``; returns the path.

    Creates the directory, expands ``~``, never overwrites an existing file
    (the name is reserved with an exclusive create, so concurrent saves from
    several threads cannot clobber each other). ``tag`` is sanitized, it can
    never leave ``directory``.
    """
    if not isinstance(img, Image.Image):
        raise TypeError(f"save_frame needs a PIL image, got {type(img).__name__}")
    folder = Path(os.path.expanduser(str(directory)))
    folder.mkdir(parents=True, exist_ok=True)
    now = time.time()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int(now * 1000) % 1000:03d}"
    safe = _TAG_RE.sub("_", str(tag or "")).strip("_")[:48] or "frame"
    out = img if img.mode in ("RGB", "RGBA", "L") else img.convert("RGB")
    n = 0
    while True:
        path = folder / (f"{stamp}_{safe}.png" if n == 0 else f"{stamp}_{safe}-{n}.png")
        n += 1
        try:
            fh = open(path, "xb")
        except FileExistsError:
            continue
        try:
            with fh:
                out.save(fh, format="PNG", compress_level=3)
        except BaseException:
            try:
                path.unlink()
            except OSError:
                pass
            raise
        return path
