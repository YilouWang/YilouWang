"""Optional always-on-top mini window (tkinter) showing the current advice.

The overlay is a small, semi-transparent, borderless window that floats above
other windows: the headline, the top 3 actions and open human requests.

Note: it only shows over the game when TFT runs in **borderless or windowed**
mode. Exclusive fullscreen owns the display and hides every other window, so
switch the game's window mode to "无边框" (borderless) to use it.

It is passive: it only reads bus events and never sends input to the game.
Tk must run in the main thread; bus events arrive from worker threads and go
through a thread-safe queue that the Tk loop polls every 200 ms.

Details that matter on the target Windows PC:
  * The app makes the process per-monitor DPI aware (for screen capture), so
    Tk fonts come out at real DPI. The window size is scaled by the same
    factor and its height follows the content, so nothing gets clipped at
    125 % / 150 % / 200 % display scaling.
  * The window is made non-activating (``WS_EX_NOACTIVATE``): clicking or
    dragging it never takes keyboard focus away from the game.
  * The "收起" button only collapses it to one line (click "展开" to restore);
    it never ends the app. ``close()`` (or Ctrl+C in the console) ends ``run()``.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from ..bus import EventBus
from ..config import UIConfig

WIDTH = 380
HEIGHT = 220
ALPHA = 0.85
POLL_MS = 200
TOPICS = ("advice", "requests", "status", "state")
COMPACT_MAX_CHARS = 24
MAX_HEIGHT_FRACTION = 0.45  # never cover more than this share of the screen height

BG = "#111418"
FG = "#e8e8e8"
DIM = "#8b949e"
P1 = "#ff9f43"
REQ = "#39c5cf"
BUSY = "#58a6ff"

_ACTION_PREFIX = {1: "!", 2: "·", 3: "·"}


def display_available() -> bool:
    """False on a headless Linux/BSD box (no X11 / Wayland display)."""
    if sys.platform in ("win32", "darwin"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _stage_text(stage: Any) -> str:
    if isinstance(stage, dict) and stage.get("stage") is not None and stage.get("round") is not None:
        return f"{stage['stage']}-{stage['round']}"
    if isinstance(stage, str):
        return stage
    return ""


def _as_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            return {}
    return obj if isinstance(obj, dict) else {}


def overlay_lines(
    advice: Any = None,
    requests: Any = None,
    state: Any = None,
    status: Any = None,
    max_actions: int = 3,
    max_requests: int = 2,
) -> list[tuple[str, str]]:
    """Pure formatting of the overlay content: a list of ``(text, style)``.

    Styles: ``info`` (top line), ``headline``, ``p1``, ``action``, ``request``,
    ``busy``, ``dim``. Kept free of tkinter so it can be unit tested.
    """
    adv = _as_dict(advice)
    st = _as_dict(state)
    stat = _as_dict(status)
    lines: list[tuple[str, str]] = []

    info = []
    stage = _stage_text(st.get("stage"))
    if stage:
        info.append(stage)
    if st.get("gold") is not None:
        info.append(f"金币 {st['gold']}")
    if st.get("level") is not None:
        info.append(f"{st['level']} 级")
    if st.get("hp") is not None:
        info.append(f"血量 {st['hp']}")
    if stat.get("auto") is False:
        info.append("自动关")
    if info:
        lines.append(("  ".join(info), "info"))
    if stat.get("busy"):
        lines.append(("分析中...", "busy"))

    headline = str(adv.get("headline") or "").strip()
    if headline:
        lines.append((headline, "headline"))
        actions = [a for a in (adv.get("actions") or []) if isinstance(a, dict) and str(a.get("text") or "").strip()]
        actions.sort(key=lambda a: _prio(a.get("priority")))
        for a in actions[:max_actions]:
            p = _prio(a.get("priority"))
            lines.append((f"{_ACTION_PREFIX.get(p, '·')} {str(a['text']).strip()}", "p1" if p == 1 else "action"))
    else:
        lines.append(("等待分析... 按 F6 立即分析", "dim"))

    reqs = []
    for r in requests or []:
        r = _as_dict(r)
        text = str(r.get("text") or "").strip()
        if not text and r.get("target_player"):
            text = f"请切到 {r['target_player']} 的棋盘后按 F7"
        if text:
            reqs.append(text)
    if not reqs and adv.get("scout_request"):
        reqs.append(str(adv["scout_request"]).strip())
    for text in reqs[:max_requests]:
        lines.append((f"» {text}", "request"))
    return lines


def compact_line(advice: Any = None, requests: Any = None, max_chars: int = COMPACT_MAX_CHARS) -> tuple[str, str]:
    """The single line shown when the overlay is collapsed: ``(text, style)``.

    The most urgent action (priority 1) if any, else the headline, plus the
    number of open human requests. Styles: ``p1``, ``compact`` or ``dim``.
    """
    adv = _as_dict(advice)
    actions = [a for a in (adv.get("actions") or []) if isinstance(a, dict) and str(a.get("text") or "").strip()]
    first = next((str(a["text"]).strip() for a in actions if _prio(a.get("priority")) == 1), "")
    headline = str(adv.get("headline") or "").strip()
    text, style = (first, "p1") if first else ((headline, "compact") if headline else ("等待分析", "dim"))
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[: max(1, max_chars - 1)] + "…"
    n = sum(1 for r in (requests or []) if isinstance(r, dict) or hasattr(r, "model_dump")) if isinstance(requests, (list, tuple)) else 0
    if n:
        text += f"  » {n} 条请求"
    return text, style


def _prio(value: Any) -> int:
    try:
        return max(1, min(3, int(value)))
    except (TypeError, ValueError):
        return 2


def _tk_scale(root: Any) -> float:
    """Pixels per 96 dpi pixel as Tk sees it (1.5 at 150 % Windows scaling)."""
    try:
        scale = float(root.winfo_fpixels("1i")) / 96.0
    except Exception:
        return 1.0
    return max(1.0, min(3.0, scale)) if scale == scale else 1.0  # NaN guard


def _screen_bounds(root: Any) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the whole desktop (all monitors on Windows)."""
    if sys.platform == "win32":
        try:
            import ctypes

            user32 = ctypes.WinDLL("user32")  # private handle: never touch shared prototypes
            x0, y0 = user32.GetSystemMetrics(76), user32.GetSystemMetrics(77)  # SM_X/YVIRTUALSCREEN
            w, h = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)  # SM_CX/CYVIRTUALSCREEN
            if w > 0 and h > 0:
                return x0, y0, x0 + w, y0 + h
        except Exception:
            pass
    return 0, 0, int(root.winfo_screenwidth()), int(root.winfo_screenheight())


def _make_non_activating(root: Any) -> bool:
    """Windows: clicks on the overlay must not steal keyboard focus from the game."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32")
        long_ptr = ctypes.c_ssize_t
        get_style = getattr(user32, "GetWindowLongPtrW", None) or user32.GetWindowLongW
        set_style = getattr(user32, "SetWindowLongPtrW", None) or user32.SetWindowLongW
        get_style.restype, get_style.argtypes = long_ptr, [wintypes.HWND, ctypes.c_int]
        set_style.restype, set_style.argtypes = long_ptr, [wintypes.HWND, ctypes.c_int, long_ptr]
        user32.GetParent.restype, user32.GetParent.argtypes = wintypes.HWND, [wintypes.HWND]
        user32.SetWindowPos.restype = wintypes.BOOL
        user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND] + [ctypes.c_int] * 4 + [wintypes.UINT]
        root.update_idletasks()
        hwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()
        gwl_exstyle = -20
        ws_ex_noactivate, ws_ex_toolwindow, ws_ex_appwindow = 0x08000000, 0x00000080, 0x00040000
        style = get_style(hwnd, gwl_exstyle)
        set_style(hwnd, gwl_exstyle, (style | ws_ex_noactivate | ws_ex_toolwindow) & ~ws_ex_appwindow)
        # SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED
        user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020)
        return True
    except Exception:
        return False


def _pos_file() -> Path:
    return Path(os.path.expanduser("~/.tft_advisor")) / "overlay_pos.json"


class Overlay:
    """Tkinter always-on-top overlay. ``run()`` must be called on the main thread.

    Only visible over the game in borderless / windowed mode (not exclusive
    fullscreen). Drag it with the left mouse button; the position is
    remembered. Click the small "x" to close it (the caller decides what
    closing means; ``AdvisorApp.wait`` then returns).
    """

    def __init__(
        self,
        bus: EventBus,
        cfg: UIConfig,
        log: Callable[[str], Any] = print,
        *,
        stop_event: Optional[threading.Event] = None,
        width: int = WIDTH,
        height: int = HEIGHT,
    ) -> None:
        self.bus = bus
        self.cfg = cfg
        self.log = log
        self.width = width
        self.height = height
        self._stop = stop_event or threading.Event()
        self._closed = threading.Event()
        self._q: "queue.Queue[tuple[str, Any]]" = queue.Queue(maxsize=200)
        self._resync = True
        self._data: dict[str, Any] = {}
        self._unsubs: list[Callable[[], None]] = []
        self._root: Any = None
        self._body: Any = None
        self._drag: tuple[int, int] = (0, 0)
        self._interrupted = False
        self._last_lines: Any = None
        self._collapsed = False
        self._scale = 1.0
        self._frame: Any = None
        self._toggle: Any = None
        self._bounds: tuple[int, int, int, int] = (0, 0, width, height)

    # ---------------------------------------------------------------- bus side
    def _on_event(self, topic: str, payload: Any) -> None:
        try:
            self._q.put_nowait((topic, payload))
        except queue.Full:
            self._resync = True  # re-read the latest payloads from the bus instead

    # ------------------------------------------------------------------ public
    def run(self) -> bool:
        """Show the overlay and block until it is closed. False when unavailable."""
        if not display_available():
            self.log("置顶小窗不可用：没有图形界面（没有显示器环境）")
            return False
        try:
            import tkinter as tk
            import tkinter.font as tkfont
        except Exception:
            self.log("置顶小窗不可用：没有安装 tkinter")
            return False
        try:
            root = tk.Tk()
        except Exception as exc:
            self.log(f"置顶小窗不可用：{exc}")
            return False
        self._root = root
        try:
            self._build(root, tk, tkfont)
        except Exception as exc:
            self.log(f"置顶小窗创建失败：{exc}")
            try:
                root.destroy()
            except Exception:
                pass
            self._root = None
            return False

        for topic in TOPICS:
            self._unsubs.append(self.bus.subscribe(topic, self._on_event))
        self._resync = True
        root.after(POLL_MS, self._poll)
        self.log("置顶小窗已打开（游戏需为无边框或窗口模式，点「收起」可缩成一行）")
        try:
            root.mainloop()
        except KeyboardInterrupt:
            # Tk checks for signals between events that ran no Python callback, so
            # Ctrl+C often lands here instead of _report_exception. Set the shared
            # stop event too: otherwise only the window closes and the app (capture,
            # hotkeys, Claude calls) keeps running until a second Ctrl+C.
            self._interrupted = True
            self._stop.set()
        finally:
            for unsub in self._unsubs:
                try:
                    unsub()
                except Exception:
                    pass
            self._unsubs.clear()
            self._save_position()
            try:
                root.destroy()
            except Exception:
                pass
            self._root = None
            self._closed.set()
        return True

    def close(self) -> None:
        """Close the window (safe from any thread: the Tk loop notices it)."""
        self._stop.set()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    @property
    def interrupted(self) -> bool:
        """True when ``run()`` ended because of Ctrl+C (the stop event is set too)."""
        return self._interrupted

    # ------------------------------------------------------------------- Tk side
    def _build(self, root: Any, tk: Any, tkfont: Any) -> None:
        root.title("TFT 助手")
        root.overrideredirect(True)
        root.configure(bg=BG)
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        try:
            root.attributes("-alpha", ALPHA)
        except Exception:
            pass
        families = set()
        try:
            families = set(tkfont.families(root))
        except Exception:
            pass
        family = next(
            (f for f in ("Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Micro Hei") if f in families),
            "TkDefaultFont",
        )
        self._fonts = {
            "info": (family, 9),
            "busy": (family, 9),
            "headline": (family, 13, "bold"),
            "p1": (family, 11, "bold"),
            "action": (family, 10),
            "request": (family, 10, "bold"),
            "dim": (family, 10),
        }
        self._fonts["compact"] = (family, 11, "bold")
        self._colors = {
            "info": DIM, "busy": BUSY, "headline": FG, "p1": P1, "action": FG, "request": REQ, "dim": DIM, "compact": FG,
        }

        # Fonts are in points, so Tk already scales them with the DPI; scale the
        # pixel geometry by the same factor or the text no longer fits.
        self._scale = _tk_scale(root)
        self._bounds = _screen_bounds(root)
        x, y = self._initial_position(root)
        root.geometry(f"{self._px(self.width)}x{self._px(self.height)}+{x}+{y}")

        frame = tk.Frame(root, bg=BG, highlightthickness=1, highlightbackground="#2a313c")
        frame.pack(fill="both", expand=True)
        body = tk.Frame(frame, bg=BG)
        body.pack(fill="both", expand=True, padx=(self._px(10), self._px(40)), pady=(self._px(6), self._px(8)))
        toggle = tk.Label(frame, text="收起", bg=BG, fg=DIM, font=(family, 9), cursor="hand2")
        toggle.place(relx=1.0, x=-self._px(6), y=self._px(3), anchor="ne")
        toggle.lift()
        toggle.bind("<Button-1>", lambda _e: self.toggle_collapsed())
        self._frame = frame
        self._body = body
        self._toggle = toggle
        self._tk = tk

        for widget in (root, frame, body):
            self._bind_drag(widget)
        root.bind("<Escape>", lambda _e: self.toggle_collapsed())
        root.protocol("WM_DELETE_WINDOW", self.toggle_collapsed)  # a WM close only collapses too
        root.report_callback_exception = self._report_exception
        _make_non_activating(root)

    def _px(self, value: float) -> int:
        return int(round(value * self._scale))

    def toggle_collapsed(self) -> None:
        """Collapse to one line / expand again (Tk thread only)."""
        self._collapsed = not self._collapsed
        if self._toggle is not None:
            try:
                self._toggle.configure(text="展开" if self._collapsed else "收起")
            except Exception:
                pass
        try:
            self._render()
        except Exception as exc:
            self.log(f"置顶小窗刷新失败：{exc!r}")

    def _bind_drag(self, widget: Any) -> None:
        widget.bind("<ButtonPress-1>", self._drag_start, add="+")
        widget.bind("<B1-Motion>", self._drag_move, add="+")

    def _drag_start(self, event: Any) -> None:
        self._drag = (event.x_root - self._root.winfo_x(), event.y_root - self._root.winfo_y())

    def _drag_move(self, event: Any) -> None:
        dx, dy = self._drag
        self._root.geometry(f"+{event.x_root - dx}+{event.y_root - dy}")

    def _report_exception(self, exc: type, val: BaseException, tb: Any) -> None:
        # Tk swallows exceptions raised in callbacks; Ctrl+C must still quit.
        if issubclass(exc, KeyboardInterrupt):
            self._interrupted = True
            self._stop.set()
            try:
                self._root.quit()
            except Exception:
                pass
            return
        try:
            self.log(f"置顶小窗出错：{val!r}")
        except Exception:
            pass

    def _initial_position(self, root: Any) -> tuple[int, int]:
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w = self._px(self.width)
        x0, y0, x1, y1 = self._bounds
        try:
            data = json.loads(_pos_file().read_text(encoding="utf-8"))
            x, y = int(data["x"]), int(data["y"])
            # Any monitor of the desktop (a second screen may have negative coordinates).
            if x0 - w // 2 <= x <= x1 - 40 and y0 <= y <= y1 - 40:
                return x, y
        except Exception:
            pass
        # Right side, below the top HUD, left of the player list.
        return max(0, sw - w - self._px(260)), max(0, int(sh * 0.12))

    def _save_position(self) -> None:
        root = self._root
        if root is None:
            return
        try:
            x, y = root.winfo_x(), root.winfo_y()
            path = _pos_file()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"x": x, "y": y}), encoding="utf-8")
        except Exception:
            pass

    def _poll(self) -> None:
        root = self._root
        if root is None:
            return
        if self._stop.is_set():
            root.quit()
            return
        try:
            changed = False
            if self._resync:
                self._resync = False
                for topic in TOPICS:
                    value = self.bus.latest(topic)
                    if value is not None:
                        self._data[topic] = value
                changed = True
            for _ in range(500):  # bounded: a flood of events cannot freeze the Tk loop
                try:
                    topic, payload = self._q.get_nowait()
                except queue.Empty:
                    break
                self._data[topic] = payload
                changed = True
            if changed:
                self._render()
        except Exception as exc:
            try:
                self.log(f"置顶小窗刷新失败：{exc!r}")
            except Exception:
                pass
        finally:
            # Always re-arm: if polling stopped, close() would never be noticed.
            try:
                root.after(POLL_MS, self._poll)
            except Exception:
                pass

    def _render(self) -> None:
        if self._collapsed:
            lines = [compact_line(self._data.get("advice"), self._data.get("requests"))]
        else:
            lines = overlay_lines(
                self._data.get("advice"),
                self._data.get("requests"),
                self._data.get("state"),
                self._data.get("status"),
            )
        key = (self._collapsed, lines)
        if key == self._last_lines:
            return
        self._last_lines = key
        tk = self._tk
        for child in self._body.winfo_children():
            child.destroy()
        wrap = max(self._px(120), self._px(self.width) - self._px(10) - self._px(40) - 4)
        for text, style in lines:
            label = tk.Label(
                self._body,
                text=text,
                bg=BG,
                fg=self._colors.get(style, FG),
                font=self._fonts.get(style, self._fonts["action"]),
                anchor="w",
                justify="left",
                wraplength=wrap,
            )
            label.pack(fill="x", anchor="w", pady=(0, self._px(3) if style == "headline" else self._px(1)))
            self._bind_drag(label)
        self._fit_height()

    def _fit_height(self) -> None:
        """Grow / shrink the window to its content (bounded), keeping its position."""
        root, frame = self._root, self._frame
        if root is None or frame is None:
            return
        try:
            root.update_idletasks()
            need = int(frame.winfo_reqheight())
            screen_h = self._bounds[3] - self._bounds[1]
            max_h = max(self._px(self.height), int(screen_h * MAX_HEIGHT_FRACTION))
            h = max(self._px(30), min(need, max_h))
            root.geometry(f"{self._px(self.width)}x{h}+{root.winfo_x()}+{root.winfo_y()}")
        except Exception:
            pass


__all__ = ["Overlay", "compact_line", "display_available", "overlay_lines"]
