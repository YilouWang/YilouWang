"""Optional always-on-top mini window (tkinter) showing the current advice.

The overlay is a small, semi-transparent, borderless window that floats above
other windows: the headline, the top 3 actions and open human requests.

Note: it only shows over the game when TFT runs in **borderless or windowed**
mode. Exclusive fullscreen owns the display and hides every other window, so
switch the game's window mode to "无边框" (borderless) to use it.

It is passive: it only reads bus events and never sends input to the game.
Tk must run in the main thread; bus events arrive from worker threads and go
through a thread-safe queue that the Tk loop polls every 200 ms.
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


def _prio(value: Any) -> int:
    try:
        return max(1, min(3, int(value)))
    except (TypeError, ValueError):
        return 2


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
        self._last_lines: list[tuple[str, str]] = []

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
        self.log("置顶小窗已打开（游戏需为无边框或窗口模式）")
        try:
            root.mainloop()
        except KeyboardInterrupt:
            self._interrupted = True
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
        self._colors = {"info": DIM, "busy": BUSY, "headline": FG, "p1": P1, "action": FG, "request": REQ, "dim": DIM}

        x, y = self._initial_position(root)
        root.geometry(f"{self.width}x{self.height}+{x}+{y}")

        frame = tk.Frame(root, bg=BG, highlightthickness=1, highlightbackground="#2a313c")
        frame.pack(fill="both", expand=True)
        body = tk.Frame(frame, bg=BG)
        body.pack(fill="both", expand=True, padx=(10, 22), pady=(6, 8))
        close = tk.Label(frame, text="×", bg=BG, fg=DIM, font=(family, 12), cursor="hand2")
        close.place(relx=1.0, x=-6, y=2, anchor="ne")
        close.lift()
        close.bind("<Button-1>", lambda _e: self.close())
        self._body = body
        self._tk = tk

        for widget in (root, frame, body):
            self._bind_drag(widget)
        root.bind("<Escape>", lambda _e: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.report_callback_exception = self._report_exception

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
        try:
            data = json.loads(_pos_file().read_text(encoding="utf-8"))
            x, y = int(data["x"]), int(data["y"])
            if -self.width // 2 <= x <= sw - 40 and 0 <= y <= sh - 40:
                return x, y
        except Exception:
            pass
        # Right side, below the top HUD, left of the player list.
        return max(0, sw - self.width - 260), max(0, int(sh * 0.12))

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
        changed = False
        if self._resync:
            self._resync = False
            for topic in TOPICS:
                value = self.bus.latest(topic)
                if value is not None:
                    self._data[topic] = value
            changed = True
        while True:
            try:
                topic, payload = self._q.get_nowait()
            except queue.Empty:
                break
            self._data[topic] = payload
            changed = True
        if changed:
            try:
                self._render()
            except Exception as exc:
                self.log(f"置顶小窗刷新失败：{exc!r}")
        root.after(POLL_MS, self._poll)

    def _render(self) -> None:
        lines = overlay_lines(
            self._data.get("advice"),
            self._data.get("requests"),
            self._data.get("state"),
            self._data.get("status"),
        )
        if lines == self._last_lines:
            return
        self._last_lines = lines
        tk = self._tk
        for child in self._body.winfo_children():
            child.destroy()
        wrap = self.width - 30
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
            label.pack(fill="x", anchor="w", pady=(0, 3 if style == "headline" else 1))
            self._bind_drag(label)


__all__ = ["Overlay", "display_available", "overlay_lines"]
