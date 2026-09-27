"""Orchestrator: capture loop -> perception -> tracker -> analysis -> advice -> UIs.

Threads:
  * capture thread  grabs frames, detects round / shop changes (auto mode)
  * worker thread   runs one perception job at a time (latest job wins)
  * strategy thread runs the Claude strategist after a job (latest wins), so a
                    slow strategy call never delays the next perception job
  * ask thread      answers free-form questions (one at a time)
  * hotkeys         Win32 message loop (Windows only)
  * dashboard       HTTP server
The main thread runs the tkinter overlay if enabled, otherwise just waits.

Frames for player-triggered jobs (F6 / F7 / F9 and the dashboard buttons) are
grabbed when the key is pressed, not when the worker gets to the job: the
player may already have switched the camera back. Automatic jobs (round or
shop change) only use frames of the game window while it is in the
foreground, so the desktop or another app is never sent to Claude.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

from .bus import EventBus
from .config import Config
from .models import Advice, Analysis, GameState, Observation, ScreenObservation, ScreenType

PRIORITY = {"scout": 4, "manual": 4, "auto": 3, "reanalyze": 2, "shop": 1}
#: Jobs whose frame is grabbed at trigger time (the player is looking at it now).
TRIGGER_GRAB = ("manual", "scout", "shop")

NO_PERCEIVER_MSG = (
    "没有可用的识别方式：请设置 ANTHROPIC_API_KEY，或在 tft-advisor 目录运行 "
    'pip install -e ".[ocr]" 安装 OCR，现在只能用「手动修正」'
)

#: Default for ``AdvisorApp(perceiver=..., fast_perceiver=...)``: detect what is
#: available (Claude vision, local OCR). ``None`` means "none", not "detect".
AUTO: Any = type("_Auto", (), {"__repr__": lambda self: "AUTO"})()
AUTO_PAUSED_MSG = "没找到游戏窗口或游戏不在前台，自动分析暂停"


def _console_safe(text: str) -> str:
    """Escape control / format characters so dashboard input cannot drive the terminal."""
    return "".join(ch if unicodedata.category(ch) not in ("Cc", "Cf") else repr(ch)[1:-1] for ch in str(text))


def _strip_controls(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch) not in ("Cc", "Cf"))


@dataclass(order=True)
class Job:
    priority: int
    purpose: str = field(compare=False)  # auto | manual | scout | shop | reanalyze
    image: Optional[Image.Image] = field(default=None, compare=False)
    extra: dict[str, Any] = field(default_factory=dict, compare=False)
    created_at: float = field(default_factory=time.time, compare=False)
    # New-game generation the job belongs to (None = whatever is current when it runs).
    generation: Optional[int] = field(default=None, compare=False)
    # Automatic job: only perceive a frame of the foreground game window.
    gated: bool = field(default=False, compare=False)


class JobSlot:
    """Holds at most one pending job; a new job replaces it unless the pending
    one is more important (a full analysis is never dropped for a shop read)."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._job: Optional[Job] = None

    def put(self, job: Job) -> bool:
        with self._cond:
            if self._job is not None and self._job.priority > job.priority:
                return False
            self._job = job
            self._cond.notify()
            return True

    def get(self, timeout: float = 0.5) -> Optional[Job]:
        with self._cond:
            if self._job is None:
                self._cond.wait(timeout)
            job, self._job = self._job, None
            return job

    def pending(self) -> Optional[Job]:
        with self._cond:
            return self._job

    def clear(self) -> None:
        with self._cond:
            self._job = None


class GameLogger:
    """Appends one JSON line per analysis to ~/.tft_advisor/logs/game-<start>.jsonl.

    One file per game. When a new file is started only the newest ``keep``
    game logs are kept (0 = keep all): they contain other players' names.
    """

    def __init__(self, directory: Path, keep: int = 0) -> None:
        self.directory = directory
        self.keep = max(0, int(keep))
        self._path: Optional[Path] = None
        self._lock = threading.Lock()

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def new_game(self) -> None:
        with self._lock:
            self._path = None

    def _new_path(self) -> Path:
        # Private directory on POSIX (mode is ignored on Windows).
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        now = time.time()
        base = time.strftime("game-%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int(now * 1000) % 1000:03d}"
        path, n = self.directory / f"{base}.jsonl", 0
        while path.exists():  # two games started in the same millisecond (tests)
            n += 1
            path = self.directory / f"{base}-{n}.jsonl"
        return path

    def prune(self) -> None:
        """Delete the oldest game logs beyond ``keep`` (never the current one)."""
        if not self.keep:
            return
        try:
            logs = sorted((p for p in self.directory.glob("game-*.jsonl") if p.is_file()), key=lambda p: p.stem)
        except OSError:
            return
        for old in logs[: max(0, len(logs) - self.keep)]:
            if old == self._path:
                continue
            try:
                old.unlink()
            except OSError:
                pass

    def write(self, record: dict[str, Any]) -> None:
        with self._lock:
            try:
                if self._path is None:
                    self._path = self._new_path()
                    self._path.touch()
                    self.prune()
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            except OSError:
                pass


def _dump(model: Any) -> Any:
    return model.model_dump(mode="json") if hasattr(model, "model_dump") else model


class AdvisorApp:
    def __init__(
        self,
        cfg: Config,
        *,
        bus: Optional[EventBus] = None,
        set_data: Any = None,
        mech: Any = None,
        capturer: Any = None,
        perceiver: Any = AUTO,
        fast_perceiver: Any = AUTO,
        llm: Any = None,
        strategist: Any = None,
        use_llm: Optional[bool] = None,
        live_client: Any = None,
        offline_data: bool = False,
        console: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        from .advisor.rules import RulesAdvisor
        from .advisor.scouting import ScoutPlanner
        from .data.comps import comps_reference_text, load_comps
        from .data.mechanics import load_mechanics
        from .data.setdata import load_set_data, set_notes
        from .engine.analyzer import Analyzer
        from .engine.tracker import GameTracker
        from .llm import LLM, has_credentials

        self.cfg = cfg
        self.bus = bus or EventBus()
        self.console = console
        self.clock = clock
        # A cache or the bundled snapshot starts the game right away; a stale
        # export is refreshed in the background and applies on the next start.
        self.set_data = set_data or load_set_data(cfg.data, offline=offline_data, log=self.info, background=True)
        self.mech = mech or load_mechanics(cfg.data.mechanics_file or None)
        self.comps = load_comps(cfg.data.comps_file or None, self.set_data, log=self.warn)
        self.tracker = GameTracker(self.set_data, self.mech)
        self.analyzer = Analyzer(self.set_data, self.mech, self.comps, cfg.advisor.comp_hint)
        self.rules = RulesAdvisor(cfg.hotkeys, mech=self.mech, set_data=self.set_data)
        self.scout_planner = ScoutPlanner(cfg.advisor.max_scout_requests_per_stage, cfg.hotkeys.scout, mech=self.mech)
        self.game_log = GameLogger(cfg.cache_dir / "logs", keep=cfg.data.keep_game_logs)

        if use_llm is None:
            use_llm = llm is not None or has_credentials()
        self.llm = llm if llm is not None else (LLM(cfg.anthropic) if use_llm else None)

        # Only AUTO auto-detects: None / False mean "no perceiver" (demo, replay
        # --mock and tests must not silently pick up a locally installed OCR).
        self.perceiver = self._default_perceiver() if perceiver is AUTO else (None if perceiver is False else perceiver)
        self.fast_perceiver = (
            self._default_fast_perceiver() if fast_perceiver is AUTO else (None if fast_perceiver is False else fast_perceiver)
        )
        if strategist is not None:
            self.strategist = strategist
        elif self.llm is not None and cfg.advisor.llm_strategy:
            from .advisor.strategist import ClaudeStrategist

            reference = "\n\n".join(
                x for x in (set_notes(self.set_data.set_number), comps_reference_text(self.comps)) if x
            )
            self.strategist = ClaudeStrategist(
                self.llm,
                cfg.anthropic,
                self.set_data,
                extra_reference=reference,
                hotkeys=cfg.hotkeys,
                scout_enabled=cfg.advisor.scout_prompts,
            )
        else:
            self.strategist = None
        # The rules text points to the Claude advice only when a strategist exists.
        self.rules.claude_enabled = self.strategist is not None

        self.capturer = capturer
        self.live = live_client
        self.auto = cfg.advisor.auto
        self.busy = False
        self.last_error: Optional[str] = None
        self._slot = JobSlot()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._ask_lock = threading.Lock()
        self._last_llm_ts = 0.0
        self._last_advice: Optional[Advice] = None
        self._pending_auto_at: Optional[float] = None
        self._pending_shop_at: Optional[float] = None
        self._last_shop_request = float("-inf")
        self._hotkeys: Any = None
        self._server: Any = None
        # cli `run --new-token`: rotate the saved LAN dashboard token.
        self.new_dashboard_token = False
        self._speaker: Any = None
        # Bumped by new_game(): results of jobs started before it are dropped.
        self._generation = 0
        self._state_lock = threading.RLock()
        self._logged_game_id: Optional[str] = self.tracker.state.game_id
        self._auto_paused = False
        # Strategist: its own single-slot thread once start() ran, inline otherwise.
        self._strategy_slot = JobSlot()
        self._strategy_async = False
        self._strategy_busy = False
        self._strategy_seq = 0
        self._unsubscribe = self.bus.subscribe("command", self._on_command)

    # ------------------------------------------------------------------ logging
    def info(self, text: str) -> None:
        self._log(text, "info")

    def warn(self, text: str) -> None:
        self._log(text, "warn")

    def _log(self, text: str, level: str) -> None:
        if self.console:
            print(f"[{time.strftime('%H:%M:%S')}] {_console_safe(text)}", flush=True)
        self.bus.log(text, level)

    # ------------------------------------------------------------ construction
    def _default_perceiver(self) -> Any:
        if self.llm is not None:
            from .vision.claude_vision import ClaudeVisionPerceiver

            return ClaudeVisionPerceiver(self.llm, self.cfg.anthropic, self.set_data)
        return self._default_fast_perceiver()

    def _default_fast_perceiver(self) -> Any:
        try:
            from .vision.ocr import OcrPerceiver, ocr_available

            if ocr_available():
                return OcrPerceiver(self.set_data)
        except Exception:
            pass
        return None

    def perception_mode(self) -> str:
        names = [getattr(p, "name", type(p).__name__) for p in (self.perceiver, self.fast_perceiver) if p is not None]
        return "+".join(dict.fromkeys(names)) or "manual"

    # ------------------------------------------------------------------ status
    def publish_status(self) -> None:
        stats = self.llm.stats.as_dict() if self.llm is not None else {}
        self.bus.publish(
            "status",
            {
                "auto": self.auto,
                "busy": self.busy or self._strategy_busy,
                "thinking": self._strategy_busy,
                "auto_paused": self._auto_paused,
                "capture_error": getattr(self.capturer, "last_error", None) if self.capturer is not None else None,
                "comp_hint": self.analyzer.comp_hint,
                "comp_names": [c.name for c in self.comps][:80],
                "llm": self.llm is not None,
                "strategist": self.strategist is not None,
                "perceiver": self.perception_mode(),
                "last_error": self.last_error,
                "llm_stats": stats,
                "calls": stats.get("calls", 0),
                "set": f"S{self.set_data.set_number} {self.set_data.set_name}",
                "data_source": self.set_data.source,
                "hotkeys": {
                    "analyze": self.cfg.hotkeys.analyze,
                    "scout": self.cfg.hotkeys.scout,
                    "toggle_auto": self.cfg.hotkeys.toggle_auto,
                    "shop": self.cfg.hotkeys.shop,
                },
                "ts": self.clock(),
            },
        )

    # ---------------------------------------------------------------- commands
    def _on_command(self, _topic: str, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        cmd = payload.get("cmd")
        try:
            if cmd == "analyze":
                self.request_analysis("manual")
            elif cmd == "scout":
                self.request_analysis("scout", player=payload.get("player"))
            elif cmd == "shop":
                self.request_analysis("shop")
            elif cmd == "toggle_auto":
                self.toggle_auto()
            elif cmd == "ask":
                q = str(payload.get("question") or "").strip()
                if q:
                    self.ask_async(q[:500])
            elif cmd == "dismiss":
                self.scout_planner.dismiss(str(payload.get("id") or ""))
                self._publish_requests()
            elif cmd == "set_field":
                self.tracker.set_field(str(payload.get("field")), payload.get("value"))
                from .engine.tracker import FIELD_ZH

                fname = _strip_controls(str(payload.get("field")))[:40]
                value = _strip_controls(str(payload.get("value")))[:40]
                self.info(f"已手动修正 {FIELD_ZH.get(fname, fname)} = {value}")
                self._slot.put(Job(PRIORITY["reanalyze"], "reanalyze"))
            elif cmd == "set_comp":
                self.analyzer.comp_hint = _strip_controls(str(payload.get("comp") or "")).strip()[:200]
                self.info(f"目标阵容设为: {self.analyzer.comp_hint or '自动'}")
                self.publish_status()
                self._slot.put(Job(PRIORITY["reanalyze"], "reanalyze"))
            elif cmd == "new_game":
                self.new_game()
        except Exception as exc:  # a bad command must not kill anything
            self.warn(f"命令 {cmd} 失败: {exc}")

    def toggle_auto(self) -> None:
        self.auto = not self.auto
        self.info("自动分析: " + ("开" if self.auto else "关"))
        self.publish_status()

    def new_game(self) -> None:
        """Manual reset (dashboard button). The target comp goes back to the
        configured default; jobs already in flight are dropped when they finish."""
        with self._state_lock:
            self._generation += 1
            self._slot.clear()
            self._strategy_slot.clear()
            state = self.tracker.reset()
            self._logged_game_id = state.game_id
            self.game_log.new_game()
            self._last_advice = None
            self._last_llm_ts = 0.0
            self.scout_planner.reset()
            self.analyzer.comp_hint = self.cfg.advisor.comp_hint
        self.info("新对局：已重置" + (f"（目标阵容: {self.analyzer.comp_hint}）" if self.analyzer.comp_hint else ""))
        self._publish_state(self.tracker.state)
        # Clear the last game's advice everywhere (dashboard, overlay, voice).
        # No gold is known yet: no economy plan and no confidence badge.
        analysis = _dump(Analysis())
        analysis["econ"] = None
        self.bus.publish("analysis", analysis)
        advice = _dump(Advice(headline="新对局，等待第一次分析", source="rules"))
        advice["confidence"] = None
        self.bus.publish("advice", advice)
        self._publish_requests()
        self.publish_status()

    def _on_game_change(self, state: GameState) -> None:
        """The tracker saw a new game on its own (stage back to 1-x): new log file."""
        if not state.game_id or state.game_id == self._logged_game_id:
            return
        self._logged_game_id = state.game_id
        self.game_log.new_game()
        self.scout_planner.reset()
        self._last_llm_ts = 0.0
        self._last_advice = None
        self._strategy_slot.clear()
        self.info("检测到新对局，开始新的对局日志")
        if self.analyzer.comp_hint:
            self.info(f"目标阵容仍是 {self.analyzer.comp_hint}（可以在看板上清空）")

    # ---------------------------------------------------------------- pipeline
    def request_analysis(self, purpose: str, image: Optional[Image.Image] = None, **extra: Any) -> bool:
        """Queue an analysis job for a player action (hotkey / dashboard button).

        For manual / scout / shop the frame is grabbed right now, in the
        caller's thread: the worker may be busy for seconds and the player
        flips the camera back right after pressing the key. If that grab
        fails the worker tries again (and explains why it failed)."""
        if image is None and purpose in TRIGGER_GRAB and self.capturer is not None:
            image = self._grab()
        return self._slot.put(Job(PRIORITY.get(purpose, 2), purpose, image, extra, generation=self._generation))

    def _request_auto(self, purpose: str) -> bool:
        """Queue an automatic job (capture loop); its frame is grabbed and
        checked by the worker (game window in the foreground only)."""
        return self._slot.put(Job(PRIORITY.get(purpose, 2), purpose, generation=self._generation, gated=True))

    def _game_frame_ok(self) -> bool:
        """True when the capturer's last frame shows the game: taken from the
        game window (not the whole-monitor fallback) while the game is in the
        foreground. Capturers without that information (files, tests) pass."""
        cap = self.capturer
        if cap is None:
            return True
        source = getattr(cap, "last_source", None)
        if source is None:
            return True
        if source != "window" and not (source == "monitor" and not self.cfg.capture.use_window):
            return False  # window not found / minimized: the monitor shows something else
        foreground = getattr(cap, "game_foreground", None)
        if callable(foreground) and self.cfg.capture.require_foreground:
            try:
                if foreground() is False:  # None = unknown (not Windows)
                    return False
            except Exception:
                pass
        return True

    def _set_auto_paused(self, paused: bool) -> None:
        if paused == self._auto_paused:
            return
        self._auto_paused = paused
        if paused:
            self.info(AUTO_PAUSED_MSG)
        self.publish_status()

    def _grab(self) -> Optional[Image.Image]:
        if self.capturer is None:
            return None
        try:
            return self.capturer.grab()
        except Exception as exc:
            self.warn(f"截图失败: {exc}")
            return None

    def _perceive(self, image: Image.Image, purpose: str, extra: dict[str, Any]) -> Optional[Observation]:
        from .vision.base import PerceptionError, PerceptionHint

        from .vision.base import clean_name

        # "Likely on screen" hints: what the tracker saw last time (board, bench,
        # shop, items, traits). The full name lists already live in the cached
        # system prompt, so repeating them here would only add tokens.
        last = self.tracker.state
        player = clean_name(extra.get("player")) if extra.get("player") else None
        hint = PerceptionHint(
            champion_names=[u.name for u in (*last.board, *last.bench)] + [u.name for u in last.shop_units if u],
            item_names=list(last.item_bench) + [i for u in last.board for i in u.items],
            trait_names=[t.name for t in last.traits],
            self_name=last.self_name,
            scouting_player=player,
        )
        order = [self.fast_perceiver, self.perceiver] if purpose == "shop" else [self.perceiver]
        obs: Optional[Observation] = None
        for perceiver in order:
            if perceiver is None:
                continue
            try:
                obs = perceiver.perceive(image, purpose=purpose, hint=hint)
                break
            except PerceptionError as exc:
                self.last_error = str(exc)
                self.warn(f"识别失败 ({getattr(perceiver, 'name', '?')}): {exc}")
        if obs is None:
            return None
        # Optional OCR cross-check of the numbers read by the model (costs a
        # second pass over the frame, so it is off unless configured).
        if (
            self.cfg.advisor.ocr_crosscheck
            and self.fast_perceiver is not None
            and self.fast_perceiver is not self.perceiver
            and purpose != "shop"
        ):
            try:
                fast = self.fast_perceiver.perceive(image, purpose=purpose, hint=hint)
                from .vision.merge import merge_observations

                obs = Observation(
                    screen=merge_observations(obs.screen, fast.screen, prefer_secondary={"gold", "stage"}),
                    captured_at=obs.captured_at,
                    source=obs.source + "+" + fast.source,
                    purpose=obs.purpose,
                    latency_s=obs.latency_s,
                )
            except Exception:
                pass
        live = self._live_observation()
        if live is not None:
            from .vision.merge import merge_observations

            obs = obs.model_copy(update={"screen": merge_observations(obs.screen, live, prefer_secondary={"level"})})
        if (
            purpose == "scout"
            and player
            and not obs.screen.viewed_player_name
            and obs.screen.viewing_own_board is not True  # still our own board: never file it under `player`
        ):
            obs.screen.viewed_player_name = player
        if purpose == "scout":
            obs.purpose = "scout"
        return obs

    def _live_observation(self) -> Optional[ScreenObservation]:
        if self.live is None:
            return None
        try:
            data = self.live.fetch()
            if data and self.live.is_tft(data):
                return self.live.to_screen_observation(data)
        except Exception:
            return None
        return None

    def run_job(self, job: Job) -> Optional[Advice]:
        """Run one job synchronously (worker thread, replay and tests)."""
        generation = job.generation if job.generation is not None else self._generation
        self.busy = True
        self.publish_status()
        try:
            if job.purpose != "reanalyze":
                image = job.image if job.image is not None else self._grab()
                if image is None:
                    if self.capturer is not None:
                        reason = getattr(self.capturer, "last_error", None)
                        if job.gated:
                            return None  # the capture loop already reports this state
                        self.warn(
                            "没有截到游戏画面"
                            + (f"：{reason}" if reason else "（检查游戏是否在运行，或运行 tft-advisor calibrate）")
                        )
                        return None
                    # No capturer (demo / tests): perceivers that replay data ignore the frame.
                    image = Image.new("RGB", (1920, 1080))
                if job.gated and job.image is None and not self._game_frame_ok():
                    self._set_auto_paused(True)
                    return None
                if self.perceiver is None and self.fast_perceiver is None:
                    self.last_error = NO_PERCEIVER_MSG
                    self.warn(NO_PERCEIVER_MSG)
                    return None
                if self.cfg.capture.save_screenshots and job.purpose != "shop":
                    self._save_frame(image, job.purpose)
                obs = self._perceive(image, job.purpose, job.extra)
                if obs is None:
                    return None
                with self._state_lock:
                    if generation != self._generation:
                        self.info("已开始新对局，丢弃上一局的分析结果")
                        return None
                    before = (
                        {k: _dump(v) for k, v in self.tracker.state.opponents.items()} if obs.purpose == "scout" else {}
                    )
                    state = self.tracker.ingest(obs)
                    self._on_game_change(state)
                if obs.purpose == "scout":
                    self._report_scout(before, state)
            else:
                state = self.tracker.state
            return self._advise(state, job, generation)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.warn(f"分析出错: {self.last_error}")
            if self.console:
                traceback.print_exc()
            return None
        finally:
            self.busy = False
            self.publish_status()

    def _report_scout(self, before: dict[str, Any], state: GameState) -> None:
        """Say whose board was stored, or that nothing was (own board on screen)."""
        from .engine.tracker import UNKNOWN_PLAYER

        stored = [k for k, v in state.opponents.items() if before.get(k) != _dump(v)]
        if not stored:
            self.warn(f"没有记录到对手：截到的像是你自己的棋盘（先切到对手的棋盘，再按 {self.cfg.hotkeys.scout}）")
            return
        who = "对手（没读到名字）" if stored[0] == UNKNOWN_PLAYER else stored[0]
        self.info(f"已记录 {who} 的棋盘")

    def _advise(self, state: GameState, job: Job, generation: Optional[int] = None) -> Advice:
        analysis: Analysis = self.analyzer.analyze(state, self.tracker.taken_by_player())
        if self.cfg.advisor.scout_prompts:
            analysis.scout_requests = self.scout_planner.plan(state, analysis)
        advice = self.rules.advise(state, analysis)
        self._publish_state(state)
        self.bus.publish("analysis", _dump(analysis))
        self._publish_requests()
        self._publish_advice(advice)

        use_llm = (
            self.strategist is not None
            and job.purpose in ("auto", "manual", "reanalyze")
            and state.screen_type not in (ScreenType.LOADING, ScreenType.POST_GAME)
            and (job.purpose == "manual" or self.clock() - self._last_llm_ts >= self.cfg.advisor.min_seconds_between_llm)
        )
        record = {
            "ts": self.clock(),
            "game_id": state.game_id,
            "purpose": job.purpose,
            "state": _dump(state),
            "analysis": _dump(analysis),
            "advice": _dump(advice),
        }
        if use_llm:
            self._last_llm_ts = self.clock()
            if self._strategy_async:
                # The rules advice is already on screen; Claude's advice follows
                # from the strategy thread without holding up the next frame.
                self.game_log.write(record)
                self._strategy_seq += 1
                task = {
                    "seq": self._strategy_seq,
                    "generation": self._generation if generation is None else generation,
                    "game_id": state.game_id,
                    "stage": str(state.stage) if state.stage else None,
                    "state": state,
                    "analysis": analysis,
                    "advice": advice,
                }
                self._strategy_slot.put(Job(0, "strategy", extra=task))
                return advice
            llm_advice = self._call_strategist(state, analysis, advice)
            if llm_advice is not None:
                advice = llm_advice
                self._publish_advice(advice)
                record["advice"] = _dump(advice)
        self.game_log.write(record)
        return advice

    def _call_strategist(self, state: GameState, analysis: Analysis, advice: Advice) -> Optional[Advice]:
        """Claude's advice, or None (the rules advice stays; the reason goes to last_error)."""
        llm_advice = self.strategist.advise(state, analysis, advice)
        if llm_advice is not advice and getattr(llm_advice, "source", "") == "llm":
            return llm_advice
        if getattr(self.strategist, "last_error", None):
            self.last_error = str(self.strategist.last_error)
        return None

    def _strategy_current(self, task: dict[str, Any]) -> bool:
        """A strategy result is shown only if nothing newer replaced its basis."""
        if task["seq"] != self._strategy_seq or task["generation"] != self._generation:
            return False
        now = self.tracker.state
        return now.game_id == task["game_id"] and (str(now.stage) if now.stage else None) == task["stage"]

    def _run_strategy(self, task: dict[str, Any]) -> None:
        self._strategy_busy = True
        self.publish_status()
        try:
            llm_advice = self._call_strategist(task["state"], task["analysis"], task["advice"])
            if llm_advice is None:
                return
            if not self._strategy_current(task):
                return  # the round moved on (or a newer request is queued): stale advice
            self._publish_advice(llm_advice)
            self.game_log.write(
                {
                    "ts": self.clock(),
                    "game_id": task["game_id"],
                    "purpose": "strategy",
                    "stage": task["stage"],
                    "advice": _dump(llm_advice),
                }
            )
        except Exception as exc:  # the strategy thread must survive anything
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.warn(f"Claude 策略出错: {self.last_error}")
        finally:
            self._strategy_busy = False
            self.publish_status()

    def _publish_state(self, state: GameState) -> None:
        self.bus.publish("state", _dump(state))

    def _publish_requests(self) -> None:
        self.bus.publish("requests", [_dump(r) for r in self.scout_planner.open_requests()])

    def _publish_advice(self, advice: Advice) -> None:
        self._last_advice = advice
        self.bus.publish("advice", _dump(advice))
        if self.console:
            acts = " | ".join(a.text for a in advice.actions[:3])
            self.info(f"[{advice.source}] {advice.headline}  {acts}")

    def _save_frame(self, image: Image.Image, tag: str) -> None:
        try:
            from .capture.screen import save_frame

            save_frame(image, self.cfg.capture.screenshot_dir, tag)
        except Exception:
            pass

    # --------------------------------------------------------------------- ask
    def ask_async(self, question: str) -> None:
        def run() -> None:
            if not self._ask_lock.acquire(blocking=False):
                self.bus.publish("answer", {"question": question, "answer": "上一个问题还在处理中，请稍等", "ts": self.clock()})
                return
            try:
                try:
                    answer = self.ask(question)
                except Exception as exc:  # the page waits for an answer: always send one
                    self.warn(f"提问失败: {exc}")
                    answer = f"出错了，没能回答（{type(exc).__name__}）"
                self.bus.publish("answer", {"question": question, "answer": answer, "ts": self.clock()})
            finally:
                self._ask_lock.release()

        threading.Thread(target=run, name="ask", daemon=True).start()

    def ask(self, question: str) -> str:
        if self.strategist is None:
            return "未启用 Claude（没有 API Key），无法回答自由提问"
        state = self.tracker.state
        analysis = self.analyzer.analyze(state, self.tracker.taken_by_player())
        # The advice on screen, so "why should I save?" is answered consistently.
        return self.strategist.ask(question, state, analysis, rules_advice=self._last_advice)

    # ----------------------------------------------------------------- threads
    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            job = self._slot.get(timeout=0.5)
            if job is not None:
                self.run_job(job)

    def _strategy_loop(self) -> None:
        while not self._stop.is_set():
            job = self._strategy_slot.get(timeout=0.5)
            if job is not None:
                self._run_strategy(job.extra)

    def _shop_read_allowed(self) -> bool:
        """Automatic shop reads through Claude must not starve the round analysis."""
        if self.fast_perceiver is not None or self.llm is None:
            return True  # local OCR is free
        limiter = getattr(self.llm, "limiter", None)
        remaining = getattr(limiter, "remaining", None)
        if not callable(remaining):
            return True
        return remaining() > max(0, self.cfg.advisor.shop_reserve_calls)

    def _capture_loop(self) -> None:
        from .capture.change import RoundWatcher

        watcher = RoundWatcher(self.cfg.capture.change_threshold)
        interval = max(0.2, self.cfg.capture.poll_interval_s)
        while not self._stop.wait(interval):
            if not self.auto or self.capturer is None:
                continue
            now = self.clock()
            if self._pending_auto_at is not None and now >= self._pending_auto_at:
                self._pending_auto_at = None
                self._request_auto("auto")
                continue
            if self._pending_shop_at is not None and now >= self._pending_shop_at:
                self._pending_shop_at = None
                self._request_auto("shop")
                continue
            image = self._grab()
            if image is None:
                continue
            if not self._game_frame_ok():
                # Not the game (window not found, minimized, covered by another
                # app): never analyse it, and start fresh when the game is back.
                watcher.reset()
                self._pending_auto_at = None
                self._pending_shop_at = None
                self._set_auto_paused(True)
                continue
            self._set_auto_paused(False)
            try:
                events = watcher.update(image)
            except Exception:
                continue
            if "round_changed" in events:
                self._pending_auto_at = now + self.cfg.capture.settle_delay_s
                self._pending_shop_at = None
            elif "shop_changed" in events and self.cfg.advisor.shop_watch and self._pending_auto_at is None:
                if self.tracker.state.screen_type in (ScreenType.PLANNING, ScreenType.OTHER) and self._shop_read_allowed():
                    # Coalesce rapid rerolls: one read at most every shop_min_interval_s.
                    earliest = self._last_shop_request + max(0.0, self.cfg.advisor.shop_min_interval_s)
                    self._pending_shop_at = max(now + 0.4, earliest)
                    self._last_shop_request = self._pending_shop_at

    def _scout_hotkey(self) -> None:
        # With exactly one open request, the player is almost surely looking at
        # that board: pass the name as a hint, like the dashboard button does.
        open_reqs = [r for r in self.scout_planner.open_requests() if r.target_player]
        if len(open_reqs) == 1:
            self.request_analysis("scout", player=open_reqs[0].target_player)
        else:
            self.request_analysis("scout")

    def start(self, *, dashboard: bool = True, hotkeys: bool = True, voice: Optional[bool] = None, capture: bool = True) -> Optional[str]:
        """Start background threads. Returns the dashboard URL if started.

        The dashboard starts first: if its port cannot be bound the OSError
        propagates before any thread was started."""
        url = None
        self._stop.clear()
        if dashboard:
            from .ui.server import DashboardServer

            self._server = DashboardServer(
                self.bus,
                self.cfg.ui,
                log=self.info,
                token_file=Path(self.cfg.cache_dir) / "dashboard_token",
                new_token=self.new_dashboard_token,
            )
            try:
                url = self._server.start()
            except BaseException:
                self._server = None
                raise
        loops = [(self._worker_loop, "worker"), (self._capture_loop, "capture")]
        if self.strategist is not None:
            loops.append((self._strategy_loop, "strategy"))
            self._strategy_async = True
        for target, name in loops:
            if name == "capture" and not capture:
                continue
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        if self.perception_mode() == "manual":
            self.last_error = NO_PERCEIVER_MSG
            self.warn(NO_PERCEIVER_MSG)
        if hotkeys and self.cfg.hotkeys.enabled:
            from .capture.hotkeys import HotkeyManager

            wanted = [
                (self.cfg.hotkeys.analyze, lambda: self.request_analysis("manual")),
                (self.cfg.hotkeys.scout, self._scout_hotkey),
                (self.cfg.hotkeys.toggle_auto, self.toggle_auto),
                (self.cfg.hotkeys.shop, lambda: self.request_analysis("shop")),
            ]
            bindings: dict[str, Callable[[], None]] = {}
            for key, fn in wanted:
                norm = key.strip().lower()
                if norm in {k.strip().lower() for k in bindings}:
                    self.warn(f"热键 {key} 在配置里重复了，只保留第一个用途")
                    continue
                bindings[key] = fn
            self._hotkeys = HotkeyManager(bindings, log=self.info)
            if not self._hotkeys.start():
                failed = list(getattr(self._hotkeys, "failed", None) or [])
                if getattr(self._hotkeys, "registered", None):
                    self.info(f"部分热键不可用（{'、'.join(failed)}），其余热键正常；不可用的请用网页上的按钮")
                else:
                    self.info("全局热键不可用：请用网页上的按钮")
        # Rules and scouting texts must not name keys that do nothing (hotkeys
        # off, not on Windows, or taken by another program): they point to the
        # dashboard buttons instead.
        registered = {str(k).strip().lower() for k in (getattr(self._hotkeys, "registered", None) or [])}
        keys = [self.cfg.hotkeys.analyze, self.cfg.hotkeys.scout, self.cfg.hotkeys.toggle_auto, self.cfg.hotkeys.shop]
        self.rules.unavailable_keys = {k for k in keys if k.strip().lower() not in registered}
        self.scout_planner.hotkey_available = self.cfg.hotkeys.scout.strip().lower() in registered
        # Claude must not name keys that do nothing: without any registered
        # hotkey it points the player to the dashboard buttons instead.
        if self.strategist is not None and hasattr(self.strategist, "hotkeys"):
            live = self._hotkeys is not None and bool(getattr(self._hotkeys, "registered", None))
            if self.cfg.hotkeys.enabled and not live:
                from dataclasses import replace as _replace

                self.strategist.hotkeys = _replace(self.cfg.hotkeys, enabled=False)
        if voice if voice is not None else self.cfg.ui.voice:
            from .ui.voice import Speaker

            self._speaker = Speaker(self.cfg.ui, log=self.info)
            if self._speaker.start():
                self._speaker.attach(self.bus)
        self.publish_status()
        self._publish_state(self.tracker.state)
        return url

    def stop(self) -> None:
        self._stop.set()
        for obj in (self._hotkeys, self._server, self._speaker):
            if obj is not None:
                try:
                    obj.stop()
                except Exception:
                    pass
        if self.capturer is not None and hasattr(self.capturer, "close"):
            try:
                self.capturer.close()
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        self._strategy_async = False
        self._unsubscribe()

    def wait(self) -> None:
        """Block the main thread (overlay if enabled) until Ctrl+C."""
        try:
            if self.cfg.ui.overlay:
                from .ui.overlay import Overlay

                overlay = Overlay(self.bus, self.cfg.ui, log=self.info, stop_event=self._stop)
                overlay.run()  # returns when closed or unavailable; keep running either way
            while not self._stop.wait(0.5):
                pass
        except KeyboardInterrupt:
            pass
