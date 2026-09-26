"""Orchestrator: capture loop -> perception -> tracker -> analysis -> advice -> UIs.

Threads:
  * capture thread  grabs frames, detects round / shop changes (auto mode)
  * worker thread   runs one analysis job at a time (latest job wins)
  * ask thread      answers free-form questions (one at a time)
  * hotkeys         Win32 message loop (Windows only)
  * dashboard       HTTP server
The main thread runs the tkinter overlay if enabled, otherwise just waits.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

from .bus import EventBus
from .config import Config
from .models import Advice, Analysis, GameState, Observation, ScreenObservation, ScreenType

PRIORITY = {"scout": 4, "manual": 4, "auto": 3, "reanalyze": 2, "shop": 1}


@dataclass(order=True)
class Job:
    priority: int
    purpose: str = field(compare=False)  # auto | manual | scout | shop | reanalyze
    image: Optional[Image.Image] = field(default=None, compare=False)
    extra: dict[str, Any] = field(default_factory=dict, compare=False)
    created_at: float = field(default_factory=time.time, compare=False)


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


class GameLogger:
    """Appends one JSON line per analysis to ~/.tft_advisor/logs/<game>.jsonl."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._path: Optional[Path] = None
        self._lock = threading.Lock()

    def new_game(self) -> None:
        with self._lock:
            self._path = None

    def write(self, record: dict[str, Any]) -> None:
        with self._lock:
            try:
                if self._path is None:
                    self.directory.mkdir(parents=True, exist_ok=True)
                    self._path = self.directory / time.strftime("game-%Y%m%d-%H%M%S.jsonl")
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
        perceiver: Any = None,
        fast_perceiver: Any = None,
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
        from .data.comps import load_comps
        from .data.mechanics import load_mechanics
        from .data.setdata import load_set_data
        from .engine.analyzer import Analyzer
        from .engine.tracker import GameTracker
        from .llm import LLM, has_credentials

        self.cfg = cfg
        self.bus = bus or EventBus()
        self.console = console
        self.clock = clock
        self.set_data = set_data or load_set_data(cfg.data, offline=offline_data, log=self.info)
        self.mech = mech or load_mechanics(cfg.data.mechanics_file or None)
        self.comps = load_comps(cfg.data.comps_file or None, self.set_data)
        self.tracker = GameTracker(self.set_data, self.mech)
        self.analyzer = Analyzer(self.set_data, self.mech, self.comps, cfg.advisor.comp_hint)
        self.rules = RulesAdvisor(cfg.hotkeys)
        self.scout_planner = ScoutPlanner(cfg.advisor.max_scout_requests_per_stage, cfg.hotkeys.scout)
        self.game_log = GameLogger(cfg.cache_dir / "logs")

        if use_llm is None:
            use_llm = llm is not None or has_credentials()
        self.llm = llm if llm is not None else (LLM(cfg.anthropic) if use_llm else None)

        self.perceiver = perceiver if perceiver is not None else self._default_perceiver()
        self.fast_perceiver = fast_perceiver if fast_perceiver is not None else self._default_fast_perceiver()
        if strategist is not None:
            self.strategist = strategist
        elif self.llm is not None and cfg.advisor.llm_strategy:
            from .advisor.strategist import ClaudeStrategist

            self.strategist = ClaudeStrategist(self.llm, cfg.anthropic, self.set_data)
        else:
            self.strategist = None

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
        self._hotkeys: Any = None
        self._server: Any = None
        self._speaker: Any = None
        self._unsubscribe = self.bus.subscribe("command", self._on_command)

    # ------------------------------------------------------------------ logging
    def info(self, text: str) -> None:
        self._log(text, "info")

    def warn(self, text: str) -> None:
        self._log(text, "warn")

    def _log(self, text: str, level: str) -> None:
        if self.console:
            print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)
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
                "busy": self.busy,
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
                self.info(f"已手动修正 {payload.get('field')} = {payload.get('value')}")
                self._slot.put(Job(PRIORITY["reanalyze"], "reanalyze"))
            elif cmd == "set_comp":
                self.analyzer.comp_hint = str(payload.get("comp") or "")
                self.info(f"目标阵容设为: {self.analyzer.comp_hint or '自动'}")
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
        self.tracker.reset()
        self.game_log.new_game()
        self._last_advice = None
        self.info("新对局：已重置")
        self._publish_state(self.tracker.state)
        self.publish_status()

    # ---------------------------------------------------------------- pipeline
    def request_analysis(self, purpose: str, image: Optional[Image.Image] = None, **extra: Any) -> bool:
        """Queue an analysis job. The frame is grabbed by the worker if not given."""
        return self._slot.put(Job(PRIORITY.get(purpose, 2), purpose, image, extra))

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

        hint = PerceptionHint(
            champion_names=self.set_data.champion_names(),
            item_names=self.set_data.item_names(),
            trait_names=sorted({t.name for t in self.set_data.traits.values()}),
            self_name=self.tracker.state.self_name,
            scouting_player=extra.get("player"),
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
        # Cheap exact sources refine numbers read by the model.
        if self.fast_perceiver is not None and self.fast_perceiver is not self.perceiver and purpose != "shop":
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
        if purpose == "scout" and extra.get("player") and not obs.screen.viewed_player_name:
            obs.screen.viewed_player_name = str(extra["player"])
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
        self.busy = True
        self.publish_status()
        try:
            if job.purpose != "reanalyze":
                image = job.image if job.image is not None else self._grab()
                if image is None and self.perceiver is not None and getattr(self.perceiver, "name", "") != "mock":
                    self.warn("没有截到游戏画面（检查游戏是否在运行，或运行 tft-advisor calibrate）")
                    return None
                if image is None:
                    image = Image.new("RGB", (1920, 1080))
                if self.perceiver is None and self.fast_perceiver is None:
                    self.warn("没有可用的识别方式：请设置 ANTHROPIC_API_KEY 或安装 OCR (pip install rapidocr-onnxruntime)")
                    return None
                if self.cfg.capture.save_screenshots and job.purpose != "shop":
                    self._save_frame(image, job.purpose)
                obs = self._perceive(image, job.purpose, job.extra)
                if obs is None:
                    return None
                state = self.tracker.ingest(obs)
                if obs.purpose == "scout":
                    who = obs.screen.viewed_player_name or "对手"
                    self.info(f"已记录 {who} 的棋盘")
            else:
                state = self.tracker.state
            return self._advise(state, job)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.warn(f"分析出错: {self.last_error}")
            if self.console:
                traceback.print_exc()
            return None
        finally:
            self.busy = False
            self.publish_status()

    def _advise(self, state: GameState, job: Job) -> Advice:
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
        if use_llm:
            self._last_llm_ts = self.clock()
            llm_advice = self.strategist.advise(state, analysis, advice)
            if llm_advice is not advice and getattr(llm_advice, "source", "") == "llm":
                advice = llm_advice
                self._publish_advice(advice)
            elif getattr(self.strategist, "last_error", None):
                self.last_error = str(self.strategist.last_error)
        self.game_log.write(
            {
                "ts": self.clock(),
                "purpose": job.purpose,
                "state": _dump(state),
                "analysis": _dump(analysis),
                "advice": _dump(advice),
            }
        )
        return advice

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
                answer = self.ask(question)
                self.bus.publish("answer", {"question": question, "answer": answer, "ts": self.clock()})
            finally:
                self._ask_lock.release()

        threading.Thread(target=run, name="ask", daemon=True).start()

    def ask(self, question: str) -> str:
        if self.strategist is None:
            return "未启用 Claude（没有 API Key），无法回答自由提问"
        state = self.tracker.state
        analysis = self.analyzer.analyze(state, self.tracker.taken_by_player())
        return self.strategist.ask(question, state, analysis)

    # ----------------------------------------------------------------- threads
    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            job = self._slot.get(timeout=0.5)
            if job is not None:
                self.run_job(job)

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
                self.request_analysis("auto")
                continue
            if self._pending_shop_at is not None and now >= self._pending_shop_at:
                self._pending_shop_at = None
                self.request_analysis("shop")
                continue
            image = self._grab()
            if image is None:
                continue
            try:
                events = watcher.update(image)
            except Exception:
                continue
            if "round_changed" in events:
                self._pending_auto_at = now + self.cfg.capture.settle_delay_s
                self._pending_shop_at = None
            elif "shop_changed" in events and self.cfg.advisor.shop_watch and self._pending_auto_at is None:
                if self.tracker.state.screen_type in (ScreenType.PLANNING, ScreenType.OTHER):
                    self._pending_shop_at = now + 0.4

    def start(self, *, dashboard: bool = True, hotkeys: bool = True, voice: Optional[bool] = None, capture: bool = True) -> Optional[str]:
        """Start background threads. Returns the dashboard URL if started."""
        url = None
        self._stop.clear()
        for target, name in ((self._worker_loop, "worker"), (self._capture_loop, "capture")):
            if name == "capture" and not capture:
                continue
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        if dashboard:
            from .ui.server import DashboardServer

            self._server = DashboardServer(self.bus, self.cfg.ui, log=self.info)
            url = self._server.start()
        if hotkeys and self.cfg.hotkeys.enabled:
            from .capture.hotkeys import HotkeyManager

            bindings = {
                self.cfg.hotkeys.analyze: lambda: self.request_analysis("manual"),
                self.cfg.hotkeys.scout: lambda: self.request_analysis("scout"),
                self.cfg.hotkeys.toggle_auto: self.toggle_auto,
                self.cfg.hotkeys.shop: lambda: self.request_analysis("shop"),
            }
            self._hotkeys = HotkeyManager(bindings, log=self.info)
            if not self._hotkeys.start():
                self.info("全局热键不可用：请用网页上的按钮")
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
        self._unsubscribe()

    def wait(self) -> None:
        """Block the main thread (overlay if enabled) until Ctrl+C."""
        try:
            if self.cfg.ui.overlay:
                from .ui.overlay import Overlay

                overlay = Overlay(self.bus, self.cfg.ui, log=self.info)
                if overlay.run() is not False:
                    return
            while not self._stop.wait(0.5):
                pass
        except KeyboardInterrupt:
            pass
