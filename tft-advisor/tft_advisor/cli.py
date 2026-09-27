"""Command line entry point: ``tft-advisor <command>`` (or ``python -m tft_advisor``)."""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Optional, Sequence

from . import __version__
from .config import Config, load_config


def _utf8_console() -> None:
    # Windows consoles default to a legacy code page; Chinese output needs UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def _load(args: argparse.Namespace) -> Config:
    cfg = load_config(getattr(args, "config", None))
    if getattr(args, "no_llm", False):
        cfg.advisor.llm_strategy = False
    if getattr(args, "no_auto", False):
        cfg.advisor.auto = False
    if getattr(args, "host", None):
        cfg.ui.host = args.host
    if getattr(args, "port", None):
        cfg.ui.port = args.port
    if getattr(args, "overlay", False):
        cfg.ui.overlay = True
    if getattr(args, "voice", False):
        cfg.ui.voice = True
    cfg.validate()  # command line overrides (--port 70000) get the same checks as the file
    return cfg


def _config_line(cfg: Config) -> str:
    return f"配置文件: {cfg.source_path or '(默认值，没有配置文件)'}"


def _open(url: Optional[str], cfg: Config, server: Any = None, wait_s: float = 2.0) -> None:
    """Open the dashboard unless a page from an earlier run is already connected.

    Old dashboard tabs reconnect on their own within a couple of seconds after
    a restart; opening another tab each run piles them up (browsers allow only
    6 live connections per host).
    """
    if not (url and cfg.ui.open_browser):
        return
    if server is not None:
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if getattr(server, "sse_clients", 0) > 0:
                print("看板页面已经打开（已有页面重新连上），不再打开新标签页")
                return
            time.sleep(0.1)
    try:
        webbrowser.open(url)
    except Exception:
        pass


# ------------------------------------------------------------------------ run


def cmd_run(args: argparse.Namespace) -> int:
    from .app import AdvisorApp
    from .capture.screen import ScreenCapturer, set_dpi_awareness
    from .llm import has_credentials
    from .vision.liveclient import LiveClient

    cfg = _load(args)
    print(_config_line(cfg))
    if not args.no_dashboard and cfg.ui.host.strip().lower() not in ("127.0.0.1", "localhost", "::1"):
        print(f"注意: 看板会对局域网开放（host = {cfg.ui.host}），同一网络里拿到访问口令的设备都能打开")
    set_dpi_awareness()
    if not has_credentials():
        print("提示: 没有检测到 ANTHROPIC_API_KEY。没有它就无法用 Claude 识别画面和给策略，只能用 OCR/手动输入。")
    app: Optional[AdvisorApp] = None

    def capture_log(message: str) -> None:
        # Precise capture problems (minimized, black frames, no display) reach the dashboard log.
        if app is not None:
            app.warn(message)
        else:
            print(message)

    capturer = ScreenCapturer(cfg.capture, log=capture_log)
    app = AdvisorApp(cfg, capturer=capturer, live_client=LiveClient(), use_llm=None if not args.no_vision_llm else False)
    app.new_dashboard_token = bool(getattr(args, "new_token", False))
    try:
        try:
            url = app.start(dashboard=not args.no_dashboard)
        except OSError as exc:
            print(f"错误: {exc}（可以加 --no-dashboard 只用热键）", file=sys.stderr)
            return 2
        hk = cfg.hotkeys
        print(f"TFT 助手已启动 ({app.set_data.set_number} {app.set_data.set_name})  识别: {app.perception_mode()}")
        if url:
            print(f"看板: {url}")
        print(f"热键: {hk.analyze}=分析  {hk.scout}=记录对手棋盘  {hk.shop}=读商店  {hk.toggle_auto}=自动开关   Ctrl+C 退出")
        _open(url, cfg, app._server)
        app.wait()
    finally:
        app.stop()
    return 0


# ----------------------------------------------------------------------- demo


def _demo_fixture_dir() -> Path:
    from importlib import resources

    return Path(str(resources.files("tft_advisor.data").joinpath("bundled", "demo_s18")))


def cmd_demo(args: argparse.Namespace) -> int:
    """Replay a scripted sample game through the whole pipeline (no game, no API key)."""
    from .app import AdvisorApp, GameLogger, Job
    from .data.setdata import SetData, bundled_sample, bundled_snapshot
    from .vision.mock import MockPerceiver

    cfg = _load(args)
    print(_config_line(cfg))
    cfg.advisor.auto = False
    fixtures = Path(args.fixtures) if args.fixtures else _demo_fixture_dir()
    if not fixtures.exists():
        print(f"错误: 找不到演示数据 {fixtures}", file=sys.stderr)
        return 2
    perceiver = MockPerceiver(fixtures, loop=True)
    if args.fixtures:
        set_data = SetData.from_cdragon(bundled_sample(), source="bundled-sample")
    else:
        # The scripted demo game uses real Set 18 names (bundled snapshot).
        set_data = SetData.from_cdragon(bundled_snapshot("zh_cn"), bundled_snapshot("en_us"), source="bundled-snapshot")
    app = AdvisorApp(cfg, set_data=set_data, perceiver=perceiver, fast_perceiver=None, use_llm=args.llm, offline_data=True)
    # Keep the scripted game out of the real logs, so `review` never reviews the demo.
    app.game_log = GameLogger(cfg.cache_dir / "logs" / "demo", keep=5)
    stop = threading.Event()
    try:
        try:
            url = app.start(dashboard=not args.no_dashboard, hotkeys=False, capture=False)
        except OSError as exc:
            print(f"错误: {exc}（可以加 --no-dashboard）", file=sys.stderr)
            return 2
        print("演示模式：用一局预先写好的 S18 对局驱动整个流程（不需要游戏）")
        if url:
            print(f"看板: {url}")
        _open(url, cfg, app._server)

        def driver() -> None:
            steps = 0
            while not stop.is_set() and (args.steps <= 0 or steps < args.steps):
                app.run_job(Job(4, "manual"))
                steps += 1
                if stop.wait(args.interval):
                    break

        t = threading.Thread(target=driver, name="demo", daemon=True)
        t.start()
        if args.steps > 0 and args.no_dashboard:
            # Timed joins: an untimed join ignores Ctrl+C on Windows.
            while t.is_alive():
                t.join(0.5)
        else:
            app.wait()
    finally:
        stop.set()
        app.stop()
    return 0


# --------------------------------------------------------------------- replay


def cmd_replay(args: argparse.Namespace) -> int:
    """Run the pipeline over saved screenshots and print the advice."""
    from .app import AdvisorApp, Job
    from .capture.screen import FileCapturer

    cfg = _load(args)
    cap = FileCapturer(args.images)
    if cap.missing:
        print(f"找不到这些文件: {', '.join(cap.missing)}")
    if not cap.paths:
        print("没有找到截图")
        return 2
    perceiver = None
    if args.mock:
        from .vision.mock import MockPerceiver

        if not Path(args.mock).exists():
            print(f"错误: 找不到观察数据 {args.mock}", file=sys.stderr)
            return 2
        perceiver = MockPerceiver(Path(args.mock))
    app = AdvisorApp(cfg, perceiver=perceiver, use_llm=None if not args.no_llm else False)
    # Driven by the capturer, so an unreadable file cannot shift the labels.
    while True:
        start = cap.index
        img = cap.grab()
        skipped = cap.paths[start : cap.index - (1 if img is not None else 0)]
        for bad in skipped:
            print(f"\n=== {bad.name} ===\n无法读取图片，已跳过")
        if img is None or cap.current_path is None:
            break
        print(f"\n=== {cap.current_path.name} ===")
        advice = app.run_job(Job(4, args.purpose, img))
        if advice is None:
            print("(没有结果" + (f": {app.last_error}" if app.last_error else "") + ")")
            continue
        print(f"{advice.headline}")
        for a in advice.actions:
            print(f"  [{a.priority}] {a.text}")
        if advice.plan:
            print(f"  计划: {advice.plan}")
    return 0


# ----------------------------------------------------------------------- odds


def cmd_odds(args: argparse.Namespace) -> int:
    from .data.mechanics import load_mechanics
    from .engine.probability import best_roll_level, expected_gold_to_goal, p_shop_shows, p_unit_per_slot, rolldown_probability

    cfg = load_config(getattr(args, "config", None))
    mech = load_mechanics(cfg.data.mechanics_file or None)
    n_champs = args.champs
    if n_champs is None:
        try:
            from .data.setdata import load_set_data

            sd = load_set_data(cfg.data, offline=True, log=lambda *_: None)
            n_champs = sd.champion_count_by_cost().get(args.cost) or 13
        except Exception:
            n_champs = 13
    size = mech.pool_size[args.cost]
    have = args.have
    goal_copies = 3 ** (args.star - 1)
    need = max(0, goal_copies - have)
    rem_unit = size - have - args.taken
    rem_cost = n_champs * size - have - args.taken - args.other_taken
    p_slot = p_unit_per_slot(mech, args.level, args.cost, rem_unit, rem_cost)
    print(f"{args.cost} 费卡，等级 {args.level}，已有 {have} 张，目标 {args.star} 星（还差 {need} 张），场上别人有 {args.taken} 张")
    print(f"卡池剩余: 这张 {rem_unit} / 同费用 {rem_cost}   单格概率 {p_slot:.2%}   一次刷新至少出现一张 {p_shop_shows(mech, args.level, args.cost, rem_unit, rem_cost):.1%}")
    eg = expected_gold_to_goal(mech, args.level, args.cost, need, rem_unit, rem_cost)
    print("期望花费: " + (f"{eg:.0f} 金币" if eg is not None else "不可能（卡池不够）"))
    for g in (10, 20, 30, 40, 50, 60, 80, 100):
        p = rolldown_probability(mech, args.level, args.cost, need, rem_unit, rem_cost, g)
        print(f"  {g:>3} 金币: {p:6.1%}")
    if args.gold:
        print(f"\n用 {args.gold} 金币：当前等级搜 vs 先升级再搜")
        for lvl, left, p in best_roll_level(mech, args.level, args.xp, args.gold, args.cost, need, rem_unit, rem_cost):
            print(f"  等级 {lvl}: 剩 {left} 金币搜牌 -> {p:.1%}")
    return 0


# ----------------------------------------------------------------------- data


def cmd_data(args: argparse.Namespace) -> int:
    from .data.setdata import load_set_data

    cfg = load_config(getattr(args, "config", None))
    if args.action == "update":
        cfg.data.refresh_hours = 0
    sd = load_set_data(cfg.data, offline=args.action == "show" and args.offline)
    print(f"赛季: {sd.set_number} {sd.set_name}  来源: {sd.source}")
    print(f"英雄 {len(sd.champions)}  羁绊 {len(sd.traits)}  装备 {len(sd.items)}  各费用数量 {sd.champion_count_by_cost()}")
    if args.action == "show" and args.full:
        print(sd.summary_text())
    return 0


# ---------------------------------------------------------------------- doctor


def _check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK' if ok else '!!'}] {label}" + (f": {detail}" if detail else ""))


def cmd_doctor(args: argparse.Namespace) -> int:
    from .llm import has_credentials

    cfg = load_config(getattr(args, "config", None))
    print(f"tft-advisor {__version__}  Python {platform.python_version()}  {platform.system()} {platform.release()}")
    print(f"配置文件: {cfg.source_path or '(默认值)'}")
    print("依赖:")
    for mod, need in (("anthropic", True), ("pydantic", True), ("PIL", True), ("numpy", True), ("mss", True), ("rapidocr_onnxruntime", False), ("pyttsx3", False), ("tkinter", False)):
        try:
            m = importlib.import_module(mod)
            _check(mod, True, getattr(m, "__version__", ""))
        except Exception as exc:
            _check(mod + ("" if need else " (可选)"), not need, f"未安装 ({type(exc).__name__})")
    print("Claude:")
    _check("API 凭证", has_credentials(), "ANTHROPIC_API_KEY 已设置" if os.environ.get("ANTHROPIC_API_KEY") else "未找到 ANTHROPIC_API_KEY")
    _check("视觉模型", True, cfg.anthropic.vision_model + f" (effort {cfg.anthropic.vision_effort})")
    _check("策略模型", True, cfg.anthropic.strategy_model + f" (effort {cfg.anthropic.strategy_effort})")
    print("数据:")
    try:
        from .data.setdata import load_set_data

        sd = load_set_data(cfg.data, offline=args.offline, log=lambda *_: None)
        _check("赛季数据", sd.source != "bundled-sample", f"S{sd.set_number} {sd.set_name}, {len(sd.champions)} 英雄 ({sd.source})")
    except Exception as exc:
        _check("赛季数据", False, str(exc))
    print("截图:")
    try:
        from .capture.screen import ScreenCapturer, find_window_rect, set_dpi_awareness

        set_dpi_awareness()
        rect = find_window_rect(cfg.capture.window_title)
        _check("游戏窗口", rect is not None, str(rect) if rect else f"没找到窗口 '{cfg.capture.window_title}'（游戏没开时正常）")
        cap = ScreenCapturer(cfg.capture)
        try:
            img = cap.grab()
            where = {"window": "游戏窗口", "monitor": "整个显示器"}.get(cap.last_source, cap.last_source)
            _check(
                "截屏",
                img is not None,
                f"{img.size[0]}x{img.size[1]}（{where}）" if img is not None else (cap.last_error or "失败，原因未知"),
            )
        finally:
            cap.close()
    except Exception as exc:
        _check("截屏", False, str(exc))
    try:
        from .vision.liveclient import LiveClient

        data = LiveClient().fetch()
        _check("Riot Live Client API", data is not None, "游戏中可用" if data else "未连接（不在对局中时正常）")
    except Exception as exc:
        _check("Riot Live Client API", False, str(exc))
    return 0


# ------------------------------------------------------------------- calibrate


def cmd_calibrate(args: argparse.Namespace) -> int:
    from PIL import Image

    from .capture.calibrate import draw_regions

    cfg = load_config(getattr(args, "config", None))
    out = Path(os.path.expanduser(args.out)) if args.out else cfg.cache_dir / "calibrate.png"
    if out.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
        print(f"错误: 输出文件需要是 .png 或 .jpg: {out}", file=sys.stderr)
        return 2
    if args.image:
        src = Path(os.path.expanduser(args.image))
        if not src.is_file():
            print(f"错误: 找不到图片 {src}", file=sys.stderr)
            return 2
        try:
            with Image.open(src) as im:
                img = im.convert("RGB")
        except Exception as exc:  # noqa: BLE001 - corrupt / not an image
            print(f"错误: 无法读取图片 {src}（{type(exc).__name__}）", file=sys.stderr)
            return 2
    else:
        from .capture.screen import ScreenCapturer, set_dpi_awareness

        set_dpi_awareness()
        if args.delay:
            print(f"{args.delay} 秒后截图，请切到游戏画面...")
            time.sleep(args.delay)
        cap = ScreenCapturer(cfg.capture)
        try:
            img = cap.grab()
        finally:
            cap.close()
        if img is None:
            print(f"截图失败: {cap.last_error or '原因未知'}")
            return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    draw_regions(img).save(out)
    print(f"已保存区域标注图: {out}  (尺寸 {img.size[0]}x{img.size[1]})")
    print("打开图片检查每个框是否框住了对应的界面元素；若偏移，可在 GitHub issue 里附上此图。")
    return 0


# ---------------------------------------------------------------------- review


def cmd_review(args: argparse.Namespace) -> int:
    from .data.mechanics import load_mechanics
    from .review import format_summary, latest_log, llm_review, load_records, summarize_log

    cfg = load_config(getattr(args, "config", None))
    path = Path(args.log) if args.log else latest_log(cfg.cache_dir / "logs")
    if path is None or not path.is_file():
        print("没有找到对局日志（运行 tft-advisor run 打一局后会自动生成）")
        return 1
    mech = load_mechanics(cfg.data.mechanics_file or None)
    try:
        records = load_records(path)
    except OSError as exc:
        print(f"无法读取日志 {path}（{exc}）")
        return 1
    summary = summarize_log(records, mech)
    print(f"对局日志: {path}")
    print(format_summary(summary))
    if args.llm:
        from .llm import LLM, LLMError, has_credentials

        if not has_credentials():
            print("\n没有检测到 ANTHROPIC_API_KEY，跳过 Claude 复盘")
            return 0
        try:
            text = llm_review(LLM(cfg.anthropic), cfg.anthropic.strategy_model, cfg.anthropic.strategy_effort, summary)
            print("\nClaude 复盘:\n" + text)
        except LLMError as exc:
            print(f"Claude 复盘失败: {exc}")
        except Exception as exc:  # noqa: BLE001 - SDK / transport errors must not show a traceback
            print(f"Claude 复盘失败（{type(exc).__name__}）: {exc}")
    return 0


# ------------------------------------------------------------------------ init


def cmd_init(args: argparse.Namespace) -> int:
    from importlib import resources

    default = Path(os.path.expanduser("~/.tft_advisor/config.toml"))
    dest = Path(os.path.expanduser(args.path)) if args.path else default
    if dest.is_dir():
        print(f"错误: {dest} 是一个目录，请给出文件路径（例如 {dest / 'config.toml'}）", file=sys.stderr)
        return 2
    if dest.exists() and not args.force:
        print(f"{dest} 已存在（用 --force 覆盖）")
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = resources.files("tft_advisor.data").joinpath("bundled", "config.example.toml").read_text(encoding="utf-8")
    dest.write_text(text, encoding="utf-8")
    print(f"已写入配置: {dest}")
    if dest.resolve() != default.resolve():
        print(f"这个位置不会被自动读取：用 tft-advisor --config \"{dest}\" run，或设置环境变量 TFT_ADVISOR_CONFIG")
    return 0


# ----------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tft-advisor", description="云顶之弈实时助手 (个人使用)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--config", help="配置文件路径 (TOML)")
    # "--config" is accepted after the command too (tft-advisor run --config x.toml);
    # SUPPRESS keeps a top-level value when the subcommand does not repeat it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS, help="配置文件路径 (TOML)")
    _sub = p.add_subparsers(dest="command", required=True)

    class _Sub:
        @staticmethod
        def add_parser(name: str, **kw):  # type: ignore[no-untyped-def]
            return _sub.add_parser(name, parents=[common], **kw)

    sub = _Sub()

    r = sub.add_parser("run", help="启动实时助手")
    r.add_argument("--no-llm", action="store_true", help="不用 Claude 给策略（只用规则引擎）")
    r.add_argument("--no-vision-llm", action="store_true", help="完全不调用 Claude（只用 OCR/手动）")
    r.add_argument("--no-auto", action="store_true", help="关闭自动分析，只用热键")
    r.add_argument("--no-dashboard", action="store_true")
    r.add_argument("--overlay", action="store_true", help="显示置顶小窗（无边框模式）")
    r.add_argument("--voice", action="store_true", help="语音播报")
    r.add_argument("--host")
    r.add_argument("--port", type=int)
    r.add_argument("--new-token", action="store_true", help="换一个新的手机看板访问口令（旧链接失效）")
    r.set_defaults(func=cmd_run)

    d = sub.add_parser("demo", help="用内置样例对局演示（不需要游戏和 API Key）")
    d.add_argument("--interval", type=float, default=6.0, help="每一步间隔秒数")
    d.add_argument("--steps", type=int, default=0, help="跑多少步后停止 (0 = 一直循环)")
    d.add_argument("--fixtures", help="观察数据目录 (obs_*.json)")
    d.add_argument("--llm", action="store_true", help="演示时也调用 Claude 策略（需要 API Key）")
    d.add_argument("--no-dashboard", action="store_true")
    d.add_argument("--overlay", action="store_true")
    d.add_argument("--host")
    d.add_argument("--port", type=int)
    d.set_defaults(func=cmd_demo)

    rp = sub.add_parser("replay", help="对保存的截图跑一遍分析")
    rp.add_argument("images", nargs="+", help="截图文件或目录")
    rp.add_argument("--mock", help="用 JSON 观察数据代替 Claude 识别")
    rp.add_argument("--purpose", default="manual", choices=["manual", "scout", "shop"])
    rp.add_argument("--no-llm", action="store_true")
    rp.set_defaults(func=cmd_replay)

    o = sub.add_parser("odds", help="搜牌概率计算器")
    o.add_argument("--level", type=int, required=True)
    o.add_argument("--cost", type=int, required=True, choices=[1, 2, 3, 4, 5])
    o.add_argument("--have", type=int, default=0, help="已有几张 (1星=1, 2星=3)")
    o.add_argument("--star", type=int, default=2, choices=[2, 3], help="目标星级")
    o.add_argument("--taken", type=int, default=0, help="别人拿走了几张这张卡")
    o.add_argument("--other-taken", type=int, default=0, help="同费用其他卡被拿走的总张数")
    o.add_argument("--champs", type=int, help="这个费用有多少种英雄（默认从赛季数据读取）")
    o.add_argument("--gold", type=int, default=0, help="比较：先升级再搜 vs 直接搜")
    o.add_argument("--xp", type=int, default=0, help="当前经验（配合 --gold）")
    o.set_defaults(func=cmd_odds)

    da = sub.add_parser("data", help="赛季数据")
    da.add_argument("action", choices=["update", "show"])
    da.add_argument("--full", action="store_true")
    da.add_argument("--offline", action="store_true")
    da.set_defaults(func=cmd_data)

    doc = sub.add_parser("doctor", help="检查环境")
    doc.add_argument("--offline", action="store_true")
    doc.set_defaults(func=cmd_doctor)

    c = sub.add_parser("calibrate", help="截图并画出识别区域，检查分辨率适配")
    c.add_argument("--image", help="用已有截图")
    c.add_argument("--out")
    c.add_argument("--delay", type=float, default=0.0)
    c.set_defaults(func=cmd_calibrate)

    rv = sub.add_parser("review", help="赛后复盘（读取对局日志）")
    rv.add_argument("log", nargs="?", help="日志文件（默认最近一局）")
    rv.add_argument("--llm", action="store_true", help="让 Claude 写复盘建议")
    rv.set_defaults(func=cmd_review)

    i = sub.add_parser("init", help="生成配置文件")
    i.add_argument("--path")
    i.add_argument("--force", action="store_true")
    i.set_defaults(func=cmd_init)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    from PIL import UnidentifiedImageError

    _utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, UnidentifiedImageError) as exc:
        # Config / input problems: one Chinese line instead of a traceback.
        if os.environ.get("TFT_ADVISOR_DEBUG") == "1":
            raise
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
