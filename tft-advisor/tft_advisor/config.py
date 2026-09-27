"""Configuration: dataclasses with defaults, optionally overridden by a TOML file.

Lookup order for the config file: explicit ``--config`` path, ``$TFT_ADVISOR_CONFIG``,
``~/.tft_advisor/config.toml``. No file = defaults. The current directory is
deliberately not searched: a ``tft_advisor.toml`` shipped in a downloaded
folder could otherwise open the dashboard to the LAN or redirect data silently.

Every error (missing explicit file, bad encoding, bad TOML, unknown key, wrong
type, out of range value, unknown hotkey) is raised as ``ValueError`` or
``FileNotFoundError`` with a short Chinese message the CLI prints as is.
"""

from __future__ import annotations

import dataclasses
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class AnthropicConfig:
    # Models are configurable; claude-opus-5 is the default for both roles.
    # For lower latency / cost you can set vision_model = "claude-sonnet-5".
    vision_model: str = "claude-opus-5"
    strategy_model: str = "claude-opus-5"
    vision_effort: str = "low"  # low | medium | high | xhigh | max
    strategy_effort: str = "medium"
    use_fallbacks: bool = True  # server-side refusal fallbacks (beta)
    max_calls_per_minute: int = 12  # hard client-side rate limit across all calls
    timeout_s: float = 60.0
    max_image_edge: int = 1568  # downscale screenshots so the long edge fits this


@dataclass
class CaptureConfig:
    monitor: int = 1  # mss monitor index (1 = primary)
    window_title: str = "Teamfight Tactics"  # also matched by process name (TFT.exe / TFTClient-Win64-Shipping.exe)
    use_window: bool = True  # crop to the game window if it can be found (Windows)
    require_foreground: bool = True  # auto mode only reads frames while the game window has focus
    poll_interval_s: float = 1.0
    settle_delay_s: float = 1.2  # wait after a round change before analysing
    change_threshold: float = 0.08  # mean abs diff (0..1) that counts as a change
    save_screenshots: bool = False
    screenshot_dir: str = "~/.tft_advisor/captures"


@dataclass
class AdvisorConfig:
    auto: bool = True  # analyse automatically on round change
    llm_strategy: bool = True  # use Claude for strategy on top of the rules engine
    min_seconds_between_llm: float = 8.0
    scout_prompts: bool = True  # ask the human to show other boards
    shop_watch: bool = True  # re-read the shop when it changes (rolls)
    max_scout_requests_per_stage: int = 2
    ocr_crosscheck: bool = False
    shop_min_interval_s: float = 3.0  # automatic shop reads at most this often (roll-downs reroll fast)
    shop_reserve_calls: int = 3  # skip automatic Claude shop reads when fewer calls/min remain  # also run OCR on every frame to double-check gold / stage (slower)
    comp_hint: str = ""  # optional: the comp you want to play, free text


@dataclass
class DataConfig:
    locale: str = "zh_cn"  # display names; en_us is also loaded for matching
    cache_dir: str = "~/.tft_advisor"
    set_number: int = 0  # 0 = latest set found in the data
    comps_file: str = ""  # optional user comps JSON/TOML
    mechanics_file: str = ""  # optional override of shop odds / pool sizes / xp
    cdragon_base: str = "https://raw.communitydragon.org/latest/cdragon/tft"
    refresh_hours: float = 24.0
    keep_game_logs: int = 30  # newest game logs kept in <cache_dir>/logs (0 = keep all)


@dataclass
class UIConfig:
    host: str = "127.0.0.1"  # use 0.0.0.0 to open the dashboard from your phone
    port: int = 8765
    open_browser: bool = True
    overlay: bool = False  # tkinter always-on-top overlay (borderless game mode)
    voice: bool = False  # speak the headline (pyttsx3 / Windows SAPI)
    voice_rate: int = 190


@dataclass
class HotkeyConfig:
    enabled: bool = True
    analyze: str = "F6"  # full analysis now
    scout: str = "F7"  # I am looking at another player's board: record it
    toggle_auto: str = "F8"
    shop: str = "F9"  # read the shop now and tell me what to buy


@dataclass
class Config:
    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    advisor: AdvisorConfig = field(default_factory=AdvisorConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    source_path: Optional[str] = None

    @property
    def cache_dir(self) -> Path:
        p = Path(os.path.expanduser(self.data.cache_dir))
        p.mkdir(parents=True, exist_ok=True)
        return p

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> None:
        """Range / format checks. Raises ``ValueError`` with a Chinese message."""
        a, c, ad, d, u, h = self.anthropic, self.capture, self.advisor, self.data, self.ui, self.hotkeys

        def need(ok: bool, key: str, rule: str) -> None:
            if not ok:
                raise ValueError(f"配置项 {key} {rule}")

        for key in ("vision_effort", "strategy_effort"):
            need(getattr(a, key) in EFFORTS, f"[anthropic].{key}", f"只能是 {' / '.join(EFFORTS)}")
        for key in ("vision_model", "strategy_model"):
            need(bool(getattr(a, key).strip()), f"[anthropic].{key}", "不能为空")
        need(a.max_calls_per_minute >= 1, "[anthropic].max_calls_per_minute", "至少是 1")
        need(a.timeout_s > 0, "[anthropic].timeout_s", "必须大于 0")
        need(a.max_image_edge >= 256, "[anthropic].max_image_edge", "至少是 256")
        need(c.monitor >= 0, "[capture].monitor", "不能小于 0")
        need(c.poll_interval_s > 0, "[capture].poll_interval_s", "必须大于 0")
        need(c.settle_delay_s >= 0, "[capture].settle_delay_s", "不能小于 0")
        need(0 < c.change_threshold < 1, "[capture].change_threshold", "必须在 0 到 1 之间（不含 0 和 1）")
        need(ad.min_seconds_between_llm >= 0, "[advisor].min_seconds_between_llm", "不能小于 0")
        need(ad.max_scout_requests_per_stage >= 0, "[advisor].max_scout_requests_per_stage", "不能小于 0")
        need(d.set_number >= 0, "[data].set_number", "不能小于 0")
        need(d.refresh_hours >= 0, "[data].refresh_hours", "不能小于 0")
        need(d.keep_game_logs >= 0, "[data].keep_game_logs", "不能小于 0（0 = 全部保留）")
        need(bool(d.cache_dir.strip()), "[data].cache_dir", "不能为空")
        need(d.cdragon_base.strip().lower().startswith("https://"), "[data].cdragon_base", "必须是 https:// 开头的地址")
        for key, value in (("[data].cache_dir", d.cache_dir), ("[capture].screenshot_dir", c.screenshot_dir)):
            need(not _is_network_path(value), key, "不能是网络共享路径（\\\\ 或 // 开头），请用本机目录")
        need(bool(u.host.strip()), "[ui].host", "不能为空")
        need(1 <= u.port <= 65535, "[ui].port", "必须在 1 到 65535 之间")
        need(u.voice_rate > 0, "[ui].voice_rate", "必须大于 0")
        if h.enabled:
            from .capture.hotkeys import parse_hotkey

            for key in ("analyze", "scout", "toggle_auto", "shop"):
                value = getattr(h, key)
                try:
                    parse_hotkey(value)
                except ValueError:
                    raise ValueError(
                        f"配置项 [hotkeys].{key} = {value!r} 不是可用的按键（例如 \"F6\" 或 \"ctrl+shift+a\"）"
                    ) from None


EFFORTS = ("low", "medium", "high", "xhigh", "max")


def _is_network_path(value: str) -> bool:
    text = os.path.expanduser(str(value)).strip()
    return text.startswith("\\\\") or text.startswith("//")


def _merge(section_obj: Any, values: dict[str, Any], section: str) -> None:
    known = {f.name: f for f in dataclasses.fields(section_obj)}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"配置里没有 [{section}].{key} 这一项（检查拼写，或删掉这一行）")
        current = getattr(section_obj, key)
        if isinstance(current, bool):
            if not isinstance(value, bool):
                raise ValueError(f"配置项 [{section}].{key} 需要是 true 或 false")
        elif isinstance(current, (int, float)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"配置项 [{section}].{key} 需要是数字（不要加引号）")
            if isinstance(current, int) and isinstance(value, float) and not value.is_integer():
                raise ValueError(f"配置项 [{section}].{key} 需要是整数")
            value = type(current)(value)
        elif isinstance(current, str) and not isinstance(value, str):
            raise ValueError(f"配置项 [{section}].{key} 需要是字符串（用引号括起来）")
        setattr(section_obj, key, value)


def default_config_paths() -> list[Path]:
    """Implicit config locations: ``$TFT_ADVISOR_CONFIG`` (if set), then the home file."""
    paths = []
    env = os.environ.get("TFT_ADVISOR_CONFIG")
    if env:
        paths.append(Path(os.path.expanduser(env)))
    paths.append(Path(os.path.expanduser("~/.tft_advisor/config.toml")))
    return paths


def _read_toml(p: Path) -> dict[str, Any]:
    data = p.read_bytes()
    try:
        text = data.decode("utf-8-sig")  # PowerShell 5.1 writes UTF-8 with a BOM
    except UnicodeDecodeError:
        raise ValueError(f"配置文件 {p} 不是 UTF-8 编码，请用记事本另存为 UTF-8") from None
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"配置文件 {p} 格式错误: {exc}") from None


def load_config(path: Optional[str | Path] = None) -> Config:
    cfg = Config()
    env = os.environ.get("TFT_ADVISOR_CONFIG")
    if path:
        p = Path(os.path.expanduser(str(path)))
        if not p.is_file():
            raise FileNotFoundError(f"找不到配置文件: {p}")
        candidates = [p]
    else:
        candidates = default_config_paths()
        if env and not candidates[0].is_file():
            raise FileNotFoundError(f"环境变量 TFT_ADVISOR_CONFIG 指向的配置文件不存在: {candidates[0]}")
    for p in candidates:
        if not p.is_file():
            continue
        raw = _read_toml(p)
        for section, values in raw.items():
            if not isinstance(values, dict) or section == "source_path" or not hasattr(cfg, section):
                raise ValueError(f"配置文件 {p} 里有未知的段 [{section}]")
            try:
                _merge(getattr(cfg, section), values, section)
            except ValueError as exc:
                raise ValueError(f"{exc}（配置文件 {p}）") from None
        cfg.source_path = str(p)
        break
    try:
        cfg.validate()
    except ValueError as exc:
        where = f"（配置文件 {cfg.source_path}）" if cfg.source_path else ""
        raise ValueError(f"{exc}{where}") from None
    return cfg
