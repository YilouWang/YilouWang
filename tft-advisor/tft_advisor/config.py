"""Configuration: dataclasses with defaults, optionally overridden by a TOML file.

Lookup order for the config file: explicit ``--config`` path, ``$TFT_ADVISOR_CONFIG``,
``./tft_advisor.toml``, ``~/.tft_advisor/config.toml``. Missing file = defaults.
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
    ocr_crosscheck: bool = False  # also run OCR on every frame to double-check gold / stage (slower)
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


def _merge(section_obj: Any, values: dict[str, Any], section: str) -> None:
    known = {f.name: f for f in dataclasses.fields(section_obj)}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"Unknown config key [{section}].{key}")
        current = getattr(section_obj, key)
        if isinstance(current, bool) and not isinstance(value, bool):
            raise ValueError(f"[{section}].{key} must be true/false")
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"[{section}].{key} must be a number")
            value = type(current)(value)
        setattr(section_obj, key, value)


def default_config_paths() -> list[Path]:
    paths = []
    env = os.environ.get("TFT_ADVISOR_CONFIG")
    if env:
        paths.append(Path(env))
    paths.append(Path.cwd() / "tft_advisor.toml")
    paths.append(Path(os.path.expanduser("~/.tft_advisor/config.toml")))
    return paths


def load_config(path: Optional[str | Path] = None) -> Config:
    cfg = Config()
    candidates = [Path(path)] if path else default_config_paths()
    for p in candidates:
        if p.is_file():
            with open(p, "rb") as fh:
                raw = tomllib.load(fh)
            for section, values in raw.items():
                if not isinstance(values, dict) or not hasattr(cfg, section) or section == "source_path":
                    raise ValueError(f"Unknown config section [{section}] in {p}")
                _merge(getattr(cfg, section), values, section)
            cfg.source_path = str(p)
            break
        if path:
            raise FileNotFoundError(f"Config file not found: {p}")
    return cfg
