# Architecture

Personal, passive TFT advisor. It never sends input to the game and never reads
game memory: it only takes screenshots (like OBS would), reads the Riot local
Live Client Data API when available, and shows advice in a browser dashboard,
an optional overlay, and optional voice.

```
 ┌──────────────┐  frames   ┌──────────────┐ Observation ┌──────────────┐ GameState ┌──────────────┐
 │ capture      │ ────────> │ vision       │ ──────────> │ engine.      │ ────────> │ engine.      │
 │ screen.py    │           │ claude/ocr/  │             │ tracker      │           │ analyzer     │
 │ change.py    │           │ mock         │             └──────────────┘           └──────┬───────┘
 │ hotkeys.py   │           └──────────────┘                                               │ Analysis
 └──────┬───────┘                                                                          v
        │ round/shop changed                                                      ┌──────────────────┐
        └──────────────────────────> app.AdvisorApp (orchestrator) <──────────────│ advisor.rules    │
                                              │  ^                                │ advisor.strategist│ (Claude)
                                     publish  v  │ commands                       │ advisor.scouting │
                                         bus.EventBus                             └──────────────────┘
                                      ┌───────┼────────┐
                                      v       v        v
                               ui.server   ui.overlay  ui.voice
                              (dashboard)  (tkinter)   (TTS)
```

## Module contracts

All shared types live in `tft_advisor/models.py` (pydantic v2). Every module
below exposes exactly the public API listed; internal helpers are free.

### `tft_advisor/data/mechanics.py`
```python
@dataclass(frozen=True)
class Mechanics:
    shop_odds: dict[int, tuple[float, float, float, float, float]]  # level -> P(cost 1..5), sums to 1
    pool_size: dict[int, int]            # cost -> copies of EACH champion of that cost
    xp_to_level: dict[int, int]          # level L -> XP needed to go from L to L+1
    xp_per_buy: int = 4
    buy_xp_cost: int = 4
    roll_cost: int = 2
    interest_step: int = 10
    interest_cap: int = 5
    streak_bonus: tuple[tuple[int, int], ...]  # (min streak length, gold), ascending
    base_income: int = 5                 # per round from stage 2
    pvp_win_gold: int = 1
    passive_xp_per_round: int = 2
    bench_size: int = 9
    shop_slots: int = 5
    max_level: int = 10
    augment_rounds: tuple[str, ...]      # e.g. ("2-1", "3-2", "4-2")
    carousel_round: int = 4              # x-4 is the carousel from stage 2
    stage_damage: dict[int, int]         # stage -> base damage on loss
    standard_levels: dict[str, int]      # round -> level a standard line reaches ("2-1": 4, ...)
    def interest(self, gold: int) -> int
    def streak_gold(self, streak: int | None) -> int      # abs(streak)
    def income(self, gold: int, streak: int | None, won: bool | None = None) -> int
    def xp_to_next(self, level: int, xp_current: int | None) -> int | None
    def gold_to_reach(self, level: int, xp_current: int | None, target_level: int) -> int | None
    def is_pve(self, stage: StageRound) -> bool
    def is_carousel(self, stage: StageRound) -> bool
    def is_augment(self, stage: StageRound) -> bool
def load_mechanics(override_path: str | None = None) -> Mechanics
```
Defaults live in `data/bundled/mechanics.toml`; a user TOML overrides any key.

### `tft_advisor/data/setdata.py`
```python
@dataclass
class Champion: api_name: str; name: str; name_en: str; cost: int; traits: list[str]; icon: str | None
@dataclass
class Trait: api_name: str; name: str; name_en: str; breakpoints: list[int]; desc: str
@dataclass
class Item: api_name: str; name: str; name_en: str; composition: list[str]; kind: str  # component|completed|emblem|special
class SetData:
    set_number: int; set_name: str; source: str
    champions: dict[str, Champion]; traits: dict[str, Trait]; items: dict[str, Item]
    def resolve_champion(self, name: str) -> Champion | None   # zh/en/api, normalized + fuzzy
    def resolve_item(self, name: str) -> Item | None
    def resolve_trait(self, name: str) -> Trait | None
    def champions_by_cost(self, cost: int) -> list[Champion]
    def components(self) -> list[Item]
    def recipe(self, a: str, b: str) -> Item | None             # two component names/api names
    def summary_text(self) -> str                              # compact text for LLM prompts
    @classmethod
    def from_cdragon(cls, data: dict, data_en: dict | None = None, set_number: int = 0) -> "SetData"
def load_set_data(cfg: DataConfig, offline: bool = False, log=print) -> SetData
```
Runtime source: CommunityDragon `…/cdragon/tft/{zh_cn,en_us}.json`, cached in
`~/.tft_advisor/`. Falls back to cache, then to `data/bundled/sample_set.json`.

### `tft_advisor/data/comps.py`
```python
@dataclass
class CompDef: name: str; units: list[str]; carry: str | None; carry_items: list[str];
               traits: list[str]; style: str  # "fast8" | "reroll" | "standard"
               tier: str; notes: str
def load_comps(path: str | None, set_data: SetData) -> list[CompDef]
```

### `tft_advisor/engine/`
* `probability.py`: pool model + rolldown Markov chain.
  `compute_hit_odds(state, set_data, mech, taken: dict[str,int], targets=None) -> list[HitOdds]`,
  `rolldown_probability(...)`, `p_unit_per_slot(...)`.
* `economy.py`: `plan_economy(state, mech, style="standard") -> EconPlan`.
* `items.py`: `plan_items(state, set_data, comp=None) -> list[ItemSuggestion]`.
* `comps_engine.py`: `suggest_comps(state, set_data, comps, taken_by_player) -> list[CompSuggestion]`.
* `tracker.py`: `GameTracker(set_data, mech).ingest(obs) -> GameState`, `.state`,
  `.taken_copies()`, `.taken_by_player()`, `.set_field(name, value)`, `.reset()`.
* `analyzer.py`: `Analyzer(set_data, mech, comps).analyze(state, taken) -> Analysis`.

### `tft_advisor/vision/`
* `base.py`: `Perceiver` protocol `perceive(image, purpose="auto", hint=None) -> Observation`,
  `PerceptionHint`, `PerceptionError`.
* `claude_vision.py`: `ClaudeVisionPerceiver(client, cfg: AnthropicConfig, set_data)`.
* `ocr.py`: optional RapidOCR fast path for stage / gold / level / shop names.
* `mock.py`: `MockPerceiver` replays `ScreenObservation` JSON files.
* `liveclient.py`: optional poller for `https://127.0.0.1:2999/liveclientdata/*`.

### `tft_advisor/llm.py` (shared Claude access, already implemented)
`LLM(cfg: AnthropicConfig, client=None)` with `.parse(model=, effort=, system=, content=, schema=, purpose=)`
(structured output, adaptive thinking, fallbacks, cached system prompt) and `.text(...)`.
Errors: `LLMError` (base), `LLMUnavailable`, `LLMRefusal`, `LLMRateLimited`.
Helpers `image_block(png_bytes)`, `text_block(text)`, `has_credentials()`.
Tests use `tests/fakeapi.py` (`FakeAnthropic`), a real local HTTP server that
records requests and returns scripted replies, so the real SDK path runs offline.

### `tft_advisor/capture/`
* `regions.py`: named HUD regions as fractions of a 16:9 frame, `crop(img, name)`,
  `region_box(size, name) -> (x0, y0, x1, y1)` in pixels. Region names (fixed):
  `stage`, `gold`, `level`, `xp`, `streak`, `shop`, `bench`, `board`, `players`,
  `traits`, `items`, `augments`, `hud_bottom` (whole bottom HUD strip incl. shop,
  gold, level), `top_banner` (name of the board owner while scouting).
* `screen.py`: `ScreenCapturer(cfg)` (mss + Win32 window rect), `FileCapturer(paths)`.
* `change.py`: `RoundWatcher` emits `round_changed` / `shop_changed` from cheap diffs.
* `hotkeys.py`: global hotkeys via Win32 `RegisterHotKey` (ctypes, no deps).

### `tft_advisor/advisor/`
* `rules.py`: `RulesAdvisor().advise(state, analysis) -> Advice` (Chinese, offline).
* `strategist.py`: `ClaudeStrategist(client, cfg, set_data).advise(state, analysis, rules_advice, question=None) -> Advice`
  and `.ask(question, state, analysis) -> str`.
* `scouting.py`: `ScoutPlanner(max_per_stage).plan(state, analysis) -> list[ScoutRequest]`.

### `tft_advisor/ui/`
* `server.py`: `DashboardServer(bus, cfg: UIConfig)`: static page, `/api/snapshot`,
  `/api/events` (Server-Sent Events), `POST /api/command`.
* `overlay.py`: optional tkinter always-on-top overlay.
* `voice.py`: optional TTS (pyttsx3).

### `tft_advisor/app.py` / `cli.py`
`AdvisorApp` wires everything, owns threads (capture loop, analysis worker,
hotkeys, dashboard). `cli.py` exposes `run`, `demo`, `replay`, `odds`, `data`,
`doctor`, `calibrate`.

## Threading model

* capture thread: grabs a frame every `poll_interval_s`, feeds `RoundWatcher`.
* worker thread: single-slot job queue (latest job wins) runs
  perceive → ingest → analyze → rules → (LLM strategist) → publish.
* dashboard: `ThreadingHTTPServer` in a daemon thread.
* hotkeys: Win32 message loop thread.
* main thread: tkinter overlay mainloop when enabled, otherwise waits on a stop event.

## Human in the loop

The advisor asks the player for information it cannot see (other boards).
Requests appear in the dashboard / overlay / voice ("请点开 XXX 的棋盘后按 F7").
The player presses the scout hotkey (or taps the dashboard button on a phone),
the frame is tagged `purpose="scout"` and stored as an `OpponentSnapshot`. The
pool model then subtracts those copies when computing odds.

## Conventions

* Python 3.11+, type hints, pydantic v2 models from `models.py` for anything crossing modules.
* Code comments / identifiers in English. Everything the player reads (advice,
  UI labels, requests, logs meant for the dashboard) is Simplified Chinese.
  Never use the em dash / 破折号 characters ("—", "——") in player-facing text;
  use commas, colons or parentheses instead.
* No module may send keyboard/mouse input to the game or read game memory.
* Optional dependencies (rapidocr, pyttsx3, tkinter, mss on CI) are imported
  lazily; the app must degrade gracefully when they are missing.
* Tests: `pytest` under `tests/`, fully offline. Claude calls go through
  `tests/fakeapi.py`. Synthetic images are generated with Pillow in tests.
