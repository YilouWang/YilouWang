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
    wisp_every: int = 0                  # 1 shop in N hides a champion slot under a Set 18 Wisp
    extra: dict[str, Any]                # [extra] tables (team_slots, wisp_every_shop, upgraded_slots, ...)
    def shop_schedule(self, wisp_every: int | None = None) -> tuple[int, ...]  # champion slots per shop, repeating
    def interest(self, gold: int) -> int
    def streak_gold(self, streak: int | None) -> int      # abs(streak)
    def income(self, gold: int, streak: int | None, won: bool | None = None) -> int
    def xp_to_next(self, level: int, xp_current: int | None) -> int | None
    def gold_to_reach(self, level: int, xp_current: int | None, target_level: int) -> int | None
    def is_pve(self, stage: StageRound) -> bool
    def is_carousel(self, stage: StageRound) -> bool
    def is_augment(self, stage: StageRound) -> bool
    def max_round_damage(self, stage: int | None) -> int   # stage damage + units, caps HP drops (misread check)
    def standard_level_at(self, sr: StageRound | None) -> int | None
    patch: str                           # e.g. "18.3b", printed by `doctor` / `data show`
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
def load_set_data(cfg: DataConfig, offline: bool = False, log=print, background: bool = False) -> SetData
```
Runtime source: CommunityDragon `…/cdragon/tft/{zh_cn,en_us}.json`, cached in
`~/.tft_advisor/`. Falls back to cache, then to the bundled real snapshot
(`data/bundled/snapshot_{locale}.json`), then to `data/bundled/sample_set.json`.
Downloads use gzip, conditional requests, a 15 s socket timeout and a 45 s
whole-file deadline; after a failure they are not retried for 6 hours
(`data update` forces one). `background=True` (used by `AdvisorApp`): when a
cache or the bundled snapshot can be used right away, a stale export is
refreshed in a daemon thread and applies on the next start.

### `tft_advisor/data/textio.py`
`read_user_text(path, what)` / `read_user_data(path, what, fmt=None)`: the
player's own comps / mechanics files, UTF-8 with or without BOM, JSON or
TOML; errors name the file in Chinese.

### `tft_advisor/data/comps.py`
```python
@dataclass
class CompDef: name: str; units: list[str]; carry: str | None; carry_items: list[str];
               traits: list[str]; style: str  # standard | fast8 | fast9 | reroll1 | reroll2 | reroll3
               tier: str; notes: str; early; item_holders; positions; stars; difficulty;
               reroll_units: list[str]  # units a reroll line slow rolls for (3-star targets)
               name_en: str             # name as written in the file; `name` is name_zh when the set data is Chinese
def load_comps(path: str | None, set_data: SetData, log=None) -> list[CompDef]
```
A missing, broken or empty custom comps file falls back to the bundled
library (with a warning); unresolved API ids get readable names.

### `tft_advisor/data/augments.py`
`AugmentData.load(set_number)` (bundled `augments_set18.json`), `.lookup(name)`
(zh / en, fuzzy), `effect_kinds(augment) -> list[str]` (经济 / 装备 / 搜牌 / 战力 ...,
from the category or the effect text), and
`pick_augment(choices, data, stage, hp, comp_traits, level=None, round_no=None, augment_rounds=None)
-> (choice, reason) | None`. The rules pass the state's level and round and
`Mechanics.augment_rounds`, so effects whose trigger already passed (a level
reached, the last augment round) or pay out too late at low HP score lower.

### `tft_advisor/engine/`
* `probability.py`: pool model + rolldown Markov chain.
  `compute_hit_odds(state, set_data, mech, taken: dict[str,int], targets=None, budgets=..., level=None, shop=None) -> list[HitOdds]`
  (`level`: odds at the level after a planned level up; `shop`: `ShopModel` with the Wisp / Inferno
  shop shape, default from `shop_model(state, set_data, mech)`),
  `rolldown_probability(..., shop=None)`, `rolldown_curve(mech, level, cost, need, remaining_unit,
  remaining_cost_total, budgets, unit_price=None, shop=None) -> {budget: p}` (one iterative table for
  every budget, no recursion limit), `p_shop_shows(...)`, `p_unit_per_slot(...)`.
  `HitOdds.p_at(budget)` reads the exact budget when computed (the analyzer adds the plan's roll
  budget), otherwise the largest computed budget below it.
* `economy.py`: `plan_economy(state, mech, style="standard", key_star=None, carry_cost=None) -> EconPlan`
  (`key_star`: star level of the reroll target / carry; a reroll line levels normally once it is 3-star;
  `carry_cost`: cost of the line's carry, a 5-cost carry turns a fast 8 into a fast 9).
* `items.py`: `plan_items(state, set_data, comp=None, hp_bucket="unknown", profile_hints=None) -> list[ItemSuggestion]`
  (`profile_hints`: unit -> "ad" / "ap" / "tank" from the comp library's item plans).
* `comps_engine.py`: `suggest_comps(state, set_data, comps, taken_by_player, top_n=3, hint="") -> list[CompSuggestion]`
  (`hint` matches comp names in both languages, traits and champion names).
* `tracker.py`: `GameTracker(set_data, mech).ingest(obs) -> GameState`, `.state`,
  `.taken_copies()`, `.taken_by_player()`, `.set_field(name, value)`, `.reset()`,
  `.self_name_confirmed` (the local name was flagged on 2+ frames: only then is it
  sent to vision as a hint). `field_age["round"]` is when the current round started.
  Suspicious readings (a lower stage, a gold jump or a lost digit, a lower level,
  an HP drop larger than the fights since can deal) need a second consistent
  reading; a level rises at most once per round and only as far as the gold spent
  since can buy. Combat frames never change a known board, and a double stage-1
  reading resets the game only when nothing on screen shows the old game going on.
* `analyzer.py`: `Analyzer(set_data, mech, comps).analyze(state, taken) -> Analysis`;
  `Analyzer.shop_is_stale(state)` (carousel frame, or a shop read before the round
  started: no shop picks; the dashboard labels it 上回合的商店).
  Shop picks are paid from the roll budget (`Analysis.shop_picks_cost`), and the
  odds count the copies they buy. Team slots neither the board nor the bench can
  fill get fill picks first (comp units, then trait fit, then cost; inside the hard
  gold limit, not the interest limit); the rules headline then says 补满空位. `CompSuggestion` carries the library's `style`,
  `tier`, `positions` (unit -> [row, col]) and `item_holders`.
* The tracker takes an opponent's level only from `ScreenObservation.viewed_player_level`
  (or the note "对手等级 N"), never from the local HUD `level`.

### `tft_advisor/vision/`
* `base.py`: `Perceiver` protocol `perceive(image, purpose="auto", hint=None) -> Observation`,
  `PerceptionHint`, `PerceptionError`.
* `claude_vision.py`: `ClaudeVisionPerceiver(llm, cfg: AnthropicConfig, set_data)`. The structured
  output schema is `models.ScreenObservationWire` (every field required, no unions, an
  `unreadable` list instead of nulls) because `ScreenObservation` is over the API's schema
  complexity limits (24 optional / 16 union parameters); `.to_screen()` converts back.
* `ocr.py`: optional RapidOCR fast path for stage / gold / level / shop names
  (`rapidocr_onnxruntime` on Python < 3.13, `rapidocr` 3.x + `onnxruntime` from 3.13).
  Item icons vision cannot name are sent as `"?"` (`base.UNKNOWN_ITEM`): the slot
  counts as used, but the item is never guessed.
* `mock.py`: `MockPerceiver` replays `ScreenObservation` JSON files.
* `liveclient.py`: optional poller for `https://127.0.0.1:2999/liveclientdata/*`.

### `tft_advisor/llm.py` (shared Claude access, already implemented)
`LLM(cfg: AnthropicConfig, client=None, limiter=None, clock=time.monotonic)` with `.parse(model=, effort=, system=, content=, schema=, purpose=)`
(structured output, adaptive thinking, fallbacks, cached system prompt) and `.text(...)`.
The system prompt uses the default 5 minute cache while calls come less than 5
minutes apart; once a prompt sat idle past that, the next write asks for the 1 hour
TTL (priced at 2x input in the cost estimate) and keeps it while the entry lives.
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
  The app sets `unavailable_keys` / `unavailable_actions` from the registered hotkeys,
  so a key that does not work for an action is never named (the dashboard button is).
  Level plans use `economy._wanted_level(..., key_star)` and a save for a catch-up
  level quotes `economy.catch_up_keep`, so the texts match when the econ levels.
* `strategist.py`: `ClaudeStrategist(llm, cfg, set_data, *, clock=time.time, extra_reference="", hotkeys=None, scout_enabled=True, augments=None)`
  `.advise(state, analysis, rules_advice, question=None) -> Advice` and `.ask(question, state, analysis) -> str`.
  `AdvisorApp` passes `[hotkeys]` per key (`app.strategist_hotkeys`: a key that does not
  work for its action, not registered or repeated in the config, is sent as "off"; the
  whole field is "off" when neither the analyze nor the scout key works) and
  `[advisor] scout_prompts` (False: scout requests are dropped).
* `scouting.py`: `ScoutPlanner(max_per_stage).plan(state, analysis) -> list[ScoutRequest]`.

### `tft_advisor/ui/`
* `server.py`: `DashboardServer(bus, cfg: UIConfig)`: static page, `/api/snapshot`,
  `/api/events` (Server-Sent Events), `POST /api/command`.
* `overlay.py`: optional tkinter always-on-top overlay. On Windows 10 2004+ it is
  excluded from screen capture (`SetWindowDisplayAffinity`, WDA_EXCLUDEFROMCAPTURE),
  so it never reaches the frames sent to Claude (nor OBS / Discord). When that is not
  possible it starts in the top-left corner, above the board, and logs a warning.
* `voice.py`: optional TTS (pyttsx3).

### `tft_advisor/app.py` / `cli.py` / `config.py` / `review.py`
`AdvisorApp` wires everything, owns threads (capture loop, analysis worker,
strategy worker, hotkeys, dashboard). `GameLogger` writes one
`~/.tft_advisor/logs/game-<start>.jsonl` per game (rotated when the tracker
sees a new game or on 新对局; the newest `[data] keep_game_logs` are kept).
`perceiver` / `fast_perceiver` default to `app.AUTO` (detect Claude vision /
local OCR); `None` means none, so demo, `replay --mock` and tests never pick
up an installed OCR by accident.
`cli.py` exposes `run`, `demo`, `replay`, `odds`, `data`, `doctor`,
`calibrate`, `review`, `init`; `--config` is accepted before or after the
command. Config / input errors print one Chinese line (`错误: ...`, exit 2);
`TFT_ADVISOR_DEBUG=1` shows the traceback. stdout / stderr are UTF-8 on a
terminal; when redirected on Windows they use the console code page (when it
can hold Chinese) so PowerShell pipes do not garble the text. `doctor` shows
why an optional module failed to import (a "DLL load failed" gets the VC++
runtime link) and checks OCR as `rapidocr_onnxruntime` or `rapidocr`. `run`
prints the phone URL (`手机访问: ...`) in LAN mode.

`config.py`: lookup order `--config`, `$TFT_ADVISOR_CONFIG` (must exist when
set), `~/.tft_advisor/config.toml`; the current directory is never searched.
`load_config` accepts UTF-8 with or without BOM, checks types (bool / number /
string), ranges (`Config.validate()`), hotkey names, `https://` for
`cdragon_base` and rejects network paths for `cache_dir` / `screenshot_dir`.
Relative `comps_file` / `mechanics_file` / `cache_dir` / `screenshot_dir`
values are resolved against the config file's directory. A TOML error caused
by a Windows path in double quotes gets a Chinese hint (single quotes or `/`),
and path values holding control characters (`"D:\tft\new.json"` parsed as
TAB / LF) are rejected with the corrected single-quoted line.

`review.py`: `latest_log`, `load_records` (skips broken / non-object lines),
`summarize_log` (last game in the file only, one row per round with HP,
gold, level, streak, board with items, target comp, econ style and call,
advice; advice-only `purpose="strategy"` records update that round's
advice; `level_timing` with `standard_round` / `rounds_late` (a lower bound
backed by logged rounds, None when a log gap hides it);
`interest_short_on_save_rounds` counts only stage 3+ rounds where the
assistant said save for interest (PvE rounds, low / critical HP and rounds
whose plan buys shop units, `shop_picks_cost > 0`, are left out); `augments` with the round first seen;
`final_item_bench`), `format_summary`, `llm_review` (`REVIEW_SYSTEM` explains
every field and asks to skip categories without data; em dashes removed).

## Threading model

* capture thread: grabs a frame every `poll_interval_s`, feeds `RoundWatcher`.
  Frames that are not the game window in the foreground (window not found,
  minimized, covered; `last_source != "window"` or
  `capturer.game_foreground()` is False) are skipped and pause auto mode, so
  the desktop or another app is never sent to Claude. Auto jobs are "gated":
  the worker grabs their frame and checks it again.
* worker thread: `JobSlot` queue runs
  perceive → ingest → analyze → rules → publish. Player-triggered jobs
  (manual / scout / shop, hotkeys and dashboard buttons) carry the frame
  grabbed at trigger time. Scout jobs wait in their own FIFO (up to 7, run
  first) and are never replaced. Otherwise one main job: a new job replaces
  it unless the pending one is more important; a job that loses but would see
  something newer (a shop read grabbed after the pending analysis' frame, a
  round change after an F6 frame, a shop read displaced by a correction) is
  kept as one follow-up. A job that must be dropped is reported as
  `命令 <按钮> 失败：...`, so the dashboard toast says 未执行. Each job
  remembers the 新对局 generation it was created in; a result from before a
  reset is dropped. The check is made before ingest and again, under
  `_advice_lock`, right before publishing and logging (new_game() and a
  tracker-detected new game take that lock too, lock order `_state_lock` then
  `_advice_lock`), so nothing of the old game is shown or logged after a reset.
* strategy thread (when a strategist exists and `start()` ran): single slot,
  runs the Claude strategist after auto / manual / reanalyze jobs and
  publishes its advice only if the game, round and request are still current.
  If the shop changed while Claude was thinking, its buy actions are fitted to
  the current shop (`_merge_buy_actions`, `source="rules+llm"`): a Claude buy
  whose units are all still in the same slots stays; the others are replaced
  by the rules' buys, minus buys of units Claude saw and did not buy; advice
  without a buy action gets none; at most `MAX_ACTIONS` actions. The publish
  and the log line happen under `_advice_lock`.
  Without `start()` (replay, tests) the strategist runs inline in `run_job`.
* Shop reads and scouts never ask Claude, and they do not replace Claude's
  advice for the same game and round: it stays up (buy actions redone for a
  changed shop); the shop card and odds update from the new analysis.
* Automatic shop reads run on every screen except carousel / loading /
  post-game (an augment-screen auto frame must not block the round's rerolls).
* `status.last_error` / `last_error_ts`: the latest error per source
  (perception, strategy, job); a later success of that source clears it
  (the dashboard shows its age).
* `status.hotkeys_live`: action -> the global hotkey really works (registered and
  bound to that action); the dashboard names only those keys.
  `status.asking` = `{question, ts}` while Claude answers a question (every open
  page shows 思考中), `status.comp_hint_matched` is False when the target comp
  matches nothing. 新对局 and a detected new game publish `answer = None`.
* Advice goes to the dashboard log as 「规则建议：…」 / 「Claude 建议：…」
  (headline only); the console line also lists the first actions.
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
