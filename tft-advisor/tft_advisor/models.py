"""Shared data contracts for every module.

Layers:
  * Perception (``ScreenObservation``): what one screenshot shows, raw names as
    displayed in the client (any language). Produced by ``tft_advisor.vision``.
  * State (``GameState``): the tracker's merged, normalized view of the game.
    Produced by ``tft_advisor.engine.tracker``.
  * Analysis (``Analysis``): deterministic math (econ, odds, items, comps).
    Produced by ``tft_advisor.engine``.
  * Advice (``Advice``): what we show / say to the player. Produced by the
    rules engine (offline) or the Claude strategist.

Board coordinates: ``row`` 0 is the front row (closest to the enemy), 3 is the
back row (closest to your bench). ``col`` 0 is the leftmost hex as seen on
screen, 6 the rightmost.
"""

from __future__ import annotations

import re
import time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Stage / round
# --------------------------------------------------------------------------

_STAGE_RE = re.compile(r"^\s*(\d{1,2})\s*[-–—:]\s*(\d{1,2})\s*$")


class StageRound(BaseModel):
    """A TFT round such as ``3-2`` (stage 3, round 2)."""

    stage: int
    round: int

    @classmethod
    def parse(cls, text: Optional[str]) -> Optional["StageRound"]:
        if not text:
            return None
        m = _STAGE_RE.match(text)
        if not m:
            return None
        stage, rnd = int(m.group(1)), int(m.group(2))
        if stage < 1 or rnd < 1 or stage > 9 or rnd > 9:
            return None
        return cls(stage=stage, round=rnd)

    @property
    def key(self) -> tuple[int, int]:
        return (self.stage, self.round)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.stage}-{self.round}"

    def __lt__(self, other: "StageRound") -> bool:
        return self.key < other.key

    def __le__(self, other: "StageRound") -> bool:
        return self.key <= other.key

    def __hash__(self) -> int:
        return hash(self.key)


# --------------------------------------------------------------------------
# Perception layer (one screenshot)
# --------------------------------------------------------------------------


class ScreenType(str, Enum):
    PLANNING = "planning"  # own or other board, planning phase (shop usable)
    COMBAT = "combat"  # fight in progress
    CAROUSEL = "carousel"  # shared draft
    AUGMENT_SELECT = "augment_select"  # augment choice overlay visible
    LOADING = "loading"
    POST_GAME = "post_game"
    OTHER = "other"  # client, menus, not TFT, unreadable


class UnitObs(BaseModel):
    """A champion seen on a board or bench."""

    name: str = Field(description="Champion name exactly as displayed / recognized")
    star: int = Field(default=1, description="Star level 1-3 (4 only for special cases)")
    items: list[str] = Field(default_factory=list, description="Item names held by the unit")
    row: Optional[int] = Field(default=None, description="Board row 0=front .. 3=back; null on bench")
    col: Optional[int] = Field(default=None, description="Board column 0=left .. 6=right; bench slot 0-8 when on bench")


class ShopSlot(BaseModel):
    name: Optional[str] = Field(default=None, description="Champion name, null when the slot is empty")
    cost: Optional[int] = Field(default=None, description="Gold cost 1-5 (or higher for special units)")


class PlayerObs(BaseModel):
    """One row of the player list on the right side of the HUD."""

    name: str
    hp: Optional[int] = None
    is_self: bool = False


class TraitObs(BaseModel):
    """A trait row from the trait tracker on the left side."""

    name: str
    count: int = Field(description="Number of unique units contributing")
    active: bool = Field(default=False, description="True when a breakpoint is reached")
    next_breakpoint: Optional[int] = None


class ScreenObservation(BaseModel):
    """Everything a perceiver could read from a single screenshot.

    Every field is optional: unreadable or not-visible means ``None`` (not 0,
    not empty list). An empty list means "visible and empty".
    """

    screen_type: ScreenType = ScreenType.OTHER
    viewing_own_board: Optional[bool] = Field(
        default=None, description="False when the camera shows another player's board (scouting)"
    )
    viewed_player_name: Optional[str] = Field(default=None, description="Owner of the board shown, if readable")
    stage: Optional[str] = Field(default=None, description="Round indicator like '3-2'")
    gold: Optional[int] = None
    level: Optional[int] = None
    xp_current: Optional[int] = None
    xp_needed: Optional[int] = None
    hp: Optional[int] = Field(default=None, description="HP of the player whose board is shown")
    streak: Optional[int] = Field(default=None, description="+N win streak, -N loss streak")
    shop: Optional[list[ShopSlot]] = Field(default=None, description="5 shop slots left to right")
    shop_locked: Optional[bool] = None
    board: Optional[list[UnitObs]] = None
    bench: Optional[list[UnitObs]] = None
    item_bench: Optional[list[str]] = Field(default=None, description="Unequipped items / components")
    players: Optional[list[PlayerObs]] = Field(default=None, description="Player list with HP, top to bottom")
    traits: Optional[list[TraitObs]] = None
    augment_choices: Optional[list[str]] = Field(default=None, description="Augments currently offered")
    augments: Optional[list[str]] = Field(default=None, description="Augments already owned, if visible")
    notes: list[str] = Field(default_factory=list, description="Anything uncertain or notable")


class Observation(BaseModel):
    """A ScreenObservation plus capture metadata."""

    screen: ScreenObservation
    captured_at: float = Field(default_factory=time.time)
    source: str = "unknown"  # "claude", "ocr", "liveclient", "mock", "manual", "merged"
    purpose: str = "auto"  # "auto", "manual", "scout", "shop"
    image_path: Optional[str] = None
    latency_s: Optional[float] = None


# --------------------------------------------------------------------------
# Normalized game state (tracker output)
# --------------------------------------------------------------------------


class Unit(BaseModel):
    """A champion resolved against set data."""

    api_name: str  # canonical id from set data, or "?"+raw name if unresolved
    name: str  # display name
    cost: Optional[int] = None
    star: int = 1
    items: list[str] = Field(default_factory=list)  # display names of items
    row: Optional[int] = None
    col: Optional[int] = None
    traits: list[str] = Field(default_factory=list)

    @property
    def copies(self) -> int:
        """Number of 1-star copies this unit represents (1, 3 or 9)."""
        return 3 ** (max(1, self.star) - 1)


class OpponentSnapshot(BaseModel):
    player: str
    stage: Optional[str] = None
    hp: Optional[int] = None
    level: Optional[int] = None
    board: list[Unit] = Field(default_factory=list)
    bench: list[Unit] = Field(default_factory=list)
    traits: list[TraitObs] = Field(default_factory=list)
    items: list[str] = Field(default_factory=list)
    captured_at: float = Field(default_factory=time.time)


class HistoryPoint(BaseModel):
    stage: str
    hp: Optional[int] = None
    gold: Optional[int] = None
    level: Optional[int] = None
    captured_at: float = Field(default_factory=time.time)


class GameState(BaseModel):
    """Tracker's best estimate of the current game."""

    game_id: str = ""
    stage: Optional[StageRound] = None
    screen_type: ScreenType = ScreenType.OTHER
    gold: Optional[int] = None
    level: Optional[int] = None
    xp_current: Optional[int] = None
    xp_needed: Optional[int] = None
    hp: Optional[int] = None
    streak: Optional[int] = None
    shop: list[ShopSlot] = Field(default_factory=list)
    shop_units: list[Optional[Unit]] = Field(default_factory=list)  # resolved shop, None for empty slot
    board: list[Unit] = Field(default_factory=list)
    bench: list[Unit] = Field(default_factory=list)
    item_bench: list[str] = Field(default_factory=list)
    traits: list[TraitObs] = Field(default_factory=list)
    augments: list[str] = Field(default_factory=list)
    augment_choices: list[str] = Field(default_factory=list)
    players: list[PlayerObs] = Field(default_factory=list)
    self_name: Optional[str] = None
    opponents: dict[str, OpponentSnapshot] = Field(default_factory=dict)
    history: list[HistoryPoint] = Field(default_factory=list)
    last_update: float = 0.0
    field_age: dict[str, float] = Field(
        default_factory=dict, description="Timestamp of last confirmed update per field name"
    )

    def all_units(self) -> list[Unit]:
        return [*self.board, *self.bench]

    def alive_players(self) -> list[PlayerObs]:
        return [p for p in self.players if p.hp is None or p.hp > 0]


# --------------------------------------------------------------------------
# Analysis (deterministic engine output)
# --------------------------------------------------------------------------


class EconAction(str, Enum):
    SAVE = "save"  # hold gold, build interest
    LEVEL = "level"  # buy XP now
    ROLL = "roll"  # reroll shop
    LEVEL_AND_ROLL = "level_and_roll"
    SLOW_ROLL = "slow_roll"  # roll only gold above the interest threshold
    ALL_IN = "all_in"  # spend everything, HP critical
    HOLD = "hold"  # nothing special this round


class EconPlan(BaseModel):
    gold: int = 0
    interest: int = 0
    streak_gold: int = 0
    base_income: int = 0
    expected_income: int = 0  # next round, excluding PvP win gold
    xp_to_next: Optional[int] = None
    gold_to_next_level: Optional[int] = None  # gold needed to level via XP buys
    recommendation: EconAction = EconAction.HOLD
    roll_budget: int = 0  # gold that can be spent on rolls now under the plan
    target_level: Optional[int] = None
    reason: str = ""


class HitOdds(BaseModel):
    unit: str  # display name
    api_name: str
    cost: int
    owned_copies: int  # 1-star equivalents we hold
    goal_copies: int  # 1-star equivalents needed for the goal star level
    goal_star: int
    seen_elsewhere: int  # copies known to be held by other players (scouted)
    remaining_in_pool: int
    level: int
    p_per_slot: float  # probability a single shop slot shows this unit
    p_in_shop: float  # probability one fresh shop shows at least one copy
    expected_gold_to_goal: Optional[float] = None
    p_goal_by_gold: dict[int, float] = Field(default_factory=dict)  # gold budget -> P(reach goal)


class ItemSuggestion(BaseModel):
    item: str  # completed item display name
    components: list[str]
    holder: Optional[str] = None  # suggested unit
    priority: int = 2  # 1 = slam now
    reason: str = ""


class CompSuggestion(BaseModel):
    name: str
    score: float  # 0..1 fit
    core_units: list[str] = Field(default_factory=list)
    have_units: list[str] = Field(default_factory=list)
    missing_units: list[str] = Field(default_factory=list)
    carry: Optional[str] = None
    carry_items: list[str] = Field(default_factory=list)
    contested_by: list[str] = Field(default_factory=list)  # player names playing similar units
    reason: str = ""


class ScoutRequest(BaseModel):
    """Something we want the human to do (e.g. switch camera to a player)."""

    id: str
    text: str  # Chinese instruction shown/spoken to the player
    target_player: Optional[str] = None
    reason: str = ""
    created_at: float = Field(default_factory=time.time)
    stage: Optional[str] = None


class Analysis(BaseModel):
    stage: Optional[str] = None
    econ: EconPlan = Field(default_factory=EconPlan)
    odds: list[HitOdds] = Field(default_factory=list)
    items: list[ItemSuggestion] = Field(default_factory=list)
    comps: list[CompSuggestion] = Field(default_factory=list)
    shop_picks: list[str] = Field(default_factory=list)  # shop units worth buying, in priority order
    sell_candidates: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    scout_requests: list[ScoutRequest] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)


# --------------------------------------------------------------------------
# Advice (what the player sees / hears)
# --------------------------------------------------------------------------


class ActionType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    LEVEL = "level"
    ROLL = "roll"
    SAVE = "save"
    ITEM = "item"
    POSITION = "position"
    PIVOT = "pivot"
    SCOUT = "scout"
    AUGMENT = "augment"
    CAROUSEL = "carousel"
    OTHER = "other"


class AdviceAction(BaseModel):
    type: ActionType
    text: str = Field(description="Short imperative instruction in Chinese, <= 40 chars")
    priority: int = Field(default=2, description="1 = do it now, 2 = this round, 3 = nice to have")


class Advice(BaseModel):
    headline: str = Field(description="One-line summary in Chinese, <= 30 chars")
    actions: list[AdviceAction] = Field(default_factory=list)
    plan: str = Field(default="", description="Plan for the next 2-3 rounds, Chinese, <= 120 chars")
    comp: Optional[str] = Field(default=None, description="Target comp direction")
    items: Optional[str] = Field(default=None, description="Item plan")
    positioning: Optional[str] = Field(default=None, description="Positioning tip")
    augment: Optional[str] = Field(default=None, description="Augment pick advice when choices are visible")
    scout_request: Optional[str] = Field(default=None, description="Ask the human to show another board")
    confidence: float = Field(default=0.5, description="0..1")
    source: str = "rules"  # "rules" | "llm" | "rules+llm"
    stage: Optional[str] = None
    created_at: float = Field(default_factory=time.time)


class StrategistAdvice(BaseModel):
    """Schema the Claude strategist must return (subset of Advice, no metadata)."""

    headline: str = Field(description="One-line summary in Chinese, <= 30 chars")
    actions: list[AdviceAction] = Field(description="1-6 concrete actions ordered by priority")
    plan: str = Field(description="Plan for the next 2-3 rounds, Chinese, <= 120 chars")
    comp: Optional[str] = Field(default=None, description="Target comp direction, Chinese")
    items: Optional[str] = Field(default=None, description="Item plan, Chinese")
    positioning: Optional[str] = Field(default=None, description="Positioning tip, Chinese")
    augment: Optional[str] = Field(default=None, description="Augment pick advice if choices visible")
    scout_request: Optional[str] = Field(
        default=None, description="If more info is needed, ask the player to open a specific board"
    )
    confidence: float = Field(description="0..1 confidence in this advice")
