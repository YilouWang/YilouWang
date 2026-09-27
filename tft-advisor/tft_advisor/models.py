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
from typing import Literal, Optional

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
    viewed_player_level: Optional[int] = Field(
        default=None,
        description="Level on the plate above the viewed player's arena while scouting (``level`` is always the local HUD)",
    )
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


# --------------------------------------------------------------------------
# Wire format for Claude vision (structured output)
# --------------------------------------------------------------------------
#
# ScreenObservation has ~30 optional fields and ~24 nullable unions. Claude's
# structured outputs cap a request at 24 optional parameters and 16 union
# typed parameters (platform docs, "Schema complexity limits"), so it cannot
# be sent as ``output_config.format``. The wire models below carry the same
# information with every field required and no unions: unknown scalars use
# sentinels and ``unreadable`` lists the fields that were not visible, which
# keeps the "null vs empty list" distinction. ``to_screen()`` converts back.

WireField = Literal[
    "viewing_own_board",
    "viewed_player_name",
    "viewed_player_level",
    "stage",
    "gold",
    "level",
    "xp_current",
    "xp_needed",
    "hp",
    "streak",
    "shop",
    "shop_locked",
    "board",
    "bench",
    "item_bench",
    "players",
    "traits",
    "augment_choices",
    "augments",
]
WIRE_FIELDS: tuple[str, ...] = WireField.__args__  # type: ignore[attr-defined]


class UnitObsWire(BaseModel):
    name: str = Field(description="Champion name exactly as displayed / recognized")
    star: int = Field(description="Star level 1-3 (4 only for special cases)")
    items: list[str] = Field(
        description="Item names held by the unit; \"?\" for each icon that cannot be named; [] when the unit holds no item"
    )
    row: int = Field(description="Board row 0=front .. 3=back; -1 on the bench")
    col: int = Field(description="Board column 0=left .. 6=right; bench slot 0-8 on the bench; -1 unknown")

    def to_obs(self) -> UnitObs:
        return UnitObs(
            name=self.name,
            star=self.star,
            items=list(self.items),
            row=self.row if self.row >= 0 else None,
            col=self.col if self.col >= 0 else None,
        )


def _is_wisp_label(name: str) -> bool:
    return name.strip().lower().startswith(("wisp", "精灵", "灵火"))


class ShopSlotWire(BaseModel):
    name: str = Field(description="Champion name ('Wisp: <name>' for a Wisp); empty string for an empty slot")
    cost: int = Field(description="Gold price; -1 when unreadable; 0 for an empty slot or a free (0 gold) Wisp")

    def to_obs(self) -> ShopSlot:
        name = self.name.strip() or None
        if name is None:
            return ShopSlot(name=None, cost=None)
        if self.cost < 0 or (self.cost == 0 and not _is_wisp_label(name)):
            return ShopSlot(name=name, cost=None)  # unknown price (a champion never costs 0)
        return ShopSlot(name=name, cost=self.cost)


class PlayerObsWire(BaseModel):
    name: str
    hp: int = Field(description="HP; -1 when unreadable")
    is_self: bool

    def to_obs(self) -> PlayerObs:
        return PlayerObs(name=self.name, hp=self.hp if self.hp >= 0 else None, is_self=self.is_self)


class TraitObsWire(BaseModel):
    name: str
    count: int = Field(description="Number of unique units contributing")
    active: bool = Field(description="True when a breakpoint is reached")
    next_breakpoint: int = Field(description="Next breakpoint; 0 when none or unreadable")

    def to_obs(self) -> TraitObs:
        return TraitObs(
            name=self.name,
            count=self.count,
            active=self.active,
            next_breakpoint=self.next_breakpoint if self.next_breakpoint > 0 else None,
        )


# Structured-output schema for Claude vision (see the comment above). Every
# field is required. A field listed in ``unreadable`` becomes ``None`` in the
# ScreenObservation whatever placeholder value it carries; an empty list NOT
# listed there means "visible and empty". The class docstring and the field
# descriptions are sent to the model: keep them model-facing and in line with
# rule 2 of the vision system prompt.
class ScreenObservationWire(BaseModel):
    """What one TFT screenshot shows. Fill every field. List every top-level field that is not visible or not readable in "unreadable" (its value is then ignored). An empty list that is not listed in "unreadable" means "visible and empty"."""

    screen_type: ScreenType
    unreadable: list[WireField] = Field(
        description="Fields that are not visible or not readable on this screenshot (they become null)"
    )
    viewing_own_board: bool = Field(description="False when the camera shows another player's board (scouting)")
    viewed_player_name: str = Field(description="Owner of the board shown; empty string if unreadable")
    viewed_player_level: int = Field(
        description="Level on the plate above the viewed player's arena when it is another player's; "
        "0 when not shown (the top level field is always the local player's HUD level)"
    )
    stage: str = Field(description="Round indicator like '3-2'; empty string if unreadable")
    gold: int = Field(description="Local player's gold (bottom HUD); -1 when unreadable (also list it in unreadable)")
    level: int = Field(description="Local player's level (bottom HUD); 0 when unreadable (also list it in unreadable)")
    xp_current: int = Field(description="XP bar value x of 'x/y'; -1 when unreadable (also list it in unreadable)")
    xp_needed: int = Field(description="XP bar value y of 'x/y'; 0 when unreadable (also list it in unreadable)")
    hp: int = Field(description="HP of the player whose board is shown; -1 when unreadable (also list it in unreadable)")
    streak: int = Field(
        description="+N win streak, -N loss streak, 0 for none; 0 and listed in unreadable when the gold area is not visible"
    )
    shop: list[ShopSlotWire] = Field(description="5 shop slots left to right")
    shop_locked: bool
    board: list[UnitObsWire]
    bench: list[UnitObsWire]
    item_bench: list[str] = Field(description="Unequipped items / components")
    players: list[PlayerObsWire] = Field(description="Player list with HP, top to bottom")
    traits: list[TraitObsWire]
    augment_choices: list[str] = Field(description="Augments currently offered")
    augments: list[str] = Field(description="Augments already owned, if visible")
    notes: list[str] = Field(description="Anything uncertain or notable")

    @classmethod
    def from_screen(cls, screen: "ScreenObservation | dict") -> "ScreenObservationWire":
        """Inverse of ``to_screen`` (``None`` fields go to ``unreadable``). Used by
        tests and tools that script replies in the ScreenObservation shape."""
        s = screen if isinstance(screen, ScreenObservation) else ScreenObservation.model_validate(screen)
        unreadable = [f for f in WIRE_FIELDS if getattr(s, f) is None]

        def unit(u: UnitObs) -> dict:
            return {
                "name": u.name,
                "star": u.star,
                "items": list(u.items),
                "row": -1 if u.row is None else u.row,
                "col": -1 if u.col is None else u.col,
            }

        return cls(
            screen_type=s.screen_type,
            unreadable=unreadable,  # type: ignore[arg-type]
            viewing_own_board=bool(s.viewing_own_board),
            viewed_player_name=s.viewed_player_name or "",
            viewed_player_level=s.viewed_player_level or 0,
            stage=s.stage or "",
            gold=-1 if s.gold is None else s.gold,
            level=0 if s.level is None else s.level,
            xp_current=-1 if s.xp_current is None else s.xp_current,
            xp_needed=0 if s.xp_needed is None else s.xp_needed,
            hp=-1 if s.hp is None else s.hp,
            streak=s.streak or 0,
            shop=[
                {"name": x.name or "", "cost": 0 if not x.name else (-1 if x.cost is None else x.cost)}
                for x in s.shop or []
            ],
            shop_locked=bool(s.shop_locked),
            board=[unit(u) for u in s.board or []],
            bench=[unit(u) for u in s.bench or []],
            item_bench=list(s.item_bench or []),
            players=[{"name": p.name, "hp": -1 if p.hp is None else p.hp, "is_self": p.is_self} for p in s.players or []],
            traits=[
                {"name": t.name, "count": t.count, "active": t.active, "next_breakpoint": t.next_breakpoint or 0}
                for t in s.traits or []
            ],
            augment_choices=list(s.augment_choices or []),
            augments=list(s.augments or []),
            notes=list(s.notes),
        )

    def to_screen(self) -> ScreenObservation:
        missing = set(self.unreadable)

        def val(name: str, value):
            return None if name in missing else value

        stage = self.stage.strip()
        name = self.viewed_player_name.strip()
        return ScreenObservation(
            screen_type=self.screen_type,
            viewing_own_board=val("viewing_own_board", self.viewing_own_board),
            viewed_player_name=val("viewed_player_name", name or None),
            viewed_player_level=val(
                "viewed_player_level", self.viewed_player_level if self.viewed_player_level > 0 else None
            ),
            stage=val("stage", stage or None),
            gold=val("gold", self.gold if self.gold >= 0 else None),
            level=val("level", self.level if self.level > 0 else None),
            xp_current=val("xp_current", self.xp_current if self.xp_current >= 0 else None),
            xp_needed=val("xp_needed", self.xp_needed if self.xp_needed > 0 else None),
            hp=val("hp", self.hp if self.hp >= 0 else None),
            streak=val("streak", self.streak),
            shop=val("shop", [s.to_obs() for s in self.shop]),
            shop_locked=val("shop_locked", self.shop_locked),
            board=val("board", [u.to_obs() for u in self.board]),
            bench=val("bench", [u.to_obs() for u in self.bench]),
            item_bench=val("item_bench", list(self.item_bench)),
            players=val("players", [p.to_obs() for p in self.players]),
            traits=val("traits", [t.to_obs() for t in self.traits]),
            augment_choices=val("augment_choices", list(self.augment_choices)),
            augments=val("augments", list(self.augments)),
            notes=list(self.notes),
        )


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
        """Number of 1-star copies this unit represents (1, 3 or 9).

        A 4-star (Set 18 Solar 8 ascension) is an upgraded 3-star and still
        holds 9 copies of the pool, not 27.
        """
        return 3 ** (min(max(1, self.star), 3) - 1)


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
    style: str = "standard"  # standard | fast8 | fast9 | reroll1 | reroll2 | reroll3
    level_estimated: bool = False  # level was not read, a guess was used
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

    def p_at(self, budget: int) -> float:
        """P(reach goal) with ``budget`` gold: the exact entry when computed,
        otherwise the largest computed budget below it (a lower bound)."""
        if budget in self.p_goal_by_gold:
            return self.p_goal_by_gold[budget]
        options = [g for g in self.p_goal_by_gold if g <= budget]
        return self.p_goal_by_gold[max(options)] if options else 0.0


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
    # From the comp library (None / empty for auto comps built from traits):
    style: Optional[str] = None  # standard | fast8 | fast9 | reroll1 | reroll2 | reroll3
    tier: Optional[str] = None
    positions: dict[str, list[int]] = Field(default_factory=dict)  # unit -> [row, col], row 0 = front (as the board)
    item_holders: dict[str, list[str]] = Field(default_factory=dict)  # unit -> recommended items (tanks included)


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
    shop_picks_cost: int = 0  # gold the shop picks cost (already taken out of econ.roll_budget)
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
    comp: Optional[str] = Field(default=None, description="Target comp direction, Chinese, one sentence, <= 60 chars")
    items: Optional[str] = Field(default=None, description="Item plan, Chinese, one sentence, <= 60 chars")
    positioning: Optional[str] = Field(default=None, description="Positioning tip, Chinese, one sentence, <= 60 chars")
    augment: Optional[str] = Field(
        default=None, description="Augment pick advice if choices visible, Chinese, one sentence, <= 60 chars"
    )
    scout_request: Optional[str] = Field(
        default=None,
        description="If more info is needed, ask the player to open a specific board, Chinese, one sentence, <= 60 chars",
    )
    confidence: float = Field(description="0..1 confidence in this advice")
