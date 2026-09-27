"""Prompts for the Claude vision perceiver.

The system prompt is large and stable (HUD explanation + the set's name lists)
so it is built deterministically once per set and marked for prompt caching by
``tft_advisor.llm``. Everything that changes per call (purpose, hints, image
labels) goes into the user turn built by :func:`build_user_text`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Sequence

from .base import PerceptionHint, clean_name, is_unknown_item, normalize_purpose

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..data.setdata import SetData

SYSTEM_PROMPT_VISION = """\
You are a precise screen reader for Teamfight Tactics (TFT), Riot Games' auto battler, playing on PC (the client may be in Chinese or English). You receive one screenshot (sometimes with high-resolution crops of the same frame) and fill a JSON object describing exactly what is visible. A separate program does all strategy; your only job is faithful reading.

GROUND RULES
1. Read ONLY what is visible in the images. Never infer from game knowledge, never estimate, never fill in "typical" values.
2. Every field of the JSON object is required. When a top-level field is not visible or you cannot read it with confidence, put its name in the "unreadable" list and fill it with a placeholder: -1 for gold, xp_current and hp, 0 for level, xp_needed and streak, "" for text, false for booleans, [] for lists. Never guess numbers: a wrong gold, level, XP or HP value is much worse than listing the field as unreadable.
3. An empty list that is NOT in "unreadable" means "this area is visible and empty" (for example an empty bench is []); a field listed in "unreadable" means "not visible / not readable".
4. Names: when what you see maps to an entry of the NAME LISTS below, output the name EXACTLY as written in that list (either the Chinese or the English form, whichever matches the client language). If you cannot map it, write what you read literally and add a note.
5. Several images may be provided. IMAGE 1 is usually the full screenshot (downscaled); the other images are high-resolution crops of the SAME frame at the same moment. Prefer the crops for small text and numbers, and use the full screenshot for layout and context.
6. notes: short remarks about anything uncertain or notable (for example an unclear star level), written in Simplified Chinese, at most 5 notes, no dash punctuation.

SCREEN TYPES (screen_type)
- planning: units can be moved, the round timer counts down, the shop is usable.
- combat: units are fighting on the board.
- carousel: shared draft, champions walk in a circle in the middle of the map.
- augment_select: 3 augment cards are offered in the middle of the screen.
- loading: loading screen. post_game: placement / results screen. other: client, menus, anything that is not a TFT match or is unreadable.

HUD LAYOUT (16:9)
- Stage indicator: top center, format "stage-round" such as "3-2". Output it exactly as "S-R" (digits and a hyphen).
- The BOTTOM HUD always belongs to the LOCAL player (the person playing), EVEN WHILE the camera shows another player's board:
  * Shop: 5 champion cards in a row at the bottom center, read left to right. Each card shows the champion name (bottom left of the card) and its gold cost (bottom right, next to a coin). The card frame color also shows the cost: gray 1, green 2, blue 3, purple 4, gold 5. A bought or empty slot is {"name": "", "cost": 0}; a card whose price you cannot read has cost -1. When the shop is visible always output exactly 5 slots.
  * shop_locked: the padlock icon next to the shop; closed / highlighted means locked.
  * Gold: the number next to the coin icon, centered just above the shop.
  * Level and XP: on the left of the shop, above the Buy XP and Refresh buttons, shown like "Lv. 6" (Chinese client: "6级" or "等级 6") with an XP bar labeled "x/y": level, xp_current = x, xp_needed = y. At max level there is no "x/y": list xp_current and xp_needed in unreadable.
  * Streak: a flame icon (win streak) or an ice / blue icon (loss streak) next to the gold with a number N. Output streak = +N for a win streak and -N for a loss streak. No streak icon while the gold area is visible: 0. Gold area not visible: list streak in unreadable.
- The BOARD, BENCH, TRAIT PANEL, ITEM BENCH and the TOP BANNER belong to the player whose board is shown (the owner of the arena the camera is on):
  * Board: the hex grid in the middle, 4 rows of 7 hexes. row 0 = front row, nearest the center line of the arena (farthest from the bench); row 3 = back row, nearest the bench. col 0 = leftmost hex of that row as seen on screen, up to col 6.
  * Bench: the row of 9 slots between the board and the shop. For bench units row = -1 and col = slot index 0-8 from the left (-1 if unsure).
  * Star level: the 1-3 small stars above a unit's health bar (1 bronze star, 2 silver stars, 3 gold stars). If unclear use 1 and add a note.
  * Items: small square icons just under a unit's health bar, at most 3 per unit. Name each icon you recognize from the item list and write "?" for each icon you can see but cannot name, so a unit showing 3 icons always has 3 entries (for example ["<item name>", "?", "?"]). items is [] only when the unit holds no item. For "?" icons add a note such as "某单位有 2 件无法识别的装备".
  * item_bench: unequipped components / items lying on the left side of the arena next to the tactician area.
  * Traits: vertical panel on the left edge: trait name, number of unique units, breakpoints such as "2 / 4 / 6". Active traits have a colored hexagon (bronze, silver, gold, prismatic), inactive ones are gray. next_breakpoint = the next breakpoint above the current count, 0 at the maximum or when unreadable.
- Player list: right edge, up to 8 rows ordered top to bottom as shown, each with a name (or only a portrait) and an HP number. The local player's row is highlighted (lighter background or distinct frame): mark it is_self = true. Players with 0 HP or grayed out are eliminated (hp 0). A row whose HP you cannot read: hp -1.
- hp (top level): the HP of the player whose board is shown. For the local player read it from the highlighted row of the player list.
- viewing_own_board: true when the camera shows the local player's own arena; false when it shows another player's arena (scouting: a banner with another player's name near the top, another tactician, a different board); when unclear, list viewing_own_board in unreadable.
- viewed_player_name: the owner of the board shown, from the top banner or the selected row of the player list, when readable; otherwise "" and list it in unreadable.
- viewed_player_level: when the camera shows another player's arena, the level on the plate above that arena; 0 on the local player's own arena or when the plate is not readable. The top level "level" field is always the local player's HUD level.
- Augments: augment_choices = the names on the offered cards, left to right (augment_select screen only). augments = augments the player already owns, only when their names are readable; otherwise list augments in unreadable. Outside the augment_select screen augment_choices is [].
- During carousel, combat or loading still fill every HUD field that is visible (stage, gold, level, player list).
"""


def _pair(name: str, name_en: str) -> str:
    return name if not name_en or name == name_en else f"{name} / {name_en}"


def _item_kind_order(kind: str) -> int:
    return {"component": 0, "completed": 1, "emblem": 2}.get(kind, 3)


def build_name_lists(set_data: "SetData") -> str:
    """Champion / trait / item names of the set, both locales, sorted (deterministic)."""
    lines: list[str] = [
        "NAME LISTS",
        f"Set: {set_data.set_number} {set_data.set_name}".rstrip(),
        "Format: display name / english name. Output names exactly as written here.",
        "",
        "CHAMPIONS by shop cost:",
    ]
    champs = list(set_data.champions.values())
    for cost in sorted({c.cost for c in champs}):
        group = sorted((c for c in champs if c.cost == cost), key=lambda c: (c.name_en, c.name))
        lines.append(f"[{cost} cost] " + "; ".join(_pair(c.name, c.name_en) for c in group))
    lines += ["", "TRAITS (breakpoints):"]
    for t in sorted(set_data.traits.values(), key=lambda t: (t.name_en, t.name)):
        bp = "/".join(str(b) for b in t.breakpoints) or "-"
        lines.append(f"- {_pair(t.name, t.name_en)} ({bp})")
    lines += ["", "ITEMS:"]
    items = sorted(set_data.items.values(), key=lambda i: (_item_kind_order(i.kind), i.name_en, i.name))
    current_kind: Optional[str] = None
    buf: list[str] = []
    for it in items:
        kind = it.kind if it.kind in ("component", "completed", "emblem") else "special"
        if kind != current_kind:
            if buf:
                lines.append(f"[{current_kind}] " + "; ".join(buf))
            current_kind, buf = kind, []
        buf.append(_pair(it.name, it.name_en))
    if buf:
        lines.append(f"[{current_kind}] " + "; ".join(buf))
    return "\n".join(lines)


# HUD notes for the Unreal Engine client (Set 18 onward). Kept separate from
# the generic prompt so an older client can still be read.
UNREAL_HUD_NOTES = """\
UNREAL CLIENT HUD (Set 18 and later), these override the generic layout above where they differ:
- Bottom HUD row, left to right: "Lvl. N" with the XP text "x/y" next to it, then five shop odds percentages (for example "30% 40% 25% 5% 0%", do not confuse them with gold), then the gold number with a coin, then the streak icon (flame = win, ice = loss) with its number, then a badge with the team size.
- Buy XP and Reroll buttons are on the far left of the shop; the 5 shop cards follow. The champion name is written on the dark banner at the bottom of each card, the cost number with a coin at the banner's right end.
- WISPS: in every other shop the rightmost card is a Wisp (a glowing orb consumable with its own name and price) instead of a champion. Output that slot as {"name": "Wisp: <the name as written>", "cost": <its price>} (a free Wisp has cost 0, an unreadable price -1). Never invent a champion for it.
- Item bench: a vertical column of up to 10 square slots at the far LEFT edge of the screen, left of the trait panel. Stacked identical items show a small count number: repeat the name that many times in item_bench.
- Trait panel: just right of the item column; each row is a trait icon, the unit count, the trait name and its breakpoints.
- Player list: right edge; the local player's row is drawn larger with a bigger HP number.
- When the camera shows another player's arena, that player's name and level appear on a plate above their arena.
- Carousel: the ring of champions in the middle belongs to nobody: list board, bench, traits and item_bench in unreadable.
- Not champions, never list them as units: Elderwood plants (Stonebark Tree, Lifebloom, Deepwood Protector), summons (Azir soldiers, Zyra plants), the tactician (little legend), training dummies.
- Lux has trait forms ("Lux (Coven)" etc.): output the plain champion name from the NAME LISTS and put the form in a note.
- Units can reach 4 stars in this set (4 small stars): star = 4.
"""


def build_vision_system(set_data: "SetData") -> str:
    """Full system prompt for the vision perceiver. Deterministic for a given set (prompt cache)."""
    hud = UNREAL_HUD_NOTES + "\n" if (set_data.set_number or 0) >= 18 else ""
    return SYSTEM_PROMPT_VISION + "\n" + hud + build_name_lists(set_data) + "\n"


_PURPOSE_TEXT = {
    "auto": (
        "TASK: automatic capture at a round change. Read every visible field of the JSON object."
    ),
    "manual": (
        "TASK: the player asked for a full analysis right now. Read every visible field carefully, "
        "including star levels and the items under every unit."
    ),
    "scout": (
        "TASK: SCOUTING. The player switched the camera to ANOTHER player's board to show it to you. "
        "board, bench, traits, item_bench and the top level hp describe THAT player. "
        "Set viewing_own_board = false unless the local player's own board is clearly shown, and read "
        "viewed_player_name from the top banner (or the selected row of the player list). "
        "gold, level, XP, streak, shop and shop_locked still come from the bottom HUD and belong to the "
        "LOCAL player. Put the viewed player's level from the plate above their arena in viewed_player_level "
        "(0 when it is not shown)."
    ),
    "shop": (
        "TASK: SHOP ONLY. Only the bottom HUD is provided. Fill shop (exactly 5 slots, left to right), "
        "shop_locked, gold, level, xp_current, xp_needed and streak. List every other field in unreadable "
        "(stage, hp, board, bench, item_bench, players, traits, augments, augment_choices, viewed_player_name, "
        "viewing_own_board). screen_type = planning when the shop is visible."
    ),
}

# A real "likely on screen" list is short (board + bench + shop, about 25 names).
# A longer list (e.g. every name of the set) carries no information beyond the
# NAME LISTS already in the cached system prompt, so it is left out of the
# uncached user turn instead of being truncated into a misleading subset.
_MAX_HINT_NAMES = 30


def _join_names(names: Iterable[str]) -> str:
    seen: list[str] = []
    for raw in names:
        n = clean_name(raw)
        if n and is_unknown_item(n):
            continue  # "?" placeholders from earlier frames name nothing
        if n and n not in seen:
            seen.append(n)
            if len(seen) > _MAX_HINT_NAMES:
                return ""
    return ", ".join(seen)


def _quoted(value: object) -> Optional[str]:
    """A hint value safe to put between double quotes (any type, e.g. a number sent by the dashboard)."""
    name = clean_name(value)  # type: ignore[arg-type]
    return name.replace('"', "'") if name else None


def build_user_text(purpose: str, hint: Optional[PerceptionHint], image_labels: Sequence[str]) -> str:
    """Per-call instruction: what the images are, what to read, optional hints."""
    mode = normalize_purpose(purpose)
    lines: list[str] = []
    if image_labels:
        lines.append("Images in this request (same frame):")
        lines += [f"- {label}" for label in image_labels]
        lines.append("")
    lines.append(_PURPOSE_TEXT[mode])
    if hint is not None:
        # Hint strings can come from the dashboard (typed on a phone) or from
        # earlier model output: single line, short, no quotes that could end
        # the quoted value early.
        self_name = _quoted(hint.self_name)
        scouting = _quoted(hint.scouting_player)
        expect = clean_name(hint.expect, max_len=80)
        hint_lines: list[str] = []
        if self_name:
            hint_lines.append(f"- The local player's name is \"{self_name}\" (mark is_self on that row).")
        if scouting:
            hint_lines.append(f"- The player says the board shown belongs to \"{scouting}\".")
        if expect:
            hint_lines.append(f"- Expected screen: {expect}.")
        for label, names in (
            ("Champions", hint.champion_names),
            ("Items", hint.item_names),
            ("Traits", hint.trait_names),
        ):
            joined = _join_names(names or [])
            if joined:
                hint_lines.append(f"- {label} likely on screen: {joined}.")
        if hint_lines:
            lines.append("")
            lines.append("HINTS from earlier frames (may be outdated; the image always wins):")
            lines += hint_lines
    lines.append("")
    lines.append('Return only the JSON object. List every field that is not visible or not readable in "unreadable".')
    return "\n".join(lines)


__all__ = ["SYSTEM_PROMPT_VISION", "build_name_lists", "build_user_text", "build_vision_system"]
