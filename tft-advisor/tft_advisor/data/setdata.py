"""Set data (champions, traits, items) from CommunityDragon.

CommunityDragon publishes the full TFT data export for every locale:
``https://raw.communitydragon.org/latest/cdragon/tft/<locale>.json``. Top level:
``items`` (all items of all sets), ``setData`` (one entry per set / mutator with
``champions``, ``traits`` and, in recent exports, ``items``/``augments`` id lists),
``sets``. We load the configured display locale (default zh_cn) plus en_us so
both Chinese and English names resolve.

Quirks handled here (seen in the Set 18 "Enchanted Wilds" export, Unreal era):
  * apiNames use ``DA_`` prefixes (``DA_18_Akali_AD``, ``DA_InfinityEdge``,
    ``DA_Component_BFSword``) next to legacy ``TFT_Item_*`` duplicates.
  * Trait-variant champions (``Lux (Coven)`` ...) are the same shop unit: they
    are folded into the base champion so pool sizes stay correct.
  * The set entry's ``name`` can be stale ("Set10" for Set 18).
"""

from __future__ import annotations

import difflib
import json
import os
import re
import time
import unicodedata
import urllib.request
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from ..config import DataConfig

COMPONENT_IDS: tuple[str, ...] = (
    "TFT_Item_BFSword",
    "TFT_Item_RecurveBow",
    "TFT_Item_NeedlesslyLargeRod",
    "TFT_Item_TearOfTheGoddess",
    "TFT_Item_ChainVest",
    "TFT_Item_NegatronCloak",
    "TFT_Item_GiantsBelt",
    "TFT_Item_SparringGloves",
    "TFT_Item_Spatula",
    "TFT_Item_FryingPan",
)
EMBLEM_COMPONENTS = {"TFT_Item_Spatula", "TFT_Item_FryingPan"}

KNOWN_SET_NAMES = {
    13: "Into the Arcane",
    14: "Cyber City",
    15: "K.O. Coliseum",
    16: "Lore & Legends",
    17: "Space Gods",
    18: "Enchanted Wilds",
}

# apiName fragments that mark items/units we never want to suggest.
_ITEM_BLOCKLIST = re.compile(
    r"(Tutorial|Debug|_HR$|Deprecated|Consumable|Grant|Dummy|Blank|Unusable|Placeholder|Test)", re.I
)
# PvE monsters have no traits and are dropped by that rule; this only catches
# odd leftovers. Do NOT add real monster names here (Set 18 sells Krug, Murkwolf...).
_UNIT_BLOCKLIST = re.compile(r"(TrainingDummy|Tutorial|ArmoryKey|MercenaryChest)", re.I)
_VARIANT_NAME = re.compile(r"^(.+?)\s*[（(]\s*([^()（）]+?)\s*[)）]\s*$")

_SET_MUTATOR = re.compile(r"^TFTSet(\d+)(?:_(Stage2|Act2))?$", re.I)


def normalize_name(text: str) -> str:
    """Lowercase, NFKC, strip spaces and punctuation (keeps CJK)."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    return "".join(ch for ch in t if ch.isalnum())


def canonical_component(api: str) -> Optional[str]:
    """Map DA_Component_X / TFT_Item_X component ids to the TFT_Item_X id."""
    if api in COMPONENT_IDS:
        return api
    if api.startswith("DA_Component_"):
        cand = "TFT_Item_" + api[len("DA_Component_") :]
        if cand in COMPONENT_IDS:
            return cand
    return None


@dataclass
class Champion:
    api_name: str
    name: str
    name_en: str
    cost: int
    traits: list[str] = field(default_factory=list)  # display names (locale)
    traits_en: list[str] = field(default_factory=list)
    icon: Optional[str] = None
    variants: list[str] = field(default_factory=list)  # api names folded into this unit


@dataclass
class Trait:
    api_name: str
    name: str
    name_en: str
    breakpoints: list[int] = field(default_factory=list)
    desc: str = ""


@dataclass
class Item:
    api_name: str
    name: str
    name_en: str
    composition: list[str] = field(default_factory=list)  # canonical component api names
    kind: str = "special"  # component | completed | emblem | special
    desc: str = ""
    aliases: list[str] = field(default_factory=list)  # other api names / names for the same item


def _strip_markup(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    text = re.sub(r"@[^@]+@", "X", text)
    text = text.replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", text).strip()


class SetData:
    def __init__(
        self,
        set_number: int,
        set_name: str,
        champions: dict[str, Champion],
        traits: dict[str, Trait],
        items: dict[str, Item],
        source: str = "",
        champion_aliases: Optional[dict[str, list[str]]] = None,
    ) -> None:
        self.set_number = set_number
        self.set_name = set_name
        self.champions = champions
        self.traits = traits
        self.items = items
        self.source = source
        champion_aliases = champion_aliases or {}
        self._champ_index = self._build_index(
            (
                c.api_name,
                [c.name, c.name_en, c.api_name, *_api_tail_names(c.api_name), *champion_aliases.get(c.api_name, [])],
            )
            for c in champions.values()
        )
        self._trait_index = self._build_index(
            (t.api_name, [t.name, t.name_en, t.api_name, *_api_tail_names(t.api_name)]) for t in traits.values()
        )
        self._item_index = self._build_index(
            (i.api_name, [i.name, i.name_en, i.api_name, *_api_tail_names(i.api_name), *i.aliases])
            for i in items.values()
        )
        self._recipes: dict[tuple[str, str], str] = {}
        for it in items.values():
            if it.kind in ("completed", "emblem") and len(it.composition) == 2:
                self._recipes[_pair_key(*it.composition)] = it.api_name

    # ---- indexes ---------------------------------------------------------------
    @staticmethod
    def _build_index(entries: Iterable[tuple[str, list[str]]]) -> dict[str, str]:
        index: dict[str, str] = {}
        for api, names in entries:
            for n in names:
                key = normalize_name(n)
                if key and key not in index:
                    index[key] = api
        return index

    @staticmethod
    def _resolve(name: Optional[str], index: dict[str, str], cutoff: float) -> Optional[str]:
        if not name:
            return None
        key = normalize_name(name)
        if not key:
            return None
        if key in index:
            return index[key]
        if len(key) < 2:
            return None
        # CJK names are short; require a higher similarity for them.
        is_cjk = any("一" <= ch <= "鿿" for ch in key)
        match = difflib.get_close_matches(key, index.keys(), n=1, cutoff=0.85 if is_cjk else cutoff)
        if match:
            return index[match[0]]
        # "Lux (Coven)" / "拉克丝（魔女）" style names fall back to the base name.
        m = _VARIANT_NAME.match(name)
        if m:
            return SetData._resolve(m.group(1), index, cutoff)
        return None

    def resolve_champion(self, name: Optional[str]) -> Optional[Champion]:
        api = self._resolve(name, self._champ_index, 0.78)
        return self.champions.get(api) if api else None

    def resolve_trait(self, name: Optional[str]) -> Optional[Trait]:
        api = self._resolve(name, self._trait_index, 0.78)
        return self.traits.get(api) if api else None

    def resolve_item(self, name: Optional[str]) -> Optional[Item]:
        api = self._resolve(name, self._item_index, 0.8)
        return self.items.get(api) if api else None

    # ---- queries ---------------------------------------------------------------
    def champions_by_cost(self, cost: int) -> list[Champion]:
        return sorted((c for c in self.champions.values() if c.cost == cost), key=lambda c: c.name_en)

    def champion_count_by_cost(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for c in self.champions.values():
            out[c.cost] = out.get(c.cost, 0) + 1
        return dict(sorted(out.items()))

    def components(self) -> list[Item]:
        return [i for i in self.items.values() if i.kind == "component"]

    def completed_items(self) -> list[Item]:
        return [i for i in self.items.values() if i.kind in ("completed", "emblem")]

    def recipe(self, a: str, b: str) -> Optional[Item]:
        ia, ib = self.resolve_item(a), self.resolve_item(b)
        if not ia or not ib:
            return None
        api = self._recipes.get(_pair_key(ia.api_name, ib.api_name))
        return self.items.get(api) if api else None

    def trait_units(self, trait_name: str) -> list[Champion]:
        t = self.resolve_trait(trait_name)
        if not t:
            return []
        return [c for c in self.champions.values() if t.name in c.traits or t.name_en in c.traits_en]

    def champion_names(self) -> list[str]:
        return sorted({c.name for c in self.champions.values()} | {c.name_en for c in self.champions.values()})

    def item_names(self) -> list[str]:
        return sorted({i.name for i in self.items.values()} | {i.name_en for i in self.items.values()})

    def summary_text(self) -> str:
        """Compact reference for LLM prompts. Deterministic (prompt-cache friendly)."""
        title = f"TFT Set {self.set_number} {self.set_name}".strip()
        lines = [title, "", "CHAMPIONS (cost: name / english - traits):"]
        for cost in sorted({c.cost for c in self.champions.values()}):
            lines.append(f"[{cost} cost]")
            for c in self.champions_by_cost(cost):
                nm = c.name if c.name == c.name_en else f"{c.name} / {c.name_en}"
                extra = f" (variants: {len(c.variants)})" if c.variants else ""
                lines.append(f"- {nm}: {', '.join(c.traits)}{extra}")
        lines += ["", "TRAITS (name / english: breakpoints):"]
        for t in sorted(self.traits.values(), key=lambda t: (t.name_en, t.api_name)):
            nm = t.name if t.name == t.name_en else f"{t.name} / {t.name_en}"
            bp = "/".join(str(b) for b in t.breakpoints) or "-"
            lines.append(f"- {nm}: {bp}")
        lines += ["", "ITEM RECIPES (component + component = item):"]
        for it in sorted(self.completed_items(), key=lambda i: (i.name_en, i.api_name)):
            if len(it.composition) != 2:
                continue
            a, b = (self.items.get(x) for x in it.composition)
            if not a or not b:
                continue
            nm = it.name if it.name == it.name_en else f"{it.name} / {it.name_en}"
            lines.append(f"- {a.name} + {b.name} = {nm}")
        return "\n".join(lines)

    # ---- construction ----------------------------------------------------------
    @classmethod
    def from_cdragon(
        cls, data: dict[str, Any], data_en: Optional[dict[str, Any]] = None, set_number: int = 0, source: str = ""
    ) -> "SetData":
        data_en = data_en or data
        entry = _pick_set_entry(data.get("setData") or [], set_number)
        entry_en = _match_entry(data_en.get("setData") or [], entry) or entry
        en_champs = {c.get("apiName"): c for c in entry_en.get("champions", [])}
        en_traits = {t.get("apiName"): t for t in entry_en.get("traits", [])}
        number = _entry_number(entry)

        traits: dict[str, Trait] = {}
        for t in entry.get("traits", []):
            api = t.get("apiName")
            if not api:
                continue
            te = en_traits.get(api, t)
            bps = sorted({int(e.get("minUnits") or 0) for e in t.get("effects", []) if e.get("minUnits")})
            traits[api] = Trait(
                api_name=api,
                name=t.get("name") or api,
                name_en=te.get("name") or t.get("name") or api,
                breakpoints=bps,
                desc=_strip_markup(t.get("desc", ""))[:300],
            )

        raw_champs: list[Champion] = []
        for c in entry.get("champions", []):
            api = c.get("apiName") or c.get("characterName")
            cost = c.get("cost")
            tr = [x for x in (c.get("traits") or []) if x]
            if not api or not isinstance(cost, int) or not 1 <= cost <= 5 or not tr:
                continue
            if _UNIT_BLOCKLIST.search(api):
                continue
            ce = en_champs.get(api, c)
            raw_champs.append(
                Champion(
                    api_name=api,
                    name=c.get("name") or api,
                    name_en=ce.get("name") or c.get("name") or api,
                    cost=cost,
                    traits=tr,
                    traits_en=[x for x in (ce.get("traits") or tr) if x],
                    icon=c.get("squareIcon") or c.get("tileIcon") or c.get("icon"),
                )
            )
        champions, aliases = _fold_variants(raw_champs)

        set_items = entry.get("items")
        set_item_ids = {x if isinstance(x, str) else x.get("apiName") for x in set_items} if set_items else None
        items = _parse_items(data.get("items") or [], data_en.get("items") or [], number, set_item_ids)

        name = str(entry.get("name") or "")
        sets = data_en.get("sets") if isinstance(data_en.get("sets"), dict) else {}
        set_name = KNOWN_SET_NAMES.get(number) or (sets.get(str(number)) or {}).get("name") or name
        if re.fullmatch(r"Set\s*\d+", set_name or "") and set_name.replace(" ", "") != f"Set{number}":
            set_name = ""  # stale export label
        return cls(
            set_number=number,
            set_name=set_name or str(entry.get("mutator") or ""),
            champions=champions,
            traits=traits,
            items=items,
            source=source,
            champion_aliases=aliases,
        )


def _api_tail_names(api: str) -> list[str]:
    """Readable fragments of an api name: DA_18_Akali_AD -> Akali, TFT_Item_InfinityEdge -> InfinityEdge."""
    parts = [p for p in re.split(r"_", api) if p and not p.isdigit() and p not in ("DA", "TFT", "Item", "AD", "AP", "Component")]
    return [re.sub(r"\d+$", "", p) for p in parts[:1]] if parts else []


def _fold_variants(raw: list[Champion]) -> tuple[dict[str, Champion], dict[str, list[str]]]:
    """Fold 'Lux (Coven)' style variants into the base unit of the same cost."""
    by_name: dict[tuple[str, int], Champion] = {}
    for c in sorted(raw, key=lambda c: c.api_name):
        by_name.setdefault((c.name_en, c.cost), c)
    champions: dict[str, Champion] = {}
    aliases: dict[str, list[str]] = {}
    variants: list[tuple[Champion, Champion]] = []
    for c in sorted(raw, key=lambda c: c.api_name):
        m = _VARIANT_NAME.match(c.name_en)
        base = by_name.get((m.group(1).strip(), c.cost)) if m else None
        if base is not None and base is not c:
            variants.append((c, base))
            continue
        if (c.name_en, c.cost) in by_name and by_name[(c.name_en, c.cost)] is not c:
            variants.append((c, by_name[(c.name_en, c.cost)]))  # exact duplicate name
            continue
        champions[c.api_name] = c
    for v, base in variants:
        base.variants.append(v.api_name)
        aliases.setdefault(base.api_name, []).extend([v.api_name, v.name, v.name_en])
    return champions, aliases


def _pair_key(a: str, b: str) -> tuple[str, str]:
    x, y = sorted((a, b))
    return (x, y)


def _entry_number(entry: dict[str, Any]) -> int:
    m = _SET_MUTATOR.match(str(entry.get("mutator") or ""))
    if m:
        return int(m.group(1))
    try:
        return int(float(entry.get("number") or 0))
    except (TypeError, ValueError):
        return 0


def _pick_set_entry(entries: list[dict[str, Any]], set_number: int) -> dict[str, Any]:
    if not entries:
        raise ValueError("CommunityDragon data has no setData entries")
    standard = [e for e in entries if _SET_MUTATOR.match(str(e.get("mutator") or ""))]
    pool = standard or entries
    if set_number:
        pool = [e for e in pool if _entry_number(e) == set_number] or pool
    top = max(_entry_number(e) for e in pool)
    same = [e for e in pool if _entry_number(e) == top]

    def rank(e: dict[str, Any]) -> tuple[int, int]:
        m = _SET_MUTATOR.match(str(e.get("mutator") or ""))
        stage2 = 1 if (m and m.group(2)) else 0
        return (stage2, len(e.get("champions") or []))

    return max(same, key=rank)


def _match_entry(entries: list[dict[str, Any]], ref: dict[str, Any]) -> Optional[dict[str, Any]]:
    for e in entries:
        if e.get("mutator") == ref.get("mutator") and e.get("number") == ref.get("number"):
            return e
    return None


def _parse_items(
    raw: list[dict[str, Any]],
    raw_en: list[dict[str, Any]],
    set_number: int,
    set_item_ids: Optional[set[str]] = None,
) -> dict[str, Item]:
    en = {i.get("apiName"): i for i in raw_en}
    by_api = {i.get("apiName"): i for i in raw}
    in_set = (lambda api: api in set_item_ids) if set_item_ids else (lambda api: True)
    new_era = bool(set_item_ids) and any(a.startswith("DA_") for a in set_item_ids)

    def en_name(api: str, src: dict[str, Any]) -> str:
        return (en.get(api) or src).get("name") or src.get("name") or api

    # Components: canonical TFT_Item_X id; display name from the set's own
    # version (DA_Component_X in the Unreal era) when present.
    items: dict[str, Item] = {}
    for cid in COMPONENT_IDS:
        versions = [a for a in (f"DA_Component_{cid[len('TFT_Item_'):]}", cid) if a in by_api]
        preferred = [a for a in versions if in_set(a)] or versions
        if new_era:
            preferred.sort(key=lambda a: not a.startswith("DA_"))
        else:
            preferred.sort(key=lambda a: not a.startswith("TFT_Item_"))
        if not preferred:
            continue
        main = by_api[preferred[0]]
        aliases: list[str] = []
        for a in versions:
            aliases += [a, by_api[a].get("name") or "", en_name(a, by_api[a])]
        items[cid] = Item(
            api_name=cid,
            name=main.get("name") or cid,
            name_en=en_name(preferred[0], main),
            kind="component",
            desc=_strip_markup(main.get("desc", ""))[:200],
            aliases=[x for x in aliases if x],
        )

    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for it in raw:
        api = it.get("apiName") or ""
        comp = [canonical_component(c) for c in (it.get("composition") or [])]
        if len(comp) != 2 or not all(c in items for c in comp) or _ITEM_BLOCKLIST.search(api):
            continue
        if not it.get("name") or not in_set(api):
            continue
        if not set_item_ids and not it.get("icon"):
            continue
        by_pair.setdefault(_pair_key(comp[0], comp[1]), []).append(it)  # type: ignore[arg-type]

    set_prefixes = (f"TFT{set_number}_", f"DA_{set_number}_")
    for pair, cands in by_pair.items():
        emblem = bool(EMBLEM_COMPONENTS & set(pair))

        def score(it: dict[str, Any]) -> tuple[int, int, int]:
            api = it.get("apiName") or ""
            current_set = 1 if api.startswith(set_prefixes) else 0
            if new_era:
                primary = 2 if api.startswith("DA_") else (1 if api.startswith("TFT_Item_") else 0)
            elif emblem:
                primary = current_set
            else:
                # The generic TFT_Item_ wins, then the current set's override.
                primary = (2 if api.startswith("TFT_Item_") else 0) + current_set
            return (primary, current_set, -len(api))

        best = max(cands, key=score)
        api = best["apiName"]
        # Emblem recipes from other sets are noise unless the set list vouches for them.
        if emblem and not set_item_ids and not (api.startswith(set_prefixes) or api.startswith("TFT_Item_")):
            continue
        aliases = []
        for other in cands:
            if other is not best:
                aliases += [other["apiName"], other.get("name") or "", en_name(other["apiName"], other)]
        items[api] = Item(
            api_name=api,
            name=best.get("name") or api,
            name_en=en_name(api, best),
            composition=list(pair),
            kind="emblem" if emblem and "Emblem" in api else "completed",
            desc=_strip_markup(best.get("desc", ""))[:200],
            aliases=[x for x in aliases if x],
        )
    return items


# ---- loading ---------------------------------------------------------------------


def _download(url: str, dest: Path, timeout: float = 60.0) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "tft-advisor/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read()
    json.loads(payload)  # validate before replacing the cache
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, dest)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _bundled_json(name: str) -> Optional[dict[str, Any]]:
    try:
        text = resources.files("tft_advisor.data").joinpath("bundled", name).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    return json.loads(text)


def bundled_sample() -> dict[str, Any]:
    data = _bundled_json("sample_set.json")
    assert data is not None, "bundled sample_set.json missing"
    return data


def bundled_snapshot(locale: str) -> Optional[dict[str, Any]]:
    """Trimmed real CommunityDragon export of the latest set known at release time."""
    return _bundled_json(f"snapshot_{locale}.json")


def load_set_data(cfg: DataConfig, offline: bool = False, log: Callable[[str], None] = print) -> SetData:
    """Download (or reuse cached) CommunityDragon exports and build SetData.

    Order: fresh download -> cache -> bundled snapshot (real data, may be a
    patch behind) -> bundled synthetic sample (tests only).
    """
    cache = Path(os.path.expanduser(cfg.cache_dir))
    cache.mkdir(parents=True, exist_ok=True)
    locales = [cfg.locale] + ([] if cfg.locale == "en_us" else ["en_us"])
    paths: dict[str, Path] = {}
    for loc in locales:
        dest = cache / f"cdragon_tft_{loc}.json"
        fresh = dest.is_file() and (time.time() - dest.stat().st_mtime) < cfg.refresh_hours * 3600
        if not offline and not fresh:
            url = f"{cfg.cdragon_base.rstrip('/')}/{loc}.json"
            try:
                log(f"下载 TFT 数据 {url} ...")
                _download(url, dest)
            except Exception as exc:  # network / proxy / JSON problems
                log(f"下载失败 ({exc})" + ("，使用缓存" if dest.is_file() else ""))
        if dest.is_file():
            paths[loc] = dest

    def build(data: dict[str, Any], data_en: Optional[dict[str, Any]], source: str) -> Optional[SetData]:
        try:
            sd = SetData.from_cdragon(data, data_en, cfg.set_number, source=source)
        except Exception as exc:
            log(f"解析数据失败 ({source}): {exc}")
            return None
        return sd if sd.champions else None

    if cfg.locale in paths or "en_us" in paths:
        main = paths.get(cfg.locale) or paths["en_us"]
        data = _read_json(main)
        data_en = _read_json(paths["en_us"]) if "en_us" in paths and main != paths["en_us"] else None
        sd = build(data, data_en, str(main))
        if sd:
            return sd

    snap = bundled_snapshot(cfg.locale) or bundled_snapshot("en_us")
    if snap is not None:
        snap_en = bundled_snapshot("en_us")
        sd = build(snap, snap_en, f"bundled-snapshot-{cfg.locale}")
        if sd:
            log(f"使用内置赛季快照 S{sd.set_number}（可能落后一个补丁）。联网后运行 `tft-advisor data update` 获取最新数据。")
            return sd

    log("警告: 使用内置样例数据 (不是当前赛季!)。请联网后运行 `tft-advisor data update`。")
    sample = bundled_sample()
    return SetData.from_cdragon(sample, sample, 0, source="bundled-sample")
