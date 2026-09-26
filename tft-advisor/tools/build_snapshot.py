"""Build the trimmed offline set snapshots shipped in tft_advisor/data/bundled/.

The full CommunityDragon export is ~25 MB per locale and covers every set; the
advisor only needs the current set's champions, traits and craftable items.

Usage (from the tft-advisor directory):

    python tools/build_snapshot.py --en en_us.json --zh zh_cn.json
    python tools/build_snapshot.py --en en_us.json --ddragon-zh TFT_DDragon/data/zh_CN

``--en``/``--zh`` take CommunityDragon exports (full, or the per-set trimmed
format with top-level ``champions``/``traits``/``items``/``mutator``/``set``).
``--ddragon-zh`` takes a Data Dragon locale folder (champion.json, trait.json,
item.json) when a CommunityDragon zh_cn export is not available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tft_advisor.data.setdata import COMPONENT_IDS, SetData, canonical_component  # noqa: E402

OUT = ROOT / "tft_advisor" / "data" / "bundled"


def as_full_export(data: dict[str, Any]) -> dict[str, Any]:
    """Accept either a full export or a per-set trimmed one."""
    if "setData" in data:
        return data
    entry = {
        "mutator": data.get("mutator") or f"TFTSet{data.get('set')}",
        "number": data.get("set"),
        "name": data.get("name") or f"Set{data.get('set')}",
        "champions": data.get("champions", []),
        "traits": data.get("traits", []),
        "items": [i.get("apiName") for i in data.get("items", []) if i.get("apiName")],
    }
    return {"items": data.get("items", []), "setData": [entry], "sets": {}}


def trim(full: dict[str, Any], set_number: int, keep_items: Optional[set[str]] = None) -> dict[str, Any]:
    sd = SetData.from_cdragon(full, None, set_number)
    entry = next(
        e
        for e in full["setData"]
        if e.get("mutator") == f"TFTSet{sd.set_number}" or str(e.get("number")) == str(sd.set_number)
    )
    by_api = {i.get("apiName"): i for i in full.get("items", [])}
    if keep_items is None:
        keep_items = set()
        for it in sd.items.values():
            keep_items.add(it.api_name)
            for alias in it.aliases:
                if alias in by_api:
                    keep_items.add(alias)
        for cid in COMPONENT_IDS:
            keep_items.add(cid)
            keep_items.add("DA_Component_" + cid[len("TFT_Item_") :])
    items = []
    for api in sorted(keep_items):
        src = by_api.get(api)
        if not src:
            continue
        items.append(
            {
                "apiName": api,
                "name": src.get("name"),
                "composition": list(src.get("composition") or []),
                "desc": (src.get("desc") or "")[:400],
                "icon": src.get("icon") or "",
            }
        )
    champions = [
        {
            "apiName": c.get("apiName"),
            "characterName": c.get("characterName"),
            "name": c.get("name"),
            "cost": c.get("cost"),
            "traits": c.get("traits") or [],
            "role": c.get("role"),
            "squareIcon": c.get("squareIcon"),
        }
        for c in entry.get("champions", [])
        if isinstance(c.get("cost"), int) and 1 <= c["cost"] <= 5 and c.get("traits")
    ]
    traits = [
        {
            "apiName": t.get("apiName"),
            "name": t.get("name"),
            "desc": (t.get("desc") or "")[:600],
            "effects": [
                {"minUnits": e.get("minUnits"), "maxUnits": e.get("maxUnits"), "style": e.get("style")}
                for e in t.get("effects", [])
            ],
        }
        for t in entry.get("traits", [])
    ]
    return {
        "items": items,
        "setData": [
            {
                "mutator": entry.get("mutator"),
                "number": sd.set_number,
                "name": sd.set_name,
                "champions": champions,
                "traits": traits,
                "items": [i["apiName"] for i in items],
            }
        ],
        "sets": {str(sd.set_number): {"name": sd.set_name}},
    }


def localize_from_ddragon(snapshot: dict[str, Any], ddragon_dir: Path) -> dict[str, Any]:
    def load(name: str) -> dict[str, str]:
        data = json.loads((ddragon_dir / f"{name}.json").read_text(encoding="utf-8"))["data"]
        return {v["id"]: v["name"] for v in data.values() if v.get("id") and v.get("name")}

    champs, traits, items = load("champion"), load("trait"), load("item")
    out = json.loads(json.dumps(snapshot))
    entry = out["setData"][0]
    trait_en_to_api = {t["name"]: t["apiName"] for t in entry["traits"]}
    missing: list[str] = []
    for t in entry["traits"]:
        t["name"] = traits.get(t["apiName"]) or (missing.append(t["apiName"]) or t["name"])
    for c in entry["champions"]:
        c["name"] = champs.get(c["apiName"]) or (missing.append(c["apiName"]) or c["name"])
        c["traits"] = [traits.get(trait_en_to_api.get(x, ""), x) for x in c["traits"]]
    for it in out["items"]:
        it["name"] = items.get(it["apiName"]) or (missing.append(it["apiName"]) or it["name"])
    if missing:
        print(f"warning: {len(missing)} ids without a localized name: {missing[:10]}", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--en", required=True)
    ap.add_argument("--zh")
    ap.add_argument("--ddragon-zh")
    ap.add_argument("--set", type=int, default=0)
    ap.add_argument("--source-note", default="")
    args = ap.parse_args()

    en_full = as_full_export(json.loads(Path(args.en).read_text(encoding="utf-8")))
    en = trim(en_full, args.set)
    keep = {i["apiName"] for i in en["items"]}
    if args.zh:
        zh = trim(as_full_export(json.loads(Path(args.zh).read_text(encoding="utf-8"))), args.set, keep)
    elif args.ddragon_zh:
        zh = localize_from_ddragon(en, Path(args.ddragon_zh))
    else:
        zh = None
    for loc, snap in (("en_us", en), ("zh_cn", zh)):
        if snap is None:
            continue
        snap["_meta"] = {"note": args.source_note, "locale": loc}
        path = OUT / f"snapshot_{loc}.json"
        path.write_text(json.dumps(snap, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        sd = SetData.from_cdragon(snap, en, 0)
        print(f"{path.name}: S{sd.set_number} {sd.set_name} champions={len(sd.champions)} {sd.champion_count_by_cost()} items={len(sd.items)} ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
