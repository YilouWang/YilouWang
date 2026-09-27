"""Build the bundled augment reference (names, effects, rarity, category).

    python tools/build_augments.py --ddragon TFT_DDragon/data --metatft metatft.json \
        --set 18 --s-tier "Latent Forge,Booster Pack+,..." --note "MetaTFT Plat+ 2026-09-26"

``--ddragon`` is a Data Dragon folder with ``en_US/augments.json`` and
``zh_CN/augments.json`` (names and descriptions with numbers filled in).
``--metatft`` is a MetaTFT lookup export (rarity and category tags per
augment). ``--s-tier`` is an optional comma separated list of English names
that a public tier list rated S at build time (static, pre-game information).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tft_advisor" / "data" / "bundled"

CATEGORY_ZH = {
    "Economic": "经济",
    "Combat": "战力",
    "Item": "装备",
    "GrantsItems": "装备",
    "GrantsEmblem": "纹章",
    "Trait": "羁绊",
    "Unit": "英雄",
    "Reroll": "搜牌",
    "RandomAugment": "随机",
    "Utility": "功能",
}
_VERSION = re.compile(r"\d+\.\d+\.\d+")  # Data Dragon zh text sometimes has the game version where a number belongs


def clean(text: str) -> str:
    text = re.sub(r"<rules>.*?</rules>", "", text or "", flags=re.S)
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddragon", required=True)
    ap.add_argument("--metatft")
    ap.add_argument("--set", type=int, default=18)
    ap.add_argument("--s-tier", default="")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    base = Path(args.ddragon)
    zh = json.loads((base / "zh_CN" / "augments.json").read_text(encoding="utf-8"))["data"]
    en = json.loads((base / "en_US" / "augments.json").read_text(encoding="utf-8"))["data"]
    meta: dict[str, dict] = {}
    if args.metatft:
        m = json.loads(Path(args.metatft).read_text(encoding="utf-8"))
        meta = {a.get("apiName"): a for a in m.get("augments", []) if a.get("apiName")}
    s_tier = {x.strip().lower() for x in args.s_tier.split(",") if x.strip()}

    out = {}
    for key, e in en.items():
        api = e.get("id") or ""
        if not api.startswith("DA_"):
            continue
        z = zh.get(key, {})
        mt = meta.get(api, {})
        cats = [t.split(".")[-1] for t in mt.get("tags", []) if t.startswith("Augment.Category.")]
        name_en = (e.get("name") or api).strip()
        desc_en = clean(e.get("description") or "")
        desc_zh = clean(z.get("description") or "")
        if _VERSION.search(desc_zh) or not desc_zh:
            desc_zh = desc_en  # broken localized number: the English text is correct
        category = []
        for c in cats:
            label = CATEGORY_ZH.get(c)
            if label and label not in category:
                category.append(label)
        out[api] = {
            "name": (z.get("name") or name_en).strip(),
            "name_en": name_en,
            "desc": desc_zh[:220],
            "desc_en": desc_en[:260],
            "rarity": mt.get("rarity") or "",
            "category": category,
            "meta_tier": "S" if name_en.lower() in s_tier else "",
        }
    payload = {"meta": {"set": args.set, "note": args.note, "count": len(out)}, "augments": out}
    path = OUT / f"augments_set{args.set}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{path.name}: {len(out)} augments, {sum(1 for a in out.values() if a['meta_tier'])} S tier, {path.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
