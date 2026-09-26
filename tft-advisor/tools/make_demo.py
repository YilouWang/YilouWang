"""Generate the scripted Set 18 demo game used by ``tft-advisor demo``.

Writes ``tft_advisor/data/bundled/demo_s18/obs_*.json``: one ScreenObservation
per step, with names in the bundled zh_cn snapshot's language, telling the story
of a Spirit Blossom (Ahri) game: early loss streak, 3-2 stabilize, a scouted
contest at 3-3, fast 8 at 4-2.

    python tools/make_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tft_advisor.data.setdata import SetData, bundled_snapshot  # noqa: E402
from tft_advisor.models import ScreenObservation  # noqa: E402

OUT = ROOT / "tft_advisor" / "data" / "bundled" / "demo_s18"
SD = SetData.from_cdragon(bundled_snapshot("zh_cn"), bundled_snapshot("en_us"))

PLAYERS = ["我", "星星点灯", "晚安玛卡巴卡", "北极熊", "小火龙", "Kaiser", "Nina", "Leo"]


def c(name: str) -> str:
    champ = SD.resolve_champion(name)
    assert champ, name
    return champ.name


def i(name: str) -> str:
    item = SD.resolve_item(name)
    assert item, name
    return item.name


def unit(name: str, star: int = 1, items=(), row=None, col=None) -> dict:
    return {"name": c(name), "star": star, "items": [i(x) for x in items], "row": row, "col": col}


def shop(*names: str) -> list[dict]:
    out = []
    for n in names:
        if n is None:
            out.append({"name": None, "cost": None})
        elif n.startswith("Wisp"):
            out.append({"name": n, "cost": 3})
        else:
            out.append({"name": c(n), "cost": SD.resolve_champion(n).cost})
    return out


def players(hps: list[int]) -> list[dict]:
    return [{"name": p, "hp": hp, "is_self": p == "我"} for p, hp in zip(PLAYERS, hps)]


STEPS = [
    dict(
        tag="1-3", screen_type="planning", viewing_own_board=True, stage="1-3", gold=4, level=3, xp_current=2, xp_needed=6,
        hp=100, streak=0, shop=shop("Karma", "Yorick", "Xayah", "Varus", "Karma"),
        board=[unit("Karma", 1, row=3, col=1), unit("Yorick", 1, row=0, col=3)], bench=[unit("Rakan")],
        item_bench=["Tear of the Goddess"], players=players([100] * 8),
    ),
    dict(
        tag="2-1_augment", screen_type="augment_select", viewing_own_board=True, stage="2-1", gold=12, level=4, xp_current=0,
        xp_needed=10, hp=100, streak=0, shop=shop("Karma", "Camille", "Alistar", "Yunara", "Leona"),
        board=[unit("Karma", 2, row=3, col=1), unit("Yorick", 1, row=0, col=3), unit("Rakan", 1, row=0, col=2), unit("Yunara", 1, row=3, col=5)],
        bench=[], item_bench=["Tear of the Goddess", "Needlessly Large Rod"],
        augment_choices=["Latent Forge", "Booster Pack", "Spreading Roots"], players=players([100] * 8),
    ),
    dict(
        tag="2-5", screen_type="planning", viewing_own_board=True, stage="2-5", gold=28, level=5, xp_current=4, xp_needed=20,
        hp=82, streak=-3, shop=shop("Vi", "Elise", "Karma", "Caitlyn", "Wisp: Grow Up"),
        board=[unit("Karma", 2, ["Archangel's Staff"], 3, 1), unit("Yorick", 1, row=0, col=3), unit("Rakan", 1, row=0, col=2),
               unit("Yunara", 1, row=3, col=5), unit("Vi", 1, row=0, col=4)],
        bench=[unit("Karma"), unit("Caitlyn")], item_bench=["B.F. Sword"], augments=["Latent Forge"],
        players=players([82, 100, 96, 90, 88, 94, 100, 92]),
    ),
    dict(
        tag="3-2", screen_type="planning", viewing_own_board=True, stage="3-2", gold=34, level=5, xp_current=12, xp_needed=20,
        hp=62, streak=-5, shop=shop("Ahri", "Caitlyn", "Karma", "Sett", None),
        board=[unit("Karma", 2, ["Archangel's Staff"], 3, 0), unit("Yorick", 1, row=0, col=3), unit("Vi", 1, row=0, col=4),
               unit("Yunara", 1, row=3, col=6), unit("Ahri", 1, row=3, col=5)],
        bench=[unit("Karma"), unit("Caitlyn"), unit("Elise")], item_bench=["B.F. Sword", "Sparring Gloves"],
        augments=["Latent Forge"], players=players([62, 100, 94, 86, 80, 78, 97, 88]),
    ),
    dict(
        tag="3-3_scout", screen_type="planning", viewing_own_board=False, viewed_player_name="星星点灯", stage="3-3", gold=18,
        level=6, xp_current=6, xp_needed=36, hp=100, streak=-5, shop=shop("Sett", "Master Yi", "Karma", "Teemo", "Wisp: Search Party"),
        board=[unit("Ahri", 2, ["Jeweled Gauntlet", "Spear of Shojin"], 3, 5), unit("Sett", 1, row=0, col=3), unit("Morgana", 1, row=3, col=1),
               unit("Karma", 2, row=3, col=0), unit("Vi", 2, row=0, col=4), unit("Yorick", 1, row=0, col=2)],
        bench=[unit("Ahri"), unit("Sett")], traits=[{"name": "灵魂莲华", "count": 4, "active": True, "next_breakpoint": 5}],
        players=players([62, 100, 94, 86, 80, 78, 97, 88]),
    ),
    dict(
        tag="3-4_carousel", screen_type="carousel", viewing_own_board=True, stage="3-4", gold=24, level=6, xp_current=8,
        xp_needed=36, hp=58, streak=-5, players=players([58, 100, 92, 83, 77, 74, 95, 85]),
    ),
    dict(
        tag="4-1", screen_type="planning", viewing_own_board=True, stage="4-1", gold=56, level=6, xp_current=26, xp_needed=36,
        hp=48, streak=1, shop=shop("Aphelios", "Sivir", "Amumu", "Diana", "Kha'Zix"),
        board=[unit("Karma", 2, ["Archangel's Staff"], 3, 0), unit("Yorick", 2, row=0, col=3), unit("Vi", 2, ["Warmog's Armor"], 0, 4),
               unit("Yunara", 1, row=3, col=6), unit("Ahri", 1, row=3, col=5), unit("Sett", 1, row=0, col=2)],
        bench=[unit("Caitlyn"), unit("Elise"), unit("Master Yi")], item_bench=["Negatron Cloak"],
        augments=["Latent Forge"], players=players([48, 94, 88, 75, 70, 64, 90, 71]),
    ),
    dict(
        tag="4-2_augment", screen_type="augment_select", viewing_own_board=True, stage="4-2", gold=38, level=7, xp_current=4,
        xp_needed=56, hp=48, streak=2, shop=shop("Sett", "Zyra", "Morgana", "Hecarim", "Lillia"),
        board=[unit("Karma", 2, ["Archangel's Staff"], 3, 0), unit("Yorick", 2, row=0, col=3), unit("Vi", 2, ["Warmog's Armor"], 0, 4),
               unit("Yunara", 1, row=3, col=6), unit("Ahri", 1, ["Jeweled Gauntlet"], 3, 5), unit("Sett", 1, row=0, col=2),
               unit("Master Yi", 1, row=1, col=4)],
        bench=[unit("Sett"), unit("Zyra")], item_bench=[], augments=["Latent Forge"],
        augment_choices=["Epic Rolldown", "Frontline Foundation", "Magic Roll"], players=players([48, 90, 85, 70, 66, 58, 88, 66]),
    ),
    dict(
        tag="4-5", screen_type="planning", viewing_own_board=True, stage="4-5", gold=22, level=8, xp_current=10, xp_needed=68,
        hp=39, streak=-1, shop=shop("Ahri", "Soraka", "Ahri", "Taric", "Wisp: Freeroller"),
        board=[unit("Karma", 2, ["Archangel's Staff"], 3, 0), unit("Yorick", 2, row=0, col=3), unit("Vi", 2, ["Warmog's Armor", "Bramble Vest"], 0, 4),
               unit("Yunara", 1, row=3, col=6), unit("Ahri", 1, ["Jeweled Gauntlet", "Spear of Shojin"], 3, 5),
               unit("Sett", 2, row=0, col=2), unit("Master Yi", 1, row=1, col=4), unit("Zyra", 1, row=3, col=1)],
        bench=[unit("Ahri"), unit("Morgana")], item_bench=["Chain Vest"], augments=["Latent Forge", "Epic Rolldown"],
        players=players([39, 86, 80, 64, 57, 45, 84, 0]),
    ),
    dict(
        tag="5-1", screen_type="planning", viewing_own_board=True, stage="5-1", gold=48, level=8, xp_current=30, xp_needed=68,
        hp=33, streak=1, shop=shop("Ashe", "Morgana", "Sett", "Karma", "Soraka"),
        board=[unit("Karma", 2, ["Archangel's Staff"], 3, 0), unit("Yorick", 2, row=0, col=3), unit("Vi", 2, ["Warmog's Armor", "Bramble Vest"], 0, 4),
               unit("Yunara", 1, row=3, col=6), unit("Ahri", 2, ["Jeweled Gauntlet", "Spear of Shojin"], 3, 5),
               unit("Sett", 2, row=0, col=2), unit("Morgana", 1, row=3, col=1), unit("Zyra", 1, row=3, col=3)],
        bench=[unit("Master Yi")], item_bench=["Tear of the Goddess", "Needlessly Large Rod"], augments=["Latent Forge", "Epic Rolldown"],
        players=players([33, 80, 72, 55, 44, 0, 79, 0]),
    ),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("obs_*.json"):
        old.unlink()
    for n, step in enumerate(STEPS, 1):
        tag = step.pop("tag")
        obs = ScreenObservation.model_validate(step)
        path = OUT / f"obs_{n:02d}_{tag}.json"
        path.write_text(json.dumps(obs.model_dump(mode="json", exclude_none=True), ensure_ascii=False, indent=1), encoding="utf-8")
        print(path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
