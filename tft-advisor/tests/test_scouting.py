from __future__ import annotations

import pytest

from tft_advisor.advisor.scouting import REQUEST_TEXT, ScoutPlanner, request_id, round_index
from tft_advisor.models import (
    Analysis,
    CompSuggestion,
    GameState,
    OpponentSnapshot,
    PlayerObs,
    ScreenType,
    StageRound,
    Unit,
)

EM_DASHES = ("\u2014", "\u2015")


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def players(*hp_by_name: tuple[str, int | None], me: str = "Me") -> list[PlayerObs]:
    return [PlayerObs(name=me, hp=70, is_self=True)] + [PlayerObs(name=n, hp=hp) for n, hp in hp_by_name]


def snap(player: str, stage: str, *names: str, captured_at: float = 0.0) -> OpponentSnapshot:
    board = [Unit(api_name=f"TFT99_{n}", name=n, cost=3) for n in names]
    return OpponentSnapshot(player=player, stage=stage, board=board, captured_at=captured_at)


def state_at(stage: str, plist, opponents=None, **kw) -> GameState:
    return GameState(
        stage=StageRound.parse(stage),
        players=plist,
        self_name="Me",
        opponents=opponents or {},
        screen_type=kw.pop("screen_type", ScreenType.PLANNING),
        **kw,
    )


P7 = players(("A", 100), ("B", 90), ("C", 80), ("D", 70), ("E", 60), ("F", 50), ("G", 40))


def targets(reqs) -> list[str]:
    return [r.target_player for r in reqs]


def test_round_index_is_monotonic():
    order = ["1-1", "1-4", "2-1", "2-7", "3-1", "3-7", "4-1", "5-2"]
    idx = [round_index(StageRound.parse(s)) for s in order]
    assert idx == sorted(idx) and len(set(idx)) == len(idx)
    assert round_index(StageRound.parse("3-1")) - round_index(StageRound.parse("2-1")) == 7


@pytest.mark.parametrize("stage", ["1-1", "1-3", "2-1", "2-4", "3-4", "4-4"])
def test_no_requests_in_stage_one_early_stage_two_or_carousel(stage):
    planner = ScoutPlanner(clock=Clock())
    assert planner.plan(state_at(stage, P7), Analysis()) == []
    assert planner.open_requests() == []


@pytest.mark.parametrize("screen", [ScreenType.CAROUSEL, ScreenType.AUGMENT_SELECT, ScreenType.LOADING])
def test_no_new_requests_on_busy_screens(screen):
    planner = ScoutPlanner(clock=Clock())
    assert planner.plan(state_at("3-2", P7, screen_type=screen), Analysis()) == []


def test_combat_screen_still_allowed():
    planner = ScoutPlanner(clock=Clock())
    assert len(planner.plan(state_at("3-2", P7, screen_type=ScreenType.COMBAT), Analysis())) == 2


def test_per_stage_limit_and_text():
    planner = ScoutPlanner(max_per_stage=2, hotkey="F9", clock=Clock())
    reqs = planner.plan(state_at("2-5", P7), Analysis())
    # All boards unknown: highest HP first.
    assert targets(reqs) == ["A", "B"]
    r = reqs[0]
    assert r.text == REQUEST_TEXT.format(name="A", hotkey="F9")
    assert r.text == "请点开右侧玩家列表里「A」的棋盘，然后按 F9 记录（看完点自己头像回来）"
    assert r.stage == "2-5" and r.created_at == 1000.0 and r.reason
    assert r.id == request_id(2, "A")
    for req in reqs:
        for dash in EM_DASHES:
            assert dash not in req.text and dash not in req.reason
    # Same stage, later round: quota exhausted, nothing new.
    again = planner.plan(state_at("2-6", P7), Analysis())
    assert targets(again) == ["A", "B"]
    # Next stage: the stage-2 requests expired (3 rounds), fresh quota and ids.
    nxt = planner.plan(state_at("3-1", P7), Analysis())
    assert targets(nxt) == ["A", "B"]
    assert [r.id for r in nxt] == [request_id(3, "A"), request_id(3, "B")]


def test_zero_quota():
    planner = ScoutPlanner(max_per_stage=0, clock=Clock())
    assert planner.plan(state_at("3-2", P7), Analysis()) == []


def test_ids_are_stable_per_stage_and_player():
    a = ScoutPlanner(clock=Clock()).plan(state_at("3-2", P7), Analysis())
    b = ScoutPlanner(clock=Clock()).plan(state_at("3-3", P7), Analysis())
    assert [r.id for r in a] == [r.id for r in b]
    assert request_id(3, "A") != request_id(4, "A")
    assert request_id(3, "A") != request_id(3, "B")
    assert request_id(3, "玩家一") == request_id(3, "玩家一")


def test_contest_priority_beats_hp():
    opponents = {
        "F": snap("F", "2-5", "Draven", "Aatrox"),  # low HP but holds our carry and a comp unit
        "E": snap("E", "2-5", "Shen"),
    }
    comp = CompSuggestion(name="Blademaster", score=0.8, core_units=["Draven", "Aatrox", "Shen"], carry="Draven")
    planner = ScoutPlanner(clock=Clock())
    reqs = planner.plan(state_at("3-5", P7, opponents), Analysis(comps=[comp]))
    assert targets(reqs) == ["F", "E"]
    assert "Draven" in reqs[0].reason and "抢" in reqs[0].reason


def test_contested_by_flag_counts_as_contest():
    comp = CompSuggestion(name="X", score=0.5, contested_by=["G"])
    reqs = ScoutPlanner(clock=Clock()).plan(state_at("3-2", P7), Analysis(comps=[comp]))
    assert targets(reqs)[0] == "G"


def test_unknown_board_preferred_over_stale_at_similar_hp():
    plist = players(("A", 100), ("B", 100), ("C", 95))
    opponents = {"A": snap("A", "2-5"), "B": snap("B", "2-5")}  # stale by 3-5 (7 rounds)
    reqs = ScoutPlanner(max_per_stage=1, clock=Clock()).plan(state_at("3-5", plist, opponents), Analysis())
    assert targets(reqs) == ["C"]
    assert "还没记录" in reqs[0].reason


def test_skips_self_dead_fresh_and_already_scouted_this_stage():
    plist = [
        PlayerObs(name="Me", hp=70, is_self=False),  # self detected by self_name
        PlayerObs(name="Dead", hp=0),
        PlayerObs(name="Fresh", hp=100),
        PlayerObs(name="ThisStage", hp=99),
        PlayerObs(name="Old", hp=20),
    ]
    opponents = {
        "Fresh": snap("Fresh", "3-7"),  # 1 round old at 4-1: fresh
        "ThisStage": snap("ThisStage", "4-1"),
        "Old": snap("Old", "3-1"),  # 7 rounds old: stale
    }
    reqs = ScoutPlanner(clock=Clock()).plan(state_at("4-1", plist, opponents), Analysis())
    assert targets(reqs) == ["Old"]
    assert "3-1" in reqs[0].reason


def test_snapshot_without_stage_uses_clock():
    clock = Clock()
    plist = players(("A", 100))
    planner = ScoutPlanner(clock=clock, stale_seconds=300)
    assert planner.plan(state_at("3-2", plist, {"A": OpponentSnapshot(player="A", captured_at=900.0)}), Analysis()) == []
    clock.t = 2000.0
    planner = ScoutPlanner(clock=clock, stale_seconds=300)
    assert targets(planner.plan(state_at("3-2", plist, {"A": OpponentSnapshot(player="A", captured_at=900.0)}), Analysis())) == ["A"]


def test_expires_after_two_rounds_and_not_rerequested_same_stage():
    planner = ScoutPlanner(clock=Clock())
    assert targets(planner.plan(state_at("3-1", P7), Analysis())) == ["A", "B"]
    assert targets(planner.plan(state_at("3-2", P7), Analysis())) == ["A", "B"]
    assert planner.plan(state_at("3-3", P7), Analysis()) == []  # expired, quota used
    assert planner.open_requests() == []


def test_expires_when_player_scouted():
    planner = ScoutPlanner(clock=Clock())
    planner.plan(state_at("3-1", P7), Analysis())
    scouted = {"A": snap("A", "3-1", "Garen")}
    left = planner.plan(state_at("3-1", P7, scouted), Analysis())
    assert targets(left) == ["B"]
    # A scouted this stage: never asked again in stage 3.
    assert "A" not in targets(planner.plan(state_at("3-2", P7, scouted), Analysis()))


def test_expires_when_player_dies():
    planner = ScoutPlanner(clock=Clock())
    planner.plan(state_at("3-1", P7), Analysis())
    dead = players(("A", 0), ("B", 90), ("C", 80))
    assert targets(planner.plan(state_at("3-1", dead), Analysis())) == ["B"]


def test_dismiss_and_mark_scouted():
    planner = ScoutPlanner(clock=Clock())
    reqs = planner.plan(state_at("3-1", P7), Analysis())
    assert planner.dismiss(reqs[0].id) is True
    assert planner.dismiss(reqs[0].id) is False
    assert targets(planner.open_requests()) == ["B"]
    # Dismissed request counts toward the quota and is not re-created.
    assert targets(planner.plan(state_at("3-1", P7), Analysis())) == ["B"]
    planner.mark_scouted("b")
    assert planner.open_requests() == []


def test_new_game_resets():
    planner = ScoutPlanner(clock=Clock())
    planner.plan(state_at("4-1", P7, game_id="g1"), Analysis())
    assert planner.open_requests()
    fresh = planner.plan(state_at("2-5", P7, game_id="g2"), Analysis())
    assert targets(fresh) == ["A", "B"]
    assert all(r.stage == "2-5" for r in fresh)


def test_open_requests_are_copies():
    planner = ScoutPlanner(clock=Clock())
    planner.plan(state_at("3-1", P7), Analysis())
    reqs = planner.open_requests()
    reqs[0].text = "changed"
    assert planner.open_requests()[0].text != "changed"


def test_large_hp_gap_beats_unknown_bonus():
    plist = players(("Strong", 100), ("Weak", 40))
    opponents = {"Strong": snap("Strong", "2-5")}  # stale at 3-5, Weak never recorded
    reqs = ScoutPlanner(max_per_stage=1, clock=Clock()).plan(state_at("3-5", plist, opponents), Analysis())
    assert targets(reqs) == ["Strong"]


def test_ocr_name_noise_does_not_expire_request():
    planner = ScoutPlanner(clock=Clock())
    planner.plan(state_at("3-1", players(("Alexander", 90), ("Bob", 80))), Analysis())
    noisy = players(("A1exander", 90), ("Bob", 80))
    assert targets(planner.plan(state_at("3-1", noisy), Analysis())) == ["Alexander", "Bob"]
