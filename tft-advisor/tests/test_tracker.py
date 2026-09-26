from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from tft_advisor.engine.tracker import UNKNOWN_PLAYER, GameTracker
from tft_advisor.models import (
    Observation,
    PlayerObs,
    ScreenObservation,
    ScreenType,
    ShopSlot,
    TraitObs,
)

from .conftest import FakeClock, obs, players, uo

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def tracker(set_data, mech, clock):
    return GameTracker(set_data, mech, clock=clock)


def own_frame(**kw):
    kw.setdefault("viewing_own_board", True)
    return obs(**kw)


# ---------------------------------------------------------------------------
# Basic merge rules
# ---------------------------------------------------------------------------


def test_ingest_resolves_units_items_and_shop(tracker):
    st = tracker.ingest(
        own_frame(
            stage="2-1",
            gold=12,
            level=4,
            hp=90,
            board=[uo("graves", 2, ["infinity edge", "Mystery Blade"], row=3, col=1), uo("Braum", row=0, col=3)],
            bench=[uo("Miss Fortun"), uo("Zzzzzz")],
            shop=[ShopSlot(name="Lucian", cost=2), ShopSlot(name=None), ShopSlot(name="Garen", cost=1)],
            item_bench=["b.f. sword", "Weird Thing"],
        )
    )
    g = st.board[0]
    assert (g.api_name, g.name, g.cost, g.star, g.row, g.col) == ("TFT99_Graves", "Graves", 1, 2, 3, 1)
    assert g.items == ["Infinity Edge", "Mystery Blade"]  # unresolved item keeps raw text
    assert g.traits == ["Gunslinger", "Pirate"]
    assert st.bench[0].api_name == "TFT99_MissFortune"  # fuzzy
    assert st.bench[1].api_name == "?Zzzzzz" and st.bench[1].cost is None and st.bench[1].name == "Zzzzzz"
    assert [s.name for s in st.shop] == ["Lucian", None, "Garen"]
    assert st.shop_units[0].api_name == "TFT99_Lucian" and st.shop_units[1] is None
    assert st.item_bench == ["B.F. Sword", "Weird Thing"]
    assert st.gold == 12 and st.level == 4 and st.hp == 90
    assert str(st.stage) == "2-1"
    for f in ("stage", "gold", "level", "hp", "board", "bench", "shop", "item_bench"):
        assert st.field_age[f] == 1000.0


def test_shop_cost_disambiguates_fuzzy_name(tracker):
    # "Ashee" fuzzy-resolves to Ashe (3 cost); a misread cost must not override an exact name.
    st = tracker.ingest(own_frame(shop=[ShopSlot(name="Ashee", cost=3), ShopSlot(name="Graves", cost=4)]))
    assert st.shop_units[0].api_name == "TFT99_Ashe"
    assert st.shop_units[1].api_name == "TFT99_Graves"
    assert st.shop[1].cost == 4  # raw slot kept as read


def test_none_keeps_old_values_and_empty_list_clears(tracker, clock):
    tracker.ingest(own_frame(stage="3-1", gold=40, level=6, hp=70, streak=3, board=["Graves"], bench=["Ahri"], item_bench=["B.F. Sword"]))
    clock.tick(5)
    st = tracker.ingest(own_frame(gold=None, level=None, hp=None, board=None, bench=[], item_bench=None))
    assert st.gold == 40 and st.level == 6 and st.hp == 70 and st.streak == 3
    assert [u.name for u in st.board] == ["Graves"]
    assert st.bench == []
    assert st.item_bench == ["B.F. Sword"]
    assert st.field_age["gold"] == 1000.0
    assert st.field_age["bench"] == 1005.0
    assert st.last_update == 1005.0


@pytest.mark.parametrize(
    "field,bad",
    [("gold", 999), ("gold", -3), ("level", 0), ("level", 11), ("hp", 250), ("hp", -1), ("streak", 40), ("xp_current", -5)],
)
def test_implausible_values_rejected(tracker, field, bad):
    tracker.ingest(own_frame(stage="3-1", gold=30, level=6, hp=60, streak=2, xp_current=4))
    before = tracker.state
    st = tracker.ingest(own_frame(**{field: bad}))
    assert getattr(st, field) == getattr(before, field)


def test_star_and_positions_are_sanitized(tracker):
    st = tracker.ingest(own_frame(board=[uo("Graves", star=7, row=9, col=-1)], bench=[uo("Ahri", star=0, row=2, col=4)]))
    assert st.board[0].star == 4 and st.board[0].row is None and st.board[0].col is None
    assert st.bench[0].star == 1 and st.bench[0].row is None and st.bench[0].col == 4


def test_positions_kept_when_missing(tracker):
    tracker.ingest(own_frame(board=[uo("Graves", row=3, col=2), uo("Braum", row=0, col=3)]))
    st = tracker.ingest(own_frame(board=[uo("Graves"), uo("Braum", row=1, col=1)]))
    assert (st.board[0].row, st.board[0].col) == (3, 2)
    assert (st.board[1].row, st.board[1].col) == (1, 1)


def test_bench_truncated_to_bench_size(tracker, mech):
    st = tracker.ingest(own_frame(bench=["Graves"] * (mech.bench_size + 3)))
    assert len(st.bench) == mech.bench_size


def test_state_is_a_copy(tracker):
    st = tracker.ingest(own_frame(gold=10, board=["Graves"]))
    st.gold = 99
    st.board[0].star = 3
    st.board.clear()
    assert tracker.state.gold == 10
    assert tracker.state.board[0].star == 1
    s2 = tracker.state
    s2.hp = 1
    assert tracker.state.hp is None


def test_traits_names_resolved(tracker):
    st = tracker.ingest(own_frame(traits=[TraitObs(name="gunslinger", count=2, active=True), TraitObs(name="Mystery", count=1)]))
    assert [t.name for t in st.traits] == ["Gunslinger", "Mystery"]
    assert st.traits[0].active


def test_augment_choices_cleared_on_new_round(tracker):
    st = tracker.ingest(obs(screen_type=ScreenType.AUGMENT_SELECT, stage="2-1", augment_choices=["A", "B", "C"]))
    assert st.augment_choices == ["A", "B", "C"]
    st = tracker.ingest(own_frame(stage="2-1"))
    assert st.augment_choices == ["A", "B", "C"]
    st = tracker.ingest(own_frame(stage="2-2", augments=["B"]))
    assert st.augment_choices == [] and st.augments == ["B"]


# ---------------------------------------------------------------------------
# Stage handling, history, new game
# ---------------------------------------------------------------------------


def test_history_appended_on_stage_change(tracker, clock):
    tracker.ingest(own_frame(stage="2-1", gold=5, hp=100, level=4))
    clock.tick()
    tracker.ingest(own_frame(stage="2-1", gold=9))  # same round: no new point
    clock.tick()
    st = tracker.ingest(own_frame(stage="2-2", gold=11))
    assert [h.stage for h in st.history] == ["2-1", "2-2"]
    assert st.history[0].gold == 5 and st.history[1].gold == 11
    assert st.history[1].hp == 100 and st.history[1].level == 4
    assert st.history[1].captured_at == 1002.0


def test_history_fills_missing_values_same_round(tracker):
    tracker.ingest(own_frame(stage="3-1"))
    st = tracker.ingest(own_frame(gold=33, hp=55))
    assert st.history[-1].gold == 33 and st.history[-1].hp == 55


def test_lower_stage_once_is_ignored_twice_is_a_correction(tracker):
    tracker.ingest(own_frame(stage="3-1", gold=10))
    tracker.ingest(own_frame(stage="4-1"))  # misread (adjacent stage: accepted at first)
    st = tracker.ingest(own_frame(stage="3-2", gold=20))
    assert str(st.stage) == "4-1"
    assert st.gold == 20  # the rest of the frame still merges
    st = tracker.ingest(own_frame(stage="3-2"))
    assert str(st.stage) == "3-2"
    assert [h.stage for h in st.history] == ["3-1", "3-2"]


def test_stage_jump_over_a_whole_stage_needs_confirmation(tracker):
    tracker.ingest(own_frame(stage="3-1", gold=10))
    st = tracker.ingest(own_frame(stage="8-1", gold=12))  # "3" read as "8"
    assert str(st.stage) == "3-1" and st.gold == 12
    st = tracker.ingest(own_frame(stage="3-2"))
    assert str(st.stage) == "3-2" and [h.stage for h in st.history] == ["3-1", "3-2"]
    # A real jump (advisor paused for a while) is accepted on the second reading.
    tracker.ingest(own_frame(stage="5-1"))
    st = tracker.ingest(own_frame(stage="5-2"))
    assert str(st.stage) == "5-2"


def test_new_game_after_post_game_even_without_stage_one(tracker):
    tracker.ingest(own_frame(stage="5-3", level=8, hp=30, board=["Kayle"], players=players(("Me", 30), ("A", 20), me="Me")))
    old_id = tracker.state.game_id
    tracker.ingest(obs(screen_type=ScreenType.POST_GAME))
    st = tracker.ingest(own_frame(stage="2-1", level=4, hp=100, players=players(("Me", 100), ("B", 100), me="Me")))
    assert str(st.stage) == "2-1" and st.game_id != old_id
    assert st.board == [] and [p.name for p in st.players] == ["Me", "B"]


def test_misclassified_post_game_does_not_reset_on_a_misread(tracker):
    tracker.ingest(own_frame(stage="4-2", board=["Graves"]))
    tracker.ingest(obs(screen_type=ScreenType.POST_GAME))
    tracker.ingest(own_frame(stage="4-2"))  # the game goes on
    st = tracker.ingest(own_frame(stage="3-2"))  # one misread
    assert str(st.stage) == "4-2" and st.board


def test_stage_one_after_stage_two_is_a_new_game(tracker, clock):
    tracker.ingest(own_frame(stage="4-3", gold=50, level=8, hp=20, board=["Graves"], players=players(("Me", 20), ("A", 50), me="Me")))
    tracker.ingest(obs(purpose="scout", viewing_own_board=False, viewed_player_name="A", board=["Ahri"]))
    old_id = tracker.state.game_id
    clock.tick(600)
    st = tracker.ingest(own_frame(stage="1-2", gold=1))
    assert str(st.stage) == "1-2"
    assert st.game_id != old_id
    assert st.board == [] and st.level is None and st.hp is None and st.opponents == {} and st.players == []
    assert st.gold == 1
    assert [h.stage for h in st.history] == ["1-2"]
    assert tracker.taken_copies() == {}


def test_stage_1_lower_within_stage_1_is_misread(tracker):
    tracker.ingest(own_frame(stage="1-4"))
    st = tracker.ingest(own_frame(stage="1-2"))
    assert str(st.stage) == "1-4"


def test_post_game_keeps_state(tracker):
    tracker.ingest(own_frame(stage="5-3", gold=12, board=["Graves"]))
    st = tracker.ingest(obs(screen_type=ScreenType.POST_GAME, gold=0, board=[]))
    assert st.screen_type == ScreenType.POST_GAME
    assert st.gold == 12 and len(st.board) == 1


def test_reset_with_game_id(tracker):
    tracker.ingest(own_frame(stage="3-1", gold=10))
    st = tracker.reset("g42")
    assert st.game_id == "g42" and st.gold is None and st.stage is None


# ---------------------------------------------------------------------------
# Players and scouting
# ---------------------------------------------------------------------------


def test_players_self_detection_and_hp(tracker):
    st = tracker.ingest(own_frame(players=players(("Me", 64), ("Alice", 80), ("Bob", 12), me="Me")))
    assert st.self_name == "Me" and st.hp == 64
    assert [p.is_self for p in st.players] == [True, False, False]
    # A later list without is_self still knows who we are; a garbled HP keeps the old one.
    st = tracker.ingest(obs(players=[PlayerObs(name="Alice", hp=78), PlayerObs(name="Me", hp=None), PlayerObs(name="Bob", hp=999)]))
    me = next(p for p in st.players if p.name == "Me")
    assert me.is_self and me.hp == 64
    assert next(p for p in st.players if p.name == "Bob").hp == 12


def test_player_names_fuzzy_but_distinct(tracker):
    tracker.ingest(obs(players=players(("Player1", 50), ("Faker", 70))))
    st = tracker.ingest(obs(players=players(("Player1", 45), ("Player2", 60), ("Fakerr", 66))))
    names = sorted(p.name for p in st.players)
    assert names == ["Faker", "Player1", "Player2"]
    assert next(p for p in st.players if p.name == "Faker").hp == 66


def test_scout_goes_to_opponent_snapshot(tracker, clock):
    tracker.ingest(
        own_frame(
            stage="3-2", gold=30, level=6, hp=70, xp_current=4,
            board=["Graves", "Braum"], bench=["Ahri"], item_bench=["B.F. Sword"],
            traits=[TraitObs(name="Gunslinger", count=1)],
            players=players(("Me", 70), ("Alice", 55), me="Me"),
        )
    )
    clock.tick(3)
    st = tracker.ingest(
        obs(
            purpose="scout",
            viewing_own_board=False,
            viewed_player_name="alice",
            stage="3-2",
            gold=31,
            level=8,
            hp=55,
            xp_current=50,
            board=[uo("Miss Fortune", 2), "Draven"],
            bench=["Garen"],
            item_bench=["Recurve Bow"],
            traits=[TraitObs(name="Pirate", count=1)],
            shop=["Lucian", "Pyke"],
        )
    )
    # Own state: only gold and shop changed.
    assert st.gold == 31 and st.level == 6 and st.hp == 70 and st.xp_current == 4
    assert [u.name for u in st.board] == ["Graves", "Braum"] and [u.name for u in st.bench] == ["Ahri"]
    assert st.item_bench == ["B.F. Sword"] and st.traits[0].name == "Gunslinger"
    assert [s.name for s in st.shop] == ["Lucian", "Pyke"]
    snap = st.opponents["Alice"]  # canonical name from the player list
    assert snap.level == 8 and snap.hp == 55 and snap.stage == "3-2" and snap.captured_at == 1003.0
    assert [u.api_name for u in snap.board] == ["TFT99_MissFortune", "TFT99_Draven"]
    assert snap.items == ["Recurve Bow"] and snap.traits[0].name == "Pirate"
    assert tracker.taken_copies() == {"TFT99_MissFortune": 3, "TFT99_Draven": 1, "TFT99_Garen": 1}
    assert tracker.taken_by_player() == {"Alice": {"TFT99_MissFortune": 3, "TFT99_Draven": 1, "TFT99_Garen": 1}}


def test_viewing_own_board_false_without_scout_purpose(tracker):
    tracker.ingest(own_frame(board=["Graves"], players=players(("Me", 70), ("Bob", 40), me="Me")))
    st = tracker.ingest(obs(viewing_own_board=False, viewed_player_name="Bob", board=["Ahri"]))
    assert [u.name for u in st.board] == ["Graves"]
    assert "Bob" in st.opponents


def test_scout_snapshot_merges_none_fields(tracker):
    tracker.ingest(obs(purpose="scout", viewed_player_name="Bob", board=["Ahri"], bench=["Garen"], level=7))
    st = tracker.ingest(obs(purpose="scout", viewed_player_name="Bob", board=None, bench=["Graves"], level=None))
    snap = st.opponents["Bob"]
    assert [u.name for u in snap.board] == ["Ahri"] and [u.name for u in snap.bench] == ["Graves"] and snap.level == 7


def test_scout_fallback_by_hp_or_unknown(tracker):
    tracker.ingest(own_frame(players=players(("Me", 70), ("Alice", 55), ("Bob", 31), me="Me")))
    st = tracker.ingest(obs(purpose="scout", viewing_own_board=False, hp=31, board=["Ahri"]))
    assert "Bob" in st.opponents
    st = tracker.ingest(obs(purpose="scout", viewing_own_board=False, hp=None, board=["Pyke"]))
    assert UNKNOWN_PLAYER in st.opponents


def test_scout_of_own_board_is_not_an_opponent(tracker):
    tracker.ingest(own_frame(board=["Graves", uo("Braum", 2)], players=players(("Me", 70), ("Bob", 50), me="Me")))
    st = tracker.ingest(obs(purpose="scout", viewed_player_name="Me", board=["Graves", uo("Braum", 2)], level=5))
    assert st.opponents == {} and st.level == 5
    st = tracker.ingest(obs(purpose="scout", board=["Graves", uo("Braum", 2)]))
    assert st.opponents == {}


def test_named_opponent_with_board_identical_to_ours(tracker):
    # Early boards often match by chance: the banner name must win.
    tracker.ingest(own_frame(stage="2-1", level=3, hp=100, board=["Garen"], item_bench=["B.F. Sword"], players=players(("Me", 100), ("Alice", 100), me="Me")))
    st = tracker.ingest(obs(purpose="scout", viewing_own_board=False, viewed_player_name="Alice", level=4, board=["Garen"], bench=["Graves"], item_bench=["Chain Vest"]))
    assert st.level == 3 and st.bench == [] and st.item_bench == ["B.F. Sword"]
    assert [u.name for u in st.opponents["Alice"].bench] == ["Graves"]


def test_scout_key_pressed_on_own_board(tracker):
    # The vision says "own board" explicitly (it defaults scout frames to False).
    tracker.ingest(own_frame(stage="3-1", board=["Graves", "Braum"], players=players(("Me", 70), ("A", 60), me="Me")))
    st = tracker.ingest(obs(purpose="scout", viewing_own_board=True, board=["Graves", "Braum", "Ahri"], level=6))
    assert st.opponents == {} and [u.name for u in st.board] == ["Graves", "Braum", "Ahri"] and st.level == 6
    assert tracker.taken_copies() == {}


def test_players_list_sanity(tracker):
    st = tracker.ingest(obs(players=[PlayerObs(name=f"Player{i}_{i * 7}", hp=50) for i in range(12)]))
    assert len(st.players) == 8
    t2 = GameTracker(tracker.set_data, tracker._mech, clock=FakeClock())
    t2.ingest(obs(players=[PlayerObs(name="★★", hp=50), PlayerObs(name="Bob", hp=30)]))
    st = t2.ingest(obs(players=[PlayerObs(name="!!", hp=10), PlayerObs(name="Bob", hp=30)]))
    assert next(p for p in st.players if p.name == "★★").hp == 50  # symbol-only names do not merge
    # A row flagged is_self by mistake does not steal the known self.
    t2.ingest(obs(players=players(("Me", 60), ("Bob", 30), me="Me")))
    st = t2.ingest(obs(players=[PlayerObs(name="Bob", hp=30, is_self=True), PlayerObs(name="Me", hp=55)]))
    assert st.self_name == "Me" and st.hp == 55


def test_eliminated_player_copies_ignored(tracker):
    tracker.ingest(own_frame(players=players(("Me", 70), ("Alice", 40), ("Bob", 20), me="Me")))
    tracker.ingest(obs(purpose="scout", viewed_player_name="Alice", board=[uo("Graves", 2)]))
    tracker.ingest(obs(purpose="scout", viewed_player_name="Bob", board=[uo("Graves", 1), "Ahri"]))
    assert tracker.taken_copies() == {"TFT99_Graves": 4, "TFT99_Ahri": 1}
    tracker.ingest(obs(players=players(("Me", 60), ("Alice", 30), ("Bob", 0))))
    assert tracker.taken_copies() == {"TFT99_Graves": 3}
    assert set(tracker.taken_by_player()) == {"Alice"}


def test_unresolved_units_not_counted(tracker):
    tracker.ingest(obs(purpose="scout", viewed_player_name="Bob", board=["Qwertyuiop", "Graves"]))
    assert tracker.taken_copies() == {"TFT99_Graves": 1}


def test_self_name_learned_from_own_board_banner(tracker):
    tracker.ingest(obs(players=players(("Me", 70), ("Bob", 50))))
    st = tracker.ingest(obs(viewing_own_board=True, viewed_player_name="Me", gold=3))
    assert st.self_name == "Me"
    assert [p.is_self for p in st.players] == [True, False]


# ---------------------------------------------------------------------------
# Manual corrections
# ---------------------------------------------------------------------------


def test_set_field_valid(tracker, clock):
    tracker.ingest(own_frame(stage="3-1", gold=10, players=players(("Me", 50), me="Me")))
    clock.tick(2)
    st = tracker.set_field("gold", "42")
    assert st.gold == 42 and st.field_age["gold"] == 1002.0
    assert tracker.set_field("level", 7).level == 7
    st = tracker.set_field("hp", 33)
    assert st.hp == 33 and st.players[0].hp == 33
    assert tracker.set_field("streak", -4).streak == -4
    assert tracker.set_field("xp_current", 12).xp_current == 12
    st = tracker.set_field("stage", "3-2")
    assert str(st.stage) == "3-2" and st.history[-1].stage == "3-2"
    st = tracker.set_field("stage", "2-7")  # manual correction may go back
    assert str(st.stage) == "2-7" and all(h.stage != "3-2" for h in st.history)
    assert tracker.set_field("gold", None).gold is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("gold", 301), ("gold", "abc"), ("level", 0), ("level", 11), ("hp", 201), ("streak", 21), ("stage", "x-y"),
        ("stage", "10-1"), ("board", []), ("gold", True), ("gold", "inf"), ("gold", float("inf")), ("hp", "1e400"),
        ("level", float("nan")),
    ],
)
def test_set_field_invalid(tracker, field, value):
    with pytest.raises(ValueError):
        tracker.set_field(field, value)


# ---------------------------------------------------------------------------
# Thread safety and replay
# ---------------------------------------------------------------------------


def test_thread_safety_smoke(set_data, mech):
    tracker = GameTracker(set_data, mech, clock=FakeClock())
    errors: list[BaseException] = []

    def writer(k: int) -> None:
        try:
            for i in range(40):
                tracker.ingest(own_frame(gold=(k * 7 + i) % 50, board=["Graves", "Ahri"][: 1 + i % 2]))
                tracker.ingest(obs(purpose="scout", viewed_player_name=f"P{k}", board=[uo("Braum", 1 + i % 2)]))
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    def reader() -> None:
        try:
            for _ in range(80):
                st = tracker.state
                assert 0 <= (st.gold or 0) < 50
                tracker.taken_copies()
                tracker.taken_by_player()
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(k,)) for k in range(4)] + [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    taken = tracker.taken_copies()
    assert set(tracker.taken_by_player()) == {"P0", "P1", "P2", "P3"}
    assert taken["TFT99_Braum"] == 4 * 3  # last frame of each writer: 2-star Braum


def test_replay_fixture_game(tracker):
    files = sorted(FIXTURES.glob("obs_*.json"))
    if not files:
        pytest.skip("no observation fixtures")
    for f in files:
        screen = ScreenObservation.model_validate(json.loads(f.read_text(encoding="utf-8")))
        purpose = "scout" if "scout" in f.name else "auto"
        st = tracker.ingest(Observation(screen=screen, purpose=purpose))
    assert st.stage is not None and st.stage.stage >= 4
    assert len(st.history) >= len(files) - 2
    assert st.board, "own board survives the scouting frame"
    if any("scout" in f.name for f in files):
        assert st.opponents
    assert all(not u.api_name.startswith("?") for u in st.board)


def test_chinese_locale_names(zh_set_data, mech, clock):
    tracker = GameTracker(zh_set_data, mech, clock=clock)
    st = tracker.ingest(
        own_frame(
            board=[uo("厄运小姐", 2, ["无尽之刃", "Giant Slayer"], row=3, col=3)],
            bench=[uo("Graves")],
            item_bench=["暴风大剑", "Sparring Gloves"],
            shop=[ShopSlot(name="卢锡安", cost=2), ShopSlot(name="Lucian", cost=2)],
            traits=[TraitObs(name="Gunslinger", count=2, active=True)],
        )
    )
    mf = st.board[0]
    assert mf.api_name == "TFT99_MissFortune" and mf.name == "厄运小姐" and mf.traits == ["枪手", "海盗"]
    assert mf.items == ["无尽之刃", "巨人杀手"]
    assert st.bench[0].name == "格雷福斯"
    assert st.item_bench == ["暴风大剑", "拳套"]
    assert [u.name for u in st.shop_units] == ["卢锡安", "卢锡安"]
    assert st.traits[0].name == "枪手"


def test_stage_one_misread_mid_game_needs_confirmation(tracker):
    tracker.ingest(own_frame(stage="4-1", gold=40, level=7, hp=55, board=["Graves"]))
    st = tracker.ingest(own_frame(stage="1-1", hp=55, level=7))  # "4-1" misread as "1-1"
    assert str(st.stage) == "4-1" and st.board and st.hp == 55
    st = tracker.ingest(own_frame(stage="1-1", hp=55))  # read twice: trust it
    assert str(st.stage) == "1-1" and st.board == []


def test_shop_only_frame_while_away_does_not_touch_snapshots(tracker, clock):
    tracker.ingest(obs(purpose="scout", viewed_player_name="Bob", board=["Ahri"]))
    clock.tick(30)
    st = tracker.ingest(obs(purpose="shop", viewing_own_board=False, viewed_player_name="Bob", shop=["Graves"], gold=7))
    assert st.opponents["Bob"].captured_at == 1000.0
    st = tracker.ingest(obs(purpose="shop", viewing_own_board=False, viewed_player_name="Eve", shop=["Graves"]))
    assert "Eve" not in st.opponents and st.gold == 7


def test_game_ids_unique_with_frozen_clock(tracker):
    ids = {tracker.reset().game_id for _ in range(3)}
    assert len(ids) == 3


def test_carousel_units_are_not_our_board(tracker):
    tracker.ingest(own_frame(stage="3-3", board=["Graves", "Braum"], bench=["Ahri"], item_bench=["B.F. Sword"]))
    st = tracker.ingest(obs(screen_type=ScreenType.CAROUSEL, stage="3-4", gold=40, board=["Kayle", "Akali", "Draven"], bench=[], item_bench=[]))
    assert [u.name for u in st.board] == ["Graves", "Braum"] and [u.name for u in st.bench] == ["Ahri"]
    assert st.item_bench == ["B.F. Sword"] and st.gold == 40 and str(st.stage) == "3-4"
    st = tracker.ingest(obs(purpose="scout", screen_type=ScreenType.CAROUSEL, viewed_player_name="Bob", board=["Kayle"]))
    assert st.opponents == {}


def test_unknown_snapshot_not_counted_twice(tracker):
    tracker.ingest(own_frame(stage="3-2", board=["Graves", "Braum", "Pyke"], players=players(("Me", 70), ("Bob", 50), ("Eve", 50), me="Me")))
    # Nameless scout of Bob (HP ambiguous), then the same board with his name.
    tracker.ingest(obs(purpose="scout", viewing_own_board=False, hp=50, board=[uo("Ahri", 2), "Morgana", "Warwick"]))
    assert tracker.taken_copies() == {"TFT99_Ahri": 3, "TFT99_Morgana": 1, "TFT99_Warwick": 1}
    tracker.ingest(obs(purpose="scout", viewing_own_board=False, viewed_player_name="Bob", board=[uo("Ahri", 2), "Morgana", "Warwick", "Gnar"]))
    assert tracker.taken_copies() == {"TFT99_Ahri": 3, "TFT99_Morgana": 1, "TFT99_Warwick": 1, "TFT99_Gnar": 1}
    assert set(tracker.taken_by_player()) == {"Bob"}
    # A nameless snapshot that is really our own board (vision said "not own") is dropped too.
    tracker.ingest(obs(purpose="scout", viewing_own_board=False, board=["Graves", "Braum", "Pyke", "Lucian"]))
    assert set(tracker.taken_by_player()) == {"Bob"}
    # A genuinely different nameless board still counts.
    tracker.ingest(obs(purpose="scout", viewing_own_board=False, board=["Kayle", "Draven"]))
    assert tracker.taken_by_player()[UNKNOWN_PLAYER] == {"TFT99_Kayle": 1, "TFT99_Draven": 1}
