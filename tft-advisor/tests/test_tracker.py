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
    # One 1-x reading is not enough (a digit misread early in stage 2 would
    # wipe the game); the next consistent reading confirms the new game.
    st = tracker.ingest(own_frame(stage="1-2", gold=1))
    assert str(st.stage) == "4-3" and st.game_id == old_id
    st = tracker.ingest(own_frame(stage="1-3", gold=3))
    assert str(st.stage) == "1-3"
    assert st.game_id != old_id
    assert st.board == [] and st.level is None and st.hp is None and st.opponents == {} and st.players == []
    assert st.gold == 3
    assert [h.stage for h in st.history] == ["1-3"]
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
            viewed_player_level=7,
            stage="3-2",
            gold=31,
            level=8,  # the local player's HUD level: neither ours to take nor Alice's
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
    assert snap.level == 7 and snap.hp == 55 and snap.stage == "3-2" and snap.captured_at == 1003.0
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
    tracker.ingest(obs(purpose="scout", viewed_player_name="Bob", board=["Ahri"], bench=["Garen"], viewed_player_level=7))
    st = tracker.ingest(obs(purpose="scout", viewed_player_name="Bob", board=None, bench=["Graves"], viewed_player_level=None))
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


# ---------------------------------------------------------------------------
# 4-star units and the vision wire format
# ---------------------------------------------------------------------------


def test_four_star_unit_holds_nine_copies(set_data, mech):
    from tft_advisor.engine.probability import owned_copies, pool_remaining
    from tft_advisor.models import Unit

    assert [Unit(api_name="x", name="x", star=s).copies for s in (0, 1, 2, 3, 4)] == [1, 1, 3, 9, 9]
    garen = set_data.resolve_champion("Garen")
    four = Unit(api_name=garen.api_name, name=garen.name, cost=garen.cost, star=4)
    per_unit, per_cost = pool_remaining(set_data, mech, owned_copies([four]))
    size = mech.pool_size[garen.cost]
    assert per_unit[garen.api_name] == size - 9
    assert per_cost[garen.cost] == size * len(set_data.champions_by_cost(garen.cost)) - 9


def test_tracker_counts_a_four_star_as_nine_copies(tracker):
    st = tracker.ingest(own_frame(stage="5-1", board=[uo("Garen", 4, row=0, col=3)], bench=[]))
    assert st.board[0].star == 4
    assert sum(u.copies for u in st.all_units()) == 9


def _schema_complexity(model) -> tuple[int, int]:
    """(optional parameters, union typed parameters) the API counts for a schema."""
    from anthropic import transform_schema
    from pydantic import TypeAdapter

    schema = transform_schema(TypeAdapter(model).json_schema())
    optional = unions = 0

    def walk(node) -> None:
        nonlocal optional, unions
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict):
                required = set(node.get("required", []))
                for key, spec in props.items():
                    optional += key not in required
                    if isinstance(spec, dict) and ("anyOf" in spec or isinstance(spec.get("type"), list)):
                        unions += 1
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return optional, unions


def test_structured_output_schemas_fit_the_api_limits():
    # Platform docs, structured outputs "Schema complexity limits": at most 24
    # optional parameters and 16 union typed parameters per request.
    from tft_advisor.models import ScreenObservation, ScreenObservationWire, StrategistAdvice

    for model in (ScreenObservationWire, StrategistAdvice):
        optional, unions = _schema_complexity(model)
        assert optional <= 24 and unions <= 16, (model.__name__, optional, unions)
    assert _schema_complexity(ScreenObservationWire) == (0, 0)
    # The rich ScreenObservation itself is over the limits: never send it as output_format.
    optional, unions = _schema_complexity(ScreenObservation)
    assert optional > 24 or unions > 16


def test_wire_observation_converts_back(tracker):
    from tft_advisor.models import ScreenObservationWire

    wire = ScreenObservationWire.model_validate(
        {
            "screen_type": "planning",
            "unreadable": ["hp", "players", "streak"],
            "viewing_own_board": True,
            "viewed_player_name": "",
            "viewed_player_level": 0,
            "stage": "3-2",
            "gold": 34,
            "level": 6,
            "xp_current": 4,
            "xp_needed": 36,
            "hp": 0,
            "streak": 0,
            "shop": [{"name": "Graves", "cost": 1}, {"name": "", "cost": 0}, {"name": "Wisp: Grow Up", "cost": 3}],
            "shop_locked": False,
            "board": [{"name": "Garen", "star": 2, "items": [], "row": 0, "col": 3}],
            "bench": [{"name": "Lucian", "star": 1, "items": [], "row": -1, "col": 2}],
            "item_bench": [],
            "players": [],
            "traits": [{"name": "Knight", "count": 1, "active": False, "next_breakpoint": 0}],
            "augment_choices": [],
            "augments": [],
            "notes": [],
        }
    )
    screen = wire.to_screen()
    assert screen.hp is None and screen.players is None and screen.streak is None  # unreadable -> null
    assert screen.item_bench == [] and screen.augments == []  # visible and empty stays []
    assert screen.viewed_player_name is None and screen.stage == "3-2" and screen.gold == 34
    assert screen.viewed_player_level is None  # 0 = not shown
    assert [(s.name, s.cost) for s in screen.shop] == [("Graves", 1), (None, None), ("Wisp: Grow Up", 3)]
    assert screen.board[0].row == 0 and screen.bench[0].row is None and screen.bench[0].col == 2
    assert screen.traits[0].next_breakpoint is None
    st = tracker.ingest(Observation(screen=screen, source="claude", purpose="auto"))
    assert st.gold == 34 and st.level == 6 and [u.name for u in st.board] == ["Garen"]


# ---------------------------------------------------------------------------
# Regressions (vision-wire review): whose level / board / HP a frame carries
# ---------------------------------------------------------------------------


def _wire_screen(**over):
    from tft_advisor.models import ScreenObservationWire

    base = ScreenObservation(screen_type=ScreenType.PLANNING, stage="3-3", gold=20, level=6, hp=70)
    wire = ScreenObservationWire.from_screen(base).model_dump(mode="json")
    wire.update(over)
    return ScreenObservationWire.model_validate(wire).to_screen()


def test_scouted_opponent_never_gets_our_level(tracker):
    tracker.ingest(own_frame(stage="3-2", level=6, board=["Graves"], players=players(("Me", 70), ("Opp", 50), me="Me")))
    # Wire round trip of a scout frame: HUD level 6 is ours, the plate says 8.
    screen = _wire_screen(
        viewing_own_board=False, viewed_player_name="Opp", viewed_player_level=8, level=6,
        board=[{"name": "Ahri", "star": 1, "items": [], "row": 3, "col": 0}], unreadable=[],
    )
    st = tracker.ingest(Observation(screen=screen, purpose="scout", source="claude"))
    assert st.opponents["Opp"].level == 8 and st.level == 6
    # Plate not shown: unknown, not the local level.
    screen = _wire_screen(viewing_own_board=False, viewed_player_name="Opp", viewed_player_level=0, level=6, unreadable=[])
    t2 = GameTracker(tracker.set_data, tracker._mech)
    t2.ingest(own_frame(stage="3-2", level=6, players=players(("Me", 70), ("Opp", 50), me="Me")))
    st = t2.ingest(Observation(screen=screen.model_copy(update={"board": [uo("Ahri")]}), purpose="scout"))
    assert st.opponents["Opp"].level is None
    # The older free-text note still works.
    st = t2.ingest(obs(purpose="scout", viewed_player_name="Opp", level=6, board=["Ahri"], notes=["对手等级 7"]))
    assert st.opponents["Opp"].level == 7


def test_uncertain_frame_does_not_overwrite_our_board_or_hp(tracker):
    tracker.ingest(own_frame(stage="3-2", level=6, hp=70, board=["Garen", "Graves"], players=players(("Me", 70), ("Opp", 41), me="Me")))
    st = tracker.ingest(
        obs(stage="3-3", viewing_own_board=None, viewed_player_name=None, hp=41, level=6, gold=12,
            board=["Ahri", "Shen"], players=players(("Me", 70), ("Opp", 41), me="Me"))
    )
    assert [u.name for u in st.board] == ["Garen", "Graves"] and st.hp == 70
    assert st.gold == 12 and str(st.stage) == "3-3"  # the HUD still applies
    # An uncertain frame whose board is clearly ours (one unit bought) is applied.
    tracker.ingest(own_frame(hp=70, board=["Garen", "Graves", "Braum", "Lucian"]))
    st = tracker.ingest(obs(viewing_own_board=None, hp=68, board=["Garen", "Graves", "Braum", "Lucian", "Pyke"]))
    assert len(st.board) == 5 and st.hp == 68


def test_augment_choices_cleared_by_a_scout_frame_next_round(tracker, clock):
    tracker.ingest(own_frame(stage="2-1", screen_type=ScreenType.AUGMENT_SELECT, augment_choices=["A", "B", "C"], players=players(("Me", 100), ("Opp", 100), me="Me")))
    clock.tick(70)
    st = tracker.ingest(obs(purpose="scout", viewing_own_board=False, viewed_player_name="Opp", stage="2-2", board=["Ahri"], augment_choices=[]))
    assert st.augment_choices == []


def test_false_on_carousel_still_merges_the_hud(tracker):
    tracker.ingest(own_frame(stage="3-2", level=6, xp_current=10, xp_needed=36, streak=3, hp=70, board=["Garen"]))
    st = tracker.ingest(obs(stage="3-4", screen_type=ScreenType.CAROUSEL, viewing_own_board=False, level=7, xp_current=2, xp_needed=56, streak=4))
    assert st.level == 7 and st.xp_current == 2 and st.streak == 4
    assert st.opponents == {}


def test_false_placeholder_on_our_frame_is_not_filed_under_an_opponent(tracker):
    tracker.ingest(
        own_frame(stage="3-2", hp=60, board=[uo("Garen", 2), uo("Graves", 2), "Braum"], players=players(("Me", 60), ("Opp", 60), me="Me"))
    )
    st = tracker.ingest(obs(stage="3-2", viewing_own_board=False, hp=60, board=[uo("Garen", 2), uo("Graves", 2), "Braum", "Lucian"]))
    assert "Opp" not in st.opponents and tracker.taken_by_player() == {}
    assert [u.name for u in st.board][-1] == "Lucian"
    # A real scout request still falls back to the HP match.
    st = tracker.ingest(obs(purpose="scout", viewing_own_board=False, hp=60, board=["Ahri", "Shen"]))
    assert "Opp" in st.opponents


def test_level_change_without_xp_drops_the_old_xp(tracker, mech):
    tracker.ingest(own_frame(stage="3-2", level=6, xp_current=30, xp_needed=36))
    st = tracker.ingest(own_frame(stage="3-3", level=7, xp_current=None, xp_needed=None))
    assert st.level == 7 and st.xp_current is None and st.xp_needed == mech.xp_to_level[7]
    st = tracker.ingest(own_frame(level=7, xp_current=4, xp_needed=56))
    st = tracker.set_field("level", 8)
    assert st.xp_current is None and st.xp_needed == mech.xp_to_level[8]


def test_placeholder_values_do_not_wipe_known_state(tracker):
    tracker.ingest(own_frame(stage="4-1", streak=-4, shop=["Graves", "Lucian"], board=["Garen", "Graves", "Braum", "Ahri"], bench=["Pyke"]))
    # Same round, streak placeholder 0 and a closed shop ([]): keep what we know.
    st = tracker.ingest(own_frame(stage="4-1", streak=0, shop=[]))
    assert st.streak == -4 and [s.name for s in st.shop] == ["Graves", "Lucian"]
    # The next round may change the streak; a real value still replaces a 0.
    st = tracker.ingest(own_frame(stage="4-2", streak=0))
    assert st.streak == 0
    st = tracker.ingest(own_frame(stage="4-2", streak=-5))
    assert st.streak == -5
    # Augment cards cover the arena: an empty board read there is not a sold board.
    st = tracker.ingest(own_frame(stage="4-2", screen_type=ScreenType.AUGMENT_SELECT, board=[], bench=[], augment_choices=["X", "Y", "Z"]))
    assert len(st.board) == 4 and [u.name for u in st.bench] == ["Pyke"]
    assert st.augment_choices == ["X", "Y", "Z"]


def test_wire_schema_text_is_model_facing():
    from anthropic import transform_schema
    from pydantic import TypeAdapter

    from tft_advisor.models import ScreenObservationWire

    schema = transform_schema(TypeAdapter(ScreenObservationWire).json_schema())
    root = schema.get("description", "")
    assert "comment above" not in root and "ScreenObservation" not in root and "``" not in root
    props = schema["properties"]
    for name, placeholder in (("gold", "-1"), ("xp_current", "-1"), ("hp", "-1"), ("level", "0"), ("xp_needed", "0")):
        desc = props[name].get("description", "")
        assert placeholder in desc and "unreadable" in desc, name
    assert "viewed_player_level" in props


# ---------------------------------------------------------------------------
# Vision noise: stage runs, new games, suspicious numbers, ownership
# ---------------------------------------------------------------------------

LOBBY1 = (("Me", 40), ("Kaiser", 55), ("Bolt", 30), ("Cyan", 20), ("Dora", 61), ("Echo", 12), ("Fay", 0), ("Gus", 0))
LOBBY2 = (("Me", 100), ("Anna", 100), ("Bert", 100), ("Cleo", 100), ("Dan", 100), ("Ezra", 100), ("Finn", 100), ("Gia", 100))


def test_new_game_found_without_stage_one_or_post_game(tracker):
    tracker.ingest(own_frame(stage="5-2", level=8, hp=40, board=["Graves", "Ahri"], players=players(*LOBBY1, me="Me")))
    tracker.ingest(obs(purpose="scout", viewing_own_board=False, viewed_player_name="Kaiser", board=[uo("Lucian", 3), "Ashe"]))
    tracker.ingest(own_frame(stage="5-3", hp=40))
    old_id = tracker.state.game_id
    # The window closed after elimination; the next game starts at 2-1 with a new lobby.
    st = tracker.ingest(own_frame(stage="2-1", level=4, hp=100, board=["Garen"], players=players(*LOBBY2, me="Me")))
    assert st.game_id != old_id and str(st.stage) == "2-1"
    assert st.opponents == {} and tracker.taken_copies() == {}
    assert [h.stage for h in st.history] == ["2-1"] and st.hp == 100
    st = tracker.ingest(own_frame(stage="2-2", level=4, hp=100, players=players(*LOBBY2, me="Me")))
    assert str(st.stage) == "2-2" and st.game_id != old_id


def test_consistent_lower_run_is_followed_even_without_lobby_evidence(tracker):
    # Only the stage is readable: two consecutive consistent lower readings
    # still move the stage (a correction), so it never stays frozen at 5-3.
    tracker.ingest(own_frame(stage="5-3", level=8, hp=40))
    st = tracker.ingest(own_frame(stage="2-1"))
    assert str(st.stage) == "5-3"
    st = tracker.ingest(own_frame(stage="2-2"))
    assert str(st.stage) == "2-2"


def test_forward_misread_is_corrected_by_later_consistent_readings(tracker):
    tracker.ingest(own_frame(stage="3-2", gold=36, level=6, hp=70))
    tracker.ingest(own_frame(stage="4-3"))  # misread of 3-3, one stage ahead: accepted at first
    st = tracker.ingest(own_frame(stage="3-5"))
    assert str(st.stage) == "4-3"
    st = tracker.ingest(own_frame(stage="3-6"))
    assert str(st.stage) == "3-6"
    assert [h.stage for h in st.history] == ["3-2", "3-6"]


def test_confirmed_big_jump_misread_does_not_freeze_the_stage(tracker):
    tracker.ingest(own_frame(stage="3-2"))
    tracker.ingest(own_frame(stage="8-2"))
    st = tracker.ingest(own_frame(stage="8-2"))  # the same misread twice (auto + F6)
    assert str(st.stage) == "8-2"
    tracker.ingest(own_frame(stage="3-3"))
    st = tracker.ingest(own_frame(stage="3-5"))
    assert str(st.stage) == "3-5"


def test_single_stage_one_misread_early_in_stage_two_keeps_the_game(tracker):
    lobby = players(("Me", 100), ("Kaiser", 100), ("Bolt", 100), ("Cyan", 100), me="Me")
    tracker.ingest(own_frame(stage="2-1", level=4, hp=100, board=["Garen"], players=lobby))
    tracker.ingest(obs(purpose="scout", viewing_own_board=False, viewed_player_name="Kaiser", board=["Ahri"]))
    tracker.ingest(own_frame(stage="2-2", level=4, hp=100, players=lobby))
    old_id = tracker.state.game_id
    st = tracker.ingest(own_frame(stage="1-2", level=4, hp=100, players=lobby))  # "2-2" misread
    assert st.game_id == old_id and str(st.stage) == "2-2"
    st = tracker.ingest(own_frame(stage="2-3", level=4, hp=100, players=lobby))
    assert st.game_id == old_id and "Kaiser" in st.opponents
    assert [h.stage for h in st.history] == ["2-1", "2-2", "2-3"]


def test_misclassified_post_game_then_lower_misread_keeps_the_game(tracker):
    tracker.ingest(own_frame(stage="4-1", hp=50, board=["Graves"], players=players(("Me", 50), ("Kaiser", 60), me="Me")))
    tracker.ingest(own_frame(stage="4-2", hp=50))
    tracker.ingest(obs(purpose="scout", viewing_own_board=False, viewed_player_name="Kaiser", board=["Ahri"]))
    old_id = tracker.state.game_id
    # A round result banner called post-game (the stage is still on screen)...
    tracker.ingest(obs(screen_type=ScreenType.POST_GAME, stage="4-2"))
    st = tracker.ingest(own_frame(stage="4-1"))  # ...then a round digit misread
    assert st.game_id == old_id and "Kaiser" in st.opponents and str(st.stage) == "4-2"
    # Even a plausible end screen resets only on an early stage or a new lobby.
    tracker.ingest(obs(screen_type=ScreenType.POST_GAME))
    st = tracker.ingest(own_frame(stage="4-1"))
    assert st.game_id == old_id


def test_gold_jump_needs_a_second_reading(tracker):
    tracker.ingest(own_frame(stage="4-1", gold=42, level=7, board=["Graves"]))
    st = tracker.ingest(own_frame(gold=142))  # "42" with a stray digit
    assert st.gold == 42
    st = tracker.ingest(own_frame(gold=44))
    assert st.gold == 44
    tracker.ingest(own_frame(gold=95))  # a real +51 (sold a board, augment gold)
    st = tracker.ingest(own_frame(gold=95))
    assert st.gold == 95
    # A new round brings income: +20 is fine without a second reading.
    st = tracker.ingest(own_frame(stage="4-2", gold=115))
    assert st.gold == 115


def test_level_never_goes_down_on_one_reading_and_xp_survives_a_flicker(tracker, mech):
    tracker.ingest(own_frame(stage="4-1", level=7, xp_current=20, xp_needed=56))
    st = tracker.ingest(own_frame(level=1, xp_current=20, xp_needed=56))  # "7" read as "1"
    assert st.level == 7 and st.xp_current == 20
    st = tracker.ingest(own_frame(level=8))  # one frame reads 8 with no XP
    assert st.level == 8 and st.xp_current is None
    tracker.ingest(own_frame(level=7))
    st = tracker.ingest(own_frame(level=7))  # read twice: the 8 was the misread
    assert st.level == 7 and st.xp_current == 20 and st.xp_needed == 56
    # A dashboard correction applies at once.
    st = tracker.set_field("level", 6)
    assert st.level == 6


def test_hp_digit_drop_prefers_the_player_list(tracker):
    tracker.ingest(own_frame(stage="4-1", hp=60, players=players(("Me", 60), ("Kaiser", 70), me="Me")))
    st = tracker.ingest(own_frame(stage="4-2", hp=6, players=players(("Me", 60), ("Kaiser", 70), me="Me")))
    assert st.hp == 60 and next(p for p in st.players if p.name == "Me").hp == 60
    assert st.history[-1].hp == 60
    # Only the top HP, implausible: needs a second reading.
    st = tracker.ingest(own_frame(hp=6))
    assert st.hp == 60
    st = tracker.ingest(own_frame(hp=6))
    assert st.hp == 6 and next(p for p in st.players if p.name == "Me").hp == 6
    # HP does not come back up on one reading.
    st = tracker.ingest(own_frame(hp=60))
    assert st.hp == 6


def test_big_real_loss_is_accepted_when_both_readings_agree(tracker):
    tracker.ingest(own_frame(stage="5-1", hp=70, players=players(("Me", 70), ("Kaiser", 70), me="Me")))
    st = tracker.ingest(own_frame(stage="5-2", hp=35, players=players(("Me", 35), ("Kaiser", 70), me="Me")))
    assert st.hp == 35


def test_scout_frame_labeled_own_does_not_replace_our_board(tracker):
    lobby = players(("Me", 62), ("Kaiser", 48), ("Bolt", 30), me="Me")
    tracker.ingest(own_frame(
        stage="4-1", hp=62, level=7, board=[uo("Ahri", 2), uo("Morgana", 2), "Braum", "Shen", "Garen", "Pyke"],
        bench=["Kayle"], item_bench=["B.F. Sword"], players=lobby,
    ))
    for purpose in ("scout", "manual"):
        st = tracker.ingest(obs(
            purpose=purpose, stage="4-1", viewing_own_board=True, hp=48, level=7,
            board=[uo("Lucian", 2), uo("Ashe", 2), "Vayne", uo("Darius", 2), "Draven"], bench=["Ashe"],
            item_bench=["Recurve Bow", "Recurve Bow"], players=lobby,
        ))
        assert [u.name for u in st.board][:2] == ["Ahri", "Morgana"] and [u.name for u in st.bench] == ["Kayle"]
        assert st.item_bench == ["B.F. Sword"] and st.hp == 62
        assert next(p for p in st.players if p.name == "Me").hp == 62
        # Filed under the player whose listed HP is on screen.
        assert [u.name for u in st.opponents["Kaiser"].board][0] == "Lucian"
    # The scout key pressed on our own board (one unit bought) is still ours.
    st = tracker.ingest(obs(purpose="scout", viewing_own_board=True, hp=62, players=lobby,
                            board=[uo("Ahri", 2), uo("Morgana", 2), "Braum", "Shen", "Garen", "Pyke", "Kayle"]))
    assert len(st.board) == 7


def test_wrong_first_self_flag_is_corrected(tracker):
    tracker.ingest(obs(stage="1-2", players=[PlayerObs(name="Kaiser", hp=100, is_self=True), PlayerObs(name="Me", hp=100)]))
    assert tracker.state.self_name == "Kaiser"
    rows = [PlayerObs(name="Me", hp=72, is_self=True), PlayerObs(name="Kaiser", hp=91), PlayerObs(name="Bolt", hp=50)]
    tracker.ingest(own_frame(stage="3-1", hp=72, board=["Garen", "Graves"], players=rows))
    st = tracker.ingest(own_frame(stage="3-2", hp=72, players=rows))
    assert st.self_name == "Me" and st.hp == 72
    assert next(p for p in st.players if p.name == "Kaiser").hp == 91
    assert [p.name for p in st.players if p.is_self] == ["Me"]
    # Kaiser can be scouted now.
    st = tracker.ingest(obs(purpose="scout", viewing_own_board=False, viewed_player_name="Kaiser", board=["Ahri", "Shen"]))
    assert "Kaiser" in st.opponents and "Kaiser" in tracker.taken_by_player()
    assert tracker.self_name_confirmed


def test_one_stray_self_flag_does_not_switch(tracker):
    tracker.ingest(own_frame(stage="3-1", players=players(("Me", 70), ("Kaiser", 80), me="Me")))
    tracker.ingest(own_frame(stage="3-1", players=players(("Me", 70), ("Kaiser", 80), me="Me")))
    st = tracker.ingest(own_frame(stage="3-2", players=players(("Me", 70), ("Kaiser", 80), me="Kaiser")))
    assert st.self_name == "Me"
    st = tracker.ingest(own_frame(stage="3-2", players=players(("Me", 70), ("Kaiser", 80), me="Me")))
    assert st.self_name == "Me"


def test_own_board_with_a_model_read_name_is_ours(tracker):
    lobby = players(("Me", 70), ("Kaiser", 60), me="Me")
    board = ["Ahri", "Garen", "Graves", "Braum", "Shen"]
    tracker.ingest(own_frame(stage="3-2", board=board, players=lobby))
    st = tracker.ingest(obs(stage="3-3", viewing_own_board=False, viewed_player_name="Kaiser", level=6, board=board, players=lobby))
    assert st.opponents == {} and tracker.taken_by_player() == {} and st.level == 6


def test_star_ups_after_a_roll_still_look_like_our_board(tracker):
    tracker.ingest(own_frame(stage="4-1", level=7, board=["Ahri", "Garen", "Graves", "Braum", "Shen", "Pyke", "Lucian"]))
    st = tracker.ingest(obs(
        stage="4-2", viewing_own_board=False, level=8,
        board=[uo("Ahri", 2), uo("Garen", 2), uo("Graves", 2), uo("Braum", 2), "Shen", "Pyke", "Lucian", "Kayle"],
    ))
    assert st.level == 8 and len(st.board) == 8 and st.opponents == {}
    # The bottom HUD applies even when the board is filed under an opponent.
    st = tracker.ingest(obs(stage="4-2", viewing_own_board=False, viewed_player_name="Bob", level=9, board=["Draven", "Ashe"]))
    assert st.level == 9 and "Bob" in st.opponents


def test_augment_screen_bench_read_is_not_an_unknown_opponent(tracker):
    tracker.ingest(own_frame(stage="4-1", level=7, board=["Ahri", "Garen", "Graves", "Braum", "Shen", "Pyke", "Lucian"],
                             bench=["Kayle", "Akali", "Draven"]))
    st = tracker.ingest(obs(stage="4-2", screen_type=ScreenType.AUGMENT_SELECT, viewing_own_board=False, level=8,
                            board=["Ahri"], bench=["Kayle", "Akali", "Draven"]))
    assert st.opponents == {} and tracker.taken_by_player() == {} and st.level == 8


def test_stale_unknown_snapshot_is_superseded(tracker):
    tracker.ingest(own_frame(stage="3-2", board=["Garen"], players=players(("Me", 70), ("Kaiser", 50), ("Bolt", 50), me="Me")))
    tracker.ingest(obs(purpose="scout", viewing_own_board=False, hp=50, board=["Lucian", "Ashe", "Vayne", uo("Kayle", 2)]))
    assert UNKNOWN_PLAYER in tracker.taken_by_player()
    tracker.ingest(own_frame(stage="4-2"))
    assert UNKNOWN_PLAYER not in tracker.taken_by_player()  # a stage old
    tracker.ingest(obs(purpose="scout", viewing_own_board=False, hp=50, board=["Draven"]))
    tracker.ingest(obs(purpose="scout", viewing_own_board=False, viewed_player_name="Kaiser",
                       board=[uo("Lucian", 2), uo("Ashe", 2), "Vayne", uo("Kayle", 2)]))
    assert set(tracker.taken_by_player()) == {"Kaiser"}
    assert tracker.taken_copies()["TFT99_Kayle"] == 3


def test_short_chinese_names_match_with_one_wrong_character(tracker):
    lobby = players(("我自己", 70), ("月光", 60), ("小明", 50), me="我自己")
    tracker.ingest(own_frame(stage="3-1", board=["Garen"], players=lobby))
    st = tracker.ingest(obs(purpose="scout", viewing_own_board=False, viewed_player_name="月先", board=["Kayle", "Kayle"],
                            players=players(("我自己", 70), ("月先", 60), ("小明", 50))))
    assert sorted(p.name for p in st.players) == sorted(["我自己", "月光", "小明"])
    assert set(st.opponents) == {"月光"}
    tracker.ingest(obs(purpose="scout", viewing_own_board=False, viewed_player_name="月光", board=["Kayle", "Kayle", "Teemo"]))
    assert set(tracker.taken_by_player()) == {"月光"}
    # Different digits are different players, not a misread.
    assert tracker._canonical_player("P2", ["P1"]) == "P2"
    assert tracker._canonical_player("Zcd", ["Zed", "Kaiser"]) == "Zed"


def test_snapshot_of_a_name_missing_from_a_full_lobby_is_ignored(tracker):
    lobby = players(("Me", 70), ("A1", 60), ("B1", 60), ("C1", 60), ("D1", 60), ("E1", 60), ("F1", 60), ("G1", 60), me="Me")
    tracker.ingest(obs(purpose="scout", viewing_own_board=False, viewed_player_name="Ghost", board=["Kayle"]))
    assert "Ghost" in tracker.taken_by_player()
    tracker.ingest(own_frame(stage="3-1", players=lobby))
    assert "Ghost" not in tracker.taken_by_player()


def test_combat_frame_with_enemy_units_keeps_our_board(tracker):
    own = ["Ahri", "Garen", "Graves", "Braum", "Shen", "Pyke", "Lucian"]
    tracker.ingest(own_frame(stage="4-3", level=7, board=own))
    enemies = [uo("Ashe", 2), uo("Vayne", 2), uo("Darius", 2), "Draven", uo("Kayle", 2)]
    st = tracker.ingest(own_frame(screen_type=ScreenType.COMBAT, level=7, board=[*own, *enemies]))
    assert [u.name for u in st.board] == own
    # A combat board that is not ours is ignored too (HUD still applies).
    st = tracker.ingest(own_frame(screen_type=ScreenType.COMBAT, level=7, streak=2, board=enemies))
    assert [u.name for u in st.board] == own and st.streak == 2


def test_unit_missed_for_one_frame_is_kept_with_its_hex(tracker):
    board = [uo("Ahri", 2, row=3, col=0), uo("Garen", row=0, col=3), uo("Graves", row=3, col=6), uo("Braum", row=0, col=2)]
    tracker.ingest(own_frame(stage="3-3", gold=30, board=board, bench=["Kayle"]))
    st = tracker.ingest(own_frame(stage="3-5", gold=30, board=board[1:], bench=["Kayle"]))  # Ahri missed
    assert [u.name for u in st.board].count("Ahri") == 1 and len(st.board) == 4
    # Back, read without hexes: the remembered hex returns.
    st = tracker.ingest(own_frame(stage="3-6", gold=30, board=[uo("Ahri", 2), "Garen", "Graves", "Braum"]))
    ahri = next(u for u in st.board if u.name == "Ahri")
    assert (ahri.row, ahri.col) == (3, 0)
    # Gone on two frames in a row: really gone.
    tracker.ingest(own_frame(gold=30, board=["Garen", "Graves", "Braum"]))
    st = tracker.ingest(own_frame(gold=30, board=["Garen", "Graves", "Braum"]))
    assert len(st.board) == 3
    # Sold (gold went up by its value) or benched: gone at once.
    st = tracker.ingest(own_frame(gold=30, board=["Garen", "Graves", "Braum", "Lucian"]))
    st = tracker.ingest(own_frame(gold=33, board=["Garen", "Graves", "Braum"]))
    assert len(st.board) == 3
    tracker.ingest(own_frame(gold=33, board=["Garen", "Graves", "Braum", "Lucian"]))
    st = tracker.ingest(own_frame(gold=33, board=["Garen", "Graves", "Braum"], bench=["Lucian"]))
    assert len(st.board) == 3


def test_manual_corrections_are_bounded_and_accept_full_width(tracker, mech):
    tracker.set_field("level", 8)
    with pytest.raises(ValueError):
        tracker.set_field("xp_current", mech.xp_to_level[8])
    assert tracker.set_field("xp_current", mech.xp_to_level[8] - 1).xp_current == mech.xp_to_level[8] - 1
    with pytest.raises(ValueError):
        tracker.set_field("stage", "9-9")
    with pytest.raises(ValueError):
        tracker.set_field("stage", "3-8")
    assert tracker.set_field("gold", "５５").gold == 55
    assert str(tracker.set_field("stage", "３－２").stage) == "3-2"
    assert tracker.set_field("streak", "－３").streak == -3


def test_hex_positions_are_remembered_for_a_few_rounds(tracker):
    board = [uo("Ahri", 2, row=3, col=0), uo("Shen", row=0, col=1), uo("Garen", row=0, col=3), uo("Graves", row=3, col=6)]
    tracker.ingest(own_frame(stage="3-3", board=board))
    tracker.ingest(own_frame(stage="3-5", board=board[2:]))  # two units missed: not kept
    st = tracker.ingest(own_frame(stage="3-6", board=[uo("Ahri", 2), "Shen", "Garen", "Graves"]))
    pos = {u.name: (u.row, u.col) for u in st.board}
    assert pos["Ahri"] == (3, 0) and pos["Shen"] == (0, 1)


def test_named_opponent_with_a_board_close_to_our_small_one(tracker):
    # Early boards share cheap units: the banner name still wins when the
    # boards differ (only the vision's own "own board" claim is lenient).
    tracker.ingest(own_frame(stage="2-1", board=["Garen", "Graves"], players=players(("Me", 100), ("Alice", 100), me="Me")))
    st = tracker.ingest(obs(stage="2-1", viewing_own_board=False, viewed_player_name="Alice", board=["Garen", "Graves", "Ahri"]))
    assert "Alice" in st.opponents and [u.name for u in st.board] == ["Garen", "Graves"]


def test_self_flags_on_scout_frames_do_not_move_the_local_player(tracker):
    lobby = [PlayerObs(name="Me", hp=70, is_self=True), PlayerObs(name="Kaiser", hp=50)]
    tracker.ingest(own_frame(stage="3-1", hp=70, board=["Garen"], players=lobby))
    viewed = [PlayerObs(name="Me", hp=70), PlayerObs(name="Kaiser", hp=50, is_self=True)]  # the viewed row is lit
    for _ in range(3):
        st = tracker.ingest(obs(purpose="scout", viewing_own_board=False, viewed_player_name="Kaiser", hp=50,
                                board=["Ahri"], players=viewed))
    assert st.self_name == "Me" and st.hp == 70 and "Kaiser" in st.opponents


def test_gold_augment_on_an_augment_round_is_taken_at_once(tracker, mech):
    rnd = mech.augment_rounds[-1]
    tracker.ingest(own_frame(stage=rnd, gold=50, screen_type=ScreenType.AUGMENT_SELECT, augment_choices=["A", "B", "C"]))
    st = tracker.ingest(own_frame(stage=rnd, gold=95))
    assert st.gold == 95
    st = tracker.ingest(own_frame(stage=rnd, gold=195))  # still a misread
    assert st.gold == 95
