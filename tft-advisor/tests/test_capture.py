from __future__ import annotations

import math
import sys
import threading
import types

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tft_advisor.capture import calibrate, change, hotkeys, regions, screen
from tft_advisor.capture.change import RoundWatcher, diff, local_diff, signature
from tft_advisor.capture.hotkeys import (
    MOD_ALT,
    MOD_CONTROL,
    MOD_SHIFT,
    MOD_WIN,
    HotkeyManager,
    parse_hotkey,
)
from tft_advisor.capture.regions import crop, list_regions, region_box, viewport
from tft_advisor.capture.screen import FileCapturer, ScreenCapturer, find_window_rect, save_frame, set_dpi_awareness
from tft_advisor.config import CaptureConfig

EXPECTED_REGIONS = {
    "stage",
    "gold",
    "level",
    "xp",
    "streak",
    "shop",
    "bench",
    "board",
    "players",
    "traits",
    "items",
    "augments",
    "hud_bottom",
    "top_banner",
}

EM_DASHES = (chr(0x2014), chr(0x2015))  # em dash, horizontal bar


def _expected_box(size, name, ox=0.0, oy=0.0, vw=None, vh=None):
    w, h = size
    vw = w if vw is None else vw
    vh = h if vh is None else vh
    fx0, fy0, fx1, fy1 = regions.REGIONS[name]
    return (
        math.floor(ox + fx0 * vw + 1e-6),
        math.floor(oy + fy0 * vh + 1e-6),
        math.ceil(ox + fx1 * vw - 1e-6),
        math.ceil(oy + fy1 * vh - 1e-6),
    )


# --------------------------------------------------------------------------
# regions
# --------------------------------------------------------------------------


def test_region_names_are_fixed_and_fractions_valid():
    assert set(list_regions()) == EXPECTED_REGIONS
    assert list_regions() == list(regions.REGIONS)
    for name, (x0, y0, x1, y1) in regions.REGIONS.items():
        assert 0.0 <= x0 < x1 <= 1.0, name
        assert 0.0 <= y0 < y1 <= 1.0, name
        assert name in regions.ANCHORS and name in regions.REGION_LABELS


def test_region_labels_have_no_em_dash():
    for label in regions.REGION_LABELS.values():
        assert not any(d in label for d in EM_DASHES)


def test_region_box_1080p_matches_fractions():
    size = (1920, 1080)
    for name in list_regions():
        assert region_box(size, name) == _expected_box(size, name), name
    # A couple of hand-checked values.
    assert region_box(size, "stage") == (739, 0, 912, 42)
    assert region_box(size, "shop") == (326, 912, 1594, 1080)
    assert region_box(size, "players")[2] == 1920


def test_region_box_1440p_scales_with_resolution():
    for name in list_regions():
        a = region_box((1920, 1080), name)
        b = region_box((2560, 1440), name)
        assert b == _expected_box((2560, 1440), name)
        for va, vb in zip(a, b, strict=True):
            assert abs(va * 4 / 3 - vb) <= 2, name


def test_ultrawide_is_pillarboxed_and_centered():
    size = (3440, 1440)
    assert viewport(size) == (440.0, 0.0, 2560.0, 1440.0)
    for name in list_regions():
        ref = region_box((2560, 1440), name)
        box = region_box(size, name)
        assert box == (ref[0] + 440, ref[1], ref[2] + 440, ref[3]), name
        assert 440 <= box[0] < box[2] <= 3000


def test_16_10_is_letterboxed_vertically():
    size = (1920, 1200)
    assert viewport(size) == (0.0, 60.0, 1920.0, 1080.0)
    for name in list_regions():
        ref = region_box((1920, 1080), name)
        assert region_box(size, name) == (ref[0], ref[1] + 60, ref[2], ref[3] + 60), name


def test_anchored_layout_sticks_to_edges():
    size = (3440, 1440)
    players = region_box(size, "players", layout="anchored")
    assert players[2] == 3440
    assert region_box(size, "items", layout="anchored")[0] == 0
    assert region_box(size, "stage", layout="anchored") == region_box(size, "stage")
    tall = (1920, 1200)
    assert region_box(tall, "shop", layout="anchored")[3] == 1200
    assert region_box(tall, "stage", layout="anchored")[1] == 0
    for name in list_regions():  # identical on a 16:9 frame
        assert region_box((1920, 1080), name, layout="anchored") == region_box((1920, 1080), name)
    with pytest.raises(ValueError):
        region_box((1920, 1080), "gold", layout="stretched")


def test_layout_env_override(monkeypatch):
    monkeypatch.setenv("TFT_ADVISOR_HUD_LAYOUT", "anchored")
    assert region_box((3440, 1440), "players")[2] == 3440
    monkeypatch.delenv("TFT_ADVISOR_HUD_LAYOUT")
    assert region_box((3440, 1440), "players")[2] == 3000


def test_crop_sizes_and_content():
    for size in [(1920, 1080), (2560, 1440), (3440, 1440), (1920, 1200)]:
        img = Image.new("RGB", size, (0, 0, 0))
        for name in list_regions():
            x0, y0, x1, y1 = region_box(size, name)
            assert crop(img, name).size == (x1 - x0, y1 - y0)
    img = Image.new("RGB", (1920, 1080), (0, 0, 0))
    box = region_box(img.size, "gold")
    ImageDraw.Draw(img).rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=(255, 0, 0))
    assert crop(img, "gold").getextrema() == ((255, 255), (0, 0), (0, 0))


def test_pad_grows_and_clamps():
    size = (1920, 1080)
    for name in list_regions():
        a = region_box(size, name)
        b = region_box(size, name, pad=0.1)
        assert b[0] <= a[0] and b[1] <= a[1] and b[2] >= a[2] and b[3] >= a[3]
        assert 0 <= b[0] < b[2] <= 1920 and 0 <= b[1] < b[3] <= 1080
    gold = region_box(size, "gold")
    padded = region_box(size, "gold", pad=0.5)
    assert padded[2] - padded[0] >= 2 * (gold[2] - gold[0]) - 2
    img = Image.new("RGB", size)
    assert crop(img, "shop", pad=0.2).size[1] > crop(img, "shop").size[1]
    assert region_box(size, "shop", pad=0.5)[3] == 1080


def test_region_box_errors_and_tiny_frames():
    with pytest.raises(KeyError):
        region_box((1920, 1080), "minimap")
    with pytest.raises(ValueError):
        region_box((0, 1080), "gold")
    for name in list_regions():
        x0, y0, x1, y1 = region_box((16, 9), name)
        assert x1 > x0 and y1 > y0 and x1 <= 16 and y1 <= 9


# --------------------------------------------------------------------------
# change detection
# --------------------------------------------------------------------------

# 7-segment digits: segments a b c d e f g
_SEGMENTS = {
    "0": "abcdef",
    "1": "bc",
    "2": "abdeg",
    "3": "abcdg",
    "4": "bcfg",
    "5": "acdfg",
    "6": "acdefg",
    "7": "abc",
    "8": "abcdefg",
    "9": "abcdfg",
    "-": "g",
}


def _draw_char(draw, x, y, w, h, ch, color=(240, 240, 240)):
    t = max(2, w // 5)
    segs = {
        "a": (x, y, x + w, y + t),
        "b": (x + w - t, y, x + w, y + h // 2),
        "c": (x + w - t, y + h // 2, x + w, y + h),
        "d": (x, y + h - t, x + w, y + h),
        "e": (x, y + h // 2, x + t, y + h),
        "f": (x, y, x + t, y + h // 2),
        "g": (x, y + h // 2 - t // 2, x + w, y + h // 2 + t // 2),
    }
    for s in _SEGMENTS[ch]:
        draw.rectangle(segs[s], fill=color)


def _hud(size=(1280, 720), stage="3-5", shop_seed=0, base=(35, 40, 55)):
    """Synthetic HUD frame: dark background, stage text, shop cards, a board."""
    img = Image.new("RGB", size, base)
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = region_box(size, "stage")
    ch_h = int((y1 - y0) * 0.55)
    ch_w = max(6, ch_h // 2)
    x = x0 + (x1 - x0) // 8
    for ch in stage:
        _draw_char(draw, x, y0 + (y1 - y0) // 5, ch_w, ch_h, ch)
        x += ch_w + ch_w // 2
    sx0, sy0, sx1, sy1 = region_box(size, "shop")
    rng = np.random.default_rng(shop_seed)
    card_w = (sx1 - sx0) // 5
    for i in range(5):
        color = tuple(int(c) for c in rng.integers(40, 255, 3))
        draw.rectangle([sx0 + i * card_w + 4, sy0 + 4, sx0 + (i + 1) * card_w - 4, sy1 - 4], fill=color)
    bx0, by0, bx1, by1 = region_box(size, "board")
    draw.rectangle([bx0, by0, bx1, by1], outline=(90, 90, 120), width=3)
    return img


def test_signature_shape_range_and_diff():
    img = _hud()
    sig = signature(img)
    assert sig.shape == (18, 32) and sig.dtype == np.float32
    assert 0.0 <= float(sig.min()) and float(sig.max()) <= 1.0
    s2 = signature(img, "shop", size=(64, 10))
    assert s2.shape == (10, 64)
    box = region_box(img.size, "shop")
    assert np.allclose(signature(img, box, size=(64, 10)), s2)
    assert diff(sig, sig) == 0.0
    assert diff(sig, s2) == 1.0 and diff(None, sig) == 1.0
    white = signature(Image.new("RGB", (320, 180), (255, 255, 255)))
    black = signature(Image.new("RGB", (320, 180), (0, 0, 0)))
    assert diff(white, black) == pytest.approx(1.0)
    rgba = Image.new("RGBA", (320, 180), (255, 255, 255, 255))
    assert signature(rgba).shape == (18, 32)


def test_local_diff_sees_digit_change_that_mean_diff_misses():
    a = signature(_hud(stage="3-5"), "stage", size=(48, 12))
    b = signature(_hud(stage="3-6"), "stage", size=(48, 12))
    assert diff(a, b) < 0.08
    assert local_diff(a, b) > 0.08
    assert local_diff(a, a) == 0.0


def test_round_watcher_fires_once_after_stabilizing():
    w = RoundWatcher(threshold=0.08, stable_frames=2)
    base = _hud(stage="3-5")
    assert w.update(base) == set()
    assert w.update(base) == set()  # first stable view: baseline only
    assert w.update(base) == set()
    changed = _hud(stage="3-6")
    assert w.update(changed) == set()  # not stable yet
    assert w.update(changed) == {"round_changed"}
    for _ in range(5):
        assert w.update(changed) == set()
    assert w.counts["round_changed"] == 1
    assert w.last_scores["round_changed"] == pytest.approx(0.0)


def test_round_watcher_ignores_animation_noise():
    w = RoundWatcher(threshold=0.08, stable_frames=2)
    base = _hud(stage="3-5")
    w.update(base)
    w.update(base)
    rng = np.random.default_rng(1)
    arr = np.asarray(base).astype(np.int16)
    x0, y0, x1, y1 = region_box(base.size, "stage")
    fired: set[str] = set()
    for i in range(12):
        # A glow moving across the stage region, plus heavy noise over the whole frame.
        frame = arr + rng.integers(-90, 90, arr.shape, dtype=np.int16)
        img = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(img)
        gx = x0 + (i * 17) % max(1, x1 - x0 - 12)
        d.ellipse([gx, y0 + 2, gx + 12, y1 - 2], fill=(255, 255, 200))
        fired |= w.update(img)
    assert fired == set()
    # Back to the same stable view: nothing changed compared to the baseline.
    assert w.update(base) == set()
    assert w.update(base) == set()
    assert w.update(base) == set()


def _glow(img: Image.Image) -> Image.Image:
    """Same frame with a bright pulsing icon at the right end of the stage region."""
    out = img.copy()
    x0, y0, x1, y1 = region_box(out.size, "stage")
    ImageDraw.Draw(out).rectangle([x1 - (x1 - x0) // 5, y0 + 3, x1 - 3, y1 - 3], fill=(255, 230, 120))
    return out


def test_pulsing_icon_fires_round_changed_at_most_once():
    w = RoundWatcher()
    base = _hud(stage="3-5")
    lit = _glow(base)
    fired = []
    for _ in range(6):  # glow on for 2 frames, off for 2 frames (long enough to look stable)
        for frame in (base, base, lit, lit):
            fired.append(w.update(frame))
    assert sum("round_changed" in e for e in fired) <= 1
    # A real round change afterwards is still reported, once.
    nxt = _hud(stage="3-6")
    assert w.update(nxt) == set()
    assert "round_changed" in w.update(nxt)
    assert w.update(nxt) == set()
    # Without the novelty memory the same pulsing fires on every stable switch.
    plain = RoundWatcher(history=0)
    count = sum("round_changed" in plain.update(f) for _ in range(3) for f in (base, base, lit, lit))
    assert count >= 4


def test_screen_changed_fires_again_when_returning_to_own_board():
    w = RoundWatcher()
    own, other = _hud(), _hud(base=(170, 110, 50), shop_seed=3)
    events = [w.update(f) for f in (own, own, other, other, own, own)]
    assert sum("screen_changed" in e for e in events) == 2


def test_screen_changed_fires_once_after_camera_settles():
    w = RoundWatcher()
    a = _hud()
    for _ in range(3):
        assert "screen_changed" not in w.update(a)
    rng = np.random.default_rng(2)
    for _ in range(5):  # camera pan / combat: large blobs, every frame different
        blobs = rng.integers(0, 255, (9, 16, 3), dtype=np.uint8)
        frame = Image.fromarray(blobs).resize((1280, 720), Image.Resampling.NEAREST)
        assert "screen_changed" not in w.update(frame)
    other = _hud(base=(160, 120, 60), shop_seed=5)
    assert "screen_changed" not in w.update(other)
    assert "screen_changed" in w.update(other)
    assert "screen_changed" not in w.update(other)
    assert w.counts["screen_changed"] == 1


def test_shop_changed_on_roll():
    w = RoundWatcher()
    a = _hud(shop_seed=0)
    w.update(a)
    w.update(a)
    rolled = _hud(shop_seed=42)
    assert w.update(rolled) == set()
    events = w.update(rolled)
    assert "shop_changed" in events and "round_changed" not in events
    assert w.update(rolled) == set()


def test_buying_cards_is_not_a_shop_change():
    w = RoundWatcher()
    shop = _hud(shop_seed=7)
    w.update(shop)
    w.update(shop)
    sx0, sy0, sx1, sy1 = region_box(shop.size, "shop")
    card_w = (sx1 - sx0) // 5
    bought = shop.copy()
    draw = ImageDraw.Draw(bought)
    for i in (1, 3):  # two cards bought: their slots go dark
        draw.rectangle([sx0 + i * card_w + 4, sy0 + 4, sx0 + (i + 1) * card_w - 4, sy1 - 4], fill=(20, 20, 25))
    events = [w.update(bought) for _ in range(3)]
    assert all("shop_changed" not in e for e in events)
    rolled = _hud(shop_seed=8)
    assert [("shop_changed" in w.update(rolled)) for _ in range(3)] == [False, True, False]


def test_column_diff_counts_columns():
    a = np.zeros((10, 64), dtype=np.float32)
    b = a.copy()
    b[:, :20] = 1.0  # a third of the width changed
    assert change.column_diff(a, b) == 0.0
    b[:, :50] = 1.0  # most of the width changed
    assert change.column_diff(a, b) == pytest.approx(1.0)
    assert change.column_diff(a, a[:, :10]) == 1.0


def test_slow_drift_does_not_accumulate():
    w = RoundWatcher()
    fired: set[str] = set()
    for i in range(40):  # brightness creeping up 0.01 per frame, 0.4 in total
        v = 40 + int(i * 2.55)
        fired |= w.update(Image.new("RGB", (320, 180), (v, v, v)))
    assert fired == set()


def test_stable_frames_and_reset():
    w = RoundWatcher(stable_frames=3)
    a, b = _hud(stage="3-5"), _hud(stage="4-1")
    for _ in range(3):
        w.update(a)
    assert w.update(b) == set()
    assert w.update(b) == set()
    assert w.update(b) == {"round_changed"}
    w.reset()
    assert w.update(a) == set()
    assert w.update(a) == set()
    assert w.update(a) == set()  # baseline re-established without an event
    assert w.update(None) == set()


def test_stable_frames_one_fires_immediately():
    w = RoundWatcher(stable_frames=1)
    a, b = _hud(stage="3-5"), _hud(stage="3-6")
    assert w.update(a) == set()
    assert w.update(b) == {"round_changed"}
    assert w.update(b) == set()


def test_round_watcher_is_thread_safe():
    w = RoundWatcher()
    frames = [_hud(stage="3-5"), _hud(stage="3-6")]
    errors: list[BaseException] = []

    def run(k: int) -> None:
        try:
            for i in range(20):
                w.update(frames[(i // 3 + k) % 2])
                if i == 10:
                    w.reset()
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(k,)) for k in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


# --------------------------------------------------------------------------
# hotkeys
# --------------------------------------------------------------------------


def test_parse_hotkey_valid():
    assert parse_hotkey("F6") == (0, 0x75)
    assert parse_hotkey("f1") == (0, 0x70)
    assert parse_hotkey("F24") == (0, 0x87)
    assert parse_hotkey("ctrl+shift+a") == (MOD_CONTROL | MOD_SHIFT, 0x41)
    assert parse_hotkey("alt+F1") == (MOD_ALT, 0x70)
    assert parse_hotkey(" Ctrl + 1 ") == (MOD_CONTROL, 0x31)
    assert parse_hotkey("win+space") == (MOD_WIN, 0x20)
    assert parse_hotkey("Control+Alt+Z") == (MOD_CONTROL | MOD_ALT, ord("Z"))
    assert parse_hotkey("numpad5") == (0, 0x65)
    assert parse_hotkey("shift+`") == (MOD_SHIFT, 0xC0)


@pytest.mark.parametrize("bad", ["", "   ", "F25", "F0", "ctrl+", "ctrl+shift", "a+b", "hyper+x", "ctrl++", "foo", None])
def test_parse_hotkey_invalid(bad):
    with pytest.raises(ValueError):
        parse_hotkey(bad)


def test_default_config_hotkeys_parse():
    from tft_advisor.config import HotkeyConfig

    cfg = HotkeyConfig()
    for combo in (cfg.analyze, cfg.scout, cfg.toggle_auto, cfg.shop):
        mods, vk = parse_hotkey(combo)
        assert mods == 0 and 0x70 <= vk <= 0x87


@pytest.mark.skipif(sys.platform == "win32", reason="non-Windows behaviour")
def test_hotkey_manager_start_returns_false_off_windows():
    logs: list[str] = []
    called: list[str] = []
    mgr = HotkeyManager({"F6": lambda: called.append("F6")}, log=logs.append)
    assert mgr.start() is False
    assert not mgr.running
    assert logs and "Windows" in logs[0]
    assert not any(d in logs[0] for d in EM_DASHES)
    mgr.stop()  # safe without a running loop
    mgr.stop()
    assert called == []


def test_hotkey_callback_errors_are_logged_not_fatal():
    logs: list[str] = []
    called: list[str] = []

    def boom() -> None:
        raise RuntimeError("kaput")

    mgr = HotkeyManager({"F6": boom, "F7": lambda: called.append("F7")}, log=logs.append)
    mgr._worker = threading.Thread(target=mgr._work, daemon=True)
    mgr._worker.start()
    mgr._jobs.put(boom)
    mgr._jobs.put(mgr.bindings["F7"])
    mgr._stop_worker()
    assert called == ["F7"]
    assert any("kaput" in line for line in logs)


# --------------------------------------------------------------------------
# screen capture
# --------------------------------------------------------------------------


def test_screen_capturer_headless_never_raises():
    cap = ScreenCapturer(CaptureConfig())
    img = cap.grab()
    assert img is None or isinstance(img, Image.Image)
    if img is None:
        assert isinstance(cap.last_error, str) and cap.last_error
        assert not any(d in cap.last_error for d in EM_DASHES)
    else:  # pragma: no cover - only with a real display
        assert img.mode == "RGB"
    result: list[object] = []
    t = threading.Thread(target=lambda: result.append(cap.grab()))
    t.start()
    t.join()
    assert result and (result[0] is None or isinstance(result[0], Image.Image))
    cap.close()
    assert cap.grab() is None


def test_windows_helpers_are_noops_elsewhere():
    if sys.platform == "win32":  # pragma: no cover
        pytest.skip("Windows only behaviour differs")
    assert find_window_rect("League of Legends (TM) Client") is None
    assert set_dpi_awareness() is False


class _FakeShot:
    def __init__(self, width: int, height: int, bgra: bytes) -> None:
        self.size = (width, height)
        self.bgra = bgra


class _FakeMSS:
    instances: list["_FakeMSS"] = []
    pixel = (10, 20, 30, 255)  # B, G, R, A

    def __init__(self) -> None:
        self.closed = False
        self.thread = threading.get_ident()
        self.grabs: list[dict] = []
        _FakeMSS.instances.append(self)

    @property
    def monitors(self):
        return [
            {"left": 0, "top": 0, "width": 200, "height": 100},
            {"left": 0, "top": 0, "width": 160, "height": 90},
        ]

    def grab(self, region):
        assert threading.get_ident() == self.thread, "mss instance used from another thread"
        self.grabs.append(dict(region))
        w, h = region["width"], region["height"]
        return _FakeShot(w, h, bytes(self.pixel) * (w * h))

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_mss(monkeypatch):
    _FakeMSS.instances = []
    _FakeMSS.pixel = (10, 20, 30, 255)
    module = types.ModuleType("mss")
    module.MSS = _FakeMSS
    monkeypatch.setitem(sys.modules, "mss", module)
    # Deterministic on a Windows dev box too (a running game would otherwise be captured).
    monkeypatch.setattr(screen, "_on_windows", lambda: False)
    return _FakeMSS


def test_screen_capturer_with_fake_mss(fake_mss):
    cap = ScreenCapturer(CaptureConfig(monitor=1, use_window=True))
    img = cap.grab()
    assert isinstance(img, Image.Image) and img.mode == "RGB"
    assert img.size == (160, 90)
    assert img.getpixel((5, 5)) == (30, 20, 10)  # BGRA -> RGB
    assert cap.last_source == "monitor" and cap.last_rect == (0, 0, 160, 90)
    assert cap.last_error is None

    other: list[object] = []
    t = threading.Thread(target=lambda: other.append(cap.grab()))
    t.start()
    t.join()
    assert isinstance(other[0], Image.Image)
    assert len(fake_mss.instances) == 2  # one mss instance per thread
    cap.close()
    assert all(inst.closed for inst in fake_mss.instances)


def test_screen_capturer_bad_monitor_index_falls_back(fake_mss):
    cap = ScreenCapturer(CaptureConfig(monitor=7))
    img = cap.grab()
    assert img is not None and img.size == (160, 90)
    cap.close()


def test_screen_capturer_black_frames(fake_mss):
    fake_mss.pixel = (0, 0, 0, 255)
    logs: list[str] = []
    cap = ScreenCapturer(CaptureConfig(), log=logs.append)
    results = [cap.grab() for _ in range(ScreenCapturer.BLACK_FRAMES_HINT)]
    assert results == [None] * ScreenCapturer.BLACK_FRAMES_HINT
    assert cap.last_error and "无边框" in cap.last_error
    assert len(logs) == 1
    fake_mss.pixel = (10, 20, 30, 255)
    assert cap.grab() is not None and cap.last_error is None
    cap.close()


def test_screen_capturer_grab_failure_returns_none(monkeypatch):
    class Broken:
        def __init__(self) -> None:
            raise RuntimeError("no screen")

    module = types.ModuleType("mss")
    module.MSS = Broken
    monkeypatch.setitem(sys.modules, "mss", module)
    monkeypatch.setattr(screen, "_on_windows", lambda: False)
    cap = ScreenCapturer(CaptureConfig())
    assert cap.grab() is None
    assert cap.grab() is None
    assert cap.failures == 2 and "no screen" in (cap.last_error or "")


def _write_png(path, color, size=(64, 36)):
    Image.new("RGB", size, color).save(path)
    return path


def test_file_capturer_cycles(tmp_path):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    paths = [_write_png(tmp_path / f"f{i}.png", c) for i, c in enumerate(colors)]
    cap = FileCapturer(paths)
    assert len(cap) == 3
    got = [cap.grab() for _ in range(3)]
    assert [g.getpixel((0, 0)) for g in got] == colors
    assert cap.current_path == paths[2]
    assert cap.grab() is None and cap.remaining == 0
    cap.reset()
    assert cap.grab().getpixel((0, 0)) == colors[0]

    looping = FileCapturer([str(p) for p in paths], loop=True)
    seq = [looping.grab().getpixel((0, 0)) for _ in range(7)]
    assert seq == colors * 2 + colors[:1]


def test_file_capturer_dirs_globs_missing_and_bad_files(tmp_path):
    _write_png(tmp_path / "b.png", (0, 0, 9))
    _write_png(tmp_path / "a.png", (0, 0, 1))
    (tmp_path / "c.png").write_bytes(b"not a png")
    (tmp_path / "notes.txt").write_text("x")
    rgba = Image.new("RGBA", (8, 8), (1, 2, 3, 128))
    rgba.save(tmp_path / "d.png")

    cap = FileCapturer([tmp_path, tmp_path / "missing.png"])
    assert [p.name for p in cap.paths] == ["a.png", "b.png", "c.png", "d.png"]
    assert cap.missing == [str(tmp_path / "missing.png")]
    frames = []
    while (f := cap.grab()) is not None:
        frames.append((cap.current_path.name, f.mode))
    assert frames == [("a.png", "RGB"), ("b.png", "RGB"), ("d.png", "RGB")]
    assert cap.last_error and "c.png" in cap.last_error

    globbed = FileCapturer(str(tmp_path / "*.png"))
    assert len(globbed) == 4
    assert FileCapturer([]).grab() is None
    assert FileCapturer([], loop=True).grab() is None


def test_save_frame(tmp_path, monkeypatch):
    img = Image.new("RGB", (40, 20), (1, 2, 3))
    target = tmp_path / "a" / "b"
    p1 = save_frame(img, target, "scout/玩家 1")
    p2 = save_frame(img, target, "scout/玩家 1")
    assert p1 != p2 and p1.exists() and p2.exists()
    assert p1.parent == target and p1.suffix == ".png"
    assert "/" not in p1.name[:-4] and "玩家" in p1.name
    with Image.open(p1) as back:
        assert back.size == (40, 20) and back.convert("RGB").getpixel((0, 0)) == (1, 2, 3)

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    p3 = save_frame(img, "~/caps", "")
    assert p3.parent == tmp_path / "caps" and p3.name.endswith("_frame.png")

    pal = img.convert("P")
    assert save_frame(pal, tmp_path, "pal").exists()


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(1920, 1080), (3440, 1440), (1920, 1200), (640, 360)])
def test_draw_regions_output(size):
    img = Image.new("RGB", size, (20, 20, 20))
    out = calibrate.draw_regions(img)
    assert out.size == size and out.mode == "RGB"
    assert out is not img
    assert img.getextrema() == ((20, 20), (20, 20), (20, 20))  # input untouched
    assert out.getextrema() != img.getextrema()
    # Region outlines are drawn on the box edges.
    x0, y0, x1, y1 = region_box(size, "board")
    assert out.getpixel((x0, (y0 + y1) // 2)) != (20, 20, 20)


def test_draw_regions_accepts_other_modes_and_subsets():
    img = Image.new("RGBA", (800, 450), (0, 0, 0, 255))
    out = calibrate.draw_regions(img, names=["gold", "shop"], layout="anchored", chinese=False)
    assert out.size == (800, 450) and out.mode == "RGB"


def test_save_calibration_and_main(tmp_path):
    img = _hud(size=(960, 540))
    raw, annotated = calibrate.save_calibration(img, tmp_path)
    assert raw.exists() and annotated.exists() and raw != annotated
    src = tmp_path / "frame.png"
    img.save(src)
    out_dir = tmp_path / "out"
    assert calibrate.main([str(src), "--out", str(out_dir)]) == 0
    assert any(p.name.endswith("_calibrate.png") for p in out_dir.iterdir())
    assert calibrate.main([str(tmp_path / "nope.png"), "--out", str(out_dir)]) == 1


def test_capture_modules_have_no_em_dash_in_player_text():
    import inspect

    for mod in (regions, screen, change, hotkeys, calibrate):
        src = inspect.getsource(mod)
        assert not any(d in src for d in EM_DASHES), mod.__name__


# --------------------------------------------------------------------------
# regression tests (adversarial review)
# --------------------------------------------------------------------------


def test_bad_layout_env_var_falls_back_instead_of_breaking_every_crop(monkeypatch, capsys):
    good = region_box((1920, 1080), "stage", layout="inscribed")
    for bad in ("bogus", "  ", "ANCHORED "):
        monkeypatch.setenv("TFT_ADVISOR_HUD_LAYOUT", bad)
        box = region_box((1920, 1080), "stage")
        if bad.strip().lower() == "anchored":
            assert box == region_box((1920, 1080), "stage", layout="anchored")
        else:
            assert box == good
    assert "bogus" in capsys.readouterr().err
    with pytest.raises(ValueError):  # an explicit argument is still validated
        region_box((1920, 1080), "stage", layout="bogus")
    img = _hud(size=(640, 360))
    monkeypatch.setenv("TFT_ADVISOR_HUD_LAYOUT", "bogus")
    assert crop(img, "shop").size[0] > 0
    assert RoundWatcher().update(img) == set()


def test_pad_is_validated_and_clamped():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            region_box((1920, 1080), "shop", pad=bad)
    x0, y0, x1, y1 = region_box((1920, 1080), "shop", pad=-5.0)  # clamped to the center line
    assert x1 - x0 == 1 and y1 - y0 == 1
    full = region_box((1920, 1080), "shop")
    assert full[0] <= x0 < full[2] and full[1] <= y0 < full[3]


def test_signature_clamps_explicit_boxes():
    img = _hud(size=(320, 180))
    assert signature(img, (300, 170, 900, 900)).shape == (18, 32)
    assert signature(img, (50, 50, 10, 10)).shape == (18, 32)
    assert signature(img, (-20, -20, 40, 40)).shape == (18, 32)


def _sparkle(img: Image.Image, k: int) -> Image.Image:
    """A looping animation inside the stage region (changes every frame)."""
    out = img.copy()
    x0, y0, x1, y1 = region_box(out.size, "stage")
    sx = x1 - (x1 - x0) // 6
    v = (255, 140, 40)[k % 3]
    ImageDraw.Draw(out).ellipse([sx, y0 + 3, x1 - 3, y1 - 3], fill=(v, v, 90))
    return out


def test_looping_animation_does_not_block_round_detection():
    # Before the fix a looping icon kept the round channel "unstable" forever,
    # so no round change was ever reported while it animated.
    w = RoundWatcher()
    a, b = _hud(stage="3-5"), _hud(stage="3-6")
    events = [w.update(f) for f in [a, a] + [_sparkle(a, k) for k in range(8)]]
    assert all("round_changed" not in e for e in events)  # the animation alone is not a round change
    events = [w.update(_sparkle(b, k)) for k in range(8, 16)]
    assert sum("round_changed" in e for e in events) == 1
    assert 0.0 < w._channels[0].animated_fraction <= change.ANIM_MAX_FRACTION
    # The same stream without masking never fires (documents what the mask is for).
    plain = RoundWatcher()
    plain._channels[0].mask_animated = False
    frames = [a, a] + [_sparkle(a, k) for k in range(8)] + [_sparkle(b, k) for k in range(8, 16)]
    assert not any("round_changed" in plain.update(f) for f in frames)


def test_animation_masking_ignores_scene_changes():
    # Everything changing every frame (combat camera, noise) is not "a small
    # animation": nothing is masked and nothing fires until the picture settles.
    w = RoundWatcher()
    rng = np.random.default_rng(5)
    base = _hud(stage="2-1")
    w.update(base)
    w.update(base)
    for _ in range(6):
        noisy = Image.fromarray(rng.integers(0, 255, (360, 640, 3), dtype=np.uint8))
        assert "round_changed" not in w.update(noisy)
        assert w._channels[0].animated_fraction == 0.0
    nxt = _hud(size=(1280, 720), stage="2-2")
    assert "round_changed" not in w.update(nxt)
    assert "round_changed" in w.update(nxt)


# ---- hotkeys on a fake Win32 API (exercises the real message loop code) ----


class _FakeMsg:
    def __init__(self) -> None:
        self.message = 0
        self.wParam = 0


class _FakeUser32:
    def __init__(self, taken=()) -> None:
        import queue as _queue

        self.taken = set(taken)
        self.registered: dict[int, tuple[int, int]] = {}
        self.unregistered: list[int] = []
        self.loop_thread: list[int] = []
        self.q: "_queue.Queue" = _queue.Queue()
        self.gets = 0
        self.last_error = 0

    def PeekMessageW(self, msg, hwnd, a, b, flag):
        return 0

    def RegisterHotKey(self, hwnd, ident, mods, vk):
        self.loop_thread.append(threading.get_ident())
        key = (mods & ~hotkeys.MOD_NOREPEAT, vk)
        if key in self.taken or key in self.registered.values():
            self.last_error = 1409
            return 0
        assert mods & hotkeys.MOD_NOREPEAT
        self.registered[ident] = key
        return 1

    def UnregisterHotKey(self, hwnd, ident):
        self.unregistered.append(ident)
        return 1

    def GetMessageW(self, msg, hwnd, a, b):
        item = self.q.get()
        self.gets += 1
        if item is None:
            return 0
        msg.message, msg.wParam = hotkeys.WM_HOTKEY, item
        return 1

    def PostThreadMessageW(self, tid, message, wparam, lparam):
        assert message == hotkeys.WM_QUIT
        self.q.put(None)
        return 1

    def press(self, combo: str) -> None:
        key = parse_hotkey(combo)
        ident = next(i for i, k in self.registered.items() if k == key)
        self.q.put(ident)


@pytest.fixture
def fake_win32(monkeypatch):
    user32 = _FakeUser32(taken={(0, 0x76)})  # F7 is taken by "another program"
    api = {
        "user32": user32,
        "kernel32": types.SimpleNamespace(GetCurrentThreadId=lambda: threading.get_ident() & 0xFFFFFFFF),
        "ctypes": types.SimpleNamespace(byref=lambda x: x, get_last_error=lambda: user32.last_error),
        "wintypes": types.SimpleNamespace(MSG=_FakeMsg),
    }
    monkeypatch.setattr(hotkeys, "sys", types.SimpleNamespace(platform="win32"))
    monkeypatch.setattr(hotkeys, "_load_api", lambda: api)
    return user32


def _wait_for(pred, timeout=3.0):
    import time as _time

    end = _time.monotonic() + timeout
    while _time.monotonic() < end:
        if pred():
            return True
        _time.sleep(0.01)
    return pred()


def test_hotkey_manager_registers_dispatches_and_stops(fake_win32):
    logs: list[str] = []
    calls: list[tuple[str, int]] = []
    release = threading.Event()

    def slow() -> None:
        calls.append(("F6", threading.get_ident()))
        release.wait(2.0)

    def boom() -> None:
        raise RuntimeError("kaput")

    bindings = {
        "F6": slow,
        "f6": lambda: calls.append(("dup", 0)),  # same key, other spelling
        "F7": lambda: calls.append(("F7", 0)),  # taken
        "ctrl+shift+a": boom,
        "F9": lambda: calls.append(("F9", threading.get_ident())),
    }
    mgr = HotkeyManager(bindings, log=logs.append)
    assert mgr.start() is False  # not every binding could be registered
    assert mgr.running
    assert sorted(mgr.registered) == ["F6", "F9", "ctrl+shift+a"]
    assert set(mgr.failed) == {"f6", "F7"}
    assert "F6" in mgr.failed["f6"] and "占用" in mgr.failed["F7"]
    assert len(set(fake_win32.loop_thread)) == 1  # registered on the loop thread
    assert fake_win32.loop_thread[0] != threading.get_ident()

    fake_win32.press("F6")  # blocks the callback worker ...
    fake_win32.press("ctrl+shift+a")  # ... raises
    fake_win32.press("F9")
    assert _wait_for(lambda: fake_win32.gets >= 3)  # the message loop kept pumping
    assert [c[0] for c in calls] == ["F6"]
    release.set()
    assert _wait_for(lambda: [c[0] for c in calls] == ["F6", "F9"])
    assert calls[0][1] != fake_win32.loop_thread[0]  # callbacks never run on the loop thread
    assert any("kaput" in line for line in logs)

    mgr.stop()
    assert not mgr.running
    assert sorted(fake_win32.unregistered) == sorted(fake_win32.registered)
    assert not any(d in line for line in logs for d in EM_DASHES)


def test_hotkey_manager_all_taken_returns_false_and_cleans_up(fake_win32):
    fake_win32.taken |= {(0, 0x75)}
    logs: list[str] = []
    mgr = HotkeyManager({"F6": lambda: None, "F7": lambda: None}, log=logs.append)
    assert mgr.start() is False
    assert mgr.registered == [] and set(mgr.failed) == {"F6", "F7"}
    assert _wait_for(lambda: not mgr.running)
    assert mgr._worker is None
    mgr.stop()


def test_parse_hotkey_rejects_non_ascii_digits():
    for bad in ("f²", "F١"):
        with pytest.raises(ValueError, match="unknown"):
            parse_hotkey(bad)


# ---- screen capture: minimized window, thread churn, paths, saving ----


def _fake_window(monkeypatch, state: dict) -> None:
    monkeypatch.setattr(screen, "_on_windows", lambda: True)
    monkeypatch.setattr(
        screen,
        "_find_game_hwnd",
        lambda title, process_names=screen.KNOWN_PROCESS_NAMES, include_minimized=False: (
            42 if state["exists"] and (include_minimized or not state["minimized"]) else None
        ),
    )
    monkeypatch.setattr(screen, "_is_minimized", lambda hwnd: bool(hwnd) and state["exists"] and state["minimized"])
    monkeypatch.setattr(
        screen, "_client_rect", lambda hwnd: (10, 5, 120, 60) if hwnd and state["exists"] and not state["minimized"] else None
    )


def test_minimized_game_window_pauses_capture_instead_of_grabbing_desktop(fake_mss, monkeypatch):
    state = {"exists": True, "minimized": False}
    _fake_window(monkeypatch, state)
    logs: list[str] = []
    cap = ScreenCapturer(CaptureConfig(use_window=True), log=logs.append)
    img = cap.grab()
    assert img is not None and img.size == (120, 60) and cap.last_source == "window"
    assert fake_mss.instances[0].grabs[-1] == {"left": 10, "top": 5, "width": 120, "height": 60}

    state["minimized"] = True
    n_grabs = len(fake_mss.instances[0].grabs)
    assert cap.grab() is None and cap.grab() is None
    assert len(fake_mss.instances[0].grabs) == n_grabs  # the desktop was not captured
    assert cap.last_source == "minimized" and "最小化" in (cap.last_error or "")
    assert sum("最小化" in line for line in logs) == 1  # logged once, not every poll

    state["minimized"] = False
    img = cap.grab()
    assert img is not None and img.size == (120, 60) and cap.last_error is None

    # Minimized when the capturer starts: still detected (not the monitor).
    state["minimized"] = True
    cap2 = ScreenCapturer(CaptureConfig(use_window=True))
    assert cap2.grab() is None and cap2.last_source == "minimized"
    # Game not running at all: monitor fallback as specified.
    state["exists"] = False
    cap3 = ScreenCapturer(CaptureConfig(use_window=True))
    img = cap3.grab()
    assert img is not None and img.size == (160, 90) and cap3.last_source == "monitor"
    for c in (cap, cap2, cap3):
        c.close()


def test_mss_instances_of_finished_threads_are_released(fake_mss):
    cap = ScreenCapturer(CaptureConfig())
    for _ in range(6):  # e.g. short lived helper threads grabbing a frame
        t = threading.Thread(target=cap.grab)
        t.start()
        t.join()
    assert len(cap._instances) <= 1
    assert sum(not inst.closed for inst in fake_mss.instances) <= 1
    cap.close()
    assert all(inst.closed for inst in fake_mss.instances)
    assert cap.grab() is None and cap.last_error is None


def test_file_capturer_literal_path_with_glob_characters(tmp_path):
    folder = tmp_path / "录像[旧]"
    folder.mkdir()
    _write_png(folder / "a.png", (9, 9, 9))
    assert len(FileCapturer([folder])) == 1
    assert len(FileCapturer([folder / "a.png"])) == 1
    assert FileCapturer([str(folder / "a.png")]).grab().getpixel((0, 0)) == (9, 9, 9)


def test_save_frame_concurrent_saves_never_overwrite(tmp_path):
    img = Image.new("RGB", (8, 8), (4, 5, 6))
    paths: list = []
    lock = threading.Lock()
    start = threading.Barrier(8)

    def run() -> None:
        start.wait()
        p = save_frame(img, tmp_path, "scout")
        with lock:
            paths.append(p)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(paths)) == 8 and all(p.exists() for p in paths)
    assert len(list(tmp_path.iterdir())) == 8
    with pytest.raises(TypeError):
        save_frame(None, tmp_path, "x")


def test_calibrate_main_reports_capture_error_once(tmp_path, monkeypatch, capsys):
    class Broken:
        def __init__(self) -> None:
            raise RuntimeError("no screen")

    module = types.ModuleType("mss")
    module.MSS = Broken
    monkeypatch.setitem(sys.modules, "mss", module)
    monkeypatch.setattr(screen, "_on_windows", lambda: False)
    monkeypatch.setenv("TFT_ADVISOR_CONFIG", str(tmp_path / "missing.toml"))
    assert calibrate.main(["--out", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "no screen" in out and "截图失败：截图失败" not in out
    assert not any(d in out for d in EM_DASHES)
