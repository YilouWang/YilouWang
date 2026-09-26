"""Cheap change detection on screenshots (no LLM, a few ms per frame).

``signature`` reduces a frame (or one HUD region) to a small grayscale
thumbnail; ``diff`` is the mean absolute difference of two signatures.

``RoundWatcher`` turns a stream of frames into debounced events:

* ``round_changed``: the round indicator (``stage`` region) changed,
* ``shop_changed``: the shop (``shop`` region) changed, e.g. after a roll,
* ``screen_changed``: the whole frame changed (combat -> planning, the camera
  moved to another board, carousel, augment overlay ...).

Each channel keeps the signature of the last *stable* view. An event fires
once when the view has moved away from that baseline by more than
``threshold`` and has then stayed stable (frame to frame change below
``threshold / 2``) for ``stable_frames`` consecutive frames, so animations and
camera pans do not fire until the picture settles. While a view is stable the
baseline follows it, so slow drifts (a draining timer, idle animations) never
accumulate into an event. The very first stable view only sets the baseline.

The round indicator is a few small digits: a "3-5" -> "3-6" change moves the
mean of the stage region by well under 1 %, which a mean-based diff cannot
tell from noise. The ``round_changed`` channel therefore uses ``local_diff``
(mean of the largest per-cell differences) on a finer thumbnail, with the
same threshold semantics.

Shop cards share frames, name banners and trait rows, so a full roll moves
the shop's global mean by less than 0.08 (about 0.075 on a real Set 18 frame),
barely more than buying two cards. The ``shop_changed`` channel uses
``column_diff`` (median over thumbnail columns of the per-column change):
a roll changes nearly every card column (0.055 to 0.075 on the same frame),
buying one or two cards or hovering a card changes a minority of columns and
scores 0. Its default threshold is ``0.4 * threshold``.

Round text and shop rolls never legitimately return to a view seen a moment
ago, while a pulsing icon or a hover highlight does. The ``round_changed`` and
``shop_changed`` channels therefore remember the last few distinct stable
views and only fire for a view that differs from all of them (a glow cycling
between two states fires at most once). ``screen_changed`` has no such memory:
going back to your own board after scouting is a real change.

A looping animation (a glowing current-round icon, a sparkle) changes the
same cells on every frame. With ``local_diff`` that alone would keep the
round channel "unstable" forever and no round change could ever fire. The
round channel therefore ignores cells that changed in at least
``ANIM_MIN_CHANGES`` of the last ``ANIM_WINDOW`` frames, as long as they are
a small part of the region (at most ``ANIM_MAX_FRACTION``; beyond that it is
a scene change, not a local animation). A digit changes once, so it is never
masked.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np
from PIL import Image

from .regions import region_box

RegionSpec = Union[str, tuple[int, int, int, int], None]

ROUND_CHANGED = "round_changed"
SHOP_CHANGED = "shop_changed"
SCREEN_CHANGED = "screen_changed"
EVENTS = (ROUND_CHANGED, SHOP_CHANGED, SCREEN_CHANGED)

#: Animated-cell masking (round channel): frames looked at, changes needed, max masked share.
ANIM_WINDOW = 4
ANIM_MIN_CHANGES = 3
ANIM_MAX_FRACTION = 0.25


def signature(img: Image.Image, region: RegionSpec = None, size: tuple[int, int] = (32, 18)) -> np.ndarray:
    """Grayscale thumbnail of the frame or of one region, float32 in 0..1, shape (h, w).

    ``region`` is a region name from ``regions.REGIONS``, a pixel box
    ``(x0, y0, x1, y1)`` or None for the whole frame. Area averaging (BOX)
    keeps it stable against single-pixel noise.
    """
    if region is None:
        box = None
    elif isinstance(region, str):
        box = region_box(img.size, region)
    else:  # explicit pixel box: clamp to the frame, never empty
        iw, ih = img.size
        x0, y0, x1, y1 = (int(v) for v in region)
        x0, y0 = min(max(x0, 0), max(iw - 1, 0)), min(max(y0, 0), max(ih - 1, 0))
        box = (x0, y0, min(max(x1, x0 + 1), iw), min(max(y1, y0 + 1), ih))
    w, h = max(1, int(size[0])), max(1, int(size[1]))
    src = img if img.mode in ("L", "RGB") else img.convert("RGB")
    # Resize straight from the source box (no full-size crop / convert copy), then gray.
    thumb = src.resize((w, h), Image.Resampling.BOX, box=box)
    if thumb.mode != "L":
        thumb = thumb.convert("L")
    return np.asarray(thumb, dtype=np.float32) / 255.0


def diff(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Mean absolute difference (0..1). Missing or shape-mismatched signatures count as 1.0."""
    if a is None or b is None or a.shape != b.shape:
        return 1.0
    return float(np.mean(np.abs(a - b)))


def local_diff(a: Optional[np.ndarray], b: Optional[np.ndarray], fraction: float = 0.03, min_cells: int = 4) -> float:
    """Mean of the largest per-cell differences (top ``fraction`` of cells, at least ``min_cells``).

    Sensitive to small, high-contrast changes (digits, icons) that a global
    mean dilutes, while a single noisy cell cannot reach a meaningful value.
    """
    if a is None or b is None or a.shape != b.shape:
        return 1.0
    d = np.abs(a - b).ravel()
    if d.size == 0:
        return 0.0
    k = int(min(d.size, max(min_cells, round(fraction * d.size))))
    top = np.partition(d, d.size - k)[d.size - k :]
    return float(np.mean(top))


def column_diff(a: Optional[np.ndarray], b: Optional[np.ndarray], quantile: float = 0.5) -> float:
    """Quantile (default median) over thumbnail columns of the mean per-column change.

    High only when most of the width changed (every shop card after a roll),
    zero when a minority of columns changed (one bought card, a hover glow).
    """
    if a is None or b is None or a.shape != b.shape:
        return 1.0
    if a.size == 0:
        return 0.0
    cols = np.abs(a - b).reshape(a.shape[0], -1).mean(axis=0)
    return float(np.quantile(cols, quantile))


Metric = Callable[[Optional[np.ndarray], Optional[np.ndarray]], float]


@dataclass
class _Channel:
    event: str
    region: RegionSpec
    size: tuple[int, int]
    metric: Metric
    threshold: float
    history_size: int = 0  # remembered distinct stable views (0 = no novelty check)
    mask_animated: bool = False  # ignore small, persistently changing cells (looping animations)
    baseline: Optional[np.ndarray] = None
    history: deque = field(default_factory=deque)
    changes: deque = field(default_factory=lambda: deque(maxlen=ANIM_WINDOW))
    animated_fraction: float = 0.0  # share of cells currently masked as animated
    prev: Optional[np.ndarray] = None
    run: int = 0  # length (in frames) of the current stable streak
    last_step: float = 0.0  # change vs previous frame
    last_drift: float = 0.0  # change vs baseline
    fired: int = 0

    def reset(self) -> None:
        self.baseline = None
        self.history.clear()
        self.prev = None
        self.run = 0
        self.last_step = 0.0
        self.last_drift = 0.0
        self.changes.clear()
        self.animated_fraction = 0.0

    def _animated_mask(self) -> Optional[np.ndarray]:
        """Cells that kept changing over the last frames (None = mask nothing)."""
        self.animated_fraction = 0.0
        if not self.mask_animated or len(self.changes) < ANIM_MIN_CHANGES:
            return None
        mask = np.sum(np.stack(tuple(self.changes)), axis=0) >= ANIM_MIN_CHANGES
        frac = float(mask.mean()) if mask.size else 0.0
        if frac == 0.0 or frac > ANIM_MAX_FRACTION:
            return None
        self.animated_fraction = frac
        return mask

    def _cmp(self, sig: np.ndarray, ref: Optional[np.ndarray], mask: Optional[np.ndarray]) -> float:
        """``metric(sig, ref)`` with the animated cells treated as unchanged."""
        if mask is not None and ref is not None and ref.shape == sig.shape:
            sig = np.where(mask, ref, sig)
        return self.metric(sig, ref)

    def _remember(self, sig: np.ndarray, mask: Optional[np.ndarray]) -> None:
        if self.history_size <= 0:
            return
        if not self.history or self._cmp(sig, self.history[-1], mask) >= self.threshold / 2.0:
            self.history.append(sig)
            while len(self.history) > self.history_size:
                self.history.popleft()

    def _novel(self, sig: np.ndarray, mask: Optional[np.ndarray]) -> bool:
        return all(self._cmp(sig, old, mask) > self.threshold for old in self.history)

    def update(self, sig: np.ndarray, stable_frames: int) -> bool:
        if self.prev is None:
            self.prev, self.run = sig, 1
            self.changes.clear()
            if stable_frames <= 1:
                self.baseline = sig
                self._remember(sig, None)
            return False
        mask = self._animated_mask()  # from the frames before this one
        if self.mask_animated:
            if self.prev.shape == sig.shape:
                self.changes.append(np.abs(sig - self.prev) >= self.threshold / 2.0)
            else:
                self.changes.clear()
        self.last_step = self._cmp(sig, self.prev, mask)
        self.run = self.run + 1 if self.last_step < self.threshold / 2.0 else 1
        self.prev = sig
        if self.run < stable_frames:
            return False
        if self.baseline is None:  # first stable view: baseline only
            self.baseline = sig
            self._remember(sig, mask)
            return False
        self.last_drift = self._cmp(sig, self.baseline, mask)
        self.baseline = sig  # follow the stable view (slow drifts never accumulate)
        fire = self.last_drift > self.threshold and self._novel(sig, mask)
        self._remember(sig, mask)
        if fire:
            self.fired += 1
        return fire


class RoundWatcher:
    """Debounced ``round_changed`` / ``shop_changed`` / ``screen_changed`` events.

    ``update(img)`` returns the set of events that fired for this frame
    (usually empty). Thread safe; ``reset()`` forgets all baselines (new game).
    ``last_scores`` exposes the latest drift per event for diagnostics.

    Optional per-channel thresholds override ``threshold``; ``history`` is how
    many distinct stable views the round / shop channels remember (0 disables
    the novelty check).
    """

    def __init__(
        self,
        threshold: float = 0.08,
        stable_frames: int = 2,
        *,
        round_threshold: Optional[float] = None,
        shop_threshold: Optional[float] = None,
        screen_threshold: Optional[float] = None,
        history: int = 4,
    ) -> None:
        self.threshold = float(threshold)
        self.stable_frames = max(1, int(stable_frames))
        t = self.threshold
        self._channels: list[_Channel] = [
            # stage region is ~0.09 x 0.038 of the frame (about 5:1): 48 x 12 cells of ~3.5 px at 1080p.
            _Channel(
                ROUND_CHANGED, "stage", (48, 12), local_diff,
                float(round_threshold if round_threshold is not None else t), history_size=history,
                mask_animated=True,
            ),
            # shop region is ~0.66 x 0.155 of the frame (about 7.5:1): 64 columns, ~13 per card.
            _Channel(
                SHOP_CHANGED, "shop", (64, 10), column_diff,
                float(shop_threshold if shop_threshold is not None else 0.4 * t), history_size=history,
            ),
            _Channel(SCREEN_CHANGED, None, (32, 18), diff, float(screen_threshold if screen_threshold is not None else t)),
        ]
        self._lock = threading.Lock()
        self.frames = 0

    def update(self, img: Optional[Image.Image]) -> set[str]:
        if img is None:
            return set()
        with self._lock:
            self.frames += 1
            events: set[str] = set()
            for ch in self._channels:
                sig = signature(img, ch.region, ch.size)
                if ch.update(sig, self.stable_frames):
                    events.add(ch.event)
            return events

    def reset(self) -> None:
        with self._lock:
            for ch in self._channels:
                ch.reset()
            self.frames = 0

    @property
    def last_scores(self) -> dict[str, float]:
        with self._lock:
            return {ch.event: round(ch.last_drift, 4) for ch in self._channels}

    @property
    def counts(self) -> dict[str, int]:
        """How many times each event fired since construction."""
        with self._lock:
            return {ch.event: ch.fired for ch in self._channels}
