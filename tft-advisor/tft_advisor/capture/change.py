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
"""

from __future__ import annotations

import threading
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
    else:
        box = tuple(int(v) for v in region)
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


Metric = Callable[[Optional[np.ndarray], Optional[np.ndarray]], float]


@dataclass
class _Channel:
    event: str
    region: RegionSpec
    size: tuple[int, int]
    metric: Metric
    threshold: float
    baseline: Optional[np.ndarray] = None
    prev: Optional[np.ndarray] = None
    run: int = 0  # length (in frames) of the current stable streak
    last_step: float = 0.0  # change vs previous frame
    last_drift: float = 0.0  # change vs baseline
    fired: int = 0

    def reset(self) -> None:
        self.baseline = None
        self.prev = None
        self.run = 0
        self.last_step = 0.0
        self.last_drift = 0.0

    def update(self, sig: np.ndarray, stable_frames: int) -> bool:
        if self.prev is None:
            self.prev, self.run = sig, 1
            if stable_frames <= 1:
                self.baseline = sig
            return False
        self.last_step = self.metric(sig, self.prev)
        self.run = self.run + 1 if self.last_step < self.threshold / 2.0 else 1
        self.prev = sig
        if self.run < stable_frames:
            return False
        if self.baseline is None:  # first stable view: baseline only
            self.baseline = sig
            return False
        self.last_drift = self.metric(sig, self.baseline)
        self.baseline = sig  # follow the stable view (slow drifts never accumulate)
        if self.last_drift > self.threshold:
            self.fired += 1
            return True
        return False


class RoundWatcher:
    """Debounced ``round_changed`` / ``shop_changed`` / ``screen_changed`` events.

    ``update(img)`` returns the set of events that fired for this frame
    (usually empty). Thread safe; ``reset()`` forgets all baselines (new game).
    ``last_scores`` exposes the latest drift per event for diagnostics.
    """

    def __init__(
        self,
        threshold: float = 0.08,
        stable_frames: int = 2,
        *,
        round_threshold: Optional[float] = None,
        shop_threshold: Optional[float] = None,
        screen_threshold: Optional[float] = None,
    ) -> None:
        self.threshold = float(threshold)
        self.stable_frames = max(1, int(stable_frames))
        t = self.threshold
        self._channels: list[_Channel] = [
            # stage region is ~0.09 x 0.038 of the frame (about 5:1): 48 x 12 cells of ~3.5 px at 1080p.
            _Channel(ROUND_CHANGED, "stage", (48, 12), local_diff, float(round_threshold if round_threshold is not None else t)),
            # shop region is ~0.66 x 0.155 of the frame (about 7.5:1).
            _Channel(SHOP_CHANGED, "shop", (64, 10), diff, float(shop_threshold if shop_threshold is not None else t)),
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
