"""Mock perceiver: replays ``ScreenObservation`` JSON files (demo mode and tests).

Accepted inputs:

* a list of ``ScreenObservation`` (or plain dicts),
* a directory: every ``*.json`` file in name order, each holding one observation
  (or a list of them),
* a single ``.json`` file holding one observation or a list of them.

A JSON object may also be a serialized ``Observation`` (with a ``screen`` key);
only its ``screen`` part is replayed.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from pydantic import ValidationError

from ..models import Observation, ScreenObservation
from .base import PerceptionError, PerceptionHint

ObservationSource = Union[Iterable[Union[ScreenObservation, dict]], str, Path]


def _to_screen(obj: Any, where: str) -> ScreenObservation:
    if isinstance(obj, ScreenObservation):
        return obj.model_copy(deep=True)
    if isinstance(obj, Observation):
        return obj.screen.model_copy(deep=True)
    if isinstance(obj, dict):
        payload = obj.get("screen") if isinstance(obj.get("screen"), dict) else obj
        try:
            return ScreenObservation.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"invalid observation in {where}: {exc}") from exc
    raise ValueError(f"unsupported observation type in {where}: {type(obj).__name__}")


def _load_json_file(path: Path) -> list[ScreenObservation]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    items = data if isinstance(data, list) else [data]
    return [_to_screen(obj, f"{path.name}[{i}]") for i, obj in enumerate(items)]


def load_observations(source: ObservationSource) -> list[ScreenObservation]:
    """Load observations from a list, a directory of ``*.json`` files or one ``.json`` file."""
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if path.is_dir():
            files = sorted(p for p in path.glob("*.json") if p.is_file())
            out: list[ScreenObservation] = []
            for f in files:
                out.extend(_load_json_file(f))
            return out
        if path.is_file():
            return _load_json_file(path)
        raise FileNotFoundError(f"observation source not found: {path}")
    return [_to_screen(obj, f"item {i}") for i, obj in enumerate(source)]


class MockPerceiver:
    """Returns the next prepared observation on each ``perceive`` call (image is ignored)."""

    def __init__(self, observations: ObservationSource, loop: bool = False, name: str = "mock") -> None:
        self.name = name
        self.loop = loop
        self.observations: list[ScreenObservation] = load_observations(observations)
        self._index = 0
        self._lock = threading.Lock()
        # (purpose, hint) of the most recent calls, for tests and debugging.
        self.calls: deque[tuple[str, Optional[PerceptionHint]]] = deque(maxlen=100)

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def remaining(self) -> int:
        if self.loop and self.observations:
            return len(self.observations)
        return max(0, len(self.observations) - self._index)

    def reset(self) -> None:
        with self._lock:
            self._index = 0

    def perceive(self, image: Any = None, purpose: str = "auto", hint: Optional[PerceptionHint] = None) -> Observation:
        started = time.monotonic()
        with self._lock:
            self.calls.append((purpose, hint))
            if not self.observations:
                raise PerceptionError("模拟数据为空")
            if self._index >= len(self.observations):
                if not self.loop:
                    raise PerceptionError("模拟数据已全部播放完毕")
                self._index = 0
            screen = self.observations[self._index].model_copy(deep=True)
            self._index += 1
        return Observation(
            screen=screen,
            source="mock",
            purpose=purpose or "auto",
            latency_s=round(time.monotonic() - started, 3),
        )


__all__ = ["MockPerceiver", "load_observations"]
