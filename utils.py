from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class FPSCounter:
    window: int = 30
    samples: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    last_time: float = field(default_factory=time.perf_counter)

    def update(self) -> float:
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.samples.append(1.0 / dt)
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)
