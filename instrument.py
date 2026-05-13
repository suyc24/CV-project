from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import config


@dataclass(frozen=True)
class Zone:
    label: str
    sound_id: str
    x1: int
    y1: int
    x2: int
    y2: int
    kind: str
    press_ratio: float = config.PRESS_RATIO
    release_ratio: float = config.RELEASE_RATIO

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)

    @property
    def press_y(self) -> float:
        return self.y1 + self.press_ratio * self.height

    @property
    def release_y(self) -> float:
        return self.y1 + self.release_ratio * self.height

    def contains(self, point: Tuple[int, int]) -> bool:
        x, y = point
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class InstrumentLayout:
    """Builds simple screen-space drum pads or piano keys inside the desk ROI."""

    DRUM_LABELS = ["KICK", "SNARE", "HIHAT", "TOM1", "TOM2", "CRASH"]
    PIANO_LABELS = [
        "C4", "D4", "E4", "F4", "G4", "A4", "B4",
        "C5", "D5", "E5", "F5", "G5", "A5", "B5", "C6",
    ]

    def __init__(self, mode: str = "drum", roi_ratios: Tuple[float, float, float, float] | None = None) -> None:
        self.mode = mode
        self.roi_ratios = roi_ratios

    def set_mode(self, mode: str) -> None:
        if mode not in {"drum", "piano"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode

    def toggle_mode(self) -> str:
        self.mode = "piano" if self.mode == "drum" else "drum"
        return self.mode

    def set_roi_ratios(self, roi_ratios: Tuple[float, float, float, float] | None) -> None:
        self.roi_ratios = roi_ratios

    def get_zones(self, frame_shape: Tuple[int, int, int]) -> List[Zone]:
        height, width = frame_shape[:2]
        if self.mode == "piano":
            roi = self._piano_roi(width, height)
            return self._piano_zones(roi)
        roi = self._roi(width, height)
        return self._drum_zones(roi)

    def zone_at(self, zones: Iterable[Zone], point: Tuple[int, int]) -> Optional[Zone]:
        for zone in zones:
            if zone.contains(point):
                return zone
        return None

    def _roi(self, width: int, height: int) -> Tuple[int, int, int, int]:
        if self.roi_ratios is not None:
            x_min, y_min, x_max, y_max = self.roi_ratios
            return (
                int(width * x_min),
                int(height * y_min),
                int(width * x_max),
                int(height * y_max),
            )
        return (
            int(width * config.ROI_X_MIN),
            int(height * config.ROI_Y_MIN),
            int(width * config.ROI_X_MAX),
            int(height * config.ROI_Y_MAX),
        )

    def _piano_roi(self, width: int, height: int) -> Tuple[int, int, int, int]:
        if self.roi_ratios is not None:
            return self._roi(width, height)
        return (
            int(width * config.PIANO_ROI_X_MIN),
            int(height * config.PIANO_ROI_Y_MIN),
            int(width * config.PIANO_ROI_X_MAX),
            int(height * config.PIANO_ROI_Y_MAX),
        )

    def _drum_zones(self, roi: Tuple[int, int, int, int]) -> List[Zone]:
        x1, y1, x2, y2 = roi
        gap = 14
        cols, rows = 3, 2
        cell_w = (x2 - x1 - gap * (cols + 1)) // cols
        cell_h = (y2 - y1 - gap * (rows + 1)) // rows
        zones: List[Zone] = []
        for idx, label in enumerate(self.DRUM_LABELS):
            row = idx // cols
            col = idx % cols
            zx1 = x1 + gap + col * (cell_w + gap)
            zy1 = y1 + gap + row * (cell_h + gap)
            zones.append(Zone(label, label.lower(), zx1, zy1, zx1 + cell_w, zy1 + cell_h, "drum"))
        return zones

    def _piano_zones(self, roi: Tuple[int, int, int, int]) -> List[Zone]:
        x1, y1, x2, y2 = roi
        piano_height = int((y2 - y1) * config.PIANO_AREA_HEIGHT_RATIO)
        y1 = y2 - max(80, piano_height)
        key_count = len(self.PIANO_LABELS)
        key_w = (x2 - x1) // key_count
        zones: List[Zone] = []
        for idx, label in enumerate(self.PIANO_LABELS):
            zx1 = x1 + idx * key_w
            zx2 = x2 if idx == key_count - 1 else zx1 + key_w
            zones.append(
                Zone(
                    label,
                    label.lower(),
                    zx1,
                    y1,
                    zx2,
                    y2,
                    "piano",
                    press_ratio=config.PIANO_PRESS_RATIO,
                    release_ratio=config.PIANO_RELEASE_RATIO,
                )
            )
        return zones
