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
    polygon: Optional[Tuple[Tuple[int, int], ...]] = None

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        if self.polygon:
            x = sum(point[0] for point in self.polygon) / len(self.polygon)
            y = sum(point[1] for point in self.polygon) / len(self.polygon)
            return (int(x), int(y))
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)

    @property
    def press_y(self) -> float:
        return self.y1 + self.press_ratio * self.height

    @property
    def release_y(self) -> float:
        return self.y1 + self.release_ratio * self.height

    def contains(self, point: Tuple[int, int]) -> bool:
        if self.polygon:
            return _point_in_polygon(point, self.polygon)
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
        width = x2 - x1
        height = y2 - y1
        top_inset = int(width * config.PIANO_PLANE_TOP_INSET_RATIO)
        top_lift = int(height * config.PIANO_PLANE_TOP_LIFT_RATIO)
        top_y = max(0, y1 - top_lift)
        top_left = (x1 + top_inset, top_y)
        top_right = (x2 - top_inset, top_y)
        bottom_left = (x1, y2)
        bottom_right = (x2, y2)
        zones: List[Zone] = []
        for idx, label in enumerate(self.PIANO_LABELS):
            t0 = idx / key_count
            t1 = (idx + 1) / key_count
            poly = (
                _lerp_point(top_left, top_right, t0),
                _lerp_point(top_left, top_right, t1),
                _lerp_point(bottom_left, bottom_right, t1),
                _lerp_point(bottom_left, bottom_right, t0),
            )
            xs = [point[0] for point in poly]
            ys = [point[1] for point in poly]
            zones.append(
                Zone(
                    label,
                    label.lower(),
                    min(xs),
                    min(ys),
                    max(xs),
                    max(ys),
                    "piano",
                    press_ratio=config.PIANO_PRESS_RATIO,
                    release_ratio=config.PIANO_RELEASE_RATIO,
                    polygon=poly,
                )
            )
        return zones


def _lerp_point(a: Tuple[int, int], b: Tuple[int, int], t: float) -> Tuple[int, int]:
    return (int(round(a[0] + (b[0] - a[0]) * t)), int(round(a[1] + (b[1] - a[1]) * t)))


def _point_in_polygon(point: Tuple[int, int], polygon: Tuple[Tuple[int, int], ...]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, pi in enumerate(polygon):
        xi, yi = pi
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            denominator = yj - yi
            if abs(denominator) < 1e-9:
                j = i
                continue
            x_at_y = (xj - xi) * (y - yi) / denominator + xi
            if x <= x_at_y:
                inside = not inside
        j = i
    return inside
