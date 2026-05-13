from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Deque, Dict, Iterable, List, Optional, Tuple

import config
from instrument import Zone
from utils import clamp

if TYPE_CHECKING:
    from hand_tracker import HandLandmarks


@dataclass
class HitEvent:
    note_id: str
    sound_id: str
    zone_label: str
    finger_id: int
    hand_id: int
    timestamp: float
    velocity: float
    volume: float


@dataclass
class FingerState:
    previous_position: Optional[Tuple[int, int]] = None
    previous_timestamp: Optional[float] = None
    smoothed_velocity_y: float = 0.0
    is_pressed: bool = False
    pressed_zone_id: Optional[str] = None
    last_hit_time: float = -999.0
    trail: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=config.TRAIL_LENGTH))


FINGER_NAMES = {
    4: "thumb",
    8: "index",
    12: "middle",
    16: "ring",
    20: "pinky",
}


class HitDetector:
    def __init__(self, finger_ids: Iterable[int] = config.FINGER_TIP_IDS) -> None:
        self.finger_ids = tuple(finger_ids)
        self._states: Dict[Tuple[int, int], FingerState] = {}
        self._diagnostics: List[Dict[str, object]] = []

    def reset(self) -> None:
        self._states.clear()
        self._diagnostics.clear()

    def update(self, hands: Iterable["HandLandmarks"], zones: List[Zone], current_time: float) -> List[HitEvent]:
        hits: List[HitEvent] = []
        self._diagnostics = []
        for hand in hands:
            for finger_id in self.finger_ids:
                if finger_id >= len(hand.landmarks):
                    continue
                state = self._states.setdefault((hand.hand_id, finger_id), FingerState())
                x, y, _ = hand.landmarks[finger_id]
                position = (x, y)
                state.trail.append(position)
                previous_position = state.previous_position
                velocity_y = self._update_velocity(state, position, current_time)
                zone = self._zone_at(zones, position)
                self._update_release_state(state, zone, y)

                reason = self._miss_reason(state, zone, y, velocity_y, current_time, previous_position)
                if zone and reason == "hit":
                    volume = self._velocity_to_volume(velocity_y)
                    hit = HitEvent(
                        note_id=zone.label,
                        sound_id=zone.sound_id,
                        zone_label=zone.label,
                        finger_id=finger_id,
                        hand_id=hand.hand_id,
                        timestamp=current_time,
                        velocity=velocity_y,
                        volume=volume,
                    )
                    hits.append(hit)
                    state.is_pressed = True
                    state.pressed_zone_id = zone.sound_id
                    state.last_hit_time = current_time
                self._diagnostics.append(
                    {
                        "hand_id": hand.hand_id,
                        "finger_id": finger_id,
                        "finger_name": FINGER_NAMES.get(finger_id, f"F{finger_id}"),
                        "x": x,
                        "y": y,
                        "velocity_y": velocity_y,
                        "zone_label": zone.label if zone else None,
                        "zone_kind": zone.kind if zone else None,
                        "pressed": state.is_pressed,
                        "reason": reason,
                        "threshold": self._threshold_for(zone) if zone else None,
                        "press_y": zone.press_y if zone else None,
                    }
                )
        return hits

    def get_trails(self) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        return {key: list(state.trail) for key, state in self._states.items()}

    def diagnostics(self) -> List[Dict[str, object]]:
        return list(self._diagnostics)

    def debug_snapshot(self) -> List[str]:
        lines: List[str] = []
        for (hand_id, finger_id), state in sorted(self._states.items()):
            if abs(state.smoothed_velocity_y) < 80 and not state.is_pressed:
                continue
            name = FINGER_NAMES.get(finger_id, f"F{finger_id}")
            lines.append(
                f"H{hand_id} {name}: vy={state.smoothed_velocity_y:6.0f} "
                f"pressed={int(state.is_pressed)}"
            )
        return lines[:10]

    def _update_velocity(self, state: FingerState, position: Tuple[int, int], current_time: float) -> float:
        if state.previous_position is None or state.previous_timestamp is None:
            state.previous_position = position
            state.previous_timestamp = current_time
            return state.smoothed_velocity_y

        dt = max(1e-3, current_time - state.previous_timestamp)
        raw_velocity_y = (position[1] - state.previous_position[1]) / dt
        alpha = config.VELOCITY_SMOOTHING_ALPHA
        state.smoothed_velocity_y = alpha * raw_velocity_y + (1.0 - alpha) * state.smoothed_velocity_y
        state.previous_position = position
        state.previous_timestamp = current_time
        return state.smoothed_velocity_y

    def _update_release_state(self, state: FingerState, zone: Optional[Zone], finger_y: int) -> None:
        if not state.is_pressed:
            return
        if zone is None:
            state.is_pressed = False
            state.pressed_zone_id = None
            return
        if state.pressed_zone_id and zone.sound_id != state.pressed_zone_id:
            state.is_pressed = False
            state.pressed_zone_id = None
            return
        if finger_y < zone.release_y:
            state.is_pressed = False
            state.pressed_zone_id = None

    def _is_hit_candidate(
        self,
        state: FingerState,
        zone: Zone,
        finger_y: int,
        velocity_y: float,
        current_time: float,
    ) -> bool:
        if state.is_pressed:
            return False
        if current_time - state.last_hit_time < config.HIT_COOLDOWN:
            return False
        threshold = self._threshold_for(zone)
        return velocity_y > threshold and finger_y > zone.press_y

    def _miss_reason(
        self,
        state: FingerState,
        zone: Optional[Zone],
        finger_y: int,
        velocity_y: float,
        current_time: float,
        previous_position: Optional[Tuple[int, int]],
    ) -> str:
        if zone is None:
            return "no_zone"
        if state.is_pressed:
            return "pressed"
        if current_time - state.last_hit_time < config.HIT_COOLDOWN:
            return "cooldown"
        if finger_y <= zone.press_y:
            return "press_line"
        if zone.kind == "piano":
            crossed = previous_position is not None and previous_position[1] <= zone.press_y < finger_y
            if crossed and velocity_y > config.PIANO_CROSSING_VELOCITY_THRESHOLD:
                return "hit"
            if velocity_y > self._threshold_for(zone):
                return "hit"
            if crossed:
                return "crossing_velocity"
            return "velocity"
        if velocity_y <= self._threshold_for(zone):
            return "velocity"
        return "hit"

    def _threshold_for(self, zone: Zone) -> float:
        return config.PIANO_HIT_VELOCITY_THRESHOLD if zone.kind == "piano" else config.HIT_VELOCITY_THRESHOLD

    def _velocity_to_volume(self, velocity_y: float) -> float:
        normalized = (velocity_y - config.HIT_MIN_VELOCITY) / (config.HIT_MAX_VELOCITY - config.HIT_MIN_VELOCITY)
        return clamp(normalized, 0.2, 1.0)

    def _zone_at(self, zones: Iterable[Zone], point: Tuple[int, int]) -> Optional[Zone]:
        piano_candidates: List[Zone] = []
        for zone in zones:
            if zone.contains(point):
                return zone
            if zone.kind == "piano" and self._contains_with_piano_margin(zone, point):
                piano_candidates.append(zone)
        if piano_candidates:
            x, _ = point
            return min(piano_candidates, key=lambda zone: abs(zone.center[0] - x))
        return None

    def _contains_with_piano_margin(self, zone: Zone, point: Tuple[int, int]) -> bool:
        x, y = point
        x_margin = zone.width * config.PIANO_HIT_X_MARGIN_RATIO
        top_margin = zone.height * config.PIANO_HIT_TOP_MARGIN_RATIO
        bottom_margin = zone.height * config.PIANO_HIT_BOTTOM_MARGIN_RATIO
        return (
            zone.x1 - x_margin <= x <= zone.x2 + x_margin
            and zone.y1 - top_margin <= y <= zone.y2 + bottom_margin
        )
