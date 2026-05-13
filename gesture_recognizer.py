from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple

import config

if TYPE_CHECKING:
    from hand_tracker import HandLandmarks


class Gesture(str, Enum):
    FIST = "FIST"
    OPEN_PALM = "OPEN_PALM"
    THUMB_UP = "THUMB_UP"
    UNKNOWN = "UNKNOWN"


class GestureRecognizer:
    """Rule-based recognizer over MediaPipe's 21 hand landmarks."""

    LONG_FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))

    def recognize(self, hand: "HandLandmarks") -> Gesture:
        points = hand.landmarks
        if len(points) < 21:
            return Gesture.UNKNOWN

        extended = [self._finger_extended(points, tip, pip) for tip, pip in self.LONG_FINGERS]
        extended_count = sum(extended)
        thumb_up = self._thumb_up(points)

        if thumb_up and extended_count <= 1:
            return Gesture.THUMB_UP
        if extended_count == 0:
            return Gesture.FIST
        if extended_count >= 3:
            return Gesture.OPEN_PALM
        return Gesture.UNKNOWN

    def _finger_extended(self, points: List[Tuple[int, int, float]], tip: int, pip: int) -> bool:
        return points[tip][1] < points[pip][1] - 8

    def _thumb_up(self, points: List[Tuple[int, int, float]]) -> bool:
        wrist_y = points[0][1]
        thumb_tip_y = points[4][1]
        thumb_ip_y = points[3][1]
        return thumb_tip_y < thumb_ip_y - 8 and thumb_tip_y < wrist_y


@dataclass
class GestureUpdate:
    gesture: Gesture
    triggered: Optional[Gesture] = None
    action: str = ""


class GestureController:
    def __init__(
        self,
        stable_frames: int = config.GESTURE_STABLE_FRAMES,
        cooldown: float = config.GESTURE_COOLDOWN,
    ) -> None:
        self.stable_frames = stable_frames
        self.cooldown = cooldown
        self._candidate = Gesture.UNKNOWN
        self._stable_count = 0
        self._last_trigger_time = -999.0
        self._triggered_candidate: Optional[Gesture] = None
        self.last_action = ""

    def update(self, gesture: Gesture, current_time: float, loop_station) -> GestureUpdate:
        if gesture == self._candidate:
            self._stable_count += 1
        else:
            self._candidate = gesture
            self._stable_count = 1
            self._triggered_candidate = None

        if gesture == Gesture.UNKNOWN:
            self._triggered_candidate = None
            return GestureUpdate(gesture=gesture)
        if self._stable_count < self.stable_frames:
            return GestureUpdate(gesture=gesture)
        if self._triggered_candidate == gesture:
            return GestureUpdate(gesture=gesture)
        if current_time - self._last_trigger_time < self.cooldown:
            return GestureUpdate(gesture=gesture)

        self._last_trigger_time = current_time
        self._triggered_candidate = gesture
        action = self._apply(gesture, loop_station, current_time)
        self.last_action = action
        return GestureUpdate(gesture=gesture, triggered=gesture, action=action)

    def _apply(self, gesture: Gesture, loop_station, current_time: float) -> str:
        if gesture == Gesture.FIST:
            loop_station.toggle_recording(current_time)
            return "toggle recording"
        if gesture == Gesture.OPEN_PALM:
            loop_station.toggle_playback(current_time)
            return "toggle playback"
        if gesture == Gesture.THUMB_UP:
            loop_station.clear()
            return "clear loop"
        return ""
