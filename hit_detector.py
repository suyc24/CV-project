from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Deque, Dict, Iterable, List, Mapping, Optional, Tuple

import config
from instrument import Zone
from utils import clamp

if TYPE_CHECKING:
    from depth_contact import DepthObservation
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
    previous_relative_y: Optional[float] = None
    smoothed_velocity_y: float = 0.0
    raw_velocity_y: float = 0.0
    smoothed_relative_velocity_y: float = 0.0
    raw_relative_velocity_y: float = 0.0
    is_pressed: bool = False
    pressed_zone_id: Optional[str] = None
    pressed_y: Optional[int] = None
    pressed_relative_y: Optional[float] = None
    last_hit_time: float = -999.0
    motion_state: str = "idle"
    armed_zone_id: Optional[str] = None
    lift_start_y: Optional[int] = None
    lift_start_relative_y: Optional[float] = None
    release_ready_frames: int = 0
    falling_frames: int = 0
    peak_y: Optional[int] = None
    peak_relative_y: Optional[float] = None
    max_down_velocity: float = 0.0
    max_down_relative_velocity: float = 0.0
    last_zone_id: Optional[str] = None
    depth_motion_state: str = "idle"
    previous_depth_height_m: Optional[float] = None
    previous_depth_timestamp: Optional[float] = None
    raw_depth_down_velocity_mps: float = 0.0
    smoothed_depth_down_velocity_mps: float = 0.0
    depth_peak_height_m: Optional[float] = None
    depth_falling_frames: int = 0
    depth_arm_ready_frames: int = 0
    depth_release_ready_frames: int = 0
    recent_motion: Deque[Tuple[float, int, float]] = field(default_factory=lambda: deque(maxlen=8))
    trail: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=config.TRAIL_LENGTH))


FINGER_NAMES = {
    4: "thumb",
    8: "index",
    12: "middle",
    16: "ring",
    20: "pinky",
}

FINGER_BASE_IDS = {
    4: 2,
    8: 5,
    12: 9,
    16: 13,
    20: 17,
}


class HitDetector:
    def __init__(self, finger_ids: Iterable[int] = config.TRIGGER_FINGER_IDS) -> None:
        self.finger_ids = tuple(finger_ids)
        self._states: Dict[Tuple[int, int], FingerState] = {}
        self._diagnostics: List[Dict[str, object]] = []
        self._last_piano_hand_hit_time: Dict[int, float] = {}
        self._stable_hand_centers: Dict[int, Tuple[float, float, float, str, int]] = {}
        self._next_stable_hand_id = 0

    def reset(self) -> None:
        self._states.clear()
        self._diagnostics.clear()
        self._last_piano_hand_hit_time.clear()
        self._stable_hand_centers.clear()
        self._next_stable_hand_id = 0

    def update(
        self,
        hands: Iterable["HandLandmarks"],
        zones: List[Zone],
        current_time: float,
        depth_observations: Optional[Mapping[Tuple[int, int], "DepthObservation"]] = None,
    ) -> List[HitEvent]:
        hits: List[HitEvent] = []
        self._diagnostics = []
        hand_list = list(hands)
        stable_hand_ids = self._assign_stable_hand_ids(hand_list, current_time)
        raw_depth_observations = depth_observations or {}
        depth_aliases: Dict[Tuple[int, int], "DepthObservation"] = {}
        for hand, stable_hand_id in zip(hand_list, stable_hand_ids):
            if stable_hand_id == hand.hand_id:
                continue
            for finger_id in self.finger_ids:
                observation = raw_depth_observations.get((hand.hand_id, finger_id))
                if observation is not None:
                    depth_aliases[(stable_hand_id, finger_id)] = observation
        for hand, hand_id in zip(hand_list, stable_hand_ids):
            hand_candidates: List[Tuple[float, FingerState, Zone, int, int, int, float, float, Dict[str, object]]] = []
            for finger_id in self.finger_ids:
                if finger_id >= len(hand.landmarks):
                    continue
                state = self._states.setdefault((hand_id, finger_id), FingerState())
                x, y, _ = hand.landmarks[finger_id]
                position = (x, y)
                relative_y = y - self._finger_anchor_y(hand.landmarks, finger_id)
                state.trail.append(position)
                previous_position = state.previous_position
                previous_relative_y = state.previous_relative_y
                unstable_tracking = self._finger_tracking_unstable(hand, finger_id)
                if unstable_tracking:
                    velocity_y = self._observe_unstable_tracking(state, position, relative_y, current_time)
                    zone = self._zone_for_state(zones, position, state, previous_position)
                    depth_observation = raw_depth_observations.get((hand.hand_id, finger_id)) or depth_aliases.get(
                        (hand_id, finger_id)
                    )
                    self._diagnostics.append(
                        {
                            "hand_id": hand_id,
                            "raw_hand_id": hand.hand_id,
                            "finger_id": finger_id,
                            "finger_name": FINGER_NAMES.get(finger_id, f"F{finger_id}"),
                            "x": x,
                            "y": y,
                            "relative_y": relative_y,
                            "velocity_y": velocity_y,
                            "relative_velocity_y": state.smoothed_relative_velocity_y,
                            "zone_label": zone.label if zone else None,
                            "zone_kind": zone.kind if zone else None,
                            "pressed": state.is_pressed,
                            "motion_state": state.motion_state,
                            "reason": "unstable_tracking" if zone else "no_zone",
                            "threshold": self._threshold_for(zone) if zone else None,
                            "press_y": zone.press_y if zone else None,
                            "depth_contact": depth_observation.contact if depth_observation else None,
                            "depth_height_m": depth_observation.height_above_desk_m if depth_observation else None,
                            "depth_reason": depth_observation.reason if depth_observation else None,
                            "depth_state": state.depth_motion_state,
                            "depth_velocity_mps": state.smoothed_depth_down_velocity_mps,
                            "tracking_source": getattr(hand, "tracking_source", "mediapipe"),
                            "missed_frames": getattr(hand, "missed_frames", 0),
                            "unstable_tracking": True,
                        }
                    )
                    continue
                velocity_y = self._update_velocity(state, position, relative_y, current_time)
                zone = self._zone_for_state(zones, position, state, previous_position)
                depth_observation = raw_depth_observations.get((hand.hand_id, finger_id)) or depth_aliases.get(
                    (hand_id, finger_id)
                )
                self._update_release_state(state, zone, y, relative_y, depth_observation)

                reason = self._miss_reason(
                    state,
                    zone,
                    finger_id,
                    y,
                    velocity_y,
                    relative_y,
                    current_time,
                    previous_position,
                    previous_relative_y,
                    depth_observation,
                )
                diagnostic = {
                    "hand_id": hand_id,
                    "raw_hand_id": hand.hand_id,
                    "finger_id": finger_id,
                    "finger_name": FINGER_NAMES.get(finger_id, f"F{finger_id}"),
                    "x": x,
                    "y": y,
                    "relative_y": relative_y,
                    "velocity_y": velocity_y,
                    "relative_velocity_y": state.smoothed_relative_velocity_y,
                    "zone_label": zone.label if zone else None,
                    "zone_kind": zone.kind if zone else None,
                    "pressed": state.is_pressed,
                    "motion_state": state.motion_state,
                    "reason": reason,
                    "threshold": self._threshold_for(zone) if zone else None,
                    "press_y": zone.press_y if zone else None,
                    "depth_contact": depth_observation.contact if depth_observation else None,
                    "depth_height_m": depth_observation.height_above_desk_m if depth_observation else None,
                    "depth_reason": depth_observation.reason if depth_observation else None,
                    "depth_state": state.depth_motion_state,
                    "depth_velocity_mps": state.smoothed_depth_down_velocity_mps,
                    "tracking_source": getattr(hand, "tracking_source", "mediapipe"),
                    "missed_frames": getattr(hand, "missed_frames", 0),
                    "unstable_tracking": False,
                }
                self._diagnostics.append(diagnostic)
                if zone and reason == "hit":
                    if zone.kind == "piano":
                        score = self._hit_score(state, finger_id, y, relative_y)
                        hand_candidates.append(
                            (score, state, zone, hand_id, finger_id, y, relative_y, velocity_y, diagnostic)
                        )
                    else:
                        hits.append(self._commit_hit(state, zone, hand_id, finger_id, current_time, y, relative_y, velocity_y))
            if hand_candidates:
                hand_candidates.sort(key=lambda candidate: candidate[0], reverse=True)
                max_hits = max(1, int(config.PIANO_MAX_HITS_PER_HAND_PER_FRAME))
                committed = 0
                for candidate in hand_candidates:
                    _, state, zone, hand_id, finger_id, y, relative_y, velocity_y, diagnostic = candidate
                    if self._blocked_by_recent_piano_hand_hit(hand_id, current_time):
                        diagnostic["reason"] = "suppressed_by_hand_cooldown"
                        state.motion_state = "suppressed"
                        state.max_down_velocity = 0.0
                        state.max_down_relative_velocity = 0.0
                    elif self._blocked_by_active_piano_finger(hand_id, finger_id):
                        diagnostic["reason"] = "suppressed_by_active_finger"
                        state.motion_state = "suppressed"
                        state.max_down_velocity = 0.0
                        state.max_down_relative_velocity = 0.0
                    elif committed < max_hits:
                        hits.append(self._commit_hit(state, zone, hand_id, finger_id, current_time, y, relative_y, velocity_y))
                        self._last_piano_hand_hit_time[hand_id] = current_time
                        committed += 1
                    else:
                        diagnostic["reason"] = "suppressed_by_finger"
                        state.motion_state = "suppressed"
                        state.max_down_velocity = 0.0
                        state.max_down_relative_velocity = 0.0
        return hits

    def get_trails(self) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        return {key: list(state.trail) for key, state in self._states.items()}

    def diagnostics(self) -> List[Dict[str, object]]:
        return list(self._diagnostics)

    def debug_snapshot(self) -> List[str]:
        lines: List[str] = []
        for (hand_id, finger_id), state in sorted(self._states.items()):
            display_velocity = self._motion_velocity(state)
            if abs(display_velocity) < 80 and not state.is_pressed:
                continue
            name = FINGER_NAMES.get(finger_id, f"F{finger_id}")
            lines.append(
                f"H{hand_id} {name}: vy={display_velocity:6.0f} "
                f"{state.motion_state} pressed={int(state.is_pressed)}"
            )
        return lines[:10]

    def _assign_stable_hand_ids(self, hands: List["HandLandmarks"], current_time: float) -> List[int]:
        if not config.PIANO_DETECTOR_STABLE_HAND_IDS:
            return [hand.hand_id for hand in hands]
        self._prune_stable_hand_centers(current_time)
        if not hands:
            return []

        detected_info = [
            (
                self._hand_center(hand.landmarks),
                self._hand_scale(hand.landmarks),
                getattr(hand, "label", "Unknown"),
                hand.hand_id,
            )
            for hand in hands
        ]
        candidates: List[Tuple[float, int, int]] = []
        for detected_idx, (center, scale, label, raw_id) in enumerate(detected_info):
            max_distance = max(float(config.PIANO_DETECTOR_HAND_ID_MAX_DISTANCE_PX), scale * 0.95)
            for stable_id, (px, py, _, previous_label, previous_raw_id) in self._stable_hand_centers.items():
                distance = math.hypot(center[0] - px, center[1] - py)
                if distance > max_distance:
                    continue
                label_matches = previous_label == label or "Unknown" in {previous_label, label}
                label_penalty = 0.0 if label_matches else min(45.0, max_distance * 0.18)
                raw_bonus = -12.0 if previous_raw_id == raw_id else 0.0
                candidates.append((distance + label_penalty + raw_bonus, detected_idx, stable_id))

        assignments: Dict[int, int] = {}
        used_stable_ids: set[int] = set()
        for _, detected_idx, stable_id in sorted(candidates):
            if detected_idx in assignments or stable_id in used_stable_ids:
                continue
            assignments[detected_idx] = stable_id
            used_stable_ids.add(stable_id)

        stable_ids: List[int] = []
        next_centers = dict(self._stable_hand_centers)
        for detected_idx, (center, _, label, raw_id) in enumerate(detected_info):
            stable_id = assignments.get(detected_idx)
            if stable_id is None:
                stable_id = self._next_stable_hand_id
                self._next_stable_hand_id += 1
            stable_ids.append(stable_id)
            next_centers[stable_id] = (center[0], center[1], current_time, label, raw_id)
        self._stable_hand_centers = next_centers
        return stable_ids

    def _prune_stable_hand_centers(self, current_time: float) -> None:
        max_age = float(config.PIANO_DETECTOR_HAND_ID_MEMORY_SECONDS)
        if max_age <= 0:
            self._stable_hand_centers.clear()
            return
        for stable_id, (_, _, last_seen, _, _) in list(self._stable_hand_centers.items()):
            if current_time - last_seen > max_age:
                del self._stable_hand_centers[stable_id]

    def _hand_center(self, landmarks: List[Tuple[int, int, float]]) -> Tuple[float, float]:
        anchor_ids = (0, 5, 9, 13, 17)
        points = [landmarks[idx] for idx in anchor_ids if idx < len(landmarks)]
        if not points:
            points = landmarks
        if not points:
            return (0.0, 0.0)
        return (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )

    def _hand_scale(self, landmarks: List[Tuple[int, int, float]]) -> float:
        if not landmarks:
            return 0.0
        xs = [point[0] for point in landmarks]
        ys = [point[1] for point in landmarks]
        return max(max(xs) - min(xs), max(ys) - min(ys))

    def _commit_hit(
        self,
        state: FingerState,
        zone: Zone,
        hand_id: int,
        finger_id: int,
        current_time: float,
        finger_y: int,
        relative_y: float,
        velocity_y: float,
    ) -> HitEvent:
        hit_velocity = self._hit_velocity(state, velocity_y)
        volume = self._velocity_to_volume(hit_velocity, zone)
        hit = HitEvent(
            note_id=zone.label,
            sound_id=zone.sound_id,
            zone_label=zone.label,
            finger_id=finger_id,
            hand_id=hand_id,
            timestamp=current_time,
            velocity=hit_velocity,
            volume=volume,
        )
        state.is_pressed = True
        state.pressed_zone_id = zone.sound_id
        state.pressed_y = finger_y
        state.pressed_relative_y = relative_y
        state.last_hit_time = current_time
        state.motion_state = "pressed"
        state.armed_zone_id = zone.sound_id
        state.lift_start_y = None
        state.lift_start_relative_y = None
        state.release_ready_frames = 0
        state.falling_frames = 0
        state.peak_y = finger_y
        state.peak_relative_y = relative_y
        state.max_down_velocity = 0.0
        state.max_down_relative_velocity = 0.0
        state.depth_motion_state = "pressed"
        state.depth_falling_frames = 0
        state.depth_arm_ready_frames = 0
        state.depth_release_ready_frames = 0
        state.depth_peak_height_m = None
        return hit

    def _hit_score(self, state: FingerState, finger_id: int, finger_y: int, relative_y: float) -> float:
        drop = self._drop_distance(state, finger_y, relative_y)
        score = self._hit_velocity(state, state.smoothed_velocity_y) + drop * 8.0
        if finger_id == 4:
            score *= config.PIANO_THUMB_SCORE_WEIGHT
        return score

    def _blocked_by_active_piano_finger(self, hand_id: int, finger_id: int) -> bool:
        if not config.PIANO_MONOPHONIC_PER_HAND:
            return False
        for (state_hand_id, state_finger_id), state in self._states.items():
            if state_hand_id == hand_id and state_finger_id != finger_id and state.is_pressed:
                return True
        return False

    def _blocked_by_recent_piano_hand_hit(self, hand_id: int, current_time: float) -> bool:
        cooldown = float(config.PIANO_HAND_HIT_COOLDOWN)
        if cooldown <= 0:
            return False
        last_time = self._last_piano_hand_hit_time.get(hand_id)
        return last_time is not None and current_time - last_time < cooldown

    def _hit_velocity(self, state: FingerState, velocity_y: float) -> float:
        if config.PIANO_USE_RELATIVE_FINGER_MOTION:
            return max(state.smoothed_relative_velocity_y, state.raw_relative_velocity_y, state.max_down_relative_velocity)
        return max(velocity_y, state.max_down_velocity)

    def _motion_velocity(self, state: FingerState) -> float:
        return state.smoothed_relative_velocity_y if config.PIANO_USE_RELATIVE_FINGER_MOTION else state.smoothed_velocity_y

    def _release_up_velocity(self, state: FingerState) -> float:
        if config.PIANO_USE_RELATIVE_FINGER_MOTION:
            return min(state.smoothed_relative_velocity_y, state.raw_relative_velocity_y)
        return min(state.smoothed_velocity_y, state.raw_velocity_y)

    def _finger_tracking_unstable(self, hand: "HandLandmarks", finger_id: int) -> bool:
        if not config.PIANO_BLOCK_UNSTABLE_LANDMARK_HITS:
            return False
        if getattr(hand, "tracking_source", "mediapipe") == "optical_flow":
            return True
        if getattr(hand, "missed_frames", 0) > 0:
            return True
        return finger_id in set(getattr(hand, "unstable_landmark_ids", ()))

    def _update_velocity(self, state: FingerState, position: Tuple[int, int], relative_y: float, current_time: float) -> float:
        if state.previous_position is None or state.previous_timestamp is None:
            state.previous_position = position
            state.previous_relative_y = relative_y
            state.previous_timestamp = current_time
            state.recent_motion.append((current_time, position[1], relative_y))
            return state.smoothed_velocity_y

        dt = max(1e-3, current_time - state.previous_timestamp)
        raw_velocity_y = (position[1] - state.previous_position[1]) / dt
        raw_relative_velocity_y = (relative_y - (state.previous_relative_y if state.previous_relative_y is not None else relative_y)) / dt
        state.raw_velocity_y = raw_velocity_y
        state.raw_relative_velocity_y = raw_relative_velocity_y
        alpha = config.VELOCITY_SMOOTHING_ALPHA
        state.smoothed_velocity_y = alpha * raw_velocity_y + (1.0 - alpha) * state.smoothed_velocity_y
        state.smoothed_relative_velocity_y = (
            alpha * raw_relative_velocity_y + (1.0 - alpha) * state.smoothed_relative_velocity_y
        )
        state.previous_position = position
        state.previous_relative_y = relative_y
        state.previous_timestamp = current_time
        state.recent_motion.append((current_time, position[1], relative_y))
        return state.smoothed_velocity_y

    def _observe_unstable_tracking(
        self,
        state: FingerState,
        position: Tuple[int, int],
        relative_y: float,
        current_time: float,
    ) -> float:
        state.previous_position = position
        state.previous_relative_y = relative_y
        state.previous_timestamp = current_time
        state.recent_motion.append((current_time, position[1], relative_y))
        state.raw_velocity_y = 0.0
        state.raw_relative_velocity_y = 0.0
        state.smoothed_velocity_y *= 0.35
        state.smoothed_relative_velocity_y *= 0.35
        state.max_down_velocity = 0.0
        state.max_down_relative_velocity = 0.0
        if not state.is_pressed and state.motion_state == "falling":
            state.motion_state = "raised"
            state.falling_frames = 0
            state.peak_y = position[1]
            state.peak_relative_y = relative_y
        return state.smoothed_velocity_y

    def _update_release_state(
        self,
        state: FingerState,
        zone: Optional[Zone],
        finger_y: int,
        relative_y: float,
        depth_observation: Optional["DepthObservation"] = None,
    ) -> None:
        if not state.is_pressed:
            return
        if zone is None:
            state.is_pressed = False
            state.pressed_zone_id = None
            state.pressed_y = None
            state.pressed_relative_y = None
            state.motion_state = "idle"
            state.armed_zone_id = None
            state.lift_start_y = None
            state.lift_start_relative_y = None
            state.release_ready_frames = 0
            state.falling_frames = 0
            state.peak_y = None
            state.peak_relative_y = None
            state.max_down_velocity = 0.0
            state.max_down_relative_velocity = 0.0
            state.last_zone_id = None
            return
        if state.pressed_zone_id and zone.sound_id != state.pressed_zone_id and zone.kind != "piano":
            self._clear_pressed_state(state, "idle")
            return
        if zone.kind == "piano":
            if self._release_piano_zone_change_state(state, zone, finger_y, relative_y, depth_observation):
                return
            if self._release_piano_depth_state(state, depth_observation):
                self._clear_pressed_state(state, "raised", zone, finger_y, relative_y)
                state.depth_motion_state = "armed"
                return
            if config.PIANO_USE_RELATIVE_FINGER_MOTION:
                lift_amount = (
                    state.pressed_relative_y - relative_y
                    if state.pressed_relative_y is not None
                    else 0.0
                )
                lifted_enough = (
                    state.pressed_relative_y is not None
                    and lift_amount >= config.PIANO_RELEASE_LIFT_PX
                )
            else:
                lift_amount = (
                    state.pressed_y - finger_y
                    if state.pressed_y is not None
                    else 0.0
                )
                lifted_enough = state.pressed_y is not None and lift_amount >= config.PIANO_RELEASE_LIFT_PX
            release_up_velocity = self._release_up_velocity(state)
            deliberate_lift = (
                release_up_velocity <= -config.PIANO_RELEASE_MIN_UP_VELOCITY
                or lift_amount >= config.PIANO_RELEASE_LIFT_PX * config.PIANO_RELEASE_STRONG_LIFT_MULTIPLIER
                or (state.release_ready_frames > 0 and lift_amount >= config.PIANO_RELEASE_LIFT_PX)
            )
            lifted_enough = lifted_enough and deliberate_lift and self._passes_release_motion_guard(state)
            if lifted_enough and self._depth_still_contacting(depth_observation):
                lifted_enough = False
            if lifted_enough:
                state.release_ready_frames += 1
            else:
                state.release_ready_frames = 0
            if state.release_ready_frames >= config.PIANO_RELEASE_STABLE_FRAMES:
                self._clear_pressed_state(state, "raised", zone, finger_y, relative_y)
            return
        if finger_y < zone.release_y:
            self._clear_pressed_state(state, "raised", zone, finger_y, relative_y)

    def _release_piano_zone_change_state(
        self,
        state: FingerState,
        zone: Zone,
        finger_y: int,
        relative_y: float,
        depth_observation: Optional["DepthObservation"],
    ) -> bool:
        if not config.PIANO_RELEASE_ON_ZONE_CHANGE:
            return False
        if not state.pressed_zone_id or state.pressed_zone_id == zone.sound_id:
            return False
        if self._depth_still_contacting(depth_observation):
            return False
        if config.PIANO_USE_RELATIVE_FINGER_MOTION:
            lift_amount = (
                state.pressed_relative_y - relative_y
                if state.pressed_relative_y is not None
                else 0.0
            )
        else:
            lift_amount = state.pressed_y - finger_y if state.pressed_y is not None else 0.0
        release_lift = config.PIANO_RELEASE_LIFT_PX * config.PIANO_ZONE_CHANGE_RELEASE_LIFT_RATIO
        if lift_amount >= release_lift or self._motion_velocity(state) <= -config.PIANO_RELEASE_MIN_UP_VELOCITY:
            self._clear_pressed_state(state, "raised", zone, finger_y, relative_y)
            return True
        return False

    def _clear_pressed_state(
        self,
        state: FingerState,
        motion_state: str,
        zone: Optional[Zone] = None,
        finger_y: Optional[int] = None,
        relative_y: Optional[float] = None,
    ) -> None:
        state.is_pressed = False
        state.pressed_zone_id = None
        state.pressed_y = None
        state.pressed_relative_y = None
        state.motion_state = motion_state
        state.armed_zone_id = zone.sound_id if zone is not None and motion_state == "raised" else None
        state.lift_start_y = None
        state.lift_start_relative_y = None
        state.release_ready_frames = 0
        state.falling_frames = 0
        state.peak_y = finger_y
        state.peak_relative_y = relative_y
        state.max_down_velocity = 0.0
        state.max_down_relative_velocity = 0.0
        state.last_zone_id = zone.sound_id if zone is not None else None

    def _reset_depth_motion_state(self, state: FingerState) -> None:
        state.depth_motion_state = "idle"
        state.previous_depth_height_m = None
        state.previous_depth_timestamp = None
        state.raw_depth_down_velocity_mps = 0.0
        state.smoothed_depth_down_velocity_mps = 0.0
        state.depth_peak_height_m = None
        state.depth_falling_frames = 0
        state.depth_arm_ready_frames = 0
        state.depth_release_ready_frames = 0

    def _release_piano_depth_state(
        self,
        state: FingerState,
        observation: Optional["DepthObservation"],
    ) -> bool:
        height = self._depth_height(observation)
        if height is None:
            state.depth_release_ready_frames = 0
            return False
        if height >= config.PIANO_3D_RELEASE_HEIGHT_M:
            state.depth_release_ready_frames += 1
        else:
            state.depth_release_ready_frames = 0
        return state.depth_release_ready_frames >= max(1, int(config.PIANO_3D_RELEASE_STABLE_FRAMES))

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
        finger_id: int,
        finger_y: int,
        velocity_y: float,
        relative_y: float,
        current_time: float,
        previous_position: Optional[Tuple[int, int]],
        previous_relative_y: Optional[float],
        depth_observation: Optional["DepthObservation"],
    ) -> str:
        if zone is None:
            self._update_air_motion_state(state, finger_y, relative_y)
            return "no_zone"
        if state.is_pressed:
            return "pressed"
        if current_time - state.last_hit_time < config.HIT_COOLDOWN:
            return "cooldown"
        if zone.kind == "piano":
            return self._piano_miss_reason(
                state,
                zone,
                finger_id,
                finger_y,
                velocity_y,
                relative_y,
                previous_position,
                previous_relative_y,
                current_time,
                depth_observation,
            )
        if finger_y <= zone.press_y:
            return "press_line"
        if velocity_y <= self._threshold_for(zone):
            return "velocity"
        return "hit"

    def _piano_miss_reason(
        self,
        state: FingerState,
        zone: Zone,
        finger_id: int,
        finger_y: int,
        velocity_y: float,
        relative_y: float,
        previous_position: Optional[Tuple[int, int]],
        previous_relative_y: Optional[float],
        current_time: float,
        depth_observation: Optional["DepthObservation"],
    ) -> str:
        depth_reason = self._piano_3d_miss_reason(state, depth_observation, current_time)
        if depth_reason is not None:
            if depth_reason in {"depth_lifting", "depth_armed"}:
                self._arm_piano_from_current_position(state, zone, finger_y, relative_y)
            return depth_reason

        arm_y = zone.y1 + config.PIANO_ARM_RATIO * zone.height
        motion_y = relative_y if config.PIANO_USE_RELATIVE_FINGER_MOTION else finger_y
        motion_velocity = self._motion_velocity(state)

        if previous_position is not None:
            previous_motion_y = previous_relative_y if config.PIANO_USE_RELATIVE_FINGER_MOTION else previous_position[1]
            if previous_motion_y is None:
                previous_motion_y = motion_y
        else:
            previous_motion_y = motion_y

        if state.motion_state == "lifting" or (
            motion_velocity < -config.PIANO_LIFT_VELOCITY_THRESHOLD
            and state.motion_state not in {"falling", "pressed"}
        ):
            if state.motion_state != "lifting":
                state.motion_state = "lifting"
                state.armed_zone_id = zone.sound_id
                state.lift_start_y = previous_position[1] if previous_position is not None else finger_y
                state.lift_start_relative_y = previous_motion_y
                state.peak_y = min(state.lift_start_y, finger_y)
                state.peak_relative_y = min(state.lift_start_relative_y, motion_y)
                state.max_down_velocity = 0.0
                state.max_down_relative_velocity = 0.0
            else:
                state.armed_zone_id = zone.sound_id
                state.peak_y = finger_y if state.peak_y is None else min(state.peak_y, finger_y)
                state.peak_relative_y = motion_y if state.peak_relative_y is None else min(state.peak_relative_y, motion_y)

            lift_px = self._lift_distance(state, finger_y, relative_y)
            passive_arm_allowed = not self._depth_blocks_passive_arm(depth_observation)
            if lift_px >= config.PIANO_ARM_MIN_LIFT_PX or (finger_y <= arm_y and passive_arm_allowed):
                state.motion_state = "raised"
                state.falling_frames = 0
                return "armed"
            if finger_y <= arm_y and not passive_arm_allowed:
                return "contact_arm_guard"
            if motion_velocity > config.PIANO_FALLING_VELOCITY_THRESHOLD:
                state.motion_state = "idle"
                state.armed_zone_id = None
                state.lift_start_y = None
                state.lift_start_relative_y = None
                state.peak_y = None
                state.peak_relative_y = None
                state.max_down_velocity = 0.0
                state.max_down_relative_velocity = 0.0
                state.falling_frames = 0
                return "short_lift"
            return "lifting"

        if state.motion_state in {"raised", "falling"} and zone.sound_id != state.armed_zone_id:
            state.armed_zone_id = zone.sound_id
            state.peak_y = min(state.peak_y if state.peak_y is not None else finger_y, finger_y)
            state.peak_relative_y = min(
                state.peak_relative_y if state.peak_relative_y is not None else motion_y,
                motion_y,
            )

        can_start_fall = state.motion_state in {"raised", "falling"}
        if motion_velocity > config.PIANO_FALLING_VELOCITY_THRESHOLD and can_start_fall:
            if state.motion_state != "falling":
                candidate_peak = previous_position[1] if previous_position is not None else finger_y
                state.peak_y = min(state.peak_y if state.peak_y is not None else candidate_peak, candidate_peak, finger_y)
                state.peak_relative_y = min(
                    state.peak_relative_y if state.peak_relative_y is not None else motion_y,
                    previous_relative_y if previous_relative_y is not None else motion_y,
                    motion_y,
                )
                state.falling_frames = 1
            else:
                state.falling_frames += 1
            state.motion_state = "falling"
            state.armed_zone_id = state.armed_zone_id or zone.sound_id
            state.max_down_velocity = max(state.max_down_velocity, velocity_y)
            state.max_down_relative_velocity = max(state.max_down_relative_velocity, motion_velocity)
        elif state.motion_state == "falling":
            state.falling_frames += 1
            state.max_down_velocity = max(state.max_down_velocity, velocity_y)
            state.max_down_relative_velocity = max(state.max_down_relative_velocity, motion_velocity)

        if finger_y <= arm_y and state.motion_state != "falling":
            if self._depth_blocks_passive_arm(depth_observation):
                state.motion_state = "idle"
                state.armed_zone_id = None
                state.lift_start_y = None
                state.lift_start_relative_y = None
                state.falling_frames = 0
                state.peak_y = None
                state.peak_relative_y = None
                state.max_down_velocity = 0.0
                state.max_down_relative_velocity = 0.0
                return "contact_arm_guard"
            state.motion_state = "raised"
            state.armed_zone_id = zone.sound_id
            state.lift_start_y = None
            state.lift_start_relative_y = None
            state.falling_frames = 0
            state.peak_y = finger_y if state.peak_y is None else min(state.peak_y, finger_y)
            state.peak_relative_y = motion_y if state.peak_relative_y is None else min(state.peak_relative_y, motion_y)
            state.max_down_velocity = 0.0
            state.max_down_relative_velocity = 0.0
            return "armed"

        if state.motion_state == "falling" and motion_velocity < -config.PIANO_LIFT_VELOCITY_THRESHOLD:
            state.motion_state = "raised"
            state.armed_zone_id = zone.sound_id
            state.lift_start_y = None
            state.lift_start_relative_y = None
            state.falling_frames = 0
            state.peak_y = finger_y
            state.peak_relative_y = motion_y
            state.max_down_velocity = 0.0
            state.max_down_relative_velocity = 0.0
            return "lifted"

        if state.motion_state != "falling":
            return "not_armed"

        if state.falling_frames < config.PIANO_MIN_FALL_FRAMES:
            return "falling"

        drop_px = self._drop_distance(state, finger_y, relative_y)
        if drop_px < config.PIANO_STRIKE_MIN_DROP_PX:
            return "short_drop"

        strike_velocity = self._hit_velocity(state, velocity_y)
        if config.PIANO_JITTER_GUARD_ENABLED and not self._passes_piano_jitter_guard(state, strike_velocity, drop_px):
            return "jitter_guard"
        depth_reason = self._depth_contact_block_reason(depth_observation)
        if depth_reason:
            return depth_reason
        key_depth_reason = self._key_depth_block_reason(zone, finger_id, finger_y)
        if key_depth_reason:
            return key_depth_reason
        if strike_velocity >= config.PIANO_STRIKE_MIN_VELOCITY:
            return "hit"
        return "velocity"

    def _arm_piano_from_current_position(
        self,
        state: FingerState,
        zone: Zone,
        finger_y: int,
        relative_y: float,
    ) -> None:
        if state.is_pressed or state.motion_state == "falling":
            return
        motion_y = relative_y if config.PIANO_USE_RELATIVE_FINGER_MOTION else finger_y
        state.motion_state = "raised"
        state.armed_zone_id = zone.sound_id
        state.lift_start_y = None
        state.lift_start_relative_y = None
        state.falling_frames = 0
        state.peak_y = finger_y if state.peak_y is None else min(state.peak_y, finger_y)
        state.peak_relative_y = motion_y if state.peak_relative_y is None else min(state.peak_relative_y, motion_y)
        state.max_down_velocity = 0.0
        state.max_down_relative_velocity = 0.0

    def _piano_3d_miss_reason(
        self,
        state: FingerState,
        observation: Optional["DepthObservation"],
        current_time: float,
    ) -> Optional[str]:
        if not config.PIANO_3D_TRIGGER_ENABLED or config.DEPTH_CONTACT_MODE == "off":
            return None
        height = self._depth_height(observation)
        if height is None:
            if observation is not None and config.PIANO_3D_BLOCK_ON_UNKNOWN_DEPTH:
                reason = getattr(observation, "reason", "unknown")
                return f"depth_{reason}"
            return None
        if self._depth_blocks_passive_arm(observation):
            state.depth_motion_state = "resting"
            state.depth_peak_height_m = height
            state.depth_falling_frames = 0
            return "contact_arm_guard"

        down_velocity = self._update_depth_velocity(state, height, current_time)
        contact = self._depth_is_contact(observation, height)

        if contact and state.depth_motion_state == "lifting":
            peak = state.depth_peak_height_m if state.depth_peak_height_m is not None else height
            drop_m = max(0.0, peak - height)
            if drop_m >= config.PIANO_3D_MIN_DROP_M and down_velocity >= config.PIANO_3D_MIN_DOWN_VELOCITY_MPS:
                state.depth_motion_state = "falling"
                return self._piano_3d_direct_hit_reason()

        if contact and state.depth_motion_state not in {"armed", "falling"}:
            state.depth_motion_state = "resting"
            state.depth_peak_height_m = height
            state.depth_falling_frames = 0
            return "depth_resting"

        if height >= config.PIANO_3D_ARM_HEIGHT_M:
            state.depth_arm_ready_frames += 1
            if state.depth_arm_ready_frames < max(1, int(config.PIANO_3D_ARM_STABLE_FRAMES)):
                state.depth_motion_state = "lifting"
                state.depth_peak_height_m = max(state.depth_peak_height_m or height, height)
                return "depth_lifting"
            if state.depth_motion_state not in {"armed", "falling"}:
                state.depth_motion_state = "armed"
                state.depth_falling_frames = 0
            state.depth_peak_height_m = max(state.depth_peak_height_m or height, height)
            if down_velocity <= 0:
                return "depth_armed"
        else:
            state.depth_arm_ready_frames = 0

        if state.depth_motion_state in {"armed", "falling"}:
            peak = state.depth_peak_height_m if state.depth_peak_height_m is not None else height
            drop_m = max(0.0, peak - height)
            if down_velocity >= config.PIANO_3D_MIN_DOWN_VELOCITY_MPS or drop_m >= config.PIANO_3D_MIN_DROP_M * 0.5:
                state.depth_motion_state = "falling"
                state.depth_falling_frames += 1

            if not contact:
                return "depth_falling" if state.depth_motion_state == "falling" else "depth_armed"
            if drop_m < config.PIANO_3D_MIN_DROP_M:
                return "depth_short_drop"
            if (
                down_velocity < config.PIANO_3D_MIN_DOWN_VELOCITY_MPS
                and state.depth_falling_frames < max(1, int(config.PIANO_MIN_FALL_FRAMES))
            ):
                return "depth_velocity"
            return self._piano_3d_direct_hit_reason()

        if height >= config.PIANO_3D_ARM_HEIGHT_M:
            state.depth_motion_state = "armed"
            state.depth_peak_height_m = height
            state.depth_falling_frames = 0
            return "depth_armed"

        return "depth_not_armed"

    def _piano_3d_direct_hit_reason(self) -> Optional[str]:
        return "hit" if config.PIANO_3D_DIRECT_TRIGGER_ENABLED else None

    def _depth_height(self, observation: Optional["DepthObservation"]) -> Optional[float]:
        if observation is None:
            return None
        confidence = getattr(observation, "confidence", 1.0)
        if confidence is not None and confidence < config.DEPTH_MIN_CONFIDENCE:
            return None
        height = getattr(observation, "height_above_desk_m", None)
        if height is not None:
            return max(0.0, float(height))
        if getattr(observation, "contact", None) is True:
            return 0.0
        return None

    def _depth_is_contact(self, observation: Optional["DepthObservation"], height: float) -> bool:
        if height <= config.PIANO_3D_CONTACT_HEIGHT_M:
            return True
        if observation is not None and getattr(observation, "contact", None) is True:
            return getattr(observation, "height_above_desk_m", None) is None
        return False

    def _update_depth_velocity(self, state: FingerState, height: float, current_time: float) -> float:
        if state.previous_depth_height_m is None or state.previous_depth_timestamp is None:
            state.previous_depth_height_m = height
            state.previous_depth_timestamp = current_time
            state.raw_depth_down_velocity_mps = 0.0
            state.smoothed_depth_down_velocity_mps = 0.0
            return 0.0
        dt = max(1e-3, current_time - state.previous_depth_timestamp)
        raw = (state.previous_depth_height_m - height) / dt
        state.raw_depth_down_velocity_mps = raw
        alpha = config.VELOCITY_SMOOTHING_ALPHA
        state.smoothed_depth_down_velocity_mps = (
            alpha * raw + (1.0 - alpha) * state.smoothed_depth_down_velocity_mps
        )
        state.previous_depth_height_m = height
        state.previous_depth_timestamp = current_time
        return state.smoothed_depth_down_velocity_mps

    def _key_depth_block_reason(self, zone: Zone, finger_id: int, finger_y: int) -> Optional[str]:
        min_ratio = (
            config.PIANO_THUMB_HIT_MIN_KEY_Y_RATIO
            if finger_id == 4
            else config.PIANO_HIT_MIN_KEY_Y_RATIO
        )
        if min_ratio <= 0:
            return None
        key_ratio = (finger_y - zone.y1) / max(1.0, float(zone.height))
        if key_ratio < min_ratio:
            return "key_depth_guard"
        return None

    def _drop_distance(self, state: FingerState, finger_y: int, relative_y: float) -> float:
        if config.PIANO_USE_RELATIVE_FINGER_MOTION:
            peak = state.peak_relative_y if state.peak_relative_y is not None else relative_y
            return relative_y - peak
        peak = state.peak_y if state.peak_y is not None else finger_y
        return float(finger_y - peak)

    def _lift_distance(self, state: FingerState, finger_y: int, relative_y: float) -> float:
        if config.PIANO_USE_RELATIVE_FINGER_MOTION:
            start = state.lift_start_relative_y if state.lift_start_relative_y is not None else relative_y
            peak = state.peak_relative_y if state.peak_relative_y is not None else relative_y
            return start - peak
        start_y = state.lift_start_y if state.lift_start_y is not None else finger_y
        peak_y = state.peak_y if state.peak_y is not None else finger_y
        return float(start_y - peak_y)

    def _passes_piano_jitter_guard(self, state: FingerState, strike_velocity: float, drop_px: float) -> bool:
        if len(state.recent_motion) < 2:
            return False
        net_drop = self._recent_net_drop(state)
        if net_drop < config.PIANO_STRIKE_MIN_NET_DROP_PX:
            return False

        if state.falling_frames >= max(1, config.PIANO_MIN_FALL_FRAMES + 1):
            return True

        strong_drop = drop_px >= config.PIANO_STRIKE_MIN_DROP_PX * config.PIANO_STRIKE_STRONG_DROP_MULTIPLIER
        strong_velocity = strike_velocity >= config.PIANO_STRIKE_MIN_VELOCITY * config.PIANO_STRIKE_STRONG_VELOCITY_MULTIPLIER
        if strong_drop and strong_velocity:
            return True

        return self._last_frame_drop(state) <= config.PIANO_STRIKE_MAX_SINGLE_FRAME_DROP_PX

    def _passes_release_motion_guard(self, state: FingerState) -> bool:
        if len(state.recent_motion) < 2:
            return False
        net_lift = self._recent_net_lift(state)
        if net_lift < config.PIANO_RELEASE_MIN_NET_LIFT_PX:
            return False
        return self._last_frame_lift(state) <= config.PIANO_RELEASE_MAX_SINGLE_FRAME_LIFT_PX

    def _depth_still_contacting(self, observation: Optional["DepthObservation"]) -> bool:
        if not config.PIANO_DEPTH_RELEASE_GUARD or config.DEPTH_CONTACT_MODE == "off":
            return False
        if config.DEPTH_CONTACT_MODE != "required" and not config.PIANO_DEPTH_RELEASE_GUARD_ASSIST:
            return False
        if observation is None:
            return False
        if observation.contact is True:
            return True
        height = observation.height_above_desk_m
        if height is None:
            return False
        return height <= config.DEPTH_CONTACT_THRESHOLD_M

    def _depth_blocks_passive_arm(self, observation: Optional["DepthObservation"]) -> bool:
        if not config.PIANO_BLOCK_PASSIVE_ARM_WHILE_DEPTH_CONTACT or config.DEPTH_CONTACT_MODE == "off":
            return False
        if observation is None:
            return False
        height = observation.height_above_desk_m
        if height is not None:
            return height <= config.PIANO_PASSIVE_ARM_MAX_CONTACT_HEIGHT_M
        return observation.contact is True

    def _recent_net_drop(self, state: FingerState) -> float:
        values = [entry[2] if config.PIANO_USE_RELATIVE_FINGER_MOTION else float(entry[1]) for entry in state.recent_motion]
        if not values:
            return 0.0
        return values[-1] - min(values)

    def _recent_net_lift(self, state: FingerState) -> float:
        values = [entry[2] if config.PIANO_USE_RELATIVE_FINGER_MOTION else float(entry[1]) for entry in state.recent_motion]
        if not values:
            return 0.0
        return max(values) - values[-1]

    def _last_frame_drop(self, state: FingerState) -> float:
        if len(state.recent_motion) < 2:
            return 0.0
        previous = state.recent_motion[-2]
        current = state.recent_motion[-1]
        if config.PIANO_USE_RELATIVE_FINGER_MOTION:
            return current[2] - previous[2]
        return float(current[1] - previous[1])

    def _last_frame_lift(self, state: FingerState) -> float:
        if len(state.recent_motion) < 2:
            return 0.0
        previous = state.recent_motion[-2]
        current = state.recent_motion[-1]
        if config.PIANO_USE_RELATIVE_FINGER_MOTION:
            return previous[2] - current[2]
        return float(previous[1] - current[1])

    def _depth_contact_block_reason(self, observation: Optional["DepthObservation"]) -> Optional[str]:
        mode = config.DEPTH_CONTACT_MODE
        if mode == "off":
            return None
        if observation is None:
            return "depth_unknown" if mode == "required" else None
        if observation.contact is True:
            return None
        if observation.contact is None:
            return f"depth_{observation.reason}" if mode == "required" else None
        height = observation.height_above_desk_m
        if mode == "required":
            return "depth_air"
        if height is not None and height > config.DEPTH_RELEASE_THRESHOLD_M:
            return "depth_air"
        return None

    def _update_air_motion_state(self, state: FingerState, finger_y: int, relative_y: float) -> None:
        if state.is_pressed:
            return
        motion_y = relative_y if config.PIANO_USE_RELATIVE_FINGER_MOTION else finger_y
        motion_velocity = self._motion_velocity(state)
        if motion_velocity < -config.PIANO_LIFT_VELOCITY_THRESHOLD:
            state.motion_state = "lifting"
            state.armed_zone_id = None
            if state.lift_start_y is None:
                state.lift_start_y = finger_y
            if state.lift_start_relative_y is None:
                state.lift_start_relative_y = motion_y
            state.peak_y = finger_y if state.peak_y is None else min(state.peak_y, finger_y)
            state.peak_relative_y = motion_y if state.peak_relative_y is None else min(state.peak_relative_y, motion_y)
            state.max_down_velocity = 0.0
            state.max_down_relative_velocity = 0.0
            return
        if state.motion_state in {"lifting", "raised", "falling"} and motion_velocity > config.PIANO_FALLING_VELOCITY_THRESHOLD:
            if state.motion_state == "lifting" and self._lift_distance(state, finger_y, relative_y) < config.PIANO_ARM_MIN_LIFT_PX:
                state.motion_state = "idle"
                state.armed_zone_id = None
                state.lift_start_y = None
                state.lift_start_relative_y = None
                state.peak_y = None
                state.peak_relative_y = None
                state.max_down_velocity = 0.0
                state.max_down_relative_velocity = 0.0
                state.falling_frames = 0
                return
            if state.motion_state != "falling":
                state.peak_y = state.peak_y if state.peak_y is not None else finger_y
                state.peak_relative_y = state.peak_relative_y if state.peak_relative_y is not None else motion_y
                state.falling_frames = 1
            else:
                state.falling_frames += 1
            state.motion_state = "falling"
            state.max_down_velocity = max(state.max_down_velocity, state.smoothed_velocity_y)
            state.max_down_relative_velocity = max(state.max_down_relative_velocity, motion_velocity)
            return
        if state.motion_state not in {"lifting", "raised", "falling"}:
            state.motion_state = "idle"
            state.armed_zone_id = None
            state.lift_start_y = None
            state.lift_start_relative_y = None
            state.release_ready_frames = 0
            state.falling_frames = 0
            state.peak_y = None
            state.peak_relative_y = None
            state.max_down_velocity = 0.0
            state.max_down_relative_velocity = 0.0

    def _finger_anchor_y(self, landmarks: List[Tuple[int, int, float]], finger_id: int) -> float:
        base_id = FINGER_BASE_IDS.get(finger_id)
        if base_id is not None and base_id < len(landmarks):
            return float(landmarks[base_id][1])
        anchor_ids = (0, 5, 9, 13, 17)
        values = [landmarks[idx][1] for idx in anchor_ids if idx < len(landmarks)]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _threshold_for(self, zone: Zone) -> float:
        return config.PIANO_HIT_VELOCITY_THRESHOLD if zone.kind == "piano" else config.HIT_VELOCITY_THRESHOLD

    def _velocity_to_volume(self, velocity_y: float, zone: Zone) -> float:
        if zone.kind == "piano":
            normalized = (
                (velocity_y - config.PIANO_HIT_MIN_VELOCITY)
                / (config.PIANO_HIT_MAX_VELOCITY - config.PIANO_HIT_MIN_VELOCITY)
            )
            return clamp(normalized, config.PIANO_MIN_VOLUME, 1.0)
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

    def _zone_for_state(
        self,
        zones: Iterable[Zone],
        point: Tuple[int, int],
        state: FingerState,
        previous_position: Optional[Tuple[int, int]],
    ) -> Optional[Zone]:
        zone_list = list(zones)
        current = self._zone_at(zone_list, point)
        sticky = self._sticky_piano_zone(zone_list, point, state, previous_position, current)
        selected = sticky or current
        state.last_zone_id = selected.sound_id if selected else None
        return selected

    def _sticky_piano_zone(
        self,
        zones: Iterable[Zone],
        point: Tuple[int, int],
        state: FingerState,
        previous_position: Optional[Tuple[int, int]],
        current: Optional[Zone],
    ) -> Optional[Zone]:
        if not config.PIANO_ZONE_STICKY_ENABLED or previous_position is None or not state.last_zone_id:
            return None
        if state.motion_state == "falling" or self._motion_velocity(state) > config.PIANO_FALLING_VELOCITY_THRESHOLD:
            return None
        if current is not None and current.kind != "piano":
            return None
        previous_zone = next((zone for zone in zones if zone.sound_id == state.last_zone_id), None)
        if previous_zone is None or previous_zone.kind != "piano":
            return None
        if current is not None and current.sound_id == previous_zone.sound_id:
            return None
        step_x = abs(point[0] - previous_position[0])
        if step_x > config.PIANO_ZONE_STICKY_MAX_STEP_PX:
            return None
        if self._contains_with_piano_sticky_margin(previous_zone, point):
            return previous_zone
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

    def _contains_with_piano_sticky_margin(self, zone: Zone, point: Tuple[int, int]) -> bool:
        x, y = point
        x_margin = zone.width * config.PIANO_ZONE_STICKY_X_MARGIN_RATIO
        top_margin = zone.height * config.PIANO_HIT_TOP_MARGIN_RATIO
        bottom_margin = zone.height * config.PIANO_HIT_BOTTOM_MARGIN_RATIO
        return (
            zone.x1 - x_margin <= x <= zone.x2 + x_margin
            and zone.y1 - top_margin <= y <= zone.y2 + bottom_margin
        )
