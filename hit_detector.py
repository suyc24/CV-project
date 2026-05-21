from __future__ import annotations

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

    def reset(self) -> None:
        self._states.clear()
        self._diagnostics.clear()

    def update(
        self,
        hands: Iterable["HandLandmarks"],
        zones: List[Zone],
        current_time: float,
        depth_observations: Optional[Mapping[Tuple[int, int], "DepthObservation"]] = None,
    ) -> List[HitEvent]:
        hits: List[HitEvent] = []
        self._diagnostics = []
        for hand in hands:
            hand_candidates: List[Tuple[float, FingerState, Zone, int, int, int, float, float, Dict[str, object]]] = []
            for finger_id in self.finger_ids:
                if finger_id >= len(hand.landmarks):
                    continue
                state = self._states.setdefault((hand.hand_id, finger_id), FingerState())
                x, y, _ = hand.landmarks[finger_id]
                position = (x, y)
                relative_y = y - self._finger_anchor_y(hand.landmarks, finger_id)
                state.trail.append(position)
                previous_position = state.previous_position
                previous_relative_y = state.previous_relative_y
                unstable_tracking = self._finger_tracking_unstable(hand, finger_id)
                if unstable_tracking:
                    velocity_y = self._observe_unstable_tracking(state, position, relative_y, current_time)
                    zone = self._zone_at(zones, position)
                    depth_observation = (depth_observations or {}).get((hand.hand_id, finger_id))
                    self._diagnostics.append(
                        {
                            "hand_id": hand.hand_id,
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
                            "tracking_source": getattr(hand, "tracking_source", "mediapipe"),
                            "missed_frames": getattr(hand, "missed_frames", 0),
                            "unstable_tracking": True,
                        }
                    )
                    continue
                velocity_y = self._update_velocity(state, position, relative_y, current_time)
                zone = self._zone_at(zones, position)
                depth_observation = (depth_observations or {}).get((hand.hand_id, finger_id))
                self._update_release_state(state, zone, y, relative_y, depth_observation)

                reason = self._miss_reason(
                    state,
                    zone,
                    y,
                    velocity_y,
                    relative_y,
                    current_time,
                    previous_position,
                    previous_relative_y,
                    depth_observation,
                )
                diagnostic = {
                    "hand_id": hand.hand_id,
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
                    "tracking_source": getattr(hand, "tracking_source", "mediapipe"),
                    "missed_frames": getattr(hand, "missed_frames", 0),
                    "unstable_tracking": False,
                }
                self._diagnostics.append(diagnostic)
                if zone and reason == "hit":
                    if zone.kind == "piano":
                        score = self._hit_score(state, finger_id, y, relative_y)
                        hand_candidates.append(
                            (score, state, zone, hand.hand_id, finger_id, y, relative_y, velocity_y, diagnostic)
                        )
                    else:
                        hits.append(self._commit_hit(state, zone, hand.hand_id, finger_id, current_time, y, relative_y, velocity_y))
            if hand_candidates:
                hand_candidates.sort(key=lambda candidate: candidate[0], reverse=True)
                max_hits = max(1, int(config.PIANO_MAX_HITS_PER_HAND_PER_FRAME))
                for idx, candidate in enumerate(hand_candidates):
                    _, state, zone, hand_id, finger_id, y, relative_y, velocity_y, diagnostic = candidate
                    if idx < max_hits:
                        hits.append(self._commit_hit(state, zone, hand_id, finger_id, current_time, y, relative_y, velocity_y))
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
        return hit

    def _hit_score(self, state: FingerState, finger_id: int, finger_y: int, relative_y: float) -> float:
        drop = self._drop_distance(state, finger_y, relative_y)
        score = self._hit_velocity(state, state.smoothed_velocity_y) + drop * 8.0
        if finger_id == 4:
            score *= config.PIANO_THUMB_SCORE_WEIGHT
        return score

    def _hit_velocity(self, state: FingerState, velocity_y: float) -> float:
        if config.PIANO_USE_RELATIVE_FINGER_MOTION:
            return max(state.smoothed_relative_velocity_y, state.raw_relative_velocity_y, state.max_down_relative_velocity)
        return max(velocity_y, state.max_down_velocity)

    def _motion_velocity(self, state: FingerState) -> float:
        return state.smoothed_relative_velocity_y if config.PIANO_USE_RELATIVE_FINGER_MOTION else state.smoothed_velocity_y

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
            return
        if state.pressed_zone_id and zone.sound_id != state.pressed_zone_id and zone.kind != "piano":
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
            return
        if zone.kind == "piano":
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
            deliberate_lift = (
                self._motion_velocity(state) <= -config.PIANO_RELEASE_MIN_UP_VELOCITY
                or lift_amount >= config.PIANO_RELEASE_LIFT_PX * config.PIANO_RELEASE_STRONG_LIFT_MULTIPLIER
            )
            lifted_enough = lifted_enough and deliberate_lift and self._passes_release_motion_guard(state)
            if lifted_enough and self._depth_still_contacting(depth_observation):
                lifted_enough = False
            if lifted_enough:
                state.release_ready_frames += 1
            else:
                state.release_ready_frames = 0
            if state.release_ready_frames >= config.PIANO_RELEASE_STABLE_FRAMES:
                state.is_pressed = False
                state.pressed_zone_id = None
                state.pressed_y = None
                state.pressed_relative_y = None
                state.motion_state = "raised"
                state.armed_zone_id = zone.sound_id
                state.lift_start_y = None
                state.lift_start_relative_y = None
                state.release_ready_frames = 0
                state.falling_frames = 0
                state.peak_y = finger_y
                state.peak_relative_y = relative_y
                state.max_down_velocity = 0.0
                state.max_down_relative_velocity = 0.0
            return
        if finger_y < zone.release_y:
            state.is_pressed = False
            state.pressed_zone_id = None
            state.pressed_y = None
            state.pressed_relative_y = None
            state.motion_state = "raised"
            state.armed_zone_id = zone.sound_id
            state.lift_start_y = None
            state.lift_start_relative_y = None
            state.release_ready_frames = 0
            state.falling_frames = 0
            state.peak_y = finger_y
            state.peak_relative_y = relative_y
            state.max_down_velocity = 0.0
            state.max_down_relative_velocity = 0.0

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
                finger_y,
                velocity_y,
                relative_y,
                previous_position,
                previous_relative_y,
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
        finger_y: int,
        velocity_y: float,
        relative_y: float,
        previous_position: Optional[Tuple[int, int]],
        previous_relative_y: Optional[float],
        depth_observation: Optional["DepthObservation"],
    ) -> str:
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
            if lift_px >= config.PIANO_ARM_MIN_LIFT_PX or finger_y <= arm_y:
                state.motion_state = "raised"
                state.falling_frames = 0
                return "armed"
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
        if strike_velocity >= config.PIANO_STRIKE_MIN_VELOCITY:
            return "hit"
        return "velocity"

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

    def _contains_with_piano_margin(self, zone: Zone, point: Tuple[int, int]) -> bool:
        x, y = point
        x_margin = zone.width * config.PIANO_HIT_X_MARGIN_RATIO
        top_margin = zone.height * config.PIANO_HIT_TOP_MARGIN_RATIO
        bottom_margin = zone.height * config.PIANO_HIT_BOTTOM_MARGIN_RATIO
        return (
            zone.x1 - x_margin <= x <= zone.x2 + x_margin
            and zone.y1 - top_margin <= y <= zone.y2 + bottom_margin
        )
