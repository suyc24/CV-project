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
    previous_depth_height_m: Optional[float] = None
    previous_depth_timestamp: Optional[float] = None
    smoothed_depth_down_velocity_m_s: float = 0.0
    raw_depth_down_velocity_m_s: float = 0.0
    depth_motion_state: str = "idle"
    depth_armed_zone_id: Optional[str] = None
    depth_falling_frames: int = 0
    depth_release_ready_frames: int = 0
    depth_peak_height_m: Optional[float] = None
    max_down_depth_velocity_m_s: float = 0.0
    depth_valid_frames: int = 0
    depth_missing_frames: int = 0
    fallback_rest_y: Optional[int] = None
    fallback_rest_relative_y: Optional[float] = None
    fallback_rest_zone_id: Optional[str] = None
    fallback_rest_frames: int = 0
    last_zone_id: Optional[str] = None
    recent_motion: Deque[Tuple[float, int, float]] = field(default_factory=lambda: deque(maxlen=8))
    trail: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=config.TRAIL_LENGTH))


@dataclass
class HandTapState:
    missing_depth_owner_finger_id: Optional[int] = None
    missing_depth_owner_until: float = -999.0
    missing_depth_owner_lift_px: float = 0.0
    missing_depth_owner_committed_until: float = -999.0


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
        self._hand_tap_states: Dict[int, HandTapState] = {}
        self._diagnostics: List[Dict[str, object]] = []

    def reset(self) -> None:
        self._states.clear()
        self._hand_tap_states.clear()
        self._diagnostics.clear()

    def _state_for(self, hand_id: int, finger_id: int, current_time: float) -> FingerState:
        key = (hand_id, finger_id)
        state = self._states.get(key)
        if state is not None:
            return state
        state = FingerState()
        if self._piano_trigger_mode() == "3d":
            donor = self._recent_state_for_finger(finger_id, current_time)
            if donor is not None:
                self._copy_3d_continuity(donor, state)
        self._states[key] = state
        return state

    def _hand_tap_state_for(self, hand_id: int) -> HandTapState:
        state = self._hand_tap_states.get(hand_id)
        if state is None:
            state = HandTapState()
            self._hand_tap_states[hand_id] = state
        return state

    def _recent_state_for_finger(self, finger_id: int, current_time: float) -> Optional[FingerState]:
        max_age = float(getattr(config, "PIANO_3D_STATE_CONTINUITY_SECONDS", 0.75))
        candidates = [
            state
            for (candidate_hand_id, candidate_finger_id), state in self._states.items()
            if candidate_finger_id == finger_id
            and state.previous_timestamp is not None
            and 0.0 <= current_time - state.previous_timestamp <= max_age
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda candidate: candidate.previous_timestamp or -999.0)

    def _copy_3d_continuity(self, source: FingerState, target: FingerState) -> None:
        target.last_hit_time = source.last_hit_time
        target.motion_state = source.motion_state
        target.armed_zone_id = source.armed_zone_id
        target.peak_y = source.peak_y
        target.peak_relative_y = source.peak_relative_y
        target.falling_frames = source.falling_frames
        target.depth_motion_state = source.depth_motion_state
        target.depth_armed_zone_id = source.depth_armed_zone_id
        target.depth_peak_height_m = source.depth_peak_height_m
        target.max_down_depth_velocity_m_s = source.max_down_depth_velocity_m_s
        target.fallback_rest_y = source.fallback_rest_y
        target.fallback_rest_relative_y = source.fallback_rest_relative_y
        target.fallback_rest_zone_id = source.fallback_rest_zone_id
        target.fallback_rest_frames = source.fallback_rest_frames

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
                state = self._state_for(hand.hand_id, finger_id, current_time)
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
                            "depth_down_velocity_m_s": state.smoothed_depth_down_velocity_m_s,
                            "depth_raw_down_velocity_m_s": state.raw_depth_down_velocity_m_s,
                            "depth_motion_state": state.depth_motion_state,
                            "depth_reason": depth_observation.reason if depth_observation else None,
                            "depth_valid_frames": state.depth_valid_frames,
                            "depth_missing_frames": state.depth_missing_frames,
                            "depth_finger_samples": getattr(depth_observation, "finger_sample_count", None) if depth_observation else None,
                            "depth_desk_samples": getattr(depth_observation, "desk_sample_count", None) if depth_observation else None,
                            "depth_sample_radius_px": getattr(depth_observation, "sample_radius_px", None) if depth_observation else None,
                            "depth_desk_source": getattr(depth_observation, "desk_depth_source", None) if depth_observation else None,
                            "tracking_source": getattr(hand, "tracking_source", "mediapipe"),
                            "missed_frames": getattr(hand, "missed_frames", 0),
                            "unstable_tracking": True,
                        }
                    )
                    continue
                velocity_y = self._update_velocity(state, position, relative_y, current_time)
                zone = self._zone_for_state(zones, position, state, previous_position)
                depth_observation = (depth_observations or {}).get((hand.hand_id, finger_id))
                hand_has_valid_depth = self._hand_has_valid_depth(depth_observations or {}, hand.hand_id, finger_id)
                self._update_release_state(state, zone, y, relative_y, current_time, depth_observation)

                reason = self._miss_reason(
                    state,
                    zone,
                    hand.hand_id,
                    finger_id,
                    y,
                    velocity_y,
                    relative_y,
                    current_time,
                    previous_position,
                    previous_relative_y,
                    depth_observation,
                    hand_has_valid_depth,
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
                    "depth_down_velocity_m_s": state.smoothed_depth_down_velocity_m_s,
                    "depth_raw_down_velocity_m_s": state.raw_depth_down_velocity_m_s,
                    "depth_motion_state": state.depth_motion_state,
                    "depth_reason": depth_observation.reason if depth_observation else None,
                    "depth_valid_frames": state.depth_valid_frames,
                    "depth_missing_frames": state.depth_missing_frames,
                    "depth_finger_samples": getattr(depth_observation, "finger_sample_count", None) if depth_observation else None,
                    "depth_desk_samples": getattr(depth_observation, "desk_sample_count", None) if depth_observation else None,
                    "depth_sample_radius_px": getattr(depth_observation, "sample_radius_px", None) if depth_observation else None,
                    "depth_desk_source": getattr(depth_observation, "desk_depth_source", None) if depth_observation else None,
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
        state.depth_motion_state = "pressed"
        state.depth_armed_zone_id = zone.sound_id
        state.depth_falling_frames = 0
        state.depth_release_ready_frames = 0
        state.depth_peak_height_m = None
        state.max_down_depth_velocity_m_s = 0.0
        if zone.kind == "piano" and self._piano_trigger_mode() == "3d":
            self._seed_missing_depth_rest(state, zone, finger_y, relative_y)
            self._commit_missing_depth_owner(hand_id, finger_id, current_time)
        return hit

    def _hit_score(self, state: FingerState, finger_id: int, finger_y: int, relative_y: float) -> float:
        drop = self._drop_distance(state, finger_y, relative_y)
        score = self._hit_velocity(state, state.smoothed_velocity_y) + drop * 8.0
        if state.depth_peak_height_m is not None and state.previous_depth_height_m is not None:
            score += max(0.0, state.depth_peak_height_m - state.previous_depth_height_m) * 8000.0
            score += state.max_down_depth_velocity_m_s * 300.0
        if finger_id == 4:
            score *= config.PIANO_THUMB_SCORE_WEIGHT
        return score

    def _hit_velocity(self, state: FingerState, velocity_y: float) -> float:
        if self._piano_trigger_mode() != "2d" and state.max_down_depth_velocity_m_s > 0:
            depth_velocity_as_px = state.max_down_depth_velocity_m_s * 1000.0
            velocity_y = max(velocity_y, depth_velocity_as_px)
        if config.PIANO_USE_RELATIVE_FINGER_MOTION:
            return max(
                velocity_y,
                state.smoothed_relative_velocity_y,
                state.raw_relative_velocity_y,
                state.max_down_relative_velocity,
            )
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
        current_time: float,
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
            state.depth_motion_state = "idle"
            state.depth_armed_zone_id = None
            state.depth_falling_frames = 0
            state.depth_release_ready_frames = 0
            state.depth_peak_height_m = None
            state.max_down_depth_velocity_m_s = 0.0
            state.last_zone_id = None
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
            state.depth_motion_state = "idle"
            state.depth_armed_zone_id = None
            state.depth_falling_frames = 0
            state.depth_release_ready_frames = 0
            state.depth_peak_height_m = None
            state.max_down_depth_velocity_m_s = 0.0
            state.last_zone_id = None
            return
        if zone.kind == "piano":
            depth_release_handled = self._update_piano_depth_release_state(
                state,
                zone,
                finger_y,
                relative_y,
                current_time,
                depth_observation,
            )
            if depth_release_handled:
                return
            if self._piano_trigger_mode() == "3d":
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
                state.depth_motion_state = "raised"
                state.depth_armed_zone_id = zone.sound_id
                state.depth_falling_frames = 0
                state.depth_release_ready_frames = 0
                state.depth_peak_height_m = self._depth_height(depth_observation)
                state.max_down_depth_velocity_m_s = 0.0
                state.max_down_velocity = 0.0
                state.max_down_relative_velocity = 0.0
                state.last_zone_id = zone.sound_id
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
            state.last_zone_id = zone.sound_id

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
        hand_id: int,
        finger_id: int,
        finger_y: int,
        velocity_y: float,
        relative_y: float,
        current_time: float,
        previous_position: Optional[Tuple[int, int]],
        previous_relative_y: Optional[float],
        depth_observation: Optional["DepthObservation"],
        hand_has_valid_depth: bool = False,
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
                hand_id,
                finger_id,
                finger_y,
                velocity_y,
                relative_y,
                current_time,
                previous_position,
                previous_relative_y,
                depth_observation,
                hand_has_valid_depth,
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
        hand_id: int,
        finger_id: int,
        finger_y: int,
        velocity_y: float,
        relative_y: float,
        current_time: float,
        previous_position: Optional[Tuple[int, int]],
        previous_relative_y: Optional[float],
        depth_observation: Optional["DepthObservation"],
        hand_has_valid_depth: bool = False,
    ) -> str:
        trigger_mode = self._piano_trigger_mode()
        if trigger_mode != "2d":
            depth_reason = self._piano_depth_miss_reason(
                state,
                zone,
                hand_id,
                finger_id,
                finger_y,
                relative_y,
                current_time,
                depth_observation,
            )
            if depth_reason == "hit":
                return "hit"
            if trigger_mode == "3d":
                fallback_reason = self._piano_missing_depth_miss_reason(
                    state,
                    zone,
                    hand_id,
                    finger_id,
                    finger_y,
                    velocity_y,
                    relative_y,
                    current_time,
                    depth_observation,
                    hand_has_valid_depth,
                )
                if fallback_reason == "hit":
                    return "hit"
                return fallback_reason or depth_reason or "depth_unavailable"

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
        if strike_velocity >= config.PIANO_STRIKE_MIN_VELOCITY:
            return "hit"
        return "velocity"

    def _piano_depth_miss_reason(
        self,
        state: FingerState,
        zone: Zone,
        hand_id: int,
        finger_id: int,
        finger_y: int,
        relative_y: float,
        current_time: float,
        observation: Optional["DepthObservation"],
    ) -> Optional[str]:
        if self._piano_trigger_mode() == "3d" and self._piano_3d_ignores_finger(finger_id):
            return "depth_finger_ignored"
        height = self._depth_height(observation)
        if height is None:
            state.depth_missing_frames += 1
            if state.depth_missing_frames > max(2, config.PIANO_RELEASE_STABLE_FRAMES) and not state.is_pressed:
                state.depth_motion_state = "idle"
                state.depth_armed_zone_id = None
                state.depth_falling_frames = 0
                state.depth_release_ready_frames = 0
                state.depth_peak_height_m = None
                state.max_down_depth_velocity_m_s = 0.0
                state.depth_valid_frames = 0
            return None

        state.depth_missing_frames = 0
        state.depth_valid_frames += 1
        depth_velocity = self._update_depth_velocity(state, height, current_time)
        (
            arm_height,
            release_height,
            press_height,
            min_drop,
            falling_velocity,
            strike_velocity,
        ) = self._piano_depth_thresholds()

        if state.depth_motion_state in {"armed", "raised", "falling"}:
            state.depth_armed_zone_id = zone.sound_id

        if height >= arm_height and state.depth_motion_state != "falling":
            state.depth_motion_state = "armed"
            state.depth_armed_zone_id = zone.sound_id
            state.depth_falling_frames = 0
            state.depth_release_ready_frames = 0
            state.depth_peak_height_m = max(state.depth_peak_height_m or height, height)
            state.max_down_depth_velocity_m_s = 0.0
            self._clear_missing_depth_rest_if_air(state, height, release_height)
            return "depth_armed"

        if state.depth_motion_state not in {"armed", "raised", "falling"}:
            state.depth_peak_height_m = height
            state.max_down_depth_velocity_m_s = 0.0
            if height <= press_height:
                state.depth_motion_state = "resting"
                state.depth_armed_zone_id = zone.sound_id
                self._seed_missing_depth_rest(state, zone, finger_y, relative_y)
                return "depth_resting"
            self._clear_missing_depth_rest_if_air(state, height, release_height)
            return "depth_lift_too_low"

        peak = max(state.depth_peak_height_m if state.depth_peak_height_m is not None else height, height)
        state.depth_peak_height_m = peak
        drop = peak - height

        if state.depth_motion_state == "falling" and height >= release_height and depth_velocity <= 0.0:
            state.depth_motion_state = "armed"
            state.depth_armed_zone_id = zone.sound_id
            state.depth_falling_frames = 0
            state.depth_peak_height_m = height
            state.max_down_depth_velocity_m_s = 0.0
            return "depth_lifted"

        can_start_fall = state.depth_motion_state in {"armed", "raised", "falling"}
        falling_by_velocity = depth_velocity >= falling_velocity
        falling_by_drop = drop >= min_drop * 0.5
        falling_by_landing = height <= press_height and drop > 0.0
        if can_start_fall and (falling_by_velocity or falling_by_drop or falling_by_landing):
            if state.depth_motion_state != "falling":
                state.depth_falling_frames = 1
            else:
                state.depth_falling_frames += 1
            state.depth_motion_state = "falling"
            state.depth_armed_zone_id = zone.sound_id
            state.max_down_depth_velocity_m_s = max(state.max_down_depth_velocity_m_s, depth_velocity)
        elif state.depth_motion_state == "falling":
            state.depth_falling_frames += 1
            state.max_down_depth_velocity_m_s = max(state.max_down_depth_velocity_m_s, depth_velocity)

        if state.depth_motion_state != "falling":
            return "depth_armed"
        if state.depth_falling_frames < config.PIANO_MIN_FALL_FRAMES:
            return "depth_falling"
        if self._piano_trigger_mode() == "3d" and state.depth_valid_frames < 3:
            return "depth_warming"
        if height > press_height:
            return "depth_air"

        drop = peak - height
        if drop < min_drop:
            state.depth_motion_state = "resting"
            state.depth_armed_zone_id = zone.sound_id
            state.depth_falling_frames = 0
            state.depth_peak_height_m = height
            state.max_down_depth_velocity_m_s = 0.0
            self._seed_missing_depth_rest(state, zone, finger_y, relative_y)
            return "depth_short_drop"
        strong_drop = drop >= min_drop * 1.35
        if state.max_down_depth_velocity_m_s < strike_velocity and not strong_drop:
            return "depth_velocity"
        recent_velocity = float(getattr(config, "PIANO_3D_STRIKE_RECENT_VELOCITY_M_S", 0.0))
        if recent_velocity > 0.0 and depth_velocity < recent_velocity:
            return "depth_recent_velocity"
        if self._piano_trigger_mode() == "3d" and not self._depth_hit_has_2d_landing_support(zone, finger_id, finger_y):
            return "depth_landing_guard"
        if self._piano_trigger_mode() == "3d":
            owner_reason = self._claim_missing_depth_owner(
                hand_id,
                finger_id,
                current_time,
                max(0.0, drop * 1000.0),
            )
            if owner_reason:
                return owner_reason
        return "hit"

    def _piano_trigger_mode(self) -> str:
        mode = getattr(config, "PIANO_TRIGGER_MODE", "2d")
        if mode in {"2d", "hybrid", "3d"}:
            if mode == "2d" and getattr(config, "PIANO_DEPTH_TRIGGER_ENABLED", False):
                return "hybrid"
            return mode
        return "hybrid" if getattr(config, "PIANO_DEPTH_TRIGGER_ENABLED", False) else "2d"

    def _piano_depth_thresholds(self) -> Tuple[float, float, float, float, float, float]:
        if self._piano_trigger_mode() == "3d":
            return (
                config.PIANO_3D_ARM_HEIGHT_M,
                config.PIANO_3D_RELEASE_HEIGHT_M,
                config.PIANO_3D_PRESS_HEIGHT_M,
                config.PIANO_3D_MIN_DROP_M,
                config.PIANO_3D_FALLING_VELOCITY_M_S,
                config.PIANO_3D_STRIKE_MIN_VELOCITY_M_S,
            )
        return (
            config.PIANO_DEPTH_ARM_HEIGHT_M,
            config.PIANO_DEPTH_RELEASE_HEIGHT_M,
            config.PIANO_DEPTH_PRESS_HEIGHT_M,
            config.PIANO_DEPTH_MIN_DROP_M,
            config.PIANO_DEPTH_FALLING_VELOCITY_M_S,
            config.PIANO_DEPTH_STRIKE_MIN_VELOCITY_M_S,
        )

    def _piano_3d_ignores_finger(self, finger_id: int) -> bool:
        return bool(getattr(config, "PIANO_3D_IGNORE_THUMB", True)) and finger_id == 4

    def _hand_has_valid_depth(
        self,
        observations: Mapping[Tuple[int, int], "DepthObservation"],
        hand_id: int,
        excluded_finger_id: int,
    ) -> bool:
        for (candidate_hand_id, candidate_finger_id), observation in observations.items():
            if candidate_hand_id != hand_id or candidate_finger_id == excluded_finger_id:
                continue
            if candidate_finger_id == 4:
                continue
            if self._depth_height(observation) is not None:
                return True
        return False

    def _depth_hit_has_2d_landing_support(self, zone: Zone, finger_id: int, finger_y: int) -> bool:
        min_ratio = float(getattr(config, "PIANO_3D_DEPTH_HIT_MIN_KEY_Y_RATIO", 0.20))
        if finger_id == 4:
            min_ratio = max(
                min_ratio,
                float(getattr(config, "PIANO_3D_THUMB_DEPTH_HIT_MIN_KEY_Y_RATIO", min_ratio)),
            )
        if min_ratio <= 0.0:
            return True
        if zone.height <= 0:
            return True
        return (finger_y - zone.y1) / zone.height >= min_ratio

    def _missing_depth_key_y_supported(self, zone: Zone, finger_id: int, finger_y: int) -> bool:
        min_ratio = float(getattr(config, "PIANO_3D_MISSING_DEPTH_MIN_KEY_Y_RATIO", 0.10))
        if finger_id == 4:
            min_ratio = max(
                min_ratio,
                float(getattr(config, "PIANO_3D_THUMB_MISSING_DEPTH_MIN_KEY_Y_RATIO", min_ratio)),
            )
        if min_ratio <= 0.0:
            return True
        if zone.height <= 0:
            return True
        return (finger_y - zone.y1) / zone.height >= min_ratio

    def _seed_missing_depth_rest(
        self,
        state: FingerState,
        zone: Zone,
        finger_y: int,
        relative_y: float,
    ) -> None:
        state.fallback_rest_y = finger_y
        state.fallback_rest_relative_y = relative_y
        state.fallback_rest_zone_id = zone.sound_id
        state.fallback_rest_frames = max(state.fallback_rest_frames, 1)

    def _clear_missing_depth_rest_if_air(
        self,
        state: FingerState,
        height: float,
        release_height: float,
    ) -> None:
        if height < release_height:
            return
        state.fallback_rest_y = None
        state.fallback_rest_relative_y = None
        state.fallback_rest_zone_id = None
        state.fallback_rest_frames = 0

    def _piano_missing_depth_miss_reason(
        self,
        state: FingerState,
        zone: Zone,
        hand_id: int,
        finger_id: int,
        finger_y: int,
        velocity_y: float,
        relative_y: float,
        current_time: float,
        observation: Optional["DepthObservation"],
        hand_has_valid_depth: bool = False,
    ) -> Optional[str]:
        if not bool(getattr(config, "PIANO_3D_MISSING_DEPTH_FALLBACK_ENABLED", True)):
            return None
        if self._piano_3d_ignores_finger(finger_id):
            return "missing_depth_finger_ignored"
        if self._depth_height(observation) is not None:
            return None
        if observation is None:
            return None
        fallback_fingers = tuple(getattr(config, "PIANO_3D_MISSING_DEPTH_FINGER_IDS", (8, 16, 20)))
        if finger_id not in fallback_fingers:
            return "missing_depth_finger_ignored"
        if not self._missing_depth_zone_allowed(finger_id, zone):
            return "missing_depth_zone_ignored"
        if hand_has_valid_depth:
            return "missing_depth_hand_depth_guard"
        if not self._missing_depth_key_y_supported(zone, finger_id, finger_y):
            self._reset_missing_depth_lift(state)
            return "missing_depth_key_top"

        motion_y = relative_y if config.PIANO_USE_RELATIVE_FINGER_MOTION else float(finger_y)
        rest_y = (
            state.fallback_rest_relative_y
            if config.PIANO_USE_RELATIVE_FINGER_MOTION
            else float(state.fallback_rest_y) if state.fallback_rest_y is not None else None
        )
        min_rest = float(getattr(config, "PIANO_3D_MISSING_DEPTH_MIN_REST_RELATIVE_Y", 70.0))
        if rest_y is None:
            if self._missing_depth_initial_tap_supported(state, zone, finger_y):
                state.motion_state = "falling"
                state.armed_zone_id = zone.sound_id
                state.peak_y = None
                state.peak_relative_y = None
                state.max_down_velocity = max(state.max_down_velocity, velocity_y)
                state.max_down_relative_velocity = max(
                    state.max_down_relative_velocity,
                    self._motion_velocity(state),
                )
                return "hit"
            if (
                relative_y >= min_rest
                and abs(self._motion_velocity(state)) <= config.PIANO_3D_MISSING_DEPTH_REST_MAX_VELOCITY
            ):
                state.fallback_rest_frames += 1
                if state.fallback_rest_frames >= config.PIANO_3D_MISSING_DEPTH_REST_STABLE_FRAMES:
                    self._seed_missing_depth_rest(state, zone, finger_y, relative_y)
                    state.motion_state = "resting"
                    return "missing_depth_resting"
                return "missing_depth_rest_warming"
            state.fallback_rest_frames = 0
            return "depth_unavailable"

        lift_px = rest_y - motion_y
        lift_threshold = float(getattr(config, "PIANO_3D_MISSING_DEPTH_LIFT_PX", 16.0))
        absolute_lift = (
            float(state.fallback_rest_y - finger_y)
            if state.fallback_rest_y is not None
            else lift_px
        )
        min_absolute_lift = float(getattr(config, "PIANO_3D_MISSING_DEPTH_MIN_ABSOLUTE_LIFT_PX", 0.0))
        if lift_px >= lift_threshold:
            if absolute_lift < min_absolute_lift:
                return "missing_depth_abs_lift"
            owner_reason = self._claim_missing_depth_owner(
                hand_id,
                finger_id,
                current_time,
                max(lift_px, absolute_lift),
            )
            if owner_reason:
                return owner_reason
            if state.motion_state not in {"raised", "falling"}:
                state.motion_state = "raised"
                state.armed_zone_id = zone.sound_id
                state.peak_y = finger_y
                state.peak_relative_y = relative_y
                state.falling_frames = 0
                state.max_down_velocity = 0.0
                state.max_down_relative_velocity = 0.0
            else:
                state.armed_zone_id = zone.sound_id
                state.peak_y = finger_y if state.peak_y is None else min(state.peak_y, finger_y)
                state.peak_relative_y = relative_y if state.peak_relative_y is None else min(
                    state.peak_relative_y,
                    relative_y,
                )
            return "missing_depth_armed"

        if state.motion_state not in {"raised", "falling"}:
            self._maybe_update_missing_depth_rest(state, zone, finger_y, relative_y)
            return "missing_depth_resting"

        if state.motion_state != "falling":
            state.motion_state = "falling"
            state.falling_frames = 1
        else:
            state.falling_frames += 1
        state.armed_zone_id = zone.sound_id
        state.max_down_velocity = max(state.max_down_velocity, velocity_y)
        state.max_down_relative_velocity = max(state.max_down_relative_velocity, self._motion_velocity(state))
        owner_reason = self._claim_missing_depth_owner(
            hand_id,
            finger_id,
            current_time,
            max(self._drop_distance(state, finger_y, relative_y), self._absolute_drop_distance(state, finger_y)),
        )
        if owner_reason:
            return owner_reason

        drop_px = self._drop_distance(state, finger_y, relative_y)
        absolute_drop_px = self._absolute_drop_distance(state, finger_y)
        contact_margin = float(getattr(config, "PIANO_3D_MISSING_DEPTH_CONTACT_MARGIN_PX", 16.0))
        if motion_y < rest_y - contact_margin:
            return "missing_depth_air"
        if drop_px < config.PIANO_3D_MISSING_DEPTH_MIN_DROP_PX:
            return "missing_depth_short_drop"
        min_absolute_drop = float(getattr(config, "PIANO_3D_MISSING_DEPTH_MIN_ABSOLUTE_DROP_PX", 0.0))
        if absolute_drop_px < min_absolute_drop:
            return "missing_depth_abs_short_drop"
        min_absolute_velocity = float(getattr(config, "PIANO_3D_MISSING_DEPTH_MIN_ABSOLUTE_DOWN_VELOCITY", 0.0))
        if state.max_down_velocity < min_absolute_velocity:
            return "missing_depth_abs_velocity"
        max_lateral_step = float(getattr(config, "PIANO_3D_MISSING_DEPTH_MAX_LATERAL_STEP_PX", 0.0))
        if max_lateral_step > 0.0 and self._last_frame_lateral_step(state) > max_lateral_step:
            return "missing_depth_lateral_jump"

        strike_velocity = self._hit_velocity(state, velocity_y)
        strong_drop = drop_px >= config.PIANO_3D_MISSING_DEPTH_STRONG_DROP_PX
        if strike_velocity < config.PIANO_3D_MISSING_DEPTH_MIN_DOWN_VELOCITY and not strong_drop:
            return "missing_depth_velocity"
        if not self._missing_depth_finger_is_isolated(hand_id, finger_id, state, drop_px, current_time):
            return "missing_depth_not_isolated"
        return "hit"

    def _claim_missing_depth_owner(
        self,
        hand_id: int,
        finger_id: int,
        current_time: float,
        lift_or_drop_px: float,
    ) -> Optional[str]:
        if not bool(getattr(config, "PIANO_3D_MISSING_DEPTH_HAND_OWNER_ENABLED", True)):
            return None
        hand_state = self._hand_tap_state_for(hand_id)
        if current_time > hand_state.missing_depth_owner_until:
            hand_state.missing_depth_owner_finger_id = None
            hand_state.missing_depth_owner_lift_px = 0.0
        owner = hand_state.missing_depth_owner_finger_id
        if owner is None or owner == finger_id:
            hand_state.missing_depth_owner_finger_id = finger_id
            hand_state.missing_depth_owner_lift_px = max(
                hand_state.missing_depth_owner_lift_px,
                lift_or_drop_px,
            )
            hand_state.missing_depth_owner_until = (
                current_time + float(getattr(config, "PIANO_3D_MISSING_DEPTH_OWNER_TIMEOUT_SECONDS", 0.45))
            )
            return None

        committed = current_time <= hand_state.missing_depth_owner_committed_until
        steal_margin = float(getattr(config, "PIANO_3D_MISSING_DEPTH_OWNER_STEAL_MARGIN_PX", 5.0))
        if not committed and lift_or_drop_px >= hand_state.missing_depth_owner_lift_px + steal_margin:
            hand_state.missing_depth_owner_finger_id = finger_id
            hand_state.missing_depth_owner_lift_px = lift_or_drop_px
            hand_state.missing_depth_owner_until = (
                current_time + float(getattr(config, "PIANO_3D_MISSING_DEPTH_OWNER_TIMEOUT_SECONDS", 0.45))
            )
            return None
        return "missing_depth_other_finger_active"

    def _commit_missing_depth_owner(self, hand_id: int, finger_id: int, current_time: float) -> None:
        if not bool(getattr(config, "PIANO_3D_MISSING_DEPTH_HAND_OWNER_ENABLED", True)):
            return
        hand_state = self._hand_tap_state_for(hand_id)
        hand_state.missing_depth_owner_finger_id = finger_id
        hand_state.missing_depth_owner_until = (
            current_time + float(getattr(config, "PIANO_3D_MISSING_DEPTH_OWNER_TIMEOUT_SECONDS", 0.45))
        )
        hand_state.missing_depth_owner_committed_until = (
            current_time + float(getattr(config, "PIANO_3D_MISSING_DEPTH_OWNER_AFTER_HIT_SECONDS", 0.10))
        )

    def _missing_depth_finger_is_isolated(
        self,
        hand_id: int,
        finger_id: int,
        state: FingerState,
        drop_px: float,
        current_time: float,
    ) -> bool:
        if not bool(getattr(config, "PIANO_3D_MISSING_DEPTH_FINGER_ISOLATION_ENABLED", True)):
            return True
        lookback = float(getattr(config, "PIANO_3D_MISSING_DEPTH_ISOLATION_LOOKBACK_SECONDS", 0.16))
        target_drop = max(drop_px, self._recent_net_drop(state), self._absolute_drop_distance(state, state.previous_position[1] if state.previous_position else 0))
        target_velocity = max(
            state.max_down_relative_velocity,
            state.smoothed_relative_velocity_y,
            state.max_down_velocity,
            state.smoothed_velocity_y,
        )
        other_drops: List[float] = []
        other_velocities: List[float] = []
        for (candidate_hand_id, candidate_finger_id), candidate in self._states.items():
            if candidate_hand_id != hand_id or candidate_finger_id == finger_id:
                continue
            if candidate.previous_timestamp is None or current_time - candidate.previous_timestamp > lookback:
                continue
            if self._piano_3d_ignores_finger(candidate_finger_id):
                continue
            other_drops.append(max(0.0, self._recent_net_drop(candidate)))
            other_velocities.append(
                max(
                    0.0,
                    candidate.max_down_relative_velocity,
                    candidate.smoothed_relative_velocity_y,
                    candidate.max_down_velocity,
                    candidate.smoothed_velocity_y,
                )
            )
        if not other_drops and not other_velocities:
            return True
        max_other_drop = max(other_drops) if other_drops else 0.0
        max_other_velocity = max(other_velocities) if other_velocities else 0.0
        drop_margin = float(getattr(config, "PIANO_3D_MISSING_DEPTH_ISOLATION_MARGIN_PX", 4.0))
        velocity_margin = float(getattr(config, "PIANO_3D_MISSING_DEPTH_ISOLATION_VELOCITY_MARGIN", 24.0))
        return (
            target_drop >= max_other_drop + drop_margin
            or target_velocity >= max_other_velocity + velocity_margin
        )

    def _maybe_update_missing_depth_rest(
        self,
        state: FingerState,
        zone: Zone,
        finger_y: int,
        relative_y: float,
    ) -> None:
        if abs(self._motion_velocity(state)) > config.PIANO_3D_MISSING_DEPTH_REST_MAX_VELOCITY:
            return
        if relative_y < config.PIANO_3D_MISSING_DEPTH_MIN_REST_RELATIVE_Y:
            return
        state.fallback_rest_frames = min(
            config.PIANO_3D_MISSING_DEPTH_REST_STABLE_FRAMES,
            state.fallback_rest_frames + 1,
        )
        alpha = 0.20
        if state.fallback_rest_y is None:
            state.fallback_rest_y = finger_y
        else:
            state.fallback_rest_y = int(round(alpha * finger_y + (1.0 - alpha) * state.fallback_rest_y))
        if state.fallback_rest_relative_y is None:
            state.fallback_rest_relative_y = relative_y
        else:
            state.fallback_rest_relative_y = (
                alpha * relative_y + (1.0 - alpha) * state.fallback_rest_relative_y
            )
        state.fallback_rest_zone_id = zone.sound_id

    def _reset_missing_depth_lift(self, state: FingerState) -> None:
        if state.motion_state in {"raised", "falling"}:
            state.motion_state = "resting" if state.fallback_rest_relative_y is not None else "idle"
        state.armed_zone_id = None
        state.peak_y = None
        state.peak_relative_y = None
        state.falling_frames = 0
        state.max_down_velocity = 0.0
        state.max_down_relative_velocity = 0.0

    def _missing_depth_initial_tap_supported(self, state: FingerState, zone: Zone, finger_y: int) -> bool:
        if not bool(getattr(config, "PIANO_3D_MISSING_DEPTH_INITIAL_HIT_ENABLED", False)):
            return False
        if zone.height <= 0:
            return False
        min_ratio = float(getattr(config, "PIANO_3D_MISSING_DEPTH_INITIAL_KEY_Y_RATIO", 0.28))
        if (finger_y - zone.y1) / zone.height < min_ratio:
            return False
        return self._recent_net_drop(state) >= config.PIANO_3D_MISSING_DEPTH_INITIAL_DROP_PX

    def _missing_depth_zone_allowed(self, finger_id: int, zone: Zone) -> bool:
        allowed_by_finger = getattr(config, "PIANO_3D_MISSING_DEPTH_ALLOWED_ZONES_BY_FINGER", {})
        allowed = allowed_by_finger.get(finger_id) if isinstance(allowed_by_finger, dict) else None
        if not allowed:
            return True
        return zone.label in set(allowed)

    def _update_depth_velocity(self, state: FingerState, height: float, current_time: float) -> float:
        if state.previous_depth_height_m is None or state.previous_depth_timestamp is None:
            state.previous_depth_height_m = height
            state.previous_depth_timestamp = current_time
            state.raw_depth_down_velocity_m_s = 0.0
            state.smoothed_depth_down_velocity_m_s = 0.0
            return 0.0

        dt = max(1e-3, current_time - state.previous_depth_timestamp)
        raw_velocity = (state.previous_depth_height_m - height) / dt
        alpha = config.PIANO_DEPTH_VELOCITY_SMOOTHING_ALPHA
        state.raw_depth_down_velocity_m_s = raw_velocity
        state.smoothed_depth_down_velocity_m_s = (
            alpha * raw_velocity + (1.0 - alpha) * state.smoothed_depth_down_velocity_m_s
        )
        state.previous_depth_height_m = height
        state.previous_depth_timestamp = current_time
        return max(state.raw_depth_down_velocity_m_s, state.smoothed_depth_down_velocity_m_s)

    def _depth_height(self, observation: Optional["DepthObservation"]) -> Optional[float]:
        if observation is None or observation.contact is None:
            return None
        height = observation.height_above_desk_m
        if height is None:
            return None
        return max(0.0, float(height))

    def _drop_distance(self, state: FingerState, finger_y: int, relative_y: float) -> float:
        if config.PIANO_USE_RELATIVE_FINGER_MOTION:
            peak = state.peak_relative_y if state.peak_relative_y is not None else relative_y
            return relative_y - peak
        peak = state.peak_y if state.peak_y is not None else finger_y
        return float(finger_y - peak)

    def _absolute_drop_distance(self, state: FingerState, finger_y: int) -> float:
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

    def _update_piano_depth_release_state(
        self,
        state: FingerState,
        zone: Zone,
        finger_y: int,
        relative_y: float,
        current_time: float,
        observation: Optional["DepthObservation"],
    ) -> bool:
        trigger_mode = self._piano_trigger_mode()
        if trigger_mode == "2d":
            return False
        height = self._depth_height(observation)
        if height is None:
            if trigger_mode == "3d":
                return self._update_missing_depth_release_state(state, zone, finger_y, relative_y)
            return trigger_mode == "3d"
        state.previous_depth_height_m = height
        state.previous_depth_timestamp = current_time

        _, release_height, _, _, _, _ = self._piano_depth_thresholds()
        if height >= release_height:
            state.depth_release_ready_frames += 1
        else:
            state.depth_release_ready_frames = 0

        if state.depth_release_ready_frames >= config.PIANO_RELEASE_STABLE_FRAMES:
            state.is_pressed = False
            state.pressed_zone_id = None
            state.pressed_y = None
            state.pressed_relative_y = None
            state.motion_state = "raised"
            state.armed_zone_id = zone.sound_id
            state.lift_start_y = None
            state.lift_start_relative_y = None
            state.release_ready_frames = 0
            state.depth_release_ready_frames = 0
            state.falling_frames = 0
            state.peak_y = None
            state.peak_relative_y = None
            state.depth_motion_state = "armed"
            state.depth_armed_zone_id = zone.sound_id
            state.depth_falling_frames = 0
            state.depth_peak_height_m = height
            state.depth_missing_frames = 0
            state.max_down_velocity = 0.0
            state.max_down_relative_velocity = 0.0
            state.max_down_depth_velocity_m_s = 0.0
            state.last_zone_id = zone.sound_id
            return True
        return trigger_mode == "3d"

    def _update_missing_depth_release_state(
        self,
        state: FingerState,
        zone: Zone,
        finger_y: int,
        relative_y: float,
    ) -> bool:
        rest_y = (
            state.fallback_rest_relative_y
            if config.PIANO_USE_RELATIVE_FINGER_MOTION
            else float(state.fallback_rest_y) if state.fallback_rest_y is not None else None
        )
        if rest_y is None:
            return True
        motion_y = relative_y if config.PIANO_USE_RELATIVE_FINGER_MOTION else float(finger_y)
        lifted = rest_y - motion_y >= config.PIANO_3D_MISSING_DEPTH_LIFT_PX
        if lifted:
            state.depth_release_ready_frames += 1
        else:
            state.depth_release_ready_frames = 0
            self._maybe_update_missing_depth_rest(state, zone, finger_y, relative_y)

        if state.depth_release_ready_frames < config.PIANO_RELEASE_STABLE_FRAMES:
            return True

        state.is_pressed = False
        state.pressed_zone_id = None
        state.pressed_y = None
        state.pressed_relative_y = None
        state.motion_state = "raised"
        state.armed_zone_id = zone.sound_id
        state.lift_start_y = None
        state.lift_start_relative_y = None
        state.release_ready_frames = 0
        state.depth_release_ready_frames = 0
        state.falling_frames = 0
        state.peak_y = finger_y
        state.peak_relative_y = relative_y
        state.depth_motion_state = "armed"
        state.depth_armed_zone_id = zone.sound_id
        state.depth_falling_frames = 0
        state.depth_peak_height_m = None
        state.max_down_velocity = 0.0
        state.max_down_relative_velocity = 0.0
        state.max_down_depth_velocity_m_s = 0.0
        state.last_zone_id = zone.sound_id
        return True

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

    def _last_frame_lateral_step(self, state: FingerState) -> float:
        if len(state.trail) < 2:
            return 0.0
        previous = state.trail[-2]
        current = state.trail[-1]
        return abs(float(current[0] - previous[0]))

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
        if (
            self._piano_trigger_mode() != "3d"
            and (state.motion_state == "falling" or self._motion_velocity(state) > config.PIANO_FALLING_VELOCITY_THRESHOLD)
        ):
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
