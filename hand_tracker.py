from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import math
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import cv2
import numpy as np

import config
from fingertip_refiner import FingertipRefiner


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass
class HandLandmarks:
    hand_id: int
    label: str
    landmarks: List[Tuple[int, int, float]]
    normalized_landmarks: List[Tuple[float, float, float]]
    tracking_source: str = "mediapipe"
    missed_frames: int = 0
    unstable_landmark_ids: Tuple[int, ...] = ()


class HandTracker:
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.55,
        min_tracking_confidence: float = 0.55,
        input_max_width: int = config.TRACKING_MAX_WIDTH,
        smooth_landmarks: bool = True,
        smoothing_alpha: float = config.LANDMARK_SMOOTHING_ALPHA,
        refine_fingertips: bool = config.FINGERTIP_REFINEMENT_ENABLED,
    ) -> None:
        self._backend = ""
        self._hands = None
        self._landmarker = None
        self._mp = None
        self._max_num_hands = max(1, int(max_num_hands))
        self._last_timestamp_ms = 0
        self._frame_index = 0
        self._input_max_width = input_max_width
        self._smooth_landmarks = smooth_landmarks
        self._smoothing_alpha = smoothing_alpha
        self._smoothed_points: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
        self._tracked_hand_centers: Dict[int, Tuple[float, float, str]] = {}
        self._next_stable_hand_id = 0
        self._last_gray = None
        self._flow_points: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self._last_good_hands: List[HandLandmarks] = []
        self._missed_frame_count = 0
        self._empty_detection_frames = 0
        self._reacquire_guard_frames = 0
        self._new_hand_guard_frames: Dict[int, int] = {}
        self._fingertip_refiner = (
            FingertipRefiner(
                finger_ids=config.TRIGGER_FINGER_IDS,
                radius_px=config.FINGERTIP_REFINEMENT_RADIUS_PX,
                max_shift_px=config.FINGERTIP_REFINEMENT_MAX_SHIFT_PX,
                max_perpendicular_shift_px=config.FINGERTIP_REFINEMENT_MAX_PERPENDICULAR_SHIFT_PX,
                min_edge_score=config.FINGERTIP_REFINEMENT_MIN_EDGE_SCORE,
                blend_alpha=config.FINGERTIP_REFINEMENT_BLEND_ALPHA,
                forward_bias=config.FINGERTIP_REFINEMENT_FORWARD_BIAS,
            )
            if refine_fingertips
            else None
        )

        try:
            self._init_legacy_hands(
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            return
        except Exception:
            pass

        try:
            self._init_tasks_landmarker(
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except Exception as exc:  # pragma: no cover - depends on local install
            raise RuntimeError(
                "MediaPipe hand tracking could not be initialized. This project supports "
                "both legacy `solutions.hands` and the newer Tasks Hand Landmarker API. "
                "If the model download failed, manually download "
                f"{config.HAND_LANDMARKER_MODEL_URL} to {config.HAND_LANDMARKER_MODEL_PATH}. "
                f"Original error: {exc}"
            ) from exc

    def _init_legacy_hands(
        self,
        max_num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        self._mp_hands = self._load_legacy_hands_module()
        try:
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                model_complexity=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"MediaPipe Hands initialization failed: {exc}") from exc
        self._backend = "legacy"

    def _load_legacy_hands_module(self):
        import mediapipe as mp

        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "hands"):
            return solutions.hands

        # Newer wheels may not expose mp.solutions at package top level, while
        # the legacy modules can still be imported from the internal path.
        return import_module("mediapipe.python.solutions.hands")

    def _init_tasks_landmarker(
        self,
        max_num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = self._ensure_landmarker_model()
        options = vision.HandLandmarkerOptions(
            # Passing an absolute Windows path can be misinterpreted by some
            # MediaPipe wheels as a package-relative resource. A byte buffer is
            # portable across Windows paths, spaces, and non-ASCII directories.
            base_options=mp_python.BaseOptions(model_asset_buffer=model_path.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._backend = "tasks"

    def _ensure_landmarker_model(self) -> Path:
        model_path = Path(config.HAND_LANDMARKER_MODEL_PATH)
        if model_path.exists() and model_path.stat().st_size > 0:
            return model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe Hand Landmarker model to {model_path}...")
        urlretrieve(config.HAND_LANDMARKER_MODEL_URL, model_path)
        return model_path

    def process(self, frame_bgr, roi: Optional[Tuple[int, int, int, int]] = None) -> List[HandLandmarks]:
        self._frame_index += 1
        hands = self._detect(frame_bgr, roi)
        if not hands and roi is not None and config.TRACKING_FULL_FRAME_REACQUIRE:
            hands = self._detect(frame_bgr, None)
        elif roi is not None and 0 < len(hands) < self._max_num_hands and self._should_partial_full_frame_reacquire():
            full_frame_hands = self._detect(frame_bgr, None)
            if len(full_frame_hands) > len(hands):
                hands = full_frame_hands

        if not hands:
            bridged = self._bridge_missing_hands(frame_bgr)
            if bridged:
                return bridged
            self._empty_detection_frames += 1
            self._clear_after_miss(frame_bgr)
            return []

        missed_before = max(self._missed_frame_count, self._empty_detection_frames)
        full_miss_before = self._empty_detection_frames
        self._missed_frame_count = 0
        self._empty_detection_frames = 0
        hands = self._assign_stable_hand_ids(hands)
        hands = self._smooth(hands)
        hands = self._refine_fingertips(frame_bgr, hands)
        hands = self._stabilize_with_optical_flow(frame_bgr, hands)
        if missed_before > 0:
            guard_frames = (
                config.TRACKING_FULL_MISS_REACQUIRE_HIT_BLOCK_FRAMES
                if full_miss_before > 0
                else config.TRACKING_REACQUIRE_HIT_BLOCK_FRAMES
            )
            self._reacquire_guard_frames = max(
                self._reacquire_guard_frames,
                int(guard_frames),
            )
        hands = self._apply_hit_guards(hands)
        self._last_good_hands = self._copy_hands(hands)
        return hands

    def _should_partial_full_frame_reacquire(self) -> bool:
        if not config.TRACKING_PARTIAL_FULL_FRAME_REACQUIRE:
            return False
        interval = max(1, int(config.TRACKING_PARTIAL_REACQUIRE_INTERVAL_FRAMES))
        return self._frame_index % interval == 0

    def _refine_fingertips(self, frame_bgr, hands: List[HandLandmarks]) -> List[HandLandmarks]:
        if self._fingertip_refiner is None:
            return hands
        refined_hands: List[HandLandmarks] = []
        for hand in hands:
            refined_landmarks = self._fingertip_refiner.refine_landmarks(frame_bgr, hand.landmarks)
            refined_hands.append(
                HandLandmarks(
                    hand_id=hand.hand_id,
                    label=hand.label,
                    landmarks=refined_landmarks,
                    normalized_landmarks=hand.normalized_landmarks,
                    tracking_source=f"{hand.tracking_source}+refined",
                    missed_frames=hand.missed_frames,
                    unstable_landmark_ids=hand.unstable_landmark_ids,
                )
            )
        return refined_hands

    def _detect(self, frame_bgr, roi: Optional[Tuple[int, int, int, int]] = None) -> List[HandLandmarks]:
        prepared_frame, offset, scale = self._prepare_frame(frame_bgr, roi)
        if self._backend == "tasks":
            return self._process_tasks(prepared_frame, offset, scale)
        return self._process_legacy(prepared_frame, offset, scale)

    def reset(self) -> None:
        self._smoothed_points.clear()
        self._tracked_hand_centers.clear()
        self._flow_points.clear()
        self._last_good_hands = []
        self._missed_frame_count = 0
        self._empty_detection_frames = 0
        self._reacquire_guard_frames = 0
        self._new_hand_guard_frames.clear()
        self._last_gray = None

    def _process_legacy(
        self,
        frame_bgr,
        offset: Tuple[int, int],
        scale: float,
    ) -> List[HandLandmarks]:
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._hands.process(frame_rgb)

        hands: List[HandLandmarks] = []
        if not results.multi_hand_landmarks:
            return hands

        handedness = results.multi_handedness or []
        for hand_id, hand_lms in enumerate(results.multi_hand_landmarks):
            label = "Unknown"
            if hand_id < len(handedness):
                label = handedness[hand_id].classification[0].label
            normalized = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]
            pixels = [
                self._map_point(lm.x, lm.y, lm.z, width, height, offset, scale)
                for lm in hand_lms.landmark
            ]
            hands.append(
                HandLandmarks(
                    hand_id=hand_id,
                    label=label,
                    landmarks=pixels,
                    normalized_landmarks=normalized,
                    tracking_source="legacy",
                )
            )
        return hands

    def _process_tasks(
        self,
        frame_bgr,
        offset: Tuple[int, int],
        scale: float,
    ) -> List[HandLandmarks]:
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.perf_counter() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        hands: List[HandLandmarks] = []
        for hand_id, hand_lms in enumerate(result.hand_landmarks or []):
            label = "Unknown"
            if result.handedness and hand_id < len(result.handedness) and result.handedness[hand_id]:
                category = result.handedness[hand_id][0]
                label = getattr(category, "category_name", None) or getattr(category, "display_name", None) or "Unknown"
            normalized = [(lm.x, lm.y, lm.z) for lm in hand_lms]
            pixels = [self._map_point(lm.x, lm.y, lm.z, width, height, offset, scale) for lm in hand_lms]
            hands.append(
                HandLandmarks(
                    hand_id=hand_id,
                    label=label,
                    landmarks=pixels,
                    normalized_landmarks=normalized,
                    tracking_source="tasks",
                )
            )
        return hands

    def _prepare_frame(
        self,
        frame_bgr,
        roi: Optional[Tuple[int, int, int, int]],
    ):
        frame_height, frame_width = frame_bgr.shape[:2]
        if roi is None:
            x1, y1, x2, y2 = 0, 0, frame_width, frame_height
        else:
            x1, y1, x2, y2 = roi
            x1 = max(0, min(frame_width - 1, x1))
            x2 = max(x1 + 1, min(frame_width, x2))
            y1 = max(0, min(frame_height - 1, y1))
            y2 = max(y1 + 1, min(frame_height, y2))

        cropped = frame_bgr[y1:y2, x1:x2]
        scale = 1.0
        if self._input_max_width > 0 and cropped.shape[1] > self._input_max_width:
            scale = self._input_max_width / float(cropped.shape[1])
            target_size = (self._input_max_width, max(1, int(cropped.shape[0] * scale)))
            cropped = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
        return cropped, (x1, y1), scale

    def _map_point(
        self,
        norm_x: float,
        norm_y: float,
        norm_z: float,
        width: int,
        height: int,
        offset: Tuple[int, int],
        scale: float,
    ) -> Tuple[int, int, float]:
        x = int(norm_x * width / scale + offset[0])
        y = int(norm_y * height / scale + offset[1])
        return (x, y, float(norm_z))

    def _smooth(self, hands: List[HandLandmarks]) -> List[HandLandmarks]:
        if not self._smooth_landmarks:
            return hands
        alpha = self._smoothing_alpha
        next_keys = set()
        smoothed_hands: List[HandLandmarks] = []
        for hand in hands:
            smoothed_landmarks = []
            for idx, (x, y, z) in enumerate(hand.landmarks):
                key = (hand.hand_id, idx)
                next_keys.add(key)
                previous = self._smoothed_points.get(key)
                if previous is None:
                    sx, sy, sz = float(x), float(y), float(z)
                else:
                    sx = alpha * x + (1.0 - alpha) * previous[0]
                    sy = alpha * y + (1.0 - alpha) * previous[1]
                    sz = alpha * z + (1.0 - alpha) * previous[2]
                self._smoothed_points[key] = (sx, sy, sz)
                smoothed_landmarks.append((int(sx), int(sy), float(sz)))
            smoothed_hands.append(
                HandLandmarks(
                    hand_id=hand.hand_id,
                    label=hand.label,
                    landmarks=smoothed_landmarks,
                    normalized_landmarks=hand.normalized_landmarks,
                    tracking_source=hand.tracking_source,
                    missed_frames=hand.missed_frames,
                    unstable_landmark_ids=hand.unstable_landmark_ids,
                )
            )
        for key in list(self._smoothed_points):
            if key not in next_keys:
                del self._smoothed_points[key]
        return smoothed_hands

    def _assign_stable_hand_ids(self, hands: List[HandLandmarks]) -> List[HandLandmarks]:
        if not hands:
            return hands

        previous = dict(self._tracked_hand_centers)
        assignments: Dict[int, int] = {}
        used_detected: set[int] = set()
        used_previous: set[int] = set()
        next_centers: Dict[int, Tuple[float, float, str]] = {}

        detected_info = [
            (self._hand_center(hand.landmarks), self._hand_scale(hand.landmarks), hand.label)
            for hand in hands
        ]
        candidates: List[Tuple[float, float, int, int]] = []
        for detected_idx, (center, scale, detected_label) in enumerate(detected_info):
            max_distance = max(float(config.STABLE_HAND_ID_MAX_DISTANCE_PX), scale * 0.75)
            for stable_id, (px, py, previous_label) in previous.items():
                distance = math.hypot(center[0] - px, center[1] - py)
                if distance > max_distance:
                    continue
                label_matches = (
                    previous_label == detected_label
                    or "Unknown" in {previous_label, detected_label}
                )
                label_penalty = 0.0 if label_matches else min(35.0, max_distance * 0.18)
                candidates.append((distance + label_penalty, distance, detected_idx, stable_id))

        for _, _, detected_idx, stable_id in sorted(candidates):
            if detected_idx in used_detected or stable_id in used_previous:
                continue
            assignments[detected_idx] = stable_id
            used_detected.add(detected_idx)
            used_previous.add(stable_id)

        assigned: List[HandLandmarks] = []
        for detected_idx, detected in enumerate(hands):
            center, _, _ = detected_info[detected_idx]
            best_id = assignments.get(detected_idx)
            if best_id is None:
                if self._is_duplicate_detection(detected_idx, detected_info, set(assignments)):
                    continue
                best_id = self._next_stable_hand_id
                self._next_stable_hand_id += 1
                self._new_hand_guard_frames[best_id] = max(
                    self._new_hand_guard_frames.get(best_id, 0),
                    int(config.TRACKING_NEW_HAND_HIT_BLOCK_FRAMES),
                )
            next_centers[best_id] = (center[0], center[1], detected.label)
            assigned.append(
                HandLandmarks(
                    hand_id=best_id,
                    label=detected.label,
                    landmarks=detected.landmarks,
                    normalized_landmarks=detected.normalized_landmarks,
                    tracking_source=detected.tracking_source,
                    missed_frames=detected.missed_frames,
                    unstable_landmark_ids=detected.unstable_landmark_ids,
                )
            )

        self._tracked_hand_centers = next_centers
        return assigned

    def _is_duplicate_detection(
        self,
        detected_idx: int,
        detected_info: List[Tuple[Tuple[float, float], float, str]],
        assigned_detected: set[int],
    ) -> bool:
        if not assigned_detected:
            return False
        center, scale, _ = detected_info[detected_idx]
        duplicate_distance = max(48.0, min(scale * 0.35, float(config.STABLE_HAND_ID_MAX_DISTANCE_PX) * 0.65))
        for assigned_idx in assigned_detected:
            assigned_center, assigned_scale, _ = detected_info[assigned_idx]
            threshold = max(
                duplicate_distance,
                max(48.0, min(assigned_scale * 0.35, float(config.STABLE_HAND_ID_MAX_DISTANCE_PX) * 0.65)),
            )
            if math.hypot(center[0] - assigned_center[0], center[1] - assigned_center[1]) <= threshold:
                return True
        return False

    def _stabilize_with_optical_flow(self, frame_bgr, hands: List[HandLandmarks]) -> List[HandLandmarks]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if not config.FINGERTIP_OPTICAL_FLOW_STABILIZATION:
            self._last_gray = gray
            self._flow_points = self._current_flow_points(hands)
            return hands

        if self._last_gray is None or not self._flow_points:
            self._last_gray = gray
            self._flow_points = self._current_flow_points(hands)
            return hands

        optical_predictions = self._optical_flow_predictions(gray, hands)
        next_flow_points: Dict[Tuple[int, int], Tuple[float, float]] = {}
        stabilized_hands: List[HandLandmarks] = []

        for hand in hands:
            scale = self._hand_scale(hand.landmarks)
            outlier_distance = config.OPTICAL_FLOW_OUTLIER_DISTANCE_PX + scale * 0.035
            max_step = config.OPTICAL_FLOW_MAX_POINT_STEP_PX + scale * 0.04
            stabilized_landmarks = []
            unstable_ids = set(hand.unstable_landmark_ids)
            for idx, (x, y, z) in enumerate(hand.landmarks):
                key = (hand.hand_id, idx)
                current = (float(x), float(y))
                previous = self._flow_points.get(key)
                prediction = optical_predictions.get(key)

                if idx in config.OPTICAL_FLOW_LANDMARK_IDS and prediction is not None:
                    distance_to_model = math.hypot(current[0] - prediction[0], current[1] - prediction[1])
                    if distance_to_model > outlier_distance:
                        unstable_ids.add(idx)
                    raw_weight = (
                        config.OPTICAL_FLOW_OUTLIER_RAW_WEIGHT
                        if distance_to_model > outlier_distance
                        else config.OPTICAL_FLOW_RAW_WEIGHT
                    )
                    sx = raw_weight * current[0] + (1.0 - raw_weight) * prediction[0]
                    sy = raw_weight * current[1] + (1.0 - raw_weight) * prediction[1]
                elif idx in config.OPTICAL_FLOW_LANDMARK_IDS and previous is not None:
                    sx, sy = self._limit_point_step(previous, current, max_step)
                else:
                    sx, sy = current

                if previous is not None and idx in config.OPTICAL_FLOW_LANDMARK_IDS:
                    if math.hypot(sx - previous[0], sy - previous[1]) > max_step:
                        unstable_ids.add(idx)
                    sx, sy = self._limit_point_step(previous, (sx, sy), max_step)
                next_flow_points[key] = (sx, sy)
                stabilized_landmarks.append((int(round(sx)), int(round(sy)), z))

            stabilized_hands.append(
                HandLandmarks(
                    hand_id=hand.hand_id,
                    label=hand.label,
                    landmarks=stabilized_landmarks,
                    normalized_landmarks=hand.normalized_landmarks,
                    tracking_source=hand.tracking_source,
                    missed_frames=hand.missed_frames,
                    unstable_landmark_ids=tuple(sorted(unstable_ids)),
                )
            )

        self._last_gray = gray
        self._flow_points = next_flow_points
        return stabilized_hands

    def _apply_hit_guards(self, hands: List[HandLandmarks]) -> List[HandLandmarks]:
        active_ids = {hand.hand_id for hand in hands}
        guarded: List[HandLandmarks] = []
        global_guard_active = self._reacquire_guard_frames > 0
        for hand in hands:
            hand_guard_active = self._new_hand_guard_frames.get(hand.hand_id, 0) > 0
            if global_guard_active or hand_guard_active:
                guarded.extend(self._mark_unstable([hand], config.TRIGGER_FINGER_IDS))
            else:
                guarded.append(hand)

        if self._reacquire_guard_frames > 0:
            self._reacquire_guard_frames -= 1

        next_new_hand_guards: Dict[int, int] = {}
        for hand_id in active_ids:
            remaining = self._new_hand_guard_frames.get(hand_id, 0)
            if remaining > 1:
                next_new_hand_guards[hand_id] = remaining - 1
        self._new_hand_guard_frames = next_new_hand_guards
        return guarded

    def _bridge_missing_hands(self, frame_bgr) -> List[HandLandmarks]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if (
            not config.TRACKING_BRIDGE_MISSED_FRAMES
            or not config.FINGERTIP_OPTICAL_FLOW_STABILIZATION
            or self._last_gray is None
            or not self._last_good_hands
            or not self._flow_points
        ):
            self._last_gray = gray
            return []

        self._missed_frame_count += 1
        if self._missed_frame_count > config.TRACKING_BRIDGE_MAX_FRAMES:
            self._clear_after_miss(frame_bgr)
            return []

        predictions = self._optical_flow_predictions_for_keys(
            gray,
            set(self._flow_points),
            max_error=config.TRACKING_BRIDGE_MAX_FLOW_ERROR,
        )
        tracked_tip_count = sum(
            1 for hand in self._last_good_hands for finger_id in config.TRIGGER_FINGER_IDS
            if (hand.hand_id, finger_id) in predictions
        )
        if tracked_tip_count < max(1, min(config.TRACKING_BRIDGE_MIN_POINTS, len(config.TRIGGER_FINGER_IDS))):
            self._clear_after_miss(frame_bgr)
            return []

        height, width = gray.shape[:2]
        next_flow_points: Dict[Tuple[int, int], Tuple[float, float]] = {}
        bridged_hands: List[HandLandmarks] = []

        for hand in self._last_good_hands:
            deltas: List[Tuple[float, float]] = []
            for idx, (x, y, _) in enumerate(hand.landmarks):
                key = (hand.hand_id, idx)
                previous = self._flow_points.get(key)
                prediction = predictions.get(key)
                if previous is None or prediction is None:
                    continue
                px, py = self._limit_point_step(previous, prediction, config.TRACKING_BRIDGE_MAX_POINT_STEP_PX)
                deltas.append((px - previous[0], py - previous[1]))

            if deltas:
                dx = sum(delta[0] for delta in deltas) / len(deltas)
                dy = sum(delta[1] for delta in deltas) / len(deltas)
            else:
                dx = dy = 0.0

            landmarks: List[Tuple[int, int, float]] = []
            for idx, (x, y, z) in enumerate(hand.landmarks):
                key = (hand.hand_id, idx)
                previous = self._flow_points.get(key)
                prediction = predictions.get(key)
                if previous is not None and prediction is not None:
                    sx, sy = self._limit_point_step(previous, prediction, config.TRACKING_BRIDGE_MAX_POINT_STEP_PX)
                else:
                    sx, sy = float(x) + dx, float(y) + dy
                sx = float(np.clip(sx, 0, width - 1))
                sy = float(np.clip(sy, 0, height - 1))
                next_flow_points[key] = (sx, sy)
                landmarks.append((int(round(sx)), int(round(sy)), z))

            bridged_hands.append(
                HandLandmarks(
                    hand_id=hand.hand_id,
                    label=hand.label,
                    landmarks=landmarks,
                    normalized_landmarks=hand.normalized_landmarks,
                    tracking_source="optical_flow",
                    missed_frames=self._missed_frame_count,
                    unstable_landmark_ids=tuple(config.TRIGGER_FINGER_IDS),
                )
            )

        self._last_gray = gray
        self._flow_points = next_flow_points
        self._tracked_hand_centers = {
            hand.hand_id: (*self._hand_center(hand.landmarks), hand.label)
            for hand in bridged_hands
        }
        self._last_good_hands = self._copy_hands(bridged_hands)
        return bridged_hands

    def _clear_after_miss(self, frame_bgr) -> None:
        self._smoothed_points.clear()
        self._tracked_hand_centers.clear()
        self._flow_points.clear()
        self._last_good_hands = []
        self._missed_frame_count = 0
        self._reacquire_guard_frames = 0
        self._new_hand_guard_frames.clear()
        self._last_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def _optical_flow_predictions(
        self,
        gray,
        hands: List[HandLandmarks],
    ) -> Dict[Tuple[int, int], Tuple[float, float]]:
        current_keys = {
            (hand.hand_id, idx)
            for hand in hands
            for idx, _ in enumerate(hand.landmarks)
            if idx in config.OPTICAL_FLOW_LANDMARK_IDS
        }
        return self._optical_flow_predictions_for_keys(gray, current_keys, config.OPTICAL_FLOW_MAX_ERROR)

    def _optical_flow_predictions_for_keys(
        self,
        gray,
        candidate_keys: set[Tuple[int, int]],
        max_error: float,
    ) -> Dict[Tuple[int, int], Tuple[float, float]]:
        keys = []
        points = []
        for key, point in self._flow_points.items():
            if key in candidate_keys:
                keys.append(key)
                points.append(point)
        if not points:
            return {}

        previous_points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            self._last_gray,
            gray,
            previous_points,
            None,
            winSize=(config.OPTICAL_FLOW_WINDOW_SIZE, config.OPTICAL_FLOW_WINDOW_SIZE),
            maxLevel=config.OPTICAL_FLOW_MAX_LEVEL,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 16, 0.03),
        )
        if next_points is None or status is None:
            return {}

        backtrack_errors = np.zeros(len(keys), dtype=np.float32)
        if config.OPTICAL_FLOW_FORWARD_BACKWARD_CHECK:
            back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray,
                self._last_gray,
                next_points,
                None,
                winSize=(config.OPTICAL_FLOW_WINDOW_SIZE, config.OPTICAL_FLOW_WINDOW_SIZE),
                maxLevel=config.OPTICAL_FLOW_MAX_LEVEL,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 16, 0.03),
            )
            if back_points is None or back_status is None:
                return {}
            back_status_flat = back_status.reshape(-1)
            back_deltas = back_points.reshape(-1, 2) - previous_points.reshape(-1, 2)
            backtrack_errors = np.linalg.norm(back_deltas, axis=1).astype(np.float32)
        else:
            back_status_flat = np.ones(len(keys), dtype=np.uint8)

        height, width = gray.shape[:2]
        predictions: Dict[Tuple[int, int], Tuple[float, float]] = {}
        flat_status = status.reshape(-1)
        flat_errors = errors.reshape(-1) if errors is not None else np.zeros(len(keys), dtype=np.float32)
        for idx, key in enumerate(keys):
            if not flat_status[idx] or not back_status_flat[idx]:
                continue
            error = float(flat_errors[idx])
            if error > max_error:
                continue
            if backtrack_errors[idx] > config.OPTICAL_FLOW_MAX_BACKTRACK_ERROR_PX:
                continue
            x, y = next_points[idx, 0]
            if 0 <= x < width and 0 <= y < height:
                predictions[key] = (float(x), float(y))
        return predictions

    def _current_flow_points(self, hands: List[HandLandmarks]) -> Dict[Tuple[int, int], Tuple[float, float]]:
        points: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for hand in hands:
            for idx, (x, y, _) in enumerate(hand.landmarks):
                if idx in config.OPTICAL_FLOW_LANDMARK_IDS:
                    points[(hand.hand_id, idx)] = (float(x), float(y))
        return points

    def _limit_point_step(
        self,
        previous: Tuple[float, float],
        current: Tuple[float, float],
        max_step: float,
    ) -> Tuple[float, float]:
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        distance = math.hypot(dx, dy)
        if distance <= max_step or distance <= 1e-6:
            return current
        ratio = max_step / distance
        return (previous[0] + dx * ratio, previous[1] + dy * ratio)

    def _hand_center(self, landmarks: List[Tuple[int, int, float]]) -> Tuple[float, float]:
        if not landmarks:
            return (0.0, 0.0)
        xs = [point[0] for point in landmarks]
        ys = [point[1] for point in landmarks]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _hand_scale(self, landmarks: List[Tuple[int, int, float]]) -> float:
        if not landmarks:
            return 1.0
        xs = [point[0] for point in landmarks]
        ys = [point[1] for point in landmarks]
        return max(1.0, math.hypot(max(xs) - min(xs), max(ys) - min(ys)))

    def _mark_unstable(self, hands: List[HandLandmarks], landmark_ids: Tuple[int, ...]) -> List[HandLandmarks]:
        marked: List[HandLandmarks] = []
        mark_ids = set(landmark_ids)
        for hand in hands:
            unstable = tuple(sorted(set(hand.unstable_landmark_ids) | mark_ids))
            marked.append(
                HandLandmarks(
                    hand_id=hand.hand_id,
                    label=hand.label,
                    landmarks=hand.landmarks,
                    normalized_landmarks=hand.normalized_landmarks,
                    tracking_source=hand.tracking_source,
                    missed_frames=hand.missed_frames,
                    unstable_landmark_ids=unstable,
                )
            )
        return marked

    def _copy_hands(self, hands: List[HandLandmarks]) -> List[HandLandmarks]:
        return [
            HandLandmarks(
                hand_id=hand.hand_id,
                label=hand.label,
                landmarks=list(hand.landmarks),
                normalized_landmarks=list(hand.normalized_landmarks),
                tracking_source=hand.tracking_source,
                missed_frames=hand.missed_frames,
                unstable_landmark_ids=tuple(hand.unstable_landmark_ids),
            )
            for hand in hands
        ]

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
        if self._landmarker is not None:
            self._landmarker.close()
