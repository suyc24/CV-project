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


class HandTracker:
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.55,
        min_tracking_confidence: float = 0.55,
        input_max_width: int = config.TRACKING_MAX_WIDTH,
        smooth_landmarks: bool = True,
        smoothing_alpha: float = config.LANDMARK_SMOOTHING_ALPHA,
    ) -> None:
        self._backend = ""
        self._hands = None
        self._landmarker = None
        self._mp = None
        self._last_timestamp_ms = 0
        self._input_max_width = input_max_width
        self._smooth_landmarks = smooth_landmarks
        self._smoothing_alpha = smoothing_alpha
        self._smoothed_points: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
        self._tracked_hand_centers: Dict[int, Tuple[float, float, str]] = {}
        self._next_stable_hand_id = 0
        self._last_gray = None
        self._flow_points: Dict[Tuple[int, int], Tuple[float, float]] = {}

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
        prepared_frame, offset, scale = self._prepare_frame(frame_bgr, roi)
        if self._backend == "tasks":
            hands = self._process_tasks(prepared_frame, offset, scale)
        else:
            hands = self._process_legacy(prepared_frame, offset, scale)
        if not hands:
            self._smoothed_points.clear()
            self._tracked_hand_centers.clear()
            self._flow_points.clear()
            self._last_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            return hands
        hands = self._assign_stable_hand_ids(hands)
        hands = self._smooth(hands)
        return self._stabilize_with_optical_flow(frame_bgr, hands)

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
            hands.append(HandLandmarks(hand_id=hand_id, label=label, landmarks=pixels, normalized_landmarks=normalized))
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
        used_previous: set[int] = set()
        assigned: List[HandLandmarks] = []
        next_centers: Dict[int, Tuple[float, float, str]] = {}

        for detected in hands:
            center = self._hand_center(detected.landmarks)
            scale = self._hand_scale(detected.landmarks)
            best_id: Optional[int] = None
            best_distance = float("inf")
            max_distance = max(float(config.STABLE_HAND_ID_MAX_DISTANCE_PX), scale * 0.75)
            for stable_id, (px, py, label) in previous.items():
                if stable_id in used_previous:
                    continue
                label_matches = label == detected.label or "Unknown" in {label, detected.label}
                if not label_matches:
                    continue
                distance = math.hypot(center[0] - px, center[1] - py)
                if distance < best_distance and distance <= max_distance:
                    best_distance = distance
                    best_id = stable_id

            if best_id is None:
                best_id = self._next_stable_hand_id
                self._next_stable_hand_id += 1
            used_previous.add(best_id)
            next_centers[best_id] = (center[0], center[1], detected.label)
            assigned.append(
                HandLandmarks(
                    hand_id=best_id,
                    label=detected.label,
                    landmarks=detected.landmarks,
                    normalized_landmarks=detected.normalized_landmarks,
                )
            )

        self._tracked_hand_centers = next_centers
        return assigned

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
            for idx, (x, y, z) in enumerate(hand.landmarks):
                key = (hand.hand_id, idx)
                current = (float(x), float(y))
                previous = self._flow_points.get(key)
                prediction = optical_predictions.get(key)

                if idx in config.OPTICAL_FLOW_LANDMARK_IDS and prediction is not None:
                    distance_to_model = math.hypot(current[0] - prediction[0], current[1] - prediction[1])
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
                    sx, sy = self._limit_point_step(previous, (sx, sy), max_step)
                next_flow_points[key] = (sx, sy)
                stabilized_landmarks.append((int(round(sx)), int(round(sy)), z))

            stabilized_hands.append(
                HandLandmarks(
                    hand_id=hand.hand_id,
                    label=hand.label,
                    landmarks=stabilized_landmarks,
                    normalized_landmarks=hand.normalized_landmarks,
                )
            )

        self._last_gray = gray
        self._flow_points = next_flow_points
        return stabilized_hands

    def _optical_flow_predictions(
        self,
        gray,
        hands: List[HandLandmarks],
    ) -> Dict[Tuple[int, int], Tuple[float, float]]:
        keys = []
        points = []
        current_keys = {
            (hand.hand_id, idx)
            for hand in hands
            for idx, _ in enumerate(hand.landmarks)
            if idx in config.OPTICAL_FLOW_LANDMARK_IDS
        }
        for key, point in self._flow_points.items():
            if key in current_keys:
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

        height, width = gray.shape[:2]
        predictions: Dict[Tuple[int, int], Tuple[float, float]] = {}
        flat_status = status.reshape(-1)
        flat_errors = errors.reshape(-1) if errors is not None else np.zeros(len(keys), dtype=np.float32)
        for idx, key in enumerate(keys):
            if not flat_status[idx]:
                continue
            error = float(flat_errors[idx])
            if error > config.OPTICAL_FLOW_MAX_ERROR:
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

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
        if self._landmarker is not None:
            self._landmarker.close()
