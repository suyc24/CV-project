from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import cv2

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
            return hands
        return self._smooth(hands)

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

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
        if self._landmarker is not None:
            self._landmarker.close()
