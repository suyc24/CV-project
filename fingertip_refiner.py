from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


Landmark = Tuple[int, int, float]

FINGER_DISTAL_IDS = {
    4: 3,
    8: 7,
    12: 11,
    16: 15,
    20: 19,
}


@dataclass(frozen=True)
class RefinedPoint:
    x: float
    y: float
    score: float
    accepted: bool


class FingertipRefiner:
    """Conservative local fingertip refinement on the original camera frame.

    MediaPipe gives a strong coarse hand pose, but its fingertip landmarks can
    jump by a few pixels frame-to-frame. This refiner only searches a small
    patch around each MediaPipe fingertip and moves the point toward a strong
    local edge in the distal finger direction. It is intentionally conservative:
    low-confidence patches return the original landmark unchanged.
    """

    def __init__(
        self,
        finger_ids: Iterable[int],
        radius_px: int = 22,
        max_shift_px: int = 14,
        min_edge_score: float = 28.0,
        blend_alpha: float = 0.45,
        forward_bias: float = 0.65,
    ) -> None:
        self.finger_ids = tuple(finger_ids)
        self.radius_px = max(4, int(radius_px))
        self.max_shift_px = max(2, int(max_shift_px))
        self.min_edge_score = float(min_edge_score)
        self.blend_alpha = min(1.0, max(0.0, float(blend_alpha)))
        self.forward_bias = max(0.0, float(forward_bias))

    def refine_landmarks(self, frame_bgr, landmarks: Sequence[Landmark]) -> List[Landmark]:
        if frame_bgr is None or len(landmarks) < 21:
            return list(landmarks)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        refined = list(landmarks)
        for finger_id in self.finger_ids:
            if finger_id >= len(refined):
                continue
            point = self.refine_point(gray, refined, finger_id)
            if not point.accepted:
                continue
            _, _, z = refined[finger_id]
            refined[finger_id] = (int(round(point.x)), int(round(point.y)), z)
        return refined

    def refine_point(self, gray, landmarks: Sequence[Landmark], finger_id: int) -> RefinedPoint:
        x, y, _ = landmarks[finger_id]
        direction = self._finger_direction(landmarks, finger_id)
        if direction is None:
            return RefinedPoint(float(x), float(y), 0.0, False)

        height, width = gray.shape[:2]
        x1, y1, x2, y2 = self._patch_bounds(int(x), int(y), width, height)
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0 or patch.shape[0] < 5 or patch.shape[1] < 5:
            return RefinedPoint(float(x), float(y), 0.0, False)

        candidate = self._best_edge_candidate(patch, (float(x - x1), float(y - y1)), direction)
        if candidate is None:
            return RefinedPoint(float(x), float(y), 0.0, False)

        candidate_x = x1 + candidate[0]
        candidate_y = y1 + candidate[1]
        score = candidate[2]
        if score < self.min_edge_score:
            return RefinedPoint(float(x), float(y), score, False)

        dx = candidate_x - float(x)
        dy = candidate_y - float(y)
        distance = math.hypot(dx, dy)
        if distance > self.max_shift_px:
            scale = self.max_shift_px / max(distance, 1e-6)
            candidate_x = float(x) + dx * scale
            candidate_y = float(y) + dy * scale

        refined_x = (1.0 - self.blend_alpha) * float(x) + self.blend_alpha * candidate_x
        refined_y = (1.0 - self.blend_alpha) * float(y) + self.blend_alpha * candidate_y
        return RefinedPoint(refined_x, refined_y, score, True)

    def _patch_bounds(self, x: int, y: int, width: int, height: int) -> tuple[int, int, int, int]:
        radius = self.radius_px
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(width, x + radius + 1)
        y2 = min(height, y + radius + 1)
        return x1, y1, x2, y2

    def _finger_direction(self, landmarks: Sequence[Landmark], finger_id: int) -> tuple[float, float] | None:
        distal_id = FINGER_DISTAL_IDS.get(finger_id)
        if distal_id is None or distal_id >= len(landmarks):
            return None
        tip_x, tip_y, _ = landmarks[finger_id]
        distal_x, distal_y, _ = landmarks[distal_id]
        dx = float(tip_x - distal_x)
        dy = float(tip_y - distal_y)
        length = math.hypot(dx, dy)
        if length < 3.0:
            return None
        return (dx / length, dy / length)

    def _best_edge_candidate(
        self,
        patch,
        center: tuple[float, float],
        direction: tuple[float, float],
    ) -> tuple[float, float, float] | None:
        blurred = cv2.GaussianBlur(patch, (5, 5), 0)
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(grad_x, grad_y)
        if not np.isfinite(gradient).any():
            return None

        h, w = gradient.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        rel_x = xx - float(center[0])
        rel_y = yy - float(center[1])
        distance = np.sqrt(rel_x * rel_x + rel_y * rel_y)
        within_shift = distance <= float(self.max_shift_px)

        dir_x, dir_y = direction
        parallel = rel_x * dir_x + rel_y * dir_y
        perpendicular = np.abs(rel_x * dir_y - rel_y * dir_x)
        not_far_behind = parallel >= -0.45 * float(self.max_shift_px)

        front_bonus = 1.0 + self.forward_bias * np.clip(
            parallel / max(float(self.max_shift_px), 1.0),
            0.0,
            1.0,
        )
        center_weight = 1.0 / (1.0 + distance / max(float(self.radius_px) * 0.55, 1.0))
        axial_weight = np.exp(-perpendicular / max(float(self.radius_px) * 0.45, 1.0))
        score = gradient * front_bonus * center_weight * axial_weight
        score = np.where(within_shift & not_far_behind, score, 0.0)

        flat_idx = int(np.argmax(score))
        best_score = float(score.reshape(-1)[flat_idx])
        if best_score <= 0.0:
            return None
        best_y, best_x = np.unravel_index(flat_idx, score.shape)
        return (float(best_x), float(best_y), best_score)
