from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

import config
from hand_tracker import HandLandmarks
from instrument import Zone
from rgbd_camera import RGBDFrame


@dataclass
class DepthObservation:
    hand_id: int
    finger_id: int
    x: int
    y: int
    finger_depth_m: Optional[float]
    desk_depth_m: Optional[float]
    height_above_desk_m: Optional[float]
    contact: Optional[bool]
    confidence: float
    reason: str
    finger_sample_count: int = 0
    desk_sample_count: int = 0
    sample_radius_px: int = 0
    desk_depth_source: str = ""


class DepthContactEstimator:
    """Estimate fingertip contact from Record3D depth.

    The estimator stores a desk-depth baseline captured with the user's hands
    away from the virtual keyboard. Runtime contact is then a local comparison
    between fingertip depth and the calibrated desk depth at the same image
    coordinate. This is intentionally conservative: when depth is missing or
    low-confidence, it returns "unknown" instead of pretending to know.
    """

    def __init__(
        self,
        mode: str = "assist",
        contact_threshold_m: float = config.DEPTH_CONTACT_THRESHOLD_M,
        release_threshold_m: float = config.DEPTH_RELEASE_THRESHOLD_M,
        sample_radius_px: int = config.DEPTH_SAMPLE_RADIUS_PX,
        baseline_frames: int = config.DEPTH_BASELINE_FRAMES,
        min_confidence: float = config.DEPTH_MIN_CONFIDENCE,
    ) -> None:
        self.mode = mode
        self.contact_threshold_m = contact_threshold_m
        self.release_threshold_m = release_threshold_m
        self.sample_radius_px = sample_radius_px
        self.baseline_frames = max(1, baseline_frames)
        self.min_confidence = min_confidence
        self._baseline_accumulator: List[np.ndarray] = []
        self._baseline_depth_m: Optional[np.ndarray] = None
        self._baseline_shape: Optional[Tuple[int, int]] = None
        self._baseline_plane: Optional[Tuple[float, float, float]] = None
        self._last_observations: Dict[Tuple[int, int], DepthObservation] = {}
        self._last_status = "depth: unavailable"

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def calibrated(self) -> bool:
        return self._baseline_depth_m is not None

    def reset(self) -> None:
        self._baseline_accumulator.clear()
        self._baseline_depth_m = None
        self._baseline_shape = None
        self._baseline_plane = None
        self._last_observations = {}
        self._last_status = "Depth: uncalibrated (press d with hands away)"

    def status_text(self) -> str:
        if self.mode == "off":
            return "Depth: off"
        if not self.calibrated:
            return "Depth: uncalibrated (press d with hands away)"
        return self._last_status

    def calibrate(self, frame: RGBDFrame, zones: Iterable[Zone]) -> bool:
        depth = self._aligned_depth(frame)
        if depth is None:
            self._last_status = "Depth: no frame for calibration"
            return False
        mask = self._zones_mask(depth.shape, zones)
        if mask is None or np.count_nonzero(mask) < 256:
            mask = np.isfinite(depth) & (depth > 0)
        else:
            mask = mask & np.isfinite(depth) & (depth > 0)
        if np.count_nonzero(mask) < 256:
            self._last_status = "Depth: not enough valid desk samples"
            return False

        filtered = depth.copy()
        filtered[~mask] = np.nan
        self._baseline_accumulator.append(filtered)
        if len(self._baseline_accumulator) > self.baseline_frames:
            self._baseline_accumulator.pop(0)

        stack = np.stack(self._baseline_accumulator, axis=0)
        self._baseline_depth_m = np.nanmedian(stack, axis=0).astype(np.float32)
        self._baseline_shape = self._baseline_depth_m.shape
        self._baseline_plane = self._fit_depth_plane(self._baseline_depth_m)
        valid_ratio = float(np.mean(np.isfinite(self._baseline_depth_m) & (self._baseline_depth_m > 0)))
        plane_status = " plane=ok" if self._baseline_plane is not None else " plane=none"
        self._last_status = f"Depth: calibrated valid={valid_ratio * 100:.0f}%{plane_status}"
        return True

    def update(
        self,
        frame: Optional[RGBDFrame],
        hands: Iterable[HandLandmarks],
        zones: Iterable[Zone],
    ) -> Dict[Tuple[int, int], DepthObservation]:
        self._last_observations = {}
        if not self.enabled:
            return self._last_observations
        if frame is None or frame.depth is None:
            self._last_status = "Depth: no RGB-D frame"
            return self._last_observations

        if not self.calibrated and config.DEPTH_AUTO_BASELINE_WHEN_UNCALIBRATED:
            self.calibrate(frame, zones)

        depth = self._aligned_depth(frame)
        confidence = self._aligned_confidence(frame, depth.shape if depth is not None else None)
        if depth is None:
            self._last_status = "Depth: no depth"
            return self._last_observations

        for hand in hands:
            for finger_id in config.TRIGGER_FINGER_IDS:
                if finger_id >= len(hand.landmarks):
                    continue
                x, y, _ = hand.landmarks[finger_id]
                self._last_observations[(hand.hand_id, finger_id)] = self._observe_point(
                    hand.hand_id,
                    finger_id,
                    x,
                    y,
                    depth,
                    confidence,
                )

        known = [obs for obs in self._last_observations.values() if obs.contact is not None]
        unknown = len(self._last_observations) - len(known)
        contacts = sum(1 for obs in known if obs.contact)
        self._last_status = (
            f"Depth: mode={self.mode} contact={contacts}/{len(known)} unknown={unknown}"
            if self.calibrated
            else "Depth: uncalibrated (press d)"
        )
        return dict(self._last_observations)

    def observation_for(self, hand_id: int, finger_id: int) -> Optional[DepthObservation]:
        return self._last_observations.get((hand_id, finger_id))

    def estimate_piano_quad(
        self,
        frame_shape: Tuple[int, int, int],
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
        if self._baseline_depth_m is None:
            return None
        height, width = frame_shape[:2]
        depth = self._baseline_depth_m
        if depth.shape[:2] != (height, width):
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)

        roi_y1 = int(height * 0.42)
        roi_y2 = int(height * 0.98)
        roi_x1 = int(width * 0.05)
        roi_x2 = int(width * 0.95)
        roi = depth[roi_y1:roi_y2, roi_x1:roi_x2]
        valid = np.isfinite(roi) & (roi > 0)
        if np.count_nonzero(valid) >= 512:
            ys, xs = np.where(valid)
            step = max(1, len(xs) // 5000)
            xs = xs[::step].astype(np.float32) + roi_x1
            ys = ys[::step].astype(np.float32) + roi_y1
            zs = roi[valid][::step].astype(np.float32)
            angle = self._desk_roll_angle(xs, ys, zs)
        else:
            angle = 0.0

        max_roll = np.deg2rad(config.DEPTH_KEYBED_MAX_ROLL_DEGREES)
        angle = float(np.clip(angle, -max_roll, max_roll))
        hdir = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        up = np.array([hdir[1], -hdir[0]], dtype=np.float32)
        if up[1] > 0:
            up = -up

        bottom_center = np.array(
            [width * 0.5, height * (1.0 - config.DEPTH_KEYBED_BOTTOM_MARGIN_RATIO)],
            dtype=np.float32,
        )
        bottom_width = width * config.DEPTH_KEYBED_WIDTH_RATIO
        keybed_height = bottom_width * config.DEPTH_KEYBED_LENGTH_TO_WIDTH_RATIO
        keybed_height = float(
            np.clip(
                keybed_height,
                height * config.DEPTH_KEYBED_MIN_HEIGHT_RATIO,
                height * config.DEPTH_KEYBED_MAX_HEIGHT_RATIO,
            )
        )
        top_width = bottom_width * config.DEPTH_KEYBED_TOP_SCALE
        top_center = bottom_center + up * keybed_height

        bottom_left = bottom_center - hdir * (bottom_width / 2.0)
        bottom_right = bottom_center + hdir * (bottom_width / 2.0)
        top_left = top_center - hdir * (top_width / 2.0)
        top_right = top_center + hdir * (top_width / 2.0)
        quad = (top_left, top_right, bottom_right, bottom_left)
        return tuple(self._clamp_point(point, width, height) for point in quad)

    def _desk_roll_angle(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> float:
        if len(xs) < 3:
            return 0.0
        design = np.column_stack([xs, ys, np.ones_like(xs)])
        try:
            a, b, _ = np.linalg.lstsq(design, zs, rcond=None)[0]
        except np.linalg.LinAlgError:
            return 0.0
        if abs(a) + abs(b) < 1e-6:
            return 0.0
        direction = np.array([b, -a], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return 0.0
        direction /= norm
        if direction[0] < 0:
            direction = -direction
        return float(np.arctan2(direction[1], direction[0]))

    def _fit_depth_plane(self, depth: np.ndarray) -> Optional[Tuple[float, float, float]]:
        valid = np.isfinite(depth) & (depth > 0)
        if np.count_nonzero(valid) < config.DEPTH_PLANE_MIN_SAMPLES:
            return None
        ys, xs = np.where(valid)
        values = depth[valid].astype(np.float32)
        step = max(1, len(values) // 12000)
        xs = xs[::step].astype(np.float32)
        ys = ys[::step].astype(np.float32)
        values = values[::step]
        design = np.column_stack([xs, ys, np.ones_like(xs)])
        try:
            a, b, c = np.linalg.lstsq(design, values, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        return (float(a), float(b), float(c))

    def _plane_depth(self, x: int, y: int, shape: Tuple[int, int]) -> Optional[float]:
        if self._baseline_plane is None:
            return None
        height, width = shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return None
        a, b, c = self._baseline_plane
        value = a * float(x) + b * float(y) + c
        if not np.isfinite(value) or value <= 0:
            return None
        return float(value)

    def _clamp_point(self, point: np.ndarray, width: int, height: int) -> Tuple[int, int]:
        x = int(round(float(np.clip(point[0], 0, width - 1))))
        y = int(round(float(np.clip(point[1], 0, height - 1))))
        return (x, y)

    def _observe_point(
        self,
        hand_id: int,
        finger_id: int,
        x: int,
        y: int,
        depth: np.ndarray,
        confidence: Optional[np.ndarray],
    ) -> DepthObservation:
        finger_depth, finger_count, finger_radius = self._local_depth_stats(
            depth,
            x,
            y,
            percentile=config.DEPTH_FINGER_DEPTH_PERCENTILE,
        )
        conf_value = self._local_confidence(confidence, x, y)
        if finger_depth is None:
            return self._unknown(
                hand_id,
                finger_id,
                x,
                y,
                None,
                None,
                conf_value,
                "no_finger_depth",
                finger_count=finger_count,
                sample_radius=finger_radius,
            )
        if not self.calibrated or self._baseline_depth_m is None:
            return self._unknown(
                hand_id,
                finger_id,
                x,
                y,
                finger_depth,
                None,
                conf_value,
                "uncalibrated",
                finger_count=finger_count,
                sample_radius=finger_radius,
            )

        baseline = self._baseline_depth_m
        if baseline.shape != depth.shape:
            baseline = cv2.resize(baseline, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        desk_depth, desk_count, desk_radius = self._local_depth_stats(
            baseline,
            x,
            y,
            percentile=config.DEPTH_DESK_DEPTH_PERCENTILE,
        )
        desk_source = "local"
        if desk_depth is None:
            desk_depth = self._plane_depth(x, y, baseline.shape)
            desk_source = "plane" if desk_depth is not None else "missing"
        if desk_depth is None:
            return self._unknown(
                hand_id,
                finger_id,
                x,
                y,
                finger_depth,
                None,
                conf_value,
                "no_desk_depth",
                finger_count=finger_count,
                desk_count=desk_count,
                sample_radius=max(finger_radius, desk_radius),
                desk_source=desk_source,
            )
        if conf_value < self.min_confidence:
            return self._unknown(
                hand_id,
                finger_id,
                x,
                y,
                finger_depth,
                desk_depth,
                conf_value,
                "low_confidence",
                finger_count=finger_count,
                desk_count=desk_count,
                sample_radius=max(finger_radius, desk_radius),
                desk_source=desk_source,
            )

        height = max(0.0, float(desk_depth - finger_depth))
        contact = height <= self.contact_threshold_m
        return DepthObservation(
            hand_id=hand_id,
            finger_id=finger_id,
            x=x,
            y=y,
            finger_depth_m=float(finger_depth),
            desk_depth_m=float(desk_depth),
            height_above_desk_m=height,
            contact=contact,
            confidence=conf_value,
            reason="contact" if contact else "above_desk",
            finger_sample_count=finger_count,
            desk_sample_count=desk_count,
            sample_radius_px=max(finger_radius, desk_radius),
            desk_depth_source=desk_source,
        )

    def _unknown(
        self,
        hand_id: int,
        finger_id: int,
        x: int,
        y: int,
        finger_depth: Optional[float],
        desk_depth: Optional[float],
        confidence: float,
        reason: str,
        finger_count: int = 0,
        desk_count: int = 0,
        sample_radius: int = 0,
        desk_source: str = "",
    ) -> DepthObservation:
        height = None if finger_depth is None or desk_depth is None else max(0.0, float(desk_depth - finger_depth))
        return DepthObservation(
            hand_id=hand_id,
            finger_id=finger_id,
            x=x,
            y=y,
            finger_depth_m=finger_depth,
            desk_depth_m=desk_depth,
            height_above_desk_m=height,
            contact=None,
            confidence=confidence,
            reason=reason,
            finger_sample_count=finger_count,
            desk_sample_count=desk_count,
            sample_radius_px=sample_radius,
            desk_depth_source=desk_source,
        )

    def _aligned_depth(self, frame: RGBDFrame) -> Optional[np.ndarray]:
        if frame.depth is None:
            return None
        depth = self._as_image_array(frame.depth)
        if depth is None:
            return None
        depth = depth.astype(np.float32)
        height, width = frame.color_bgr.shape[:2]
        if depth.shape[:2] != (height, width):
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        depth[~np.isfinite(depth)] = np.nan
        depth[depth <= 0] = np.nan
        return depth

    def _aligned_confidence(self, frame: RGBDFrame, shape: Optional[Tuple[int, int]]) -> Optional[np.ndarray]:
        if frame.confidence is None or shape is None:
            return None
        confidence = self._as_image_array(frame.confidence)
        if confidence is None:
            return None
        confidence = confidence.astype(np.float32)
        if confidence.shape[:2] != shape:
            confidence = cv2.resize(confidence, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        max_value = float(np.nanmax(confidence)) if confidence.size else 0.0
        if max_value > 1.0:
            confidence = confidence / max_value
        return confidence

    def _as_image_array(self, value) -> Optional[np.ndarray]:
        array = np.asarray(value)
        if array.size == 0 or array.ndim < 2:
            return None
        array = np.squeeze(array)
        if array.size == 0 or array.ndim < 2:
            return None
        if array.ndim > 2:
            array = array[:, :, 0]
        if array.shape[0] <= 0 or array.shape[1] <= 0:
            return None
        return np.ascontiguousarray(array)

    def _local_depth(self, depth: np.ndarray, x: int, y: int) -> Optional[float]:
        value, _, _ = self._local_depth_stats(depth, x, y, percentile=50.0)
        return value

    def _local_depth_stats(
        self,
        depth: np.ndarray,
        x: int,
        y: int,
        percentile: float,
    ) -> Tuple[Optional[float], int, int]:
        height, width = depth.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return None, 0, 0
        min_samples = max(1, int(config.DEPTH_MIN_SAMPLE_COUNT))
        start_radius = max(0, int(self.sample_radius_px))
        max_radius = max(start_radius, int(config.DEPTH_MAX_SAMPLE_RADIUS_PX))
        last_count = 0
        for radius in range(start_radius, max_radius + 1):
            x1, x2 = max(0, x - radius), min(width, x + radius + 1)
            y1, y2 = max(0, y - radius), min(height, y + radius + 1)
            patch = depth[y1:y2, x1:x2]
            values = patch[np.isfinite(patch) & (patch > 0)]
            last_count = int(values.size)
            if values.size >= min_samples:
                clipped_percentile = float(np.clip(percentile, 0.0, 100.0))
                return float(np.percentile(values, clipped_percentile)), int(values.size), radius
        return None, last_count, max_radius

    def _local_confidence(self, confidence: Optional[np.ndarray], x: int, y: int) -> float:
        if confidence is None:
            return 1.0
        height, width = confidence.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return 0.0
        radius = max(1, self.sample_radius_px)
        x1, x2 = max(0, x - radius), min(width, x + radius + 1)
        y1, y2 = max(0, y - radius), min(height, y + radius + 1)
        patch = confidence[y1:y2, x1:x2]
        values = patch[np.isfinite(patch)]
        if values.size == 0:
            return 0.0
        return float(np.median(values))

    def _zones_mask(self, shape: Tuple[int, int], zones: Iterable[Zone]) -> Optional[np.ndarray]:
        mask = np.zeros(shape, dtype=bool)
        found = False
        for zone in zones:
            if zone.polygon:
                mask_u8 = mask.astype(np.uint8)
                poly = np.array(zone.polygon, dtype=np.int32)
                cv2.fillConvexPoly(mask_u8, poly, 1)
                mask = mask_u8.astype(bool)
                found = True
            else:
                y1, y2 = max(0, zone.y1), min(shape[0], zone.y2)
                x1, x2 = max(0, zone.x1), min(shape[1], zone.x2)
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = True
                    found = True
        return mask if found else None
