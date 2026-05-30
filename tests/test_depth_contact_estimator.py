from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from depth_contact import DepthContactEstimator
from instrument import Zone
from rgbd_camera import RGBDFrame


def rgbd(depth, confidence=None):
    height, width = depth.shape[:2]
    return RGBDFrame(
        color_bgr=np.zeros((height, width, 3), dtype=np.uint8),
        depth=depth.astype(np.float32),
        confidence=None if confidence is None else confidence.astype(np.float32),
        timestamp=0.0,
    )


def full_zone(width: int, height: int) -> Zone:
    return Zone("C4", "c4", 0, 0, width - 1, height - 1, "piano")


def hand_at(x: int, y: int):
    landmarks = [(0, 0, 0.0)] * 21
    landmarks[8] = (x, y, 0.0)
    return SimpleNamespace(hand_id=0, landmarks=landmarks)


def calibrated_estimator(depth: np.ndarray) -> DepthContactEstimator:
    estimator = DepthContactEstimator(sample_radius_px=2, baseline_frames=1)
    assert estimator.calibrate(rgbd(depth), [full_zone(depth.shape[1], depth.shape[0])])
    return estimator


def test_adaptive_fingertip_patch_uses_nearby_valid_depth():
    width, height = 80, 60
    x, y = 40, 30
    estimator = calibrated_estimator(np.full((height, width), 0.50, dtype=np.float32))
    frame_depth = np.full((height, width), np.nan, dtype=np.float32)
    frame_depth[y - 6 : y - 4, x - 1 : x + 2] = 0.47
    frame_depth[y + 4 : y + 6, x - 1 : x + 2] = 0.47

    observations = estimator.update(rgbd(frame_depth), [hand_at(x, y)], [full_zone(width, height)])
    obs = observations[(0, 8)]

    assert obs.finger_depth_m is not None
    assert obs.height_above_desk_m is not None
    assert obs.height_above_desk_m > 0.02
    assert obs.finger_sample_count >= 3
    assert obs.sample_radius_px > 2


def test_baseline_plane_fills_local_desk_hole():
    width, height = 96, 72
    x, y = 48, 38
    xs = np.arange(width, dtype=np.float32)[None, :]
    ys = np.arange(height, dtype=np.float32)[:, None]
    baseline = 0.45 + xs * 0.0002 + ys * 0.0001
    baseline[y - 14 : y + 15, x - 14 : x + 15] = np.nan
    estimator = calibrated_estimator(baseline)

    frame_depth = np.full((height, width), np.nan, dtype=np.float32)
    frame_depth[y - 2 : y + 3, x - 2 : x + 3] = 0.42
    observations = estimator.update(rgbd(frame_depth), [hand_at(x, y)], [full_zone(width, height)])
    obs = observations[(0, 8)]

    assert obs.desk_depth_m is not None
    assert obs.desk_depth_source == "plane"
    assert obs.height_above_desk_m is not None
    assert obs.height_above_desk_m > 0.02


def test_low_confidence_returns_unknown_not_contact():
    width, height = 80, 60
    x, y = 40, 30
    estimator = calibrated_estimator(np.full((height, width), 0.50, dtype=np.float32))
    frame_depth = np.full((height, width), 0.49, dtype=np.float32)
    confidence = np.zeros((height, width), dtype=np.float32)

    observations = estimator.update(rgbd(frame_depth, confidence), [hand_at(x, y)], [full_zone(width, height)])
    obs = observations[(0, 8)]

    assert obs.contact is None
    assert obs.reason == "low_confidence"
    assert obs.height_above_desk_m is not None


if __name__ == "__main__":
    test_adaptive_fingertip_patch_uses_nearby_valid_depth()
    test_baseline_plane_fills_local_desk_hole()
    test_low_confidence_returns_unknown_not_contact()
    print("depth contact estimator tests passed")
