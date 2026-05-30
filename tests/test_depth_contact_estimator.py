from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from depth_contact import DepthContactEstimator
from hand_tracker import HandLandmarks
from instrument import Zone


def make_frame(depth):
    return SimpleNamespace(color_bgr=np.zeros((30, 30, 3), dtype=np.uint8), depth=depth, confidence=None)


def make_hand(x: int = 15, y: int = 15):
    landmarks = [(0, 0, 0.0)] * 21
    landmarks[8] = (x, y, 0.0)
    return HandLandmarks(
        hand_id=0,
        label="Right",
        landmarks=landmarks,
        normalized_landmarks=[(0.0, 0.0, 0.0)] * 21,
    )


def make_zone():
    return Zone("C4", "c4", 0, 0, 30, 30, "piano")


def test_fingertip_depth_uses_local_patch_when_exact_pixel_is_missing():
    estimator = DepthContactEstimator(sample_radius_px=4, baseline_frames=1)
    estimator.calibrate(make_frame(np.ones((30, 30), dtype=np.float32)), [make_zone()])
    depth = np.full((30, 30), np.nan, dtype=np.float32)
    depth[13:18, 13:18] = 0.96
    depth[15, 15] = np.nan

    observations = estimator.update(make_frame(depth), [make_hand()], [make_zone()])
    observation = observations[(0, 8)]

    assert observation.contact is True
    assert observation.reason == "contact"


def test_desk_depth_uses_calibrated_plane_when_local_baseline_has_hole():
    baseline = np.ones((30, 30), dtype=np.float32)
    baseline[9:22, 9:22] = np.nan
    estimator = DepthContactEstimator(sample_radius_px=4, baseline_frames=1)
    estimator.calibrate(make_frame(baseline), [make_zone()])
    runtime = np.ones((30, 30), dtype=np.float32)
    runtime[13:18, 13:18] = 0.96

    observations = estimator.update(make_frame(runtime), [make_hand()], [make_zone()])
    observation = observations[(0, 8)]

    assert observation.desk_depth_m is not None
    assert abs(observation.desk_depth_m - 1.0) < 0.01
    assert observation.contact is True


if __name__ == "__main__":
    test_fingertip_depth_uses_local_patch_when_exact_pixel_is_missing()
    test_desk_depth_uses_calibrated_plane_when_local_baseline_has_hole()
    print("depth contact estimator tests passed")
