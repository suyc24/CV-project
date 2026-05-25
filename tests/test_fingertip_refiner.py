from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from fingertip_refiner import FingertipRefiner


def landmarks_with_index_tip(tip_x: int, tip_y: int):
    landmarks = [(0, 0, 0.0)] * 21
    landmarks[7] = (40, tip_y, 0.0)
    landmarks[8] = (tip_x, tip_y, 0.0)
    return landmarks


def test_refiner_moves_tip_toward_strong_distal_edge():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(frame, (63, 43), (70, 57), (255, 255, 255), thickness=-1)
    refiner = FingertipRefiner(
        finger_ids=(8,),
        radius_px=18,
        max_shift_px=12,
        max_perpendicular_shift_px=5,
        min_edge_score=5.0,
        blend_alpha=1.0,
    )

    refined = refiner.refine_landmarks(frame, landmarks_with_index_tip(58, 50))

    assert refined[8][0] > 58
    assert abs(refined[8][1] - 50) <= 4


def test_refiner_leaves_blank_patch_unchanged():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    refiner = FingertipRefiner(
        finger_ids=(8,),
        radius_px=18,
        max_shift_px=12,
        max_perpendicular_shift_px=5,
        min_edge_score=5.0,
        blend_alpha=1.0,
    )
    landmarks = landmarks_with_index_tip(58, 50)

    refined = refiner.refine_landmarks(frame, landmarks)

    assert refined[8] == landmarks[8]


def test_refiner_rejects_large_sideways_edge():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(frame, (57, 62), (71, 69), (255, 255, 255), thickness=-1)
    refiner = FingertipRefiner(
        finger_ids=(8,),
        radius_px=18,
        max_shift_px=14,
        max_perpendicular_shift_px=3,
        min_edge_score=5.0,
        blend_alpha=1.0,
    )
    landmarks = landmarks_with_index_tip(58, 50)

    refined = refiner.refine_landmarks(frame, landmarks)

    assert refined[8] == landmarks[8]


if __name__ == "__main__":
    test_refiner_moves_tip_toward_strong_distal_edge()
    test_refiner_leaves_blank_patch_unchanged()
    test_refiner_rejects_large_sideways_edge()
    print("fingertip refiner tests passed")
