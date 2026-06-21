from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import config
from hand_tracker import HandLandmarks, HandTracker


def make_hand(hand_id: int = 0, x: int = 100, y: int = 120, label: str = "Right") -> HandLandmarks:
    landmarks = [(x + idx, y + idx, 0.0) for idx in range(21)]
    return HandLandmarks(
        hand_id=hand_id,
        label=label,
        landmarks=landmarks,
        normalized_landmarks=[(0.0, 0.0, 0.0)] * 21,
    )


class FakeTracker(HandTracker):
    def __init__(self, detections):
        self.detections = list(detections)
        self._max_num_hands = 2
        self._frame_index = 0
        self._smoothed_points = {}
        self._tracked_hand_centers = {}
        self._next_stable_hand_id = 0
        self._flow_points = {}
        self._last_good_hands = []
        self._missed_frame_count = 0
        self._empty_detection_frames = 0
        self._reacquire_guard_frames = 0
        self._new_hand_guard_frames = {}
        self._last_gray = None
        self._fingertip_refiner = None

    def _detect(self, frame_bgr, roi=None):
        return self.detections.pop(0) if self.detections else []

    def _bridge_missing_hands(self, frame_bgr):
        return []

    def _smooth(self, hands):
        return hands

    def _refine_fingertips(self, frame_bgr, hands):
        return hands

    def _stabilize_with_optical_flow(self, frame_bgr, hands):
        return hands


def trigger_ids_are_guarded(hand: HandLandmarks) -> bool:
    return set(config.TRIGGER_FINGER_IDS).issubset(set(hand.unstable_landmark_ids))


def test_new_hand_is_marked_unstable_for_guard_frames():
    guard_frames = int(config.TRACKING_NEW_HAND_HIT_BLOCK_FRAMES)
    tracker = FakeTracker([[make_hand()] for _ in range(guard_frames + 1)])
    frame = np.zeros((80, 80, 3), dtype=np.uint8)

    guarded = [trigger_ids_are_guarded(tracker.process(frame)[0]) for _ in range(guard_frames + 1)]

    assert all(guarded[:guard_frames])
    assert guarded[guard_frames] is False


def test_full_miss_reacquire_gets_longer_guard():
    guard_frames = max(
        int(config.TRACKING_NEW_HAND_HIT_BLOCK_FRAMES),
        int(config.TRACKING_FULL_MISS_REACQUIRE_HIT_BLOCK_FRAMES),
    )
    detections = [[], []] + [[make_hand()] for _ in range(guard_frames + 1)]
    tracker = FakeTracker(detections)
    frame = np.zeros((80, 80, 3), dtype=np.uint8)

    assert tracker.process(frame) == []
    assert tracker.process(frame) == []
    guarded = [trigger_ids_are_guarded(tracker.process(frame)[0]) for _ in range(guard_frames + 1)]

    assert all(guarded[:guard_frames])
    assert guarded[guard_frames] is False


def test_partial_roi_detection_reacquires_full_frame_for_second_hand():
    tracker = FakeTracker([[make_hand(0)], [make_hand(0), make_hand(1, x=280)]])
    tracker._frame_index = 5
    frame = np.zeros((80, 80, 3), dtype=np.uint8)

    hands = tracker.process(frame, roi=(0, 20, 80, 80))

    assert len(hands) == 2


def test_single_hand_label_flip_keeps_stable_id():
    tracker = FakeTracker(
        [
            [make_hand(label="Left")],
            [make_hand(label="Right")],
        ]
    )
    frame = np.zeros((80, 80, 3), dtype=np.uint8)

    first = tracker.process(frame)
    second = tracker.process(frame)

    assert first[0].hand_id == second[0].hand_id


def test_extra_detection_does_not_steal_existing_hand_id():
    tracker = FakeTracker(
        [
            [make_hand(x=100, y=120, label="Left")],
            [
                make_hand(x=210, y=120, label="Left"),
                make_hand(x=102, y=121, label="Left"),
            ],
        ]
    )
    frame = np.zeros((80, 80, 3), dtype=np.uint8)

    first = tracker.process(frame)
    second = tracker.process(frame)

    assert second[1].hand_id == first[0].hand_id


def test_overlapping_extra_detection_is_dropped():
    tracker = FakeTracker(
        [
            [make_hand(x=100, y=120, label="Left")],
            [
                make_hand(x=112, y=124, label="Left"),
                make_hand(x=101, y=120, label="Right"),
            ],
        ]
    )
    frame = np.zeros((80, 80, 3), dtype=np.uint8)

    first = tracker.process(frame)
    second = tracker.process(frame)

    assert len(second) == 1
    assert second[0].hand_id == first[0].hand_id


def test_short_detection_gap_keeps_stable_id():
    tracker = FakeTracker(
        [
            [make_hand(x=100, y=120, label="Left")],
            [],
            [make_hand(x=112, y=128, label="Left")],
        ]
    )
    frame = np.zeros((80, 80, 3), dtype=np.uint8)

    first = tracker.process(frame)
    assert tracker.process(frame) == []
    reacquired = tracker.process(frame)

    assert reacquired[0].hand_id == first[0].hand_id


if __name__ == "__main__":
    test_new_hand_is_marked_unstable_for_guard_frames()
    test_full_miss_reacquire_gets_longer_guard()
    test_partial_roi_detection_reacquires_full_frame_for_second_hand()
    test_single_hand_label_flip_keeps_stable_id()
    test_extra_detection_does_not_steal_existing_hand_id()
    test_overlapping_extra_detection_is_dropped()
    test_short_detection_gap_keeps_stable_id()
    print("hand tracker guard tests passed")
