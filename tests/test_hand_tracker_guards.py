from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import config
from hand_tracker import HandLandmarks, HandTracker


def make_hand(hand_id: int = 0) -> HandLandmarks:
    landmarks = [(100 + idx, 120 + idx, 0.0) for idx in range(21)]
    return HandLandmarks(
        hand_id=hand_id,
        label="Right",
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
    tracker = FakeTracker([[make_hand()], [make_hand()], [make_hand()]])
    frame = np.zeros((80, 80, 3), dtype=np.uint8)

    first = tracker.process(frame)
    second = tracker.process(frame)
    third = tracker.process(frame)

    assert trigger_ids_are_guarded(first[0])
    assert trigger_ids_are_guarded(second[0])
    assert not trigger_ids_are_guarded(third[0])


def test_full_miss_reacquire_gets_longer_guard():
    detections = [[], [], [make_hand()], [make_hand()], [make_hand()], [make_hand()], [make_hand()]]
    tracker = FakeTracker(detections)
    frame = np.zeros((80, 80, 3), dtype=np.uint8)

    assert tracker.process(frame) == []
    assert tracker.process(frame) == []
    guarded = [trigger_ids_are_guarded(tracker.process(frame)[0]) for _ in range(5)]

    assert guarded[:4] == [True, True, True, True]
    assert guarded[4] is False


def test_partial_roi_detection_reacquires_full_frame_for_second_hand():
    tracker = FakeTracker([[make_hand(0)], [make_hand(0), make_hand(1)]])
    tracker._frame_index = 5
    frame = np.zeros((80, 80, 3), dtype=np.uint8)

    hands = tracker.process(frame, roi=(0, 20, 80, 80))

    assert len(hands) == 2


if __name__ == "__main__":
    test_new_hand_is_marked_unstable_for_guard_frames()
    test_full_miss_reacquire_gets_longer_guard()
    test_partial_roi_detection_reacquires_full_frame_for_second_hand()
    print("hand tracker guard tests passed")
