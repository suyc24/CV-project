from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hit_detector import HitDetector
from instrument import InstrumentLayout


def hand_with_finger(hand_id: int, finger_id: int, x: int, y: int):
    landmarks = [(0, 0, 0.0)] * 21
    landmarks[finger_id] = (x, y, 0.0)
    return SimpleNamespace(hand_id=hand_id, landmarks=landmarks)


def test_all_fingertips_can_trigger():
    zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
    zone = zones[0]
    detector = HitDetector()
    finger_ids = (4, 8, 12, 16, 20)
    x = zone.center[0]
    y_start = zone.y1 + 5
    y_hit = int(zone.y1 + zone.height * 0.78)
    time_base = 100.0

    for idx, finger_id in enumerate(finger_ids):
        detector.reset()
        detector.update([hand_with_finger(0, finger_id, x, y_start)], zones, time_base + idx)
        hits = detector.update([hand_with_finger(0, finger_id, x, y_hit)], zones, time_base + idx + 0.035)
        assert len(hits) == 1, f"finger {finger_id} did not trigger"
        assert hits[0].finger_id == finger_id


def test_press_line_blocks_shallow_motion():
    zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
    zone = zones[0]
    detector = HitDetector()
    x = zone.center[0]
    y_start = zone.y1 + 5
    y_shallow = int(zone.y1 + zone.height * 0.30)
    detector.update([hand_with_finger(0, 8, x, y_start)], zones, 100.0)
    hits = detector.update([hand_with_finger(0, 8, x, y_shallow)], zones, 100.035)
    assert not hits
    reasons = [diag["reason"] for diag in detector.diagnostics()]
    assert "press_line" in reasons or "velocity" in reasons


if __name__ == "__main__":
    test_all_fingertips_can_trigger()
    test_press_line_blocks_shallow_motion()
    print("synthetic hit detector tests passed")
