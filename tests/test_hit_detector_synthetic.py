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
    detector = HitDetector(finger_ids=(4, 8, 12, 16, 20))
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


def test_short_drop_does_not_trigger():
    zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
    zone = zones[0]
    detector = HitDetector()
    x = zone.center[0]
    y_start = zone.y1 + 5
    y_shallow = y_start + 4
    detector.update([hand_with_finger(0, 8, x, y_start)], zones, 100.0)
    hits = detector.update([hand_with_finger(0, 8, x, y_shallow)], zones, 100.035)
    assert not hits
    reasons = [diag["reason"] for diag in detector.diagnostics()]
    assert "hit" not in reasons


def test_default_trigger_fingers_include_thumb():
    zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
    zone = zones[0]
    detector = HitDetector()
    x = zone.center[0]
    y_start = zone.y1 + 5
    y_landing = int(zone.y1 + zone.height * 0.78)
    detector.update([hand_with_finger(0, 4, x, y_start)], zones, 100.0)
    hits = detector.update([hand_with_finger(0, 4, x, y_landing)], zones, 100.035)
    assert len(hits) == 1
    assert hits[0].finger_id == 4


def test_upper_key_landing_can_trigger():
    zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
    zone = zones[0]
    detector = HitDetector()
    x = zone.center[0]
    y_start = zone.y1 + 5
    y_landing = int(zone.y1 + zone.height * 0.32)
    detector.update([hand_with_finger(0, 8, x, y_start)], zones, 100.0)
    hits = detector.update([hand_with_finger(0, 8, x, y_landing)], zones, 100.035)
    assert len(hits) == 1


def test_perspective_key_mapping_uses_landing_x():
    zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
    detector = HitDetector()
    points = [
        ((724, 551), "D5"),
        ((801, 608), "E5"),
        ((910, 569), "F5"),
        ((418, 502), "G4"),
    ]
    for point, expected_label in points:
        zone = detector._zone_at(zones, point)
        assert zone is not None
        assert zone.label == expected_label, f"{point} mapped to {zone.label}, expected {expected_label}"


if __name__ == "__main__":
    test_all_fingertips_can_trigger()
    test_short_drop_does_not_trigger()
    test_default_trigger_fingers_include_thumb()
    test_upper_key_landing_can_trigger()
    test_perspective_key_mapping_uses_landing_x()
    print("synthetic hit detector tests passed")
