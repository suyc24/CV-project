from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from hit_detector import HitDetector
from instrument import InstrumentLayout


def hand_with_finger(hand_id: int, finger_id: int, x: int, y: int):
    landmarks = [(0, 0, 0.0)] * 21
    landmarks[finger_id] = (x, y, 0.0)
    return SimpleNamespace(hand_id=hand_id, landmarks=landmarks)


def contact_observation():
    return SimpleNamespace(contact=True, height_above_desk_m=0.0, reason="contact")


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


def test_pressed_finger_jitter_does_not_retrigger():
    zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
    zone = zones[0]
    detector = HitDetector()
    x = zone.center[0]
    y_start = zone.y1 + 5
    y_hit = int(zone.y1 + zone.height * 0.78)
    detector.update([hand_with_finger(0, 8, x, y_start)], zones, 100.0)
    hits = detector.update([hand_with_finger(0, 8, x, y_hit)], zones, 100.05)
    assert len(hits) == 1

    jitter_positions = [y_hit - 8, y_hit + 5, y_hit - 6, y_hit + 6, y_hit - 7, y_hit + 4]
    for idx, y in enumerate(jitter_positions, start=1):
        hits = detector.update([hand_with_finger(0, 8, x, y)], zones, 100.05 + idx * 0.05)
        assert not hits, f"stationary jitter retriggered at y={y}"


def test_short_lift_before_drop_does_not_trigger():
    zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
    zone = zones[0]
    detector = HitDetector()
    x = zone.center[0]
    y_rest = int(zone.y1 + zone.height * 0.78)
    detector.update([hand_with_finger(0, 8, x, y_rest)], zones, 100.0)
    detector.update([hand_with_finger(0, 8, x, y_rest - 10)], zones, 100.05)
    hits = detector.update([hand_with_finger(0, 8, x, y_rest + 4)], zones, 100.10)
    assert not hits


def test_clear_lift_allows_retrigger():
    zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
    zone = zones[0]
    detector = HitDetector()
    x = zone.center[0]
    y_start = zone.y1 + 5
    y_hit = int(zone.y1 + zone.height * 0.78)
    y_lifted = y_hit - 24
    detector.update([hand_with_finger(0, 8, x, y_start)], zones, 100.0)
    hits = detector.update([hand_with_finger(0, 8, x, y_hit)], zones, 100.05)
    assert len(hits) == 1
    detector.update([hand_with_finger(0, 8, x, y_lifted)], zones, 100.25)
    detector.update([hand_with_finger(0, 8, x, y_lifted)], zones, 100.30)
    hits = detector.update([hand_with_finger(0, 8, x, y_hit)], zones, 100.38)
    assert len(hits) == 1


def test_depth_contact_prevents_false_release_retrigger():
    previous_mode = config.DEPTH_CONTACT_MODE
    config.DEPTH_CONTACT_MODE = "required"
    try:
        zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
        zone = zones[0]
        detector = HitDetector()
        x = zone.center[0]
        y_start = zone.y1 + 5
        y_hit = int(zone.y1 + zone.height * 0.78)
        y_jitter_up = y_hit - 26
        depth = {(0, 8): contact_observation()}

        detector.update([hand_with_finger(0, 8, x, y_start)], zones, 100.0, depth)
        hits = detector.update([hand_with_finger(0, 8, x, y_hit)], zones, 100.05, depth)
        assert len(hits) == 1

        detector.update([hand_with_finger(0, 8, x, y_jitter_up)], zones, 100.25, depth)
        detector.update([hand_with_finger(0, 8, x, y_jitter_up)], zones, 100.30, depth)
        hits = detector.update([hand_with_finger(0, 8, x, y_hit)], zones, 100.38, depth)
        assert not hits
    finally:
        config.DEPTH_CONTACT_MODE = previous_mode


def test_perspective_key_mapping_uses_landing_x():
    zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
    detector = HitDetector()
    for expected_zone in zones:
        point = expected_zone.center
        zone = detector._zone_at(zones, point)
        assert zone is not None
        assert zone.label == expected_zone.label, f"{point} mapped to {zone.label}, expected {expected_zone.label}"


if __name__ == "__main__":
    test_all_fingertips_can_trigger()
    test_short_drop_does_not_trigger()
    test_default_trigger_fingers_include_thumb()
    test_upper_key_landing_can_trigger()
    test_pressed_finger_jitter_does_not_retrigger()
    test_short_lift_before_drop_does_not_trigger()
    test_clear_lift_allows_retrigger()
    test_depth_contact_prevents_false_release_retrigger()
    test_perspective_key_mapping_uses_landing_x()
    print("synthetic hit detector tests passed")
