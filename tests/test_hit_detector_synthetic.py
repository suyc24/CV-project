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


def hand_with_fingers(hand_id: int, points: dict[int, tuple[int, int]]):
    landmarks = [(0, 0, 0.0)] * 21
    for finger_id, (x, y) in points.items():
        landmarks[finger_id] = (x, y, 0.0)
    return SimpleNamespace(hand_id=hand_id, landmarks=landmarks)


def contact_observation():
    return SimpleNamespace(contact=True, height_above_desk_m=0.0, reason="contact")


def air_observation():
    return SimpleNamespace(contact=False, height_above_desk_m=0.12, reason="air")


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


def test_active_finger_blocks_other_same_hand_piano_hit():
    zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
    detector = HitDetector()
    left_zone = zones[0]
    right_zone = zones[1]
    y_start = left_zone.y1 + 5
    y_hit = int(left_zone.y1 + left_zone.height * 0.78)

    detector.update([hand_with_finger(0, 8, left_zone.center[0], y_start)], zones, 100.0)
    hits = detector.update([hand_with_finger(0, 8, left_zone.center[0], y_hit)], zones, 100.05)
    assert len(hits) == 1

    detector.update(
        [hand_with_fingers(0, {8: (left_zone.center[0], y_hit), 12: (right_zone.center[0], y_start)})],
        zones,
        100.20,
    )
    hits = detector.update(
        [hand_with_fingers(0, {8: (left_zone.center[0], y_hit), 12: (right_zone.center[0], y_hit)})],
        zones,
        100.25,
    )
    assert not hits
    assert "suppressed_by_active_finger" in [diag["reason"] for diag in detector.diagnostics()]


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
        air_depth = {(0, 8): air_observation()}
        contact_depth = {(0, 8): contact_observation()}

        detector.update([hand_with_finger(0, 8, x, y_start)], zones, 100.0, air_depth)
        hits = detector.update([hand_with_finger(0, 8, x, y_hit)], zones, 100.05, contact_depth)
        assert len(hits) == 1

        detector.update([hand_with_finger(0, 8, x, y_jitter_up)], zones, 100.25, contact_depth)
        detector.update([hand_with_finger(0, 8, x, y_jitter_up)], zones, 100.30, contact_depth)
        hits = detector.update([hand_with_finger(0, 8, x, y_hit)], zones, 100.38, contact_depth)
        assert not hits
    finally:
        config.DEPTH_CONTACT_MODE = previous_mode


def test_depth_contact_blocks_passive_piano_arm():
    previous = config.PIANO_BLOCK_PASSIVE_ARM_WHILE_DEPTH_CONTACT
    config.PIANO_BLOCK_PASSIVE_ARM_WHILE_DEPTH_CONTACT = True
    try:
        zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
        zone = zones[0]
        detector = HitDetector()
        x = zone.center[0]
        y_start = zone.y1 + 5
        y_hit = int(zone.y1 + zone.height * 0.78)
        depth = {(0, 8): contact_observation()}

        detector.update([hand_with_finger(0, 8, x, y_start)], zones, 100.0, depth)
        assert "contact_arm_guard" in [diag["reason"] for diag in detector.diagnostics()]
        hits = detector.update([hand_with_finger(0, 8, x, y_hit)], zones, 100.05, depth)
        assert not hits
    finally:
        config.PIANO_BLOCK_PASSIVE_ARM_WHILE_DEPTH_CONTACT = previous


def test_piano_zone_mapping_sticks_near_boundary():
    zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
    detector = HitDetector()
    left_zone = zones[0]
    y = left_zone.center[1]
    assert left_zone.polygon is not None
    top_edge = left_zone.polygon[1]
    bottom_edge = left_zone.polygon[2]
    edge_t = (y - top_edge[1]) / (bottom_edge[1] - top_edge[1])
    boundary_x = top_edge[0] + edge_t * (bottom_edge[0] - top_edge[0])
    x_left = int(boundary_x - 1)
    x_right = int(boundary_x + 1)

    detector.update([hand_with_finger(0, 8, x_left, y)], zones, 100.0)
    detector.update([hand_with_finger(0, 8, x_right, y)], zones, 100.05)
    zones_seen = [
        diag["zone_label"]
        for diag in detector.diagnostics()
        if diag["hand_id"] == 0 and diag["finger_id"] == 8
    ]
    assert zones_seen == [left_zone.label]


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
    test_active_finger_blocks_other_same_hand_piano_hit()
    test_depth_contact_prevents_false_release_retrigger()
    test_depth_contact_blocks_passive_piano_arm()
    test_piano_zone_mapping_sticks_near_boundary()
    test_perspective_key_mapping_uses_landing_x()
    print("synthetic hit detector tests passed")
