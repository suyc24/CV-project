from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from hit_detector import FINGER_BASE_IDS, HitDetector
from instrument import InstrumentLayout


def hand_with_finger(hand_id: int, finger_id: int, x: int, y: int):
    landmarks = [(0, 0, 0.0)] * 21
    landmarks[finger_id] = (x, y, 0.0)
    return SimpleNamespace(hand_id=hand_id, landmarks=landmarks)


def hand_with_finger_positions(hand_id: int, positions: dict[int, tuple[int, int, int]]):
    landmarks = [(0, 0, 0.0)] * 21
    for finger_id, (x, y, base_y) in positions.items():
        landmarks[finger_id] = (x, y, 0.0)
        base_id = FINGER_BASE_IDS.get(finger_id)
        if base_id is not None:
            landmarks[base_id] = (x, base_y, 0.0)
    return SimpleNamespace(hand_id=hand_id, landmarks=landmarks)


def contact_observation():
    return SimpleNamespace(contact=True, height_above_desk_m=0.0, reason="contact")


def air_observation():
    return SimpleNamespace(contact=False, height_above_desk_m=0.12, reason="air")


def unknown_depth_observation():
    return SimpleNamespace(contact=None, height_above_desk_m=None, reason="no_finger_depth")


def depth_height_observation(height_m: float):
    return SimpleNamespace(
        contact=height_m <= config.PIANO_DEPTH_PRESS_HEIGHT_M,
        height_above_desk_m=height_m,
        reason="contact" if height_m <= config.PIANO_DEPTH_PRESS_HEIGHT_M else "above_desk",
    )


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
    previous_depth_trigger = config.PIANO_DEPTH_TRIGGER_ENABLED
    config.PIANO_DEPTH_TRIGGER_ENABLED = False
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
        config.PIANO_DEPTH_TRIGGER_ENABLED = previous_depth_trigger


def test_depth_height_can_trigger_with_flat_2d_motion():
    previous = config.PIANO_DEPTH_TRIGGER_ENABLED
    previous_mode = config.PIANO_TRIGGER_MODE
    config.PIANO_DEPTH_TRIGGER_ENABLED = True
    config.PIANO_TRIGGER_MODE = "2d"
    try:
        zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
        zone = zones[0]
        detector = HitDetector()
        x = zone.center[0]
        y = zone.center[1]

        detector.update(
            [hand_with_finger(0, 8, x, y)],
            zones,
            100.0,
            {(0, 8): depth_height_observation(0.090)},
        )
        hits = detector.update(
            [hand_with_finger(0, 8, x, y)],
            zones,
            100.08,
            {(0, 8): depth_height_observation(0.012)},
        )

        assert len(hits) == 1
        assert hits[0].finger_id == 8
    finally:
        config.PIANO_DEPTH_TRIGGER_ENABLED = previous
        config.PIANO_TRIGGER_MODE = previous_mode


def test_depth_height_release_allows_retrigger_with_flat_2d_motion():
    previous = config.PIANO_DEPTH_TRIGGER_ENABLED
    previous_mode = config.PIANO_TRIGGER_MODE
    config.PIANO_DEPTH_TRIGGER_ENABLED = True
    config.PIANO_TRIGGER_MODE = "2d"
    try:
        zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
        zone = zones[0]
        detector = HitDetector()
        x = zone.center[0]
        y = zone.center[1]

        detector.update([hand_with_finger(0, 8, x, y)], zones, 100.0, {(0, 8): depth_height_observation(0.090)})
        detector.update([hand_with_finger(0, 8, x, y)], zones, 100.04, {(0, 8): depth_height_observation(0.040)})
        hits = detector.update([hand_with_finger(0, 8, x, y)], zones, 100.08, {(0, 8): depth_height_observation(0.012)})
        assert len(hits) == 1

        detector.update([hand_with_finger(0, 8, x, y)], zones, 100.20, {(0, 8): depth_height_observation(0.080)})
        detector.update([hand_with_finger(0, 8, x, y)], zones, 100.28, {(0, 8): depth_height_observation(0.080)})
        hits = detector.update([hand_with_finger(0, 8, x, y)], zones, 100.36, {(0, 8): depth_height_observation(0.012)})

        assert len(hits) == 1
    finally:
        config.PIANO_DEPTH_TRIGGER_ENABLED = previous
        config.PIANO_TRIGGER_MODE = previous_mode


def test_3d_trigger_mode_ignores_2d_hit_without_depth():
    previous = config.PIANO_DEPTH_TRIGGER_ENABLED
    previous_mode = config.PIANO_TRIGGER_MODE
    config.PIANO_DEPTH_TRIGGER_ENABLED = True
    config.PIANO_TRIGGER_MODE = "3d"
    try:
        zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
        zone = zones[0]
        detector = HitDetector()
        x = zone.center[0]
        y_start = zone.y1 + 5
        y_hit = int(zone.y1 + zone.height * 0.78)

        detector.update([hand_with_finger(0, 8, x, y_start)], zones, 100.0)
        hits = detector.update([hand_with_finger(0, 8, x, y_hit)], zones, 100.035)

        assert not hits
        assert "depth_unavailable" in [diag["reason"] for diag in detector.diagnostics()]
    finally:
        config.PIANO_DEPTH_TRIGGER_ENABLED = previous
        config.PIANO_TRIGGER_MODE = previous_mode


def test_3d_trigger_mode_uses_depth_for_flat_2d_motion():
    previous = config.PIANO_DEPTH_TRIGGER_ENABLED
    previous_mode = config.PIANO_TRIGGER_MODE
    config.PIANO_DEPTH_TRIGGER_ENABLED = True
    config.PIANO_TRIGGER_MODE = "3d"
    try:
        zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
        zone = zones[0]
        detector = HitDetector()
        x = zone.center[0]
        y = zone.center[1]

        detector.update([hand_with_finger(0, 8, x, y)], zones, 100.0, {(0, 8): depth_height_observation(0.090)})
        detector.update([hand_with_finger(0, 8, x, y)], zones, 100.04, {(0, 8): depth_height_observation(0.040)})
        hits = detector.update([hand_with_finger(0, 8, x, y)], zones, 100.08, {(0, 8): depth_height_observation(0.012)})

        assert len(hits) == 1
        assert hits[0].finger_id == 8
    finally:
        config.PIANO_DEPTH_TRIGGER_ENABLED = previous
        config.PIANO_TRIGGER_MODE = previous_mode


def test_3d_resting_contact_does_not_trigger():
    previous = config.PIANO_DEPTH_TRIGGER_ENABLED
    previous_mode = config.PIANO_TRIGGER_MODE
    config.PIANO_DEPTH_TRIGGER_ENABLED = True
    config.PIANO_TRIGGER_MODE = "3d"
    try:
        zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
        zone = zones[4]
        detector = HitDetector()
        x, y = zone.center

        for idx, height in enumerate([0.006, 0.004, 0.005, 0.004, 0.006]):
            hits = detector.update(
                [hand_with_finger(0, 8, x, y)],
                zones,
                100.0 + idx * 0.04,
                {(0, 8): depth_height_observation(height)},
            )
            assert not hits
        assert "depth_resting" in [diag["reason"] for diag in detector.diagnostics()]
    finally:
        config.PIANO_DEPTH_TRIGGER_ENABLED = previous
        config.PIANO_TRIGGER_MODE = previous_mode


def test_3d_hover_without_contact_does_not_trigger():
    previous = config.PIANO_DEPTH_TRIGGER_ENABLED
    previous_mode = config.PIANO_TRIGGER_MODE
    config.PIANO_DEPTH_TRIGGER_ENABLED = True
    config.PIANO_TRIGGER_MODE = "3d"
    try:
        zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
        zone = zones[4]
        detector = HitDetector()
        x, y = zone.center

        for idx, height in enumerate([0.040, 0.035, 0.032, 0.030, 0.034, 0.038]):
            hits = detector.update(
                [hand_with_finger(0, 8, x, y)],
                zones,
                100.0 + idx * 0.04,
                {(0, 8): depth_height_observation(height)},
            )
            assert not hits
    finally:
        config.PIANO_DEPTH_TRIGGER_ENABLED = previous
        config.PIANO_TRIGGER_MODE = previous_mode


def test_3d_low_lift_ring_tap_triggers_once():
    previous = config.PIANO_DEPTH_TRIGGER_ENABLED
    previous_mode = config.PIANO_TRIGGER_MODE
    config.PIANO_DEPTH_TRIGGER_ENABLED = True
    config.PIANO_TRIGGER_MODE = "3d"
    try:
        zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
        zone = zones[2]
        detector = HitDetector(finger_ids=(16,))
        x, y = zone.center

        detector.update([hand_with_finger(0, 16, x, y)], zones, 100.0, {(0, 16): depth_height_observation(0.030)})
        detector.update([hand_with_finger(0, 16, x, y)], zones, 100.04, {(0, 16): depth_height_observation(0.020)})
        hits = detector.update([hand_with_finger(0, 16, x, y)], zones, 100.08, {(0, 16): depth_height_observation(0.010)})
        assert len(hits) == 1
        assert hits[0].finger_id == 16
        assert hits[0].note_id == zone.label

        hits = detector.update([hand_with_finger(0, 16, x, y)], zones, 100.12, {(0, 16): depth_height_observation(0.009)})
        assert not hits
    finally:
        config.PIANO_DEPTH_TRIGGER_ENABLED = previous
        config.PIANO_TRIGGER_MODE = previous_mode


def test_3d_release_allows_retrigger_without_exaggerated_lift():
    previous = config.PIANO_DEPTH_TRIGGER_ENABLED
    previous_mode = config.PIANO_TRIGGER_MODE
    config.PIANO_DEPTH_TRIGGER_ENABLED = True
    config.PIANO_TRIGGER_MODE = "3d"
    try:
        zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
        zone = zones[4]
        detector = HitDetector()
        x, y = zone.center

        detector.update([hand_with_finger(0, 8, x, y)], zones, 100.0, {(0, 8): depth_height_observation(0.030)})
        detector.update([hand_with_finger(0, 8, x, y)], zones, 100.03, {(0, 8): depth_height_observation(0.020)})
        hits = detector.update([hand_with_finger(0, 8, x, y)], zones, 100.06, {(0, 8): depth_height_observation(0.010)})
        assert len(hits) == 1

        detector.update([hand_with_finger(0, 8, x, y)], zones, 100.22, {(0, 8): depth_height_observation(0.028)})
        detector.update([hand_with_finger(0, 8, x, y)], zones, 100.27, {(0, 8): depth_height_observation(0.029)})
        hits = detector.update([hand_with_finger(0, 8, x, y)], zones, 100.33, {(0, 8): depth_height_observation(0.010)})
        assert len(hits) == 1
    finally:
        config.PIANO_DEPTH_TRIGGER_ENABLED = previous
        config.PIANO_TRIGGER_MODE = previous_mode


def test_3d_adjacent_key_uses_landing_zone_for_note():
    previous = config.PIANO_DEPTH_TRIGGER_ENABLED
    previous_mode = config.PIANO_TRIGGER_MODE
    config.PIANO_DEPTH_TRIGGER_ENABLED = True
    config.PIANO_TRIGGER_MODE = "3d"
    try:
        zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
        g4 = next(zone for zone in zones if zone.label == "G4")
        a4 = next(zone for zone in zones if zone.label == "A4")
        detector = HitDetector()
        y = g4.center[1]

        detector.update([hand_with_finger(0, 8, g4.center[0], y)], zones, 100.0, {(0, 8): depth_height_observation(0.032)})
        detector.update([hand_with_finger(0, 8, g4.center[0], y)], zones, 100.03, {(0, 8): depth_height_observation(0.021)})
        hits = detector.update([hand_with_finger(0, 8, a4.center[0], y)], zones, 100.06, {(0, 8): depth_height_observation(0.010)})
        assert len(hits) == 1
        assert hits[0].note_id == "A4"
    finally:
        config.PIANO_DEPTH_TRIGGER_ENABLED = previous
        config.PIANO_TRIGGER_MODE = previous_mode


def test_3d_depth_mode_all_fingertips_can_trigger():
    previous = config.PIANO_DEPTH_TRIGGER_ENABLED
    previous_mode = config.PIANO_TRIGGER_MODE
    previous_ignore_thumb = config.PIANO_3D_IGNORE_THUMB
    config.PIANO_DEPTH_TRIGGER_ENABLED = True
    config.PIANO_TRIGGER_MODE = "3d"
    config.PIANO_3D_IGNORE_THUMB = False
    try:
        zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
        zone = zones[4]
        for finger_id in (4, 8, 12, 16, 20):
            detector = HitDetector(finger_ids=(finger_id,))
            x, y = zone.center
            detector.update([hand_with_finger(0, finger_id, x, y)], zones, 100.0, {(0, finger_id): depth_height_observation(0.090)})
            detector.update([hand_with_finger(0, finger_id, x, y)], zones, 100.04, {(0, finger_id): depth_height_observation(0.040)})
            hits = detector.update([hand_with_finger(0, finger_id, x, y)], zones, 100.08, {(0, finger_id): depth_height_observation(0.010)})
            assert len(hits) == 1, f"finger {finger_id} did not trigger in 3d mode"
            assert hits[0].finger_id == finger_id
    finally:
        config.PIANO_DEPTH_TRIGGER_ENABLED = previous
        config.PIANO_TRIGGER_MODE = previous_mode
        config.PIANO_3D_IGNORE_THUMB = previous_ignore_thumb


def test_3d_missing_depth_owner_suppresses_passive_finger():
    previous = config.PIANO_DEPTH_TRIGGER_ENABLED
    previous_mode = config.PIANO_TRIGGER_MODE
    previous_owner = config.PIANO_3D_MISSING_DEPTH_HAND_OWNER_ENABLED
    previous_isolation = config.PIANO_3D_MISSING_DEPTH_FINGER_ISOLATION_ENABLED
    config.PIANO_DEPTH_TRIGGER_ENABLED = True
    config.PIANO_TRIGGER_MODE = "3d"
    config.PIANO_3D_MISSING_DEPTH_HAND_OWNER_ENABLED = True
    config.PIANO_3D_MISSING_DEPTH_FINGER_ISOLATION_ENABLED = True
    try:
        zones = InstrumentLayout("piano").get_zones((720, 1280, 3))
        e4 = next(zone for zone in zones if zone.label == "E4")
        g4 = next(zone for zone in zones if zone.label == "G4")
        detector = HitDetector(finger_ids=(8, 16))
        unknown = {(0, 8): unknown_depth_observation(), (0, 16): unknown_depth_observation()}
        rest_hand = hand_with_finger_positions(
            0,
            {
                8: (g4.center[0], g4.center[1], g4.center[1] - 150),
                16: (e4.center[0], e4.center[1], e4.center[1] - 150),
            },
        )

        detector.update([rest_hand], zones, 100.0, unknown)
        detector.update([rest_hand], zones, 100.05, unknown)
        detector.update(
            [
                hand_with_finger_positions(
                    0,
                    {
                        8: (g4.center[0], g4.center[1] - 30, g4.center[1] - 150),
                        16: (e4.center[0], e4.center[1] - 18, e4.center[1] - 150),
                    },
                )
            ],
            zones,
            100.10,
            unknown,
        )
        hits = detector.update(
            [
                hand_with_finger_positions(
                    0,
                    {
                        8: (g4.center[0], g4.center[1] + 2, g4.center[1] - 150),
                        16: (e4.center[0], e4.center[1] + 2, e4.center[1] - 150),
                    },
                )
            ],
            zones,
            100.15,
            unknown,
        )

        assert len(hits) == 1
        assert hits[0].finger_id == 8
        assert hits[0].note_id == "G4"
    finally:
        config.PIANO_DEPTH_TRIGGER_ENABLED = previous
        config.PIANO_TRIGGER_MODE = previous_mode
        config.PIANO_3D_MISSING_DEPTH_HAND_OWNER_ENABLED = previous_owner
        config.PIANO_3D_MISSING_DEPTH_FINGER_ISOLATION_ENABLED = previous_isolation


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
    test_depth_contact_prevents_false_release_retrigger()
    test_depth_contact_blocks_passive_piano_arm()
    test_depth_height_can_trigger_with_flat_2d_motion()
    test_depth_height_release_allows_retrigger_with_flat_2d_motion()
    test_3d_trigger_mode_ignores_2d_hit_without_depth()
    test_3d_trigger_mode_uses_depth_for_flat_2d_motion()
    test_3d_resting_contact_does_not_trigger()
    test_3d_hover_without_contact_does_not_trigger()
    test_3d_low_lift_ring_tap_triggers_once()
    test_3d_release_allows_retrigger_without_exaggerated_lift()
    test_3d_adjacent_key_uses_landing_zone_for_note()
    test_3d_depth_mode_all_fingertips_can_trigger()
    test_3d_missing_depth_owner_suppresses_passive_finger()
    test_piano_zone_mapping_sticks_near_boundary()
    test_perspective_key_mapping_uses_landing_x()
    print("synthetic hit detector tests passed")
