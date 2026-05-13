from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

import config
from gesture_recognizer import Gesture, GestureUpdate
from hand_tracker import HAND_CONNECTIONS, HandLandmarks
from hit_detector import HitEvent, HitDetector
from instrument import Zone
from loop_station import LoopState, LoopStation


WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
GRAY = (150, 150, 150)
RED = (40, 60, 230)
GREEN = (80, 210, 80)
BLUE = (230, 140, 60)
YELLOW = (80, 220, 245)


def draw_scene(
    frame,
    zones: List[Zone],
    hands: Iterable[HandLandmarks],
    loop_station: LoopStation,
    mode: str,
    fps: float,
    gesture_update: GestureUpdate,
    recent_hit: Optional[HitEvent],
    highlights: Dict[str, float],
    current_time: float,
    hit_detector: HitDetector,
    debug: bool = False,
    debug_lines: Optional[List[str]] = None,
) -> None:
    camera_layer = frame.copy()
    hands = list(hands)
    draw_zones(frame, zones, highlights, current_time)
    if config.HAND_CUTOUT_ENABLED and hands:
        draw_hand_cutouts(frame, camera_layer, hands)
    if config.SHOW_FINGERTIP_MARKERS and hands:
        draw_fingertip_markers(frame, hands)
    draw_status(frame, loop_station, mode, fps, gesture_update, recent_hit)
    if debug:
        lines = list(debug_lines or [])
        lines.extend(hit_detector.debug_snapshot())
        draw_debug(frame, lines)


def draw_zones(frame, zones: List[Zone], highlights: Dict[str, float], current_time: float) -> None:
    if not zones:
        return
    if zones[0].kind == "piano":
        draw_piano_zones(frame, zones, highlights, current_time)
        return
    roi_x1 = min(zone.x1 for zone in zones)
    roi_y1 = min(zone.y1 for zone in zones)
    roi_x2 = max(zone.x2 for zone in zones)
    roi_y2 = max(zone.y2 for zone in zones)
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (120, 120, 120), 2)

    for zone in zones:
        active = highlights.get(zone.sound_id, 0.0) > current_time
        if zone.kind == "piano":
            fill = (225, 225, 215) if not active else (90, 230, 250)
            border = (40, 40, 40)
            text = BLACK
        else:
            fill = (55, 70, 92) if not active else (65, 185, 245)
            border = (190, 210, 220)
            text = WHITE

        overlay = frame.copy()
        cv2.rectangle(overlay, (zone.x1, zone.y1), (zone.x2, zone.y2), fill, -1)
        zone_alpha = 0.45 if zone.kind == "drum" else 0.38
        cv2.addWeighted(overlay, zone_alpha, frame, 1.0 - zone_alpha, 0, frame)
        cv2.rectangle(frame, (zone.x1, zone.y1), (zone.x2, zone.y2), border, 2)
        cv2.line(frame, (zone.x1, int(zone.press_y)), (zone.x2, int(zone.press_y)), (90, 180, 255), 1)
        cv2.line(frame, (zone.x1, int(zone.release_y)), (zone.x2, int(zone.release_y)), (110, 110, 110), 1)
        _put_centered(frame, zone.label, zone.center, text, scale=0.72, thickness=2)


def draw_piano_zones(frame, zones: List[Zone], highlights: Dict[str, float], current_time: float) -> None:
    roi_x1 = min(zone.x1 for zone in zones)
    roi_y1 = min(zone.y1 for zone in zones)
    roi_x2 = max(zone.x2 for zone in zones)
    roi_y2 = max(zone.y2 for zone in zones)
    width = roi_x2 - roi_x1
    height = roi_y2 - roi_y1
    if width <= 0 or height <= 0:
        return

    keybed = _generate_piano_keybed(width, height, len(zones))
    frame[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.addWeighted(
        frame[roi_y1:roi_y2, roi_x1:roi_x2],
        1.0 - config.PIANO_KEYBED_OPACITY,
        keybed,
        config.PIANO_KEYBED_OPACITY,
        0,
    )

    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (24, 24, 24), 3)
    for zone in zones:
        active = highlights.get(zone.sound_id, 0.0) > current_time
        if active:
            overlay = frame.copy()
            cv2.rectangle(overlay, (zone.x1, zone.y1), (zone.x2, zone.y2), (60, 215, 255), -1)
            cv2.addWeighted(overlay, 0.42, frame, 0.58, 0, frame)

        cv2.rectangle(frame, (zone.x1, zone.y1), (zone.x2, zone.y2), (34, 34, 34), 1)
        cv2.line(frame, (zone.x1, int(zone.press_y)), (zone.x2, int(zone.press_y)), (40, 185, 255), 1)
        label_center = (zone.center[0], int(zone.y1 + zone.height * 0.78))
        _put_centered(frame, zone.label, label_center, BLACK, scale=0.52, thickness=2)

    _draw_black_keys(frame, zones)


def _draw_black_keys(frame, zones: List[Zone]) -> None:
    if len(zones) < 2:
        return
    black_after_scale_degrees = {0, 1, 3, 4, 5}
    key_w = zones[0].width
    black_w = max(8, int(key_w * 0.55))
    black_h = int(zones[0].height * 0.58)
    y1 = zones[0].y1
    for idx, zone in enumerate(zones[:-1]):
        if idx % 7 not in black_after_scale_degrees:
            continue
        if idx + 1 >= len(zones):
            continue
        cx = zone.x2
        x1 = int(cx - black_w / 2)
        x2 = x1 + black_w
        _draw_black_key(frame, x1, y1, x2, y1 + black_h)


def _generate_piano_keybed(width: int, height: int, white_key_count: int):
    keybed = np.full((height, width, 3), (238, 234, 224), dtype=np.uint8)
    top_bar_h = max(5, int(height * 0.05))
    bottom_bar_h = max(5, int(height * 0.06))
    cv2.rectangle(keybed, (0, 0), (width, top_bar_h), (10, 10, 10), -1)
    cv2.rectangle(keybed, (0, top_bar_h), (width, top_bar_h + 4), (25, 25, 25), -1)
    cv2.rectangle(keybed, (0, top_bar_h + 1), (width, top_bar_h + 3), (20, 20, 180), -1)
    cv2.rectangle(keybed, (0, height - bottom_bar_h), (width, height), (12, 12, 12), -1)

    for idx in range(white_key_count):
        x1 = int(idx * width / white_key_count)
        x2 = int((idx + 1) * width / white_key_count)
        _draw_white_key(keybed, x1, top_bar_h, x2, height - bottom_bar_h, idx)

    black_after_scale_degrees = {0, 1, 3, 4, 5}
    key_w = width / white_key_count
    black_w = max(8, int(key_w * 0.56))
    black_h = int(height * 0.62)
    for idx in range(white_key_count - 1):
        if idx % 7 not in black_after_scale_degrees:
            continue
        cx = int((idx + 1) * key_w)
        x1 = int(cx - black_w / 2)
        x2 = x1 + black_w
        _draw_black_key(keybed, x1, top_bar_h, x2, black_h)
    return keybed


def _draw_white_key(image, x1: int, y1: int, x2: int, y2: int, index: int) -> None:
    fill = (244, 240, 230) if index % 2 else (250, 247, 239)
    cv2.rectangle(image, (x1, y1), (x2, y2), fill, -1)
    cv2.line(image, (x1, y1), (x1, y2), (45, 42, 38), 1)
    cv2.line(image, (x2 - 1, y1), (x2 - 1, y2), (80, 76, 68), 1)
    shadow_w = max(2, int((x2 - x1) * 0.06))
    cv2.rectangle(image, (x2 - shadow_w, y1), (x2 - 1, y2), (222, 217, 205), -1)
    highlight_h = max(8, int((y2 - y1) * 0.16))
    cv2.rectangle(image, (x1 + 2, y1 + 2), (x2 - shadow_w - 1, y1 + highlight_h), (255, 253, 246), -1)
    cv2.rectangle(image, (x1 + 2, y2 - 8), (x2 - 2, y2 - 2), (224, 220, 210), -1)


def _draw_black_key(image, x1: int, y1: int, x2: int, y2: int) -> None:
    x1 = max(0, x1)
    x2 = min(image.shape[1] - 1, x2)
    y1 = max(0, y1)
    y2 = min(image.shape[0] - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(image, (x1, y1), (x2, y2), (8, 8, 10), -1)
    cv2.rectangle(image, (x1 + 2, y1 + 2), (x2 - 3, y2 - 4), (20, 20, 22), -1)
    cv2.rectangle(image, (x1 + 5, y1 + 6), (x2 - 7, y2 - 14), (34, 34, 36), -1)
    cv2.rectangle(image, (x1 + 5, y2 - 18), (x2 - 7, y2 - 5), (54, 54, 56), -1)
    cv2.line(image, (x1 + 2, y1 + 2), (x1 + 2, y2 - 5), (72, 72, 74), 1)
    cv2.line(image, (x2 - 3, y1 + 2), (x2 - 3, y2 - 5), (2, 2, 2), 1)


def draw_hand_cutouts(frame, camera_layer, hands: Iterable[HandLandmarks]) -> None:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for hand in hands:
        points = [(int(x), int(y)) for x, y, _ in hand.landmarks]
        if not points:
            continue
        hull = cv2.convexHull(np.array(points, dtype=np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        for a, b in HAND_CONNECTIONS:
            if a < len(points) and b < len(points):
                cv2.line(mask, points[a], points[b], 255, config.HAND_CUTOUT_LINE_THICKNESS)
        for point in points:
            cv2.circle(mask, point, config.HAND_CUTOUT_HULL_PADDING, 255, -1)

    if not np.any(mask):
        return

    mask = cv2.dilate(mask, np.ones((9, 9), dtype=np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    alpha = (mask.astype(np.float32) / 255.0) * config.HAND_CUTOUT_ALPHA
    alpha_3 = alpha[:, :, None]
    blended = camera_layer.astype(np.float32) * alpha_3 + frame.astype(np.float32) * (1.0 - alpha_3)
    frame[:] = np.clip(blended, 0, 255).astype(np.uint8)


def draw_fingertip_markers(frame, hands: Iterable[HandLandmarks]) -> None:
    for hand in hands:
        for finger_id in config.FINGER_TIP_IDS:
            if finger_id >= len(hand.landmarks):
                continue
            x, y, _ = hand.landmarks[finger_id]
            color = YELLOW if finger_id == 8 else (130, 225, 255)
            cv2.circle(frame, (x, y), 8, (15, 15, 15), 2)
            cv2.circle(frame, (x, y), 5, color, -1)


def draw_hands(frame, hands: Iterable[HandLandmarks]) -> None:
    for hand in hands:
        points = hand.landmarks
        for a, b in HAND_CONNECTIONS:
            if a < len(points) and b < len(points):
                cv2.line(frame, points[a][:2], points[b][:2], (70, 220, 190), 2)
        for idx, (x, y, _) in enumerate(points):
            radius = 6 if idx in {8, 12} else 3
            color = YELLOW if idx == 8 else (120, 220, 255) if idx == 12 else (80, 200, 170)
            cv2.circle(frame, (x, y), radius, color, -1)
        if points:
            cv2.putText(frame, f"H{hand.hand_id} {hand.label}", points[0][:2], cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)


def draw_trails(frame, trails: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> None:
    for (_, finger_id), points in trails.items():
        color = YELLOW if finger_id == 8 else (120, 220, 255)
        for idx in range(1, len(points)):
            thickness = max(1, int(3 * idx / len(points)))
            cv2.line(frame, points[idx - 1], points[idx], color, thickness)


def draw_status(
    frame,
    loop_station: LoopStation,
    mode: str,
    fps: float,
    gesture_update: GestureUpdate,
    recent_hit: Optional[HitEvent],
) -> None:
    panel_w, panel_h = 430, 150
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (12 + panel_w, 12 + panel_h), (15, 18, 22), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    cv2.rectangle(frame, (12, 12), (12 + panel_w, 12 + panel_h), (100, 110, 120), 1)

    y = 40
    cv2.putText(frame, f"Mode: {mode.upper()}   FPS: {fps:4.1f}", (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, WHITE, 2)
    y += 30
    state_color = _loop_color(loop_station.state)
    cv2.putText(
        frame,
        f"Loop: {loop_station.state.value}  events={loop_station.event_count}  len={loop_station.loop_duration:0.1f}s",
        (28, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        state_color,
        2,
    )
    y += 30
    gesture = gesture_update.gesture.value if gesture_update.gesture else Gesture.UNKNOWN.value
    action = f" -> {gesture_update.action}" if gesture_update.action else ""
    cv2.putText(frame, f"Gesture: {gesture}{action}", (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, WHITE, 2)
    y += 30
    if recent_hit:
        hit_text = f"Hit: {recent_hit.sound_id}  vy={recent_hit.velocity:0.0f}  vol={recent_hit.volume:0.2f}"
    else:
        hit_text = "Hit: none"
    cv2.putText(frame, hit_text, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, WHITE, 2)


def draw_debug(frame, lines: List[str]) -> None:
    height, width = frame.shape[:2]
    x = width - 330
    y = 34
    cv2.putText(frame, "Debug", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, WHITE, 2)
    for line in lines[:10]:
        y += 24
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 230, 230), 1)


def _loop_color(state: LoopState) -> Tuple[int, int, int]:
    if state == LoopState.RECORDING:
        return RED
    if state == LoopState.PLAYING:
        return GREEN
    return GRAY


def _put_centered(frame, text: str, center: Tuple[int, int], color: Tuple[int, int, int], scale: float, thickness: int) -> None:
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = int(center[0] - size[0] / 2)
    y = int(center[1] + size[1] / 2)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
