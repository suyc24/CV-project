from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import config
from hit_detector import HitDetector, HitEvent
from instrument import Zone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay an AirDesk recorded session through HitDetector")
    parser.add_argument("session_dir", help="Session directory created by --record-session")
    parser.add_argument("--piano-velocity-threshold", type=float, default=None)
    parser.add_argument("--piano-strike-velocity", type=float, default=None)
    parser.add_argument("--piano-strike-drop", type=float, default=None)
    parser.add_argument("--piano-strike-net-drop", type=float, default=None)
    parser.add_argument("--piano-release-lift", type=float, default=None)
    parser.add_argument("--piano-arm-lift", type=float, default=None)
    parser.add_argument("--piano-release-stable-frames", type=int, default=None)
    parser.add_argument("--depth-contact-mode", choices=["off", "assist", "required"], default=None)
    parser.add_argument("--piano-trigger-mode", choices=["2d", "hybrid", "3d"], default=None)
    parser.add_argument("--piano-depth-trigger", dest="piano_depth_trigger", action="store_true", default=None)
    parser.add_argument("--no-piano-depth-trigger", dest="piano_depth_trigger", action="store_false")
    parser.add_argument("--piano-depth-release-guard-assist", action="store_true")
    parser.add_argument("--piano-arm-ratio", type=float, default=None)
    parser.add_argument("--drum-velocity-threshold", type=float, default=None)
    parser.add_argument("--piano-press-ratio", type=float, default=None)
    parser.add_argument("--piano-release-ratio", type=float, default=None)
    parser.add_argument("--output-prefix", default="replay", help="Output filename prefix inside the session directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    apply_overrides(args)
    session_dir = Path(args.session_dir)
    frames_path = session_dir / "frames.jsonl"
    if not frames_path.exists():
        raise SystemExit(f"Missing {frames_path}")

    detector = HitDetector()
    replay_hits: List[Dict[str, object]] = []
    online_hits = 0
    reason_counts: Counter[str] = Counter()
    frame_count = 0

    with frames_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            entry = json.loads(line)
            frame_count += 1
            hands = [_hand_from_dict(hand) for hand in entry.get("hands", [])]
            zones = [_zone_from_dict(zone) for zone in entry.get("zones", [])]
            depth_observations = _depth_observations_from_dict(entry.get("depth_observations", []))
            timestamp = float(entry.get("relative_time", entry.get("timestamp", 0.0)))
            hits = detector.update(hands, zones, timestamp, depth_observations)
            online_hits += len(entry.get("hits", []))
            for diag in detector.diagnostics():
                reason_counts[str(diag.get("reason", "unknown"))] += 1
            for hit in hits:
                replay_hits.append(_hit_to_row(entry.get("frame_index", frame_count - 1), hit))

    write_hits_csv(session_dir / f"{args.output_prefix}_hits.csv", replay_hits)
    write_reason_csv(session_dir / f"{args.output_prefix}_miss_reasons.csv", reason_counts)
    summary = {
        "frames": frame_count,
        "online_hits": online_hits,
        "replay_hits": len(replay_hits),
        "reason_counts": dict(reason_counts),
        "config": {
            "PIANO_HIT_VELOCITY_THRESHOLD": config.PIANO_HIT_VELOCITY_THRESHOLD,
            "PIANO_CROSSING_VELOCITY_THRESHOLD": config.PIANO_CROSSING_VELOCITY_THRESHOLD,
            "HIT_VELOCITY_THRESHOLD": config.HIT_VELOCITY_THRESHOLD,
            "PIANO_PRESS_RATIO": config.PIANO_PRESS_RATIO,
            "PIANO_RELEASE_RATIO": config.PIANO_RELEASE_RATIO,
            "PIANO_ARM_RATIO": config.PIANO_ARM_RATIO,
            "PIANO_ARM_MIN_LIFT_PX": config.PIANO_ARM_MIN_LIFT_PX,
            "PIANO_RELEASE_STABLE_FRAMES": config.PIANO_RELEASE_STABLE_FRAMES,
            "PIANO_MIN_FALL_FRAMES": config.PIANO_MIN_FALL_FRAMES,
            "PIANO_LIFT_VELOCITY_THRESHOLD": config.PIANO_LIFT_VELOCITY_THRESHOLD,
            "PIANO_FALLING_VELOCITY_THRESHOLD": config.PIANO_FALLING_VELOCITY_THRESHOLD,
            "PIANO_STRIKE_MIN_DROP_PX": config.PIANO_STRIKE_MIN_DROP_PX,
            "PIANO_STRIKE_MIN_VELOCITY": config.PIANO_STRIKE_MIN_VELOCITY,
            "PIANO_RELEASE_LIFT_PX": config.PIANO_RELEASE_LIFT_PX,
            "PIANO_RELEASE_MIN_UP_VELOCITY": config.PIANO_RELEASE_MIN_UP_VELOCITY,
            "PIANO_RELEASE_STRONG_LIFT_MULTIPLIER": config.PIANO_RELEASE_STRONG_LIFT_MULTIPLIER,
            "PIANO_DEPTH_RELEASE_GUARD": config.PIANO_DEPTH_RELEASE_GUARD,
            "PIANO_DEPTH_RELEASE_GUARD_ASSIST": config.PIANO_DEPTH_RELEASE_GUARD_ASSIST,
            "PIANO_TRIGGER_MODE": config.PIANO_TRIGGER_MODE,
            "PIANO_DEPTH_TRIGGER_ENABLED": config.PIANO_DEPTH_TRIGGER_ENABLED,
            "PIANO_DEPTH_ARM_HEIGHT_M": config.PIANO_DEPTH_ARM_HEIGHT_M,
            "PIANO_DEPTH_RELEASE_HEIGHT_M": config.PIANO_DEPTH_RELEASE_HEIGHT_M,
            "PIANO_DEPTH_PRESS_HEIGHT_M": config.PIANO_DEPTH_PRESS_HEIGHT_M,
            "PIANO_DEPTH_MIN_DROP_M": config.PIANO_DEPTH_MIN_DROP_M,
            "PIANO_DEPTH_FALLING_VELOCITY_M_S": config.PIANO_DEPTH_FALLING_VELOCITY_M_S,
            "PIANO_DEPTH_STRIKE_MIN_VELOCITY_M_S": config.PIANO_DEPTH_STRIKE_MIN_VELOCITY_M_S,
            "PIANO_3D_ARM_HEIGHT_M": config.PIANO_3D_ARM_HEIGHT_M,
            "PIANO_3D_RELEASE_HEIGHT_M": config.PIANO_3D_RELEASE_HEIGHT_M,
            "PIANO_3D_PRESS_HEIGHT_M": config.PIANO_3D_PRESS_HEIGHT_M,
            "PIANO_3D_MIN_DROP_M": config.PIANO_3D_MIN_DROP_M,
            "PIANO_3D_FALLING_VELOCITY_M_S": config.PIANO_3D_FALLING_VELOCITY_M_S,
            "PIANO_3D_STRIKE_MIN_VELOCITY_M_S": config.PIANO_3D_STRIKE_MIN_VELOCITY_M_S,
            "PIANO_RELEASE_MIN_NET_LIFT_PX": config.PIANO_RELEASE_MIN_NET_LIFT_PX,
            "PIANO_RELEASE_MAX_SINGLE_FRAME_LIFT_PX": config.PIANO_RELEASE_MAX_SINGLE_FRAME_LIFT_PX,
            "PIANO_JITTER_GUARD_ENABLED": config.PIANO_JITTER_GUARD_ENABLED,
            "PIANO_STRIKE_MIN_NET_DROP_PX": config.PIANO_STRIKE_MIN_NET_DROP_PX,
            "PIANO_STRIKE_MAX_SINGLE_FRAME_DROP_PX": config.PIANO_STRIKE_MAX_SINGLE_FRAME_DROP_PX,
            "PIANO_BLOCK_PASSIVE_ARM_WHILE_DEPTH_CONTACT": config.PIANO_BLOCK_PASSIVE_ARM_WHILE_DEPTH_CONTACT,
            "PIANO_PASSIVE_ARM_MAX_CONTACT_HEIGHT_M": config.PIANO_PASSIVE_ARM_MAX_CONTACT_HEIGHT_M,
            "PIANO_HIT_MIN_VELOCITY": config.PIANO_HIT_MIN_VELOCITY,
            "PIANO_HIT_MAX_VELOCITY": config.PIANO_HIT_MAX_VELOCITY,
            "PIANO_MIN_VOLUME": config.PIANO_MIN_VOLUME,
            "PIANO_USE_RELATIVE_FINGER_MOTION": config.PIANO_USE_RELATIVE_FINGER_MOTION,
            "PIANO_MAX_HITS_PER_HAND_PER_FRAME": config.PIANO_MAX_HITS_PER_HAND_PER_FRAME,
            "TRIGGER_FINGER_IDS": config.TRIGGER_FINGER_IDS,
            "PIANO_HIT_X_MARGIN_RATIO": config.PIANO_HIT_X_MARGIN_RATIO,
            "PIANO_HIT_TOP_MARGIN_RATIO": config.PIANO_HIT_TOP_MARGIN_RATIO,
            "PIANO_HIT_BOTTOM_MARGIN_RATIO": config.PIANO_HIT_BOTTOM_MARGIN_RATIO,
            "PIANO_ZONE_STICKY_ENABLED": config.PIANO_ZONE_STICKY_ENABLED,
            "PIANO_ZONE_STICKY_X_MARGIN_RATIO": config.PIANO_ZONE_STICKY_X_MARGIN_RATIO,
            "PIANO_ZONE_STICKY_MAX_STEP_PX": config.PIANO_ZONE_STICKY_MAX_STEP_PX,
            "DEPTH_CONTACT_MODE": config.DEPTH_CONTACT_MODE,
            "OPTICAL_FLOW_FORWARD_BACKWARD_CHECK": config.OPTICAL_FLOW_FORWARD_BACKWARD_CHECK,
            "OPTICAL_FLOW_MAX_BACKTRACK_ERROR_PX": config.OPTICAL_FLOW_MAX_BACKTRACK_ERROR_PX,
            "TRACKING_REACQUIRE_HIT_BLOCK_FRAMES": config.TRACKING_REACQUIRE_HIT_BLOCK_FRAMES,
            "TRACKING_FULL_MISS_REACQUIRE_HIT_BLOCK_FRAMES": config.TRACKING_FULL_MISS_REACQUIRE_HIT_BLOCK_FRAMES,
            "TRACKING_NEW_HAND_HIT_BLOCK_FRAMES": config.TRACKING_NEW_HAND_HIT_BLOCK_FRAMES,
        },
    }
    (session_dir / f"{args.output_prefix}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


def apply_overrides(args: argparse.Namespace) -> None:
    if args.piano_velocity_threshold is not None:
        config.PIANO_HIT_VELOCITY_THRESHOLD = args.piano_velocity_threshold
        config.PIANO_STRIKE_MIN_VELOCITY = args.piano_velocity_threshold
    if args.piano_strike_velocity is not None:
        config.PIANO_STRIKE_MIN_VELOCITY = args.piano_strike_velocity
    if args.piano_strike_drop is not None:
        config.PIANO_STRIKE_MIN_DROP_PX = args.piano_strike_drop
    if args.piano_strike_net_drop is not None:
        config.PIANO_STRIKE_MIN_NET_DROP_PX = args.piano_strike_net_drop
    if args.piano_release_lift is not None:
        config.PIANO_RELEASE_LIFT_PX = args.piano_release_lift
    if args.piano_arm_lift is not None:
        config.PIANO_ARM_MIN_LIFT_PX = args.piano_arm_lift
    if args.piano_release_stable_frames is not None:
        config.PIANO_RELEASE_STABLE_FRAMES = args.piano_release_stable_frames
    if args.depth_contact_mode is not None:
        config.DEPTH_CONTACT_MODE = args.depth_contact_mode
    if args.piano_trigger_mode is not None:
        config.PIANO_TRIGGER_MODE = args.piano_trigger_mode
        config.PIANO_DEPTH_TRIGGER_ENABLED = args.piano_trigger_mode != "2d"
    if args.piano_depth_trigger is not None:
        config.PIANO_TRIGGER_MODE = "hybrid" if args.piano_depth_trigger else "2d"
        config.PIANO_DEPTH_TRIGGER_ENABLED = args.piano_depth_trigger
    if args.piano_depth_release_guard_assist:
        config.PIANO_DEPTH_RELEASE_GUARD_ASSIST = True
    if args.piano_arm_ratio is not None:
        config.PIANO_ARM_RATIO = args.piano_arm_ratio
    if args.drum_velocity_threshold is not None:
        config.HIT_VELOCITY_THRESHOLD = args.drum_velocity_threshold
    if args.piano_press_ratio is not None:
        config.PIANO_PRESS_RATIO = args.piano_press_ratio
    if args.piano_release_ratio is not None:
        config.PIANO_RELEASE_RATIO = args.piano_release_ratio


def _hand_from_dict(hand: Dict[str, object]):
    return SimpleNamespace(
        hand_id=int(hand["hand_id"]),
        label=str(hand.get("label", "Unknown")),
        landmarks=[tuple(point) for point in hand.get("landmarks", [])],
        tracking_source=str(hand.get("tracking_source", "mediapipe")),
        missed_frames=int(hand.get("missed_frames", 0)),
        unstable_landmark_ids=tuple(int(value) for value in hand.get("unstable_landmark_ids", ())),
    )


def _zone_from_dict(zone: Dict[str, object]) -> Zone:
    height = max(1.0, float(zone["y2"]) - float(zone["y1"]))
    press_ratio = (float(zone.get("press_y", zone["y1"])) - float(zone["y1"])) / height
    release_ratio = (float(zone.get("release_y", zone["y1"])) - float(zone["y1"])) / height
    kind = str(zone["kind"])
    if kind == "piano":
        press_ratio = config.PIANO_PRESS_RATIO
        release_ratio = config.PIANO_RELEASE_RATIO
    elif kind == "drum":
        press_ratio = config.PRESS_RATIO
        release_ratio = config.RELEASE_RATIO
    return Zone(
        label=str(zone["label"]),
        sound_id=str(zone["sound_id"]),
        x1=int(zone["x1"]),
        y1=int(zone["y1"]),
        x2=int(zone["x2"]),
        y2=int(zone["y2"]),
        kind=kind,
        press_ratio=press_ratio,
        release_ratio=release_ratio,
        polygon=_polygon_from_dict(zone),
    )


def _polygon_from_dict(zone: Dict[str, object]):
    polygon = zone.get("polygon")
    if not polygon:
        return None
    return tuple((int(point[0]), int(point[1])) for point in polygon)


def _depth_observations_from_dict(rows: object) -> Dict[tuple[int, int], SimpleNamespace]:
    observations: Dict[tuple[int, int], SimpleNamespace] = {}
    if not isinstance(rows, list):
        return observations
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            hand_id = int(row["hand_id"])
            finger_id = int(row["finger_id"])
        except (KeyError, TypeError, ValueError):
            continue
        observations[(hand_id, finger_id)] = SimpleNamespace(
            hand_id=hand_id,
            finger_id=finger_id,
            x=int(row.get("x", 0) or 0),
            y=int(row.get("y", 0) or 0),
            finger_depth_m=_optional_float(row.get("finger_depth_m")),
            desk_depth_m=_optional_float(row.get("desk_depth_m")),
            height_above_desk_m=_optional_float(row.get("height_above_desk_m")),
            contact=_optional_bool(row.get("contact")),
            confidence=float(row.get("confidence", 0.0) or 0.0),
            reason=str(row.get("reason", "")),
        )
    return observations


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def _hit_to_row(frame_index: int, hit: HitEvent) -> Dict[str, object]:
    return {
        "frame_index": frame_index,
        "timestamp": hit.timestamp,
        "note_id": hit.note_id,
        "sound_id": hit.sound_id,
        "hand_id": hit.hand_id,
        "finger_id": hit.finger_id,
        "velocity": hit.velocity,
        "volume": hit.volume,
    }


def write_hits_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["frame_index", "timestamp", "note_id", "sound_id", "hand_id", "finger_id", "velocity", "volume"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_reason_csv(path: Path, counter: Counter[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["reason", "count"])
        writer.writeheader()
        for reason, count in counter.most_common():
            writer.writerow({"reason": reason, "count": count})


if __name__ == "__main__":
    raise SystemExit(main())
