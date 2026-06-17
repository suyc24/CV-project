from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

import config
from camera_utils import tracking_roi
from hand_tracker import HandLandmarks, HandTracker
from hit_detector import HitDetector, HitEvent
from instrument import InstrumentLayout, Zone
from replay_session import _depth_observations_from_dict, _zone_from_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run the current HandTracker on a recorded raw_video.avi and write a new "
            "frames.jsonl. This makes tracker/refiner changes testable without recording "
            "a fresh camera session."
        )
    )
    parser.add_argument("session_dir", help="Existing AirDesk session containing raw_video.avi and frames.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for reprocessed frames.jsonl")
    parser.add_argument("--variant", default="baseline", help="Label stored in metadata")
    parser.add_argument("--fingertip-refinement", action="store_true")
    parser.add_argument("--no-optical-stabilization", action="store_true")
    parser.add_argument("--no-tracking-roi", action="store_true")
    parser.add_argument("--tracking-roi-y", type=float, default=config.TRACKING_ROI_Y_MIN)
    parser.add_argument("--tracking-max-width", type=int, default=config.TRACKING_MAX_WIDTH)
    parser.add_argument("--hand-model-complexity", type=int, choices=[0, 1], default=config.HAND_MODEL_COMPLEXITY)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--min-detection-confidence", type=float, default=config.HAND_MIN_DETECTION_CONFIDENCE)
    parser.add_argument("--min-tracking-confidence", type=float, default=config.HAND_MIN_TRACKING_CONFIDENCE)
    parser.add_argument("--no-landmark-smoothing", action="store_true")
    parser.add_argument("--landmark-smoothing-alpha", type=float, default=config.LANDMARK_SMOOTHING_ALPHA)
    parser.add_argument(
        "--use-recorded-depth",
        action="store_true",
        help=(
            "Reuse recorded depth observations in HitDetector. This is approximate because "
            "reprocessed stable hand ids can differ from the original run."
        ),
    )
    parser.add_argument(
        "--rebuild-zones",
        action="store_true",
        help="Rebuild instrument zones from current layout/config instead of replaying recorded zones",
    )
    parser.add_argument("--piano-left-trim-keys", type=float, default=config.PIANO_LEFT_TRIM_KEYS)
    parser.add_argument("--piano-right-trim-keys", type=float, default=config.PIANO_RIGHT_TRIM_KEYS)
    parser.add_argument("--limit-frames", type=int, default=0, help="Optional smoke-test frame limit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.no_optical_stabilization:
        config.FINGERTIP_OPTICAL_FLOW_STABILIZATION = False
    config.PIANO_LEFT_TRIM_KEYS = max(0.0, float(args.piano_left_trim_keys))
    config.PIANO_RIGHT_TRIM_KEYS = max(0.0, float(args.piano_right_trim_keys))

    session_dir = Path(args.session_dir)
    source_frames_path = session_dir / "frames.jsonl"
    video_path = session_dir / "raw_video.avi"
    if not source_frames_path.exists():
        raise SystemExit(f"Missing {source_frames_path}")
    if not video_path.exists():
        raise SystemExit(f"Missing {video_path}; record sessions with video enabled for tracker benchmarks")

    source_frames = load_jsonl(source_frames_path)
    source_metadata = load_metadata(session_dir / "metadata.json")
    layout = InstrumentLayout(
        str(source_metadata.get("mode", "piano")),
        roi_ratios=parse_roi(source_metadata.get("instrument_roi")),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_session": str(session_dir),
        "variant": args.variant,
        "args": vars(args),
        "format": "aird_esktop_reprocessed_tracking_v1",
        "files": {"frames": "frames.jsonl", "video": None},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {video_path}")

    try:
        tracker = HandTracker(
            max_num_hands=args.max_hands,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            input_max_width=args.tracking_max_width,
            model_complexity=args.hand_model_complexity,
            smooth_landmarks=not args.no_landmark_smoothing,
            smoothing_alpha=args.landmark_smoothing_alpha,
            refine_fingertips=args.fingertip_refinement,
        )
    except RuntimeError as exc:
        raise SystemExit(
            "Could not initialize HandTracker for offline reprocessing. "
            "Run this tool with the same Python environment you use for `main.py` "
            "(Python 3.10/3.11 with requirements.txt installed). "
            f"Original error: {exc}"
        ) from exc
    detector = HitDetector()
    reason_counts: Counter[str] = Counter()
    hit_count = 0
    frame_count = 0
    mismatch_count = 0

    try:
        with (output_dir / "frames.jsonl").open("w", encoding="utf-8") as out:
            for source_entry in source_frames:
                if args.limit_frames and frame_count >= args.limit_frames:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                source_shape = tuple(source_entry.get("frame_shape", frame.shape))
                if tuple(frame.shape) != source_shape:
                    mismatch_count += 1

                roi = None if args.no_tracking_roi else tracking_roi(frame.shape, args.tracking_roi_y)
                hands = tracker.process(frame, roi=roi)
                if args.rebuild_zones:
                    zones = layout.get_zones(tuple(frame.shape))
                else:
                    zones = transform_zones(
                        [_zone_from_dict(zone) for zone in source_entry.get("zones", [])],
                        source_shape=source_shape,
                        target_shape=tuple(frame.shape),
                    )
                depth_observations = (
                    _depth_observations_from_dict(source_entry.get("depth_observations", []))
                    if args.use_recorded_depth
                    else {}
                )
                timestamp = float(source_entry.get("relative_time", source_entry.get("timestamp", 0.0)) or 0.0)
                hits = detector.update(hands, zones, timestamp, depth_observations)
                diagnostics = detector.diagnostics()
                hit_count += len(hits)
                for diag in diagnostics:
                    reason_counts[str(diag.get("reason", "unknown"))] += 1

                out.write(
                    json.dumps(
                        {
                            "frame_index": frame_count,
                            "source_frame_index": source_entry.get("frame_index", frame_count),
                            "timestamp": timestamp,
                            "relative_time": timestamp,
                            "fps": source_entry.get("fps", 0.0),
                            "mode": source_entry.get("mode", ""),
                            "frame_shape": list(frame.shape),
                            "metrics": source_entry.get("metrics", {}),
                            "hands": [_hand_to_dict(hand) for hand in hands],
                            "zones": [_zone_to_dict(zone) for zone in zones],
                            "hits": [_hit_to_dict(hit) for hit in hits],
                            "diagnostics": diagnostics,
                            "depth_observations": source_entry.get("depth_observations", []) if args.use_recorded_depth else [],
                            "depth_status": source_entry.get("depth_status", ""),
                            "gesture": source_entry.get("gesture", ""),
                            "loop_state": source_entry.get("loop_state", ""),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                frame_count += 1
    finally:
        tracker.close()
        cap.release()

    summary = {
        "source_session": str(session_dir),
        "variant": args.variant,
        "frames": frame_count,
        "source_frames": len(source_frames),
        "shape_mismatch_frames": mismatch_count,
        "hits": hit_count,
        "reason_counts": dict(reason_counts),
        "config": {
            "FINGERTIP_REFINEMENT_ENABLED": args.fingertip_refinement,
            "FINGERTIP_OPTICAL_FLOW_STABILIZATION": config.FINGERTIP_OPTICAL_FLOW_STABILIZATION,
            "TRACKING_REACQUIRE_HIT_BLOCK_FRAMES": config.TRACKING_REACQUIRE_HIT_BLOCK_FRAMES,
            "TRACKING_FULL_MISS_REACQUIRE_HIT_BLOCK_FRAMES": config.TRACKING_FULL_MISS_REACQUIRE_HIT_BLOCK_FRAMES,
            "TRACKING_NEW_HAND_HIT_BLOCK_FRAMES": config.TRACKING_NEW_HAND_HIT_BLOCK_FRAMES,
            "TRACKING_MAX_WIDTH": args.tracking_max_width,
            "HAND_MODEL_COMPLEXITY": args.hand_model_complexity,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def parse_roi(value: object) -> Optional[tuple[float, float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def transform_zones(zones: Iterable[Zone], source_shape: tuple[int, ...], target_shape: tuple[int, ...]) -> list[Zone]:
    if len(source_shape) < 2 or len(target_shape) < 2 or source_shape[:2] == target_shape[:2]:
        return list(zones)
    src_h, src_w = int(source_shape[0]), int(source_shape[1])
    dst_h, dst_w = int(target_shape[0]), int(target_shape[1])
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return list(zones)
    scale = min(dst_w / src_w, dst_h / src_h)
    fit_w = max(1, int(round(src_w * scale)))
    fit_h = max(1, int(round(src_h * scale)))
    offset_x = (dst_w - fit_w) // 2
    offset_y = (dst_h - fit_h) // 2
    return [_transform_zone(zone, scale, offset_x, offset_y) for zone in zones]


def _transform_zone(zone: Zone, scale: float, offset_x: int, offset_y: int) -> Zone:
    polygon = None
    if zone.polygon:
        polygon = tuple(_transform_point(point, scale, offset_x, offset_y) for point in zone.polygon)
    x1, y1 = _transform_point((zone.x1, zone.y1), scale, offset_x, offset_y)
    x2, y2 = _transform_point((zone.x2, zone.y2), scale, offset_x, offset_y)
    return Zone(
        label=zone.label,
        sound_id=zone.sound_id,
        x1=min(x1, x2),
        y1=min(y1, y2),
        x2=max(x1, x2),
        y2=max(y1, y2),
        kind=zone.kind,
        press_ratio=zone.press_ratio,
        release_ratio=zone.release_ratio,
        polygon=polygon,
    )


def _transform_point(point: tuple[int, int], scale: float, offset_x: int, offset_y: int) -> tuple[int, int]:
    return (int(round(point[0] * scale + offset_x)), int(round(point[1] * scale + offset_y)))


def _hand_to_dict(hand: HandLandmarks) -> dict[str, Any]:
    return {
        "hand_id": hand.hand_id,
        "label": hand.label,
        "tracking_source": getattr(hand, "tracking_source", "mediapipe"),
        "missed_frames": getattr(hand, "missed_frames", 0),
        "unstable_landmark_ids": list(getattr(hand, "unstable_landmark_ids", ())),
        "landmarks": [[int(x), int(y), float(z)] for x, y, z in hand.landmarks],
        "normalized_landmarks": [[float(x), float(y), float(z)] for x, y, z in hand.normalized_landmarks],
    }


def _zone_to_dict(zone: Zone) -> dict[str, Any]:
    return {
        "label": zone.label,
        "sound_id": zone.sound_id,
        "kind": zone.kind,
        "x1": zone.x1,
        "y1": zone.y1,
        "x2": zone.x2,
        "y2": zone.y2,
        "press_y": zone.press_y,
        "release_y": zone.release_y,
        "polygon": zone.polygon,
    }


def _hit_to_dict(hit: HitEvent) -> dict[str, Any]:
    return {
        "note_id": hit.note_id,
        "sound_id": hit.sound_id,
        "zone_label": hit.zone_label,
        "finger_id": hit.finger_id,
        "hand_id": hit.hand_id,
        "timestamp": hit.timestamp,
        "velocity": hit.velocity,
        "volume": hit.volume,
    }


if __name__ == "__main__":
    raise SystemExit(main())
