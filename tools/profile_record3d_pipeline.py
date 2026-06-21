from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import config
from camera_utils import tracking_roi
from depth_contact import DepthContactEstimator
from gesture_recognizer import Gesture, GestureUpdate
from hand_tracker import HandTracker
from hit_detector import HitDetector
from instrument import InstrumentLayout
from loop_station import LoopStation
from rgbd_camera import AsyncRecord3DCamera, Record3DCamera, list_record3d_devices
from ui import draw_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the live Record3D -> tracking -> depth -> hit -> UI pipeline. "
            "Use this to find why the app is capped near 10 FPS."
        )
    )
    parser.add_argument("--frames", type=int, default=300, help="Measured frames after warmup")
    parser.add_argument("--warmup-frames", type=int, default=30, help="Frames to discard before measuring")
    parser.add_argument("--output", default="data/benchmarks/record3d_profile")
    parser.add_argument("--record3d-device", type=int, default=0)
    parser.add_argument("--record3d-timeout", type=float, default=2.0)
    parser.add_argument("--record3d-rotate", type=int, choices=[0, 90, 180, 270], default=0)
    parser.add_argument("--record3d-mirror", action="store_true")
    parser.add_argument("--record3d-depth-unit", choices=["auto", "m", "cm", "mm"], default="auto")
    parser.add_argument("--async-record3d", dest="async_record3d", action="store_true", default=True)
    parser.add_argument("--no-async-record3d", dest="async_record3d", action="store_false")
    parser.add_argument("--mode", choices=["piano", "drum"], default="piano")
    parser.add_argument("--instrument-roi", default="0.05,0.45,0.95,0.95")
    parser.add_argument("--paper-keyboard", action="store_true", default=True)
    parser.add_argument("--tracking-max-width", type=int, default=config.TRACKING_MAX_WIDTH)
    parser.add_argument("--tracking-roi-y", type=float, default=config.TRACKING_ROI_Y_MIN)
    parser.add_argument("--no-tracking-roi", action="store_true")
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--min-detection-confidence", type=float, default=config.HAND_MIN_DETECTION_CONFIDENCE)
    parser.add_argument("--min-tracking-confidence", type=float, default=config.HAND_MIN_TRACKING_CONFIDENCE)
    parser.add_argument("--no-landmark-smoothing", action="store_true")
    parser.add_argument("--landmark-smoothing-alpha", type=float, default=config.LANDMARK_SMOOTHING_ALPHA)
    parser.add_argument("--no-optical-stabilization", action="store_true")
    parser.add_argument("--fingertip-refinement", action="store_true")
    parser.add_argument("--piano-left-trim-keys", type=float, default=config.PIANO_LEFT_TRIM_KEYS)
    parser.add_argument("--piano-right-trim-keys", type=float, default=config.PIANO_RIGHT_TRIM_KEYS)
    parser.add_argument("--depth-contact-mode", choices=["off", "assist", "required"], default="off")
    parser.add_argument("--auto-depth-baseline", action="store_true")
    parser.add_argument("--depth-baseline-frames", type=int, default=config.DEPTH_BASELINE_FRAMES)
    parser.add_argument("--depth-min-confidence", type=float, default=config.DEPTH_MIN_CONFIDENCE)
    parser.add_argument("--skip-hand-tracking", action="store_true", help="Measure Record3D input and UI without MediaPipe")
    parser.add_argument("--skip-depth", action="store_true", help="Skip depth contact estimator")
    parser.add_argument("--skip-hit-detector", action="store_true")
    parser.add_argument("--draw", action="store_true", help="Include draw_scene cost")
    parser.add_argument("--display", action="store_true", help="Open a window and include imshow/waitKey cost")
    parser.add_argument("--display-scale", type=float, default=1.0)
    parser.add_argument("--window-width", type=int, default=0)
    parser.add_argument("--window-height", type=int, default=0)
    parser.add_argument("--report-every", type=int, default=60)
    parser.add_argument("--list-devices", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_devices:
        devices = list_record3d_devices()
        print("Record3D devices:")
        for index, device in enumerate(devices):
            print(f"  index {index}: {device}")
        if not devices:
            print("  none found")
        return 0

    apply_runtime_config(args)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    roi = parse_roi(args.instrument_roi)
    layout = InstrumentLayout(args.mode, roi_ratios=roi)
    tracker = None
    if not args.skip_hand_tracking:
        tracker = HandTracker(
            max_num_hands=args.max_hands,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            input_max_width=args.tracking_max_width,
            smooth_landmarks=not args.no_landmark_smoothing,
            smoothing_alpha=args.landmark_smoothing_alpha,
            refine_fingertips=args.fingertip_refinement,
        )
    depth_estimator = None
    if not args.skip_depth and args.depth_contact_mode != "off":
        depth_estimator = DepthContactEstimator(
            mode=args.depth_contact_mode,
            baseline_frames=args.depth_baseline_frames,
            min_confidence=args.depth_min_confidence,
        )
    hit_detector = None if args.skip_hit_detector else HitDetector()
    loop_station = LoopStation()
    gesture_update = GestureUpdate(gesture=Gesture.UNKNOWN)
    recent_hit = None
    highlights: dict[str, float] = {}
    window_name = "AirDesk Record3D Profiler"

    record3d_cls = AsyncRecord3DCamera if args.async_record3d else Record3DCamera
    cap = record3d_cls(
        device_index=args.record3d_device,
        timeout_seconds=args.record3d_timeout,
        rotate_degrees=args.record3d_rotate,
        mirror=args.record3d_mirror,
        depth_unit=args.record3d_depth_unit,
    )

    rows: list[dict[str, Any]] = []
    stats: dict[str, list[float]] = defaultdict(list)
    input_intervals = deque(maxlen=max(1, args.frames + args.warmup_frames))
    last_frame_timestamp: Optional[float] = None
    measured_start: Optional[float] = None
    measured_end: Optional[float] = None
    ok_count = 0
    timeout_count = 0

    try:
        if args.display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        target_total = max(0, args.warmup_frames) + max(1, args.frames)
        frame_index = 0
        while ok_count < target_total:
            loop_start = time.perf_counter()
            ok, rgbd_frame, read_timings = cap.read_profiled()
            if not ok or rgbd_frame is None:
                timeout_count += 1
                continue
            ok_count += 1
            if last_frame_timestamp is not None:
                input_intervals.append(max(0.0, rgbd_frame.timestamp - last_frame_timestamp))
            last_frame_timestamp = rgbd_frame.timestamp

            measuring = ok_count > args.warmup_frames
            if measuring and measured_start is None:
                measured_start = loop_start

            frame = rgbd_frame.color_bgr
            stage = dict(read_timings)
            hands = []
            hits = []
            zones = []

            timed(stage, "zones_ms", lambda: zones.extend(layout.get_zones(frame.shape)))

            def run_tracking() -> None:
                nonlocal hands
                if tracker is None:
                    hands = []
                    return
                hand_roi = None if args.no_tracking_roi else tracking_roi(frame.shape, args.tracking_roi_y)
                hands = tracker.process(frame, roi=hand_roi)

            timed(stage, "hand_tracking_ms", run_tracking)

            depth_observations = {}

            def run_depth() -> None:
                nonlocal depth_observations
                if depth_estimator is None:
                    depth_observations = {}
                    return
                if args.auto_depth_baseline and not depth_estimator.calibrated and not hands:
                    depth_estimator.calibrate(rgbd_frame, zones)
                depth_observations = depth_estimator.update(rgbd_frame, hands, zones)

            timed(stage, "depth_contact_ms", run_depth)

            def run_hit_detector() -> None:
                nonlocal hits
                if hit_detector is None:
                    hits = []
                    return
                hits = hit_detector.update(hands, zones, time.perf_counter(), depth_observations)

            timed(stage, "hit_detector_ms", run_hit_detector)

            display_frame = frame
            if args.draw or args.display:
                display_frame = frame.copy()

                def run_draw() -> None:
                    draw_scene(
                        frame=display_frame,
                        zones=zones,
                        hands=hands,
                        loop_station=loop_station,
                        mode=args.mode,
                        fps=input_fps(input_intervals),
                        gesture_update=gesture_update,
                        recent_hit=recent_hit,
                        highlights=highlights,
                        current_time=time.perf_counter(),
                        hit_detector=hit_detector or HitDetector(),
                        debug=True,
                        debug_lines=profile_debug_lines(stage),
                        draw_instrument_overlay=not args.paper_keyboard,
                    )

                timed(stage, "draw_scene_ms", run_draw)
            else:
                stage["draw_scene_ms"] = 0.0

            def run_display() -> None:
                if not args.display:
                    return
                output = display_frame
                if args.window_width > 0 and args.window_height > 0:
                    output = fit_frame_to_size(output, args.window_width, args.window_height)
                elif abs(args.display_scale - 1.0) > 1e-3:
                    output = cv2.resize(
                        output,
                        (
                            max(1, int(round(output.shape[1] * args.display_scale))),
                            max(1, int(round(output.shape[0] * args.display_scale))),
                        ),
                        interpolation=cv2.INTER_LINEAR,
                    )
                cv2.imshow(window_name, output)
                cv2.waitKey(1)

            timed(stage, "display_ms", run_display)

            stage["loop_total_ms"] = (time.perf_counter() - loop_start) * 1000.0
            if measuring:
                measured_end = time.perf_counter()
                row = {
                    "frame_index": frame_index,
                    "input_fps": input_fps(input_intervals),
                    "loop_fps_instant": 1000.0 / stage["loop_total_ms"] if stage["loop_total_ms"] > 0 else 0.0,
                    "hands": len(hands),
                    "hits": len(hits),
                    "color_width": int(frame.shape[1]),
                    "color_height": int(frame.shape[0]),
                    "depth_width": int(rgbd_frame.depth.shape[1]) if rgbd_frame.depth is not None else 0,
                    "depth_height": int(rgbd_frame.depth.shape[0]) if rgbd_frame.depth is not None else 0,
                    **stage,
                }
                rows.append(row)
                for key, value in stage.items():
                    stats[key].append(float(value))
                if args.report_every > 0 and len(rows) % args.report_every == 0:
                    print(live_report(len(rows), args.frames, rows, stats), flush=True)
                frame_index += 1
            elif ok_count == args.warmup_frames:
                print(f"Warmup complete ({args.warmup_frames} frame(s)); measuring {args.frames} frame(s)...")
    finally:
        if tracker is not None:
            tracker.close()
        cap.release()
        if args.display:
            cv2.destroyWindow(window_name)

    summary = build_summary(
        rows=rows,
        stats=stats,
        measured_start=measured_start,
        measured_end=measured_end,
        timeouts=timeout_count,
        args=args,
    )
    (output_dir / "record3d_pipeline_profile.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(output_dir / "record3d_pipeline_profile_frames.csv", rows)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output_dir / 'record3d_pipeline_profile.json'}")
    print(f"Wrote {output_dir / 'record3d_pipeline_profile_frames.csv'}")
    return 0


def timed(stage: dict[str, float], key: str, fn) -> None:
    start = time.perf_counter()
    fn()
    stage[key] = (time.perf_counter() - start) * 1000.0


def apply_runtime_config(args: argparse.Namespace) -> None:
    if args.no_optical_stabilization:
        config.FINGERTIP_OPTICAL_FLOW_STABILIZATION = False
    config.PIANO_LEFT_TRIM_KEYS = max(0.0, float(args.piano_left_trim_keys))
    config.PIANO_RIGHT_TRIM_KEYS = max(0.0, float(args.piano_right_trim_keys))


def build_summary(
    rows: list[dict[str, Any]],
    stats: dict[str, list[float]],
    measured_start: Optional[float],
    measured_end: Optional[float],
    timeouts: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    duration = max(1e-9, (measured_end or 0.0) - (measured_start or 0.0))
    stage_summary = {key: summarize_values(values) for key, values in sorted(stats.items())}
    bottlenecks = sorted(
        (
            {
                "stage": key,
                "mean_ms": value["mean_ms"],
                "p90_ms": value["p90_ms"],
                "share_of_loop": value["mean_ms"] / max(1e-9, stage_summary.get("loop_total_ms", {}).get("mean_ms", 0.0)),
            }
            for key, value in stage_summary.items()
            if key != "loop_total_ms" and key.endswith("_ms")
        ),
        key=lambda item: item["mean_ms"],
        reverse=True,
    )
    return {
        "frames": len(rows),
        "timeouts": timeouts,
        "measured_duration_s": duration,
        "effective_loop_fps": len(rows) / duration,
        "mean_input_fps": mean([float(row["input_fps"]) for row in rows]) if rows else 0.0,
        "mean_callback_fps": callback_fps(stage_summary),
        "mean_hands": mean([float(row["hands"]) for row in rows]) if rows else 0.0,
        "stage_summary": stage_summary,
        "bottlenecks": bottlenecks[:10],
        "args": vars(args),
        "diagnosis": diagnose(stage_summary, rows),
    }


def diagnose(stage_summary: dict[str, dict[str, float]], rows: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    loop_mean = stage_summary.get("loop_total_ms", {}).get("mean_ms", 0.0)
    wait_mean = stage_summary.get("record3d_wait_ms", {}).get("mean_ms", 0.0)
    tracking_mean = stage_summary.get("hand_tracking_ms", {}).get("mean_ms", 0.0)
    draw_mean = stage_summary.get("draw_scene_ms", {}).get("mean_ms", 0.0)
    display_mean = stage_summary.get("display_ms", {}).get("mean_ms", 0.0)
    async_age_mean = stage_summary.get("async_frame_age_ms", {}).get("mean_ms", 0.0)
    async_dropped_mean = stage_summary.get("async_dropped_frames", {}).get("mean_ms", 0.0)
    input_fps_mean = mean([float(row["input_fps"]) for row in rows]) if rows else 0.0
    callback_mean = stage_summary.get("record3d_callback_interval_ms", {}).get("mean_ms", 0.0)
    callback_fps_mean = 1000.0 / callback_mean if callback_mean > 0 else 0.0
    if wait_mean > 0.45 * max(1e-9, loop_mean):
        notes.append("Record3D frame wait dominates; input stream/callback cadence is likely the FPS cap.")
    if callback_fps_mean and callback_fps_mean < 20.0:
        notes.append("Record3D callback cadence is below 20 FPS; the device/app/USB stream is the bottleneck before our pipeline.")
    if input_fps_mean and input_fps_mean < 20.0:
        notes.append("Measured Record3D input FPS is below 20; verify Record3D app streaming mode, USB bandwidth, and device settings.")
    if tracking_mean > 0.35 * max(1e-9, loop_mean):
        notes.append("MediaPipe hand tracking is a major bottleneck; reduce --tracking-max-width or ROI/full-frame reacquire work.")
    if draw_mean + display_mean > 0.25 * max(1e-9, loop_mean):
        notes.append("Drawing/display is a major bottleneck; disable hand cutout/markers or use a smaller window.")
    if async_age_mean > 80.0:
        notes.append("Async Record3D frame age is high; the consumer loop is lagging behind the latest camera frame.")
    if async_dropped_mean >= 0.5:
        notes.append("Async Record3D is dropping old frames, which is good for latency but means downstream processing is slower than input.")
    if not notes:
        notes.append("No single dominant bottleneck found; inspect stage_summary p90/p99 for spikes.")
    return notes


def callback_fps(stage_summary: dict[str, dict[str, float]]) -> float:
    interval_ms = stage_summary.get("record3d_callback_interval_ms", {}).get("mean_ms", 0.0)
    if interval_ms <= 0:
        return 0.0
    return 1000.0 / interval_ms


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
    sorted_values = sorted(values)
    return {
        "mean_ms": float(mean(sorted_values)),
        "p50_ms": percentile(sorted_values, 0.50),
        "p90_ms": percentile(sorted_values, 0.90),
        "p99_ms": percentile(sorted_values, 0.99),
        "max_ms": float(sorted_values[-1]),
    }


def percentile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * ratio))))
    return float(sorted_values[index])


def input_fps(intervals: Iterable[float]) -> float:
    values = [value for value in intervals if value > 1e-9]
    if not values:
        return 0.0
    return 1.0 / mean(values)


def live_report(frame_count: int, target: int, rows: list[dict[str, Any]], stats: dict[str, list[float]]) -> str:
    loop_fps = mean([float(row["loop_fps_instant"]) for row in rows[-min(len(rows), 30):]])
    in_fps = mean([float(row["input_fps"]) for row in rows[-min(len(rows), 30):]])
    means = {key: mean(values[-min(len(values), 30):]) for key, values in stats.items() if values}
    top = sorted(((key, value) for key, value in means.items() if key != "loop_total_ms"), key=lambda item: item[1], reverse=True)[:4]
    top_text = " ".join(f"{key}={value:.1f}ms" for key, value in top)
    return f"{frame_count}/{target} loop_fps={loop_fps:.1f} input_fps={in_fps:.1f} {top_text}"


def profile_debug_lines(stage: dict[str, float]) -> list[str]:
    lines = [
        f"cam wait {stage.get('record3d_wait_ms', 0.0):.1f} ms",
        f"track {stage.get('hand_tracking_ms', 0.0):.1f} ms",
        f"depth {stage.get('depth_contact_ms', 0.0):.1f} ms",
        f"draw {stage.get('draw_scene_ms', 0.0):.1f} ms",
    ]
    if "async_frame_age_ms" in stage:
        lines.append(
            f"async age {stage.get('async_frame_age_ms', 0.0):.1f} ms "
            f"drop {stage.get('async_dropped_frames', 0.0):.0f}"
        )
    return lines


def parse_roi(raw: Optional[str]) -> Optional[tuple[float, float, float, float]]:
    if raw is None or raw == "":
        return None
    parts = [float(part.strip()) for part in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("--instrument-roi expects x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("--instrument-roi values must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    return (x1, y1, x2, y2)


def fit_frame_to_size(frame, width: int, height: int):
    src_h, src_w = frame.shape[:2]
    scale = min(width / src_w, height / src_h)
    fit_w = max(1, int(round(src_w * scale)))
    fit_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (fit_w, fit_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, frame.shape[2]), dtype=frame.dtype)
    x = (width - fit_w) // 2
    y = (height - fit_h) // 2
    canvas[y : y + fit_h, x : x + fit_w] = resized
    return canvas


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
