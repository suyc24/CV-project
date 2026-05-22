from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, Iterable, Optional

try:
    import cv2
except Exception as exc:  # pragma: no cover - depends on local install
    print(
        "Startup error: OpenCV could not be imported. Install dependencies with "
        "`pip install -r requirements.txt`.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

import numpy as np

import config
from audio_engine import AudioEngine
from camera_utils import (
    CameraSettings,
    calibrate_camera,
    configure_capture,
    create_capture,
    enhance_frame,
    list_cameras,
    load_camera_profile,
    measure_frame_quality,
    read_capture_settings,
    save_camera_profile,
    tracking_roi,
)
from depth_contact import DepthContactEstimator
from gesture_recognizer import Gesture, GestureController, GestureRecognizer, GestureUpdate
from hand_tracker import HandTracker
from hit_detector import HitDetector, HitEvent
from instrument import InstrumentLayout
from loop_station import LoopStation
from rgbd_camera import RGBDFrame, Record3DCamera, list_record3d_devices
from session_recorder import SessionRecorder
from ui import draw_scene
from utils import FPSCounter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AirDesk Instrument: monocular virtual desktop instrument")
    parser.add_argument("--camera-source", choices=["webcam", "record3d"], default="webcam", help="Video source")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--mode", choices=["drum", "piano"], default="drum", help="Instrument mode")
    parser.add_argument("--debug", action="store_true", help="Show velocity and state debug overlays")
    parser.add_argument("--display-scale", type=float, default=config.DISPLAY_SCALE, help="Scale factor for the OpenCV display window")
    parser.add_argument("--window-width", type=int, default=config.DISPLAY_WINDOW_WIDTH, help="Optional display window width")
    parser.add_argument("--window-height", type=int, default=config.DISPLAY_WINDOW_HEIGHT, help="Optional display window height")
    parser.add_argument("--fullscreen", action="store_true", help="Start in fullscreen display mode")
    parser.add_argument("--air-test", action="store_true", help="Move instrument ROI upward for testing with a laptop camera")
    parser.add_argument(
        "--paper-keyboard",
        action="store_true",
        help=(
            "Use a real paper/iPad keyboard as the visible instrument. "
            "Keeps piano zones active without requiring Record3D depth calibration "
            "and hides the virtual keybed overlay by default."
        ),
    )
    parser.add_argument(
        "--hide-instrument-overlay",
        action="store_true",
        help="Do not draw virtual pads/keys. Detection zones remain active and recorded.",
    )
    parser.add_argument("--instrument-roi", default=None, help="ROI ratios x1,y1,x2,y2 for pads/keys, e.g. 0.05,0.25,0.95,0.85")
    parser.add_argument("--list-cameras", action="store_true", help="List visible camera devices and exit")
    parser.add_argument("--calibrate-camera", action="store_true", help="Auto-tune exposure/focus and save camera_profile.json")
    parser.add_argument("--camera-profile", default=config.CAMERA_PROFILE_PATH, help="Camera profile JSON path")
    parser.add_argument("--no-camera-profile", action="store_true", help="Do not load saved camera profile")
    parser.add_argument("--show-camera-profile", action="store_true", help="Print effective camera settings and exit")
    parser.add_argument(
        "--quality",
        choices=["fast", "balanced", "high", "max"],
        default=None,
        help="Convenience preset for capture resolution and MediaPipe input size",
    )
    parser.add_argument("--backend", choices=["auto", "dshow", "msmf", "v4l2"], default="auto", help="OpenCV camera backend")
    parser.add_argument("--width", type=int, default=config.CAMERA_WIDTH, help="Requested camera width")
    parser.add_argument("--height", type=int, default=config.CAMERA_HEIGHT, help="Requested camera height")
    parser.add_argument("--fps", type=int, default=config.CAMERA_FPS, help="Requested camera FPS")
    parser.add_argument("--auto-exposure", dest="auto_exposure", action="store_true", help="Enable camera auto exposure")
    parser.add_argument("--manual-exposure", dest="auto_exposure", action="store_false", help="Disable camera auto exposure")
    parser.set_defaults(auto_exposure=config.CAMERA_AUTO_EXPOSURE)
    parser.add_argument("--exposure", type=float, default=config.CAMERA_EXPOSURE, help="Manual exposure value, e.g. -4 to -8 on many Windows webcams")
    parser.add_argument("--brightness", type=float, default=None, help="Optional camera brightness override")
    parser.add_argument("--contrast", type=float, default=None, help="Optional camera contrast override")
    parser.add_argument("--gain", type=float, default=None, help="Optional camera gain override")
    parser.add_argument("--autofocus", dest="autofocus", action="store_true", help="Enable camera autofocus when supported")
    parser.add_argument("--no-autofocus", dest="autofocus", action="store_false", help="Disable camera autofocus when supported")
    parser.set_defaults(autofocus=None)
    parser.add_argument("--focus", type=float, default=None, help="Optional manual focus value when supported")
    parser.add_argument("--enhance", choices=["auto", "clahe", "none"], default="none", help="Software contrast/exposure enhancement")
    parser.add_argument("--camera-warmup-frames", type=int, default=12, help="Frames to discard before reapplying camera settings")
    parser.add_argument("--no-reapply-camera-settings", action="store_true", help="Do not reapply camera settings after warmup")
    parser.add_argument("--no-tracking-roi", action="store_true", help="Run hand tracking on the whole frame")
    parser.add_argument("--tracking-roi-y", type=float, default=config.TRACKING_ROI_Y_MIN, help="Top y ratio for hand-tracking ROI")
    parser.add_argument("--tracking-max-width", type=int, default=config.TRACKING_MAX_WIDTH, help="Max width passed to MediaPipe")
    parser.add_argument("--no-landmark-smoothing", action="store_true", help="Disable landmark temporal smoothing")
    parser.add_argument("--landmark-smoothing-alpha", type=float, default=config.LANDMARK_SMOOTHING_ALPHA, help="Landmark smoothing alpha")
    parser.add_argument("--no-optical-stabilization", action="store_true", help="Disable optical-flow fingertip stabilization")
    parser.add_argument("--max-hands", type=int, default=2, help="Maximum number of hands to track")
    parser.add_argument("--no-hand-cutout", action="store_true", help="Do not composite the real hand above the piano layer")
    parser.add_argument("--no-fingertip-markers", action="store_true", help="Hide fingertip marker dots")
    parser.add_argument("--trigger-thumb", action="store_true", help="Allow thumb tips to trigger notes; enabled by default")
    parser.add_argument("--no-trigger-thumb", action="store_true", help="Disable thumb note triggers for unstable camera angles")
    parser.add_argument("--min-detection-confidence", type=float, default=config.HAND_MIN_DETECTION_CONFIDENCE, help="MediaPipe hand detection confidence")
    parser.add_argument("--min-tracking-confidence", type=float, default=config.HAND_MIN_TRACKING_CONFIDENCE, help="MediaPipe hand tracking confidence")
    parser.add_argument(
        "--piano-sensitivity",
        choices=["stable", "balanced", "sensitive"],
        default="stable",
        help="Piano hit tuning preset. stable reduces jitter triggers; sensitive accepts softer taps.",
    )
    parser.add_argument(
        "--calibration-exposures",
        default=",".join(str(value) for value in config.CALIBRATION_EXPOSURES),
        help="Comma-separated exposure values for --calibrate-camera, e.g. -4,-5,-6,-7,-8",
    )
    parser.add_argument(
        "--calibration-resolutions",
        default=",".join(f"{w}x{h}" for w, h in config.CALIBRATION_RESOLUTIONS),
        help="Comma-separated resolutions for --calibrate-camera, e.g. 640x480,1280x720,1920x1080",
    )
    parser.add_argument("--calibrate-focus", action="store_true", help="Also sweep manual focus values during calibration")
    parser.add_argument(
        "--calibration-focus-values",
        default=",".join(str(value) for value in config.CALIBRATION_FOCUS_VALUES),
        help="Comma-separated focus values used with --calibrate-focus",
    )
    parser.add_argument("--no-calibration-preview", action="store_true", help="Hide calibration preview window")
    parser.add_argument("--record-session", default=None, help="Directory to save replayable session data")
    parser.add_argument("--no-record-video", action="store_true", help="Record JSONL landmarks/diagnostics without AVI video")
    parser.add_argument("--record3d-device", type=int, default=0, help="Record3D device index")
    parser.add_argument("--record3d-timeout", type=float, default=2.0, help="Seconds to wait for each Record3D frame")
    parser.add_argument("--record3d-rotate", type=int, choices=[0, 90, 180, 270], default=0, help="Rotate Record3D frames")
    parser.add_argument("--record3d-mirror", action="store_true", help="Mirror Record3D frames horizontally")
    parser.add_argument("--record3d-depth-unit", choices=["auto", "m", "cm", "mm"], default="auto", help="Record3D depth unit")
    parser.add_argument("--depth-contact-mode", choices=["auto", "off", "assist", "required"], default="auto", help="How RGB-D contact evidence gates piano hits")
    parser.add_argument("--depth-contact-threshold", type=float, default=config.DEPTH_CONTACT_THRESHOLD_M, help="Meters above calibrated desk considered contact")
    parser.add_argument("--depth-release-threshold", type=float, default=config.DEPTH_RELEASE_THRESHOLD_M, help="Meters above calibrated desk considered definitely in air")
    parser.add_argument("--depth-baseline-frames", type=int, default=config.DEPTH_BASELINE_FRAMES, help="Frames used for desk depth calibration")
    parser.add_argument("--depth-min-confidence", type=float, default=config.DEPTH_MIN_CONFIDENCE, help="Minimum Record3D confidence for contact gating")
    parser.add_argument("--auto-depth-baseline", action="store_true", help="Continuously build a depth baseline while uncalibrated")
    return parser.parse_args()


def open_camera(camera_index: int, settings: CameraSettings, warmup_frames: int = 12, reapply: bool = True) -> cv2.VideoCapture:
    cap = create_capture(camera_index, settings.backend)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. Run `python main.py --list-cameras` "
            "and try a visible camera index/backend."
        )
    actual = configure_capture(cap, settings)
    for _ in range(max(0, warmup_frames)):
        cap.read()
    if reapply:
        actual = configure_capture(cap, settings)
        for _ in range(max(0, warmup_frames // 2)):
            cap.read()
        actual = read_capture_settings(cap)
    print(
        "Camera:",
        f"requested={settings.width}x{settings.height}",
        f"{int(actual['width'])}x{int(actual['height'])}",
        f"fps={actual['fps']:.1f}",
        f"exposure={actual['exposure']:.2f}",
        f"brightness={actual['brightness']:.2f}",
        f"contrast={actual['contrast']:.2f}",
        f"gain={actual['gain']:.2f}",
        f"focus={actual['focus']:.2f}",
        f"auto_exp={actual['auto_exposure']:.2f}",
    )
    return cap


def main() -> int:
    args = parse_args()
    apply_runtime_tracking_config(args)
    camera_settings = build_camera_settings(args) if args.camera_source == "webcam" else CameraSettings()
    try:
        instrument_roi = parse_roi(args.instrument_roi)
    except ValueError as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 1
    if args.air_test and instrument_roi is None:
        instrument_roi = (0.05, 0.25, 0.95, 0.85)
        args.tracking_roi_y = 0.0
    if args.show_camera_profile:
        if args.camera_source != "webcam":
            print("Record3D source does not use OpenCV camera profiles.")
            return 0
        print_camera_settings("Effective camera settings", camera_settings)
        return 0

    if args.list_cameras:
        if args.camera_source == "record3d":
            devices = list_record3d_devices()
            print("Record3D devices:")
            if devices:
                for index, device in enumerate(devices):
                    print(f"  index {index}: {device}")
            else:
                print("  none found")
        else:
            list_cameras(backend=args.backend)
        return 0

    if args.calibrate_camera:
        if args.camera_source != "webcam":
            print("Calibration error: --calibrate-camera is only for OpenCV webcam sources.", file=sys.stderr)
            return 1
        try:
            result = calibrate_camera(
                camera_index=args.camera,
                base_settings=camera_settings,
                profile_path=args.camera_profile,
                exposures=parse_number_list(args.calibration_exposures),
                resolutions=parse_resolution_list(args.calibration_resolutions),
                focus_values=parse_number_list(args.calibration_focus_values) if args.calibrate_focus else None,
                show_preview=not args.no_calibration_preview,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"Calibration error: {exc}", file=sys.stderr)
            return 1
        print("Calibration saved:", result.profile_path)
        print(
            "Best:",
            f"exposure={result.settings.exposure}",
            f"gain={result.settings.gain}",
            f"focus={result.settings.focus}",
            f"resolution={result.settings.width}x{result.settings.height}",
            f"score={result.metrics.score:.2f}",
            f"sharpness={result.metrics.sharpness:.1f}",
            f"overexposed={result.metrics.overexposed_ratio * 100:.1f}%",
        )
        print(f"Run: python main.py --camera {args.camera} --backend {result.settings.backend} --mode {args.mode}")
        return 0

    try:
        if args.camera_source == "record3d":
            cap = Record3DCamera(
                device_index=args.record3d_device,
                timeout_seconds=args.record3d_timeout,
                rotate_degrees=args.record3d_rotate,
                mirror=args.record3d_mirror,
                depth_unit=args.record3d_depth_unit,
            )
        else:
            cap = open_camera(
                args.camera,
                camera_settings,
                warmup_frames=args.camera_warmup_frames,
                reapply=not args.no_reapply_camera_settings,
            )
        audio_engine = AudioEngine()
        hand_tracker = HandTracker(
            max_num_hands=args.max_hands,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            input_max_width=args.tracking_max_width,
            smooth_landmarks=not args.no_landmark_smoothing,
            smoothing_alpha=args.landmark_smoothing_alpha,
        )
    except RuntimeError as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        return 1

    config.HAND_CUTOUT_ENABLED = config.HAND_CUTOUT_ENABLED and not args.no_hand_cutout
    config.SHOW_FINGERTIP_MARKERS = config.SHOW_FINGERTIP_MARKERS and not args.no_fingertip_markers

    layout = InstrumentLayout(args.mode, roi_ratios=instrument_roi)
    trigger_fingers = tuple(finger_id for finger_id in config.TRIGGER_FINGER_IDS if finger_id != 4) if args.no_trigger_thumb else config.TRIGGER_FINGER_IDS
    hit_detector = HitDetector(finger_ids=trigger_fingers)
    loop_station = LoopStation()
    gesture_recognizer = GestureRecognizer()
    gesture_controller = GestureController()
    fps_counter = FPSCounter()
    depth_estimator = DepthContactEstimator(
        mode=config.DEPTH_CONTACT_MODE,
        contact_threshold_m=args.depth_contact_threshold,
        release_threshold_m=args.depth_release_threshold,
        baseline_frames=args.depth_baseline_frames,
        min_confidence=args.depth_min_confidence,
    ) if config.DEPTH_CONTACT_MODE != "off" else None

    recent_hit: Optional[HitEvent] = None
    highlights: Dict[str, float] = {}
    gesture_update = GestureUpdate(gesture=Gesture.UNKNOWN)
    window_name = "AirDesk Instrument"
    display_scale = max(0.25, float(args.display_scale))
    fullscreen = bool(args.fullscreen)
    last_window_image_size: Optional[tuple[int, int]] = None
    webcam_rotation = 0
    depth_calibration_frames_remaining = 0
    frame_metrics_text = ""
    last_debug_metrics_time = 0.0
    recorder: Optional[SessionRecorder] = None
    if args.record_session:
        recorder = SessionRecorder(
            output_dir=args.record_session,
            metadata={
                "args": vars(args),
                "camera_settings": camera_settings,
                "camera_source": args.camera_source,
                "instrument_roi": instrument_roi,
                "mode": layout.mode,
                "depth_contact_mode": config.DEPTH_CONTACT_MODE,
            },
            record_video=not args.no_record_video,
            fps=camera_settings.fps,
        )
        print(f"Recording session to {args.record_session}")

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        _configure_window(window_name, fullscreen, args.window_width, args.window_height)
        while True:
            rgbd_frame: Optional[RGBDFrame] = None
            if args.camera_source == "record3d":
                ok, rgbd_frame = cap.read()
                frame = rgbd_frame.color_bgr if ok and rgbd_frame is not None else None
            else:
                ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera frame read failed. Exiting.", file=sys.stderr)
                break

            if args.camera_source == "webcam":
                frame = cv2.flip(frame, 1)
                frame = _rotate_frame(frame, webcam_rotation)
            current_time = time.perf_counter()
            fps = fps_counter.update()
            metrics = None
            should_sample_debug_metrics = (
                args.debug
                and current_time - last_debug_metrics_time >= config.DEBUG_METRICS_INTERVAL
            )
            if recorder is not None or should_sample_debug_metrics:
                metrics = measure_frame_quality(frame)
                if should_sample_debug_metrics:
                    last_debug_metrics_time = current_time
            if args.debug and metrics is not None:
                frame_metrics_text = (
                    f"Frame: luma={metrics.mean_luma:.0f} "
                    f"over={metrics.overexposed_ratio * 100:.1f}% "
                    f"sharp={metrics.sharpness:.0f}"
                )
            display_frame = enhance_frame(frame, args.enhance)

            hand_roi = None if args.no_tracking_roi else tracking_roi(frame.shape, args.tracking_roi_y)
            hands = hand_tracker.process(frame, roi=hand_roi)
            provisional_zones = layout.get_zones(display_frame.shape)
            if depth_calibration_frames_remaining > 0 and depth_estimator is not None and rgbd_frame is not None:
                depth_estimator.calibrate(rgbd_frame, provisional_zones)
                depth_calibration_frames_remaining -= 1
                if depth_calibration_frames_remaining == 0:
                    quad = depth_estimator.estimate_piano_quad(display_frame.shape)
                    if quad is not None:
                        layout.set_piano_quad(quad)
                        hit_detector.reset()
                        highlights.clear()
                        print(f"Depth calibration complete. Piano plane: {quad}")
                    else:
                        print("Depth calibration complete, but piano plane estimation failed. Using fallback keybed.")
            hide_piano_until_calibrated = (
                args.camera_source == "record3d"
                and layout.mode == "piano"
                and not args.paper_keyboard
                and depth_estimator is not None
                and (not depth_estimator.calibrated or depth_calibration_frames_remaining > 0)
            )
            zones = [] if hide_piano_until_calibrated else layout.get_zones(display_frame.shape)
            if depth_estimator is not None and args.auto_depth_baseline and not hands and not depth_estimator.calibrated and rgbd_frame is not None:
                depth_estimator.calibrate(rgbd_frame, provisional_zones)
            depth_observations = depth_estimator.update(rgbd_frame, hands, zones) if depth_estimator is not None else {}

            gesture_hands = [hand for hand in hands if getattr(hand, "tracking_source", "mediapipe") != "optical_flow"]
            if gesture_hands:
                gesture = gesture_recognizer.recognize(gesture_hands[0])
                gesture_update = gesture_controller.update(gesture, current_time, loop_station)
            else:
                gesture_update = gesture_controller.update(Gesture.UNKNOWN, current_time, loop_station)

            hits = hit_detector.update(hands, zones, current_time, depth_observations=depth_observations)
            diagnostics = hit_detector.diagnostics()
            for hit in hits:
                audio_engine.play(hit.sound_id, hit.volume)
                loop_station.record_event(hit)
                recent_hit = hit
                highlights[hit.sound_id] = current_time + config.HIGHLIGHT_SECONDS

            loop_station.update(current_time, audio_engine)
            _expire_highlights(highlights, current_time)

            if recorder is not None:
                recorder.record_frame(
                    frame=frame,
                    timestamp=current_time,
                    fps=fps,
                    metrics=metrics,
                    hands=hands,
                    zones=zones,
                    hits=hits,
                    diagnostics=diagnostics,
                    depth_observations=depth_observations,
                    depth_status=depth_estimator.status_text() if depth_estimator is not None else "Depth: off",
                    gesture=gesture_update.gesture.value,
                    loop_state=loop_station.state.value,
                    mode=layout.mode,
                )

            draw_scene(
                frame=display_frame,
                zones=zones,
                hands=hands,
                loop_station=loop_station,
                mode=layout.mode,
                fps=fps,
                gesture_update=gesture_update,
                recent_hit=recent_hit,
                highlights=highlights,
                current_time=current_time,
                hit_detector=hit_detector,
                debug=args.debug,
                debug_lines=_debug_lines(frame_metrics_text, depth_estimator, depth_calibration_frames_remaining),
                draw_instrument_overlay=not (args.hide_instrument_overlay or args.paper_keyboard),
            )

            output_frame = _display_frame(display_frame, display_scale, args.window_width, args.window_height)
            output_size = (output_frame.shape[1], output_frame.shape[0])
            if (
                not fullscreen
                and args.window_width <= 0
                and args.window_height <= 0
                and output_size != last_window_image_size
            ):
                cv2.resizeWindow(window_name, output_size[0], output_size[1])
                last_window_image_size = output_size
            cv2.imshow(window_name, output_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                loop_station.clear()
            elif key == ord(" "):
                loop_station.toggle_playback(current_time)
            elif key == ord("m"):
                layout.toggle_mode()
                hit_detector.reset()
                highlights.clear()
                if layout.mode == "piano" and depth_estimator is not None and not depth_estimator.calibrated:
                    layout.set_piano_quad(None)
            elif key == ord("e"):
                loop_station.toggle_recording(current_time)
            elif key == ord("d"):
                if depth_estimator is None or rgbd_frame is None:
                    print("Depth calibration unavailable for this camera source.")
                else:
                    depth_estimator.reset()
                    layout.set_piano_quad(None)
                    hit_detector.reset()
                    depth_calibration_frames_remaining = max(1, int(args.depth_baseline_frames))
                    print(f"Depth desk calibration started: collecting {depth_calibration_frames_remaining} frame(s).")
            elif key == ord("o"):
                if args.camera_source == "record3d":
                    rotation = cap.rotate_clockwise()
                    print(f"Record3D rotation: {rotation} degrees")
                else:
                    webcam_rotation = (webcam_rotation + 90) % 360
                    print(f"Webcam rotation: {webcam_rotation} degrees")
                hand_tracker.reset()
                hit_detector.reset()
                highlights.clear()
                if depth_estimator is not None:
                    depth_estimator.reset()
                    layout.set_piano_quad(None)
                    depth_calibration_frames_remaining = 0
            elif key in {ord("="), ord("+")}:
                display_scale = min(4.0, display_scale + 0.1)
                last_window_image_size = None
                print(f"Display scale: {display_scale:.2f}")
            elif key in {ord("-"), ord("_")}:
                display_scale = max(0.35, display_scale - 0.1)
                last_window_image_size = None
                print(f"Display scale: {display_scale:.2f}")
            elif key == ord("f"):
                fullscreen = not fullscreen
                _configure_window(window_name, fullscreen, args.window_width, args.window_height)
                last_window_image_size = None
            elif key == ord("["):
                if args.camera_source == "webcam":
                    adjust_exposure(cap, camera_settings, -1.0)
            elif key == ord("]"):
                if args.camera_source == "webcam":
                    adjust_exposure(cap, camera_settings, 1.0)
            elif key == ord("a"):
                if args.camera_source != "webcam":
                    continue
                camera_settings.auto_exposure = not camera_settings.auto_exposure
                actual = configure_capture(cap, camera_settings)
                print(f"Auto exposure={camera_settings.auto_exposure} actual={actual['auto_exposure']:.2f}")
            elif key == ord("p"):
                if args.camera_source != "webcam":
                    continue
                metrics = measure_frame_quality(frame)
                save_camera_profile(args.camera_profile, camera_settings, metrics)
                print(
                    f"Saved camera profile to {args.camera_profile}: "
                    f"exposure={camera_settings.exposure} over={metrics.overexposed_ratio * 100:.1f}%"
                )

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        hand_tracker.close()
        audio_engine.close()
        if recorder is not None:
            recorder.close()
        cv2.destroyAllWindows()
    return 0


def build_camera_settings(args: argparse.Namespace) -> CameraSettings:
    settings = CameraSettings()
    if not args.no_camera_profile:
        profile = load_camera_profile(args.camera_profile)
        if profile:
            apply_camera_profile(settings, profile)
            print(f"Loaded camera profile: {args.camera_profile}")
            print_camera_settings("Profile settings", settings)
            if settings.width <= 640 or settings.height <= 480:
                print(
                    "Profile hint: this profile uses a low capture resolution. "
                    "Re-run `--calibrate-camera` or add `--quality balanced/high` to override it."
                )

    if args.quality:
        apply_quality_preset(settings, args.quality)

    overrides = {
        "backend": ("backend",),
        "width": ("width",),
        "height": ("height",),
        "fps": ("fps",),
        "exposure": ("exposure",),
        "brightness": ("brightness",),
        "contrast": ("contrast",),
        "gain": ("gain",),
        "focus": ("focus",),
    }
    for attr, flags in overrides.items():
        if arg_was_provided(flags):
            setattr(settings, attr, getattr(args, attr))
    if arg_was_provided(("--auto-exposure", "--manual-exposure")):
        settings.auto_exposure = args.auto_exposure
    if arg_was_provided(("--autofocus", "--no-autofocus")):
        settings.autofocus = args.autofocus
    return settings


def apply_quality_preset(settings: CameraSettings, preset: str) -> None:
    if preset == "fast":
        settings.width, settings.height, settings.fps = 640, 480, 30
    elif preset == "balanced":
        settings.width, settings.height, settings.fps = 1280, 720, 30
    elif preset == "high":
        settings.width, settings.height, settings.fps = 1920, 1080, 30
    elif preset == "max":
        settings.width, settings.height, settings.fps = 1920, 1080, 60


def apply_runtime_tracking_config(args: argparse.Namespace) -> None:
    if args.no_optical_stabilization:
        config.FINGERTIP_OPTICAL_FLOW_STABILIZATION = False

    if args.piano_sensitivity == "sensitive":
        config.PIANO_STRIKE_MIN_DROP_PX = 14.0
        config.PIANO_STRIKE_MIN_VELOCITY = 90.0
        config.PIANO_STRIKE_MIN_NET_DROP_PX = 10.0
        config.PIANO_RELEASE_MIN_UP_VELOCITY = 10.0
        config.PIANO_RELEASE_STRONG_LIFT_MULTIPLIER = 1.25
    elif args.piano_sensitivity == "balanced":
        config.PIANO_STRIKE_MIN_DROP_PX = 16.0
        config.PIANO_STRIKE_MIN_VELOCITY = 100.0
        config.PIANO_STRIKE_MIN_NET_DROP_PX = 12.0
        config.PIANO_RELEASE_MIN_UP_VELOCITY = 14.0
        config.PIANO_RELEASE_STRONG_LIFT_MULTIPLIER = 1.4

    if args.paper_keyboard and args.depth_contact_mode == "auto":
        config.DEPTH_CONTACT_MODE = "off"
    elif args.depth_contact_mode == "auto":
        config.DEPTH_CONTACT_MODE = "assist" if args.camera_source == "record3d" else "off"
    else:
        config.DEPTH_CONTACT_MODE = args.depth_contact_mode
    config.DEPTH_CONTACT_THRESHOLD_M = args.depth_contact_threshold
    config.DEPTH_RELEASE_THRESHOLD_M = args.depth_release_threshold
    config.DEPTH_BASELINE_FRAMES = args.depth_baseline_frames
    config.DEPTH_MIN_CONFIDENCE = args.depth_min_confidence
    config.DEPTH_AUTO_BASELINE_WHEN_UNCALIBRATED = args.auto_depth_baseline


def print_camera_settings(label: str, settings: CameraSettings) -> None:
    print(
        f"{label}:",
        f"backend={settings.backend}",
        f"resolution={settings.width}x{settings.height}",
        f"fps={settings.fps}",
        f"auto_exposure={settings.auto_exposure}",
        f"exposure={settings.exposure}",
        f"gain={settings.gain}",
        f"brightness={settings.brightness}",
        f"contrast={settings.contrast}",
        f"autofocus={settings.autofocus}",
        f"focus={settings.focus}",
    )


def adjust_exposure(cap, settings: CameraSettings, delta: float) -> None:
    current = settings.exposure if settings.exposure is not None else 0.0
    settings.auto_exposure = False
    settings.exposure = float(current + delta)
    actual = configure_capture(cap, settings)
    print(
        "Exposure adjusted:",
        f"requested={settings.exposure}",
        f"actual={actual['exposure']:.2f}",
        f"auto_exp={actual['auto_exposure']:.2f}",
    )


def apply_camera_profile(settings: CameraSettings, profile: Dict[str, object]) -> None:
    for key, value in profile.items():
        if hasattr(settings, key):
            setattr(settings, key, value)


def arg_was_provided(dest_or_flags: Iterable[str]) -> bool:
    names = set(dest_or_flags)
    flag_names = set()
    for name in names:
        if name.startswith("--"):
            flag_names.add(name)
        else:
            flag_names.add(f"--{name.replace('_', '-')}")
    for raw in sys.argv[1:]:
        flag = raw.split("=", 1)[0]
        if flag in flag_names:
            return True
    return False


def parse_number_list(raw: str) -> list[float]:
    values = []
    for piece in raw.split(","):
        piece = piece.strip()
        if piece:
            values.append(float(piece))
    return values


def parse_resolution_list(raw: str) -> list[tuple[int, int]]:
    values = []
    for piece in raw.split(","):
        piece = piece.strip().lower()
        if not piece:
            continue
        if "x" not in piece:
            raise ValueError(f"Invalid resolution: {piece}")
        width, height = piece.split("x", 1)
        values.append((int(width), int(height)))
    return values


def parse_roi(raw: Optional[str]) -> Optional[tuple[float, float, float, float]]:
    if not raw:
        return None
    parts = [float(piece.strip()) for piece in raw.split(",") if piece.strip()]
    if len(parts) != 4:
        raise ValueError("--instrument-roi expects four comma-separated ratios: x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("--instrument-roi values must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    return (x1, y1, x2, y2)


def _configure_window(window_name: str, fullscreen: bool, width: int, height: int) -> None:
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        return
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    if width > 0 and height > 0:
        cv2.resizeWindow(window_name, width, height)


def _display_frame(frame, scale: float, width: int, height: int):
    if width > 0 and height > 0:
        return _fit_frame_to_size(frame, width, height)
    if abs(scale - 1.0) < 1e-3:
        return frame
    target = (
        max(1, int(round(frame.shape[1] * scale))),
        max(1, int(round(frame.shape[0] * scale))),
    )
    return cv2.resize(frame, target, interpolation=cv2.INTER_LINEAR)


def _fit_frame_to_size(frame, width: int, height: int):
    """Resize without changing aspect ratio, then pad to the requested canvas."""
    src_h, src_w = frame.shape[:2]
    if src_w <= 0 or src_h <= 0 or width <= 0 or height <= 0:
        return frame
    scale = min(width / src_w, height / src_h)
    fit_w = max(1, int(round(src_w * scale)))
    fit_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (fit_w, fit_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, frame.shape[2]), dtype=frame.dtype)
    x = (width - fit_w) // 2
    y = (height - fit_h) // 2
    canvas[y : y + fit_h, x : x + fit_w] = resized
    return canvas


def _rotate_frame(frame, rotation_degrees: int):
    rotation = rotation_degrees % 360
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _debug_lines(
    frame_metrics_text: str,
    depth_estimator: Optional[DepthContactEstimator],
    depth_calibration_frames_remaining: int = 0,
) -> list[str]:
    lines = []
    if frame_metrics_text:
        lines.append(frame_metrics_text)
    if depth_estimator is not None:
        if depth_calibration_frames_remaining > 0:
            lines.append(f"Depth: calibrating {depth_calibration_frames_remaining} frame(s)")
        lines.append(depth_estimator.status_text())
    return lines


def _expire_highlights(highlights: Dict[str, float], current_time: float) -> None:
    expired = [sound_id for sound_id, until in highlights.items() if until <= current_time]
    for sound_id in expired:
        del highlights[sound_id]


if __name__ == "__main__":
    raise SystemExit(main())
