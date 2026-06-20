from __future__ import annotations

"""Decompose fingertip jitter by filter stage on a recorded raw_video.avi.

Re-runs the current HandTracker over a recorded clip in several filter
configurations (raw MediaPipe -> +EMA -> +EMA+optical-flow -> full incl.
One-Euro) and reports, per fingertip, how much frame-to-frame jitter each
stage leaves behind. On a *resting* clip the true hand motion is ~0, so the
remaining frame-to-frame displacement is almost entirely tracker noise --
this attributes the jitter the user sees to MediaPipe vs. each filter.

Caveat: the One-Euro stage in HandTracker keys off time.perf_counter(), so its
behaviour offline (process() called back-to-back) differs from a live ~12-20
FPS run. Treat the 'full' column's One-Euro contribution as indicative only;
the raw / +ema / +ema+flow columns are frame-indexed and faithful.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import config
from camera_utils import tracking_roi
from hand_tracker import HandTracker

FINGER_IDS = (4, 8, 12, 16, 20)
FINGER_NAMES = {4: "thumb", 8: "index", 12: "middle", 16: "ring", 20: "pinky"}

# (label, smooth_ema, optical_flow, one_euro)
STAGES: Tuple[Tuple[str, bool, bool, bool], ...] = (
    ("raw_mediapipe", False, False, False),
    ("+ema", True, False, False),
    ("+ema+flow", True, True, False),
    ("full(+oneeuro)", True, True, True),
)


def _pick_hand(hands) -> Optional[object]:
    if not hands:
        return None
    # Single-hand benchmark clips: take the hand with the most spread (largest
    # wrist-to-middle-tip distance), which is the real hand rather than a stub.
    best = None
    best_span = -1.0
    for hand in hands:
        lm = hand.landmarks
        if len(lm) <= 12:
            continue
        span = abs(lm[12][0] - lm[0][0]) + abs(lm[12][1] - lm[0][1])
        if span > best_span:
            best_span = span
            best = hand
    return best or hands[0]


def _run_stage(
    video_path: Path,
    smooth_ema: bool,
    optical_flow: bool,
    one_euro: bool,
    roi_y: float,
    max_width: int,
) -> Dict[int, List[Tuple[float, float]]]:
    config.FINGERTIP_OPTICAL_FLOW_STABILIZATION = optical_flow
    tracker = HandTracker(
        max_num_hands=2,
        min_detection_confidence=config.HAND_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.HAND_MIN_TRACKING_CONFIDENCE,
        input_max_width=max_width,
        model_complexity=config.HAND_MODEL_COMPLEXITY,
        smooth_landmarks=smooth_ema,
        fingertip_one_euro_enabled=one_euro,
    )
    series: Dict[int, List[Tuple[float, float]]] = {fid: [] for fid in FINGER_IDS}
    cap = cv2.VideoCapture(str(video_path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            roi = tracking_roi(frame.shape, roi_y)
            hands = tracker.process(frame, roi=roi)
            hand = _pick_hand(hands)
            for fid in FINGER_IDS:
                if hand is not None and fid < len(hand.landmarks):
                    x, y, _ = hand.landmarks[fid]
                    series[fid].append((float(x), float(y)))
                else:
                    series[fid].append((np.nan, np.nan))
    finally:
        tracker.close()
        cap.release()
    return series


def _jitter_metrics(points: List[Tuple[float, float]]) -> Dict[str, float]:
    arr = np.asarray(points, dtype=np.float64)
    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    coverage = float(np.mean(valid)) if len(arr) else 0.0
    xy = arr[valid]
    if len(xy) < 4:
        return {"coverage": coverage, "step_median": float("nan"),
                "step_p95": float("nan"), "hf_std": float("nan"), "n": int(len(xy))}
    # Frame-to-frame displacement = per-frame "speed" in px. On a resting clip
    # this is pure jitter; on a tapping clip it mixes real motion + jitter.
    deltas = np.diff(xy, axis=0)
    step = np.hypot(deltas[:, 0], deltas[:, 1])
    # High-frequency residual: detrend with a 5-frame moving average, then std.
    k = 5
    kernel = np.ones(k) / k
    trend_x = np.convolve(xy[:, 0], kernel, mode="same")
    trend_y = np.convolve(xy[:, 1], kernel, mode="same")
    resid = np.hypot(xy[:, 0] - trend_x, xy[:, 1] - trend_y)
    # Drop edge effects of 'same' convolution.
    resid = resid[k:-k] if len(resid) > 2 * k else resid
    return {
        "coverage": coverage,
        "step_median": float(np.median(step)),
        "step_p95": float(np.percentile(step, 95)),
        "hf_std": float(np.std(resid)),
        "n": int(len(xy)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", help="Session dir containing raw_video.avi")
    parser.add_argument("--roi-y", type=float, default=config.TRACKING_ROI_Y_MIN)
    parser.add_argument("--max-width", type=int, default=config.TRACKING_MAX_WIDTH)
    parser.add_argument("--fingers", default="8,12,16,20",
                        help="Comma fingertip ids to print (default skip thumb=4)")
    args = parser.parse_args()

    video_path = Path(args.session_dir) / "raw_video.avi"
    if not video_path.exists():
        raise SystemExit(f"Missing {video_path}")
    print_fingers = [int(f) for f in args.fingers.split(",") if f.strip()]

    results: Dict[str, Dict[int, Dict[str, float]]] = {}
    for label, ema, flow, euro in STAGES:
        series = _run_stage(video_path, ema, flow, euro, args.roi_y, args.max_width)
        results[label] = {fid: _jitter_metrics(series[fid]) for fid in FINGER_IDS}
        print(f"[done] stage={label}", file=sys.stderr)

    print(f"\n=== fingertip jitter by stage: {Path(args.session_dir).name} ===")
    print("metric = frame-to-frame displacement (px). lower = less jitter on a resting clip.\n")
    header = f"{'finger':8} {'stage':16} {'cover':>6} {'step_med':>9} {'step_p95':>9} {'hf_std':>7}"
    print(header)
    print("-" * len(header))
    for fid in print_fingers:
        for label, *_ in STAGES:
            m = results[label][fid]
            print(f"{FINGER_NAMES.get(fid, fid):8} {label:16} "
                  f"{m['coverage']*100:5.0f}% {m['step_median']:9.2f} "
                  f"{m['step_p95']:9.2f} {m['hf_std']:7.2f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
