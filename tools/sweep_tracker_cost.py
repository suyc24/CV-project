from __future__ import annotations

"""Sweep HandTracker cost vs. fingertip jitter vs. detection dropout.

For each (model_complexity, input_max_width[, min_detection_confidence])
configuration this re-runs the FULL production filter stack over a recorded
clip and reports, per config:

  - cover   : fraction of frames a hand was detected (dropout = 1 - cover)
  - hf_std  : avg high-frequency fingertip residual over index/mid/ring/pinky
              (px; on a resting clip this is pure jitter -- lower is better)
  - step_p95: avg 95th-pct frame-to-frame fingertip jump (px; big-pop jitter)
  - trk_ms  : median HandTracker.process() time (the live FPS bottleneck)
  - est_fps : 1000 / (trk_ms + OVERHEAD_MS), capped at the camera rate

OVERHEAD_MS is the fixed non-tracking loop cost measured by
tools/profile_record3d_pipeline.py (loop_total - hand_tracking ~= 11.2ms).

Run on a *resting* clip so the jitter columns isolate tracker noise.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import config
from camera_utils import tracking_roi
from hand_tracker import HandTracker

FINGER_IDS = (8, 12, 16, 20)  # skip thumb; it rests differently
OVERHEAD_MS = 11.2
CAMERA_FPS = 30.0

# (model_complexity, input_max_width, min_detection_confidence)
DEFAULT_GRID: Tuple[Tuple[int, int, float], ...] = (
    (0, 320, 0.45),
    (0, 416, 0.45),  # current production baseline
    (0, 512, 0.45),
    (0, 640, 0.45),  # native width, no downscale
    (1, 416, 0.45),
    (1, 640, 0.45),
)


def _pick_hand(hands):
    if not hands:
        return None
    best, best_span = None, -1.0
    for hand in hands:
        lm = hand.landmarks
        if len(lm) <= 12:
            continue
        span = abs(lm[12][0] - lm[0][0]) + abs(lm[12][1] - lm[0][1])
        if span > best_span:
            best_span, best = span, hand
    return best or hands[0]


def _hf_std(points: List[Tuple[float, float]]) -> float:
    arr = np.asarray(points, dtype=np.float64)
    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    xy = arr[valid]
    if len(xy) < 12:
        return float("nan")
    k = 5
    kernel = np.ones(k) / k
    tx = np.convolve(xy[:, 0], kernel, mode="same")
    ty = np.convolve(xy[:, 1], kernel, mode="same")
    resid = np.hypot(xy[:, 0] - tx, xy[:, 1] - ty)[k:-k]
    return float(np.std(resid))


def _step_p95(points: List[Tuple[float, float]]) -> float:
    arr = np.asarray(points, dtype=np.float64)
    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    xy = arr[valid]
    if len(xy) < 4:
        return float("nan")
    d = np.diff(xy, axis=0)
    return float(np.percentile(np.hypot(d[:, 0], d[:, 1]), 95))


def run_config(video_path: Path, mc: int, width: int, conf: float, roi_y: float):
    config.FINGERTIP_OPTICAL_FLOW_STABILIZATION = True
    tracker = HandTracker(
        max_num_hands=2,
        min_detection_confidence=conf,
        min_tracking_confidence=conf,
        input_max_width=width,
        smooth_landmarks=True,
    )
    series: Dict[int, List[Tuple[float, float]]] = {f: [] for f in FINGER_IDS}
    times_ms: List[float] = []
    present = 0
    total = 0
    cap = cv2.VideoCapture(str(video_path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            roi = tracking_roi(frame.shape, roi_y)
            t0 = time.perf_counter()
            hands = tracker.process(frame, roi=roi)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            hand = _pick_hand(hands)
            total += 1
            if hand is not None:
                present += 1
            for f in FINGER_IDS:
                if hand is not None and f < len(hand.landmarks):
                    x, y, _ = hand.landmarks[f]
                    series[f].append((float(x), float(y)))
                else:
                    series[f].append((np.nan, np.nan))
    finally:
        tracker.close()
        cap.release()

    # Drop the first few frames (model warmup) from timing.
    warm = times_ms[5:] if len(times_ms) > 10 else times_ms
    trk_ms = float(np.median(warm)) if warm else float("nan")
    hf = float(np.nanmean([_hf_std(series[f]) for f in FINGER_IDS]))
    p95 = float(np.nanmean([_step_p95(series[f]) for f in FINGER_IDS]))
    cover = present / total if total else 0.0
    est_fps = min(CAMERA_FPS, 1000.0 / (trk_ms + OVERHEAD_MS)) if np.isfinite(trk_ms) else float("nan")
    return {"cover": cover, "hf": hf, "p95": p95, "trk_ms": trk_ms, "fps": est_fps}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", help="Resting-clip session dir with raw_video.avi")
    parser.add_argument("--roi-y", type=float, default=config.TRACKING_ROI_Y_MIN)
    args = parser.parse_args()
    video_path = Path(args.session_dir) / "raw_video.avi"
    if not video_path.exists():
        raise SystemExit(f"Missing {video_path}")

    print(f"\n=== tracker cost vs jitter vs dropout: {Path(args.session_dir).name} ===")
    print(f"OVERHEAD_MS={OVERHEAD_MS}  camera_cap={CAMERA_FPS:.0f}fps  "
          f"jitter avg over fingers {FINGER_IDS}\n")
    header = (f"{'mc':>2} {'width':>5} {'conf':>5} | {'cover':>6} {'hf_std':>7} "
              f"{'step_p95':>9} | {'trk_ms':>7} {'est_fps':>7}")
    print(header)
    print("-" * len(header))
    for mc, width, conf in DEFAULT_GRID:
        r = run_config(video_path, mc, width, conf, args.roi_y)
        tag = " <- baseline" if (mc, width, conf) == (0, 416, 0.45) else ""
        print(f"{mc:>2} {width:>5} {conf:>5.2f} | {r['cover']*100:5.0f}% "
              f"{r['hf']:7.2f} {r['p95']:9.2f} | {r['trk_ms']:7.1f} {r['fps']:7.1f}{tag}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
