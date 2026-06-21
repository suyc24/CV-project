from __future__ import annotations

"""Probe where MediaPipe hand-detection dropout happens in a recorded clip.

Runs the production HandTracker over raw_video.avi and reports whether the
"no hand detected" frames are concentrated at the clip edges (hand entering /
leaving -- harmless) or scattered through the middle (real in-play instability
that will cause missed/wrong triggers). Saves a few mid-clip dropout frames as
PNGs so they can be eyeballed for the cause.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import config
from camera_utils import tracking_roi
from hand_tracker import HandTracker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir")
    parser.add_argument("--roi-y", type=float, default=config.TRACKING_ROI_Y_MIN)
    parser.add_argument("--max-width", type=int, default=config.TRACKING_MAX_WIDTH)
    parser.add_argument("--mc", type=int, default=1)
    parser.add_argument("--save-dir", default=None, help="Where to dump sample dropout frames")
    parser.add_argument("--save-n", type=int, default=4)
    args = parser.parse_args()

    session = Path(args.session_dir)
    video_path = session / "raw_video.avi"
    if not video_path.exists():
        raise SystemExit(f"Missing {video_path}")
    save_dir = Path(args.save_dir) if args.save_dir else session / "_dropout_frames"

    config.FINGERTIP_OPTICAL_FLOW_STABILIZATION = True
    tracker = HandTracker(
        max_num_hands=2,
        min_detection_confidence=config.HAND_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.HAND_MIN_TRACKING_CONFIDENCE,
        input_max_width=args.max_width,
        smooth_landmarks=True,
    )

    present = []   # bool per frame
    frames = []    # keep refs only for dropout frames we may save
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            roi = tracking_roi(frame.shape, args.roi_y)
            hands = tracker.process(frame, roi=roi)
            hit = bool(hands)
            present.append(hit)
            frames.append(frame if not hit else None)
            idx += 1
    finally:
        tracker.close()
        cap.release()

    p = np.asarray(present, dtype=bool)
    n = len(p)
    if n == 0:
        raise SystemExit("No frames decoded")
    drop = ~p
    total_drop = int(drop.sum())

    # First/last frame where a hand ever appears = the "in-play" window.
    where = np.where(p)[0]
    first_hand = int(where[0]) if where.size else n
    last_hand = int(where[-1]) if where.size else -1
    inplay = drop[first_hand:last_hand + 1] if last_hand >= first_hand else np.array([], bool)
    inplay_drop = int(inplay.sum())
    inplay_len = len(inplay)

    # Longest consecutive dropout run inside the in-play window.
    longest = cur = 0
    for v in inplay:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)

    print(f"\n=== detection dropout: {session.name} (mc={args.mc} width={args.max_width}) ===")
    print(f"frames                : {n}")
    print(f"overall dropout       : {total_drop}/{n} = {total_drop/n*100:.0f}%")
    print(f"in-play window        : frames [{first_hand}..{last_hand}] ({inplay_len} frames)")
    print(f"  edge frames (enter/leave) excluded: {n - inplay_len}")
    print(f"in-play dropout       : {inplay_drop}/{inplay_len} = "
          f"{(inplay_drop/inplay_len*100 if inplay_len else 0):.0f}%  <-- the number that matters")
    print(f"longest dropout run in-play: {longest} frames")

    # Save a few scattered mid-clip dropout frames to look at.
    mid_drop_idx = [i for i in range(first_hand, last_hand + 1) if drop[i]]
    if mid_drop_idx and args.save_n > 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        pick = np.linspace(0, len(mid_drop_idx) - 1, min(args.save_n, len(mid_drop_idx))).astype(int)
        for j in pick:
            fi = mid_drop_idx[j]
            out = save_dir / f"drop_frame_{fi:04d}.png"
            cv2.imwrite(str(out), frames[fi])
        print(f"saved {len(pick)} mid-clip dropout frame(s) to {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
