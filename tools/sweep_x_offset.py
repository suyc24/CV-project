from __future__ import annotations
"""Sweep PIANO_ZONE_X_OFFSET_PX on the Twinkle demo + hover guardrail."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from tools.run_demo_benchmark import evaluate_session

DEMO = Path("data/sessions/shuxingxing01")
HOVER = Path("data/sessions/benchmarks_3d/twinkle_right_hand/dev/02_hover_both_hands")


def ev(session):
    return evaluate_session(
        session, frame_tolerance=4, time_tolerance=0.22,
        start_frame=None, end_frame=None, annotation_padding_frames=45, use_depth=True,
    )


def main():
    print(f"{'xoff':>5} | {'match':>5} {'miss':>4} {'extra':>5} {'wrong':>5} | {'hover_false':>11}")
    for off in [-30, -24, -18, -14, -10, -6, 0, 6, 14]:
        config.PIANO_ZONE_X_OFFSET_PX = off
        d = ev(DEMO)
        h = ev(HOVER)
        print(f"{off:5d} | {d['matched_count']:5d} {d['miss_count']:4d} "
              f"{d['extra_count']:5d} {d['wrong_near_count']:5d} | {h['predicted_count']:11d}")


if __name__ == "__main__":
    main()
