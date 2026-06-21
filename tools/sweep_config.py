from __future__ import annotations
"""Sweep one config attribute and report demo + hover-guardrail scores.
Usage: sweep_config.py ATTR v1 v2 v3 ...   (values parsed as float)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from tools.run_demo_benchmark import evaluate_session

DEMO = Path("data/sessions/shuxingxing01")
HOVER = Path("data/sessions/benchmarks_3d/twinkle_right_hand/dev/02_hover_both_hands")
REST = Path("data/sessions/benchmarks_3d/twinkle_right_hand/dev/01_rest_both_hands")


def ev(s):
    return evaluate_session(s, frame_tolerance=4, time_tolerance=0.22, start_frame=None,
                            end_frame=None, annotation_padding_frames=45, use_depth=True)


def main():
    attr = sys.argv[1]
    vals = [float(v) for v in sys.argv[2:]]
    orig = getattr(config, attr)
    print(f"sweeping {attr} (orig={orig})")
    print(f"{'val':>8} | {'match':>5} {'miss':>4} {'extra':>5} {'wrong':>5} | {'hover':>5} {'rest':>4}")
    for v in vals:
        setattr(config, attr, type(orig)(v) if not isinstance(orig, bool) else bool(v))
        d = ev(DEMO); h = ev(HOVER); r = ev(REST)
        print(f"{v:8.3f} | {d['matched_count']:5d} {d['miss_count']:4d} "
              f"{d['extra_count']:5d} {d['wrong_near_count']:5d} | "
              f"{h['predicted_count']:5d} {r['predicted_count']:4d}")
    setattr(config, attr, orig)


if __name__ == "__main__":
    main()
