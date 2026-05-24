from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config


@dataclass
class FingerSample:
    time_s: float
    x: float
    y: float
    zone_label: Optional[str]
    reason: Optional[str]
    tracking_source: str
    unstable: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize fingertip tracking quality from an AirDesk session. "
            "Use this before/after tracker changes to compare jitter, jumps, "
            "unstable frames, zone flicker, and rapid repeat hits."
        )
    )
    parser.add_argument("session_dir", help="Directory containing frames.jsonl")
    parser.add_argument("--output-prefix", default="tracking_quality")
    parser.add_argument(
        "--finger-ids",
        default=",".join(str(value) for value in config.TRIGGER_FINGER_IDS),
        help="Comma-separated MediaPipe landmark ids to evaluate",
    )
    parser.add_argument("--jump-threshold-px", type=float, default=25.0)
    parser.add_argument("--zone-jitter-step-px", type=float, default=18.0)
    parser.add_argument("--rapid-repeat-s", type=float, default=0.10)
    parser.add_argument("--min-samples", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_dir = Path(args.session_dir)
    frames_path = session_dir / "frames.jsonl"
    if not frames_path.exists():
        raise SystemExit(f"Missing {frames_path}")

    finger_ids = parse_int_list(args.finger_ids)
    frames = load_jsonl(frames_path)
    trajectories = collect_finger_samples(frames, finger_ids)
    per_finger = summarize_trajectories(
        trajectories,
        jump_threshold_px=args.jump_threshold_px,
        zone_jitter_step_px=args.zone_jitter_step_px,
        min_samples=args.min_samples,
    )
    hits = collect_hits(frames)
    hit_summary = summarize_hits(hits, args.rapid_repeat_s)
    reason_counts = count_reasons(frames)
    summary = {
        "session": str(session_dir),
        "frames": len(frames),
        "duration_s": session_duration(frames),
        "finger_ids": finger_ids,
        "overall": summarize_overall(per_finger),
        "hits": hit_summary,
        "diagnostic_reason_counts": dict(reason_counts),
        "parameters": {
            "jump_threshold_px": args.jump_threshold_px,
            "zone_jitter_step_px": args.zone_jitter_step_px,
            "rapid_repeat_s": args.rapid_repeat_s,
            "min_samples": args.min_samples,
        },
    }

    json_path = session_dir / f"{args.output_prefix}_summary.json"
    csv_path = session_dir / f"{args.output_prefix}_per_finger.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_per_finger_csv(csv_path, per_finger)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collect_finger_samples(frames: Iterable[dict], finger_ids: tuple[int, ...]) -> dict[tuple[int, int], list[FingerSample]]:
    trajectories: dict[tuple[int, int], list[FingerSample]] = defaultdict(list)
    finger_set = set(finger_ids)
    for frame in frames:
        time_s = float(frame.get("relative_time", frame.get("timestamp", 0.0)) or 0.0)
        diagnostics = {
            (int(diag.get("hand_id", -1)), int(diag.get("finger_id", -1))): diag
            for diag in frame.get("diagnostics", [])
            if isinstance(diag, dict)
        }
        for hand in frame.get("hands", []):
            try:
                hand_id = int(hand.get("hand_id", -1))
            except (TypeError, ValueError):
                continue
            landmarks = hand.get("landmarks", [])
            unstable_ids = set(int(value) for value in hand.get("unstable_landmark_ids", []) or [])
            tracking_source = str(hand.get("tracking_source", "unknown"))
            for finger_id in finger_ids:
                if finger_id >= len(landmarks):
                    continue
                point = landmarks[finger_id]
                if not isinstance(point, list | tuple) or len(point) < 2:
                    continue
                diag = diagnostics.get((hand_id, finger_id), {})
                unstable = (
                    finger_id in unstable_ids
                    or bool(diag.get("unstable_tracking", False))
                    or tracking_source == "optical_flow"
                )
                if finger_id not in finger_set:
                    continue
                trajectories[(hand_id, finger_id)].append(
                    FingerSample(
                        time_s=time_s,
                        x=float(point[0]),
                        y=float(point[1]),
                        zone_label=_optional_str(diag.get("zone_label")),
                        reason=_optional_str(diag.get("reason")),
                        tracking_source=tracking_source,
                        unstable=unstable,
                    )
                )
    return trajectories


def summarize_trajectories(
    trajectories: dict[tuple[int, int], list[FingerSample]],
    jump_threshold_px: float,
    zone_jitter_step_px: float,
    min_samples: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (hand_id, finger_id), samples in sorted(trajectories.items()):
        if len(samples) < min_samples:
            continue
        steps = []
        velocities = []
        dts = []
        zone_jitter_switches = 0
        for previous, current in zip(samples, samples[1:]):
            dt = max(1e-6, current.time_s - previous.time_s)
            step = math.hypot(current.x - previous.x, current.y - previous.y)
            steps.append(step)
            velocities.append(step / dt)
            dts.append(dt)
            if (
                previous.zone_label
                and current.zone_label
                and previous.zone_label != current.zone_label
                and step <= zone_jitter_step_px
            ):
                zone_jitter_switches += 1

        zones = Counter(sample.zone_label for sample in samples if sample.zone_label)
        reasons = Counter(sample.reason for sample in samples if sample.reason)
        sources = Counter(sample.tracking_source for sample in samples)
        rows.append(
            {
                "hand_id": hand_id,
                "finger_id": finger_id,
                "samples": len(samples),
                "duration_s": samples[-1].time_s - samples[0].time_s,
                "median_dt_s": percentile(dts, 50),
                "median_step_px": percentile(steps, 50),
                "p95_step_px": percentile(steps, 95),
                "max_step_px": max(steps) if steps else 0.0,
                "p95_velocity_px_s": percentile(velocities, 95),
                "large_jump_count": sum(1 for step in steps if step >= jump_threshold_px),
                "unstable_sample_count": sum(1 for sample in samples if sample.unstable),
                "no_zone_sample_count": sum(1 for sample in samples if not sample.zone_label),
                "zone_jitter_switches": zone_jitter_switches,
                "top_zone": zones.most_common(1)[0][0] if zones else None,
                "top_reason": reasons.most_common(1)[0][0] if reasons else None,
                "tracking_sources": dict(sources),
            }
        )
    return rows


def collect_hits(frames: Iterable[dict]) -> list[dict]:
    hits = []
    for frame in frames:
        frame_time = float(frame.get("relative_time", frame.get("timestamp", 0.0)) or 0.0)
        for hit in frame.get("hits", []):
            if not isinstance(hit, dict):
                continue
            row = dict(hit)
            row["_frame_time"] = frame_time
            row["_relative_timestamp"] = _hit_relative_time(row, frame_time)
            hits.append(row)
    hits.sort(key=lambda hit: float(hit.get("_relative_timestamp", 0.0) or 0.0))
    return hits


def summarize_hits(hits: list[dict], rapid_repeat_s: float) -> dict[str, object]:
    note_counts = Counter(str(hit.get("note_id", hit.get("zone_label", ""))) for hit in hits)
    by_finger: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for hit in hits:
        try:
            key = (int(hit.get("hand_id", -1)), int(hit.get("finger_id", -1)))
        except (TypeError, ValueError):
            continue
        by_finger[key].append(hit)

    rapid_repeats = 0
    min_same_finger_interval = None
    for finger_hits in by_finger.values():
        for previous, current in zip(finger_hits, finger_hits[1:]):
            interval = float(current["_relative_timestamp"]) - float(previous["_relative_timestamp"])
            if min_same_finger_interval is None or interval < min_same_finger_interval:
                min_same_finger_interval = interval
            if interval <= rapid_repeat_s:
                rapid_repeats += 1

    return {
        "total_hits": len(hits),
        "rapid_same_finger_repeats": rapid_repeats,
        "rapid_repeat_s": rapid_repeat_s,
        "min_same_finger_interval_s": min_same_finger_interval,
        "top_notes": dict(note_counts.most_common(12)),
    }


def count_reasons(frames: Iterable[dict]) -> Counter[str]:
    reasons: Counter[str] = Counter()
    for frame in frames:
        for diag in frame.get("diagnostics", []):
            if isinstance(diag, dict):
                reason = diag.get("reason")
                if reason:
                    reasons[str(reason)] += 1
    return reasons


def summarize_overall(per_finger: list[dict[str, object]]) -> dict[str, object]:
    if not per_finger:
        return {}
    sample_count = sum(int(row["samples"]) for row in per_finger)
    return {
        "tracked_finger_streams": len(per_finger),
        "samples": sample_count,
        "weighted_median_step_px": weighted_average(per_finger, "median_step_px", "samples"),
        "weighted_p95_step_px": weighted_average(per_finger, "p95_step_px", "samples"),
        "large_jump_count": sum(int(row["large_jump_count"]) for row in per_finger),
        "unstable_sample_count": sum(int(row["unstable_sample_count"]) for row in per_finger),
        "zone_jitter_switches": sum(int(row["zone_jitter_switches"]) for row in per_finger),
    }


def write_per_finger_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "hand_id",
        "finger_id",
        "samples",
        "duration_s",
        "median_dt_s",
        "median_step_px",
        "p95_step_px",
        "max_step_px",
        "p95_velocity_px_s",
        "large_jump_count",
        "unstable_sample_count",
        "no_zone_sample_count",
        "zone_jitter_switches",
        "top_zone",
        "top_reason",
        "tracking_sources",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["tracking_sources"] = json.dumps(out.get("tracking_sources", {}), sort_keys=True)
            writer.writerow(out)


def percentile(values: Iterable[float], pct: float) -> float:
    data = sorted(float(value) for value in values)
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]
    rank = (len(data) - 1) * (pct / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return data[lo]
    weight = rank - lo
    return data[lo] * (1.0 - weight) + data[hi] * weight


def weighted_average(rows: list[dict[str, object]], value_key: str, weight_key: str) -> float:
    total_weight = sum(float(row.get(weight_key, 0.0) or 0.0) for row in rows)
    if total_weight <= 0:
        return 0.0
    return sum(float(row.get(value_key, 0.0) or 0.0) * float(row.get(weight_key, 0.0) or 0.0) for row in rows) / total_weight


def session_duration(frames: list[dict]) -> float:
    if not frames:
        return 0.0
    first = float(frames[0].get("relative_time", 0.0) or 0.0)
    last = float(frames[-1].get("relative_time", first) or first)
    return max(0.0, last - first)


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = []
    for piece in raw.split(","):
        piece = piece.strip()
        if piece:
            values.append(int(piece))
    return tuple(values)


def _optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _hit_relative_time(hit: dict, frame_time: float) -> float:
    timestamp = hit.get("timestamp")
    try:
        value = float(timestamp)
    except (TypeError, ValueError):
        return frame_time
    if value > 10000.0:
        return frame_time
    return value


if __name__ == "__main__":
    raise SystemExit(main())
