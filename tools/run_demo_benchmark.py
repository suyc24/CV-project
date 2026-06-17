from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from hit_detector import HitDetector, HitEvent
from replay_session import _depth_observations_from_dict, _hand_from_dict, _zone_from_dict


DEFAULT_SESSIONS = (
    "data/sessions/ipad_test01",
    "data/sessions/benchmarks_3d/bench3d_index_fast_g4",
    "data/sessions/benchmarks_3d/bench3d_index_slow_g4",
    "data/sessions/benchmarks_3d/bench3d_ring_low_lift_e4",
    "data/sessions/benchmarks_3d/bench3d_pinky_low_lift_f4",
    "data/sessions/benchmarks_3d/bench3d_adjacent_keys_g4_a4",
    "data/sessions/benchmarks_3d/bench3d_rest_on_keys",
    "data/sessions/benchmarks_3d/bench3d_hover_no_contact",
)


@dataclass
class Annotation:
    onset: float
    note: str
    frame_index: Optional[int]


@dataclass
class PredictedHit:
    timestamp: float
    frame_index: int
    note: str
    hand_id: int
    finger_id: int
    velocity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay existing AirDesk sessions through the current HitDetector and score "
            "annotated demo/benchmark clips. This is intended to reduce manual demo testing."
        )
    )
    parser.add_argument("sessions", nargs="*", default=list(DEFAULT_SESSIONS))
    parser.add_argument("--output-root", default="data/benchmarks/demo_eval")
    parser.add_argument("--frame-tolerance", type=int, default=4)
    parser.add_argument("--time-tolerance", type=float, default=0.22)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--annotation-padding-frames",
        type=int,
        default=45,
        help=(
            "For annotated sessions, ignore predictions outside the annotated song "
            "window plus this many frames. Use a negative value to disable."
        ),
    )
    parser.add_argument("--no-depth", action="store_true", help="Ignore recorded depth observations during replay")
    parser.add_argument(
        "--piano-sensitivity",
        choices=["current", "strict", "balanced", "sensitive"],
        default="current",
        help="Temporarily apply a piano tuning preset before replay.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    apply_piano_sensitivity(args.piano_sensitivity)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for raw_session in args.sessions:
        session_dir = Path(raw_session)
        if not session_dir.exists():
            print(f"Skipping missing session: {session_dir}", file=sys.stderr)
            continue
        summary = evaluate_session(
            session_dir,
            frame_tolerance=args.frame_tolerance,
            time_tolerance=args.time_tolerance,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            annotation_padding_frames=args.annotation_padding_frames,
            use_depth=not args.no_depth,
        )
        summaries.append(summary)
        session_output = output_root / session_dir.name
        session_output.mkdir(parents=True, exist_ok=True)
        (session_output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_hits_csv(session_output / "predicted_hits.csv", summary["predicted_hits"])
        write_matches_csv(session_output / "matches.csv", summary["matches"])
        print(one_line_summary(summary))

    aggregate = {
        "sessions": summaries,
        "totals": aggregate_totals(summaries),
        "config": benchmark_config(),
    }
    (output_root / "benchmark_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate["totals"], indent=2))
    return 0


def evaluate_session(
    session_dir: Path,
    frame_tolerance: int,
    time_tolerance: float,
    start_frame: Optional[int],
    end_frame: Optional[int],
    annotation_padding_frames: int,
    use_depth: bool,
) -> dict[str, Any]:
    annotations_path = find_annotations_path(session_dir)
    annotations = load_annotations(annotations_path) if annotations_path else []
    effective_start_frame = start_frame
    effective_end_frame = end_frame
    if annotations and annotation_padding_frames >= 0:
        annotation_frames = [annotation.frame_index for annotation in annotations if annotation.frame_index is not None]
        if annotation_frames:
            window_start = min(annotation_frames) - annotation_padding_frames
            window_end = max(annotation_frames) + annotation_padding_frames
            effective_start_frame = max(window_start, start_frame) if start_frame is not None else window_start
            effective_end_frame = min(window_end, end_frame) if end_frame is not None else window_end
    predicted_hits, reason_counts, frame_count = replay_hits(
        session_dir / "frames.jsonl",
        start_frame=effective_start_frame,
        end_frame=effective_end_frame,
        use_depth=use_depth,
    )
    matches, misses, extras, wrong_near = match_hits(
        annotations,
        predicted_hits,
        frame_tolerance=frame_tolerance,
        time_tolerance=time_tolerance,
    )
    note_counts = Counter(hit.note for hit in predicted_hits)
    finger_counts = Counter(f"F{hit.finger_id}" for hit in predicted_hits)
    annotated = bool(annotations)
    score = score_session(
        annotations=annotations,
        predicted_hits=predicted_hits,
        matches=matches,
        misses=misses,
        extras=extras,
        wrong_near=wrong_near,
    )
    return {
        "session": str(session_dir),
        "annotations_path": str(annotations_path) if annotations_path else None,
        "frames": frame_count,
        "evaluated_start_frame": effective_start_frame,
        "evaluated_end_frame": effective_end_frame,
        "annotated": annotated,
        "annotation_count": len(annotations),
        "predicted_count": len(predicted_hits),
        "matched_count": len(matches),
        "miss_count": len(misses),
        "extra_count": len(extras),
        "wrong_near_count": len(wrong_near),
        "precision": len(matches) / len(predicted_hits) if predicted_hits else (1.0 if not annotations else 0.0),
        "recall": len(matches) / len(annotations) if annotations else None,
        "score": score,
        "expected_notes": [annotation.note for annotation in annotations],
        "predicted_notes": [hit.note for hit in predicted_hits],
        "note_counts": dict(note_counts),
        "finger_counts": dict(finger_counts),
        "reason_counts": dict(reason_counts),
        "matches": matches,
        "misses": [annotation_to_dict(annotation) for annotation in misses],
        "extras": [hit_to_dict(hit) for hit in extras],
        "wrong_near": [hit_to_dict(hit) for hit in wrong_near],
        "predicted_hits": [hit_to_dict(hit) for hit in predicted_hits],
    }


def load_annotations(path: Path) -> list[Annotation]:
    if not path.exists():
        return []
    rows: list[Annotation] = []
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            note = str(row.get("note", "")).strip().upper()
            if not note:
                continue
            frame_index = optional_int(row.get("frame_index"))
            rows.append(
                Annotation(
                    onset=float(row.get("onset", 0.0) or 0.0),
                    note=note,
                    frame_index=frame_index,
                )
            )
    return rows


def find_annotations_path(session_dir: Path) -> Optional[Path]:
    direct = session_dir / "annotations.csv"
    if direct.exists():
        return direct
    metadata_path = session_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    source_session = metadata.get("source_session")
    if not source_session:
        return None
    source_path = Path(str(source_session))
    if not source_path.is_absolute():
        cwd_relative = (Path.cwd() / source_path).resolve()
        metadata_relative = (metadata_path.parent / source_path).resolve()
        source_path = cwd_relative if cwd_relative.exists() else metadata_relative
    source_annotations = source_path / "annotations.csv"
    return source_annotations if source_annotations.exists() else None


def replay_hits(
    frames_path: Path,
    start_frame: Optional[int],
    end_frame: Optional[int],
    use_depth: bool,
) -> tuple[list[PredictedHit], Counter[str], int]:
    if not frames_path.exists():
        raise FileNotFoundError(frames_path)
    detector = HitDetector()
    hits: list[PredictedHit] = []
    reason_counts: Counter[str] = Counter()
    frame_count = 0
    with frames_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            entry = json.loads(line)
            frame_index = int(entry.get("frame_index", frame_count))
            frame_count += 1
            if start_frame is not None and frame_index < start_frame:
                continue
            if end_frame is not None and frame_index > end_frame:
                continue
            hands = [_hand_from_dict(hand) for hand in entry.get("hands", [])]
            zones = [_zone_from_dict(zone) for zone in entry.get("zones", [])]
            depth_observations = (
                _depth_observations_from_dict(entry.get("depth_observations", []))
                if use_depth
                else {}
            )
            timestamp = float(entry.get("relative_time", entry.get("timestamp", 0.0)) or 0.0)
            frame_hits = detector.update(hands, zones, timestamp, depth_observations)
            for hit in frame_hits:
                hits.append(predicted_hit(frame_index, hit))
            for diagnostic in detector.diagnostics():
                reason_counts[str(diagnostic.get("reason", "unknown"))] += 1
    return hits, reason_counts, frame_count


def match_hits(
    annotations: list[Annotation],
    hits: list[PredictedHit],
    frame_tolerance: int,
    time_tolerance: float,
) -> tuple[list[dict[str, Any]], list[Annotation], list[PredictedHit], list[PredictedHit]]:
    used_hits: set[int] = set()
    matches: list[dict[str, Any]] = []
    misses: list[Annotation] = []

    for annotation in annotations:
        candidates = []
        for idx, hit in enumerate(hits):
            if idx in used_hits or hit.note != annotation.note:
                continue
            distance = annotation_distance(annotation, hit, frame_tolerance, time_tolerance)
            if distance is not None:
                candidates.append((distance, idx, hit))
        if candidates:
            _, idx, hit = min(candidates, key=lambda item: item[0])
            used_hits.add(idx)
            matches.append(
                {
                    "expected": annotation_to_dict(annotation),
                    "hit": hit_to_dict(hit),
                    "frame_delta": (
                        None
                        if annotation.frame_index is None
                        else hit.frame_index - annotation.frame_index
                    ),
                    "time_delta": hit.timestamp - annotation.onset,
                }
            )
        else:
            misses.append(annotation)

    extras = [hit for idx, hit in enumerate(hits) if idx not in used_hits]
    wrong_near = [
        hit
        for hit in extras
        if any(
            annotation.note != hit.note
            and annotation_distance(annotation, hit, frame_tolerance, time_tolerance) is not None
            for annotation in annotations
        )
    ]
    return matches, misses, extras, wrong_near


def annotation_distance(
    annotation: Annotation,
    hit: PredictedHit,
    frame_tolerance: int,
    time_tolerance: float,
) -> Optional[float]:
    if annotation.frame_index is not None:
        frame_delta = abs(hit.frame_index - annotation.frame_index)
        if frame_delta <= frame_tolerance:
            return float(frame_delta)
        return None
    time_delta = abs(hit.timestamp - annotation.onset)
    if time_delta <= time_tolerance:
        return time_delta * 1000.0
    return None


def score_session(
    annotations: list[Annotation],
    predicted_hits: list[PredictedHit],
    matches: list[dict[str, Any]],
    misses: list[Annotation],
    extras: list[PredictedHit],
    wrong_near: list[PredictedHit],
) -> float:
    if not annotations:
        return -float(len(predicted_hits))
    return (
        len(matches) * 10.0
        - len(misses) * 8.0
        - len(extras) * 2.0
        - len(wrong_near) * 12.0
    )


def predicted_hit(frame_index: int, hit: HitEvent) -> PredictedHit:
    return PredictedHit(
        timestamp=float(hit.timestamp),
        frame_index=frame_index,
        note=str(hit.note_id).upper(),
        hand_id=int(hit.hand_id),
        finger_id=int(hit.finger_id),
        velocity=float(hit.velocity),
    )


def aggregate_totals(summaries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(summaries)
    annotated = [row for row in rows if row["annotated"]]
    guardrail = [row for row in rows if not row["annotated"]]
    return {
        "session_count": len(rows),
        "annotated_session_count": len(annotated),
        "guardrail_session_count": len(guardrail),
        "annotations": sum(int(row["annotation_count"]) for row in annotated),
        "predicted": sum(int(row["predicted_count"]) for row in rows),
        "matched": sum(int(row["matched_count"]) for row in annotated),
        "misses": sum(int(row["miss_count"]) for row in annotated),
        "extras": sum(int(row["extra_count"]) for row in annotated),
        "wrong_near": sum(int(row["wrong_near_count"]) for row in annotated),
        "guardrail_hits": sum(int(row["predicted_count"]) for row in guardrail),
        "score": sum(float(row["score"]) for row in rows),
    }


def one_line_summary(summary: dict[str, Any]) -> str:
    name = Path(str(summary["session"])).name
    if summary["annotated"]:
        return (
            f"{name}: match={summary['matched_count']}/{summary['annotation_count']} "
            f"miss={summary['miss_count']} extra={summary['extra_count']} "
            f"wrong_near={summary['wrong_near_count']} score={summary['score']:.1f}"
        )
    return (
        f"{name}: guardrail_hits={summary['predicted_count']} "
        f"notes={summary['note_counts']} score={summary['score']:.1f}"
    )


def annotation_to_dict(annotation: Annotation) -> dict[str, Any]:
    return {
        "onset": annotation.onset,
        "note": annotation.note,
        "frame_index": annotation.frame_index,
    }


def hit_to_dict(hit: PredictedHit) -> dict[str, Any]:
    return {
        "timestamp": hit.timestamp,
        "frame_index": hit.frame_index,
        "note": hit.note,
        "hand_id": hit.hand_id,
        "finger_id": hit.finger_id,
        "velocity": hit.velocity,
    }


def write_hits_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows, ["frame_index", "timestamp", "note", "hand_id", "finger_id", "velocity"])


def write_matches_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened = []
    for row in rows:
        expected = row["expected"]
        hit = row["hit"]
        flattened.append(
            {
                "expected_frame": expected["frame_index"],
                "expected_time": expected["onset"],
                "expected_note": expected["note"],
                "hit_frame": hit["frame_index"],
                "hit_time": hit["timestamp"],
                "hit_note": hit["note"],
                "finger_id": hit["finger_id"],
                "frame_delta": row["frame_delta"],
                "time_delta": row["time_delta"],
            }
        )
    write_csv(
        path,
        flattened,
        [
            "expected_frame",
            "expected_time",
            "expected_note",
            "hit_frame",
            "hit_time",
            "hit_note",
            "finger_id",
            "frame_delta",
            "time_delta",
        ],
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def benchmark_config() -> dict[str, Any]:
    keys = (
        "PIANO_ARM_MIN_LIFT_PX",
        "PIANO_STRIKE_MIN_DROP_PX",
        "PIANO_STRIKE_MIN_NET_DROP_PX",
        "PIANO_STRIKE_MIN_VELOCITY",
        "PIANO_RELEASE_LIFT_PX",
        "PIANO_RELEASE_STABLE_FRAMES",
        "PIANO_RELEASE_MIN_NET_LIFT_PX",
        "PIANO_MONOPHONIC_PER_HAND",
        "PIANO_BLOCK_PASSIVE_ARM_WHILE_DEPTH_CONTACT",
        "PIANO_HIT_MIN_KEY_Y_RATIO",
        "PIANO_THUMB_HIT_MIN_KEY_Y_RATIO",
        "PIANO_LEFT_TRIM_KEYS",
        "PIANO_RIGHT_TRIM_KEYS",
        "PIANO_3D_TRIGGER_ENABLED",
        "PIANO_3D_DIRECT_TRIGGER_ENABLED",
        "PIANO_HAND_HIT_COOLDOWN",
        "TRACKING_NEW_HAND_HIT_BLOCK_FRAMES",
    )
    return {key: getattr(config, key) for key in keys}


def apply_piano_sensitivity(preset: str) -> None:
    if preset == "current":
        return
    if preset == "strict":
        config.PIANO_STRIKE_MIN_DROP_PX = 12.0
        config.PIANO_STRIKE_MIN_VELOCITY = 130.0
        config.PIANO_STRIKE_MIN_NET_DROP_PX = 8.0
        config.PIANO_RELEASE_LIFT_PX = 24.0
        config.PIANO_RELEASE_MIN_NET_LIFT_PX = 28.0
        config.PIANO_RELEASE_STABLE_FRAMES = 3
        config.PIANO_RELEASE_MIN_UP_VELOCITY = 10.0
        config.PIANO_RELEASE_STRONG_LIFT_MULTIPLIER = 1.25
        config.PIANO_HAND_HIT_COOLDOWN = 0.14
    elif preset == "balanced":
        config.PIANO_STRIKE_MIN_DROP_PX = 8.0
        config.PIANO_STRIKE_MIN_VELOCITY = 100.0
        config.PIANO_STRIKE_MIN_NET_DROP_PX = 5.0
        config.PIANO_RELEASE_LIFT_PX = 18.0
        config.PIANO_RELEASE_MIN_NET_LIFT_PX = 20.0
        config.PIANO_RELEASE_STABLE_FRAMES = 2
        config.PIANO_RELEASE_MIN_UP_VELOCITY = 14.0
        config.PIANO_RELEASE_STRONG_LIFT_MULTIPLIER = 1.4
        config.PIANO_HAND_HIT_COOLDOWN = 0.10
    elif preset == "sensitive":
        config.PIANO_STRIKE_MIN_DROP_PX = 5.0
        config.PIANO_STRIKE_MIN_VELOCITY = 75.0
        config.PIANO_STRIKE_MIN_NET_DROP_PX = 3.0
        config.PIANO_RELEASE_LIFT_PX = 12.0
        config.PIANO_RELEASE_MIN_NET_LIFT_PX = 14.0
        config.PIANO_RELEASE_STABLE_FRAMES = 1
        config.PIANO_RELEASE_MIN_UP_VELOCITY = 10.0
        config.PIANO_RELEASE_STRONG_LIFT_MULTIPLIER = 1.25
        config.PIANO_HAND_HIT_COOLDOWN = 0.06


def optional_int(value: object) -> Optional[int]:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
