from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from hit_detector import HitDetector
from replay_session import (
    _depth_observations_from_dict,
    _hand_from_dict,
    _hit_to_row,
    _zone_from_dict,
    write_hits_csv,
    write_reason_csv,
)
from tools.evaluate_hit_events import Event, evaluate_events, read_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay Record3D piano benchmark sessions through the 3D trigger path. "
            "Annotated sessions are evaluated against annotations.csv; unannotated "
            "sessions are reported as needing labels."
        )
    )
    parser.add_argument(
        "session_dirs",
        nargs="*",
        help="Optional benchmark session directories. Defaults to every child of --benchmark-root.",
    )
    parser.add_argument("--benchmark-root", default="data/sessions/benchmarks_3d")
    parser.add_argument("--output-root", default="data/benchmarks/3d_trigger")
    parser.add_argument("--tolerance", type=float, default=0.08)
    parser.add_argument("--min-precision", type=float, default=0.85)
    parser.add_argument("--min-recall", type=float, default=0.85)
    parser.add_argument("--min-f1", type=float, default=0.85)
    parser.add_argument("--allow-false-positives", type=int, default=0)
    parser.add_argument("--no-match-notes", action="store_true")
    parser.add_argument("--require-annotations", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sessions = discover_sessions(args)
    if not sessions:
        raise SystemExit("No benchmark sessions found")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    original_config = capture_trigger_config()
    try:
        configure_3d_replay()
        results = [run_session(session, output_root / session.name, args) for session in sessions]
    finally:
        restore_config(original_config)

    annotated_failures = [item for item in results if item["status"] == "failed"]
    missing_annotations = [item for item in results if item["status"] == "needs_annotations"]
    aggregate = {
        "benchmark_root": str(Path(args.benchmark_root)),
        "output_root": str(output_root),
        "tolerance_s": args.tolerance,
        "match_notes": not args.no_match_notes,
        "targets": {
            "min_precision": args.min_precision,
            "min_recall": args.min_recall,
            "min_f1": args.min_f1,
            "allow_false_positives": args.allow_false_positives,
        },
        "sessions": results,
        "annotated_failures": len(annotated_failures),
        "needs_annotations": len(missing_annotations),
        "all_annotated_passed": not annotated_failures,
        "all_sessions_passed": all(item["status"] == "passed" for item in results),
    }
    summary_path = output_root / "benchmark_summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"Wrote {summary_path}")

    if annotated_failures:
        return 2
    if args.require_annotations and missing_annotations:
        return 3
    return 0


def discover_sessions(args: argparse.Namespace) -> list[Path]:
    if args.session_dirs:
        sessions = [Path(path) for path in args.session_dirs]
    else:
        root = Path(args.benchmark_root)
        sessions = sorted(path for path in root.iterdir() if path.is_dir()) if root.exists() else []
    return [session for session in sessions if (session / "frames.jsonl").exists()]


def configure_3d_replay() -> None:
    config.PIANO_TRIGGER_MODE = "3d"
    config.PIANO_DEPTH_TRIGGER_ENABLED = True


def capture_trigger_config() -> dict[str, Any]:
    names = [
        "PIANO_TRIGGER_MODE",
        "PIANO_DEPTH_TRIGGER_ENABLED",
    ]
    return {name: getattr(config, name) for name in names}


def restore_config(values: dict[str, Any]) -> None:
    for name, value in values.items():
        setattr(config, name, value)


def run_session(session_dir: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    replay = replay_3d(session_dir)
    hits_path = output_dir / "predicted_hits.csv"
    reasons_path = output_dir / "miss_reasons.csv"
    write_hits_csv(hits_path, replay["hits"])
    write_reason_csv(reasons_path, replay["reason_counts"])

    annotations_path = session_dir / "annotations.csv"
    result: dict[str, Any] | None = None
    status = "needs_annotations"
    passed = False
    if annotations_path.exists():
        predicted, truth, timebase = read_benchmark_events(hits_path, annotations_path)
        result = evaluate_events(
            predicted,
            truth,
            tolerance_s=args.tolerance,
            match_notes=not args.no_match_notes,
        )
        result = normalize_empty_truth_result(result)
        result["timebase"] = timebase
        passed = passes_targets(result, args)
        status = "passed" if passed else "failed"

    summary = {
        "session": str(session_dir),
        "status": status,
        "passed": passed,
        "frames": replay["frames"],
        "predicted_hits": len(replay["hits"]),
        "online_hits": replay["online_hits"],
        "hits_csv": str(hits_path),
        "miss_reasons_csv": str(reasons_path),
        "annotations_csv": str(annotations_path) if annotations_path.exists() else None,
        "evaluation": result,
        "top_reasons": replay["reason_counts"].most_common(12),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def replay_3d(session_dir: Path) -> dict[str, Any]:
    detector = HitDetector()
    replay_hits: list[dict[str, object]] = []
    reason_counts: Counter[str] = Counter()
    online_hits = 0
    frame_count = 0

    with (session_dir / "frames.jsonl").open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            entry = json.loads(line)
            frame_count += 1
            hands = [_hand_from_dict(hand) for hand in entry.get("hands", [])]
            zones = [_zone_from_dict(zone) for zone in entry.get("zones", [])]
            depth_observations = _depth_observations_from_dict(entry.get("depth_observations", []))
            timestamp = float(entry.get("relative_time", entry.get("timestamp", 0.0)))
            hits = detector.update(hands, zones, timestamp, depth_observations)
            online_hits += len(entry.get("hits", []))
            for diagnostic in detector.diagnostics():
                reason_counts[str(diagnostic.get("reason", "unknown"))] += 1
            for hit in hits:
                replay_hits.append(_hit_to_row(int(entry.get("frame_index", frame_count - 1)), hit))

    return {
        "frames": frame_count,
        "online_hits": online_hits,
        "hits": replay_hits,
        "reason_counts": reason_counts,
    }


def normalize_empty_truth_result(result: dict[str, Any]) -> dict[str, Any]:
    if result["ground_truth_events"] != 0:
        return result
    false_positives = int(result["false_positives"])
    result = dict(result)
    result["precision"] = 1.0 if false_positives == 0 else 0.0
    result["recall"] = 1.0
    result["f1"] = 1.0 if false_positives == 0 else 0.0
    return result


def read_benchmark_events(hits_path: Path, annotations_path: Path) -> tuple[list[Event], list[Event], str]:
    truth_rows = read_annotation_rows(annotations_path)
    fps = next((row.get("fps") for row in truth_rows if row.get("fps")), None)
    has_frame_labels = bool(fps) and any(row.get("frame_index") for row in truth_rows)
    if not has_frame_labels:
        return (
            read_events(hits_path, "timestamp", "note_id", 0.0),
            read_events(annotations_path, "onset", "note", 0.0),
            "timestamp",
        )

    fps_value = float(fps)
    predicted = read_frame_events(hits_path, "frame_index", "note_id", fps_value)
    truth = rows_to_frame_events(truth_rows, "frame_index", "note", fps_value)
    return predicted, truth, "frame_index/fps"


def read_annotation_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def read_frame_events(path: Path, frame_column: str, note_column: str, fps: float) -> list[Event]:
    import csv

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    return rows_to_frame_events(rows, frame_column, note_column, fps)


def rows_to_frame_events(
    rows: list[dict[str, str]],
    frame_column: str,
    note_column: str,
    fps: float,
) -> list[Event]:
    events: list[Event] = []
    if fps <= 0:
        return events
    for row in rows:
        frame_value = row.get(frame_column)
        if frame_value is None or frame_value == "":
            continue
        try:
            frame_index = int(float(frame_value))
        except ValueError:
            continue
        note = row.get(note_column)
        events.append(Event(time_s=frame_index / fps, note=note.lower() if note else None))
    events.sort(key=lambda event: event.time_s)
    return events


def passes_targets(result: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        float(result.get("precision", 0.0) or 0.0) >= args.min_precision
        and float(result.get("recall", 0.0) or 0.0) >= args.min_recall
        and float(result.get("f1", 0.0) or 0.0) >= args.min_f1
        and int(result.get("false_positives", 0) or 0) <= args.allow_false_positives
    )


if __name__ == "__main__":
    raise SystemExit(main())
