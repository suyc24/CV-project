from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class Event:
    time_s: float
    note: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate predicted AirDesk hit events against a labeled key-onset file. "
            "This is intentionally dataset-agnostic so it can be used with PianoVAM TSV, "
            "manually annotated sessions, or future project datasets."
        )
    )
    parser.add_argument("--pred", required=True, help="Predicted hits CSV, e.g. replay_session.py *_hits.csv")
    parser.add_argument("--gt", required=True, help="Ground-truth CSV/TSV with key onset times")
    parser.add_argument("--pred-time-column", default="timestamp")
    parser.add_argument("--gt-time-column", default="onset")
    parser.add_argument("--pred-note-column", default="note_id")
    parser.add_argument("--gt-note-column", default="note")
    parser.add_argument("--pred-time-offset", type=float, default=0.0)
    parser.add_argument("--gt-time-offset", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=0.08, help="Match tolerance in seconds")
    parser.add_argument("--match-notes", action="store_true", help="Require predicted and GT notes to match")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predicted = read_events(
        Path(args.pred),
        time_column=args.pred_time_column,
        note_column=args.pred_note_column,
        time_offset=args.pred_time_offset,
    )
    ground_truth = read_events(
        Path(args.gt),
        time_column=args.gt_time_column,
        note_column=args.gt_note_column,
        time_offset=args.gt_time_offset,
    )
    result = evaluate_events(
        predicted,
        ground_truth,
        tolerance_s=args.tolerance,
        match_notes=args.match_notes,
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    return 0


def read_events(path: Path, time_column: str, note_column: str, time_offset: float) -> list[Event]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    events: list[Event] = []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        if reader.fieldnames is None:
            raise SystemExit(f"{path} has no header row")
        if time_column not in reader.fieldnames:
            raise SystemExit(f"{path} is missing time column {time_column!r}; found {reader.fieldnames}")
        for row in reader:
            value = row.get(time_column)
            if value is None or value == "":
                continue
            try:
                time_s = float(value) + time_offset
            except ValueError:
                continue
            note = row.get(note_column) if note_column in reader.fieldnames else None
            events.append(Event(time_s=time_s, note=normalize_note(note)))
    events.sort(key=lambda event: event.time_s)
    return events


def evaluate_events(
    predicted: Iterable[Event],
    ground_truth: Iterable[Event],
    tolerance_s: float,
    match_notes: bool = False,
) -> dict[str, object]:
    predictions = list(predicted)
    truth = list(ground_truth)
    used_pred: set[int] = set()
    matches = []

    for gt_idx, gt_event in enumerate(truth):
        best_idx: Optional[int] = None
        best_error = float("inf")
        for pred_idx, pred_event in enumerate(predictions):
            if pred_idx in used_pred:
                continue
            if match_notes and pred_event.note and gt_event.note and pred_event.note != gt_event.note:
                continue
            error = abs(pred_event.time_s - gt_event.time_s)
            if error <= tolerance_s and error < best_error:
                best_error = error
                best_idx = pred_idx
        if best_idx is None:
            continue
        used_pred.add(best_idx)
        matches.append(
            {
                "gt_index": gt_idx,
                "pred_index": best_idx,
                "gt_time": truth[gt_idx].time_s,
                "pred_time": predictions[best_idx].time_s,
                "abs_error_s": best_error,
                "gt_note": truth[gt_idx].note,
                "pred_note": predictions[best_idx].note,
            }
        )

    tp = len(matches)
    fp = max(0, len(predictions) - tp)
    fn = max(0, len(truth) - tp)
    precision = tp / len(predictions) if predictions else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    mean_abs_error = sum(item["abs_error_s"] for item in matches) / tp if tp else None
    return {
        "predicted_events": len(predictions),
        "ground_truth_events": len(truth),
        "matched_events": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tolerance_s": tolerance_s,
        "match_notes": match_notes,
        "mean_abs_error_s": mean_abs_error,
        "matches_preview": matches[:25],
    }


def normalize_note(note: Optional[str]) -> Optional[str]:
    if note is None:
        return None
    note = str(note).strip()
    if not note:
        return None
    return note.lower()


if __name__ == "__main__":
    raise SystemExit(main())
