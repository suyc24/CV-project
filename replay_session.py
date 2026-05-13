from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import config
from hit_detector import HitDetector, HitEvent
from instrument import Zone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay an AirDesk recorded session through HitDetector")
    parser.add_argument("session_dir", help="Session directory created by --record-session")
    parser.add_argument("--piano-velocity-threshold", type=float, default=None)
    parser.add_argument("--drum-velocity-threshold", type=float, default=None)
    parser.add_argument("--piano-press-ratio", type=float, default=None)
    parser.add_argument("--piano-release-ratio", type=float, default=None)
    parser.add_argument("--output-prefix", default="replay", help="Output filename prefix inside the session directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    apply_overrides(args)
    session_dir = Path(args.session_dir)
    frames_path = session_dir / "frames.jsonl"
    if not frames_path.exists():
        raise SystemExit(f"Missing {frames_path}")

    detector = HitDetector()
    replay_hits: List[Dict[str, object]] = []
    online_hits = 0
    reason_counts: Counter[str] = Counter()
    frame_count = 0

    with frames_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            entry = json.loads(line)
            frame_count += 1
            hands = [_hand_from_dict(hand) for hand in entry.get("hands", [])]
            zones = [_zone_from_dict(zone) for zone in entry.get("zones", [])]
            timestamp = float(entry.get("timestamp", entry.get("relative_time", 0.0)))
            hits = detector.update(hands, zones, timestamp)
            online_hits += len(entry.get("hits", []))
            for diag in detector.diagnostics():
                reason_counts[str(diag.get("reason", "unknown"))] += 1
            for hit in hits:
                replay_hits.append(_hit_to_row(entry.get("frame_index", frame_count - 1), hit))

    write_hits_csv(session_dir / f"{args.output_prefix}_hits.csv", replay_hits)
    write_reason_csv(session_dir / f"{args.output_prefix}_miss_reasons.csv", reason_counts)
    summary = {
        "frames": frame_count,
        "online_hits": online_hits,
        "replay_hits": len(replay_hits),
        "reason_counts": dict(reason_counts),
        "config": {
            "PIANO_HIT_VELOCITY_THRESHOLD": config.PIANO_HIT_VELOCITY_THRESHOLD,
            "PIANO_CROSSING_VELOCITY_THRESHOLD": config.PIANO_CROSSING_VELOCITY_THRESHOLD,
            "HIT_VELOCITY_THRESHOLD": config.HIT_VELOCITY_THRESHOLD,
            "PIANO_PRESS_RATIO": config.PIANO_PRESS_RATIO,
            "PIANO_RELEASE_RATIO": config.PIANO_RELEASE_RATIO,
            "PIANO_HIT_X_MARGIN_RATIO": config.PIANO_HIT_X_MARGIN_RATIO,
            "PIANO_HIT_TOP_MARGIN_RATIO": config.PIANO_HIT_TOP_MARGIN_RATIO,
            "PIANO_HIT_BOTTOM_MARGIN_RATIO": config.PIANO_HIT_BOTTOM_MARGIN_RATIO,
        },
    }
    (session_dir / f"{args.output_prefix}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


def apply_overrides(args: argparse.Namespace) -> None:
    if args.piano_velocity_threshold is not None:
        config.PIANO_HIT_VELOCITY_THRESHOLD = args.piano_velocity_threshold
    if args.drum_velocity_threshold is not None:
        config.HIT_VELOCITY_THRESHOLD = args.drum_velocity_threshold
    if args.piano_press_ratio is not None:
        config.PIANO_PRESS_RATIO = args.piano_press_ratio
    if args.piano_release_ratio is not None:
        config.PIANO_RELEASE_RATIO = args.piano_release_ratio


def _hand_from_dict(hand: Dict[str, object]):
    return SimpleNamespace(
        hand_id=int(hand["hand_id"]),
        label=str(hand.get("label", "Unknown")),
        landmarks=[tuple(point) for point in hand.get("landmarks", [])],
    )


def _zone_from_dict(zone: Dict[str, object]) -> Zone:
    height = max(1.0, float(zone["y2"]) - float(zone["y1"]))
    press_ratio = (float(zone.get("press_y", zone["y1"])) - float(zone["y1"])) / height
    release_ratio = (float(zone.get("release_y", zone["y1"])) - float(zone["y1"])) / height
    kind = str(zone["kind"])
    if kind == "piano":
        press_ratio = config.PIANO_PRESS_RATIO
        release_ratio = config.PIANO_RELEASE_RATIO
    elif kind == "drum":
        press_ratio = config.PRESS_RATIO
        release_ratio = config.RELEASE_RATIO
    return Zone(
        label=str(zone["label"]),
        sound_id=str(zone["sound_id"]),
        x1=int(zone["x1"]),
        y1=int(zone["y1"]),
        x2=int(zone["x2"]),
        y2=int(zone["y2"]),
        kind=kind,
        press_ratio=press_ratio,
        release_ratio=release_ratio,
    )


def _hit_to_row(frame_index: int, hit: HitEvent) -> Dict[str, object]:
    return {
        "frame_index": frame_index,
        "timestamp": hit.timestamp,
        "note_id": hit.note_id,
        "sound_id": hit.sound_id,
        "hand_id": hit.hand_id,
        "finger_id": hit.finger_id,
        "velocity": hit.velocity,
        "volume": hit.volume,
    }


def write_hits_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["frame_index", "timestamp", "note_id", "sound_id", "hand_id", "finger_id", "velocity", "volume"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_reason_csv(path: Path, counter: Counter[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["reason", "count"])
        writer.writeheader()
        for reason, count in counter.most_common():
            writer.writerow({"reason": reason, "count": count})


if __name__ == "__main__":
    raise SystemExit(main())
