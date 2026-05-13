from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a simple HTML report for an AirDesk session")
    parser.add_argument("session_dir")
    parser.add_argument("--output", default="report.html")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_dir = Path(args.session_dir)
    frames = load_jsonl(session_dir / "frames.jsonl")
    if not frames:
        raise SystemExit("No frames found")

    metrics = [frame.get("metrics", {}) for frame in frames]
    online_hits = [hit for frame in frames for hit in frame.get("hits", [])]
    diagnostics = [diag for frame in frames for diag in frame.get("diagnostics", [])]
    reason_counts = Counter(str(diag.get("reason", "unknown")) for diag in diagnostics)
    notes = Counter(str(hit.get("note_id", "unknown")) for hit in online_hits)
    fingers = Counter(f"H{hit.get('hand_id')} F{hit.get('finger_id')}" for hit in online_hits)

    report = render_report(
        session_dir=session_dir,
        frames=frames,
        metrics=metrics,
        online_hits=online_hits,
        reason_counts=reason_counts,
        notes=notes,
        fingers=fingers,
    )
    output_path = session_dir / args.output
    output_path.write_text(report, encoding="utf-8")
    write_metrics_csv(session_dir / "frame_metrics.csv", frames)
    print(f"Wrote {output_path}")
    return 0


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def render_report(session_dir: Path, frames, metrics, online_hits, reason_counts, notes, fingers) -> str:
    duration = frames[-1].get("relative_time", 0.0) if frames else 0.0
    avg_fps = average(frame.get("fps", 0.0) for frame in frames)
    avg_luma = average(metric.get("mean_luma", 0.0) for metric in metrics)
    avg_over = average(metric.get("overexposed_ratio", 0.0) for metric in metrics) * 100.0
    avg_sharp = average(metric.get("sharpness", 0.0) for metric in metrics)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>AirDesk Session Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #202124; }}
    table {{ border-collapse: collapse; margin: 14px 0 28px; min-width: 420px; }}
    th, td {{ border: 1px solid #d0d0d0; padding: 6px 10px; text-align: left; }}
    th {{ background: #f2f2f2; }}
    .metric {{ display: inline-block; margin: 0 18px 18px 0; padding: 10px 14px; background: #f7f7f7; border: 1px solid #ddd; }}
  </style>
</head>
<body>
  <h1>AirDesk Session Report</h1>
  <p>{html.escape(str(session_dir))}</p>
  <div class="metric"><b>Frames</b><br>{len(frames)}</div>
  <div class="metric"><b>Duration</b><br>{duration:.2f}s</div>
  <div class="metric"><b>Avg FPS</b><br>{avg_fps:.1f}</div>
  <div class="metric"><b>Hits</b><br>{len(online_hits)}</div>
  <div class="metric"><b>Avg luma</b><br>{avg_luma:.1f}</div>
  <div class="metric"><b>Avg overexposed</b><br>{avg_over:.2f}%</div>
  <div class="metric"><b>Avg sharpness</b><br>{avg_sharp:.0f}</div>
  <h2>Miss Reasons</h2>
  {table_from_counter(reason_counts, "Reason")}
  <h2>Notes Hit</h2>
  {table_from_counter(notes, "Note")}
  <h2>Fingers Hit</h2>
  {table_from_counter(fingers, "Finger")}
</body>
</html>
"""


def table_from_counter(counter: Counter, label: str) -> str:
    rows = "\n".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{count}</td></tr>"
        for key, count in counter.most_common()
    )
    return f"<table><tr><th>{label}</th><th>Count</th></tr>{rows}</table>"


def average(values) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0


def write_metrics_csv(path: Path, frames) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["frame_index", "relative_time", "fps", "mean_luma", "overexposed_ratio", "sharpness", "hit_count"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for frame in frames:
            metrics = frame.get("metrics", {})
            writer.writerow(
                {
                    "frame_index": frame.get("frame_index"),
                    "relative_time": frame.get("relative_time"),
                    "fps": frame.get("fps"),
                    "mean_luma": metrics.get("mean_luma"),
                    "overexposed_ratio": metrics.get("overexposed_ratio"),
                    "sharpness": metrics.get("sharpness"),
                    "hit_count": len(frame.get("hits", [])),
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
