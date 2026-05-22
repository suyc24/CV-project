from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_NOTES = ("C4", "D4", "E4", "F4", "G4", "A4", "B4")
NUMBER_KEYS = "1234567890"

cv2 = None
np = None


@dataclass
class Annotation:
    onset: float
    note: str
    frame_index: int
    source: str
    score_index: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simple video onset annotator for AirDesk piano tests. "
            "Open a video/session, pause near each key press, then press Enter "
            "for the next score note or 1-7 to label notes directly."
        )
    )
    parser.add_argument("video_or_session", help="Video file, or an AirDesk session directory containing raw_video.avi")
    parser.add_argument("--output", default=None, help="Output CSV. Defaults to <video_dir>/annotations.csv")
    parser.add_argument("--score", default=None, help="Optional score file. Can be CSV with note column, or plain text notes")
    parser.add_argument("--notes", default=",".join(DEFAULT_NOTES), help="Comma-separated direct-label notes for 1..N keys")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-display-width", type=int, default=1400)
    parser.add_argument("--playback-speed", type=float, default=0.65)
    parser.add_argument("--seek-seconds", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing output annotations")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_cv_deps()
    video_path = resolve_video_path(Path(args.video_or_session))
    output_path = Path(args.output) if args.output else video_path.with_name("annotations.csv")
    notes = parse_note_list(args.notes)
    score_notes = load_score_notes(Path(args.score)) if args.score else []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1e-6:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_index = int(np.clip(args.start_frame, 0, max(0, total_frames - 1))) if total_frames else args.start_frame
    playing = False
    playback_speed = max(0.05, args.playback_speed)
    annotations = [] if args.overwrite else load_annotations(output_path)
    score_index = next_score_index(annotations)
    last_message = "Ready"

    window_name = "AirDesk annotation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = read_frame(cap, frame_index)
            if not ok:
                if total_frames:
                    frame_index = max(0, total_frames - 1)
                    ok, frame = read_frame(cap, frame_index)
                if not ok:
                    raise SystemExit("Could not read video frame")

            display = draw_overlay(
                frame,
                video_path=video_path,
                output_path=output_path,
                frame_index=frame_index,
                total_frames=total_frames,
                fps=fps,
                annotations=annotations,
                notes=notes,
                score_notes=score_notes,
                score_index=score_index,
                playing=playing,
                message=last_message,
                max_display_width=args.max_display_width,
            )
            cv2.imshow(window_name, display)
            delay_ms = max(1, int(1000.0 / fps / playback_speed)) if playing else 0
            key = cv2.waitKeyEx(delay_ms)

            if key == -1:
                if playing:
                    frame_index = advance_frame(frame_index, total_frames, 1)
                continue

            key_char = chr(key & 0xFF).lower() if 0 <= (key & 0xFF) < 256 else ""
            if key_char == "q" or key == 27:
                save_annotations(output_path, annotations, video_path, fps, score_notes)
                last_message = f"Saved {len(annotations)} annotations"
                break
            if key_char == " ":
                playing = not playing
                last_message = "Playing" if playing else "Paused"
            elif key_char == "s":
                save_annotations(output_path, annotations, video_path, fps, score_notes)
                last_message = f"Saved {len(annotations)} annotations"
            elif key_char in {"\r", "\n"} or key in {10, 13}:
                if score_index >= len(score_notes):
                    last_message = "No next score note; use number keys for direct labels"
                else:
                    note = score_notes[score_index]
                    annotations.append(Annotation(frame_index / fps, note, frame_index, "score", score_index))
                    score_index += 1
                    save_annotations(output_path, annotations, video_path, fps, score_notes)
                    last_message = f"Marked {note} at {frame_index / fps:.3f}s"
            elif key_char in NUMBER_KEYS:
                note_index = NUMBER_KEYS.index(key_char)
                if note_index < len(notes):
                    note = notes[note_index]
                    annotations.append(Annotation(frame_index / fps, note, frame_index, "direct", None))
                    save_annotations(output_path, annotations, video_path, fps, score_notes)
                    last_message = f"Marked {note} at {frame_index / fps:.3f}s"
                else:
                    last_message = f"No note mapped to key {key_char}"
            elif key in {8, 127} or key_char in {"u", "z"}:
                if annotations:
                    removed = annotations.pop()
                    if removed.score_index is not None:
                        score_index = min(score_index, removed.score_index)
                    save_annotations(output_path, annotations, video_path, fps, score_notes)
                    last_message = f"Undid {removed.note} at {removed.onset:.3f}s"
                else:
                    last_message = "Nothing to undo"
            elif key_char in {",", "a"}:
                playing = False
                frame_index = advance_frame(frame_index, total_frames, -1)
                last_message = "Step back"
            elif key_char in {".", "d"}:
                playing = False
                frame_index = advance_frame(frame_index, total_frames, 1)
                last_message = "Step forward"
            elif key_char == "[":
                playing = False
                frame_index = advance_frame(frame_index, total_frames, -int(round(args.seek_seconds * fps)))
                last_message = f"Seek -{args.seek_seconds:g}s"
            elif key_char == "]":
                playing = False
                frame_index = advance_frame(frame_index, total_frames, int(round(args.seek_seconds * fps)))
                last_message = f"Seek +{args.seek_seconds:g}s"
            elif key_char in {"-", "_"}:
                playback_speed = max(0.1, playback_speed * 0.75)
                last_message = f"Speed {playback_speed:.2f}x"
            elif key_char in {"=", "+", "w"}:
                playback_speed = min(4.0, playback_speed * 1.25)
                last_message = f"Speed {playback_speed:.2f}x"

            if playing:
                frame_index = advance_frame(frame_index, total_frames, 1)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Saved {len(annotations)} annotations to {output_path}")
    return 0


def ensure_cv_deps() -> None:
    global cv2, np
    try:
        import cv2 as cv2_module
        import numpy as np_module
    except ImportError as exc:
        raise SystemExit(
            "OpenCV/numpy is required for video annotation. Activate the project venv "
            "and run `pip install -r requirements.txt` first."
        ) from exc
    cv2 = cv2_module
    np = np_module


def resolve_video_path(path: Path) -> Path:
    if path.is_dir():
        for name in ("raw_video.avi", "raw_video.mp4", "video.mp4", "video.avi"):
            candidate = path / name
            if candidate.exists():
                return candidate
        raise SystemExit(f"No video found in session directory: {path}")
    return path


def parse_note_list(text: str) -> list[str]:
    notes = [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]
    return notes or list(DEFAULT_NOTES)


def load_score_notes(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Score file not found: {path}")
    if path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            if reader.fieldnames:
                note_column = "note" if "note" in reader.fieldnames else reader.fieldnames[-1]
                return [row[note_column].strip() for row in reader if row.get(note_column, "").strip()]
    text = path.read_text(encoding="utf-8-sig")
    for separator in [",", "\n", "\t"]:
        text = text.replace(separator, " ")
    return [part.strip() for part in text.split(" ") if part.strip()]


def load_annotations(path: Path) -> list[Annotation]:
    if not path.exists():
        return []
    rows: list[Annotation] = []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if not row.get("onset") or not row.get("note"):
                continue
            score_index = row.get("score_index")
            rows.append(
                Annotation(
                    onset=float(row["onset"]),
                    note=row["note"],
                    frame_index=int(float(row.get("frame_index", 0) or 0)),
                    source=row.get("source", "direct"),
                    score_index=int(score_index) if score_index not in {None, ""} else None,
                )
            )
    return rows


def save_annotations(
    path: Path,
    annotations: list[Annotation],
    video_path: Path,
    fps: float,
    score_notes: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    annotations = sorted(annotations, key=lambda item: (item.onset, item.frame_index))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["onset", "note", "frame_index", "source", "score_index", "video", "fps"],
        )
        writer.writeheader()
        for item in annotations:
            writer.writerow(
                {
                    "onset": f"{item.onset:.6f}",
                    "note": item.note,
                    "frame_index": item.frame_index,
                    "source": item.source,
                    "score_index": "" if item.score_index is None else item.score_index,
                    "video": str(video_path),
                    "fps": f"{fps:.6f}",
                }
            )
    sidecar = {
        "video": str(video_path),
        "fps": fps,
        "annotations": len(annotations),
        "score_notes": score_notes,
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")


def next_score_index(annotations: list[Annotation]) -> int:
    indices = [item.score_index for item in annotations if item.score_index is not None]
    return max(indices) + 1 if indices else 0


def read_frame(cap: cv2.VideoCapture, frame_index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
    return cap.read()


def advance_frame(frame_index: int, total_frames: int, delta: int) -> int:
    next_index = frame_index + delta
    if total_frames > 0:
        return int(np.clip(next_index, 0, total_frames - 1))
    return max(0, next_index)


def draw_overlay(
    frame,
    video_path: Path,
    output_path: Path,
    frame_index: int,
    total_frames: int,
    fps: float,
    annotations: list[Annotation],
    notes: list[str],
    score_notes: list[str],
    score_index: int,
    playing: bool,
    message: str,
    max_display_width: int,
):
    image = frame.copy()
    h, w = image.shape[:2]
    time_s = frame_index / fps
    panel_h = 170
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.62, image, 0.38, 0)

    progress = frame_index / max(1, total_frames - 1) if total_frames else 0.0
    cv2.rectangle(image, (20, panel_h - 18), (w - 20, panel_h - 10), (70, 70, 70), -1)
    cv2.rectangle(image, (20, panel_h - 18), (20 + int((w - 40) * progress), panel_h - 10), (80, 190, 255), -1)

    state = "PLAY" if playing else "PAUSE"
    next_note = score_notes[score_index] if score_index < len(score_notes) else "-"
    total_text = str(total_frames) if total_frames else "?"
    lines = [
        f"{state}  t={time_s:.3f}s  frame={frame_index}/{total_text}  annotations={len(annotations)}",
        f"Next score note: {next_note}   Output: {output_path.name}",
        f"Direct keys: {format_note_keys(notes)}",
        "Space play/pause | Enter mark next score note | 1-7 direct note | ,/. frame | [/ ] seek | Backspace undo | S save | Q quit",
        message,
    ]
    y = 28
    for idx, line in enumerate(lines):
        color = (255, 255, 255) if idx < 4 else (0, 220, 255)
        cv2.putText(image, line, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
        y += 29

    draw_recent_annotations(image, annotations, fps, time_s)
    if max_display_width > 0 and w > max_display_width:
        scale = max_display_width / float(w)
        image = cv2.resize(image, (max_display_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return image


def format_note_keys(notes: list[str]) -> str:
    pairs = []
    for idx, note in enumerate(notes[:10]):
        pairs.append(f"{NUMBER_KEYS[idx]}={note}")
    return " ".join(pairs)


def draw_recent_annotations(image, annotations: list[Annotation], fps: float, time_s: float) -> None:
    h, w = image.shape[:2]
    recent = [item for item in annotations if abs(item.onset - time_s) < 3.0][-12:]
    if not recent:
        return
    x1, y1 = 20, h - 38
    x2 = w - 20
    cv2.rectangle(image, (x1, y1 - 36), (x2, y1 + 8), (0, 0, 0), -1)
    cv2.addWeighted(image, 0.96, image, 0.04, 0)
    for item in recent:
        x = int(np.interp(item.onset, [time_s - 3.0, time_s + 3.0], [x1, x2]))
        cv2.line(image, (x, y1 - 30), (x, y1), (0, 220, 255), 2)
        cv2.putText(image, item.note, (x - 12, y1 - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)


if __name__ == "__main__":
    raise SystemExit(main())
