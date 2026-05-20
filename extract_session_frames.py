from __future__ import annotations

import argparse
import json
import shutil
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover - depends on local install
    cv2 = None
    np = None

try:
    from PIL import Image, ImageDraw
except Exception as exc:  # pragma: no cover - depends on local install
    raise SystemExit("Pillow could not be imported. Install it with `pip install Pillow`.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract fresh screenshots from an AirDesk recorded session")
    parser.add_argument("session_dir", help="Session directory containing raw_video.avi and frames.jsonl")
    parser.add_argument("--count", type=int, default=5, help="Number of evenly spaced frames to export")
    parser.add_argument("--output-dir", default="extracted_frames", help="Output directory inside the session")
    parser.add_argument("--no-overlay", action="store_true", help="Do not draw zones, fingertips, and hits")
    parser.add_argument("--include-hit-frames", action="store_true", help="Also export frames where hits happened")
    parser.add_argument("--keep-existing", action="store_true", help="Do not clear the output directory before writing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_dir = Path(args.session_dir)
    frames_path = session_dir / "frames.jsonl"
    video_path = session_dir / "raw_video.avi"
    if not frames_path.exists():
        raise SystemExit(f"Missing {frames_path}")
    if not video_path.exists():
        raise SystemExit(f"Missing {video_path}")

    entries = load_jsonl(frames_path)
    if not entries:
        raise SystemExit(f"No frames found in {frames_path}")

    output_dir = session_dir / args.output_dir
    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_indices = selected_frame_indices(entries, args.count, args.include_hit_frames)
    frame_by_index = {int(entry["frame_index"]): entry for entry in entries}
    written = 0
    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise SystemExit(f"Could not open {video_path}")
        for frame_index in frame_indices:
            ok, image = read_video_frame(cap, frame_index)
            if not ok or image is None:
                print(f"Skipping frame {frame_index}: video read failed")
                continue
            entry = frame_by_index.get(frame_index)
            if entry and not args.no_overlay:
                draw_overlay(image, entry)
            output_path = output_dir / f"frame_{frame_index:04d}.jpg"
            cv2.imwrite(str(output_path), image)
            written += 1
        cap.release()
    else:
        jpeg_frames = load_mjpeg_frames(video_path)
        for frame_index in frame_indices:
            if frame_index >= len(jpeg_frames):
                print(f"Skipping frame {frame_index}: only {len(jpeg_frames)} JPEG frames found")
                continue
            image = Image.open(BytesIO(jpeg_frames[frame_index])).convert("RGB")
            entry = frame_by_index.get(frame_index)
            if entry and not args.no_overlay:
                draw_overlay_pil(image, entry)
            output_path = output_dir / f"frame_{frame_index:04d}.jpg"
            image.save(output_path, quality=92)
            written += 1

    print(f"Wrote {written} screenshots to {output_dir}")
    return 0


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def selected_frame_indices(entries: List[Dict[str, object]], count: int, include_hit_frames: bool) -> List[int]:
    max_index = len(entries) - 1
    count = max(1, min(count, len(entries)))
    indices = {round(i * max_index / max(1, count - 1)) for i in range(count)}
    if include_hit_frames:
        indices.update(int(entry["frame_index"]) for entry in entries if entry.get("hits"))
    return sorted(int(index) for index in indices)


def read_video_frame(cap, frame_index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    return cap.read()


def load_mjpeg_frames(video_path: Path) -> List[bytes]:
    data = video_path.read_bytes()
    frames: List[bytes] = []
    offset = 0
    while True:
        start = data.find(b"\xff\xd8", offset)
        if start < 0:
            break
        end = data.find(b"\xff\xd9", start + 2)
        if end < 0:
            break
        frames.append(data[start : end + 2])
        offset = end + 2
    if not frames:
        raise SystemExit(
            f"No embedded JPEG frames found in {video_path}. "
            "Install OpenCV and rerun this script to decode non-MJPG video."
        )
    return frames


def draw_overlay(image, entry: Dict[str, object]) -> None:
    zones = entry.get("zones", [])
    hands = entry.get("hands", [])
    hits = entry.get("hits", [])
    for zone in zones:
        draw_zone(image, zone)
    for hand in hands:
        draw_hand(image, hand)
    for hit in hits:
        draw_hit(image, hit, zones)
    draw_status(image, entry)


def draw_zone(image, zone: Dict[str, object]) -> None:
    polygon = zone.get("polygon")
    label = str(zone.get("label", ""))
    if polygon:
        points = np.array([(int(x), int(y)) for x, y in polygon], dtype=np.int32)
        cv2.polylines(image, [points], True, (40, 220, 255), 2)
        cx = int(sum(point[0] for point in polygon) / len(polygon))
        cy = int(sum(point[1] for point in polygon) / len(polygon))
    else:
        x1, y1 = int(zone["x1"]), int(zone["y1"])
        x2, y2 = int(zone["x2"]), int(zone["y2"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (40, 220, 255), 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.putText(image, label, (cx - 18, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3)
    cv2.putText(image, label, (cx - 18, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1)


def draw_hand(image, hand: Dict[str, object]) -> None:
    for idx, point in enumerate(hand.get("landmarks", [])):
        if idx not in {4, 8, 12, 16, 20}:
            continue
        x, y = int(point[0]), int(point[1])
        color = (50, 220, 255) if idx == 4 else (80, 255, 120)
        cv2.circle(image, (x, y), 7, (0, 0, 0), 2)
        cv2.circle(image, (x, y), 5, color, -1)
        cv2.putText(image, str(idx), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_hit(image, hit: Dict[str, object], zones: Iterable[Dict[str, object]]) -> None:
    sound_id = hit.get("sound_id")
    zone = next((candidate for candidate in zones if candidate.get("sound_id") == sound_id), None)
    if zone:
        polygon = zone.get("polygon")
        if polygon:
            points = np.array([(int(x), int(y)) for x, y in polygon], dtype=np.int32)
            overlay = image.copy()
            cv2.fillConvexPoly(overlay, points, (0, 210, 255))
            cv2.addWeighted(overlay, 0.28, image, 0.72, 0, image)
        else:
            x1, y1 = int(zone["x1"]), int(zone["y1"])
            x2, y2 = int(zone["x2"]), int(zone["y2"])
            overlay = image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 210, 255), -1)
            cv2.addWeighted(overlay, 0.28, image, 0.72, 0, image)
    text = f"HIT {hit.get('note_id')} H{hit.get('hand_id')} F{hit.get('finger_id')}"
    cv2.putText(image, text, (24, image.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3)
    cv2.putText(image, text, (24, image.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 235, 255), 2)


def draw_status(image, entry: Dict[str, object]) -> None:
    text = f"frame={entry.get('frame_index')} hits={len(entry.get('hits', []))} mode={entry.get('mode')}"
    cv2.rectangle(image, (12, 12), (420, 54), (20, 20, 20), -1)
    cv2.putText(image, text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2)


def draw_overlay_pil(image: Image.Image, entry: Dict[str, object]) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    zones = entry.get("zones", [])
    hands = entry.get("hands", [])
    hits = entry.get("hits", [])
    for zone in zones:
        draw_zone_pil(draw, zone)
    for hit in hits:
        draw_hit_pil(draw, image.size, hit, zones)
    for hand in hands:
        draw_hand_pil(draw, hand)
    draw_status_pil(draw, image.size, entry)


def draw_zone_pil(draw: ImageDraw.ImageDraw, zone: Dict[str, object]) -> None:
    polygon = zone.get("polygon")
    label = str(zone.get("label", ""))
    if polygon:
        points = [(int(x), int(y)) for x, y in polygon]
        draw.line(points + [points[0]], fill=(40, 220, 255, 230), width=2)
        cx = int(sum(point[0] for point in points) / len(points))
        cy = int(sum(point[1] for point in points) / len(points))
    else:
        x1, y1 = int(zone["x1"]), int(zone["y1"])
        x2, y2 = int(zone["x2"]), int(zone["y2"])
        draw.rectangle((x1, y1, x2, y2), outline=(40, 220, 255, 230), width=2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    draw.text((cx - 16, cy - 8), label, fill=(255, 255, 255, 230), stroke_width=2, stroke_fill=(0, 0, 0, 220))


def draw_hand_pil(draw: ImageDraw.ImageDraw, hand: Dict[str, object]) -> None:
    for idx, point in enumerate(hand.get("landmarks", [])):
        if idx not in {4, 8, 12, 16, 20}:
            continue
        x, y = int(point[0]), int(point[1])
        color = (50, 220, 255, 240) if idx == 4 else (80, 255, 120, 240)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=(0, 0, 0, 240), width=2)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        draw.text((x + 8, y - 12), str(idx), fill=color, stroke_width=1, stroke_fill=(0, 0, 0, 220))


def draw_hit_pil(
    draw: ImageDraw.ImageDraw,
    image_size: Tuple[int, int],
    hit: Dict[str, object],
    zones: Iterable[Dict[str, object]],
) -> None:
    sound_id = hit.get("sound_id")
    zone = next((candidate for candidate in zones if candidate.get("sound_id") == sound_id), None)
    if zone:
        polygon = zone.get("polygon")
        if polygon:
            points = [(int(x), int(y)) for x, y in polygon]
            draw.polygon(points, fill=(0, 210, 255, 70))
            draw.line(points + [points[0]], fill=(0, 230, 255, 255), width=4)
        else:
            x1, y1 = int(zone["x1"]), int(zone["y1"])
            x2, y2 = int(zone["x2"]), int(zone["y2"])
            draw.rectangle((x1, y1, x2, y2), fill=(0, 210, 255, 70), outline=(0, 230, 255, 255), width=4)
    width, height = image_size
    text = f"HIT {hit.get('note_id')} H{hit.get('hand_id')} F{hit.get('finger_id')}"
    draw.text((24, height - 42), text, fill=(0, 235, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 240))


def draw_status_pil(draw: ImageDraw.ImageDraw, image_size: Tuple[int, int], entry: Dict[str, object]) -> None:
    text = f"frame={entry.get('frame_index')} hits={len(entry.get('hits', []))} mode={entry.get('mode')}"
    draw.rectangle((12, 12, 420, 54), fill=(20, 20, 20, 210))
    draw.text((24, 28), text, fill=(245, 245, 245, 255))


if __name__ == "__main__":
    raise SystemExit(main())
