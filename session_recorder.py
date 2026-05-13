from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import cv2

from hand_tracker import HandLandmarks
from hit_detector import HitEvent
from instrument import Zone


class SessionRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        metadata: Dict[str, Any],
        record_video: bool = True,
        fps: float = 30.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.record_video = record_video
        self.fps = fps
        self.frame_count = 0
        self.start_time = time.perf_counter()
        self._video_writer = None
        self._frames_file = (self.output_dir / "frames.jsonl").open("w", encoding="utf-8")
        self._metadata = {
            **metadata,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "format": "aird_esktop_session_v1",
            "files": {
                "frames": "frames.jsonl",
                "video": "raw_video.avi" if record_video else None,
            },
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(_json_safe(self._metadata), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def record_frame(
        self,
        frame,
        timestamp: float,
        fps: float,
        metrics,
        hands: Iterable[HandLandmarks],
        zones: Iterable[Zone],
        hits: Iterable[HitEvent],
        diagnostics: Iterable[Dict[str, Any]],
        gesture: str,
        loop_state: str,
        mode: str,
    ) -> None:
        if self.record_video:
            self._write_video_frame(frame, fps)
        entry = {
            "frame_index": self.frame_count,
            "timestamp": timestamp,
            "relative_time": timestamp - self.start_time,
            "fps": fps,
            "mode": mode,
            "frame_shape": list(frame.shape),
            "metrics": _dataclass_to_dict(metrics),
            "hands": [_hand_to_dict(hand) for hand in hands],
            "zones": [_zone_to_dict(zone) for zone in zones],
            "hits": [_hit_to_dict(hit) for hit in hits],
            "diagnostics": list(diagnostics),
            "gesture": gesture,
            "loop_state": loop_state,
        }
        self._frames_file.write(json.dumps(_json_safe(entry), ensure_ascii=False) + "\n")
        self.frame_count += 1

    def close(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
        self._frames_file.close()
        summary = {
            "frames": self.frame_count,
            "duration_seconds": time.perf_counter() - self.start_time,
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _write_video_frame(self, frame, fps: float) -> None:
        if self._video_writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            video_path = str(self.output_dir / "raw_video.avi")
            self._video_writer = cv2.VideoWriter(video_path, fourcc, max(1.0, fps or self.fps), (width, height))
        self._video_writer.write(frame)


def _hand_to_dict(hand: HandLandmarks) -> Dict[str, Any]:
    return {
        "hand_id": hand.hand_id,
        "label": hand.label,
        "landmarks": [[int(x), int(y), float(z)] for x, y, z in hand.landmarks],
        "normalized_landmarks": [[float(x), float(y), float(z)] for x, y, z in hand.normalized_landmarks],
    }


def _zone_to_dict(zone: Zone) -> Dict[str, Any]:
    return {
        "label": zone.label,
        "sound_id": zone.sound_id,
        "kind": zone.kind,
        "x1": zone.x1,
        "y1": zone.y1,
        "x2": zone.x2,
        "y2": zone.y2,
        "press_y": zone.press_y,
        "release_y": zone.release_y,
    }


def _hit_to_dict(hit: HitEvent) -> Dict[str, Any]:
    return {
        "note_id": hit.note_id,
        "sound_id": hit.sound_id,
        "zone_label": hit.zone_label,
        "finger_id": hit.finger_id,
        "hand_id": hit.hand_id,
        "timestamp": hit.timestamp,
        "velocity": hit.velocity,
        "volume": hit.volume,
    }


def _dataclass_to_dict(value) -> Dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return value
    return {}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    return value
