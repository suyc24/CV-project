from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


TWINKLE_NOTES = (
    "C4", "C4", "G4", "G4", "A4", "A4", "G4",
    "F4", "F4", "E4", "E4", "D4", "D4", "C4",
    "G4", "G4", "F4", "F4", "E4", "E4", "D4",
    "G4", "G4", "F4", "F4", "E4", "E4", "D4",
    "C4", "C4", "G4", "G4", "A4", "A4", "G4",
    "F4", "F4", "E4", "E4", "D4", "D4", "C4",
)

SINGLE_FINGER_CHECK_NOTES = (
    ("C4",) * 6
    + ("D4",) * 6
    + ("E4",) * 6
    + ("F4",) * 6
    + ("G4",) * 6
)

ADJACENT_KEY_NOTES = ("C4", "D4", "E4", "F4", "G4", "F4", "E4", "D4", "C4") * 2

GUIDE_SEQUENCES = {
    "twinkle": TWINKLE_NOTES,
    "single_finger_checks": SINGLE_FINGER_CHECK_NOTES,
    "adjacent_keys": ADJACENT_KEY_NOTES,
}


@dataclass(frozen=True)
class GuideEvent:
    onset: float
    note: str


class GuidedNoteSequence:
    def __init__(
        self,
        name: str,
        notes: tuple[str, ...],
        first_onset: float = 1.0,
        notes_per_second: float = 2.0,
    ) -> None:
        self.name = name
        self.notes = tuple(notes)
        self.first_onset = max(0.0, float(first_onset))
        self.notes_per_second = max(0.01, float(notes_per_second))
        self.interval = 1.0 / self.notes_per_second
        self.events = tuple(
            GuideEvent(onset=self.first_onset + index * self.interval, note=note)
            for index, note in enumerate(self.notes)
        )

    @property
    def duration(self) -> float:
        if not self.events:
            return 0.0
        return self.events[-1].onset + self.interval

    def write_annotations(self, output_dir: str | Path, overwrite: bool = True) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        annotations_path = output_path / "annotations.csv"
        if annotations_path.exists() and not overwrite:
            return annotations_path
        with annotations_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["onset", "note", "frame_index"])
            writer.writeheader()
            for event in self.events:
                writer.writerow({"onset": f"{event.onset:.3f}", "note": event.note, "frame_index": ""})
        spec = {
            "guide_sequence": self.name,
            "first_onset": self.first_onset,
            "notes_per_second": self.notes_per_second,
            "interval": self.interval,
            "note_count": len(self.notes),
            "duration": self.duration,
        }
        (output_path / "guide_spec.json").write_text(
            json.dumps(spec, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return annotations_path

    def state_at(self, relative_time: float) -> dict[str, object]:
        if not self.events:
            return {
                "phase": "empty",
                "current_index": None,
                "current_note": "",
                "next_note": "",
                "time_to_next": 0.0,
                "progress": 0.0,
            }
        if relative_time < self.first_onset:
            return {
                "phase": "countdown",
                "current_index": None,
                "current_note": "",
                "next_note": self.events[0].note,
                "time_to_next": self.first_onset - relative_time,
                "progress": 0.0,
            }

        raw_index = int((relative_time - self.first_onset) // self.interval)
        if raw_index >= len(self.events):
            return {
                "phase": "done",
                "current_index": len(self.events) - 1,
                "current_note": self.events[-1].note,
                "next_note": "",
                "time_to_next": 0.0,
                "progress": 1.0,
            }
        next_index = raw_index + 1
        next_note = self.events[next_index].note if next_index < len(self.events) else ""
        next_onset = self.events[next_index].onset if next_index < len(self.events) else self.duration
        return {
            "phase": "playing",
            "current_index": raw_index,
            "current_note": self.events[raw_index].note,
            "next_note": next_note,
            "time_to_next": max(0.0, next_onset - relative_time),
            "progress": raw_index / max(1, len(self.events) - 1),
        }

    def draw_overlay(self, frame: np.ndarray, relative_time: float) -> None:
        state = self.state_at(relative_time)
        height, width = frame.shape[:2]
        panel_w = min(560, max(360, width - 80))
        panel_h = 170
        x1 = max(20, (width - panel_w) // 2)
        y1 = 18
        x2 = x1 + panel_w
        y2 = y1 + panel_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (14, 18, 24), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 150, 180), 1)

        phase = str(state["phase"])
        if phase == "countdown":
            title = f"Get ready: {state['time_to_next']:.1f}s"
            note = str(state["next_note"])
            subtitle = "First note"
            color = (80, 220, 245)
        elif phase == "done":
            title = "Done"
            note = "END"
            subtitle = "Hold still, then press q"
            color = (100, 230, 130)
        elif phase == "playing":
            current_index = int(state["current_index"]) if state["current_index"] is not None else 0
            title = f"{current_index + 1}/{len(self.events)}   next in {state['time_to_next']:.1f}s"
            note = str(state["current_note"])
            subtitle = f"Next: {state['next_note'] or 'END'}"
            color = (80, 220, 245)
        else:
            title = "Guide"
            note = ""
            subtitle = ""
            color = (220, 220, 220)

        cv2.putText(frame, title, (x1 + 22, y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2)
        note_scale = 2.2 if len(note) <= 3 else 1.8
        size, _ = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, note_scale, 4)
        note_x = x1 + (panel_w - size[0]) // 2
        cv2.putText(frame, note, (note_x, y1 + 105), cv2.FONT_HERSHEY_SIMPLEX, note_scale, color, 4)
        cv2.putText(frame, subtitle, (x1 + 22, y1 + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (225, 230, 235), 2)

        progress = float(state["progress"])
        bar_x1, bar_y1 = x1 + 20, y2 - 16
        bar_x2, bar_y2 = x2 - 20, y2 - 8
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (65, 70, 80), -1)
        fill_x = int(bar_x1 + (bar_x2 - bar_x1) * max(0.0, min(1.0, progress)))
        cv2.rectangle(frame, (bar_x1, bar_y1), (fill_x, bar_y2), color, -1)


def create_guide(
    sequence_name: str,
    first_onset: float = 1.0,
    notes_per_second: float = 2.0,
) -> Optional[GuidedNoteSequence]:
    if sequence_name in {"", "none", None}:
        return None
    try:
        notes = GUIDE_SEQUENCES[sequence_name]
    except KeyError as exc:
        known = ", ".join(sorted(GUIDE_SEQUENCES))
        raise ValueError(f"Unknown guide sequence {sequence_name!r}; expected one of: {known}") from exc
    return GuidedNoteSequence(
        name=sequence_name,
        notes=notes,
        first_onset=first_onset,
        notes_per_second=notes_per_second,
    )
