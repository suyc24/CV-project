from __future__ import annotations

import csv
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_guide import GuidedNoteSequence


def test_guided_sequence_writes_annotations_on_recorder_clock():
    guide = GuidedNoteSequence(
        name="test",
        notes=("C4", "D4", "E4"),
        first_onset=1.25,
        notes_per_second=2.0,
    )

    with TemporaryDirectory() as temp_dir:
        annotations_path = guide.write_annotations(temp_dir)
        rows = list(csv.DictReader(annotations_path.open(newline="", encoding="utf-8")))

    assert rows == [
        {"onset": "1.250", "note": "C4", "frame_index": ""},
        {"onset": "1.750", "note": "D4", "frame_index": ""},
        {"onset": "2.250", "note": "E4", "frame_index": ""},
    ]


def test_guided_sequence_state_tracks_current_and_next_note():
    guide = GuidedNoteSequence(
        name="test",
        notes=("C4", "D4", "E4"),
        first_onset=1.0,
        notes_per_second=2.0,
    )

    assert guide.state_at(0.5)["phase"] == "countdown"
    playing = guide.state_at(1.55)
    assert playing["phase"] == "playing"
    assert playing["current_note"] == "D4"
    assert playing["next_note"] == "E4"
    assert guide.state_at(2.6)["phase"] == "done"


if __name__ == "__main__":
    test_guided_sequence_writes_annotations_on_recorder_clock()
    test_guided_sequence_state_tracks_current_and_next_note()
    print("benchmark guide tests passed")
