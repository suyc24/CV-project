"""Scale definitions and keyboard remapping utilities."""
from __future__ import annotations

from typing import Dict, List

CHROMATIC = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']

SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "blues": [0, 3, 5, 6, 7, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
}

SCALE_NAMES = list(SCALE_INTERVALS.keys())

DEFAULT_KEYS = ["c4", "d4", "e4", "f4", "g4", "a4", "b4", "c5", "d5", "e5"]
KEY_POSITIONS = {k: i for i, k in enumerate(DEFAULT_KEYS)}


def build_scale_notes(scale_name: str, root_semitone: int = 0, start_octave: int = 4, count: int = 15) -> List[str]:
    intervals = SCALE_INTERVALS[scale_name]
    notes = []
    for i in range(count):
        degree = i % len(intervals)
        octave_wrap = i // len(intervals)
        semitone = root_semitone + intervals[degree] + octave_wrap * 12
        octave = start_octave + semitone // 12
        note_name = CHROMATIC[semitone % 12]
        notes.append(f"{note_name}{octave}")
    return notes


def build_scale_remap(scale_name: str) -> Dict[str, str]:
    if scale_name == "major":
        return {}
    target = build_scale_notes(scale_name, count=len(DEFAULT_KEYS))
    return dict(zip(DEFAULT_KEYS, target))


def get_chord_notes(extended_scale: List[str], key_index: int) -> List[str]:
    notes = [extended_scale[key_index]]
    if key_index + 2 < len(extended_scale):
        notes.append(extended_scale[key_index + 2])
    if key_index + 4 < len(extended_scale):
        notes.append(extended_scale[key_index + 4])
    return notes
