"""Minimal standard MIDI file writer with no external dependencies."""
from __future__ import annotations

import struct
import time as _time
from typing import Dict, List, Optional, Tuple

CHROMATIC = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
_NOTE_SEMITONE = {n: i for i, n in enumerate(CHROMATIC)}
_FLAT_SHARP = {'db': 'c#', 'eb': 'd#', 'fb': 'e', 'gb': 'f#', 'ab': 'g#', 'bb': 'a#', 'cb': 'b'}


def _note_to_midi(name: str) -> int:
    name = name.lower().strip()
    octave = int(name[-1])
    pitch = _FLAT_SHARP.get(name[:-1], name[:-1])
    return _NOTE_SEMITONE[pitch] + (octave + 1) * 12


def _vlq(value: int) -> bytes:
    parts = [value & 0x7F]
    value >>= 7
    while value:
        parts.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(parts))


class MidiRecorder:
    """Records note events and exports to a standard MIDI file."""

    def __init__(self, bpm: int = 120):
        self.bpm = bpm
        self.ticks_per_beat = 480
        self.events: List[Tuple[float, str, str, float]] = []
        self._active: Dict[str, float] = {}

    @property
    def has_events(self) -> bool:
        return len(self.events) > 0

    def note_on(self, timestamp: float, note: str, velocity: float = 1.0) -> None:
        if note in self._active:
            self.events.append((timestamp, 'off', note, 0.0))
        self.events.append((timestamp, 'on', note, velocity))
        self._active[note] = timestamp

    def note_off(self, timestamp: float, note: str) -> None:
        if note in self._active:
            self.events.append((timestamp, 'off', note, 0.0))
            del self._active[note]

    def finalize(self, timestamp: Optional[float] = None) -> None:
        t = timestamp if timestamp is not None else _time.perf_counter()
        for note in list(self._active):
            self.events.append((t, 'off', note, 0.0))
        self._active.clear()

    def save(self, path: str) -> str:
        if not self.events:
            return path
        self.events.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))
        start = self.events[0][0]
        track = bytearray()
        us_per_beat = int(60_000_000 / self.bpm)
        track += _vlq(0) + b'\xFF\x51\x03' + struct.pack('>I', us_per_beat)[1:]
        prev_tick = 0
        for ts, etype, note, vel in self.events:
            tick = int((ts - start) * self.bpm / 60 * self.ticks_per_beat)
            delta = max(0, tick - prev_tick)
            prev_tick = tick
            try:
                midi_note = _note_to_midi(note)
            except (KeyError, ValueError, IndexError):
                continue
            midi_vel = max(1, min(127, int(vel * 127)))
            status = 0x90 if etype == 'on' else 0x80
            track += _vlq(delta) + bytes([status, midi_note, midi_vel if etype == 'on' else 0])
        track += _vlq(0) + b'\xFF\x2F\x00'
        with open(path, 'wb') as f:
            f.write(b'MThd' + struct.pack('>I', 6) + struct.pack('>HHH', 0, 1, self.ticks_per_beat))
            f.write(b'MTrk' + struct.pack('>I', len(track)))
            f.write(track)
        return path

    def reset(self) -> None:
        self.events.clear()
        self._active.clear()
