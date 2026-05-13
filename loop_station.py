from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Set

import config
from hit_detector import HitEvent


class LoopState(str, Enum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"


@dataclass
class LoopEvent:
    relative_time: float
    sound_id: str
    volume: float
    note_id: str = ""


class LoopStation:
    def __init__(self) -> None:
        self.state = LoopState.IDLE
        self.events: List[LoopEvent] = []
        self.record_start_time = 0.0
        self.loop_duration = 0.0
        self.playback_start_time = 0.0
        self.paused_position = 0.0
        self._played_indices: Set[int] = set()
        self._last_loop_index = 0

    @property
    def event_count(self) -> int:
        return len(self.events)

    def start_recording(self, current_time: float) -> None:
        self.events.clear()
        self.state = LoopState.RECORDING
        self.record_start_time = current_time
        self.loop_duration = 0.0
        self.paused_position = 0.0
        self._played_indices.clear()
        self._last_loop_index = 0

    def stop_recording(self, current_time: float) -> None:
        if self.state != LoopState.RECORDING:
            return
        if not self.events:
            self.clear()
            return
        duration_from_clock = current_time - self.record_start_time
        duration_from_events = max(event.relative_time for event in self.events) + 0.25
        self.loop_duration = max(config.LOOP_MIN_DURATION, duration_from_clock, duration_from_events)
        self.events.sort(key=lambda event: event.relative_time)
        self.state = LoopState.PAUSED
        self.paused_position = 0.0
        self._played_indices.clear()

    def toggle_recording(self, current_time: float) -> None:
        if self.state == LoopState.RECORDING:
            self.stop_recording(current_time)
        else:
            self.start_recording(current_time)

    def toggle_playback(self, current_time: float) -> None:
        if self.state == LoopState.RECORDING:
            self.stop_recording(current_time)
            return
        if not self.events:
            self.state = LoopState.IDLE
            return
        if self.state == LoopState.PLAYING:
            self.paused_position = self._position(current_time)
            self.state = LoopState.PAUSED
            return

        self.state = LoopState.PLAYING
        self.playback_start_time = current_time - self.paused_position
        self._last_loop_index = int(max(0.0, current_time - self.playback_start_time) // self.loop_duration)
        self._played_indices = {
            idx for idx, event in enumerate(self.events) if event.relative_time < self.paused_position
        }

    def clear(self) -> None:
        self.state = LoopState.IDLE
        self.events.clear()
        self.record_start_time = 0.0
        self.loop_duration = 0.0
        self.playback_start_time = 0.0
        self.paused_position = 0.0
        self._played_indices.clear()
        self._last_loop_index = 0

    def record_event(self, hit_event: HitEvent) -> None:
        if self.state != LoopState.RECORDING:
            return
        relative_time = max(0.0, hit_event.timestamp - self.record_start_time)
        self.events.append(
            LoopEvent(
                relative_time=relative_time,
                sound_id=hit_event.sound_id,
                volume=hit_event.volume,
                note_id=hit_event.note_id,
            )
        )

    def update(self, current_time: float, audio_engine) -> None:
        if self.state != LoopState.PLAYING or not self.events or self.loop_duration <= 0:
            return

        elapsed = max(0.0, current_time - self.playback_start_time)
        loop_index = int(elapsed // self.loop_duration)
        position = elapsed % self.loop_duration
        if loop_index != self._last_loop_index:
            self._last_loop_index = loop_index
            self._played_indices.clear()

        for idx, event in enumerate(self.events):
            if idx in self._played_indices:
                continue
            if event.relative_time <= position:
                audio_engine.play(event.sound_id, event.volume)
                self._played_indices.add(idx)

    def _position(self, current_time: float) -> float:
        if self.loop_duration <= 0:
            return 0.0
        return (current_time - self.playback_start_time) % self.loop_duration
