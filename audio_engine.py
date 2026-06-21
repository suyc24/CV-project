from __future__ import annotations

import math
from typing import Dict

import numpy as np

import config
from utils import clamp


class AudioEngine:
    """Self-contained pygame audio engine with generated drum and piano sounds."""

    def __init__(self, velocity_curve: str = "linear") -> None:
        self._velocity_curve = velocity_curve
        self._sustain_active = False
        self._sustained_channels: list = []
        try:
            import pygame
        except Exception as exc:  # pragma: no cover - depends on local install
            raise RuntimeError(
                "pygame could not be imported. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        self._pygame = pygame
        try:
            pygame.mixer.pre_init(
                frequency=config.AUDIO_SAMPLE_RATE,
                size=-16,
                channels=config.AUDIO_CHANNELS,
                buffer=config.AUDIO_BUFFER,
            )
            pygame.init()
            pygame.mixer.init()
            pygame.mixer.set_num_channels(config.AUDIO_MIXER_CHANNELS)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "pygame mixer initialization failed. Check that your system has an "
                f"available audio device. Original error: {exc}"
            ) from exc

        self.sounds: Dict[str, object] = {}
        self._build_sounds()

    def play(self, sound_id: str, volume: float = 1.0) -> None:
        actual_id = sound_id
        actual_volume = float(volume)
        is_piano = sound_id not in {"kick", "snare", "hihat", "tom1", "tom2", "crash"} and not sound_id.startswith("metronome")
        if is_piano and self._velocity_curve != "linear":
            if actual_volume < 0.35 and f"{sound_id}_pp" in self.sounds:
                actual_id = f"{sound_id}_pp"
            elif actual_volume >= 0.70 and f"{sound_id}_ff" in self.sounds:
                actual_id = f"{sound_id}_ff"
            if self._velocity_curve == "natural":
                actual_volume = actual_volume ** 0.7
            elif self._velocity_curve == "dramatic":
                actual_volume = actual_volume ** 1.5
        if is_piano and self._sustain_active and f"{sound_id}_sus" in self.sounds:
            actual_id = f"{sound_id}_sus"
        sound = self.sounds.get(actual_id) or self.sounds.get(sound_id)
        if sound is None:
            return
        channel = self._pygame.mixer.find_channel(force=True)
        if channel is None:
            return
        channel.set_volume(clamp(actual_volume, 0.0, 1.0))
        channel.play(sound)
        if is_piano and self._sustain_active:
            self._sustained_channels = [c for c in self._sustained_channels if c.get_busy()]
            self._sustained_channels.append(channel)

    def play_chord(self, notes: list, volume: float = 1.0) -> None:
        for note in notes:
            self.play(note, volume * 0.75)

    def set_velocity_curve(self, curve: str) -> None:
        self._velocity_curve = curve

    def set_sustain(self, active: bool) -> None:
        self._sustain_active = active
        if not active:
            for ch in self._sustained_channels:
                try:
                    ch.fadeout(250)
                except Exception:
                    pass
            self._sustained_channels.clear()

    def close(self) -> None:
        self._pygame.mixer.quit()
        self._pygame.quit()

    def _build_sounds(self) -> None:
        for sound_id, samples in {
            "kick": self._kick(),
            "snare": self._snare(),
            "hihat": self._hihat(),
            "tom1": self._tom(170.0),
            "tom2": self._tom(125.0),
            "crash": self._crash(),
        }.items():
            self.sounds[sound_id] = self._to_sound(samples)

        for sound_id, frequency in self._chromatic_frequencies().items():
            self.sounds[sound_id] = self._to_sound(self._piano(frequency))
            self.sounds[f"{sound_id}_pp"] = self._to_sound(self._piano_soft(frequency))
            self.sounds[f"{sound_id}_ff"] = self._to_sound(self._piano_loud(frequency))
            self.sounds[f"{sound_id}_sus"] = self._to_sound(self._piano_sustained(frequency))

        self.sounds["metronome_tick"] = self._to_sound(self._metronome_tick())
        self.sounds["metronome_accent"] = self._to_sound(self._metronome_accent())

    def _chromatic_frequencies(self) -> Dict[str, float]:
        chromatic = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
        frequencies: Dict[str, float] = {}
        for octave in (4, 5, 6):
            for i, note in enumerate(chromatic):
                if octave == 6 and note != 'c':
                    continue
                semitone_from_a4 = (i - 9) + (octave - 4) * 12
                frequencies[f"{note}{octave}"] = 440.0 * (2.0 ** (semitone_from_a4 / 12.0))
        return frequencies

    def _to_sound(self, mono: np.ndarray):
        mono = np.clip(mono, -1.0, 1.0)
        stereo = np.column_stack([mono, mono])
        pcm = (stereo * 32767).astype(np.int16)
        return self._pygame.sndarray.make_sound(pcm.copy())

    def _time(self, seconds: float) -> np.ndarray:
        count = int(config.AUDIO_SAMPLE_RATE * seconds)
        return np.linspace(0.0, seconds, count, endpoint=False)

    def _kick(self) -> np.ndarray:
        t = self._time(0.42)
        freq = 42.0 + 115.0 * np.exp(-t * 12.0)
        phase = 2.0 * math.pi * np.cumsum(freq) / config.AUDIO_SAMPLE_RATE
        body = np.sin(phase) * np.exp(-t * 7.0)
        click = np.random.default_rng(2).uniform(-1.0, 1.0, len(t)) * np.exp(-t * 90.0) * 0.15
        return 0.95 * body + click

    def _snare(self) -> np.ndarray:
        rng = np.random.default_rng(3)
        t = self._time(0.34)
        noise = rng.uniform(-1.0, 1.0, len(t)) * np.exp(-t * 13.0)
        tone = np.sin(2.0 * math.pi * 185.0 * t) * np.exp(-t * 9.0)
        return 0.50 * noise + 0.35 * tone

    def _hihat(self) -> np.ndarray:
        rng = np.random.default_rng(4)
        t = self._time(0.13)
        noise = rng.uniform(-1.0, 1.0, len(t))
        metallic = np.sin(2.0 * math.pi * 7200.0 * t) + 0.45 * np.sin(2.0 * math.pi * 9400.0 * t)
        return (0.35 * noise + 0.35 * metallic) * np.exp(-t * 35.0)

    def _crash(self) -> np.ndarray:
        rng = np.random.default_rng(5)
        t = self._time(0.95)
        noise = rng.uniform(-1.0, 1.0, len(t))
        shimmer = np.sin(2.0 * math.pi * 5800.0 * t) + np.sin(2.0 * math.pi * 8200.0 * t)
        return (0.28 * noise + 0.16 * shimmer) * np.exp(-t * 3.2)

    def _tom(self, base_frequency: float) -> np.ndarray:
        t = self._time(0.38)
        freq = base_frequency * (0.82 + 0.18 * np.exp(-t * 8.0))
        phase = 2.0 * math.pi * np.cumsum(freq) / config.AUDIO_SAMPLE_RATE
        return np.sin(phase) * np.exp(-t * 6.5)

    def _piano(self, frequency: float) -> np.ndarray:
        t = self._time(0.82)
        attack = np.minimum(1.0, t / 0.015)
        decay = np.exp(-t * 3.2)
        wave = (
            np.sin(2.0 * math.pi * frequency * t)
            + 0.42 * np.sin(2.0 * math.pi * frequency * 2.0 * t)
            + 0.18 * np.sin(2.0 * math.pi * frequency * 3.0 * t)
        )
        return 0.45 * wave * attack * decay

    def _piano_soft(self, frequency: float) -> np.ndarray:
        t = self._time(0.65)
        attack = np.minimum(1.0, t / 0.030)
        decay = np.exp(-t * 4.5)
        wave = (
            np.sin(2.0 * math.pi * frequency * t)
            + 0.15 * np.sin(2.0 * math.pi * frequency * 2.0 * t)
        )
        return 0.30 * wave * attack * decay

    def _piano_loud(self, frequency: float) -> np.ndarray:
        t = self._time(1.0)
        attack = np.minimum(1.0, t / 0.006)
        decay = np.exp(-t * 2.2)
        wave = (
            np.sin(2.0 * math.pi * frequency * t)
            + 0.55 * np.sin(2.0 * math.pi * frequency * 2.0 * t)
            + 0.32 * np.sin(2.0 * math.pi * frequency * 3.0 * t)
            + 0.14 * np.sin(2.0 * math.pi * frequency * 4.0 * t)
            + 0.06 * np.sin(2.0 * math.pi * frequency * 5.0 * t)
        )
        return 0.55 * wave * attack * decay

    def _piano_sustained(self, frequency: float) -> np.ndarray:
        t = self._time(3.0)
        attack = np.minimum(1.0, t / 0.012)
        decay = np.exp(-t * 0.6)
        wave = (
            np.sin(2.0 * math.pi * frequency * t)
            + 0.38 * np.sin(2.0 * math.pi * frequency * 2.0 * t)
            + 0.15 * np.sin(2.0 * math.pi * frequency * 3.0 * t)
        )
        return 0.42 * wave * attack * decay

    def _metronome_tick(self) -> np.ndarray:
        t = self._time(0.03)
        return 0.5 * np.sin(2.0 * math.pi * 1200.0 * t) * np.exp(-t * 80.0)

    def _metronome_accent(self) -> np.ndarray:
        t = self._time(0.05)
        return 0.7 * np.sin(2.0 * math.pi * 1800.0 * t) * np.exp(-t * 50.0)
