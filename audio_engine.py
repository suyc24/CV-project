from __future__ import annotations

import math
from typing import Dict

import numpy as np

import config
from utils import clamp


class AudioEngine:
    """Self-contained pygame audio engine with generated drum and piano sounds."""

    def __init__(self) -> None:
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
        sound = self.sounds.get(sound_id)
        if sound is None:
            return
        channel = self._pygame.mixer.find_channel(force=True)
        if channel is None:
            return
        channel.set_volume(clamp(float(volume), 0.0, 1.0))
        channel.play(sound)

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

        for sound_id, frequency in self._piano_frequencies().items():
            self.sounds[sound_id] = self._to_sound(self._piano(frequency))

    def _piano_frequencies(self) -> Dict[str, float]:
        semitones = {
            "c": -9,
            "d": -7,
            "e": -5,
            "f": -4,
            "g": -2,
            "a": 0,
            "b": 2,
        }
        frequencies: Dict[str, float] = {}
        for octave in (4, 5, 6):
            for note, offset in semitones.items():
                if octave == 6 and note != "c":
                    continue
                semitone_from_a4 = offset + (octave - 4) * 12
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
