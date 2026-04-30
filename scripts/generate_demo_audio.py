from pathlib import Path

import numpy as np
from scipy.io import wavfile


SAMPLE_RATE = 44100
OUTPUT_PATH = Path("assets/audio/synthetic_demo.wav")


def fade(signal, fade_seconds=0.03):
    fade_length = int(SAMPLE_RATE * fade_seconds)
    if fade_length == 0 or len(signal) < fade_length * 2:
        return signal

    envelope = np.ones_like(signal)
    envelope[:fade_length] = np.linspace(0, 1, fade_length)
    envelope[-fade_length:] = np.linspace(1, 0, fade_length)
    return signal * envelope


def sine(frequency, seconds, amplitude=0.35):
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    return amplitude * np.sin(2 * np.pi * frequency * t)


def sweep(start_frequency, end_frequency, seconds, amplitude=0.35):
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    frequency = np.linspace(start_frequency, end_frequency, len(t))
    phase = 2 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
    return amplitude * np.sin(phase)


def chord(frequencies, seconds, amplitude=0.28):
    parts = [sine(frequency, seconds, amplitude / len(frequencies)) for frequency in frequencies]
    return np.sum(parts, axis=0)


def noise_bursts(seconds, burst_count=8, amplitude=0.32):
    length = int(SAMPLE_RATE * seconds)
    signal = np.zeros(length)
    rng = np.random.default_rng(7)
    burst_length = int(0.045 * SAMPLE_RATE)

    for index in np.linspace(0, length - burst_length, burst_count).astype(int):
        burst = rng.normal(0, amplitude, burst_length)
        signal[index:index + burst_length] += fade(burst, 0.006)

    return signal


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    sections = [
        fade(sine(220, 2.0)),
        fade(sine(440, 2.0)),
        fade(chord([261.63, 329.63, 392.0], 2.5)),
        fade(sweep(180, 2400, 4.0)),
        fade(noise_bursts(2.5)),
        fade(chord([196.0, 246.94, 293.66, 392.0], 2.0)),
    ]

    audio = np.concatenate(sections)
    audio = audio / max(np.max(np.abs(audio)), 1e-9)
    audio = (audio * 0.9).astype(np.float32)

    wavfile.write(OUTPUT_PATH, SAMPLE_RATE, audio)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
