# Roadmap

This project can stay simple and still be fun, but there are a lot of natural ways to push it further.

## Near-Term Improvements

- Add a cleaner UI panel for frequency range, window size, and color scale.
- Add keyboard shortcuts for common visualization modes.
- Save screenshots from the current live view.
- Add a waveform strip below the spectrogram.
- Add selectable colormaps.
- Add a logarithmic frequency axis for a more music-friendly view.

## Audio Analysis Ideas

- Detect dominant pitch.
- Track harmonics over time.
- Add beat/onset detection.
- Show separate bass, mids, and highs meters.
- Add a piano-key frequency guide.
- Compare two spectrograms side by side.

## Filtering Ideas

- Add lowpass, highpass, bandpass, and notch filter modes.
- Let selected frequency bands stack instead of replacing each other.
- Add filter order controls.
- Export filtered audio.
- Show the filter response curve.

## Microphone Ideas

- Add calibration for room noise.
- Add automatic gain for quiet microphones.
- Add a simple tuner mode.
- Add a voice range detector.
- Add live recording and replay.

## Machine Learning Direction

The spectrogram representation is a natural bridge into audio ML.

A beginner-friendly path:

1. Generate simple synthetic tones.
2. Train a tiny model to classify tone frequency ranges.
3. Train an autoencoder to reconstruct spectrograms.
4. Convert reconstructed spectrograms back into audio.
5. Explore denoising: noisy spectrogram in, cleaner spectrogram out.

For voice or sound generation, the later pipeline would look like:

```text
audio -> mel-spectrogram -> model -> predicted spectrogram -> vocoder -> audio
```

That is a much bigger project, especially for realistic voices, but this repo already contains the first conceptual step: turning sound into a representation a model can learn from.

## Repo Polish Ideas

- Add more synthetic demo clips.
- Add animated examples for different instruments.
- Add a short tutorial notebook.
- Add tests for core signal-processing helpers.
- Package the app as a small CLI tool.
