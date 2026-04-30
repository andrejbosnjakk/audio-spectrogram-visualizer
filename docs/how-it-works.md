# How It Works

This project is built around one idea: sound is easier to understand when you can see how its frequencies change over time.

## From Waveform To Spectrogram

A WAV file or microphone stream starts as waveform samples:

```text
[-0.01, 0.03, 0.08, 0.02, ...]
```

Those numbers are amplitude values. On their own, they tell you how air pressure moves over time, but they do not directly show pitch, harmonics, drums, noise, or timbre.

To see frequency content, the program repeatedly takes a short chunk of audio and runs an FFT:

```text
audio chunk -> FFT -> frequency magnitudes
```

Each FFT result becomes one vertical slice of the spectrogram.

## Window Size

The FFT window size controls how much audio is analyzed at once.

Larger windows:

- separate nearby frequencies better
- make stable pitches look cleaner
- smear fast timing changes more

Smaller windows:

- react faster to transients
- make drums and clicks sharper in time
- blur nearby frequencies together more

Good starting values:

```bash
--window-size 4096
--window-size 8192
```

## Hop Size

The hop size controls how far the analysis window moves each frame.

Smaller hop sizes update more often and look smoother:

```bash
--hop-size 256
```

Larger hop sizes do less work and move faster through the song:

```bash
--hop-size 1024
```

## Why A Window Function Is Used

The program multiplies each chunk by a Hann window before the FFT.

Without a window, the edges of each chunk can create artificial frequency energy. The Hann window fades the edges down, which gives a cleaner frequency estimate.

## Decibels

Raw FFT magnitudes can be awkward because audio has a huge dynamic range. The project converts magnitudes to decibels:

```text
dB = 20 * log10(magnitude)
```

That makes quiet and loud details fit into the same color scale.

## Why Full Songs Look Busy

Mixed music is dense. A rock or pop song usually contains:

- drums
- bass
- vocals
- guitars or synths
- cymbals
- reverb
- compression
- harmonics from every instrument

So the spectrogram can look crowded. This is not a bug. It is the sound mix showing up honestly.

To make it clearer, try limiting the visible range:

```bash
python main.py --max-display-freq 6000
```

Or use a larger FFT:

```bash
python main.py --window-size 8192 --hop-size 512
```

## Bandpass Filtering

A bandpass filter keeps frequencies between two cutoffs and reduces everything outside them.

In this project, the live filter uses:

```python
butter(..., btype="bandpass", output="sos")
```

SOS means second-order sections. This is a stable way to represent filters, especially compared with one large high-order filter polynomial.

The result is useful for isolating parts of the sound:

- low bass energy
- vocal range
- cymbal/high-frequency texture
- individual harmonics in simple tones

## Microphone Mode

Microphone mode uses a callback input stream. The callback captures audio blocks into a small queue, while the plotting loop consumes those blocks and updates the spectrogram.

That design avoids blocking the audio driver while Matplotlib is busy drawing.
