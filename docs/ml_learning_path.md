# ML Learning Path: From Spectrograms to Voice Cloning

You already have the right foundation. `main.py` turns raw audio into a time-frequency
representation (FFT -> magnitude -> dB spectrogram) and lets you filter by frequency band.
That representation — turning sound into a 2D array a model can look at — is the entry
point into almost all audio ML, including voice cloning.

This doc is a staged curriculum, not a finished pipeline. Each stage names the concept,
why it matters for the end goal, what you build, and the PyTorch/torchaudio APIs to learn.
Write the code yourself; use this as the map.

Background assumed: you know ML theory (loss functions, gradient descent, backprop) but
haven't written it in PyTorch yet, and you're new to audio-specific ML.

## Stage 0 — PyTorch mechanics (no audio yet)

Goal: get comfortable with the four things every PyTorch project reuses: tensors,
autograd, `nn.Module`, and the training loop. Do this on a toy problem (e.g. fit
`y = sin(x)` with a small MLP) so you're not debugging audio bugs and PyTorch bugs at
the same time.

Learn:
- `torch.Tensor`, `.to(device)`, broadcasting (same rules as NumPy, which you already know)
- `requires_grad`, `.backward()`, `optimizer.step()`, `optimizer.zero_grad()`
- Defining a model as a class: `class Net(nn.Module)` with `__init__` and `forward`
- The canonical loop shape:

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        prediction = model(batch.x)
        loss = loss_fn(prediction, batch.y)
        loss.backward()
        optimizer.step()
```

Once you can explain why `zero_grad()` has to be called every step, move on.

## Stage 1 — Synthetic tone classifier

This is the first ML step in the existing `docs/roadmap.md`. Goal: classify a generated
tone's frequency range (e.g. low / mid / high) from its spectrogram.

What you build:
- A synthetic data generator (extend `scripts/generate_demo_audio.py`'s approach): produce
  short tones at random frequencies with labels for which band they fall in.
- A `Dataset`/`DataLoader` that runs your existing `compute_spectrogram` on each tone and
  returns `(spectrogram_tensor, label)`.
- A small classifier — a 2-3 layer MLP or a tiny CNN over the spectrogram — trained with
  `CrossEntropyLoss`.

Why this stage matters: it's the smallest possible version of "audio in, model decision
out," and it forces you to deal with batching variable audio into fixed-size tensors,
which every later stage needs too.

## Stage 2 — Mel-spectrograms (the real audio ML feature)

Your current spectrogram uses linear frequency bins. Speech/voice models almost always use
**mel-spectrograms** instead, because the mel scale roughly matches human pitch perception
and gives more resolution where speech actually carries information (low/mid frequencies).

Learn: `torchaudio.transforms.MelSpectrogram`, `torchaudio.transforms.AmplitudeToDB`.

What you build: swap your classifier's input (Stage 1) from a linear spectrogram to a
mel-spectrogram and compare. This is also where you'll start using `torchaudio` instead of
hand-rolling FFT calls, since torchaudio's transforms are differentiable and GPU-friendly.

## Stage 3 — Autoencoder on spectrograms

Goal: train a model to compress a mel-spectrogram into a small latent vector and
reconstruct it, with no labels needed (unsupervised).

What you build: an encoder (conv layers shrinking the spectrogram) and a decoder
(transposed conv layers growing it back), trained with MSE loss between input and
reconstruction.

Why this matters: voice cloning models are built almost entirely out of this idea —
encode something (content, or speaker identity) into a compact representation, then decode
it back into a spectrogram. This stage is where you learn what a latent space actually
behaves like, with a problem small enough to debug.

## Stage 4 — Getting back to audio: vocoders

A spectrogram throws away phase information, so turning a (predicted) spectrogram back
into a waveform isn't trivial.

Start with the classical, no-training-required baseline: **Griffin-Lim** iterative phase
reconstruction (`torchaudio.transforms.GriffinLim`). Run your Stage 3 autoencoder's
reconstructed spectrograms through it and listen to the result — it'll sound robotic/
phasey. That artifact is exactly why neural vocoders exist.

Then read up on (don't need to train from scratch) **HiFi-GAN**, the most common neural
vocoder used in modern TTS/voice-cloning pipelines: a GAN that takes mel-spectrograms and
generates a much cleaner waveform. Understanding why it beats Griffin-Lim is the goal here,
not reimplementing it.

## Stage 5 — Speaker identity: what "cloning" actually means

Up to here everything has been about *content* (what sound/word is happening). Voice
cloning is about separating *content* from *speaker identity* (the timbre/voice itself),
so you can keep one and swap the other.

Learn the concept of a **speaker embedding** (sometimes called d-vector or x-vector): a
small network trained (often via contrastive/triplet loss) so that embeddings of the same
speaker are close together and different speakers are far apart, regardless of what's
being said.

What you build: a toy version using your synthetic tones or short recorded clips from
yourself/a friend as two "speakers" — train an embedding network and verify clips from the
same source cluster together (e.g. visualize with PCA/t-SNE).

## Stage 6 — Where real voice cloning pipelines diverge

This is the survey stage, not a build stage — it's where you decide which real pipeline to
aim for, because "voice cloning" branches into two different architectures:

- **TTS-based cloning**: text -> content representation -> (+ speaker embedding from a
  few seconds of reference audio) -> mel-spectrogram -> vocoder -> audio. Examples:
  Tortoise-TTS, Coqui XTTS, VITS variants. This is what's usually meant by "type text, hear
  it in someone's cloned voice."
- **Voice conversion**: source audio (your voice) -> content representation -> swap the
  speaker embedding for the target -> mel-spectrogram -> vocoder -> audio. Examples:
  so-vits-svc, RVC. This is "speak normally, output sounds like someone else."

Training either from scratch to a *good* quality bar takes large datasets and serious
compute — not realistic as a learning project. The practical path once you've done Stages
0-5 (and actually understand every block in the diagram below) is to take a small
pretrained open-source model and fine-tune its speaker embedding/adapter layers on a short
clip of a target voice. That gets you a real, working clone while keeping the part that's
actually worth doing yourself: understanding what every stage of

```text
audio -> mel-spectrogram -> content/speaker encoders -> decoder -> predicted spectrogram -> vocoder -> audio
```

is doing and why, instead of running someone else's script blind.

## Suggested new dependencies

Add when you get to Stage 1:

```text
torch
torchaudio
```

`librosa` is also commonly used for audio loading/feature utilities outside torchaudio,
optional but worth knowing.

One thing to check before Stage 1: your `.venv` is built against a very new Python
(3.14, per the installed package tags). PyTorch wheels lag behind new Python releases —
run `python -m pip install torch torchaudio` and if there's no matching wheel yet, you may
need a separate venv on an older Python (3.11/3.12) for the ML side of this project. This
kind of environment friction is a normal part of ML work, not a sign you're doing it wrong.
