# Microphone Troubleshooting

Microphone debugging can be strangely annoying because Python may open a device successfully while Windows still sends silence.

Start with the built-in test:

```bash
python main.py --list-devices
python main.py --test-mic --input-device DEVICE_NUMBER
```

Talk, clap, or tap near the microphone while the test runs.

## Reading The Level

Useful signal usually creates peaks somewhere around:

```text
-40 dBFS to -10 dBFS
```

Very quiet values like these usually mean silence or noise floor:

```text
-90 dBFS
-80 dBFS
```

If you see only noise-floor values while making sound, the issue is probably outside the spectrogram code.

## Windows Checks

Open:

```text
Settings -> System -> Sound -> Input
```

Pick the microphone and confirm the Windows input meter moves.

Then check:

```text
Settings -> Privacy & security -> Microphone
```

Enable:

```text
Microphone access
Let apps access your microphone
Let desktop apps access your microphone
```

Python launched from a terminal counts as a desktop app.

## Bluetooth Headsets

Bluetooth devices often expose separate playback and microphone endpoints.

For AirPods or similar devices, use a device named something like:

```text
Headset
Hands-Free
Input
```

Do not choose:

```text
Headphones
```

The headphones endpoint is usually playback-only.

## Try Different Backends

The same physical microphone can appear several times:

```text
MME
Windows DirectSound
Windows WASAPI
Windows WDM-KS
```

Try the different input devices with:

```bash
python main.py --test-mic --input-device 1
python main.py --test-mic --input-device 5
python main.py --test-mic --input-device 9
```

When one device shows a real peak, use that same device for live mode:

```bash
python main.py --mic --input-device DEVICE_NUMBER --mic-gain-db 10
```

## Channel Selection

Some microphone arrays expose two channels. The visualizer defaults to:

```bash
--mic-channel auto
```

This picks the loudest channel. You can force a channel:

```bash
python main.py --mic --input-device 5 --mic-channel 1
python main.py --mic --input-device 5 --mic-channel 2
```

## If It Still Does Not Work

Try recording with Windows Sound Recorder. If that is silent too, the problem is Windows settings, drivers, permissions, or hardware.

If Sound Recorder works but Python does not, run:

```bash
python main.py --list-devices
```

Then test every input endpoint that looks plausible.
