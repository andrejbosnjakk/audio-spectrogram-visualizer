from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import sounddevice as sd
import argparse
import shutil
import subprocess
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path
from scipy.signal import butter, sosfilt, sosfilt_zi

import cv2


def parse_sounddevice_device(value):
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return value


def get_input_sample_rate(input_device=None, fallback_sample_rate=44100):
    try:
        device_info = sd.query_devices(input_device, kind="input")
    except sd.PortAudioError:
        return fallback_sample_rate

    return int(device_info.get("default_samplerate", fallback_sample_rate))


def get_input_channel_count(input_device=None, fallback_channels=1):
    try:
        device_info = sd.query_devices(input_device, kind="input")
    except sd.PortAudioError:
        return fallback_channels

    return max(1, int(device_info.get("max_input_channels", fallback_channels)))


def choose_mic_block(indata, mic_channel):
    if mic_channel == "mix" or indata.shape[1] == 1:
        return np.mean(indata, axis=1).astype(np.float32, copy=False).copy()

    if mic_channel == "auto":
        channel_rms = np.sqrt(np.mean(indata ** 2, axis=0))
        channel_index = int(np.argmax(channel_rms))
        return indata[:, channel_index].copy()

    channel_index = min(max(int(mic_channel) - 1, 0), indata.shape[1] - 1)
    return indata[:, channel_index].copy()


def load_audio(path):
    sample_rate, data = wavfile.read(path)

    # Normalize depending on WAV format
    if data.dtype == np.int16:
        normalized_data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        normalized_data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        normalized_data = (data.astype(np.float32) - 128) / 128.0
    else:
        normalized_data = data.astype(np.float32)

    # Convert stereo to mono
    if normalized_data.ndim == 2:
        mono_data = normalized_data.mean(axis=1)
    else:
        mono_data = normalized_data

    return sample_rate, mono_data


def compute_spectrogram(mono_data, sample_rate, window_size=2048, hop_size=512):
    spectrogram_data = []
    window = np.hanning(window_size)

    for i in range(0, len(mono_data) - window_size, hop_size):
        chunk = mono_data[i:i + window_size]
        windowed_chunk = chunk * window

        fft_chunk = rfft(windowed_chunk)
        magnitude = np.abs(fft_chunk)

        spectrogram_data.append(magnitude)

    spectrogram_data = np.array(spectrogram_data)

    db_data = 20 * np.log10(spectrogram_data.T + 1e-10)

    return db_data

def create_bandpass_filter(sample_rate, low_cutoff, high_cutoff, order=4):
    nyquist = sample_rate / 2

    low = max(float(low_cutoff), 1.0)
    high = min(float(high_cutoff), nyquist * 0.999)

    if low >= high:
        raise ValueError("Bandpass low cutoff must be below the high cutoff.")

    sos = butter(
        order,
        [low, high],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )

    zi = sosfilt_zi(sos)

    return sos, zi


def apply_bandpass_filter(signal, sample_rate, low_cutoff, high_cutoff, order=4):
    sos, _ = create_bandpass_filter(sample_rate, low_cutoff, high_cutoff, order)
    return sosfilt(sos, signal).astype(np.float32, copy=False)

def plot_spectrogram(db_data, sample_rate, window_size, hop_size, mono_data):
    fig, ax = plt.subplots(figsize=(12, 6))

    num_frames = db_data.shape[1]
    x_max = num_frames * hop_size / sample_rate

    img = ax.imshow(
        db_data,
        aspect="auto",
        origin="lower",
        extent=[0, x_max, 0, sample_rate / 2],
        cmap="inferno"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram")

    fig.colorbar(img, label="Magnitude (dB)")

    first_click_freq = None
    first_line = None

    def on_click(event):
        nonlocal first_click_freq

        if event.ydata is None:
            return
        
        freq = event.ydata

        if first_click_freq is None:
            first_click_freq = freq

            first_line = ax.axhline(freq, linestyle="--", linewidth=2)
            fit.canvas.draw()

            print(f"First cutoff at {first_click_freq:.2f} Hz")

        else:
            low_cutoff = min(first_click_freq, freq)
            high_cutoff = max(first_click_freq, freq)

            ax.axhline(freq, linestyle="--", linewidth=2)
            ax.axhspan(low_cutoff, high_cutoff, alpha=0.25)
            fig.canvas.draw()

            print(f"Filtering {low_cutoff:.2f} to {high_cutoff:.2f} Hz")

            filtered = apply_bandpass_filter(
                mono_data, 
                sample_rate, 
                low_cutoff,
                high_cutoff
            )

            sd.play(filtered, sample_rate)

            first_click_freq = None

    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()

def create_rolling_spectrogram_plot(
    sample_rate,
    window_size,
    hop_size,
    history_frames,
    use_pyplot=True,
    max_display_freq=8000,
):
    frequencies = rfftfreq(window_size, 1 / sample_rate)

    # Only display useful frequency bins
    max_display_freq = min(max_display_freq, sample_rate / 2)
    display_bin_count = max(1, np.searchsorted(frequencies, max_display_freq))

    spectrogram_buffer = np.full(
        (display_bin_count, history_frames),
        -100.0
    )

    if use_pyplot:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = Figure(figsize=(12, 6))
        FigureCanvasAgg(fig)
        ax = fig.subplots()

    time_width = history_frames * hop_size / sample_rate

    img = ax.imshow(
        spectrogram_buffer,
        aspect="auto",
        origin="lower",
        extent=[-time_width, 0, 0, max_display_freq],
        cmap="inferno",
        vmin=-90,
        vmax=-20,
    )

    ax.set_xlabel("Time before now")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Live rolling spectrogram")
    ax.grid(alpha=0.18, linewidth=0.6)

    fig.colorbar(img, label="Magnitude (dBFS)")

    return fig, ax, img, spectrogram_buffer


def update_rolling_spectrogram_frame(
    spectrogram_buffer,
    img,
    fft_chunk_data,
    window,
):
    fft_chunk = rfft(fft_chunk_data * window)

    magnitude = (np.abs(fft_chunk) * 2) / np.sum(window)
    dB = 20 * np.log10(magnitude + 1e-10)

    visible_bins = spectrogram_buffer.shape[0]

    spectrogram_buffer = np.roll(spectrogram_buffer, -1, axis=1)
    spectrogram_buffer[:, -1] = dB[:visible_bins]
    img.set_data(spectrogram_buffer)

    return spectrogram_buffer


def find_ffmpeg_executable():
    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def write_audio_wav(path, mono_data, sample_rate):
    clipped_audio = np.clip(mono_data, -1.0, 1.0)
    int_audio = (clipped_audio * 32767).astype(np.int16)
    wavfile.write(path, sample_rate, int_audio)


def make_shareable_video(video_path, output_path, mono_data=None, sample_rate=None):
    ffmpeg_path = find_ffmpeg_executable()

    if ffmpeg_path is None:
        raise RuntimeError(
            "Shareable MP4 export needs FFmpeg. Install it on PATH or run "
            "`python -m pip install imageio-ffmpeg`."
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        command = [
            ffmpeg_path,
            "-y",
            "-i",
            str(video_path),
        ]

        if mono_data is not None and sample_rate is not None:
            temp_audio_path = Path(temp_dir) / "audio.wav"
            write_audio_wav(temp_audio_path, mono_data, sample_rate)
            command.extend(
                [
                    "-i",
                    str(temp_audio_path),
                ]
            )

        command.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ]
        )

        if mono_data is not None and sample_rate is not None:
            command.extend(
                [
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-shortest",
                ]
            )
        else:
            command.append("-an")

        command.append(str(output_path))

        result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg could not create the shareable MP4:\n{result.stderr}")


def add_audio_to_video(video_path, mono_data, sample_rate, output_path):
    make_shareable_video(video_path, output_path, mono_data, sample_rate)


def make_silent_shareable_video(video_path, output_path):
    make_shareable_video(video_path, output_path)


def build_frame_positions(max_samples, sample_rate, fps, window_size):
    frame_total = max(0, int((max_samples / sample_rate) * fps))
    last_start = max(0, max_samples - window_size)

    for frame_index in range(frame_total):
        frame_time = frame_index / fps
        sample_start = min(int(frame_time * sample_rate), last_start)
        yield frame_index + 1, sample_start, frame_total


def live_microphone_spectrogram(
    sample_rate=None,
    window_size=4096,
    hop_size=512,
    history_frames=180,
    max_display_freq=8000,
    input_device=None,
    mic_gain_db=0.0,
    mic_channel="auto",
):
    sample_rate = sample_rate or get_input_sample_rate(input_device)
    input_channels = get_input_channel_count(input_device)
    window = np.hanning(window_size)
    running = {"active": True}
    mic_gain = 10 ** (mic_gain_db / 20)

    fig, ax, img, spectrogram_buffer = create_rolling_spectrogram_plot(
        sample_rate,
        window_size,
        hop_size,
        history_frames,
        use_pyplot=True,
        max_display_freq=max_display_freq,
    )

    audio_buffer = np.zeros(window_size, dtype=np.float32)
    pending_blocks = deque(maxlen=128)
    stream_statuses = deque(maxlen=16)

    def on_key_press(event):
        if event.key == "q":
            running["active"] = False
            plt.close(fig)

    def on_audio(indata, frames, time, status):
        if status:
            stream_statuses.append(str(status))

        mono_block = choose_mic_block(indata, mic_channel)
        pending_blocks.append(mono_block)

    fig.canvas.mpl_connect("key_press_event", on_key_press)

    plt.ion()
    plt.show()
    plt.pause(0.1)

    device_info = sd.query_devices(input_device, kind="input")
    print(
        f"Listening to microphone: {device_info['name']} at {sample_rate} Hz "
        f"({input_channels} channel{'s' if input_channels != 1 else ''}, "
        f"{mic_channel} channel mode)"
    )
    print("Press q in the spectrogram window to stop microphone mode.")

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=input_channels,
        blocksize=hop_size,
        dtype="float32",
        device=input_device,
        latency="high",
        callback=on_audio,
    )

    frame_count = 0

    with stream:
        while running["active"] and plt.fignum_exists(fig.number):
            if stream_statuses:
                print(f"Input status: {stream_statuses.popleft()}")

            if not pending_blocks:
                plt.pause(0.01)
                continue

            while pending_blocks:
                new_samples = pending_blocks.popleft() * mic_gain
                block_len = len(new_samples)

                if block_len >= window_size:
                    audio_buffer[:] = new_samples[-window_size:]
                else:
                    audio_buffer[:-block_len] = audio_buffer[block_len:]
                    audio_buffer[-block_len:] = new_samples

                spectrogram_buffer = update_rolling_spectrogram_frame(
                    spectrogram_buffer,
                    img,
                    audio_buffer,
                    window,
                )

                frame_count += 1

                if frame_count % 80 == 0:
                    rms = np.sqrt(np.mean(new_samples ** 2))
                    peak = np.max(np.abs(new_samples))
                    print(
                        f"Mic level: {20 * np.log10(rms + 1e-10):.1f} dBFS "
                        f"(peak {20 * np.log10(peak + 1e-10):.1f} dBFS)"
                    )

            fig.canvas.draw_idle()
            plt.pause(0.001)

    plt.ioff()


def test_microphone_level(input_device=None, sample_rate=None, seconds=5):
    sample_rate = sample_rate or get_input_sample_rate(input_device)
    input_channels = get_input_channel_count(input_device)
    device_info = sd.query_devices(input_device, kind="input")

    print(
        f"Testing microphone: {device_info['name']} at {sample_rate} Hz "
        f"({input_channels} channel{'s' if input_channels != 1 else ''})"
    )
    print("Make sound near the microphone now...")

    recording = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=input_channels,
        dtype="float32",
        device=input_device,
    )
    sd.wait()

    mono_data = np.mean(recording, axis=1)
    rms = np.sqrt(np.mean(mono_data ** 2))
    peak = np.max(np.abs(mono_data))

    print(f"Mixed RMS:  {20 * np.log10(rms + 1e-10):.1f} dBFS")
    print(f"Mixed peak: {20 * np.log10(peak + 1e-10):.1f} dBFS")

    if input_channels > 1:
        for channel_index in range(input_channels):
            channel = recording[:, channel_index]
            channel_rms = np.sqrt(np.mean(channel ** 2))
            channel_peak = np.max(np.abs(channel))
            print(
                f"Channel {channel_index + 1}: "
                f"RMS {20 * np.log10(channel_rms + 1e-10):.1f} dBFS, "
                f"peak {20 * np.log10(channel_peak + 1e-10):.1f} dBFS"
            )


def live_rolling_spectogram(
    mono_data,
    sample_rate,
    window_size=4096,
    hop_size=512,
    history_frames=180,
    max_display_freq=8000,
):
    window = np.hanning(window_size)

    filter_state = {
        "sos": None,
        "zi": None,
    }

    fig, ax, img, spectrogram_buffer = create_rolling_spectrogram_plot(
        sample_rate,
        window_size,
        hop_size,
        history_frames,
        use_pyplot=True,
        max_display_freq=max_display_freq,
    )

    selected_band = {
        "first_click": None,
        "low": None,
        "high": None,
        "artists": [],
    }

    def clear_band_visuals():
        for artist in selected_band["artists"]:
            artist.remove()
        selected_band["artists"].clear()

    def on_click(event):
        if event.inaxes != ax or event.ydata is None:
            return

        freq = event.ydata

        # Right click clears the filter
        if event.button == 3:
            selected_band["first_click"] = None
            selected_band["low"] = None
            selected_band["high"] = None
            clear_band_visuals()
            fig.canvas.draw_idle()
            print("Bandpass filter cleared.")
            return

        # First left click
        if selected_band["first_click"] is None:
            clear_band_visuals()

            selected_band["low"] = None
            selected_band["high"] = None
            selected_band["first_click"] = freq

            line = ax.axhline(freq, linestyle="--", linewidth=2)
            selected_band["artists"].append(line)

            fig.canvas.draw_idle()
            print(f"First cutoff: {freq:.2f} Hz")

        # Second left click
        else:
            low = min(selected_band["first_click"], freq)
            high = max(selected_band["first_click"], freq)

            selected_band["low"] = low
            selected_band["high"] = high
            selected_band["first_click"] = None

            filter_state["sos"], filter_state["zi"] = create_bandpass_filter(
                sample_rate,
                low,
                high,
            )

            line = ax.axhline(freq, linestyle="--", linewidth=2)
            span = ax.axhspan(low, high, alpha=0.25)

            selected_band["artists"].append(line)
            selected_band["artists"].append(span)

            fig.canvas.draw_idle()
            print(f"Live bandpass active: {low:.2f} Hz to {high:.2f} Hz")

    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.ion()
    plt.show()
    plt.pause(0.1)

    stream = sd.OutputStream(samplerate=sample_rate, channels=1)
    stream.start()

    frame_count = 0

    try:
        for i in range(0, len(mono_data) - window_size, hop_size):

            if not plt.fignum_exists(fig.number):
                break

            play_chunk = mono_data[i:i + hop_size]

            # If a band is selected, filter the live playback chunk
            if selected_band["low"] is not None and selected_band["high"] is not None:
                play_chunk = apply_bandpass_filter(
                    play_chunk,
                    sample_rate,
                    selected_band["low"],
                    selected_band["high"],
                )

            stream.write(play_chunk.reshape(-1, 1))

            fft_chunk_data = mono_data[i:i + window_size]

            spectrogram_buffer = update_rolling_spectrogram_frame(
                spectrogram_buffer,
                img,
                fft_chunk_data,
                window,
            )

            frame_count += 1

            if frame_count % 4 == 0:
                plt.pause(0.001)

    finally:
        stream.stop()
        stream.close()
        plt.ioff()


def export_rolling_spectrogram_video(
    mono_data,
    sample_rate,
    output_folder,
    window_size=4096,
    hop_size=512,
    history_frames=180,
    duration_seconds=None,
    include_audio=True,
    video_fps=30,
    max_display_freq=8000,
):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if duration_seconds is None:
        max_samples = len(mono_data)
        duration_label = "full"
    else:
        max_samples = min(len(mono_data), sample_rate * duration_seconds)
        duration_label = f"{duration_seconds}s"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_folder / f"rolling_spectrogram_{duration_label}_{timestamp}.mp4"
    video_only_path = output_path.with_name(f"{output_path.stem}_video_only{output_path.suffix}")

    window = np.hanning(window_size)

    fig, ax, img, spectrogram_buffer = create_rolling_spectrogram_plot(
        sample_rate,
        window_size,
        hop_size,
        history_frames,
        use_pyplot=False,
        max_display_freq=max_display_freq,
    )
    fig.canvas.draw()

    frame_width, frame_height = fig.canvas.get_width_height()
    writer = cv2.VideoWriter(
        str(video_only_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        plt.close(fig)
        raise RuntimeError("Could not open the MP4 video writer.")

    try:
        for frame_index, i, frame_total in build_frame_positions(
            max_samples,
            sample_rate,
            video_fps,
            window_size,
        ):
            fft_chunk_data = mono_data[i:i + window_size]

            spectrogram_buffer = update_rolling_spectrogram_frame(
                spectrogram_buffer,
                img,
                fft_chunk_data,
                window,
            )

            fig.canvas.draw()
            rgba_frame = np.asarray(fig.canvas.buffer_rgba())
            bgr_frame = cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2BGR)
            writer.write(bgr_frame)

            if frame_index % 100 == 0 or frame_index == frame_total:
                print(f"Exported {frame_index}/{frame_total} frames")
    finally:
        writer.release()
        plt.close(fig)

    if include_audio:
        audio_data = mono_data[:max_samples]
        add_audio_to_video(video_only_path, audio_data, sample_rate, output_path)
        video_only_path.unlink(missing_ok=True)
    else:
        make_silent_shareable_video(video_only_path, output_path)
        video_only_path.unlink(missing_ok=True)

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Play or export a rolling spectrogram animation.")
    parser.add_argument(
        "--audio",
        default=r"assets\audio\synthetic_demo.wav",
        help="Path to the WAV file to visualize.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Number of seconds to play or export. Omit it to use the whole song.",
    )
    parser.add_argument(
        "--export-video",
        action="store_true",
        help="Export the rolling spectrogram animation as an MP4 video with audio.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Export a silent MP4 video.",
    )
    parser.add_argument(
        "--output-folder",
        default="exports",
        help="Folder where exported videos are saved.",
    )
    parser.add_argument(
        "--play-after-export",
        action="store_true",
        help="Play the live animation after exporting the video.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=4096,
        help="FFT window size. Larger values separate frequencies better but smear time more.",
    )
    parser.add_argument(
        "--hop-size",
        type=int,
        default=512,
        help="Audio samples between spectrogram updates. Smaller values look smoother.",
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=180,
        help="Number of spectrogram columns kept on screen.",
    )
    parser.add_argument(
        "--max-display-freq",
        type=int,
        default=8000,
        help="Highest frequency shown in Hz.",
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Use microphone input instead of a WAV file.",
    )
    parser.add_argument(
        "--test-mic",
        action="store_true",
        help="Record a few seconds from the microphone and print RMS/peak levels.",
    )
    parser.add_argument(
        "--input-device",
        default=None,
        help="Microphone device index or name. Use --list-devices to see choices.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Microphone sample rate. Omit it to use the input device default.",
    )
    parser.add_argument(
        "--mic-gain-db",
        type=float,
        default=0.0,
        help="Boost microphone visualization level in dB.",
    )
    parser.add_argument(
        "--mic-channel",
        default="auto",
        help="Microphone channel to visualize: auto, mix, 1, 2, etc.",
    )
    parser.add_argument(
        "--test-seconds",
        type=float,
        default=5.0,
        help="Number of seconds to record for --test-mic.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available audio devices and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    input_device = parse_sounddevice_device(args.input_device)

    if args.test_mic:
        test_microphone_level(
            input_device=input_device,
            sample_rate=args.sample_rate,
            seconds=args.test_seconds,
        )
        return

    if args.mic:
        live_microphone_spectrogram(
            sample_rate=args.sample_rate,
            window_size=args.window_size,
            hop_size=args.hop_size,
            history_frames=args.history_frames,
            max_display_freq=args.max_display_freq,
            input_device=input_device,
            mic_gain_db=args.mic_gain_db,
            mic_channel=args.mic_channel,
        )
        return

    sample_rate, mono_data = load_audio(args.audio)
    if args.duration is not None:
        mono_data = mono_data[:sample_rate * args.duration]

    window_size = args.window_size
    hop_size = args.hop_size
    history_frames = args.history_frames

    if args.export_video:
        output_path = export_rolling_spectrogram_video(
            mono_data,
            sample_rate,
            args.output_folder,
            window_size=window_size,
            hop_size=hop_size,
            history_frames=history_frames,
            duration_seconds=args.duration,
            include_audio=not args.no_audio,
            max_display_freq=args.max_display_freq,
        )
        print(f"Saved video to {output_path}")

    if not args.export_video or args.play_after_export:
        live_rolling_spectogram(
            mono_data,
            sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            history_frames=history_frames,
            max_display_freq=args.max_display_freq,
        )


if __name__ == "__main__":
    main()
