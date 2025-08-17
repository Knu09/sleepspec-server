import librosa
from pathlib import Path
import soundfile as sf
import numpy as np
import noisereduce as nr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import shutil
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from scipy.signal import medfilt


def check_audio_extension(input_file: Path):
    ext = input_file.suffix.lower()
    if ext != ".wav":
        audio = AudioSegment.from_file(input_file)
        output_file = input_file.with_suffix(".wav")
        audio.export(output_file, format="wav")
        print(f"Converted to WAV: {output_file}")
        return output_file
    return input_file


def load_audio_with_soundfile(input_file):
    y, sr = sf.read(input_file, always_2d=True)  # Ensure 2D output
    y = np.mean(y, axis=1)  # Convert stereo to mono
    print(f"sleepspec-app Sampling rate: {sr} Hz")
    return y, sr


def get_unique_output_dir(base_dir: Path) -> Path:
    base_path = Path(base_dir)
    output_dir = base_path
    counter = 1

    while output_dir.exists():
        output_dir = base_path.parent / f"{base_path.stem}_{counter}"
        counter += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def background_noise_removal(input_file):
    """
    Remove background noise segments from the audio file.
    Uses Fourier Transform.
    """

    y, sr = librosa.load(input_file, sr=None)

    # Compute STFT
    S_full, phase = librosa.magphase(librosa.stft(y))

    # Estimate noise profile as the median across time
    noise_power = np.mean(S_full[:, int(sr * 0.1)], axis=1)

    # Build mask
    mask = S_full > noise_power[:, None]
    mask = mask.astype(float)
    mask = medfilt(mask, kernel_size=(1, 5))
    S_clean = S_full * mask
    y_clean = librosa.istft(S_clean * phase)

    return y_clean


# Define preprocessing function
def preprocess_audio(
    input_file,
    output_dir=Path(""),
    noise_removal_flag=False,
    segment_length=15,
    target_sr=16000,
):
    """
    Preprocesses an audio file by performing noise reduction, segmentation (15s), amplitude normalization, silence removal, and downsampling (44.1kHz to 16kHz).

    - Converts non-WAV files to WAV
    - Performs background noise removal
    - Downsamples (44.1kHz → 16kHz)
    - Segments audio into 15s chunks
    - Handles both single files and directories

    Args:
        input_path (str): Path to an audio file or directory.
        output_dir (str): Directory to save the processed audio segments.
        segment_length (int): Length of each segment in seconds (default is 15s).
        target_sr (int): Target sampling rate (default is 16kHz).

    Returns:
        lists: A list of processed audio segments (NumPy arrays).
        int: The sampling rate of the processed segments.
    """

    input_file = check_audio_extension(input_file)

    # Output of subdirectory
    segmented_dir = output_dir / "segmented_audio"
    if segmented_dir.exists():
        shutil.rmtree(segmented_dir)
    segmented_dir.mkdir(parents=True, exist_ok=True)

    # Load and resample audio
    y, sr = load_audio_with_soundfile(input_file)

    # Apply background noise removal
    # y = background_noise_removal(input_file)

    # Check if noise removal is active in client-side
    if noise_removal_flag:
        # Apply noise reduction using spectral gating
        y = nr.reduce_noise(y=y, sr=sr)

    # Noise reduction in chunks (if audio is long)
    # chunk_size = sr * 5  # Process 5-second chunks
    # y_denoised = np.concatenate(
    #     [
    #         nr.reduce_noise(y[i : i + chunk_size], sr=sr)
    #         for i in range(0, len(y), chunk_size)
    #     ]
    # )

    # Normalize amplitudes to [-1, 1]
    # y_normalized = y_denoised / np.max(np.abs(y_denoised))

    # Resample from 44.1kHz to 16kHz if not in target sampling rate
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # total_samples = len(y_denoised)

    # Get the base filename of the audio (excluding extension)
    audio_filename = Path(input_file).stem

    # Calculate segment length in samples
    segment_samples = segment_length * sr

    # Split and save segments
    segments = []

    for i, start in enumerate(range(0, len(y), segment_samples)):
        end = start + segment_samples
        segment = y[start:end]
        if len(segment) == segment_samples:  # includes full-length segments only
            segments.append(segment)
            # Save segment to disk if output_dir is provided
            if output_dir:
                file = segmented_dir / f"segment_{i + 1}.wav"
                sf.write(
                    file,
                    segment,
                    sr,
                )

    return segments, sr
