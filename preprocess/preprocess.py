import subprocess
import tempfile
from .noise_reduction.noisereduction import Wiener

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


def deepfilternet_noise_reduction(y, sr, target_sr=16000):
    """
    Applies DeepFilterNet noise reduction.
    Handles resampling to 48kHz and back to the target sample rate.
    """
    print("Background noise reduction: active (using DeepFilterNet)")

    binary_path = Path("preprocess/noise_reduction/deepfilternet/deep-filter-0.5.6-x86_64-unknown-linux-musl")
    if not binary_path.is_file():
        print(f"--- ERROR: The binary was not found.")
        print("--- Skipping noise reduction.")


    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        input_path = temp_dir_path / "input_audio.wav"

        # resample 16khz to 48khz
        if sr != 48000:
            print(f"Resampling audio from {sr}Hz to 48000Hz for DeepFilterNet binary.")
            y = librosa.resample(y, orig_sr=sr, target_sr=48000)
            sr = 48000
        
        sf.write(input_path, y, sr)

        # commands of deepfilter
        command = [
            str(binary_path),
            "--out-dir", str(temp_dir_path),  
            str(input_path)                 
        ]


    try:
        # Convert numpy array to torch tensor
        audio_tensor = torch.from_numpy(y).float()

        # 1. Resample from the current sr (e.g., 16kHz) to the required 48kHz
        resampler_to_48k = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=48000)
        audio_48k = resampler_to_48k(audio_tensor)

        # 2. Enhance the audio (model expects a batch, so we add a dimension)
        # enhanced_audio_48k = enhance(df_model, df_state, audio_48k)
        enhanced_audio_48k = enhance(df_model, df_state, audio_48k.unsqueeze(0))

        # 3. Resample back down to the pipeline's target sample rate (16kHz)
        resampler_to_target = torchaudio.transforms.Resample(
            orig_freq=48000, new_freq=target_sr)
        enhanced_audio_target_sr = resampler_to_target(
            enhanced_audio_48k.squeeze(0))

        # 4. Convert back to a numpy array for the rest of the pipeline
        y_clean = enhanced_audio_target_sr.numpy()
        print("DeepFilterNet filtering applied successfully.")
        return y_clean, target_sr

    except Exception as e:
        print(f"An error occurred during DeepFilterNet filtering: {e}")
        print("Skipping noise reduction and proceeding with original audio.")
        # Return original audio if an error occurs
        return y, sr

# def wiener_noise_reduction(y, sr):
#     """Applies the Wiener filter noise reduction."""
#     print("Background noise reduction: active (using Wiener Filter)")
#     try:
#         # The Wiener filter class works on files, so we create a temporary one.
#         with tempfile.TemporaryDirectory() as temp_dir:
#             temp_dir_path = Path(temp_dir)
#             temp_noisy_path = temp_dir_path / "temp_noisy_audio.wav"
#             sf.write(temp_noisy_path, y, sr)
#
#             noise_start_time = 0.0
#             noise_end_time = 0.5
#
#             if len(y) / sr > noise_end_time:
#                 print(f"Using first {noise_end_time}s for noise profile.")
#                 wiener_filter = Wiener(
#                     str(temp_noisy_path.with_suffix('')),
#                     noise_start_time,
#                     noise_end_time
#                 )
#                 wiener_filter.wiener_two_step()
#                 cleaned_file_path = temp_dir_path / \
#                     f"{temp_noisy_path.stem}_wiener_two_step.wav"
#
#                 if cleaned_file_path.exists():
#                     y, _ = sf.read(cleaned_file_path)
#                     print("Wiener filtering applied successfully.")
#                 else:
#                     print("Warning: Wiener filter output file not found. Skipping.")
#             else:
#                 print("Warning: Audio too short for noise profiling. Skipping.")
#     except Exception as e:
#         print(f"An error occurred during Wiener filtering: {e}")
#         print("Skipping noise reduction.")
#
#     return y, sr


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


def background_noise_removal(y, sr):
    """
    Remove background noise segments from the audio file.
    Uses Fourier Transform.
    """

    # y, sr = librosa.load(input_file, sr=None)

    # Compute STFT
    S_full, phase = librosa.magphase(librosa.stft(y))

    # Use first 10% of frames to estimate noise profile
    num_frames = S_full.shape[1]
    noise_frames = int(num_frames * 0.1)

    # Estimate noise profile as the median across time
    noise_power = np.mean(S_full[:, :noise_frames], axis=1)

    # Build mask
    mask = S_full > noise_power[:, None]
    mask = mask.astype(float)
    mask = medfilt(mask, kernel_size=(1, 5))

    # Apply mask
    S_clean = S_full * mask
    y_clean = librosa.istft(S_clean * phase)

    return y_clean


def wiener_noise_reduction(y, sr):
    print("Background noise reduction: active (using Wiener Filter)")

    # The Wiener filter class works on files, so we create a temporary one.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        # Create a temporary path for the noisy input file
        temp_noisy_path = temp_dir_path / "temp_noisy_audio.wav"

        # Save the in-memory audio array 'y' to the temporary file
        sf.write(temp_noisy_path, y, sr)

        # --- NEW: Wiener Filter Implementation ---
        try:
            # The Wiener class constructor needs the base filename (without extension)
            # and the start/end time of a known noise segment.
            # We ASSUME the first 0.5 seconds of the recording is stationary noise.
            noise_start_time = 0.0
            noise_end_time = 0.5

            # Check if the audio is long enough for the noise profile
            if len(y) / sr > noise_end_time:
                print(f"Using first {noise_end_time}s for noise profile.")
                # Instantiate the filter
                wiener_filter = Wiener(
                    str(temp_noisy_path.with_suffix("")),
                    noise_start_time,
                    noise_end_time,
                )

                # Apply the two-step Wiener filter. This saves a new file.
                wiener_filter.wiener_two_step()

                # Define the path where the cleaned file was saved
                # The class appends '_wiener_two_step.wav' to the original name
                cleaned_file_path = (
                    temp_dir_path /
                    f"{temp_noisy_path.stem}_wiener_two_step.wav"
                )

                if cleaned_file_path.exists():
                    # Load the cleaned audio data back into our 'y' variable
                    y, _ = sf.read(cleaned_file_path)
                    print("Wiener filtering applied successfully.")
                else:
                    print(
                        "Warning: Wiener filter output file not found. Skipping noise reduction."
                    )
            else:
                print(
                    "Warning: Audio too short for noise profiling. Skipping noise reduction."
                )

        except Exception as e:
            print(f"An error occurred during Wiener filtering: {e}")
            print("Skipping noise reduction and proceeding with original audio.")
    # The temporary directory and its contents are automatically deleted here.
    return y, sr


# Define preprocessing function


def preprocess_audio(
    input_file,
    output_dir=Path(""),
    noise_removal_method="none",
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

    print(f"Processing audio file: {input_file}")

    input_file = check_audio_extension(input_file)

    # Output of subdirectory
    segmented_dir = output_dir / "segmented_audio"
    if segmented_dir.exists():
        shutil.rmtree(segmented_dir)
    segmented_dir.mkdir(parents=True, exist_ok=True)

    # Load and resample audio
    y, sr = load_audio_with_soundfile(input_file)

    # Resample to 16kHz if not in target sampling rate
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # selects which function to call based on the method string.
    if noise_removal_method == 'wiener':
        y, sr = wiener_noise_reduction(y, sr)
    elif noise_removal_method == 'deepfilternet':
        y, sr = deepfilternet_noise_reduction(y, sr, target_sr)
    else:
        print("Background noise reduction: inactive")

    audio_filename = Path(input_file).stem
    segment_samples = segment_length * sr
    segments = []
    for i, start in enumerate(range(0, len(y), segment_samples)):
        end = start + segment_samples
        segment = y[start:end]
        if len(segment) == segment_samples:
            segments.append(segment)
            if output_dir:
                file = segmented_dir / f"segment_{i + 1}.wav"
                sf.write(file, segment, sr)

    return segments, sr
