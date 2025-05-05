from pathlib import Path
import numpy as np
import utils
import auditory
import plotslib
import pickle
import matplotlib.pyplot as plt

# Add output_dir for Flask app initialization
output_dir = Path("feature_analysis/strf_plots")
output_dir.mkdir(parents=True, exist_ok=True)

# Define the directory containing the audio files
audio_dir = Path("preprocess/preprocessed_audio/processed_audio/")

# Define the rates and scales vectors
rates_vec: list[float] = [
    -32,
    -22.6,
    -16,
    -11.3,
    -8,
    -5.70,
    -4,
    -2,
    -1,
    -0.5,
    -0.25,
    0.25,
    0.5,
    1,
    2,
    4,
    5.70,
    8,
    11.3,
    16,
    22.6,
    32,
]
scales_vec = [0.71, 1.0, 1.41, 2.00, 2.83, 4.00, 5.66, 8.00]

# Initialize accumulators for scale-rate, freq-rate, and freq-scale
total_scale_rate = np.zeros((len(scales_vec), len(rates_vec)))
total_freq_rate = np.zeros((128, len(rates_vec)))
total_freq_scale = np.zeros((128, len(scales_vec)))

# Initialize a counter for the number of files
num_files = 0

# Loop through all audio files in the directory
for filename in audio_dir.iterdir():
    if filename.suffix == ".wav":
        # Construct the full file path
        wav_file = audio_dir / filename

        # Load the audio file
        audio, fs = utils.audio_data(wav_file)

        # Compute the STRF
        strf, auditory_spectrogram_, mod_scale, scale_rate = auditory.strf(
            audio, audio_fs=fs, duration=15, rates=rates_vec, scales=scales_vec
        )

        # Compute the average STRF
        magnitude_strf = np.abs(strf)
        real_valued_strf = np.mean(magnitude_strf, axis=0)

        # Convert STRF to average vectors
        avgvec = plotslib.strf2avgvec(strf)
        strf_scale_rate, strf_freq_rate, strf_freq_scale = plotslib.avgvec2strfavg(
            avgvec, nbScales=len(scales_vec), nbRates=len(rates_vec)
        )

        # Accumulate the results
        total_scale_rate += strf_scale_rate
        total_freq_rate += strf_freq_rate
        total_freq_scale += strf_freq_scale

        # Increment the file counter
        num_files += 1

# Average the results
avg_scale_rate = total_scale_rate / num_files
avg_freq_rate = total_freq_rate / num_files
avg_freq_scale = total_freq_scale / num_files

# Save averaged arrays as numpy files
np.save(output_dir / "avg_scale_rate.npy", avg_scale_rate)
np.save(output_dir / "avg_freq_scale.npy", avg_freq_scale)
np.save(output_dir / "avg_freq_rate.npy", avg_freq_rate)

# Create and save visualizaiton plots


def save_strf_plots(scale_rate, freq_rate, freq_scale, output_path):
    plt.style.use("seaborn")
    plt.rcParams.update({"font.size": 12})
    cmap = "viridis"

    # Scale-Rate plot
    plt.figure(figsize=(10, 6))
    plt.imshow(
        scale_rate,
        aspect="auto",
        origin="lower",
        interpolation="gaussian",
        cmap=cmap,
        extent=[rates_vec[0], rates_vec[-1], scales_vec[0], scales_vec[-1]],
    )
    plt.colorbar(label="Amplitude (a.u.)")
    plt.title("Scale-Rate Representation", pad=20)
    plt.xlabel("Modulation Rate (Hz)", labelpad=10)
    plt.ylabel("Spectral Scale (cycles/octave)", labelpad=10)
    plt.grid(False)
    plt.savefig(output_dir / "avg_scale_rate.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Frequencies-Scale plot
    plt.figure(figsize=(10, 6))
    plt.imshow(
        freq_scale,
        aspect="auto",
        origin="lower",
        interpolation="gaussian",
        cmap=cmap,
        extent=[scales_vec[0], scales_vec[-1], 0, freq_scale.shape[0]],
    )

    plt.colorbar(label="Amplitude (a.u.)")
    plt.title("Frequency-Scale Representation", pad=20)
    plt.xlabel("Scale (cycles/octaves)", labelpad=10)
    plt.ylabel("Frequency (Hz)", labelpad=10)
    plt.grid(False)

    plt.savefig(output_dir / "freq_scale.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Frequencies-Rates plot
    plt.figure(figsize=(10, 6))
    plt.imshow(
        freq_rate,
        aspect="auto",
        origin="lower",
        interpolation="gaussian",
        cmap=cmap,
        extent=[rate_values[0], rate_values[-1], scale_values[0], scale_values[-1]],
    )

    plt.colorbar(label="Amplitude (a.u.)")
    plt.title("Frequency-Rate Representation", pad=20)
    plt.xlabel("Rates (Hz)", labelpad=10)
    plt.ylabel("Frequencies (Hz)", labelpad=10)
    plt.grid(False)

    plt.savefig(output_dir / "freq_rate.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Combined Plot
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle("STRF Average Representations", y=1.02, fontsize=16)

    # Scale-Rate
    im0 = axes[0].imshow(
        scale_rate,
        aspect="auto",
        origin="lower",
        interpolation="gaussian",
        cmap=cmap,
        extent=[rates_vec[0], rates_vec[-1], scales_vec[0], scales_vec[-1]],
    )
    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    axes[0].set_title("Scale-Rate")
    axes[0].set_xlabel("Rate (Hz)")
    axes[0].set_ylabel("Scale (c/o)")

    # Frequency-Rate
    im1 = axes[1].imshow(
        freq_rate,
        aspect="auto",
        origin="lower",
        interpolation="gaussian",
        cmap=cmap,
        extent=[rates_vec[0], rates_vec[-1], 0, freq_rate.shape[0]],
    )
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    axes[1].set_title("Frequency-Rate")
    axes[1].set_xlabel("Rate (Hz)")
    axes[1].set_ylabel("Frequency Channel")

    # Frequency-Scale
    im2 = axes[2].imshow(
        freq_scale,
        aspect="auto",
        origin="lower",
        interpolation="gaussian",
        cmap=cmap,
        extent=[scales_vec[0], scales_vec[-1], 0, freq_scale.shape[0]],
    )
    fig.colorbar(im2, ax=axes[2], shrink=0.8)
    axes[2].set_title("Frequency-Scale")
    axes[2].set_xlabel("Scale (c/o)")
    axes[2].set_ylabel("Frequency Channel")

    plt.tight_layout()
    plt.savefig(output_dir / "strf_averages.png", dpi=300, bbox_inches="tight")
    plt.close()


save_strf_plots(avg_scale_rate, avg_freq_rate, avg_freq_scale, output_dir)

