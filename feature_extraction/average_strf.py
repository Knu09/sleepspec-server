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


class STRFAnalyzer:
    def __init__(self):
        self.rates_vec = [
            -32, -22.6, -16, -11.3, -8, -5.70, -4, -2, -1, -0.5, -0.25,
            0.25, 0.5, 1, 2, 4, 5.70, 8, 11.3, 16, 22.6, 32
        ]
        self.scales_vec = [0.71, 1.0, 1.41, 2.00, 2.83, 4.00, 5.66, 8.00]

    def compute_avg_strf(self, audio_dir: Path):
        # Initialize accumulators for scale-rate, freq-rate, and freq-scale
        total_scale_rate = np.zeros(
            (len(self.scales_vec), len(self.rates_vec)))
        total_freq_rate = np.zeros((128, len(self.rates_vec)))
        total_freq_scale = np.zeros((128, len(self.scales_vec)))

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
                strf, _, _, _ = auditory.strf(
                    audio, audio_fs=fs, duration=15, rates=rates_vec, scales=scales_vec
                )

                # Compute the average STRF
                magnitude_strf = np.abs(strf)

                # Convert STRF to average vectors
                avgvec = plotslib.strf2avgvec(magnitude_strf)
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
        return (
            total_scale_rate / num_files,
            total_freq_rate / num_files,
            total_freq_scale / num_files
        )

    def save_plots(self, scale_rate, freq_rate, freq_scale, output_dir: Path):
        """Save STRF visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn")
        plt.rcParams.update({"font.size": 12})

        # Individual plots
        self._save_single_plot(
            scale_rate, output_dir / "avg_scale_rate.png",
            "Scale-Rate Representation", "Modulation Rate (Hz)", "Spectral Scale (cycles/octave)",
            [self.rates_vec[0], self.rates_vec[-1], self.scales_vec[0], self.scales_vec[-1]]
        )
        
        self._save_single_plot(
            freq_rate, output_dir / "avg_freq_rate.png",
            "Frequency-Rate Representation", "Rate (Hz)", "Frequency Channel",
            [self.rates_vec[0], self.rates_vec[-1], 0, freq_rate.shape[0]]
        )
        
        self._save_single_plot(
            freq_scale, output_dir / "avg_freq_scale.png",
            "Frequency-Scale Representation", "Scale (cycles/octave)", "Frequency Channel",
            [self.scales_vec[0], self.scales_vec[-1], 0, freq_scale.shape[0]]
        )

        # Combined plot
        self._save_combined_plot(scale_rate, freq_rate, freq_scale, output_dir / "strf_averages.png")

    def _save_single_plot(self, data, path, title, xlabel, ylabel, extent):
        plt.figure(figsize=(10, 6))
        plt.imshow(
            data, aspect="auto", origin="lower", 
            interpolation="gaussian", cmap="viridis", extent=extent
        )
        plt.colorbar(label="Amplitude (a.u.)")
        plt.title(title, pad=20)
        plt.xlabel(xlabel, labelpad=10)
        plt.ylabel(ylabel, labelpad=10)
        plt.grid(False)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    def _save_combined_plot(self, scale_rate, freq_rate, freq_scale, path):
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        fig.suptitle("STRF Average Representations", y=1.02, fontsize=16)
        
        plots = [
            (scale_rate, "Scale-Rate", "Rate (Hz)", "Scale (c/o)", 
             [self.rates_vec[0], self.rates_vec[-1], self.scales_vec[0], self.scales_vec[-1]]),
            (freq_rate, "Frequency-Rate", "Rate (Hz)", "Frequency Channel",
             [self.rates_vec[0], self.rates_vec[-1], 0, freq_rate.shape[0]]),
            (freq_scale, "Frequency-Scale", "Scale (c/o)", "Frequency Channel",
             [self.scales_vec[0], self.scales_vec[-1], 0, freq_scale.shape[0]])
        ]
        
        for ax, (data, title, xlabel, ylabel, extent) in zip(axes, plots):
            im = ax.imshow(
                data, aspect="auto", origin="lower",
                interpolation="gaussian", cmap="viridis", extent=extent
            )
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
