from pathlib import Path
from typing import Any
import numpy as np
from feature_extraction import utils
from feature_extraction import auditory
from feature_extraction import plotslib
import matplotlib

from profiler import profile

matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add output_dir for Flask app initialization
output_dir = Path("feature_analysis/strf_plots")
output_dir.mkdir(parents=True, exist_ok=True)

# Define the directory containing the audio files
audio_dir = Path("preprocess/preprocessed_audio/processed_audio/")


class STRFAnalyzer:
    def __init__(self):
        self.rates_vec = [
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
        self.scales_vec = [0.71, 1.0, 1.41, 2.00, 2.83, 4.00, 5.66, 8.00]

    def segment_strf(self, wav_file: Path):
        if wav_file.suffix != ".wav":
            raise Exception("Not a wav file!")

        print(wav_file.name)
        # Load the audio file
        audio, fs = utils.audio_data(wav_file)

        # Compute the STRF
        strf, _, _, _ = auditory.strf(
            audio,
            audio_fs=fs,
            duration=15,
            rates=self.rates_vec,
            scales=self.scales_vec,
        )

        # Compute the average STRF
        magnitude_strf = np.abs(strf)

        # Convert STRF to average vectors
        avgvec = plotslib.strf2avgvec(magnitude_strf)
        strf_scale_rate, strf_freq_rate, strf_freq_scale = plotslib.avgvec2strfavg(
            avgvec,
            nbScales=len(self.scales_vec),
            nbRates=len(self.rates_vec),
        )

        return strf_scale_rate, strf_freq_rate, strf_freq_scale

    @profile
    def compute_avg_strf(self, audio_dir: Path):
        # Initialize accumulators for scale-rate, freq-rate, and freq-scale
        feature_tensors = {
            "total_scale_rate": np.zeros((len(self.scales_vec), len(self.rates_vec))),
            "total_freq_rate": np.zeros((128, len(self.rates_vec))),
            "total_freq_scale": np.zeros((128, len(self.scales_vec))),
        }

        # Batch process all audio files in the directory
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.segment_strf, wav_file)
                for wav_file in audio_dir.iterdir()
            ]

        for f in as_completed(futures):
            sr, fr, fs = f.result()
            feature_tensors["total_scale_rate"] += sr
            feature_tensors["total_freq_rate"] += fr
            feature_tensors["total_freq_scale"] += fs

        num_files = len(futures)

        # Average the results
        return (
            feature_tensors["total_scale_rate"] / num_files,
            feature_tensors["total_freq_rate"] / num_files,
            feature_tensors["total_freq_scale"] / num_files,
        )

    def save_plots(self, scale_rate, freq_rate, freq_scale, output_dir: Path):
        """Save STRF visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        # plt.style.use("seaborn")
        plt.rcParams.update({"font.size": 12})

        # Individual plots
        self._save_single_plot(
            scale_rate,
            output_dir / "avg_scale_rate.png",
            "Scale-Rate Representation",
            "Modulation Rate (Hz)",
            "Spectral Scale (cycles/octave)",
            [
                self.rates_vec[0],
                self.rates_vec[-1],
                self.scales_vec[0],
                self.scales_vec[-1],
            ],
        )

        self._save_single_plot(
            freq_rate,
            output_dir / "avg_freq_rate.png",
            "Frequency-Rate Representation",
            "Rate (Hz)",
            "Frequency Channel",
            [self.rates_vec[0], self.rates_vec[-1], 0, freq_rate.shape[0]],
        )

        self._save_single_plot(
            freq_scale,
            output_dir / "avg_freq_scale.png",
            "Frequency-Scale Representation",
            "Scale (cycles/octave)",
            "Frequency Channel",
            [self.scales_vec[0], self.scales_vec[-1], 0, freq_scale.shape[0]],
        )

        # Combined plot
        self._save_combined_plot(
            scale_rate, freq_rate, freq_scale, output_dir / "strf_averages.png"
        )

    def _save_single_plot(self, data, path, title, xlabel, ylabel, extent):
        plt.figure(figsize=(10, 6))
        plt.imshow(
            data,
            aspect="auto",
            origin="lower",
            interpolation="gaussian",
            cmap="viridis",
            extent=extent,
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
            (
                scale_rate,
                "Scale-Rate",
                "Rate (Hz)",
                "Scale (c/o)",
                [
                    self.rates_vec[0],
                    self.rates_vec[-1],
                    self.scales_vec[0],
                    self.scales_vec[-1],
                ],
            ),
            (
                freq_rate,
                "Frequency-Rate",
                "Rate (Hz)",
                "Frequency Channel",
                [self.rates_vec[0], self.rates_vec[-1], 0, freq_rate.shape[0]],
            ),
            (
                freq_scale,
                "Frequency-Scale",
                "Scale (c/o)",
                "Frequency Channel",
                [self.scales_vec[0], self.scales_vec[-1], 0, freq_scale.shape[0]],
            ),
        ]

        for ax, (data, title, xlabel, ylabel, extent) in zip(axes, plots):
            im = ax.imshow(
                data,
                aspect="auto",
                origin="lower",
                interpolation="gaussian",
                cmap="viridis",
                extent=extent,
            )
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
