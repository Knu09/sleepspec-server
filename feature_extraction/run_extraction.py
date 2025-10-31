"""
Copyright (c) Baptiste Caramiaux, Etienne Thoret
Please cite us if you use this script :)
All rights reserved

"""

import os
from feature_extraction import utils
from feature_extraction import plotslib
from feature_extraction import auditory
from numpy.ma import append
import matplotlib.pylab as plt
from librosa import feature

import scipy.io as sio
from scipy.fft import ifft2, ifftshift
import numpy as np
import pickle
import sys
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor
from profiler import profile


sys.path.append(str(Path(__file__).resolve().parent))

rates_vec = [
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


def extract_features(audio_segment, fs):
    strf, auditory_spectrogram_, mod_scale, scale_rate = auditory.strf(
        audio_segment, audio_fs=fs, duration=15, rates=rates_vec, scales=scales_vec
    )

    # prints entire array
    # np.set_printoptions(threshold=np.inf)

    # initial STRF (3750, 128, 8, 22)
    # Computation of STRF (time, frequency, scale, rate) by aggregating the time dimension (axis=0) using mean

    # Compute the magnitude of the STRF and average over time
    magnitude_strf = np.abs(strf)

    # STRF (128, 8, 22)
    real_valued_strf = np.mean(magnitude_strf, axis=0)

    # print(real_valued_strf)  ## print entire array of STRF
    return real_valued_strf, fs


# feature extraction for segmented audio in specific directory
def feature_extract_dir(input_dir: Path, output_dir: Path):
    for filename in input_dir.iterdir():
        print(f"Processing file: {filename}")

        audio_file = input_dir / filename

        audio_file, fs = utils.audio_data(audio_file)
        real_valued_strf, fs = extract_features(audio_file, fs)

        output_file = output_dir / f"{filename.stem}_strf.pkl"
        strf_data = {
            "strf": real_valued_strf,
            "fs": fs,
        }

        with open(output_file, "wb") as f:
            pickle.dump(strf_data, f)

        print(f"Saved output to: {output_file}")


def process_segment(i, segment, sample_rate):
    print(f"Processing Segment {i + 1}")

    real_valued_strf, fs = extract_features(segment, sample_rate)

    return real_valued_strf


@profile
def feature_extract_segments(segment_audio_arr, sample_rate):
    workers = os.getenv("MAX_WORKERS") or 2

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit in order and keep the futures in the same order

        futures = [
            executor.submit(process_segment, i, segment, sample_rate)
            for i, segment in enumerate(segment_audio_arr)
        ]

        # Retrieve results in the same order as submitted
        features = [future.result() for future in futures]

    return features
