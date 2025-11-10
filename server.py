import io
import os
import pickle
import sys
import zipfile
from dataclasses import dataclass
from enum import Enum
from http import HTTPStatus
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from pydub import AudioSegment
from werkzeug.utils import secure_filename

from feature_extraction.run_extraction import feature_extract_segments
from feature_extraction.strf_analyzer import STRFAnalyzer
from preprocess.preprocess import preprocess_audio
from profiler import profile
from globals import OUTDIR

sys.path.append("preprocess/")
sys.path.append("feature_extraction/")

app = Flask(__name__)
CORS(app, origins="*", supports_credentials=True)
uploads_path = Path(OUTDIR / "uploads")

strf_analyzer = STRFAnalyzer()


@app.route("/")
def home():
    return "Flask server is running."


class SD_Class(Enum):
    PRE = "pre"  # Non-sleep deprived
    POST = "post"  # Sleep deprived
    BALANCED = "balanced"


@dataclass
class Classification:
    sd: SD_Class
    classes: list[SD_Class]
    scores: list[float]
    decision_scores: list[float]
    sd_decision_score: float
    nsd_decision_score: float
    confidence_score: float
    avg_decision_score: float
    result: str
    is_success: bool
    sd_prob: float
    nsd_prob: float
    # other fields here

    def into_json(self):
        return jsonify(
            {
                "class": self.sd.value,
                "classes": [c.value for c in self.classes],
                "scores": self.scores,
                "decision_scores": self.decision_scores,
                "sd_decision_score": self.sd_decision_score,
                "nsd_decision_score": self.nsd_decision_score,
                "sd_prob": self.sd_prob,
                "nsd_prob": self.nsd_prob,
                "confidence_score": self.confidence_score,
                "decision_score": self.avg_decision_score,
                "result": self.result,
            }
        )


@app.route("/plots/<uuid:uid>/<path:filename>")
def get_plot(filename, uid):
    print(f"Requesting plot: {filename}")
    path = (Path(OUTDIR / "feature_analysis/strf_plots") /
            str(uid)).resolve(strict=True)
    return send_from_directory(path, filename)


@app.route("/segments/<uuid:uid>")
def Segments(uid):
    segments_dir = (
        Path(OUTDIR / "preprocess/preprocessed_audio/processed_audio")
        / str(uid)
        / "segmented_audio"
    )

    # Construct an in-memory zip file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for _, file in enumerate(segments_dir.glob("segment_*.wav")):
            zip_file.write(file, arcname=file.name)

    # reset buffer pointer back to 0
    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        mimetype="application/zip",
        download_name="segments.zip",
        as_attachment=True,
    )


@app.route("/upload/<uuid:uid>", methods=["POST"])
def Upload(uid):
    if "audio" not in request.files:
        return jsonify({"error": "No audio file in request."}), HTTPStatus.BAD_REQUEST

    audio_file = request.files["audio"]

    # parse noiseRemoval request
    noise_removal_flag = request.form.get(
        "wienerFiltering", "false").lower() == "true"

    if audio_file.filename:
        (uploads_path / str(uid)).mkdir(parents=True, exist_ok=True)

        file_path = uploads_path / \
            str(uid) / secure_filename(audio_file.filename)
        audio_file.save(file_path)

        wav_file = convertWAV(file_path)
        clf = classify(wav_file, uid, noise_removal_flag)

        return (
            clf.into_json(),
            HTTPStatus.OK if clf.is_success else HTTPStatus.BAD_REQUEST,
        )

    return (
        jsonify({"error": "There was a problem saving the file"}),
        HTTPStatus.INTERNAL_SERVER_ERROR,
    )


def predict_features(features, svm, pca):
    if not features:
        print("!!!!!!!!!! Error: no features accepted !!!!!!!!!!")
        print("Make sure the audio recording length is at least 15 seconds.")
        is_success = False
        return 0, 0, [], [], 0.0, is_success

    nsd_counter = 0
    sd_counter = 0
    sum_nsd_prob = 0.0
    sum_sd_prob = 0.0
    sd_prob_scores = []
    nsd_prob_scores = []
    classes = []
    sd_decision_scores = []
    nsd_decision_scores = []
    decision_scores = []
    confidence_scores = []
    avg_confidence_score = 0.0

    for i, feature in enumerate(features):
        print(f"\nProcessing feature {i + 1}")

        # Flatten and normalize
        feature_flat = np.asarray(feature).flatten()
        feature_norm = (
            feature_flat / np.max(np.abs(feature_flat))
            if np.max(np.abs(feature_flat)) != 0
            else feature_flat
        )
        feature_reshaped = feature_norm.reshape(1, -1)

        expected_features = pca.components_.shape[1]
        if feature_flat.shape[0] != expected_features:
            raise ValueError(
                f"Feature mismatch! Expected {expected_features}, got {
                    feature_flat.shape[0]
                }."
            )

        # PCA transformation
        feature_pca = pca.transform(feature_reshaped)

        # Prediction
        y_pred = svm.predict(feature_pca)
        predicted_label = y_pred[0]
        print(f"Predicted class for feature {i + 1}: {predicted_label}")
        print(f"SVM classes: {svm.classes_}")

        # Decision score (distance from hyperplane)
        decision_score = abs(float(svm.decision_function(feature_pca)[0]))
        decision_scores.append(decision_score)
        print(f"Decision Score: {decision_score:.4f}")

        # Confidence (probability) score
        sd_prob, nsd_prob = 0.0, 0.0
        if hasattr(svm, "predict_proba"):
            probs = svm.predict_proba(feature_pca)[0]
            sd_index = np.where(svm.classes_ == SD_Class.POST.value)[0][0]
            nsd_index = np.where(svm.classes_ == SD_Class.PRE.value)[0][0]
            sd_prob = float(probs[sd_index])
            nsd_prob = float(probs[nsd_index])

        # Output NSD and SD Probabilities
        print(f"Non-sleep-deprived Probability Score: {nsd_prob}")
        print(f"Sleep-deprived Probability Score: {sd_prob}")

        # Assign class and count
        if predicted_label == SD_Class.POST.value:
            sd_counter += 1
            classes.append(SD_Class.POST)
            confidence_scores.append(sd_prob)
            sd_decision_scores.append(decision_score)
        else:
            nsd_counter += 1
            classes.append(SD_Class.PRE)
            confidence_scores.append(nsd_prob)
            nsd_decision_scores.append(decision_score)

        sum_sd_prob += sd_prob
        sum_nsd_prob += nsd_prob
        sd_prob_scores.append(sd_prob)
        nsd_prob_scores.append(nsd_prob)

    # Final calculations
    avg_sd_prob = sum_sd_prob / len(sd_prob_scores)
    avg_nsd_prob = sum_nsd_prob / len(nsd_prob_scores)
    avg_decision_score = np.mean(decision_scores) if decision_scores else 0.0
    avg_sd_decision_score = np.mean(
        sd_decision_scores) if sd_decision_scores else 0.0
    avg_nsd_decision_score = (
        np.mean(nsd_decision_scores) if nsd_decision_scores else 0.0
    )

    # Adjusted confidence scoring
    if sd_counter == nsd_counter:
        adjusted_confidence_score = 50 + (avg_sd_prob - avg_nsd_prob) * 50
    elif sd_counter > nsd_counter:
        adjusted_confidence_score = 50 + (avg_sd_prob * 50)
    else:
        adjusted_confidence_score = avg_nsd_prob * 50

    # Feedback message
    if adjusted_confidence_score >= 80:
        print("\nClassification: Highly Sleep-deprived")
    elif adjusted_confidence_score >= 50:
        print("\nClassification: Moderate Sleep-deprived")
    else:
        print("\nClassification: Non-sleep-deprived")

    # Average Confidence Score
    avg_confidence_score = sum(confidence_scores) / len(confidence_scores)

    # Output summaries
    print(f"\nAverage SD Probability: {avg_sd_prob:.4f}")
    print(f"Average NSD Probability: {avg_nsd_prob:.4f}")
    print(f"Pre (NSD) features count: {nsd_counter}")
    print(f"Post (SD) features count: {sd_counter}")
    print(f"Adjusted Confidence Score: {adjusted_confidence_score:.2f}")
    print(f"Average Confidence Score: {avg_confidence_score:.2f}")
    print(f"Average Decision Score (all): {avg_decision_score:.4f}")
    print(f"Average Decision Score (sd only): {avg_sd_decision_score:.4f}")
    print(f"Average Decision Score (nsd only): {avg_nsd_decision_score:.4f}")

    is_success = True
    return (
        avg_sd_prob,
        avg_nsd_prob,
        nsd_counter,
        sd_counter,
        classes,
        confidence_scores,
        decision_scores,
        avg_sd_decision_score,
        avg_nsd_decision_score,
        adjusted_confidence_score,
        avg_confidence_score,
        avg_decision_score,
        is_success,
    )


@profile
def classify(audio_path: Path, uid, noise_removal_flag) -> Classification:
    """
    Predict the class labels for the given STM features array of 3D using the trained SVM and PCA models.

    Args:
        features (list): List of feature arrays (e.g., STRF features).
        svm_path (str): Path to the trained SVM model (.pkl file).
        pca_path (str): Path to the trained PCA model (.pkl file).
    """
    svm_path = Path("./updated_model/svm_pca_strf_ncomp24_2025-05-29.pkl")

    print(f"Model: {svm_path}")

    test_sample_path = Path("./strf_data_new.pkl")

    # Load the SVM and PCA models using pickle
    with open(svm_path, "rb") as f:
        data = pickle.load(f)
    svm = data["svm"]
    pca = data["pca"]
    # Define the output directory, if necessary to be stored
    output_dir_processed = Path(OUTDIR / "preprocess/preprocessed_audio/processed_audio") / str(
        uid
    )
    output_dir_segmented = output_dir_processed / "segmented_audio"

    # Preprocess
    segments, sr = preprocess_audio(
        audio_path, output_dir_processed, noise_removal_flag
    )

    # Print details
    print(f"Number of segments: {len(segments)}")
    print(f"Sampling rate: {sr} Hz")

    # Feature Extraction
    features = feature_extract_segments(segments, sr)
    print("Feature Extraction Complete.")

    # Compute and save STRFs
    avg_scale_rate, avg_freq_rate, avg_freq_scale = strf_analyzer.compute_avg_strf(
        features
    )
    strf_analyzer.save_plots(
        avg_scale_rate,
        avg_freq_rate,
        avg_freq_scale,
        Path(OUTDIR / "feature_analysis/strf_plots") / str(uid),
    )
    print(f"Avg STRF computation and plots complete.")

    # test_sample = pickle.load(test_sample_path)
    # with open(test_sample_path, "rb") as f:
    #     test_sample = pickle.load(f)

    # print(type(test_sample), test_sample)
    # np.set_printoptions(threshold=np.inf)
    #
    # magnitude_strf = np.abs(test_sample)
    #
    # # STRF (128, 8, 22)
    # test_sample = np.mean(magnitude_strf, axis=0)
    # print(test_sample["strf"])

    (
        avg_sd_prob,
        avg_nsd_prob,
        pre_count,
        post_count,
        classes,
        confidence_scores,
        decision_scores,
        avg_sd_decision_score,
        avg_nsd_decision_score,
        adjusted_confidence_score,
        avg_confidence_score,
        avg_decision_score,
        is_success,
    ) = predict_features(features, svm, pca)

    print(f"\nsuccess: {is_success}\n")
    if adjusted_confidence_score == 50:
        result_text = "The classification score is balanced."
        sd_class = SD_Class.BALANCED
    elif adjusted_confidence_score > 50:
        result_text = "You are sleep-deprived."
        sd_class = SD_Class.POST
    else:
        result_text = "You are non-sleep-deprived."
        sd_class = SD_Class.PRE

    return Classification(
        sd_prob=avg_sd_prob,
        nsd_prob=avg_nsd_prob,
        sd=sd_class,
        scores=confidence_scores,
        decision_scores=decision_scores,
        sd_decision_score=avg_sd_decision_score,
        nsd_decision_score=avg_nsd_decision_score,
        classes=classes,
        confidence_score=avg_confidence_score,
        avg_decision_score=avg_decision_score,
        result=result_text,
        is_success=is_success,
    )


def convertWAV(audio: Path) -> Path:
    if audio.suffix == ".wav":
        return audio

    wav = audio.with_suffix(".wav")
    file = AudioSegment.from_file(audio)
    file.export(wav, format="wav")

    audio.unlink()
    return wav
