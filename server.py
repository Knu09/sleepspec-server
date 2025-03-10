from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
from pydub import AudioSegment
from pathlib import Path
from http import HTTPStatus
from enum import Enum
from dataclasses import dataclass
import os
import pickle
import joblib
import numpy as np
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score

app = Flask(__name__)
uploads_path = "tmp/uploads"


class SD_Class(Enum):
    NSD = 0
    SD = 1


@dataclass
class Classification:
    sd: SD_Class
    confidence_score: float
    result: str
    # other fields here


@app.route("/upload", methods=["POST"])
def Upload():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file in request."}), HTTPStatus.BAD_REQUEST

    audio_file = request.files["audio"]
    if audio_file.filename:
        Path(uploads_path).mkdir(parents=True, exist_ok=True)

        uploads = Path(uploads_path)

        file_path = uploads / secure_filename(audio_file.filename)
        audio_file.save(file_path)

        wav_file = convertWAV(file_path)
        clf = classify(wav_file)

        return (
            jsonify(
                {
                    "class": clf.sd.value,
                    "result": clf.result,
                }
            ),
            HTTPStatus.OK,
        )

    return (
        jsonify({"error": "There was a problem saving the file"}),
        HTTPStatus.INTERNAL_SERVER_ERROR,
    )


def classify(audio_path: Path) -> Classification:
    svm_path = Path("./population_level_svm.pkl")
    pca_path = Path("./population_level_pca.pkl")
    test_sample_path = Path(
        # "~/Research/Sleep Deprivation Detection using voice/strf_data_new.pkl"
        # "~/Research/Sleep Deprivation Detection using voice/dataset/osf/stmtf/strf_session_post_subjectNb_01_daySession_01_segmentNb_0.pkl"
        # "~/github/16Khz-models/feature_extraction/gian_data_new.pkl"
        "./strf_data_new.pkl"
        # "~/github/16Khz-models/feature_extraction/pkls/segment_72_strf.pkl"
    )
    with open(svm_path, "rb") as f:
        svm = pickle.load(f)
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)

    # test_sample = pickle.load(test_sample_path)
    with open(test_sample_path, "rb") as f:
        test_sample = pickle.load(f)

    # print(type(test_sample), test_sample)
    np.set_printoptions(threshold=np.inf)
    #
    # magnitude_strf = np.abs(test_sample)
    #
    # # STRF (128, 8, 22)
    # test_sample = np.mean(magnitude_strf, axis=0)
    # print(test_sample["strf"])

    # original_shape = test_sample.shape
    test_sample_flattened = np.asarray(test_sample["strf"]).flatten()
    test_sample = test_sample_flattened.reshape(1, -1)

    expected_features = pca.components_.shape[1]
    if test_sample_flattened.shape[0] != expected_features:
        raise ValueError(
            f"Feature mismatch! Expected {expected_features} features, but got {
                test_sample_flattened.shape[0]
            }."
        )

    max_test_sample = np.max(np.abs(test_sample))
    if max_test_sample != 0:
        test_sample_normalized = test_sample / max_test_sample
    else:
        test_sample_normalized = test_sample

    test_sample_pca = pca.transform(test_sample_normalized)

    y_pred = svm.predict(test_sample_pca)
    logits = np.array([1.0, 0.1])
    probs = softmax(logits)
    confidence = np.max(probs)

    if y_pred == svm.classes[0]:
        return Classification(
            sd=SD_Class.SD,
            confidence_score=confidence,  # 82.6%
            result="You are sleep deprived.",
        )
    else:
        return Classification(
            sd=SD_Class.NSD,
            confidence_score=confidence,  # 82.6%
            result="You are not sleep deprived.",
        )


def convertWAV(audio: Path) -> Path:
    wav = audio.with_suffix(".wav")
    file = AudioSegment.from_file(audio)
    file.export(wav, format="wav")

    audio.unlink()
    return Path(wav)
