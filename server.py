from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
from pydub import AudioSegment
from pathlib import Path
from http import HTTPStatus
from enum import Enum
from dataclasses import dataclass

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
    # insert classification logic here
    return Classification(
        sd=SD_Class.NSD,
        confidence_score=0.826,  # 82.6%
        result="You are not sleep deprived.",
    )


def convertWAV(audio: Path) -> Path:
    wav = audio.with_suffix('.wav')
    file = AudioSegment.from_file(audio)
    file.export(wav, format="wav")

    audio.unlink()
    return Path(wav)
