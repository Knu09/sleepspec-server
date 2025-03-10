from enum import Enum
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
from pathlib import Path
from http import HTTPStatus
from dataclasses import dataclass

app = Flask(__name__)

class SD_Class(Enum):
    NSD = 0
    SD = 1

@dataclass
class Classification:
    sd: SD_Class
    confidence_score: float
    # other fields here

@app.route("/", methods=["POST"])
def Index():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file in request."}), HTTPStatus.BAD_REQUEST

    audio_file = request.files["audio"]
    if audio_file.filename:
        file_path = Path("./tmp/uploads") / secure_filename(audio_file.filename)
        audio_file.save(file_path)

        classification = classify(file_path)

        return jsonify({
            "class": classification.sd,
            "result": "You are not sleep deprived.",
        }), HTTPStatus.OK
    
    return jsonify({"error": "There was a problem saving the file"}), HTTPStatus.INTERNAL_SERVER_ERROR

def classify(audio_path: Path) -> Classification:
    # insert classification logic here
    return Classification(sd=SD_Class.NSD, confidence_score=0.826) # 82.6%
