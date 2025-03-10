from enum import Enum
from flask import Flask, request, jsonify 
from pathlib import Path
from http import HTTPStatus

app = Flask(__name__)

class Class(Enum):
    NSD = 0
    SD = 1

@app.route("/", methods=["POST"])
def Index():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file in request."}), HTTPStatus.BAD_REQUEST

    audio_file = request.files["audio"]
    if audio_file.filename:
        file_path = Path("./tmp/uploads") / audio_file.filename
        audio_file.save(file_path)

        classify(file_path)
        
        return jsonify({
            "class": Class.NSD,
            "result": "You are not sleep deprived.",
        }), HTTPStatus.OK
    
    return jsonify({"error": "There was a problem saving the file"}), HTTPStatus.INTERNAL_SERVER_ERROR

def classify(audio_path: Path) -> Class:
    # insert classification logic here

    return Class.NSD # or Class.SD
