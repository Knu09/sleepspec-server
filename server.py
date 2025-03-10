from enum import Enum
from flask import Flask, request, jsonify 
from pathlib import Path

from werkzeug.datastructures import FileStorage

app = Flask(__name__)

class Class(Enum):
    NSD = 0
    SD = 1

@app.route("/", methods=["POST"])
def Index():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file in request."}), 400

    audio_file = request.files["audio"]
    if audio_file.filename:
        file_path = Path("./tmp") / audio_file.filename
        audio_file.save(file_path)

        classify(file_path)
        
        return jsonify({
            "class": Class.NSD,
            "result": "You are not sleep deprived.",
        }), 200
    
    return jsonify({"error": "There was a problem saving the file"}), 500

def classify(audio_path: Path) -> Class:
    # insert classification logic here

    return Class.NSD # or Class.SD
