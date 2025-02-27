import os

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from transformers import AutoModel, AutoTokenizer

from models.linear_probe import LinearProbe

app = Flask(__name__)


# Load models
def load_models():
    # Load BERT model
    bert_tag = "google-bert/bert-base-uncased"
    bert_model = AutoModel.from_pretrained(bert_tag)
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_tag)

    # Load ERNIE model
    ernie_tag = "nghuyong/ernie-2.0-base-en"
    ernie_model = AutoModel.from_pretrained(ernie_tag)
    ernie_tokenizer = AutoTokenizer.from_pretrained(ernie_tag)

    # Paths to model checkpoints
    bert_lp_ckpt = os.path.expanduser(
        "~/lprobe/BERT/20250226/lightning_logs/version_0/checkpoints/last.ckpt"
    )
    bert_lp = LinearProbe(bert_model)
    bert_lp.load_state_dict(
        torch.load(bert_lp_ckpt, weights_only=True, map_location=torch.device("cpu"))[
            "state_dict"
        ],
    )

    ernie_lp_ckpt = os.path.expanduser(
        "~/lprobe/ERNIE/20250226/lightning_logs/version_0/checkpoints/last.ckpt"
    )
    ernie_lp = LinearProbe(ernie_model)
    ernie_lp.load_state_dict(
        torch.load(ernie_lp_ckpt, weights_only=True, map_location=torch.device("cpu"))[
            "state_dict"
        ],
    )

    # Set models to evaluation mode
    bert_lp.eval()
    ernie_lp.eval()

    return {
        "bert": {"model": bert_lp, "tokenizer": bert_tokenizer},
        "ernie": {"model": ernie_lp, "tokenizer": ernie_tokenizer},
    }


# Load models at server startup
models = load_models()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/comparison", methods=["POST"])
def comparison():
    # Only handle POST requests with JSON data
    if not request.is_json:
        return jsonify({"error": "Only JSON requests are accepted"}), 400

    data = request.json
    input_text = data.get("text", "")

    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    results = process_models(input_text)
    return jsonify(results)


def process_models(input_text):
    # Process with BERT
    bert_inputs = models["bert"]["tokenizer"](
        input_text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    with torch.no_grad():
        bert_outputs = models["bert"]["model"](bert_inputs)

    # Process with ERNIE
    ernie_inputs = models["ernie"]["tokenizer"](
        input_text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    with torch.no_grad():
        ernie_outputs = models["ernie"]["model"](ernie_inputs)

    # Format results according to required structure
    results = {
        "BERT": {
            f"l{i+1}": float(F.sigmoid(z).item()) for i, z in enumerate(bert_outputs)
        },
        "ERNIE": {
            f"l{i+1}": float(F.sigmoid(z).item()) for i, z in enumerate(ernie_outputs)
        },
    }

    return results


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
