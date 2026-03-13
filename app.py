"""
NephroLearn – Flask backend for FSGS kidney damage analysis.
Swap the `run_model()` function body with your real CNN inference when ready.
"""

import io
import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)
CORS(app)

# ─── Load model once at startup ────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Best_FSGS_Model.keras")
model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = (224, 224)   # ← match the size your model was trained on


# ─── Stage definitions ───────────────────────────────────────────────────────

STAGES = [
    {
        "stage": "Stage I – Minimal Change",
        "range": (0, 20),
        "description": (
            "Early-stage FSGS with minimal glomerular involvement. "
            "Kidney function is largely preserved. Close monitoring and "
            "lifestyle adjustments are recommended."
        ),
    },
    {
        "stage": "Stage II – Mild Damage",
        "range": (20, 40),
        "description": (
            "Mild focal scarring detected in glomerular tissue. "
            "Medical management with ACE inhibitors or ARBs is typically initiated "
            "to slow progression."
        ),
    },
    {
        "stage": "Stage III – Moderate Damage",
        "range": (40, 60),
        "description": (
            "Moderate segmental sclerosis affecting a significant portion of glomeruli. "
            "Immunosuppressive therapy may be indicated. Regular nephrology follow-up "
            "is essential."
        ),
    },
    {
        "stage": "Stage IV – Severe Damage",
        "range": (60, 80),
        "description": (
            "Severe glomerulosclerosis with markedly reduced GFR. "
            "Aggressive treatment and preparation for renal replacement therapy "
            "should be discussed with the care team."
        ),
    },
    {
        "stage": "Stage V – End-Stage Renal Disease",
        "range": (80, 101),
        "description": (
            "Extensive scarring with near-total loss of kidney function. "
            "Dialysis or kidney transplantation is typically required at this stage."
        ),
    },
]


def determine_stage(damage_pct: float) -> dict:
    for s in STAGES:
        lo, hi = s["range"]
        if lo <= damage_pct < hi:
            return s
    return STAGES[-1]


# ─── Real inference ─────────────────────────────────────────────────────────
def run_model(image: Image.Image) -> dict:

    # Preprocess
    img = image.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Model prediction (single probability)
    prob = float(model.predict(arr)[0][0])

    # Stage thresholds
    if prob < 0.50:
        stage = "Normal"
        damage_pct = 10
        description = "No significant glomerular sclerosis detected."

    elif prob < 0.65:
        stage = "Stage I – Mild Damage"
        damage_pct = 30
        description = "Early-stage focal glomerular damage with mild sclerosis."

    elif prob < 0.75:
        stage = "Stage II – Moderate Damage"
        damage_pct = 50
        description = "Moderate glomerular sclerosis detected in the tissue."

    elif prob < 0.85:
        stage = "Stage III – Severe Damage"
        damage_pct = 70
        description = "Significant glomerular damage indicating progressive FSGS."

    else:
        stage = "Stage IV – Critical Damage"
        damage_pct = 90
        description = "Extensive glomerular sclerosis indicating advanced disease."

    return {
        "damage_percent": damage_pct,
        "stage": stage,
        "description": description
    }

# ─── Routes ──────────────────────────────────────────────────────────────────

from flask import render_template

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not open image. Please upload a valid image file."}), 422

    result = run_model(image)
    return jsonify(result)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  NephroLearn backend running → http://127.0.0.1:{port}\n")
    app.run(debug=True, port=port)
