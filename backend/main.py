# flask backend: serves the static frontend and a /api/predict endpoint
# that runs a submitted student's info through the same preprocessing as
# training, then the blind or full model depending on whether G1/G2 were given

import os
import sys

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from preprocess import ENUM_FIELDS, RANGE_FIELDS, encode_common, performance_category  # noqa: E402

MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")

model_blind = joblib.load(os.path.join(MODELS_DIR, "model_blind.pkl"))
columns_blind = joblib.load(os.path.join(MODELS_DIR, "feature_columns_blind.pkl"))
model_full = joblib.load(os.path.join(MODELS_DIR, "model_full.pkl"))
columns_full = joblib.load(os.path.join(MODELS_DIR, "feature_columns_full.pkl"))

GRADE_FIELDS = {"G1": (0, 20), "G2": (0, 20)}
REQUIRED_FIELDS = list(ENUM_FIELDS) + list(RANGE_FIELDS)


def validate(payload):
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in payload or payload[field] in (None, ""):
            errors.append(f"missing field: {field}")

    for field, allowed in ENUM_FIELDS.items():
        if field in payload and payload[field] not in allowed:
            errors.append(f"{field} must be one of {allowed}")

    numeric_fields = {**RANGE_FIELDS, **GRADE_FIELDS}
    for field, (lo, hi) in numeric_fields.items():
        if field not in payload or payload[field] in (None, ""):
            continue
        try:
            value = float(payload[field])
        except (TypeError, ValueError):
            errors.append(f"{field} must be a number")
            continue
        if not (lo <= value <= hi):
            errors.append(f"{field} must be between {lo} and {hi}")

    return errors


@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    errors = validate(payload)
    if errors:
        return jsonify(errors=errors), 400

    has_grades = payload.get("G1") not in (None, "") and payload.get("G2") not in (None, "")

    # form values arrive as strings (FormData/JSON from the browser), but
    # the model needs real numbers - cast the numeric fields explicitly
    row = {field: payload[field] for field in REQUIRED_FIELDS}
    for field in RANGE_FIELDS:
        row[field] = float(row[field])
    if has_grades:
        row["G1"] = float(payload["G1"])
        row["G2"] = float(payload["G2"])

    encoded = encode_common(pd.DataFrame([row]))

    if has_grades:
        X = encoded.reindex(columns=columns_full, fill_value=0)
        prediction = model_full.predict(X)[0]
        variant = "full"
    else:
        X = encoded.reindex(columns=columns_blind, fill_value=0)
        prediction = model_blind.predict(X)[0]
        variant = "blind"

    prediction = max(0.0, min(20.0, float(prediction)))

    return jsonify(
        prediction=round(prediction, 1),
        category=performance_category(prediction),
        model=variant,
    )


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5050)))
