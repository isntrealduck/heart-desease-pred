"""Local web form for single-row inference with the frozen AdaBoost pipeline."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import math
import mimetypes
import os
from urllib.parse import unquote, urlparse

import joblib
import numpy as np
import pandas as pd

from .audit import CATEGORICAL, CURRENT_CATEGORICAL, FEATURES, NUMERIC

MODEL_NUMERIC = [name for name in NUMERIC if name != "BP"]
MODEL_FEATURES = MODEL_NUMERIC + CURRENT_CATEGORICAL
PREVIOUS_FEATURES = MODEL_NUMERIC + CATEGORICAL
REMOVED = {"BP", "FBS over 120"}


WEB_ROOT = Path(__file__).resolve().parents[2] / "web"
MODEL_PATH = Path("artifacts/evaluated_pipeline.joblib")
MAX_REQUEST_BYTES = 16_384

CATEGORY_MAPS = {
    "Sex": {"female": 0, "male": 1},
    "Chest pain type": {
        "typical_angina": 1,
        "atypical_angina": 2,
        "non_anginal_pain": 3,
        "asymptomatic": 4,
    },
    "EKG results": {
        "normal": 0,
        "st_t_abnormality": 1,
        "left_ventricular_hypertrophy": 2,
    },
    "Exercise angina": {"no": 0, "yes": 1},
    "Slope of ST": {"upsloping": 1, "flat": 2, "downsloping": 3},
    "Thallium": {"normal": 3, "fixed_defect": 6, "reversible_defect": 7},
}

CATEGORY_LABELS = {
    "Sex": {"female": "Female", "male": "Male"},
    "Chest pain type": {
        "typical_angina": "Typical angina",
        "atypical_angina": "Atypical angina",
        "non_anginal_pain": "Non-anginal pain",
        "asymptomatic": "Asymptomatic",
    },
    "EKG results": {
        "normal": "Normal",
        "st_t_abnormality": "ST–T wave abnormality",
        "left_ventricular_hypertrophy": "Probable/definite left ventricular hypertrophy",
    },
    "Exercise angina": {"no": "No", "yes": "Yes"},
    "Slope of ST": {
        "upsloping": "Upsloping",
        "flat": "Flat",
        "downsloping": "Downsloping",
    },
    "Thallium": {
        "normal": "Normal",
        "fixed_defect": "Fixed defect",
        "reversible_defect": "Reversible defect",
    },
}

FEATURE_INFO = {
    "Age": {
        "label": "Age",
        "description": "Age of the person in the record.",
        "unit": "years",
    },
    "Cholesterol": {
        "label": "Serum cholesterol",
        "description": "Serum cholesterol, following the source schema.",
        "unit": "mg/dL",
    },
    "Max HR": {
        "label": "Maximum heart rate",
        "description": "Maximum heart rate achieved. The source documentation does not state a unit.",
        "unit": None,
    },
    "ST depression": {
        "label": "ST depression",
        "description": "Exercise-induced ST depression relative to rest. The source documentation does not state a unit.",
        "unit": None,
    },
    "Number of vessels fluro": {
        "label": "Major vessels (fluoroscopy)",
        "description": "Number of major vessels visualized by fluoroscopy. The local data's coding range is not asserted here.",
        "unit": "count",
    },
    "Sex": {
        "label": "Sex",
        "description": "Sex category as represented in the source schema.",
        "unit": None,
    },
    "Chest pain type": {
        "label": "Chest pain type",
        "description": "Chest-pain category from the source schema.",
        "unit": None,
    },
    "EKG results": {
        "label": "Resting ECG result",
        "description": "Resting electrocardiographic result category.",
        "unit": None,
    },
    "Exercise angina": {
        "label": "Exercise-induced angina",
        "description": "Whether exercise-induced angina was recorded.",
        "unit": None,
    },
    "Slope of ST": {
        "label": "Peak-exercise ST slope",
        "description": "Slope of the peak-exercise ST segment.",
        "unit": None,
    },
    "Thallium": {
        "label": "Thallium test result",
        "description": "Thallium stress-test result category.",
        "unit": None,
    },
}


def input_frame(payload):
    """Validate one set of form values and return the model's fixed feature frame."""
    if not isinstance(payload, dict) or set(payload) != set(MODEL_FEATURES):
        raise ValueError("Provide exactly the 11 current model features.")

    row = {}
    for name in MODEL_FEATURES:
        value = payload[name]
        if value is None or value == "":
            row[name] = np.nan
            continue
        if name in CATEGORICAL:
            continue
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a number or blank.")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a number or blank.") from error
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite.")
        row[name] = number

    for name, values in CATEGORY_MAPS.items():
        value = payload[name]
        if value is None or value == "":
            row[name] = np.nan
        elif not isinstance(value, str) or value not in values:
            raise ValueError(f"Choose a listed descriptive option for {name}.")
        else:
            row[name] = float(values[value])

    if all(pd.isna(row[name]) for name in MODEL_FEATURES):
        raise ValueError("Enter at least one feature value.")
    return pd.DataFrame([{name: row[name] for name in MODEL_FEATURES}], columns=MODEL_FEATURES)


def _adapt_legacy_bp(pipeline, frame):
    """Adapt historical schemas only after proving removed columns are unused."""
    expected = list(getattr(pipeline, "feature_names_in_", ()))
    if expected == MODEL_FEATURES:
        return frame
    if expected not in (PREVIOUS_FEATURES, FEATURES):
        raise ValueError("Saved model input schema is not supported by this form.")
    try:
        transformed_names = list(
            pipeline.named_steps["preprocess"].get_feature_names_out()
        )
        removed_indexes = []
        removed_for_schema = {"FBS over 120"}
        if expected == FEATURES:
            removed_for_schema.add("BP")
        for name in removed_for_schema:
            matches = [i for i, transformed in enumerate(transformed_names)
                       if transformed == f"numeric__{name}" or
                       (name == "FBS over 120" and transformed.startswith("categorical__FBS over 120_"))]
            if not matches:
                raise ValueError(f"Cannot locate transformed {name}.")
            removed_indexes.extend(matches)
        estimators = list(pipeline.named_steps["model"].estimators_)
        if not estimators or any(not hasattr(estimator, "tree_") for estimator in estimators):
            raise ValueError("Cannot verify BP is unused by every fitted weak learner.")
        features = [estimator.tree_.feature for estimator in estimators]
        if any(np.any(((indexes < 0) & (indexes != -2)) |
                      (indexes >= len(transformed_names)))
               for indexes in features):
            raise ValueError("Unverifiable transformed tree feature index.")
        used = any(np.isin(indexes, removed_indexes).any() for indexes in features)
    except (AttributeError, KeyError, ValueError, IndexError) as error:
        raise ValueError("Cannot verify removed features are unused by the saved legacy model.") from error
    if used:
        raise ValueError("Saved legacy model uses BP or FBS; refusing to fabricate their values.")
    adapted = frame.copy()
    if expected == FEATURES:
        adapted.insert(NUMERIC.index("BP"), "BP", np.nan)
        adapted["FBS over 120"] = np.nan
    else:
        adapted["FBS over 120"] = np.nan
    return adapted[expected]


def predict_one(pipeline, payload):
    """Predict with the frozen pipeline; return only the label and score."""
    frame = _adapt_legacy_bp(pipeline, input_frame(payload))
    classes = np.asarray(pipeline.classes_)
    positive_columns = np.flatnonzero(classes == 1)
    if len(positive_columns) != 1:
        raise ValueError("Saved model does not contain the expected positive class.")
    probabilities = np.asarray(pipeline.predict_proba(frame), dtype=float)
    if probabilities.shape != (1, len(classes)):
        raise ValueError("Model returned an unexpected probability shape.")
    probability = float(probabilities[0, positive_columns[0]])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Model returned an invalid probability.")
    return {
        "predicted_label": "Presence" if probability >= 0.5 else "Absence",
        "probability_presence": probability,
        "threshold": 0.5,
    }


def category_options(pipeline):
    """Expose human-readable categories compatible with the fitted encoder."""
    encoder = (
        pipeline.named_steps["preprocess"]
        .named_transformers_["categorical"]
        .named_steps["encode"]
    )
    selectors = next(transformer[2] for transformer in
                     pipeline.named_steps["preprocess"].transformers_
                     if transformer[0] == "categorical")
    options = {}
    for name, fitted_values in zip(selectors, encoder.categories_):
        if name not in CATEGORY_MAPS:
            continue
        known_codes = {float(value) for value in fitted_values}
        options[name] = [
            {"value": key, "label": CATEGORY_LABELS[name][key]}
            for key, code in CATEGORY_MAPS[name].items()
            if float(code) in known_codes
        ]
    return options


def schema_payload(pipeline):
    """Return the public form schema without exposing raw-source-only columns."""
    return {
        "features": MODEL_FEATURES,
        "numeric": MODEL_NUMERIC,
        "categorical": CATEGORICAL,
        "category_options": category_options(pipeline),
        "feature_info": FEATURE_INFO,
    }


class PredictionHandler(BaseHTTPRequestHandler):
    pipeline = None
    web_root = WEB_ROOT

    def log_message(self, format, *args):
        # Form values are sensitive; keep request details out of server logs.
        return

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, value):
        self._send(
            status,
            "application/json; charset=utf-8",
            json.dumps(value, allow_nan=False).encode("utf-8"),
        )

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/schema":
            self._json(200, schema_payload(self.pipeline))
            return

        relative = "index.html" if path == "/" else unquote(path).lstrip("/")
        root = self.web_root.resolve()
        requested = (root / relative).resolve()
        if requested != root and root not in requested.parents:
            self._json(404, {"error": "Not found."})
            return
        if not requested.is_file():
            self._json(404, {"error": "Not found."})
            return
        content_type = mimetypes.guess_type(requested.name)[0] or "application/octet-stream"
        self._send(200, content_type, requested.read_bytes())

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            self._json(404, {"error": "Not found."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("Request must contain a small JSON form submission.")
            payload = json.loads(self.rfile.read(length))
            self._json(200, predict_one(self.pipeline, payload))
        except (ValueError, json.JSONDecodeError) as error:
            self._json(400, {"error": str(error)})
        except Exception:
            self._json(500, {"error": "Prediction failed. Check the local model setup."})


def main():
    pipeline_path = Path(os.environ.get("HEART_ML_PIPELINE", MODEL_PATH))
    if not pipeline_path.is_file():
        raise SystemExit(
            f"Saved evaluated pipeline not found: {pipeline_path}. "
            "Run the app from the project root."
        )
    handler = type("ConfiguredPredictionHandler", (PredictionHandler,), {
        "pipeline": joblib.load(pipeline_path),
        "web_root": Path(os.environ.get("HEART_ML_WEB_ROOT", WEB_ROOT)),
    })
    server = ThreadingHTTPServer(("0.0.0.0", 8000), handler)
    print("Prediction demo listening on http://localhost:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
