import numpy as np
import pandas as pd
import pytest

from heart_ml.audit import CATEGORICAL, CURRENT_CATEGORICAL, FEATURES, NUMERIC, validate_config
from heart_ml.predict_web import (
    CATEGORY_MAPS, MODEL_FEATURES, PREVIOUS_FEATURES, _adapt_legacy_bp, input_frame, predict_one,
    schema_payload,
)
from heart_ml.preprocess import build_pipeline


class FixedPipeline:
    classes_ = np.array([0, 1])
    feature_names_in_ = np.asarray(MODEL_FEATURES)

    def __init__(self, probability):
        self.probability = probability
        self.seen = None

    def predict_proba(self, frame):
        self.seen = frame.copy()
        return np.array([[1 - self.probability, self.probability]])


def valid_payload():
    payload = {feature: "1" for feature in MODEL_FEATURES}
    payload.update({
        "Sex": "female",
        "Chest pain type": "typical_angina",
        "EKG results": "normal",
        "Exercise angina": "no",
        "Slope of ST": "upsloping",
        "Thallium": "normal",
    })
    return payload


def test_input_frame_orders_schema_and_allows_blank_imputation():
    payload = valid_payload()
    payload["Age"] = None
    frame = input_frame(payload)
    assert list(frame.columns) == MODEL_FEATURES
    assert np.isnan(frame.loc[0, "Age"])
    assert frame.loc[0, "Sex"] == 0
    assert frame.loc[0, "Chest pain type"] == 1


@pytest.mark.parametrize(
    ("feature", "choice", "code"),
    [
        (feature, choice, code)
        for feature, choices in CATEGORY_MAPS.items()
        for choice, code in choices.items()
    ],
)
def test_descriptive_categories_map_to_fitted_codes(feature, choice, code):
    payload = valid_payload()
    payload[feature] = choice
    assert input_frame(payload).loc[0, feature] == code


@pytest.mark.parametrize("value", ["1", "unknown", True])
def test_invalid_categorical_value_rejected(value):
    payload = valid_payload()
    payload[CATEGORICAL[0]] = value
    with pytest.raises(ValueError):
        input_frame(payload)


def test_requires_exact_feature_set_and_at_least_one_value():
    with pytest.raises(ValueError, match="exactly the 11"):
        input_frame({"Age": "50"})
    with pytest.raises(ValueError, match="at least one"):
        input_frame({feature: None for feature in MODEL_FEATURES})


def test_predict_one_uses_positive_class_and_frozen_threshold():
    pipeline = FixedPipeline(0.5)
    result = predict_one(pipeline, valid_payload())
    assert result == {
        "predicted_label": "Presence",
        "probability_presence": 0.5,
        "threshold": 0.5,
    }
    assert list(pipeline.seen.columns) == MODEL_FEATURES


def test_predict_one_rejects_invalid_model_probabilities():
    with pytest.raises(ValueError, match="invalid probability"):
        predict_one(FixedPipeline(1.2), valid_payload())


def test_config_accepts_exact_supported_profiles_and_rejects_others():
    import json
    with open("configs/default.json") as source:
        current = json.load(source)
    validate_config(current)
    for numeric, categorical in (
        (NUMERIC.copy(), CATEGORICAL.copy()),
        (current["numeric_features"], CATEGORICAL.copy()),
    ):
        validate_config({**current, "numeric_features": numeric,
                         "categorical_features": categorical})
    for numeric, categorical in (
        (current["numeric_features"][:-1], current["categorical_features"]),
        (current["numeric_features"][::-1], current["categorical_features"]),
        (current["numeric_features"], CATEGORICAL[:-1]),
        (current["numeric_features"], current["categorical_features"] + ["FBS over 120"]),
    ):
        with pytest.raises(ValueError, match="features"):
            validate_config({**current, "numeric_features": numeric,
                             "categorical_features": categorical})


def test_new_pipeline_fits_and_predicts_with_eleven_inputs():
    import json
    with open("configs/default.json") as source:
        config = json.load(source)
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({name: rng.normal(size=40) for name in config["numeric_features"]})
    for name in CURRENT_CATEGORICAL:
        frame[name] = rng.integers(0, 2, size=40)
    y = (frame["Age"] > 0).astype(int)
    model = build_pipeline("adaboost", config).fit(frame, y)
    assert list(model.feature_names_in_) == MODEL_FEATURES
    assert model.predict_proba(frame.iloc[:3]).shape == (3, 2)
    schema = schema_payload(model)
    assert len(schema["features"]) == 11
    assert "BP" not in schema["features"] and "FBS over 120" not in schema["features"]
    assert "BP" not in schema["feature_info"] and "FBS over 120" not in schema["feature_info"]
    assert list(schema["category_options"]) == CURRENT_CATEGORICAL
    assert "Cholesterol" in schema["features"] and "Cholesterol" in schema["feature_info"]


def test_legacy_pipeline_adapter_requires_all_removed_columns_unused():
    import json
    with open("configs/default.json") as source:
        config = json.load(source)
    config["numeric_features"] = NUMERIC.copy()
    config["categorical_features"] = CATEGORICAL.copy()
    config["models"]["adaboost"] = {"n_estimators": 50, "learning_rate": 1.0}
    rng = np.random.default_rng(21)
    frame = pd.DataFrame({name: rng.normal(size=60) for name in NUMERIC})
    frame["BP"] = 120.0
    for name in CATEGORICAL:
        frame[name] = rng.integers(0, 2, size=60)
    y = (frame["Age"] > 0).astype(int)
    model = build_pipeline("adaboost", config).fit(frame[FEATURES], y)
    assert all(not np.any(est.tree_.feature == 1) for est in model.named_steps["model"].estimators_)
    adapted = _adapt_legacy_bp(model, input_frame(valid_payload()))
    assert adapted.columns.tolist() == FEATURES
    assert np.isnan(adapted.loc[0, "BP"])
    assert np.isnan(adapted.loc[0, "FBS over 120"])
    assert model.predict_proba(adapted).shape == (1, 2)


def test_legacy_adapter_refuses_model_that_uses_bp():
    from types import SimpleNamespace

    class Prep:
        def get_feature_names_out(self):
            return np.asarray([f"numeric__{name}" for name in NUMERIC] +
                              ["categorical__Sex_0", "categorical__Chest pain type_0",
                               "categorical__FBS over 120_0", "categorical__FBS over 120_1",
                               "categorical__EKG results_0", "categorical__Exercise angina_0",
                               "categorical__Slope of ST_0", "categorical__Thallium_0"])

    class UsedBPModel:
        estimators_ = [SimpleNamespace(tree_=SimpleNamespace(feature=np.asarray([1])))]

    class Legacy:
        feature_names_in_ = np.asarray(FEATURES)
        named_steps = {
            "preprocess": Prep(),
            "model": UsedBPModel(),
        }

    with pytest.raises(ValueError, match="uses BP or FBS"):
        _adapt_legacy_bp(Legacy(), input_frame(valid_payload()))


@pytest.mark.parametrize("transformed_index", [1, 8, 9])
def test_legacy_adapter_refuses_any_used_removed_transformed_column(transformed_index):
    from types import SimpleNamespace

    class Prep:
        def get_feature_names_out(self):
            return np.asarray([f"numeric__{name}" for name in NUMERIC] +
                              ["categorical__Sex_0", "categorical__Chest pain type_0",
                               "categorical__FBS over 120_0", "categorical__FBS over 120_1",
                               "categorical__EKG results_0", "categorical__Exercise angina_0",
                               "categorical__Slope of ST_0", "categorical__Thallium_0"])

    legacy = SimpleNamespace(
        feature_names_in_=np.asarray(FEATURES),
        named_steps={
            "preprocess": Prep(),
            "model": SimpleNamespace(estimators_=[
                SimpleNamespace(tree_=SimpleNamespace(
                    feature=np.asarray([transformed_index])
                ))
            ]),
        },
    )
    with pytest.raises(ValueError, match="uses BP or FBS"):
        _adapt_legacy_bp(legacy, input_frame(valid_payload()))


@pytest.mark.parametrize("profile", ["legacy13", "bp12"])
def test_legacy_category_options_follow_actual_fitted_selectors(profile):
    import json
    with open("configs/default.json") as source:
        config = json.load(source)
    config["categorical_features"] = CATEGORICAL.copy()
    if profile == "legacy13":
        config["numeric_features"] = NUMERIC.copy()
        numeric = NUMERIC
    else:
        numeric = [name for name in NUMERIC if name != "BP"]
    rng = np.random.default_rng(45)
    frame = pd.DataFrame({name: rng.normal(size=80) for name in numeric})
    for name in CATEGORICAL:
        frame[name] = rng.integers(0, 2, size=80)
    y = (frame["Age"] > 0).astype(int)
    model = build_pipeline("adaboost", config).fit(frame, y)
    schema = schema_payload(model)
    assert list(schema["category_options"]) == CURRENT_CATEGORICAL
    assert "EKG results" in schema["category_options"]
    assert "Exercise angina" in schema["category_options"]
    assert "FBS over 120" not in schema["category_options"]
    payload = valid_payload()
    if profile == "bp12":
        from types import SimpleNamespace
        model = SimpleNamespace(
            feature_names_in_=np.asarray(PREVIOUS_FEATURES),
            named_steps=model.named_steps,
        )
    adapted = _adapt_legacy_bp(model, input_frame(payload))
    assert adapted.columns.tolist() == list(model.feature_names_in_)
