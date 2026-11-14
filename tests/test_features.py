"""Verify numeric terms, fold isolation, selection rules, and experiment outputs."""

import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from heart_ml import baseline
from heart_ml.audit import CATEGORICAL, NUMERIC, SCHEMA, TARGET
from heart_ml.feature_experiments import choose_variant, development_eda, run_experiments
from heart_ml.features import NumericFeatureEngineer
from heart_ml.preprocess import build_pipeline
from heart_ml.split import create_membership, write_or_verify_splits


@pytest.fixture
def config():
    with open("configs/default.json", encoding="utf-8") as source:
        return json.load(source)


@pytest.fixture
def frame():
    rng = np.random.default_rng(82)
    data = pd.DataFrame({"id": np.arange(180)})
    for column in NUMERIC:
        data[column] = rng.normal(size=len(data))
    for column in CATEGORICAL:
        data[column] = rng.integers(0, 2, size=len(data))
    data[TARGET] = np.where(data["Age"] + data["BP"] > 0, "Presence", "Absence")
    return data[SCHEMA + [TARGET]]


@pytest.mark.parametrize("name,extra", [
    ("original", []), ("age_squared", [[4, 16]]),
    ("age_st_interaction", [[6, 8]]), ("combined", [[4, 16], [6, 8]]),
])
def test_numeric_terms_are_correct_named_and_nonmutating(name, extra):
    X = np.array([[2, 120, 200, 150, 3, 1], [4, 130, 210, 160, 2, 0]], dtype=float)
    before = X.copy()
    transformer = clone(NumericFeatureEngineer(name))
    observed = transformer.fit_transform(X, [0, 1])
    expected = np.column_stack([X, *[np.asarray(column) for column in extra]]) if extra else X
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(X, before)
    assert len(transformer.get_feature_names_out()) == expected.shape[1]
    if name == "combined":
        assert transformer.get_feature_names_out()[-2:].tolist() == ["Age_squared", "Age_x_ST_depression"]


def test_impute_then_engineer_then_scale_uses_training_medians(config):
    training = pd.DataFrame({column: [1.0, 2.0, 3.0, 4.0] for column in NUMERIC})
    training["Age"] = [2, 4, np.nan, 6]
    training["ST depression"] = [1, np.nan, 3, 5]
    for column in CATEGORICAL:
        training[column] = [0, 1, 0, 1]
    model = build_pipeline("logistic", config, feature_set="combined").fit(training, [0, 1, 0, 1])
    numeric = model.named_steps["preprocess"].named_transformers_["numeric"]
    assert list(numeric.named_steps) == ["impute", "engineer", "scale"]
    expected_age = np.array([2, 4, 4, 6], dtype=float)
    expected_st = np.array([1, 3, 3, 5], dtype=float)
    engineered = numeric[:-1].transform(training[NUMERIC])
    np.testing.assert_array_equal(engineered[:, -2], expected_age ** 2)
    np.testing.assert_array_equal(engineered[:, -1], expected_age * expected_st)
    validation = training.iloc[:1].copy()
    validation["Age"] = np.nan
    validation["ST depression"] = 20
    np.testing.assert_array_equal(numeric[:-1].transform(validation[NUMERIC])[:, -2:], [[16, 80]])
    np.testing.assert_allclose(numeric.named_steps["scale"].mean_, engineered.mean(axis=0))


def test_engineered_cv_keeps_fit_and_prediction_rows_disjoint(monkeypatch, config, frame):
    frame["Age"] = frame["id"].astype(float)
    membership = create_membership(frame, config)
    development = baseline.development_rows(frame, membership, config)
    real_builder = baseline.build_pipeline
    fits, predicts = [], []

    class RecordingPipeline(Pipeline):
        def fit(self, X, y, **params):
            result = super().fit(X, y, **params)
            numeric = self.named_steps["preprocess"].named_transformers_["numeric"]
            expected = numeric[:-1].transform(X[NUMERIC]).mean(axis=0)
            np.testing.assert_allclose(numeric.named_steps["scale"].mean_, expected)
            fits.append(set(X["Age"].astype(int)))
            return result

        def predict_proba(self, X, **params):
            predicts.append(set(X["Age"].astype(int)))
            return super().predict_proba(X, **params)

    def builder(name, cfg, feature_set="original"):
        return RecordingPipeline(real_builder(name, cfg, feature_set=feature_set).steps)

    monkeypatch.setattr(baseline, "build_pipeline", builder)
    baseline.evaluate_model(development, config, "logistic", feature_set="combined")
    holdout = set(membership.loc[membership.partition == "holdout", "id"])
    assert len(fits) == len(predicts) == 3
    for fold, (training_ids, validation_ids) in enumerate(zip(fits, predicts)):
        assert training_ids == set(development.loc[development.fold != fold, "id"])
        assert validation_ids == set(development.loc[development.fold == fold, "id"])
        assert training_ids.isdisjoint(validation_ids)
        assert holdout.isdisjoint(training_ids | validation_ids)


@pytest.mark.parametrize("name", ["age_squared", "age_st_interaction", "combined"])
def test_holdout_changes_do_not_change_engineered_predictions(name, config, frame):
    membership = create_membership(frame, config)
    development = baseline.development_rows(frame, membership, config)
    first_scores, first_predictions = baseline.evaluate_model(development, config, "logistic", name)
    changed = frame.copy()
    mask = changed["id"].isin(membership.loc[membership.partition == "holdout", "id"])
    changed.loc[mask, NUMERIC] = 1e9
    changed.loc[mask, TARGET] = "Presence"
    second = baseline.development_rows(changed, membership, config)
    second_scores, second_predictions = baseline.evaluate_model(second, config, "logistic", name)
    pd.testing.assert_frame_equal(first_predictions, second_predictions)
    pd.testing.assert_frame_equal(
        first_scores.drop(columns="elapsed_seconds"), second_scores.drop(columns="elapsed_seconds")
    )


def test_selection_rejects_tiny_or_inconsistent_gains():
    def row(name, score, gain, all_improved, extra):
        return {
            "feature_set": name, "status": "completed", "roc_auc": score,
            "mean_delta_roc_auc": gain, "all_folds_improved": all_improved,
            "convergence_warnings": 0, "added_features": extra,
        }
    records = [
        row("original", .95, 0, False, 0),
        row("age_squared", .9502, .0002, True, 1),
        row("age_st_interaction", .951, .001, False, 1),
        row("combined", .9507, .0007, True, 2),
    ]
    selected = choose_variant(records, .0005)
    assert selected["recommended_feature_set"] == "combined"
    assert selected["highest_mean_roc_auc_feature_set"] == "age_st_interaction"
    assert choose_variant(records, .002)["recommended_feature_set"] == "original"


def test_suite_saves_all_candidates_plots_and_reloadable_pipeline(tmp_path, config, frame):
    source = tmp_path / "train.csv"
    frame.to_csv(source, index=False)
    cfg = copy.deepcopy(config)
    cfg["data"]["train"] = str(source)
    cfg["data"]["test"] = str(tmp_path / "not-present.csv")
    cfg["splits_dir"] = str(tmp_path / "splits")
    membership_path, _ = write_or_verify_splits(cfg)
    before = membership_path.read_bytes()
    plan = {
        "feature_sets": ["original", "age_squared"],
        "minimum_mean_roc_auc_gain": .0005,
        "require_positive_gain_in_every_fold": True,
    }
    output = tmp_path / "experiment"
    result = run_experiments(cfg, plan, output)
    assert result["status"] == "completed"
    assert [row["feature_set"] for row in result["comparison"]] == plan["feature_sets"]
    assert result["data"]["development_rows"] == 144
    assert result["data"]["holdout_rows_excluded"] == 36
    assert result["holdout_evaluated"] is False and result["test_csv_used"] is False
    assert membership_path.read_bytes() == before
    assert json.loads((output / "development_eda.json").read_text())["rows"] == 144
    for name in ("feature_engineering_eda.png", "feature_engineering_comparison.png"):
        assert (output / name).read_bytes().startswith(b"\x89PNG")
    pipeline = joblib.load(output / "age_squared/logistic_pipeline.joblib")
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    assert "numeric__Age_squared" in names
    probability = pipeline.predict_proba(frame[NUMERIC + CATEGORICAL].iloc[:4])
    assert probability.shape == (4, 2) and np.isfinite(probability).all()
    assert (output / "feature_engineering.md").exists()
    assert (output / "feature_engineering_reference.json").exists()


def test_invalid_inputs_do_not_enable_unrelated_models(tmp_path, config, frame):
    with pytest.raises(ValueError, match="only for logistic"):
        build_pipeline("dummy", config, feature_set="age_squared")
    with pytest.raises(ValueError, match="Unknown feature"):
        NumericFeatureEngineer("unexpected").fit(np.ones((2, 6)))
    membership = create_membership(frame, config)
    joined = frame.merge(membership, on="id")
    with pytest.raises(ValueError, match="development rows only"):
        development_eda(joined, config, tmp_path)
