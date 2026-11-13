"""Check the learned baseline, fold-fitted transforms, and saved pipeline."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from heart_ml import baseline
from heart_ml.audit import CATEGORICAL, NUMERIC, SCHEMA, TARGET
from heart_ml.split import create_membership, encode_labels, write_or_verify_splits


@pytest.fixture
def config():
    with open("configs/default.json", encoding="utf-8") as source:
        return json.load(source)


@pytest.fixture
def frame():
    rng = np.random.default_rng(31)
    rows = 240
    data = pd.DataFrame({"id": np.arange(rows)})
    for column in NUMERIC:
        data[column] = rng.normal(size=rows)
    for column in CATEGORICAL:
        data[column] = rng.integers(0, 2, size=rows)
    # A known linear signal checks that this baseline really uses predictors.
    data[TARGET] = np.where(data["Age"] + 0.4 * data["BP"] > 0, "Presence", "Absence")
    return data[SCHEMA + [TARGET]]


def test_logistic_learns_a_signal_and_outputs_aligned_probabilities(config, frame):
    development = baseline.development_rows(frame, create_membership(frame, config), config)
    folds, predictions = baseline.evaluate_model(development, config, "logistic")
    assert folds["model"].eq("logistic").all()
    assert folds["roc_auc"].mean() > 0.9
    assert folds["accuracy"].mean() > 0.8
    assert not folds["convergence_warning"].any()
    assert predictions["probability_presence"].nunique() > 1
    assert predictions["probability_presence"].between(0, 1).all()
    np.testing.assert_array_equal(
        predictions["predicted_label"],
        (predictions["probability_presence"] >= config["threshold"]).astype(int),
    )
    expected_labels = frame.set_index("id")[TARGET].loc[predictions["id"]]
    np.testing.assert_array_equal(predictions["true_label"], encode_labels(expected_labels, config))
    assert (folds[["tn", "fp", "fn", "tp"]].sum(axis=1) == folds["validation_rows"]).all()


def test_logistic_preprocessing_and_fit_only_see_training_fold(monkeypatch, config, frame):
    frame["Age"] = frame["id"].astype(float)
    membership = create_membership(frame, config)
    first_development_id = membership.loc[membership.partition == "development", "id"].iloc[0]
    frame.loc[frame.id == first_development_id, "Sex"] = 999
    development = baseline.development_rows(frame, membership, config)
    original_builder = baseline.build_pipeline
    fits, predictions = [], []

    class RecordingPipeline(Pipeline):
        def fit(self, X, y, **params):
            assert "id" not in X.columns and TARGET not in X.columns
            result = super().fit(X, y, **params)
            prep = self.named_steps["preprocess"]
            observed_mean = prep.named_transformers_["numeric"].named_steps["scale"].mean_[0]
            observed_categories = prep.named_transformers_["categorical"].named_steps["encode"].categories_[0]
            assert observed_mean == pytest.approx(X["Age"].mean())
            assert set(observed_categories) == set(X["Sex"])
            fits.append(set(X["Age"].astype(int)))
            return result

        def predict_proba(self, X, **params):
            predictions.append(set(X["Age"].astype(int)))
            return super().predict_proba(X, **params)

    def recording_builder(name, cfg):
        return RecordingPipeline(original_builder(name, cfg).steps)

    monkeypatch.setattr(baseline, "build_pipeline", recording_builder)
    baseline.evaluate_model(development, config, "logistic")
    holdout = set(membership.loc[membership.partition == "holdout", "id"])
    assert len(fits) == len(predictions) == config["folds"]
    for fold, (fitted_ids, predicted_ids) in enumerate(zip(fits, predictions)):
        assert fitted_ids == set(development.loc[development.fold != fold, "id"])
        assert predicted_ids == set(development.loc[development.fold == fold, "id"])
        assert fitted_ids.isdisjoint(predicted_ids)
        assert holdout.isdisjoint(fitted_ids | predicted_ids)


def test_logistic_run_reuses_split_and_saves_full_pipeline(tmp_path, config, frame):
    source = tmp_path / "train.csv"
    frame.to_csv(source, index=False)
    config["data"]["train"] = str(source)
    config["data"]["test"] = str(tmp_path / "absent-test.csv")
    config["splits_dir"] = str(tmp_path / "splits")
    membership_path, _ = write_or_verify_splits(config)
    original_membership = membership_path.read_bytes()
    output, summary = baseline.run_baseline(config, "logistic", tmp_path / "logistic")
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["model"] == summary["model"] == "logistic"
    assert metadata["feature_set"] == "original"
    assert metadata["split_created"] is False
    assert metadata["holdout_evaluated"] is False
    assert metadata["test_csv_used"] is False
    assert metadata["pipeline_fit_partition"] == "development_only"
    assert metadata["final_fit_convergence_warning"] is False
    assert membership_path.read_bytes() == original_membership
    pipeline = joblib.load(output / "logistic_pipeline.joblib")
    features = config["numeric_features"] + config["categorical_features"]
    assert list(pipeline.feature_names_in_) == features
    assert set(pipeline.named_steps) == {"preprocess", "model"}
    assert metadata["resolved_model_parameters"]["C"] == config["models"]["logistic"]["C"]
    assert metadata["resolved_model_parameters"]["solver"] == "lbfgs"
    membership = pd.read_csv(membership_path)
    selected = frame["id"].isin(membership.loc[membership.partition == "development", "id"])
    scaler = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, frame.loc[selected, NUMERIC].mean().to_numpy())
    probability = pipeline.predict_proba(frame[features].iloc[:10])[:, 1]
    assert np.isfinite(probability).all() and np.ptp(probability) > 0
    assert (output / "development_predictions.csv").exists()
    assert not (output / "dummy_pipeline.joblib").exists()


def test_convergence_warnings_are_retained(config, frame):
    config["models"]["logistic"]["max_iter"] = 1
    development = baseline.development_rows(frame, create_membership(frame, config), config)
    folds, _ = baseline.evaluate_model(development, config, "logistic")
    assert folds["convergence_warning"].all()
    assert folds["notes"].str.contains("converg", case=False).all()
    assert folds["solver_iterations"].eq(1).all()
