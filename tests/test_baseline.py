"""Behavior checks for a reproducible, development-only dummy experiment."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
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
    rows = 120
    data = pd.DataFrame({"id": np.arange(rows)})
    for column in NUMERIC:
        data[column] = np.arange(rows, dtype=float)
    for column in CATEGORICAL:
        data[column] = np.arange(rows) % 2
    data[TARGET] = np.where(np.arange(rows) % 4 == 0, "Presence", "Absence")
    return data[SCHEMA + [TARGET]]


def test_dummy_matches_fold_prior_and_constant_score_metrics(config, frame):
    membership = create_membership(frame, config)
    # Reordering membership must not change which rows belong to each fold.
    shuffled = membership.sample(frac=1, random_state=7)
    development = baseline.development_rows(frame, shuffled, config)
    folds, predictions = baseline.evaluate_dummy(development, config)
    labels = encode_labels(development[TARGET], config)
    assert len(predictions) == len(development)
    assert predictions["id"].is_unique
    holdout_ids = set(membership.loc[membership.partition == "holdout", "id"])
    assert holdout_ids.isdisjoint(predictions["id"])
    for row in folds.to_dict("records"):
        fold = row["fold"]
        validation = development["fold"].eq(fold)
        expected_probability = labels.loc[~validation].mean()
        observed = predictions.loc[predictions["fold"].eq(fold)]
        np.testing.assert_allclose(observed["probability_presence"], expected_probability)
        assert row["roc_auc"] == pytest.approx(0.5)
        assert row["average_precision"] == pytest.approx(labels.loc[validation].mean())
        assert row["accuracy"] == pytest.approx(1 - labels.loc[validation].mean())
        assert row["precision"] == row["recall"] == row["f1"] == 0
        assert row["fp"] == row["tp"] == 0
        assert row["tn"] + row["fn"] == row["validation_rows"]
        assert "No positive predictions" in row["notes"]


def test_no_holdout_or_validation_rows_reach_training(monkeypatch, config, frame):
    membership = create_membership(frame, config)
    development = baseline.development_rows(frame, membership, config)
    training_calls, prediction_calls = [], []

    class RecordingDummy(DummyClassifier):
        def fit(self, X, y, sample_weight=None):
            assert "id" not in X.columns and TARGET not in X.columns
            training_calls.append(set(X["Age"].astype(int)))
            return super().fit(X, y, sample_weight=sample_weight)

        def predict_proba(self, X):
            prediction_calls.append(set(X["Age"].astype(int)))
            return super().predict_proba(X)

    monkeypatch.setattr(
        baseline, "build_pipeline",
        lambda name, cfg: Pipeline([("model", RecordingDummy(strategy="prior"))]),
    )
    baseline.evaluate_dummy(development, config)
    holdout = set(membership.loc[membership.partition == "holdout", "id"])
    assert len(training_calls) == len(prediction_calls) == config["folds"]
    for fold, (trained, predicted) in enumerate(zip(training_calls, prediction_calls)):
        expected_validation = set(development.loc[development.fold == fold, "id"])
        expected_training = set(development.loc[development.fold != fold, "id"])
        assert trained == expected_training
        assert predicted == expected_validation
        assert trained.isdisjoint(predicted)
        assert holdout.isdisjoint(trained | predicted)


@pytest.mark.parametrize("model_name", ["dummy", "logistic"])
def test_holdout_changes_cannot_change_development_metrics(config, frame, model_name):
    membership = create_membership(frame, config)
    original = baseline.development_rows(frame, membership, config)
    original_folds, original_predictions = baseline.evaluate_model(original, config, model_name)
    changed = frame.copy()
    holdout = changed["id"].isin(membership.loc[membership.partition == "holdout", "id"])
    changed.loc[holdout, NUMERIC] = -999
    changed.loc[holdout, TARGET] = "Presence"
    altered = baseline.development_rows(changed, membership, config)
    altered_folds, altered_predictions = baseline.evaluate_model(altered, config, model_name)
    pd.testing.assert_frame_equal(original_predictions, altered_predictions)
    pd.testing.assert_frame_equal(
        original_folds.drop(columns="elapsed_seconds"),
        altered_folds.drop(columns="elapsed_seconds"),
    )


def test_incomplete_membership_and_holdout_in_evaluation_are_rejected(config, frame):
    membership = create_membership(frame, config)
    with pytest.raises(ValueError, match="cover exactly"):
        baseline.development_rows(frame, membership.iloc[1:], config)
    membership.loc[membership.partition == "holdout", "fold"] = 0
    with pytest.raises(ValueError, match="fold -1"):
        baseline.development_rows(frame, membership, config)
    valid = baseline.development_rows(frame, create_membership(frame, config), config)
    valid.loc[0, "partition"] = "holdout"
    with pytest.raises(ValueError, match="Only development"):
        baseline.evaluate_dummy(valid, config)


def test_complete_run_saves_consistent_outputs_without_test_csv(tmp_path, config, frame):
    source = tmp_path / "train.csv"
    frame.to_csv(source, index=False)
    config["data"]["train"] = str(source)
    config["data"]["test"] = str(tmp_path / "does-not-exist.csv")
    config["splits_dir"] = str(tmp_path / "splits")
    membership_path, _ = write_or_verify_splits(config)
    original_split = membership_path.read_bytes()
    output, summary = baseline.run_dummy_baseline(config, tmp_path / "run")
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["split_created"] is False
    assert metadata["holdout_evaluated"] is False
    assert metadata["test_csv_used"] is False
    assert metadata["development_rows"] == 96
    assert metadata["holdout_rows_excluded"] == 24
    assert membership_path.read_bytes() == original_split
    folds = pd.read_csv(output / "cv_results.csv")
    predictions = pd.read_csv(output / "development_predictions.csv")
    assert len(predictions) == 96
    assert summary["metrics"]["roc_auc"]["mean"] == pytest.approx(0.5)
    assert summary["metrics"]["roc_auc"]["fold_sd"] == pytest.approx(0)
    saved_summary = json.loads((output / "summary.json").read_text())
    assert saved_summary == summary
    assert folds["validation_rows"].sum() == len(predictions)
    pipeline = joblib.load(output / "dummy_pipeline.joblib")
    features = config["numeric_features"] + config["categorical_features"]
    probability = pipeline.predict_proba(frame[features].iloc[:2])[:, 1]
    membership = pd.read_csv(membership_path)
    selected = frame["id"].isin(membership.loc[membership.partition == "development", "id"])
    expected = encode_labels(frame.loc[selected, TARGET], config).mean()
    np.testing.assert_allclose(probability, expected)
    with pytest.raises(FileExistsError):
        baseline.run_dummy_baseline(config, output)


def test_failure_is_recorded_without_fabricated_metrics(tmp_path, config):
    config["data"]["train"] = str(tmp_path / "missing.csv")
    config["splits_dir"] = str(tmp_path / "splits")
    output = tmp_path / "failed_run"
    with pytest.raises(ValueError, match="Could not read"):
        baseline.run_dummy_baseline(config, output)
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["error"]["type"] == "ValueError"
    assert not (output / "summary.json").exists()
